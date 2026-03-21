"""
tune_engine.py — Tune team_goraieb eval weights with the generic optimizer.

Three fitness modes:
  1. self_play  — candidate plays N games vs the engine with default weights
  2. test_suite — score on a set of EPD positions (does the engine find the best move?)
  3. texel      — minimize eval error against a CSV of (fen, result) pairs

Run:
    python tune_engine.py                  # default: self_play, genetic, small run
    python tune_engine.py --mode texel --texel-file positions.csv
    python tune_engine.py --mode test_suite --epd-file suite.epd
    python tune_engine.py --strategy hillclimb --games 50
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
import time

import chess
import chess.polyglot

# ── Import the engine and optimizer ───────────────────────────────────────────
import team_goraieb as engine
from optimizer import Parameter, Optimizer

# ── Snapshot the default weights so the reference engine is always the same ───
DEFAULT_WEIGHTS = dict(engine.WEIGHTS)


# ═══════════════════════════════════════════════════════════════════════════════
#  PARAMETER DEFINITIONS — edit ranges/steps here
# ═══════════════════════════════════════════════════════════════════════════════

def build_params(coarse: bool = False) -> list[Parameter]:
    """
    Build the list of tunable parameters.

    Set coarse=True for a fast first pass with wide steps,
    then re-run with coarse=False to refine around the winner.
    """
    s = 2 if coarse else 1  # step multiplier

    return [
        # ── Piece values ──────────────────────────────────────────────────
        # Pawn is the unit; we can still wiggle it slightly.
        Parameter("pawn",             80,  120,  step=5*s,   dtype=int),
        Parameter("knight",          270,  370,  step=10*s,  dtype=int),
        Parameter("bishop",          280,  380,  step=10*s,  dtype=int),
        Parameter("rook",            450,  560,  step=10*s,  dtype=int),
        Parameter("queen",           820, 1000,  step=20*s,  dtype=int),

        # ── Eval terms ────────────────────────────────────────────────────
        Parameter("mobility",        0.5,  4.0,  step=0.25*s),
        Parameter("bishop_pair",    15.0, 50.0,  step=5.0*s),
        Parameter("doubled_penalty", 8.0, 35.0,  step=3.0*s),
        Parameter("isolated_penalty",5.0, 30.0,  step=3.0*s),
        Parameter("rook_open_file", 10.0, 40.0,  step=5.0*s),
        Parameter("rook_semi_open",  4.0, 25.0,  step=3.0*s),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  WEIGHT INJECTION
# ═══════════════════════════════════════════════════════════════════════════════

def inject_weights(values: dict[str, float]):
    """Patch engine.WEIGHTS with the candidate values (keeps untuned keys)."""
    for k, v in values.items():
        engine.WEIGHTS[k] = v

def reset_weights():
    """Restore engine to its default configuration."""
    engine.WEIGHTS.update(DEFAULT_WEIGHTS)

def reset_engine_state():
    """Clear TT / killers / history so each evaluation starts clean."""
    engine.TRANSPOSITION_TABLE.clear()
    engine._HISTORY.clear()
    engine._KILLERS[:] = [[None, None] for _ in range(64)]


# ═══════════════════════════════════════════════════════════════════════════════
#  FITNESS MODE 1: Self-play
# ═══════════════════════════════════════════════════════════════════════════════

def play_game(depth: int = 3, max_moves: int = 120) -> float:
    """
    Play one game: candidate (WHITE) vs default weights (BLACK).
    Returns 1.0 for win, 0.5 for draw, 0.0 for loss.

    The trick: we swap engine.WEIGHTS between moves so the same module
    serves both sides.
    """
    board = chess.Board()
    candidate_weights = dict(engine.WEIGHTS)  # snapshot current (candidate)

    for move_num in range(max_moves):
        if board.is_game_over():
            break

        if board.turn == chess.WHITE:
            # candidate's turn
            inject_weights(candidate_weights)
        else:
            # reference's turn
            reset_weights()

        reset_engine_state()
        move = engine.get_next_move(board, board.turn, depth=depth)
        board.push(move)

    # score the result from WHITE's (candidate's) perspective
    result = board.result(claim_draw=True)
    if result == "1-0":     return 1.0
    elif result == "0-1":   return 0.0
    else:                   return 0.5


def selfplay_fitness(values: dict[str, float],
                     games: int = 20,
                     depth: int = 2) -> float:
    """
    Fitness = win rate over N games as white, then N as black.
    We alternate colors to remove first-move bias.
    """
    inject_weights(values)
    candidate_w = dict(engine.WEIGHTS)

    total = 0.0
    for i in range(games):
        # even games: candidate is white; odd: candidate is black
        if i % 2 == 0:
            inject_weights(candidate_w)
            score = play_one_game_candidate_white(candidate_w, depth)
            total += score
        else:
            score = play_one_game_candidate_black(candidate_w, depth)
            total += score

    return total / games


def play_one_game_candidate_white(candidate_w: dict, depth: int,
                                   max_moves: int = 100) -> float:
    board = chess.Board()
    for _ in range(max_moves):
        if board.is_game_over(): break
        if board.turn == chess.WHITE:
            engine.WEIGHTS.update(candidate_w)
        else:
            engine.WEIGHTS.update(DEFAULT_WEIGHTS)
        reset_engine_state()
        move = engine.get_next_move(board, board.turn, depth=depth)
        board.push(move)

    r = board.result(claim_draw=True)
    return 1.0 if r == "1-0" else (0.0 if r == "0-1" else 0.5)


def play_one_game_candidate_black(candidate_w: dict, depth: int,
                                   max_moves: int = 100) -> float:
    board = chess.Board()
    for _ in range(max_moves):
        if board.is_game_over(): break
        if board.turn == chess.BLACK:
            engine.WEIGHTS.update(candidate_w)
        else:
            engine.WEIGHTS.update(DEFAULT_WEIGHTS)
        reset_engine_state()
        move = engine.get_next_move(board, board.turn, depth=depth)
        board.push(move)

    r = board.result(claim_draw=True)
    return 1.0 if r == "0-1" else (0.0 if r == "1-0" else 0.5)


# ═══════════════════════════════════════════════════════════════════════════════
#  FITNESS MODE 2: Test suite (EPD positions)
# ═══════════════════════════════════════════════════════════════════════════════

def load_epd_suite(path: str) -> list[tuple[str, str]]:
    """
    Load an EPD file.  Each line: <fen> bm <best_move>;
    Returns list of (fen, best_move_san).
    """
    positions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # standard EPD: fen bm <move>;
            if " bm " in line:
                fen_part, rest = line.split(" bm ", 1)
                bm = rest.split(";")[0].strip()
                # EPD FEN has 4 fields; extend to full FEN
                fields = fen_part.strip().split()
                if len(fields) == 4:
                    fen_part = fen_part.strip() + " 0 1"
                positions.append((fen_part.strip(), bm))
    return positions


def testsuite_fitness(values: dict[str, float],
                      positions: list[tuple[str, str]],
                      depth: int = 3) -> float:
    """Fitness = fraction of positions where the engine finds the best move."""
    inject_weights(values)
    correct = 0
    for fen, expected_san in positions:
        board = chess.Board(fen)
        reset_engine_state()
        move = engine.get_next_move(board, board.turn, depth=depth)
        if board.san(move) == expected_san:
            correct += 1
    reset_weights()
    return correct / max(1, len(positions))


# ═══════════════════════════════════════════════════════════════════════════════
#  FITNESS MODE 3: Texel tuning
# ═══════════════════════════════════════════════════════════════════════════════

def load_texel_data(path: str) -> list[tuple[str, float]]:
    """
    Load a CSV with columns: fen, result
    result is 1.0 (white win), 0.5 (draw), 0.0 (black win).
    """
    data = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fen = row["fen"]
            result = float(row["result"])
            data.append((fen, result))
    return data


def _sigmoid(x: float, K: float = 0.0073) -> float:
    """Map centipawn eval to [0, 1] expected score."""
    return 1.0 / (1.0 + math.exp(-K * x))


def texel_fitness(values: dict[str, float],
                  data: list[tuple[str, float]],
                  K: float = 0.0073) -> float:
    """
    Fitness = -MSE between sigmoid(static_eval) and game result.
    Negative because the optimizer maximises and we want to minimise error.
    """
    inject_weights(values)
    total_error = 0.0
    for fen, result in data:
        board = chess.Board(fen)
        raw_eval = engine.evaluate(board)
        # eval is from white's perspective
        predicted = _sigmoid(raw_eval, K)
        total_error += (predicted - result) ** 2
    reset_weights()
    return -(total_error / max(1, len(data)))


# ═══════════════════════════════════════════════════════════════════════════════
#  QUICK DEMO (built-in test positions — no external files needed)
# ═══════════════════════════════════════════════════════════════════════════════

# A handful of well-known tactical/positional test positions with known best moves.
BUILTIN_POSITIONS = [
    # Lucena position — Rook endgame, white to play and win
    ("1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1", "Rc4"),
    # Simple fork: knight forks king and rook
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1", "Qxf7+"),
    # Back rank mate threat
    ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "Re8+"),
    # Pin the knight to the queen
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "e5"),
    # Starting position (test opening preference)
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e4"),
]


def demo_testsuite_fitness(values: dict[str, float], depth: int = 3) -> float:
    """Use the built-in positions as a mini test suite."""
    inject_weights(values)
    correct = 0
    for fen, expected_san in BUILTIN_POSITIONS:
        board = chess.Board(fen)
        reset_engine_state()
        try:
            move = engine.get_next_move(board, board.turn, depth=depth)
            if board.san(move) == expected_san:
                correct += 1
        except Exception:
            pass
    reset_weights()
    return correct / len(BUILTIN_POSITIONS)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Tune chess engine eval weights")
    ap.add_argument("--mode", choices=["self_play", "test_suite", "texel", "demo"],
                    default="demo", help="Fitness function to use")
    ap.add_argument("--strategy", choices=["genetic", "hillclimb", "random", "grid", "sweep"],
                    default="genetic")
    ap.add_argument("--depth", type=int, default=2,
                    help="Search depth for engine during evaluation (lower = faster)")
    ap.add_argument("--games", type=int, default=10,
                    help="Games per evaluation (self_play mode)")
    ap.add_argument("--epd-file", type=str, default=None,
                    help="Path to EPD test suite (test_suite mode)")
    ap.add_argument("--texel-file", type=str, default=None,
                    help="Path to CSV with fen,result columns (texel mode)")
    ap.add_argument("--pop", type=int, default=20,
                    help="Population size (genetic)")
    ap.add_argument("--gen", type=int, default=15,
                    help="Generations (genetic)")
    ap.add_argument("--coarse", action="store_true",
                    help="Use wider step sizes for fast first pass")
    args = ap.parse_args()

    params = build_params(coarse=args.coarse)

    # ── Build the fitness function ────────────────────────────────────────
    if args.mode == "self_play":
        games, depth = args.games, args.depth
        def fitness(values):
            return selfplay_fitness(values, games=games, depth=depth)

    elif args.mode == "test_suite":
        if not args.epd_file:
            print("Error: --epd-file required for test_suite mode", file=sys.stderr)
            sys.exit(1)
        positions = load_epd_suite(args.epd_file)
        print(f"Loaded {len(positions)} test positions from {args.epd_file}")
        depth = args.depth
        def fitness(values):
            return testsuite_fitness(values, positions, depth=depth)

    elif args.mode == "texel":
        if not args.texel_file:
            print("Error: --texel-file required for texel mode", file=sys.stderr)
            sys.exit(1)
        data = load_texel_data(args.texel_file)
        print(f"Loaded {len(data)} positions from {args.texel_file}")
        def fitness(values):
            return texel_fitness(values, data)

    else:  # demo
        depth = args.depth
        print(f"Demo mode: using {len(BUILTIN_POSITIONS)} built-in test positions at depth={depth}")
        def fitness(values):
            return demo_testsuite_fitness(values, depth=depth)

    # ── Build strategy kwargs ─────────────────────────────────────────────
    strat_kwargs = {}
    if args.strategy == "genetic":
        strat_kwargs = {"population_size": args.pop, "generations": args.gen, "mutation_rate": 0.2}
    elif args.strategy == "hillclimb":
        strat_kwargs = {"iterations": args.pop * args.gen, "restarts": 3}
    elif args.strategy == "random":
        strat_kwargs = {"n_samples": args.pop * args.gen}

    # ── Estimate total time ──────────────────────────────────────────────
    # Benchmark one fitness call, then multiply by expected total evaluations.
    print("Benchmarking one fitness evaluation...", end=" ", flush=True)
    _bench_vals = {p.name: p.clamp((p.min_val + p.max_val) / 2) for p in params}
    _t0 = time.perf_counter()
    fitness(_bench_vals)
    cost_per_eval = time.perf_counter() - _t0
    print(f"{cost_per_eval:.2f}s")

    # Estimate total evaluations for each strategy
    if args.strategy == "genetic":
        total_evals = args.pop * args.gen
    elif args.strategy == "hillclimb":
        total_evals = (args.pop * args.gen) * 3 + 3  # iterations*restarts + starts
    elif args.strategy == "random":
        total_evals = args.pop * args.gen
    elif args.strategy == "grid":
        total_evals = 1
        for p in params:
            total_evals *= p.num_steps
    elif args.strategy == "sweep":
        total_evals = sum(p.num_steps for p in params)
    else:
        total_evals = args.pop * args.gen

    eta_secs = total_evals * cost_per_eval
    if eta_secs < 60:
        eta_str = f"{eta_secs:.0f}s"
    elif eta_secs < 3600:
        eta_str = f"{int(eta_secs // 60)}m {int(eta_secs % 60)}s"
    else:
        h, m = int(eta_secs // 3600), int((eta_secs % 3600) // 60)
        eta_str = f"{h}h {m}m"
    print(f"\n  ~{total_evals} evaluations × {cost_per_eval:.2f}s each ≈ {eta_str}")
    if args.strategy == "grid" and total_evals > 50000:
        print(f"  WARNING: grid search has {total_evals} combos — consider genetic instead")
    print()

    # ── Run ───────────────────────────────────────────────────────────────
    print(f"Strategy: {args.strategy}  |  Mode: {args.mode}  |  Depth: {args.depth}")
    print(f"Parameters: {len(params)}  |  Tuning...\n")

    opt = Optimizer(params, fitness, strategy=args.strategy, verbose=True, **strat_kwargs)
    report = opt.run()

    # restore default weights before printing
    reset_weights()
    report.show(top_n=10)

    # ── Print a copy-pasteable WEIGHTS dict ───────────────────────────────
    print("  Copy-paste into your engine:")
    print("  WEIGHTS = {")
    merged = {**DEFAULT_WEIGHTS, **report.best.values}
    for k, v in merged.items():
        print(f'      "{k}": {v!r},')
    print("  }")


if __name__ == "__main__":
    main()