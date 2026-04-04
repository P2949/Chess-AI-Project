"""
tune_engine.py — Tune team_goraieb eval weights with the generic optimizer.

Modes:
  self_play     — Candidate vs default-weights opponent (win rate)
  vs_stockfish  — Candidate vs Stockfish; fitness = avg SF eval of positions
                  reached after candidate moves (rewards good positional play)
  test_suite    — Fraction of EPD best-moves found
  texel         — MSE of sigmoid(static_eval) vs game results
  demo          — Built-in 5-position test

Thread-safe: each worker gets its own engine + Stockfish instances
via threading.local(), zero shared mutable state.

Run:
    python tune_engine.py --mode vs_stockfish --games 8 --depth 2
    python tune_engine.py --mode vs_stockfish --strategy genetic --pop 30 --gen 40
    python tune_engine.py --mode self_play --games 20 --depth 2
"""

from __future__ import annotations

import argparse
import atexit
import copy
import csv
import importlib
import importlib.util
import math
import os
import random
import shutil
import sys
import threading
import time

import chess
import chess.engine
import chess.polyglot

import team_goraieb as _engine_template
from optimizer import Parameter, Optimizer

DEFAULT_WEIGHTS = dict(_engine_template.WEIGHTS)

N_WORKERS = os.cpu_count() or 1

# ── Stockfish configuration ──────────────────────────────────────────────────
SF_PATH = shutil.which("stockfish") or "stockfish"

# Opponent strength (plays the moves against the candidate)
SF_PLAY_DEPTH = 8        # FIX #2: was 20, way too slow for optimizer
SF_PLAY_SKILL = 20       # skill level 0-20

# Evaluation depth (judges positions after candidate moves)
SF_EVAL_DEPTH = 10       # FIX #1: was 200 (!), would never finish

# Game settings
SF_DECISIVE_CP  = 900
SF_DECISIVE_N   = 4
SF_MAX_MOVES    = 150
SF_RANDOM_OPEN  = 0


# ── Opening positions for game diversity ──────────────────────────────────────
OPENING_POSITIONS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkbnr/pp1ppppp/8/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
    "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
    "rnbqkb1r/pppppp1p/5np1/8/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3",
    "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1",
    "rnbqkb1r/ppp1pppp/3p1n2/8/3PP3/8/PPP2PPP/RNBQKBNR w KQkq - 0 3",
    "rnbqkbnr/pp2pppp/2p5/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
]


_thread_local = threading.local()

# Track all SF engines for cleanup
_all_sf_engines: list[chess.engine.SimpleEngine] = []
_all_sf_lock = threading.Lock()


def _register_sf(engine: chess.engine.SimpleEngine):
    with _all_sf_lock:
        _all_sf_engines.append(engine)


@atexit.register
def _cleanup_all_sf():
    with _all_sf_lock:
        for eng in _all_sf_engines:
            try:
                eng.quit()
            except Exception:
                pass
        _all_sf_engines.clear()


def _make_engine():
    spec = importlib.util.find_spec("team_goraieb")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_engines():
    if not hasattr(_thread_local, "eng_candidate"):
        _thread_local.eng_candidate = _make_engine()
        _thread_local.eng_opponent = _make_engine()
    return _thread_local.eng_candidate, _thread_local.eng_opponent


def _get_stockfish():
    if not hasattr(_thread_local, "sf_play"):
        try:
            sf_play = chess.engine.SimpleEngine.popen_uci(SF_PATH)
            sf_play.configure({"Threads": 1, "Hash": 64, "Skill Level": SF_PLAY_SKILL})
            _register_sf(sf_play)

            sf_eval = chess.engine.SimpleEngine.popen_uci(SF_PATH)
            sf_eval.configure({"Threads": 1, "Hash": 32})
            _register_sf(sf_eval)

            _thread_local.sf_play = sf_play
            _thread_local.sf_eval = sf_eval
        except Exception as e:
            print(f"[error] Failed to start Stockfish: {e}", file=sys.stderr)
            print(f"  Looked for: {SF_PATH}", file=sys.stderr)
            raise

    return _thread_local.sf_play, _thread_local.sf_eval


def _reset_engine(eng) -> None:
    """Clear TT / killers / history / EVAL_CACHE between games."""
    eng.TRANSPOSITION_TABLE.clear()
    eng._HISTORY.clear()
    eng._KILLERS[:] = [[None, None] for _ in range(64)]
    # FIX #4: Clear EVAL_CACHE — stale cache from previous WEIGHTS poisons results
    if hasattr(eng, 'EVAL_CACHE'):
        eng.EVAL_CACHE.clear()


def _material_balance(board) -> int:
    values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
              chess.ROOK: 500, chess.QUEEN: 900}
    score = 0
    for pt, val in values.items():
        score += len(board.pieces(pt, chess.WHITE)) * val
        score -= len(board.pieces(pt, chess.BLACK)) * val
    return score


def build_params(coarse: bool = False) -> list[Parameter]:
    s = 2 if coarse else 1

    return [
        # ── Piece values ──────────────────────────────────────────────────
        Parameter("pawn",             80,  120,  step=5*s,   dtype=int),
        Parameter("knight",          270,  370,  step=10*s,  dtype=int),
        Parameter("bishop",          280,  380,  step=10*s,  dtype=int),
        Parameter("rook",            450,  560,  step=10*s,  dtype=int),
        Parameter("queen",           820, 1000,  step=20*s,  dtype=int),

        # ── Core eval terms ───────────────────────────────────────────────
        Parameter("mobility",        0.5,  4.0,  step=0.25*s),
        Parameter("bishop_pair",    15.0, 50.0,  step=5.0*s),
        Parameter("doubled_penalty", 8.0, 35.0,  step=3.0*s),
        Parameter("isolated_penalty",5.0, 30.0,  step=3.0*s),
        Parameter("backward_penalty",0.0, 25.0,  step=3.0*s),
        Parameter("rook_open_file", 10.0, 40.0,  step=5.0*s),
        Parameter("rook_semi_open",  4.0, 25.0,  step=3.0*s),

        # ── New terms ─────────────────────────────────────────────────────
        Parameter("connected_rooks",  0.0, 25.0, step=5.0*s),
        Parameter("king_shield",      0.0, 20.0, step=3.0*s),
        Parameter("king_shield_miss", 0.0, 30.0, step=5.0*s),
        Parameter("knight_outpost",   5.0, 40.0, step=5.0*s),
        Parameter("king_tropism",     0.0,  5.0, step=0.5*s),
        Parameter("tempo",            0.0, 25.0, step=5.0*s),

        # FIX #5: New WEIGHTS keys that were missing
        Parameter("hanging_piece",   20.0, 90.0, step=10.0*s),
        Parameter("king_attack",      0.0, 30.0, step=3.0*s),
        Parameter("pin_penalty",     10.0, 50.0, step=5.0*s),
        Parameter("center_control",   0.0, 20.0, step=2.0*s),
        Parameter("uncastled_king",  15.0, 80.0, step=5.0*s),
        Parameter("rook_doubled_file", 5.0, 40.0, step=5.0*s),

        # ── Development ───────────────────────────────────────────────────
        Parameter("early_queen_pen", 10.0, 60.0, step=5.0*s),
        Parameter("development",      0.0, 25.0, step=3.0*s),

        # ── NN blend ──────────────────────────────────────────────────────
        Parameter("nn_weight",       0.0,  0.5,  step=0.05*s),
        Parameter("policy_weight",   0.0,  0.1,  step=0.01*s),
        Parameter("nn_phase_boost",  0.5,  2.0,  step=0.25*s),
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  VS STOCKFISH FITNESS
# ══════════════════════════════════════════════════════════════════════════════

def _sf_eval_cp(sf_eval: chess.engine.SimpleEngine, board: chess.Board,
                depth: int) -> float:
    try:
        info = sf_eval.analyse(board, chess.engine.Limit(depth=depth))
    except chess.engine.EngineTerminatedError:
        raise
    except Exception:
        return 0.0

    score = info["score"].white()
    if score.is_mate():
        m = score.mate()
        return 15000.0 if m > 0 else -15000.0
    return float(score.score())


def _play_one_vs_stockfish(candidate_w: dict, candidate_is_white: bool,
                           depth: int, opening_fen: str | None = None,
                           ) -> tuple[list[float], str]:
    eng_cand, _ = _get_engines()
    sf_play, sf_eval = _get_stockfish()

    eng_cand.WEIGHTS.update(candidate_w)
    _reset_engine(eng_cand)

    board = chess.Board(opening_fen) if opening_fen else chess.Board()
    sign = 1.0 if candidate_is_white else -1.0

    evals = []
    decisive_streak = 0

    for move_num in range(SF_MAX_MOVES):
        if board.is_game_over():
            break

        is_candidate_turn = (board.turn == chess.WHITE) == candidate_is_white

        if is_candidate_turn:
            try:
                move = eng_cand.get_next_move(board, board.turn, depth=depth)
            except Exception:
                evals.append(-10000.0)
                break
            board.push(move)

            if board.is_game_over():
                r = board.result(claim_draw=True)
                if r == "1-0":
                    evals.append(15000.0 * sign)
                elif r == "0-1":
                    evals.append(-15000.0 * sign)
                else:
                    evals.append(0.0)
                break

            cp = _sf_eval_cp(sf_eval, board, SF_EVAL_DEPTH) * sign
            evals.append(cp)

            if abs(cp) >= SF_DECISIVE_CP:
                decisive_streak += 1
            else:
                decisive_streak = 0

            if decisive_streak >= SF_DECISIVE_N:
                break
        else:
            try:
                result = sf_play.play(board, chess.engine.Limit(depth=SF_PLAY_DEPTH))
                board.push(result.move)
            except chess.engine.EngineTerminatedError:
                try:
                    sf_play = chess.engine.SimpleEngine.popen_uci(SF_PATH)
                    sf_play.configure({"Threads": 1, "Hash": 64,
                                       "Skill Level": SF_PLAY_SKILL})
                    _register_sf(sf_play)
                    _thread_local.sf_play = sf_play
                except Exception:
                    break

    if board.is_game_over():
        r = board.result(claim_draw=True)
        if candidate_is_white:
            outcome = "win" if r == "1-0" else ("loss" if r == "0-1" else "draw")
        else:
            outcome = "win" if r == "0-1" else ("loss" if r == "1-0" else "draw")
    elif decisive_streak >= SF_DECISIVE_N:
        last_cp = evals[-1] if evals else 0
        outcome = "adjudicated_win" if last_cp > 0 else "adjudicated_loss"
    else:
        outcome = "draw"

    return evals, outcome


def vs_stockfish_fitness(values: dict[str, float],
                         games: int = 8,
                         depth: int = 2) -> float:
    candidate_w = {**DEFAULT_WEIGHTS, **values}

    all_evals = []
    wins = 0
    losses = 0

    for i in range(games):
        candidate_is_white = (i % 2 == 0)
        opening = OPENING_POSITIONS[i % len(OPENING_POSITIONS)]

        evals, outcome = _play_one_vs_stockfish(
            candidate_w, candidate_is_white, depth, opening_fen=opening)

        all_evals.extend(evals)

        if "win" in outcome:
            wins += 1
        elif "loss" in outcome:
            losses += 1

    if not all_evals:
        return -10000.0

    avg_eval = sum(all_evals) / len(all_evals)
    outcome_bonus = (wins - losses) * 30.0

    return avg_eval + outcome_bonus


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-PLAY FITNESS
# ══════════════════════════════════════════════════════════════════════════════

ADJUDICATE_MATERIAL = 800
ADJUDICATE_MOVES    = 5


def _play_one_game(candidate_w: dict, candidate_is_white: bool,
                   depth: int, max_moves: int = 200,
                   opening_fen: str | None = None) -> float:
    eng_cand, eng_opp = _get_engines()

    eng_cand.WEIGHTS.update(candidate_w)
    eng_opp.WEIGHTS.update(DEFAULT_WEIGHTS)

    _reset_engine(eng_cand)
    _reset_engine(eng_opp)

    board = chess.Board(opening_fen) if opening_fen else chess.Board()
    adjudicate_count = 0

    for move_num in range(max_moves):
        if board.is_game_over():
            break

        is_candidate_turn = (board.turn == chess.WHITE) == candidate_is_white

        if is_candidate_turn:
            move = eng_cand.get_next_move(board, board.turn, depth=depth)
        else:
            move = eng_opp.get_next_move(board, board.turn, depth=depth)

        board.push(move)

        mat = _material_balance(board)
        if abs(mat) >= ADJUDICATE_MATERIAL:
            adjudicate_count += 1
        else:
            adjudicate_count = 0

        if adjudicate_count >= ADJUDICATE_MOVES:
            white_winning = mat > 0
            if candidate_is_white:
                return 1.0 if white_winning else 0.0
            else:
                return 0.0 if white_winning else 1.0

    r = board.result(claim_draw=True)
    if candidate_is_white:
        return 1.0 if r == "1-0" else (0.0 if r == "0-1" else 0.5)
    else:
        return 1.0 if r == "0-1" else (0.0 if r == "1-0" else 0.5)


def selfplay_fitness(values: dict[str, float],
                     games: int = 20,
                     depth: int = 2) -> float:
    candidate_w = {**DEFAULT_WEIGHTS, **values}
    total = 0.0
    for i in range(games):
        candidate_is_white = (i % 2 == 0)
        opening = OPENING_POSITIONS[i % len(OPENING_POSITIONS)]
        total += _play_one_game(candidate_w, candidate_is_white, depth,
                                opening_fen=opening)
    return total / games


# ══════════════════════════════════════════════════════════════════════════════
#  TEST SUITE / TEXEL FITNESS
# ══════════════════════════════════════════════════════════════════════════════

def load_epd_suite(path: str) -> list[tuple[str, str]]:
    positions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if " bm " in line:
                fen_part, rest = line.split(" bm ", 1)
                bm = rest.split(";")[0].strip()
                fields = fen_part.strip().split()
                if len(fields) == 4:
                    fen_part = fen_part.strip() + " 0 1"
                positions.append((fen_part.strip(), bm))
    return positions


def testsuite_fitness(values: dict[str, float],
                      positions: list[tuple[str, str]],
                      depth: int = 3) -> float:
    eng_cand, _ = _get_engines()
    eng_cand.WEIGHTS.update({**DEFAULT_WEIGHTS, **values})
    _reset_engine(eng_cand)  # FIX: clear cache for new WEIGHTS
    correct = 0
    for fen, expected_san in positions:
        board = chess.Board(fen)
        _reset_engine(eng_cand)
        move = eng_cand.get_next_move(board, board.turn, depth=depth)
        if board.san(move) == expected_san:
            correct += 1
    return correct / max(1, len(positions))


def load_texel_data(path: str) -> list[tuple[str, float]]:
    data = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append((row["fen"], float(row["result"])))
    return data


def _sigmoid(x: float, K: float = 0.0073) -> float:
    return 1.0 / (1.0 + math.exp(-K * x))


def texel_fitness(values: dict[str, float],
                  data: list[tuple[str, float]],
                  K: float = 0.0073) -> float:
    eng_cand, _ = _get_engines()
    eng_cand.WEIGHTS.update({**DEFAULT_WEIGHTS, **values})
    _reset_engine(eng_cand)  # FIX: clear cache for new WEIGHTS
    total_error = 0.0
    for fen, result in data:
        board = chess.Board(fen)
        raw_eval = eng_cand.evaluate(board)
        predicted = _sigmoid(raw_eval, K)
        total_error += (predicted - result) ** 2
    return -(total_error / max(1, len(data)))


BUILTIN_POSITIONS = [
    ("1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1", "Rc4"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1", "Qxf7+"),
    ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "Re8+"),
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "e5"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e4"),
]


def demo_testsuite_fitness(values: dict[str, float], depth: int = 3) -> float:
    eng_cand, _ = _get_engines()
    eng_cand.WEIGHTS.update({**DEFAULT_WEIGHTS, **values})
    correct = 0
    for fen, expected_san in BUILTIN_POSITIONS:
        board = chess.Board(fen)
        _reset_engine(eng_cand)
        try:
            move = eng_cand.get_next_move(board, board.turn, depth=depth)
            if board.san(move) == expected_san:
                correct += 1
        except Exception:
            pass
    return correct / len(BUILTIN_POSITIONS)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global SF_PLAY_DEPTH, SF_EVAL_DEPTH, SF_PLAY_SKILL

    ap = argparse.ArgumentParser(description="Tune chess engine eval weights")
    ap.add_argument("--mode",
                    choices=["self_play", "vs_stockfish", "test_suite", "texel", "demo"],
                    default="demo", help="Fitness function to use")
    ap.add_argument("--strategy",
                    choices=["genetic", "hillclimb", "random", "grid", "sweep"],
                    default="genetic")
    ap.add_argument("--depth", type=int, default=2,
                    help="Search depth for candidate engine (lower = faster)")
    ap.add_argument("--games", type=int, default=8,
                    help="Games per fitness evaluation")
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
    ap.add_argument("--workers", type=int, default=N_WORKERS,
                    help=f"Number of threads (default: {N_WORKERS} = all CPUs)")

    # vs_stockfish specific
    ap.add_argument("--sf-play-depth", type=int, default=None,
                    help=f"SF opponent depth (default: {SF_PLAY_DEPTH})")
    ap.add_argument("--sf-eval-depth", type=int, default=None,
                    help=f"SF position eval depth (default: {SF_EVAL_DEPTH})")
    ap.add_argument("--sf-skill", type=int, default=None,
                    help=f"SF skill level 0-20 (default: {SF_PLAY_SKILL})")

    args = ap.parse_args()

    if args.sf_play_depth is not None:
        SF_PLAY_DEPTH = args.sf_play_depth
    if args.sf_eval_depth is not None:
        SF_EVAL_DEPTH = args.sf_eval_depth
    if args.sf_skill is not None:
        SF_PLAY_SKILL = args.sf_skill

    n_workers = args.workers
    params = build_params(coarse=args.coarse)

    if args.mode == "vs_stockfish":
        games, depth = args.games, args.depth
        print(f"═══ vs_stockfish mode ═══")
        print(f"  Candidate depth : {depth}")
        print(f"  SF play depth   : {SF_PLAY_DEPTH}  (skill {SF_PLAY_SKILL})")
        print(f"  SF eval depth   : {SF_EVAL_DEPTH}")
        print(f"  Games/eval      : {games}")
        print(f"  Decisive        : >{SF_DECISIVE_CP}cp × {SF_DECISIVE_N} moves")
        print(f"  Max moves/game  : {SF_MAX_MOVES}")
        print()

        def fitness(values):
            return vs_stockfish_fitness(values, games=games, depth=depth)

    elif args.mode == "self_play":
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
        print(f"Demo mode: {len(BUILTIN_POSITIONS)} built-in test positions at depth={depth}")
        def fitness(values):
            return demo_testsuite_fitness(values, depth=depth)

    strat_kwargs = {}
    if args.strategy == "genetic":
        strat_kwargs = {"population_size": args.pop, "generations": args.gen,
                        "mutation_rate": 0.2}
    elif args.strategy == "hillclimb":
        strat_kwargs = {"iterations": args.pop * args.gen, "restarts": 3}
    elif args.strategy == "random":
        strat_kwargs = {"n_samples": args.pop * args.gen}

    from concurrent.futures import ThreadPoolExecutor

    if args.strategy in ("genetic", "random", "grid", "sweep"):
        bench_batch = min(n_workers, args.pop)
    else:
        bench_batch = min(n_workers, 3)

    print(f"Threads: {n_workers}")
    print(f"Benchmarking {bench_batch} parallel evaluations...", end=" ", flush=True)

    _bench_candidates = []
    for i in range(bench_batch):
        vals = {}
        for p in params:
            frac = (i + 1) / (bench_batch + 1)
            vals[p.name] = p.clamp(p.min_val + frac * (p.max_val - p.min_val))
        _bench_candidates.append(vals)

    _t0 = time.perf_counter()
    if bench_batch <= 1:
        fitness(_bench_candidates[0])
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(fitness, _bench_candidates))
    batch_wall_time = time.perf_counter() - _t0
    cost_per_eval = batch_wall_time / bench_batch
    print(f"{batch_wall_time:.2f}s  ({cost_per_eval:.2f}s/eval effective)")

    if args.strategy == "genetic":
        total_evals = args.pop * args.gen
    elif args.strategy == "hillclimb":
        total_evals = (args.pop * args.gen) * 3 + 3
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

    total_batches = math.ceil(total_evals / max(1, bench_batch))
    eta_secs = total_batches * batch_wall_time

    def _fmt_time(secs):
        if secs < 60:   return f"{secs:.0f}s"
        if secs < 3600: return f"{int(secs // 60)}m {int(secs % 60)}s"
        h, m = int(secs // 3600), int((secs % 3600) // 60)
        return f"{h}h {m}m"

    print(f"\n  ~{total_evals} evaluations in ~{total_batches} batches of {bench_batch}")
    print(f"  Estimated wall time: ~{_fmt_time(eta_secs)}")
    if args.strategy == "grid" and total_evals > 50000:
        print(f"  WARNING: grid search has {total_evals} combos — consider genetic instead")
    print()

    print(f"Strategy: {args.strategy}  |  Mode: {args.mode}  |  Depth: {args.depth}")
    print(f"Parameters: {len(params)}  |  Workers: {n_workers}  |  Tuning...\n")

    opt = Optimizer(params, fitness, strategy=args.strategy, verbose=True,
                    n_workers=n_workers, **strat_kwargs)
    report = opt.run()

    report.show(top_n=10)

    print("  Copy-paste into your engine:")
    print("  WEIGHTS = {")
    merged = {**DEFAULT_WEIGHTS, **report.best.values}
    for k, v in merged.items():
        print(f'      "{k}": {v!r},')
    print("  }")


if __name__ == "__main__":
    main()