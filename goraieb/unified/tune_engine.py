"""
tune_engine.py — Tune team_goraieb eval weights with the generic optimizer.

Modes:
  move_quality  — NEW: Measures how much each candidate move worsens the position
                  (delta-based). Works against STRONG SF. Fitness ~ 0 is perfect.
  self_play     — Candidate vs default-weights opponent (win rate).
  vs_stockfish  — Candidate vs weak Stockfish (win rate).
  test_suite    — Fraction of EPD best-moves found
  texel         — MSE of sigmoid(static_eval) vs game results
  demo          — Built-in 5-position test

Run:
    python tune_engine.py --mode move_quality --depth 3 --games 10 --pop 24 --gen 30
    python tune_engine.py --mode self_play --depth 3 --games 12 --pop 24 --gen 30
"""

from __future__ import annotations
import argparse, atexit, csv, importlib, importlib.util, math, os, random
import shutil, sys, threading, time
import chess, chess.engine, chess.polyglot
import team_goraieb as _engine_template
from optimizer import Parameter, Optimizer

DEFAULT_WEIGHTS = dict(_engine_template.WEIGHTS)
N_WORKERS = os.cpu_count() or 1
SF_PATH = shutil.which("stockfish") or "stockfish"

SF_PLAY_DEPTH = 8;  SF_PLAY_SKILL = 20;  SF_EVAL_DEPTH = 10
SF_DECISIVE_CP = 1400;  SF_DECISIVE_N = 12;  SF_MAX_MOVES = 200
ADJUDICATE_MATERIAL = 800;  ADJUDICATE_MOVES = 5

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
_all_sf_engines = [];  _all_sf_lock = threading.Lock()

def _register_sf(e):
    with _all_sf_lock: _all_sf_engines.append(e)

@atexit.register
def _cleanup_all_sf():
    with _all_sf_lock:
        for e in _all_sf_engines:
            try: e.quit()
            except: pass
        _all_sf_engines.clear()

def _make_engine():
    spec = importlib.util.find_spec("team_goraieb")
    mod = importlib.util.module_from_spec(spec);  spec.loader.exec_module(mod)
    return mod

def _get_engines():
    if not hasattr(_thread_local, "eng_candidate"):
        _thread_local.eng_candidate = _make_engine()
        _thread_local.eng_opponent = _make_engine()
    return _thread_local.eng_candidate, _thread_local.eng_opponent

def _get_stockfish():
    if not hasattr(_thread_local, "sf_play"):
        sf_play = chess.engine.SimpleEngine.popen_uci(SF_PATH)
        sf_play.configure({"Threads": 1, "Hash": 64, "Skill Level": SF_PLAY_SKILL})
        _register_sf(sf_play)
        sf_eval = chess.engine.SimpleEngine.popen_uci(SF_PATH)
        sf_eval.configure({"Threads": 1, "Hash": 32})
        _register_sf(sf_eval)
        _thread_local.sf_play = sf_play;  _thread_local.sf_eval = sf_eval
    return _thread_local.sf_play, _thread_local.sf_eval

def _reset_engine(eng):
    eng.TRANSPOSITION_TABLE.clear();  eng._HISTORY.clear()
    eng._KILLERS[:] = [[None, None] for _ in range(64)]
    if hasattr(eng, 'EVAL_CACHE'): eng.EVAL_CACHE.clear()

def _material_balance(board):
    vals = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900}
    s = 0
    for pt, v in vals.items():
        s += len(board.pieces(pt, chess.WHITE)) * v
        s -= len(board.pieces(pt, chess.BLACK)) * v
    return s

def _sf_eval_cp(sf_eval, board, depth):
    try: info = sf_eval.analyse(board, chess.engine.Limit(depth=depth))
    except chess.engine.EngineTerminatedError: raise
    except: return 0.0
    sc = info["score"].white()
    if sc.is_mate(): return 15000.0 if sc.mate() > 0 else -15000.0
    return float(sc.score())


def build_params(coarse=False):
    s = 2 if coarse else 1
    return [
        Parameter("pawn",             80,  120,  step=5*s,   dtype=int),
        Parameter("knight",          270,  370,  step=10*s,  dtype=int),
        Parameter("bishop",          280,  380,  step=10*s,  dtype=int),
        Parameter("rook",            450,  560,  step=10*s,  dtype=int),
        Parameter("queen",           820, 1000,  step=20*s,  dtype=int),
        Parameter("mobility",        0.5,  4.0,  step=0.25*s),
        Parameter("bishop_pair",    15.0, 50.0,  step=5.0*s),
        Parameter("doubled_penalty", 8.0, 35.0,  step=3.0*s),
        Parameter("isolated_penalty",5.0, 30.0,  step=3.0*s),
        Parameter("rook_open_file", 10.0, 40.0,  step=5.0*s),
        Parameter("rook_semi_open",  4.0, 25.0,  step=3.0*s),
        Parameter("rook_doubled_file", 5.0, 40.0, step=5.0*s),
        Parameter("hanging_piece",   20.0, 90.0, step=10.0*s),
        Parameter("pin_penalty",     10.0, 50.0, step=5.0*s),
        Parameter("center_control",   0.0, 20.0, step=2.0*s),
        Parameter("king_shield",      0.0, 20.0, step=3.0*s),
        Parameter("king_attack",      0.0, 30.0, step=3.0*s),
        Parameter("uncastled_king",  15.0, 80.0, step=5.0*s),
        Parameter("early_queen_pen", 10.0, 60.0, step=5.0*s),
        Parameter("development",      0.0, 25.0, step=3.0*s),
        Parameter("nn_weight",       0.15, 0.45, step=0.05*s),
        Parameter("policy_weight",       0.01, 0.5, step=0.01*s),
    ]  # 21 params


# ══════════════════════════════════════════════════════════════════════════════
#  MOVE QUALITY FITNESS — delta-based, works against STRONG SF
# ══════════════════════════════════════════════════════════════════════════════

def _play_one_move_quality(candidate_w, candidate_is_white, depth,
                           opening_fen=None):
    """
    Play one game measuring move quality via eval deltas.

    For each candidate move:
      E_before = SF eval before candidate moves (candidate's perspective)
      E_after  = SF eval after candidate moves  (candidate's perspective)
      delta    = E_after - E_before

    Perfect move: delta ~ 0.  Blunder: delta << 0.
    """
    eng_cand, _ = _get_engines()
    sf_play, sf_eval = _get_stockfish()
    eng_cand.WEIGHTS.update(candidate_w)
    _reset_engine(eng_cand)

    board = chess.Board(opening_fen) if opening_fen else chess.Board()
    sign = 1.0 if candidate_is_white else -1.0
    deltas = []
    decisive_streak = 0

    eval_before = _sf_eval_cp(sf_eval, board, SF_EVAL_DEPTH) * sign

    for _ in range(SF_MAX_MOVES):
        if board.is_game_over():
            break

        is_candidate_turn = (board.turn == chess.WHITE) == candidate_is_white

        if is_candidate_turn:
            try:
                move = eng_cand.get_next_move(board, board.turn, depth=depth)
            except Exception:
                deltas.append(-500.0)
                break

            board.push(move)

            if board.is_game_over():
                r = board.result(claim_draw=True)
                if (r == "1-0" and candidate_is_white) or \
                   (r == "0-1" and not candidate_is_white):
                    deltas.append(max(0.0, 500.0 - eval_before))
                elif r in ("1/2-1/2", "*"):
                    deltas.append(-abs(eval_before) * 0.1)
                else:
                    deltas.append(-500.0)
                break

            eval_after = _sf_eval_cp(sf_eval, board, SF_EVAL_DEPTH) * sign
            delta = eval_after - eval_before
            deltas.append(delta)

            if eval_after <= -SF_DECISIVE_CP:
                decisive_streak += 1
            else:
                decisive_streak = 0
            if decisive_streak >= SF_DECISIVE_N:
                break
        else:
            # SF's turn
            try:
                result = sf_play.play(board, chess.engine.Limit(depth=SF_PLAY_DEPTH))
                board.push(result.move)
            except chess.engine.EngineTerminatedError:
                try:
                    sf_play = chess.engine.SimpleEngine.popen_uci(SF_PATH)
                    sf_play.configure({"Threads": 1, "Hash": 64,
                                       "Skill Level": SF_PLAY_SKILL})
                    _register_sf(sf_play);  _thread_local.sf_play = sf_play
                    result = sf_play.play(board, chess.engine.Limit(depth=SF_PLAY_DEPTH))
                    board.push(result.move)
                except Exception:
                    break

            if board.is_game_over():
                break
            eval_before = _sf_eval_cp(sf_eval, board, SF_EVAL_DEPTH) * sign

    return deltas


def move_quality_fitness(values, games=10, depth=3):
    """
    Fitness = average eval delta per move.
      ~ 0    = perfect play
      ~ -30  = slight inaccuracies
      ~ -100 = frequent blunders
    """
    candidate_w = {**DEFAULT_WEIGHTS, **values}
    all_deltas = []
    for i in range(games):
        candidate_is_white = (i % 2 == 0)
        opening = OPENING_POSITIONS[i % len(OPENING_POSITIONS)]
        deltas = _play_one_move_quality(
            candidate_w, candidate_is_white, depth, opening_fen=opening)
        all_deltas.extend(deltas)

    if not all_deltas:
        return -10000.0

    clamped = [max(-500.0, min(50.0, d)) for d in all_deltas]
    return sum(clamped) / len(clamped)


# ══════════════════════════════════════════════════════════════════════════════
#  VS STOCKFISH — win rate against weak SF
# ══════════════════════════════════════════════════════════════════════════════

def _play_one_vs_stockfish(candidate_w, candidate_is_white, depth,
                           opening_fen=None):
    eng_cand, _ = _get_engines()
    sf_play, _ = _get_stockfish()
    eng_cand.WEIGHTS.update(candidate_w)
    _reset_engine(eng_cand)

    board = chess.Board(opening_fen) if opening_fen else chess.Board()
    adj = 0
    for _ in range(SF_MAX_MOVES):
        if board.is_game_over(): break
        if (board.turn == chess.WHITE) == candidate_is_white:
            try: move = eng_cand.get_next_move(board, board.turn, depth=depth)
            except: return 0.0
            board.push(move)
        else:
            try:
                r = sf_play.play(board, chess.engine.Limit(depth=SF_PLAY_DEPTH))
                board.push(r.move)
            except chess.engine.EngineTerminatedError:
                try:
                    sf_play = chess.engine.SimpleEngine.popen_uci(SF_PATH)
                    sf_play.configure({"Threads":1,"Hash":64,"Skill Level":SF_PLAY_SKILL})
                    _register_sf(sf_play); _thread_local.sf_play = sf_play
                    r = sf_play.play(board, chess.engine.Limit(depth=SF_PLAY_DEPTH))
                    board.push(r.move)
                except: return 0.5
        mat = _material_balance(board)
        if abs(mat) >= ADJUDICATE_MATERIAL: adj += 1
        else: adj = 0
        if adj >= ADJUDICATE_MOVES:
            w = mat > 0
            return (1.0 if w else 0.0) if candidate_is_white else (0.0 if w else 1.0)

    if board.is_game_over():
        r = board.result(claim_draw=True)
        if candidate_is_white:
            return 1.0 if r=="1-0" else (0.0 if r=="0-1" else 0.5)
        return 1.0 if r=="0-1" else (0.0 if r=="1-0" else 0.5)
    return 0.5

def vs_stockfish_fitness(values, games=12, depth=3):
    cw = {**DEFAULT_WEIGHTS, **values}
    t = sum(_play_one_vs_stockfish(cw, i%2==0, depth,
            OPENING_POSITIONS[i%len(OPENING_POSITIONS)]) for i in range(games))
    return t / games


# ══════════════════════════════════════════════════════════════════════════════
#  SELF-PLAY FITNESS
# ══════════════════════════════════════════════════════════════════════════════

def _play_one_game(candidate_w, candidate_is_white, depth, max_moves=200,
                   opening_fen=None):
    eng_cand, eng_opp = _get_engines()
    eng_cand.WEIGHTS.update(candidate_w)
    eng_opp.WEIGHTS.update(DEFAULT_WEIGHTS)
    _reset_engine(eng_cand);  _reset_engine(eng_opp)

    board = chess.Board(opening_fen) if opening_fen else chess.Board()
    adj = 0
    for _ in range(max_moves):
        if board.is_game_over(): break
        if (board.turn == chess.WHITE) == candidate_is_white:
            move = eng_cand.get_next_move(board, board.turn, depth=depth)
        else:
            move = eng_opp.get_next_move(board, board.turn, depth=depth)
        board.push(move)
        mat = _material_balance(board)
        if abs(mat) >= ADJUDICATE_MATERIAL: adj += 1
        else: adj = 0
        if adj >= ADJUDICATE_MOVES:
            w = mat > 0
            return (1.0 if w else 0.0) if candidate_is_white else (0.0 if w else 1.0)

    r = board.result(claim_draw=True)
    if candidate_is_white:
        return 1.0 if r=="1-0" else (0.0 if r=="0-1" else 0.5)
    return 1.0 if r=="0-1" else (0.0 if r=="1-0" else 0.5)

def selfplay_fitness(values, games=12, depth=3):
    cw = {**DEFAULT_WEIGHTS, **values}
    t = sum(_play_one_game(cw, i%2==0, depth,
            opening_fen=OPENING_POSITIONS[i%len(OPENING_POSITIONS)]) for i in range(games))
    return t / games


# ══════════════════════════════════════════════════════════════════════════════
#  TEST SUITE / TEXEL / DEMO
# ══════════════════════════════════════════════════════════════════════════════

def load_epd_suite(path):
    positions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            if " bm " in line:
                fp, rest = line.split(" bm ", 1)
                bm = rest.split(";")[0].strip()
                if len(fp.strip().split()) == 4: fp = fp.strip() + " 0 1"
                positions.append((fp.strip(), bm))
    return positions

def testsuite_fitness(values, positions, depth=3):
    eng, _ = _get_engines()
    eng.WEIGHTS.update({**DEFAULT_WEIGHTS, **values})
    _reset_engine(eng)
    c = 0
    for fen, bm in positions:
        b = chess.Board(fen); _reset_engine(eng)
        m = eng.get_next_move(b, b.turn, depth=depth)
        if b.san(m) == bm: c += 1
    return c / max(1, len(positions))

def load_texel_data(path):
    d = []
    with open(path) as f:
        for row in csv.DictReader(f): d.append((row["fen"], float(row["result"])))
    return d

def _sigmoid(x, K=0.0073): return 1.0 / (1.0 + math.exp(-K * x))

def texel_fitness(values, data, K=0.0073):
    eng, _ = _get_engines()
    eng.WEIGHTS.update({**DEFAULT_WEIGHTS, **values}); _reset_engine(eng)
    err = sum((_sigmoid(eng.evaluate(chess.Board(f)), K) - r)**2 for f, r in data)
    return -(err / max(1, len(data)))

BUILTIN_POSITIONS = [
    ("1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1", "Rc4"),
    ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1", "Qxf7+"),
    ("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", "Re8+"),
    ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "e5"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "e4"),
]

def demo_testsuite_fitness(values, depth=3):
    eng, _ = _get_engines()
    eng.WEIGHTS.update({**DEFAULT_WEIGHTS, **values})
    c = 0
    for fen, bm in BUILTIN_POSITIONS:
        b = chess.Board(fen); _reset_engine(eng)
        try:
            m = eng.get_next_move(b, b.turn, depth=depth)
            if b.san(m) == bm: c += 1
        except: pass
    return c / len(BUILTIN_POSITIONS)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global SF_PLAY_DEPTH, SF_EVAL_DEPTH, SF_PLAY_SKILL
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["move_quality","self_play","vs_stockfish",
                    "test_suite","texel","demo"], default="move_quality")
    ap.add_argument("--strategy", choices=["genetic","hillclimb","random","grid","sweep"],
                    default="genetic")
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--epd-file", type=str, default=None)
    ap.add_argument("--texel-file", type=str, default=None)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gen", type=int, default=30)
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--sf-play-depth", type=int, default=None)
    ap.add_argument("--sf-eval-depth", type=int, default=None)
    ap.add_argument("--sf-skill", type=int, default=None)
    args = ap.parse_args()

    if args.sf_play_depth is not None: SF_PLAY_DEPTH = args.sf_play_depth
    if args.sf_eval_depth is not None: SF_EVAL_DEPTH = args.sf_eval_depth
    if args.sf_skill is not None: SF_PLAY_SKILL = args.sf_skill

    n_workers = args.workers
    params = build_params(coarse=args.coarse)

    if args.mode == "move_quality":
        games, depth = args.games, args.depth
        print(f"═══ move_quality mode (DELTA-BASED) ═══")
        print(f"  Candidate depth : {depth}")
        print(f"  SF play depth   : {SF_PLAY_DEPTH}  (skill {SF_PLAY_SKILL})")
        print(f"  SF eval depth   : {SF_EVAL_DEPTH}")
        print(f"  Games/eval      : {games}")
        print(f"  Metric          : avg centipawn loss per move")
        print(f"    fitness ~ 0   : perfect play")
        print(f"    fitness ~ -30 : slight inaccuracies")
        print(f"    fitness ~ -100: frequent blunders")
        print()
        def fitness(v): return move_quality_fitness(v, games=games, depth=depth)

    elif args.mode == "vs_stockfish":
        games, depth = args.games, args.depth
        print(f"═══ vs_stockfish mode (WIN RATE) ═══")
        print(f"  SF depth={SF_PLAY_DEPTH} skill={SF_PLAY_SKILL}  |  Games={games}")
        print()
        def fitness(v): return vs_stockfish_fitness(v, games=games, depth=depth)

    elif args.mode == "self_play":
        games, depth = args.games, args.depth
        print(f"═══ self_play mode ═══")
        print(f"  Depth={depth}  |  Games={games}  |  Opponent=default weights")
        print()
        def fitness(v): return selfplay_fitness(v, games=games, depth=depth)

    elif args.mode == "test_suite":
        if not args.epd_file: sys.exit("--epd-file required")
        positions = load_epd_suite(args.epd_file)
        print(f"Loaded {len(positions)} positions"); depth = args.depth
        def fitness(v): return testsuite_fitness(v, positions, depth=depth)

    elif args.mode == "texel":
        if not args.texel_file: sys.exit("--texel-file required")
        data = load_texel_data(args.texel_file)
        print(f"Loaded {len(data)} positions")
        def fitness(v): return texel_fitness(v, data)

    else:
        depth = args.depth
        def fitness(v): return demo_testsuite_fitness(v, depth=depth)

    strat_kwargs = {}
    if args.strategy == "genetic":
        strat_kwargs = {"population_size": args.pop, "generations": args.gen, "mutation_rate": 0.2}
    elif args.strategy == "hillclimb":
        strat_kwargs = {"iterations": args.pop * args.gen, "restarts": 3}
    elif args.strategy == "random":
        strat_kwargs = {"n_samples": args.pop * args.gen}

    from concurrent.futures import ThreadPoolExecutor
    bench_batch = min(n_workers, args.pop) if args.strategy in ("genetic","random","grid","sweep") else min(n_workers, 3)

    print(f"Threads: {n_workers}")
    print(f"Benchmarking {bench_batch} parallel evaluations...", end=" ", flush=True)

    bc = []
    for i in range(bench_batch):
        vals = {}
        for p in params:
            frac = (i+1)/(bench_batch+1)
            vals[p.name] = p.clamp(p.min_val + frac*(p.max_val - p.min_val))
        bc.append(vals)

    t0 = time.perf_counter()
    if bench_batch <= 1: fitness(bc[0])
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(fitness, bc))
    bwt = time.perf_counter() - t0
    cpe = bwt / bench_batch
    print(f"{bwt:.2f}s  ({cpe:.2f}s/eval)")

    te = {"genetic": args.pop*args.gen, "hillclimb": (args.pop*args.gen)*3+3,
          "random": args.pop*args.gen}.get(args.strategy, args.pop*args.gen)
    tb = math.ceil(te / max(1, bench_batch))
    eta = tb * bwt

    def ft(s):
        if s<60: return f"{s:.0f}s"
        if s<3600: return f"{int(s//60)}m {int(s%60)}s"
        return f"{int(s//3600)}h {int((s%3600)//60)}m"

    print(f"\n  ~{te} evals in ~{tb} batches")
    print(f"  Estimated wall time: ~{ft(eta)}\n")
    print(f"Strategy: {args.strategy}  |  Mode: {args.mode}  |  Depth: {args.depth}")
    print(f"Parameters: {len(params)}  |  Workers: {n_workers}\n")

    opt = Optimizer(params, fitness, strategy=args.strategy, verbose=True,
                    n_workers=n_workers, **strat_kwargs)
    report = opt.run()
    report.show(top_n=10)

    print("  Copy-paste into your engine:")
    print("  WEIGHTS = {")
    for k, v in {**DEFAULT_WEIGHTS, **report.best.values}.items():
        print(f'      "{k}": {v!r},')
    print("  }")

if __name__ == "__main__":
    main()