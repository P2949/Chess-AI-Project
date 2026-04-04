"""
selfplay_refine.py — Self-play / vs-Stockfish refinement (depth 1–4)

Key design:
  • Depth 2-4 uses recursive alpha-beta with batched leaf evaluation.
    At depth 1 (one ply above leaves), ALL children are collected into
    the buffer and evaluated in ONE GPU call. This gives correct minimax
    with turn alternation, alpha-beta pruning at depths 2+, and batched
    GPU inference — no more one-leaf-at-a-time overhead.
  • Stockfish opponent mode with Stockfish-based position labeling.
  • Streamed buffer: only needs to hold ~40 leaves per depth-1 node,
    never the entire tree. No 8GB+ allocation needed.
  • Game replacement: finished games replaced immediately.

Build Cython first:
    python setup_cython.py build_ext --inplace
Run:
    python selfplay_refine.py
"""
from dataclasses import dataclass, field
import time
import os
import gc
import random
import chess.polyglot
import numpy as np
import torch
import torch.nn as nn
import chess
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
from math import sqrt
import chess.engine

# ── Cython ────────────────────────────────────────────────────────────────────
try:
    from fast_chess import board_to_vector_bitboard as cy_board_to_vector
    HAVE_CYTHON = True
    print("[accel] Cython loaded ✓")
except ImportError:
    HAVE_CYTHON = False
    print("[accel] No Cython — using pure Python")

try:
    from fast_chess import vectorize_into_buffer as cy_vectorize_into
    HAVE_DIRECT_WRITE = True
    print("[accel] Direct buffer write ✓")
except ImportError:
    HAVE_DIRECT_WRITE = False

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0
}

PIECE_ORDER = [
    (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE), (chess.ROOK, chess.WHITE),
    (chess.QUEEN, chess.WHITE), (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK), (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK), (chess.ROOK, chess.BLACK),
    (chess.QUEEN, chess.BLACK), (chess.KING, chess.BLACK),
]

def _py_board_to_vector(board):
    v = np.zeros(773, dtype=np.float32)
    for plane, (pt, c) in enumerate(PIECE_ORDER):
        for sq in board.pieces(pt, c):
            v[plane * 64 + sq] = 1.0
    v[768] = float(board.has_kingside_castling_rights(chess.WHITE))
    v[769] = float(board.has_queenside_castling_rights(chess.WHITE))
    v[770] = float(board.has_kingside_castling_rights(chess.BLACK))
    v[771] = float(board.has_queenside_castling_rights(chess.BLACK))
    v[772] = float(board.turn == chess.WHITE)
    return v

board_to_vector = cy_board_to_vector if HAVE_CYTHON else _py_board_to_vector

INF = float('inf')

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "model.pt")
OUTPUT_PATH = str(BASE_DIR / "model_selfplay.pt")
SCORE_CLAMP = 1500

PROBE_PATH = BASE_DIR / "probe_set.pt"
PROBE_SIZE = 256

# ── TUNE THESE ────────────────────────────────────────────────────────────────
BATCH_GAMES = 12
RANDOM_OPENING   = 3
DEPTH            = 4
ROUNDS           = 100
GAMES_PER_ROUND  = 120
CONCURRENT       = 8       # 4 concurrent games — fits buffer, avoids thread overcommit
MAX_MOVES        = 200
SKIP_OPENING     = RANDOM_OPENING
EPSILON          = 0.10
BATCH_SIZE       = 4096
EPOCHS_PER_ROUND = 15
LR               = 1e-4
WEIGHT_DECAY     = 1e-4
GPU_BATCH        = 32768
AUGMENT          = True

# ── Opponent config ───────────────────────────────────────────────────────────
OPPONENT        = "stockfish"
SF_PATH         = "/usr/bin/stockfish"
SF_DEPTH        = 20       # play depth (10 ≈ 3000 ELO, fast)
SF_SKILL        = 20
SF_THREADS      = 2        # per play-engine instance (4 instances × 1 = 4 threads)
SF_HASH         = 256

LABEL_MODE      = "stockfish"
SF_LABEL_DEPTH  = 20       # labeling depth (12 is accurate + fast, ~100ms/pos)

SF_MOVE_TIMEOUT_S = 20.0
SF_LABEL_TIMEOUT_S = 50.0
SF_LABEL_THREADS = 6
SF_LABEL_HASH = 256

# Early termination
DECISIVE_SCORE  = 900
DECISIVE_COUNT  = 6

RESULT_MAP = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}

MATE_SCORE = 10000.0
DRAW_SCORE = 0.0

# ── Global TT Setup ──
TT = {}
EXACT, LOWERBOUND, UPPERBOUND = 0, 1, 2


# ══════════════════════════════════════════════════════════════════════════════
#  CORE CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RoundStats:
    started: float = field(default_factory=time.time)
    games_finished: int = 0
    positions: int = 0
    sf_timeouts: int = 0
    buffer_overflows: int = 0
    tt_lookups: int = 0
    tt_hits: int = 0
    cutoffs: int = 0
    max_game_plies: int = 0
    sf_move_seconds: float = 0.0
    nn_search_seconds: float = 0.0
    label_seconds: float = 0.0

    def hit_rate(self):
        return self.tt_hits / self.tt_lookups if self.tt_lookups else 0.0

def save_probe_set(X, y):
    torch.save(
        {"X": X[:PROBE_SIZE].cpu(), "y": y[:PROBE_SIZE].cpu()},
        PROBE_PATH,
    )

def eval_probe_set(model, device):
    data = torch.load(PROBE_PATH, map_location="cpu")
    ds = TensorDataset(data["X"], data["y"].unsqueeze(1))
    dl = DataLoader(ds, batch_size=BATCH_SIZE)
    crit = nn.SmoothL1Loss()
    total = 0.0
    count = 0
    model.eval()
    with torch.no_grad():
        for X_b, y_b in dl:
            loss = crit(model(X_b.to(device)), y_b.to(device)).item()
            total += loss * len(y_b)
            count += len(y_b)
    return total / max(1, count)

class BufferOverflowError(RuntimeError):
    pass


class LeafBuffer:
    __slots__ = ['data', 'scores', 'count', 'capacity', 'overflow', 'strict']

    def __init__(self, capacity, strict=False):
        self.capacity = capacity
        self.data = np.zeros((capacity, 773), dtype=np.float32)
        self.scores = np.zeros(capacity, dtype=np.float32)
        self.count = 0
        self.overflow = False
        self.strict = strict

    def reset(self):
        self.count = 0
        self.overflow = False

    def write(self, board) -> int:
        idx = self.count
        if idx >= self.capacity:
            self.overflow = True
            if self.strict:
                raise BufferOverflowError(
                    f"LeafBuffer overflow ({self.count}/{self.capacity})")
            return -1
        if HAVE_DIRECT_WRITE:
            cy_vectorize_into(board, self.data, idx)
        else:
            self.data[idx] = board_to_vector(board)
        self.count += 1
        return idx

    def eval_gpu(self, model, device, gpu_tensor=None):
        if self.count == 0:
            return
        n = self.count
        model.eval()
        with torch.no_grad():
            for start in range(0, n, GPU_BATCH):
                end = min(start + GPU_BATCH, n)
                chunk_size = end - start
                if gpu_tensor is not None and chunk_size <= gpu_tensor.shape[0]:
                    gpu_tensor[:chunk_size].copy_(
                        torch.from_numpy(self.data[start:end]))
                    out = model(gpu_tensor[:chunk_size])
                else:
                    t = torch.from_numpy(self.data[start:end]).to(device)
                    out = model(t)
                    del t
                self.scores[start:end] = out.squeeze(-1).cpu().numpy() * SCORE_CLAMP


class HarvestBuffer:
    __slots__ = ['X', 'y', 'count', 'capacity']

    def __init__(self, capacity):
        self.capacity = capacity
        self.X = np.zeros((capacity, 773), dtype=np.float32)
        self.y = np.zeros(capacity, dtype=np.float32)
        self.count = 0

    def write(self, vec, label):
        if self.count >= self.capacity:
            return False
        self.X[self.count] = vec
        self.y[self.count] = label
        self.count += 1
        return True

    def get_data(self):
        return self.X[:self.count], self.y[:self.count]


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH — correct minimax with batched leaf eval + alpha-beta
# ══════════════════════════════════════════════════════════════════════════════


def terminal_score(board):
    """White-perspective terminal score."""
    if board.is_checkmate():
        return -MATE_SCORE if board.turn == chess.WHITE else MATE_SCORE
    return DRAW_SCORE


def _order_moves(board, moves):
    """Move ordering using Checks, MVV-LVA for captures, and Promotions."""
    def move_score(m):
        score = 0
        if board.gives_check(m):
            score += 1000
        if board.is_capture(m):
            if board.is_en_passant(m):
                victim_type = chess.PAWN
            else:
                victim = board.piece_at(m.to_square)
                victim_type = victim.piece_type if victim else 0

            attacker = board.piece_at(m.from_square)
            attacker_type = attacker.piece_type if attacker else 0

            score += 10 * PIECE_VALUES.get(victim_type, 0) - PIECE_VALUES.get(attacker_type, 0)
        if m.promotion is not None:
            score += 500
        return score

    moves.sort(key=move_score, reverse=True)


def search_node(board, depth, buf, model, device, gpu_tensor,
                alpha=-INF, beta=INF, stats=None):
    if stats is not None:
        stats.tt_lookups += 1
    if board.is_game_over():
        return terminal_score(board)

    orig_alpha, orig_beta = alpha, beta
    h = chess.polyglot.zobrist_hash(board)

    if h in TT:
        if stats is not None:
            stats.tt_hits += 1
        entry = TT[h]
        if entry["depth"] >= depth:
            if entry["flag"] == EXACT:
                return entry["score"]
            if entry["flag"] == LOWERBOUND:
                alpha = max(alpha, entry["score"])
            elif entry["flag"] == UPPERBOUND:
                beta = min(beta, entry["score"])
            if alpha >= beta:
                if stats is not None:
                    stats.cutoffs += 1
                return entry["score"]

    if depth == 0:
        return quiescence_search_batched(board, alpha, beta, buf, model, device, gpu_tensor)

    maximizing = (board.turn == chess.WHITE)
    moves = list(board.legal_moves)
    _order_moves(board, moves)

    if depth == 1:
        buf.reset()
        child_info = []
        for mv in moves:
            board.push(mv)
            if board.is_game_over():
                child_info.append((-1, terminal_score(board)))
            else:
                idx = buf.write(board)
                child_info.append((idx, None))
            board.pop()

        if buf.count > 0:
            buf.eval_gpu(model, device, gpu_tensor)

        best = -INF if maximizing else INF
        for buf_idx, term_score in child_info:
            sc = term_score if term_score is not None else float(buf.scores[buf_idx])
            if maximizing:
                if sc > best: best = sc
                if best > alpha: alpha = best
                if alpha >= beta: break
            else:
                if sc < best: best = sc
                if best < beta: beta = best
                if alpha >= beta: break
        flag = EXACT
        if best <= orig_alpha:
            flag = UPPERBOUND
        elif best >= beta:
            flag = LOWERBOUND
        TT[h] = {"depth": 1, "score": best, "flag": flag}
        return best

    # ── DEPTH >= 2: recursive alpha-beta ─────────────────────────────────
    best = -INF if maximizing else INF
    for mv in moves:
        board.push(mv)
        score = search_node(board, depth - 1, buf, model, device, gpu_tensor,
                            alpha, beta, stats=stats)
        board.pop()

        if maximizing:
            if score > best: best = score
            if best > alpha: alpha = best
            if alpha >= beta: 
                if stats is not None:
                    stats.cutoffs += 1
                break
        else:
            if score < best: best = score
            if best < beta: beta = best
            if alpha >= beta: 
                if stats is not None:
                    stats.cutoffs += 1
                break
    flag = EXACT
    if best <= orig_alpha:
        flag = UPPERBOUND
    elif best >= beta:
        flag = LOWERBOUND
    TT[h] = {"depth": depth, "score": best, "flag": flag}
    return best


# ══════════════════════════════════════════════════════════════════════════════
#  MOVE SELECTION — depth 1-4
# ══════════════════════════════════════════════════════════════════════════════

def _best_move_depth1_batch(games, buf, model, device, gpu_tensor, stats=None):
    """Depth 1: evaluate all children of all games in one GPU call."""
    buf.reset()
    game_ranges = []
    for gi, game in enumerate(games):
        board = game.board
        start = buf.count
        for move in board.legal_moves:
            board.push(move)
            if buf.write(board) == -1:
                board.pop()
                break
            board.pop()
        game_ranges.append((gi, start, buf.count))

    if buf.count == 0:
        return {}
    buf.eval_gpu(model, device, gpu_tensor)

    results = {}
    for gi, start, end in game_ranges:
        if start >= end:
            continue
        board = games[gi].board
        maximizing = (board.turn == chess.WHITE)
        moves = list(board.legal_moves)
        s = buf.scores[start:end]
        idx = int(np.argmax(s)) if maximizing else int(np.argmin(s))
        results[gi] = (moves[idx], float(s[idx]))
    return results


def _best_move_generic(games, buf, model, device, gpu_tensor, search_depth,stats=None):
    """
    Generic best-move finder for depth 2+.
    Uses search_node with batched leaf eval + alpha-beta.
    Processes games sequentially (buffer reused per search_node call).
    """
    results = {}
    for gi, game in enumerate(games):
        board = game.board
        maximizing = (board.turn == chess.WHITE)
        best_move = None
        best_score = -INF if maximizing else INF

        moves = list(board.legal_moves)
        _order_moves(board, moves)

        for root_move in moves:
            board.push(root_move)
            score = search_node(board, search_depth - 1, buf, model,
                                device, gpu_tensor, stats=stats)
            board.pop()

            if maximizing and score > best_score:
                best_score, best_move = score, root_move
            elif not maximizing and score < best_score:
                best_score, best_move = score, root_move

        if best_move is not None:
            results[gi] = (best_move, best_score)
    return results


def _best_move_depth2_batch(games, buf, model, device, gpu_tensor,stats=None):
    return _best_move_generic(games, buf, model, device, gpu_tensor, 2,stats=stats)

def _best_move_depth3_batch(games, buf, model, device, gpu_tensor,stats=None):
    return _best_move_generic(games, buf, model, device, gpu_tensor, 3,stats=stats)

def _best_move_depth4_batch(games, buf, model, device, gpu_tensor,stats=None):
    return _best_move_generic(games, buf, model, device, gpu_tensor, 4,stats=stats)


MOVE_FNS = {
    1: _best_move_depth1_batch,
    2: _best_move_depth2_batch,
    3: _best_move_depth3_batch,
    4: _best_move_depth4_batch,
}


# ══════════════════════════════════════════════════════════════════════════════
#  STOCKFISH
# ══════════════════════════════════════════════════════════════════════════════

def label_positions_with_stockfish(positions_boards, engine, depth, timeout_s=SF_LABEL_TIMEOUT_S):
    labels = []
    for board in positions_boards:
        try:
            info = engine.analyse(
                board,
                chess.engine.Limit(depth=depth, time=timeout_s),
            )
            score = info["score"].white()
            if score.is_mate():
                cp = 15000 if score.mate() > 0 else -15000
            else:
                cp = score.score()
            cp = max(-SCORE_CLAMP, min(SCORE_CLAMP, cp))
            labels.append(cp / SCORE_CLAMP)
        except chess.engine.EngineTerminatedError:
            raise  # let this propagate — engine is dead
        except Exception:
            labels.append(0.0)
    return labels

def quiescence_search_batched(board, alpha, beta, buf, model, device, gpu_tensor):
    if board.is_game_over():
        return terminal_score(board)

    maximizing = (board.turn == chess.WHITE)

    buf.reset()
    buf.write(board)
    buf.eval_gpu(model, device, gpu_tensor)
    stand_pat = float(buf.scores[0])

    best = stand_pat
    if maximizing:
        if best >= beta:
            return best
        alpha = max(alpha, best)
    else:
        if best <= alpha:
            return best
        beta = min(beta, best)

    captures = [m for m in board.legal_moves if board.is_capture(m)]
    if not captures:
        return best

    buf.reset()
    for m in captures:
        board.push(m)
        buf.write(board)
        board.pop()

    if buf.count == 0:
        return best

    buf.eval_gpu(model, device, gpu_tensor)

    for i in range(buf.count):
        score = float(buf.scores[i])
        if maximizing:
            if score > best:
                best = score
            alpha = max(alpha, best)
        else:
            if score < best:
                best = score
            beta = min(beta, best)

        if alpha >= beta:
            break

    return best


class StockfishPool:
    __slots__ = ['engines', 'sf_path', 'depth', 'skill', 'threads', 'hash_mb']

    def __init__(self, n_engines, sf_path, depth, skill, threads, hash_mb):
        self.sf_path = sf_path
        self.depth = depth
        self.skill = skill
        self.threads = threads
        self.hash_mb = hash_mb
        self.engines = [self._make_engine() for _ in range(n_engines)]

    def _make_engine(self):
        engine = chess.engine.SimpleEngine.popen_uci(self.sf_path)
        engine.configure({
            "Threads": self.threads,
            "Hash": self.hash_mb,
            "Skill Level": self.skill,
        })
        return engine

    def get_move(self, idx, board, timeout_s=SF_MOVE_TIMEOUT_S):
        try:
            result = self.engines[idx].play(
                board,
                chess.engine.Limit(depth=self.depth, time=timeout_s),
            )
            return result.move
        except Exception as e:
            self.replace_engine(idx)
            raise TimeoutError(f"Stockfish move failed on engine {idx}: {e}") from e


    def replace_engine(self, idx):
        try:
            self.engines[idx].quit()
        except Exception:
            pass
        self.engines[idx] = self._make_engine()

    def close_all(self):
        for eng in self.engines:
            try:
                eng.quit()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL + DATASET
# ══════════════════════════════════════════════════════════════════════════════

class ChessEvaluator(nn.Module):
    def __init__(self, input_dim=773):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),        nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),        nn.ReLU(),
            nn.Linear(128, 1),          nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)


class TensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X; self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.X[i], self.y[i]


def build_flip_perm():
    _F = [6,7,8,9,10,11,0,1,2,3,4,5]
    p = torch.zeros(773, dtype=torch.long)
    for s in range(12):
        d = _F[s]
        for sq in range(64):
            p[d*64+(sq^56)] = s*64+sq
    p[768]=770; p[769]=771; p[770]=768; p[771]=769; p[772]=772
    return p

FLIP_PERM = build_flip_perm()

def augment_gpu(X, y, device):
    perm = FLIP_PERM.to(device)
    CHUNK = 500_000
    fX = torch.empty_like(X)
    for s in range(0, len(X), CHUNK):
        e = min(s + CHUNK, len(X))
        c = X[s:e].to(device)
        f = c[:, perm]; f[:, 772] = 1.0 - f[:, 772]
        fX[s:e] = f.cpu()
        del c, f
    torch.cuda.empty_cache()
    return torch.cat([X, fX], 0), torch.cat([y, -y], 0)


# ══════════════════════════════════════════════════════════════════════════════
#  GAME STATE + GAME LOOP
# ══════════════════════════════════════════════════════════════════════════════

class GameState:
    __slots__ = ['board', 'positions', 'position_boards', 'move_count',
                 'done', 'decisive_streak', 'position_hashes',
                 'nn_is_white', 'sf_index']
    def __init__(self):
        self.board = chess.Board()
        self.positions = []
        self.position_boards = []
        self.move_count = 0
        self.done = False
        self.decisive_streak = 0
        self.position_hashes = set()
        self.nn_is_white = True
        self.sf_index = -1

    def randomize_opening(self, n_random):
        for _ in range(n_random):
            moves = list(self.board.legal_moves)
            if not moves or self.board.is_game_over():
                break
            self.board.push(random.choice(moves))
            self.move_count += 1
        if self.board.is_game_over():
            self.board = chess.Board()
            self.move_count = 0
        self.position_hashes.add(chess.polyglot.zobrist_hash(self.board))


def play_games(model, device, num_games, depth, concurrent, buf, gpu_tensor,
               sf_label_engine_ref=None):
    move_fn = MOVE_FNS[depth]
    use_sf = (OPPONENT == "stockfish")

    stats = RoundStats()
    last_log = time.time()

    def maybe_log(force=False, nn_count=0, sf_count=0):
        nonlocal last_log
        now = time.time()
        if not force and (now - last_log) < 5.0:
            return
        elapsed = now - stats.started
        print(
            f"[round] {stats.games_finished}/{num_games} finished | "
            f"active={len(games)} nn={nn_count} sf={sf_count} | "
            f"pos={stats.positions} | "
            f"timeouts={stats.sf_timeouts} | "
            f"overflows={stats.buffer_overflows} | "
            f"tt={stats.tt_hits}/{stats.tt_lookups} "
            f"({stats.hit_rate():.1%}) | "
            f"cutoffs={stats.cutoffs} | "
            f"max_plies={stats.max_game_plies} | "
            f"sf_move={stats.sf_move_seconds:.1f}s "
            f"nn_search={stats.nn_search_seconds:.1f}s "
            f"label={stats.label_seconds:.1f}s | "
            f"elapsed={elapsed/60:.1f}m"
        )
        last_log = now

    est_positions = num_games * (MAX_MOVES - SKIP_OPENING)
    harvest = HarvestBuffer(est_positions)

    sf_pool = None
    if use_sf:
        sf_pool = StockfishPool(
            concurrent, SF_PATH, SF_DEPTH, SF_SKILL, SF_THREADS, SF_HASH)
        print(f"[stockfish] {concurrent} engines, depth={SF_DEPTH}, "
              f"skill={SF_SKILL}")

    completed = 0
    game_counter = 0
    pbar = tqdm(total=num_games,
                desc=f"{'vs SF' if use_sf else 'Self-play'} depth-{depth}")

    def make_game():
        nonlocal game_counter
        g = GameState()
        g.randomize_opening(RANDOM_OPENING)
        g.nn_is_white = (game_counter % 2 == 0)
        game_counter += 1
        return g

    games = [make_game() for _ in range(min(concurrent, num_games))]
    if use_sf:
        for i, g in enumerate(games):
            g.sf_index = i
    started = len(games)

    while completed < num_games:
        for g in games:
            if not g.done and (g.board.is_game_over() or g.move_count >= MAX_MOVES):
                g.done = True

        still_active = []
        for game in games:
            if game.done:
                total = len(game.positions)
                if total > 0:

                    if LABEL_MODE == "stockfish" and sf_label_engine_ref[0] is not None:
                        n_pos = len(game.position_boards)
                        if n_pos > 20:
                            print(f"  [labeling {n_pos} pos...]",
                                  end="", flush=True)
                        t0 = time.perf_counter()
                        try:
                            sf_labels = label_positions_with_stockfish(
                                game.position_boards, sf_label_engine_ref[0], SF_LABEL_DEPTH, timeout_s=10)
                        except chess.engine.EngineTerminatedError:
                            stats.sf_timeouts += 1
                            print(f"  [labeling engine died, restarting]", flush=True)
                            try:
                                sf_label_engine_ref[0].quit()
                            except Exception:
                                pass
                            sf_label_engine_ref[0] = chess.engine.SimpleEngine.popen_uci(SF_PATH)
                            sf_label_engine_ref[0].configure({"Threads": SF_LABEL_THREADS, "Hash": SF_LABEL_HASH})
                            sf_labels = [0.0] * len(game.positions)
                        stats.label_seconds += time.perf_counter() - t0
                        for vec, label in zip(game.positions, sf_labels):
                            if(harvest.write(vec, label)):
                                stats.positions += 1
                    else:
                        result = game.board.result()
                        if use_sf:
                            if game.nn_is_white:
                                outcome = RESULT_MAP.get(result, 0.0)
                            else:
                                flipped = {"1-0": -1.0, "0-1": 1.0,
                                           "1/2-1/2": 0.0}
                                outcome = flipped.get(result, 0.0)
                        else:
                            outcome = RESULT_MAP.get(result, 0.0)
                        for i, vec in enumerate(game.positions):
                            w = sqrt((i + 1) / total)
                            if not harvest.write(vec, outcome * w):
                                break
                            else:
                                stats.positions += 1
                game.positions.clear()
                game.position_boards.clear()
                completed += 1
                pbar.update(1)
                stats.games_finished += 1
                stats.max_game_plies = max(stats.max_game_plies, game.move_count)
                maybe_log(force=True)

                if started < num_games:
                    new_game = make_game()
                    if use_sf:
                        new_game.sf_index = game.sf_index
                        sf_pool.replace_engine(game.sf_index)
                    still_active.append(new_game)
                    started += 1
            else:
                still_active.append(game)

        games = still_active
        if not games:
            break

        nn_games = []
        sf_games = []
        for g in games:
            if g.done:
                continue
            if use_sf:
                is_nn_turn = (g.board.turn == chess.WHITE) == g.nn_is_white
                if is_nn_turn:
                    nn_games.append(g)
                else:
                    sf_games.append(g)
            else:
                nn_games.append(g)
        maybe_log(nn_count=len(nn_games), sf_count=len(sf_games))
        t0 = time.perf_counter()
        for game in sf_games:
            try:
                sf_move = sf_pool.get_move(game.sf_index, game.board)
                game.board.push(sf_move)
                game.move_count += 1
                game.position_hashes.add(chess.polyglot.zobrist_hash(game.board))
                if game.board.is_game_over() or game.move_count >= MAX_MOVES:
                    game.done = True
            except TimeoutError as e:
                stats.sf_timeouts += 1
                print(f"  [SF timeout: {e}] restarting engine", flush=True)
                sf_pool.replace_engine(game.sf_index)
            except Exception as e:
                print(f"  [SF error: {e}]", end="", flush=True)
                game.done = True
        stats.sf_move_seconds += time.perf_counter() - t0

        if not nn_games:
            continue

        try:
            t0 = time.perf_counter()
            results = move_fn(nn_games, buf, model, device, gpu_tensor, stats)
            stats.nn_search_seconds += time.perf_counter() - t0
        except BufferOverflowError:
            stats.buffer_overflows += 1
            raise

        for local_idx, game in enumerate(nn_games):
            data = results.get(local_idx)
            if data is None:
                game.done = True
                continue

            best_move, best_score = data

            # Anti-repetition
            game.board.push(best_move)
            h = chess.polyglot.zobrist_hash(game.board)
            game.board.pop()
            all_moves = list(game.board.legal_moves)

            if h in game.position_hashes and len(all_moves) > 1:
                alternatives = []
                for m in all_moves:
                    if m == best_move:
                        continue
                    game.board.push(m)
                    ah = chess.polyglot.zobrist_hash(game.board)
                    game.board.pop()
                    if ah not in game.position_hashes:
                        alternatives.append(m)
                if alternatives:
                    best_move = random.choice(alternatives)
            elif random.random() < EPSILON:
                if len(all_moves) > 1:
                    best_move = random.choice(all_moves)

            # Decisive streak
            maximizing = (game.board.turn == chess.WHITE)
            if (maximizing and best_score > DECISIVE_SCORE) or \
               (not maximizing and best_score < -DECISIVE_SCORE):
                game.decisive_streak += 1
            else:
                game.decisive_streak = 0

            if game.decisive_streak >= DECISIVE_COUNT:
                game.positions.append(board_to_vector(game.board))
                if LABEL_MODE == "stockfish":
                    game.position_boards.append(game.board.copy())
                game.board.push(best_move)
                game.move_count += 1
                game.done = True
                continue

            if game.move_count >= SKIP_OPENING:
                game.positions.append(board_to_vector(game.board))
                if LABEL_MODE == "stockfish":
                    game.position_boards.append(game.board.copy())

            game.board.push(best_move)
            game.move_count += 1
            game.position_hashes.add(chess.polyglot.zobrist_hash(game.board))

            if game.board.is_game_over() or game.move_count >= MAX_MOVES:
                game.done = True

    pbar.close()
    if sf_pool:
        sf_pool.close_all()

    if harvest.count == 0:
        return None, None

    X, y = harvest.get_data()
    print(f"Generated {harvest.count} positions from {num_games} games")
    return X.copy(), y.copy()


# ══════════════════════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train(model, train_loader, val_loader, epochs, lr, device, save_path):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    crit = nn.SmoothL1Loss()
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

    best_val = INF
    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            loss = crit(model(X_b), y_b)
            loss.backward(); opt.step(); sched.step()
            tl += loss.item() * len(y_b)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                vl += crit(model(X_b.to(device)), y_b.to(device)).item() * len(y_b)

        tl /= len(train_loader.dataset)
        vl /= len(val_loader.dataset)
        tag = ""
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), save_path)
            tag = "  ← saved"
        print(f"  Epoch {epoch:3d}/{epochs}  train: {tl:.6f}  val: {vl:.6f}{tag}")

    model.load_state_dict(torch.load(save_path, map_location=device))
    return best_val


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" +
          (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))

    model = ChessEvaluator()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print(f"Loaded {MODEL_PATH}")
    else:
        print("No model found. Starting fresh.")
    model = model.to(device)
        
    # 3. Preparation & Warmup
    model.eval()
    with torch.no_grad():
        # Dry run to initialize CUDA kernels
        model(torch.zeros(1, 773, device=device))
    print("[accel] Model warmed up ✓")

    params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {params:,}")

    gpu_tensor = torch.zeros(GPU_BATCH, 773, dtype=torch.float32, device=device)
    gpu_vram_mb = GPU_BATCH * 773 * 4 / 1024 / 1024
    print(f"[accel] Persistent GPU tensor: {GPU_BATCH:,} × 773 = {gpu_vram_mb:.0f}MB VRAM")

    # Small buffer — streamed search only needs ~40 positions at a time
    buf = LeafBuffer(GPU_BATCH, strict=True)
    buf_mb = GPU_BATCH * 773 * 4 / 1024 / 1024
    print(f"Leaf buffer: {GPU_BATCH:,} × 773 = {buf_mb:.0f}MB")

    print(f"\n{'='*60}")
    print(f"  Depth:       {DEPTH}")
    print(f"  Games/round: {GAMES_PER_ROUND}")
    print(f"  Rounds:      {ROUNDS}")
    print(f"  Concurrent:  {CONCURRENT}")
    print(f"  Opponent:    {OPPONENT}" +
          (f" (depth={SF_DEPTH}, skill={SF_SKILL})" if OPPONENT == "stockfish" else ""))
    print(f"  Labeling:    {LABEL_MODE}" +
          (f" (depth={SF_LABEL_DEPTH})" if LABEL_MODE == "stockfish" else ""))
    print(f"  Decisive:    >{DECISIVE_SCORE}cp × {DECISIVE_COUNT} moves → end early")
    print(f"{'='*60}\n")

    # Persistent labeling engine — created ONCE, reused across all rounds
    sf_label_engine = [None]
    if LABEL_MODE == "stockfish":
        sf_label_engine[0] = chess.engine.SimpleEngine.popen_uci(SF_PATH)
        sf_label_engine[0].configure({"Threads": SF_LABEL_THREADS, "Hash": SF_LABEL_HASH})
        print(f"[stockfish] Labeling engine: depth={SF_LABEL_DEPTH}, threads={SF_LABEL_THREADS}")
    prev_probe_val = None
    for rnd in range(1, ROUNDS + 1):
        print(f"\n── Round {rnd}/{ROUNDS} ──")
        round_X = []
        round_y = []
        remaining = GAMES_PER_ROUND

        while remaining > 0:
            n = min(BATCH_GAMES, remaining)

            while True:
                try:
                    TT.clear()
                    X_np, y_np = play_games(
                        model, device, n, DEPTH,
                        min(n, CONCURRENT), buf, gpu_tensor,
                        sf_label_engine_ref=sf_label_engine,
                    )
                    if X_np is not None and len(y_np) > 0:
                        round_X.append(X_np)
                        round_y.append(y_np)
                    break

                except BufferOverflowError as e:
                    print(f"[overflow] {e}")
                    if n == 1:
                        raise
                    n = max(1, n // 2)

            remaining -= n

        if not round_X:
            print("Too few positions, skipping.")
            continue

        X_np = np.concatenate(round_X, axis=0)
        y_np = np.concatenate(round_y, axis=0)

        X = torch.from_numpy(X_np)
        y = torch.from_numpy(y_np)
        print(f"Dataset: {len(y):,} samples | y_mean={y.float().mean():+.4f} | y_std={y.float().std(unbiased=False):.4f}")
        del X_np, y_np; gc.collect()
        if not PROBE_PATH.exists() and len(y) >= PROBE_SIZE:
            save_probe_set(X, y)
        if AUGMENT:
            X, y = augment_gpu(X, y, device)
            print(f"Augmented to {len(y)} positions")

        n = len(y)
        perm = torch.randperm(n)
        X, y = X[perm], y[perm]
        split = int(0.9 * n)

        tl = DataLoader(TensorDataset(X[:split], y[:split].unsqueeze(1)),
                        batch_size=BATCH_SIZE, shuffle=True)
        vl = DataLoader(TensorDataset(X[split:], y[split:].unsqueeze(1)),
                        batch_size=BATCH_SIZE)
        pre_probe_val = eval_probe_set(model, device) if PROBE_PATH.exists() else float("nan")
        t_train = time.perf_counter()
        best = train(model, tl, vl, EPOCHS_PER_ROUND, LR, device, OUTPUT_PATH)
        train_wall = time.perf_counter() - t_train
        probe_val = eval_probe_set(model, device) if PROBE_PATH.exists() else float("nan")

        delta = ""
        if not np.isnan(pre_probe_val) and not np.isnan(probe_val):
            delta = f"  Δprobe={probe_val - pre_probe_val:+.6f}"
        print(
            f"Round {rnd} done — best val: {best:.6f}  "
            f"probe: {probe_val:.6f}  "
            f"pre: {pre_probe_val:.6f}  "
            f"train: {train_wall/60:.1f}m{delta}"
        )
        prev_probe_val = probe_val

        del X, y, tl, vl; gc.collect(); torch.cuda.empty_cache()

    if sf_label_engine[0] is not None:
        sf_label_engine[0].quit()

    del buf, gpu_tensor; gc.collect(); torch.cuda.empty_cache()
    print(f"\nFinal model saved to {OUTPUT_PATH}")