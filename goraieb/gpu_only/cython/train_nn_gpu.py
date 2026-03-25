"""
train_nn_gpu.py — GPU + Cython accelerated chess NN trainer

Build Cython first:
    pip install cython
    python setup_cython.py build_ext --inplace

Then run:
    python train_nn_gpu.py
"""

import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import chess
import chess.pgn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

# ── Cython imports with fallback ──────────────────────────────────────────────
try:
    from fast_chess import (
        board_to_vector_bitboard as board_to_vector,
        batch_vectorize as cy_batch_vectorize,
        expand_tree as cy_expand_tree,
        outcome_weight as cy_outcome_weight,
        extract_and_label_game as cy_extract_game,
        batch_flip_vectors as cy_batch_flip,
    )
    HAVE_CYTHON = True
    print("[accel] Cython fast_chess loaded ✓")
except ImportError:
    HAVE_CYTHON = False
    print("[accel] Cython not available — using pure Python (run setup_cython.py)")

    PIECE_ORDER = [
        (chess.PAWN,   chess.WHITE), (chess.KNIGHT, chess.WHITE),
        (chess.BISHOP, chess.WHITE), (chess.ROOK,   chess.WHITE),
        (chess.QUEEN,  chess.WHITE), (chess.KING,   chess.WHITE),
        (chess.PAWN,   chess.BLACK), (chess.KNIGHT, chess.BLACK),
        (chess.BISHOP, chess.BLACK), (chess.ROOK,   chess.BLACK),
        (chess.QUEEN,  chess.BLACK), (chess.KING,   chess.BLACK),
    ]

    def board_to_vector(board):
        v = np.zeros(773, dtype=np.float32)
        for plane, (pt, color) in enumerate(PIECE_ORDER):
            for sq in board.pieces(pt, color):
                v[plane * 64 + sq] = 1.0
        v[768] = float(board.has_kingside_castling_rights(chess.WHITE))
        v[769] = float(board.has_queenside_castling_rights(chess.WHITE))
        v[770] = float(board.has_kingside_castling_rights(chess.BLACK))
        v[771] = float(board.has_queenside_castling_rights(chess.BLACK))
        v[772] = float(board.turn == chess.WHITE)
        return v


# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR           = Path(__file__).resolve().parent
PGN_FILE           = str(BASE_DIR / "db.pgn")
NUM_POSITIONS      = 2_000_000
POSITIONS_PER_GAME = 64
SKIP_OPENING_PLY   = 6
BATCH_SIZE         = 4096
EPOCHS             = 100
LR                 = 3e-4
WEIGHT_DECAY       = 1e-4
MODEL_PATH         = "model.pt"
SCORE_CLAMP        = 1500
AUGMENT_COLOR_FLIP = True

SELF_PLAY_ENABLED  = True
SELF_PLAY_ROUNDS   = 5
SELF_PLAY_GAMES    = 2000
SELF_PLAY_DEPTH    = 3
SELF_PLAY_EPOCHS   = 15
CONCURRENT_GAMES   = 64
GPU_EVAL_BATCH     = 16384

RESULT_MAP = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


# ── GPU augmentation ──────────────────────────────────────────────────────────
def build_flip_permutation() -> torch.Tensor:
    _FLIP = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
    perm = torch.zeros(773, dtype=torch.long)
    for src in range(12):
        dst = _FLIP[src]
        for sq in range(64):
            perm[dst * 64 + (sq ^ 56)] = src * 64 + sq
    perm[768] = 770; perm[769] = 771
    perm[770] = 768; perm[771] = 769
    perm[772] = 772
    return perm


FLIP_PERM = build_flip_permutation()


def augment_gpu(X: torch.Tensor, y: torch.Tensor,
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked GPU augmentation — never puts the full dataset on VRAM."""
    perm = FLIP_PERM.to(device)
    CHUNK = 500_000  # process 500k at a time

    flipped_X = torch.empty_like(X)
    for start in range(0, len(X), CHUNK):
        end = min(start + CHUNK, len(X))
        chunk = X[start:end].to(device)
        flipped = chunk[:, perm]
        flipped[:, 772] = 1.0 - flipped[:, 772]
        flipped_X[start:end] = flipped.cpu()
        del chunk, flipped

    torch.cuda.empty_cache()
    X_aug = torch.cat([X, flipped_X], dim=0)
    y_aug = torch.cat([y, -y], dim=0)
    del flipped_X
    print(f"[GPU augment] {len(y_aug) // 2} → {len(y_aug)} positions.")
    return X_aug, y_aug


# ── Batched GPU evaluator ─────────────────────────────────────────────────────
class BatchedEvaluator:
    def __init__(self, model: nn.Module, device: torch.device,
                 max_leaves: int = 500_000):
        self.model = model
        self.device = device
        self._max_leaves = max_leaves
        self._buffer = torch.zeros(max_leaves, 773, dtype=torch.float32,
                                   pin_memory=True)
        self._vec_count = 0
        self._scores: list[float] = []
        self._entries: list[tuple[str, int | float]] = []

    def reset(self):
        self._vec_count = 0
        self._scores.clear()
        self._entries.clear()

    def enqueue(self, board) -> int:
        idx = len(self._entries)
        if board.is_checkmate():
            score = -99999.0 if board.turn == chess.WHITE else 99999.0
            self._entries.append(('terminal', score))
            return idx
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_fifty_moves() or board.is_repetition(3)):
            self._entries.append(('terminal', 0.0))
            return idx
        vec_idx = self._vec_count
        if vec_idx < self._max_leaves:
            vec = board_to_vector(board)
            self._buffer[vec_idx].copy_(torch.from_numpy(vec))
        self._vec_count += 1
        self._entries.append(('gpu', vec_idx))
        return idx

    def enqueue_batch(self, boards: list) -> list[int]:
        indices = []
        non_terminal = []
        non_terminal_vi = []
        for board in boards:
            idx = len(self._entries)
            indices.append(idx)
            if board.is_checkmate():
                s = -99999.0 if board.turn == chess.WHITE else 99999.0
                self._entries.append(('terminal', s))
            elif (board.is_stalemate() or board.is_insufficient_material()
                    or board.is_fifty_moves() or board.is_repetition(3)):
                self._entries.append(('terminal', 0.0))
            else:
                vi = self._vec_count
                self._vec_count += 1
                self._entries.append(('gpu', vi))
                non_terminal.append(board)
                non_terminal_vi.append(vi)

        if non_terminal:
            if HAVE_CYTHON:
                vecs = cy_batch_vectorize(non_terminal)
            else:
                vecs = np.stack([board_to_vector(b) for b in non_terminal])
            for i, vi in enumerate(non_terminal_vi):
                if vi < self._max_leaves:
                    self._buffer[vi].copy_(torch.from_numpy(vecs[i]))
        return indices

    def flush(self):
        if self._vec_count == 0:
            return
        n = min(self._vec_count, self._max_leaves)
        self.model.eval()
        all_scores = []
        with torch.no_grad():
            for start in range(0, n, GPU_EVAL_BATCH):
                end = min(start + GPU_EVAL_BATCH, n)
                chunk = self._buffer[start:end].to(self.device, non_blocking=True)
                out = self.model(chunk).cpu().squeeze(-1).tolist()
                all_scores.extend(out)
        self._scores = [s * SCORE_CLAMP for s in all_scores]

    def get_score(self, idx: int) -> float:
        kind, val = self._entries[idx]
        if kind == 'terminal':
            return val
        return self._scores[val]


# ── Pure Python fallback for tree expansion (when no Cython) ──────────────────
class TreeNode:
    __slots__ = ['move', 'children', 'score', 'eval_idx', 'is_maximizing', 'is_leaf']
    def __init__(self, move, is_maximizing):
        self.move = move
        self.children = []
        self.score = 0.0
        self.eval_idx = -1
        self.is_maximizing = is_maximizing
        self.is_leaf = False


def _py_expand_tree(board, depth, maximizing, evaluator):
    node = TreeNode(move=None, is_maximizing=maximizing)
    if depth == 0 or board.is_game_over():
        node.eval_idx = evaluator.enqueue(board)
        node.is_leaf = True
        return node
    for move in board.legal_moves:
        board.push(move)
        child = _py_expand_tree(board, depth - 1, not maximizing, evaluator)
        child.move = move
        node.children.append(child)
        board.pop()
    if not node.children:
        node.eval_idx = evaluator.enqueue(board)
        node.is_leaf = True
    return node


def _py_propagate(node, evaluator):
    if node.is_leaf:
        node.score = evaluator.get_score(node.eval_idx)
        return node.score
    if node.is_maximizing:
        node.score = float('-inf')
        for child in node.children:
            node.score = max(node.score, _py_propagate(child, evaluator))
    else:
        node.score = float('inf')
        for child in node.children:
            node.score = min(node.score, _py_propagate(child, evaluator))
    return node.score


def _py_outcome_weight(ply, total):
    if total <= 0:
        return 0.5
    return (ply / total) ** 0.5


# ── Self-play ─────────────────────────────────────────────────────────────────
class GameState:
    __slots__ = ['board', 'positions', 'move_count', 'done']
    def __init__(self):
        self.board = chess.Board()
        self.positions = []
        self.move_count = 0
        self.done = False


def play_games_batched(model, device, num_games, depth,
                       concurrent=CONCURRENT_GAMES):
    all_data = []
    games_completed = 0
    max_moves = 200
    max_leaves = min(concurrent * (35 ** depth) + 10000, 2_000_000)

    pbar = tqdm(total=num_games, desc="Self-play (batched)")

    while games_completed < num_games:
        batch_size = min(concurrent, num_games - games_completed)
        games = [GameState() for _ in range(batch_size)]

        while any(not g.done for g in games):
            active = [g for g in games if not g.done]
            if not active:
                break

            evaluator = BatchedEvaluator(model, device, max_leaves=max_leaves)

            if HAVE_CYTHON:
                # ── Cython path: flat C tree ──────────────────────────
                game_trees = []
                for game in active:
                    if game.board.is_game_over() or game.move_count >= max_moves:
                        game.done = True
                        continue
                    maximizing = (game.board.turn == chess.WHITE)
                    tree_wrapper, root_children = cy_expand_tree(
                        game.board, depth, maximizing, evaluator
                    )
                    if root_children:
                        game_trees.append((game, tree_wrapper, root_children, maximizing))
                    else:
                        game.done = True

                if not game_trees:
                    break

                # ONE GPU call
                evaluator.flush()

                # C-speed propagation + best move selection
                for game, tree_wrapper, root_children, maximizing in game_trees:
                    best_move, _ = tree_wrapper.propagate_and_best_move(
                        evaluator, root_children, maximizing
                    )
                    if best_move is None:
                        game.done = True
                        continue
                    if game.move_count >= SKIP_OPENING_PLY:
                        game.positions.append(board_to_vector(game.board))
                    game.board.push(best_move)
                    game.move_count += 1
                    if game.board.is_game_over() or game.move_count >= max_moves:
                        game.done = True

                del game_trees

            else:
                # ── Python fallback path ──────────────────────────────
                game_roots = []
                for game in active:
                    if game.board.is_game_over() or game.move_count >= max_moves:
                        game.done = True
                        continue
                    maximizing = (game.board.turn == chess.WHITE)
                    root = TreeNode(move=None, is_maximizing=maximizing)
                    for move in game.board.legal_moves:
                        game.board.push(move)
                        child = _py_expand_tree(game.board, depth - 1,
                                               not maximizing, evaluator)
                        child.move = move
                        root.children.append(child)
                        game.board.pop()
                    if root.children:
                        game_roots.append((game, root, maximizing))
                    else:
                        game.done = True

                if not game_roots:
                    break

                evaluator.flush()

                for game, root, maximizing in game_roots:
                    best_move = None
                    best_score = float('-inf') if maximizing else float('inf')
                    for child in root.children:
                        score = _py_propagate(child, evaluator)
                        if maximizing and score > best_score:
                            best_score, best_move = score, child.move
                        elif not maximizing and score < best_score:
                            best_score, best_move = score, child.move
                    if best_move is None:
                        game.done = True
                        continue
                    if game.move_count >= SKIP_OPENING_PLY:
                        game.positions.append(board_to_vector(game.board))
                    game.board.push(best_move)
                    game.move_count += 1
                    if game.board.is_game_over() or game.move_count >= max_moves:
                        game.done = True

            del evaluator

        # Harvest outcomes
        ow = cy_outcome_weight if HAVE_CYTHON else _py_outcome_weight
        for game in games:
            result = game.board.result()
            outcome = RESULT_MAP.get(result, 0.0)
            total = len(game.positions)
            for i, vec in enumerate(game.positions):
                all_data.append((vec, outcome * ow(i, total)))
            games_completed += 1
            pbar.update(1)

    pbar.close()
    print(f"Generated {len(all_data)} positions from {num_games} games "
          f"({concurrent} concurrent, depth {depth})")
    return all_data


# ── Phase 1: PGN parsing ─────────────────────────────────────────────────────
def positions_from_pgn(pgn_path, target):
    X = np.zeros((target, 773), dtype=np.float32)
    y = np.zeros(target, dtype=np.float32)
    count = 0
    games_read = 0

    with open(pgn_path, "r", errors="replace") as f:
        pbar = tqdm(total=target, desc="Parsing PGN + labeling")
        while count < target:
            game = chess.pgn.read_game(f)
            if game is None:
                print(f"\nEnd of PGN after {games_read} games.")
                break
            games_read += 1

            result_str = game.headers.get("Result", "*")
            if result_str not in RESULT_MAP:
                continue
            result = RESULT_MAP[result_str]

            if HAVE_CYTHON:
                # Cython path: writes directly into X, y arrays
                board = game.board()
                written = cy_extract_game(
                    board, game.mainline_moves(), result,
                    SKIP_OPENING_PLY, POSITIONS_PER_GAME,
                    X, y, count
                )
                pbar.update(written)
                count += written
            else:
                # Python fallback
                board = game.board()
                boards = []
                plies = []
                for ply, move in enumerate(game.mainline_moves()):
                    board.push(move)
                    if ply >= SKIP_OPENING_PLY and not board.is_game_over():
                        boards.append(board.copy())
                        plies.append(ply)

                if not boards:
                    continue

                total_plies = plies[-1] + 1
                if len(boards) > POSITIONS_PER_GAME:
                    indices = random.sample(range(len(boards)), POSITIONS_PER_GAME)
                else:
                    indices = list(range(len(boards)))

                for idx in indices:
                    if count >= target:
                        break
                    X[count] = board_to_vector(boards[idx])
                    w = _py_outcome_weight(plies[idx], total_plies)
                    y[count] = max(-1.0, min(1.0, result * w))
                    count += 1
                pbar.update(len(indices))

        pbar.close()

    return X[:count], y[:count]


# ── Dataset / Model ───────────────────────────────────────────────────────────
class ChessDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def get_device():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("No GPU detected — using CPU.\n")
    return torch.device("cpu")


def train(model, train_loader, val_loader, epochs, lr, device,
          save_path=MODEL_PATH):
    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)
    criterion = nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=lr,
        steps_per_epoch=len(train_loader), epochs=epochs
    )
    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimiser.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimiser.step()
            scheduler.step()
            train_loss += loss.item() * len(y_b)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b.to(device)),
                                      y_b.to(device)).item() * len(y_b)

        tl = train_loss / len(train_loader.dataset)
        vl = val_loss / len(val_loader.dataset)
        tag = ""
        if vl < best_val:
            best_val = vl
            torch.save(model.state_dict(), save_path)
            tag = "  ← saved"
        print(f"Epoch {epoch:3d}/{epochs}  "
              f"train loss: {tl:.6f}  val loss: {vl:.6f}{tag}")

    model.load_state_dict(torch.load(save_path, map_location=device))
    print(f"Best val loss: {best_val:.6f}")
    return best_val


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = get_device()

    # ── Phase 1 ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("PHASE 1: Game-outcome labeling from PGN")
    print("=" * 60)

    if not os.path.exists(PGN_FILE):
        raise SystemExit(f"PGN file not found: {PGN_FILE}")

    X_np, y_np = positions_from_pgn(PGN_FILE, NUM_POSITIONS)
    print(f"Extracted {len(y_np)} labeled positions.\n")

    if len(y_np) < 1000:
        raise SystemExit("Too few positions — check your PGN file.")

    X_t = torch.from_numpy(X_np)
    y_t = torch.from_numpy(y_np)
    del X_np, y_np
    gc.collect()

    if AUGMENT_COLOR_FLIP:
        X_t, y_t = augment_gpu(X_t, y_t, device)

    n = len(y_t)
    perm = torch.randperm(n)
    X_t = X_t[perm]
    y_t = y_t[perm]
    del perm

    split = int(0.9 * n)
    train_ds = ChessDataset(X_t[:split], y_t[:split].unsqueeze(1))
    val_ds   = ChessDataset(X_t[split:], y_t[split:].unsqueeze(1))
    del X_t, y_t
    gc.collect()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True,
                              persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              num_workers=1, pin_memory=True,
                              persistent_workers=True)

    model = ChessEvaluator()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    train(model, train_loader, val_loader, EPOCHS, LR, device)

    del train_ds, val_ds, train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    if SELF_PLAY_ENABLED:
        print("\n" + "=" * 60)
        print("PHASE 2: Batched self-play refinement")
        print("=" * 60)

        model = model.to(device)

        for rnd in range(1, SELF_PLAY_ROUNDS + 1):
            print(f"\n── Self-play round {rnd}/{SELF_PLAY_ROUNDS} ──")

            sp_data = play_games_batched(
                model, device, SELF_PLAY_GAMES, SELF_PLAY_DEPTH, CONCURRENT_GAMES
            )

            if sp_data:
                X_sp = np.stack([r[0] for r in sp_data])
                y_sp = np.array([r[1] for r in sp_data], dtype=np.float32)
                X_t = torch.from_numpy(X_sp)
                y_t = torch.from_numpy(y_sp)
                del X_sp, y_sp, sp_data

                if AUGMENT_COLOR_FLIP:
                    X_t, y_t = augment_gpu(X_t, y_t, device)

                n = len(y_t)
                perm = torch.randperm(n)
                X_t, y_t = X_t[perm], y_t[perm]
                split = int(0.9 * n)

                sp_train = DataLoader(
                    ChessDataset(X_t[:split], y_t[:split].unsqueeze(1)),
                    batch_size=BATCH_SIZE, shuffle=True)
                sp_val = DataLoader(
                    ChessDataset(X_t[split:], y_t[split:].unsqueeze(1)),
                    batch_size=BATCH_SIZE)

                train(model, sp_train, sp_val, SELF_PLAY_EPOCHS, LR * 0.3, device)
                del X_t, y_t, sp_train, sp_val
                gc.collect()
                torch.cuda.empty_cache()

    print(f"\nFinal model saved to {MODEL_PATH}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
