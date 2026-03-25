"""
train_nn_gpu.py — GPU + Cython accelerated chess NN trainer (memory-optimized)

Memory optimizations:
  • Phase 1 PGN parsing writes directly to mmap'd files on NVMe (~500MB vs 6GB)
  • GPU augmentation writes to a second mmap in chunks (~1.5GB VRAM vs 18GB)
  • Training reads from mmap-backed Dataset (only current batch in RAM)
  • Self-play writes results to mmap, not a growing Python list
  • Peak RAM: ~2-3GB regardless of dataset size

Build Cython first:
    python setup_cython.py build_ext --inplace
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

# Mmap paths (temp files on NVMe)
MMAP_X     = str(BASE_DIR / ".mmap_X.dat")
MMAP_Y     = str(BASE_DIR / ".mmap_y.dat")
MMAP_AUG_X = str(BASE_DIR / ".mmap_aug_X.dat")
MMAP_AUG_Y = str(BASE_DIR / ".mmap_aug_y.dat")
MMAP_SP_X  = str(BASE_DIR / ".mmap_sp_X.dat")
MMAP_SP_Y  = str(BASE_DIR / ".mmap_sp_y.dat")


def cleanup_mmaps():
    for p in [MMAP_X, MMAP_Y, MMAP_AUG_X, MMAP_AUG_Y, MMAP_SP_X, MMAP_SP_Y]:
        if os.path.exists(p):
            os.remove(p)


# ── GPU augmentation → mmap ───────────────────────────────────────────────────
def build_flip_permutation():
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


def augment_to_mmap(src_X_path, src_y_path, n, device,
                    dst_X_path, dst_y_path):
    """
    Read from source mmap, flip on GPU in chunks, write original + flipped
    to destination mmap. Peak RAM: ~1.5GB. Peak VRAM: ~1.5GB.
    Returns count of augmented positions (2 * n).
    """
    src_X = np.memmap(src_X_path, dtype=np.float32, mode='r', shape=(n, 773))
    src_y = np.memmap(src_y_path, dtype=np.float32, mode='r', shape=(n,))

    n2 = n * 2
    dst_X = np.memmap(dst_X_path, dtype=np.float32, mode='w+', shape=(n2, 773))
    dst_y = np.memmap(dst_y_path, dtype=np.float32, mode='w+', shape=(n2,))

    perm = FLIP_PERM.to(device)
    CHUNK = 500_000

    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)

        # Copy originals to first half
        dst_X[start:end] = src_X[start:end]
        dst_y[start:end] = src_y[start:end]

        # Flip on GPU, write to second half
        chunk = torch.from_numpy(np.array(src_X[start:end])).to(device)
        fl = chunk[:, perm]
        fl[:, 772] = 1.0 - fl[:, 772]
        dst_X[n + start:n + end] = fl.cpu().numpy()
        dst_y[n + start:n + end] = -src_y[start:end]
        del chunk, fl

    dst_X.flush()
    dst_y.flush()
    torch.cuda.empty_cache()
    print(f"[GPU augment → disk] {n} → {n2} positions.")
    return n2


def augment_gpu_tensors(X, y, device):
    """Chunked GPU augmentation for in-memory tensors (used for small self-play data)."""
    perm = FLIP_PERM.to(device)
    CHUNK = 500_000
    flipped_X = torch.empty_like(X)
    for start in range(0, len(X), CHUNK):
        end = min(start + CHUNK, len(X))
        chunk = X[start:end].to(device)
        fl = chunk[:, perm]
        fl[:, 772] = 1.0 - fl[:, 772]
        flipped_X[start:end] = fl.cpu()
        del chunk, fl
    torch.cuda.empty_cache()
    return torch.cat([X, flipped_X], dim=0), torch.cat([y, -y], dim=0)


# ── Mmap-backed Dataset ──────────────────────────────────────────────────────
class MmapDataset(Dataset):
    """
    Reads from memory-mapped files. Only the current batch pages into RAM.
    Peak RAM contribution: ~12MB (one batch of 4096 × 773 × 4 bytes).
    """
    def __init__(self, X_path, y_path, n, indices):
        self.X = np.memmap(X_path, dtype=np.float32, mode='r', shape=(n, 773))
        self.y = np.memmap(y_path, dtype=np.float32, mode='r', shape=(n,))
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (torch.from_numpy(self.X[i].copy()),
                torch.tensor(self.y[i], dtype=torch.float32).unsqueeze(0))


class TensorDataset(Dataset):
    """For small in-memory data (self-play)."""
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── Batched GPU evaluator ─────────────────────────────────────────────────────
class BatchedEvaluator:
    def __init__(self, model, device, max_leaves=500_000):
        self.model = model
        self.device = device
        self._max_leaves = max_leaves
        self._buffer = torch.zeros(max_leaves, 773, dtype=torch.float32,
                                   pin_memory=True)
        self._vec_count = 0
        self._scores = []
        self._entries = []

    def reset(self):
        self._vec_count = 0
        self._scores.clear()
        self._entries.clear()

    def enqueue(self, board):
        idx = len(self._entries)
        if board.is_checkmate():
            s = -99999.0 if board.turn == chess.WHITE else 99999.0
            self._entries.append(('terminal', s))
            return idx
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_fifty_moves() or board.is_repetition(3)):
            self._entries.append(('terminal', 0.0))
            return idx
        vi = self._vec_count
        if vi < self._max_leaves:
            vec = board_to_vector(board)
            self._buffer[vi].copy_(torch.from_numpy(vec))
        self._vec_count += 1
        self._entries.append(('gpu', vi))
        return idx

    def enqueue_batch(self, boards):
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

    def get_score(self, idx):
        kind, val = self._entries[idx]
        if kind == 'terminal':
            return val
        return self._scores[val]


# ── Python fallback tree ops ─────────────────────────────────────────────────
class TreeNode:
    __slots__ = ['move', 'children', 'score', 'eval_idx', 'is_maximizing', 'is_leaf']
    def __init__(self, move, is_maximizing):
        self.move = move; self.children = []; self.score = 0.0
        self.eval_idx = -1; self.is_maximizing = is_maximizing; self.is_leaf = False

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
        for c in node.children: node.score = max(node.score, _py_propagate(c, evaluator))
    else:
        node.score = float('inf')
        for c in node.children: node.score = min(node.score, _py_propagate(c, evaluator))
    return node.score

def _py_outcome_weight(ply, total):
    return 0.5 if total <= 0 else (ply / total) ** 0.5


# ── Self-play with mmap output ───────────────────────────────────────────────
class GameState:
    __slots__ = ['board', 'positions', 'move_count', 'done']
    def __init__(self):
        self.board = chess.Board()
        self.positions = []
        self.move_count = 0
        self.done = False


def play_games_batched_mmap(model, device, num_games, depth,
                            sp_X_path, sp_y_path,
                            concurrent=CONCURRENT_GAMES):
    """
    Self-play with results written directly to mmap files.
    Returns the number of positions written.
    """
    max_moves = 200
    max_positions = num_games * max_moves  # upper bound
    max_leaves = min(concurrent * (35 ** depth) + 10000, 2_000_000)

    sp_X = np.memmap(sp_X_path, dtype=np.float32, mode='w+', shape=(max_positions, 773))
    sp_y = np.memmap(sp_y_path, dtype=np.float32, mode='w+', shape=(max_positions,))
    sp_count = 0

    games_completed = 0
    pbar = tqdm(total=num_games, desc="Self-play (batched)")
    ow = cy_outcome_weight if HAVE_CYTHON else _py_outcome_weight

    while games_completed < num_games:
        batch_size = min(concurrent, num_games - games_completed)
        games = [GameState() for _ in range(batch_size)]

        while any(not g.done for g in games):
            active = [g for g in games if not g.done]
            if not active:
                break

            evaluator = BatchedEvaluator(model, device, max_leaves=max_leaves)

            if HAVE_CYTHON:
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

                evaluator.flush()

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

        # Harvest outcomes → write directly to mmap
        for game in games:
            result = game.board.result()
            outcome = RESULT_MAP.get(result, 0.0)
            total = len(game.positions)
            for i, vec in enumerate(game.positions):
                if sp_count < max_positions:
                    sp_X[sp_count] = vec
                    sp_y[sp_count] = outcome * ow(i, total)
                    sp_count += 1
            # Free position vectors immediately
            game.positions.clear()
            games_completed += 1
            pbar.update(1)

    pbar.close()
    sp_X.flush()
    sp_y.flush()

    print(f"Generated {sp_count} positions from {num_games} games "
          f"({concurrent} concurrent, depth {depth})")
    return sp_count


# ── Phase 1: PGN parsing → mmap ──────────────────────────────────────────────
def positions_from_pgn_mmap(pgn_path, target, X_path, y_path):
    """
    Parse PGN and write results directly to mmap files.
    Peak RAM: ~500MB (game objects + temporary boards).
    """
    X = np.memmap(X_path, dtype=np.float32, mode='w+', shape=(target, 773))
    y = np.memmap(y_path, dtype=np.float32, mode='w+', shape=(target,))
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
                board = game.board()
                written = cy_extract_game(
                    board, game.mainline_moves(), result,
                    SKIP_OPENING_PLY, POSITIONS_PER_GAME,
                    X, y, count
                )
                pbar.update(written)
                count += written
            else:
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

                total_plies = plies[len(plies) - 1] + 1
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

    X.flush()
    y.flush()
    print(f"Wrote {count} positions to disk.")
    return count


# ── Model ─────────────────────────────────────────────────────────────────────
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

    # Clean any leftover mmap files from previous runs
    cleanup_mmaps()

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 1: PGN → mmap → augment on GPU → mmap → train from mmap
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("PHASE 1: Game-outcome labeling from PGN")
    print("=" * 60)

    if not os.path.exists(PGN_FILE):
        raise SystemExit(f"PGN file not found: {PGN_FILE}")

    # Step 1a: Parse PGN → mmap (~500MB RAM, data goes to NVMe)
    n_raw = positions_from_pgn_mmap(PGN_FILE, NUM_POSITIONS, MMAP_X, MMAP_Y)

    if n_raw < 1000:
        raise SystemExit("Too few positions — check your PGN file.")

    # Step 1b: Augment on GPU → second mmap (~1.5GB VRAM peak)
    if AUGMENT_COLOR_FLIP:
        n_train = augment_to_mmap(MMAP_X, MMAP_Y, n_raw, device,
                                  MMAP_AUG_X, MMAP_AUG_Y)
        train_X_path, train_y_path = MMAP_AUG_X, MMAP_AUG_Y
    else:
        n_train = n_raw
        train_X_path, train_y_path = MMAP_X, MMAP_Y

    # Step 1c: Build mmap-backed DataLoaders (~12MB RAM per batch)
    all_indices = list(range(n_train))
    random.shuffle(all_indices)
    split = int(0.9 * n_train)

    train_loader = DataLoader(
        MmapDataset(train_X_path, train_y_path, n_train, all_indices[:split]),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        MmapDataset(train_X_path, train_y_path, n_train, all_indices[split:]),
        batch_size=BATCH_SIZE,
        num_workers=2, pin_memory=True, persistent_workers=True
    )
    del all_indices
    gc.collect()

    # Step 1d: Train
    model = ChessEvaluator()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    train(model, train_loader, val_loader, EPOCHS, LR, device)

    del train_loader, val_loader
    gc.collect()
    torch.cuda.empty_cache()

    # Clean Phase 1 mmaps (no longer needed)
    for p in [MMAP_X, MMAP_Y, MMAP_AUG_X, MMAP_AUG_Y]:
        if os.path.exists(p):
            os.remove(p)

    # ══════════════════════════════════════════════════════════════════════════
    #  PHASE 2: Self-play → mmap → augment → train
    # ══════════════════════════════════════════════════════════════════════════
    if SELF_PLAY_ENABLED:
        print("\n" + "=" * 60)
        print("PHASE 2: Batched self-play refinement")
        print("=" * 60)

        model = model.to(device)

        for rnd in range(1, SELF_PLAY_ROUNDS + 1):
            print(f"\n── Self-play round {rnd}/{SELF_PLAY_ROUNDS} ──")

            # Step 2a: Self-play → mmap
            sp_count = play_games_batched_mmap(
                model, device, SELF_PLAY_GAMES, SELF_PLAY_DEPTH,
                MMAP_SP_X, MMAP_SP_Y, CONCURRENT_GAMES
            )

            if sp_count < 100:
                print(f"Too few self-play positions ({sp_count}), skipping round.")
                continue

            # Step 2b: Augment self-play data
            if AUGMENT_COLOR_FLIP:
                n_sp = augment_to_mmap(MMAP_SP_X, MMAP_SP_Y, sp_count, device,
                                       MMAP_AUG_X, MMAP_AUG_Y)
                sp_X_path, sp_y_path = MMAP_AUG_X, MMAP_AUG_Y
            else:
                n_sp = sp_count
                sp_X_path, sp_y_path = MMAP_SP_X, MMAP_SP_Y

            # Step 2c: Train from mmap
            sp_indices = list(range(n_sp))
            random.shuffle(sp_indices)
            sp_split = int(0.9 * n_sp)

            sp_train = DataLoader(
                MmapDataset(sp_X_path, sp_y_path, n_sp, sp_indices[:sp_split]),
                batch_size=BATCH_SIZE, shuffle=True
            )
            sp_val = DataLoader(
                MmapDataset(sp_X_path, sp_y_path, n_sp, sp_indices[sp_split:]),
                batch_size=BATCH_SIZE
            )

            train(model, sp_train, sp_val, SELF_PLAY_EPOCHS, LR * 0.3, device)

            del sp_train, sp_val, sp_indices
            gc.collect()
            torch.cuda.empty_cache()

            # Clean round mmaps
            for p in [MMAP_SP_X, MMAP_SP_Y, MMAP_AUG_X, MMAP_AUG_Y]:
                if os.path.exists(p):
                    os.remove(p)

    # Final cleanup
    cleanup_mmaps()

    print(f"\nFinal model saved to {MODEL_PATH}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")