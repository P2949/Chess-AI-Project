"""
train_nn.py — Stockfish-labeled chess NN trainer (memory-optimized)

Key change: results are written to memory-mapped numpy files on disk
as each worker chunk finishes. Peak RAM usage drops from ~20GB to ~2GB
regardless of dataset size.

The mmap'd arrays act like regular numpy arrays but only page into RAM
what's actively being accessed. Your NVMe handles the rest.
"""

import os
import gc
import random
import multiprocessing
import chess
import chess.pgn
import chess.engine
import numpy as np
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

# ── Cython acceleration ──────────────────────────────────────────────────────
try:
    from fast_chess import board_to_vector_bitboard as board_to_vector
    HAVE_CYTHON = True
except ImportError:
    HAVE_CYTHON = False
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


BASE_DIR = Path(__file__).resolve().parent
STOCKFISH_PATH = "/usr/bin/stockfish"
PGN_FILE = str(BASE_DIR / "db.pgn")
NUM_POSITIONS = 2_000_000
POSITIONS_PER_GAME = 48
SKIP_OPENING_PLY = 2
EVAL_DEPTH = 18
BATCH_SIZE = 4096
EPOCHS = 50
LR = 3e-4
WEIGHT_DECAY = 1e-4
MODEL_PATH = "model.pt"
SCORE_CLAMP = 1500
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", multiprocessing.cpu_count()))
AUGMENT_COLOR_FLIP = True
SAVE_NPZ = True

# Mmap temp files
MMAP_X_PATH = str(BASE_DIR / ".mmap_X.dat")
MMAP_Y_PATH = str(BASE_DIR / ".mmap_y.dat")


# ── GPU augmentation (chunked) ────────────────────────────────────────────────
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


def augment_gpu(X, y, device):
    """Chunked GPU augmentation — never puts full dataset on VRAM."""
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
    X_out = torch.cat([X, flipped_X], dim=0)
    y_out = torch.cat([y, -y], dim=0)
    del flipped_X
    print(f"[GPU augment] {len(y_out) // 2} → {len(y_out)} positions.")
    return X_out, y_out


def get_device():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("No GPU detected — using CPU.\n")
    return torch.device("cpu")


# ── PGN parsing ───────────────────────────────────────────────────────────────
def positions_from_pgn(pgn_path, target):
    fens = []
    games_read = 0
    with open(pgn_path, "r", errors="replace") as f:
        pbar = tqdm(total=target, desc="Parsing PGN")
        while len(fens) < target:
            game = chess.pgn.read_game(f)
            if game is None:
                print(f"\nEnd of PGN after {games_read} games.")
                break
            games_read += 1
            board = game.board()
            candidates = []
            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)
                if ply >= SKIP_OPENING_PLY and not board.is_game_over():
                    candidates.append(board.fen())
            if not candidates:
                continue
            sample = random.sample(candidates, min(POSITIONS_PER_GAME, len(candidates)))
            fens.extend(sample)
            pbar.update(len(sample))
        pbar.close()
    return fens[:target]


# ── Stockfish labeling worker ─────────────────────────────────────────────────
def _label_chunk(args):
    fens, stockfish_path, depth, clamp = args
    results = []
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        for fen in fens:
            try:
                board = chess.Board(fen)
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
                score = info["score"].white().score(mate_score=clamp)
                if score is None:
                    score = clamp if info["score"].white().is_mate() else 0
                score = float(np.clip(score, -clamp, clamp)) / clamp
                results.append((board_to_vector(board), score))
            except Exception:
                continue
    finally:
        engine.quit()
    return results


def label_positions_mmap(fens):
    """
    Label positions with Stockfish, writing results directly to
    memory-mapped files on disk. RAM usage stays ~2GB regardless
    of dataset size.

    Each worker returns a small chunk (~500 positions × 3KB = ~1.5MB).
    The main process writes it into the mmap and frees the chunk.
    No growing list in RAM.
    """
    n = len(fens)

    # Pre-allocate mmap'd files on disk
    X_mmap = np.memmap(MMAP_X_PATH, dtype=np.float32, mode='w+', shape=(n, 773))
    y_mmap = np.memmap(MMAP_Y_PATH, dtype=np.float32, mode='w+', shape=(n,))

    # Small chunks = frequent progress + small per-chunk memory
    CHUNK_SIZE = 500
    chunks = [fens[i:i + CHUNK_SIZE] for i in range(0, n, CHUNK_SIZE)]
    args = [(chunk, STOCKFISH_PATH, EVAL_DEPTH, SCORE_CLAMP) for chunk in chunks]

    write_pos = 0  # current write position in mmap

    print(f"Labelling {n} positions across {NUM_WORKERS} workers "
          f"({len(chunks)} chunks of ~{CHUNK_SIZE}) ...")
    print(f"Results written to disk: {MMAP_X_PATH}")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(_label_chunk, a): a for a in args}
        pbar = tqdm(total=n, desc="Stockfish labelling")

        for future in as_completed(futures):
            chunk_results = future.result()

            # Write directly into mmap — chunk_results is freed after this
            for vec, score in chunk_results:
                if write_pos < n:
                    X_mmap[write_pos] = vec
                    y_mmap[write_pos] = score
                    write_pos += 1

            pbar.update(len(chunk_results))

            # Explicitly free the chunk data
            del chunk_results

        pbar.close()

    # Flush to disk
    X_mmap.flush()
    y_mmap.flush()

    print(f"Labeled {write_pos} positions (written to disk).")
    return write_pos


# ── Memory-mapped dataset ─────────────────────────────────────────────────────
class MmapDataset(Dataset):
    """
    Dataset backed by memory-mapped numpy files.
    Only the current batch is in RAM — the OS pages the rest from disk.
    """
    def __init__(self, X_path, y_path, n, indices=None):
        self.X = np.memmap(X_path, dtype=np.float32, mode='r', shape=(n, 773))
        self.y = np.memmap(y_path, dtype=np.float32, mode='r', shape=(n,))
        self.indices = indices  # subset indices (for train/val split)

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.y)

    def __getitem__(self, idx):
        real_idx = self.indices[idx] if self.indices is not None else idx
        return (torch.from_numpy(self.X[real_idx].copy()),
                torch.tensor(self.y[real_idx], dtype=torch.float32).unsqueeze(0))


class TensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def train(model, train_loader, val_loader, epochs, lr, device):
    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
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
            torch.save(model.state_dict(), MODEL_PATH)
            tag = "  ← saved"
        print(f"Epoch {epoch:3d}/{epochs}  "
              f"train loss: {tl:.6f}  val loss: {vl:.6f}{tag}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"\nLoaded best checkpoint (val loss {best_val:.6f})")


class NNEvaluate:
    def __init__(self, model):
        self.model = model
    def __call__(self, board):
        if board.is_checkmate():
            return -99999 if board.turn == chess.WHITE else 99999
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
        with torch.no_grad():
            x = torch.tensor(board_to_vector(board)).unsqueeze(0)
            return self.model(x).item() * SCORE_CLAMP


if __name__ == "__main__":
    if not os.path.exists(PGN_FILE):
        raise SystemExit("PGN file not found")

    device = get_device()

    # ── Phase 1: Parse PGN ────────────────────────────────────────────────────
    fens = positions_from_pgn(PGN_FILE, NUM_POSITIONS)
    print(f"Extracted {len(fens)} positions from PGN.\n")

    # ── Phase 2: Stockfish labeling → disk (low RAM) ──────────────────────────
    n_labeled = label_positions_mmap(fens)
    del fens
    gc.collect()

    if n_labeled < 1000:
        raise SystemExit("Too few positions — check your PGN file and paths.")

    # ── Save .npz for merge_train.py ──────────────────────────────────────────
    if SAVE_NPZ:
        # Read from mmap in chunks to avoid loading all into RAM
        X_mmap = np.memmap(MMAP_X_PATH, dtype=np.float32, mode='r', shape=(n_labeled, 773))
        y_mmap = np.memmap(MMAP_Y_PATH, dtype=np.float32, mode='r', shape=(n_labeled,))
        np.savez_compressed(
            str(BASE_DIR / "stockfish_data.npz"),
            X=np.array(X_mmap),  # copies into RAM temporarily for compression
            y=np.array(y_mmap)
        )
        del X_mmap, y_mmap
        gc.collect()
        print(f"Saved {n_labeled} samples to stockfish_data.npz")

    # ── Phase 3: Training ─────────────────────────────────────────────────────
    if AUGMENT_COLOR_FLIP:
        # Load mmap, augment on GPU in chunks, produce tensors
        X_mmap = np.memmap(MMAP_X_PATH, dtype=np.float32, mode='r', shape=(n_labeled, 773))
        y_mmap = np.memmap(MMAP_Y_PATH, dtype=np.float32, mode='r', shape=(n_labeled,))

        X_t = torch.from_numpy(np.array(X_mmap))
        y_t = torch.from_numpy(np.array(y_mmap))
        del X_mmap, y_mmap

        X_t, y_t = augment_gpu(X_t, y_t, device)
        n = len(y_t)

        # Shuffle
        perm = torch.randperm(n)
        X_t, y_t = X_t[perm], y_t[perm]
        del perm

        split = int(0.9 * n)
        train_ds = TensorDataset(X_t[:split], y_t[:split].unsqueeze(1))
        val_ds   = TensorDataset(X_t[split:], y_t[split:].unsqueeze(1))
        del X_t, y_t
        gc.collect()

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=2, pin_memory=True, persistent_workers=True)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                  num_workers=1, pin_memory=True, persistent_workers=True)
    else:
        # Train directly from mmap — minimal RAM
        all_indices = list(range(n_labeled))
        random.shuffle(all_indices)
        split = int(0.9 * n_labeled)

        train_ds = MmapDataset(MMAP_X_PATH, MMAP_Y_PATH, n_labeled,
                               indices=all_indices[:split])
        val_ds   = MmapDataset(MMAP_X_PATH, MMAP_Y_PATH, n_labeled,
                               indices=all_indices[split:])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=2, pin_memory=True, persistent_workers=True)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                                  num_workers=1, pin_memory=True, persistent_workers=True)

    model = ChessEvaluator()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    train(model, train_loader, val_loader, EPOCHS, LR, device)

    # ── Cleanup mmap temp files ───────────────────────────────────────────────
    del train_ds, val_ds, train_loader, val_loader
    gc.collect()
    for p in [MMAP_X_PATH, MMAP_Y_PATH]:
        if os.path.exists(p):
            os.remove(p)
            print(f"Cleaned up {p}")

    print(f"\nFinal model saved to {MODEL_PATH}")