"""
train_nn.py — Stockfish-labeled chess NN trainer (optimized)

Optimizations over original:
  • Small chunks (500 positions) for visible progress during Stockfish labeling
  • Cython board_to_vector in worker processes (if available)
  • GPU-accelerated color-flip augmentation
  • Saves stockfish_data.npz for merge_train.py
  • Memory-efficient dataset construction
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

# ── Cython acceleration (optional) ───────────────────────────────────────────
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
SAVE_NPZ = True  # save data for merge_train.py


# ── GPU augmentation ──────────────────────────────────────────────────────────
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


def _label_chunk(args):
    """Worker function — each process labels a small chunk of positions."""
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


def label_positions_parallel(fens):
    """Small chunks = frequent progress updates (not stuck at 0% for days)."""
    CHUNK_SIZE = 500
    chunks = [fens[i:i + CHUNK_SIZE] for i in range(0, len(fens), CHUNK_SIZE)]
    args = [(chunk, STOCKFISH_PATH, EVAL_DEPTH, SCORE_CLAMP) for chunk in chunks]
    data = []
    print(f"Labelling {len(fens)} positions across {NUM_WORKERS} workers "
          f"({len(chunks)} chunks of ~{CHUNK_SIZE}) ...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(_label_chunk, a): a for a in args}
        pbar = tqdm(total=len(fens), desc="Stockfish labelling")
        for future in as_completed(futures):
            chunk_results = future.result()
            data.extend(chunk_results)
            pbar.update(len(chunk_results))
        pbar.close()
    return data


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

    fens = positions_from_pgn(PGN_FILE, NUM_POSITIONS)
    print(f"Extracted {len(fens)} positions from PGN.\n")
    data = label_positions_parallel(fens)
    del fens

    if len(data) < 1000:
        raise SystemExit("Too few positions — check your PGN file and paths.")

    # Build numpy arrays and free list
    X = np.stack([r[0] for r in data])
    y = np.array([r[1] for r in data], dtype=np.float32)
    del data
    gc.collect()

    # Save for merge_train.py
    if SAVE_NPZ:
        np.savez_compressed(str(BASE_DIR / "stockfish_data.npz"), X=X, y=y)
        print(f"Saved {len(y)} samples to stockfish_data.npz")

    # Convert to tensors
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    del X, y
    gc.collect()

    # GPU augmentation
    if AUGMENT_COLOR_FLIP:
        X_t, y_t = augment_gpu(X_t, y_t, device)

    # Shuffle
    n = len(y_t)
    perm = torch.randperm(n)
    X_t, y_t = X_t[perm], y_t[perm]
    del perm

    split = int(0.9 * n)
    train_ds = ChessDataset(X_t[:split], y_t[:split].unsqueeze(1))
    val_ds   = ChessDataset(X_t[split:], y_t[split:].unsqueeze(1))
    del X_t, y_t
    gc.collect()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              num_workers=1, pin_memory=True, persistent_workers=True)

    model = ChessEvaluator()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    train(model, train_loader, val_loader, EPOCHS, LR, device)

    print(f"\nFinal model saved to {MODEL_PATH}")
