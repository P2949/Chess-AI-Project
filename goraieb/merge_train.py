"""
merge_train.py — Merge Stockfish + game-outcome training data (optimized)

Optimizations:
  • GPU-accelerated color-flip augmentation
  • Pinned memory DataLoaders with num_workers
  • Memory-efficient tensor construction
"""

import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

OUTCOME_MODEL_PATH  = str(BASE_DIR / "model_outcome.pt")
STOCKFISH_DATA_PATH = str(BASE_DIR / "stockfish_data.npz")
OUTCOME_DATA_PATH   = str(BASE_DIR / "outcome_data.npz")
MERGED_MODEL_PATH   = str(BASE_DIR / "model.pt")

BATCH_SIZE     = 4096
WEIGHT_DECAY   = 1e-4
FINETUNE_EPOCHS = 30
FINETUNE_LR     = 5e-5
JOINT_EPOCHS    = 50
JOINT_LR        = 2e-4
SF_WEIGHT       = 3.0


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

def augment_gpu(X_np, y_np, device):
    perm = FLIP_PERM.to(device)
    CHUNK = 500_000
    X = torch.from_numpy(X_np)
    flipped_X = torch.empty_like(X)
    for start in range(0, len(X), CHUNK):
        end = min(start + CHUNK, len(X))
        chunk = X[start:end].to(device)
        fl = chunk[:, perm]
        fl[:, 772] = 1.0 - fl[:, 772]
        flipped_X[start:end] = fl.cpu()
        del chunk, fl
    torch.cuda.empty_cache()
    X_out = torch.cat([X, flipped_X], dim=0).numpy()
    y_out = np.concatenate([y_np, -y_np])
    del X, flipped_X
    print(f"[GPU augment] {len(y_out) // 2} → {len(y_out)}")
    return X_out, y_out


class ChessDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X) if isinstance(X, np.ndarray) else X
        self.y = torch.from_numpy(y).unsqueeze(1) if isinstance(y, np.ndarray) else y
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


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
    def forward(self, x): return self.net(x)


def get_device():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    return torch.device("cpu")


def train(model, train_loader, val_loader, epochs, lr, device,
          save_path, label=""):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    crit = nn.SmoothL1Loss()
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)

    best_val = float('inf')
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
        print(f"  {label} Epoch {epoch:3d}/{epochs}  train: {tl:.6f}  val: {vl:.6f}{tag}")

    model.load_state_dict(torch.load(save_path, map_location=device))
    return best_val


def load_data(path):
    d = np.load(path)
    print(f"Loaded {len(d['y'])} samples from {path}")
    return d['X'], d['y']


def make_loaders(X, y, bs=BATCH_SIZE):
    n = len(y)
    idx = list(range(n)); random.shuffle(idx)
    s = int(n * 0.9)
    return (
        DataLoader(ChessDataset(X[idx[:s]], y[idx[:s]]), batch_size=bs,
                   shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True),
        DataLoader(ChessDataset(X[idx[s:]], y[idx[s:]]), batch_size=bs,
                   num_workers=1, pin_memory=True, persistent_workers=True)
    )


def strategy_sequential(device):
    print("=" * 60)
    print("STRATEGY: Sequential (outcome → Stockfish)")
    print("=" * 60)

    model = ChessEvaluator()

    if os.path.exists(OUTCOME_DATA_PATH):
        X, y = load_data(OUTCOME_DATA_PATH)
        X, y = augment_gpu(X, y, device)
        print(f"\nStep 1: {len(y)} outcome-labeled positions...")
        tl, vl = make_loaders(X, y)
        train(model, tl, vl, 40, 3e-4, device, MERGED_MODEL_PATH, "[outcome]")
        del X, y, tl, vl; gc.collect(); torch.cuda.empty_cache()
    elif os.path.exists(OUTCOME_MODEL_PATH):
        model.load_state_dict(torch.load(OUTCOME_MODEL_PATH, map_location=device))
        print(f"Loaded outcome model from {OUTCOME_MODEL_PATH}")

    if os.path.exists(STOCKFISH_DATA_PATH):
        X, y = load_data(STOCKFISH_DATA_PATH)
        X, y = augment_gpu(X, y, device)
        print(f"\nStep 2: {len(y)} Stockfish-labeled positions...")
        tl, vl = make_loaders(X, y)
        train(model, tl, vl, FINETUNE_EPOCHS, FINETUNE_LR, device,
              MERGED_MODEL_PATH, "[stockfish]")
        del X, y, tl, vl; gc.collect(); torch.cuda.empty_cache()

    print(f"\nFinal model saved to {MERGED_MODEL_PATH}")
    return model


def strategy_finetune(device):
    print("STRATEGY: Fine-tune outcome model on Stockfish data")
    model = ChessEvaluator()
    if os.path.exists(OUTCOME_MODEL_PATH):
        model.load_state_dict(torch.load(OUTCOME_MODEL_PATH, map_location=device))
    X, y = load_data(STOCKFISH_DATA_PATH)
    X, y = augment_gpu(X, y, device)
    tl, vl = make_loaders(X, y)
    train(model, tl, vl, FINETUNE_EPOCHS, FINETUNE_LR, device,
          MERGED_MODEL_PATH, "[finetune]")
    return model


def strategy_joint(device):
    print("STRATEGY: Joint training")
    sf_X, sf_y = load_data(STOCKFISH_DATA_PATH)
    oc_X, oc_y = load_data(OUTCOME_DATA_PATH)
    X = np.concatenate([sf_X, oc_X])
    y = np.concatenate([sf_y, oc_y])
    X, y = augment_gpu(X, y, device)
    weights = np.ones(len(y), dtype=np.float64)
    n_sf = len(sf_y) * 2  # doubled by augment
    weights[:n_sf] = SF_WEIGHT

    idx = list(range(len(y))); random.shuffle(idx)
    X, y, weights = X[idx], y[idx], weights[idx]
    s = int(0.9 * len(y))

    sampler = WeightedRandomSampler(weights[:s], num_samples=s, replacement=True)
    tl = DataLoader(ChessDataset(X[:s], y[:s]), batch_size=BATCH_SIZE, sampler=sampler)
    vl = DataLoader(ChessDataset(X[s:], y[s:]), batch_size=BATCH_SIZE)

    model = ChessEvaluator()
    if os.path.exists(OUTCOME_MODEL_PATH):
        model.load_state_dict(torch.load(OUTCOME_MODEL_PATH, map_location=device))
    train(model, tl, vl, JOINT_EPOCHS, JOINT_LR, device, MERGED_MODEL_PATH, "[joint]")
    return model


if __name__ == "__main__":
    device = get_device()

    have_sf = os.path.exists(STOCKFISH_DATA_PATH)
    have_oc = os.path.exists(OUTCOME_DATA_PATH)
    have_om = os.path.exists(OUTCOME_MODEL_PATH)

    print(f"Stockfish data: {'FOUND' if have_sf else 'NOT FOUND'}")
    print(f"Outcome data:   {'FOUND' if have_oc else 'NOT FOUND'}")
    print(f"Outcome model:  {'FOUND' if have_om else 'NOT FOUND'}\n")

    if have_sf and (have_oc or have_om):
        model = strategy_sequential(device)
    elif have_sf and have_om:
        model = strategy_finetune(device)
    elif have_sf and have_oc:
        model = strategy_joint(device)
    elif have_sf:
        X, y = load_data(STOCKFISH_DATA_PATH)
        X, y = augment_gpu(X, y, device)
        model = ChessEvaluator()
        tl, vl = make_loaders(X, y)
        train(model, tl, vl, JOINT_EPOCHS, JOINT_LR, device, MERGED_MODEL_PATH)
    else:
        raise SystemExit("No training data found.")

    print(f"\n{'=' * 60}")
    print(f"Done. Final model: {MERGED_MODEL_PATH}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
