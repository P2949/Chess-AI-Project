"""
merge_train.py — Merge Stockfish + game-outcome training data (optimized)

Improvements:
  • Fixed strategy selection logic (elif branches were unreachable)
  • Warm-start from existing model.pt (preserves self-play refinements)
  • Early stopping with patience
  • Cosine annealing with linear warmup
  • CLI args for hyperparameters
  • Data quality logging
"""

import argparse
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
SF_WEIGHT      = 3.0


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
          save_path, label="", patience=15):
    """Train with early stopping and cosine annealing with warmup."""
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    crit = nn.SmoothL1Loss()

    total_steps = len(train_loader) * epochs
    warmup_steps = max(1, total_steps // 20)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_val = float('inf')
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device, non_blocking=True), y_b.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(X_b), y_b)
            loss.backward()
            opt.step()
            sched.step()
            tl += loss.item() * len(y_b)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b = X_b.to(device, non_blocking=True)
                y_b = y_b.to(device, non_blocking=True)
                vl += crit(model(X_b), y_b).item() * len(y_b)

        tl /= len(train_loader.dataset)
        vl /= len(val_loader.dataset)
        tag = ""
        if vl < best_val:
            best_val = vl
            no_improve = 0
            torch.save(model.state_dict(), save_path)
            tag = "  ← saved"
        else:
            no_improve += 1

        lr_now = opt.param_groups[0]['lr']
        print(f"  {label} Epoch {epoch:3d}/{epochs}  "
              f"train: {tl:.6f}  val: {vl:.6f}  lr: {lr_now:.2e}{tag}")

        if patience > 0 and no_improve >= patience:
            print(f"  Early stop: no improvement for {patience} epochs")
            break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return best_val


def load_data(path):
    d = np.load(path)
    X, y = d['X'], d['y']
    print(f"Loaded {len(y)} samples from {path}")
    print(f"  y: mean={np.mean(y):.4f} std={np.std(y):.4f} "
          f"range=[{np.min(y):.3f}, {np.max(y):.3f}]")
    return X, y


def make_loaders(X, y, bs=BATCH_SIZE):
    n = len(y)
    idx = list(range(n))
    random.shuffle(idx)
    s = int(n * 0.9)
    return (
        DataLoader(ChessDataset(X[idx[:s]], y[idx[:s]]), batch_size=bs,
                   shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True),
        DataLoader(ChessDataset(X[idx[s:]], y[idx[s:]]), batch_size=bs,
                   num_workers=1, pin_memory=True, persistent_workers=True)
    )


def load_warm_start(model, device):
    """
    Warm-start from existing model.pt (preserves self-play refinements),
    falls back to outcome model, then random init.
    """
    if os.path.exists(MERGED_MODEL_PATH):
        model.load_state_dict(torch.load(MERGED_MODEL_PATH, map_location=device))
        print(f"[warm-start] Loaded existing model from {MERGED_MODEL_PATH}")
        return True
    elif os.path.exists(OUTCOME_MODEL_PATH):
        model.load_state_dict(torch.load(OUTCOME_MODEL_PATH, map_location=device))
        print(f"[warm-start] Loaded outcome model from {OUTCOME_MODEL_PATH}")
        return True
    else:
        print("[warm-start] No existing model found, training from scratch")
        return False


def strategy_sequential(device, epochs_outcome, epochs_sf, lr_outcome, lr_sf,
                        patience, warm_start):
    print("=" * 60)
    print("STRATEGY: Sequential (outcome → Stockfish)")
    print("=" * 60)

    model = ChessEvaluator()

    if warm_start:
        loaded = load_warm_start(model, device)
    else:
        loaded = False
        if os.path.exists(OUTCOME_MODEL_PATH):
            model.load_state_dict(torch.load(OUTCOME_MODEL_PATH, map_location=device))
            print(f"Loaded outcome model from {OUTCOME_MODEL_PATH}")
            loaded = True

    # Step 1: outcome data (skip if warm-starting from trained model)
    if os.path.exists(OUTCOME_DATA_PATH) and not loaded:
        X, y = load_data(OUTCOME_DATA_PATH)
        X, y = augment_gpu(X, y, device)
        print(f"\nStep 1: {len(y)} outcome-labeled positions...")
        tl, vl = make_loaders(X, y)
        train(model, tl, vl, epochs_outcome, lr_outcome, device,
              MERGED_MODEL_PATH, "[outcome]", patience=patience)
        del X, y, tl, vl; gc.collect(); torch.cuda.empty_cache()

    # Step 2: Stockfish data
    if os.path.exists(STOCKFISH_DATA_PATH):
        X, y = load_data(STOCKFISH_DATA_PATH)
        X, y = augment_gpu(X, y, device)
        print(f"\nStep 2: {len(y)} Stockfish-labeled positions...")
        tl, vl = make_loaders(X, y)
        train(model, tl, vl, epochs_sf, lr_sf, device,
              MERGED_MODEL_PATH, "[stockfish]", patience=patience)
        del X, y, tl, vl; gc.collect(); torch.cuda.empty_cache()

    print(f"\nFinal model saved to {MERGED_MODEL_PATH}")
    return model


def strategy_finetune(device, epochs, lr, patience, warm_start):
    print("=" * 60)
    print("STRATEGY: Fine-tune on Stockfish data")
    print("=" * 60)

    model = ChessEvaluator()
    if warm_start:
        load_warm_start(model, device)
    elif os.path.exists(OUTCOME_MODEL_PATH):
        model.load_state_dict(torch.load(OUTCOME_MODEL_PATH, map_location=device))

    X, y = load_data(STOCKFISH_DATA_PATH)
    X, y = augment_gpu(X, y, device)
    tl, vl = make_loaders(X, y)
    train(model, tl, vl, epochs, lr, device,
          MERGED_MODEL_PATH, "[finetune]", patience=patience)
    return model


def strategy_joint(device, epochs, lr, patience, warm_start):
    print("=" * 60)
    print("STRATEGY: Joint training (SF weighted higher)")
    print("=" * 60)

    sf_X, sf_y = load_data(STOCKFISH_DATA_PATH)
    oc_X, oc_y = load_data(OUTCOME_DATA_PATH)
    X = np.concatenate([sf_X, oc_X])
    y = np.concatenate([sf_y, oc_y])
    X, y = augment_gpu(X, y, device)

    weights = np.ones(len(y), dtype=np.float64)
    n_sf = len(sf_y) * 2
    weights[:n_sf] = SF_WEIGHT

    idx = list(range(len(y)))
    random.shuffle(idx)
    X, y, weights = X[idx], y[idx], weights[idx]
    s = int(0.9 * len(y))

    sampler = WeightedRandomSampler(weights[:s], num_samples=s, replacement=True)
    tl = DataLoader(ChessDataset(X[:s], y[:s]), batch_size=BATCH_SIZE,
                    sampler=sampler, num_workers=2, pin_memory=True,
                    persistent_workers=True)
    vl = DataLoader(ChessDataset(X[s:], y[s:]), batch_size=BATCH_SIZE,
                    num_workers=1, pin_memory=True, persistent_workers=True)

    model = ChessEvaluator()
    if warm_start:
        load_warm_start(model, device)
    elif os.path.exists(OUTCOME_MODEL_PATH):
        model.load_state_dict(torch.load(OUTCOME_MODEL_PATH, map_location=device))

    train(model, tl, vl, epochs, lr, device,
          MERGED_MODEL_PATH, "[joint]", patience=patience)
    return model


def strategy_sf_only(device, epochs, lr, patience, warm_start):
    print("=" * 60)
    print("STRATEGY: Stockfish data only")
    print("=" * 60)

    model = ChessEvaluator()
    if warm_start:
        load_warm_start(model, device)

    X, y = load_data(STOCKFISH_DATA_PATH)
    X, y = augment_gpu(X, y, device)
    tl, vl = make_loaders(X, y)
    train(model, tl, vl, epochs, lr, device,
          MERGED_MODEL_PATH, "[sf-only]", patience=patience)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge-train the chess evaluator NN")
    ap.add_argument("--strategy",
                    choices=["auto", "sequential", "finetune", "joint", "sf_only"],
                    default="auto",
                    help="Training strategy (default: auto-detect)")
    ap.add_argument("--epochs", type=int, default=100,
                    help="Max epochs for main training phase")
    ap.add_argument("--epochs-outcome", type=int, default=40,
                    help="Max epochs for outcome pre-training (sequential)")
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="Learning rate for fine-tuning / SF training")
    ap.add_argument("--lr-outcome", type=float, default=3e-4,
                    help="Learning rate for outcome pre-training")
    ap.add_argument("--patience", type=int, default=15,
                    help="Early stopping patience (0 = disabled)")
    ap.add_argument("--warm-start", action="store_true", default=True,
                    help="Start from existing model.pt if available (default)")
    ap.add_argument("--no-warm-start", action="store_false", dest="warm_start",
                    help="Train from scratch, ignoring existing model.pt")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = ap.parse_args()

    BATCH_SIZE = args.batch_size
    device = get_device()

    have_sf = os.path.exists(STOCKFISH_DATA_PATH)
    have_oc = os.path.exists(OUTCOME_DATA_PATH)
    have_om = os.path.exists(OUTCOME_MODEL_PATH)
    have_mm = os.path.exists(MERGED_MODEL_PATH)

    print(f"Stockfish data : {'FOUND' if have_sf else 'NOT FOUND'}")
    print(f"Outcome data   : {'FOUND' if have_oc else 'NOT FOUND'}")
    print(f"Outcome model  : {'FOUND' if have_om else 'NOT FOUND'}")
    print(f"Existing model : {'FOUND' if have_mm else 'NOT FOUND'}")
    print(f"Warm start     : {'YES' if args.warm_start else 'NO'}\n")

    strategy = args.strategy

    # Auto-detect best strategy from available data
    if strategy == "auto":
        if have_sf and have_oc:
            strategy = "sequential"
        elif have_sf and (have_om or have_mm):
            strategy = "finetune"
        elif have_sf:
            strategy = "sf_only"
        else:
            raise SystemExit("No training data found. Need at least stockfish_data.npz")

    print(f"Selected strategy: {strategy}\n")

    if strategy == "sequential":
        model = strategy_sequential(device, args.epochs_outcome, args.epochs,
                                    args.lr_outcome, args.lr, args.patience,
                                    args.warm_start)
    elif strategy == "finetune":
        model = strategy_finetune(device, args.epochs, args.lr, args.patience,
                                  args.warm_start)
    elif strategy == "joint":
        model = strategy_joint(device, args.epochs, args.lr, args.patience,
                               args.warm_start)
    elif strategy == "sf_only":
        model = strategy_sf_only(device, args.epochs, args.lr, args.patience,
                                 args.warm_start)

    print(f"\n{'=' * 60}")
    print(f"Done. Final model: {MERGED_MODEL_PATH}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")