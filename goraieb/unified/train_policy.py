"""
train_policy.py — Train the policy model used by team_goraieb.py.

What it learns
--------------
The policy model is a move-ordering helper: given a board position after a
candidate move, it assigns a higher score to positions that are more promising
for the side that just moved.

Training sources:
  --npz PATH        Use existing stockfish_data.npz (fastest, recommended)
  --pgn PATH        Sample positions from PGN games, label with teacher search
  --use-selfplay    Generate positions via engine self-play

Outputs:
  policy.pt         state_dict for PolicyEvaluator in team_goraieb.py
  policy_meta.json  training settings

Fixes over original:
  • eng module imported at module level (was missing → NameError)
  • Duplicate flip_vector removed (second shadowed first with wrong impl)
  • Dead code in pairwise_ranking_loss removed
  • NPZ path properly subsets training data (was training on validation)
  • SWA model actually saved (was computed then thrown away)
  • teacher_score docstring placement fixed
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*fast_chess.*")

import argparse
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

import chess
import chess.pgn

# FIX #15: Import engine at module level — many functions use eng.*
import team_goraieb as eng


# ──────────────────────────────────────────────────────────────────────────────
# Compatibility helpers
# ──────────────────────────────────────────────────────────────────────────────

PIECE_ORDER = [
    (chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE), (chess.ROOK, chess.WHITE),
    (chess.QUEEN, chess.WHITE), (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK), (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK), (chess.ROOK, chess.BLACK),
    (chess.QUEEN, chess.BLACK), (chess.KING, chess.BLACK),
]
_FLIP_PLANE = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]


def board_to_vector_fallback(board: chess.Board) -> np.ndarray:
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


# FIX #14: Single correct flip_vector implementation (removed duplicate)
def flip_vector(v: np.ndarray) -> np.ndarray:
    """
    Mirror the board and swap colors.
    Used as augmentation: the flipped position should have the negated label.
    """
    out = np.zeros_like(v)
    for src_plane in range(12):
        dst_plane = _FLIP_PLANE[src_plane]
        for sq in range(64):
            out[dst_plane * 64 + (sq ^ 56)] = v[src_plane * 64 + sq]
    out[768], out[769] = v[770], v[771]
    out[770], out[771] = v[768], v[769]
    out[772] = 1.0 - v[772]
    return out


def board_to_vec(board: chess.Board) -> np.ndarray:
    try:
        return eng.board_to_vector(board)
    except Exception:
        return board_to_vector_fallback(board)


def reset_engine_state() -> None:
    """Clear search state between teacher calls."""
    eng.TRANSPOSITION_TABLE.clear()
    eng._HISTORY.clear()
    eng._KILLERS[:] = [[None, None] for _ in range(64)]
    if hasattr(eng, "EVAL_CACHE"):
        eng.EVAL_CACHE.clear()


# FIX #docstring: docstring was after a function call, moved to correct position
def teacher_score(board: chess.Board, depth: int) -> float:
    """
    Score a board from White's perspective using the current engine's search.
    Higher = better for White, lower = better for Black.
    """
    reset_engine_state()
    maximizing = (board.turn == chess.WHITE)
    return float(eng.minimax(board, depth, float("-inf"), float("inf"), maximizing, 0))


def normalized_target(cp: float, scale: float) -> float:
    """Map centipawn-style scores to a stable [-1, 1] target."""
    return float(np.tanh(cp / scale))


def candidate_moves(board: chess.Board,
                    max_moves: int,
                    include_checks: bool = True,
                    include_captures: bool = True) -> list[chess.Move]:
    """
    Select a manageable set of candidate moves for teacher labeling.
    """
    legal = list(board.legal_moves)
    if len(legal) <= max_moves:
        return legal

    ordered = eng.order_moves(board, legal, ply=0, tt_move=None, root_policy=False)

    chosen: list[chess.Move] = []
    seen: set[chess.Move] = set()

    def add(move: chess.Move) -> None:
        if move not in seen:
            seen.add(move)
            chosen.append(move)

    top_n = min(max_moves, max(8, max_moves // 2))
    for mv in ordered[:top_n]:
        add(mv)

    if include_captures or include_checks:
        for mv in ordered[top_n:]:
            if len(chosen) >= max_moves:
                break
            if include_captures and board.is_capture(mv):
                add(mv)
            elif include_checks and board.gives_check(mv):
                add(mv)

    if len(chosen) < max_moves:
        remainder = [mv for mv in ordered if mv not in seen]
        random.shuffle(remainder)
        for mv in remainder:
            if len(chosen) >= max_moves:
                break
            add(mv)

    return chosen[:max_moves]


# ──────────────────────────────────────────────────────────────────────────────
# Data generation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Sample:
    vec: np.ndarray
    target: float
    weight: float


def _sample_positions_from_game(
    game: chess.pgn.Game,
    sample_plies: int,
    skip_opening_ply: int,
) -> list[chess.Board]:
    board = game.board()

    plies = list(game.mainline_moves())
    if not plies:
        return []

    boards: list[chess.Board] = []
    for ply_idx, move in enumerate(plies):
        board.push(move)
        if ply_idx >= skip_opening_ply and not board.is_game_over():
            boards.append(board.copy(stack=False))

    if not boards:
        return []
    if len(boards) <= sample_plies:
        return boards

    weights = np.linspace(1.0, 2.0, num=len(boards), dtype=np.float64)
    weights /= weights.sum()
    idxs = np.random.choice(len(boards), size=sample_plies, replace=False, p=weights)
    idxs.sort()
    return [boards[i] for i in idxs]


def _teacher_label_position(
    board: chess.Board,
    teacher_depth: int,
    max_moves: int,
    target_scale: float,
    samples_per_position: int,
    random_fraction: float,
) -> list[Sample]:
    """Label a board by scoring its candidate child positions."""
    moves = candidate_moves(board, max_moves=max_moves)
    if not moves:
        return []

    scored_children: list[tuple[np.ndarray, float]] = []
    for move in moves:
        child = board.copy(stack=False)
        child.push(move)
        cp = teacher_score(child, max(1, teacher_depth - 1))
        target = normalized_target(cp, target_scale)
        scored_children.append((board_to_vec(child), target))

    scored_children.sort(key=lambda x: x[1], reverse=True)

    keep = max(1, min(samples_per_position, len(scored_children)))
    top_keep = max(1, int(round(keep * (1.0 - random_fraction))))
    top_keep = min(top_keep, keep)

    selected: list[tuple[np.ndarray, float]] = []
    selected.extend(scored_children[:top_keep])

    remaining = scored_children[top_keep:]
    if remaining and len(selected) < keep:
        k = min(keep - len(selected), len(remaining))
        random.shuffle(remaining)
        selected.extend(remaining[:k])

    if len(scored_children) >= 2:
        top = scored_children[0][1]
        second = scored_children[1][1]
        margin = abs(top - second)
        weight = float(np.clip(0.75 + 1.5 * margin, 0.75, 2.0))
    else:
        weight = 1.0

    out: list[Sample] = []
    for vec, target in selected:
        out.append(Sample(vec=vec, target=target, weight=weight))
        if random.random() < 0.5:
            out.append(Sample(vec=flip_vector(vec), target=-target, weight=weight))

    return out


def generate_from_pgn(
    pgn_path: str,
    position_budget: int,
    sample_plies: int,
    skip_opening_ply: int,
    teacher_depth: int,
    max_moves: int,
    target_scale: float,
    samples_per_position: int,
    random_fraction: float,
    seed: int,
) -> list[Sample]:
    random.seed(seed)
    np.random.seed(seed)

    samples: list[Sample] = []
    games_seen = 0

    with open(pgn_path, "r", errors="replace") as f:
        pbar = tqdm(total=position_budget, desc="PGN positions")
        while len(samples) < position_budget:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games_seen += 1

            boards = _sample_positions_from_game(
                game, sample_plies=sample_plies,
                skip_opening_ply=skip_opening_ply)
            if not boards:
                continue

            for b in boards:
                if len(samples) >= position_budget:
                    break
                samples.extend(_teacher_label_position(
                    b, teacher_depth=teacher_depth, max_moves=max_moves,
                    target_scale=target_scale,
                    samples_per_position=samples_per_position,
                    random_fraction=random_fraction))
                pbar.update(1)

        pbar.close()

    print(f"Loaded {games_seen} PGN games and generated {len(samples)} samples.")
    return samples[:max(1, position_budget)]


def _selfplay_game_positions(
    depth: int, max_plies: int,
    opening_fens: Sequence[str], seed: int,
) -> list[chess.Board]:
    random.seed(seed)
    np.random.seed(seed)

    board = chess.Board(random.choice(list(opening_fens)))
    boards: list[chess.Board] = []

    reset_engine_state()
    for ply in range(max_plies):
        if board.is_game_over():
            break
        move = eng.get_next_move(board, board.turn, depth=depth)
        if move is None or move not in board.legal_moves:
            break
        board.push(move)
        if ply >= 4 and not board.is_game_over():
            boards.append(board.copy(stack=False))

    return boards


def generate_from_selfplay(
    games: int, selfplay_depth: int, max_plies: int,
    opening_fens: Sequence[str], teacher_depth: int,
    max_moves: int, target_scale: float,
    samples_per_position: int, random_fraction: float, seed: int,
) -> list[Sample]:
    random.seed(seed)
    np.random.seed(seed)

    samples: list[Sample] = []
    pbar = tqdm(total=games, desc="Self-play games")

    for i in range(games):
        boards = _selfplay_game_positions(
            depth=selfplay_depth, max_plies=max_plies,
            opening_fens=opening_fens, seed=seed + i)
        for b in boards:
            samples.extend(_teacher_label_position(
                b, teacher_depth=teacher_depth, max_moves=max_moves,
                target_scale=target_scale,
                samples_per_position=samples_per_position,
                random_fraction=random_fraction))
        pbar.update(1)

    pbar.close()
    print(f"Generated {len(samples)} samples from {games} self-play games.")
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# Multiprocessing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _worker_label_fen(args):
    fen, teacher_depth, max_moves, target_scale, samples_per_position, random_fraction, seed = args
    random.seed(seed)
    np.random.seed(seed)
    board = chess.Board(fen)
    return _teacher_label_position(
        board, teacher_depth=teacher_depth, max_moves=max_moves,
        target_scale=target_scale, samples_per_position=samples_per_position,
        random_fraction=random_fraction)


def _fen_positions_from_pgn(
    pgn_path: str, position_budget: int,
    sample_plies: int, skip_opening_ply: int, seed: int,
) -> list[str]:
    random.seed(seed)
    np.random.seed(seed)

    fens: list[str] = []
    games_seen = 0

    with open(pgn_path, "r", errors="replace") as f:
        while len(fens) < position_budget:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games_seen += 1
            boards = _sample_positions_from_game(game, sample_plies, skip_opening_ply)
            for b in boards:
                fens.append(b.fen())
                if len(fens) >= position_budget:
                    break

    print(f"Loaded {games_seen} PGN games and extracted {len(fens)} positions.")
    return fens


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

class PolicyDataset(Dataset):
    def __init__(self, samples: Sequence[Sample] = None,
                 X_arr: np.ndarray = None, y_arr: np.ndarray = None,
                 w_arr: np.ndarray = None, augment=False):
        self.augment = augment
        if samples is not None:
            self.X = torch.tensor(np.stack([s.vec for s in samples]).astype(np.float32))
            self.y = torch.tensor(np.array([s.target for s in samples], dtype=np.float32)).unsqueeze(1)
            self.w = torch.tensor(np.array([s.weight for s in samples], dtype=np.float32)).unsqueeze(1)
            self.is_lazy = False
        else:
            self.X_arr = X_arr
            self.y_arr = y_arr
            self.w_arr = w_arr
            self.is_lazy = True

    def __len__(self) -> int:
        return len(self.y_arr) if self.is_lazy else self.X.shape[0]

    def __getitem__(self, idx: int):
        if self.is_lazy:
            x_np = self.X_arr[idx].copy()
            y_val = float(self.y_arr[idx])
            w_val = float(self.w_arr[idx]) if self.w_arr is not None else 1.0

            if self.augment and random.random() < 0.5:
                x_np = flip_vector(x_np)
                y_val = -y_val

            return (torch.from_numpy(x_np).float(),
                    torch.tensor([y_val], dtype=torch.float32),
                    torch.tensor([w_val], dtype=torch.float32))

        return self.X[idx], self.y[idx], self.w[idx]


# ──────────────────────────────────────────────────────────────────────────────
# Model: must match team_goraieb.PolicyEvaluator
# ──────────────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dims, dims),
            nn.BatchNorm1d(dims),
            nn.ReLU(),
            nn.Linear(dims, dims),
            nn.BatchNorm1d(dims)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class PolicyEvaluator(nn.Module):
    def __init__(self, input_dim: int = 773):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU())
        self.layer1 = ResBlock(512)
        self.layer2 = ResBlock(512)
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)


def get_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# FIX #16: Removed dead code after return statement
def pairwise_ranking_loss(pred: torch.Tensor,
                          target: torch.Tensor,
                          num_pairs: int = 128) -> torch.Tensor:
    """
    Rank higher-scored positions above lower-scored positions.
    Weighted by target difference magnitude to prioritize clear distinctions.
    """
    if pred.numel() < 2:
        return pred.new_tensor(0.0)

    pred = pred.view(-1)
    target = target.view(-1)

    n = pred.shape[0]
    i = torch.randint(0, n, (num_pairs,), device=pred.device)
    j = torch.randint(0, n, (num_pairs,), device=pred.device)

    ti, tj = target[i], target[j]
    pi, pj = pred[i], pred[j]

    diff = ti - tj
    weight = diff.abs().clamp(min=0.1, max=2.0)
    sign = torch.sign(diff)
    score_diff = pi - pj

    loss = F.softplus(-sign * score_diff)
    return (loss * weight).mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    device: torch.device,
    rank_weight: float,
    grad_clip: float,
    amp: bool,
) -> tuple[float, float]:
    model.train()
    mse_sum = 0.0
    rank_sum = 0.0
    total = 0

    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    for X, y, w in tqdm(loader, desc="Train", leave=False):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=amp):
            raw = model(X)
            pred = torch.tanh(raw)
            reg = F.smooth_l1_loss(pred, y, reduction="none")
            reg = (reg * w).mean()
            rank = pairwise_ranking_loss(pred, y, num_pairs=min(256, X.shape[0] * 2))
            loss = reg + rank_weight * rank

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        bs = X.shape[0]
        mse_sum += reg.item() * bs
        rank_sum += rank.item() * bs
        total += bs

    return mse_sum / max(1, total), rank_sum / max(1, total)


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader,
                   device: torch.device) -> dict[str, float]:
    model.eval()
    reg_sum = 0.0
    pair_acc_sum = 0.0
    total = 0
    for X, y, w in tqdm(loader, desc="Val", leave=False):
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        w = w.to(device, non_blocking=True)

        raw = model(X)
        pred = torch.tanh(raw)

        reg = F.smooth_l1_loss(pred, y, reduction="none")
        reg = (reg * w).mean()

        n = pred.shape[0]
        if n >= 2:
            i = torch.randint(0, n, (min(256, n * 2),), device=device)
            j = torch.randint(0, n, (min(256, n * 2),), device=device)
            diff_t = y[i] - y[j]
            diff_p = pred[i] - pred[j]
            mask = diff_t.abs() > 1e-6
            if mask.any():
                pair_acc = ((torch.sign(diff_t[mask]) == torch.sign(diff_p[mask])).float().mean()).item()
            else:
                pair_acc = 0.0
        else:
            pair_acc = 0.0

        bs = X.shape[0]
        reg_sum += reg.item() * bs
        pair_acc_sum += pair_acc * bs
        total += bs

    return {
        "reg_loss": reg_sum / max(1, total),
        "pair_acc": pair_acc_sum / max(1, total),
    }


def build_model() -> nn.Module:
    return PolicyEvaluator()


def split_samples(samples: Sequence[Sample], val_fraction: float,
                  seed: int) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    idxs = list(range(len(samples)))
    rng.shuffle(idxs)
    n_val = max(1, int(round(len(samples) * val_fraction)))
    val_idx = set(idxs[:n_val])
    train = [s for i, s in enumerate(samples) if i not in val_idx]
    val = [s for i, s in enumerate(samples) if i in val_idx]
    return train, val


def save_model(model: nn.Module, save_path: str, meta: dict) -> None:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    meta_path = path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved model to {path}")
    print(f"Saved metadata to {meta_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the policy model")

    src = ap.add_argument_group("data source")
    src.add_argument("--npz", type=str, default=None,
                     help="Load existing stockfish_data.npz")
    src.add_argument("--pgn", type=str, default=None,
                     help="PGN file for position sampling")
    src.add_argument("--use-selfplay", action="store_true",
                     help="Generate positions by self-play")
    src.add_argument("--opening-fens", type=str, default=None,
                     help="File with one FEN per line for self-play starts")

    gen = ap.add_argument_group("generation")
    gen.add_argument("--position-budget", type=int, default=8_000)
    gen.add_argument("--sample-plies", type=int, default=256)
    gen.add_argument("--skip-opening-ply", type=int, default=0)
    gen.add_argument("--teacher-depth", type=int, default=0)
    gen.add_argument("--selfplay-depth", type=int, default=9)
    gen.add_argument("--max-plies", type=int, default=80)
    gen.add_argument("--max-moves-per-position", type=int, default=256)
    gen.add_argument("--samples-per-position", type=int, default=100)
    gen.add_argument("--random-fraction", type=float, default=1.00)
    gen.add_argument("--target-scale", type=float, default=1.0)
    gen.add_argument("--processes", type=int, default=max(1, os.cpu_count() or 1))
    gen.add_argument("--parallel", action="store_true")

    trn = ap.add_argument_group("training")
    trn.add_argument("--batch-size", type=int, default=4096)
    trn.add_argument("--epochs", type=int, default=48)
    trn.add_argument("--lr", type=float, default=1e-4)
    trn.add_argument("--weight-decay", type=float, default=1e-3)
    trn.add_argument("--rank-weight", type=float, default=0.9)
    trn.add_argument("--grad-clip", type=float, default=1.0)
    trn.add_argument("--val-fraction", type=float, default=0.1)
    trn.add_argument("--num-workers", type=int, default=1)
    trn.add_argument("--device", type=str, default=None)
    trn.add_argument("--seed", type=int, default=1337)

    out = ap.add_argument_group("output")
    out.add_argument("--save-path", type=str,
                     default=str(Path(__file__).resolve().parent / "policy.pt"))

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = get_device(args.device)
    amp = device.type == "cuda"
    if amp:
        torch.set_float32_matmul_precision("high")

    # ── Load or generate samples ─────────────────────────────────────────────
    if args.npz:
        print(f"Opening {args.npz} with memory mapping...")
        data = np.load(args.npz, mmap_mode='r')
        X_all = data['X']
        y_all = data['y']

        total_available = len(y_all)
        budget = min(args.position_budget, total_available)
        print(f"Dataset contains {total_available} positions. Using budget of {budget}.")

        indices = np.arange(total_available)
        np.random.shuffle(indices)
        indices = indices[:budget]

        n_val = max(1, int(round(len(indices) * args.val_fraction)))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        # FIX #17: Both train and val are properly subsetted
        # FIX #18: Use None for w_arr, return 1.0 in __getitem__ instead
        full_ds_train = PolicyDataset(X_arr=X_all, y_arr=y_all, w_arr=None,
                                      augment=True)
        full_ds_val = PolicyDataset(X_arr=X_all, y_arr=y_all, w_arr=None,
                                    augment=False)
        train_ds = Subset(full_ds_train, train_idx)
        val_ds = Subset(full_ds_val, val_idx)

        print(f"Train samples: {len(train_idx)}")
        print(f"Val samples:   {len(val_idx)}")

    else:
        samples: list[Sample] = []
        if args.pgn:
            if args.parallel:
                fens = _fen_positions_from_pgn(
                    args.pgn, position_budget=args.position_budget,
                    sample_plies=args.sample_plies,
                    skip_opening_ply=args.skip_opening_ply, seed=args.seed)
                jobs = [
                    (fen, args.teacher_depth, args.max_moves_per_position,
                     args.target_scale, args.samples_per_position,
                     args.random_fraction, args.seed + i)
                    for i, fen in enumerate(fens)
                ]
                print(f"Labeling {len(jobs)} positions with {args.processes} workers...")
                with ProcessPoolExecutor(max_workers=args.processes) as ex:
                    for chunk in tqdm(ex.map(_worker_label_fen, jobs),
                                     total=len(jobs), desc="Labeling"):
                        samples.extend(chunk)
            else:
                samples = generate_from_pgn(
                    pgn_path=args.pgn, position_budget=args.position_budget,
                    sample_plies=args.sample_plies,
                    skip_opening_ply=args.skip_opening_ply,
                    teacher_depth=args.teacher_depth,
                    max_moves=args.max_moves_per_position,
                    target_scale=args.target_scale,
                    samples_per_position=args.samples_per_position,
                    random_fraction=args.random_fraction, seed=args.seed)
        elif args.use_selfplay:
            opening_fens = [chess.STARTING_FEN]
            if args.opening_fens:
                path = Path(args.opening_fens)
                if path.exists():
                    opening_fens = [ln.strip() for ln in path.read_text().splitlines()
                                    if ln.strip()]
            samples = generate_from_selfplay(
                games=max(1, args.position_budget // max(1, args.samples_per_position)),
                selfplay_depth=args.selfplay_depth, max_plies=args.max_plies,
                opening_fens=opening_fens, teacher_depth=args.teacher_depth,
                max_moves=args.max_moves_per_position, target_scale=args.target_scale,
                samples_per_position=args.samples_per_position,
                random_fraction=args.random_fraction, seed=args.seed)
        else:
            raise SystemExit("Choose a source: --npz PATH, --pgn PATH, or --use-selfplay")

        if len(samples) < 32:
            raise SystemExit(f"Not enough samples collected: {len(samples)}")

        train_samples, val_samples = split_samples(samples, args.val_fraction,
                                                   args.seed)
        print(f"Train samples: {len(train_samples)}")
        print(f"Val samples:   {len(val_samples)}")

        train_ds = PolicyDataset(train_samples)
        val_ds = PolicyDataset(val_samples)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), drop_last=False)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=steps_per_epoch, epochs=args.epochs)

    best_val = float("inf")
    best_path = Path(args.save_path).with_name(
        Path(args.save_path).stem + ".best.pt")
    history: list[dict] = []

    # SWA setup
    swa_model = torch.optim.swa_utils.AveragedModel(model)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer,
                                                  swa_lr=args.lr * 0.1)
    swa_start = int(args.epochs * 0.75)

    for epoch in range(1, args.epochs + 1):
        train_reg, train_rank = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            scheduler=scheduler, device=device,
            rank_weight=args.rank_weight, grad_clip=args.grad_clip, amp=amp)

        val_metrics = evaluate_model(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_reg": train_reg,
            "train_rank": train_rank,
            **val_metrics,
        })

        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_reg={train_reg:.4f} train_rank={train_rank:.4f} | "
            f"val_reg={val_metrics['reg_loss']:.4f} "
            f"pair_acc={val_metrics['pair_acc']:.3f}")

        if val_metrics["reg_loss"] < best_val:
            best_val = val_metrics["reg_loss"]
            torch.save(model.state_dict(), best_path)
            print(f"  saved best -> {best_path}")

    # Update batch norm stats for SWA model
    torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)

    # FIX #19: Save the SWA-averaged model (was saving raw model before)
    save_model(
        swa_model.module,
        args.save_path,
        meta={
            "arch": "PolicyEvaluator(773 -> 512 -> ResBlock(512) x2 -> 128 -> 1)",
            "source": "npz" if args.npz else ("pgn" if args.pgn else "selfplay"),
            "teacher_depth": args.teacher_depth,
            "target_scale": args.target_scale,
            "rank_weight": args.rank_weight,
            "best_val_reg_loss": best_val,
            "epochs_trained": len(history),
            "swa_start": swa_start,
            "seed": args.seed,
        },
    )

    if best_path.exists():
        print(f"Best checkpoint (non-SWA) kept at {best_path}")
        print(f"SWA-averaged model saved at {args.save_path}")
        print("Tip: test both and keep whichever performs better in games.")


if __name__ == "__main__":
    main()