"""
train_nnue.py

Small training script for the BlunderBus eval net.
By default it resumes from existing output weights.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import random
import shutil
import sys
import time

import chess
import chess.engine

try:
    import numpy as np
except Exception as exc:  # pragma: no cover
    raise SystemExit("numpy is required for training.") from exc


FEATURE_ORDER = (
    (chess.PAWN, chess.WHITE),
    (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE),
    (chess.ROOK, chess.WHITE),
    (chess.QUEEN, chess.WHITE),
    (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK),
    (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.BLACK),
    (chess.QUEEN, chess.BLACK),
    (chess.KING, chess.BLACK),
)
FEATURE_BASE = {(pt, color): i * 64 for i, (pt, color) in enumerate(FEATURE_ORDER)}
FEATURE_DIM = 12 * 64 + 5
CASTLE_WK = 12 * 64 + 0
CASTLE_WQ = 12 * 64 + 1
CASTLE_BK = 12 * 64 + 2
CASTLE_BQ = 12 * 64 + 3
SIDE_TO_MOVE = 12 * 64 + 4


def build_mirror_feature_perm() -> np.ndarray:
    perm = np.arange(FEATURE_DIM, dtype=np.int32)

    # Swap white/black channels for each piece-square bucket.
    for piece_slot in range(6):
        white_base = piece_slot * 64
        black_base = (piece_slot + 6) * 64
        for sq in range(64):
            perm[white_base + sq] = black_base + sq
            perm[black_base + sq] = white_base + sq

    # Swap castling rights by color.
    perm[CASTLE_WK] = CASTLE_BK
    perm[CASTLE_WQ] = CASTLE_BQ
    perm[CASTLE_BK] = CASTLE_WK
    perm[CASTLE_BQ] = CASTLE_WQ

    # Side-to-move handled separately.
    perm[SIDE_TO_MOVE] = SIDE_TO_MOVE
    return perm


MIRROR_FEATURE_PERM = build_mirror_feature_perm()


def mirror_feature_matrix(x: np.ndarray) -> np.ndarray:
    x_mirror = x[:, MIRROR_FEATURE_PERM].copy()
    x_mirror[:, SIDE_TO_MOVE] = 1.0 - x[:, SIDE_TO_MOVE]
    return x_mirror


def feature_index(piece: chess.Piece, square: int) -> int:
    base = FEATURE_BASE[(piece.piece_type, piece.color)]
    mapped_square = chess.square_mirror(square) if piece.color == chess.WHITE else square
    return base + mapped_square


def active_features(board: chess.Board) -> list[int]:
    feats = [feature_index(piece, sq) for sq, piece in board.piece_map().items()]
    if board.has_kingside_castling_rights(chess.WHITE):
        feats.append(CASTLE_WK)
    if board.has_queenside_castling_rights(chess.WHITE):
        feats.append(CASTLE_WQ)
    if board.has_kingside_castling_rights(chess.BLACK):
        feats.append(CASTLE_BK)
    if board.has_queenside_castling_rights(chess.BLACK):
        feats.append(CASTLE_BQ)
    if board.turn == chess.WHITE:
        feats.append(SIDE_TO_MOVE)
    return feats


# Position sampling.
def weighted_random_move(board: chess.Board) -> chess.Move:
    legal = list(board.legal_moves)
    captures = [m for m in legal if board.is_capture(m)]
    checks = [m for m in legal if board.gives_check(m)]
    promotions = [m for m in legal if m.promotion]

    if promotions and random.random() < 0.50:
        return random.choice(promotions)
    if captures and random.random() < 0.45:
        return random.choice(captures)
    if checks and random.random() < 0.30:
        return random.choice(checks)
    return random.choice(legal)


def parse_csv_list(raw: str | None) -> list[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_sampler_modules(module_names: list[str]) -> list[object]:
    samplers: list[object] = []
    for name in module_names:
        mod = importlib.import_module(name)
        if not hasattr(mod, "get_next_move"):
            raise SystemExit(f"Sampler module does not expose get_next_move(): {name}")
        samplers.append(mod)
    if not samplers:
        raise SystemExit("Need at least one sampler module.")
    return samplers


def normalize_sampler_weights(raw: str | None, module_count: int) -> list[float]:
    if raw is None:
        return [1.0 for _ in range(module_count)]
    parts = parse_csv_list(raw)
    if len(parts) != module_count:
        raise SystemExit(
            "sampler weights count must match samplers count "
            f"({len(parts)} != {module_count})."
        )
    vals: list[float] = []
    for p in parts:
        v = float(p)
        if v <= 0.0:
            raise SystemExit("sampler weights must be > 0.")
        vals.append(v)
    return vals


def progress_tick(
    phase: str,
    done: int,
    total: int,
    started_at: float,
    last_log_at: float,
    log_every_seconds: float,
    force: bool = False,
) -> float:
    if log_every_seconds <= 0.0:
        return last_log_at

    now = time.perf_counter()
    if not force and (now - last_log_at) < log_every_seconds:
        return last_log_at

    elapsed = max(1e-9, now - started_at)
    done = max(0, done)
    total = max(total, done)
    rate = done / elapsed
    pct = (100.0 * done / total) if total > 0 else 0.0
    remaining = max(0, total - done)
    eta = (remaining / rate) if rate > 1e-9 else 0.0

    print(
        f"[{phase}] {done}/{total} ({pct:.1f}%) "
        f"| rate={rate:.1f}/s "
        f"| elapsed={format_duration(elapsed)} "
        f"| eta~{format_duration(eta)}"
    )
    return now


def sample_positions(
    samplers: list[object],
    sampler_weights: list[float],
    samples: int,
    min_plies: int,
    max_plies: int,
    teacher_prob: float,
    opening_random_plies: int,
    dedupe_positions: bool,
    log_every_seconds: float = 0.0,
) -> list[chess.Board]:
    positions: list[chess.Board] = []
    seen_hashes: set[int] = set()
    sampling_t0 = time.perf_counter()
    last_log = sampling_t0

    while len(positions) < samples:
        board = chess.Board()
        # Random opening rollout makes training less tied to one teacher's opening tree.
        open_plies = random.randint(0, max(0, opening_random_plies))
        for _ in range(open_plies):
            if board.is_game_over():
                break
            board.push(weighted_random_move(board))

        game_len = random.randint(min_plies, max_plies)

        for ply in range(game_len):
            if board.is_game_over():
                break

            legal = list(board.legal_moves)
            move = None

            if random.random() < teacher_prob:
                try:
                    teacher = random.choices(samplers, weights=sampler_weights, k=1)[0]
                    cand = teacher.get_next_move(board, board.turn, depth=1)
                    if cand in legal:
                        move = cand
                except Exception:
                    move = None

            if move is None:
                move = weighted_random_move(board)
            board.push(move)

            if ply >= 4 and not board.is_game_over() and random.random() < 0.42:
                if dedupe_positions:
                    key = chess.polyglot.zobrist_hash(board)
                    if key in seen_hashes:
                        continue
                    seen_hashes.add(key)
                positions.append(board.copy(stack=False))
                last_log = progress_tick(
                    phase="sampling",
                    done=len(positions),
                    total=samples,
                    started_at=sampling_t0,
                    last_log_at=last_log,
                    log_every_seconds=log_every_seconds,
                )
                if len(positions) >= samples:
                    break

    progress_tick(
        phase="sampling",
        done=len(positions),
        total=samples,
        started_at=sampling_t0,
        last_log_at=last_log,
        log_every_seconds=log_every_seconds,
        force=True,
    )
    return positions


def build_feature_matrix(positions: list[chess.Board], log_every_seconds: float = 0.0) -> np.ndarray:
    x = np.zeros((len(positions), FEATURE_DIM), dtype=np.float32)
    feature_t0 = time.perf_counter()
    last_log = feature_t0
    total = len(positions)
    for i, board in enumerate(positions):
        x[i, active_features(board)] = 1.0
        last_log = progress_tick(
            phase="features",
            done=i + 1,
            total=total,
            started_at=feature_t0,
            last_log_at=last_log,
            log_every_seconds=log_every_seconds,
        )
    progress_tick(
        phase="features",
        done=total,
        total=total,
        started_at=feature_t0,
        last_log_at=last_log,
        log_every_seconds=log_every_seconds,
        force=True,
    )
    return x


def labels_from_module(
    teacher,
    positions: list[chess.Board],
    cp_clamp: float,
    log_every_seconds: float = 0.0,
) -> np.ndarray:
    y = np.zeros((len(positions),), dtype=np.float32)
    label_t0 = time.perf_counter()
    last_log = label_t0
    total = len(positions)
    for i, board in enumerate(positions):
        raw = float(teacher.evaluate(board))
        y[i] = max(-cp_clamp, min(cp_clamp, raw))
        last_log = progress_tick(
            phase="labels(module)",
            done=i + 1,
            total=total,
            started_at=label_t0,
            last_log_at=last_log,
            log_every_seconds=log_every_seconds,
        )
    progress_tick(
        phase="labels(module)",
        done=total,
        total=total,
        started_at=label_t0,
        last_log_at=last_log,
        log_every_seconds=log_every_seconds,
        force=True,
    )
    return y


def labels_from_stockfish(
    positions: list[chess.Board],
    stockfish_path: str,
    stockfish_depth: int,
    stockfish_depth_min: int | None,
    cp_clamp: float,
    log_every_seconds: float = 0.0,
) -> np.ndarray:
    y = np.zeros((len(positions),), dtype=np.float32)
    label_t0 = time.perf_counter()
    last_log = label_t0
    total = len(positions)
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        for i, board in enumerate(positions):
            depth = stockfish_depth
            if stockfish_depth_min is not None:
                depth = random.randint(stockfish_depth_min, stockfish_depth)
            info = engine.analyse(board, chess.engine.Limit(depth=depth))
            score = info["score"].white().score(mate_score=int(cp_clamp))
            if score is None:
                score = cp_clamp if info["score"].white().is_mate() else 0.0
            y[i] = max(-cp_clamp, min(cp_clamp, float(score)))
            last_log = progress_tick(
                phase="labels(stockfish)",
                done=i + 1,
                total=total,
                started_at=label_t0,
                last_log_at=last_log,
                log_every_seconds=log_every_seconds,
            )
    finally:
        engine.quit()
    progress_tick(
        phase="labels(stockfish)",
        done=total,
        total=total,
        started_at=label_t0,
        last_log_at=last_log,
        log_every_seconds=log_every_seconds,
        force=True,
    )
    return y


def maybe_mirror_augment(
    positions: list[chess.Board],
    labels_cp: np.ndarray,
    enable: bool,
) -> tuple[list[chess.Board], np.ndarray]:
    if not enable or len(positions) == 0:
        return positions, labels_cp

    mirrored_positions = [board.mirror() for board in positions]
    mirrored_labels = -labels_cp
    merged_positions = positions + mirrored_positions
    merged_labels = np.concatenate((labels_cp, mirrored_labels), axis=0).astype(np.float32)
    return merged_positions, merged_labels


def pack_dataset(features: np.ndarray, labels_cp: np.ndarray, target_scale: float) -> tuple[np.ndarray, np.ndarray]:
    return features, labels_cp.astype(np.float32) / np.float32(target_scale)


# SGD training loop.
def evaluate_loss(
    x: np.ndarray,
    y: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.float32,
    clip: float,
) -> float:
    if len(y) == 0:
        return 0.0
    h_pre = x @ w1 + b1
    h = np.clip(h_pre, 0.0, clip)
    pred = h @ w2 + b2
    return float(np.mean((pred - y) ** 2))


def forward_pred(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.float32,
    clip: float,
) -> np.ndarray:
    h_pre = x @ w1 + b1
    h = np.clip(h_pre, 0.0, clip)
    return h @ w2 + b2


def symmetry_error_cp(
    x: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.float32,
    clip: float,
    target_scale: float,
    max_rows: int = 2048,
) -> float:
    if len(x) == 0:
        return 0.0

    rows = min(len(x), max_rows)
    x_small = x[:rows]
    x_mirror = mirror_feature_matrix(x_small)

    p = forward_pred(x_small, w1, b1, w2, b2, clip)
    p_m = forward_pred(x_mirror, w1, b1, w2, b2, clip)
    # In ideal symmetry: eval(pos) == -eval(mirror(pos)).
    sym_abs = np.abs(p + p_m)
    return float(np.mean(sym_abs) * target_scale)


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    clip: float,
    symmetry_loss_weight: float,
    target_scale: float,
    seed: int,
    initial_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.float32] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32]:
    rng = np.random.default_rng(seed)

    if initial_state is None:
        w1 = rng.normal(0.0, 0.045, size=(FEATURE_DIM, hidden)).astype(np.float32)
        b1 = np.zeros((hidden,), dtype=np.float32)
        w2 = rng.normal(0.0, 0.045, size=(hidden,)).astype(np.float32)
        b2 = np.float32(0.0)
    else:
        w1, b1, w2, b2 = initial_state
        w1 = np.array(w1, dtype=np.float32, copy=True)
        b1 = np.array(b1, dtype=np.float32, copy=True)
        w2 = np.array(w2, dtype=np.float32, copy=True)
        b2 = np.float32(b2)
        if w1.shape != (FEATURE_DIM, hidden) or b1.shape != (hidden,) or w2.shape != (hidden,):
            raise ValueError("Resume weights do not match requested hidden size.")

    n = x_train.shape[0]
    train_t0 = time.perf_counter()
    best_w1 = np.array(w1, copy=True)
    best_b1 = np.array(b1, copy=True)
    best_w2 = np.array(w2, copy=True)
    best_b2 = np.float32(b2)
    best_metric = float("inf")
    best_sym_cp = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()
        perm = rng.permutation(n)
        x_train = x_train[perm]
        y_train = y_train[perm]

        for i in range(0, n, batch_size):
            xb = x_train[i : i + batch_size]
            yb = y_train[i : i + batch_size]
            if len(yb) == 0:
                continue

            h_pre = xb @ w1 + b1
            h = np.clip(h_pre, 0.0, clip)
            pred = h @ w2 + b2

            bs = np.float32(len(yb))
            err = pred - yb
            grad_pred = (2.0 / bs) * err

            grad_w2 = h.T @ grad_pred
            grad_b2 = np.sum(grad_pred).astype(np.float32)

            grad_h = np.outer(grad_pred, w2)
            grad_h[(h_pre <= 0.0) | (h_pre >= clip)] = 0.0

            grad_w1 = xb.T @ grad_h
            grad_b1 = np.sum(grad_h, axis=0)

            if symmetry_loss_weight > 0.0:
                x_m = mirror_feature_matrix(xb)
                h_pre_m = x_m @ w1 + b1
                h_m = np.clip(h_pre_m, 0.0, clip)
                pred_m = h_m @ w2 + b2

                # Penalize asymmetry directly: pred + pred_mirror should be near 0.
                sym = pred + pred_m
                grad_sym = (2.0 * symmetry_loss_weight / bs) * sym

                # Symmetry term backprop through original side.
                grad_w2 += h.T @ grad_sym
                grad_b2 += np.sum(grad_sym).astype(np.float32)
                grad_h_sym = np.outer(grad_sym, w2)
                grad_h_sym[(h_pre <= 0.0) | (h_pre >= clip)] = 0.0
                grad_w1 += xb.T @ grad_h_sym
                grad_b1 += np.sum(grad_h_sym, axis=0)

                # ...and through mirrored side.
                grad_w2 += h_m.T @ grad_sym
                grad_b2 += np.sum(grad_sym).astype(np.float32)
                grad_h_m = np.outer(grad_sym, w2)
                grad_h_m[(h_pre_m <= 0.0) | (h_pre_m >= clip)] = 0.0
                grad_w1 += x_m.T @ grad_h_m
                grad_b1 += np.sum(grad_h_m, axis=0)

            grad_w2 += weight_decay * w2
            grad_w1 += weight_decay * w1

            w1 -= lr * grad_w1
            b1 -= lr * grad_b1
            w2 -= lr * grad_w2
            b2 -= np.float32(lr) * grad_b2

        train_mse = evaluate_loss(x_train, y_train, w1, b1, w2, b2, clip)
        val_mse = evaluate_loss(x_val, y_val, w1, b1, w2, b2, clip)
        sym_source = x_val if len(y_val) > 0 else x_train
        sym_cp = symmetry_error_cp(sym_source, w1, b1, w2, b2, clip, target_scale, max_rows=1024)
        epoch_secs = time.perf_counter() - epoch_t0
        train_elapsed = time.perf_counter() - train_t0
        eta_train = (train_elapsed / epoch) * (epochs - epoch)

        print(
            f"epoch {epoch:02d}/{epochs} "
            f"train_mse={train_mse:.5f} val_mse={val_mse:.5f} "
            f"sym_abs={sym_cp:.1f}cp "
            f"| epoch={format_duration(epoch_secs)} "
            f"| train_eta~{format_duration(eta_train)}"
        )

        metric = val_mse if len(y_val) > 0 else train_mse
        if metric < best_metric - 1e-9 or (abs(metric - best_metric) <= 1e-9 and sym_cp < best_sym_cp):
            best_metric = metric
            best_sym_cp = sym_cp
            best_epoch = epoch
            best_w1 = np.array(w1, copy=True)
            best_b1 = np.array(b1, copy=True)
            best_w2 = np.array(w2, copy=True)
            best_b2 = np.float32(b2)

    print(
        f"Using best checkpoint: epoch {best_epoch:02d} "
        f"(metric={best_metric:.5f}, sym_abs={best_sym_cp:.1f}cp)"
    )
    return best_w1, best_b1, best_w2, best_b2


# ETA helpers.
def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    if seconds < 3600.0:
        mins = int(seconds // 60)
        rem = int(seconds % 60)
        return f"{mins}m {rem}s"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m"


def benchmark_training_epoch(
    train_rows: int,
    hidden: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    clip: float,
    symmetry_loss_weight: float,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    x = np.zeros((train_rows, FEATURE_DIM), dtype=np.float32)
    y = np.zeros((train_rows,), dtype=np.float32)

    hot = min(33, FEATURE_DIM)
    for i in range(train_rows):
        idxs = rng.integers(0, FEATURE_DIM, size=hot)
        x[i, idxs] = 1.0

    w1 = rng.normal(0.0, 0.045, size=(FEATURE_DIM, hidden)).astype(np.float32)
    b1 = np.zeros((hidden,), dtype=np.float32)
    w2 = rng.normal(0.0, 0.045, size=(hidden,)).astype(np.float32)
    b2 = np.float32(0.0)

    t0 = time.perf_counter()
    for i in range(0, train_rows, batch_size):
        xb = x[i : i + batch_size]
        yb = y[i : i + batch_size]
        if len(yb) == 0:
            continue

        h_pre = xb @ w1 + b1
        h = np.clip(h_pre, 0.0, clip)
        pred = h @ w2 + b2

        bs = np.float32(len(yb))
        err = pred - yb
        grad_pred = (2.0 / bs) * err
        grad_w2 = h.T @ grad_pred
        grad_b2 = np.sum(grad_pred).astype(np.float32)

        grad_h = np.outer(grad_pred, w2)
        grad_h[(h_pre <= 0.0) | (h_pre >= clip)] = 0.0
        grad_w1 = xb.T @ grad_h
        grad_b1 = np.sum(grad_h, axis=0)

        if symmetry_loss_weight > 0.0:
            x_m = mirror_feature_matrix(xb)
            h_pre_m = x_m @ w1 + b1
            h_m = np.clip(h_pre_m, 0.0, clip)
            pred_m = h_m @ w2 + b2

            sym = pred + pred_m
            grad_sym = (2.0 * symmetry_loss_weight / bs) * sym

            grad_w2 += h.T @ grad_sym
            grad_b2 += np.sum(grad_sym).astype(np.float32)
            grad_h_sym = np.outer(grad_sym, w2)
            grad_h_sym[(h_pre <= 0.0) | (h_pre >= clip)] = 0.0
            grad_w1 += xb.T @ grad_h_sym
            grad_b1 += np.sum(grad_h_sym, axis=0)

            grad_w2 += h_m.T @ grad_sym
            grad_b2 += np.sum(grad_sym).astype(np.float32)
            grad_h_m = np.outer(grad_sym, w2)
            grad_h_m[(h_pre_m <= 0.0) | (h_pre_m >= clip)] = 0.0
            grad_w1 += x_m.T @ grad_h_m
            grad_b1 += np.sum(grad_h_m, axis=0)

        grad_w2 += weight_decay * w2
        grad_w1 += weight_decay * w1

        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
        w2 -= lr * grad_w2
        b2 -= np.float32(lr) * grad_b2

    return time.perf_counter() - t0


def estimate_runtime(
    samplers: list[object],
    sampler_weights: list[float],
    teacher,
    teacher_mode: str,
    stockfish_path: str,
    stockfish_depth: int,
    stockfish_depth_min: int | None,
    samples: int,
    min_plies: int,
    max_plies: int,
    teacher_prob: float,
    opening_random_plies: int,
    dedupe_positions: bool,
    mirror_augment: bool,
    cp_clamp: float,
    target_scale: float,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    clip: float,
    symmetry_loss_weight: float,
    seed: int,
) -> tuple[float, float, float, float]:
    bench_samples = max(80, min(360, samples // 30 if samples > 0 else 80))

    t0 = time.perf_counter()
    bench_positions = sample_positions(
        samplers=samplers,
        sampler_weights=sampler_weights,
        samples=bench_samples,
        min_plies=min_plies,
        max_plies=max_plies,
        teacher_prob=teacher_prob,
        opening_random_plies=opening_random_plies,
        dedupe_positions=dedupe_positions,
        log_every_seconds=0.0,
    )
    sample_bench_secs = time.perf_counter() - t0
    got = max(1, len(bench_positions))
    sampling_eta = (sample_bench_secs / got) * samples

    if teacher_mode == "stockfish":
        label_bench_count = max(12, min(72, len(bench_positions)))
        label_positions = bench_positions[:label_bench_count]
        t1 = time.perf_counter()
        labels_cp = labels_from_stockfish(
            positions=label_positions,
            stockfish_path=stockfish_path,
            stockfish_depth=stockfish_depth,
            stockfish_depth_min=stockfish_depth_min,
            cp_clamp=cp_clamp,
            log_every_seconds=0.0,
        )
        labels_bench_secs = time.perf_counter() - t1
        labels_eta = (labels_bench_secs / max(1, len(labels_cp))) * samples
    else:
        t1 = time.perf_counter()
        labels_cp = labels_from_module(
            teacher=teacher,
            positions=bench_positions,
            cp_clamp=cp_clamp,
            log_every_seconds=0.0,
        )
        labels_bench_secs = time.perf_counter() - t1
        labels_eta = (labels_bench_secs / max(1, len(labels_cp))) * samples

    t2 = time.perf_counter()
    x_bench = build_feature_matrix(bench_positions, log_every_seconds=0.0)
    if mirror_augment:
        mirrored_bench = [board.mirror() for board in bench_positions]
        x_mirror = build_feature_matrix(mirrored_bench, log_every_seconds=0.0)
        x_bench = np.concatenate((x_bench, x_mirror), axis=0)
    _ = pack_dataset(x_bench, np.zeros((len(x_bench),), dtype=np.float32), target_scale)
    feature_bench_secs = time.perf_counter() - t2
    feature_eta = (feature_bench_secs / max(1, len(bench_positions))) * samples
    dataset_eta = labels_eta + feature_eta

    bench_rows = max(256, min(4096, int(0.9 * samples)))
    epoch_bench = benchmark_training_epoch(
        train_rows=bench_rows,
        hidden=hidden,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        clip=clip,
        symmetry_loss_weight=symmetry_loss_weight,
        seed=seed,
    )
    augment_factor = 2 if mirror_augment else 1
    train_rows_real = max(1, int(0.9 * samples * augment_factor))
    training_eta = epoch_bench * (train_rows_real / bench_rows) * epochs

    total_eta = sampling_eta + dataset_eta + training_eta
    return sampling_eta, dataset_eta, training_eta, total_eta


# Resume helpers.
def load_weights_dict(weights_path: str) -> dict | None:
    if not os.path.exists(weights_path):
        return None
    try:
        spec = importlib.util.spec_from_file_location("resume_nnue_weights", weights_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        data = getattr(module, "NNUE_WEIGHTS", None)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def decode_resume_state(
    weights: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float32, int, float, float] | None:
    try:
        if int(weights.get("feature_dim", -1)) != FEATURE_DIM:
            return None

        raw_w1_t = weights.get("w1_t")
        raw_b1 = weights.get("b1")
        raw_w2 = weights.get("w2")
        raw_b2 = weights.get("b2")

        if (
            not isinstance(raw_w1_t, list)
            or not isinstance(raw_b1, list)
            or not isinstance(raw_w2, list)
            or not isinstance(raw_b2, (int, float))
            or len(raw_w1_t) != FEATURE_DIM
        ):
            return None

        hidden = len(raw_b1)
        if hidden == 0 or len(raw_w2) != hidden:
            return None
        if any(not isinstance(row, list) or len(row) != hidden for row in raw_w1_t):
            return None

        w1 = np.array(raw_w1_t, dtype=np.float32)
        b1 = np.array(raw_b1, dtype=np.float32)
        w2 = np.array(raw_w2, dtype=np.float32)
        b2 = np.float32(raw_b2)
        clip = float(weights.get("clip", 1.5))
        output_scale = float(weights.get("output_scale", 400.0))
        return w1, b1, w2, b2, hidden, clip, output_scale
    except Exception:
        return None


# Export helpers.
def export_weights_py(
    out_path: str,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.float32,
    clip: float,
    output_scale: float,
) -> None:
    w1_t = w1.astype(np.float32).tolist()
    b1_l = b1.astype(np.float32).tolist()
    w2_l = w2.astype(np.float32).tolist()
    b2_v = float(b2)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated by BlunderBus/train_nnue.py\n")
        f.write("# NNUE weights for BlunderBus/team_BlunderBus.py\n\n")
        f.write("NNUE_WEIGHTS = {\n")
        f.write(f"    \"feature_dim\": {FEATURE_DIM},\n")
        f.write(f"    \"hidden\": {len(b1_l)},\n")
        f.write(f"    \"clip\": {float(clip):.6f},\n")
        f.write(f"    \"output_scale\": {float(output_scale):.6f},\n")
        f.write(f"    \"b2\": {b2_v:.8f},\n")
        f.write("    \"b1\": [\n")
        for v in b1_l:
            f.write(f"        {float(v):.8f},\n")
        f.write("    ],\n")
        f.write("    \"w2\": [\n")
        for v in w2_l:
            f.write(f"        {float(v):.8f},\n")
        f.write("    ],\n")
        f.write("    \"w1_t\": [\n")
        for row in w1_t:
            row_text = ", ".join(f"{float(v):.8f}" for v in row)
            f.write(f"        [{row_text}],\n")
        f.write("    ],\n")
        f.write("}\n")


def resolve_stockfish_path(repo_root: str, user_path: str) -> str:
    resolved = user_path
    local_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "stockfish", "src", "stockfish"),
        os.path.join(repo_root, "BlunderBus", "stockfish", "src", "stockfish"),
    ]

    if os.path.sep not in resolved:
        found = shutil.which(resolved)
        if found:
            return found
        for candidate in local_candidates:
            if os.path.exists(candidate):
                return candidate
    elif os.path.exists(resolved):
        return resolved

    for candidate in local_candidates:
        if os.path.exists(candidate):
            return candidate
    return resolved


# CLI entry.
def main() -> None:
    parser = argparse.ArgumentParser(description="Train NNUE weights for BlunderBus.")
    parser.add_argument("--teacher-mode", choices=("module", "stockfish"), default="stockfish")
    parser.add_argument("--teacher", default="team_aaaaaaaaaaaaaaa")
    parser.add_argument(
        "--samplers",
        default=None,
        help="Comma-separated sampler modules for position generation. Defaults to --teacher.",
    )
    parser.add_argument(
        "--sampler-weights",
        default=None,
        help="Optional comma-separated positive weights aligned with --samplers.",
    )
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument("--stockfish-depth", type=int, default=11)
    parser.add_argument(
        "--stockfish-depth-min",
        type=int,
        default=None,
        help="Optional lower bound for random stockfish depth per sample.",
    )
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--min-plies", type=int, default=6)
    parser.add_argument("--max-plies", type=int, default=70)
    parser.add_argument("--teacher-prob", type=float, default=0.70)
    parser.add_argument("--opening-random-plies", type=int, default=8)
    parser.add_argument(
        "--no-dedupe-positions",
        action="store_true",
        help="Disable zobrist dedupe during sampling.",
    )
    parser.add_argument(
        "--no-mirror-augment",
        action="store_true",
        help="Disable board mirror augmentation.",
    )
    parser.add_argument("--cp-clamp", type=float, default=1400.0)
    parser.add_argument("--target-scale", type=float, default=400.0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.010)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--clip", type=float, default=1.5)
    parser.add_argument(
        "--symmetry-loss-weight",
        type=float,
        default=0.12,
        help="Extra penalty on eval(pos) + eval(mirror(pos)) during training.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--skip-eta",
        action="store_true",
        help="Skip runtime estimation benchmark for faster startup.",
    )
    parser.add_argument(
        "--log-every-seconds",
        type=float,
        default=60.0,
        help="Progress log interval for long phases (<=0 disables periodic logs).",
    )
    parser.add_argument("--out", default="BlunderBus/team_shay_nnue_weights.py")
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional resume path. By default resumes from --out when available.",
    )
    parser.add_argument("--fresh-start", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    out_path = args.out
    if not os.path.isabs(out_path):
        out_path = os.path.join(repo_root, out_path)

    sampler_names = parse_csv_list(args.samplers) or [args.teacher]
    print(f"Loading samplers: {', '.join(sampler_names)}")
    samplers = load_sampler_modules(sampler_names)
    sampler_weights = normalize_sampler_weights(args.sampler_weights, len(sampler_names))

    print(f"Loading label module: {args.teacher}")
    label_teacher = importlib.import_module(args.teacher)
    if args.teacher_mode == "module" and not hasattr(label_teacher, "evaluate"):
        raise SystemExit("Module label mode requires evaluate().")

    dedupe_positions = not args.no_dedupe_positions
    mirror_augment = not args.no_mirror_augment
    if args.symmetry_loss_weight < 0.0:
        raise SystemExit("--symmetry-loss-weight must be >= 0.")
    if args.log_every_seconds < 0.0:
        raise SystemExit("--log-every-seconds must be >= 0.")
    print(
        "Sampling options: "
        f"teacher_prob={args.teacher_prob:.2f}, "
        f"opening_random_plies={args.opening_random_plies}, "
        f"dedupe={'on' if dedupe_positions else 'off'}, "
        f"mirror_augment={'on' if mirror_augment else 'off'}, "
        f"symmetry_loss={args.symmetry_loss_weight:.3f}, "
        f"log_every={args.log_every_seconds:.0f}s"
    )

    if args.teacher_mode == "stockfish":
        args.stockfish_path = resolve_stockfish_path(repo_root, args.stockfish_path)
        if not os.path.exists(args.stockfish_path):
            raise SystemExit("Stockfish not found. Install it or pass --stockfish-path /full/path/to/stockfish.")
        if args.stockfish_depth_min is not None:
            if args.stockfish_depth_min < 1 or args.stockfish_depth_min > args.stockfish_depth:
                raise SystemExit("--stockfish-depth-min must be between 1 and --stockfish-depth.")
            print(
                "Label mode: Stockfish "
                f"({args.stockfish_path}, depth={args.stockfish_depth_min}..{args.stockfish_depth})"
            )
        else:
            print(f"Label mode: Stockfish ({args.stockfish_path}, depth={args.stockfish_depth})")
    else:
        print(f"Label mode: module eval ({args.teacher})")

    # default behavior: resume from current output file
    initial_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.float32] | None = None
    explicit_resume = args.resume_from is not None
    resume_path = args.resume_from if explicit_resume else out_path
    if resume_path is not None and not os.path.isabs(resume_path):
        resume_path = os.path.join(repo_root, resume_path)

    if args.fresh_start:
        print("Fresh start requested: ignoring previous weights.")
    elif resume_path is not None and os.path.exists(resume_path):
        weights_dict = load_weights_dict(resume_path)
        decoded = decode_resume_state(weights_dict) if weights_dict is not None else None
        if decoded is None:
            if explicit_resume:
                raise SystemExit(f"Could not decode resume weights: {resume_path}")
            print(f"Resume skipped (decode failed): {resume_path}")
        else:
            w1, b1, w2, b2, resume_hidden, resume_clip, resume_scale = decoded
            if args.hidden != resume_hidden:
                print(f"Resume note: hidden changed {args.hidden} -> {resume_hidden}")
                args.hidden = resume_hidden
            if abs(args.clip - resume_clip) > 1e-9:
                print(f"Resume note: clip changed {args.clip} -> {resume_clip}")
                args.clip = resume_clip
            if abs(args.target_scale - resume_scale) > 1e-9:
                print(f"Resume note: target_scale changed {args.target_scale} -> {resume_scale}")
                args.target_scale = resume_scale
            initial_state = (w1, b1, w2, b2)
            print(f"Resuming from: {resume_path}")
    elif explicit_resume:
        raise SystemExit(f"--resume-from file not found: {resume_path}")
    else:
        print("No existing output weights found. Starting from random init.")

    est_total = 0.0
    if args.skip_eta:
        print("ETA benchmark skipped (--skip-eta).")
    else:
        print("Estimating runtime...")
        est_sampling, est_build, est_train, est_total = estimate_runtime(
            samplers=samplers,
            sampler_weights=sampler_weights,
            teacher=label_teacher,
            teacher_mode=args.teacher_mode,
            stockfish_path=args.stockfish_path,
            stockfish_depth=args.stockfish_depth,
            stockfish_depth_min=args.stockfish_depth_min,
            samples=args.samples,
            min_plies=args.min_plies,
            max_plies=args.max_plies,
            teacher_prob=args.teacher_prob,
            opening_random_plies=args.opening_random_plies,
            dedupe_positions=dedupe_positions,
            mirror_augment=mirror_augment,
            cp_clamp=args.cp_clamp,
            target_scale=args.target_scale,
            hidden=args.hidden,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            clip=args.clip,
            symmetry_loss_weight=args.symmetry_loss_weight,
            seed=args.seed,
        )
        print(
            "Estimated runtime: "
            f"total ~{format_duration(est_total)} "
            f"(sampling ~{format_duration(est_sampling)}, "
            f"dataset ~{format_duration(est_build)}, "
            f"training ~{format_duration(est_train)})"
        )

    # reset seeds after ETA benchmark
    random.seed(args.seed)
    np.random.seed(args.seed)
    run_t0 = time.perf_counter()

    print("Sampling training positions...")
    t_sampling = time.perf_counter()
    positions = sample_positions(
        samplers=samplers,
        sampler_weights=sampler_weights,
        samples=args.samples,
        min_plies=args.min_plies,
        max_plies=args.max_plies,
        teacher_prob=args.teacher_prob,
        opening_random_plies=args.opening_random_plies,
        dedupe_positions=dedupe_positions,
        log_every_seconds=args.log_every_seconds,
    )
    elapsed = time.perf_counter() - run_t0
    print(
        f"Collected {len(positions)} positions in {format_duration(time.perf_counter() - t_sampling)} "
        f"(elapsed {format_duration(elapsed)}, "
        f"remaining ~{format_duration(max(0.0, est_total - elapsed))})."
    )

    print("Building dataset...")
    t_dataset = time.perf_counter()
    positions = list(positions)
    if args.teacher_mode == "stockfish":
        labels_cp = labels_from_stockfish(
            positions=positions,
            stockfish_path=args.stockfish_path,
            stockfish_depth=args.stockfish_depth,
            stockfish_depth_min=args.stockfish_depth_min,
            cp_clamp=args.cp_clamp,
            log_every_seconds=args.log_every_seconds,
        )
    else:
        labels_cp = labels_from_module(
            teacher=label_teacher,
            positions=positions,
            cp_clamp=args.cp_clamp,
            log_every_seconds=args.log_every_seconds,
        )
    positions, labels_cp = maybe_mirror_augment(positions, labels_cp, mirror_augment)
    x = build_feature_matrix(positions, log_every_seconds=args.log_every_seconds)
    x, y = pack_dataset(x, labels_cp, args.target_scale)

    # Shuffle once before train/val split so validation is less correlated.
    perm = np.random.permutation(len(y))
    x = x[perm]
    y = y[perm]

    elapsed = time.perf_counter() - run_t0
    print(
        f"Dataset built in {format_duration(time.perf_counter() - t_dataset)} "
        f"(elapsed {format_duration(elapsed)}, "
        f"remaining ~{format_duration(max(0.0, est_total - elapsed))})."
    )

    split = int(0.9 * len(y))
    if len(y) > 1:
        split = max(1, min(split, len(y) - 1))
    else:
        split = 1
    x_train, y_train = x[:split], y[:split]
    x_val, y_val = x[split:], y[split:]
    print(f"train={len(y_train)} val={len(y_val)}")

    print("Training...")
    t_train = time.perf_counter()
    w1, b1, w2, b2 = train_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        hidden=args.hidden,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        clip=args.clip,
        symmetry_loss_weight=args.symmetry_loss_weight,
        target_scale=args.target_scale,
        seed=args.seed,
        initial_state=initial_state,
    )
    elapsed = time.perf_counter() - run_t0
    print(
        f"Training finished in {format_duration(time.perf_counter() - t_train)} "
        f"(elapsed {format_duration(elapsed)}, "
        f"remaining ~{format_duration(max(0.0, est_total - elapsed))})."
    )

    final_sym_cp = symmetry_error_cp(
        x_val if len(y_val) > 0 else x_train,
        w1,
        b1,
        w2,
        b2,
        args.clip,
        args.target_scale,
        max_rows=4096,
    )
    print(f"Final symmetry abs: {final_sym_cp:.1f}cp")
    if final_sym_cp > 60.0:
        print("Warning: symmetry is still high. Consider fresh-start + lower lr.")

    print(f"Exporting weights to {out_path}")
    export_weights_py(
        out_path=out_path,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        clip=args.clip,
        output_scale=args.target_scale,
    )
    print(f"Done in {format_duration(time.perf_counter() - run_t0)}.")


if __name__ == "__main__":
    main()
