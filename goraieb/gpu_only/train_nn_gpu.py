"""
train_nn_gpu.py — Fully GPU-batched chess NN trainer

Key architecture: instead of evaluating positions one-at-a-time on GPU,
we expand the entire minimax tree, collect ALL leaf positions, evaluate
them in a single batched GPU forward pass, then propagate scores back
up the tree. This eliminates PCIe round-trip overhead per position.

Phase 1: Train on PGN game outcomes (no engine needed)
Phase 2: Batched self-play refinement (genuine GPU acceleration)

The self-play phase runs CONCURRENT_GAMES simultaneously, collecting
leaf nodes across all games into one big batch for GPU evaluation.
"""

import os
import math
import random
import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path

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

# Self-play
SELF_PLAY_ENABLED  = True
SELF_PLAY_ROUNDS   = 5
SELF_PLAY_GAMES    = 2000
SELF_PLAY_DEPTH    = 3      # can afford depth 3 now with batching
SELF_PLAY_EPOCHS   = 15
CONCURRENT_GAMES   = 64     # games running in parallel for leaf batching
GPU_EVAL_BATCH     = 8192   # max positions per GPU forward pass

PIECE_ORDER = [
    (chess.PAWN,   chess.WHITE), (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE), (chess.ROOK,   chess.WHITE),
    (chess.QUEEN,  chess.WHITE), (chess.KING,   chess.WHITE),
    (chess.PAWN,   chess.BLACK), (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK), (chess.ROOK,   chess.BLACK),
    (chess.QUEEN,  chess.BLACK), (chess.KING,   chess.BLACK),
]
_FLIP_PLANE = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
RESULT_MAP  = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


# ── Board vectorization ──────────────────────────────────────────────────────
def board_to_vector(board: chess.Board) -> np.ndarray:
    v = np.zeros(773, dtype=np.float32)
    for plane, (piece_type, color) in enumerate(PIECE_ORDER):
        for sq in board.pieces(piece_type, color):
            v[plane * 64 + sq] = 1.0
    v[768] = float(board.has_kingside_castling_rights(chess.WHITE))
    v[769] = float(board.has_queenside_castling_rights(chess.WHITE))
    v[770] = float(board.has_kingside_castling_rights(chess.BLACK))
    v[771] = float(board.has_queenside_castling_rights(chess.BLACK))
    v[772] = float(board.turn == chess.WHITE)
    return v


def flip_vector(v: np.ndarray) -> np.ndarray:
    out = np.zeros_like(v)
    for src_plane in range(12):
        dst_plane = _FLIP_PLANE[src_plane]
        for sq in range(64):
            out[dst_plane * 64 + (sq ^ 56)] = v[src_plane * 64 + sq]
    out[768], out[769] = v[770], v[771]
    out[770], out[771] = v[768], v[769]
    out[772] = 1.0 - v[772]
    return out


# ── Batched GPU evaluator ────────────────────────────────────────────────────
class BatchedEvaluator:
    """
    Collects board positions, evaluates them all at once on GPU.

    Usage:
        evaluator = BatchedEvaluator(model, device)
        idx1 = evaluator.enqueue(board1)
        idx2 = evaluator.enqueue(board2)
        evaluator.flush()
        score1 = evaluator.get_score(idx1)
        score2 = evaluator.get_score(idx2)
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self._vectors: list[np.ndarray] = []
        self._scores: list[float] = []
        self._entries: list[tuple[str, int | float]] = []
        # Each entry is ('gpu', vec_index) or ('terminal', score)

    def reset(self):
        self._vectors.clear()
        self._scores.clear()
        self._entries.clear()

    def enqueue(self, board: chess.Board) -> int:
        """
        Add a position to the evaluation queue.
        Returns an index to retrieve the score after flush().
        Terminal positions are scored immediately, no GPU needed.
        """
        idx = len(self._entries)

        # Terminal positions: score immediately, skip GPU
        if board.is_checkmate():
            score = -99999.0 if board.turn == chess.WHITE else 99999.0
            self._entries.append(('terminal', score))
            return idx
        if (board.is_stalemate() or board.is_insufficient_material()
                or board.is_fifty_moves() or board.is_repetition(3)):
            self._entries.append(('terminal', 0.0))
            return idx

        vec_idx = len(self._vectors)
        self._vectors.append(board_to_vector(board))
        self._entries.append(('gpu', vec_idx))
        return idx

    def flush(self):
        """Evaluate all enqueued non-terminal positions in one GPU batch."""
        if not self._vectors:
            return

        self.model.eval()
        all_scores: list[float] = []

        with torch.no_grad():
            for start in range(0, len(self._vectors), GPU_EVAL_BATCH):
                chunk = self._vectors[start:start + GPU_EVAL_BATCH]
                tensor = torch.tensor(np.stack(chunk), dtype=torch.float32).to(self.device)
                out = self.model(tensor).cpu().squeeze(-1).tolist()
                all_scores.extend(out)

        self._scores = [s * SCORE_CLAMP for s in all_scores]

    def get_score(self, idx: int) -> float:
        """Retrieve score for a previously enqueued position."""
        kind, val = self._entries[idx]
        if kind == 'terminal':
            return val
        return self._scores[val]

    @property
    def pending_count(self) -> int:
        return len(self._vectors)


# ── Batched minimax ──────────────────────────────────────────────────────────
class TreeNode:
    """A node in the minimax search tree."""
    __slots__ = ['move', 'children', 'score', 'eval_idx', 'is_maximizing', 'is_leaf']

    def __init__(self, move: chess.Move | None, is_maximizing: bool):
        self.move = move
        self.children: list[TreeNode] = []
        self.score: float = 0.0
        self.eval_idx: int = -1
        self.is_maximizing = is_maximizing
        self.is_leaf = False


def _expand_tree(board: chess.Board, depth: int, maximizing: bool,
                 evaluator: BatchedEvaluator) -> TreeNode:
    """
    Recursively expand the game tree WITHOUT evaluating anything.
    Leaf nodes get registered with the BatchedEvaluator for later GPU eval.
    """
    node = TreeNode(move=None, is_maximizing=maximizing)

    if depth == 0 or board.is_game_over():
        node.eval_idx = evaluator.enqueue(board)
        node.is_leaf = True
        return node

    for move in board.legal_moves:
        board.push(move)
        child = _expand_tree(board, depth - 1, not maximizing, evaluator)
        child.move = move
        node.children.append(child)
        board.pop()

    if not node.children:
        node.eval_idx = evaluator.enqueue(board)
        node.is_leaf = True

    return node


def _propagate_scores(node: TreeNode, evaluator: BatchedEvaluator) -> float:
    """
    After flush(), propagate leaf scores up the tree with minimax.
    Pure CPU arithmetic — no GPU calls.
    """
    if node.is_leaf:
        node.score = evaluator.get_score(node.eval_idx)
        return node.score

    if node.is_maximizing:
        node.score = float('-inf')
        for child in node.children:
            node.score = max(node.score, _propagate_scores(child, evaluator))
    else:
        node.score = float('inf')
        for child in node.children:
            node.score = min(node.score, _propagate_scores(child, evaluator))

    return node.score


def batched_best_move(board: chess.Board, depth: int,
                      evaluator: BatchedEvaluator) -> tuple[chess.Move | None, float]:
    """
    Find the best move using batched minimax:
    1. Expand the full tree to `depth` — CPU, collects leaves
    2. Evaluate ALL leaves in one GPU batch — evaluator.flush()
    3. Propagate scores back up — CPU arithmetic
    4. Pick the best root move
    """
    maximizing = (board.turn == chess.WHITE)
    root = TreeNode(move=None, is_maximizing=maximizing)

    for move in board.legal_moves:
        board.push(move)
        child = _expand_tree(board, depth - 1, not maximizing, evaluator)
        child.move = move
        root.children.append(child)
        board.pop()

    if not root.children:
        return None, 0.0

    # The caller is responsible for calling evaluator.flush()
    # This allows batching across multiple games before flushing
    return root, maximizing


# ── Multi-game batched self-play ─────────────────────────────────────────────
def _outcome_weight(ply: int, total_plies: int) -> float:
    if total_plies <= 0:
        return 0.5
    return (ply / total_plies) ** 0.5


class GameState:
    """Tracks one in-progress self-play game."""
    __slots__ = ['board', 'positions', 'move_count', 'done']

    def __init__(self):
        self.board = chess.Board()
        self.positions: list[np.ndarray] = []
        self.move_count = 0
        self.done = False


def play_games_batched(model: nn.Module, device: torch.device,
                       num_games: int, depth: int,
                       concurrent: int = CONCURRENT_GAMES
                       ) -> list[tuple[np.ndarray, float]]:
    """
    Run many self-play games concurrently, batching leaf evaluations
    across ALL active games into single GPU calls.

    64 concurrent games × ~30 moves × depth 3 ≈ 50k+ leaves per GPU batch.
    One kernel launch instead of 50k individual round-trips.
    """
    all_data: list[tuple[np.ndarray, float]] = []
    games_completed = 0
    max_moves = 200

    pbar = tqdm(total=num_games, desc="Self-play (batched)")

    while games_completed < num_games:
        batch_size = min(concurrent, num_games - games_completed)
        games = [GameState() for _ in range(batch_size)]

        # Play all games in this batch move-by-move
        while any(not g.done for g in games):
            active = [g for g in games if not g.done]
            if not active:
                break

            # ── Build trees for ALL active games, shared evaluator ────
            evaluator = BatchedEvaluator(model, device)
            game_roots: list[tuple[GameState, TreeNode, bool]] = []

            for game in active:
                if game.board.is_game_over() or game.move_count >= max_moves:
                    game.done = True
                    continue

                maximizing = (game.board.turn == chess.WHITE)
                root = TreeNode(move=None, is_maximizing=maximizing)

                for move in game.board.legal_moves:
                    game.board.push(move)
                    child = _expand_tree(game.board, depth - 1, not maximizing, evaluator)
                    child.move = move
                    root.children.append(child)
                    game.board.pop()

                if root.children:
                    game_roots.append((game, root, maximizing))
                else:
                    game.done = True

            if not game_roots:
                break

            # ── ONE GPU call for ALL leaves across ALL games ──────────
            evaluator.flush()

            # ── Propagate and make moves ──────────────────────────────
            for game, root, maximizing in game_roots:
                best_move = None
                best_score = float('-inf') if maximizing else float('inf')

                for child in root.children:
                    score = _propagate_scores(child, evaluator)
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

        # ── Harvest outcomes ──────────────────────────────────────────
        for game in games:
            result = game.board.result()
            outcome = RESULT_MAP.get(result, 0.0)
            total = len(game.positions)
            for i, vec in enumerate(game.positions):
                all_data.append((vec, outcome * _outcome_weight(i, total)))
            games_completed += 1
            pbar.update(1)

    pbar.close()
    print(f"Generated {len(all_data)} positions from {num_games} games "
          f"({concurrent} concurrent, depth {depth})")
    return all_data


# ── Phase 1: PGN game-outcome labeling ────────────────────────────────────────
def positions_from_pgn_with_outcomes(pgn_path: str, target: int) -> list[tuple[np.ndarray, float]]:
    data: list[tuple[np.ndarray, float]] = []
    games_read = 0

    with open(pgn_path, "r", errors="replace") as f:
        pbar = tqdm(total=target, desc="Parsing PGN + labeling")
        while len(data) < target:
            game = chess.pgn.read_game(f)
            if game is None:
                print(f"\nEnd of PGN after {games_read} games.")
                break
            games_read += 1

            result_str = game.headers.get("Result", "*")
            if result_str not in RESULT_MAP:
                continue
            result = RESULT_MAP[result_str]

            board = game.board()
            positions: list[tuple[np.ndarray, int]] = []
            for ply, move in enumerate(game.mainline_moves()):
                board.push(move)
                if ply >= SKIP_OPENING_PLY and not board.is_game_over():
                    positions.append((board_to_vector(board), ply))

            if not positions:
                continue

            total_plies = positions[-1][1] + 1
            sample = random.sample(positions, min(POSITIONS_PER_GAME, len(positions)))
            for vec, ply in sample:
                weight = _outcome_weight(ply, total_plies)
                data.append((vec, max(-1.0, min(1.0, result * weight))))

            pbar.update(len(sample))
        pbar.close()

    return data[:target]


# ── Data augmentation ─────────────────────────────────────────────────────────
def augment_data(data: list[tuple[np.ndarray, float]]) -> list[tuple[np.ndarray, float]]:
    augmented = list(data)
    for vec, score in data:
        augmented.append((flip_vector(vec), -score))
    print(f"Augmented {len(data)} → {len(augmented)} positions via color-flip.")
    return augmented


# ── Dataset / Model ───────────────────────────────────────────────────────────
class ChessDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            # Already list[tuple] format
            self.X = torch.tensor(np.stack([r[0] for r in X]), dtype=torch.float32)
            self.y = torch.tensor([r[1] for r in X], dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ChessEvaluator(nn.Module):
    def __init__(self, input_dim: int = 773):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("No GPU detected — using CPU.\n")
    return torch.device("cpu")


def train(model: nn.Module,
          train_loader: DataLoader, val_loader: DataLoader,
          epochs: int, lr: float, device: torch.device,
          save_path: str = MODEL_PATH) -> float:
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

    data = positions_from_pgn_with_outcomes(PGN_FILE, NUM_POSITIONS)
    print(f"Extracted {len(data)} labeled positions.\n")

    if len(data) < 1000:
        raise SystemExit("Too few positions — check your PGN file.")

    if AUGMENT_COLOR_FLIP:
        data = augment_data(data)

    random.shuffle(data)
    split = int(0.9 * len(data))
    # Build tensors incrementally and free the list immediately
    import gc
    X = np.stack([r[0] for r in data])
    y = np.array([r[1] for r in data], dtype=np.float32)
    del data
    gc.collect()

    train_ds = ChessDataset(X[:split], y[:split])
    val_ds   = ChessDataset(X[split:], y[split:])
    del X, y
    gc.collect()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = ChessEvaluator()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    train(model, train_loader, val_loader, EPOCHS, LR, device)

    del train_ds, val_ds, train_loader, val_loader
    gc.collect()

    # ── Phase 2 ──────────────────────────────────────────────────────────────
    if SELF_PLAY_ENABLED:
        print("\n" + "=" * 60)
        print("PHASE 2: Batched self-play refinement")
        print("=" * 60)

        model = model.to(device)

        for round_num in range(1, SELF_PLAY_ROUNDS + 1):
            print(f"\n── Self-play round {round_num}/{SELF_PLAY_ROUNDS} ──")

            sp_data = play_games_batched(
                model, device, SELF_PLAY_GAMES, SELF_PLAY_DEPTH, CONCURRENT_GAMES
            )

            if AUGMENT_COLOR_FLIP:
                sp_data = augment_data(sp_data)

            random.shuffle(sp_data)
            split = int(0.9 * len(sp_data))
            sp_train = DataLoader(ChessDataset(sp_data[:split]), batch_size=BATCH_SIZE, shuffle=True)
            sp_val   = DataLoader(ChessDataset(sp_data[split:]), batch_size=BATCH_SIZE)

            train(model, sp_train, sp_val, SELF_PLAY_EPOCHS, LR * 0.3, device)
            del sp_data, sp_train, sp_val

    print(f"\nFinal model saved to {MODEL_PATH}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")