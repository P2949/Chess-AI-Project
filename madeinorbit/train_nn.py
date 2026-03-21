### GO FORTH MY CLAUDE

import os
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

STOCKFISH_PATH = "/usr/bin/stockfish"
PGN_FILE= "/run/media/arsenicu/DATA/Development/Chess-AI-Project/madeinorbit/db.pgn"
NUM_POSITIONS = 100000
POSITIONS_PER_GAME = 8      # random sample per game; avoids over-representing long games
SKIP_OPENING_PLY = 10     # skip first N half-moves (opening theory != useful eval signal)
EVAL_DEPTH = 12
BATCH_SIZE = 512
EPOCHS = 25
LR = 1e-3
MODEL_PATH = "model.pt"
SCORE_CLAMP = 1500   #centipawns; +-1500 is already a decisive advantage
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", multiprocessing.cpu_count()))

PIECE_ORDER = [
    (chess.PAWN,   chess.WHITE), (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE), (chess.ROOK,   chess.WHITE),
    (chess.QUEEN,  chess.WHITE), (chess.KING,   chess.WHITE),
    (chess.PAWN,   chess.BLACK), (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK), (chess.ROOK,   chess.BLACK),
    (chess.QUEEN,  chess.BLACK), (chess.KING,   chess.BLACK),
]

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("No GPU detected — using CPU.\n")
    return torch.device("cpu")

def board_to_vector(board: chess.Board) -> np.ndarray:
    """773-dim: 12 piece planes * 64 squares + 4 castling bits + 1 side-to-move."""
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

def positions_from_pgn(pgn_path: str, target: int) -> list[str]:
    """Phase 1: parse FENs from PGN (fast, single-threaded, no Stockfish)."""
    fens: list[str] = []
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
            candidates: list[str] = []
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

def _label_chunk(args: tuple[list[str], str, int, int]) -> list[tuple[np.ndarray, float]]:
    """Phase 2 worker: each process owns its own Stockfish instance."""
    fens, stockfish_path, depth, clamp = args
    results: list[tuple[np.ndarray, float]] = []
    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        for fen in fens:
            try:
                board = chess.Board(fen)
                info  = engine.analyse(board, chess.engine.Limit(depth=depth))
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

def label_positions_parallel(fens: list[str]) -> list[tuple[np.ndarray, float]]:
    """Phase 2: label FENs across NUM_WORKERS processes in parallel."""
    chunks = [fens[i::NUM_WORKERS] for i in range(NUM_WORKERS)]
    args   = [(chunk, STOCKFISH_PATH, EVAL_DEPTH, SCORE_CLAMP) for chunk in chunks if chunk]
    data: list[tuple[np.ndarray, float]] = []
    print(f"Labelling {len(fens)} positions across {NUM_WORKERS} workers ...")
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(_label_chunk, a): a for a in args}
        pbar    = tqdm(total=len(fens), desc="Stockfish labelling")
        for future in as_completed(futures):
            chunk_results = future.result()
            data.extend(chunk_results)
            pbar.update(len(chunk_results))
        pbar.close()
    return data

class ChessDataset(Dataset):
    def __init__(self, records: list[tuple[np.ndarray, float]]):
        self.X = torch.tensor(np.stack([r[0] for r in records]), dtype=torch.float32)
        self.y = torch.tensor([r[1] for r in records], dtype=torch.float32).unsqueeze(1)
 
    def __len__(self):
        return len(self.y)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ChessEvaluator(nn.Module):
    """
    3-hidden-layer MLP.  Tanh output -> [-1, 1].
    Rescale by SCORE_CLAMP at inference time to get centipawns.
    """
    def __init__(self, input_dim: int = 773):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), 
            nn.ReLU(), 
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def train(model: nn.Module,
          train_loader: DataLoader, val_loader: DataLoader,
          epochs: int, lr: float, device: torch.device) -> None:
    model     = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
 
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimiser.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() * len(y_b)
        scheduler.step()
 
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                val_loss += criterion(model(X_b.to(device)),
                                      y_b.to(device)).item() * len(y_b)
 
        print(f"Epoch {epoch:3d}/{epochs}  "
              f"train MSE: {train_loss/len(train_loader.dataset):.6f}  "
              f"val MSE: {val_loss/len(val_loader.dataset):.6f}")


class NNEvaluate:
    """
    Drop-in replacement for evaluate() in team_astar.py.
 
        from train_chess_nn import NNEvaluate, ChessEvaluator, SCORE_CLAMP
        import torch
 
        _model = ChessEvaluator()
        _model.load_state_dict(torch.load("chess_eval.pt", map_location="cpu"))
        _model.eval()
        evaluate = NNEvaluate(_model)   # replaces your old evaluate()
    """
    def __init__(self, model: ChessEvaluator):
        self.model = model
 
    def __call__(self, board: chess.Board) -> float:
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
 
    fens = positions_from_pgn(PGN_FILE, NUM_POSITIONS)
    print(f"Extracted {len(fens)} positions from PGN.\n")
    data = label_positions_parallel(fens)
 
    if len(data) < 1000:
        raise SystemExit("Too few positions - check your PGN file and paths.")
 
    random.shuffle(data)
    split = int(0.9 * len(data))
    train_loader = DataLoader(ChessDataset(data[:split]), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ChessDataset(data[split:]), batch_size=BATCH_SIZE)
 
    device = get_device()
    model  = ChessEvaluator()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    train(model, train_loader, val_loader, EPOCHS, LR, device)
 
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nSaved {MODEL_PATH}")