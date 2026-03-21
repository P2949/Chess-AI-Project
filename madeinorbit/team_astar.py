import os
import math
import chess
import torch

from madeinorbit.train_nn import ChessEvaluator, board_to_vector, SCORE_CLAMP

model = ChessEvaluator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("madeinorbit/model.pt", map_location=device))
model.eval()

# ── Piece values (centipawns) ─────────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN:   1,
    chess.KNIGHT: 4,
    chess.BISHOP: 4.1,
    chess.ROOK:   6,
    chess.QUEEN:  12,
    chess.KING:   1,
}

# ── Heuristic ─────────────────────────────────────────────────────────────────
def evaluate(board: chess.Board) -> float:
    if board.is_checkmate():
        return -99999 if board.turn == chess.WHITE else 99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    #run model
    with torch.no_grad():
        x = torch.tensor(board_to_vector(board)).unsqueeze(0)
        return model(x).item() * SCORE_CLAMP

# ── Minimax with Alpha-Beta Pruning ───────────────────────────────────────────
def minimax(board: chess.Board, depth: int,
            alpha: float, beta: float,
            maximizing: bool) -> float:
    """
    Standard Minimax search with Alpha-Beta cutoffs.
    maximizing=True means we are searching for the best move for White.
    """
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    if maximizing:
        best = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            best = max(best, minimax(board, depth - 1, alpha, beta, False))
            board.pop()
            alpha = max(alpha, best)
            if beta <= alpha:
                break       # Beta cutoff — opponent won't allow this path
        return best
    else:
        best = float('inf')
        for move in board.legal_moves:
            board.push(move)
            best = min(best, minimax(board, depth - 1, alpha, beta, True))
            board.pop()
            beta = min(beta, best)
            if beta <= alpha:
                break       # Alpha cutoff
        return best


# ── Entry point called by the tournament harness ──────────────────────────────
def get_next_move(board: chess.Board,
                  color: chess.Color,
                  depth: int = 3) -> chess.Move | None:
    """
    Return the best move for `color` from the current `board` position.
    DO NOT rename or change this signature — the harness calls it directly.
    """
    best_move  = None
    maximizing = (color == chess.WHITE)
    best_score = float('-inf') if maximizing else float('inf')

    b = board.copy()   # never modify the board passed in
    for move in b.legal_moves:
        b.push(move)
        score = minimax(b, depth - 1,
                        float('-inf'), float('inf'),
                        not maximizing)
        b.pop()

        if maximizing and score > best_score:
            best_score, best_move = score, move
        elif not maximizing and score < best_score:
            best_score, best_move = score, move

    return best_move


# ── Quick self-test ───────────────────────────────────────────────────────────
if __name__ == '__main__':
    b = chess.Board()
    move = get_next_move(b, chess.WHITE, depth=3)
    if move is not None:
        print(f"[team_alpha] Opening move: {b.san(move)}")
