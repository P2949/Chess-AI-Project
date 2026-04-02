"""
team_creepers.py  —  Chess Bot using Minimax + Alpha-Beta Pruning
Heuristic: Material value

Install dependency:  pip install python-chess
"""

import chess

# ── Piece values (centipawns) ─────────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN:   1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3.5,
    chess.ROOK:   5,
    chess.QUEEN:  9,
    chess.KING:   1,
}

# ── Heuristic ─────────────────────────────────────────────────────────────────

def bishop_control(board: chess.Board, square: int, color: chess.Color) -> float:
    score = 0.0
    control_influence = 0.1
    ally_block_influence = 0.2
    enemy_block_influence = 0.05

    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for file_delta, rank_delta in directions:
        f = chess.square_file(square)
        r = chess.square_rank(square)

        while True:
            f += file_delta
            r += rank_delta

            if not (0 <= f <= 7 and 0 <= r <= 7):
                break

            target = chess.square(f, r)
            piece = board.piece_at(target)

            if piece is None:
                score += control_influence
            elif piece.color == color:
                score -= ally_block_influence
                break
            else:
                score -= enemy_block_influence
                break

    return score

def rook_control(board: chess.Board, square: int, color: chess.Color) -> float:
    score = 0.0
    control_influence = 0.1
    ally_block_influence = 0.2
    enemy_block_influence = 0.05

    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    for file_delta, rank_delta in directions:
        f = chess.square_file(square)
        r = chess.square_rank(square)

        while True:
            f += file_delta
            r += rank_delta

            if not (0 <= f <= 7 and 0 <= r <= 7):
                break

            target = chess.square(f, r)
            piece = board.piece_at(target)

            if piece is None:
                score += control_influence
            elif piece.color == color:
                score -= ally_block_influence
                break
            else:
                score -= enemy_block_influence
                break

    return score

def pawn_structure(board: chess.Board) -> float:
    strong_bonus = 0.2
    week_bonus = 0.1

    score = 0.0

    for color, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
        pawns = board.pieces(chess.PAWN, color)

        for square in pawns:
            f = chess.square_file(square)
            r = chess.square_rank(square)

            if color == chess.WHITE:
                defending_squares = [
                    chess.square(f - 1, r - 1) if f > 0 else None,
                    chess.square(f + 1, r - 1) if f < 7 else None
                ]
            else:
                defending_squares = [
                    chess.square(f - 1, r + 1) if f > 0 and r < 7 else None,
                    chess.square(f + 1, r + 1) if f < 7 and r < 7 else None
                ]

            for def_square in defending_squares:
                if def_square is None:
                    continue
                piece = board.piece_at(def_square)

                # If the square behind-diagonally holds a friendly pawn, we are in a chain
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    score += sign * strong_bonus   # the defended pawn gets the main bonus
                    score += sign * week_bonus  # the defending pawn gets a smaller one

    return score

def week_pawns(board: chess.Board) -> float:
    isolated = 0.3
    doubled = 0.2

    score = 0.0

    for color, sign in [(chess.WHITE, 1), (chess.BLACK, -1)]:
        pawns = board.pieces(chess.PAWN, color)
        files = [chess.square_file(sq) for sq in pawns]

        for square in pawns:
            f = chess.square_file(square)

            if (f - 1) not in files and (f + 1) not in files:
                score -= sign * isolated

        for f in set(files):
            if files.count(f) > 1:
                score -= sign * doubled * (files.count(f) - 1)

    return score

def evaluate(board: chess.Board) -> float:
    """

    Material:  Sum of piece values for White minus Black.

    Score > 0  =>  White is better.
    Score < 0  =>  Black is better.
    """
    if board.is_checkmate():
        # The side to move is in checkmate — they lose
        return -99999 if board.turn == chess.WHITE else 99999
    if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition():
        return 0

    # ── Example Heuristic ─────────────────────────────────────────────────────────
    # The evaluation function counts the number of pieces White has minus the number
    # of pieces Black has, multiplied by a value of 1.

    # In the minimax algorithm, this evaluation is applied at the leaf nodes and
    # reflects the advantage of one player over the other. A positive score means
    # White is in a better position, while a negative score indicates that Black
    # is ahead.

    # The algorithm itself does not need to care about the player's colour,
    # because the evaluation function already represents the position from
    # both players' perspectives.

    # Heat maps for pieces, I believe its upside down, left-top corner is A1

    no_heat = [
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1
    ]

    knight_heat = [
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.5,
        0.5, 0.8, 1, 1, 1, 1, 0.8, 0.5,
        0.5, 0.8, 1, 1, 1, 1, 0.8, 0.5,
        0.5, 0.8, 1, 1, 1, 1, 0.8, 0.5,
        0.5, 0.8, 1, 1, 1, 1, 0.8, 0.5,
        0.5, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5
    ]

    king_heat = [
        1, 1, 0.9, -0.5, 0, -0.5, 1, 1,
        0.8, 0.7, -1, -1, -1, 0.6, 0.7, 0.8,
        0.3, 0.2, 0.1, 0, 0, 0.1, 0.2, 0.3,
        -0.3, -0.4, -0.5, -0.6, -0.6, -0.5, -0.4, -0.3,
        -0.3, -0.4, -0.5, -0.6, -0.6, -0.5, -0.4, -0.3,
        0.3, 0.2, 0.1, 0, 0, 0.1, 0.2, 0.3,
        0.8, 0.7, -1, -1, -1, 0.6, 0.7, 0.8,
        1, 1, 0.9, -0.5, 0, -0.5, 1, 1
    ]

    heat_dictionary = {
        chess.PAWN: no_heat,
        chess.KNIGHT: knight_heat,
        chess.BISHOP: no_heat,
        chess.ROOK: no_heat,
        chess.QUEEN: no_heat,
        chess.KING: king_heat
    }

    score = 0
    for piece_type, value in PIECE_VALUES.items():
        heat = heat_dictionary[piece_type]

        for square in board.pieces(piece_type, chess.WHITE):
            if piece_type == chess.BISHOP:
                score += value + bishop_control(board, square, chess.WHITE)
            elif piece_type == chess.ROOK:
                score += value + rook_control(board, square, chess.WHITE)
            elif piece_type == chess.QUEEN:
                score += value + rook_control(board, square, chess.WHITE) + bishop_control(board, square, chess.WHITE)
            else:
                score += value * heat[square]

        for square in board.pieces(piece_type, chess.BLACK):
            if piece_type == chess.BISHOP:
                score -= value + bishop_control(board, square, chess.BLACK)
            elif piece_type == chess.ROOK:
                score -= value + rook_control(board, square, chess.BLACK)
            elif piece_type == chess.QUEEN:
                score -= value + rook_control(board, square, chess.BLACK) + bishop_control(board, square, chess.BLACK)
            else:
                score -= value * heat[square]

    score += pawn_structure(board)
    score += week_pawns(board)

    if board.is_check():
        if board.turn == chess.WHITE:
            score -= 0.3
        else:
            score += 0.3

    threat_influence = 0.1

    def get_lowest_attacker_value(board: chess.Board, square: int, color: chess.Color) -> float:
        """Returns the value of the cheapest piece of `color` attacking `square`, or inf if none."""
        for piece_type, value in sorted(PIECE_VALUES.items(), key=lambda x: x[1]):
            for attacker in board.pieces(piece_type, color):
                if board.is_attacked_by(color, square):
                    attacks = board.attacks(attacker)
                    if square in attacks:
                        return value
        return float('inf')

    for piece_type, value in PIECE_VALUES.items():
        for square in board.pieces(piece_type, chess.WHITE):
            if board.is_attacked_by(chess.BLACK, square):
                lowest_attacker = get_lowest_attacker_value(board, square, chess.BLACK)
                if lowest_attacker <= value:  # only penalise if the trade is good or equal for Black
                    score -= value * threat_influence

        for square in board.pieces(piece_type, chess.BLACK):
            if board.is_attacked_by(chess.WHITE, square):
                lowest_attacker = get_lowest_attacker_value(board, square, chess.WHITE)
                if lowest_attacker <= value:  # only reward if the trade is good or equal for White
                    score += value * threat_influence

    return score


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
                  depth: int = 3) -> chess.Move:
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
    print(f"[team_alpha] Opening move: {b.san(move)}")
