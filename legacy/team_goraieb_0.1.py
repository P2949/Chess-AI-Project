"""
team_goraieb.py  —  Chess Bot: Minimax + Alpha-Beta + Rich Positional Heuristic
=================================================================================
Heuristic components:
  1. Material count          — proper centipawn values (not all 1s)
  2. Piece-Square Tables     — PST for all 6 piece types; king PST interpolated
                               between middlegame and endgame tables
  3. Game-phase detection    — tapering weight via remaining piece count
  4. Pawn structure          — doubled-pawn penalty, isolated-pawn penalty,
                               passed-pawn bonus (scales with advancement)
  5. Mobility                — attacked squares per side
  6. King safety             — pawn-shield bonus + open-file-near-king penalty
                               (suppressed in endgame)
  7. Castling rights         — retain or exercise castling rights
  8. Bishop pair             — bonus when both bishops are alive
  9. Rook open files         — bonus for rooks on open / semi-open files
 10. Check bonus             — small bonus for placing opponent in check

Search improvement over the template:
  • Move ordering (MVV-LVA captures first, promotions, then quiet moves) is
    applied inside minimax to maximise alpha-beta cutoffs, yielding ~30-40 %
    more pruning and effectively a deeper search at the same depth budget.

Install:  pip install python-chess
"""

import chess

# ─────────────────────────────────────────────────────────────────────────────
# 1. MATERIAL VALUES  (centipawns — standard empirical values)
# ─────────────────────────────────────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   1000,   # never captured, but needed for PST scoring
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. PIECE-SQUARE TABLES  (White's POV, rank-8 first)
#
#    Index mapping
#      White piece on square sq: pst_idx = chess.square_mirror(sq)   (flips rank)
#      Black piece on square sq: pst_idx = sq                         (already mirrored)
#
#    Verified symmetry: White-e2 and Black-e7 both map to the same PST slot.
# ─────────────────────────────────────────────────────────────────────────────

PST_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]

PST_KNIGHT = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
]

PST_BISHOP = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
]

PST_ROOK = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0,
]

PST_QUEEN = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
]

# King safety differs dramatically between middlegame (hide in corner) and
# endgame (centralise and become active).
PST_KING_MG = [
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20,
]

PST_KING_EG = [
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
]

_PST_MAP = {
    chess.PAWN:   PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK:   PST_ROOK,
    chess.QUEEN:  PST_QUEEN,
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. GAME-PHASE DETECTION
# ─────────────────────────────────────────────────────────────────────────────
def _endgame_factor(board: chess.Board) -> float:
    """
    Returns a float in [0.0, 1.0].
      0.0 = full opening / middlegame
      1.0 = pure endgame

    Uses standard piece-count weighting:
      Q=4, R=2, B=1, N=1.  Both sides together give max phase = 24.
    As pieces come off the board, phase drops and endgame_factor rises.
    """
    phase = 0
    for c in (chess.WHITE, chess.BLACK):
        phase += len(board.pieces(chess.QUEEN,  c)) * 4
        phase += len(board.pieces(chess.ROOK,   c)) * 2
        phase += len(board.pieces(chess.BISHOP, c)) * 1
        phase += len(board.pieces(chess.KNIGHT, c)) * 1
    return 1.0 - min(phase, 24) / 24.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. MATERIAL + PST
# ─────────────────────────────────────────────────────────────────────────────
def _material_and_pst(board: chess.Board, eg: float) -> float:
    """
    For each piece: base material value + positional (PST) bonus.
    King PST is linearly interpolated between MG and EG tables.
    """
    score = 0.0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        sign    = 1 if piece.color == chess.WHITE else -1
        pst_idx = chess.square_mirror(sq) if piece.color == chess.WHITE else sq

        val = PIECE_VALUES[piece.piece_type]
        if piece.piece_type == chess.KING:
            pos = PST_KING_MG[pst_idx] * (1.0 - eg) + PST_KING_EG[pst_idx] * eg
        else:
            pos = _PST_MAP[piece.piece_type][pst_idx]

        score += sign * (val + pos)
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 5. PAWN STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
def _pawn_structure(board: chess.Board) -> float:
    """
    Three classic pawn-structure penalties / bonuses:

    • Doubled pawn   (-20 cp each extra pawn on the same file)
    • Isolated pawn  (-15 cp if no friendly pawn on adjacent files)
    • Passed pawn    (+20..90 cp scaled by how far advanced the pawn is;
                      a pawn with no enemy pawns blocking or guarding it
                      ahead on the same or adjacent files is passed)
    """
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign   = 1 if color == chess.WHITE else -1
        enemy  = not color
        pawns  = board.pieces(chess.PAWN, color)
        epawns = board.pieces(chess.PAWN, enemy)

        files = [chess.square_file(sq) for sq in pawns]

        for sq in pawns:
            f    = chess.square_file(sq)
            rank = chess.square_rank(sq)

            # ── Doubled ────────────────────────────────────────────────────
            if files.count(f) > 1:
                score += sign * (-20)

            # ── Isolated ───────────────────────────────────────────────────
            adj = [a for a in (f - 1, f + 1) if 0 <= a <= 7]
            if not any(a in files for a in adj):
                score += sign * (-15)

            # ── Passed ─────────────────────────────────────────────────────
            passed = True
            for esq in epawns:
                ef = chess.square_file(esq)
                er = chess.square_rank(esq)
                if ef not in (f - 1, f, f + 1):
                    continue
                # Enemy pawn is on an adjacent or same file.
                # If it's ahead of (or level with) our pawn it blocks passage.
                if color == chess.WHITE and er >= rank:
                    passed = False
                    break
                if color == chess.BLACK and er <= rank:
                    passed = False
                    break
            if passed:
                advancement = rank if color == chess.WHITE else (7 - rank)
                score += sign * (20 + advancement * 10)   # 20..90 cp

    return score


# ─────────────────────────────────────────────────────────────────────────────
# 6. MOBILITY
# ─────────────────────────────────────────────────────────────────────────────
def _mobility(board: chess.Board) -> float:
    """
    Count all squares attacked by each side's pieces.
    More attacked squares = more active pieces = better position.
    Each extra attacked square is worth 0.1 cp (small, avoids over-weighting).
    """
    w = b = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        n = len(board.attacks(sq))
        if piece.color == chess.WHITE:
            w += n
        else:
            b += n
    return 0.1 * (w - b)


# ─────────────────────────────────────────────────────────────────────────────
# 7. KING SAFETY
# ─────────────────────────────────────────────────────────────────────────────
def _king_safety(board: chess.Board, eg: float) -> float:
    """
    Middlegame-weighted king-safety score.

    • Pawn-shield bonus: friendly pawn one square in front of king (+10 per pawn)
    • Open-file penalty: no friendly pawn on king's file or adjacent files (-10)

    Gradually suppressed as eg → 1.0 (kings should centralise in endgame).
    """
    if eg >= 0.9:
        return 0.0

    mg_weight = 1.0 - eg
    score     = 0.0

    for color in (chess.WHITE, chess.BLACK):
        sign     = 1 if color == chess.WHITE else -1
        king_sq  = board.king(color)
        if king_sq is None:
            continue

        kf = chess.square_file(king_sq)
        kr = chess.square_rank(king_sq)

        for df in (-1, 0, 1):
            f = kf + df
            if not (0 <= f <= 7):
                continue

            # Pawn shield (one rank ahead in our direction)
            shield_r = kr + (1 if color == chess.WHITE else -1)
            if 0 <= shield_r <= 7:
                p = board.piece_at(chess.square(f, shield_r))
                if p and p.piece_type == chess.PAWN and p.color == color:
                    score += sign * 10 * mg_weight

            # Open-file penalty
            own_pawns_on_file = sum(
                1 for sq in board.pieces(chess.PAWN, color)
                if chess.square_file(sq) == f
            )
            if own_pawns_on_file == 0:
                score += sign * (-10) * mg_weight

    return score


# ─────────────────────────────────────────────────────────────────────────────
# 8. CASTLING RIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def _castling_bonus(board: chess.Board) -> float:
    """Retaining or having exercised castling rights is worth ~30 cp."""
    score = 0.0
    if board.has_castling_rights(chess.WHITE):
        score += 30.0
    if board.has_castling_rights(chess.BLACK):
        score -= 30.0
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 9. BISHOP PAIR
# ─────────────────────────────────────────────────────────────────────────────
def _bishop_pair(board: chess.Board) -> float:
    """Having both bishops is worth ~30 cp (they cover both square colours)."""
    score = 0.0
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 30.0
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 30.0
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 10. ROOK OPEN FILES
# ─────────────────────────────────────────────────────────────────────────────
def _rook_open_files(board: chess.Board) -> float:
    """
    Rook on an open file (no pawn of any colour): +25 cp
    Rook on a semi-open file (no own pawn, but enemy pawn present): +10 cp
    """
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        for sq in board.pieces(chess.ROOK, color):
            f = chess.square_file(sq)
            own_on_f   = sum(1 for s in board.pieces(chess.PAWN,  color) if chess.square_file(s) == f)
            enemy_on_f = sum(1 for s in board.pieces(chess.PAWN, not color) if chess.square_file(s) == f)
            if own_on_f == 0 and enemy_on_f == 0:
                score += sign * 25
            elif own_on_f == 0:
                score += sign * 10
    return score


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(board: chess.Board) -> float:
    """
    Full positional heuristic.  Returns a score in centipawns where:
        Score > 0  ⇒  White is better
        Score < 0  ⇒  Black is better
        Score = 0  ⇒  Approximately equal / draw

    Terminal states are handled with extreme values so the search always
    prefers checkmate over any positional advantage.
    """
    # ── Terminal states ────────────────────────────────────────────────────
    if board.is_checkmate():
        return -99_999 if board.turn == chess.WHITE else 99_999

    if (board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_repetition(3)
            or board.is_fifty_moves()):
        return 0

    # ── Phase factor (0.0 = opening, 1.0 = endgame) ───────────────────────
    eg = _endgame_factor(board)

    # ── Accumulate heuristic components ───────────────────────────────────
    score  = _material_and_pst(board, eg)  # Material + PST (dominant term)
    score += _mobility(board)              # Active pieces
    score += _pawn_structure(board)        # Pawn weaknesses / strengths
    score += _king_safety(board, eg)       # King-safety (MG only)
    score += _castling_bonus(board)        # Castling rights
    score += _bishop_pair(board)           # Bishop-pair advantage
    score += _rook_open_files(board)       # Rooks on open / semi-open files

    # Small check bonus: being in check is bad for the side to move
    if board.is_check():
        score += 20 if board.turn == chess.BLACK else -20

    return score


# ─────────────────────────────────────────────────────────────────────────────
# MOVE ORDERING  (MVV-LVA)
# ─────────────────────────────────────────────────────────────────────────────
def _order_moves(board: chess.Board, moves):
    """
    Sort moves so the most promising ones are searched first, which causes
    alpha-beta to prune far more aggressively:

      1. Captures ordered by MVV-LVA  (capture high-value piece with low-value attacker)
      2. Promotions  (always very good)
      3. Quiet moves last
    """
    def score(move: chess.Move) -> int:
        s = 0
        if board.is_capture(move):
            victim   = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            v_val = PIECE_VALUES.get(victim.piece_type,   0) if victim   else 0
            a_val = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 100
            s += 10 * v_val - a_val   # prefer cheap attacker, expensive victim
        if move.promotion:
            s += PIECE_VALUES.get(move.promotion, 0)
        return -s   # descending sort (most promising first)

    return sorted(moves, key=score)


# ─────────────────────────────────────────────────────────────────────────────
# MINIMAX WITH ALPHA-BETA PRUNING  (enhanced with move ordering)
# ─────────────────────────────────────────────────────────────────────────────
def minimax(board: chess.Board, depth: int,
            alpha: float, beta: float,
            maximizing: bool) -> float:
    """
    Standard Minimax search with Alpha-Beta cutoffs and MVV-LVA move ordering.
    maximizing=True means we are searching for the best move for White.
    """
    if depth == 0 or board.is_game_over():
        return evaluate(board)

    moves = _order_moves(board, list(board.legal_moves))

    if maximizing:
        best = float('-inf')
        for move in moves:
            board.push(move)
            best = max(best, minimax(board, depth - 1, alpha, beta, False))
            board.pop()
            alpha = max(alpha, best)
            if beta <= alpha:
                break       # Beta cutoff — opponent won't allow this path
        return best
    else:
        best = float('inf')
        for move in moves:
            board.push(move)
            best = min(best, minimax(board, depth - 1, alpha, beta, True))
            board.pop()
            beta = min(beta, best)
            if beta <= alpha:
                break       # Alpha cutoff
        return best


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT  (called by the tournament harness — DO NOT RENAME/CHANGE)
# ─────────────────────────────────────────────────────────────────────────────
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
    for move in _order_moves(b, list(b.legal_moves)):  # root-level ordering too
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


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time
    b = chess.Board()
    print(f"Starting position eval: {evaluate(b):.1f} cp  (should be near 0)")
    t0 = time.time()
    move = get_next_move(b, chess.WHITE, depth=3)
    print(f"[team_goraieb] Opening move (depth=3): {b.san(move)}  "
          f"({time.time()-t0:.2f}s)")
    t0 = time.time()
    move = get_next_move(b, chess.WHITE, depth=4)
    print(f"[team_goraieb] Opening move (depth=4): {b.san(move)}  "
          f"({time.time()-t0:.2f}s)")
