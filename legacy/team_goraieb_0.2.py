"""
team_goraieb.py  —  Chess Bot: Championship Edition
=====================================================
Search enhancements (over the previous version):
  • Iterative deepening      — searches depth 1..TARGET_DEPTH; PV from
                               previous iteration is reused for move ordering,
                               giving far better alpha-beta cutoffs
  • Quiescence search        — at depth 0, continues searching captures until
                               the position is "quiet", completely eliminating
                               the horizon effect (the single biggest weakness
                               of the base bot)
  • Transposition table      — Zobrist hash (chess.polyglot); EXACT / LOWER /
                               UPPER flags; avoids re-searching repeated
                               positions and massively widens the effective
                               search depth
  • Null-move pruning        — R=2/3 reduction; if passing gives score >= beta
                               the position is already too good, prune.
                               Skipped in check and deep endgames (zugzwang)
  • Killer heuristic         — 2 quiet moves per ply that caused beta cutoffs
                               in sibling nodes; ordered right after captures
  • History heuristic        — [from][to] table incremented by depth² on
                               cutoffs; decayed between outer iterations
  • Late Move Reductions     — moves 4+ that are quiet, not check, not
                               promotions are reduced by 1 ply; re-searched
                               at full depth only if they beat alpha
  • Check extension          — when the side to move is in check, extend
                               depth by 1 to avoid misevaluation

Evaluation enhancements (added on top of the existing rich heuristic):
  • Tempo bonus              — +15 cp for the side to move (initiative)
  • Knight outposts          — knight on ranks 4-6 protected by own pawn
                               and not attackable by enemy pawns (+20 cp)
  • Rook on 7th rank         — trapping enemy king / pawns (+25 cp)
  • Connected rooks          — rooks on same rank/file, nothing between (+15 cp)
  • Backward pawns           — pawn that can't advance safely and has no
                               friendly pawn support behind it (-10 cp)
  • Hanging pieces           — penalise attacked but undefended pieces
                               (large bonus/penalty by piece value)
  • King proximity mop-up    — in won endgames, drive the opponent's king to
                               a corner and approach with our king (+scaled cp)

Install: pip install python-chess
"""

import chess
import chess.polyglot

# ─────────────────────────────────────────────────────────────────────────────
# 1. MATERIAL VALUES  (centipawns)
# ─────────────────────────────────────────────────────────────────────────────
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. PIECE-SQUARE TABLES  (White's POV, rank-8 first)
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
    Returns 0.0 (full middlegame) .. 1.0 (pure endgame).
    Q=4, R=2, B=N=1; max phase = 24.
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
    Doubled (-20), isolated (-15), passed (+20..90), backward (-10).
    """
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign   = 1 if color == chess.WHITE else -1
        enemy  = not color
        pawns  = board.pieces(chess.PAWN, color)
        epawns = board.pieces(chess.PAWN, enemy)
        files  = [chess.square_file(sq) for sq in pawns]

        for sq in pawns:
            f    = chess.square_file(sq)
            rank = chess.square_rank(sq)

            # ── Doubled ──────────────────────────────────────────────────────
            if files.count(f) > 1:
                score += sign * (-20)

            # ── Isolated ─────────────────────────────────────────────────────
            adj = [a for a in (f - 1, f + 1) if 0 <= a <= 7]
            if not any(a in files for a in adj):
                score += sign * (-15)

            # ── Passed ───────────────────────────────────────────────────────
            passed = True
            for esq in epawns:
                ef = chess.square_file(esq)
                er = chess.square_rank(esq)
                if ef not in (f - 1, f, f + 1):
                    continue
                if color == chess.WHITE and er >= rank:
                    passed = False; break
                if color == chess.BLACK and er <= rank:
                    passed = False; break
            if passed:
                advancement = rank if color == chess.WHITE else (7 - rank)
                score += sign * (20 + advancement * 10)

            # ── Backward ─────────────────────────────────────────────────────
            # A pawn is backward if it cannot safely advance and has no
            # friendly pawns behind it on adjacent files.
            support_ranks = range(0, rank) if color == chess.WHITE else range(rank + 1, 8)
            has_support = any(
                chess.square_file(s) in (f - 1, f + 1) and chess.square_rank(s) in support_ranks
                for s in pawns
            )
            if not has_support and not passed:
                score += sign * (-10)

    return score


# ─────────────────────────────────────────────────────────────────────────────
# 6. MOBILITY
# ─────────────────────────────────────────────────────────────────────────────
def _mobility(board: chess.Board) -> float:
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
    if eg >= 0.9:
        return 0.0
    mg_weight = 1.0 - eg
    score     = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign    = 1 if color == chess.WHITE else -1
        king_sq = board.king(color)
        if king_sq is None:
            continue
        kf = chess.square_file(king_sq)
        kr = chess.square_rank(king_sq)
        for df in (-1, 0, 1):
            f = kf + df
            if not (0 <= f <= 7):
                continue
            shield_r = kr + (1 if color == chess.WHITE else -1)
            if 0 <= shield_r <= 7:
                p = board.piece_at(chess.square(f, shield_r))
                if p and p.piece_type == chess.PAWN and p.color == color:
                    score += sign * 10 * mg_weight
            own_pawns_on_file = sum(
                1 for s in board.pieces(chess.PAWN, color)
                if chess.square_file(s) == f
            )
            if own_pawns_on_file == 0:
                score += sign * (-10) * mg_weight
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 8. CASTLING RIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def _castling_bonus(board: chess.Board) -> float:
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
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        for sq in board.pieces(chess.ROOK, color):
            f = chess.square_file(sq)
            own_f   = sum(1 for s in board.pieces(chess.PAWN, color)   if chess.square_file(s) == f)
            enemy_f = sum(1 for s in board.pieces(chess.PAWN, not color) if chess.square_file(s) == f)
            if own_f == 0 and enemy_f == 0:
                score += sign * 25
            elif own_f == 0:
                score += sign * 10
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 11. KNIGHT OUTPOSTS  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
def _knight_outposts(board: chess.Board) -> float:
    """
    A knight on rank 4-6 (White) / rank 3-5 (Black) that is:
      • protected by a friendly pawn, AND
      • cannot be attacked by any enemy pawn
    is an outpost — a powerful, stable position worth +20 cp.
    """
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign   = 1 if color == chess.WHITE else -1
        enemy  = not color
        epawns = board.pieces(chess.PAWN, enemy)

        for sq in board.pieces(chess.KNIGHT, color):
            rank = chess.square_rank(sq)
            f    = chess.square_file(sq)

            # Outpost ranks
            if color == chess.WHITE and rank not in (3, 4, 5):
                continue
            if color == chess.BLACK and rank not in (2, 3, 4):
                continue

            # Protected by a friendly pawn?
            pawn_rank = rank - 1 if color == chess.WHITE else rank + 1
            if not (0 <= pawn_rank <= 7):
                continue
            protected = any(
                chess.square_file(s) in (f - 1, f + 1) and chess.square_rank(s) == pawn_rank
                for s in board.pieces(chess.PAWN, color)
            )
            if not protected:
                continue

            # Can any enemy pawn attack it?
            attack_rank = rank + 1 if color == chess.WHITE else rank - 1
            if not (0 <= attack_rank <= 7):
                attackable = False
            else:
                attackable = any(
                    chess.square_file(s) in (f - 1, f + 1) and chess.square_rank(s) == attack_rank
                    for s in epawns
                )
            if not attackable:
                score += sign * 20

    return score


# ─────────────────────────────────────────────────────────────────────────────
# 12. ROOK ON 7TH RANK  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
def _rook_on_seventh(board: chess.Board) -> float:
    """
    A rook on the 7th rank (rank index 6 for White, rank index 1 for Black)
    pressures the enemy king and back-rank pawns. +25 cp.
    """
    score = 0.0
    for sq in board.pieces(chess.ROOK, chess.WHITE):
        if chess.square_rank(sq) == 6:     # rank 7 in human notation
            score += 25
    for sq in board.pieces(chess.ROOK, chess.BLACK):
        if chess.square_rank(sq) == 1:     # rank 2 in human notation
            score -= 25
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 13. CONNECTED ROOKS  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
def _connected_rooks(board: chess.Board) -> float:
    """
    Two rooks are connected if they share the same rank or file and no piece
    stands between them. Connected rooks coordinate strongly. +15 cp.
    """
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign  = 1 if color == chess.WHITE else -1
        rooks = list(board.pieces(chess.ROOK, color))
        if len(rooks) < 2:
            continue
        r1, r2 = rooks[0], rooks[1]
        f1, rk1 = chess.square_file(r1), chess.square_rank(r1)
        f2, rk2 = chess.square_file(r2), chess.square_rank(r2)

        connected = False
        if f1 == f2:
            lo, hi = (min(rk1, rk2) + 1, max(rk1, rk2))
            if all(board.piece_at(chess.square(f1, r)) is None for r in range(lo, hi)):
                connected = True
        elif rk1 == rk2:
            lo, hi = (min(f1, f2) + 1, max(f1, f2))
            if all(board.piece_at(chess.square(f, rk1)) is None for f in range(lo, hi)):
                connected = True

        if connected:
            score += sign * 15
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 14. HANGING PIECES  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
def _hanging_pieces(board: chess.Board) -> float:
    """
    A piece is "hanging" if it is attacked by at least one enemy piece and
    defended by fewer pieces than it is attacked by (simplified: just check if
    it is attacked at all and the defender count is 0).
    Penalty = piece_value * 0.25 (we don't give the full value since the
    engine will find the capture itself, but this nudges eval away from
    hanging pieces proactively).
    """
    score = 0.0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type == chess.KING:
            continue
        enemy = not piece.color
        if board.is_attacked_by(enemy, sq):
            # Count defenders
            defenders = len(board.attackers(piece.color, sq))
            attackers = len(board.attackers(enemy, sq))
            if defenders < attackers:
                val   = PIECE_VALUES[piece.piece_type] * 0.25
                sign  = -1 if piece.color == chess.WHITE else 1
                score += sign * val
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 15. ENDGAME KING PROXIMITY (mop-up)  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
def _king_proximity_mop_up(board: chess.Board, eg: float, material_score: float) -> float:
    """
    In the endgame, when one side is clearly winning (|material_score| > 300 cp):
      • Drive the losing king toward a corner (corner proximity bonus).
      • Bring the winning king close to the losing king.
    This is how engines win material-up endgames efficiently.
    """
    if eg < 0.4:
        return 0.0

    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return 0.0

    wkf, wkr = chess.square_file(wk), chess.square_rank(wk)
    bkf, bkr = chess.square_file(bk), chess.square_rank(bk)

    king_dist = abs(wkf - bkf) + abs(wkr - bkr)

    def corner_dist(f, r):
        return min(f + r, (7 - f) + r, f + (7 - r), (7 - f) + (7 - r))

    result = 0.0
    if material_score > 300:          # White is winning
        bk_corner = corner_dist(bkf, bkr)
        result += eg * (14 - king_dist) * 4   # White king approaches Black king
        result += eg * bk_corner * 8          # Push Black king to corner
    elif material_score < -300:       # Black is winning
        wk_corner = corner_dist(wkf, wkr)
        result -= eg * (14 - king_dist) * 4
        result -= eg * wk_corner * 8

    return result


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(board: chess.Board) -> float:
    """
    Full positional heuristic (centipawns, White-positive).
    """
    # ── Terminal states ────────────────────────────────────────────────────
    if board.is_checkmate():
        return -99_999 if board.turn == chess.WHITE else 99_999
    if (board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_repetition(3)
            or board.is_fifty_moves()):
        return 0

    eg = _endgame_factor(board)

    # Dominant term: material + PST
    mat = _material_and_pst(board, eg)

    score  = mat
    score += _mobility(board)
    score += _pawn_structure(board)
    score += _king_safety(board, eg)
    score += _castling_bonus(board)
    score += _bishop_pair(board)
    score += _rook_open_files(board)
    score += _knight_outposts(board)
    score += _rook_on_seventh(board)
    score += _connected_rooks(board)
    score += _hanging_pieces(board)
    score += _king_proximity_mop_up(board, eg, mat)

    # Tempo: small bonus for having the move
    score += 15 if board.turn == chess.WHITE else -15

    # Check: being in check is a positional liability
    if board.is_check():
        score += 20 if board.turn == chess.BLACK else -20

    return score


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
_TT      = {}                                      # Transposition table
_KILLERS = [[None, None] for _ in range(128)]      # 2 killers per ply
_HISTORY = {}                                      # (from_sq, to_sq) -> int

TT_EXACT = 0
TT_LOWER = 1   # score >= beta (fail high)
TT_UPPER = 2   # score <= alpha (fail low)

MAX_TT_SIZE = 2_000_000    # ~200 MB in Python — tune down if memory is tight

# Minimum search depth regardless of what the harness requests.
# No time limit => go deep.  Reduce if games time out on slow hardware.
TARGET_DEPTH = 3


# ─────────────────────────────────────────────────────────────────────────────
# TT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _tt_store(h: int, depth: int, flag: int, score: float, move) -> None:
    existing = _TT.get(h)
    # Always-replace with depth preference: deeper entries take priority
    if existing is None or depth >= existing[0] or len(_TT) < MAX_TT_SIZE:
        _TT[h] = (depth, flag, score, move.uci() if move else None)


def _tt_probe(h: int):
    """Returns (depth, flag, score, move_or_None) or None."""
    entry = _TT.get(h)
    if entry is None:
        return None
    depth, flag, score, uci = entry
    move = None
    if uci:
        try:
            move = chess.Move.from_uci(uci)
        except Exception:
            pass
    return depth, flag, score, move


# ─────────────────────────────────────────────────────────────────────────────
# KILLER & HISTORY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _update_killers(move: chess.Move, ply: int) -> None:
    if ply >= 128:
        return
    if _KILLERS[ply][0] != move:
        _KILLERS[ply][1] = _KILLERS[ply][0]
        _KILLERS[ply][0] = move


def _update_history(move: chess.Move, depth: int) -> None:
    key = (move.from_square, move.to_square)
    _HISTORY[key] = min(_HISTORY.get(key, 0) + depth * depth, 50_000)


# ─────────────────────────────────────────────────────────────────────────────
# MOVE ORDERING
# ─────────────────────────────────────────────────────────────────────────────
def _order_moves(board: chess.Board, moves, tt_move=None, ply: int = 0):
    """
    Priority (highest first):
      1. TT / hash move                  (+30 000)
      2. Winning captures (MVV-LVA)      (+20 000 + gain)
      3. Queen promotion                 (+19 000)
      4. Killer moves                    (+10 000)
      5. Quiet moves ordered by history
      6. Losing captures                 (-10 000 + loss)
    """
    tt_uci     = tt_move.uci() if tt_move is not None else None
    ks         = _KILLERS[ply] if ply < 128 else [None, None]
    killer_set = {k.uci() for k in ks if k is not None}

    def _score(move: chess.Move) -> int:
        uci = move.uci()
        if uci == tt_uci:
            return 30_000

        is_cap = board.is_capture(move)
        if is_cap:
            victim   = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            v_val = PIECE_VALUES.get(victim.piece_type,   0) if victim   else 0
            a_val = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
            gain  = 10 * v_val - a_val
            return (20_000 + gain) if gain >= 0 else (-10_000 + gain)

        if move.promotion == chess.QUEEN:
            return 19_000

        if uci in killer_set:
            return 10_000

        return _HISTORY.get((move.from_square, move.to_square), 0)

    return sorted(moves, key=_score, reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# QUIESCENCE SEARCH
# ─────────────────────────────────────────────────────────────────────────────
def _qsearch(board: chess.Board, alpha: float, beta: float, maximizing: bool) -> float:
    """
    Continue searching until no captures remain (the position is "quiet").
    Prevents the horizon effect: without this, a position where a piece is
    about to be captured looks deceptively good.

    Uses stand-pat: if the static eval already exceeds beta we prune immediately.
    When in check, all legal moves must be considered (not just captures).
    """
    if board.is_checkmate():
        return -99_999 if board.turn == chess.WHITE else 99_999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    stand_pat = evaluate(board)

    if maximizing:
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Escape check with any legal move; otherwise only captures
        if board.is_check():
            candidates = list(board.legal_moves)
        else:
            candidates = [m for m in board.legal_moves
                          if board.is_capture(m) or m.promotion == chess.QUEEN]

        for move in _order_moves(board, candidates):
            board.push(move)
            score = _qsearch(board, alpha, beta, False)
            board.pop()
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        return alpha

    else:
        if stand_pat <= alpha:
            return alpha
        if stand_pat < beta:
            beta = stand_pat

        if board.is_check():
            candidates = list(board.legal_moves)
        else:
            candidates = [m for m in board.legal_moves
                          if board.is_capture(m) or m.promotion == chess.QUEEN]

        for move in _order_moves(board, candidates):
            board.push(move)
            score = _qsearch(board, alpha, beta, True)
            board.pop()
            if score <= alpha:
                return alpha
            if score < beta:
                beta = score
        return beta


# ─────────────────────────────────────────────────────────────────────────────
# MINIMAX WITH ALPHA-BETA  (greatly enhanced)
# ─────────────────────────────────────────────────────────────────────────────
def minimax(board: chess.Board, depth: int,
            alpha: float, beta: float,
            maximizing: bool,
            ply: int = 0,
            null_ok: bool = True) -> float:
    """
    Alpha-beta minimax with:
      • Transposition table cutoffs
      • Check extension (+1 depth when in check)
      • Null-move pruning (R=2/3; skipped near endgame and at root)
      • Killer + history move ordering
      • Late Move Reductions for quiet moves (move 4+, depth >= 3)
      • Quiescence search at the leaves
    """
    # ── Transposition table probe ──────────────────────────────────────────
    h       = chess.polyglot.zobrist_hash(board)
    tt_data = _tt_probe(h)
    tt_move = None

    if tt_data is not None:
        tt_depth, tt_flag, tt_score, tt_move_raw = tt_data
        if tt_depth >= depth:
            if tt_flag == TT_EXACT:
                return tt_score
            if tt_flag == TT_LOWER and tt_score >= beta:
                return tt_score
            if tt_flag == TT_UPPER and tt_score <= alpha:
                return tt_score
        # Use TT move for ordering even if we can't cut
        if tt_move_raw is not None and tt_move_raw in board.legal_moves:
            tt_move = tt_move_raw

    # ── Terminal / leaf ───────────────────────────────────────────────────
    if board.is_game_over():
        return evaluate(board)

    if depth <= 0:
        return _qsearch(board, alpha, beta, maximizing)

    # ── Check extension ────────────────────────────────────────────────────
    in_check = board.is_check()
    if in_check:
        depth += 1

    # ── Null-move pruning ─────────────────────────────────────────────────
    # Skip at root (ply=0), in check, deep endgame (zugzwang risk), depth<3
    eg = _endgame_factor(board)
    if null_ok and ply > 0 and not in_check and depth >= 3 and eg < 0.85:
        R = 3 if depth >= 5 else 2
        board.push(chess.Move.null())
        if maximizing:
            null_score = minimax(board, depth - 1 - R,
                                 beta - 1, beta, False, ply + 1, False)
        else:
            null_score = minimax(board, depth - 1 - R,
                                 alpha, alpha + 1, True, ply + 1, False)
        board.pop()
        if maximizing and null_score >= beta:
            return beta
        if not maximizing and null_score <= alpha:
            return alpha

    # ── Generate + order moves ────────────────────────────────────────────
    moves = _order_moves(board, list(board.legal_moves), tt_move, ply)
    if not moves:          # Should not happen if game_over checked above
        return evaluate(board)

    orig_alpha   = alpha
    orig_beta    = beta
    best_move    = None
    moves_done   = 0

    if maximizing:
        best = float('-inf')
        for move in moves:
            is_cap = board.is_capture(move)
            board.push(move)
            gives_check = board.is_check()

            # Late Move Reductions
            reduce = 0
            if (moves_done >= 3 and depth >= 3
                    and not is_cap and not in_check
                    and not gives_check and move.promotion is None):
                reduce = 1

            if reduce:
                score = minimax(board, depth - 1 - reduce,
                                alpha, beta, False, ply + 1)
                if score > alpha:          # Fail-high on reduced search: re-search
                    score = minimax(board, depth - 1,
                                    alpha, beta, False, ply + 1)
            else:
                score = minimax(board, depth - 1,
                                alpha, beta, False, ply + 1)

            board.pop()
            moves_done += 1

            if score > best:
                best      = score
                best_move = move
            if score > alpha:
                alpha = score
            if beta <= alpha:
                if not is_cap:
                    _update_killers(move, ply)
                    _update_history(move, depth)
                break

        flag = (TT_EXACT if orig_alpha < best < orig_beta
                else (TT_LOWER if best >= orig_beta else TT_UPPER))
        _tt_store(h, depth, flag, best, best_move)
        return best

    else:
        best = float('inf')
        for move in moves:
            is_cap = board.is_capture(move)
            board.push(move)
            gives_check = board.is_check()

            reduce = 0
            if (moves_done >= 3 and depth >= 3
                    and not is_cap and not in_check
                    and not gives_check and move.promotion is None):
                reduce = 1

            if reduce:
                score = minimax(board, depth - 1 - reduce,
                                alpha, beta, True, ply + 1)
                if score < beta:           # Fail-low on reduced search: re-search
                    score = minimax(board, depth - 1,
                                    alpha, beta, True, ply + 1)
            else:
                score = minimax(board, depth - 1,
                                alpha, beta, True, ply + 1)

            board.pop()
            moves_done += 1

            if score < best:
                best      = score
                best_move = move
            if score < beta:
                beta = score
            if beta <= alpha:
                if not is_cap:
                    _update_killers(move, ply)
                    _update_history(move, depth)
                break

        flag = (TT_EXACT if orig_alpha < best < orig_beta
                else (TT_UPPER if best <= orig_alpha else TT_LOWER))
        _tt_store(h, depth, flag, best, best_move)
        return best


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT  (called by the tournament harness)
# ─────────────────────────────────────────────────────────────────────────────
def get_next_move(board: chess.Board,
                  color: chess.Color,
                  depth: int = 3) -> chess.Move:
    """
    Return the best move for `color` from the current `board` position.
    DO NOT rename or change this signature — the harness calls it directly.

    Uses iterative deepening: searches depths 1, 2, … up to
    max(depth, TARGET_DEPTH).  Each iteration re-uses transposition table
    entries from the previous one for superior move ordering and therefore
    far better alpha-beta cutoffs.
    """
    global _KILLERS, _HISTORY

    # Reset per-search state; keep TT (it spans the whole game)
    _KILLERS = [[None, None] for _ in range(128)]
    # Decay history so early-game data does not dominate
    for k in list(_HISTORY.keys()):
        _HISTORY[k] = _HISTORY[k] >> 1   # halve all values
        if _HISTORY[k] == 0:
            del _HISTORY[k]

    target     = max(depth, TARGET_DEPTH)
    maximizing = (color == chess.WHITE)
    best_move  = None
    best_score = float('-inf') if maximizing else float('inf')

    for d in range(1, target + 1):
        iter_best_move  = None
        iter_best_score = float('-inf') if maximizing else float('inf')

        b = board.copy()
        # At the root, order moves using whatever the TT already has
        h_root = chess.polyglot.zobrist_hash(b)
        tt_root = _tt_probe(h_root)
        root_tt_move = tt_root[3] if tt_root and tt_root[3] in b.legal_moves else None

        moves = _order_moves(b, list(b.legal_moves), root_tt_move, ply=0)

        for move in moves:
            b.push(move)
            score = minimax(b, d - 1,
                            float('-inf'), float('inf'),
                            not maximizing, ply=1)
            b.pop()

            if maximizing and score > iter_best_score:
                iter_best_score = score
                iter_best_move  = move
            elif not maximizing and score < iter_best_score:
                iter_best_score = score
                iter_best_move  = move

        if iter_best_move is not None:
            best_move  = iter_best_move
            best_score = iter_best_score

        # Store root result in TT so next iteration uses it for ordering
        if best_move is not None:
            _tt_store(
                chess.polyglot.zobrist_hash(board),
                d,
                TT_EXACT,
                best_score,
                best_move,
            )

    return best_move


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import time

    b = chess.Board()
    print(f"Starting position eval: {evaluate(b):.1f} cp  (should be near 0)")

    for d in (3, 5, TARGET_DEPTH):
        _TT.clear()
        _KILLERS[:] = [[None, None] for _ in range(128)]
        _HISTORY.clear()
        t0   = time.time()
        move = get_next_move(b, chess.WHITE, depth=d)
        elapsed = time.time() - t0
        print(f"[team_goraieb] depth={d:2d}: {b.san(move):6s}  ({elapsed:.2f}s)")
