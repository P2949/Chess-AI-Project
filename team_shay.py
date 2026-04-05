from __future__ import annotations

import chess
import chess.polyglot

# Terminal scores
MATE_SCORE = 100_000

# PeSTO phase
PHASE_MAX = 24
PHASE_ADD_KNIGHT = 1
PHASE_ADD_BISHOP = 1
PHASE_ADD_ROOK = 2
PHASE_ADD_QUEEN = 4

# Material
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}

# Tempo
TEMPO_MG = 28
TEMPO_EG = 12

# Mobility (pseudo-legal attacks per piece, then scaled)
MOBILITY_WEIGHTS = {
    chess.KNIGHT: 4,
    chess.BISHOP: 5,
    chess.ROOK: 3,
    chess.QUEEN: 2,
}
MOBILITY_WEIGHT = 0.06

# Bishop pair imbalance
BISHOP_PAIR_BONUS = 32.0

# Pawn structure
PAWN_DOUBLED_PENALTY = 20.0
PAWN_ISOLATED_PENALTY = 15.0
# Passed-pawn bonus by rank advance
PASSED_PAWN_RANK_BONUS = [0, 10, 20, 35, 60, 90, 130, 0]

# Rook placement
ROOK_OPEN_FILE_BONUS = 26.0
ROOK_SEMI_OPEN_BONUS = 13.0
ROOK_SEVENTH_RANK_BONUS = 12.0
CONNECTED_ROOKS_BONUS = 14.0

# King safety (scaled by middlegame weight)
KING_CASTLED_BONUS = 20.0
KING_UNCASTLED_CENTER_PENALTY = 18.0

# Draw detection
DRAW_REPETITION_PLY = 3

# MVV–LVA capture ordering
MVV_LVA_VICTIM_SCALE = 10
MVV_LVA_FALLBACK_CP = 100

# Move-ordering bonuses
ORDER_TT_MOVE = 20_000
ORDER_CAPTURE_BASE = 10_000
ORDER_PROMOTION_BASE = 9_000
ORDER_KILLER_PRIMARY = 8_000
ORDER_KILLER_SECONDARY = 7_000
ORDER_COUNTERMOVE = 18_500
ORDER_QUIET_CHECK_BONUS = 6_500

# Quiescence
QUIESCENCE_MAX_PLIES = 9

# Delta pruning
QUIESCENCE_DELTA_CP = PIECE_VALUES[chess.QUEEN] + 75
# Skip clearly losing captures in quiescence unless they give check.
QUIESCENCE_BAD_CAPTURE_CP = 180

# Selective pruning / reductions in main search
FUTILITY_MARGIN_CP = 220
LMR_MIN_DEPTH = 3
LMR_START_MOVE = 4
LMR_REDUCTION_PLIES = 1
LMP_CUTOFFS = {1: 24, 2: 36, 3: 48}

# Null-move pruning (big speed gain when tuned conservatively)
NULL_MOVE_ENABLE = True
NULL_MOVE_MIN_DEPTH = 3
NULL_MOVE_BASE_REDUCTION = 2
NULL_MOVE_DEPTH_DIVISOR = 4
NULL_MOVE_STATIC_MARGIN_CP = 120

# Memory 
TT_MAX_ENTRIES = 600_000
HISTORY_MAX_ENTRIES = 120_000
COUNTERMOVE_MAX_ENTRIES = 80_000
STATIC_EVAL_CACHE_MAX_ENTRIES = 300_000

# Killer slots are indexed by ply; keep ≥ maximum search depth you expect
KILLER_TABLE_PLIES = 64

# Lightweight opening book (no external engine; hand-crafted principled lines).
OPENING_BOOK_MAX_PLIES = 10

# PeSTO tables middlegame
PST_MG = {
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        98, 134, 61, 95, 68, 126, 34, -11,
        -6, 7, 26, 31, 62, 22, -8, -17,
        -14, 13, 6, 21, 23, 12, 17, -23,
        -27, -2, -5, 12, 17, 6, 10, -25,
        -26, -4, -4, -10, 3, 3, 33, -12,
        -35, -1, -20, -23, -15, 24, 38, -22,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    chess.KNIGHT: [
        -167, -89, -34, -49, 61, -97, -15, -107,
        -73, -41, 72, 36, 23, 62, 7, -17,
        -47, 60, 37, 65, 84, 129, 73, 44,
        -9, 17, 19, 53, 37, 69, 18, 22,
        -13, 4, 16, 13, 28, 19, 21, -8,
        -23, -9, 12, 10, 19, 17, 25, -16,
        -29, -53, -12, -3, -1, 18, -14, -19,
        -105, -21, -58, -33, -17, -28, -19, -23,
    ],
    chess.BISHOP: [
        -29, 4, -82, -37, -25, -42, 7, -8,
        -26, 16, -18, -13, 30, 59, 18, -47,
        -16, 37, 43, 40, 35, 50, 37, -2,
        -4, 5, 19, 50, 37, 37, 7, -2,
        -6, 13, 13, 26, 34, 12, 10, 4,
        0, 15, 15, 15, 14, 27, 18, 10,
        4, 15, 16, 0, 7, 21, 33, 1,
        -33, -3, -14, -21, -13, -12, -39, -21,
    ],
    chess.ROOK: [
        32, 42, 32, 51, 63, 9, 31, 43,
        27, 32, 58, 62, 80, 67, 26, 44,
        -5, 19, 26, 36, 17, 45, 61, 16,
        -24, -11, 7, 26, 24, 35, -8, -20,
        -36, -26, -12, -1, 9, -7, 6, -23,
        -45, -25, -16, -17, 3, 0, -5, -33,
        -44, -16, -20, -9, -1, 11, -6, -71,
        -19, -13, 1, 17, 16, 7, -37, -26,
    ],
    chess.QUEEN: [
        -28, 0, 29, 12, 59, 44, 43, 45,
        -24, -39, -5, 1, -16, 57, 28, 54,
        -13, -17, 7, 8, 29, 56, 47, 57,
        -27, -27, -16, -16, -1, 17, -2, 1,
        -9, -26, -9, -10, -2, -4, 3, -3,
        -14, 2, -11, -2, -5, 2, 14, 5,
        -35, -8, 11, 2, 8, 15, -3, 1,
        -1, -18, -9, 10, -15, -25, -31, -50,
    ],
    chess.KING: [
        -65, 23, 16, -15, -56, -34, 2, 13,
        29, -1, -20, -7, -8, -4, -38, -29,
        -9, 24, 2, -16, -20, 6, 22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49, -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
        1, 7, -8, -64, -43, -16, 9, 8,
        -15, 36, 12, -54, 8, -28, 24, 14,
    ],
}

# PeSTO tables endgame
PST_EG = {
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        178, 173, 158, 134, 147, 132, 165, 187,
        94, 100, 85, 67, 56, 53, 82, 84,
        32, 24, 13, 5, -2, 4, 17, 17,
        13, 9, -3, -7, -7, -8, 3, -1,
        4, 7, -6, 1, 0, -5, -1, -8,
        13, 8, 8, 10, 13, 0, 2, -7,
        0, 0, 0, 0, 0, 0, 0, 0,
    ],
    chess.KNIGHT: [
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25, -8, -25, -2, -9, -25, -24, -52,
        -24, -20, 10, 9, -1, -9, -19, -41,
        -17, 3, 22, 22, 22, 11, 8, -18,
        -18, -6, 16, 25, 16, 17, 4, -18,
        -23, -3, -1, 15, 10, -3, -20, -22,
        -42, -20, -10, -5, -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    ],
    chess.BISHOP: [
        -14, -21, -11, -8, -7, -9, -17, -24,
        -8, -4, 7, -12, -3, -13, -4, -14,
        2, -8, 0, -1, -2, 6, 0, 4,
        -3, 9, 12, 9, 14, 10, 3, 2,
        -6, 3, 13, 19, 7, 10, -3, -9,
        -12, -3, 8, 10, 13, 3, -7, -15,
        -14, -18, -7, -1, 4, -9, -15, -27,
        -23, -9, -23, -5, -9, -16, -5, -17,
    ],
    chess.ROOK: [
        13, 10, 18, 15, 12, 12, 8, 5,
        11, 13, 13, 11, -3, 3, 8, 3,
        7, 7, 7, 5, 4, -3, -5, -3,
        4, 3, 13, 1, 2, 1, -1, 2,
        3, 5, 8, 4, -5, -6, -8, -11,
        -4, 0, -5, -1, -7, -12, -8, -16,
        -6, -6, 0, 2, -9, -9, -11, -3,
        -9, 2, 3, -1, -5, -13, 4, -20,
    ],
    chess.QUEEN: [
        -9, 22, 22, 27, 27, 19, 10, 20,
        -17, 20, 32, 41, 58, 25, 30, 0,
        -20, 6, 9, 49, 47, 35, 19, 9,
        3, 22, 24, 45, 57, 40, 57, 36,
        -18, 28, 19, 47, 31, 34, 12, 11,
        -16, -27, 15, 6, 9, 17, 10, 5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43, -5, -32, -20, -41,
    ],
    chess.KING: [
        -74, -35, -18, -18, -11, 15, 4, -17,
        -12, 17, 14, 17, 17, 38, 23, 11,
        10, 17, 23, 15, 20, 45, 44, 13,
        -8, 22, 24, 27, 26, 33, 26, 3,
        -18, -4, 21, 24, 27, 23, 9, -11,
        -19, -3, 11, 21, 23, 16, 7, -9,
        -27, -11, 4, 13, 14, 4, -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43,
    ],
}

# Search globals
TRANSPOSITION_TABLE: dict[int, tuple[int, int, float, chess.Move | None]] = {}
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
STATIC_EVAL_CACHE: dict[int, float] = {}

_KILLERS: list[list[chess.Move | None]] = [
    [None, None] for _ in range(KILLER_TABLE_PLIES)
]
_HISTORY: dict[tuple[int, int], int] = {}
_COUNTERMOVE: dict[tuple[int, int], chess.Move] = {}

_OPENING_BOOK: dict[tuple[str, ...], list[str]] = {
    # Start position (White)
    (): ["e2e4", "d2d4", "c2c4", "g1f3"],

    # First Black responses
    ("e2e4",): ["c7c5", "e7e5", "e7e6", "c7c6"],
    ("d2d4",): ["g8f6", "d7d5", "e7e6"],
    ("c2c4",): ["e7e5", "g8f6", "c7c5"],
    ("g1f3",): ["d7d5", "g8f6", "c7c5"],

    # White follow-ups
    ("e2e4", "c7c5"): ["g1f3", "d2d4", "c2c3"],
    ("e2e4", "e7e5"): ["g1f3", "f1c4", "d2d4"],
    ("d2d4", "g8f6"): ["c2c4", "g1f3", "c1g5"],
    ("d2d4", "d7d5"): ["c2c4", "g1f3"],
    ("c2c4", "e7e5"): ["g2g3", "b1c3", "g1f3"],

    # Black follow-ups
    ("e2e4", "c7c5", "g1f3"): ["d7d6", "b8c6", "e7e6"],
    ("e2e4", "e7e5", "g1f3"): ["b8c6", "g8f6"],
    ("d2d4", "g8f6", "c2c4"): ["e7e6", "g7g6", "c7c5"],
    ("d2d4", "d7d5", "c2c4"): ["e7e6", "c7c6", "d5c4"],
}

# Pawn-structure caches
_SQ_FILE = [chess.square_file(s) for s in chess.SQUARES]
_SQ_RANK = [chess.square_rank(s) for s in chess.SQUARES]


def _opening_book_move(board: chess.Board) -> chess.Move | None:
    """Return a principled opening move if in the small internal repertoire."""
    if len(board.move_stack) > OPENING_BOOK_MAX_PLIES:
        return None
    key = tuple(m.uci() for m in board.move_stack)
    for uci in _OPENING_BOOK.get(key, []):
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            return move
    return None


def _game_phase(board: chess.Board) -> int:
    """PeSTO-style phase: N/B=1, R=2, Q=4 per side; capped at 24."""
    ph = 0
    for c in (chess.WHITE, chess.BLACK):
        ph += PHASE_ADD_KNIGHT * len(board.pieces(chess.KNIGHT, c))
        ph += PHASE_ADD_BISHOP * len(board.pieces(chess.BISHOP, c))
        ph += PHASE_ADD_ROOK * len(board.pieces(chess.ROOK, c))
        ph += PHASE_ADD_QUEEN * len(board.pieces(chess.QUEEN, c))
    return min(ph, PHASE_MAX)


def _pst_index(sq: int, color: chess.Color) -> int:
    return chess.square_mirror(sq) if color == chess.WHITE else sq


def _pawn_structure(board: chess.Board, color: chess.Color, eg_w: float) -> float:
    pawns = list(board.pieces(chess.PAWN, color))
    if not pawns:
        return 0.0

    opp = list(board.pieces(chess.PAWN, not color))
    fcnt: dict[int, int] = {}
    for sq in pawns:
        f = _SQ_FILE[sq]
        fcnt[f] = fcnt.get(f, 0) + 1

    opp_by_file: dict[int, list[int]] = {}
    for sq in opp:
        f = _SQ_FILE[sq]
        opp_by_file.setdefault(f, []).append(_SQ_RANK[sq])

    score = 0.0
    for sq in pawns:
        f, r = _SQ_FILE[sq], _SQ_RANK[sq]
        if fcnt[f] > 1:
            score -= PAWN_DOUBLED_PENALTY

        has_neighbor = (f > 0 and (f - 1) in fcnt) or (f < 7 and (f + 1) in fcnt)
        if not has_neighbor:
            score -= PAWN_ISOLATED_PENALTY

        passed = True
        for df in (-1, 0, 1):
            for orank in opp_by_file.get(f + df, []):
                if color == chess.WHITE and orank > r:
                    passed = False
                    break
                if color == chess.BLACK and orank < r:
                    passed = False
                    break
            if not passed:
                break
        if passed:
            adv = r if color == chess.WHITE else 7 - r
            score += float(PASSED_PAWN_RANK_BONUS[adv]) * (0.60 + eg_w)
    return score


def _has_non_pawn_material(board: chess.Board, color: chess.Color) -> bool:
    """Used to disable null-move in likely zugzwang-ish pawn endings."""
    return any(
        board.pieces(pt, color)
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )


def _rooks_connected(board: chess.Board, sq1: int, sq2: int) -> bool:
    """
    True when two rooks see each other on same rank/file with no blockers.
    Cheap positional signal; rewarding coordination is usually helpful at low depth.
    """
    f1, r1 = chess.square_file(sq1), chess.square_rank(sq1)
    f2, r2 = chess.square_file(sq2), chess.square_rank(sq2)
    if f1 == f2:
        step = 8 if r2 > r1 else -8
        for sq in range(sq1 + step, sq2, step):
            if board.piece_at(sq) is not None:
                return False
        return True
    if r1 == r2:
        step = 1 if f2 > f1 else -1
        for sq in range(sq1 + step, sq2, step):
            if board.piece_at(sq) is not None:
                return False
        return True
    return False


def _update_history(move: chess.Move, delta: int) -> None:
    """History with clamping to avoid runaway scores."""
    key = (move.from_square, move.to_square)
    _HISTORY[key] = max(-50_000, min(50_000, _HISTORY.get(key, 0) + delta))


def _hanging_and_loose(board: chess.Board, mg_w: float) -> float:
    """
    Tactical pressure term for shallow search:
    - attacked & undefended pieces (hanging) get penalized strongly
    - attacked by a cheaper piece gets an extra "loose-piece" penalty
    """
    if mg_w <= 0.05:
        return 0.0

    hanging = {
        chess.PAWN: 10.0,
        chess.KNIGHT: 28.0,
        chess.BISHOP: 30.0,
        chess.ROOK: 45.0,
        chess.QUEEN: 70.0,
    }
    loose = {
        chess.PAWN: 4.0,
        chess.KNIGHT: 9.0,
        chess.BISHOP: 10.0,
        chess.ROOK: 14.0,
        chess.QUEEN: 18.0,
    }

    score = 0.0
    for sq, piece in board.piece_map().items():
        pt = piece.piece_type
        if pt == chess.KING:
            continue

        enemy = not piece.color
        attackers = board.attackers(enemy, sq)
        if not attackers:
            continue
        defenders = board.attackers(piece.color, sq)

        # White piece under fire is bad for White (negative), and vice-versa.
        sign = -1.0 if piece.color == chess.WHITE else 1.0

        if len(defenders) == 0:
            score += sign * hanging.get(pt, 0.0)
        elif len(attackers) > len(defenders):
            score += sign * (0.55 * hanging.get(pt, 0.0))

        min_attacker = PIECE_VALUES[chess.QUEEN]
        for a_sq in attackers:
            a_piece = board.piece_at(a_sq)
            if a_piece is not None:
                min_attacker = min(min_attacker, PIECE_VALUES[a_piece.piece_type])
        if min_attacker < PIECE_VALUES[pt]:
            score += sign * loose.get(pt, 0.0)

    return score * mg_w


def evaluate(board: chess.Board) -> float:
    if board.is_checkmate():
        return -float(MATE_SCORE) if board.turn == chess.WHITE else float(MATE_SCORE)

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_repetition(DRAW_REPETITION_PLY)
        or board.is_fifty_moves()
    ):
        return 0.0

    hash_key = chess.polyglot.zobrist_hash(board)
    cached = STATIC_EVAL_CACHE.get(hash_key)
    if cached is not None:
        return cached

    mg_score = 0.0
    eg_score = 0.0
    phase = 0
    w_mob = 0.0
    b_mob = 0.0

    for sq, piece in board.piece_map().items():
        sign = 1.0 if piece.color == chess.WHITE else -1.0
        idx = _pst_index(sq, piece.color)
        pt = piece.piece_type
        base = PIECE_VALUES[pt]
        mg_score += sign * (base + PST_MG[pt][idx])
        eg_score += sign * (base + PST_EG[pt][idx])

        if pt == chess.KNIGHT:
            phase += PHASE_ADD_KNIGHT
        elif pt == chess.BISHOP:
            phase += PHASE_ADD_BISHOP
        elif pt == chess.ROOK:
            phase += PHASE_ADD_ROOK
        elif pt == chess.QUEEN:
            phase += PHASE_ADD_QUEEN

        mob_w = MOBILITY_WEIGHTS.get(pt, 0)
        if mob_w:
            atk_count = len(board.attacks(sq))
            if piece.color == chess.WHITE:
                w_mob += atk_count * mob_w
            else:
                b_mob += atk_count * mob_w

    phase = min(phase, PHASE_MAX)
    eg_w = (PHASE_MAX - phase) / float(PHASE_MAX)
    mg_w = 1.0 - eg_w
    score = mg_score * mg_w + eg_score * eg_w

    # Tapered tempo
    tempo = TEMPO_MG * mg_w + TEMPO_EG * eg_w
    if board.turn == chess.WHITE:
        score += tempo
    else:
        score -= tempo

    score += (w_mob - b_mob) * MOBILITY_WEIGHT

    # Bishop pair
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += BISHOP_PAIR_BONUS
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= BISHOP_PAIR_BONUS

    score += _pawn_structure(board, chess.WHITE, eg_w)
    score -= _pawn_structure(board, chess.BLACK, eg_w)

    w_files = {_SQ_FILE[s] for s in board.pieces(chess.PAWN, chess.WHITE)}
    b_files = {_SQ_FILE[s] for s in board.pieces(chess.PAWN, chess.BLACK)}
    w_rooks = list(board.pieces(chess.ROOK, chess.WHITE))
    b_rooks = list(board.pieces(chess.ROOK, chess.BLACK))
    for sq in w_rooks:
        f = _SQ_FILE[sq]
        if f not in w_files and f not in b_files:
            score += ROOK_OPEN_FILE_BONUS
        elif f not in w_files:
            score += ROOK_SEMI_OPEN_BONUS
        if _SQ_RANK[sq] == 6:
            score += ROOK_SEVENTH_RANK_BONUS
    for sq in b_rooks:
        f = _SQ_FILE[sq]
        if f not in b_files and f not in w_files:
            score -= ROOK_OPEN_FILE_BONUS
        elif f not in b_files:
            score -= ROOK_SEMI_OPEN_BONUS
        if _SQ_RANK[sq] == 1:
            score -= ROOK_SEVENTH_RANK_BONUS

    if len(w_rooks) >= 2 and _rooks_connected(board, w_rooks[0], w_rooks[1]):
        score += CONNECTED_ROOKS_BONUS
    if len(b_rooks) >= 2 and _rooks_connected(board, b_rooks[0], b_rooks[1]):
        score -= CONNECTED_ROOKS_BONUS

    # Castled kings are safer in the middlegame; central uncastled kings are
    # usually more vulnerable once castling rights are gone.
    w_king = board.king(chess.WHITE)
    b_king = board.king(chess.BLACK)
    if w_king in (chess.G1, chess.C1):
        score += KING_CASTLED_BONUS * mg_w
    elif (w_king in (chess.E1, chess.D1, chess.F1)) and not board.has_castling_rights(chess.WHITE):
        score -= KING_UNCASTLED_CENTER_PENALTY * mg_w
    if b_king in (chess.G8, chess.C8):
        score -= KING_CASTLED_BONUS * mg_w
    elif (b_king in (chess.E8, chess.D8, chess.F8)) and not board.has_castling_rights(chess.BLACK):
        score += KING_UNCASTLED_CENTER_PENALTY * mg_w

    score += _hanging_and_loose(board, mg_w)

    STATIC_EVAL_CACHE[hash_key] = score
    return score


def _mvv_lva_score(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    v = (
        PIECE_VALUES.get(victim.piece_type, MVV_LVA_FALLBACK_CP)
        if victim
        else MVV_LVA_FALLBACK_CP
    )
    a = (
        PIECE_VALUES.get(attacker.piece_type, MVV_LVA_FALLBACK_CP)
        if attacker
        else MVV_LVA_FALLBACK_CP
    )
    return v * MVV_LVA_VICTIM_SCALE - a


def score_move(board: chess.Board,move: chess.Move,ply: int = 0,tt_move: chess.Move | None = None,prev_move: chess.Move | None = None) -> int:
    if tt_move is not None and move == tt_move:
        return ORDER_TT_MOVE
    if prev_move is not None:
        cm = _COUNTERMOVE.get((prev_move.from_square, prev_move.to_square))
        if cm == move:
            return ORDER_COUNTERMOVE
    if board.is_capture(move):
        return ORDER_CAPTURE_BASE + _mvv_lva_score(board, move)
    if move.promotion:
        return ORDER_PROMOTION_BASE + PIECE_VALUES.get(move.promotion, 0)
    if ply < KILLER_TABLE_PLIES and move == _KILLERS[ply][0]:
        return ORDER_KILLER_PRIMARY
    if ply < KILLER_TABLE_PLIES and move == _KILLERS[ply][1]:
        return ORDER_KILLER_SECONDARY
    if board.gives_check(move):
        return ORDER_QUIET_CHECK_BONUS
    return _HISTORY.get((move.from_square, move.to_square), 0)


def order_moves(
    board: chess.Board,
    moves: list[chess.Move],
    ply: int = 0,
    tt_move: chess.Move | None = None,
    prev_move: chess.Move | None = None,
) -> list[chess.Move]:
    return sorted(moves, key=lambda m: score_move(board, m, ply, tt_move, prev_move), reverse=True)


def quiescence(board: chess.Board,alpha: float,beta: float,maximizing: bool,qply: int = 0) -> float:
    if qply > QUIESCENCE_MAX_PLIES:
        return evaluate(board)

    in_check = board.is_check()

    if not in_check:
        stand = evaluate(board)
        if maximizing:
            if stand >= beta:
                return beta
            if stand + QUIESCENCE_DELTA_CP < alpha:
                return alpha
            alpha = max(alpha, stand)
        else:
            if stand <= alpha:
                return alpha
            if stand - QUIESCENCE_DELTA_CP > beta:
                return beta
            beta = min(beta, stand)
        tactical = [m for m in board.legal_moves if board.is_capture(m) or m.promotion == chess.QUEEN]
    else:
        tactical = list(board.legal_moves)

    for move in order_moves(board, tactical, qply, None):
        # Cheap SEE-like filter: losing captures are usually noise in qsearch.
        if (
            not in_check
            and board.is_capture(move)
            and not move.promotion
            and _mvv_lva_score(board, move) < -QUIESCENCE_BAD_CAPTURE_CP
            and not board.gives_check(move)
        ):
            continue

        board.push(move)
        score = quiescence(board, alpha, beta, not maximizing, qply + 1)
        board.pop()
        if maximizing:
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        else:
            beta = min(beta, score)
            if beta <= alpha:
                break

    return alpha if maximizing else beta


def minimax(
    board: chess.Board,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    ply: int = 0,
    last_capture_sq: int | None = None,
    prev_move: chess.Move | None = None,
) -> float:
    in_check = board.is_check()
    rem = depth + (1 if in_check else 0)

    hash_key = chess.polyglot.zobrist_hash(board)
    tt_move: chess.Move | None = None

    if hash_key in TRANSPOSITION_TABLE:
        tt_depth, tt_flag, tt_score, tt_move = TRANSPOSITION_TABLE[hash_key]
        if tt_depth >= rem:
            if tt_flag == TT_EXACT:
                return tt_score
            if tt_flag == TT_LOWER:
                alpha = max(alpha, tt_score)
            elif tt_flag == TT_UPPER:
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score

    if board.is_checkmate():
        return -float(MATE_SCORE - ply) if board.turn == chess.WHITE else float(MATE_SCORE - ply)
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_repetition(DRAW_REPETITION_PLY)
        or board.is_fifty_moves()
    ):
        return 0.0
    if rem <= 0:
        return quiescence(board, alpha, beta, maximizing, 0)

    # Null-move pruning: if "doing nothing" already fails high/low in a reduced
    # search, this node is very likely cuttable. Keep guards conservative to
    # avoid zugzwang pathologies.
    if (
        NULL_MOVE_ENABLE
        and rem >= NULL_MOVE_MIN_DEPTH
        and ply > 0
        and not in_check
        and _has_non_pawn_material(board, board.turn)
        and _has_non_pawn_material(board, not board.turn)
        and abs(alpha) < MATE_SCORE - 1_000
        and abs(beta) < MATE_SCORE - 1_000
    ):
        static_eval = evaluate(board)
        R = NULL_MOVE_BASE_REDUCTION + (rem // NULL_MOVE_DEPTH_DIVISOR)
        board.push(chess.Move.null())
        if maximizing and static_eval >= beta - NULL_MOVE_STATIC_MARGIN_CP:
            null_score = minimax(
                board, rem - 1 - R, beta - 1.0, beta, False, ply + 1, None, None
            )
            board.pop()
            if null_score >= beta:
                return beta
        elif (not maximizing) and static_eval <= alpha + NULL_MOVE_STATIC_MARGIN_CP:
            null_score = minimax(
                board, rem - 1 - R, alpha, alpha + 1.0, True, ply + 1, None, None
            )
            board.pop()
            if null_score <= alpha:
                return alpha
        else:
            board.pop()

    moves = order_moves(board, list(board.legal_moves), ply, tt_move, prev_move)
    if not moves:
        return evaluate(board)

    orig_alpha = alpha
    orig_beta = beta
    best_score = float("-inf") if maximizing else float("inf")
    best_move: chess.Move | None = None

    # Near frontier, skip hopeless quiet moves (futility pruning).
    futility_eval = evaluate(board) if (rem == 1 and not in_check) else None
    searched_any = False
    quiet_count = 0
    quiet_tried: list[chess.Move] = []

    if maximizing:
        for move_idx, move in enumerate(moves):
            is_capture = board.is_capture(move)
            is_quiet = (not is_capture) and (not move.promotion)

            if (
                futility_eval is not None
                and is_quiet
                and not board.gives_check(move)
                and futility_eval + FUTILITY_MARGIN_CP <= alpha
            ):
                quiet_count += 1
                continue
            if (
                is_quiet
                and not in_check
                and rem in LMP_CUTOFFS
                and quiet_count >= LMP_CUTOFFS[rem]
                and not board.gives_check(move)
            ):
                continue

            recapture = (
                last_capture_sq is not None
                and is_capture
                and move.to_square == last_capture_sq
            )
            child_rem = rem - 1 + (1 if recapture else 0)
            cap_sq = move.to_square if is_capture else None
            do_lmr = (
                rem >= LMR_MIN_DEPTH
                and move_idx >= LMR_START_MOVE
                and is_quiet
                and not in_check
                and not recapture
            )

            board.push(move)
            searched_any = True
            if move_idx == 0:
                score = minimax(board, child_rem, alpha, beta, False, ply + 1, cap_sq, move)
            elif do_lmr and alpha != float("-inf"):
                reduced = max(0, child_rem - LMR_REDUCTION_PLIES)
                score = minimax(board, reduced, alpha, alpha + 1.0, False, ply + 1, cap_sq, move)
                if score > alpha:
                    score = minimax(board, child_rem, alpha, beta, False, ply + 1, cap_sq, move)
            else:
                # PVS: null-window probe first, full re-search only on fail-high.
                score = minimax(board, child_rem, alpha, alpha + 1.0, False, ply + 1, cap_sq, move)
                if score > alpha:
                    score = minimax(board, child_rem, alpha, beta, False, ply + 1, cap_sq, move)
            board.pop()
            if is_quiet:
                quiet_count += 1
                quiet_tried.append(move)

            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if beta <= alpha:
                if is_quiet:
                    if ply < KILLER_TABLE_PLIES and move != _KILLERS[ply][0]:
                        _KILLERS[ply][1] = _KILLERS[ply][0]
                        _KILLERS[ply][0] = move
                    _update_history(move, rem * rem)
                    malus = max(1, rem)
                    for qm in quiet_tried:
                        if qm != move:
                            _update_history(qm, -malus)
                    if prev_move is not None:
                        _COUNTERMOVE[(prev_move.from_square, prev_move.to_square)] = move
                break
    else:
        for move_idx, move in enumerate(moves):
            is_capture = board.is_capture(move)
            is_quiet = (not is_capture) and (not move.promotion)

            if (
                futility_eval is not None
                and is_quiet
                and not board.gives_check(move)
                and futility_eval - FUTILITY_MARGIN_CP >= beta
            ):
                quiet_count += 1
                continue
            if (
                is_quiet
                and not in_check
                and rem in LMP_CUTOFFS
                and quiet_count >= LMP_CUTOFFS[rem]
                and not board.gives_check(move)
            ):
                continue

            recapture = (
                last_capture_sq is not None
                and is_capture
                and move.to_square == last_capture_sq
            )
            child_rem = rem - 1 + (1 if recapture else 0)
            cap_sq = move.to_square if is_capture else None
            do_lmr = (
                rem >= LMR_MIN_DEPTH
                and move_idx >= LMR_START_MOVE
                and is_quiet
                and not in_check
                and not recapture
            )

            board.push(move)
            searched_any = True
            if move_idx == 0:
                score = minimax(board, child_rem, alpha, beta, True, ply + 1, cap_sq, move)
            elif do_lmr and beta != float("inf"):
                reduced = max(0, child_rem - LMR_REDUCTION_PLIES)
                score = minimax(board, reduced, beta - 1.0, beta, True, ply + 1, cap_sq, move)
                if score < beta:
                    score = minimax(board, child_rem, alpha, beta, True, ply + 1, cap_sq, move)
            else:
                # PVS: null-window probe first, full re-search only on fail-low.
                score = minimax(board, child_rem, beta - 1.0, beta, True, ply + 1, cap_sq, move)
                if score < beta:
                    score = minimax(board, child_rem, alpha, beta, True, ply + 1, cap_sq, move)
            board.pop()
            if is_quiet:
                quiet_count += 1
                quiet_tried.append(move)

            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, score)
            if beta <= alpha:
                if is_quiet:
                    if ply < KILLER_TABLE_PLIES and move != _KILLERS[ply][0]:
                        _KILLERS[ply][1] = _KILLERS[ply][0]
                        _KILLERS[ply][0] = move
                    _update_history(move, rem * rem)
                    malus = max(1, rem)
                    for qm in quiet_tried:
                        if qm != move:
                            _update_history(qm, -malus)
                    if prev_move is not None:
                        _COUNTERMOVE[(prev_move.from_square, prev_move.to_square)] = move
                break

    if not searched_any:
        return futility_eval if futility_eval is not None else evaluate(board)

    tt_flag = TT_EXACT
    if best_score <= orig_alpha:
        tt_flag = TT_UPPER
    elif best_score >= orig_beta:
        tt_flag = TT_LOWER
    TRANSPOSITION_TABLE[hash_key] = (rem, tt_flag, best_score, best_move)
    return best_score


def get_next_move(board: chess.Board, color: chess.Color, depth: int = 3) -> chess.Move:
    global TRANSPOSITION_TABLE, _KILLERS, _HISTORY, STATIC_EVAL_CACHE, _COUNTERMOVE

    # New game boundary: clear long-lived heuristics to avoid cross-game bias.
    if len(board.move_stack) == 0:
        TRANSPOSITION_TABLE.clear()
        STATIC_EVAL_CACHE.clear()
        _HISTORY.clear()
        _COUNTERMOVE.clear()

    book_move = _opening_book_move(board)
    if book_move is not None:
        return book_move

    if len(TRANSPOSITION_TABLE) > TT_MAX_ENTRIES:
        TRANSPOSITION_TABLE.clear()
    if len(STATIC_EVAL_CACHE) > STATIC_EVAL_CACHE_MAX_ENTRIES:
        STATIC_EVAL_CACHE.clear()
    if len(_HISTORY) > HISTORY_MAX_ENTRIES:
        _HISTORY.clear()
    if len(_COUNTERMOVE) > COUNTERMOVE_MAX_ENTRIES:
        _COUNTERMOVE.clear()
    _KILLERS[:] = [[None, None] for _ in range(KILLER_TABLE_PLIES)]

    b = board.copy()
    maximizing = color == chess.WHITE
    legal_moves = list(b.legal_moves)
    if not legal_moves:
        return None

    # Iterative deepening to requested depth. At depth=3 this is still cheap, and
    # gives much better ordering for the deepest iteration.
    best_move: chess.Move | None = legal_moves[0]
    prev_score = 0.0

    for d in range(1, max(1, depth) + 1):
        # Aspiration windows from previous iteration score.
        use_asp = d >= 2 and best_move is not None
        window = 70.0
        alpha = prev_score - window if use_asp else float("-inf")
        beta = prev_score + window if use_asp else float("inf")
        retries = 0

        while True:
            root_tt_move = None
            root_tt = TRANSPOSITION_TABLE.get(chess.polyglot.zobrist_hash(b))
            if root_tt is not None and root_tt[3] in legal_moves:
                root_tt_move = root_tt[3]

            moves = order_moves(b, legal_moves, 0, root_tt_move, None)
            if best_move is not None and best_move in moves:
                moves.remove(best_move)
                moves.insert(0, best_move)

            cur_best_move: chess.Move | None = None
            cur_best_score = float("-inf") if maximizing else float("inf")
            a, bw = alpha, beta

            for move in moves:
                root_cap_sq = move.to_square if b.is_capture(move) else None
                b.push(move)
                score = minimax(
                    b,
                    d - 1,
                    a,
                    bw,
                    not maximizing,
                    1,
                    root_cap_sq,
                    move,
                )
                b.pop()

                if maximizing:
                    if score > cur_best_score:
                        cur_best_score, cur_best_move = score, move
                    a = max(a, score)
                    if bw <= a:
                        break
                else:
                    if score < cur_best_score:
                        cur_best_score, cur_best_move = score, move
                    bw = min(bw, score)
                    if bw <= a:
                        break

            if cur_best_move is not None:
                best_move = cur_best_move
                prev_score = cur_best_score

            if not use_asp:
                break

            if prev_score <= alpha:
                alpha -= window
                window *= 2.0
                retries += 1
                if retries >= 5:
                    alpha, beta = float("-inf"), float("inf")
                    use_asp = False
                continue
            if prev_score >= beta:
                beta += window
                window *= 2.0
                retries += 1
                if retries >= 5:
                    alpha, beta = float("-inf"), float("inf")
                    use_asp = False
                continue
            break

    return best_move if best_move is not None else legal_moves[0]


if __name__ == "__main__":
    _b = chess.Board()
    _m = get_next_move(_b, chess.WHITE, depth=3)
    print(f"[team_shay] Opening move: {_b.san(_m)}")
