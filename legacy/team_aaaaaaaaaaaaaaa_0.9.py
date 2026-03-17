
import chess
import chess.polyglot

MATE_SCORE = 100000
INF = 10**9

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000,
}

# PeSTO piece-square tables (middlegame / endgame)
PST_MG = {
    chess.PAWN: [
          0,   0,   0,   0,   0,   0,   0,   0,
         98, 134,  61,  95,  68, 126,  34, -11,
         -6,   7,  26,  31,  62,  22,  -8, -17,
        -14,  13,   6,  21,  23,  12,  17, -23,
        -27,  -2,  -5,  12,  17,   6,  10, -25,
        -26,  -4,  -4, -10,   3,   3,  33, -12,
        -35,  -1, -20, -23, -15,  24,  38, -22,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    chess.KNIGHT: [
        -167, -89, -34, -49,  61, -97, -15, -107,
         -73, -41,  72,  36,  23,  62,   7,  -17,
         -47,  60,  37,  65,  84, 129,  73,   44,
          -9,  17,  19,  53,  37,  69,  18,   22,
         -13,   4,  16,  13,  28,  19,  21,   -8,
         -23,  -9,  12,  10,  19,  17,  25,  -16,
         -29, -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    ],
    chess.BISHOP: [
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
         -4,   5,  19,  50,  37,  37,   7,  -2,
         -6,  13,  13,  26,  34,  12,  10,   4,
          0,  15,  15,  15,  14,  27,  18,  10,
          4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    ],
    chess.ROOK: [
         32,  42,  32,  51,  63,   9,  31,  43,
         27,  32,  58,  62,  80,  67,  26,  44,
         -5,  19,  26,  36,  17,  45,  61,  16,
        -24, -11,   7,  26,  24,  35,  -8, -20,
        -36, -26, -12,  -1,   9,  -7,   6, -23,
        -45, -25, -16, -17,   3,   0,  -5, -33,
        -44, -16, -20,  -9,  -1,  11,  -6, -71,
        -19, -13,   1,  17,  16,   7, -37, -26,
    ],
    chess.QUEEN: [
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
         -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
         -1, -18,  -9,  10, -15, -25, -31, -50,
    ],
    chess.KING: [
        -65,  23,  16, -15, -56, -34,   2,  13,
         29,  -1, -20,  -7,  -8,  -4, -38, -29,
         -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
          1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,
    ],
}

PST_EG = {
    chess.PAWN: [
          0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
         94, 100,  85,  67,  56,  53,  82,  84,
         32,  24,  13,   5,  -2,   4,  17,  17,
         13,   9,  -3,  -7,  -7,  -8,   3,  -1,
          4,   7,  -6,   1,   0,  -5,  -1,  -8,
         13,   8,   8,  10,  13,   0,   2,  -7,
          0,   0,   0,   0,   0,   0,   0,   0,
    ],
    chess.KNIGHT: [
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    ],
    chess.BISHOP: [
        -14, -21, -11,  -8,  -7,  -9, -17, -24,
         -8,  -4,   7, -12,  -3, -13,  -4, -14,
          2,  -8,   0,  -1,  -2,   6,   0,   4,
         -3,   9,  12,   9,  14,  10,   3,   2,
         -6,   3,  13,  19,   7,  10,  -3,  -9,
        -12,  -3,   8,  10,  13,   3,  -7, -15,
        -14, -18,  -7,  -1,   4,  -9, -15, -27,
        -23,  -9, -23,  -5,  -9, -16,  -5, -17,
    ],
    chess.ROOK: [
         13,  10,  18,  15,  12,  12,   8,   5,
         11,  13,  13,  11,  -3,   3,   8,   3,
          7,   7,   7,   5,   4,  -3,  -5,  -3,
          4,   3,  13,   1,   2,   1,  -1,   2,
          3,   5,   8,   4,  -5,  -6,  -8, -11,
         -4,   0,  -5,  -1,  -7, -12,  -8, -16,
         -6,  -6,   0,   2,  -9,  -9, -11,  -3,
         -9,   2,   3,  -1,  -5, -13,   4, -20,
    ],
    chess.QUEEN: [
         -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
          3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  12,  11,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    ],
    chess.KING: [
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
         10,  17,  23,  15,  20,  45,  44,  13,
         -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43,
    ],
}

TRANSPOSITION_TABLE = {}
EVAL_CACHE = {}
HISTORY_TABLE = {}
KILLER_MOVES = {}
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
MAX_TT_SIZE = 500000
MAX_EVAL_CACHE = 200000

CENTER_SQUARES = (chess.D4, chess.E4, chess.D5, chess.E5)
EXTENDED_CENTER = (
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4, chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
)
CORE_CENTER = set(CENTER_SQUARES)
RIM_FILES = {0, 7}
RIM_RANKS = {0, 7}
MINOR_PIECES = {chess.KNIGHT, chess.BISHOP}

OPENING_PHASE_CUTOFF = 0.68
EARLY_PHASE_CUTOFF = 0.82

ATTACK_WEIGHTS = {
    chess.PAWN: 0.20,
    chess.KNIGHT: 0.55,
    chess.BISHOP: 0.65,
    chess.ROOK: 0.70,
    chess.QUEEN: 0.45,
    chess.KING: 0.10,
}

PIN_PENALTY = {
    chess.PAWN: 0.25,
    chess.KNIGHT: 0.30,
    chess.BISHOP: 0.25,
    chess.ROOK: 0.18,
    chess.QUEEN: 0.12,
    chess.KING: 0.00,
}

HANGING_PENALTY = {
    chess.PAWN: 0.10,
    chess.KNIGHT: 0.18,
    chess.BISHOP: 0.18,
    chess.ROOK: 0.14,
    chess.QUEEN: 0.10,
    chess.KING: 0.00,
}

PROMOTION_BONUS = {
    chess.QUEEN: 900,
    chess.ROOK: 500,
    chess.BISHOP: 330,
    chess.KNIGHT: 320,
}

STARTING_SQUARES = {
    chess.WHITE: {
        chess.KNIGHT: {chess.B1, chess.G1},
        chess.BISHOP: {chess.C1, chess.F1},
        chess.ROOK: {chess.A1, chess.H1},
        chess.QUEEN: {chess.D1},
        chess.KING: {chess.E1},
    },
    chess.BLACK: {
        chess.KNIGHT: {chess.B8, chess.G8},
        chess.BISHOP: {chess.C8, chess.F8},
        chess.ROOK: {chess.A8, chess.H8},
        chess.QUEEN: {chess.D8},
        chess.KING: {chess.E8},
    },
}

IDEAL_MINOR_SQUARES = {
    chess.WHITE: {
        chess.KNIGHT: {chess.C3, chess.D4, chess.E4, chess.F3, chess.C2, chess.F2},
        chess.BISHOP: {chess.B2, chess.G2, chess.C4, chess.F4, chess.B3, chess.G3},
    },
    chess.BLACK: {
        chess.KNIGHT: {chess.C6, chess.D5, chess.E5, chess.F6, chess.C7, chess.F7},
        chess.BISHOP: {chess.B7, chess.G7, chess.C5, chess.F5, chess.B6, chess.G6},
    },
}

def _phase_weight(board):
    phase = 24
    for piece in board.piece_map().values():
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            phase -= 1
        elif piece.piece_type == chess.ROOK:
            phase -= 2
        elif piece.piece_type == chess.QUEEN:
            phase -= 4
    phase = max(0, min(24, phase))
    return phase / 24.0  # 1.0 = middlegame/opening, 0.0 = endgame

def _square_color(color, square):
    return square if color == chess.BLACK else chess.square_mirror(square)

def _center_distance(square):
    f = chess.square_file(square)
    r = chess.square_rank(square)
    return abs(f - 3.5) + abs(r - 3.5)

def _king_ring(square):
    f = chess.square_file(square)
    r = chess.square_rank(square)
    squares = []
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            nf = f + df
            nr = r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                squares.append(chess.square(nf, nr))
    return tuple(squares)

def _count_attacking_pawns_after_one_push(board, square, color):
    enemy = not color
    target_file = chess.square_file(square)
    target_rank = chess.square_rank(square)
    direction = 1 if enemy == chess.WHITE else -1

    count = 0
    for pawn_sq in board.pieces(chess.PAWN, enemy):
        pf = chess.square_file(pawn_sq)
        pr = chess.square_rank(pawn_sq)
        if abs(pf - target_file) != 1:
            continue
        if target_rank != pr + 2 * direction:
            continue
        step_rank = pr + direction
        if 0 <= step_rank < 8:
            step_sq = chess.square(pf, step_rank)
            if board.piece_at(step_sq) is None:
                count += 1
    return count

def _pawn_file_counts(board, color):
    counts = [0] * 8
    for sq in board.pieces(chess.PAWN, color):
        counts[chess.square_file(sq)] += 1
    return counts

def _passed_pawn_bonus(board, square, color):
    file_ = chess.square_file(square)
    rank = chess.square_rank(square)
    enemy = not color
    for ep in board.pieces(chess.PAWN, enemy):
        ef = chess.square_file(ep)
        er = chess.square_rank(ep)
        if abs(ef - file_) > 1:
            continue
        if color == chess.WHITE:
            if er > rank:
                return 0.0
        else:
            if er < rank:
                return 0.0
    advance = rank if color == chess.WHITE else (7 - rank)
    return 18.0 + advance * 6.0

def _rook_file_bonus(board, square, color):
    file_ = chess.square_file(square)
    own_pawns = 0
    enemy_pawns = 0
    for sq in board.pieces(chess.PAWN, color):
        if chess.square_file(sq) == file_:
            own_pawns += 1
    for sq in board.pieces(chess.PAWN, not color):
        if chess.square_file(sq) == file_:
            enemy_pawns += 1
    if own_pawns == 0 and enemy_pawns == 0:
        return 14.0
    if own_pawns == 0 and enemy_pawns > 0:
        return 8.0
    return 0.0

def _development_score(board, color, phase):
    """Opening-specific incentives to keep pieces active and avoid wasteful knight trips."""
    if phase < OPENING_PHASE_CUTOFF:
        return 0.0

    score = 0.0
    home_rank = 0 if color == chess.WHITE else 7
    back_rank_bonus = 0.0 if phase < EARLY_PHASE_CUTOFF else 1.0

    for sq in board.pieces(chess.KNIGHT, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        if r == home_rank:
            score -= 10.0 + 5.0 * back_rank_bonus
        if f in RIM_FILES or r in RIM_RANKS:
            score -= 26.0 + 14.0 * phase
        if sq in IDEAL_MINOR_SQUARES[color][chess.KNIGHT]:
            score += 14.0 + 10.0 * (1.0 - phase)
        # Extra penalty for central knights that are unsupported or chased.
        enemy = not color
        attackers = len(board.attackers(enemy, sq))
        defenders = len(board.attackers(color, sq))
        if attackers and defenders == 0:
            score -= 14.0 + 8.0 * phase
        elif attackers > defenders:
            score -= 8.0 * (attackers - defenders)

    for sq in board.pieces(chess.BISHOP, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        if r == home_rank:
            score -= 8.0 + 4.0 * phase
        if sq in IDEAL_MINOR_SQUARES[color][chess.BISHOP]:
            score += 10.0 + 8.0 * (1.0 - phase)
        # If bishop is blocked by its own pawns, encourage freeing it.
        blocked = 0
        for pawn_sq in board.pieces(chess.PAWN, color):
            if abs(chess.square_file(pawn_sq) - f) <= 1:
                if color == chess.WHITE and chess.square_rank(pawn_sq) <= r:
                    blocked += 1
                elif color == chess.BLACK and chess.square_rank(pawn_sq) >= r:
                    blocked += 1
        if blocked >= 2:
            score -= 5.0

    return score

def _center_control(board):
    score = 0.0
    for sq in CENTER_SQUARES:
        score += 7.5 * (len(board.attackers(chess.WHITE, sq)) - len(board.attackers(chess.BLACK, sq)))
    for sq in EXTENDED_CENTER:
        score += 2.5 * (len(board.attackers(chess.WHITE, sq)) - len(board.attackers(chess.BLACK, sq)))
    return score

def _king_safety(board, color, phase):
    king_sq = board.king(color)
    if king_sq is None:
        return 0.0

    enemy = not color
    bonus = 0.0
    home_castled = (color == chess.WHITE and king_sq in (chess.G1, chess.C1)) or (color == chess.BLACK and king_sq in (chess.G8, chess.C8))
    if home_castled:
        bonus += 34.0
    elif board.has_castling_rights(color):
        bonus += 10.0 * phase
    else:
        bonus -= 8.0 * phase

    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
    shield_dir = 1 if color == chess.WHITE else -1
    shield_rank = king_rank + shield_dir
    shield = 0

    if 0 <= shield_rank < 8:
        for df in (-1, 0, 1):
            nf = king_file + df
            if not (0 <= nf < 8):
                continue
            sq = chess.square(nf, shield_rank)
            piece = board.piece_at(sq)
            if piece is None:
                bonus -= 2.5 * phase
            elif piece.color == color and piece.piece_type == chess.PAWN:
                shield += 1

    ring_attacks = 0
    for sq in _king_ring(king_sq):
        if board.is_attacked_by(enemy, sq):
            ring_attacks += 1

    direct_check = len(board.attackers(enemy, king_sq))
    bonus -= (10.0 + 16.0 * phase) * ring_attacks
    bonus -= (14.0 + 20.0 * phase) * direct_check
    bonus += shield * (6.0 + 6.0 * phase)

    for file_ in (king_file - 1, king_file + 1):
        if 0 <= file_ < 8:
            own_pawn_here = any(chess.square_file(sq) == file_ for sq in board.pieces(chess.PAWN, color))
            enemy_pawn_here = any(chess.square_file(sq) == file_ for sq in board.pieces(chess.PAWN, enemy))
            if not own_pawn_here and enemy_pawn_here:
                bonus -= 7.0 * phase
            elif not own_pawn_here and not enemy_pawn_here:
                bonus -= 4.0 * phase

    if phase < 0.35:
        dist = _center_distance(king_sq)
        bonus += (7.0 - dist) * (1.0 - phase) * 2.5

    return bonus

def _material_and_positional(board, phase):
    mg_score = 0.0
    eg_score = 0.0
    extra = 0.0

    white_pawn_files = _pawn_file_counts(board, chess.WHITE)
    black_pawn_files = _pawn_file_counts(board, chess.BLACK)

    for sq, piece in board.piece_map().items():
        color = piece.color
        sign = 1.0 if color == chess.WHITE else -1.0
        idx = _square_color(color, sq)
        ptype = piece.piece_type
        value = PIECE_VALUES[ptype]

        mg_score += sign * (value + PST_MG[ptype][idx])
        eg_score += sign * (value + PST_EG[ptype][idx])

        activity = len(board.attacks(sq))
        extra += sign * ATTACK_WEIGHTS[ptype] * activity

        enemy = not color
        attackers = len(board.attackers(enemy, sq))
        defenders = len(board.attackers(color, sq))

        if ptype != chess.KING and attackers:
            if defenders == 0:
                extra -= sign * value * HANGING_PENALTY[ptype] * (1.0 + 0.5 * phase)
            elif attackers > defenders:
                extra -= sign * value * 0.04 * (attackers - defenders)

        if ptype != chess.KING and board.is_pinned(color, sq):
            extra -= sign * value * PIN_PENALTY[ptype] * (1.0 + 0.4 * phase)

        if ptype == chess.PAWN:
            file_ = chess.square_file(sq)
            rank = chess.square_rank(sq)
            own_files = white_pawn_files if color == chess.WHITE else black_pawn_files

            if own_files[file_] > 1:
                extra -= sign * (12.0 + 6.0 * (own_files[file_] - 1))

            left = own_files[file_ - 1] if file_ > 0 else 0
            right = own_files[file_ + 1] if file_ < 7 else 0
            if left == 0 and right == 0:
                extra -= sign * (10.0 + 2.5 * phase)

            extra += sign * _passed_pawn_bonus(board, sq, color) * (0.45 + 0.75 * (1.0 - phase))

            if board.is_attacked_by(color, sq):
                extra += sign * 4.0

            push_threats = _count_attacking_pawns_after_one_push(board, sq, color)
            if push_threats:
                extra -= sign * value * (0.08 * push_threats)

            # Encourage healthy pawn chains toward the center.
            if (color == chess.WHITE and rank >= 3) or (color == chess.BLACK and rank <= 4):
                if chess.square_file(sq) in (3, 4):
                    extra += sign * 6.0

        elif ptype == chess.ROOK:
            extra += sign * _rook_file_bonus(board, sq, color) * (0.7 + 0.3 * phase)

        elif ptype == chess.BISHOP:
            bishop_activity = len(board.attacks(sq))
            extra += sign * max(0.0, bishop_activity - 6) * (2.0 + 3.0 * phase)

        elif ptype == chess.KNIGHT:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            if f in RIM_FILES or r in RIM_RANKS:
                extra -= sign * (12.0 + 8.0 * phase)
            dist = _center_distance(sq)
            extra += sign * max(0.0, 6.0 - dist) * (6.0 + 4.0 * phase)

        elif ptype == chess.QUEEN:
            # Early queen adventure penalty; later activity bonus.
            if phase > 0.5:
                if sq in CORE_CENTER:
                    extra -= sign * 8.0
                extra -= sign * 3.5 * max(0.0, 4.5 - _center_distance(sq))

    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        extra += 24.0 * phase + 10.0 * (1.0 - phase)
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        extra -= 24.0 * phase + 10.0 * (1.0 - phase)

    if len(board.pieces(chess.ROOK, chess.WHITE)) >= 2:
        extra += 4.0
    if len(board.pieces(chess.ROOK, chess.BLACK)) >= 2:
        extra -= 4.0

    extra += _development_score(board, chess.WHITE, phase)
    extra -= _development_score(board, chess.BLACK, phase)

    return mg_score, eg_score, extra

def _evaluate_uncached(board):
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner is None:
            return 0.0
        ply = board.ply()
        mate = MATE_SCORE - ply
        return mate if outcome.winner == chess.WHITE else -mate

    phase = _phase_weight(board)
    mg_score, eg_score, extra = _material_and_positional(board, phase)
    score = mg_score * phase + eg_score * (1.0 - phase) + extra

    score += _center_control(board)
    score += _king_safety(board, chess.WHITE, phase)
    score -= _king_safety(board, chess.BLACK, phase)

    if board.is_check():
        score += -22.0 if board.turn == chess.WHITE else 22.0

    legal_moves = sum(1 for _ in board.legal_moves)
    score += (5.0 + 0.25 * legal_moves) if board.turn == chess.WHITE else -(5.0 + 0.25 * legal_moves)

    # Mild penalty for being very low on mobility in the opening.
    if phase > OPENING_PHASE_CUTOFF and legal_moves <= 10:
        score += -8.0 if board.turn == chess.WHITE else 8.0

    return score

def evaluate(board):
    key = chess.polyglot.zobrist_hash(board)
    cached = EVAL_CACHE.get(key)
    if cached is not None:
        return cached
    score = _evaluate_uncached(board)
    EVAL_CACHE[key] = score
    if len(EVAL_CACHE) > MAX_EVAL_CACHE:
        EVAL_CACHE.clear()
    return score

def _piece_key(move):
    return (move.from_square, move.to_square, move.promotion)

def _move_uci(move):
    return move.uci()

def _killer_bonus(depth, move):
    killers = KILLER_MOVES.get(depth)
    if not killers:
        return 0
    u = move.uci()
    if u == killers[0]:
        return 25000
    if len(killers) > 1 and u == killers[1]:
        return 20000
    return 0

def score_move(board, move, depth=0, tt_move=None):
    score = 0
    uci = move.uci()

    if tt_move is not None and uci == tt_move:
        score += 100000

    if board.gives_check(move):
        score += 70000

    if move.promotion:
        score += 60000 + PROMOTION_BONUS.get(move.promotion, 0)

    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        victim_val = PIECE_VALUES[chess.PAWN] if victim is None else PIECE_VALUES[victim.piece_type]
        attacker_val = PIECE_VALUES[chess.PAWN] if attacker is None else PIECE_VALUES[attacker.piece_type]
        score += 50000 + victim_val * 12 - attacker_val

    if board.is_castling(move):
        score += 2500

    score += _killer_bonus(depth, move)
    score += HISTORY_TABLE.get(_piece_key(move), 0)

    if move.to_square in CORE_CENTER:
        score += 150
    elif move.to_square in EXTENDED_CENTER:
        score += 40

    return score

def order_moves(board, moves, depth=0, tt_move=None):
    return sorted(moves, key=lambda m: score_move(board, m, depth=depth, tt_move=tt_move), reverse=True)

def _register_killer(depth, move):
    u = move.uci()
    killers = KILLER_MOVES.setdefault(depth, [])
    if u in killers:
        return
    killers.insert(0, u)
    del killers[2:]

def _register_history(move, depth):
    key = _piece_key(move)
    HISTORY_TABLE[key] = HISTORY_TABLE.get(key, 0) + depth * depth

def qsearch(board, alpha, beta, maximizing, qdepth=6):
    stand_pat = evaluate(board)

    if maximizing:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)

    if qdepth <= 0:
        return stand_pat

    if board.is_check():
        tactical_moves = list(board.legal_moves)
    else:
        tactical_moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion or board.gives_check(m)]

    tactical_moves = order_moves(board, tactical_moves, depth=0)

    for move in tactical_moves:
        board.push(move)
        score = qsearch(board, alpha, beta, not maximizing, qdepth - 1)
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

def minimax(board, depth, alpha, beta, maximizing):
    hash_key = chess.polyglot.zobrist_hash(board)
    tt_entry = TRANSPOSITION_TABLE.get(hash_key)
    tt_move = None

    if tt_entry is not None:
        tt_depth, tt_flag, tt_score, tt_move = tt_entry
        if tt_depth >= depth:
            if tt_flag == TT_EXACT:
                return tt_score
            elif tt_flag == TT_LOWER:
                alpha = max(alpha, tt_score)
            elif tt_flag == TT_UPPER:
                beta = min(beta, tt_score)
            if alpha >= beta:
                return tt_score

    if depth <= 0:
        return qsearch(board, alpha, beta, maximizing)

    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        return evaluate(board)

    moves = list(board.legal_moves)
    if not moves:
        return evaluate(board)

    moves = order_moves(board, moves, depth=depth, tt_move=tt_move)
    alpha_orig = alpha
    beta_orig = beta
    best_score = -INF if maximizing else INF
    best_move = None

    if maximizing:
        for move in moves:
            board.push(move)
            score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, score)
            if alpha >= beta:
                _register_killer(depth, move)
                _register_history(move, depth)
                break
    else:
        for move in moves:
            board.push(move)
            score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, score)
            if alpha >= beta:
                _register_killer(depth, move)
                _register_history(move, depth)
                break

    if best_score <= alpha_orig:
        tt_flag = TT_UPPER
    elif best_score >= beta_orig:
        tt_flag = TT_LOWER
    else:
        tt_flag = TT_EXACT

    TRANSPOSITION_TABLE[hash_key] = (
        depth,
        tt_flag,
        best_score,
        best_move.uci() if best_move else None,
    )

    return best_score

def get_next_move(board, color, depth=3):
    global TRANSPOSITION_TABLE

    if len(TRANSPOSITION_TABLE) > MAX_TT_SIZE:
        TRANSPOSITION_TABLE.clear()

    moves = list(board.legal_moves)
    if not moves:
        return chess.Move.null()

    maximizing = (color == chess.WHITE)
    best_move = moves[0]
    best_score = -INF if maximizing else INF
    alpha = -INF
    beta = INF

    tt_entry = TRANSPOSITION_TABLE.get(chess.polyglot.zobrist_hash(board))
    tt_move = tt_entry[3] if tt_entry is not None else None
    moves = order_moves(board, moves, depth=depth, tt_move=tt_move)

    for move in moves:
        board.push(move)
        score = minimax(board, depth - 1, alpha, beta, not maximizing)
        board.pop()

        if maximizing:
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
        else:
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)

    return best_move

if __name__ == "__main__":
    b = chess.Board()
    mv = get_next_move(b, chess.WHITE, depth=3)
    print("Opening move:", b.san(mv))
