
import chess
import chess.polyglot

# ── Core constants ─────────────────────────────────────────────────────────────
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

# ── Search state ──────────────────────────────────────────────────────────────
TRANSPOSITION_TABLE = {}
EVAL_CACHE = {}
HISTORY_TABLE = {}
KILLER_MOVES = {}  # depth -> [uci, uci]
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2
MAX_TT_SIZE = 500000
MAX_EVAL_CACHE = 200000

# ── Helpers ───────────────────────────────────────────────────────────────────
CENTER_SQUARES = (chess.D4, chess.E4, chess.D5, chess.E5)
EXTENDED_CENTER = (
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.F4, chess.C5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
)
CORE_CENTER = set(CENTER_SQUARES)

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

def _piece_key(move: chess.Move) -> tuple:
    return (move.from_square, move.to_square, move.promotion)

def _move_uci(move: chess.Move) -> str:
    return move.uci()

def _king_ring(square: chess.Square) -> tuple[int, ...]:
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

def _forward_pawn_attack_threats(board: chess.Board, square: chess.Square, color: chess.Color) -> int:
    """How many enemy pawns can attack this square after a single pawn push."""
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
        # Pawn pushes one square, then attacks one diagonally forward.
        if target_rank != pr + 2 * direction:
            continue

        step_rank = pr + direction
        if 0 <= step_rank < 8:
            step_sq = chess.square(pf, step_rank)
            if board.piece_at(step_sq) is None:
                count += 1
    return count

def _phase_weight(board: chess.Board) -> float:
    phase = 24
    for piece in board.piece_map().values():
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            phase -= 1
        elif piece.piece_type == chess.ROOK:
            phase -= 2
        elif piece.piece_type == chess.QUEEN:
            phase -= 4
    phase = max(0, min(24, phase))
    return phase / 24.0  # 1.0 = middlegame, 0.0 = endgame

def _pawn_file_counts(board: chess.Board, color: chess.Color) -> list[int]:
    counts = [0] * 8
    for sq in board.pieces(chess.PAWN, color):
        counts[chess.square_file(sq)] += 1
    return counts

def _passed_pawn_bonus(board: chess.Board, square: chess.Square, color: chess.Color) -> float:
    file_ = chess.square_file(square)
    rank = chess.square_rank(square)
    enemy = not color
    enemy_pawns = board.pieces(chess.PAWN, enemy)

    for ep in enemy_pawns:
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

def _rook_file_bonus(board: chess.Board, square: chess.Square, color: chess.Color) -> float:
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

def _center_control(board: chess.Board) -> float:
    score = 0.0
    for sq in CENTER_SQUARES:
        score += 7.5 * (len(board.attackers(chess.WHITE, sq)) - len(board.attackers(chess.BLACK, sq)))
    for sq in EXTENDED_CENTER:
        score += 2.5 * (len(board.attackers(chess.WHITE, sq)) - len(board.attackers(chess.BLACK, sq)))
    return score

def _king_safety(board: chess.Board, color: chess.Color, phase: float) -> float:
    king_sq = board.king(color)
    if king_sq is None:
        return 0.0

    enemy = not color
    bonus = 0.0

    # Reward castled king placement, and prefer keeping castling rights early.
    if (color == chess.WHITE and king_sq in (chess.G1, chess.C1)) or (
        color == chess.BLACK and king_sq in (chess.G8, chess.C8)
    ):
        bonus += 32.0
    elif board.has_castling_rights(color):
        bonus += 10.0 * phase

    # Pawn shield in front of the king.
    shield_dir = 1 if color == chess.WHITE else -1
    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)
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

    # Attacks on the king ring.
    ring_attacks = 0
    for sq in _king_ring(king_sq):
        if board.is_attacked_by(enemy, sq):
            ring_attacks += 1

    checkers = len(board.attackers(enemy, king_sq))
    bonus -= (10.0 + 16.0 * phase) * ring_attacks
    bonus -= (14.0 + 20.0 * phase) * checkers
    bonus += shield * (6.0 + 6.0 * phase)

    # Open files next to the king are dangerous.
    left_file = king_file - 1
    right_file = king_file + 1
    for file_ in (left_file, right_file):
        if 0 <= file_ < 8:
            own_pawn_here = any(chess.square_file(sq) == file_ for sq in board.pieces(chess.PAWN, color))
            enemy_pawn_here = any(chess.square_file(sq) == file_ for sq in board.pieces(chess.PAWN, enemy))
            if not own_pawn_here and enemy_pawn_here:
                bonus -= 7.0 * phase
            elif not own_pawn_here and not enemy_pawn_here:
                bonus -= 4.0 * phase

    # Endgame king activity.
    if phase < 0.35:
        center = (3.5, 3.5)
        dist = abs(chess.square_file(king_sq) - center[0]) + abs(chess.square_rank(king_sq) - center[1])
        bonus += (7.0 - dist) * (1.0 - phase) * 2.5

    return bonus

def _material_and_positional(board: chess.Board, phase: float) -> float:
    mg_score = 0.0
    eg_score = 0.0
    extra = 0.0

    white_pawn_files = _pawn_file_counts(board, chess.WHITE)
    black_pawn_files = _pawn_file_counts(board, chess.BLACK)

    for sq, piece in board.piece_map().items():
        color = piece.color
        sign = 1.0 if color == chess.WHITE else -1.0
        idx = chess.square_mirror(sq) if color == chess.WHITE else sq
        ptype = piece.piece_type
        value = PIECE_VALUES[ptype]

        mg_score += sign * (value + PST_MG[ptype][idx])
        eg_score += sign * (value + PST_EG[ptype][idx])

        # Piece activity / mobility via attacked squares.
        activity = len(board.attacks(sq))
        extra += sign * ATTACK_WEIGHTS[ptype] * activity

        # Pins and loose pieces / en prise pieces.
        enemy = not color
        attackers = len(board.attackers(enemy, sq))
        defenders = len(board.attackers(color, sq))
        if ptype != chess.KING and attackers:
            if defenders == 0:
                extra -= sign * value * HANGING_PENALTY[ptype]
            elif attackers > defenders:
                extra -= sign * value * 0.04 * (attackers - defenders)

        if ptype != chess.KING and board.is_pinned(color, sq):
            extra -= sign * value * PIN_PENALTY[ptype]

        # Pawn structures and immediate pawn-advance threats.
        if ptype == chess.PAWN:
            file_ = chess.square_file(sq)
            rank = chess.square_rank(sq)
            own_files = white_pawn_files if color == chess.WHITE else black_pawn_files
            enemy_files = black_pawn_files if color == chess.WHITE else white_pawn_files

            # Doubled pawns
            if own_files[file_] > 1:
                extra -= sign * (12.0 + 6.0 * (own_files[file_] - 1))

            # Isolated pawns
            left = own_files[file_ - 1] if file_ > 0 else 0
            right = own_files[file_ + 1] if file_ < 7 else 0
            if left == 0 and right == 0:
                extra -= sign * (10.0 + 2.5 * phase)

            # Passed pawns
            extra += sign * _passed_pawn_bonus(board, sq, color) * (0.45 + 0.75 * (1.0 - phase))

            # Protected / connected pawns
            if board.is_attacked_by(color, sq):
                extra += sign * 4.0

            # Threatened by an enemy pawn push next move.
            push_threats = _forward_pawn_attack_threats(board, sq, color)
            if push_threats:
                extra -= sign * value * (0.06 * push_threats)

        # Rooks like open and semi-open files.
        if ptype == chess.ROOK:
            extra += sign * _rook_file_bonus(board, sq, color) * (0.7 + 0.3 * phase)

        # Bishops like long diagonals.
        if ptype == chess.BISHOP:
            bishop_activity = len(board.attacks(sq))
            extra += sign * max(0.0, bishop_activity - 6) * (2.0 + 3.0 * phase)

    # Bishop pair.
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        extra += 24.0 * phase + 10.0 * (1.0 - phase)
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        extra -= 24.0 * phase + 10.0 * (1.0 - phase)

    # Rooks on open files / semi-open files were handled per rook; this also adds a mild pair bonus.
    if len(board.pieces(chess.ROOK, chess.WHITE)) >= 2:
        extra += 4.0
    if len(board.pieces(chess.ROOK, chess.BLACK)) >= 2:
        extra -= 4.0

    return mg_score, eg_score, extra

def _evaluate_uncached(board: chess.Board) -> float:
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        if outcome.winner is None:
            return 0.0
        ply = board.ply()
        mate = MATE_SCORE - ply
        return mate if outcome.winner == chess.WHITE else -mate

    phase = _phase_weight(board)  # 1 = middlegame, 0 = endgame
    mg_score, eg_score, extra = _material_and_positional(board, phase)
    score = mg_score * phase + eg_score * (1.0 - phase) + extra

    # Center control, king safety, and tempo.
    score += _center_control(board)
    score += _king_safety(board, chess.WHITE, phase)
    score -= _king_safety(board, chess.BLACK, phase)

    if board.is_check():
        score += -22.0 if board.turn == chess.WHITE else 22.0

    # Small tempo / mobility term for the side to move.
    legal_moves = sum(1 for _ in board.legal_moves)
    score += (5.0 + 0.25 * legal_moves) if board.turn == chess.WHITE else -(5.0 + 0.25 * legal_moves)

    return score

def evaluate(board: chess.Board) -> float:
    key = chess.polyglot.zobrist_hash(board)
    cached = EVAL_CACHE.get(key)
    if cached is not None:
        return cached

    score = _evaluate_uncached(board)
    EVAL_CACHE[key] = score
    if len(EVAL_CACHE) > MAX_EVAL_CACHE:
        EVAL_CACHE.clear()
    return score

# ── Move ordering ─────────────────────────────────────────────────────────────
def _killer_bonus(depth: int, move: chess.Move) -> int:
    killers = KILLER_MOVES.get(depth)
    if not killers:
        return 0
    u = move.uci()
    if u == killers[0]:
        return 25000
    if len(killers) > 1 and u == killers[1]:
        return 20000
    return 0

def score_move(board: chess.Board, move: chess.Move, depth: int = 0, tt_move: str | None = None) -> int:
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

    # Mild encouragement for central moves.
    if move.to_square in CORE_CENTER:
        score += 150
    elif move.to_square in EXTENDED_CENTER:
        score += 40

    return score

def order_moves(board: chess.Board, moves: list[chess.Move], depth: int = 0, tt_move: str | None = None) -> list[chess.Move]:
    return sorted(moves, key=lambda m: score_move(board, m, depth=depth, tt_move=tt_move), reverse=True)

def _register_killer(depth: int, move: chess.Move) -> None:
    u = move.uci()
    killers = KILLER_MOVES.setdefault(depth, [])
    if u in killers:
        return
    killers.insert(0, u)
    del killers[2:]

def _register_history(move: chess.Move, depth: int) -> None:
    key = _piece_key(move)
    HISTORY_TABLE[key] = HISTORY_TABLE.get(key, 0) + depth * depth

# ── Quiescence search ─────────────────────────────────────────────────────────
def qsearch(board: chess.Board, alpha: float, beta: float, maximizing: bool, qdepth: int = 6) -> float:
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
        tactical_moves = list(board.legal_moves)  # All evasions when in check.
    else:
        tactical_moves = [
            m for m in board.legal_moves
            if board.is_capture(m) or m.promotion or board.gives_check(m)
        ]

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

# ── Minimax / alpha-beta ───────────────────────────────────────────────────────
def minimax(board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
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

# ── Core API ──────────────────────────────────────────────────────────────────
def get_next_move(board: chess.Board, color: chess.Color, depth: int = 3) -> chess.Move:
    """
    Main entrypoint called by the tournament harness.
    """
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

# ── Self-test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    b = chess.Board()
    mv = get_next_move(b, chess.WHITE, depth=3)
    print("Opening move:", b.san(mv))
