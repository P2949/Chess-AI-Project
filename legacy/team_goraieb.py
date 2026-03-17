import chess
import chess.polyglot
from math import log as _log

PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}

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

PST_PAWN_MG = [
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  62,  22,  -8, -17,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
]
PST_PAWN_EG = [
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
]

PST_KNIGHT_MG = [
    -167, -89, -34, -49,  61, -97, -15, -107,
     -73, -41,  72,  36,  23,  62,   7,  -17,
     -47,  60,  37,  65,  84, 129,  73,   44,
      -9,  17,  19,  53,  37,  69,  18,   22,
     -13,   4,  16,  13,  28,  19,  21,   -8,
     -23,  -9,  12,  10,  19,  17,  25,  -16,
     -29, -53, -12,  -3,  -1,  18, -14,  -19,
    -105, -21, -58, -33, -17, -28, -19,  -23,
]
PST_KNIGHT_EG = [
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
]

PST_BISHOP_MG = [
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
]
PST_BISHOP_EG = [
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
]

PST_ROOK_MG = [
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
]
PST_ROOK_EG = [
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
]

PST_QUEEN_MG = [
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
]
PST_QUEEN_EG = [
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  12,  11,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
]

PST_KING_MG = [
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49, -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
]
PST_KING_EG = [
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
]

_PST_MAP = {
    chess.PAWN:   PST_PAWN,
    chess.KNIGHT: PST_KNIGHT,
    chess.BISHOP: PST_BISHOP,
    chess.ROOK:   PST_ROOK,
    chess.QUEEN:  PST_QUEEN,
}

_ROOT_COLOR = chess.WHITE
CONTEMPT    = 60


def _contempt_score() -> float:
    return -CONTEMPT if _ROOT_COLOR == chess.WHITE else CONTEMPT

def _chebyshev(sq1: int, sq2: int) -> int:
    return max(abs(chess.square_file(sq1) - chess.square_file(sq2)),abs(chess.square_rank(sq1) - chess.square_rank(sq2)))


def _manhattan(sq1: int, sq2: int) -> int:
    return (abs(chess.square_file(sq1) - chess.square_file(sq2)) + abs(chess.square_rank(sq1) - chess.square_rank(sq2)))

def _endgame_factor(board: chess.Board) -> float:
    phase = 0
    phase += len(board.pieces(chess.KNIGHT, chess.WHITE)) * 1
    phase += len(board.pieces(chess.KNIGHT, chess.BLACK)) * 1
    phase += len(board.pieces(chess.BISHOP, chess.WHITE)) * 1
    phase += len(board.pieces(chess.BISHOP, chess.BLACK)) * 1
    phase += len(board.pieces(chess.ROOK, chess.WHITE)) * 2
    phase += len(board.pieces(chess.ROOK, chess.BLACK)) * 2
    phase += len(board.pieces(chess.QUEEN, chess.WHITE)) * 4
    phase += len(board.pieces(chess.QUEEN, chess.BLACK)) * 4
    for c in (chess.WHITE, chess.BLACK):
        phase += len(board.pieces(chess.QUEEN,  c)) * 4
        phase += len(board.pieces(chess.ROOK,   c)) * 2
        phase += len(board.pieces(chess.BISHOP, c))
        phase += len(board.pieces(chess.KNIGHT, c))

    return 1.0 - min(phase, 24) / 24.0

def _get_pst(piece_type, eg):
    mg_weight = 1.0 - eg
    if piece_type == chess.PAWN: return [(PST_PAWN_MG[i]*mg_weight + PST_PAWN_EG[i]*eg) for i in range(64)]
    elif piece_type == chess.KNIGHT: return [(PST_KNIGHT_MG[i]*mg_weight + PST_KNIGHT_EG[i]*eg) for i in range(64)]
    elif piece_type == chess.BISHOP: return [(PST_BISHOP_MG[i]*mg_weight + PST_BISHOP_EG[i]*eg) for i in range(64)]
    elif piece_type == chess.ROOK: return [(PST_ROOK_MG[i]*mg_weight + PST_ROOK_EG[i]*eg) for i in range(64)]
    elif piece_type == chess.QUEEN: return [(PST_QUEEN_MG[i]*mg_weight + PST_QUEEN_EG[i]*eg) for i in range(64)]
    else: return [(PST_KING_MG[i]*mg_weight + PST_KING_EG[i]*eg) for i in range(64)]

def _evaluate_draw(board: chess.Board, current_eval: float) -> float:
    if not (board.is_stalemate() or board.is_repetition(2) or board.is_fifty_moves() or board.is_insufficient_material()):
        return 0.0
    
    perspective_eval = current_eval if _ROOT_COLOR == chess.WHITE else -current_eval
    if perspective_eval > 50:
        return -99000 if _ROOT_COLOR == chess.WHITE else 99000
    elif perspective_eval < -50:
        return 99000 if _ROOT_COLOR == chess.WHITE else -99000
    return 0.0

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


def _pawn_structure(board: chess.Board) -> float:
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

            if files.count(f) > 1:
                score += sign * (-20)

            adj = [a for a in (f - 1, f + 1) if 0 <= a <= 7]
            if not any(a in files for a in adj):
                score += sign * (-15)

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
                base_bonus  = 20 + advancement * 10
                prot_rank = rank - 1 if color == chess.WHITE else rank + 1
                if 0 <= prot_rank <= 7:
                    for df in (-1, 1):
                        pf = f + df
                        if 0 <= pf <= 7:
                            p = board.piece_at(chess.square(pf, prot_rank))
                            if p and p.piece_type == chess.PAWN and p.color == color:
                                base_bonus += 15
                                break
                score += sign * base_bonus

            support_ranks = range(0, rank) if color == chess.WHITE else range(rank + 1, 8)
            has_support = any(
                chess.square_file(s) in (f - 1, f + 1) and chess.square_rank(s) in support_ranks
                for s in pawns
            )
            if not has_support and not passed:
                score += sign * (-10)

    return score


def _pawn_chain_bonus(board: chess.Board) -> float:
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        for sq in board.pieces(chess.PAWN, color):
            f    = chess.square_file(sq)
            rank = chess.square_rank(sq)
            prot_rank = rank - 1 if color == chess.WHITE else rank + 1
            if not (0 <= prot_rank <= 7):
                continue
            for df in (-1, 1):
                pf = f + df
                if 0 <= pf <= 7:
                    p = board.piece_at(chess.square(pf, prot_rank))
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        score += sign * 7
                        break
    return score


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
            if not any(chess.square_file(s) == f for s in board.pieces(chess.PAWN, color)):
                score += sign * (-10) * mg_weight
    return score


def _pawn_storm(board: chess.Board, eg: float) -> float:
    if eg >= 0.75:
        return 0.0
    mg_weight = 1.0 - eg
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign   = 1 if color == chess.WHITE else -1
        enemy  = not color
        ek_sq  = board.king(enemy)
        if ek_sq is None:
            continue
        ekf = chess.square_file(ek_sq)
        for sq in board.pieces(chess.PAWN, color):
            pf = chess.square_file(sq)
            pr = chess.square_rank(sq)
            if abs(pf - ekf) > 2:
                continue
            advancement = pr if color == chess.WHITE else (7 - pr)
            if advancement >= 4:
                score += sign * (advancement - 3) * 10 * mg_weight
    return score


def _castling_bonus(board: chess.Board) -> float:
    score = 0.0
    if board.has_castling_rights(chess.WHITE):
        score += 30.0
    if board.has_castling_rights(chess.BLACK):
        score -= 30.0
    return score


def _bishop_pair(board: chess.Board) -> float:
    score = 0.0
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += 30.0
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= 30.0
    return score


def _rook_open_files(board: chess.Board) -> float:
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        for sq in board.pieces(chess.ROOK, color):
            f = chess.square_file(sq)
            own_f   = sum(1 for s in board.pieces(chess.PAWN, color)     if chess.square_file(s) == f)
            enemy_f = sum(1 for s in board.pieces(chess.PAWN, not color) if chess.square_file(s) == f)
            if own_f == 0 and enemy_f == 0:
                score += sign * 25
            elif own_f == 0:
                score += sign * 10
    return score


def _knight_outposts(board: chess.Board) -> float:
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign   = 1 if color == chess.WHITE else -1
        epawns = board.pieces(chess.PAWN, not color)
        for sq in board.pieces(chess.KNIGHT, color):
            rank = chess.square_rank(sq)
            f    = chess.square_file(sq)
            if color == chess.WHITE and rank not in (3, 4, 5):
                continue
            if color == chess.BLACK and rank not in (2, 3, 4):
                continue
            pawn_rank = rank - 1 if color == chess.WHITE else rank + 1
            if not (0 <= pawn_rank <= 7):
                continue
            protected = any(
                chess.square_file(s) in (f - 1, f + 1) and chess.square_rank(s) == pawn_rank
                for s in board.pieces(chess.PAWN, color)
            )
            if not protected:
                continue
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


def _rook_on_seventh(board: chess.Board) -> float:
    score = 0.0
    for sq in board.pieces(chess.ROOK, chess.WHITE):
        if chess.square_rank(sq) == 6:
            score += 25
    for sq in board.pieces(chess.ROOK, chess.BLACK):
        if chess.square_rank(sq) == 1:
            score -= 25
    return score


def _connected_rooks(board: chess.Board) -> float:
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
            lo, hi = min(rk1, rk2) + 1, max(rk1, rk2)
            if all(board.piece_at(chess.square(f1, r)) is None for r in range(lo, hi)):
                connected = True
        elif rk1 == rk2:
            lo, hi = min(f1, f2) + 1, max(f1, f2)
            if all(board.piece_at(chess.square(f, rk1)) is None for f in range(lo, hi)):
                connected = True
        if connected:
            score += sign * 15
    return score


def _hanging_pieces(board: chess.Board) -> float:
    score = 0.0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type == chess.KING:
            continue
        enemy = not piece.color
        if board.is_attacked_by(enemy, sq):
            defenders = len(board.attackers(piece.color, sq))
            attackers = len(board.attackers(enemy, sq))
            if defenders < attackers:
                val  = PIECE_VALUES[piece.piece_type] * 0.25
                sign = -1 if piece.color == chess.WHITE else 1
                score += sign * val
    return score


def _fork_evaluation(board: chess.Board) -> float:
    score = 0.0
    FORK_PIECES = {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}
    VALUABLE    = {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING}

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type not in FORK_PIECES:
            continue
        sign  = 1 if piece.color == chess.WHITE else -1
        enemy = not piece.color

        attacked_vals = []
        for tsq in board.attacks(sq):
            tp = board.piece_at(tsq)
            if tp and tp.color == enemy and tp.piece_type in VALUABLE:
                attacked_vals.append(PIECE_VALUES[tp.piece_type])

        if len(attacked_vals) >= 2:
            vals = sorted(attacked_vals, reverse=True)[:2]
            score += sign * (vals[0] + vals[1]) * 0.10

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type != chess.KNIGHT:
            continue
        sign  = 1 if piece.color == chess.WHITE else -1
        enemy = not piece.color

        for tsq in chess.SquareSet(chess.BB_KNIGHT_ATTACKS[sq]):
            occupant = board.piece_at(tsq)
            if occupant is not None and occupant.color == piece.color:
                continue

            future_vals = []
            for atsq in chess.SquareSet(chess.BB_KNIGHT_ATTACKS[tsq]):
                if atsq == sq:
                    continue
                tp = board.piece_at(atsq)
                if tp and tp.color == enemy and tp.piece_type in VALUABLE:
                    future_vals.append(PIECE_VALUES[tp.piece_type])

            if len(future_vals) >= 2:
                vals = sorted(future_vals, reverse=True)[:2]
                weight = 0.06 if occupant is None else 0.03
                score += sign * (vals[0] + vals[1]) * weight

    return score


def _king_proximity_mop_up(board: chess.Board, eg: float, material_score: float) -> float:
    if eg < 0.4:
        return 0.0
    result = 0.0
    
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return 0.0

    if material_score > 200:
        bkf, bkr = chess.square_file(bk), chess.square_rank(bk)
        edge_dist   = min(bkf, 7 - bkf, bkr, 7 - bkr)
        corner_dist = min(bkf + bkr, (7-bkf) + bkr, bkf + (7-bkr), (7-bkf) + (7-bkr))
        result += eg * edge_dist   * 20
        result += eg * corner_dist * 12
        result += eg * (14 - _chebyshev(wk, bk)) * 7

        for q_sq in board.pieces(chess.QUEEN, chess.WHITE):
            result += eg * (7 - _chebyshev(q_sq, bk)) * 6

        for r_sq in board.pieces(chess.ROOK, chess.WHITE):
            rf, rr = chess.square_file(r_sq), chess.square_rank(r_sq)
            if rf == bkf or rr == bkr:
                result += eg * 40
            result += eg * (14 - _manhattan(r_sq, bk)) * 3

        for pt in (chess.BISHOP, chess.KNIGHT):
            for sq in board.pieces(pt, chess.WHITE):
                result += eg * (7 - _chebyshev(sq, bk)) * 2

    elif material_score < -200:
        wkf, wkr = chess.square_file(wk), chess.square_rank(wk)
        edge_dist   = min(wkf, 7 - wkf, wkr, 7 - wkr)
        corner_dist = min(wkf + wkr, (7-wkf) + wkr, wkf + (7-wkr), (7-wkf) + (7-wkr))
        result -= eg * edge_dist   * 20
        result -= eg * corner_dist * 12
        result -= eg * (14 - _chebyshev(wk, bk)) * 7

        for q_sq in board.pieces(chess.QUEEN, chess.BLACK):
            result -= eg * (7 - _chebyshev(q_sq, wk)) * 6

        for r_sq in board.pieces(chess.ROOK, chess.BLACK):
            rf, rr = chess.square_file(r_sq), chess.square_rank(r_sq)
            if rf == wkf or rr == wkr:
                result -= eg * 40
            result -= eg * (14 - _manhattan(r_sq, wk)) * 3

        for pt in (chess.BISHOP, chess.KNIGHT):
            for sq in board.pieces(pt, chess.BLACK):
                result -= eg * (7 - _chebyshev(sq, wk)) * 2

    wkf, wkr = chess.square_file(wk), chess.square_rank(wk)
    bkf, bkr = chess.square_file(bk), chess.square_rank(bk)
    king_dist = abs(wkf - bkf) + abs(wkr - bkr)

    def get_corner_dist(f, r):
        return min(f + r, (7 - f) + r, f + (7 - r), (7 - f) + (7 - r))

    if material_score > 300:
        result += eg * (14 - king_dist) * 4
        result += eg * get_corner_dist(bkf, bkr) * 8
    elif material_score < -300:
        result -= eg * (14 - king_dist) * 4
        result -= eg * get_corner_dist(wkf, wkr) * 8
        
    return result


_KA_WEIGHTS = {
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  5,
}

def _rule_of_square(board: chess.Board, eg: float) -> float:
    if eg < 0.55:
        return 0.0
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign  = 1 if color == chess.WHITE else -1
        enemy = not color
        ek    = board.king(enemy)
        if ek is None:
            continue
        enemy_has_pieces = any(
            len(board.pieces(pt, enemy)) > 0
            for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
        )
        if enemy_has_pieces:
            continue
        for sq in board.pieces(chess.PAWN, color):
            f    = chess.square_file(sq)
            rank = chess.square_rank(sq)
            passed = True
            for esq in board.pieces(chess.PAWN, enemy):
                ef = chess.square_file(esq)
                er = chess.square_rank(esq)
                if abs(ef - f) <= 1:
                    if color == chess.WHITE and er >= rank:
                        passed = False; break
                    if color == chess.BLACK and er <= rank:
                        passed = False; break
            if not passed:
                continue
            promo_rank = 7 if color == chess.WHITE else 0
            promo_sq   = chess.square(f, promo_rank)
            steps      = abs(promo_rank - rank)
            king_dist  = _chebyshev(ek, promo_sq)
            steps_adj  = steps if board.turn == color else steps + 1
            if king_dist > steps_adj:
                bonus = min(400 + (7 - steps) * 60, 800)
                score += sign * bonus * eg
    return score


def _king_activity_endgame(board: chess.Board, eg: float) -> float:
    if eg < 0.45:
        return 0.0
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        ksq  = board.king(color)
        if ksq is None:
            continue
        kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
        center_dist = abs(kf - 3.5) + abs(kr - 3.5)
        score += sign * (7 - center_dist) * 5 * eg
    return score


def _king_opposition(board: chess.Board, eg: float) -> float:
    if eg < 0.60:
        return 0.0
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is None or bk is None:
        return 0.0
    wkf, wkr = chess.square_file(wk), chess.square_rank(wk)
    bkf, bkr = chess.square_file(bk), chess.square_rank(bk)
    score = 0.0
    if wkf == bkf and abs(wkr - bkr) == 2:
        bonus = 30 * eg
        score += bonus if board.turn == chess.BLACK else -bonus
    elif wkr == bkr and abs(wkf - bkf) == 2:
        bonus = 30 * eg
        score += bonus if board.turn == chess.BLACK else -bonus
    return score

def _king_attack_zone(board: chess.Board, eg: float) -> float:
    if eg > 0.80:
        return 0.0
    mg_weight = 1.0 - eg
    score     = 0.0
    for defender_color in (chess.WHITE, chess.BLACK):
        defender_sign  = 1 if defender_color == chess.WHITE else -1
        attacker_color = not defender_color
        king_sq = board.king(defender_color)
        if king_sq is None:
            continue
        kf = chess.square_file(king_sq)
        kr = chess.square_rank(king_sq)
        zone_sqs = []
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                f2, r2 = kf + df, kr + dr
                if 0 <= f2 <= 7 and 0 <= r2 <= 7:
                    zone_sqs.append(chess.square(f2, r2))
        zone_set = chess.SquareSet(zone_sqs)
        attack_units   = 0
        attacker_count = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None or piece.color != attacker_color:
                continue
            w = _KA_WEIGHTS.get(piece.piece_type, 0)
            if w == 0:
                continue
            atk = board.attacks(sq) & zone_set
            if atk:
                attacker_count += 1
                attack_units   += w * len(atk)
        if attacker_count >= 3:
            danger = attack_units * attacker_count * 1.2
        elif attacker_count == 2:
            danger = attack_units * 1.6
        else:
            danger = attack_units * 0.6
        score -= defender_sign * danger * mg_weight * 1.8
    return score


_WHITE_SPACE_SQS = chess.SquareSet([
    chess.C2, chess.D2, chess.E2, chess.F2,
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
])
_BLACK_SPACE_SQS = chess.SquareSet([
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
    chess.C7, chess.D7, chess.E7, chess.F7,
])

def _space_advantage(board: chess.Board, eg: float) -> float:
    if eg > 0.65:
        return 0.0
    mg_weight = 1.0 - eg
    w_space = b_space = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        atk = board.attacks(sq)
        if piece.color == chess.WHITE:
            w_space += len(atk & _BLACK_SPACE_SQS)
        else:
            b_space += len(atk & _WHITE_SPACE_SQS)
    return (w_space - b_space) * 1.5 * mg_weight


def _trapped_pieces(board: chess.Board) -> float:
    score = 0.0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        if piece.piece_type not in (chess.BISHOP, chess.ROOK):
            continue
        sign     = 1 if piece.color == chess.WHITE else -1
        mobility = len(board.attacks(sq))
        if piece.piece_type == chess.BISHOP and mobility <= 2:
            score += sign * (-30)
        elif piece.piece_type == chess.ROOK and mobility <= 3:
            score += sign * (-25)
    return score


def _queen_tropism(board: chess.Board, eg: float) -> float:
    if eg > 0.70:
        return 0.0
    mg_weight = 1.0 - eg
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign  = 1 if color == chess.WHITE else -1
        enemy = not color
        ek_sq = board.king(enemy)
        if ek_sq is None:
            continue
        ekf = chess.square_file(ek_sq)
        ekr = chess.square_rank(ek_sq)
        for sq in board.pieces(chess.QUEEN, color):
            qf  = chess.square_file(sq)
            qr  = chess.square_rank(sq)
            dist = abs(qf - ekf) + abs(qr - ekr)
            score += sign * (7 - dist) * 3 * mg_weight
    return score


def _fifty_move_pressure(board: chess.Board) -> float:
    hmc = board.halfmove_clock
    if hmc < 30:
        return 0.0
    pressure = (hmc - 30) * 3
    return -pressure if _ROOT_COLOR == chess.WHITE else pressure


import chess
from math import log as _log

def _anti_ai_complexity(board: chess.Board, eg: float) -> float:
    score = 0.0
    

    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2: score += 40
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2: score -= 40
    
    if eg < 0.5:
        w_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
        b_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
        score += (w_pawns - b_pawns) * 10
        
        w_pieces = len(board.piece_map())
        if w_pieces > 20: 
            score += 20 if board.turn == chess.WHITE else -20
            
    return score

def _fork_and_threat_evaluation(board: chess.Board) -> float:
    score = 0.0
    VALUABLE = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 3, chess.KING: 4}

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        enemy = not color
        
        for sq, piece in board.piece_map().items():
            if piece.color != color:
                continue
                
            attacked_squares = board.attacks(sq)
            high_value_targets = 0
            
            for target_sq in attacked_squares:
                target_piece = board.piece_at(target_sq)
                if target_piece and target_piece.color == enemy:
                    if target_piece.piece_type in VALUABLE:
                        if PIECE_VALUES[piece.piece_type] < PIECE_VALUES[target_piece.piece_type]:
                            high_value_targets += 1
                        elif not board.is_attacked_by(color, target_sq):
                            high_value_targets += 1
                            
            if high_value_targets >= 2:
                score += sign * 150 
                
    return score

def evaluate(board: chess.Board) -> float:
    if board.is_checkmate():
        return -99999 if board.turn == chess.WHITE else 99999
    if (board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_repetition(3)
            or board.is_fifty_moves()):
        return -60 if _ROOT_COLOR == chess.WHITE else 60
    if board.is_repetition(2):
        return (-60 if _ROOT_COLOR == chess.WHITE else 60) * 3.0

    eg = _endgame_factor(board)
    score = 0.0

    score += _material_and_pst(board, eg)

    score += _pawn_structure(board)
    score += _pawn_chain_bonus(board)

    w = b = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            n = len(board.attacks(sq))
            if piece.color == chess.WHITE:
                w += n
            else:
                b += n
    score += 0.1 * (w - b)

    score += _king_safety(board, eg)
    score += _king_attack_zone(board, eg)  

    score += _pawn_storm(board, eg)
    score += _castling_bonus(board)
    score += _bishop_pair(board)
    score += _rook_open_files(board)
    score += _knight_outposts(board)
    score += _rook_on_seventh(board)
    score += _connected_rooks(board)

    score += _hanging_pieces(board)

    score += _fork_evaluation(board)

    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.piece_type != chess.KNIGHT:
            continue
        sign = 1 if piece.color == chess.WHITE else -1
        enemy = not piece.color

        for move in board.legal_moves:
            if move.from_square != sq:
                continue

            tsq = move.to_square
            occupant = board.piece_at(tsq)
            if occupant is not None and occupant.color == piece.color:
                continue

            future_vals = []
            for atsq in chess.SquareSet(chess.BB_KNIGHT_ATTACKS[tsq]):
                if atsq == sq:
                    continue
                tp = board.piece_at(atsq)
                if tp and tp.color == enemy and tp.piece_type in {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING}:
                    future_vals.append(PIECE_VALUES[tp.piece_type])

            if len(future_vals) >= 2:
                vals = sorted(future_vals, reverse=True)[:2]
                weight = 0.06 if occupant is None else 0.03
                score += sign * (vals[0] + vals[1]) * weight 

    score += _king_proximity_mop_up(board, eg, _material_and_pst(board, eg))
    score += _space_advantage(board, eg)
    score += _trapped_pieces(board)
    score += _queen_tropism(board, eg)
    score += _fifty_move_pressure(board)
    score += _rule_of_square(board, eg)
    score += _king_activity_endgame(board, eg)
    score += _king_opposition(board, eg)

    material_score = 0.0
    for pt in PIECE_VALUES:
        pst = _get_pst(pt, eg)
        for sq in board.pieces(pt, chess.WHITE):
            material_score += PIECE_VALUES[pt] + pst[chess.square_mirror(sq)]
        for sq in board.pieces(pt, chess.BLACK):
            material_score -= PIECE_VALUES[pt] + pst[sq]
                
    score += material_score
        
    draw_score = _evaluate_draw(board, score)
    if draw_score != 0.0:
        return draw_score
     
    w_attacks = 0
    b_attacks = 0
    for sq, piece in board.piece_map().items():
        if piece.color == chess.WHITE:
            w_attacks += len(board.attacks(sq))
        else:
            b_attacks += len(board.attacks(sq))
    score += 0.05 * (w_attacks - b_attacks)
    
    score += _fork_and_threat_evaluation(board)
    score += _anti_ai_complexity(board, eg)

    score += (15 if board.turn == chess.WHITE else -15)
    if board.is_check():
        score += (20 if board.turn == chess.BLACK else -20)
    return score


_TT           = {}
_KILLERS      = [[None, None] for _ in range(128)]
_HISTORY      = {}
_COUNTERMOVE  = {}

TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

TARGET_DEPTH = 10

_FUTILITY_MARGINS = {1: 120, 2: 300, 3: 500}
_LMP_CUTOFFS      = {1: 8, 2: 12, 3: 20}
_RFP_MARGIN       = 120


def _tt_store(h: int, depth: int, flag: int, score: float, move) -> None:
    existing = _TT.get(h)
    if existing is None or depth >= existing[0]:
        _TT[h] = (depth, flag, score, move)


def _tt_probe(h: int):
    return _TT.get(h)


def _update_killers(move: chess.Move, ply: int) -> None:
    if ply >= 128:
        return
    if _KILLERS[ply][0] != move:
        _KILLERS[ply][1] = _KILLERS[ply][0]
        _KILLERS[ply][0] = move


def _update_history(move: chess.Move, depth: int) -> None:
    key = (move.from_square, move.to_square)
    _HISTORY[key] = min(_HISTORY.get(key, 0) + depth * depth, 50_000)


def _update_countermove(move: chess.Move, prev_move) -> None:
    if prev_move is not None:
        _COUNTERMOVE[(prev_move.from_square, prev_move.to_square)] = move


def _order_moves(board: chess.Board, moves, tt_move=None, ply: int = 0, prev_move=None):
    tt_move_obj = tt_move
    ks          = _KILLERS[ply] if ply < 128 else [None, None]
    killer_set  = {k for k in ks if k is not None}
    cm = None
    if prev_move is not None:
        cm = _COUNTERMOVE.get((prev_move.from_square, prev_move.to_square))

    def _score(move: chess.Move) -> int:
        if move == tt_move_obj:
            return 30_000
        is_cap = board.is_capture(move)
        if is_cap:
            victim   = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            v_val = PIECE_VALUES.get(victim.piece_type,   0)   if victim   else 0
            a_val = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
            gain  = 10 * v_val - a_val
            return (20_000 + gain) if gain >= 0 else (-10_000 + gain)
        if move.promotion == chess.QUEEN:
            return 19_000
        if move in killer_set:
            return 10_000
        if cm is not None and move == cm:
            return 9_000
        return _HISTORY.get((move.from_square, move.to_square), 0)

    return sorted(moves, key=_score, reverse=True)


_DELTA_PRUNE_MARGIN = 900

def _qsearch(board: chess.Board, alpha: float, beta: float, maximizing: bool) -> float:
    if board.is_checkmate():
        return -99_999 if board.turn == chess.WHITE else 99_999
    if (board.is_stalemate() or board.is_insufficient_material()
            or board.is_repetition(3) or board.is_fifty_moves()):
        return _contempt_score()

    stand_pat = evaluate(board)

    if maximizing:
        if stand_pat >= beta:
            return beta
        if stand_pat + _DELTA_PRUNE_MARGIN < alpha:
            return alpha
        if stand_pat > alpha:
            alpha = stand_pat
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
        if stand_pat - _DELTA_PRUNE_MARGIN > beta:
            return beta
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


def minimax(board: chess.Board, depth: int,
            alpha: float, beta: float,
            maximizing: bool,
            ply: int = 0,
            null_ok: bool = True,
            prev_move: chess.Move = None) -> float:

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
        if tt_move_raw is not None and tt_move_raw in board.legal_moves:
            tt_move = tt_move_raw

    if board.is_game_over():
        return evaluate(board)

    if depth <= 0:
        return _qsearch(board, alpha, beta, maximizing)

    in_check = board.is_check()
    if in_check:
        depth += 1

    eg = _endgame_factor(board)

    if (not in_check and ply > 0 and depth <= 6 and eg < 0.90
            and abs(beta) < 90_000):
        static_eval = evaluate(board)
        rfp_margin  = depth * _RFP_MARGIN
        if maximizing and static_eval - rfp_margin >= beta:
            return static_eval
        if not maximizing and static_eval + rfp_margin <= alpha:
            return static_eval
    else:
        static_eval = None

    if null_ok and ply > 0 and not in_check and depth >= 3 and eg < 0.85:
        R = 3 if depth >= 4 else 2
        board.push(chess.Move.null())
        if maximizing:
            null_score = minimax(board, depth - 1 - R,
                                 beta - 1, beta, False, ply + 1, False, None)
        else:
            null_score = minimax(board, depth - 1 - R,
                                 alpha, alpha + 1, True, ply + 1, False, None)
        board.pop()
        if maximizing and null_score >= beta:
            return beta
        if not maximizing and null_score <= alpha:
            return alpha

    if tt_move is None and depth >= 4 and null_ok:
        depth -= 1

    moves = _order_moves(board, list(board.legal_moves), tt_move, ply, prev_move)
    if not moves:
        return evaluate(board)

    orig_alpha  = alpha
    orig_beta   = beta
    best_move   = None
    quiet_count = 0

    if maximizing:
        best = float('-inf')
        for move_idx, move in enumerate(moves):
            is_cap   = board.is_capture(move)
            is_quiet = (not is_cap and move.promotion is None)

            if (is_quiet and not in_check and depth in _FUTILITY_MARGINS
                    and move_idx > 0 and abs(alpha) < 90_000):
                if static_eval is None:
                    static_eval = evaluate(board)
                if static_eval + _FUTILITY_MARGINS[depth] <= alpha:
                    quiet_count += 1
                    continue

            if (is_quiet and not in_check and depth in _LMP_CUTOFFS
                    and quiet_count >= _LMP_CUTOFFS[depth]
                    and abs(alpha) < 90_000):
                continue

            board.push(move)
            gives_check = board.is_check()

            reduce = 0
            if (move_idx >= 2 and depth >= 3 and is_quiet
                    and not in_check and not gives_check):
                reduce = max(1, int(_log(max(depth, 1)) * _log(max(move_idx + 1, 1)) / 2.25))

            if move_idx == 0:
                score = minimax(board, depth - 1, alpha, beta, False, ply + 1, True, move)
            elif reduce > 0:
                score = minimax(board, depth - 1 - reduce, alpha, alpha + 1, False, ply + 1, True, move)
                if score > alpha:
                    score = minimax(board, depth - 1, alpha, beta, False, ply + 1, True, move)
            else:
                score = minimax(board, depth - 1, alpha, alpha + 1, False, ply + 1, True, move)
                if score > alpha:
                    score = minimax(board, depth - 1, alpha, beta, False, ply + 1, True, move)

            board.pop()

            if is_quiet:
                quiet_count += 1

            if score > best:
                best      = score
                best_move = move
            if score > alpha:
                alpha = score
            if beta <= alpha:
                if is_quiet:
                    _update_killers(move, ply)
                    _update_history(move, depth)
                    _update_countermove(move, prev_move)
                break

        flag = (TT_EXACT if orig_alpha < best < orig_beta
                else (TT_LOWER if best >= orig_beta else TT_UPPER))
        _tt_store(h, depth, flag, best, best_move)
        return best

    else:
        best = float('inf')
        for move_idx, move in enumerate(moves):
            is_cap   = board.is_capture(move)
            is_quiet = (not is_cap and move.promotion is None)

            if (is_quiet and not in_check and depth in _FUTILITY_MARGINS
                    and move_idx > 0 and abs(beta) < 90_000):
                if static_eval is None:
                    static_eval = evaluate(board)
                if static_eval - _FUTILITY_MARGINS[depth] >= beta:
                    quiet_count += 1
                    continue

            if (is_quiet and not in_check and depth in _LMP_CUTOFFS
                    and quiet_count >= _LMP_CUTOFFS[depth]
                    and abs(beta) < 90_000):
                continue

            board.push(move)
            gives_check = board.is_check()

            reduce = 0
            if (move_idx >= 2 and depth >= 3 and is_quiet
                    and not in_check and not gives_check):
                reduce = max(1, int(_log(max(depth, 1)) * _log(max(move_idx + 1, 1)) / 2.25))

            if move_idx == 0:
                score = minimax(board, depth - 1, alpha, beta, True, ply + 1, True, move)
            elif reduce > 0:
                score = minimax(board, depth - 1 - reduce, beta - 1, beta, True, ply + 1, True, move)
                if score < beta:
                    score = minimax(board, depth - 1, alpha, beta, True, ply + 1, True, move)
            else:
                score = minimax(board, depth - 1, beta - 1, beta, True, ply + 1, True, move)
                if score < beta:
                    score = minimax(board, depth - 1, alpha, beta, True, ply + 1, True, move)

            board.pop()

            if is_quiet:
                quiet_count += 1

            if score < best:
                best      = score
                best_move = move
            if score < beta:
                beta = score
            if beta <= alpha:
                if is_quiet:
                    _update_killers(move, ply)
                    _update_history(move, depth)
                    _update_countermove(move, prev_move)
                break

        flag = (TT_EXACT if orig_alpha < best < orig_beta
                else (TT_UPPER if best <= orig_alpha else TT_LOWER))
        _tt_store(h, depth, flag, best, best_move)
        return best


def get_next_move(board: chess.Board,
                  color: chess.Color,
                  depth: int = depth) -> chess.Move:
    global _KILLERS, _HISTORY, _ROOT_COLOR

    _ROOT_COLOR = color

    _KILLERS = [[None, None] for _ in range(128)]
    for k in list(_HISTORY.keys()):
        _HISTORY[k] = _HISTORY[k] >> 1
        if _HISTORY[k] == 0:
            del _HISTORY[k]

    target     = max(depth, TARGET_DEPTH)
    maximizing = (color == chess.WHITE)
    best_move  = None
    prev_score = 0.0

    b = board.copy()

    all_legal = list(b.legal_moves)

    def _causes_repetition(move: chess.Move) -> bool:
        b.push(move)
        result = b.is_repetition(3)
        b.pop()
        return result

    non_rep_moves = [m for m in all_legal if not _causes_repetition(m)]
    repetition_only = len(non_rep_moves) == 0

    for d in range(1, target + 1):

        if d >= 4 and best_move is not None:
            asp_delta = 50
            asp_lo    = prev_score - asp_delta
            asp_hi    = prev_score + asp_delta
        else:
            asp_lo    = float('-inf')
            asp_hi    = float('inf')
            asp_delta = float('inf')

        retry_limit = 0
        while True:
            h_root      = chess.polyglot.zobrist_hash(b)
            tt_root     = _tt_probe(h_root)
            root_tt_move = None
            if tt_root is not None:
                _, _, _, root_tt_move = tt_root
                if root_tt_move is not None and root_tt_move not in b.legal_moves:
                    root_tt_move = None

            candidate_pool = all_legal if repetition_only else non_rep_moves
            moves = _order_moves(b, candidate_pool, root_tt_move, ply=0)

            cur_best_move  = None
            cur_best_score = float('-inf') if maximizing else float('inf')
            a  = asp_lo
            bw = asp_hi

            for move in moves:
                b.push(move)
                score = minimax(b, d - 1, a, bw,
                                not maximizing, ply=1, prev_move=move)
                b.pop()

                if maximizing:
                    if score > cur_best_score:
                        cur_best_score = score
                        cur_best_move  = move
                    a = max(a, score)
                    if bw <= a:
                        cur_best_score = score
                        cur_best_move  = move
                        break
                else:
                    if score < cur_best_score:
                        cur_best_score = score
                        cur_best_move  = move
                    bw = min(bw, score)
                    if bw <= a:
                        cur_best_score = score
                        cur_best_move  = move
                        break

            if asp_delta == float('inf'):
                if cur_best_move is not None:
                    best_move  = cur_best_move
                    prev_score = cur_best_score
                break

            retry_limit += 1
            if retry_limit >= 6:
                if cur_best_move is not None:
                    best_move  = cur_best_move
                    prev_score = cur_best_score
                break

            if maximizing:
                if cur_best_score <= asp_lo:
                    asp_lo    -= asp_delta
                    asp_delta *= 2
                elif cur_best_score >= asp_hi:
                    asp_hi    += asp_delta
                    asp_delta *= 2
                else:
                    best_move  = cur_best_move
                    prev_score = cur_best_score
                    break
            else:
                if cur_best_score >= asp_hi:
                    asp_hi    += asp_delta
                    asp_delta *= 2
                elif cur_best_score <= asp_lo:
                    asp_lo    -= asp_delta
                    asp_delta *= 2
                else:
                    best_move  = cur_best_move
                    prev_score = cur_best_score
                    break

        if best_move is not None:
            _tt_store(
                chess.polyglot.zobrist_hash(board),
                d, TT_EXACT, prev_score, best_move,
            )
        manager = mp.Manager()
        shared_tt = manager.dict()
        result_queue = mp.Queue()
        
        cores = mp.cpu_count()
        processes = []
        
        # Lazy SMP: Multiple depths to find a stable best move quickly
        for i in range(cores):
            # Workers search depth, depth+1, depth+2 etc since no time limit
            p = mp.Process(target=search_worker, args=(board.fen(), depth + (i % 2), shared_tt, result_queue))
            p.start()
            processes.append(p)
            
        best_move = None
        max_completed_depth = -1
        
        # Since no time limit, we gather the results from the deepest available search
        for _ in range(cores):
            score, move, d = result_queue.get()
            if d > max_completed_depth:
                max_completed_depth = d
                best_move = move
                print(f"Depth {d}/{target} complete. Best move: {best_move}, Score: {prev_score:.2f}")
                
        for p in processes:
            p.terminate()

    return best_move if best_move else list(board.legal_moves)[0]


if __name__ == '__main__':
    import time

    b = chess.Board()
    print(f"Starting position eval: {evaluate(b):.1f} cp  (should be near 0)")

    for d in (3, 5, TARGET_DEPTH):
        _TT.clear()
        _KILLERS[:] = [[None, None] for _ in range(128)]
        _HISTORY.clear()
        _COUNTERMOVE.clear()
        t0      = time.time()
        move    = get_next_move(b, chess.WHITE, depth=d)
        elapsed = time.time() - t0
        print(f"[team_goraieb] depth={d:2d}: {b.san(move):6s}  ({elapsed:.2f}s)")

