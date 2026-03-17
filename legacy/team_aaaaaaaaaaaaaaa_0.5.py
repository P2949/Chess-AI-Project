"""
team_aaaaaaaaaaaaaaa.py – Championship Chess Bot
=========================================
Minimax + Alpha-Beta with iterative deepening, aspiration windows, PVS,
transposition table, killer/history/countermove heuristics, null‑move pruning,
reverse futility pruning, futility pruning, late move pruning, late move reductions
(log‑log formula), check extensions, quiescence search with delta pruning,
and contempt for draws.

Evaluation combines 20+ positional terms, all phase‑aware through PST interpolation.
Optimized with Bitboards and piece_map() for Python execution speed.
"""

import chess
import chess.polyglot
from math import log as _log
import time

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
PIECE_VALUES = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   20000,
}

# Middlegame piece‑square tables (index 0 = a8 … 63 = h1)
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

# Endgame piece‑square tables
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

PST_MG = {
    chess.PAWN:   PST_PAWN_MG,
    chess.KNIGHT: PST_KNIGHT_MG,
    chess.BISHOP: PST_BISHOP_MG,
    chess.ROOK:   PST_ROOK_MG,
    chess.QUEEN:  PST_QUEEN_MG,
    chess.KING:   PST_KING_MG,
}

PST_EG = {
    chess.PAWN:   PST_PAWN_EG,
    chess.KNIGHT: PST_KNIGHT_EG,
    chess.BISHOP: PST_BISHOP_EG,
    chess.ROOK:   PST_ROOK_EG,
    chess.QUEEN:  PST_QUEEN_EG,
    chess.KING:   PST_KING_EG,
}

# ------------------------------------------------------------------------------
# Global search state
# ------------------------------------------------------------------------------
_TT      = {}                      
_MAX_TT_SIZE = 1_000_000            # Prevent memory leaks
_KILLERS = [[None, None] for _ in range(128)]
_HISTORY = {}                       
_COUNTERMOVE = {}                   

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
CONTEMPT = 60                       

_ROOT_COLOR = chess.WHITE

# Pruning margins
_FUTILITY_MARGINS = {1: 120, 2: 300, 3: 500}
_LMP_CUTOFFS      = {1: 8, 2: 12, 3: 20}
_RFP_MARGIN       = 120
_DELTA_PRUNE      = 900

# Bitboard Masks
WHITE_SPACE_MASK = 0x00000000003C3C3C
BLACK_SPACE_MASK = 0x3C3C3C0000000000

# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def _endgame_factor(board: chess.Board) -> float:
    """0.0 = opening/middlegame, 1.0 = pure endgame. Optimized with bitboards."""
    phase = (
        chess.popcount(board.pieces_mask(chess.QUEEN, chess.WHITE) | board.pieces_mask(chess.QUEEN, chess.BLACK)) * 4 +
        chess.popcount(board.pieces_mask(chess.ROOK, chess.WHITE) | board.pieces_mask(chess.ROOK, chess.BLACK)) * 2 +
        chess.popcount(board.pieces_mask(chess.BISHOP, chess.WHITE) | board.pieces_mask(chess.BISHOP, chess.BLACK)) +
        chess.popcount(board.pieces_mask(chess.KNIGHT, chess.WHITE) | board.pieces_mask(chess.KNIGHT, chess.BLACK))
    )
    return 1.0 - min(phase, 24) / 24.0

def _chebyshev(a: int, b: int) -> int:
    return max(abs(chess.square_file(a) - chess.square_file(b)),
               abs(chess.square_rank(a) - chess.square_rank(b)))

def _manhattan(a: int, b: int) -> int:
    return abs(chess.square_file(a) - chess.square_file(b)) + \
           abs(chess.square_rank(a) - chess.square_rank(b))

# ------------------------------------------------------------------------------
# Evaluation helpers (all return floats, positive = good for White)
# ------------------------------------------------------------------------------
def _material_and_pst(board: chess.Board, eg: float) -> float:
    """Material + interpolated piece‑square value for every piece using piece_map."""
    score = 0.0
    for sq, piece in board.piece_map().items():
        sign = 1 if piece.color == chess.WHITE else -1
        idx = chess.square_mirror(sq) if piece.color == chess.WHITE else sq
        mg = PST_MG[piece.piece_type][idx]
        eg_tbl = PST_EG[piece.piece_type][idx]
        pos = mg * (1.0 - eg) + eg_tbl * eg
        score += sign * (PIECE_VALUES[piece.piece_type] + pos)
    return score

def _pawn_structure(board: chess.Board) -> float:
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        pawns = board.pieces(chess.PAWN, color)
        epawns = board.pieces(chess.PAWN, not color)
        
        file_counts = [0] * 8
        for f in (chess.square_file(sq) for sq in pawns):
            file_counts[f] += 1

        for sq in pawns:
            f = chess.square_file(sq)
            rank = chess.square_rank(sq)

            if file_counts[f] > 1: score += sign * -20 # doubled
            if (f == 0 or file_counts[f-1] == 0) and (f == 7 or file_counts[f+1] == 0): score += sign * -15 # isolated

            # passed
            passed = True
            for esq in epawns:
                if chess.square_file(esq) in (f-1, f, f+1):
                    er = chess.square_rank(esq)
                    if (color == chess.WHITE and er >= rank) or (color == chess.BLACK and er <= rank):
                        passed = False
                        break
            if passed:
                advancement = rank if color == chess.WHITE else 7 - rank
                bonus = 20 + advancement * 10
                prot_rank = rank - 1 if color == chess.WHITE else rank + 1
                if 0 <= prot_rank <= 7:
                    for df in (-1, 1):
                        pf = f + df
                        if 0 <= pf <= 7 and board.piece_at(chess.square(pf, prot_rank)) == chess.Piece(chess.PAWN, color):
                            bonus += 15
                            break
                score += sign * bonus

            # backward
            support_ranks = range(0, rank) if color == chess.WHITE else range(rank+1, 8)
            has_support = any(chess.square_file(s) in (f-1, f+1) and chess.square_rank(s) in support_ranks for s in pawns)
            if not has_support and not passed:
                score += sign * -10
    return score

def _pawn_chain_bonus(board: chess.Board) -> float:
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        for sq in board.pieces(chess.PAWN, color):
            f, rank = chess.square_file(sq), chess.square_rank(sq)
            prot_rank = rank - 1 if color == chess.WHITE else rank + 1
            if 0 <= prot_rank <= 7:
                for df in (-1, 1):
                    pf = f + df
                    if 0 <= pf <= 7 and board.piece_at(chess.square(pf, prot_rank)) == chess.Piece(chess.PAWN, color):
                        score += sign * 7
                        break
    return score

def _mobility(board: chess.Board) -> float:
    w = sum(chess.popcount(board.attacks_mask(sq)) for sq in chess.SquareSet(board.occupied_co[chess.WHITE]))
    b = sum(chess.popcount(board.attacks_mask(sq)) for sq in chess.SquareSet(board.occupied_co[chess.BLACK]))
    return 0.1 * (w - b)

def _king_safety(board: chess.Board, eg: float) -> float:
    if eg >= 0.9: return 0.0
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        ksq = board.king(color)
        if ksq is None: continue
        sign = 1 if color == chess.WHITE else -1
        kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
        for df in (-1, 0, 1):
            f = kf + df
            if not (0 <= f <= 7): continue
            shield_r = kr + (1 if color == chess.WHITE else -1)
            if 0 <= shield_r <= 7:
                if board.piece_at(chess.square(f, shield_r)) == chess.Piece(chess.PAWN, color):
                    score += sign * 12 * (1.0 - eg)
                else:
                    score += sign * -18 * (1.0 - eg)
            if not any(chess.square_file(s) == f for s in board.pieces(chess.PAWN, color)):
                score += sign * -15 * (1.0 - eg)
    return score

def _castling_bonus(board: chess.Board) -> float:
    return (30.0 if board.has_castling_rights(chess.WHITE) else 0) - (30.0 if board.has_castling_rights(chess.BLACK) else 0)

def _bishop_pair(board: chess.Board) -> float:
    return (30.0 if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2 else 0) - (30.0 if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2 else 0)

def _rook_open_files(board: chess.Board) -> float:
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        for sq in board.pieces(chess.ROOK, color):
            f = chess.square_file(sq)
            own = sum(1 for s in board.pieces(chess.PAWN, color) if chess.square_file(s) == f)
            enemy = sum(1 for s in board.pieces(chess.PAWN, not color) if chess.square_file(s) == f)
            if own == 0 and enemy == 0: score += sign * 25
            elif own == 0: score += sign * 10
    return score

def _knight_outposts(board: chess.Board) -> float:
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        epawns = board.pieces(chess.PAWN, not color)
        for sq in board.pieces(chess.KNIGHT, color):
            rank, f = chess.square_rank(sq), chess.square_file(sq)
            if (color == chess.WHITE and rank not in (3, 4, 5)) or (color == chess.BLACK and rank not in (2, 3, 4)): continue
            pawn_rank = rank - 1 if color == chess.WHITE else rank + 1
            if not (0 <= pawn_rank <= 7): continue
            protected = any(chess.square_file(s) in (f-1, f+1) and chess.square_rank(s) == pawn_rank for s in board.pieces(chess.PAWN, color))
            if not protected: continue
            
            attack_rank = rank + 1 if color == chess.WHITE else rank - 1
            attackable = 0 <= attack_rank <= 7 and any(chess.square_file(s) in (f-1, f+1) and chess.square_rank(s) == attack_rank for s in epawns)
            if not attackable: score += sign * 20
    return score

def _connected_rooks(board: chess.Board) -> float:
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        rooks = list(board.pieces(chess.ROOK, color))
        if len(rooks) < 2: continue
        r1, r2 = rooks[0], rooks[1]
        f1, rk1 = chess.square_file(r1), chess.square_rank(r1)
        f2, rk2 = chess.square_file(r2), chess.square_rank(r2)
        if (f1 == f2 and all(board.piece_at(chess.square(f1, r)) is None for r in range(min(rk1, rk2) + 1, max(rk1, rk2)))) or \
           (rk1 == rk2 and all(board.piece_at(chess.square(f, rk1)) is None for f in range(min(f1, f2) + 1, max(f1, f2)))):
            score += 15 if color == chess.WHITE else -15
    return score

def _hanging_pieces(board: chess.Board) -> float:
    score = 0.0
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING: continue
        enemy = not piece.color
        if board.is_attacked_by(enemy, sq):
            defenders = chess.popcount(board.attackers_mask(piece.color, sq))
            attackers = chess.popcount(board.attackers_mask(enemy, sq))
            if defenders < attackers:
                val = PIECE_VALUES[piece.piece_type] * 0.25
                score += -val if piece.color == chess.WHITE else val
    return score

def _space_advantage(board: chess.Board, eg: float) -> float:
    if eg > 0.65: return 0.0
    w_space = sum(chess.popcount(board.attacks_mask(sq) & BLACK_SPACE_MASK) for sq in chess.SquareSet(board.occupied_co[chess.WHITE]))
    b_space = sum(chess.popcount(board.attacks_mask(sq) & WHITE_SPACE_MASK) for sq in chess.SquareSet(board.occupied_co[chess.BLACK]))
    return (w_space - b_space) * 1.5 * (1.0 - eg)

def _trapped_pieces(board: chess.Board) -> float:
    score = 0.0
    for sq in board.pieces(chess.BISHOP, chess.WHITE):
        if chess.popcount(board.attacks_mask(sq)) <= 2: score -= 30
    for sq in board.pieces(chess.BISHOP, chess.BLACK):
        if chess.popcount(board.attacks_mask(sq)) <= 2: score += 30
    for sq in board.pieces(chess.ROOK, chess.WHITE):
        if chess.popcount(board.attacks_mask(sq)) <= 3: score -= 25
    for sq in board.pieces(chess.ROOK, chess.BLACK):
        if chess.popcount(board.attacks_mask(sq)) <= 3: score += 25
    return score

def _contempt_score() -> float:
    return -CONTEMPT if _ROOT_COLOR == chess.WHITE else CONTEMPT

# ------------------------------------------------------------------------------
# Main evaluation function
# ------------------------------------------------------------------------------
def evaluate(board: chess.Board) -> float:
    if board.is_checkmate(): return -99_999 if board.turn == chess.WHITE else 99_999
    if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3) or board.is_fifty_moves():
        return _contempt_score()
    if board.is_repetition(2): return _contempt_score() * 0.7

    eg = _endgame_factor(board)
    score = _material_and_pst(board, eg)
    score += _pawn_structure(board)
    score += _pawn_chain_bonus(board)
    score += _mobility(board)
    score += _king_safety(board, eg)
    score += _castling_bonus(board)
    score += _bishop_pair(board)
    score += _rook_open_files(board)
    score += _knight_outposts(board)
    score += _connected_rooks(board)
    score += _hanging_pieces(board)
    score += _space_advantage(board, eg)
    score += _trapped_pieces(board)

    # Tempo bonus (side to move)
    score += 15 if board.turn == chess.WHITE else -15
    if board.is_check():
        score += 20 if board.turn == chess.BLACK else -20

    return score

# ------------------------------------------------------------------------------
# Move ordering & Transposition
# ------------------------------------------------------------------------------
def _capture_gain(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_at(move.to_square)
    victim_value = PIECE_VALUES[chess.PAWN] if (victim is None and board.is_en_passant(move)) else PIECE_VALUES.get(victim.piece_type, 0) if victim else 0
    attacker = board.piece_at(move.from_square)
    attacker_value = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
    return 10 * victim_value - attacker_value

def _order_moves(board: chess.Board, moves, tt_move=None, ply: int = 0, prev_move=None):
    ks = _KILLERS[ply] if ply < 128 else [None, None]
    killer_set = {k for k in ks if k is not None}
    cm = _COUNTERMOVE.get((prev_move.from_square, prev_move.to_square)) if prev_move else None

    def _score(move: chess.Move) -> int:
        if move == tt_move: return 30_000
        if board.is_capture(move):
            gain = _capture_gain(board, move)
            return (20_000 + gain) if gain >= 0 else (-10_000 + gain)
        if move.promotion == chess.QUEEN: return 19_000
        if move in killer_set: return 10_000
        if move == cm: return 9_000
        return _HISTORY.get((move.from_square, move.to_square), 0)

    return sorted(moves, key=_score, reverse=True)

def _tt_store(h: int, depth: int, flag: int, score: float, move) -> None:
    if len(_TT) > _MAX_TT_SIZE: _TT.clear()
    existing = _TT.get(h)
    if existing is None or depth >= existing[0]:
        _TT[h] = (depth, flag, score, move)

def _tt_probe(h: int):
    return _TT.get(h)

def _update_killers(move: chess.Move, ply: int) -> None:
    if ply < 128 and _KILLERS[ply][0] != move:
        _KILLERS[ply][1] = _KILLERS[ply][0]
        _KILLERS[ply][0] = move

def _update_history(move: chess.Move, depth: int) -> None:
    key = (move.from_square, move.to_square)
    _HISTORY[key] = min(_HISTORY.get(key, 0) + depth * depth, 50_000)

def _update_countermove(move: chess.Move, prev_move) -> None:
    if prev_move: _COUNTERMOVE[(prev_move.from_square, prev_move.to_square)] = move

# ------------------------------------------------------------------------------
# Search Engines
# ------------------------------------------------------------------------------
def _qsearch(board: chess.Board, alpha: float, beta: float, maximizing: bool) -> float:
    if board.is_checkmate(): return -99_999 if board.turn == chess.WHITE else 99_999
    if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3) or board.is_fifty_moves():
        return _contempt_score()

    stand_pat = evaluate(board)
    if maximizing:
        if stand_pat >= beta: return beta
        if stand_pat + _DELTA_PRUNE < alpha: return alpha
        if stand_pat > alpha: alpha = stand_pat
        candidates = list(board.legal_moves) if board.is_check() else [m for m in board.legal_moves if board.is_capture(m) or m.promotion == chess.QUEEN]
        for move in _order_moves(board, candidates):
            board.push(move)
            score = _qsearch(board, alpha, beta, False)
            board.pop()
            if score >= beta: return beta
            if score > alpha: alpha = score
        return alpha
    else:
        if stand_pat <= alpha: return alpha
        if stand_pat - _DELTA_PRUNE > beta: return beta
        if stand_pat < beta: beta = stand_pat
        candidates = list(board.legal_moves) if board.is_check() else [m for m in board.legal_moves if board.is_capture(m) or m.promotion == chess.QUEEN]
        for move in _order_moves(board, candidates):
            board.push(move)
            score = _qsearch(board, alpha, beta, True)
            board.pop()
            if score <= alpha: return alpha
            if score < beta: beta = score
        return beta

def minimax(board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool, ply: int = 0, null_ok: bool = True, prev_move: chess.Move = None) -> float:
    h = chess.polyglot.zobrist_hash(board)
    tt_data = _tt_probe(h)
    tt_move = None

    if tt_data:
        tt_depth, tt_flag, tt_score, tt_move_raw = tt_data
        if tt_depth >= depth:
            if tt_flag == TT_EXACT: return tt_score
            if tt_flag == TT_LOWER and tt_score >= beta: return tt_score
            if tt_flag == TT_UPPER and tt_score <= alpha: return tt_score
        if tt_move_raw and tt_move_raw in board.legal_moves:
            tt_move = tt_move_raw

    if board.is_game_over(): return evaluate(board)
    if depth <= 0: return _qsearch(board, alpha, beta, maximizing)

    in_check = board.is_check()
    if in_check: depth += 1
    eg = _endgame_factor(board)

    static_eval = None
    if not in_check and ply > 0 and depth <= 6 and eg < 0.90 and abs(beta) < 90_000:
        static_eval = evaluate(board)
        rfp_margin = depth * _RFP_MARGIN
        if maximizing and static_eval - rfp_margin >= beta: return static_eval
        if not maximizing and static_eval + rfp_margin <= alpha: return static_eval

    if null_ok and ply > 0 and not in_check and depth >= 3 and eg < 0.85:
        board.push(chess.Move.null())
        null_score = minimax(board, depth - 1 - (3 if depth >= 4 else 2), beta - 1 if maximizing else alpha, beta if maximizing else alpha + 1, not maximizing, ply + 1, False, None)
        board.pop()
        if maximizing and null_score >= beta: return beta
        if not maximizing and null_score <= alpha: return alpha

    if tt_move is None and depth >= 4 and null_ok:
        minimax(board, depth - 2, alpha, beta, maximizing, ply, False, prev_move)
        tt_data = _tt_probe(h)
        if tt_data: tt_move = tt_data[3]

    moves = _order_moves(board, list(board.legal_moves), tt_move, ply, prev_move)
    if not moves: return evaluate(board)

    orig_alpha, orig_beta = alpha, beta
    best_move, best = None, float('-inf') if maximizing else float('inf')
    quiet_count = 0

    for move_idx, move in enumerate(moves):
        is_quiet = not board.is_capture(move) and move.promotion is None
        
        if is_quiet and not in_check and depth in _FUTILITY_MARGINS and move_idx > 0 and abs(alpha if maximizing else beta) < 90_000:
            if static_eval is None: static_eval = evaluate(board)
            if (maximizing and static_eval + _FUTILITY_MARGINS[depth] <= alpha) or (not maximizing and static_eval - _FUTILITY_MARGINS[depth] >= beta):
                quiet_count += 1
                continue

        if is_quiet and not in_check and depth in _LMP_CUTOFFS and quiet_count >= _LMP_CUTOFFS[depth] and abs(alpha if maximizing else beta) < 90_000: continue

        board.push(move)
        gives_check = board.is_check()
        reduce = max(1, int(_log(max(depth, 1)) * _log(move_idx + 1) / 2.25)) if (move_idx >= 2 and depth >= 3 and is_quiet and not in_check and not gives_check) else 0

        if maximizing:
            if move_idx == 0: score = minimax(board, depth - 1, alpha, beta, False, ply + 1, True, move)
            elif reduce > 0:
                score = minimax(board, depth - 1 - reduce, alpha, alpha + 1, False, ply + 1, True, move)
                if score > alpha: score = minimax(board, depth - 1, alpha, beta, False, ply + 1, True, move)
            else:
                score = minimax(board, depth - 1, alpha, alpha + 1, False, ply + 1, True, move)
                if score > alpha: score = minimax(board, depth - 1, alpha, beta, False, ply + 1, True, move)
            
            board.pop()
            if is_quiet: quiet_count += 1
            if score > best: best, best_move = score, move
            if score > alpha: alpha = score
            if beta <= alpha:
                if is_quiet: _update_killers(move, ply); _update_history(move, depth); _update_countermove(move, prev_move)
                break
        else:
            if move_idx == 0: score = minimax(board, depth - 1, alpha, beta, True, ply + 1, True, move)
            elif reduce > 0:
                score = minimax(board, depth - 1 - reduce, beta - 1, beta, True, ply + 1, True, move)
                if score < beta: score = minimax(board, depth - 1, alpha, beta, True, ply + 1, True, move)
            else:
                score = minimax(board, depth - 1, beta - 1, beta, True, ply + 1, True, move)
                if score < beta: score = minimax(board, depth - 1, alpha, beta, True, ply + 1, True, move)

            board.pop()
            if is_quiet: quiet_count += 1
            if score < best: best, best_move = score, move
            if score < beta: beta = score
            if beta <= alpha:
                if is_quiet: _update_killers(move, ply); _update_history(move, depth); _update_countermove(move, prev_move)
                break

    flag = TT_EXACT if orig_alpha < best < orig_beta else (TT_LOWER if best >= orig_beta else TT_UPPER) if maximizing else (TT_UPPER if best <= orig_alpha else TT_LOWER)
    _tt_store(h, depth, flag, best, best_move)
    return best

# ------------------------------------------------------------------------------
# Root search with iterative deepening and aspiration windows
# ------------------------------------------------------------------------------
def get_next_move(board: chess.Board, color: chess.Color, depth: int = 3) -> chess.Move:
    """Return the best move for `color` from the current board."""
    global _KILLERS, _HISTORY, _ROOT_COLOR
    _ROOT_COLOR = color

    _KILLERS = [[None, None] for _ in range(128)]
    for k in list(_HISTORY.keys()):
        _HISTORY[k] >>= 1
        if _HISTORY[k] == 0: del _HISTORY[k]

    maximizing = (color == chess.WHITE)
    best_move = None
    prev_score = 0.0
    b = board.copy()
    all_legal = list(b.legal_moves)

    # Strictly adhere to depth param to prevent timeout penalties
    for d in range(1, depth + 1):
        asp_lo, asp_hi, asp_delta = (-float('inf'), float('inf'), float('inf')) if d < 4 or best_move is None else (prev_score - 50, prev_score + 50, 50)
        
        retry_limit = 0
        while True:
            tt_root = _tt_probe(chess.polyglot.zobrist_hash(b))
            root_tt_move = tt_root[3] if (tt_root and tt_root[3] in all_legal) else None
            moves = _order_moves(b, all_legal, root_tt_move, ply=0)
            cur_best_move, cur_best_score = None, -float('inf') if maximizing else float('inf')
            a, bw = asp_lo, asp_hi

            for move in moves:
                b.push(move)
                score = minimax(b, d - 1, a, bw, not maximizing, ply=1, prev_move=move)
                b.pop()

                if maximizing:
                    if score > cur_best_score: cur_best_score, cur_best_move = score, move
                    a = max(a, score)
                    if bw <= a: cur_best_score, cur_best_move = score, move; break
                else:
                    if score < cur_best_score: cur_best_score, cur_best_move = score, move
                    bw = min(bw, score)
                    if bw <= a: cur_best_score, cur_best_move = score, move; break

            if asp_delta == float('inf') or retry_limit >= 6:
                if cur_best_move: best_move, prev_score = cur_best_move, cur_best_score
                break

            retry_limit += 1
            if maximizing:
                if cur_best_score <= asp_lo: asp_lo -= asp_delta; asp_delta *= 2
                elif cur_best_score >= asp_hi: asp_hi += asp_delta; asp_delta *= 2
                else: best_move, prev_score = cur_best_move, cur_best_score; break
            else:
                if cur_best_score >= asp_hi: asp_hi += asp_delta; asp_delta *= 2
                elif cur_best_score <= asp_lo: asp_lo -= asp_delta; asp_delta *= 2
                else: best_move, prev_score = cur_best_move, cur_best_score; break

        if best_move: _tt_store(chess.polyglot.zobrist_hash(board), d, TT_EXACT, prev_score, best_move)

    return best_move if best_move else all_legal[0]

# ------------------------------------------------------------------------------
# Quick self‑test
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    b = chess.Board()
    print(f"Starting eval: {evaluate(b):.1f} cp")
    for d in (3, 5):
        t0 = time.time()
        move = get_next_move(b, chess.WHITE, depth=d)
        print(f"depth={d:2d}: {b.san(move):6s}  ({time.time() - t0:.2f}s)")