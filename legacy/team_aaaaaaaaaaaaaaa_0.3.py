"""Stronger chess bot based on team_aaaaaaaaaaaaaaa.py.

This version focuses on three things:
- a cleaner, higher-signal evaluation function
- correct PST orientation and phase interpolation
- simpler but more reliable alpha-beta search with TT, move ordering, and quiescence
"""

from __future__ import annotations

import time
from math import inf

import chess
import chess.polyglot

# -----------------------------------------------------------------------------
# Piece values and piece-square tables
# -----------------------------------------------------------------------------
PIECE_VALUES = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330, chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}

PST_PAWN_MG = [0, 0, 0, 0, 0, 0, 0, 0, 98, 134, 61, 95, 68, 126, 34, -11, -6, 7, 26, 31, 62, 22, -8, -17, -14, 13, 6, 21, 23, 12, 17, -23, -27, -2, -5, 12, 17, 6, 10, -25, -26, -4, -4, -10, 3, 3, 33, -12, -35, -1, -20, -23, -15, 24, 38, -22, 0, 0, 0, 0, 0, 0, 0, 0]
PST_PAWN_EG = [0, 0, 0, 0, 0, 0, 0, 0, 178, 173, 158, 134, 147, 132, 165, 187, 94, 100, 85, 67, 56, 53, 82, 84, 32, 24, 13, 5, -2, 4, 17, 17, 13, 9, -3, -7, -7, -8, 3, -1, 4, 7, -6, 1, 0, -5, -1, -8, 13, 8, 8, 10, 13, 0, 2, -7, 0, 0, 0, 0, 0, 0, 0, 0]
PST_KNIGHT_MG = [-167, -89, -34, -49, 61, -97, -15, -107, -73, -41, 72, 36, 23, 62, 7, -17, -47, 60, 37, 65, 84, 129, 73, 44, -9, 17, 19, 53, 37, 69, 18, 22, -13, 4, 16, 13, 28, 19, 21, -8, -23, -9, 12, 10, 19, 17, 25, -16, -29, -53, -12, -3, -1, 18, -14, -19, -105, -21, -58, -33, -17, -28, -19, -23]
PST_KNIGHT_EG = [-58, -38, -13, -28, -31, -27, -63, -99, -25, -8, -25, -2, -9, -25, -24, -52, -24, -20, 10, 9, -1, -9, -19, -41, -17, 3, 22, 22, 22, 11, 8, -18, -18, -6, 16, 25, 16, 17, 4, -18, -23, -3, -1, 15, 10, -3, -20, -22, -42, -20, -10, -5, -2, -20, -23, -44, -29, -51, -23, -15, -22, -18, -50, -64]
PST_BISHOP_MG = [-29, 4, -82, -37, -25, -42, 7, -8, -26, 16, -18, -13, 30, 59, 18, -47, -16, 37, 43, 40, 35, 50, 37, -2, -4, 5, 19, 50, 37, 37, 7, -2, -6, 13, 13, 26, 34, 12, 10, 4, 0, 15, 15, 15, 14, 27, 18, 10, 4, 15, 16, 0, 7, 21, 33, 1, -33, -3, -14, -21, -13, -12, -39, -21]
PST_BISHOP_EG = [-14, -21, -11, -8, -7, -9, -17, -24, -8, -4, 7, -12, -3, -13, -4, -14, 2, -8, 0, -1, -2, 6, 0, 4, -3, 9, 12, 9, 14, 10, 3, 2, -6, 3, 13, 19, 7, 10, -3, -9, -12, -3, 8, 10, 13, 3, -7, -15, -14, -18, -7, -1, 4, -9, -15, -27, -23, -9, -23, -5, -9, -16, -5, -17]
PST_ROOK_MG = [32, 42, 32, 51, 63, 9, 31, 43, 27, 32, 58, 62, 80, 67, 26, 44, -5, 19, 26, 36, 17, 45, 61, 16, -24, -11, 7, 26, 24, 35, -8, -20, -36, -26, -12, -1, 9, -7, 6, -23, -45, -25, -16, -17, 3, 0, -5, -33, -44, -16, -20, -9, -1, 11, -6, -71, -19, -13, 1, 17, 16, 7, -37, -26]
PST_ROOK_EG = [13, 10, 18, 15, 12, 12, 8, 5, 11, 13, 13, 11, -3, 3, 8, 3, 7, 7, 7, 5, 4, -3, -5, -3, 4, 3, 13, 1, 2, 1, -1, 2, 3, 5, 8, 4, -5, -6, -8, -11, -4, 0, -5, -1, -7, -12, -8, -16, -6, -6, 0, 2, -9, -9, -11, -3, -9, 2, 3, -1, -5, -13, 4, -20]
PST_QUEEN_MG = [-28, 0, 29, 12, 59, 44, 43, 45, -24, -39, -5, 1, -16, 57, 28, 54, -13, -17, 7, 8, 29, 56, 47, 57, -27, -27, -16, -16, -1, 17, -2, 1, -9, -26, -9, -10, -2, -4, 3, -3, -14, 2, -11, -2, -5, 2, 14, 5, -35, -8, 11, 2, 8, 15, -3, 1, -1, -18, -9, 10, -15, -25, -31, -50]
PST_QUEEN_EG = [-9, 22, 22, 27, 27, 19, 10, 20, -17, 20, 32, 41, 58, 25, 30, 0, -20, 6, 9, 49, 47, 35, 19, 9, 3, 22, 24, 45, 57, 40, 57, 36, -18, 28, 19, 47, 31, 34, 12, 11, -16, -27, 15, 6, 9, 17, 10, 5, -22, -23, -30, -16, -16, -23, -36, -32, -33, -28, -22, -43, -5, -32, -20, -41]
PST_KING_MG = [-65, 23, 16, -15, -56, -34, 2, 13, 29, -1, -20, -7, -8, -4, -38, -29, -9, 24, 2, -16, -20, 6, 22, -22, -17, -20, -12, -27, -30, -25, -14, -36, -49, -1, -27, -39, -46, -44, -33, -51, -14, -14, -22, -46, -44, -30, -15, -27, 1, 7, -8, -64, -43, -16, 9, 8, -15, 36, 12, -54, 8, -28, 24, 14]
PST_KING_EG = [-74, -35, -18, -18, -11, 15, 4, -17, -12, 17, 14, 17, 17, 38, 23, 11, 10, 17, 23, 15, 20, 45, 44, 13, -8, 22, 24, 27, 26, 33, 26, 3, -18, -4, 21, 24, 27, 23, 9, -11, -19, -3, 11, 21, 23, 16, 7, -9, -27, -11, 4, 13, 14, 4, -5, -17, -53, -34, -21, -11, -28, -14, -24, -43]

PST_MG = {
    chess.PAWN: PST_PAWN_MG,
    chess.KNIGHT: PST_KNIGHT_MG,
    chess.BISHOP: PST_BISHOP_MG,
    chess.ROOK: PST_ROOK_MG,
    chess.QUEEN: PST_QUEEN_MG,
    chess.KING: PST_KING_MG,
}

PST_EG = {
    chess.PAWN: PST_PAWN_EG,
    chess.KNIGHT: PST_KNIGHT_EG,
    chess.BISHOP: PST_BISHOP_EG,
    chess.ROOK: PST_ROOK_EG,
    chess.QUEEN: PST_QUEEN_EG,
    chess.KING: PST_KING_EG,
}

# -----------------------------------------------------------------------------
# Search / engine constants
# -----------------------------------------------------------------------------
MATE_SCORE = 100_000
MAX_PLY = 128
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2

_TT = {}
_KILLERS = [[None, None] for _ in range(MAX_PLY)]
_HISTORY = {}

CENTER = chess.SquareSet([chess.D4, chess.E4, chess.D5, chess.E5])
EXT_CENTER = chess.SquareSet([
    chess.C3, chess.D3, chess.E3, chess.F3,
    chess.C4, chess.D4, chess.E4, chess.F4,
    chess.C5, chess.D5, chess.E5, chess.F5,
    chess.C6, chess.D6, chess.E6, chess.F6,
])

MOVE_WEIGHTS = {
    chess.PAWN: 0.20,
    chess.KNIGHT: 1.05,
    chess.BISHOP: 1.00,
    chess.ROOK: 0.80,
    chess.QUEEN: 0.55,
    chess.KING: 0.20,
}

CENTER_OCC_WEIGHTS = {
    chess.PAWN: 12,
    chess.KNIGHT: 16,
    chess.BISHOP: 10,
    chess.ROOK: 6,
    chess.QUEEN: 5,
    chess.KING: 0,
}

CENTER_ATTACK_WEIGHTS = {
    chess.PAWN: 4,
    chess.KNIGHT: 3,
    chess.BISHOP: 2,
    chess.ROOK: 1,
    chess.QUEEN: 1,
    chess.KING: 0,
}

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _to_move_score(board: chess.Board, white_score: float) -> float:
    return white_score if board.turn == chess.WHITE else -white_score


def _mirror_index(sq: int, color: chess.Color) -> int:
    return sq if color == chess.WHITE else chess.square_mirror(sq)


def _phase(board: chess.Board) -> float:
    """0.0 = opening/middlegame, 1.0 = endgame."""
    phase = 0
    for color in (chess.WHITE, chess.BLACK):
        phase += len(board.pieces(chess.KNIGHT, color))
        phase += len(board.pieces(chess.BISHOP, color))
        phase += 2 * len(board.pieces(chess.ROOK, color))
        phase += 4 * len(board.pieces(chess.QUEEN, color))
    return 1.0 - min(phase, 24) / 24.0


def _group_pieces(board: chess.Board):
    by_type = {
        chess.WHITE: {pt: [] for pt in range(1, 7)},
        chess.BLACK: {pt: [] for pt in range(1, 7)},
    }
    records = {chess.WHITE: [], chess.BLACK: []}
    occupied = {chess.WHITE: set(), chess.BLACK: set()}

    for sq, piece in board.piece_map().items():
        by_type[piece.color][piece.piece_type].append(sq)
        records[piece.color].append((sq, piece.piece_type, board.attacks(sq)))
        occupied[piece.color].add(sq)

    return by_type, records, occupied


def _piece_square_value(piece_type: int, sq: int, color: chess.Color, endgame: float) -> float:
    idx = _mirror_index(sq, color)
    mg = PST_MG[piece_type][idx]
    eg = PST_EG[piece_type][idx]
    return mg * (1.0 - endgame) + eg * endgame


def _terminal_white_score(board: chess.Board, ply: int = 0):
    if board.is_checkmate():
        return (-MATE_SCORE + ply) if board.turn == chess.WHITE else (MATE_SCORE - ply)

    outcome = board.outcome()
    if outcome is not None and outcome.winner is None:
        return 0.0

    if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3) or board.is_fifty_moves():
        return 0.0

    return None


def _is_draw(board: chess.Board) -> bool:
    outcome = board.outcome()
    if outcome is not None and outcome.winner is None:
        return True
    return (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_repetition(3)
        or board.is_fifty_moves()
    )

# -----------------------------------------------------------------------------
# Evaluation terms
# -----------------------------------------------------------------------------
def _material_pst(board, by_type, endgame):
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1.0 if color == chess.WHITE else -1.0
        for pt in range(1, 7):
            if pt == chess.KING:
                # King safety is handled separately; PST still adds positional value.
                for sq in by_type[color][pt]:
                    score += sign * _piece_square_value(pt, sq, color, endgame)
                continue
            value = PIECE_VALUES[pt]
            for sq in by_type[color][pt]:
                score += sign * (value + _piece_square_value(pt, sq, color, endgame))
    return score


def _pawn_structure(board, by_type):
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1.0 if color == chess.WHITE else -1.0
        pawns = by_type[color][chess.PAWN]
        enemy_pawns = by_type[not color][chess.PAWN]
        if not pawns:
            continue

        file_counts = [0] * 8
        for sq in pawns:
            file_counts[chess.square_file(sq)] += 1

        for sq in pawns:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)

            # doubled / isolated pawns
            if file_counts[f] > 1:
                score -= sign * 14
            if (f == 0 or file_counts[f - 1] == 0) and (f == 7 or file_counts[f + 1] == 0):
                score -= sign * 10

            # passed pawns
            blocked = False
            for esq in enemy_pawns:
                ef = chess.square_file(esq)
                er = chess.square_rank(esq)
                if abs(ef - f) <= 1:
                    if color == chess.WHITE and er > r:
                        blocked = True
                        break
                    if color == chess.BLACK and er < r:
                        blocked = True
                        break
            if not blocked:
                advancement = r if color == chess.WHITE else 7 - r
                bonus = 18 + advancement * 10
                # even better if the pawn is supported by another pawn.
                support_rank = r - 1 if color == chess.WHITE else r + 1
                supported = False
                if 0 <= support_rank <= 7:
                    for df in (-1, 1):
                        sf = f + df
                        if 0 <= sf <= 7:
                            support_sq = chess.square(sf, support_rank)
                            support_piece = board.piece_at(support_sq)
                            if support_piece and support_piece.piece_type == chess.PAWN and support_piece.color == color:
                                supported = True
                                break
                if supported:
                    bonus += 10
                score += sign * bonus

    return score


def _center_control(records):
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1.0 if color == chess.WHITE else -1.0
        for sq, pt, attacks in records[color]:
            if sq in CENTER:
                score += sign * CENTER_OCC_WEIGHTS[pt]
            elif sq in EXT_CENTER:
                score += sign * (CENTER_OCC_WEIGHTS[pt] * 0.5)
            score += sign * CENTER_ATTACK_WEIGHTS[pt] * len(attacks & CENTER)
    return score


def _mobility(records, occupied):
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1.0 if color == chess.WHITE else -1.0
        own_occ = occupied[color]
        for sq, pt, attacks in records[color]:
            legalish = 0
            for dst in attacks:
                if dst not in own_occ:
                    legalish += 1
            score += sign * MOVE_WEIGHTS[pt] * legalish
    return score * 1.5


def _bishop_pair(by_type):
    score = 0.0
    if len(by_type[chess.WHITE][chess.BISHOP]) >= 2:
        score += 24.0
    if len(by_type[chess.BLACK][chess.BISHOP]) >= 2:
        score -= 24.0
    return score


def _rook_and_queen_activity(board, by_type, endgame):
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1.0 if color == chess.WHITE else -1.0
        enemy_pawns = by_type[not color][chess.PAWN]
        own_pawns = by_type[color][chess.PAWN]

        # rooks on open / semi-open files and on the 7th rank
        for sq in by_type[color][chess.ROOK]:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            own_file_pawns = sum(1 for p in own_pawns if chess.square_file(p) == f)
            enemy_file_pawns = sum(1 for p in enemy_pawns if chess.square_file(p) == f)
            if own_file_pawns == 0 and enemy_file_pawns == 0:
                score += sign * 18
            elif own_file_pawns == 0:
                score += sign * 10
            if (color == chess.WHITE and r == 6) or (color == chess.BLACK and r == 1):
                score += sign * 16

        # queen activity near the enemy king is stronger in the middlegame.
        if endgame < 0.8:
            enemy_king = board.king(not color)
            if enemy_king is not None:
                ekf = chess.square_file(enemy_king)
                ekr = chess.square_rank(enemy_king)
                for sq in by_type[color][chess.QUEEN]:
                    qf = chess.square_file(sq)
                    qr = chess.square_rank(sq)
                    dist = abs(qf - ekf) + abs(qr - ekr)
                    score += sign * max(0, 7 - dist) * 2.0 * (1.0 - endgame)

    return score


def _knight_outposts(board, by_type):
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1.0 if color == chess.WHITE else -1.0
        enemy_pawns = by_type[not color][chess.PAWN]
        for sq in by_type[color][chess.KNIGHT]:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            if color == chess.WHITE and r < 3:
                continue
            if color == chess.BLACK and r > 4:
                continue

            # supported by own pawn from behind.
            back_rank = r - 1 if color == chess.WHITE else r + 1
            supported = False
            if 0 <= back_rank <= 7:
                for df in (-1, 1):
                    sf = f + df
                    if 0 <= sf <= 7:
                        p = board.piece_at(chess.square(sf, back_rank))
                        if p and p.piece_type == chess.PAWN and p.color == color:
                            supported = True
                            break
            if not supported:
                continue

            # not easily chased by enemy pawns.
            attack_rank = r + 1 if color == chess.WHITE else r - 1
            chased = False
            if 0 <= attack_rank <= 7:
                for ep in enemy_pawns:
                    if abs(chess.square_file(ep) - f) == 1 and chess.square_rank(ep) == attack_rank:
                        chased = True
                        break
            if not chased:
                score += sign * 18

    return score


def _king_safety(board, by_type, records, endgame):
    # King safety matters most before the endgame.
    if endgame > 0.88:
        return 0.0

    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1.0 if color == chess.WHITE else -1.0
        king_sq = board.king(color)
        if king_sq is None:
            continue

        kf = chess.square_file(king_sq)
        kr = chess.square_rank(king_sq)

        # Castling rights and king shelter.
        if board.has_castling_rights(color):
            score += sign * 10
        else:
            score -= sign * 8

        # Pawn shield in front of the king.
        shield_rank = kr + 1 if color == chess.WHITE else kr - 1
        if 0 <= shield_rank <= 7:
            for df in (-1, 0, 1):
                sf = kf + df
                if 0 <= sf <= 7:
                    p = board.piece_at(chess.square(sf, shield_rank))
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        score += sign * 8
                    else:
                        score -= sign * 5

        # Enemy pressure around the king ring.
        ring = []
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                nf, nr = kf + df, kr + dr
                if 0 <= nf <= 7 and 0 <= nr <= 7:
                    ring.append(chess.square(nf, nr))

        attackers = 0
        weighted_attack = 0.0
        for sq in ring:
            attackers += len(board.attackers(not color, sq))
            piece = board.piece_at(sq)
            if piece and piece.color == color and piece.piece_type != chess.KING:
                defenders = len(board.attackers(color, sq))
                enemy_attackers = len(board.attackers(not color, sq))
                if enemy_attackers > defenders:
                    weighted_attack += PIECE_VALUES[piece.piece_type] * 0.20

        score -= sign * attackers * 2.0 * (1.0 - endgame)
        score -= sign * weighted_attack * 0.8

    return score


def _hanging_pieces(board):
    score = 0.0
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING:
            continue
        attackers = len(board.attackers(not piece.color, sq))
        if attackers == 0:
            continue
        defenders = len(board.attackers(piece.color, sq))
        if attackers > defenders:
            score += (-1.0 if piece.color == chess.WHITE else 1.0) * PIECE_VALUES[piece.piece_type] * 0.20
    return score


def _passed_pawn_push(board, by_type, endgame):
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1.0 if color == chess.WHITE else -1.0
        enemy_pawns = by_type[not color][chess.PAWN]
        for sq in by_type[color][chess.PAWN]:
            f = chess.square_file(sq)
            r = chess.square_rank(sq)
            blocked = False
            for ep in enemy_pawns:
                ef = chess.square_file(ep)
                er = chess.square_rank(ep)
                if abs(ef - f) <= 1:
                    if color == chess.WHITE and er > r:
                        blocked = True
                        break
                    if color == chess.BLACK and er < r:
                        blocked = True
                        break
            if blocked:
                continue
            advance = r if color == chess.WHITE else 7 - r
            bonus = (20 + advance * 12) * (0.7 + 0.6 * endgame)
            # very strong on 6th/7th ranks.
            if advance >= 5:
                bonus += 18 + advance * 3
            score += sign * bonus
    return score


def _space_advantage(records, endgame):
    if endgame > 0.6:
        return 0.0
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1.0 if color == chess.WHITE else -1.0
        for _, pt, attacks in records[color]:
            score += sign * len(attacks & EXT_CENTER) * (0.8 if pt in (chess.KNIGHT, chess.BISHOP) else 0.5)
    return score * (1.0 - endgame)


def _king_activity_endgame(board, endgame):
    if endgame < 0.45:
        return 0.0
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1.0 if color == chess.WHITE else -1.0
        king_sq = board.king(color)
        if king_sq is None:
            continue
        kf = chess.square_file(king_sq)
        kr = chess.square_rank(king_sq)
        dist = abs(kf - 3.5) + abs(kr - 3.5)
        score += sign * (7.0 - dist) * 6.0 * endgame
    return score


def _tempo_and_checks(board):
    score = 6.0 if board.turn == chess.WHITE else -6.0
    if board.is_check():
        score += -12.0 if board.turn == chess.WHITE else 12.0
    return score

# -----------------------------------------------------------------------------
# Public evaluation function
# -----------------------------------------------------------------------------
def evaluate(board: chess.Board) -> float:
    terminal = _terminal_white_score(board)
    if terminal is not None:
        return terminal

    if _is_draw(board):
        return 0.0

    endgame = _phase(board)
    by_type, records, occupied = _group_pieces(board)

    score = 0.0
    score += _material_pst(board, by_type, endgame)
    score += _pawn_structure(board, by_type)
    score += _passed_pawn_push(board, by_type, endgame)
    score += _center_control(records)
    score += _mobility(records, occupied)
    score += _bishop_pair(by_type)
    score += _rook_and_queen_activity(board, by_type, endgame)
    score += _knight_outposts(board, by_type)
    score += _king_safety(board, by_type, records, endgame)
    score += _hanging_pieces(board)
    score += _space_advantage(records, endgame)
    score += _king_activity_endgame(board, endgame)
    score += _tempo_and_checks(board)

    return float(score)

# -----------------------------------------------------------------------------
# Move ordering / search helpers
# -----------------------------------------------------------------------------
def _capture_gain(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_at(move.to_square)
    if victim is None and board.is_en_passant(move):
        victim_value = PIECE_VALUES[chess.PAWN]
    else:
        victim_value = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0

    attacker = board.piece_at(move.from_square)
    attacker_value = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
    return 10 * victim_value - attacker_value


def _order_moves(board: chess.Board, moves, tt_move=None, ply: int = 0):
    killer_moves = _KILLERS[ply] if ply < MAX_PLY else [None, None]
    killers = {m for m in killer_moves if m is not None}
    tt = tt_move

    def score(move: chess.Move):
        if tt is not None and move == tt:
            return 1_000_000
        promo_bonus = PIECE_VALUES.get(move.promotion, 0) if move.promotion is not None else 0

        if board.is_capture(move):
            return 500_000 + _capture_gain(board, move) + promo_bonus
        if board.gives_check(move):
            return 250_000 + promo_bonus
        if move in killers:
            return 150_000 + promo_bonus
        return _HISTORY.get((move.from_square, move.to_square), 0) + promo_bonus

    return sorted(moves, key=score, reverse=True)


def _qsearch(board: chess.Board, alpha: float, beta: float, ply: int = 0) -> float:
    terminal = _terminal_white_score(board, ply)
    if terminal is not None:
        return _to_move_score(board, terminal)

    stand_pat = _to_move_score(board, evaluate(board))
    if stand_pat >= beta:
        return stand_pat
    if stand_pat > alpha:
        alpha = stand_pat

    tactical = []
    for move in board.legal_moves:
        if board.is_capture(move) or move.promotion is not None or board.gives_check(move):
            tactical.append(move)

    if not tactical:
        return stand_pat

    best = stand_pat
    for move in _order_moves(board, tactical, ply=ply):
        board.push(move)
        score = -_qsearch(board, -beta, -alpha, ply + 1)
        board.pop()

        if score > best:
            best = score
        if best >= beta:
            return best
        if best > alpha:
            alpha = best

    return best


def _tt_probe(key, depth, alpha, beta):
    entry = _TT.get(key)
    if entry is None:
        return None

    entry_depth, flag, value, move = entry
    if entry_depth < depth:
        return move

    if flag == TT_EXACT:
        return value, move, True
    if flag == TT_LOWER and value >= beta:
        return value, move, True
    if flag == TT_UPPER and value <= alpha:
        return value, move, True

    return move


def _tt_store(key, depth, flag, value, move):
    _TT[key] = (depth, flag, value, move)


def _update_killers(move, ply):
    if ply >= MAX_PLY:
        return
    slot = _KILLERS[ply]
    if slot[0] == move:
        return
    slot[1] = slot[0]
    slot[0] = move


def _update_history(move, depth):
    key = (move.from_square, move.to_square)
    _HISTORY[key] = _HISTORY.get(key, 0) + depth * depth


def _search(board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0):
    terminal = _terminal_white_score(board, ply)
    if terminal is not None:
        return _to_move_score(board, terminal), None

    if depth <= 0:
        return _qsearch(board, alpha, beta, ply), None

    key = chess.polyglot.zobrist_hash(board)
    tt_result = _tt_probe(key, depth, alpha, beta)
    tt_move = None
    if isinstance(tt_result, tuple):
        value, move, exact = tt_result
        if exact:
            return value, move
    elif tt_result is not None:
        tt_move = tt_result

    in_check = board.is_check()
    if in_check:
        depth += 1

    moves = list(board.legal_moves)
    if not moves:
        # should only happen in terminal states, but keep it safe.
        return _qsearch(board, alpha, beta, ply), None

    moves = _order_moves(board, moves, tt_move=tt_move, ply=ply)

    original_alpha = alpha
    best_score = -inf
    best_move = moves[0]

    for idx, move in enumerate(moves):
        is_quiet = (not board.is_capture(move) and move.promotion is None and not board.gives_check(move))

        board.push(move)
        score, _ = _search(board, depth - 1, -beta, -alpha, ply + 1)
        score = -score
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move
        if score > alpha:
            alpha = score

        if alpha >= beta:
            if is_quiet:
                _update_killers(move, ply)
                _update_history(move, depth)
            break

        # Light late-move reduction for quiet late moves.
        if idx >= 4 and is_quiet and depth >= 3 and not in_check:
            reduced_depth = max(0, depth - 2)
            board.push(move)
            reduced_score, _ = _search(board, reduced_depth - 1, -alpha - 1, -alpha, ply + 1)
            reduced_score = -reduced_score
            board.pop()
            if reduced_score > alpha:
                board.push(move)
                full_score, _ = _search(board, depth - 1, -beta, -alpha, ply + 1)
                full_score = -full_score
                board.pop()
                if full_score > best_score:
                    best_score = full_score
                    best_move = move
                if full_score > alpha:
                    alpha = full_score
                if alpha >= beta:
                    if is_quiet:
                        _update_killers(move, ply)
                        _update_history(move, depth)
                    break

    flag = TT_EXACT
    if best_score <= original_alpha:
        flag = TT_UPPER
    elif best_score >= beta:
        flag = TT_LOWER

    _tt_store(key, depth, flag, best_score, best_move)
    return best_score, best_move


# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------
def get_next_move(board: chess.Board, color: chess.Color, depth: int = 3) -> chess.Move:
    """Return the best move for `color` from the current board position."""
    if board.turn != color:
        # Tournament harness should always keep these aligned, but never return
        # an illegal move if the caller passes a mismatched colour.
        color = board.turn

    legal_moves = list(board.legal_moves)
    if not legal_moves:
        raise RuntimeError("No legal moves available")

    # Small history decay so the ordering signal stays fresh.
    for key in list(_HISTORY.keys()):
        _HISTORY[key] >>= 1
        if _HISTORY[key] == 0:
            del _HISTORY[key]

    working = board.copy()
    best_move = legal_moves[0]
    best_score = -inf

    # Iterative deepening with aspiration windows.
    guess = 0.0
    for d in range(1, max(1, depth) + 1):
        window = 60.0 if d >= 3 else inf
        alpha, beta = (-inf, inf) if window == inf else (guess - window, guess + window)

        while True:
            score, move = _search(working, d, alpha, beta, 0)
            if window != inf and score <= alpha:
                alpha -= window
                window *= 2
                continue
            if window != inf and score >= beta:
                beta += window
                window *= 2
                continue

            guess = score
            if move is not None:
                best_move, best_score = move, score
            break

    return best_move


# -----------------------------------------------------------------------------
# Quick self-test
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    board = chess.Board()
    t0 = time.time()
    move = get_next_move(board, chess.WHITE, depth=3)
    elapsed = time.time() - t0
    print(f"Opening move: {board.san(move)} ({elapsed:.2f}s)")
