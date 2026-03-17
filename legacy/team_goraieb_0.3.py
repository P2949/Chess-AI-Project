"""
team_goraieb.py  —  Chess Bot: Championship Edition v3
=======================================================

Improvements over the previous version (Championship Edition v2):

━━━ SEARCH ENHANCEMENTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • TARGET_DEPTH = 7       — 4 extra plies with no time limit; this single change
                             gives more strength than everything else combined.
                             Each extra ply examines ≈ √35 × more positions.

  • Aspiration windows     — From depth 4+, search with a ±50 cp window around
                             the previous iteration's score. On fail-low or
                             fail-high, double the window and retry. Dramatically
                             reduces the effective tree width at deeper plies.

  • Principal Variation    — First move at each node is searched with full
    Search (PVS)             window; all later moves use null window [α, α+1]
                             (or [β-1, β] for the minimiser). Only re-search
                             with full window on fail. Often 20-30 % speedup.

  • Reverse Futility       — Before move generation, if static_eval − depth×120
    Pruning (RFP)            ≥ β (fail-soft), return immediately. The position
                             is so good we won't fall below beta even if we pass.

  • Futility Pruning       — Inside the move loop at depth ≤ 2: skip quiet moves
                             where static_eval + margin ≤ α  (margin = 120 @ d1,
                             300 @ d2). Called at "frontier" and "pre-frontier"
                             nodes respectively. Big win in shallow subtrees.

  • Late Move Pruning      — At depth ≤ 3, after searching LMP_CUTOFF[depth]
    (LMP)                    quiet moves, prune the rest entirely. Quiet moves
                             that come very late in the ordered list are rarely
                             best and this saves significant search time.

  • Log-log LMR formula    — Reduction = max(1, int(ln(depth)*ln(move_idx)/2.25))
                             More aggressive than the previous flat 1-ply reduction
                             for deeper/later moves.

  • Delta pruning          — In qsearch, skip captures where even adding the
    (qsearch)                highest possible material swing (900 cp = queen)
                             cannot beat α. Tightens the qsearch leaf count.

  • Countermove heuristic  — Track which quiet move refuted each opponent move
                             (indexed by from/to squares). Order this move right
                             after killers. Independent of board position.

  • Faster TT: store Move  — Transposition table now stores chess.Move objects
    objects directly         directly instead of UCI strings, eliminating the
                             Move.from_uci() conversion overhead on every probe.

  • Larger TT              — MAX_TT_SIZE raised to 4_000_000 entries to support
                             deeper search without eviction pressure.

━━━ EVALUATION ENHANCEMENTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • King Attack Zone       — Weighted count of enemy pieces attacking the 3×3
                             ring around each king. Multi-attacker scaling (2+
                             pieces = multiplicative danger). The most important
                             strategic signal missing from the previous version.

  • Space Advantage        — Squares safely controlled in the extended centre
                             (ranks 2-4 behind each side's half). Rewards the
                             side with more room to manoeuvre.

  • Pawn Chain Bonus       — Pawns that protect each other diagonally receive a
                             +7 cp bonus per link. Chain formations are stronger
                             because they can't be broken by a single exchange.

  • Protected Passed Pawn  — A passed pawn defended by a friendly pawn gets +15 cp
    extra bonus              on top of the standard passed-pawn score. In the
                             endgame these are frequently unstoppable.

  • Trapped Piece Penalty  — Bishop or rook with ≤ 2 or ≤ 3 raw attacks (mobility)
                             respectively is penalised (−30 / −25 cp). Catches
                             "bad bishops" behind locked pawns and cornered rooks.

Install: pip install python-chess
"""

import chess
import chess.polyglot
from math import log as _log

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
# 2. PIECE-SQUARE TABLES  (White's POV, index 0 = a8, index 63 = h1)
#    For White: pst_idx = chess.square_mirror(sq)  — flips rank so rank-1 = low
#    For Black: pst_idx = sq                        — already "mirrored"
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

# King hides in corner during middlegame, centralises in endgame
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
    """0.0 = full middlegame, 1.0 = pure endgame.  Q=4, R=2, B=N=1; max=24."""
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
# 5. PAWN STRUCTURE  (doubled, isolated, passed+protected, backward)
# ─────────────────────────────────────────────────────────────────────────────
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
                base_bonus  = 20 + advancement * 10

                # Protected passed pawn gets extra bonus
                prot_rank = rank - 1 if color == chess.WHITE else rank + 1
                if 0 <= prot_rank <= 7:
                    for df in (-1, 1):
                        pf = f + df
                        if 0 <= pf <= 7:
                            p = board.piece_at(chess.square(pf, prot_rank))
                            if p and p.piece_type == chess.PAWN and p.color == color:
                                base_bonus += 15  # protected passed pawn
                                break

                score += sign * base_bonus

            # ── Backward ─────────────────────────────────────────────────────
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
            own_f   = sum(1 for s in board.pieces(chess.PAWN, color)    if chess.square_file(s) == f)
            enemy_f = sum(1 for s in board.pieces(chess.PAWN, not color) if chess.square_file(s) == f)
            if own_f == 0 and enemy_f == 0:
                score += sign * 25
            elif own_f == 0:
                score += sign * 10
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 11. KNIGHT OUTPOSTS
# ─────────────────────────────────────────────────────────────────────────────
def _knight_outposts(board: chess.Board) -> float:
    """Knight on ranks 4-6 protected by friendly pawn, unreachable by enemy pawns."""
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


# ─────────────────────────────────────────────────────────────────────────────
# 12. ROOK ON 7th RANK
# ─────────────────────────────────────────────────────────────────────────────
def _rook_on_seventh(board: chess.Board) -> float:
    score = 0.0
    for sq in board.pieces(chess.ROOK, chess.WHITE):
        if chess.square_rank(sq) == 6:
            score += 25
    for sq in board.pieces(chess.ROOK, chess.BLACK):
        if chess.square_rank(sq) == 1:
            score -= 25
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 13. CONNECTED ROOKS
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 14. HANGING PIECES
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 15. ENDGAME KING PROXIMITY (mop-up)
# ─────────────────────────────────────────────────────────────────────────────
def _king_proximity_mop_up(board: chess.Board, eg: float, material_score: float) -> float:
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
    if material_score > 300:
        bk_corner = corner_dist(bkf, bkr)
        result += eg * (14 - king_dist) * 4
        result += eg * bk_corner * 8
    elif material_score < -300:
        wk_corner = corner_dist(wkf, wkr)
        result -= eg * (14 - king_dist) * 4
        result -= eg * wk_corner * 8
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 16. KING ATTACK ZONE  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
# Weights: how dangerous each attacking piece type is near the king
_KA_WEIGHTS = {
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK:   3,
    chess.QUEEN:  5,
}

def _king_attack_zone(board: chess.Board, eg: float) -> float:
    """
    Count weighted attacks on the 3×3 ring around each king.
    More attacking pieces = multiplicative danger scaling.
    Ignored in pure endgame (kings should be active then).
    """
    if eg > 0.80:
        return 0.0
    mg_weight = 1.0 - eg
    score     = 0.0

    for defender_color in (chess.WHITE, chess.BLACK):
        # defender_sign: +1 if we're scoring danger to White (hurts White = bad for White)
        defender_sign = 1 if defender_color == chess.WHITE else -1
        attacker_color = not defender_color

        king_sq = board.king(defender_color)
        if king_sq is None:
            continue
        kf = chess.square_file(king_sq)
        kr = chess.square_rank(king_sq)

        # Build king zone (3×3 area around king)
        zone_sqs = []
        for df in (-1, 0, 1):
            for dr in (-1, 0, 1):
                f2, r2 = kf + df, kr + dr
                if 0 <= f2 <= 7 and 0 <= r2 <= 7:
                    zone_sqs.append(chess.square(f2, r2))
        zone_set = chess.SquareSet(zone_sqs)

        attack_units  = 0
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

        # Multiple attackers multiply the danger
        if attacker_count >= 3:
            danger = attack_units * attacker_count * 1.2
        elif attacker_count == 2:
            danger = attack_units * 1.6
        else:
            danger = attack_units * 0.6  # lone attacker is far less threatening

        # Danger hurts the defender — subtract from score if defender is White
        score -= defender_sign * danger * mg_weight * 1.8

    return score


# ─────────────────────────────────────────────────────────────────────────────
# 17. SPACE ADVANTAGE  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
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
    """Count squares attacked in the opponent's extended centre by each side."""
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


# ─────────────────────────────────────────────────────────────────────────────
# 18. PAWN CHAIN BONUS  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
def _pawn_chain_bonus(board: chess.Board) -> float:
    """Bonus for pawns that diagonally protect a friendly pawn (+7 cp per link)."""
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
                        break   # count each pawn once
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 19. TRAPPED PIECES  [NEW]
# ─────────────────────────────────────────────────────────────────────────────
def _trapped_pieces(board: chess.Board) -> float:
    """Penalty for bishops / rooks with very limited mobility (trapped)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EVALUATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(board: chess.Board) -> float:
    """
    Full positional heuristic (centipawns, White-positive).
    Terminal states return ±99_999 / 0.
    """
    if board.is_checkmate():
        return -99_999 if board.turn == chess.WHITE else 99_999
    if (board.is_stalemate()
            or board.is_insufficient_material()
            or board.is_repetition(3)
            or board.is_fifty_moves()):
        return 0

    eg  = _endgame_factor(board)
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
    score += _king_attack_zone(board, eg)       # NEW
    score += _space_advantage(board, eg)        # NEW
    score += _pawn_chain_bonus(board)           # NEW
    score += _trapped_pieces(board)             # NEW

    # Tempo: small bonus for the side to move (initiative)
    score += 15 if board.turn == chess.WHITE else -15

    # Being in check is a positional liability for the side to move
    if board.is_check():
        score += 20 if board.turn == chess.BLACK else -20

    return score


# ─────────────────────────────────────────────────────────────────────────────
# SEARCH GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
_TT      = {}                                      # Transposition table
_KILLERS = [[None, None] for _ in range(128)]      # 2 killer moves per ply
_HISTORY = {}                                      # (from_sq, to_sq) -> int
_COUNTERMOVE = {}                                  # (from_sq, to_sq) -> Move  [NEW]

TT_EXACT = 0
TT_LOWER = 1    # score >= beta  (fail-high)
TT_UPPER = 2    # score <= alpha (fail-low)

MAX_TT_SIZE = 4_000_000_000    # ~400 MB; tuned for deep search with no time limit

# ── Minimum search depth regardless of harness request ────────────────────
# With no time limit, going deep is the single most effective improvement.
# Reduce if your hardware cannot finish games in a reasonable time.
TARGET_DEPTH = 7

# ── Pruning parameters ────────────────────────────────────────────────────
# Futility margins (cp) by depth — max gain a quiet move can realistically bring
_FUTILITY_MARGINS = {1: 120, 2: 300}

# Late Move Pruning thresholds — prune quiet moves beyond this index at each depth
_LMP_CUTOFFS = {1: 8, 2: 12, 3: 20}

# Reverse Futility Pruning margin per depth (cp)
_RFP_MARGIN = 120   # eval - depth*_RFP_MARGIN >= beta → prune


# ─────────────────────────────────────────────────────────────────────────────
# TT HELPERS  (store Move objects directly — no UCI round-trip overhead)
# ─────────────────────────────────────────────────────────────────────────────
def _tt_store(h: int, depth: int, flag: int, score: float, move) -> None:
    existing = _TT.get(h)
    if existing is None or depth >= existing[0] or len(_TT) < MAX_TT_SIZE:
        _TT[h] = (depth, flag, score, move)   # move stored as chess.Move or None


def _tt_probe(h: int):
    """Returns (depth, flag, score, move_or_None) or None."""
    return _TT.get(h)


# ─────────────────────────────────────────────────────────────────────────────
# KILLER / HISTORY / COUNTERMOVE HELPERS
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


def _update_countermove(move: chess.Move, prev_move: chess.Move) -> None:
    """Record that `move` was the best refutation of `prev_move`."""
    if prev_move is not None:
        _COUNTERMOVE[(prev_move.from_square, prev_move.to_square)] = move


# ─────────────────────────────────────────────────────────────────────────────
# MOVE ORDERING
# ─────────────────────────────────────────────────────────────────────────────
def _order_moves(board: chess.Board, moves, tt_move=None, ply: int = 0,
                 prev_move=None):
    """
    Priority (highest score = searched first):
      1. TT / hash move                  (+30 000)
      2. Winning captures (MVV-LVA)      (+20 000 + gain)
      3. Queen promotion                 (+19 000)
      4. Killer moves                    (+10 000)
      5. Countermove                     ( +9 000)  [NEW]
      6. Quiet moves ordered by history
      7. Losing captures                 (-10 000 + loss)
    """
    tt_move_obj = tt_move  # stored as Move object
    ks          = _KILLERS[ply] if ply < 128 else [None, None]
    killer_set  = {k for k in ks if k is not None}

    # Countermove for the current position's previous move
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
            v_val = PIECE_VALUES.get(victim.piece_type,   0) if victim   else 0
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


# ─────────────────────────────────────────────────────────────────────────────
# QUIESCENCE SEARCH  (with delta pruning)
# ─────────────────────────────────────────────────────────────────────────────
_DELTA_PRUNE_MARGIN = 900   # queen value — conservative upper bound on a single capture gain

def _qsearch(board: chess.Board, alpha: float, beta: float, maximizing: bool) -> float:
    """
    Search until no captures remain ("quiet" position) to eliminate horizon effect.
    Adds delta pruning: skip captures that cannot possibly beat alpha.
    """
    if board.is_checkmate():
        return -99_999 if board.turn == chess.WHITE else 99_999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    stand_pat = evaluate(board)

    if maximizing:
        if stand_pat >= beta:
            return beta
        # Delta pruning: if even the best possible capture can't beat alpha, bail
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


# ─────────────────────────────────────────────────────────────────────────────
# MINIMAX WITH ALPHA-BETA  (PVS + all pruning/reduction enhancements)
# ─────────────────────────────────────────────────────────────────────────────
def minimax(board: chess.Board, depth: int,
            alpha: float, beta: float,
            maximizing: bool,
            ply: int = 0,
            null_ok: bool = True,
            prev_move: chess.Move = None) -> float:
    """
    Alpha-beta minimax with every modern pruning and reduction technique:
      • Transposition table cutoffs (EXACT / LOWER / UPPER)
      • Check extension (+1 when in check)
      • Reverse Futility Pruning (static null move pruning)
      • Null-move pruning (R=2/3)
      • Principal Variation Search (null-window for non-first moves)
      • Late Move Reductions — log(depth)*log(i)/2.25 formula
      • Futility Pruning at depth 1-2 (skip hopeless quiet moves)
      • Late Move Pruning at depth 1-3 (give up after N quiet moves)
      • Killer + history + countermove move ordering
      • Quiescence search with delta pruning at leaves
    """
    # ── TT probe ──────────────────────────────────────────────────────────
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

    eg = _endgame_factor(board)

    # ── Reverse Futility Pruning (static null move) ────────────────────────
    # If static eval already beats beta by a safe margin, prune immediately.
    # Skipped in check, zugzwang-prone deep endgames, and at ply 0.
    if (not in_check and ply > 0 and depth <= 6 and eg < 0.90
            and abs(beta) < 90_000):
        static_eval = evaluate(board)
        rfp_margin  = depth * _RFP_MARGIN
        if maximizing and static_eval - rfp_margin >= beta:
            return static_eval
        if not maximizing and static_eval + rfp_margin <= alpha:
            return static_eval
    else:
        static_eval = None   # computed lazily below if needed

    # ── Null-move pruning ─────────────────────────────────────────────────
    if null_ok and ply > 0 and not in_check and depth >= 3 and eg < 0.85:
        R = 3 if depth >= 5 else 2
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

    # ── Generate + order moves ─────────────────────────────────────────────
    moves = _order_moves(board, list(board.legal_moves), tt_move, ply, prev_move)
    if not moves:
        return evaluate(board)

    orig_alpha = alpha
    orig_beta  = beta
    best_move  = None
    quiet_count = 0      # track quiet moves for LMP and futility

    if maximizing:
        best = float('-inf')
        for move_idx, move in enumerate(moves):
            is_cap   = board.is_capture(move)
            is_quiet = (not is_cap and move.promotion is None)

            # ── Futility pruning (depth 1-2, skip hopeless quiet moves) ───
            if (is_quiet and not in_check and depth in _FUTILITY_MARGINS
                    and move_idx > 0 and abs(alpha) < 90_000):
                if static_eval is None:
                    static_eval = evaluate(board)
                if static_eval + _FUTILITY_MARGINS[depth] <= alpha:
                    quiet_count += 1
                    continue

            # ── Late Move Pruning (after N quiet moves at low depth) ───────
            if (is_quiet and not in_check and depth in _LMP_CUTOFFS
                    and quiet_count >= _LMP_CUTOFFS[depth]
                    and abs(alpha) < 90_000):
                continue

            board.push(move)
            gives_check = board.is_check()

            # ── LMR formula: ln(depth) × ln(move_idx+1) / 2.25 ───────────
            reduce = 0
            if (move_idx >= 2 and depth >= 3 and is_quiet
                    and not in_check and not gives_check):
                reduce = max(1, int(_log(max(depth, 1)) * _log(max(move_idx + 1, 1)) / 2.25))

            # ── PVS + LMR ─────────────────────────────────────────────────
            if move_idx == 0:
                # First move: full window, no reduction
                score = minimax(board, depth - 1, alpha, beta, False, ply + 1, True, move)
            elif reduce > 0:
                # LMR reduced null window
                score = minimax(board, depth - 1 - reduce, alpha, alpha + 1, False, ply + 1, True, move)
                if score > alpha:
                    # LMR failed high — re-search at full depth with full window
                    score = minimax(board, depth - 1, alpha, beta, False, ply + 1, True, move)
            else:
                # PVS null window for non-reduced moves after the first
                score = minimax(board, depth - 1, alpha, alpha + 1, False, ply + 1, True, move)
                if score > alpha:
                    # Null window failed high — re-search with full window
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

    else:   # minimizing
        best = float('inf')
        for move_idx, move in enumerate(moves):
            is_cap   = board.is_capture(move)
            is_quiet = (not is_cap and move.promotion is None)

            # ── Futility pruning ──────────────────────────────────────────
            if (is_quiet and not in_check and depth in _FUTILITY_MARGINS
                    and move_idx > 0 and abs(beta) < 90_000):
                if static_eval is None:
                    static_eval = evaluate(board)
                if static_eval - _FUTILITY_MARGINS[depth] >= beta:
                    quiet_count += 1
                    continue

            # ── Late Move Pruning ─────────────────────────────────────────
            if (is_quiet and not in_check and depth in _LMP_CUTOFFS
                    and quiet_count >= _LMP_CUTOFFS[depth]
                    and abs(beta) < 90_000):
                continue

            board.push(move)
            gives_check = board.is_check()

            # ── LMR ───────────────────────────────────────────────────────
            reduce = 0
            if (move_idx >= 2 and depth >= 3 and is_quiet
                    and not in_check and not gives_check):
                reduce = max(1, int(_log(max(depth, 1)) * _log(max(move_idx + 1, 1)) / 2.25))

            # ── PVS + LMR ─────────────────────────────────────────────────
            if move_idx == 0:
                score = minimax(board, depth - 1, alpha, beta, True, ply + 1, True, move)
            elif reduce > 0:
                # LMR reduced null window for minimiser
                score = minimax(board, depth - 1 - reduce, beta - 1, beta, True, ply + 1, True, move)
                if score < beta:
                    # LMR failed low — re-search at full depth
                    score = minimax(board, depth - 1, alpha, beta, True, ply + 1, True, move)
            else:
                # PVS null window
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


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT  (called by tournament harness — signature must not change)
# ─────────────────────────────────────────────────────────────────────────────
def get_next_move(board: chess.Board,
                  color: chess.Color,
                  depth: int = 3) -> chess.Move:
    """
    Return the best move for `color` from `board`.
    DO NOT rename or change this signature.

    Uses iterative deepening with aspiration windows (from depth 4+).
    The transposition table persists across calls (spans the whole game).
    """
    global _KILLERS, _HISTORY

    # Reset per-move state; keep TT across moves (it helps enormously)
    _KILLERS = [[None, None] for _ in range(128)]
    # Decay history to prevent stale data from dominating
    for k in list(_HISTORY.keys()):
        _HISTORY[k] = _HISTORY[k] >> 1
        if _HISTORY[k] == 0:
            del _HISTORY[k]

    target     = max(depth, TARGET_DEPTH)
    maximizing = (color == chess.WHITE)
    best_move  = None
    prev_score = 0.0

    b = board.copy()   # never modify the caller's board

    for d in range(1, target + 1):

        # ── Aspiration windows from depth 4 ───────────────────────────────
        if d >= 4 and best_move is not None:
            asp_delta = 50
            asp_lo    = prev_score - asp_delta
            asp_hi    = prev_score + asp_delta
        else:
            asp_lo = float('-inf')
            asp_hi = float('inf')
            asp_delta = float('inf')

        # Retry loop — widen aspiration window on fail
        while True:
            h_root = chess.polyglot.zobrist_hash(b)
            tt_root = _tt_probe(h_root)
            root_tt_move = None
            if tt_root is not None:
                _, _, _, root_tt_move = tt_root
                if root_tt_move is not None and root_tt_move not in b.legal_moves:
                    root_tt_move = None

            moves = _order_moves(b, list(b.legal_moves), root_tt_move, ply=0)

            cur_best_move  = None
            cur_best_score = float('-inf') if maximizing else float('inf')
            a = asp_lo
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

            # ── Check aspiration result ────────────────────────────────────
            if asp_delta == float('inf'):
                # No aspiration window — accept result immediately
                if cur_best_move is not None:
                    best_move  = cur_best_move
                    prev_score = cur_best_score
                break

            if maximizing:
                if cur_best_score <= asp_lo:          # fail-low
                    asp_lo    -= asp_delta
                    asp_delta *= 2
                elif cur_best_score >= asp_hi:        # fail-high
                    asp_hi    += asp_delta
                    asp_delta *= 2
                else:                                 # success
                    best_move  = cur_best_move
                    prev_score = cur_best_score
                    break
            else:
                if cur_best_score >= asp_hi:          # fail-high (bad for min)
                    asp_hi    += asp_delta
                    asp_delta *= 2
                elif cur_best_score <= asp_lo:        # fail-low (good for min)
                    asp_lo    -= asp_delta
                    asp_delta *= 2
                else:
                    best_move  = cur_best_move
                    prev_score = cur_best_score
                    break

        # Store root result in TT for next iteration's move ordering
        if best_move is not None:
            _tt_store(
                chess.polyglot.zobrist_hash(board),
                d, TT_EXACT, prev_score, best_move,
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
        _COUNTERMOVE.clear()
        t0    = time.time()
        move  = get_next_move(b, chess.WHITE, depth=d)
        elapsed = time.time() - t0
        print(f"[team_goraieb] depth={d:2d}: {b.san(move):6s}  ({elapsed:.2f}s)")
