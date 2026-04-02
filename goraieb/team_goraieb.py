"""
team_goraieb.py — Hybrid NN + tunable HCE with enhanced evaluation

Merges team_astar's NN evaluation with team_goraieb's tunable WEIGHTS dict.
All eval terms are in WEIGHTS for optimizer compatibility.

Enhanced eval terms beyond original:
  • King safety: pawn shield around castled king
  • Knight outpost: knight on rank 4-5, pawn-defended, no enemy pawn attackers
  • Connected rooks: both rooks on same file/rank bonus
  • King tropism: pieces near enemy king get bonus
  • Tempo: small bonus for side to move
  • Backward pawn: pawn that can't advance without being captured

Search: TT, quiescence, killer moves, history heuristic, MVV-LVA ordering
Eval:   PeSTO tapered + structure + NN correction (tunable blend)
"""

import chess
import chess.polyglot
import torch
import numpy as np
from pathlib import Path

# ── Cython acceleration ──────────────────────────────────────────────────────
try:
    from fast_chess import board_to_vector_bitboard as board_to_vector
    _HAVE_CYTHON = True
except ImportError:
    _HAVE_CYTHON = False
    PIECE_ORDER = [
        (chess.PAWN,   chess.WHITE), (chess.KNIGHT, chess.WHITE),
        (chess.BISHOP, chess.WHITE), (chess.ROOK,   chess.WHITE),
        (chess.QUEEN,  chess.WHITE), (chess.KING,   chess.WHITE),
        (chess.PAWN,   chess.BLACK), (chess.KNIGHT, chess.BLACK),
        (chess.BISHOP, chess.BLACK), (chess.ROOK,   chess.BLACK),
        (chess.QUEEN,  chess.BLACK), (chess.KING,   chess.BLACK),
    ]
    def board_to_vector(board):
        v = np.zeros(773, dtype=np.float32)
        for plane, (pt, color) in enumerate(PIECE_ORDER):
            for sq in board.pieces(pt, color):
                v[plane * 64 + sq] = 1.0
        v[768] = float(board.has_kingside_castling_rights(chess.WHITE))
        v[769] = float(board.has_queenside_castling_rights(chess.WHITE))
        v[770] = float(board.has_kingside_castling_rights(chess.BLACK))
        v[771] = float(board.has_queenside_castling_rights(chess.BLACK))
        v[772] = float(board.turn == chess.WHITE)
        return v

# ── NN model (imported inline to avoid issues when optimizer reloads module) ─
import torch.nn as nn

class ChessEvaluator(nn.Module):
    def __init__(self, input_dim=773):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(1024, 512),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256),        nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128),        nn.ReLU(),
            nn.Linear(128, 1),          nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)

SCORE_CLAMP = 1500

# ── Load model ────────────────────────────────────────────────────────────────
_device = torch.device("cpu")
_model = ChessEvaluator()
_model_path = Path(__file__).resolve().parent / "model.pt"
if _model_path.exists():
    _model.load_state_dict(torch.load(str(_model_path), map_location=_device))
    _model.eval()
    _HAS_NN = True
else:
    _HAS_NN = False

_nn_input = torch.zeros(1, 773, dtype=torch.float32)

# Warm up
if _HAS_NN:
    with torch.no_grad():
        _model(_nn_input)


# ══════════════════════════════════════════════════════════════════════════════
#  TUNABLE WEIGHTS — optimizer patches this dict directly
# ══════════════════════════════════════════════════════════════════════════════

WEIGHTS = {
    # ── Piece values ──────────────────────────────────────────────────────
    "pawn":             100,
    "knight":           320,
    "bishop":           330,
    "rook":             500,
    "queen":            900,
    "king":           20000,

    # ── Eval terms ────────────────────────────────────────────────────────
    "mobility":          1.5,     # per legal-move delta
    "bishop_pair":      30.0,     # bonus for 2+ bishops
    "doubled_penalty":  20.0,     # penalty per doubled pawn
    "isolated_penalty": 15.0,     # penalty per isolated pawn
    "backward_penalty": 10.0,     # penalty per backward pawn
    "rook_open_file":   25.0,     # rook on fully open file
    "rook_semi_open":   12.0,     # rook on semi-open file
    "connected_rooks":  10.0,     # both rooks on same file/rank
    "king_shield":      10.0,     # per pawn in king's shield
    "king_shield_miss":  15.0,    # penalty per missing shield pawn
    "knight_outpost":   20.0,     # knight on outpost square
    "king_tropism":      2.0,     # per-piece closeness to enemy king
    "tempo":            10.0,     # bonus for side to move

    # ── Passed pawn bonuses by rank ───────────────────────────────────────
    "passed_r1":  0,  "passed_r2": 10, "passed_r3": 20, "passed_r4": 35,
    "passed_r5": 60,  "passed_r6": 90, "passed_r7": 130, "passed_r8": 0,

    # ── NN blend ──────────────────────────────────────────────────────────
    "nn_weight":        0.25,     # 0.0 = pure HCE, 1.0 = pure NN
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _piece_values():
    return {
        chess.PAWN:   WEIGHTS["pawn"],
        chess.KNIGHT: WEIGHTS["knight"],
        chess.BISHOP: WEIGHTS["bishop"],
        chess.ROOK:   WEIGHTS["rook"],
        chess.QUEEN:  WEIGHTS["queen"],
        chess.KING:   WEIGHTS["king"],
    }

def _passed_table():
    return [WEIGHTS[f"passed_r{i}"] for i in range(1, 9)]


# ── PeSTO piece-square tables ─────────────────────────────────────────────────
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

# ── Globals ───────────────────────────────────────────────────────────────────
TRANSPOSITION_TABLE = {}
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
_KILLERS = [[None, None] for _ in range(64)]
_HISTORY = {}

_SQ_FILE = [chess.square_file(s) for s in chess.SQUARES]
_SQ_RANK = [chess.square_rank(s) for s in chess.SQUARES]

# Chebyshev distance table for king tropism
_DISTANCE = [[max(abs(chess.square_file(a) - chess.square_file(b)),
                   abs(chess.square_rank(a) - chess.square_rank(b)))
               for b in chess.SQUARES] for a in chess.SQUARES]

# King shield squares: for each king square, the 3 pawn squares in front
_SHIELD_SQUARES = {}
for _ksq in chess.SQUARES:
    _kf, _kr = chess.square_file(_ksq), chess.square_rank(_ksq)
    _SHIELD_SQUARES[_ksq] = {}
    for _color in [chess.WHITE, chess.BLACK]:
        _dr = 1 if _color == chess.WHITE else -1
        _shield_r = _kr + _dr
        sqs = []
        if 0 <= _shield_r <= 7:
            for _df in [-1, 0, 1]:
                _sf = _kf + _df
                if 0 <= _sf <= 7:
                    sqs.append(chess.square(_sf, _shield_r))
        _SHIELD_SQUARES[_ksq][_color] = sqs


# ══════════════════════════════════════════════════════════════════════════════
#  NN EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _nn_eval(board):
    if not _HAS_NN:
        return 0.0
    vec = board_to_vector(board)
    _nn_input[0].copy_(torch.from_numpy(vec))
    with torch.no_grad():
        return _model(_nn_input).item() * SCORE_CLAMP


# ══════════════════════════════════════════════════════════════════════════════
#  PAWN STRUCTURE
# ══════════════════════════════════════════════════════════════════════════════

def _pawn_bonus(board, color):
    pawns = list(board.pieces(chess.PAWN, color))
    if not pawns:
        return 0.0
    opp = list(board.pieces(chess.PAWN, not color))
    passed_table = _passed_table()

    fcnt = {}
    for sq in pawns:
        f = _SQ_FILE[sq]; fcnt[f] = fcnt.get(f, 0) + 1

    opp_f = {}
    for sq in opp:
        f = _SQ_FILE[sq]; opp_f.setdefault(f, []).append(_SQ_RANK[sq])

    score = 0.0
    doubled_pen = WEIGHTS["doubled_penalty"]
    isolated_pen = WEIGHTS["isolated_penalty"]
    backward_pen = WEIGHTS["backward_penalty"]

    for sq in pawns:
        f, r = _SQ_FILE[sq], _SQ_RANK[sq]

        # Doubled
        if fcnt[f] > 1:
            score -= doubled_pen

        # Isolated
        has_neighbor = (f > 0 and f-1 in fcnt) or (f < 7 and f+1 in fcnt)
        if not has_neighbor:
            score -= isolated_pen

        # Backward: no friendly pawn on adjacent files at same or behind rank,
        # AND the stop square is controlled by enemy pawn
        if has_neighbor:
            behind_ranks = []
            for df in [-1, 1]:
                af = f + df
                if 0 <= af <= 7 and af in fcnt:
                    for osq in pawns:
                        if _SQ_FILE[osq] == af:
                            behind_ranks.append(_SQ_RANK[osq])
            if behind_ranks:
                if color == chess.WHITE:
                    if all(br > r for br in behind_ranks):
                        # stop square attacked by enemy?
                        stop_f = f
                        stop_r = r + 1
                        if stop_r <= 7:
                            for df2 in [-1, 1]:
                                ef = stop_f + df2
                                if 0 <= ef <= 7:
                                    for er in opp_f.get(ef, []):
                                        if er == stop_r + 1:
                                            score -= backward_pen
                                            break
                else:
                    if all(br < r for br in behind_ranks):
                        stop_r = r - 1
                        if stop_r >= 0:
                            for df2 in [-1, 1]:
                                ef = f + df2
                                if 0 <= ef <= 7:
                                    for er in opp_f.get(ef, []):
                                        if er == stop_r - 1:
                                            score -= backward_pen
                                            break

        # Passed
        passed = True
        for df in (-1, 0, 1):
            for or_ in opp_f.get(f + df, []):
                if color == chess.WHITE and or_ > r: passed = False; break
                if color == chess.BLACK and or_ < r: passed = False; break
            if not passed: break
        if passed:
            score += passed_table[r if color == chess.WHITE else 7 - r]

    return score


# ══════════════════════════════════════════════════════════════════════════════
#  KING SAFETY
# ══════════════════════════════════════════════════════════════════════════════

def _king_safety(board, color):
    """Evaluate pawn shield around the king. Rewards intact shield, penalizes holes."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0.0

    shield_sqs = _SHIELD_SQUARES[king_sq].get(color, [])
    if not shield_sqs:
        return 0.0

    our_pawns = board.pieces(chess.PAWN, color)
    shield_count = sum(1 for sq in shield_sqs if sq in our_pawns)
    missing = len(shield_sqs) - shield_count

    return shield_count * WEIGHTS["king_shield"] - missing * WEIGHTS["king_shield_miss"]


# ══════════════════════════════════════════════════════════════════════════════
#  KNIGHT OUTPOSTS
# ══════════════════════════════════════════════════════════════════════════════

def _knight_outposts(board, color):
    """
    Knight on rank 4-6 (relative) supported by own pawn, not attackable by enemy pawn.
    """
    score = 0.0
    knights = board.pieces(chess.KNIGHT, color)
    opp_pawns = board.pieces(chess.PAWN, not color)

    for sq in knights:
        r = _SQ_RANK[sq]
        f = _SQ_FILE[sq]
        rel_rank = r if color == chess.WHITE else 7 - r

        # Must be on rank 4, 5, or 6 (deep in enemy territory)
        if rel_rank < 3 or rel_rank > 5:
            continue

        # Supported by own pawn?
        pawn_dir = -1 if color == chess.WHITE else 1
        supported = False
        for df in [-1, 1]:
            pf = f + df
            pr = r + pawn_dir
            if 0 <= pf <= 7 and 0 <= pr <= 7:
                psq = chess.square(pf, pr)
                p = board.piece_at(psq)
                if p and p.piece_type == chess.PAWN and p.color == color:
                    supported = True
                    break

        if not supported:
            continue

        # Can enemy pawn attack this square?
        enemy_can_attack = False
        opp_dir = 1 if color == chess.WHITE else -1
        for df in [-1, 1]:
            af = f + df
            # Check all ranks behind this square for enemy pawns that could advance
            for check_r in range(r + opp_dir, 8 if opp_dir > 0 else -1, opp_dir):
                if 0 <= af <= 7 and 0 <= check_r <= 7:
                    csq = chess.square(af, check_r)
                    if csq in opp_pawns:
                        enemy_can_attack = True
                        break
            if enemy_can_attack:
                break

        if not enemy_can_attack:
            score += WEIGHTS["knight_outpost"]

    return score


# ══════════════════════════════════════════════════════════════════════════════
#  KING TROPISM
# ══════════════════════════════════════════════════════════════════════════════

def _king_tropism(board, color):
    """Bonus for having pieces close to the enemy king."""
    enemy_king = board.king(not color)
    if enemy_king is None:
        return 0.0

    score = 0.0
    trop_w = WEIGHTS["king_tropism"]

    # Knights, bishops, rooks, queens get tropism bonus
    for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for sq in board.pieces(pt, color):
            dist = _DISTANCE[sq][enemy_king]
            # Closer = higher bonus, max distance is 7
            score += trop_w * (7 - dist)

    return score


# ══════════════════════════════════════════════════════════════════════════════
#  CONNECTED ROOKS
# ══════════════════════════════════════════════════════════════════════════════

def _connected_rooks(board, color):
    """Bonus if both rooks are on the same rank or file with nothing between them."""
    rooks = list(board.pieces(chess.ROOK, color))
    if len(rooks) < 2:
        return 0.0

    r1, r2 = rooks[0], rooks[1]
    f1, f2 = _SQ_FILE[r1], _SQ_FILE[r2]
    rk1, rk2 = _SQ_RANK[r1], _SQ_RANK[r2]

    connected = False

    # Same file
    if f1 == f2:
        lo, hi = min(rk1, rk2), max(rk1, rk2)
        blocked = False
        for rk in range(lo + 1, hi):
            if board.piece_at(chess.square(f1, rk)) is not None:
                blocked = True
                break
        if not blocked:
            connected = True

    # Same rank
    elif rk1 == rk2:
        lo, hi = min(f1, f2), max(f1, f2)
        blocked = False
        for fl in range(lo + 1, hi):
            if board.piece_at(chess.square(fl, rk1)) is not None:
                blocked = True
                break
        if not blocked:
            connected = True

    return WEIGHTS["connected_rooks"] if connected else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  HANDCRAFTED EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _hce(board):
    """Full handcrafted evaluation with all tunable terms."""
    pv = _piece_values()

    mg_score = eg_score = 0.0
    game_phase = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None: continue
        sign    = 1 if piece.color == chess.WHITE else -1
        pst_idx = chess.square_mirror(sq) if piece.color == chess.WHITE else sq
        val     = pv[piece.piece_type]
        mg_score += sign * (val + PST_MG[piece.piece_type][pst_idx])
        eg_score += sign * (val + PST_EG[piece.piece_type][pst_idx])
        pt = piece.piece_type
        if pt in (chess.KNIGHT, chess.BISHOP): game_phase += 1
        elif pt == chess.ROOK:                 game_phase += 2
        elif pt == chess.QUEEN:                game_phase += 4

    eg_phase = 1.0 - min(game_phase, 24) / 24.0
    mg_phase = 1.0 - eg_phase
    score    = mg_score * mg_phase + eg_score * eg_phase

    # Mobility
    mob = WEIGHTS["mobility"]
    if board.turn == chess.WHITE:
        score += len(list(board.legal_moves)) * mob
        board.turn = chess.BLACK
        score -= len(list(board.legal_moves)) * mob
        board.turn = chess.WHITE
    else:
        score -= len(list(board.legal_moves)) * mob
        board.turn = chess.WHITE
        score += len(list(board.legal_moves)) * mob
        board.turn = chess.BLACK

    # Bishop pair
    bp = WEIGHTS["bishop_pair"]
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2: score += bp
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2: score -= bp

    # Pawn structure (doubled, isolated, backward, passed)
    score += _pawn_bonus(board, chess.WHITE)
    score -= _pawn_bonus(board, chess.BLACK)

    # Rook on open / semi-open file
    rof = WEIGHTS["rook_open_file"]
    rsf = WEIGHTS["rook_semi_open"]
    wpf = {_SQ_FILE[s] for s in board.pieces(chess.PAWN, chess.WHITE)}
    bpf = {_SQ_FILE[s] for s in board.pieces(chess.PAWN, chess.BLACK)}
    for sq in board.pieces(chess.ROOK, chess.WHITE):
        f = _SQ_FILE[sq]
        score += rof if (f not in wpf and f not in bpf) else (rsf if f not in wpf else 0)
    for sq in board.pieces(chess.ROOK, chess.BLACK):
        f = _SQ_FILE[sq]
        score -= rof if (f not in bpf and f not in wpf) else (rsf if f not in bpf else 0)

    # Connected rooks
    score += _connected_rooks(board, chess.WHITE)
    score -= _connected_rooks(board, chess.BLACK)

    # King safety (pawn shield)
    score += _king_safety(board, chess.WHITE)
    score -= _king_safety(board, chess.BLACK)

    # Knight outposts
    score += _knight_outposts(board, chess.WHITE)
    score -= _knight_outposts(board, chess.BLACK)

    # King tropism
    score += _king_tropism(board, chess.WHITE)
    score -= _king_tropism(board, chess.BLACK)

    # Tempo
    if board.turn == chess.WHITE:
        score += WEIGHTS["tempo"]
    else:
        score -= WEIGHTS["tempo"]

    return score


# ══════════════════════════════════════════════════════════════════════════════
#  HYBRID EVALUATION — full eval for leaf nodes in main search
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(board):
    if board.is_checkmate():
        return -99999.0 if board.turn == chess.WHITE else 99999.0
    if (board.is_stalemate() or board.is_insufficient_material() or
            board.is_repetition(3) or board.is_fifty_moves()):
        return 0.0

    hce_score = _hce(board)

    nn_w = WEIGHTS["nn_weight"]
    if _HAS_NN and nn_w > 0.0:
        nn_score = _nn_eval(board)
        return (1.0 - nn_w) * hce_score + nn_w * nn_score
    else:
        return hce_score


# ── Fast eval for quiescence (HCE only — speed matters here) ─────────────────
def _fast_evaluate(board):
    if board.is_checkmate():
        return -99999.0 if board.turn == chess.WHITE else 99999.0
    if (board.is_stalemate() or board.is_insufficient_material() or
            board.is_repetition(3) or board.is_fifty_moves()):
        return 0.0
    return _hce(board)


# ══════════════════════════════════════════════════════════════════════════════
#  MOVE ORDERING
# ══════════════════════════════════════════════════════════════════════════════

def score_move(board, move, ply=0, tt_move=None):
    pv = _piece_values()
    if tt_move and move == tt_move:           return 20_000
    if board.is_capture(move):
        victim   = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        v = pv.get(victim.piece_type,   100) if victim   else 100
        a = pv.get(attacker.piece_type, 100) if attacker else 100
        return 10_000 + v * 10 - a
    if move.promotion:                        return 9_000 + pv.get(move.promotion, 0)
    if move == _KILLERS[ply][0]:              return 8_000
    if move == _KILLERS[ply][1]:              return 7_000
    return _HISTORY.get((move.from_square, move.to_square), 0)


def order_moves(board, moves, ply=0, tt_move=None):
    return sorted(moves, key=lambda m: score_move(board, m, ply, tt_move), reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
#  QUIESCENCE SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def qsearch(board, alpha, beta, maximizing):
    stand_pat = _fast_evaluate(board)
    if maximizing:
        if stand_pat >= beta:  return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha: return alpha
        beta = min(beta, stand_pat)
    tactical = order_moves(board,
                           [m for m in board.legal_moves
                            if board.is_capture(m) or m.promotion])
    for move in tactical:
        board.push(move)
        score = qsearch(board, alpha, beta, not maximizing)
        board.pop()
        if maximizing:
            alpha = max(alpha, score)
            if alpha >= beta: break
        else:
            beta = min(beta, score)
            if beta <= alpha: break
    return alpha if maximizing else beta


# ══════════════════════════════════════════════════════════════════════════════
#  MINIMAX WITH ALPHA-BETA
# ══════════════════════════════════════════════════════════════════════════════

def minimax(board, depth, alpha, beta, maximizing, ply=0):
    hash_key = chess.polyglot.zobrist_hash(board)
    tt_move  = None
    if hash_key in TRANSPOSITION_TABLE:
        entry = TRANSPOSITION_TABLE[hash_key]
        tt_depth, tt_flag, tt_score = entry[0], entry[1], entry[2]
        tt_move = entry[3] if len(entry) > 3 else None
        if tt_depth >= depth:
            if   tt_flag == TT_EXACT: return tt_score
            elif tt_flag == TT_LOWER: alpha = max(alpha, tt_score)
            elif tt_flag == TT_UPPER: beta  = min(beta,  tt_score)
            if alpha >= beta:         return tt_score

    if depth <= 0:       return qsearch(board, alpha, beta, maximizing)
    if board.is_game_over(): return evaluate(board)

    moves      = order_moves(board, list(board.legal_moves), ply, tt_move)
    orig_alpha = alpha
    best_score = float('-inf') if maximizing else float('inf')
    best_move  = None

    if maximizing:
        for move in moves:
            board.push(move)
            score = minimax(board, depth - 1, alpha, beta, False, ply + 1)
            board.pop()
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if beta <= alpha:
                if not board.is_capture(move) and not move.promotion:
                    if move != _KILLERS[ply][0]:
                        _KILLERS[ply][1] = _KILLERS[ply][0]
                        _KILLERS[ply][0] = move
                    k = (move.from_square, move.to_square)
                    _HISTORY[k] = _HISTORY.get(k, 0) + depth * depth
                break
    else:
        for move in moves:
            board.push(move)
            score = minimax(board, depth - 1, alpha, beta, True, ply + 1)
            board.pop()
            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, score)
            if beta <= alpha:
                if not board.is_capture(move) and not move.promotion:
                    if move != _KILLERS[ply][0]:
                        _KILLERS[ply][1] = _KILLERS[ply][0]
                        _KILLERS[ply][0] = move
                    k = (move.from_square, move.to_square)
                    _HISTORY[k] = _HISTORY.get(k, 0) + depth * depth
                break

    tt_flag = TT_EXACT
    if   best_score <= orig_alpha: tt_flag = TT_UPPER
    elif best_score >= beta:       tt_flag = TT_LOWER
    TRANSPOSITION_TABLE[hash_key] = (depth, tt_flag, best_score, best_move)
    return best_score


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def get_next_move(board, color, depth=3):
    global TRANSPOSITION_TABLE, _KILLERS, _HISTORY

    if len(TRANSPOSITION_TABLE) > 500_000:
        TRANSPOSITION_TABLE.clear()
    if len(_HISTORY) > 100_000:
        _HISTORY.clear()
    _KILLERS[:] = [[None, None] for _ in range(64)]

    best_move  = None
    maximizing = (color == chess.WHITE)
    best_score = float('-inf') if maximizing else float('inf')

    alpha = float('-inf')
    beta  = float('inf')
    moves = order_moves(board, list(board.legal_moves), 0)

    for move in moves:
        board.push(move)
        score = minimax(board, depth - 1, alpha, beta, not maximizing, 1)
        board.pop()
        if maximizing:
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
        else:
            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, best_score)

    return best_move if best_move else moves[0]


if __name__ == '__main__':
    b = chess.Board()
    move = get_next_move(b, chess.WHITE, depth=3)
    if move is not None:
        print(f"[team_goraieb] Opening move: {b.san(move)}")
        print(f"  NN available: {_HAS_NN}")
        print(f"  Cython: {_HAVE_CYTHON}")
        print(f"  NN weight: {WEIGHTS['nn_weight']}")
