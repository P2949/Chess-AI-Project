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
  • Hanging pieces: penalty for undefended pieces under attack
  • King attack pressure: bonus for attacking squares near enemy king

Search: TT, quiescence with delta/bad-capture pruning, killer moves,
        history heuristic with malus, countermove heuristic, MVV-LVA ordering
        Iterative deepening with aspiration windows, PVS (Principal Variation Search)
        Null-move pruning, LMR (Late Move Reductions), LMP (Late Move Pruning)
        Futility pruning, check/single-reply/recapture extensions
        Mate-distance pruning, draw contempt, opening book
        Policy-guided root ordering
Eval:   PeSTO tapered + structure + NN correction (tunable blend)
"""

import chess
import chess.polyglot
import torch
import numpy as np
import sys
from pathlib import Path
import math

# Ensure this file's directory is in sys.path so fast_chess.so can be found
_THIS_DIR = str(Path(__file__).resolve().parent)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

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

class ResBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dims, dims),
            nn.BatchNorm1d(dims),
            nn.ReLU(),
            nn.Linear(dims, dims),
            nn.BatchNorm1d(dims)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))

class PolicyEvaluator(nn.Module):
    def __init__(self, input_dim: int = 773):
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU())
        self.layer1 = ResBlock(512)
        self.layer2 = ResBlock(512)
        self.head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)

SCORE_CLAMP = 1500
MATE_SCORE = 100_000

# ── Feature toggles ──────────────────────────────────────────────────────────
USE_OPENING_BOOK = False

# ── Search tuning constants ───────────────────────────────────────────────────
# Switch this later to "normal" or "aggressive" without touching the search code.
PRUNING_PROFILE = "off"  # "off", "conservative", "normal", "aggressive"

PRUNING_PRESETS = {
    "off": {
        "NULL_MOVE_ENABLE": False,
        "NULL_MOVE_MIN_DEPTH": 99,
        "NULL_MOVE_REDUCTION": 0,
        "LMR_MIN_DEPTH": 99,
        "LMR_START_MOVE": 99,
        "LMR_REDUCTION": 0,
        "LMP_CUTOFFS": {1: 999, 2: 999, 3: 999},
        "FUTILITY_MARGIN_CP": 0,
        "BAD_CAPTURE_MARGIN": -999,
    },
    "conservative": {
        "NULL_MOVE_ENABLE": True,
        "NULL_MOVE_MIN_DEPTH": 4,
        "NULL_MOVE_REDUCTION": 1,
        "LMR_MIN_DEPTH": 4,
        "LMR_START_MOVE": 6,
        "LMR_REDUCTION": 1,
        "LMP_CUTOFFS": {1: 36, 2: 54, 3: 72},
        "FUTILITY_MARGIN_CP": 120,
        "BAD_CAPTURE_MARGIN": 120,
    },
    "normal": {
        "NULL_MOVE_ENABLE": True,
        "NULL_MOVE_MIN_DEPTH": 3,
        "NULL_MOVE_REDUCTION": 1,
        "LMR_MIN_DEPTH": 3,
        "LMR_START_MOVE": 5,
        "LMR_REDUCTION": 1,
        "LMP_CUTOFFS": {1: 28, 2: 40, 3: 56},
        "FUTILITY_MARGIN_CP": 160,
        "BAD_CAPTURE_MARGIN": 150,
    },
    "aggressive": {
        "NULL_MOVE_ENABLE": True,
        "NULL_MOVE_MIN_DEPTH": 3,
        "NULL_MOVE_REDUCTION": 2,
        "LMR_MIN_DEPTH": 3,
        "LMR_START_MOVE": 4,
        "LMR_REDUCTION": 1,
        "LMP_CUTOFFS": {1: 24, 2: 36, 3: 48},
        "FUTILITY_MARGIN_CP": 200,
        "BAD_CAPTURE_MARGIN": 180,
    },
}

_SEARCH = PRUNING_PRESETS[PRUNING_PROFILE]

# Draw contempt
DRAW_CONTEMPT_CP = 15.0

# Null-move pruning
NULL_MOVE_ENABLE = _SEARCH["NULL_MOVE_ENABLE"]
NULL_MOVE_MIN_DEPTH = _SEARCH["NULL_MOVE_MIN_DEPTH"]
NULL_MOVE_REDUCTION = _SEARCH["NULL_MOVE_REDUCTION"]
NULL_MOVE_MARGIN_CP = 120

# Late Move Reductions
LMR_MIN_DEPTH = _SEARCH["LMR_MIN_DEPTH"]
LMR_START_MOVE = _SEARCH["LMR_START_MOVE"]
LMR_REDUCTION = _SEARCH["LMR_REDUCTION"]

# Late Move Pruning
LMP_CUTOFFS = _SEARCH["LMP_CUTOFFS"]

# Futility pruning margin
FUTILITY_MARGIN_CP = _SEARCH["FUTILITY_MARGIN_CP"]

# Bad capture threshold for qsearch filtering
BAD_CAPTURE_MARGIN = _SEARCH["BAD_CAPTURE_MARGIN"]

# Aspiration window half-width
ASP_WINDOW = 80


# ── Opening book ──────────────────────────────────────────────────────────────
_OPENING_BOOK_MAX_PLIES = 16

_OPENING_BOOK = {
    # Root moves
    (): ["e2e4", "d2d4", "c2c4", "g1f3", "b1c3", "g2g3"],

    # 1.e4
    ("e2e4",): ["e7e5", "c7c5", "e7e6", "c7c6", "d7d5", "g7g6"],
    ("e2e4", "e7e5"): ["g1f3", "f1c4", "b1c3", "f2f4"],
    ("e2e4", "e7e5", "g1f3"): ["b8c6", "g8f6", "d7d6"],
    ("e2e4", "e7e5", "g1f3", "b8c6"): ["f1b5", "f1c4", "d2d4", "b1c3"],
    ("e2e4", "e7e5", "g1f3", "g8f6"): ["b1c3", "d2d4", "f1c4"],
    ("e2e4", "e7e5", "f1c4"): ["g8f6", "b8c6", "c7c6"],
    ("e2e4", "e7e5", "b1c3"): ["g8f6", "b8c6"],
    ("e2e4", "e7e5", "f2f4"): ["e5f4", "g1f3", "g8f6"],
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5"): ["a7a6", "g8f6"],
    ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4"): ["f8c5", "g8f6"],

    ("e2e4", "c7c5"): ["g1f3", "b1c3", "c2c3", "d2d4"],
    ("e2e4", "c7c5", "g1f3"): ["d7d6", "b8c6", "e7e6", "g7g6"],
    ("e2e4", "c7c5", "g1f3", "d7d6"): ["d2d4", "b1c3"],
    ("e2e4", "c7c5", "g1f3", "b8c6"): ["d2d4", "b1c3"],
    ("e2e4", "c7c5", "g1f3", "e7e6"): ["d2d4", "b1c3"],
    ("e2e4", "c7c5", "d2d4"): ["c5d4", "g1f3", "b1c3"],
    ("e2e4", "c7c5", "d2d4", "c5d4"): ["d1d4", "g1f3"],
    ("e2e4", "c7c5", "c2c3"): ["d7d5", "e4d5"],
    ("e2e4", "c7c5", "b1c3"): ["b8c6", "g1f3"],
    ("e2e4", "c7c5", "g1f3", "d7d6", "d2d4"): ["c5d4", "g1f3"],
    ("e2e4", "c7c5", "g1f3", "b8c6", "d2d4"): ["c5d4", "g1f3"],

    ("e2e4", "e7e6"): ["d2d4", "g1f3", "b1c3"],
    ("e2e4", "e7e6", "d2d4"): ["d7d5", "g8f6", "c7c5"],
    ("e2e4", "e7e6", "d2d4", "d7d5"): ["b1c3", "g1f3"],
    ("e2e4", "e7e6", "g1f3"): ["d7d5", "d7d6", "c7c5"],
    ("e2e4", "e7e6", "b1c3"): ["d7d5", "g8f6"],

    ("e2e4", "c7c6"): ["d2d4", "g1f3"],
    ("e2e4", "c7c6", "d2d4"): ["d7d5", "g8f6"],
    ("e2e4", "c7c6", "d2d4", "d7d5"): ["b1c3", "g1f3"],

    ("e2e4", "d7d5"): ["e4d5", "d2d4", "g1f3"],
    ("e2e4", "d7d5", "e4d5"): ["d8d5", "g8f6", "e7e6"],
    ("e2e4", "d7d5", "e4d5", "d8d5"): ["b1c3", "g1f3"],

    ("e2e4", "g7g6"): ["d2d4", "g1f3"],
    ("e2e4", "g7g6", "d2d4"): ["f8g7", "d7d6"],
    ("e2e4", "g7g6", "g1f3"): ["f8g7", "d7d6"],

    # 1.d4
    ("d2d4",): ["d7d5", "g8f6", "e7e6", "g7g6", "c7c6"],
    ("d2d4", "d7d5"): ["c2c4", "g1f3", "b1c3"],
    ("d2d4", "d7d5", "c2c4"): ["e7e6", "c7c6", "d5c4", "g8f6"],
    ("d2d4", "d7d5", "c2c4", "e7e6"): ["b1c3", "g1f3"],
    ("d2d4", "d7d5", "c2c4", "c7c6"): ["g1f3", "b1c3"],
    ("d2d4", "d7d5", "g1f3"): ["g8f6", "e7e6", "c7c6"],
    ("d2d4", "d7d5", "b1c3"): ["g8f6", "e7e6"],

    ("d2d4", "g8f6"): ["c2c4", "g1f3", "g2g3"],
    ("d2d4", "g8f6", "c2c4"): ["e7e6", "g7g6", "c7c5", "d7d5"],
    ("d2d4", "g8f6", "c2c4", "e7e6"): ["g1f3", "b1c3"],
    ("d2d4", "g8f6", "c2c4", "g7g6"): ["b1c3", "g1f3"],
    ("d2d4", "g8f6", "g1f3"): ["d7d5", "e7e6", "g7g6"],

    ("d2d4", "e7e6"): ["c2c4", "g1f3", "b1c3"],
    ("d2d4", "e7e6", "c2c4"): ["d7d5", "g8f6"],
    ("d2d4", "g7g6"): ["c2c4", "g1f3"],
    ("d2d4", "g7g6", "c2c4"): ["f8g7", "d7d6"],

    ("d2d4", "c7c6"): ["c2c4", "g1f3"],
    ("d2d4", "c7c6", "c2c4"): ["d7d5", "g8f6"],

    # 1.c4
    ("c2c4",): ["e7e5", "g8f6", "c7c5", "e7e6", "g7g6"],
    ("c2c4", "e7e5"): ["g1f3", "b1c3", "g2g3"],
    ("c2c4", "e7e5", "g1f3"): ["g8f6", "b8c6"],
    ("c2c4", "e7e5", "b1c3"): ["g8f6", "b8c6"],

    ("c2c4", "g8f6"): ["b1c3", "g1f3", "g2g3"],
    ("c2c4", "g8f6", "g1f3"): ["e7e6", "g7g6", "c7c5"],
    ("c2c4", "g8f6", "b1c3"): ["e7e6", "g7g6"],

    ("c2c4", "c7c5"): ["g1f3", "b1c3", "g2g3"],
    ("c2c4", "c7c5", "g1f3"): ["g8f6", "d7d6", "b8c6"],
    ("c2c4", "c7c5", "b1c3"): ["g8f6", "b8c6"],

    ("c2c4", "e7e6"): ["g1f3", "b1c3", "g2g3"],
    ("c2c4", "g7g6"): ["b1c3", "g1f3"],

    # 1.Nf3
    ("g1f3",): ["d7d5", "g8f6", "c7c5", "e7e6", "g7g6"],
    ("g1f3", "d7d5"): ["d2d4", "g2g3", "c2c4"],
    ("g1f3", "d7d5", "d2d4"): ["g8f6", "e7e6"],
    ("g1f3", "d7d5", "g2g3"): ["g8f6", "c7c6"],

    ("g1f3", "g8f6"): ["d2d4", "c2c4", "g2g3"],
    ("g1f3", "g8f6", "d2d4"): ["d7d5", "e7e6", "g7g6"],
    ("g1f3", "g8f6", "c2c4"): ["e7e6", "g7g6", "c7c5"],

    ("g1f3", "c7c5"): ["d2d4", "g2g3"],
    ("g1f3", "e7e6"): ["d2d4", "c2c4"],
    ("g1f3", "g7g6"): ["d2d4", "c2c4"],

    # 1.Nc3
    ("b1c3",): ["d7d5", "e7e5", "g8f6", "c7c6"],
    ("b1c3", "d7d5"): ["d2d4", "e2e4"],
    ("b1c3", "e7e5"): ["e2e4", "g1f3"],
    ("b1c3", "g8f6"): ["d2d4", "e2e4"],

    # 1.g3
    ("g2g3",): ["d7d5", "g8f6", "e7e5"],
    ("g2g3", "d7d5"): ["f1g2", "g1f3"],
    ("g2g3", "g8f6"): ["f1g2", "d2d4"],
    ("g2g3", "e7e5"): ["f1g2", "g1f3"],
}


def _opening_book_move(board):
    """Return a principled opening move if in the small internal repertoire."""
    if not USE_OPENING_BOOK:
        return None
    if len(board.move_stack) > _OPENING_BOOK_MAX_PLIES:
        return None
    key = tuple(m.uci() for m in board.move_stack)
    for uci in _OPENING_BOOK.get(key, []):
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            return move
    return None

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

_policy_model = PolicyEvaluator()
_policy_model_path = Path(__file__).resolve().parent / "policy.pt"
if _policy_model_path.exists():
    _policy_model.load_state_dict(torch.load(str(_policy_model_path), map_location=_device))
    _policy_model.eval()
    _HAS_POLICY = True
else:
    _HAS_POLICY = False

_policy_input = torch.zeros(1, 773, dtype=torch.float32)
_nn_input = torch.zeros(1, 773, dtype=torch.float32)

# Warm up
if _HAS_NN:
    with torch.no_grad():
        _model(_nn_input)


# ══════════════════════════════════════════════════════════════════════════════
#  TUNABLE WEIGHTS — optimizer patches this dict directly
# ══════════════════════════════════════════════════════════════════════════════

MATERIAL_MULT = 1.5

WEIGHTS = {
      "pawn": 100 * MATERIAL_MULT,
      "knight": 320 * MATERIAL_MULT,
      "bishop": 330 * MATERIAL_MULT,
      "rook": 500 * MATERIAL_MULT,
      "queen": 900 * MATERIAL_MULT,
      "king": 20000,
      "mobility": 4.0,
      "bishop_pair": 40.0,
      "doubled_penalty": 32.0,
      "isolated_penalty": 5.0,
      "backward_penalty": 24.0,
      "rook_open_file": 30.0,
      "rook_semi_open": 10.0,
      "connected_rooks": 25.0,
      "king_shield": 12.0,
      "king_shield_miss": 0.0,
      "knight_outpost": 40.0,
      "king_tropism": 4.0,
      "tempo": 10.0,
      "hanging_piece": 40.0,
      "king_attack": 0.0,
      "pin_penalty": 25.0,
      "center_control": 0.0,
      "uncastled_king": 50.0,
      "rook_doubled_file": 35.0,
      "policy_weight": 0.01,
      "nn_phase_boost": 1.5,
      "early_queen_pen": 35.0,
      "development": 12.0,
      "passed_r1": 0,
      "passed_r2": 10,
      "passed_r3": 20,
      "passed_r4": 35,
      "passed_r5": 60,
      "passed_r6": 90,
      "passed_r7": 130,
      "passed_r8": 0,
      "nn_weight": 0.02,
  }

# ── Eval cache ────────────────────────────────────────────────────────────────
# FIX #6: Cache is keyed by zobrist hash. Must be cleared whenever WEIGHTS
# changes (the optimizer does this via _reset_engine). The cache only stores
# HCE results which depend on WEIGHTS.
EVAL_CACHE = {}
EVAL_CACHE_MAX = 200_000_000_000_000_000


# ── Helpers ───────────────────────────────────────────────────────────────────

def _policy_eval(board):
    if not _HAS_POLICY:
        return 0.0
    vec = board_to_vector(board)
    _policy_input[0].copy_(torch.from_numpy(vec))
    with torch.no_grad():
        return float(torch.tanh(_policy_model(_policy_input)).item())


def _root_policy_bias(board, move):
    if not _HAS_POLICY:
        return 0.0
    mover = board.turn
    board.push(move)
    try:
        s = _policy_eval(board)
    finally:
        board.pop()
    return s if mover == chess.WHITE else -s


def _cache_key(board):
    return chess.polyglot.zobrist_hash(board)


def _cached_hce(board):
    key = _cache_key(board)
    if key in EVAL_CACHE:
        return EVAL_CACHE[key]
    val = _hce(board)
    if len(EVAL_CACHE) >= EVAL_CACHE_MAX:
        EVAL_CACHE.clear()
    EVAL_CACHE[key] = val
    return val


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
_COUNTERMOVE = {}   # (from_sq, to_sq) -> best reply Move


def _has_non_pawn_material(board, color):
    """True if side has any piece besides pawns and king. Used by null-move."""
    return any(
        board.pieces(pt, color)
        for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
    )


def _update_history(move, bonus):
    """History with clamping to avoid runaway scores."""
    key = (move.from_square, move.to_square)
    _HISTORY[key] = max(-50_000, min(50_000, _HISTORY.get(key, 0) + bonus))

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

# King ring: precomputed for each square
_KING_RING = {}
for _ksq in chess.SQUARES:
    _kf = chess.square_file(_ksq)
    _kr = chess.square_rank(_ksq)
    ring = []
    for _df in (-1, 0, 1):
        for _dr in (-1, 0, 1):
            nf, nr = _kf + _df, _kr + _dr
            if 0 <= nf <= 7 and 0 <= nr <= 7:
                ring.append(chess.square(nf, nr))
    _KING_RING[_ksq] = ring


# ── Status (prints on import) ────────────────────────────────────────────────
print(f"[team_goraieb] NN: {'✓' if _HAS_NN else '✗'}  "
      f"Cython: {'✓' if _HAVE_CYTHON else '✗'}  "
      f"NN weight: {WEIGHTS['nn_weight']} "
      f"Policy: {'✓' if _HAS_POLICY else '✗'}")


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

        if fcnt[f] > 1:
            score -= doubled_pen

        has_neighbor = (f > 0 and f-1 in fcnt) or (f < 7 and f+1 in fcnt)
        if not has_neighbor:
            score -= isolated_pen

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
    king_sq = board.king(color)
    if king_sq is None:
        return 0.0

    enemy = not color
    score = 0.0

    # Pawn shield in front of the king.
    shield_sqs = _SHIELD_SQUARES[king_sq].get(color, [])
    our_pawns = board.pieces(chess.PAWN, color)
    shield_count = sum(1 for sq in shield_sqs if sq in our_pawns)
    missing = len(shield_sqs) - shield_count

    score += shield_count * WEIGHTS["king_shield"]
    score -= missing * max(WEIGHTS["king_shield_miss"], 4.0)

    # Direct attacks on the king square.
    attackers = board.attackers(enemy, king_sq)
    defenders = board.attackers(color, king_sq)
    score -= 20.0 * len(attackers)
    if attackers and not defenders:
        score -= 18.0

    # Weighted pressure on the king ring.
    piece_weight = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 1.8,
        chess.BISHOP: 1.8,
        chess.ROOK: 2.4,
        chess.QUEEN: 3.2,
    }
    ring_pressure = 0.0
    for sq in _KING_RING[king_sq]:
        for atk_sq in board.attackers(enemy, sq):
            atk_piece = board.piece_at(atk_sq)
            if atk_piece is not None:
                ring_pressure += piece_weight.get(atk_piece.piece_type, 1.0)
    score -= 2.0 * ring_pressure

    # Missing pawns on the king file and adjacent files.
    own_pawn_files = {_SQ_FILE[s] for s in our_pawns}
    king_file = _SQ_FILE[king_sq]
    for f in (king_file - 1, king_file, king_file + 1):
        if 0 <= f <= 7 and f not in own_pawn_files:
            score -= 3.5 if f == king_file else 2.0

    # Enemy rook/queen aligned with the king on an open file.
    for pt, weight in ((chess.ROOK, 8.0), (chess.QUEEN, 10.0)):
        for sq in board.pieces(pt, enemy):
            if _SQ_FILE[sq] != king_file:
                continue
            step = 8 if _SQ_RANK[sq] < _SQ_RANK[king_sq] else -8
            clear = True
            for between in range(sq + step, king_sq, step):
                if board.piece_at(between) is not None:
                    clear = False
                    break
            if clear:
                score -= weight

    # Preference for actual castled squares; central king squares are worse.
    if king_sq in (chess.G1, chess.C1, chess.G8, chess.C8):
        score += 6.0
    elif king_sq in (chess.E1, chess.D1, chess.F1, chess.E8, chess.D8, chess.F8):
        score -= 8.0
    elif king_file in (2, 3, 4, 5):
        score -= 4.0

    return score



# ══════════════════════════════════════════════════════════════════════════════
#  KNIGHT OUTPOSTS
# ══════════════════════════════════════════════════════════════════════════════

def _knight_outposts(board, color):
    score = 0.0
    knights = board.pieces(chess.KNIGHT, color)
    opp_pawns = board.pieces(chess.PAWN, not color)

    for sq in knights:
        r = _SQ_RANK[sq]
        f = _SQ_FILE[sq]
        rel_rank = r if color == chess.WHITE else 7 - r

        if rel_rank < 3 or rel_rank > 5:
            continue

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

        enemy_can_attack = False
        opp_dir = 1 if color == chess.WHITE else -1
        for df in [-1, 1]:
            af = f + df
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
#  HANGING PIECES & SIMPLIFIED SEE
# ══════════════════════════════════════════════════════════════════════════════

# Piece type to rough value for SEE ordering
_SEE_VALUES = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000,
}


def _lowest_attacker_value(board, sq, color):
    """Value of the cheapest piece of `color` attacking `sq`."""
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING):
        attackers = board.pieces(pt, color) & board.attackers(color, sq)
        if attackers:
            return _SEE_VALUES[pt]
    return 0


def _hanging_pieces(board, color):
    """
    Penalize pieces that are:
    1. Attacked and completely undefended (classic "hanging")
    2. Attacked by a lower-value piece (losing exchange even if defended)
    
    This catches situations like: knight on e4 attacked by bishop on f5
    where Bxf5 wins material even though the knight might be "defended".
    """
    score = 0.0
    pv = _piece_values()
    enemy = not color
    hang_w = WEIGHTS["hanging_piece"]

    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            attackers = board.attackers(enemy, sq)
            if not attackers:
                continue

            defenders = board.attackers(color, sq)
            piece_val = pv[pt]

            if not defenders:
                # Completely undefended and attacked — severe penalty
                score -= hang_w * (1.0 + piece_val / 500.0)
            else:
                # Has defenders, but check if cheapest attacker wins the trade
                cheapest_attacker = _lowest_attacker_value(board, sq, enemy)
                if cheapest_attacker < piece_val * 0.8:
                    # Attacker is significantly cheaper — this is a losing trade
                    # e.g., pawn attacks knight, bishop attacks rook
                    trade_loss = piece_val - cheapest_attacker
                    score -= hang_w * 0.4 * (trade_loss / 500.0)

    return score


# ══════════════════════════════════════════════════════════════════════════════
#  KING ATTACK PRESSURE
# ══════════════════════════════════════════════════════════════════════════════

def _king_attack_pressure(board, color):
    enemy_king = board.king(not color)
    if enemy_king is None:
        return 0.0

    ring = _KING_RING[enemy_king]
    pressure = 0.0
    for sq in ring:
        pressure += len(board.attackers(color, sq))

    return pressure * WEIGHTS["king_attack"]


# ══════════════════════════════════════════════════════════════════════════════
#  KING TROPISM
# ══════════════════════════════════════════════════════════════════════════════

def _king_tropism(board, color):
    enemy_king = board.king(not color)
    if enemy_king is None:
        return 0.0

    score = 0.0
    trop_w = WEIGHTS["king_tropism"]

    for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        for sq in board.pieces(pt, color):
            dist = _DISTANCE[sq][enemy_king]
            score += trop_w * (7 - dist)

    return score


# ══════════════════════════════════════════════════════════════════════════════
#  PIN DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _pin_penalty(board, color):
    """
    Penalize having pieces absolutely pinned to the king.

    A pinned piece can't move without exposing the king to check, making it
    nearly useless defensively and offensively. Higher-value pinned pieces
    get a bigger penalty (a pinned queen is worse than a pinned pawn).

    This directly addresses mistakes like Be6 when Nf6 is pinned to the
    queen by Bg5 — the engine will prefer moves that break the pin.
    """
    king_sq = board.king(color)
    if king_sq is None:
        return 0.0

    score = 0.0
    pv = _piece_values()
    pen = WEIGHTS["pin_penalty"]

    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            if board.is_pinned(color, sq):
                # Scale penalty by piece value: pinned queen > pinned knight > pinned pawn
                score -= pen * (pv[pt] / 300.0)

    return score


# ══════════════════════════════════════════════════════════════════════════════
#  CONNECTED ROOKS
# ══════════════════════════════════════════════════════════════════════════════

def _development_score(board, color):
    """
    Reward developing minor pieces, penalize early queen moves.

    Early queen development is a common amateur mistake: the queen gets
    chased around losing tempo while the opponent develops pieces.
    We penalize the queen leaving the back rank when fewer than 3 minor
    pieces (knights + bishops) have been developed.
    """
    back_rank = 0 if color == chess.WHITE else 7
    score = 0.0

    # Count developed minor pieces (knights/bishops off the back rank)
    developed_minors = 0
    for pt in (chess.KNIGHT, chess.BISHOP):
        for sq in board.pieces(pt, color):
            if _SQ_RANK[sq] != back_rank:
                developed_minors += 1
                score += WEIGHTS["development"]

    # Penalize queen off back rank when minors aren't developed yet
    queen_sqs = board.pieces(chess.QUEEN, color)
    if queen_sqs:
        queen_sq = list(queen_sqs)[0]
        if _SQ_RANK[queen_sq] != back_rank and developed_minors < 3:
            # Bigger penalty the fewer pieces are developed
            score -= WEIGHTS["early_queen_pen"] * (3 - developed_minors)

    return score

def _connected_rooks(board, color):
    rooks = list(board.pieces(chess.ROOK, color))
    if len(rooks) < 2:
        return 0.0

    r1, r2 = rooks[0], rooks[1]
    f1, f2 = _SQ_FILE[r1], _SQ_FILE[r2]
    rk1, rk2 = _SQ_RANK[r1], _SQ_RANK[r2]

    connected = False

    if f1 == f2:
        lo, hi = min(rk1, rk2), max(rk1, rk2)
        blocked = False
        for rk in range(lo + 1, hi):
            if board.piece_at(chess.square(f1, rk)) is not None:
                blocked = True
                break
        if not blocked:
            connected = True
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
#  CENTER CONTROL
# ══════════════════════════════════════════════════════════════════════════════

# Central squares: d4, d5, e4, e5 (extended center: c3-f3 to c6-f6)
_CENTER_SQUARES = {chess.D4, chess.D5, chess.E4, chess.E5}
_EXTENDED_CENTER = {chess.C3, chess.D3, chess.E3, chess.F3,
                    chess.C4, chess.D4, chess.E4, chess.F4,
                    chess.C5, chess.D5, chess.E5, chess.F5,
                    chess.C6, chess.D6, chess.E6, chess.F6}


def _center_control(board, color):
    """
    Reward pieces and pawns that occupy or attack central squares.

    Pieces on d4/d5/e4/e5 get full bonus. Pieces on extended center
    (c3-f6) get half bonus. Pawns on center get extra bonus because
    center pawns control space and restrict the opponent.

    This directly addresses the a6-vs-Nxe4 problem: Nxe4 places
    a knight on a central square, a6 is a flank move with no center impact.
    """
    score = 0.0
    w = WEIGHTS["center_control"]

    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, color):
            if sq in _CENTER_SQUARES:
                # Piece/pawn on center: full bonus, extra for pawns
                bonus = w * 2.0 if pt == chess.PAWN else w * 1.5
                score += bonus
            elif sq in _EXTENDED_CENTER:
                score += w * 0.5

    return score


# ══════════════════════════════════════════════════════════════════════════════
#  UNCASTLED KING PENALTY
# ══════════════════════════════════════════════════════════════════════════════

def _uncastled_king_penalty(board, color):
    """
    Penalize an exposed king even when castling rights are already gone.
    This stays broad on purpose: many bad positions are "moved king, still unsafe".
    """
    king_sq = board.king(color)
    if king_sq is None:
        return 0.0

    if king_sq in (chess.G1, chess.C1, chess.G8, chess.C8):
        return 0.0

    back_rank = 0 if color == chess.WHITE else 7
    king_rank = _SQ_RANK[king_sq]
    king_file = _SQ_FILE[king_sq]

    # Open center = no full pawn barrier on d/e files.
    w_pawns_d = any(_SQ_FILE[s] == 3 for s in board.pieces(chess.PAWN, chess.WHITE))
    w_pawns_e = any(_SQ_FILE[s] == 4 for s in board.pieces(chess.PAWN, chess.WHITE))
    b_pawns_d = any(_SQ_FILE[s] == 3 for s in board.pieces(chess.PAWN, chess.BLACK))
    b_pawns_e = any(_SQ_FILE[s] == 4 for s in board.pieces(chess.PAWN, chess.BLACK))
    open_center = int(not (w_pawns_d and b_pawns_d)) + int(not (w_pawns_e and b_pawns_e))

    penalty = 0.0

    if king_rank == back_rank and king_file == 4:
        penalty = 1.2 * WEIGHTS["uncastled_king"] + 0.7 * open_center * WEIGHTS["uncastled_king"]
    elif king_rank == back_rank and king_file in (3, 5):
        penalty = 0.8 * WEIGHTS["uncastled_king"] + 0.4 * open_center * WEIGHTS["uncastled_king"]
    elif king_rank in (1, 6):
        penalty = 0.6 * WEIGHTS["uncastled_king"] + 0.3 * open_center * WEIGHTS["uncastled_king"]
    elif king_file in (2, 3, 4, 5):
        penalty = 0.35 * WEIGHTS["uncastled_king"]

    if not board.has_castling_rights(color) and king_sq not in (chess.C1, chess.G1, chess.C8, chess.G8):
        penalty += 0.2 * WEIGHTS["uncastled_king"]

    return -penalty


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

    # FIX #7: Mobility — in-place turn flip, no board copy
    mob = WEIGHTS["mobility"]
    w_mob = b_mob = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece and piece.piece_type in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
            attacks = len(board.attacks(sq))
            if piece.color == chess.WHITE:
                w_mob += attacks
            else:
                b_mob += attacks
    score += mob * (w_mob - b_mob)

    # Bishop pair
    bp = WEIGHTS["bishop_pair"]
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2: score += bp
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2: score -= bp

    # Pawn structure
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

    # Doubled rooks on file (two rooks on the same file, especially open files)
    rdf = WEIGHTS["rook_doubled_file"]
    for color_side, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        rooks = list(board.pieces(chess.ROOK, color_side))
        if len(rooks) >= 2:
            rook_files = [_SQ_FILE[r] for r in rooks]
            if rook_files[0] == rook_files[1]:
                f = rook_files[0]
                # Extra bonus if it's an open file
                is_open = f not in wpf and f not in bpf
                score += sign * rdf * (2.0 if is_open else 1.0)

    # King safety
    score += _king_safety(board, chess.WHITE)
    score -= _king_safety(board, chess.BLACK)

    # Knight outposts
    score += _knight_outposts(board, chess.WHITE)
    score -= _knight_outposts(board, chess.BLACK)

    # King tropism
    score += _king_tropism(board, chess.WHITE)
    score -= _king_tropism(board, chess.BLACK)

    # Hanging pieces
    score += _hanging_pieces(board, chess.WHITE)
    score -= _hanging_pieces(board, chess.BLACK)

    # King attack pressure
    score += _king_attack_pressure(board, chess.WHITE)
    score -= _king_attack_pressure(board, chess.BLACK)

    # Pin penalty
    score += _pin_penalty(board, chess.WHITE)
    score -= _pin_penalty(board, chess.BLACK)

    # Development / early queen penalty
    score += _development_score(board, chess.WHITE)
    score -= _development_score(board, chess.BLACK)

    # Center control
    score += _center_control(board, chess.WHITE)
    score -= _center_control(board, chess.BLACK)

    # Uncastled king penalty (opening/middlegame only)
    if game_phase > 6:  # at least some minor pieces still on board
        score += _uncastled_king_penalty(board, chess.WHITE)
        score -= _uncastled_king_penalty(board, chess.BLACK)

    # Tempo
    if board.turn == chess.WHITE:
        score += WEIGHTS["tempo"]
    else:
        score -= WEIGHTS["tempo"]

    return score


# ══════════════════════════════════════════════════════════════════════════════
#  HYBRID EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _game_phase(board):
    phase = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None:
            continue
        pt = piece.piece_type
        if pt in (chess.KNIGHT, chess.BISHOP):
            phase += 1
        elif pt == chess.ROOK:
            phase += 2
        elif pt == chess.QUEEN:
            phase += 4
    return min(phase, 24) / 24.0


def _raw_material(board):
    """Raw material balance (white - black) in centipawns. No PST, no terms."""
    pv = _piece_values()
    score = 0
    for pt in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        score += len(board.pieces(pt, chess.WHITE)) * pv[pt]
        score -= len(board.pieces(pt, chess.BLACK)) * pv[pt]
    return score


def evaluate(board):
    if board.is_checkmate():
        return -float(MATE_SCORE) if board.turn == chess.WHITE else float(MATE_SCORE)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if board.is_repetition(3) or board.is_fifty_moves():
        # Draw contempt: slightly penalize the side that could avoid the draw
        return -DRAW_CONTEMPT_CP if board.turn == chess.WHITE else DRAW_CONTEMPT_CP

    hce_score = _cached_hce(board)

    nn_w = WEIGHTS["nn_weight"]
    if _HAS_NN and nn_w > 0.0:
        phase = _game_phase(board)
        gate = nn_w * WEIGHTS.get("nn_phase_boost", 1.0) * (0.35 + 0.65 * phase)
        gate = min(gate, 0.9)

        if gate > 0.02:
            nn_score = _nn_eval(board)

            # ── Material-anchored NN clamping ─────────────────────────────
            # The NN must not override clear material imbalances.
            # If the NN disagrees with raw material by more than ~1 pawn,
            # reduce its influence proportionally. This prevents the NN
            # from making moves like f5 (hanging a pawn) look acceptable.
            material = _raw_material(board)
            nn_material_gap = abs(nn_score - material)
            if nn_material_gap > 120:  # more than ~1 pawn disagreement
                # Reduce gate proportionally: 120cp gap = full gate, 
                # 360cp gap = gate/3, etc.
                gate *= min(1.0, 120.0 / nn_material_gap)

            return (1.0 - gate) * hce_score + gate * nn_score

    return hce_score


# ── Fast eval for quiescence (HCE only — speed matters here) ─────────────────
def _fast_evaluate(board):
    if board.is_checkmate():
        return -float(MATE_SCORE) if board.turn == chess.WHITE else float(MATE_SCORE)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if board.is_repetition(3) or board.is_fifty_moves():
        return -DRAW_CONTEMPT_CP if board.turn == chess.WHITE else DRAW_CONTEMPT_CP
    return _cached_hce(board)


# ══════════════════════════════════════════════════════════════════════════════
#  MOVE ORDERING
# ══════════════════════════════════════════════════════════════════════════════

def score_move(board, move, ply=0, tt_move=None, prev_move=None):
    pv = _piece_values()
    if tt_move and move == tt_move:           return 20_000
    # Countermove heuristic: if this move previously refuted prev_move, try it early
    if prev_move is not None:
        cm = _COUNTERMOVE.get((prev_move.from_square, prev_move.to_square))
        if cm == move:
            return 18_500
    if board.is_capture(move):
        victim   = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        v = pv.get(victim.piece_type,   100) if victim   else 100
        a = pv.get(attacker.piece_type, 100) if attacker else 100
        return 10_000 + v * 10 - a
    if move.promotion:                        return 9_000 + pv.get(move.promotion, 0)
    if move == _KILLERS[ply % 64][0]:              return 8_000
    if move == _KILLERS[ply % 64][1]:              return 7_000
    # FIX #10: only check for giving check at root (ply==0), not every node
    if ply == 0 and board.gives_check(move):
        return 11_000
    return _HISTORY.get((move.from_square, move.to_square), 0)


# FIX #9: only use policy at root (root_policy=True), never at ply <= 1
def order_moves(board, moves, ply=0, tt_move=None, root_policy=False, prev_move=None):
    """
    Optimized move ordering. 
    At depth 3, we minimize dictionary lookups and restrict NN usage to the root.
    """
    pw = WEIGHTS["policy_weight"]
    
    # Pre-calculating scores in a list is generally faster than 
    # repeated lambda calls inside sorted() in high-frequency functions.
    scored_moves = []
    
    # We only use the expensive NN bias if we are at the root (ply 0).
    # Even if root_policy is passed as True, we guard it with ply == 0.
    use_policy = (pw > 0.0 and root_policy and ply == 0 and _HAS_POLICY)

    for m in moves:
        # 1. Start with the base heuristic score (MVV-LVA, Killers, etc.)
        score = score_move(board, m, ply, tt_move, prev_move)
        
        # 2. Add NN policy boost ONLY at the root
        if use_policy:
            score += pw * _root_policy_bias(board, m)
            
        scored_moves.append((score, m))

    # Sort based on the calculated scores
    scored_moves.sort(key=lambda x: x[0], reverse=True)
    
    # Return just the moves
    return [m for _, m in scored_moves]


# ══════════════════════════════════════════════════════════════════════════════
#  QUIESCENCE SEARCH
# ══════════════════════════════════════════════════════════════════════════════

def _mvv_lva_score(board, move):
    """MVV-LVA score for capture ordering; also used for bad capture filtering."""
    victim = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    v = _piece_values().get(victim.piece_type, 100) if victim else 100
    a = _piece_values().get(attacker.piece_type, 100) if attacker else 100
    return v * 10 - a


def qsearch(board, alpha, beta, maximizing, qs_depth=0):
    # Hard limit on qsearch recursion (checks can cause deep chains)
    if qs_depth >= 12:
        return _fast_evaluate(board)

    in_check = board.is_check()
    stand_pat = _fast_evaluate(board)

    if not in_check:
        if maximizing:
            if stand_pat >= beta:
                return beta
            # Delta pruning: if even capturing the queen can't raise alpha, prune
            if stand_pat + WEIGHTS["queen"] + 75 < alpha:
                return alpha
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            if stand_pat - WEIGHTS["queen"] - 75 > beta:
                return beta
            beta = min(beta, stand_pat)

    # In check: must consider all legal moves (evasions)
    if in_check:
        tactical = order_moves(board, list(board.legal_moves))
    else:
        # Generates much faster by skipping quiet move validation entirely
        tactical = [m for m in board.legal_moves if board.is_capture(m) or m.promotion == chess.QUEEN]
        tactical = order_moves(board, tactical)

    for move in tactical:
        # Bad capture filtering: skip captures where we lose material
        # (e.g., queen takes defended pawn) unless it gives check
        if (not in_check and board.is_capture(move)
                and not move.promotion
                and _mvv_lva_score(board, move) < -BAD_CAPTURE_MARGIN
                and not board.gives_check(move)):
            continue

        board.push(move)
        score = qsearch(board, alpha, beta, not maximizing, qs_depth + 1)
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


# ══════════════════════════════════════════════════════════════════════════════
#  MINIMAX WITH ALPHA-BETA
# ══════════════════════════════════════════════════════════════════════════════

def minimax(board, depth, alpha, beta, maximizing, ply=0,
            prev_move=None, last_capture_sq=None, allow_null=True):
    orig_alpha = alpha
    orig_beta = beta
    alpha = max(alpha, -MATE_SCORE + ply)
    beta = min(beta, MATE_SCORE - ply)
    if alpha >= beta:
        return alpha

    in_check = board.is_check()
    # Check extension: add 1 ply when in check (few evasions, cheap)
    rem = depth + (1 if in_check else 0)

    hash_key = chess.polyglot.zobrist_hash(board)
    tt_move  = None
    if hash_key in TRANSPOSITION_TABLE:
        entry = TRANSPOSITION_TABLE[hash_key]
        tt_depth, tt_flag, tt_score = entry[0], entry[1], entry[2]
        tt_move = entry[3] if len(entry) > 3 else None
        if tt_depth >= rem:
            if   tt_flag == TT_EXACT: return tt_score
            elif tt_flag == TT_LOWER: alpha = max(alpha, tt_score)
            elif tt_flag == TT_UPPER: beta  = min(beta,  tt_score)
            if alpha >= beta:         return tt_score

    # ── Mate-distance pruning ────────────────────────────────────────────
    # Prefer shorter mates: can't do better than mate-in-ply
    if board.is_checkmate():
        return -float(MATE_SCORE - ply) if board.turn == chess.WHITE else float(MATE_SCORE - ply)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if board.is_repetition(3) or board.is_fifty_moves():
        return -DRAW_CONTEMPT_CP if board.turn == chess.WHITE else DRAW_CONTEMPT_CP

    if rem <= 0:
        return qsearch(board, alpha, beta, maximizing)

    # ── Null-move pruning ────────────────────────────────────────────────
    # If "doing nothing" still fails high, this node is likely cuttable.
    # Guards: not in check, not at root, both sides have pieces (no zugzwang).
    if (NULL_MOVE_ENABLE
            and allow_null
            and rem > NULL_MOVE_MIN_DEPTH
            and ply > 0
            and not in_check
            and _has_non_pawn_material(board, board.turn)
            and _has_non_pawn_material(board, not board.turn)
            and abs(alpha) < MATE_SCORE - 1000
            and abs(beta) < MATE_SCORE - 1000):
        
        static_eval = _fast_evaluate(board)
        # Scale reduction based on depth (like team_shay does)
        R = NULL_MOVE_REDUCTION + (rem // 4) 
        
        if maximizing and static_eval >= beta - NULL_MOVE_MARGIN_CP:
            board.push(chess.Move.null())
            null_score = minimax(board, rem - 1 - R, beta - 1, beta, False, ply + 1, allow_null=False)
            board.pop()
            if null_score >= beta:
                return beta
                
        elif not maximizing and static_eval <= alpha + NULL_MOVE_MARGIN_CP:
            board.push(chess.Move.null())
            null_score = minimax(board, rem - 1 - R, alpha, alpha + 1, True, ply + 1, allow_null=False)
            board.pop()
            if null_score <= alpha:
                return alpha

    # ── Futility pruning (depth 1, not in check) ────────────────────────
    futility_eval = None
    if rem == 1 and not in_check:
        futility_eval = _fast_evaluate(board)

    moves = order_moves(board, list(board.legal_moves), ply, tt_move, prev_move=prev_move)

    # Single-reply extension
    extend_single = (len(moves) == 1) and ply < 20


    best_score = float('-inf') if maximizing else float('inf')
    best_move  = None
    quiet_count = 0
    quiet_tried = []   # for history malus

    if maximizing:
        for move_idx, move in enumerate(moves):
            is_capture = board.is_capture(move)
            is_quiet = not is_capture and not move.promotion

            # ── Futility pruning: skip hopeless quiet moves at depth 1 ──
            if (futility_eval is not None and is_quiet
                    and not board.gives_check(move)
                    and futility_eval + FUTILITY_MARGIN_CP <= alpha):
                quiet_count += 1
                continue

            # ── LMP: skip late quiet moves at shallow depth ──────────────
            if (is_quiet and not in_check
                    and rem in LMP_CUTOFFS
                    and quiet_count >= LMP_CUTOFFS[rem]
                    and not board.gives_check(move)):
                continue

            # ── Recapture extension ──────────────────────────────────────
            recapture = (last_capture_sq is not None
                         and is_capture
                         and move.to_square == last_capture_sq)
            child_depth = rem - 1 + (1 if recapture else 0)
            if extend_single:
                child_depth = rem  # don't decrement for forced moves

            cap_sq = move.to_square if is_capture else None

            # ── LMR: reduce depth for late quiet moves ───────────────────
            do_lmr = (rem >= LMR_MIN_DEPTH
                       and move_idx >= LMR_START_MOVE
                       and is_quiet
                       and not in_check
                       and not recapture)

            board.push(move)

            if move_idx == 0:
                # First move: full window
                score = minimax(board, child_depth, alpha, beta, False,
                                ply + 1, move, cap_sq, allow_null=True)
            elif do_lmr:
                # LMR: reduced null-window search
                reduced = max(0, child_depth - LMR_REDUCTION)
                score = minimax(board, reduced, alpha, alpha + 1, False,
                                ply + 1, move, cap_sq, allow_null=True)
                if score > alpha:
                    # Re-search at full depth + full window
                    score = minimax(board, child_depth, alpha, beta, False,
                                    ply + 1, move, cap_sq, allow_null=True)
            else:
                # PVS: null-window probe
                score = minimax(board, child_depth, alpha, alpha + 1, False,
                                ply + 1, move, cap_sq, allow_null=True)
                if alpha < score < beta:
                    score = minimax(board, child_depth, alpha, beta, False,
                                    ply + 1, move, cap_sq, allow_null=True)

            board.pop()

            if is_quiet:
                quiet_count += 1
                quiet_tried.append(move)

            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, score)
            if beta <= alpha:
                # ── Cutoff: update killers, history, countermove ──────────
                if is_quiet:
                    if move != _KILLERS[ply % 64][0]:
                        _KILLERS[ply % 64][1] = _KILLERS[ply % 64][0]
                        _KILLERS[ply % 64][0] = move
                    _update_history(move, rem * rem)
                    # History malus: penalize quiet moves that didn't cause cutoff
                    malus = max(1, rem)
                    for qm in quiet_tried:
                        if qm != move:
                            _update_history(qm, -malus)
                    # Countermove: remember this move as the refutation
                    if prev_move is not None:
                        _COUNTERMOVE[(prev_move.from_square, prev_move.to_square)] = move
                break
    else:
        for move_idx, move in enumerate(moves):
            is_capture = board.is_capture(move)
            is_quiet = not is_capture and not move.promotion

            if (futility_eval is not None and is_quiet
                    and not board.gives_check(move)
                    and futility_eval - FUTILITY_MARGIN_CP >= beta):
                quiet_count += 1
                continue

            if (is_quiet and not in_check
                    and rem in LMP_CUTOFFS
                    and quiet_count >= LMP_CUTOFFS[rem]
                    and not board.gives_check(move)):
                continue

            recapture = (last_capture_sq is not None
                         and is_capture
                         and move.to_square == last_capture_sq)
            child_depth = rem - 1 + (1 if recapture else 0)
            if extend_single:
                child_depth = rem

            cap_sq = move.to_square if is_capture else None

            do_lmr = (rem >= LMR_MIN_DEPTH
                       and move_idx >= LMR_START_MOVE
                       and is_quiet
                       and not in_check
                       and not recapture)

            board.push(move)

            if move_idx == 0:
                score = minimax(board, child_depth, alpha, beta, True,
                                ply + 1, move, cap_sq, allow_null=True)
            elif do_lmr:
                reduced = max(0, child_depth - LMR_REDUCTION)
                score = minimax(board, reduced, beta - 1, beta, True,
                                ply + 1, move, cap_sq, allow_null=True)
                if score < beta:
                    score = minimax(board, child_depth, alpha, beta, True,
                                    ply + 1, move, cap_sq, allow_null=True)
            else:
                score = minimax(board, child_depth, beta - 1, beta, True,
                                ply + 1, move, cap_sq, allow_null=True)
                if alpha < score < beta:
                    score = minimax(board, child_depth, alpha, beta, True,
                                    ply + 1, move, cap_sq, allow_null=True)

            board.pop()

            if is_quiet:
                quiet_count += 1
                quiet_tried.append(move)

            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, score)
            if beta <= alpha:
                if is_quiet:
                    if move != _KILLERS[ply][0]:
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

    # If no move was searched (all pruned), return futility eval or static eval
    if best_move is None:
        if futility_eval is not None:
            return futility_eval
        return _fast_evaluate(board)

    tt_flag = TT_EXACT
    if best_score <= orig_alpha:
        tt_flag = TT_UPPER
    elif best_score >= orig_beta:
        tt_flag = TT_LOWER
    TRANSPOSITION_TABLE[hash_key] = (rem, tt_flag, best_score, best_move)
    return best_score


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def _root_search(board, depth, maximizing, alpha, beta):
    moves = order_moves(board, list(board.legal_moves), 0, root_policy=True)
    best_score = float('-inf') if maximizing else float('inf')
    best_move = None

    for idx, move in enumerate(moves):
        cap_sq = move.to_square if board.is_capture(move) else None

        board.push(move)

        if idx == 0:
            score = minimax(board, depth - 1, alpha, beta, not maximizing,
                            1, move, cap_sq, allow_null=True)
        else:
            # PVS at root
            if maximizing:
                score = minimax(board, depth - 1, alpha, alpha + 1, False,
                                1, move, cap_sq, allow_null=True)
                if alpha < score < beta:
                    score = minimax(board, depth - 1, alpha, beta, False,
                                    1, move, cap_sq, allow_null=True)
            else:
                score = minimax(board, depth - 1, beta - 1, beta, True,
                                1, move, cap_sq, allow_null=True)
                if alpha < score < beta:
                    score = minimax(board, depth - 1, alpha, beta, True,
                                    1, move, cap_sq, allow_null=True)

        board.pop()

        if maximizing:
            if score > best_score:
                best_score, best_move = score, move
            alpha = max(alpha, best_score)
        else:
            if score < best_score:
                best_score, best_move = score, move
            beta = min(beta, best_score)

        if beta <= alpha:
            break

    return best_score, best_move


def get_next_move(board, color, depth=3):
    global TRANSPOSITION_TABLE, _KILLERS, _HISTORY, EVAL_CACHE, _COUNTERMOVE

    # ── Opening book: instant return, no search ──────────────────────────
    book_move = _opening_book_move(board)
    if book_move is not None:
        return book_move

    if len(TRANSPOSITION_TABLE) > 500_000:
        TRANSPOSITION_TABLE.clear()
    if len(_HISTORY) > 100_000:
        _HISTORY.clear()
    if len(EVAL_CACHE) > EVAL_CACHE_MAX:
        EVAL_CACHE.clear()
    if len(_COUNTERMOVE) > 80_000:
        _COUNTERMOVE.clear()

    _KILLERS[:] = [[None, None] for _ in range(64)]

    maximizing = (color == chess.WHITE)
    best_move = None
    best_score = float('-inf') if maximizing else float('inf')

    # Iterative deepening with resilient aspiration windows
    prev_score = 0.0
    for d in range(1, depth + 1):
        use_asp = d >= 2 and best_move is not None
        window = float(ASP_WINDOW)

        if use_asp:
            alpha = prev_score - window
            beta = prev_score + window
        else:
            alpha = float("-inf")
            beta = float("inf")

        retries = 0
        while True:
            score, move = _root_search(board, d, maximizing, alpha, beta)

            if move is not None:
                best_score, best_move = score, move
                prev_score = score

            if not use_asp:
                break

            if score <= alpha:
                alpha -= window
                window *= 2.0
                retries += 1
                if retries >= 5:
                    alpha = float("-inf")
                    beta = float("inf")
                    use_asp = False
                continue

            if score >= beta:
                beta += window
                window *= 2.0
                retries += 1
                if retries >= 5:
                    alpha = float("-inf")
                    beta = float("inf")
                    use_asp = False
                continue

            break

    return best_move if best_move else next(iter(board.legal_moves), None)


if __name__ == '__main__':
    b = chess.Board()
    move = get_next_move(b, chess.WHITE, depth=3)
    if move is not None:
        print(f"[team_goraieb] Opening move: {b.san(move)}")
        print(f"  NN available: {_HAS_NN}")
        print(f"  Cython: {_HAVE_CYTHON}")
        print(f"  NN weight: {WEIGHTS['nn_weight']}")