#no idea how this works, it is probably worse than my past one

import chess
import chess.polyglot
from math import log as _log
import time
import multiprocessing as mp
from functools import lru_cache

    
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
_PST_MAP = {
    chess.PAWN: PST_PAWN_MG,
    chess.KNIGHT: PST_KNIGHT_MG,
    chess.BISHOP: PST_BISHOP_MG,
    chess.ROOK: PST_ROOK_MG,
    chess.QUEEN: PST_QUEEN_MG,
    chess.KING: PST_KING_MG,  # <-- ADD THIS LINE
}
_EG_PST_MAP = {
    chess.PAWN: PST_PAWN_EG,
    chess.KNIGHT: PST_KNIGHT_EG,
    chess.BISHOP: PST_BISHOP_EG,
    chess.ROOK: PST_ROOK_EG,
    chess.QUEEN: PST_QUEEN_EG,
    chess.KING: PST_KING_EG,
}


TT = {}
KILLER_MOVES = [[None] * 2 for _ in range(100)]
HISTORY = {}
CONTEMPT = 60

PST = {
    chess.PAWN: [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ],
    chess.ROOK: [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        20, 20,  0,  0,  0,  0, 20, 20,
        20, 30, 10,  0,  0, 10, 30, 20
    ]
}


TENSOR_FEATURES = (
    "material", "king_safety", "threat", "opportunity",
    "mobility", "structure", "initiative", "risk", "opening",
)

TENSOR_WEIGHTS = {
    "material": 1.00,
    "king_safety": 1.05,
    "threat": 0.60,
    "opportunity": 0.45,
    "mobility": 0.30,
    "structure": 0.40,
    "initiative": 0.15,
    "risk": -0.80,
    "opening": 0.70,
}
_EG_PST_MAP = {
    chess.PAWN: PST_PAWN_EG,
    chess.KNIGHT: PST_KNIGHT_EG,
    chess.BISHOP: PST_BISHOP_EG,
    chess.ROOK: PST_ROOK_EG,
    chess.QUEEN: PST_QUEEN_EG,
    chess.KING: PST_KING_EG,
}

class IncrementalEval:
    def __init__(self, board: chess.Board):
        self.board = board
        self.mg_score = 0.0
        self.eg_score = 0.0
        self.phase = 0
        self._full_recalc()

    def _full_recalc(self):
        self.mg_score = 0.0
        self.eg_score = 0.0
        self.phase = 0
        for sq in chess.SQUARES:
            piece = self.board.piece_at(sq)
            if piece:
                self._add_piece(piece, sq)

    def _add_piece(self, piece: chess.Piece, sq: chess.Square):
        # CORRECT MIRRORING: White uses standard index, Black uses mirrored
        pst_idx = sq if piece.color == chess.WHITE else chess.square_mirror(sq)
        
        # Track game phase for endgame blending
        if piece.piece_type == chess.KNIGHT: self.phase += 1
        elif piece.piece_type == chess.BISHOP: self.phase += 1
        elif piece.piece_type == chess.ROOK: self.phase += 2
        elif piece.piece_type == chess.QUEEN: self.phase += 4
        
        # Look up values (Ensure these maps include the King)
        mg_val = _PST_MAP[piece.piece_type][pst_idx]
        eg_val = _EG_PST_MAP[piece.piece_type][pst_idx]
        piece_val = PIECE_VALUES[piece.piece_type]
        
        # Apply scores once
        if piece.color == chess.WHITE:
            self.mg_score += (piece_val + mg_val)
            self.eg_score += (piece_val + eg_val)
        else:
            self.mg_score -= (piece_val + mg_val)
            self.eg_score -= (piece_val + eg_val)

    def get_score(self) -> float:
        # Phase 24 is standard for a full board (excluding pawns/kings)
        mg_phase = min(self.phase, 24)
        eg_phase = 24 - mg_phase
        return (self.mg_score * mg_phase + self.eg_score * eg_phase) / 24.0


OT_FULL_DEPTH = 10
TARGET_DEPTH = 10
BOOK_PATH = None              

def _capture_gain(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_at(move.to_square)
    if victim is None and board.is_en_passant(move):
        victim_value = PIECE_VALUES[chess.PAWN]
    else:
        victim_value = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0

    attacker = board.piece_at(move.from_square)
    attacker_value = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
    return 10 * victim_value - attacker_value

def _position_tensor(board: chess.Board) -> dict:
    eg = _endgame_factor(board)
    mat = _material_and_pst(board, eg)

    king_safety = (_king_safety(board, eg) + _king_shelter(board, eg) + _king_center_penalty(board, eg) + _castling_bonus(board) + 0.25 * _king_activity_endgame(board, eg) + 0.25 * _king_opposition(board, eg))
    threat = (_hanging_pieces(board) + _fork_evaluation(board) + _fork_and_threat_evaluation(board) + _king_attack_zone(board, eg))
    opportunity = (_pawn_storm(board, eg) + _queen_tropism(board, eg) + _rule_of_square(board, eg) + _king_proximity_mop_up(board, eg, mat))
    mobility = _mobility(board) + _space_advantage(board, eg)
    structure = (_pawn_structure(board) + _pawn_chain_bonus(board) + _bishop_pair(board) + _rook_open_files(board) + _knight_outposts(board) + _rook_on_seventh(board) + _connected_rooks(board) +  _fifty_move_pressure(board))

    initiative = 12.0 if board.turn == chess.WHITE else -12.0
    if board.is_check():
        initiative += 18.0 if board.turn == chess.BLACK else -18.0

    risk = (_trapped_pieces(board) + _hanging_pieces(board) - _king_safety(board, eg))
    opening = _opening_factor(board) * _opening_eval(board)

    return {
        "material": float(mat), "king_safety": float(king_safety), "threat": float(threat),
        "opportunity": float(opportunity), "mobility": float(mobility), "structure": float(structure),
        "initiative": float(initiative), "risk": float(risk), "opening": float(opening),
    }


def _tensor_scalarize(t: dict) -> float:
    return sum(TENSOR_WEIGHTS[k] * t.get(k, 0.0) for k in TENSOR_WEIGHTS)

def _ot_goodness(tensor: dict, maximizing: bool) -> dict:
    side = 1.0 if maximizing else -1.0
    return {k: (side if k != "risk" else -side) * tensor.get(k, 0.0) for k in TENSOR_WEIGHTS}

def _ot_violation_vector(before: dict, after: dict, maximizing: bool):
    b = _ot_goodness(before, maximizing)
    a = _ot_goodness(after, maximizing)
    return tuple(max(0.0, b[k] - a[k]) for k in TENSOR_WEIGHTS)

def _ot_score(board: chess.Board, move: chess.Move, before_tensor: dict, maximizing: bool, tt_move=None, ply: int = 0, prev_move=None) -> float:
    if move == tt_move: return -10**15
    board.push(move)
    after_tensor = _position_tensor(board)
    board.pop()

    viol = _ot_violation_vector(before_tensor, after_tensor, maximizing)
    score = (viol[0] * 1_000_000_000 + viol[1] * 10_000_000 + viol[2] * 100_000 + viol[3] * 1_000 + viol[4] * 100 + viol[5] * 10 + viol[6] * 5 + viol[7] * 1 + viol[8] * 500_000)

    if _opening_factor(board) > 0.3 and _is_rook_anti_development_move(board, move): score += 200_000_000
    if _opening_factor(board) > 0.3 and _is_king_weakening_move(board, move): score += 150_000_000
    if _opening_factor(board) > 0.3 and _is_development_move(board, move): score -= 25_000_000

    killer0, killer1 = _KILLERS[ply] if ply < 128 else (None, None)
    if move == killer0 or move == killer1: score -= 20_000_000

    if board.is_capture(move): score -= 250_000_000 + _capture_gain(board, move) * 1000
    if board.gives_check(move): score -= 100_000_000
    if move.promotion == chess.QUEEN: score -= 80_000_000

    if prev_move is not None:
        cm = _COUNTERMOVE.get((prev_move.from_square, prev_move.to_square))
        if cm is not None and move == cm: score -= 15_000_000

    hist_key = (board.turn, move.from_square, move.to_square)
    score -= min(_HISTORY.get(hist_key, 0), 50_000)
    
    return _HISTORY.get((board.turn, move.from_square, move.to_square), 0)

def _legacy_move_score(board: chess.Board, move: chess.Move, tt_move=None, ply: int = 0, prev_move=None) -> int:
    if move == tt_move: return 30_000
    if _opening_factor(board) > 0.3 and _is_rook_anti_development_move(board, move): return -50_000
    if _opening_factor(board) > 0.3 and _is_king_weakening_move(board, move): return -45_000
    if _opening_factor(board) > 0.3 and _is_development_move(board, move): return _HISTORY.get((board.turn, move.from_square, move.to_square), 0) + 3_000

    if board.is_capture(move):
        gain = _capture_gain(board, move)
        return (20_000 + gain) if gain >= 0 else (-10_000 + gain)

    if move.promotion == chess.QUEEN: return 19_000

    ks = _KILLERS[ply] if ply < 128 else [None, None]
    if move in {k for k in ks if k is not None}: return 10_000

    cm = None
    if prev_move is not None: cm = _COUNTERMOVE.get((prev_move.from_square, prev_move.to_square))
    if cm is not None and move == cm: return 9_000

    return _HISTORY.get((board.turn, move.from_square, move.to_square), 0)

    
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

_PST_MAP = {
    chess.PAWN: PST_PAWN_MG,
    chess.KNIGHT: PST_KNIGHT_MG,
    chess.BISHOP: PST_BISHOP_MG,
    chess.ROOK: PST_ROOK_MG,
    chess.QUEEN: PST_QUEEN_MG,
    chess.KING: PST_KING_MG,  # <-- ADD THIS LINE
}

_ROOT_COLOR = chess.WHITE

def _contempt_score(root_color=None) -> float:
    return -CONTEMPT if (root_color or _ROOT_COLOR) == chess.WHITE else CONTEMPT

def _chebyshev(sq1: int, sq2: int) -> int:
    return max(abs(chess.square_file(sq1) - chess.square_file(sq2)),abs(chess.square_rank(sq1) - chess.square_rank(sq2)))

def _manhattan(sq1: int, sq2: int) -> int:
    return (abs(chess.square_file(sq1) - chess.square_file(sq2)) + abs(chess.square_rank(sq1) - chess.square_rank(sq2)))

def _endgame_factor(board: chess.Board) -> float:
    phase = len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
    phase += len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK))
    phase += 2 * len(board.pieces(chess.ROOK, chess.WHITE)) + 2 * len(board.pieces(chess.ROOK, chess.BLACK))
    phase += 4 * len(board.pieces(chess.QUEEN, chess.WHITE)) + 4 * len(board.pieces(chess.QUEEN, chess.BLACK))
    return 1.0 - min(phase, 24) / 24.0

def _opening_factor(board: chess.Board) -> float:
    fm = board.fullmove_number
    if fm <= 10: return 1.0
    if fm >= 16: return 0.0
    return 1.0 - (fm - 10) / 6.0

_ROOK_ANTI_SQUARES_WHITE = frozenset({chess.A3, chess.B3, chess.G3, chess.H3})
_ROOK_ANTI_SQUARES_BLACK = frozenset({chess.A6, chess.B6, chess.G6, chess.H6})
_KNIGHT_DEVELOPMENT_WHITE = frozenset({chess.C3, chess.F3, chess.C4, chess.F4, chess.D2, chess.E2})
_KNIGHT_DEVELOPMENT_BLACK = frozenset({chess.C6, chess.F6, chess.C5, chess.F5, chess.D7, chess.E7})

def _is_rook_anti_development_move(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.ROOK: return False
    to_sq = move.to_square
    if piece.color == chess.WHITE and to_sq in _ROOK_ANTI_SQUARES_WHITE: return True
    if piece.color == chess.BLACK and to_sq in _ROOK_ANTI_SQUARES_BLACK: return True
    return False

def _is_development_move(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if not piece: return False
    to_sq = move.to_square
    to_r, to_f = chess.square_rank(to_sq), chess.square_file(to_sq)
    color = piece.color
    back_rank = 0 if color == chess.WHITE else 7
    from_r = chess.square_rank(move.from_square)

    if piece.piece_type == chess.KNIGHT and from_r == back_rank:
        return (color == chess.WHITE and to_sq in (chess.C3, chess.F3, chess.C4, chess.F4)) or \
               (color == chess.BLACK and to_sq in (chess.C6, chess.F6, chess.C5, chess.F5))
    if piece.piece_type == chess.BISHOP and from_r == back_rank: return to_r != back_rank
    if piece.piece_type == chess.PAWN and to_f in (3, 4): return True
    return board.is_castling(move)

def _is_king_weakening_move(board: chess.Board, move: chess.Move) -> bool:
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.PAWN: return False
    color = piece.color
    ksq = board.king(color)
    if ksq is None or chess.square_file(ksq) in (2, 6): return False
    from_f = chess.square_file(move.from_square)
    from_r = chess.square_rank(move.from_square)
    return (color == chess.WHITE and from_r == 1 and from_f in (5, 6, 7)) or \
           (color == chess.BLACK and from_r == 6 and from_f in (5, 6, 7))

def _opening_eval(board: chess.Board) -> float:
    score = 0.0
    if _opening_factor(board) <= 0.0: return 0.0

    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        for sq in board.pieces(chess.ROOK, color):
            if (color == chess.WHITE and sq in _ROOK_ANTI_SQUARES_WHITE) or (color == chess.BLACK and sq in _ROOK_ANTI_SQUARES_BLACK):
                score -= sign * 120
        for sq in board.pieces(chess.KNIGHT, color):
            if (color == chess.WHITE and sq in _KNIGHT_DEVELOPMENT_WHITE) or (color == chess.BLACK and sq in _KNIGHT_DEVELOPMENT_BLACK):
                score += sign * 25
            r = chess.square_rank(sq)
            if (color == chess.WHITE and r == 0) or (color == chess.BLACK and r == 7): score -= sign * 15
        back_rank = 0 if color == chess.WHITE else 7
        for sq in board.pieces(chess.BISHOP, color):
            if chess.square_rank(sq) != back_rank: score += sign * 18
        for sq in board.pieces(chess.PAWN, color):
            if chess.square_file(sq) in (3, 4): score += sign * 12

        ksq = board.king(color)
        if ksq is not None and not board.has_castling_rights(color):
            if chess.square_file(ksq) in (2, 6): score += sign * 35
        if board.has_castling_rights(color): score += sign * 12

        if board.has_castling_rights(color):
            shield_rank = 2 if color == chess.WHITE else 7
            for f in (5, 6, 7):
                p = board.piece_at(chess.square(f, shield_rank))
                if not (p and p.piece_type == chess.PAWN and p.color == color): score -= sign * 90
    return score

def _get_pst(piece_type, eg):
    mg_weight = 1.0 - eg
    if piece_type == chess.PAWN: return [(PST_PAWN_MG[i]*mg_weight + PST_PAWN_EG[i]*eg) for i in range(64)]
    elif piece_type == chess.KNIGHT: return [(PST_KNIGHT_MG[i]*mg_weight + PST_KNIGHT_EG[i]*eg) for i in range(64)]
    elif piece_type == chess.BISHOP: return [(PST_BISHOP_MG[i]*mg_weight + PST_BISHOP_EG[i]*eg) for i in range(64)]
    elif piece_type == chess.ROOK: return [(PST_ROOK_MG[i]*mg_weight + PST_ROOK_EG[i]*eg) for i in range(64)]
    elif piece_type == chess.QUEEN: return [(PST_QUEEN_MG[i]*mg_weight + PST_QUEEN_EG[i]*eg) for i in range(64)]
    else: return [(PST_KING_MG[i]*mg_weight + PST_KING_EG[i]*eg) for i in range(64)]

def _evaluate_draw(board: chess.Board, current_eval: float, root_color=None) -> float:
    if root_color is None: root_color = _ROOT_COLOR
    if not (board.is_stalemate() or board.is_repetition(2) or board.is_fifty_moves() or board.is_insufficient_material()):
        return 0.0
    perspective_eval = current_eval if root_color == chess.WHITE else -current_eval
    if perspective_eval > 50: return -99000 if root_color == chess.WHITE else 99000
    elif perspective_eval < -50: return 99000 if root_color == chess.WHITE else -99000
    return 0.0

def _material_and_pst(board: chess.Board, eg: float) -> float:
    score = 0.0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None: continue
        sign = 1 if piece.color == chess.WHITE else -1
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
        sign = 1 if color == chess.WHITE else -1
        enemy = not color
        pawns = board.pieces(chess.PAWN, color)
        epawns = board.pieces(chess.PAWN, enemy)
        files = [chess.square_file(sq) for sq in pawns]

        file_counts = [0] * 8
        for sq in pawns: file_counts[chess.square_file(sq)] += 1

        for sq in pawns:
            f, rank = chess.square_file(sq), chess.square_rank(sq)
            if file_counts[f] > 1: score += sign * (-20)
            adj = [a for a in (f - 1, f + 1) if 0 <= a <= 7]
            if not any(a in files for a in adj): score += sign * (-15)

            passed = True
            for esq in epawns:
                ef, er = chess.square_file(esq), chess.square_rank(esq)
                if ef in (f - 1, f, f + 1):
                    if (color == chess.WHITE and er >= rank) or (color == chess.BLACK and er <= rank):
                        passed = False; break

            if passed:
                advancement = rank if color == chess.WHITE else (7 - rank)
                base_bonus = 20 + advancement * 10
                prot_rank = rank - 1 if color == chess.WHITE else rank + 1
                if 0 <= prot_rank <= 7:
                    for df in (-1, 1):
                        pf = f + df
                        if 0 <= pf <= 7:
                            p = board.piece_at(chess.square(pf, prot_rank))
                            if p and p.piece_type == chess.PAWN and p.color == color: base_bonus += 15; break
                score += sign * base_bonus

            support_ranks = range(0, rank) if color == chess.WHITE else range(rank + 1, 8)
            has_support = any(chess.square_file(s) in (f - 1, f + 1) and chess.square_rank(s) in support_ranks for s in pawns)
            if not has_support and not passed: score += sign * (-10)
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
                    if 0 <= pf <= 7:
                        p = board.piece_at(chess.square(pf, prot_rank))
                        if p and p.piece_type == chess.PAWN and p.color == color: score += sign * 7; break
    return score

def _mobility(board: chess.Board) -> float:
    w = b = 0
    for sq, piece in board.piece_map().items():
        if piece.color == chess.WHITE: w += len(board.attacks(sq))
        else: b += len(board.attacks(sq))
    return 0.1 * (w - b)

def _king_safety(board: chess.Board, eg: float) -> float:
    if eg >= 0.9: return 0.0
    mg_weight, score = 1.0 - eg, 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign, king_sq = (1 if color == chess.WHITE else -1), board.king(color)
        if not king_sq: continue
        kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
        for df in (-1, 0, 1):
            f = kf + df
            if not (0 <= f <= 7): continue
            shield_r = kr + (1 if color == chess.WHITE else -1)
            if 0 <= shield_r <= 7:
                p = board.piece_at(chess.square(f, shield_r))
                if p and p.piece_type == chess.PAWN and p.color == color: score += sign * 10 * mg_weight
            if not any(chess.square_file(s) == f for s in board.pieces(chess.PAWN, color)): score += sign * (-10) * mg_weight
    return score

def _pawn_storm(board: chess.Board, eg: float) -> float:
    if eg >= 0.75: return 0.0
    mg_weight, score = 1.0 - eg, 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign, ek_sq = (1 if color == chess.WHITE else -1), board.king(not color)
        if not ek_sq: continue
        ekf = chess.square_file(ek_sq)
        for sq in board.pieces(chess.PAWN, color):
            pf, pr = chess.square_file(sq), chess.square_rank(sq)
            if abs(pf - ekf) > 2: continue
            advancement = pr if color == chess.WHITE else (7 - pr)
            if advancement >= 4: score += sign * (advancement - 3) * 10 * mg_weight
    return score

def _king_shelter(board: chess.Board, eg: float) -> float:
    if eg >= 0.85: return 0.0
    mg_weight, score = 1.0 - eg, 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign, ksq = (1 if color == chess.WHITE else -1), board.king(color)
        if not ksq or (chess.square_file(ksq) in (2, 6) and chess.square_rank(ksq) in (0, 7)): continue
        shield_rank = 2 if color == chess.WHITE else 7
        for f in (5, 6, 7):
            p = board.piece_at(chess.square(f, shield_rank))
            if p and p.piece_type == chess.PAWN and p.color == color: score += sign * 35 * mg_weight
            else: score -= sign * 65 * mg_weight
    return score

def _king_center_penalty(board: chess.Board, eg: float) -> float:
    if eg >= 0.6: return 0.0
    mg_weight, score = 1.0 - eg, 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign, ksq = (1 if color == chess.WHITE else -1), board.king(color)
        if not ksq: continue
        kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
        if kf not in (2, 6) and kf in (3, 4) and kr in (2, 3, 4, 5): score -= sign * 35 * mg_weight
    return score

def _castling_bonus(board: chess.Board) -> float:
    score = 0.0
    if board.has_castling_rights(chess.WHITE): score += 30.0
    if board.has_castling_rights(chess.BLACK): score -= 30.0
    return score

def _bishop_pair(board: chess.Board) -> float:
    score = 0.0
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2: score += 30.0
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2: score -= 30.0
    return score

def _rook_open_files(board: chess.Board) -> float:
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        for sq in board.pieces(chess.ROOK, color):
            f = chess.square_file(sq)
            own_f = sum(1 for s in board.pieces(chess.PAWN, color) if chess.square_file(s) == f)
            enemy_f = sum(1 for s in board.pieces(chess.PAWN, not color) if chess.square_file(s) == f)
            if own_f == 0 and enemy_f == 0: score += sign * 25
            elif own_f == 0: score += sign * 10
    return score

def _knight_outposts(board: chess.Board) -> float:
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign, epawns = (1 if color == chess.WHITE else -1), board.pieces(chess.PAWN, not color)
        for sq in board.pieces(chess.KNIGHT, color):
            rank, f = chess.square_rank(sq), chess.square_file(sq)
            if (color == chess.WHITE and rank not in (3, 4, 5)) or (color == chess.BLACK and rank not in (2, 3, 4)): continue
            pawn_rank = rank - 1 if color == chess.WHITE else rank + 1
            if not (0 <= pawn_rank <= 7): continue
            protected = any(chess.square_file(s) in (f - 1, f + 1) and chess.square_rank(s) == pawn_rank for s in board.pieces(chess.PAWN, color))
            if not protected: continue
            attack_rank = rank + 1 if color == chess.WHITE else rank - 1
            attackable = (0 <= attack_rank <= 7) and any(chess.square_file(s) in (f - 1, f + 1) and chess.square_rank(s) == attack_rank for s in epawns)
            if not attackable: score += sign * 20
    return score

def _rook_on_seventh(board: chess.Board) -> float:
    score = 0.0
    for sq in board.pieces(chess.ROOK, chess.WHITE):
        if chess.square_rank(sq) == 6: score += 25
    for sq in board.pieces(chess.ROOK, chess.BLACK):
        if chess.square_rank(sq) == 1: score -= 25
    return score

def _connected_rooks(board: chess.Board) -> float:
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color == chess.WHITE else -1
        rooks = list(board.pieces(chess.ROOK, color))
        if len(rooks) < 2: continue
        r1, r2 = rooks[0], rooks[1]
        f1, rk1, f2, rk2 = chess.square_file(r1), chess.square_rank(r1), chess.square_file(r2), chess.square_rank(r2)
        if f1 == f2 and all(board.piece_at(chess.square(f1, r)) is None for r in range(min(rk1, rk2) + 1, max(rk1, rk2))): score += sign * 15
        elif rk1 == rk2 and all(board.piece_at(chess.square(f, rk1)) is None for f in range(min(f1, f2) + 1, max(f1, f2))): score += sign * 15
    return score

def _hanging_pieces(board: chess.Board) -> float:
    score = 0.0
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.KING: continue
        enemy = not piece.color
        if board.is_attacked_by(enemy, sq) and len(board.attackers(piece.color, sq)) < len(board.attackers(enemy, sq)):
            score += (-1 if piece.color == chess.WHITE else 1) * PIECE_VALUES[piece.piece_type] * 0.25
    return score

def _fork_evaluation(board: chess.Board) -> float:
    score = 0.0
    FORK_PIECES = {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN}
    VALUABLE = {chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING}

    for sq, piece in board.piece_map().items():
        if piece.piece_type not in FORK_PIECES: continue
        sign, enemy = (1 if piece.color == chess.WHITE else -1), not piece.color
        attacked_vals = [PIECE_VALUES[tp.piece_type] for tsq in board.attacks(sq) if (tp := board.piece_at(tsq)) and tp.color == enemy and tp.piece_type in VALUABLE]
        if len(attacked_vals) >= 2: score += sign * sum(sorted(attacked_vals, reverse=True)[:2]) * 0.10

    for sq, piece in board.piece_map().items():
        if piece.piece_type != chess.KNIGHT: continue
        sign, enemy = (1 if piece.color == chess.WHITE else -1), not piece.color
        for tsq in chess.SquareSet(chess.BB_KNIGHT_ATTACKS[sq]):
            occupant = board.piece_at(tsq)
            if occupant and occupant.color == piece.color: continue
            future_vals = [PIECE_VALUES[tp.piece_type] for atsq in chess.SquareSet(chess.BB_KNIGHT_ATTACKS[tsq]) if atsq != sq and (tp := board.piece_at(atsq)) and tp.color == enemy and tp.piece_type in VALUABLE]
            if len(future_vals) >= 2: score += sign * sum(sorted(future_vals, reverse=True)[:2]) * (0.06 if not occupant else 0.03)
    return score

def _king_proximity_mop_up(board: chess.Board, eg: float, material_score: float) -> float:
    if eg < 0.4: return 0.0
    wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
    if not wk or not bk: return 0.0
    result = 0.0

    if material_score > 200:
        bkf, bkr = chess.square_file(bk), chess.square_rank(bk)
        result += eg * (min(bkf, 7 - bkf, bkr, 7 - bkr) * 20 + min(bkf + bkr, (7-bkf) + bkr, bkf + (7-bkr), (7-bkf) + (7-bkr)) * 12 + (14 - _chebyshev(wk, bk)) * 7)
        for q_sq in board.pieces(chess.QUEEN, chess.WHITE): result += eg * (7 - _chebyshev(q_sq, bk)) * 6
        for r_sq in board.pieces(chess.ROOK, chess.WHITE):
            if chess.square_file(r_sq) == bkf or chess.square_rank(r_sq) == bkr: result += eg * 40
            result += eg * (14 - _manhattan(r_sq, bk)) * 3
        for pt in (chess.BISHOP, chess.KNIGHT):
            for sq in board.pieces(pt, chess.WHITE): result += eg * (7 - _chebyshev(sq, bk)) * 2

    elif material_score < -200:
        wkf, wkr = chess.square_file(wk), chess.square_rank(wk)
        result -= eg * (min(wkf, 7 - wkf, wkr, 7 - wkr) * 20 + min(wkf + wkr, (7-wkf) + wkr, wkf + (7-wkr), (7-wkf) + (7-wkr)) * 12 + (14 - _chebyshev(wk, bk)) * 7)
        for q_sq in board.pieces(chess.QUEEN, chess.BLACK): result -= eg * (7 - _chebyshev(q_sq, wk)) * 6
        for r_sq in board.pieces(chess.ROOK, chess.BLACK):
            if chess.square_file(r_sq) == wkf or chess.square_rank(r_sq) == wkr: result -= eg * 40
            result -= eg * (14 - _manhattan(r_sq, wk)) * 3
        for pt in (chess.BISHOP, chess.KNIGHT):
            for sq in board.pieces(pt, chess.BLACK): result -= eg * (7 - _chebyshev(sq, wk)) * 2

    wkf, wkr = chess.square_file(wk), chess.square_rank(wk)
    bkf, bkr = chess.square_file(bk), chess.square_rank(bk)
    king_dist = abs(wkf - bkf) + abs(wkr - bkr)
    def get_corner_dist(f, r): return min(f + r, (7 - f) + r, f + (7 - r), (7 - f) + (7 - r))

    if material_score > 300: result += eg * ((14 - king_dist) * 4 + get_corner_dist(bkf, bkr) * 8)
    elif material_score < -300: result -= eg * ((14 - king_dist) * 4 + get_corner_dist(wkf, wkr) * 8)
    return result

_KA_WEIGHTS = {chess.KNIGHT: 2, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 5}

def _rule_of_square(board: chess.Board, eg: float) -> float:
    if eg < 0.55: return 0.0
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        sign, enemy = (1 if color == chess.WHITE else -1), not color
        ek = board.king(enemy)
        if not ek or any(len(board.pieces(pt, enemy)) > 0 for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)): continue
        for sq in board.pieces(chess.PAWN, color):
            f, rank = chess.square_file(sq), chess.square_rank(sq)
            passed = True
            for esq in board.pieces(chess.PAWN, enemy):
                ef, er = chess.square_file(esq), chess.square_rank(esq)
                if abs(ef - f) <= 1 and ((color == chess.WHITE and er >= rank) or (color == chess.BLACK and er <= rank)): passed = False; break
            if not passed: continue
            promo_rank, steps = (7 if color == chess.WHITE else 0), abs((7 if color == chess.WHITE else 0) - rank)
            if _chebyshev(ek, chess.square(f, promo_rank)) > (steps if board.turn == color else steps + 1):
                score += sign * min(400 + (7 - steps) * 60, 800) * eg
    return score

def _king_activity_endgame(board: chess.Board, eg: float) -> float:
    if eg < 0.45: return 0.0
    score = 0.0
    for color in (chess.WHITE, chess.BLACK):
        ksq = board.king(color)
        if ksq: score += (1 if color == chess.WHITE else -1) * (7 - (abs(chess.square_file(ksq) - 3.5) + abs(chess.square_rank(ksq) - 3.5))) * 5 * eg
    return score

def _king_opposition(board: chess.Board, eg: float) -> float:
    if eg < 0.60: return 0.0
    wk, bk = board.king(chess.WHITE), board.king(chess.BLACK)
    if not wk or not bk: return 0.0
    wkf, wkr, bkf, bkr = chess.square_file(wk), chess.square_rank(wk), chess.square_file(bk), chess.square_rank(bk)
    if (wkf == bkf and abs(wkr - bkr) == 2) or (wkr == bkr and abs(wkf - bkf) == 2):
        return (30 * eg) if board.turn == chess.BLACK else (-30 * eg)
    return 0.0

def _king_attack_zone(board: chess.Board, eg: float) -> float:
    if eg > 0.80: return 0.0
    mg_weight, score = 1.0 - eg, 0.0
    for defender_color in (chess.WHITE, chess.BLACK):
        defender_sign, attacker_color, king_sq = (1 if defender_color == chess.WHITE else -1), not defender_color, board.king(defender_color)
        if not king_sq: continue
        kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
        zone_set = chess.SquareSet([chess.square(kf + df, kr + dr) for df in (-1, 0, 1) for dr in (-1, 0, 1) if 0 <= kf + df <= 7 and 0 <= kr + dr <= 7])
        attack_units, attacker_count = 0, 0
        for sq, piece in board.piece_map().items():
            if piece.color != attacker_color or not (w := _KA_WEIGHTS.get(piece.piece_type, 0)): continue
            atk = board.attacks(sq) & zone_set
            if atk:
                attacker_count += 1
                attack_units += w * len(atk)
        if attacker_count >= 3: danger = attack_units * attacker_count * 1.2
        elif attacker_count == 2: danger = attack_units * 1.6
        else: danger = attack_units * 0.6
        score -= defender_sign * danger * mg_weight * 1.8
    return score

_WHITE_SPACE_SQS = chess.SquareSet([chess.C2, chess.D2, chess.E2, chess.F2, chess.C3, chess.D3, chess.E3, chess.F3, chess.C4, chess.D4, chess.E4, chess.F4])
_BLACK_SPACE_SQS = chess.SquareSet([chess.C5, chess.D5, chess.E5, chess.F5, chess.C6, chess.D6, chess.E6, chess.F6, chess.C7, chess.D7, chess.E7, chess.F7])

def _space_advantage(board: chess.Board, eg: float) -> float:
    if eg > 0.65: return 0.0
    w_space, b_space = 0, 0
    for sq, piece in board.piece_map().items():
        if piece.color == chess.WHITE: w_space += len(board.attacks(sq) & _BLACK_SPACE_SQS)
        else: b_space += len(board.attacks(sq) & _WHITE_SPACE_SQS)
    return (w_space - b_space) * 1.5 * (1.0 - eg)

def _trapped_pieces(board: chess.Board) -> float:
    score = 0.0
    for sq, piece in board.piece_map().items():
        if piece.piece_type not in (chess.BISHOP, chess.ROOK): continue
        mobility = len(board.attacks(sq))
        if piece.piece_type == chess.BISHOP and mobility <= 2: score += (1 if piece.color == chess.WHITE else -1) * (-30)
        elif piece.piece_type == chess.ROOK and mobility <= 3: score += (1 if piece.color == chess.WHITE else -1) * (-25)
    return score

def _queen_tropism(board: chess.Board, eg: float) -> float:
    if eg > 0.70: return 0.0
    mg_weight, score = 1.0 - eg, 0.0
    for color in (chess.WHITE, chess.BLACK):
        ek_sq = board.king(not color)
        if not ek_sq: continue
        ekf, ekr = chess.square_file(ek_sq), chess.square_rank(ek_sq)
        for sq in board.pieces(chess.QUEEN, color):
            score += (1 if color == chess.WHITE else -1) * (7 - (abs(chess.square_file(sq) - ekf) + abs(chess.square_rank(sq) - ekr))) * 3 * mg_weight
    return score

def _fifty_move_pressure(board: chess.Board) -> float:
    hmc = board.halfmove_clock
    if hmc < 30: return 0.0
    pressure = (hmc - 30) * 3
    return -pressure if _ROOT_COLOR == chess.WHITE else pressure

def _anti_ai_complexity(board: chess.Board, eg: float) -> float:
    score = 0.0
    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2: score += 40
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2: score -= 40
    if eg < 0.5: score += (len(board.pieces(chess.PAWN, chess.WHITE)) - len(board.pieces(chess.PAWN, chess.BLACK))) * 10
    return score

def _fork_and_threat_evaluation(board: chess.Board) -> float:
    score = 0.0
    VALUABLE = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 3, chess.KING: 4}
    for color in (chess.WHITE, chess.BLACK):
        sign, enemy = 1 if color == chess.WHITE else -1, not color
        for sq, piece in board.piece_map().items():
            if piece.color != color: continue
            high_value_targets = sum(1 for target_sq in board.attacks(sq) if (tp := board.piece_at(target_sq)) and tp.color == enemy and tp.piece_type in VALUABLE and (PIECE_VALUES[piece.piece_type] < PIECE_VALUES[tp.piece_type] or not board.is_attacked_by(color, target_sq)))
            if high_value_targets >= 2: score += sign * 150 
    return score


def mobility_and_safety(board):
    bonus = 0
    white_mobility = board.legal_moves.count()
    board.push(chess.Move.null())
    black_mobility = board.legal_moves.count()
    board.pop()
    
    bonus += (white_mobility - black_mobility) * 5
    
    if board.has_castling_rights(chess.WHITE): bonus += 30
    if board.has_castling_rights(chess.BLACK): bonus -= 30
    
    return bonus
def detect_forks(board):
    fork_score = 0
    for color in [chess.WHITE, chess.BLACK]:
        multiplier = 1 if color == chess.WHITE else -1
        for piece_type in [chess.KNIGHT, chess.PAWN, chess.BISHOP]:
            pieces = board.pieces(piece_type, color)
            for sq in pieces:
                attacks = board.attacks(sq)
                targets = []
                for target_sq in attacks:
                    target_piece = board.piece_at(target_sq)
                    if target_piece and target_piece.color != color:
                        if PIECE_VALUES[target_piece.piece_type] > PIECE_VALUES[piece_type]:
                            targets.append(target_piece)
                        elif not board.is_attacked_by(not color, target_sq):
                            targets.append(target_piece)
                
                if len(targets) >= 2:
                    fork_score += multiplier * 150
                    
    return fork_score

def evaluate(board: chess.Board, root_color=None, ply: int = 0) -> float:
    if root_color is None: root_color = _ROOT_COLOR
    if board.is_checkmate():
        return -99999 + ply if board.turn == chess.WHITE else 99999 - ply
    if board.is_stalemate() or board.is_insufficient_material() or board.is_repetition(3) or board.is_fifty_moves():
        return -60 if root_color == chess.WHITE else 60
    if board.is_repetition(2):
        return (-60 if root_color == chess.WHITE else 60) * 3.0

    eg = _endgame_factor(board)
    score = _tensor_scalarize(_position_tensor(board))
    draw_score = _evaluate_draw(board, score, root_color)
    if draw_score != 0.0: return draw_score,None

    return float(score + 0.25 * _anti_ai_complexity(board, eg))


_TT, _KILLERS, _HISTORY, _COUNTERMOVE = {}, [[None, None] for _ in range(128)], {}, {}
TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
_FUTILITY_MARGINS, _LMP_CUTOFFS, _RFP_MARGIN = {1: 100, 2: 280, 3: 450, 4: 600}, {1: 6, 2: 10, 3: 16, 4: 22}, 100



_FUTILITY_MARGINS = {1: 100, 2: 280, 3: 450, 4: 600}
_LMP_CUTOFFS      = {1: 6, 2: 10, 3: 16, 4: 22}  # Late move pruning: skip quiet moves after N good ones
_RFP_MARGIN       = 100


def _tt_store(h: int, depth: int, flag: int, score: float, move) -> None:
    existing = _TT.get(h)
    if existing is None or depth >= existing[0]: _TT[h] = (depth, flag, score, move)

def _tt_probe(h: int): return _TT.get(h)

def get_move_score(board, move, ply, tt_move):
    if move == tt_move:
        return 1000000
    if _opening_factor(board) > 0.3 and _is_rook_anti_development_move(board, move):
        return -100000
    if _opening_factor(board) > 0.3 and _is_king_weakening_move(board, move):
        return -80000
    if _opening_factor(board) > 0.3 and _is_development_move(board, move):
        return 850000 + _HISTORY.get((board.turn, move.from_square, move.to_square), 0)
    if board.is_capture(move):
        vic = board.piece_at(move.to_square)
        att = board.piece_at(move.from_square)
        v_val = PIECE_VALUES[vic.piece_type] if vic else 100
        a_val = PIECE_VALUES[att.piece_type] if att else 100
        return 900000 + (v_val * 10) - a_val
    if move in _KILLERS[ply]:
        return 800000
    return _HISTORY.get((board.turn, move.from_square, move.to_square), 0)

def pvs(board, depth, alpha, beta, ply, tt):
    h = chess.polyglot.zobrist_hash(board)
    tt_move = None
    if h in tt:
        entry = tt[h]
        if entry['depth'] >= depth:
            if entry['type'] == TT_EXACT: return entry['score'], entry['move']
            if entry['type'] == TT_LOWER: alpha = max(alpha, entry['score'])
            if entry['type'] == TT_UPPER: beta = min(beta, entry['score'])
            if alpha >= beta: return entry['score'], entry['move']
        tt_move = entry['move']

    if board.is_game_over():
        score = evaluate(board, _ROOT_COLOR, ply)
        # Convert absolute game-over score to relative for Negamax
        return (score if board.turn == chess.WHITE else -score), None
        
    if depth <= 0:
        return q_search(board, alpha, beta, ply), None

    # Internal Iterative Deepening (IID)
    if tt_move is None and depth >= 5:
        _, tt_move = pvs(board, depth - 2, alpha, beta, ply, tt)

    # Adaptive Null Move Pruning
    if depth >= 3 and not board.is_check() and ply > 0:
        R = 3 if depth > 6 else 2
        board.push(chess.Move.null())
        nm_score, _ = pvs(board, depth - 1 - R, -beta, -beta + 1, ply + 1, tt)
        board.pop()
        if -nm_score >= beta: return beta, None

    moves = sorted(board.legal_moves, key=lambda m: get_move_score(board, m, ply, tt_move), reverse=True)
    best_move, best_score = None, -float('inf')
    
    for i, move in enumerate(moves):
        reduction = 0
        if i >= 4 and depth >= 3 and not board.is_capture(move) and not board.is_check():
            reduction = 1 if i < 12 else 2

        board.push(move)
        if i == 0:
            score, _ = pvs(board, depth - 1, -beta, -alpha, ply + 1, tt)
            score = -score
        else:
            score, _ = pvs(board, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, tt)
            score = -score
            if score > alpha and reduction > 0:
                score, _ = pvs(board, depth - 1, -beta, -alpha, ply + 1, tt)
                score = -score
            elif score > alpha:
                score, _ = pvs(board, depth - 1, -beta, -alpha, ply + 1, tt)
                score = -score
        board.pop()

        if score > best_score:
            best_score, best_move = score, move
        alpha = max(alpha, score)
        if alpha >= beta:
            if not board.is_capture(move):
                _KILLERS[ply][1] = _KILLERS[ply][0]
                _KILLERS[ply][0] = move
                key = (board.turn, move.from_square, move.to_square)
                _HISTORY[key] = min(_HISTORY.get(key, 0) + depth * depth, 50_000)
            break

    tt_type = TT_EXACT if alpha < best_score < beta else (TT_LOWER if best_score >= beta else TT_UPPER)
    tt[h] = {'score': best_score, 'move': best_move, 'depth': depth, 'type': tt_type}
    return best_score, best_move

def q_search(board, alpha, beta, ply=0):
    stand_pat = evaluate(board, _ROOT_COLOR, ply)
    if board.turn == chess.BLACK:
        stand_pat = -stand_pat
    if stand_pat >= beta: return beta
    if alpha < stand_pat: alpha = stand_pat
    for move in board.legal_moves:
        if board.is_capture(move):
            board.push(move)
            score = -q_search(board, -beta, -alpha, ply + 1)
            board.pop()
            if score >= beta: return beta
            if score > alpha: alpha = score
    return alpha

def search_worker(fen, depth, alpha, beta, shared_tt, result_queue):
    board = chess.Board(fen)
    global _ROOT_COLOR
    _ROOT_COLOR = board.turn 
    
    # Convert Absolute root bounds to Relative bounds for the Negamax PVS
    rel_alpha = alpha if board.turn == chess.WHITE else -beta
    rel_beta = beta if board.turn == chess.WHITE else -alpha
    
    prev_score = 0.0
    best_move = None

    for d in range(1, depth + 1):
        score, move = pvs(board, d, rel_alpha, rel_beta, 0, shared_tt)
        
        if move is not None:
            best_move = move
            # Convert Relative score back to Absolute score for the root aggregator
            prev_score = score if board.turn == chess.WHITE else -score
            
            _tt_store(
                chess.polyglot.zobrist_hash(board),
                d, 
                TT_EXACT, 
                prev_score, 
                best_move
            )

    result_queue.put((prev_score, best_move, depth))




def _update_killers(move: chess.Move, ply: int) -> None:
    if ply >= 128: return # Safety check for depth limit
    if move != _KILLERS[ply][0]:
        _KILLERS[ply][1] = _KILLERS[ply][0]
        _KILLERS[ply][0] = move

def _update_history(board: chess.Board, move: chess.Move, depth: int) -> None:
    key = (board.turn, move.from_square, move.to_square)
    _HISTORY[key] = min(_HISTORY.get(key, 0) + depth * depth, 50_000)


def _update_countermove(move: chess.Move, prev_move) -> None:
    if prev_move is not None: _COUNTERMOVE[(prev_move.from_square, prev_move.to_square)] = move

def order_moves(board, moves, ply, tt_move):
    def score_move(move):
        if move == tt_move:
            return 100000
        if board.is_capture(move):
            victim = board.piece_at(move.to_square)
            attacker = board.piece_at(move.from_square)
            v_val = PIECE_VALUES[victim.piece_type] if victim else 0
            a_val = PIECE_VALUES[attacker.piece_type] if attacker else 100
            return 90000 + v_val - a_val
        if ply < 128 and move in _KILLERS[ply]:
            return 80000
        return _HISTORY.get((board.turn, move.from_square, move.to_square), 0)
    return sorted(moves, key=score_move, reverse=True)



def _order_moves(board, moves, tt_move=None, ply: int = 0, prev_move=None, maximizing: bool = True):
    moves = list(moves)
    if not moves: return moves
    if ply <= OT_FULL_DEPTH:
        before_tensor = _position_tensor(board)
        scored = [(_ot_score(board, m, before_tensor, maximizing, tt_move, ply, prev_move), m) for m in moves]
        scored.sort(key=lambda x: x[0])
        return [m for _, m in scored]
    scored = [(_legacy_move_score(board, m, tt_move, ply, prev_move), m) for m in moves]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored]


def _ot_prune_root_moves(board: chess.Board, moves, maximizing: bool, tt_move=None, ply: int = 0, prev_move=None):
    return _order_moves(
        board,
        moves,
        tt_move=tt_move,
        ply=ply,
        prev_move=prev_move,
        maximizing=maximizing,
    )

_DELTA_PRUNE_MARGIN = 900

def _qsearch(board: chess.Board, alpha: float, beta: float, maximizing: bool, root_color=None) -> float:
    if root_color is None: root_color = _ROOT_COLOR
    if board.is_game_over(): return evaluate(board, root_color)
    stand_pat = evaluate(board, root_color)

    if maximizing:
        if stand_pat >= beta: return beta
        if stand_pat + _DELTA_PRUNE_MARGIN < alpha: return alpha
        if stand_pat > alpha: alpha = stand_pat
        candidates = list(board.legal_moves) if board.is_check() else [m for m in board.legal_moves if board.is_capture(m) or m.promotion == chess.QUEEN]
        for move in _order_moves(board, candidates, maximizing=maximizing, ply=0):
            board.push(move)
            score = _qsearch(board, alpha, beta, False, root_color)
            board.pop()
            if score >= beta: return beta
            if score > alpha: alpha = score
        return alpha
    else:
        if stand_pat <= alpha: return alpha
        if stand_pat - _DELTA_PRUNE_MARGIN > beta: return beta
        if stand_pat < beta: beta = stand_pat
        candidates = list(board.legal_moves) if board.is_check() else [m for m in board.legal_moves if board.is_capture(m) or m.promotion == chess.QUEEN]
        for move in _order_moves(board, candidates, maximizing=maximizing, ply=0):
            board.push(move)
            score = _qsearch(board, alpha, beta, True, root_color)
            board.pop()
            if score <= alpha: return alpha
            if score < beta: beta = score
        return beta

def minimax(board: chess.Board, depth: int, alpha: float, beta: float, maximizing: bool, ply: int = 0, null_ok: bool = True, prev_move: chess.Move = None, root_color=None, inc_eval: IncrementalEval = None) -> float:
    if root_color is None: root_color = _ROOT_COLOR
    if inc_eval is None: inc_eval = IncrementalEval(board)

    h = chess.polyglot.zobrist_hash(board)
    tt_data = _tt_probe(h)
    tt_move = None
    tt_score = None

    if tt_data is not None:
        tt_depth, tt_flag, tt_score, tt_move_raw = tt_data
        if tt_depth >= depth:
            if tt_flag == TT_EXACT: return tt_score,None
            if tt_flag == TT_LOWER and tt_score >= beta: return tt_score,None
            if tt_flag == TT_UPPER and tt_score <= alpha: return tt_score,None
        if tt_move_raw is not None and tt_move_raw in board.legal_moves: tt_move = tt_move_raw

    if board.is_game_over():
        return evaluate(board, root_color, ply), None
    if depth <= 0:
        return _qsearch(board, alpha, beta, maximizing, root_color),None

    in_check = board.is_check()
    if in_check: depth += 1

    eg = _endgame_factor(board)
    static_eval = _material_and_pst(board, eg)

    if depth >= 4 and tt_move is None and not in_check:
        minimax(board, depth - 2, alpha, beta, maximizing, ply, False, prev_move, root_color, inc_eval)
        if (tt_data := _tt_probe(h)): tt_move = tt_data[3]

    if not in_check and ply > 0 and depth <= 6 and eg < 0.90 and abs(beta) < 90_000:
        rfp_margin = depth * _RFP_MARGIN
        if maximizing and static_eval - rfp_margin >= beta: return static_eval,None
        if not maximizing and static_eval + rfp_margin <= alpha: return static_eval,None

    # --- FIX 1: NMP Logic Correction ---
    if null_ok and ply > 0 and not in_check and depth >= 3 and eg < 0.85:
        R = 3 if depth >= 6 else 2
        if (maximizing and static_eval - beta > 200) or (not maximizing and alpha - static_eval > 200): R += 1
        
        board.push(chess.Move.null())
        if maximizing:
            null_score, _ = minimax(board, depth - 1 - R, beta - 1, beta, False, ply + 1, False, None, root_color, inc_eval)
            board.pop()
            if null_score >= beta: return beta,None
        else:
            null_score, _ = minimax(board, depth - 1 - R, alpha, alpha + 1, True, ply + 1, False, None, root_color, inc_eval)
            board.pop()
            if null_score <= alpha: return alpha,None

    moves = _order_moves(board, list(board.legal_moves), tt_move, ply, prev_move, maximizing=maximizing)
    if not moves: return evaluate(board, root_color, ply),None

    orig_alpha, orig_beta = alpha, beta
    best_move, best = None, float('-inf') if maximizing else float('inf')
    quiet_count = 0

    for move_idx, move in enumerate(moves):
        is_cap = board.is_capture(move)
        is_quiet = (not is_cap and move.promotion is None)

        if is_quiet and not in_check and depth in _LMP_CUTOFFS and quiet_count >= _LMP_CUTOFFS[depth]: continue

        extension = 1 if (move == tt_move and tt_score is not None and not in_check and depth >= 4 and abs(tt_score - static_eval) > 150) else 0

        board.push(move)
        gives_check = board.is_check()

        reduce = max(1, int(_log(max(depth, 1)) * _log(max(move_idx + 1, 1)) / 2.25)) if (move_idx >= 2 and depth >= 3 and is_quiet and not in_check and not gives_check) else 0
        search_depth = depth - 1 + extension

        if maximizing:
            if move_idx == 0:
                score, _= minimax(board, search_depth, alpha, beta, False, ply + 1, True, move, root_color, inc_eval)
            else:
                score, _ = minimax(board, search_depth - reduce, alpha, alpha + 1, False, ply + 1, True, move, root_color, inc_eval)
                if score > alpha:
                    score, _ = minimax(board, search_depth, alpha, beta, False, ply + 1, True, move, root_color, inc_eval)
            board.pop()
            if is_quiet: quiet_count += 1
            if score > best: best, best_move = score, move
            if score > alpha: alpha = score
            if beta <= alpha:
                if is_quiet: _update_killers(move, ply); _update_history(board, move, depth)
                if prev_move is not None: _update_countermove(move, prev_move)
                break
        else:
            if move_idx == 0:
                score, _ = minimax(board, search_depth, alpha, beta, True, ply + 1, True, move, root_color, inc_eval)
            else:
                score, _ = minimax(board, search_depth - reduce, beta - 1, beta, True, ply + 1, True, move, root_color, inc_eval)
                if score < beta:
                    score, _ = minimax(board, search_depth, alpha, beta, True, ply + 1, True, move, root_color, inc_eval)
            board.pop()
            if is_quiet: quiet_count += 1
            if score < best: best, best_move = score, move
            if score < beta: beta = score
            if beta <= alpha:
                if is_quiet: _update_killers(move, ply); _update_history(board, move, depth)
                if prev_move is not None: _update_countermove(move, prev_move)
                break

    flag = TT_EXACT if orig_alpha < best < orig_beta else (TT_LOWER if best >= orig_beta else TT_UPPER) if maximizing else (TT_UPPER if best <= orig_alpha else TT_LOWER)
    _tt_store(h, depth, flag, best, best_move)
    return best, best_move

_WORKER_POOL = None


def _get_book_move(board: chess.Board):
    if BOOK_PATH is None: return None
    try:
        with chess.polyglot.open_reader(BOOK_PATH) as reader:
            entries = list(reader.find_all(board))
            if not entries: return None
            total = sum(e.weight for e in entries)
            if total <= 0: return entries[0].move
            r = total * (hash(str(board.fen())) % 1000) / 1000.0
            for e in entries:
                r -= e.weight
                if r <= 0: return e.move
            return entries[-1].move
    except (OSError, IOError): return None


def _root_search_single_thread(board: chess.Board, target_depth: int, color: chess.Color) -> tuple[chess.Move | None, float]:
    global _ROOT_COLOR
    _ROOT_COLOR = color
    maximizing = (color == chess.WHITE)
    inf = float("inf")
    best_move = None
    prev_score = 0.0
    all_legal = list(board.legal_moves)
    if not all_legal: return None, 0.0

    for d in range(1, target_depth + 1):
        alpha, beta = -inf, inf
        asp_delta = 40
        if d >= 4:
            asp_delta = 40
            alpha = prev_score - asp_delta
            beta = prev_score + asp_delta
            
        retry_limit = 0
        while retry_limit < 3:
            best_worker_score, best_worker_move = minimax(board, d, alpha, beta, 0)
            tt_entry = _tt_probe(chess.polyglot.zobrist_hash(board))
            tt_move = tt_entry[3] if tt_entry else None

            def root_sort_key(m): return (m == tt_move, board.is_capture(m), _HISTORY.get((board.turn, m.from_square, m.to_square), 0))
            sorted_moves = sorted(all_legal, key=root_sort_key, reverse=True)
            
            best_score = -inf if maximizing else inf
            best_at_d = None
            if best_worker_score <= alpha:
                alpha -= asp_delta
                asp_delta *= 2
                retry_limit += 1
            elif best_worker_score >= beta:
                beta += asp_delta
                asp_delta *= 2
                retry_limit += 1
            else:
                best_move = best_worker_move
                prev_score = best_worker_score
                break
            for i, move in enumerate(sorted_moves):
                board.push(move)
                # Introduce Native PVS into Root search wrapper to capitalize on move ordering
                if i == 0:
                    score, _ = minimax(board, d - 1, alpha, beta, not maximizing, 1, True, move, color, None)
                else:
                    if maximizing:
                        score, _ = minimax(board, d - 1, alpha, alpha + 1, not maximizing, 1, True, move, color, None)
                        if score > alpha:
                            score, _ = minimax(board, d - 1, alpha, beta, not maximizing, 1, True, move, color, None)
                    else:
                        score, _ = minimax(board, d - 1, beta - 1, beta, not maximizing, 1, True, move, color, None)
                        if score < beta:
                            score, _ = minimax(board, d - 1, alpha, beta, not maximizing, 1, True, move, color, None)
                board.pop()

                if maximizing:
                    if score > best_score: best_score, best_at_d = score, move
                    alpha = max(alpha, score)
                else:
                    if score < best_score: best_score, best_at_d = score, move
                    beta = min(beta, score)
                    
            if best_score <= alpha and retry_limit < 3:
                alpha -= asp_delta
                asp_delta *= 2
                retry_limit += 1
                continue
            elif best_score >= beta and retry_limit < 3:
                beta += asp_delta
                asp_delta *= 2
                retry_limit += 1
                continue
            else:
                break
        else:
            # If retry limit hit, do one final search with infinite bounds
            best_worker_score, best_worker_move = minimax(board, d, -float('inf'), float('inf'), 0)
            best_move = best_worker_move
            prev_score = best_worker_score
        
        if best_at_d is not None:
            best_move = best_at_d
            prev_score = best_score
        
        san = board.san(best_move) if best_move else "?"
        print(f"Depth {d}/{target_depth} | Best: {san} | Eval: {prev_score:.2f}")
        _tt_store(chess.polyglot.zobrist_hash(board), d, TT_EXACT, prev_score, best_move)

    return best_move, prev_score


def get_next_move(board: chess.Board, color: chess.Color, depth: int = TARGET_DEPTH) -> chess.Move:
    global _KILLERS, _HISTORY, _ROOT_COLOR

    _ROOT_COLOR = color
    for k in list(_HISTORY.keys()):
        _HISTORY[k] >>= 1
        if _HISTORY[k] == 0: del _HISTORY[k]

    all_legal = list(board.legal_moves)
    if not all_legal: raise RuntimeError("No legal moves")

    if (book_move := _get_book_move(board)) and book_move in all_legal: return book_move

    # Funnel completely into the fixed iterative-deepening PVS root.
    best_move, _ = _root_search_single_thread(board, depth, color)
    return best_move if best_move is not None else all_legal[0]


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

