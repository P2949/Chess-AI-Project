"""BlunderBus chess bot — minimax + alpha-beta, get_next_move(board, color, depth)."""

from __future__ import annotations

import importlib.util
import os
import random

import chess
import chess.polyglot


# Core tuning values
MATE_SCORE = 100_000
MAX_SEARCH_DEPTH = 3
DRAW_REPETITION_PLY = 3
DRAW_CONTEMPT_CP = 18.0
NNUE_BLEND_OPENING = 0.35
NNUE_BLEND_ENDGAME = 0.25
NNUE_DELTA_CLAMP_OPENING = 220.0
NNUE_DELTA_CLAMP_ENDGAME = 320.0
NNUE_HEALTH_MIRROR_CAP = 85.0
NNUE_HEALTH_MIN = 0.35
NNUE_HEALTH_PROBE_POSITIONS = 24

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2
CHECK_EXTENSION = 1
RECAPTURE_EXTENSION = 1
NULL_MOVE_ENABLE = True
NULL_MOVE_MIN_DEPTH = 3
NULL_MOVE_REDUCTION = 2
LMR_MIN_DEPTH = 3
LMR_START_MOVE = 4
LMR_REDUCTION = 1
ASPIRATION_WINDOW = 120.0
QUIESCENCE_LIMIT = 10
QUIESCENCE_DELTA = 900
BAD_CAPTURE_MARGIN = 220
FUTILITY_MARGIN_WHITE = 180
FUTILITY_MARGIN_BLACK = 600

ORDER_TT = 20_000
ORDER_CAPTURE = 10_000
ORDER_PROMO = 9_000
ORDER_KILLER_1 = 8_000
ORDER_KILLER_2 = 7_000
ORDER_COUNTER = 6_500
ORDER_CHECK = 4_500

# optional extra move-order hints (still minimax); I left it off
POLICY_ORDERING_ON = False
POLICY_SCALE = 140
POLICY_CAPTURE_WEIGHT = 0.22
POLICY_CENTER_WEIGHT = 0.30
POLICY_RECENT_RECAPTURE = 0.70
POLICY_CHECK_BONUS = 0.55
POLICY_PROMO_BONUS = 1.20

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20_000,
}

BISHOP_PAIR_BONUS = 30.0
DOUBLED_PAWN_PENALTY = 20.0
ISOLATED_PAWN_PENALTY = 15.0
ROOK_OPEN_FILE_BONUS = 25.0
ROOK_SEMI_OPEN_FILE_BONUS = 12.0
PASSED_PAWN_BONUS = [0.0, 10.0, 20.0, 35.0, 60.0, 90.0, 130.0, 0.0]

PHASE_WEIGHTS = {
    chess.KNIGHT: 1,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 4,
}
PHASE_MAX = 24

POLICY_ACTIVITY = {
    chess.PAWN: 0.45,
    chess.KNIGHT: 1.05,
    chess.BISHOP: 1.00,
    chess.ROOK: 0.65,
    chess.QUEEN: 0.45,
    chess.KING: 0.20,
}

WHITE_MINOR_HOME = {chess.B1, chess.G1, chess.C1, chess.F1}
BLACK_MINOR_HOME = {chess.B8, chess.G8, chess.C8, chess.F8}
WHITE_QUEEN_HOME = chess.D1
BLACK_QUEEN_HOME = chess.D8

_CENTER_DISTANCE = [
    abs(chess.square_file(sq) - 3.5) + abs(chess.square_rank(sq) - 3.5)
    for sq in chess.SQUARES
]

# NNUE features: piece planes + castling + side
FEATURE_ORDER = (
    (chess.PAWN, chess.WHITE),
    (chess.KNIGHT, chess.WHITE),
    (chess.BISHOP, chess.WHITE),
    (chess.ROOK, chess.WHITE),
    (chess.QUEEN, chess.WHITE),
    (chess.KING, chess.WHITE),
    (chess.PAWN, chess.BLACK),
    (chess.KNIGHT, chess.BLACK),
    (chess.BISHOP, chess.BLACK),
    (chess.ROOK, chess.BLACK),
    (chess.QUEEN, chess.BLACK),
    (chess.KING, chess.BLACK),
)
FEATURE_BASE = {(pt, color): i * 64 for i, (pt, color) in enumerate(FEATURE_ORDER)}
FEATURE_DIM = 12 * 64 + 5
CASTLE_WK = 12 * 64 + 0
CASTLE_WQ = 12 * 64 + 1
CASTLE_BK = 12 * 64 + 2
CASTLE_BQ = 12 * 64 + 3
SIDE_TO_MOVE = 12 * 64 + 4


def _load_nnue_dict() -> dict | None:
    local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "team_blunderbus_nnue_weights.py")
    if os.path.exists(local_path):
        try:
            spec = importlib.util.spec_from_file_location("blunderbus_weights", local_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                data = getattr(module, "NNUE_WEIGHTS", None)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass

    try:
        from team_blunderbus_nnue_weights import NNUE_WEIGHTS as data  # type: ignore

        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None


NNUE_W1_T = None
NNUE_B1 = None
NNUE_W2 = None
NNUE_B2 = None
NNUE_HIDDEN = 0
NNUE_CLIP = 1.5
NNUE_SCALE = 400.0
NNUE_HEALTH = 1.0
NNUE_HEALTH_MIRROR_BIAS = 0.0
NNUE_HEALTH_START_BIAS = 0.0

_nnue = _load_nnue_dict()
if isinstance(_nnue, dict):
    raw_w1_t = _nnue.get("w1_t")
    raw_b1 = _nnue.get("b1")
    raw_w2 = _nnue.get("w2")
    raw_b2 = _nnue.get("b2")
    feature_dim = int(_nnue.get("feature_dim", -1))
    ok = (
        isinstance(raw_w1_t, list)
        and isinstance(raw_b1, list)
        and isinstance(raw_w2, list)
        and isinstance(raw_b2, (int, float))
        and feature_dim == FEATURE_DIM
        and len(raw_w1_t) == FEATURE_DIM
    )
    if ok:
        hidden = len(raw_b1)
        rows_ok = all(isinstance(row, list) and len(row) == hidden for row in raw_w1_t)
        if hidden > 0 and rows_ok and len(raw_w2) == hidden:
            NNUE_W1_T = [[float(v) for v in row] for row in raw_w1_t]
            NNUE_B1 = [float(v) for v in raw_b1]
            NNUE_W2 = [float(v) for v in raw_w2]
            NNUE_B2 = float(raw_b2)
            NNUE_HIDDEN = hidden
            NNUE_CLIP = float(_nnue.get("clip", 1.5))
            NNUE_SCALE = float(_nnue.get("output_scale", 400.0))


TT_MAX = 500_000
EVAL_CACHE_MAX = 350_000
HISTORY_MAX = 120_000
COUNTERMOVE_MAX = 80_000
KILLER_PLIES = 64
# keep some tables between games in a match (tourney sims); tweak if you want isolation
PERSIST_ORDERING_ACROSS_GAMES = True
PERSIST_TT_ACROSS_GAMES = True
PERSIST_EVAL_CACHE_ACROSS_GAMES = True

TRANSPOSITION_TABLE = {}
STATIC_EVAL_CACHE = {}
HISTORY = {}
COUNTERMOVE = {}
KILLERS = [[None, None] for _ in range(KILLER_PLIES)]

CENTER4 = (chess.D4, chess.E4, chess.D5, chess.E5)


def _feature_index(piece: chess.Piece, square: int) -> int:
    base = FEATURE_BASE[(piece.piece_type, piece.color)]
    mapped_sq = chess.square_mirror(square) if piece.color == chess.WHITE else square
    return base + mapped_sq


def _active_features(board: chess.Board) -> list[int]:
    feats = [_feature_index(piece, sq) for sq, piece in board.piece_map().items()]
    if board.has_kingside_castling_rights(chess.WHITE):
        feats.append(CASTLE_WK)
    if board.has_queenside_castling_rights(chess.WHITE):
        feats.append(CASTLE_WQ)
    if board.has_kingside_castling_rights(chess.BLACK):
        feats.append(CASTLE_BK)
    if board.has_queenside_castling_rights(chess.BLACK):
        feats.append(CASTLE_BQ)
    if board.turn == chess.WHITE:
        feats.append(SIDE_TO_MOVE)
    return feats


def _has_non_pawn_material(board: chess.Board, color: chess.Color) -> bool:
    return any(board.pieces(pt, color) for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))


def _nnue_eval(board: chess.Board) -> float | None:
    if NNUE_W1_T is None or NNUE_B1 is None or NNUE_W2 is None or NNUE_B2 is None:
        return None

    hidden = NNUE_B1[:]
    for idx in _active_features(board):
        row = NNUE_W1_T[idx]
        for h in range(NNUE_HIDDEN):
            hidden[h] += row[h]

    out = NNUE_B2
    for h in range(NNUE_HIDDEN):
        v = hidden[h]
        if v <= 0.0:
            continue
        if v > NNUE_CLIP:
            v = NNUE_CLIP
        out += v * NNUE_W2[h]
    return out * NNUE_SCALE


def _init_nnue_health() -> None:
    # if the net is badly asymmetric on random boards, blend it down a bit
    global NNUE_HEALTH, NNUE_HEALTH_MIRROR_BIAS, NNUE_HEALTH_START_BIAS

    if NNUE_W1_T is None:
        NNUE_HEALTH = 1.0
        NNUE_HEALTH_MIRROR_BIAS = 0.0
        NNUE_HEALTH_START_BIAS = 0.0
        return

    rng = random.Random(7)
    mirror_biases = []

    for _ in range(NNUE_HEALTH_PROBE_POSITIONS):
        b = chess.Board()
        plies = rng.randint(4, 28)
        for _ in range(plies):
            if b.is_game_over():
                break
            legal = list(b.legal_moves)
            b.push(rng.choice(legal))

        v = _nnue_eval(b)
        vm = _nnue_eval(b.mirror())
        if v is None or vm is None:
            continue
        mirror_biases.append(abs(v + vm))

    if not mirror_biases:
        NNUE_HEALTH = 1.0
        NNUE_HEALTH_MIRROR_BIAS = 0.0
        NNUE_HEALTH_START_BIAS = 0.0
        return

    mirror_bias = sum(mirror_biases) / len(mirror_biases)
    health = 1.0 if mirror_bias <= NNUE_HEALTH_MIRROR_CAP else (NNUE_HEALTH_MIRROR_CAP / mirror_bias)

    NNUE_HEALTH = max(NNUE_HEALTH_MIN, min(1.0, health))
    NNUE_HEALTH_MIRROR_BIAS = mirror_bias
    NNUE_HEALTH_START_BIAS = 0.0


_init_nnue_health()


def _classical_eval(board: chess.Board) -> float:
    score = 0.0
    for sq, piece in board.piece_map().items():
        sign = 1.0 if piece.color == chess.WHITE else -1.0
        score += sign * PIECE_VALUES[piece.piece_type]

        if piece.piece_type in (chess.KNIGHT, chess.BISHOP, chess.PAWN) and sq in CENTER4:
            score += sign * 14.0

        if piece.piece_type == chess.KING:
            file_idx = chess.square_file(sq)
            rank_idx = chess.square_rank(sq)
            if piece.color == chess.WHITE and (file_idx, rank_idx) in ((6, 0), (2, 0)):
                score += 16.0
            if piece.color == chess.BLACK and (file_idx, rank_idx) in ((6, 7), (2, 7)):
                score -= 16.0

    mobility = board.legal_moves.count()
    score += 0.7 * mobility if board.turn == chess.WHITE else -0.7 * mobility

    if len(board.pieces(chess.BISHOP, chess.WHITE)) >= 2:
        score += BISHOP_PAIR_BONUS
    if len(board.pieces(chess.BISHOP, chess.BLACK)) >= 2:
        score -= BISHOP_PAIR_BONUS

    score += _pawn_structure_bonus(board, chess.WHITE)
    score -= _pawn_structure_bonus(board, chess.BLACK)
    score += _rook_file_bonus(board, chess.WHITE)
    score -= _rook_file_bonus(board, chess.BLACK)
    return score


def _pawn_structure_bonus(board: chess.Board, color: chess.Color) -> float:
    pawns = list(board.pieces(chess.PAWN, color))
    if not pawns:
        return 0.0

    opp_pawns = list(board.pieces(chess.PAWN, not color))
    file_counts: dict[int, int] = {}
    for sq in pawns:
        f = chess.square_file(sq)
        file_counts[f] = file_counts.get(f, 0) + 1

    opp_files: dict[int, list[int]] = {}
    for sq in opp_pawns:
        f = chess.square_file(sq)
        opp_files.setdefault(f, []).append(chess.square_rank(sq))

    bonus = 0.0
    for sq in pawns:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)

        if file_counts[f] > 1:
            bonus -= DOUBLED_PAWN_PENALTY

        has_left_friend = (f - 1) in file_counts
        has_right_friend = (f + 1) in file_counts
        if not has_left_friend and not has_right_friend:
            bonus -= ISOLATED_PAWN_PENALTY

        passed = True
        for df in (-1, 0, 1):
            ranks = opp_files.get(f + df, [])
            for opp_r in ranks:
                if color == chess.WHITE and opp_r > r:
                    passed = False
                    break
                if color == chess.BLACK and opp_r < r:
                    passed = False
                    break
            if not passed:
                break
        if passed:
            idx = r if color == chess.WHITE else 7 - r
            bonus += PASSED_PAWN_BONUS[idx]
    return bonus


def _rook_file_bonus(board: chess.Board, color: chess.Color) -> float:
    rooks = board.pieces(chess.ROOK, color)
    if not rooks:
        return 0.0

    own_pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, color)}
    opp_pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, not color)}

    bonus = 0.0
    for sq in rooks:
        f = chess.square_file(sq)
        if f not in own_pawn_files and f not in opp_pawn_files:
            bonus += ROOK_OPEN_FILE_BONUS
        elif f not in own_pawn_files:
            bonus += ROOK_SEMI_OPEN_FILE_BONUS
    return bonus


def _game_phase_ratio(board: chess.Board) -> float:
    phase = 0
    for pt, weight in PHASE_WEIGHTS.items():
        phase += (len(board.pieces(pt, chess.WHITE)) + len(board.pieces(pt, chess.BLACK))) * weight
    return min(1.0, max(0.0, phase / PHASE_MAX))


def _nnue_blend_weight(board: chess.Board) -> float:
    phase_ratio = _game_phase_ratio(board)
    base = NNUE_BLEND_ENDGAME + (NNUE_BLEND_OPENING - NNUE_BLEND_ENDGAME) * phase_ratio
    return base * NNUE_HEALTH


def evaluate(board: chess.Board) -> float:
    # + = white better
    if board.is_checkmate():
        return -float(MATE_SCORE) if board.turn == chess.WHITE else float(MATE_SCORE)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if board.is_repetition(DRAW_REPETITION_PLY) or board.is_fifty_moves():
        return -DRAW_CONTEMPT_CP if board.turn == chess.WHITE else DRAW_CONTEMPT_CP

    key = chess.polyglot.zobrist_hash(board)
    cached = STATIC_EVAL_CACHE.get(key)
    if cached is not None:
        return cached

    classical = _classical_eval(board)
    nnue = _nnue_eval(board)
    blend = _nnue_blend_weight(board)
    if nnue is None:
        score = classical
    else:
        phase_ratio = _game_phase_ratio(board)
        delta_cap = NNUE_DELTA_CLAMP_ENDGAME + (NNUE_DELTA_CLAMP_OPENING - NNUE_DELTA_CLAMP_ENDGAME) * phase_ratio
        delta = max(-delta_cap, min(delta_cap, nnue - classical))
        score = classical + blend * delta

    if len(STATIC_EVAL_CACHE) > EVAL_CACHE_MAX:
        STATIC_EVAL_CACHE.clear()
    STATIC_EVAL_CACHE[key] = score
    return score


def _mvv_lva(board: chess.Board, move: chess.Move) -> int:
    victim = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)
    victim_val = PIECE_VALUES.get(victim.piece_type, 100) if victim else 100
    attacker_val = PIECE_VALUES.get(attacker.piece_type, 100) if attacker else 100
    return 10 * victim_val - attacker_val


def _least_attacker_value(board: chess.Board, color: chess.Color, square: int) -> int:
    least = 99_999
    for attacker_sq in board.attackers(color, square):
        attacker = board.piece_at(attacker_sq)
        if attacker is None:
            continue
        value = PIECE_VALUES.get(attacker.piece_type, 100)
        if value < least:
            least = value
    return least


def _is_probably_bad_capture(board: chess.Board, move: chess.Move) -> bool:
    # rough filter for dumb captures in qsearch (not full SEE)
    if not board.is_capture(move) or move.promotion or board.gives_check(move):
        return False

    attacker = board.piece_at(move.from_square)
    if attacker is None:
        return False
    attacker_val = PIECE_VALUES.get(attacker.piece_type, 100)

    if board.is_en_passant(move):
        victim_val = PIECE_VALUES[chess.PAWN]
    else:
        victim = board.piece_at(move.to_square)
        victim_val = PIECE_VALUES.get(victim.piece_type, 100) if victim else 100

    if victim_val + BAD_CAPTURE_MARGIN >= attacker_val:
        return False

    least_recapture = _least_attacker_value(board, not board.turn, move.to_square)
    return least_recapture <= attacker_val


def _update_history(move: chess.Move, bonus: int) -> None:
    key = (move.from_square, move.to_square)
    HISTORY[key] = max(-50_000, min(50_000, HISTORY.get(key, 0) + bonus))


def _policy_order_bonus(
    board: chess.Board,
    move: chess.Move,
    is_capture: bool,
    gives_check: bool,
    prev_move: chess.Move | None,
    phase_ratio: float,
) -> int:
    if not POLICY_ORDERING_ON:
        return 0

    piece = board.piece_at(move.from_square)
    if piece is None:
        return 0

    score = 0.0

    if is_capture:
        score += POLICY_CAPTURE_WEIGHT * (_mvv_lva(board, move) / 1000.0)

    if gives_check:
        score += POLICY_CHECK_BONUS

    if move.promotion:
        score += POLICY_PROMO_BONUS

    center_gain = _CENTER_DISTANCE[move.from_square] - _CENTER_DISTANCE[move.to_square]
    activity = POLICY_ACTIVITY.get(piece.piece_type, 0.5)
    score += POLICY_CENTER_WEIGHT * center_gain * activity * (0.35 + 0.65 * phase_ratio)

    if prev_move is not None and is_capture and move.to_square == prev_move.to_square:
        score += POLICY_RECENT_RECAPTURE

    if phase_ratio > 0.65 and piece.piece_type in (chess.KNIGHT, chess.BISHOP):
        if piece.color == chess.WHITE and move.from_square in WHITE_MINOR_HOME:
            score += 0.8
        if piece.color == chess.BLACK and move.from_square in BLACK_MINOR_HOME:
            score += 0.8

    if phase_ratio > 0.75 and piece.piece_type == chess.QUEEN and not is_capture and not gives_check:
        if (piece.color == chess.WHITE and move.from_square == WHITE_QUEEN_HOME) or (
            piece.color == chess.BLACK and move.from_square == BLACK_QUEEN_HOME
        ):
            score -= 0.8

    return int(POLICY_SCALE * score)


def score_move(
    board: chess.Board,
    move: chess.Move,
    ply: int = 0,
    tt_move: chess.Move | None = None,
    prev_move: chess.Move | None = None,
    phase_hint: float | None = None,
) -> int:
    if tt_move is not None and move == tt_move:
        return ORDER_TT

    is_capture = board.is_capture(move)
    gives_check = board.gives_check(move)

    base = 0
    if prev_move is not None:
        cm = COUNTERMOVE.get((prev_move.from_square, prev_move.to_square))
        if cm == move:
            base = max(base, ORDER_COUNTER)

    if is_capture:
        base = max(base, ORDER_CAPTURE + _mvv_lva(board, move))

    if move.promotion:
        base = max(base, ORDER_PROMO + PIECE_VALUES.get(move.promotion, 0))

    if ply < KILLER_PLIES and move == KILLERS[ply][0]:
        base = max(base, ORDER_KILLER_1)
    if ply < KILLER_PLIES and move == KILLERS[ply][1]:
        base = max(base, ORDER_KILLER_2)

    hist = HISTORY.get((move.from_square, move.to_square), 0)
    if gives_check:
        base = max(base, ORDER_CHECK + hist)
    else:
        base += hist

    phase_ratio = _game_phase_ratio(board) if phase_hint is None else phase_hint
    policy = _policy_order_bonus(board, move, is_capture, gives_check, prev_move, phase_ratio)
    return base + policy


def order_moves(
    board: chess.Board,
    moves: list[chess.Move],
    ply: int = 0,
    tt_move: chess.Move | None = None,
    prev_move: chess.Move | None = None,
) -> list[chess.Move]:
    phase_hint = _game_phase_ratio(board)
    return sorted(
        moves,
        key=lambda m: score_move(board, m, ply, tt_move, prev_move, phase_hint),
        reverse=True,
    )


def _lmr_reduction(
    board: chess.Board,
    move: chess.Move,
    depth_left: int,
    move_idx: int,
    in_check: bool,
    recapture: bool,
) -> int:
    if depth_left < LMR_MIN_DEPTH or move_idx < LMR_START_MOVE:
        return 0
    if in_check or recapture or board.is_capture(move) or move.promotion or board.gives_check(move):
        return 0
    reduction = LMR_REDUCTION
    if depth_left >= 5 and move_idx >= 8:
        reduction += 1
    return reduction


def quiescence(board: chess.Board, alpha: float, beta: float, maximizing: bool, qply: int = 0) -> float:
    if qply >= QUIESCENCE_LIMIT:
        return evaluate(board)
    if board.is_checkmate():
        return -float(MATE_SCORE - qply) if board.turn == chess.WHITE else float(MATE_SCORE - qply)
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if board.is_repetition(DRAW_REPETITION_PLY) or board.is_fifty_moves():
        return -DRAW_CONTEMPT_CP if board.turn == chess.WHITE else DRAW_CONTEMPT_CP

    in_check = board.is_check()
    stand = evaluate(board)
    if not in_check:
        if maximizing:
            if stand >= beta:
                return beta
            if stand + QUIESCENCE_DELTA < alpha:
                return alpha
            alpha = max(alpha, stand)
        else:
            if stand <= alpha:
                return alpha
            if stand - QUIESCENCE_DELTA > beta:
                return beta
            beta = min(beta, stand)
        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]
    else:
        moves = list(board.legal_moves)

    for move in order_moves(board, moves, qply):
        if not in_check and _is_probably_bad_capture(board, move):
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
    prev_move: chess.Move | None = None,
    last_capture_sq: int | None = None,
) -> float:
    in_check = board.is_check()
    depth_left = depth + (CHECK_EXTENSION if in_check else 0)

    key = chess.polyglot.zobrist_hash(board)
    tt_move = None
    tt = TRANSPOSITION_TABLE.get(key)
    if tt is not None:
        tt_depth, tt_flag, tt_score, tt_move = tt
        if tt_depth >= depth_left:
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
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if board.is_repetition(DRAW_REPETITION_PLY) or board.is_fifty_moves():
        return -DRAW_CONTEMPT_CP if board.turn == chess.WHITE else DRAW_CONTEMPT_CP
    if depth_left <= 0:
        return quiescence(board, alpha, beta, maximizing, 0)

    static_eval = evaluate(board)
    if not in_check and depth_left == 1 and abs(alpha) < MATE_SCORE - 1000 and abs(beta) < MATE_SCORE - 1000:
        futility_margin = FUTILITY_MARGIN_WHITE if board.turn == chess.WHITE else FUTILITY_MARGIN_BLACK
        if maximizing and static_eval + futility_margin <= alpha:
            return static_eval
        if (not maximizing) and static_eval - futility_margin >= beta:
            return static_eval

    if (
        NULL_MOVE_ENABLE
        and depth_left >= NULL_MOVE_MIN_DEPTH
        and ply > 0
        and not in_check
        and _has_non_pawn_material(board, board.turn)
        and _has_non_pawn_material(board, not board.turn)
    ):
        board.push(chess.Move.null())
        null_score = None
        if maximizing and static_eval >= beta - 90:
            null_score = minimax(board, depth_left - 1 - NULL_MOVE_REDUCTION, beta - 1.0, beta, False, ply + 1)
        elif (not maximizing) and static_eval <= alpha + 90:
            null_score = minimax(board, depth_left - 1 - NULL_MOVE_REDUCTION, alpha, alpha + 1.0, True, ply + 1)
        board.pop()
        if maximizing and null_score is not None and null_score >= beta:
            return beta
        if (not maximizing) and null_score is not None and null_score <= alpha:
            return alpha

    moves = order_moves(board, list(board.legal_moves), ply, tt_move, prev_move)
    if not moves:
        return static_eval

    orig_alpha = alpha
    orig_beta = beta
    best_score = float("-inf") if maximizing else float("inf")
    best_move = None

    for idx, move in enumerate(moves):
        is_cap = board.is_capture(move)
        recapture = last_capture_sq is not None and is_cap and move.to_square == last_capture_sq
        child_depth = depth_left - 1 + (RECAPTURE_EXTENSION if recapture else 0)
        reduction = _lmr_reduction(board, move, depth_left, idx, in_check, recapture)
        cap_sq = move.to_square if is_cap else None

        board.push(move)
        if idx == 0:
            score = minimax(board, child_depth, alpha, beta, not maximizing, ply + 1, move, cap_sq)
        elif reduction > 0:
            reduced = max(0, child_depth - reduction)
            if maximizing:
                score = minimax(board, reduced, alpha, alpha + 1.0, False, ply + 1, move, cap_sq)
                if score > alpha:
                    score = minimax(board, child_depth, alpha, beta, False, ply + 1, move, cap_sq)
            else:
                score = minimax(board, reduced, beta - 1.0, beta, True, ply + 1, move, cap_sq)
                if score < beta:
                    score = minimax(board, child_depth, alpha, beta, True, ply + 1, move, cap_sq)
        else:
            if maximizing:
                score = minimax(board, child_depth, alpha, alpha + 1.0, False, ply + 1, move, cap_sq)
                if score > alpha:
                    score = minimax(board, child_depth, alpha, beta, False, ply + 1, move, cap_sq)
            else:
                score = minimax(board, child_depth, beta - 1.0, beta, True, ply + 1, move, cap_sq)
                if score < beta:
                    score = minimax(board, child_depth, alpha, beta, True, ply + 1, move, cap_sq)
        board.pop()

        if maximizing:
            if score > best_score:
                best_score, best_move = score, move
            if score > alpha:
                alpha = score
            if alpha >= beta:
                if not is_cap and not move.promotion:
                    if ply < KILLER_PLIES and move != KILLERS[ply][0]:
                        KILLERS[ply][1] = KILLERS[ply][0]
                        KILLERS[ply][0] = move
                    _update_history(move, depth_left * depth_left)
                    if prev_move is not None:
                        COUNTERMOVE[(prev_move.from_square, prev_move.to_square)] = move
                break
        else:
            if score < best_score:
                best_score, best_move = score, move
            if score < beta:
                beta = score
            if beta <= alpha:
                if not is_cap and not move.promotion:
                    if ply < KILLER_PLIES and move != KILLERS[ply][0]:
                        KILLERS[ply][1] = KILLERS[ply][0]
                        KILLERS[ply][0] = move
                    _update_history(move, depth_left * depth_left)
                    if prev_move is not None:
                        COUNTERMOVE[(prev_move.from_square, prev_move.to_square)] = move
                break

    flag = TT_EXACT
    if best_score <= orig_alpha:
        flag = TT_UPPER
    elif best_score >= orig_beta:
        flag = TT_LOWER

    if len(TRANSPOSITION_TABLE) > TT_MAX:
        TRANSPOSITION_TABLE.clear()
    if len(HISTORY) > HISTORY_MAX:
        HISTORY.clear()
    if len(COUNTERMOVE) > COUNTERMOVE_MAX:
        COUNTERMOVE.clear()
    TRANSPOSITION_TABLE[key] = (depth_left, flag, best_score, best_move)
    return best_score


def get_next_move(board: chess.Board, color: chess.Color, depth: int = 3) -> chess.Move:
    # start position: wipe tt/cache (or not, see PERSIST_* above)
    if len(board.move_stack) == 0:
        if not PERSIST_TT_ACROSS_GAMES:
            TRANSPOSITION_TABLE.clear()
        if not PERSIST_EVAL_CACHE_ACROSS_GAMES:
            STATIC_EVAL_CACHE.clear()
        if not PERSIST_ORDERING_ACROSS_GAMES:
            HISTORY.clear()
            COUNTERMOVE.clear()
        KILLERS[:] = [[None, None] for _ in range(KILLER_PLIES)]

    depth = max(1, min(MAX_SEARCH_DEPTH, int(depth)))
    legal = list(board.legal_moves)
    if not legal:
        return None

    maximizing = color == chess.WHITE
    root = board.copy()
    best_move = legal[0]
    prev_score = evaluate(root)

    for current_depth in range(1, depth + 1):
        use_asp = current_depth >= 2
        window = ASPIRATION_WINDOW
        alpha = prev_score - window if use_asp else float("-inf")
        beta = prev_score + window if use_asp else float("inf")
        asp_retry = 0

        while True:
            root_key = chess.polyglot.zobrist_hash(root)
            tt = TRANSPOSITION_TABLE.get(root_key)
            tt_move = tt[3] if tt is not None and tt[3] in legal else None
            moves = order_moves(root, legal, 0, tt_move, None)
            if best_move in moves:
                moves.remove(best_move)
                moves.insert(0, best_move)

            best_here_move = None
            best_here_score = float("-inf") if maximizing else float("inf")
            a = alpha
            b = beta

            for idx, move in enumerate(moves):
                cap_sq = move.to_square if root.is_capture(move) else None
                root.push(move)
                if idx == 0:
                    score = minimax(root, current_depth - 1, a, b, not maximizing, 1, move, cap_sq)
                else:
                    if maximizing:
                        score = minimax(root, current_depth - 1, a, a + 1.0, not maximizing, 1, move, cap_sq)
                        if score > a:
                            score = minimax(root, current_depth - 1, a, b, not maximizing, 1, move, cap_sq)
                    else:
                        score = minimax(root, current_depth - 1, b - 1.0, b, not maximizing, 1, move, cap_sq)
                        if score < b:
                            score = minimax(root, current_depth - 1, a, b, not maximizing, 1, move, cap_sq)
                root.pop()

                if maximizing:
                    if score > best_here_score:
                        best_here_score, best_here_move = score, move
                    a = max(a, score)
                    if b <= a:
                        break
                else:
                    if score < best_here_score:
                        best_here_score, best_here_move = score, move
                    b = min(b, score)
                    if b <= a:
                        break

            if best_here_move is not None:
                best_move = best_here_move
                prev_score = best_here_score

            if not use_asp:
                break
            if prev_score <= alpha:
                alpha -= window
                window *= 2.0
                asp_retry += 1
                if asp_retry >= 4:
                    use_asp = False
                    alpha, beta = float("-inf"), float("inf")
                continue
            if prev_score >= beta:
                beta += window
                window *= 2.0
                asp_retry += 1
                if asp_retry >= 4:
                    use_asp = False
                    alpha, beta = float("-inf"), float("inf")
                continue
            break

    return best_move
