import chess
from team_goraieb import _hce, _pawn_bonus, _king_safety, _hanging_pieces, _king_tropism, _knight_outposts, _pin_penalty, _development_score, _center_control, _uncastled_king_penalty, _king_attack_pressure, _connected_rooks, _piece_values, _raw_material, PST_MG, PST_EG
import team_goraieb
from team_goraieb import get_next_move
b = chess.Board()
moves = "e2e4 b8c6 g1f3 g8f6 e4e5 f6g4 d2d4 d7d5 c2c4 c8f5 c4d5 d8d5 b1c3 d5d7 f1b5 e8c8 h2h3 g4f6 e5f6".split()
for m in moves:
    b.push_uci(m)

print(f"Position after Nf6 exf6: {b.fen()}")
print(f"Raw material: {_raw_material(b):.1f}")
print(f"Full HCE: {_hce(b):.1f}")

# Break down each eval term
for name, func in [
    ("pawn_bonus W", lambda: _pawn_bonus(b, chess.WHITE)),
    ("pawn_bonus B", lambda: -_pawn_bonus(b, chess.BLACK)),
    ("king_safety W", lambda: _king_safety(b, chess.WHITE)),
    ("king_safety B", lambda: -_king_safety(b, chess.BLACK)),
    ("hanging W", lambda: _hanging_pieces(b, chess.WHITE)),
    ("hanging B", lambda: -_hanging_pieces(b, chess.BLACK)),
    ("tropism W", lambda: _king_tropism(b, chess.WHITE)),
    ("tropism B", lambda: -_king_tropism(b, chess.BLACK)),
    ("outposts W", lambda: _knight_outposts(b, chess.WHITE)),
    ("outposts B", lambda: -_knight_outposts(b, chess.BLACK)),
    ("pin W", lambda: _pin_penalty(b, chess.WHITE)),
    ("pin B", lambda: -_pin_penalty(b, chess.BLACK)),
    ("development W", lambda: _development_score(b, chess.WHITE)),
    ("development B", lambda: -_development_score(b, chess.BLACK)),
    ("center W", lambda: _center_control(b, chess.WHITE)),
    ("center B", lambda: -_center_control(b, chess.BLACK)),
    ("uncastled W", lambda: _uncastled_king_penalty(b, chess.WHITE)),
    ("uncastled B", lambda: -_uncastled_king_penalty(b, chess.BLACK)),
    ("king_atk W", lambda: _king_attack_pressure(b, chess.WHITE)),
    ("king_atk B", lambda: -_king_attack_pressure(b, chess.BLACK)),
]:
    val = func()
    if abs(val) > 1.0:
        print(f"  {name:20s}: {val:+.1f}")
b = chess.Board()
moves = "e2e4 b8c6 g1f3 g8f6 e4e5 f6g4 d2d4 d7d5 c2c4 c8f5 c4d5 d8d5 b1c3 d5d7 f1b5 e8c8 h2h3".split()
for m in moves:
    b.push_uci(m)
move = get_next_move(b, chess.BLACK, depth=3)
print(f"Engine picks: {b.san(move)} ({move.uci()})")