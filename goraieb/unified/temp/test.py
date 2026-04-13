import chess
from team_goraieb import get_next_move, evaluate, _overextended_pieces

# Position after 1.e4 Nf6 2.e5 — Black must retreat the knight
b = chess.Board()
b.push_uci("e2e4")
b.push_uci("g8f6")
b.push_uci("e4e5")

print("Position:", b.fen())
move = get_next_move(b, chess.BLACK, depth=3)
print(f"Engine picks: {b.san(move)} ({move.uci()})")

# Check overextended penalty for each knight retreat
for uci, name in [("f6e4", "Ne4"), ("f6d5", "Nd5"), ("f6g4", "Ng4"), ("f6g8", "Ng8")]:
    m = chess.Move.from_uci(uci)
    if m in b.legal_moves:
        b.push(m)
        oe_w = _overextended_pieces(b, chess.WHITE)
        oe_b = _overextended_pieces(b, chess.BLACK)
        print(f"  After {name}: overext_W={oe_w:.1f} overext_B={oe_b:.1f} eval={evaluate(b):.1f}")
        b.pop()