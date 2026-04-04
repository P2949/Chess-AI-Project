"""
visualize.py  —  Chess game visualizer with random colour draw

At startup a draw screen spins through the teams and randomly assigns
White / Black.  The game then plays with that assignment.

Requirements:
    pip install python-chess
    Python standard library: tkinter  (ships with most Python installs)

Run:
    python visualize.py

All three files must be in the same directory:
    visualize.py
    team_alpha.py
    team_goraieb.py
"""

team_a = "goraieb.unified.team_goraieb"
team_b = "goraieb.unified.team_goraieb"

import tkinter as tk
from tkinter import font as tkfont
import chess
import importlib
import random
import sys
import os

# ── Configuration ─────────────────────────────────────────────────────────────
DEPTH       = 3        # search depth — keep at 3 for fast moves; raise for stronger play
MOVE_DELAY  = 0.1        # seconds between moves just for visualising
SQUARE_SIZE = 72         # pixels per square
BOARD_SIZE  = SQUARE_SIZE * 8

# ── Colours ───────────────────────────────────────────────────────────────────
LIGHT_SQ    = "#F0D9B5"
DARK_SQ     = "#B58863"
HIGHLIGHT   = "#F6F669"   # last-move highlight (from-square)
HIGHLIGHT2  = "#CDD16E"   # last-move highlight (to-square)
WHITE_PIECE = "#FFFFFF"
BLACK_PIECE = "#1A1A1A"
BORDER_COL  = "#8B6914"
BG          = "#2C2C2C"
TEXT_COL    = "#E8E8E8"
ACCENT      = "#5B9BD5"
WIN_WHITE   = "#F0D9B5"
WIN_BLACK   = "#B58863"

# ── Unicode chess pieces ──────────────────────────────────────────────────────
PIECE_UNICODE = {
    (chess.KING,   chess.WHITE): "♔",
    (chess.QUEEN,  chess.WHITE): "♕",
    (chess.ROOK,   chess.WHITE): "♖",
    (chess.BISHOP, chess.WHITE): "♗",
    (chess.KNIGHT, chess.WHITE): "♘",
    (chess.PAWN,   chess.WHITE): "♙",
    (chess.KING,   chess.BLACK): "♚",
    (chess.QUEEN,  chess.BLACK): "♛",
    (chess.ROOK,   chess.BLACK): "♜",
    (chess.BISHOP, chess.BLACK): "♝",
    (chess.KNIGHT, chess.BLACK): "♞",
    (chess.PAWN,   chess.BLACK): "♟",
}


# ─────────────────────────────────────────────────────────────────────────────
class ChessGUI:
    def __init__(self, root: tk.Tk, bot_white, bot_black,
                 name_white: str = team_b, name_black: str = team_a):
        self.root       = root
        self.bot_white  = bot_white   # module with get_next_move()
        self.bot_black  = bot_black
        self.name_white = name_white  # human-readable team name for White
        self.name_black = name_black  # human-readable team name for Black
        self.board      = chess.Board()
        self.last_move  = None
        self.move_log   = []
        self.running    = True

        root.title(f"Chess Bot Battle — {name_white} (White) vs {name_black} (Black)")
        root.configure(bg=BG)
        root.resizable(False, False)

        self._build_ui()
        self._draw_board()

        # Start the event-driven game loop (no background thread needed)
        self.root.after(200, self._game_loop)   # small initial delay so window renders first

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        # Left: board + rank/file labels
        board_frame = tk.Frame(self.root, bg=BG)
        board_frame.grid(row=0, column=0, padx=16, pady=16)

        # File labels (a–h) above
        file_bar_top = tk.Frame(board_frame, bg=BG)
        file_bar_top.pack()
        tk.Label(file_bar_top, text="  ", bg=BG, width=2).pack(side=tk.LEFT)
        for f in "abcdefgh":
            tk.Label(file_bar_top, text=f, bg=BG, fg="#888888",
                     width=SQUARE_SIZE // 10, font=("Arial", 10)).pack(side=tk.LEFT, padx=SQUARE_SIZE//2 - 8)

        inner = tk.Frame(board_frame, bg=BG)
        inner.pack()

        # Rank labels left
        rank_frame = tk.Frame(inner, bg=BG)
        rank_frame.pack(side=tk.LEFT)
        for r in range(8, 0, -1):
            tk.Label(rank_frame, text=str(r), bg=BG, fg="#888888",
                     font=("Arial", 10), height=SQUARE_SIZE // 16).pack(pady=SQUARE_SIZE // 2 - 8)

        # Canvas
        self.canvas = tk.Canvas(inner,
                                width=BOARD_SIZE, height=BOARD_SIZE,
                                bg=BG, highlightthickness=2,
                                highlightbackground=BORDER_COL)
        self.canvas.pack(side=tk.LEFT)

        # Right panel
        right = tk.Frame(self.root, bg=BG, width=240)
        right.grid(row=0, column=1, padx=(0, 16), pady=16, sticky="ns")
        right.grid_propagate(False)

        # Title
        tk.Label(right, text="♟  CHESS BOT BATTLE", bg=BG, fg=ACCENT,
                 font=("Arial", 13, "bold")).pack(pady=(8, 2))

        # Players
        tk.Label(right, text=f"⬜  {self.name_white}  (White)", bg=BG, fg=TEXT_COL,
                 font=("Arial", 10)).pack()
        tk.Label(right, text=f"⬛  {self.name_black}   (Black)", bg=BG, fg=TEXT_COL,
                 font=("Arial", 10)).pack(pady=(0, 12))

        # Status label
        self.status_var = tk.StringVar(value="Game starting…")
        self.status_lbl = tk.Label(right, textvariable=self.status_var,
                                   bg=BG, fg="#F0C040",
                                   font=("Arial", 11, "bold"),
                                   wraplength=220, justify=tk.CENTER)
        self.status_lbl.pack(pady=(0, 10))

        # Move counter
        self.move_count_var = tk.StringVar(value="Move: 0")
        tk.Label(right, textvariable=self.move_count_var,
                 bg=BG, fg=TEXT_COL, font=("Arial", 10)).pack()

        # Eval label
        self.eval_var = tk.StringVar(value="Eval: —")
        tk.Label(right, textvariable=self.eval_var,
                 bg=BG, fg="#88CC88", font=("Arial", 10)).pack(pady=(0, 8))

        # Separator
        tk.Frame(right, bg="#444444", height=1).pack(fill=tk.X, pady=4)

        # Move log
        tk.Label(right, text="Move Log", bg=BG, fg="#AAAAAA",
                 font=("Arial", 9, "bold")).pack()

        log_frame = tk.Frame(right, bg=BG)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        scrollbar = tk.Scrollbar(log_frame, bg="#444444")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(log_frame, width=22, bg="#1E1E1E", fg=TEXT_COL,
                                font=("Courier", 9), state=tk.DISABLED,
                                yscrollcommand=scrollbar.set,
                                relief=tk.FLAT, borderwidth=4)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _draw_board(self):
        self.canvas.delete("all")
        piece_font = tkfont.Font(family="Segoe UI Symbol", size=int(SQUARE_SIZE * 0.58))

        last_from = self.last_move.from_square if self.last_move else None
        last_to   = self.last_move.to_square   if self.last_move else None

        for rank in range(7, -1, -1):
            for file in range(8):
                sq = chess.square(file, rank)
                x0 = file       * SQUARE_SIZE
                y0 = (7 - rank) * SQUARE_SIZE
                x1, y1 = x0 + SQUARE_SIZE, y0 + SQUARE_SIZE

                # Square colour
                if sq == last_from:
                    col = HIGHLIGHT
                elif sq == last_to:
                    col = HIGHLIGHT2
                elif (rank + file) % 2 == 0:
                    col = DARK_SQ
                else:
                    col = LIGHT_SQ

                self.canvas.create_rectangle(x0, y0, x1, y1, fill=col, outline="")

                # Piece
                piece = self.board.piece_at(sq)
                if piece:
                    symbol = PIECE_UNICODE[(piece.piece_type, piece.color)]
                    fg = WHITE_PIECE if piece.color == chess.WHITE else BLACK_PIECE
                    # Shadow for white pieces on light squares
                    cx, cy = x0 + SQUARE_SIZE // 2, y0 + SQUARE_SIZE // 2
                    if piece.color == chess.WHITE:
                        self.canvas.create_text(cx + 1, cy + 1, text=symbol,
                                                font=piece_font, fill="#555555")
                    self.canvas.create_text(cx, cy, text=symbol,
                                            font=piece_font, fill=fg)

    # ── Move log helper ───────────────────────────────────────────────────────
    def _append_log(self, move_num: int, white_san: str, black_san: str = ""):
        self.log_text.config(state=tk.NORMAL)
        line = f"{move_num:>3}.  {white_san:<8} {black_san}\n"
        self.log_text.insert(tk.END, line)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    # ── Game loop — event-driven via root.after() ─────────────────────────────

    # Instead we use a two-phase approach entirely on the UI thread:
    #   Phase 1 (_think_and_move):  run the bot (fast at depth 2-3), push the
    #                               move, redraw the board immediately.
    #   Phase 2 (scheduled):        root.after(MOVE_DELAY_MS, _think_and_move)
    #                               fires AFTER the frame is painted, giving
    #                               the user the full delay to look at the board.
    #
    def _game_loop(self):
        """Kick off the first move. All subsequent moves are self-scheduling."""
        self.move_number       = 1
        self.white_san_pending = ""
        self._think_and_move()

    def _think_and_move(self):
        """
        Called once per ply, always on the UI thread via root.after().
        Order of events:
          1. Show "thinking…" status
          2. Ask the bot for its move  (blocks UI briefly — acceptable at depth 2-3)
          3. Push the move & redraw   (board is now visibly updated)
          4. Wait MOVE_DELAY_MS       (user sees the new position for the full delay)
          5. Schedule the next call
        """
        if self.board.is_game_over() or not self.running:
            self._show_result()
            return

        current_color = self.board.turn
        if current_color == chess.WHITE:
            bot   = self.bot_white
            label = f"⚪  {self.name_white} thinking…"
        else:
            bot   = self.bot_black
            label = f"⚫  {self.name_black} thinking…"

        self.status_var.set(label)
        self.root.update_idletasks()   # flush the status label to screen NOW

        # ── Ask bot for move ──────────────────────────────────────────────────
        move = bot.get_next_move(self.board, current_color, DEPTH)

        if move is None or move not in self.board.legal_moves:
            result_text = (
                f"{self.name_black} WINS — illegal move by {self.name_white}"
                if current_color == chess.WHITE
                else f"{self.name_white} WINS — illegal move by {self.name_black}"
            )
            self.status_var.set(result_text)
            return

        # ── Record SAN before pushing ─────────────────────────────────────────
        san = self.board.san(move)
        if current_color == chess.WHITE:
            self.white_san_pending = san
        else:
            self._append_log(self.move_number, self.white_san_pending, san)
            self.white_san_pending = ""
            self.move_number += 1

        # ── Update board state ────────────────────────────────────────────────
        self.last_move = move
        self.board.push(move)

        # ── Eval ──────────────────────────────────────────────────────────────
        try:
            eval_score = self.bot_white.evaluate(self.board)
            eval_str   = f"Eval: {eval_score:+.0f}"
        except Exception:
            eval_str = "Eval: —"

        self.move_count_var.set(
            f"Move: {self.board.fullmove_number}  |  "
            f"Ply: {len(self.board.move_stack)}"
        )
        self.eval_var.set(eval_str)

        # ── Draw the new position — this paints to screen before the delay ────
        self._draw_board()
        self.root.update_idletasks()   # force tkinter to render the frame NOW

        # ── Check again after the push ────────────────────────────────────────
        if self.board.is_game_over():
            self._show_result()
            return

        # ── Schedule the next move after the full visible delay ───────────────
        # root.after() fires only AFTER the current frame is rendered, so the
        # user sees the board for exactly MOVE_DELAY_MS milliseconds.
        self.root.after(int(MOVE_DELAY * 1000), self._think_and_move)

    def _show_result(self):
        if self.white_san_pending:
            self._append_log(self.move_number, self.white_san_pending, "")

        result = self.board.result()
        if result == "1-0":
            msg = f"✅  {self.name_white} (White) WINS!"
        elif result == "0-1":
            msg = f"✅  {self.name_black} (Black) WINS!"
        else:
            msg = "🤝  Draw!"

        reason = ""
        if self.board.is_checkmate():
            reason = " by Checkmate"
        elif self.board.is_stalemate():
            reason = " by Stalemate"
        elif self.board.is_insufficient_material():
            reason = " — Insufficient Material"
        elif self.board.is_seventyfive_moves():
            reason = " — 75-Move Rule"
        elif self.board.is_fivefold_repetition():
            reason = " — Repetition"

        self.root.after(0, self.status_var.set, msg + reason)
        self.root.after(0, self._draw_board)


# ── Colour Draw Screen ────────────────────────────────────────────────────────
class ColorDrawScreen:
    """
    A pre-game screen that randomly assigns White / Black to the two teams
    via a coin-flip animation.

    Flow:
      1. Both team names shown side by side.
      2. User clicks "🎲 Draw Colors!" button.
      3. Slot-machine spin: rapidly alternates which team is highlighted
         White vs Black, slowing down over ~2 seconds.
      4. Final assignment revealed with a banner.
      5. "▶  Start Game" button launches ChessGUI.
    """

    SPIN_SCHEDULE = (
        # (interval_ms, number_of_flips)  — starts fast, slows down
        [50]  * 10 +   # fast  rattles
        [100] * 6  +   # medium
        [200] * 4  +   # slow
        [400] * 2  +   # very slow
        [600] * 1      # final settle
    )

    def __init__(self, root: tk.Tk, mod_a, mod_b, name_a: str, name_b: str):
        self.root   = root
        self.mod_a  = mod_a
        self.mod_b  = mod_b
        self.name_a = name_a   # e.g. "team_alpha"
        self.name_b = name_b   # e.g. "team_goraieb"

        # Will be set after the draw
        self.white_mod  = None
        self.black_mod  = None
        self.white_name = None
        self.black_name = None

        self._spin_index   = 0     # position in SPIN_SCHEDULE
        self._current_flip = 0    # 0 = A is White, 1 = B is White
        self._spinning     = False

        root.title("Chess Bot Battle — Colour Draw")
        root.configure(bg=BG)
        root.resizable(False, False)
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        pad = dict(padx=30, pady=10)

        # Title
        tk.Label(self.root, text="⚔  COLOUR DRAW", bg=BG, fg=ACCENT,
                 font=("Arial", 20, "bold")).pack(pady=(30, 4))
        tk.Label(self.root, text="Click the button to randomly assign White & Black",
                 bg=BG, fg="#AAAAAA", font=("Arial", 10)).pack(pady=(0, 24))

        # Team cards frame
        cards = tk.Frame(self.root, bg=BG)
        cards.pack(**pad)

        # Team A card
        self.card_a = tk.Frame(cards, bg="#333333", relief=tk.FLAT,
                               padx=20, pady=16, bd=2)
        self.card_a.grid(row=0, column=0, padx=18)
        self.label_a_name = tk.Label(self.card_a, text=self.name_a,
                                     bg="#333333", fg=TEXT_COL,
                                     font=("Arial", 14, "bold"))
        self.label_a_name.pack()
        self.label_a_color = tk.Label(self.card_a, text="?",
                                      bg="#333333", fg="#888888",
                                      font=("Arial", 28))
        self.label_a_color.pack(pady=(6, 0))
        self.label_a_badge = tk.Label(self.card_a, text="",
                                      bg="#333333", fg="#888888",
                                      font=("Arial", 10))
        self.label_a_badge.pack()

        # VS label
        tk.Label(cards, text="vs", bg=BG, fg="#666666",
                 font=("Arial", 16, "bold")).grid(row=0, column=1, padx=10)

        # Team B card
        self.card_b = tk.Frame(cards, bg="#333333", relief=tk.FLAT,
                               padx=20, pady=16, bd=2)
        self.card_b.grid(row=0, column=2, padx=18)
        self.label_b_name = tk.Label(self.card_b, text=self.name_b,
                                     bg="#333333", fg=TEXT_COL,
                                     font=("Arial", 14, "bold"))
        self.label_b_name.pack()
        self.label_b_color = tk.Label(self.card_b, text="?",
                                      bg="#333333", fg="#888888",
                                      font=("Arial", 28))
        self.label_b_color.pack(pady=(6, 0))
        self.label_b_badge = tk.Label(self.card_b, text="",
                                      bg="#333333", fg="#888888",
                                      font=("Arial", 10))
        self.label_b_badge.pack()

        # Result banner (hidden until draw is done)
        self.banner_var = tk.StringVar(value="")
        self.banner_lbl = tk.Label(self.root, textvariable=self.banner_var,
                                   bg=BG, fg="#F0C040",
                                   font=("Arial", 13, "bold"),
                                   wraplength=400, justify=tk.CENTER)
        self.banner_lbl.pack(pady=(18, 4))

        # Buttons
        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(pady=(10, 30))

        self.draw_btn = tk.Button(
            btn_frame, text="🎲  Draw Colors!",
            bg=ACCENT, fg="white", activebackground="#3A7DC9",
            font=("Arial", 12, "bold"), relief=tk.FLAT,
            padx=18, pady=8, cursor="hand2",
            command=self._start_spin
        )
        self.draw_btn.pack(side=tk.LEFT, padx=10)

        self.start_btn = tk.Button(
            btn_frame, text="▶  Start Game",
            bg="#27AE60", fg="white", activebackground="#1E8449",
            font=("Arial", 12, "bold"), relief=tk.FLAT,
            padx=18, pady=8, cursor="hand2",
            command=self._launch_game,
            state=tk.DISABLED
        )
        self.start_btn.pack(side=tk.LEFT, padx=10)

    # ── Spin logic ────────────────────────────────────────────────────────────
    def _start_spin(self):
        if self._spinning:
            return
        self._spinning   = True
        self._spin_index = 0
        self.draw_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.DISABLED)
        self.banner_var.set("")

        # Pre-determine the final result so the spin lands on it
        self._final_flip = random.randint(0, 1)   # 0 = A is White, 1 = B is White
        self._current_flip = random.randint(0, 1)  # random starting state

        self._do_spin_step()

    def _do_spin_step(self):
        if self._spin_index >= len(self.SPIN_SCHEDULE):
            # Animation done — lock in the final result
            self._current_flip = self._final_flip
            self._update_cards(final=True)
            self._spinning = False
            self._reveal_result()
            return

        # Flip the display
        self._current_flip ^= 1
        self._update_cards(final=False)

        delay = self.SPIN_SCHEDULE[self._spin_index]
        self._spin_index += 1
        self.root.after(delay, self._do_spin_step)

    def _update_cards(self, final: bool):
        """Redraw both cards to reflect current_flip assignment."""
        # current_flip=0 → A is White, B is Black
        # current_flip=1 → A is Black, B is White
        if self._current_flip == 0:
            a_sym, a_col, a_badge = "♔", "#FFFFFF", "WHITE"
            b_sym, b_col, b_badge = "♚", "#888888", "BLACK"
            a_bg, b_bg = "#4A4A2A", "#2A2A2A"
        else:
            a_sym, a_col, a_badge = "♚", "#888888", "BLACK"
            b_sym, b_col, b_badge = "♔", "#FFFFFF", "WHITE"
            a_bg, b_bg = "#2A2A2A", "#4A4A2A"

        badge_col = "#F0C040" if final else "#888888"

        for card, lbl_color, lbl_badge, sym, col, badge, bg in [
            (self.card_a, self.label_a_color, self.label_a_badge,
             a_sym, a_col, a_badge, a_bg),
            (self.card_b, self.label_b_color, self.label_b_badge,
             b_sym, b_col, b_badge, b_bg),
        ]:
            card.config(bg=bg)
            self.label_a_name.config(bg=self.card_a.cget("bg"))
            self.label_b_name.config(bg=self.card_b.cget("bg"))
            lbl_color.config(text=sym, fg=col, bg=card.cget("bg"))
            lbl_badge.config(text=badge, fg=badge_col, bg=card.cget("bg"))

    def _reveal_result(self):
        """Store the assignment and show the result banner."""
        if self._final_flip == 0:
            self.white_mod,  self.black_mod  = self.mod_a,  self.mod_b
            self.white_name, self.black_name = self.name_a, self.name_b
        else:
            self.white_mod,  self.black_mod  = self.mod_b,  self.mod_a
            self.white_name, self.black_name = self.name_b, self.name_a

        self.banner_var.set(
            f"⬜ {self.white_name} plays WHITE   ·   "
            f"⬛ {self.black_name} plays BLACK"
        )
        self.draw_btn.config(state=tk.NORMAL,   text="🔄  Re-draw")
        self.start_btn.config(state=tk.NORMAL)

    # ── Launch ────────────────────────────────────────────────────────────────
    def _launch_game(self):
        if self.white_mod is None:
            return
        # Destroy this screen and open the game window
        self.root.destroy()

        game_root = tk.Tk()
        gui = ChessGUI(game_root,
                       self.white_mod, self.black_mod,
                       self.white_name, self.black_name)

        def on_close():
            gui.running = False
            game_root.destroy()

        game_root.protocol("WM_DELETE_WINDOW", on_close)
        game_root.mainloop()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        team_a_mod = importlib.import_module(team_a)
        team_b_mod  = importlib.import_module(team_b)
    except ModuleNotFoundError as e:
        print(f"Error importing bot: {e}")
        print("Make sure team_creepers.py and team_goraieb.py are in the same folder as visualize.py")
        sys.exit(1)

    root = tk.Tk()
    ColorDrawScreen(root, team_a_mod, team_b_mod, team_a, team_b)
    root.mainloop()


if __name__ == "__main__":
    main()
