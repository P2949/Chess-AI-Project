"""
visualize.py  —  Chess visualizer  (Human vs AI  ·  Bot vs Bot)

At startup a mode-select screen lets you choose:
  • Human vs AI  — pick your opponent bot, pick your colour
  • Bot vs Bot   — random colour draw, then watch

Requirements:
    pip install python-chess
    Python standard library: tkinter  (ships with most Python installs)

Run:
    python visualize.py

The following files must be in the same directory:
    visualize.py
    team_alpha.py
    team_goraieb.py

Each bot module must expose:
    get_next_move(board, color, depth) -> chess.Move
    evaluate(board)                    -> float   (optional — used for eval bar)
"""

import tkinter as tk
from tkinter import font as tkfont
import chess
import importlib
import random
import sys
import os

# ── Configuration ─────────────────────────────────────────────────────────────
DEPTH       = 2       # bot search depth  (keep ≤3 for sub-second moves)
MOVE_DELAY  = 1200    # ms pause after each bot move so you can see the board
SQUARE_SIZE = 72      # pixels per square
BOARD_SIZE  = SQUARE_SIZE * 8

# ── Colours ───────────────────────────────────────────────────────────────────
LIGHT_SQ   = "#F0D9B5"
DARK_SQ    = "#B58863"
HIGHLIGHT  = "#F6F669"   # last-move: from-square
HIGHLIGHT2 = "#CDD16E"   # last-move: to-square
SELECT_SQ  = "#7FC97F"   # human-selected piece
CHECK_SQ   = "#FF5555"   # king in check
WHITE_PIECE = "#FFFFFF"
BLACK_PIECE = "#1A1A1A"
BORDER_COL  = "#8B6914"
BG          = "#2C2C2C"
TEXT_COL    = "#E8E8E8"
ACCENT      = "#5B9BD5"
LEGAL_COL   = "#3AAA3A"

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
    """
    Renders a chess game between two players, each of which may be either a bot
    module (exposing get_next_move / evaluate) or a human (pass None).

    bot_white = None  →  the human plays White
    bot_black = None  →  the human plays Black
    Both non-None      →  watch mode (Bot vs Bot)
    """

    def __init__(self, root: tk.Tk,
                 bot_white, bot_black,
                 name_white: str, name_black: str):
        self.root       = root
        self.bot_white  = bot_white
        self.bot_black  = bot_black
        self.name_white = name_white
        self.name_black = name_black
        self.board      = chess.Board()
        self.last_move  = None
        self.running    = True

        # Human interaction state
        self.selected_sq        = None   # square the human has clicked on
        self.legal_moves_for_sq = []     # legal moves from selected_sq

        root.title(f"Chess — {name_white} (White)  vs  {name_black} (Black)")
        root.configure(bg=BG)
        root.resizable(False, False)
        self._build_ui()
        self._draw_board()
        self.root.after(200, self._start_game)

    # ── UI construction ───────────────────────────────────────────────────────
    def _build_ui(self):
        board_frame = tk.Frame(self.root, bg=BG)
        board_frame.grid(row=0, column=0, padx=16, pady=16)

        # File labels (a–h) above the board
        file_bar = tk.Frame(board_frame, bg=BG)
        file_bar.pack()
        tk.Label(file_bar, text="  ", bg=BG, width=2).pack(side=tk.LEFT)
        for f in "abcdefgh":
            tk.Label(file_bar, text=f, bg=BG, fg="#888888",
                     width=SQUARE_SIZE // 10,
                     font=("Arial", 10)).pack(side=tk.LEFT,
                                              padx=SQUARE_SIZE // 2 - 8)

        inner = tk.Frame(board_frame, bg=BG)
        inner.pack()

        # Rank labels (8–1) to the left
        rank_frame = tk.Frame(inner, bg=BG)
        rank_frame.pack(side=tk.LEFT)
        for r in range(8, 0, -1):
            tk.Label(rank_frame, text=str(r), bg=BG, fg="#888888",
                     font=("Arial", 10),
                     height=SQUARE_SIZE // 16).pack(pady=SQUARE_SIZE // 2 - 8)

        self.canvas = tk.Canvas(inner,
                                width=BOARD_SIZE, height=BOARD_SIZE,
                                bg=BG, highlightthickness=2,
                                highlightbackground=BORDER_COL)
        self.canvas.pack(side=tk.LEFT)

        # ── Right panel ──────────────────────────────────────────────────────
        right = tk.Frame(self.root, bg=BG, width=240)
        right.grid(row=0, column=1, padx=(0, 16), pady=16, sticky="ns")
        right.grid_propagate(False)

        tk.Label(right, text="♟  CHESS", bg=BG, fg=ACCENT,
                 font=("Arial", 13, "bold")).pack(pady=(8, 2))
        tk.Label(right, text=f"⬜  {self.name_white}  (White)",
                 bg=BG, fg=TEXT_COL, font=("Arial", 10)).pack()
        tk.Label(right, text=f"⬛  {self.name_black}  (Black)",
                 bg=BG, fg=TEXT_COL, font=("Arial", 10)).pack(pady=(0, 12))

        self.status_var = tk.StringVar(value="Starting…")
        tk.Label(right, textvariable=self.status_var,
                 bg=BG, fg="#F0C040", font=("Arial", 11, "bold"),
                 wraplength=220, justify=tk.CENTER).pack(pady=(0, 10))

        self.move_count_var = tk.StringVar(value="Move: 0")
        tk.Label(right, textvariable=self.move_count_var,
                 bg=BG, fg=TEXT_COL, font=("Arial", 10)).pack()

        self.eval_var = tk.StringVar(value="Eval: —")
        tk.Label(right, textvariable=self.eval_var,
                 bg=BG, fg="#88CC88", font=("Arial", 10)).pack(pady=(0, 8))

        tk.Frame(right, bg="#444444", height=1).pack(fill=tk.X, pady=4)
        tk.Label(right, text="Move Log", bg=BG, fg="#AAAAAA",
                 font=("Arial", 9, "bold")).pack()

        log_frame = tk.Frame(right, bg=BG)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        sb = tk.Scrollbar(log_frame, bg="#444444")
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text = tk.Text(log_frame, width=22, bg="#1E1E1E", fg=TEXT_COL,
                                font=("Courier", 9), state=tk.DISABLED,
                                yscrollcommand=sb.set, relief=tk.FLAT, borderwidth=4)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self.log_text.yview)

    # ── Board rendering ───────────────────────────────────────────────────────
    def _draw_board(self):
        self.canvas.delete("all")
        piece_font = tkfont.Font(family="Segoe UI Symbol",
                                 size=int(SQUARE_SIZE * 0.58))

        last_from  = self.last_move.from_square if self.last_move else None
        last_to    = self.last_move.to_square   if self.last_move else None
        legal_dests = {m.to_square for m in self.legal_moves_for_sq}

        # Checked king square (highlight red)
        check_sq = None
        if self.board.is_check():
            check_sq = self.board.king(self.board.turn)

        for rank in range(7, -1, -1):
            for file in range(8):
                sq = chess.square(file, rank)
                x0 = file       * SQUARE_SIZE
                y0 = (7 - rank) * SQUARE_SIZE
                x1, y1 = x0 + SQUARE_SIZE, y0 + SQUARE_SIZE

                # ── Square colour ──────────────────────────────────────────
                if sq == self.selected_sq:
                    col = SELECT_SQ
                elif sq == check_sq:
                    col = CHECK_SQ
                elif sq == last_from:
                    col = HIGHLIGHT
                elif sq == last_to:
                    col = HIGHLIGHT2
                elif (rank + file) % 2 == 0:
                    col = DARK_SQ
                else:
                    col = LIGHT_SQ

                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill=col, outline="")

                # ── Legal-move indicators ──────────────────────────────────
                if sq in legal_dests:
                    cx, cy = x0 + SQUARE_SIZE // 2, y0 + SQUARE_SIZE // 2
                    if self.board.piece_at(sq):
                        # Capture: draw a ring around the target square
                        self.canvas.create_oval(x0 + 3, y0 + 3,
                                                x1 - 3, y1 - 3,
                                                outline=LEGAL_COL, width=4,
                                                fill="")
                    else:
                        # Quiet move: small filled dot
                        r = SQUARE_SIZE // 6
                        self.canvas.create_oval(cx - r, cy - r,
                                                cx + r, cy + r,
                                                fill=LEGAL_COL, outline="")

                # ── Piece glyph ────────────────────────────────────────────
                piece = self.board.piece_at(sq)
                if piece:
                    sym = PIECE_UNICODE[(piece.piece_type, piece.color)]
                    fg  = WHITE_PIECE if piece.color == chess.WHITE else BLACK_PIECE
                    cx, cy = x0 + SQUARE_SIZE // 2, y0 + SQUARE_SIZE // 2
                    if piece.color == chess.WHITE:
                        # Thin shadow for white pieces
                        self.canvas.create_text(cx + 1, cy + 1, text=sym,
                                                font=piece_font, fill="#555555")
                    self.canvas.create_text(cx, cy, text=sym,
                                            font=piece_font, fill=fg)

    # ── Move-log helper ───────────────────────────────────────────────────────
    def _append_log(self, move_num: int, white_san: str, black_san: str = ""):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END,
                             f"{move_num:>3}.  {white_san:<8} {black_san}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    # ── Game flow ─────────────────────────────────────────────────────────────
    def _start_game(self):
        self.move_number       = 1
        self.white_san_pending = ""
        self._next_ply()

    def _next_ply(self):
        """Dispatch the current half-move to the bot or the human handler."""
        if self.board.is_game_over() or not self.running:
            self._show_result()
            return

        color = self.board.turn
        bot   = self.bot_white if color == chess.WHITE else self.bot_black

        if bot is None:
            self._human_turn(color)
        else:
            self._bot_turn(color, bot)

    # ── Bot turn ──────────────────────────────────────────────────────────────
    def _bot_turn(self, color, bot):
        name = self.name_white if color == chess.WHITE else self.name_black
        sym  = "⚪" if color == chess.WHITE else "⚫"
        self.status_var.set(f"{sym}  {name} thinking…")
        self.root.update_idletasks()

        move = bot.get_next_move(self.board, color, DEPTH)
        if move is None or move not in self.board.legal_moves:
            opponent = self.name_black if color == chess.WHITE else self.name_white
            self.status_var.set(
                f"{opponent} WINS — illegal move by {name}")
            return

        self._push_move(move)
        self.root.after(MOVE_DELAY, self._next_ply)

    # ── Human turn ────────────────────────────────────────────────────────────
    def _human_turn(self, color):
        name = self.name_white if color == chess.WHITE else self.name_black
        sym  = "⚪" if color == chess.WHITE else "⚫"
        self.status_var.set(f"{sym}  {name} — your turn")
        # Bind click; _on_human_click will unbind itself once a move is made
        self.canvas.bind("<Button-1>", self._on_human_click)

    def _on_human_click(self, event):
        file = event.x // SQUARE_SIZE
        rank = 7 - event.y // SQUARE_SIZE
        if not (0 <= file < 8 and 0 <= rank < 8):
            return
        sq = chess.square(file, rank)

        if self.selected_sq is None:
            self._try_select(sq)
        else:
            self._try_move(sq)

    def _try_select(self, sq):
        """Highlight a friendly piece and compute its legal moves."""
        piece = self.board.piece_at(sq)
        if piece and piece.color == self.board.turn:
            self.selected_sq        = sq
            self.legal_moves_for_sq = [m for m in self.board.legal_moves
                                        if m.from_square == sq]
            self._draw_board()

    def _try_move(self, sq):
        """
        Attempt to move the selected piece to sq.
        Handles: normal moves, re-selection of another friendly piece,
                 deselection, and pawn promotion.
        """
        candidates = [m for m in self.legal_moves_for_sq
                      if m.to_square == sq]

        if not candidates:
            # Clicked elsewhere — maybe a different friendly piece
            piece = self.board.piece_at(sq)
            if piece and piece.color == self.board.turn:
                self._try_select(sq)
            else:
                # Deselect
                self.selected_sq        = None
                self.legal_moves_for_sq = []
                self._draw_board()
            return

        # Determine the exact move
        if len(candidates) == 1:
            move = candidates[0]
        else:
            # Multiple candidates → pawn promotion (same from/to, 4 piece types)
            promo = self._ask_promotion()
            move  = next((m for m in candidates if m.promotion == promo),
                         candidates[0])

        # Commit the move
        self.canvas.unbind("<Button-1>")
        self.selected_sq        = None
        self.legal_moves_for_sq = []
        self._push_move(move)
        # Tiny delay so the board repaints before we dispatch the next ply
        self.root.after(50, self._next_ply)

    # ── Promotion dialog ──────────────────────────────────────────────────────
    def _ask_promotion(self) -> int:
        """Modal dialog — returns the chosen piece type (chess.QUEEN etc.)."""
        result = [chess.QUEEN]
        color  = self.board.turn

        dlg = tk.Toplevel(self.root)
        dlg.title("Promote Pawn")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        dlg.grab_set()
        dlg.lift()

        tk.Label(dlg, text="Promote to:", bg=BG, fg=TEXT_COL,
                 font=("Arial", 12, "bold")).pack(padx=24, pady=(18, 8))

        options = [
            (chess.QUEEN,  PIECE_UNICODE[(chess.QUEEN,  color)], "Queen"),
            (chess.ROOK,   PIECE_UNICODE[(chess.ROOK,   color)], "Rook"),
            (chess.BISHOP, PIECE_UNICODE[(chess.BISHOP, color)], "Bishop"),
            (chess.KNIGHT, PIECE_UNICODE[(chess.KNIGHT, color)], "Knight"),
        ]
        bf = tk.Frame(dlg, bg=BG)
        bf.pack(padx=24, pady=(0, 20))
        for pt, sym, name in options:
            tk.Button(
                bf, text=f"{sym}\n{name}",
                bg="#3A3A3A", fg=TEXT_COL,
                activebackground="#555555",
                font=("Arial", 14), width=5, height=2, relief=tk.FLAT,
                command=lambda p=pt: [result.__setitem__(0, p), dlg.destroy()]
            ).pack(side=tk.LEFT, padx=4)

        dlg.wait_window()
        return result[0]

    # ── Shared move-push logic ────────────────────────────────────────────────
    def _push_move(self, move: chess.Move):
        """Record, push, and redraw for any committed move (bot or human)."""
        color = self.board.turn
        san   = self.board.san(move)

        if color == chess.WHITE:
            self.white_san_pending = san
        else:
            self._append_log(self.move_number, self.white_san_pending, san)
            self.white_san_pending = ""
            self.move_number += 1

        self.last_move = move
        self.board.push(move)

        # Try to get an eval from whichever bot module is available
        eval_bot = self.bot_white or self.bot_black
        try:
            score    = eval_bot.evaluate(self.board)
            eval_str = f"Eval: {score:+.0f}"
        except Exception:
            eval_str = "Eval: —"

        self.move_count_var.set(
            f"Move: {self.board.fullmove_number}  |  "
            f"Ply: {len(self.board.move_stack)}")
        self.eval_var.set(eval_str)
        self._draw_board()
        self.root.update_idletasks()

    # ── End of game ───────────────────────────────────────────────────────────
    def _show_result(self):
        if getattr(self, "white_san_pending", ""):
            self._append_log(self.move_number, self.white_san_pending, "")

        res = self.board.result()
        if res == "1-0":
            msg = f"✅  {self.name_white} (White) WINS!"
        elif res == "0-1":
            msg = f"✅  {self.name_black} (Black) WINS!"
        else:
            msg = "🤝  Draw!"

        reason = ""
        if   self.board.is_checkmate():               reason = " by Checkmate"
        elif self.board.is_stalemate():               reason = " by Stalemate"
        elif self.board.is_insufficient_material():   reason = " — Insufficient Material"
        elif self.board.is_seventyfive_moves():       reason = " — 75-Move Rule"
        elif self.board.is_fivefold_repetition():     reason = " — Fivefold Repetition"

        self.status_var.set(msg + reason)
        self._draw_board()


# ── Colour-draw screen (Bot vs Bot only) ──────────────────────────────────────
class ColorDrawScreen:
    """
    Slot-machine colour draw for the Bot vs Bot mode.
    Identical to the original implementation.
    """
    SPIN_SCHEDULE = (
        [50]  * 10 +
        [100] * 6  +
        [200] * 4  +
        [400] * 2  +
        [600] * 1
    )

    def __init__(self, root: tk.Tk, mod_a, mod_b, name_a: str, name_b: str):
        self.root   = root
        self.mod_a  = mod_a
        self.mod_b  = mod_b
        self.name_a = name_a
        self.name_b = name_b
        self.white_mod  = self.black_mod  = None
        self.white_name = self.black_name = None
        self._spin_index   = 0
        self._current_flip = 0
        self._spinning     = False

        root.title("Chess Bot Battle — Colour Draw")
        root.configure(bg=BG)
        root.resizable(False, False)
        self._build_ui()

    def _build_ui(self):
        pad = dict(padx=30, pady=10)
        tk.Label(self.root, text="⚔  COLOUR DRAW", bg=BG, fg=ACCENT,
                 font=("Arial", 20, "bold")).pack(pady=(30, 4))
        tk.Label(self.root,
                 text="Click the button to randomly assign White & Black",
                 bg=BG, fg="#AAAAAA", font=("Arial", 10)).pack(pady=(0, 24))

        cards = tk.Frame(self.root, bg=BG)
        cards.pack(**pad)

        self.card_a = tk.Frame(cards, bg="#333333", relief=tk.FLAT,
                               padx=20, pady=16, bd=2)
        self.card_a.grid(row=0, column=0, padx=18)
        self.label_a_name  = tk.Label(self.card_a, text=self.name_a,
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

        tk.Label(cards, text="vs", bg=BG, fg="#666666",
                 font=("Arial", 16, "bold")).grid(row=0, column=1, padx=10)

        self.card_b = tk.Frame(cards, bg="#333333", relief=tk.FLAT,
                               padx=20, pady=16, bd=2)
        self.card_b.grid(row=0, column=2, padx=18)
        self.label_b_name  = tk.Label(self.card_b, text=self.name_b,
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

        self.banner_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.banner_var,
                 bg=BG, fg="#F0C040", font=("Arial", 13, "bold"),
                 wraplength=400, justify=tk.CENTER).pack(pady=(18, 4))

        btn_frame = tk.Frame(self.root, bg=BG)
        btn_frame.pack(pady=(10, 30))
        self.draw_btn = tk.Button(
            btn_frame, text="🎲  Draw Colors!",
            bg=ACCENT, fg="white", activebackground="#3A7DC9",
            font=("Arial", 12, "bold"), relief=tk.FLAT,
            padx=18, pady=8, cursor="hand2",
            command=self._start_spin)
        self.draw_btn.pack(side=tk.LEFT, padx=10)
        self.start_btn = tk.Button(
            btn_frame, text="▶  Start Game",
            bg="#27AE60", fg="white", activebackground="#1E8449",
            font=("Arial", 12, "bold"), relief=tk.FLAT,
            padx=18, pady=8, cursor="hand2",
            command=self._launch_game, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=10)

    def _start_spin(self):
        if self._spinning:
            return
        self._spinning    = True
        self._spin_index  = 0
        self.draw_btn.config(state=tk.DISABLED)
        self.start_btn.config(state=tk.DISABLED)
        self.banner_var.set("")
        self._final_flip   = random.randint(0, 1)
        self._current_flip = random.randint(0, 1)
        self._do_spin_step()

    def _do_spin_step(self):
        if self._spin_index >= len(self.SPIN_SCHEDULE):
            self._current_flip = self._final_flip
            self._update_cards(final=True)
            self._spinning = False
            self._reveal_result()
            return
        self._current_flip ^= 1
        self._update_cards(final=False)
        delay = self.SPIN_SCHEDULE[self._spin_index]
        self._spin_index += 1
        self.root.after(delay, self._do_spin_step)

    def _update_cards(self, final: bool):
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
        if self._final_flip == 0:
            self.white_mod,  self.black_mod  = self.mod_a,  self.mod_b
            self.white_name, self.black_name = self.name_a, self.name_b
        else:
            self.white_mod,  self.black_mod  = self.mod_b,  self.mod_a
            self.white_name, self.black_name = self.name_b, self.name_a
        self.banner_var.set(
            f"⬜ {self.white_name} plays WHITE   ·   "
            f"⬛ {self.black_name} plays BLACK")
        self.draw_btn.config(state=tk.NORMAL, text="🔄  Re-draw")
        self.start_btn.config(state=tk.NORMAL)

    def _launch_game(self):
        if self.white_mod is None:
            return
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


# ── Mode-select screen (entry point) ─────────────────────────────────────────
class ModeSelectScreen:
    """
    First screen shown at startup.

    Offers two modes:
      👤 Human vs AI   — pick the opponent bot and your colour
      🤖 Bot vs Bot    — goes to ColorDrawScreen for the classic colour draw
    """

    def __init__(self, root: tk.Tk, mods: dict, names: list):
        """
        mods  : {name: module}  — all loaded bot modules
        names : list of name strings (same order as mods)
        """
        self.root  = root
        self.mods  = mods
        self.names = names

        root.title("Chess — Select Mode")
        root.configure(bg=BG)
        root.resizable(False, False)
        self._build_ui()

    def _build_ui(self):
        tk.Label(self.root, text="♟  CHESS", bg=BG, fg=ACCENT,
                 font=("Arial", 26, "bold")).pack(pady=(44, 4))
        tk.Label(self.root, text="Choose your game mode",
                 bg=BG, fg="#AAAAAA",
                 font=("Arial", 11)).pack(pady=(0, 28))

        tk.Button(
            self.root, text="👤   Human vs AI",
            bg=ACCENT, fg="white", activebackground="#3A7DC9",
            font=("Arial", 14, "bold"), relief=tk.FLAT,
            padx=30, pady=12, cursor="hand2", width=22,
            command=self._show_human_options
        ).pack(pady=6)

        tk.Button(
            self.root, text="🤖   Bot vs Bot",
            bg="#555555", fg="white", activebackground="#777777",
            font=("Arial", 14, "bold"), relief=tk.FLAT,
            padx=30, pady=12, cursor="hand2", width=22,
            command=self._launch_bot_vs_bot
        ).pack(pady=6)

        # Human-options panel (packed on demand)
        self._human_panel = tk.Frame(self.root, bg=BG)
        self._build_human_panel()

        # Bottom padding
        tk.Label(self.root, text="", bg=BG).pack(pady=16)

    def _build_human_panel(self):
        panel = self._human_panel

        tk.Frame(panel, bg="#444444", height=1).pack(fill=tk.X,
                                                     padx=30, pady=(16, 10))

        tk.Label(panel, text="Choose AI opponent:",
                 bg=BG, fg=TEXT_COL, font=("Arial", 11)).pack()
        self.ai_var = tk.StringVar(value=self.names[0])
        for name in self.names:
            tk.Radiobutton(panel, text=name, variable=self.ai_var, value=name,
                           bg=BG, fg=TEXT_COL, selectcolor="#444444",
                           activebackground=BG,
                           font=("Arial", 11)).pack(anchor="w", padx=80)

        tk.Label(panel, text="Play as:", bg=BG, fg=TEXT_COL,
                 font=("Arial", 11)).pack(pady=(10, 2))
        self.color_var = tk.StringVar(value="white")
        for text, val in [("⬜  White", "white"),
                           ("⬛  Black", "black"),
                           ("🎲  Random", "random")]:
            tk.Radiobutton(panel, text=text, variable=self.color_var, value=val,
                           bg=BG, fg=TEXT_COL, selectcolor="#444444",
                           activebackground=BG,
                           font=("Arial", 11)).pack(anchor="w", padx=80)

        tk.Button(
            panel, text="▶  Start Game",
            bg="#27AE60", fg="white", activebackground="#1E8449",
            font=("Arial", 12, "bold"), relief=tk.FLAT,
            padx=18, pady=8, cursor="hand2",
            command=self._launch_human_vs_ai
        ).pack(pady=(14, 4))

    def _show_human_options(self):
        """Reveal the human-vs-AI option panel (idempotent)."""
        self._human_panel.pack(pady=(0, 10))

    def _launch_human_vs_ai(self):
        ai_name = self.ai_var.get()
        ai_mod  = self.mods[ai_name]
        color   = self.color_var.get()
        if color == "random":
            color = random.choice(["white", "black"])

        if color == "white":
            bot_white, bot_black   = None,   ai_mod
            name_white, name_black = "You",  ai_name
        else:
            bot_white, bot_black   = ai_mod,   None
            name_white, name_black = ai_name, "You"

        self.root.destroy()
        game_root = tk.Tk()
        gui = ChessGUI(game_root, bot_white, bot_black, name_white, name_black)
        def on_close():
            gui.running = False
            game_root.destroy()
        game_root.protocol("WM_DELETE_WINDOW", on_close)
        game_root.mainloop()

    def _launch_bot_vs_bot(self):
        mod_list  = list(self.mods.values())
        name_list = list(self.mods.keys())
        self.root.destroy()
        draw_root = tk.Tk()
        ColorDrawScreen(draw_root, mod_list[0], mod_list[1],
                        name_list[0], name_list[1])
        draw_root.mainloop()


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    try:
        team_alpha   = importlib.import_module("team_alpha")
        team_goraieb = importlib.import_module("team_goraieb")
    except ModuleNotFoundError as e:
        print(f"Error importing bot: {e}")
        print("Make sure team_alpha.py and team_goraieb.py are "
              "in the same folder as visualize.py")
        sys.exit(1)

    mods = {
        "team_alpha":   team_alpha,
        "team_goraieb": team_goraieb,
    }
    root = tk.Tk()
    ModeSelectScreen(root, mods, list(mods.keys()))
    root.mainloop()


if __name__ == "__main__":
    main()
