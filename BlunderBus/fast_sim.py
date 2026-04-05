"""
Small bot-vs-bot runner for quick testing.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import random
import time
from dataclasses import dataclass

import chess


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    if seconds < 3600.0:
        mins = int(seconds // 60)
        rem = int(seconds % 60)
        return f"{mins}m {rem}s"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m"


def load_module_from_path(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class BotRef:
    name: str
    path: str
    module: object
    depth: int

    def pick_move(self, board: chess.Board, color: chess.Color) -> chess.Move:
        move = self.module.get_next_move(board, color, self.depth)
        if move is None:
            raise RuntimeError(f"{self.name} returned None")
        if move not in board.legal_moves:
            raise RuntimeError(f"{self.name} returned illegal move: {move}")
        return move


def play_game(
    white: BotRef,
    black: BotRef,
    start_fen: str | None = None,
    max_plies: int = 300,
    verbose: bool = False,
) -> tuple[str, int, float]:
    board = chess.Board(start_fen) if start_fen else chess.Board()
    t0 = time.perf_counter()

    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break

        side = white if board.turn == chess.WHITE else black
        move = side.pick_move(board, board.turn)
        san = board.san(move)
        board.push(move)

        if verbose:
            print(f"{ply + 1:03d}. {side.name}: {san} ({move.uci()})")

    elapsed = time.perf_counter() - t0
    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"
    return result, len(board.move_stack), elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple bot-vs-bot chess runner.")
    parser.add_argument("--bot-a", default="BlunderBus/team_BlunderBus.py")
    parser.add_argument("--bot-b", default="team_aaaaaaaaaaaaaaa.py")
    parser.add_argument("--name-a", default="team_BlunderBus")
    parser.add_argument("--name-b", default="team_aaaaaaaaaaaaaaa")
    parser.add_argument("--depth-a", type=int, default=3)
    parser.add_argument("--depth-b", type=int, default=3)
    parser.add_argument("--games", type=int, default=2)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start-fen", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    bot_a_path = args.bot_a if os.path.isabs(args.bot_a) else os.path.join(repo_root, args.bot_a)
    bot_b_path = args.bot_b if os.path.isabs(args.bot_b) else os.path.join(repo_root, args.bot_b)

    mod_a = load_module_from_path("bot_a_mod", bot_a_path)
    mod_b = load_module_from_path("bot_b_mod", bot_b_path)
    if not hasattr(mod_a, "get_next_move") or not hasattr(mod_b, "get_next_move"):
        raise RuntimeError("Both bot modules must expose get_next_move(board, color, depth)")

    random.seed(args.seed)

    score_a = 0.0
    score_b = 0.0
    draws = 0
    unfinished = 0
    total_plies = 0
    total_secs = 0.0
    match_t0 = time.perf_counter()

    print(f"Bot A: {args.name_a} ({bot_a_path}) depth={args.depth_a}")
    print(f"Bot B: {args.name_b} ({bot_b_path}) depth={args.depth_b}")
    print(f"Games: {args.games}  max_plies: {args.max_plies}")
    print("-" * 72)

    for g in range(args.games):
        a_is_white = (g % 2 == 0)
        if a_is_white:
            white = BotRef(args.name_a, bot_a_path, mod_a, args.depth_a)
            black = BotRef(args.name_b, bot_b_path, mod_b, args.depth_b)
        else:
            white = BotRef(args.name_b, bot_b_path, mod_b, args.depth_b)
            black = BotRef(args.name_a, bot_a_path, mod_a, args.depth_a)

        result, plies, elapsed = play_game(
            white=white,
            black=black,
            start_fen=args.start_fen,
            max_plies=args.max_plies,
            verbose=args.verbose,
        )
        total_plies += plies
        total_secs += elapsed

        if result == "1-0":
            winner = white.name
            if a_is_white:
                score_a += 1.0
            else:
                score_b += 1.0
        elif result == "0-1":
            winner = black.name
            if a_is_white:
                score_b += 1.0
            else:
                score_a += 1.0
        elif result == "1/2-1/2":
            winner = "Draw"
            draws += 1
            score_a += 0.5
            score_b += 0.5
        else:
            winner = "Unfinished"
            unfinished += 1

        games_done = g + 1
        avg_secs_so_far = total_secs / games_done
        eta_remaining = avg_secs_so_far * max(0, args.games - games_done)
        elapsed_match = time.perf_counter() - match_t0

        print(
            f"Game {games_done:02d} | "
            f"{white.name} (W) vs {black.name} (B) | "
            f"result={result} | winner={winner} | plies={plies} | {elapsed:.2f}s | "
            f"match_elapsed={format_duration(elapsed_match)} | "
            f"eta_remaining={format_duration(eta_remaining)}"
        )

    avg_plies = total_plies / max(1, args.games)
    avg_secs = total_secs / max(1, args.games)
    print("-" * 72)
    print(f"Final score: {args.name_a} {score_a:.1f} - {score_b:.1f} {args.name_b}")
    print(f"Draws: {draws}/{args.games}")
    print(f"Unfinished: {unfinished}/{args.games}")
    print(f"Average plies/game: {avg_plies:.1f}")
    print(f"Average wall-time/game: {avg_secs:.2f}s")


if __name__ == "__main__":
    main()
