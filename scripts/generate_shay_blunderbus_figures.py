#!/usr/bin/env python3
"""Run team_shay vs BlunderBus matches and write PNG figures for the report."""

from __future__ import annotations

import importlib.util
import os
import random
import sys
import time
from dataclasses import dataclass

import chess
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def load_module(module_name: str, path: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Bot:
    name: str
    mod: object
    depth: int

    def move(self, board: chess.Board, color: chess.Color) -> chess.Move:
        m = self.mod.get_next_move(board, color, self.depth)
        if m is None or m not in board.legal_moves:
            raise RuntimeError(f"{self.name} illegal move")
        return m


def play_game(
    white: Bot,
    black: Bot,
    max_plies: int,
    trace_eval: bool,
    shay_mod,
    blunder_mod,
):
    board = chess.Board()
    t0 = time.perf_counter()
    ply_vals: list[tuple[int, float, float]] = []

    for ply in range(max_plies):
        if board.is_game_over(claim_draw=True):
            break
        side = white if board.turn == chess.WHITE else black
        mv = side.move(board, board.turn)
        board.push(mv)
        if trace_eval:
            ply_vals.append(
                (
                    len(board.move_stack),
                    float(shay_mod.evaluate(board.copy(stack=False))),
                    float(blunder_mod.evaluate(board.copy(stack=False))),
                )
            )

    elapsed = time.perf_counter() - t0
    result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "*"

    return result, len(board.move_stack), elapsed, ply_vals


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "figures"))
    args = p.parse_args()

    shay_path = os.path.join(REPO_ROOT, "team_shay.py")
    bb_path = os.path.join(REPO_ROOT, "BlunderBus", "team_BlunderBus.py")
    shay_mod = load_module("team_shay_report", shay_path)
    bb_mod = load_module("team_blunderbus_report", bb_path)

    shay = Bot("team_shay", shay_mod, args.depth)
    blunder = Bot("BlunderBus", bb_mod, args.depth)

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    wins_shay = draws = wins_bb = unfinished = 0
    plies_list: list[int] = []
    match_points_shay = 0.0
    trace_ply: list[tuple[int, float, float]] = []

    for g in range(args.games):
        a_white = g % 2 == 0
        white = shay if a_white else blunder
        black = blunder if a_white else shay
        trace = g == 0
        result, plies, elapsed, ply_vals = play_game(
            white,
            black,
            args.max_plies,
            trace_eval=trace,
            shay_mod=shay_mod,
            blunder_mod=bb_mod,
        )
        plies_list.append(plies)
        if trace and ply_vals:
            trace_ply = ply_vals

        if result == "*":
            unfinished += 1
        elif result == "1/2-1/2":
            draws += 1
            match_points_shay += 0.5
        elif (result == "1-0" and white.name == "team_shay") or (
            result == "0-1" and black.name == "team_shay"
        ):
            wins_shay += 1
            match_points_shay += 1.0
        elif (result == "1-0" and white.name == "BlunderBus") or (
            result == "0-1" and black.name == "BlunderBus"
        ):
            wins_bb += 1
        else:
            unfinished += 1

        print(f"game {g + 1}/{args.games}  result={result}  plies={plies}  {elapsed:.1f}s")

    # --- Figure 1: outcomes ---
    fig, ax = plt.subplots(figsize=(6, 4))
    cats = ["team_shay\nwins", "Draws", "BlunderBus\nwins"]
    vals = [wins_shay, draws, wins_bb]
    colors = ["#2e7d32", "#757575", "#c62828"]
    ax.bar(cats, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel("Games")
    ax.set_title(
        f"Head-to-head at depth {args.depth} ({args.games} games, seed {args.seed})\n"
        f"Match points (Shay perspective): {match_points_shay:.1f} / {args.games}"
    )
    for i, v in enumerate(vals):
        ax.text(i, v + 0.15, str(v), ha="center", fontsize=11)
    fig.tight_layout()
    p1 = os.path.join(args.out_dir, "shay_vs_blunderbus_outcomes.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # --- Figure 2: game length ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(plies_list, bins=min(20, max(5, args.games // 2)), color="#1565c0", edgecolor="white")
    ax.set_xlabel("Plies (half-moves) per game")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of game length (same match session)")
    ax.axvline(sum(plies_list) / len(plies_list), color="orange", linestyle="--", label="mean")
    ax.legend()
    fig.tight_layout()
    p2 = os.path.join(args.out_dir, "shay_vs_blunderbus_plies.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    # --- Figure 3: eval trace game 1 ---
    if trace_ply:
        xs = [t[0] for t in trace_ply]
        ys = [t[1] for t in trace_ply]
        yb = [t[2] for t in trace_ply]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, label="team_shay.evaluate()", color="#2e7d32", linewidth=1.2)
        ax.plot(xs, yb, label="BlunderBus.evaluate()", color="#c62828", linewidth=1.0, alpha=0.85)
        ax.axhline(0, color="#999", linewidth=0.8)
        ax.set_xlabel("Ply (after each move)")
        ax.set_ylabel("Static score (White-positive, centipawn scale)")
        ax.set_title("Evaluator disagreement along one sample game (game 1 of the session)")
        ax.legend(loc="best")
        fig.tight_layout()
        p3 = os.path.join(args.out_dir, "shay_vs_blunderbus_eval_trace.png")
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        print("Wrote:", p1, p2, p3)
    else:
        print("Wrote:", p1, p2)

    summary_path = os.path.join(args.out_dir, "shay_vs_blunderbus_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(
            f"games={args.games} depth={args.depth} seed={args.seed}\n"
            f"shay_wins={wins_shay} draws={draws} blunderbus_wins={wins_bb} unfinished={unfinished}\n"
            f"match_points_shay={match_points_shay}\n"
        )
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
