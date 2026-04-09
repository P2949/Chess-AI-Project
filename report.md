---
pdf-engine: xelatex

mainfont: Ubuntu
monofont: JetBrains Mono

title: Chess Bot
documentclass: article
fontsize: 12pt
geometry: margin=0.7in
---

Team "Я don't falo angielsku"

| Name | Student ID |
| -- | -- |
| Mykhailo Fedenko | 24426806 |
| Shay Power | 24377295 |
| Pedro Goraieb Fernandes | 24377619 |
| Kuba Rodak | 24436755 |

**Contributions**  
Mykhailo: Creepers engine  
Shay: Blunderbuss engine  
Pedro: AAAA engine  
Kuba: Compiling report

## Introduction

The challenge is to create a minimax-based chessbot with a limited look-ahead search. While the base architecture is provided, the difficulty comes from the open problem of designing a heuristic function that can accurately tell whether a scenario is favorable or not.

Our objective therefore is to make an advanced `evaluate()` function that follows standards where positive values favor White, negative Black, and zero represents a balance or stalemate.

Our function must use the `python-chess` library, be contained in a `.py` file, and must not make use of external engines such as Stockfish. Additionally, we must not modify the code that allows our function to integrate with other teams in the tournament. The engine operates at a depth of 3, meaning that considering performance is not as immediately important as when designing for higher depths.

Our team adopted a divide-and-conquer strategy. We developed multiple unique functions, ranging from pure minimax + alpha-beta material weighting to a complex hybrid neural network. This let us determine not just the best implementation, but also the best overall approach.

## Background / Theory / Prior Knowledge

At the core of the engine is the minimax algorithm, a simple decision making framework using a branching tree of paths to find the most favorable outcome. Additionally, Alpha-Beta pruning is used to eliminate branches that cannot possibly influence the final decision.

The average number of legal moves in chess is ~30-50, which makes it impossible to calculate every move to the end of each game. We instead use a heuristic `evaluate()` function at the leaf nodes of the search tree - this assigns a numerical value to the board state, based on metrics like material advantage, king safety, and control of the center.

To interface with the chess board, we import the `python-chess` library. The `chess.Board` object gives access to the data we need; piece positions, legal move counts, and checkmate detection.

In general, existing chess engines follow these metrics:

- Material, assigning weighted values to different piece types;
- Mobility, a higher number of available legal moves correlates with a better position;
- King Safety, eliminates blunders and allows for high value of castling;
- Positioning, encourages pieces to move to places where they are most useful, using Piece-Square tables. This maximizes piece potential.

## Work / Approaches / Individual (maybe? since we did a few algos each)

So, for each of the algos (pedro's, misha's, shay's):

- overview of the approach (phases or steps)
- core eval strategy (the actual heuristic)
- whether you're using alpha beta pruning, why/why not, banking on opponent mistake, etc
- strengths/weaknesses
- code snippets either here or in the appendix

Essentially, here we want to show not just what we have done and thought about, but also the diversity of our approaches, like I think shay's one will be standing out most here due to using CNN. we can also show our experimentation, what worked/didnt. This is probably our bread and butter of the report.

### Team Shay

After our initial discussion on how we would tackle this project, we concluded that it would be best to approach this from a divide-and-conquer prospective. We would each implement our own evaluation, upon completion we would then pit each implementation against each other. The winner would then be the evaluation method selected to be submitted for the assigment. My original idea was to implement a Convolutional Neural Network (CNN) to play against Stockfish and learn from it, and also include rounds of self play. After a quick prototype was developed, I quickly realized this approach was not viable due to the lack of computation power and research indicating CNNs aren't actually great at chess, even with thousands of compute hours. I decided to then go back to the basics of making a classic heuristic evaluation, and building upon that. After researching how Stockfish functions, I became interested in what is known as the Efficiently Updatable Neural Network (NNUE). This functions as a small neural network that works in conjunction with the classic heuristics to improve overall performance.

#### Overview of Approach

1. **Phase 1 - Baseline**
   Build a strong hand-crafted evaluator first (material + positional terms), then optimize search stability and speed.
2. **Phase 2 - Search quality improvements**
   Keep depth practical (`depth=3`) but improve move quality through ordering and pruning (transposition table, killer/history, quiescence, null move, LMR/PVS).
3. **Phase 3 - NNUE hybrid (`BlunderBus`)**
   Add a compact NNUE that does not replace classical eval, but adjusts it with a bounded correction term.
4. **Phase 4 - Practical training loop**
   Train with generated positions and Stockfish/module labels using mirror augmentation and symmetry loss to reduce evaluator bias.

#### Core eval strategy

- In `team_shay.py`, the core heuristic is a tapered eval (middlegame/endgame blend) with:
  - material values
  - PeSTO-style piece-square tables
  - mobility
  - bishop pair
  - pawn structure (doubled/isolated/passed pawns)
  - rook placement (open/semi-open files, 7th rank, connected rooks)
  - king safety signals
  - tactical pressure on hanging/loose pieces
- In `BlunderBus/team_BlunderBus.py`, the final eval is:
  - `score = classical + blend * clamp(nnue - classical)`
  - This keeps the classical model as the anchor and lets the NNUE provide a controlled positional correction, preventing unstable swings.

#### Alpha-beta pruning / search choices / opponent mistakes

- Yes, alpha-beta pruning is used in both implementations (inside minimax), mainly to increase effective search quality under tight depth limits.
- I did **not** bank on opponent mistakes as the main plan. The strategy is to force better move selection through:
  - strong move ordering (TT move, MVV-LVA captures, killer/history/countermove)
  - quiescence search (to reduce horizon blunders in tactical positions)
  - selective pruning/reductions (null-move, futility, late-move pruning/reduction, aspiration windows)
- Since competition depth is constrained, the goal was "smart depth 3" rather than brute-force deeper search.

#### Strengths / weaknesses

##### Strengths

- Robust classical fallback: if NNUE is weak/unavailable, engine still plays coherent chess.
- Good practical speed-quality tradeoff for constrained compute.
- Hybrid design improves positional nuance without fully depending on expensive deep learning.
- Training pipeline in `BlunderBus/train_nnue.py` includes symmetry checks and mirror augmentation, which improves consistency.

##### Weaknesses

- The limitation of depth 3 highly limits the maximium peformance
- NNUE quality is hit or miss and can sometimes actually hurt the peformance compared to classical.
- Manual feature/weight tuning can be time-consuming and may overfit to observed matchups.
- Not as globally optimized as top engines with massive compute and long training cycles.

## Internal Tournament

Pedro and Misha pitted their implementations against eachother, by playing on Chess.com.

Cover in depth heuristic comparison
We pitted our different solutions against eachother, both to directly improve the engines and to indirectly learn from eachother what works best. We can mention things like what metrics we used, ELO rating on chess.com or otherwise, why/why not. We can give concrete results, even show them over time in a table as we improved the engine.

Go into detail why one engine beat the other. Mention tradeoffs, speed vs accuracy vs depth vs precompute etc.

This is probably also an important section, they love introspection

## Final engine we settled on

Whatever that may be lol - this section should wrap up the previous one with definite conclusions. What we used in the final engine, why. explain with explicit justification.

Final eval function in detail - what features, what piece value, aggresiveness, pawn advantage, etc. Good to link the actual results (higher elo etc) with specific design decisions

Mention that `evaluate()` explicitly returns 0 for stalemates, threefold repetition, and fifty move rule.

## Complexity analysis

We analyzed the final engine in two parts: search complexity first, then evaluation complexity.

### Search

Search is by far the dominant cost. As is well known, the base case for minimax without pruning is $O(b^{d})$ in both time and space, where $b$ is the branching factor (average legal moves) and $d$ is the search depth. As we are operating at a fixed depth of 3, the growth is polynomial relative to the branching factor but exponential relative to depth.

Alpha-beta pruning in the best case (with perfect move ordering) reduces time to $O(b^{d/2})$ - this effectively doubles the depth we can search in the same time. In the worst case scenario with terrible ordering, the advantage degrades back to $O(b^{d})$. In practice, with average ordering, we can expect to get somewhere inbetween, which we have seen to be often cited as $O(b^{3d/4})$. The space complexity remains at $O(d)$ for the entire call stack since we only need to store the current path.

### Evaluation

The `evaluate()` function is called at every leaf node of the search tree. Each call is $O(n)$, where $n$ is the number of owned pieces on the board. The worst case $n = 32$ is effectively equal to $O(1)$ for asymptotic analysis.

Each call to `evaluate()` is $O(n)$ linear time, where $n$ is the number of pieces on the board. The worst case $n = 32$ is effectively equal to $O(1)$ for asymptotic analysis, though this is a consideration for real world performance.

### Real world performance

Given our constraints, we can estimate the workload for typical mid-game positions:

- Branching factor $b$, $$ moves;
- Depth $d$, fixed to 3;
- Nodes visited: at most ~42,000.

At this scale, the bot is expected to complete its move calculation well within realtime play. However, as shown in our depth analysis, the complexity jump to $d$ = 5 and beyond will require much more aggresive pruning and efficient ordering.

## Actual Tourament

Idk if it will happen before we submit this report, but I would also add this section at the end where we analyze how far we got in the tournament. Maybe if we know why we failed or why the competition didnt do as well. What we did differently.

## Conclusion

What worked. What didnt. What we'd do differently. What we'd improve upon more with hindsight.
Why did our different engines behave different from each other?

## Appendix

Code snippets for teams.
Annotated & simplified `evaluate()` for the final engine.
Graph for time & space complexity

### Team Shay snippets

`team_shay.py` (classical heuristic + alpha-beta search):

```python
def evaluate(board: chess.Board) -> float:
    # tapered mg/eg score + pawn structure + rook files + king safety + tactical pressure
    ...

def minimax(board, depth, alpha, beta, maximizing, ply=0, ...):
    # alpha-beta with TT, null-move, PVS/LMR, quiescence frontier
    ...
```

`BlunderBus/team_BlunderBus.py` (NNUE + classical blend):

```python
def evaluate(board: chess.Board) -> float:
    classical = _classical_eval(board)
    nnue = _nnue_eval(board)
    blend = _nnue_blend_weight(board)
    if nnue is None:
        return classical
    delta = max(-delta_cap, min(delta_cap, nnue - classical))
    return classical + blend * delta
```
