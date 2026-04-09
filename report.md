# Chess Report

Names, roles, contributions, attributions, etc.

## Introduction

- Define problem (heuristic hard/arbitrary w/e)
- define objective here (create heuristic algo etc)
- define constraints (minmax depth 3, etc)
- outline our approach/philosophy

## Background / Theory / Prior Knowledge

We wanna show we understand what we're doing.
Outline some basic chess & engine knowledge, alpha beta pruning, game tree, minmax, etc
Evaluation functions, heuristic algorithms, existing chess engines

## Work / Approaches / Individual (maybe? since we did a few algos each)

So, for each of the algos (pedro's, misha's, shay's):

- overview of the approach (phases or steps)
- core eval strategy (the actual heuristic)
- whether you're using alpha beta pruning, why/why not, banking on opponent mistake, etc
- strengths/weaknesses
- code snippets either here or in the appendix

Essentially, here we want to show not just what we have done and thought about, but also the diversity of our approaches, like I think shay's one will be standing out most here due to using CNN. we can also show our experimentation, what worked/didnt. This is probably our bread and butter of the report.

### Team Shay

After our initial discuession on how we would tackle this project, we concluded that it would be best to approach this from a divide-and-conquer prospective. 3 of us would each impliment our own evalation, upon completition we would then pit each implimentation against each other. The winner would then be the evalation method selected to be submitted for the assigment. My original idea was to impliment a Convolutional neural network (CNN) to play against stockfish and learn from it, and also include rounds of self play. after a quick prototype was developed, i quickly realized this approach was not viable due to the lack of computation power and research indicating CNN's arent actually great at chess, even with thousands of compute hours. i decided to then go back to the basics of making a classical heuristic evalation, and building upon that. after researching how stockfish functions, i became intrested in what is known as the Efficiently updatable neural network (NNUE). this functions as a small neural network that works in combination with the classical heuristics to improve overall performance.

#### Overview of the approach (phases or steps)

1. **Phase 1 - Classical baseline (`team_shay.py`)**
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

**Strengths**
- Robust classical fallback: if NNUE is weak/unavailable, engine still plays coherent chess.
- Good practical speed-quality tradeoff for constrained compute.
- Hybrid design improves positional nuance without fully depending on expensive deep learning.
- Training pipeline in `BlunderBus/train_nnue.py` includes symmetry checks and mirror augmentation, which improves consistency.

**Weaknesses**
- The limitation of depth 3 highly limits the maximium peformance
- NNUE quality is hit or miss and can sometimes actually hurt the peformance compared to classical.
- Manual feature/weight tuning can be time-consuming and may overfit to observed matchups.
- Not as globally optimized as top engines with massive compute and long training cycles.

## Internal Tournament

We pitted our different solutions against eachother, both to directly improve the engines and to indirectly learn from eachother what works best. We can mention things like what metrics we used, ELO rating on chess.com or otherwise, why/why not. We can give concrete results, even show them over time in a table as we improved the engine.

Go into detail why one engine beat the other. Mention tradeoffs, speed vs accuracy vs depth vs precompute etc.

This is probably also an important section, they love introspection

## Final engine we settled on

Whatever that may be lol - this section should wrap up the previous one with definite conclusions. What we used in the final engine, why. explain

Final eval function in detail - what features, what piece value, aggresiveness, pawn advantage, etc. Good to link the actual results (higher elo etc) with specific design decisions

## Complexity analysis

Depending on how much we can write about this, we either take this out into its own section, or tack it onto the end of the last one. We want to show analysis of the time complexity of the algorithm. We can definitely include the minmax itself in this, even though its provided from the start. Talk about the practicality of using this engine in real life, with real constraints on resources, talking about how its not just theoretically a good engine, but practically too. Why it is so.

## Actual Tourament

Idk if it will happen before we submit this report, but I would also add this section at the end where we analyze how far we got in the tournament. Maybe if we know why we failed or why the competition didnt do as well. What we did differently.

## Conclusion

What worked. What didnt. What we'd do differently. What we'd improve upon more with hindsight.
Why did our different engines behave different from each other?

## Appendix

If we want to include pseudocode and code snippets, here they go.
We could add graphs and tables/logs, or whatever else here.

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
