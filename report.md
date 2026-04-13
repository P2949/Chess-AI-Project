---
pdf-engine: xelatex

mainfont: Carlito
monofont: Consolas

title: Assignment 2 - Chess Bot
documentclass: article
fontsize: 12pt
geometry: margin=0.8in
---

Team "Я don't falo angielsku"

| Name | Student ID |
| -- | -- |
| Mykhailo Fedenko | 24426806 |
| Shay Power | 24377295 |
| Pedro Goraieb Fernandes | 24377619 |
| Kuba Rodak | 24436755 |

**Work Contributions**  
Contributions to the project are equal among members.

**Code Repository**
The code is available on GitHub:  
https://github.com/P2949/Chess-AI-Project

## Introduction

The challenge is to create an evaluation algorithm for a Minimax-based chessbot with a limited look-ahead search depth. While the base architecture is provided, the difficulty comes from the open problem of designing a heuristic function that can accurately tell whether a scenario is favorable or not.

Our objective therefore is to make an advanced `evaluate()` function that follows standards where positive values favor White, negative Black, and zero represents a balance or stalemate.

Our function must use the `python-chess` library, be contained in a `.py` file, and must not make use of external engines such as Stockfish. Additionally, we must not modify the code that allows our function to integrate with other teams in the tournament. The engine operates at a depth of 3, meaning that considering performance is not as immediately important as when designing for higher depths.

Our team adopted a divide-and-conquer strategy. We developed multiple unique functions, ranging from pure Minimax + alpha-beta material weighting to a complex hybrid neural network. This let us determine not just the best implementation, but also the best overall approach.

## Background & Prior Knowledge

At the core of the engine is the Minimax algorithm, a simple decision making framework using a branching tree of paths to find the most favorable outcome. Additionally, alpha-beta pruning is used to eliminate branches that cannot possibly influence the final decision.

The average number of legal moves in chess is approximately $35$, and the average game length is about $40$, which gives use the impractically large $35^{40} \approx 10^{61}$ estimate of possible states. This in turn makes it impossible to brute-force calculate every move to the end of each game. We instead use a heuristic `evaluate()` function at the leaf nodes of the search tree - this assigns a numerical value to the board state, based on metrics like material advantage, king safety, and control of the center. The function acts as a compass, pointing the engine in the direction most favorable overall.

To interface with the chess board, we import the `python-chess` library. The `chess.Board` object gives access to the data we need; piece positions, legal move counts, and checkmate detection. It allows us to iterate through all pieces and possible moves, and is able to import/export a game using standard chess notation.

In general, existing chess engines follow these metrics:

- Material, assigning weighted values to different piece types;
- Mobility, a higher number of available legal moves correlates with a better position;
- King Safety, eliminates blunders and allows for high value of castling;
- Positioning, encourages pieces to move to places where they are most useful, using Piece-Square tables. This maximizes piece potential.

## Approach

We decided during our first team meeting that we will tackle this with a divide-and-conquer strategy. We would each implement our own version of the evaluation function, with different strategies. Upon completion, we pitted each one against eachother. The winner is then selected as the final submission to the assignment.

### Team Shay

My initial idea was to build a convolutional neural network (CNN) to play chess. After building my prototype, I determined that this would not be viable given our time constraints and available compute power. Further research also indicated that CNNs are not ideal for chess.

My second approach was more traditional, to build a simple classical heuristic-based engine. This is where `team_shay.py` came from. After a few days of tweaking, I determined I had hit a hard limit at depth 3 and could no longer make significant gains in performance. I then further researched how chess engines evolved over time, and became interested in a dual approach of both classical heuristics working in conjunction with a small neural network.

This is how engines like Stockfish eventually overcame pure neural-network-style approaches in practical play. Stockfish specifically implements an Efficiently Updatable Neural Network (NNUE), a lightweight neural network that tracks board state and assists evaluation by spotting patterns that classical heuristics may miss.

**Search and quiescence**  

Search relies on Minimax (White maximizes, Black minimizes) with alpha-beta pruning and strong move ordering. When the main search hits the depth limit of 3 (excluding check extensions), the engine does not stop immediately. It moves into quiescence search and explores forcing lines, mainly captures and queen promotions, until a ply cap or cutoff condition stabilises the position. This is required to reduce horizon effect errors. If the side to move is in check, the search expands legal moves rather than only captures. Game-ending states are handled inside search, and static evaluations are cached using Zobrist hashing so repeated positions are not rescored from scratch.

I have also added important steps to the search, such as the transposition table, killer moves, null-move pruning, and ordering heuristics. They are really important for strength at shallow depths like 3.

Here is an outline of the most important eval metrics used:

**Material + PeSTO-style piece-square tables (tapered)**  

Every piece adds a fixed centipawn value (pawn 100, knight 320, bishop 330, rook 500, queen 900, king 20000) in combination with two table bonuses: one for middlegame (`PST_MG`) and one for endgame (`PST_EG`), indexed by square. This encodes "good" squares for each phase of the game and guides piece development into useful board areas.

**Phase-based tapering**  

The evaluator tracks game phase from non-pawn material (knight/bishop +1, rook +2, queen +4 each, capped at 24). These values help transition from middlegame to endgame scores. A king in the center during middlegame is usually undesirable, while in endgame it can be normal. Tapering prevents phase-inappropriate penalties or bonuses from causing problems.

**Tempo**  

Tempo is a bonus for the side to move, interpolated between middlegame (28 cp) and endgame (12 cp). Chess is not strictly symmetric; the side to move can act first, improve strategy, or create new threats. Without tempo, many practically favorable positions become false ties. This bonus is stronger in middlegame and reduced in endgame where precision and zugzwang-like effects matter more.

**Mobility**  

Mobility estimates piece activity by counting attacked squares. I apply it to knights, bishops, rooks, and queens. Attacked-square counts are weighted per piece (bishop 5, knight 4, rook 3, queen 2), summed White vs Black, and scaled by a mobility factor (0.06). More active pieces generally mean more tactical and positional options.

**Bishop pairs**  

If a side keeps both bishops, the score shifts by $\plusmn$32 centipawns. Two bishops scale well in open positions. Without this term, many positions treat bishop+knight and bishop+bishop too similarly, which can become dangerous in endgames, forcing stalemates.

**Pawn structure**  

Pawns have more rules:

- Doubled pawns on a file: penalty per extra pawn.
- Isolated pawns (no friendly pawns on adjacent files): penalty.
- Passed pawns: if enemy pawns cannot block/capture on forward adjacent files, apply a rank-based bonus scaled toward endgame.

Pawn structure is treated as long-term. Weak pawns can become targets, while passed pawns often decide endgames. Endgame scaling matters because depth-3 search often cannot see promotion directly.

**Rook placement**  

Rooks receive bonuses to their positions in the following conditions:

- Open or semi-open file;
- White rook on rank 6 / Black rook on rank 1 (seventh-rank pressure);
- Two friendly rooks connected on same rank/file with no blockers, creating a rook battery.

Rooks improve a lot on open files and seventh-rank access. Rook batteries increase pressure by creating strong trades & protection for other pieces.

**King safety**  

The king should remain on g1/c1 (or mirrored for Black), this gives a bonus in the middlegame. If this king is on e1/d1/f1 (also mirrored for Black), it receives a penalty, since it did not castle.

This rewards safer king setups in middlegame and discourage exposed central kings once castling is not available.

**Tactical pressure (hanging / loose pieces)**  

I built a dedicated tactical term that scans non-king pieces using attacker/defender counts. Undefended attacked pieces receive larger penalties; pieces with fewer defenders than attackers receive partial penalties. An additional loose-piece penalty applies when a lower-value attacker can pressure a higher-value piece.

**Team BlunderBus (classical + NNUE hybrid)**  

After finishing `team_shay`, I wanted to test whether a small NNUE could improve practical strength without removing the classical backbone. BlunderBus uses the same search philosophy but changes the evaluator into a hybrid.

BlunderBus computes:

1. a classical score,
2. an NNUE score,
3. a bounded correction based on the difference between them.

The difference (`nnue - classical`) is clamped by a phase-dependent cap and then scaled by a blend factor. Final score is classical plus this bounded correction. If NNUE weights are missing or unusable, BlunderBus falls back to classical-only evaluation.

The design goal is to keep classical stability while allowing the network to add positional pattern recognition where hand-crafted terms are weaker.

**Training the NNUE**  

Training is offline and exports a small weight file loaded by `team_BlunderBus.py` during runtime. In practice, my script samples positions from mixed rollouts (weighted-random plus shallow teacher-guided moves), encodes each board as sparse binary piece-square features plus castling and side-to-move bits, labels positions using either a teacher module or Stockfish centipawn scores, optionally mirror-augments data with sign-flipped labels, and then trains a compact two-layer network with minibatch SGD, weight decay, and optional symmetry loss.

A representative heavier run, chosen to approximate around 40 hours on a Ryzen 5 3600 (mostly due to Stockfish labeling cost), was:

```sh
python BlunderBus/train_nnue.py   --teacher-mode stockfish   --samples 95000   --stockfish-depth 13   --stockfish-depth-min 10   --min-plies 8   --max-plies 86   --teacher-prob 0.72   --epochs 20   --hidden 96   --batch-size 256   --lr 0.01   --symmetry-loss-weight 0.12   --seed 11
```

At these settings, Stockfish labeling dominates wall time; the SGD phase is comparatively short.

**Team Shay vs BlunderBus**  

In internal depth-3 head-to-head testing, `team_shay` was the stronger practical baseline under our constraints.

**Team Shay strengths:**  

- highly interpretable and easy to tune,
- stable tactical behaviour at shallow depth,
- strong hand-crafted phase-aware structure.

**Team Shay weaknesses:**  

- diminishing returns once depth-3 tuning saturates,
- some subtle positional patterns remain hard to encode manually.

**BlunderBus strengths:**  

- potential to capture positional patterns classical heuristics miss,
- classical fallback keeps behaviour robust,
- bounded blend reduces catastrophic NNUE errors.

**BlunderBus weaknesses:**  

- performance depends heavily on data quality and labeling depth,
- limited compute/time can make NNUE corrections inconsistent,
- extra complexity does not guarantee immediate Elo gain.

Despite `team_shay` winning all 8 test games in my scripted comparison, I don't think that hybrid designs are invalid. Under my 'training budget' and the amount of compute available for data labelling, the classical evaluator simply remained stronger.

### Team Goraieb/aaaaa

A lot of what Team Shay implements was also implemented in code code for Team Goraieb, so here the differences will be discussed, and some of the main differences where:

- parameter optimizer
  - Inside tune_engine.py and optimizer.py we have a automated weight tuning via genetic algorithm with multiple fitness modes (move_quality, self_play, vs_stockfish).
- Policy network NN
  - A seperate NN to help with move ordering, ResBlock-based PolicyEvaluator trained with pairwise       ranking loss on 500K positions. Code that generates it is found in train_policy.py
- Self-play training pipeline NN
  - Iterative self-play with probe-loss convergence tracking, similar to Shays implementation of using Stockfish labels but running self-play and then labeling all the positions on the games that the NN just played training and reapeating.
- Cython acceleration
  - compiled board-to-vector conversion, just accelerating the code to be able to run more traning.

The pipeline for improving was simple;

- 1: Baseline training, set up the NN model and the policy model.
- 2: run the tune_engine code, and create the most optimal wheights.
- 3: test the engine against chess.com engines or some other engine.
- 4: run the selfplay code, change the code trying to imrpove it, etc etc
- 5: repeat step 2, 3 and 4 until satisfied.

My code uses delta based Stockfish labeling, checking how much the moves worsen the position compared to the previous position; if the position does not change the eval, it is a perfect move. Simply evaluating the positions created and making an average makes the engine want to play as passive as possible so the eval stays high for a long time at the start of the game and then it loses as fast as possible so the avg looks good.

Tuning the engine weights takes a long time to run:

```py
python tune_engine.py \
  --mode move_quality \
  --depth 2 --games 15 \
  --sf-play-depth 4 --sf-eval-depth 10 \
  --pop 35 --gen 30  --sf-skill 20 \
  --workers 12
═══ move_quality mode (DELTA-BASED) ═══
  Candidate depth : 2
  SF play depth   : 4  (skill 20)
  SF eval depth   : 10
  Games/eval      : 15
  Metric          : avg centipawn loss per move
    fitness ~ 0   : perfect play
    fitness ~ -30 : slight inaccuracies
    fitness ~ -100: frequent blunders

Threads: 12
Benchmarking 12 parallel evaluations...
1933.48s  (161.12s/eval)

  ~1050 evals in ~88 batches
  Estimated wall time: ~47h 15m

Strategy: genetic  |  Mode: move_quality  |  Depth: 2
Parameters: 21  |  Workers: 12

[genetic] pop=35 gen=30 patience=20 (12 workers)
```

As seen above 47 hours for a depth 2 simple training, very slow, two solutions where the depth 2 shown above and the patience feature that makes the code do an early stop if there's no improvement for a few generations.

Depth 2 did cause an issue, it makes the engine become blind to the value of pieces, `tune_engine` would constantly set all the pieces to the lowest values I allowed and bump all the tactical options up, this happened mainly when running it in self-play mode, the engine tries to confuse the other engine by doing more tatical moves, but better engines will punish any innacuracies, the fix was to do a run at depth 4 for the piece values and then lock the values at those.

### Team Creepers

As I have played chess before my approach was just to represent in heuristics basic principles of Chess that I follow in my games.

- Bishop And rook controls
  - Positions where Bishops and Rooks "see" more squares would be evaluated higher for each new square in "control"
  - The blocked Bishops and Rooks e. g. those that cant do anything because are being blocked by enemy or ally piece are evaluated as advantage for enemy.
- Knight And King heat maps of the board
  - Knights do significantly more when they are in the center of the board.
  - King safety is one of the core principles in chess so by a heat map I encourage the King to move into the edges.
  - Heat maps are abolished in the endgame(e.g. when there are 4 or less pieces on both sides) as those principles no longer are hold.
- Pawn structures
  - A chain of pawns protect each other earning the "brithing space" for the player which makes them good.
  - Isolated pawns are obvious weakneses in the positions.
  - its good to have a diffrence in the color of the pawn blockad and the bishop so that bishop can move freely.
- Attackers and Defenders count
  - Basic principle to count how many pieces and pawns attack or defen a square so that engine wouldnt blunder so easily.

The evaluation for each of those parameters was adjusted by hand when it was playing against team Goraieb.

## Internal Tournament

We pitted the Creepers, Shay, BlunderBus and Goraieb implementations against eachother, by playing on Chess.com. We used the built-in analysis to rate our bots.

The main observations when watching the games were that many times inaccuracies would go unpunished, very different from what happens when you play against a semi-perfect engine like Stockfish. The engines had different flaws; a small amount of change in the values for Team Goraieb would give it a higher accuracy in theory, but it would start losing to Team Shay.

The interesting information we gather from what was mentioned above is that having a stronger engine does not mean it is stronger against all other engines. An example is raising the value for center control usually improves the opening, but the engine suffers in the endgame against an engine that has a high value for king tropism. It makes analysis of "what engine is the best" very complicated; it's all about the weakness of the engine you are playing against.

Team Creepers also caused us to see how important it was to stick with the main ideas of chess; a well-written and well-designed engine with fewer than 350 lines of code was better than anything we had done up to this point. Using the basics of chess and writing them as clearly as possible, we were achieving better results than adding as much information as possible and hoping it would fix the issues. Having more was important, but only after implementing the basics perfectly.

Additionally, we used the expert bots on Chess.com. These bots have tweakable difficulty, allowing us to figure out the exact ELO rating. Goraieb scores a consistent accuracy of 80-82%, translating to an ELO rating of 1750-1850.

## Final Engine

The information we got from our internal tournament put Goraieb on top, and so we have decided to settle on it as our submission.

Goraieb uses a sophisticated search with AB, TT, killers, history, and qsearch. The eval is also strongest, using tapered PeSTO with integrated pawn structure and mobility tables. It is also optimizer-ready - the weights are tunable, and we have used a substatial amount of compute time to tweak them. This meant that the strongest of our internal engines would get stronger over time.

## Complexity analysis

We analyzed the final engine in two parts: search complexity first, then evaluation complexity.

### Search

Search is by far the dominant cost. As is well known, the base case for minimax without pruning is $O(b^{d})$ in both time and space, where $b$ is the branching factor (average legal moves) and $d$ is the search depth. As we are operating at a fixed depth of 3, the growth is polynomial relative to the branching factor but exponential relative to depth.

Alpha-beta pruning in the best case (with perfect move ordering) reduces time to $O(b^{d/2})$ - this effectively doubles the depth we can search in the same time. In the worst case scenario with terrible ordering, the advantage degrades back to $O(b^{d})$. In practice, with average ordering, we can expect to get somewhere inbetween, which we have seen to be often cited as $O(b^{3d/4})$. The space complexity remains at $O(d)$ for the entire call stack since we only need to store the current path.

### Evaluation

The `evaluate()` function is called at every leaf node of the search tree. Each call is $O(n)$, where $n$ is the number of owned pieces on the board. The worst case $n = 32$ is effectively equal to $O(1)$ for asymptotic analysis.

### Real world performance

Given our constraints, we can estimate the workload for typical mid-game positions:

- Branching factor $b$, ~35 moves;
- Depth $d$, fixed to 3;
- Nodes visited: at most ~42,000.

At this scale, the bot is expected to complete its move calculation well within realtime play. However, as shown in our depth analysis, the complexity jump to $d$ = 5 and beyond will require much more aggresive pruning and efficient ordering.

## Conclusion

While we do not yet have the tournament results on hand, we can confidently say that we are happy with the current state of our submission. We have reasonably covered practical approaches to the solution, and balanced our resources between research and improvement.

## AI Usage Disclosure

Generative AI (in the form of LLMs) was used in a limited capacity to assist with the layout of this report. Certain sections (such as the complexity analysis) integrated some LLM provided suggestions after manual research confirming their validity.

All text contained in this report is human-written. No AI output is provided without complete rewording.

## Appendix

Code snippets for teams.
Annotated & simplified `evaluate()` for the final engine.
Graph for time & space complexity.

### Team Shay snippets

`team_shay.py` (classical heuristic + alpha-beta search):

```py
def evaluate(board: chess.Board) -> float:
    # tapered mg/eg score 
    #+ pawn structure 
    #+ rook files 
    #+ king safety 
    #+ tactical pressure
    ...

def minimax(board, depth, alpha, beta, maximizing, ply=0, ...):
    # alpha-beta with TT
    #+ null-move
    #+ PVS/LMR
    #+ quiescence frontier
    ...
```

`BlunderBus/team_BlunderBus.py` (NNUE + classical blend):

```py
def evaluate(board: chess.Board) -> float:
    classical = _classical_eval(board)
    nnue = _nnue_eval(board)
    blend = _nnue_blend_weight(board)
    if nnue is None:
        return classical
    delta = max(-delta_cap, min(delta_cap, nnue - classical))
    return classical + blend * delta
```
