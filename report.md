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
