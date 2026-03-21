"""
Generic Parameter Optimizer / Benchmarker

A framework for finding optimal parameter values for any program or function.
Supports multiple search strategies: grid search, genetic/evolutionary,
random search, hill climbing, and linear sweep.

Usage (inline Python function):

    from optimizer import Parameter, Optimizer

    params = [
        Parameter("weight_pawn", 80, 120, step=5),
        Parameter("weight_knight", 280, 360, step=10),
        Parameter("mobility", 5, 20, step=1),
    ]

    def evaluate(values: dict[str, float]) -> float:
        # your fitness function — higher = better
        return run_engine_test(values)

    opt = Optimizer(params, evaluate, strategy="genetic")
    report = opt.run()
    report.show()

Usage (external command):

    from optimizer import Parameter, Optimizer

    params = [
        Parameter("threads", 1, 8, step=1),
        Parameter("hash_mb", 16, 256, step=16),
    ]

    opt = Optimizer.from_command(
        params,
        command="./engine --threads={threads} --hash={hash_mb}",
        fitness_pattern=r"score:\\s*([\\d.]+)",  # regex to extract fitness from stdout
        strategy="genetic",
    )
    report = opt.run()
    report.show()
"""

from __future__ import annotations

import copy
import math
import random
import re
import statistics
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable




@dataclass
class Parameter:
    """A single tunable parameter with bounds and step size."""

    name: str
    min_val: float
    max_val: float
    step: float = 1.0
    dtype: type = float  # set to int if the param must be integral

    def __post_init__(self):
        if self.min_val > self.max_val:
            raise ValueError(f"{self.name}: min_val ({self.min_val}) > max_val ({self.max_val})")
        if self.step <= 0:
            raise ValueError(f"{self.name}: step must be positive")

    @property
    def num_steps(self) -> int:
        return int((self.max_val - self.min_val) / self.step) + 1

    def clamp(self, value: float) -> float:
        """Clamp value to [min_val, max_val] and snap to step grid."""
        value = max(self.min_val, min(self.max_val, value))
        # snap to nearest step
        steps_from_min = round((value - self.min_val) / self.step)
        value = self.min_val + steps_from_min * self.step
        value = max(self.min_val, min(self.max_val, value))
        if self.dtype == int:
            return int(value)
        return value

    def random_value(self) -> float:
        """Pick a random value on the step grid."""
        steps = random.randint(0, self.num_steps - 1)
        val = self.min_val + steps * self.step
        if self.dtype == int:
            return int(val)
        return val

    def all_values(self) -> list[float]:
        """Return every value on the step grid."""
        vals = []
        v = self.min_val
        while v <= self.max_val + 1e-9:
            vals.append(self.dtype(v) if self.dtype == int else round(v, 10))
            v += self.step
        return vals




@dataclass
class Candidate:
    values: dict[str, float]
    fitness: float | None = None

    def copy(self) -> Candidate:
        return Candidate(values=dict(self.values), fitness=self.fitness)

    def __repr__(self):
        vals = ", ".join(f"{k}={v}" for k, v in self.values.items())
        return f"Candidate({vals} | fitness={self.fitness})"




@dataclass
class Report:
    best: Candidate
    all_evaluated: list[Candidate]
    history: list[dict[str, Any]]  # per-generation/iteration stats
    elapsed: float = 0.0
    strategy_name: str = ""

    def show(self, top_n: int = 10):
        """Print a human-readable summary."""
        print("\n" + "=" * 70)
        print(f"  OPTIMIZATION REPORT — {self.strategy_name}")
        print("=" * 70)
        print(f"  Total evaluations : {len(self.all_evaluated)}")
        print(f"  Elapsed time      : {self.elapsed:.2f}s")
        print(f"  Best fitness       : {self.best.fitness}")
        print()
        print("  Best parameters:")
        for name, val in self.best.values.items():
            print(f"    {name:.<30s} {val}")
        print()

        # convergence summary
        if self.history:
            print("  Convergence:")
            for entry in self.history:
                label = entry.get("label", "?")
                best_f = entry.get("best_fitness", "?")
                avg_f = entry.get("avg_fitness", None)
                avg_str = f"  avg={avg_f:.4f}" if avg_f is not None else ""
                print(f"    {label:.<40s} best={best_f:.4f}{avg_str}")
            print()

        # top N
        ranked = sorted(self.all_evaluated, key=lambda c: c.fitness or float("-inf"), reverse=True)
        print(f"  Top {min(top_n, len(ranked))} candidates:")
        for i, c in enumerate(ranked[:top_n]):
            vals = ", ".join(f"{k}={v}" for k, v in c.values.items())
            print(f"    #{i + 1:>3d}  fitness={c.fitness:<12.4f}  {vals}")
        print("=" * 70 + "\n")

    def to_dict(self) -> dict:
        """Serialisable summary."""
        return {
            "strategy": self.strategy_name,
            "elapsed": self.elapsed,
            "total_evaluations": len(self.all_evaluated),
            "best_fitness": self.best.fitness,
            "best_params": dict(self.best.values),
            "history": self.history,
        }




class Strategy(ABC):
    """Base class for all search strategies."""

    name: str = "base"

    @abstractmethod
    def run(
        self,
        params: list[Parameter],
        fitness_fn: Callable[[dict[str, float]], float],
        *,
        verbose: bool = False,
    ) -> Report: ...


class GridSearch(Strategy):
    """Exhaustive cartesian product of all parameter grids."""

    name = "grid"

    def run(self, params, fitness_fn, *, verbose=False) -> Report:
        import itertools

        grids = [p.all_values() for p in params]
        names = [p.name for p in params]
        combos = list(itertools.product(*grids))

        total = len(combos)
        if verbose:
            print(f"[grid] {total} combinations to evaluate")

        evaluated: list[Candidate] = []
        best: Candidate | None = None

        for i, combo in enumerate(combos):
            values = dict(zip(names, combo))
            fitness = fitness_fn(values)
            c = Candidate(values=values, fitness=fitness)
            evaluated.append(c)

            if best is None or fitness > best.fitness:
                best = c.copy()

            if verbose and (i + 1) % max(1, total // 20) == 0:
                print(f"  [{i + 1}/{total}] current best={best.fitness:.4f}")

        history = [{"label": "full_sweep", "best_fitness": best.fitness}]
        return Report(best=best, all_evaluated=evaluated, history=history, strategy_name="Grid Search")


class LinearSweep(Strategy):
    """Sweep a single parameter while holding others at their midpoint."""

    name = "sweep"

    def run(self, params, fitness_fn, *, verbose=False) -> Report:
        evaluated: list[Candidate] = []
        history: list[dict] = []
        global_best: Candidate | None = None

        for param in params:
            midpoints = {}
            for other in params:
                if other.name != param.name:
                    mid = other.clamp((other.min_val + other.max_val) / 2)
                    midpoints[other.name] = mid

            if verbose:
                print(f"[sweep] sweeping {param.name}: {param.min_val} → {param.max_val} (step={param.step})")

            best_for_param: Candidate | None = None
            for val in param.all_values():
                values = {**midpoints, param.name: val}
                fitness = fitness_fn(values)
                c = Candidate(values=values, fitness=fitness)
                evaluated.append(c)

                if best_for_param is None or fitness > best_for_param.fitness:
                    best_for_param = c.copy()

            history.append({
                "label": f"sweep_{param.name}",
                "best_fitness": best_for_param.fitness,
                "best_value": best_for_param.values[param.name],
            })

            if global_best is None or best_for_param.fitness > global_best.fitness:
                global_best = best_for_param.copy()

            if verbose:
                print(f"  best {param.name}={best_for_param.values[param.name]} (fitness={best_for_param.fitness:.4f})")

        return Report(best=global_best, all_evaluated=evaluated, history=history, strategy_name="Linear Sweep")


class RandomSearch(Strategy):
    """Uniformly random sampling from the parameter space."""

    name = "random"

    def __init__(self, n_samples: int = 200):
        self.n_samples = n_samples

    def run(self, params, fitness_fn, *, verbose=False) -> Report:
        evaluated: list[Candidate] = []
        best: Candidate | None = None
        history: list[dict] = []

        if verbose:
            print(f"[random] sampling {self.n_samples} random points")

        for i in range(self.n_samples):
            values = {p.name: p.random_value() for p in params}
            fitness = fitness_fn(values)
            c = Candidate(values=values, fitness=fitness)
            evaluated.append(c)

            if best is None or fitness > best.fitness:
                best = c.copy()

            if verbose and (i + 1) % max(1, self.n_samples // 10) == 0:
                history.append({
                    "label": f"sample_{i + 1}",
                    "best_fitness": best.fitness,
                })
                print(f"  [{i + 1}/{self.n_samples}] best so far={best.fitness:.4f}")

        return Report(best=best, all_evaluated=evaluated, history=history, strategy_name="Random Search")


class HillClimb(Strategy):
    """Stochastic hill climbing with optional restarts."""

    name = "hillclimb"

    def __init__(self, iterations: int = 500, restarts: int = 3):
        self.iterations = iterations
        self.restarts = restarts

    def run(self, params, fitness_fn, *, verbose=False) -> Report:
        evaluated: list[Candidate] = []
        global_best: Candidate | None = None
        history: list[dict] = []

        for restart in range(self.restarts):
            # random start
            current_vals = {p.name: p.random_value() for p in params}
            current_fitness = fitness_fn(current_vals)
            current = Candidate(values=current_vals, fitness=current_fitness)
            evaluated.append(current.copy())

            if global_best is None or current_fitness > global_best.fitness:
                global_best = current.copy()

            if verbose:
                print(f"[hillclimb] restart {restart + 1}/{self.restarts}, start fitness={current_fitness:.4f}")

            for it in range(self.iterations):
                # pick a random param and nudge it by ±step
                p = random.choice(params)
                direction = random.choice([-1, 1])
                new_val = p.clamp(current.values[p.name] + direction * p.step)

                new_vals = {**current.values, p.name: new_val}
                new_fitness = fitness_fn(new_vals)
                neighbor = Candidate(values=new_vals, fitness=new_fitness)
                evaluated.append(neighbor.copy())

                if new_fitness > current.fitness:
                    current = neighbor

                if new_fitness > global_best.fitness:
                    global_best = neighbor.copy()

            history.append({
                "label": f"restart_{restart + 1}",
                "best_fitness": global_best.fitness,
            })

        return Report(best=global_best, all_evaluated=evaluated, history=history, strategy_name="Hill Climbing")


class Genetic(Strategy):
    """Genetic / evolutionary search — the main workhorse for multi-parameter tuning."""

    name = "genetic"

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 60,
        elite_ratio: float = 0.1,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
    ):
        self.population_size = population_size
        self.generations = generations
        self.elite_ratio = elite_ratio
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def _tournament_select(self, population: list[Candidate]) -> Candidate:
        contestants = random.sample(population, min(self.tournament_size, len(population)))
        return max(contestants, key=lambda c: c.fitness or float("-inf"))

    def _crossover(self, a: Candidate, b: Candidate, params: list[Parameter]) -> Candidate:
        """Uniform crossover with arithmetic blending."""
        child_vals = {}
        for p in params:
            if random.random() < 0.5:
                child_vals[p.name] = a.values[p.name]
            else:
                child_vals[p.name] = b.values[p.name]

            # occasional arithmetic blend
            if random.random() < 0.3:
                alpha = random.random()
                blended = alpha * a.values[p.name] + (1 - alpha) * b.values[p.name]
                child_vals[p.name] = p.clamp(blended)

        return Candidate(values=child_vals)

    def _mutate(self, candidate: Candidate, params: list[Parameter]) -> Candidate:
        """Mutate random parameters by ±(1..3) steps."""
        vals = dict(candidate.values)
        for p in params:
            if random.random() < self.mutation_rate:
                magnitude = random.randint(1, 3)
                direction = random.choice([-1, 1])
                vals[p.name] = p.clamp(vals[p.name] + direction * magnitude * p.step)
        return Candidate(values=vals)

    def run(self, params, fitness_fn, *, verbose=False) -> Report:
        # initialise random population
        population: list[Candidate] = []
        for _ in range(self.population_size):
            values = {p.name: p.random_value() for p in params}
            population.append(Candidate(values=values))

        all_evaluated: list[Candidate] = []
        history: list[dict] = []
        global_best: Candidate | None = None
        n_elite = max(1, int(self.population_size * self.elite_ratio))

        for gen in range(self.generations):
            # evaluate
            for c in population:
                if c.fitness is None:
                    c.fitness = fitness_fn(c.values)
                    all_evaluated.append(c.copy())

            # sort by fitness descending
            population.sort(key=lambda c: c.fitness or float("-inf"), reverse=True)

            gen_best = population[0]
            gen_avg = statistics.mean(c.fitness for c in population if c.fitness is not None)

            if global_best is None or gen_best.fitness > global_best.fitness:
                global_best = gen_best.copy()

            history.append({
                "label": f"gen_{gen + 1:>3d}",
                "best_fitness": global_best.fitness,
                "avg_fitness": round(gen_avg, 6),
                "gen_best": gen_best.fitness,
            })

            if verbose:
                print(
                    f"  gen {gen + 1:>3d}/{self.generations}"
                    f"  best={global_best.fitness:.4f}"
                    f"  gen_best={gen_best.fitness:.4f}"
                    f"  avg={gen_avg:.4f}"
                )

            # === build next generation ===
            next_gen: list[Candidate] = []

            # elitism — carry forward top performers unchanged
            for elite in population[:n_elite]:
                next_gen.append(elite.copy())

            # fill the rest via selection + crossover + mutation
            while len(next_gen) < self.population_size:
                if random.random() < self.crossover_rate:
                    parent_a = self._tournament_select(population)
                    parent_b = self._tournament_select(population)
                    child = self._crossover(parent_a, parent_b, params)
                else:
                    child = self._tournament_select(population).copy()

                child = self._mutate(child, params)
                child.fitness = None  # needs re-evaluation
                next_gen.append(child)

            population = next_gen

        return Report(
            best=global_best,
            all_evaluated=all_evaluated,
            history=history,
            strategy_name=f"Genetic (pop={self.population_size}, gen={self.generations})",
        )



STRATEGIES: dict[str, type[Strategy]] = {
    "grid": GridSearch,
    "sweep": LinearSweep,
    "random": RandomSearch,
    "hillclimb": HillClimb,
    "genetic": Genetic,
}




class Optimizer:
    """
    Main interface. Ties together parameters, a fitness function, and a strategy.

    Parameters
    ----------
    params : list[Parameter]
        The tunable parameter definitions.
    fitness_fn : callable
        A function that takes dict[str, float] and returns a float (higher = better).
    strategy : str or Strategy instance
        One of "grid", "sweep", "random", "hillclimb", "genetic",
        or a pre-configured Strategy object.
    verbose : bool
        Print progress during optimisation.
    **strategy_kwargs
        Extra keyword arguments passed to the strategy constructor
        (e.g. population_size=100 for genetic).
    """

    def __init__(
        self,
        params: list[Parameter],
        fitness_fn: Callable[[dict[str, float]], float],
        strategy: str | Strategy = "genetic",
        verbose: bool = True,
        **strategy_kwargs,
    ):
        self.params = params
        self.fitness_fn = fitness_fn
        self.verbose = verbose

        if isinstance(strategy, str):
            if strategy not in STRATEGIES:
                raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES.keys())}")
            self.strategy = STRATEGIES[strategy](**strategy_kwargs)
        else:
            self.strategy = strategy

    @classmethod
    def from_command(
        cls,
        params: list[Parameter],
        command: str,
        fitness_pattern: str,
        strategy: str | Strategy = "genetic",
        timeout: float = 60.0,
        verbose: bool = True,
        invert: bool = False,
        **strategy_kwargs,
    ) -> Optimizer:
        """
        Create an optimizer that runs an external command.

        The command string should contain {param_name} placeholders.
        fitness_pattern is a regex with one capture group extracting
        the numeric fitness from stdout.

        Set invert=True if a lower value is better (e.g. error rate) —
        the framework always maximises, so this will negate the parsed value.
        """
        regex = re.compile(fitness_pattern)

        def fitness_fn(values: dict[str, float]) -> float:
            cmd = command.format(**values)
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return float("-inf")

            output = result.stdout + "\n" + result.stderr
            match = regex.search(output)
            if not match:
                print(f"  [warn] no fitness match in output for: {cmd}", file=sys.stderr)
                return float("-inf")

            val = float(match.group(1))
            return -val if invert else val

        return cls(params, fitness_fn, strategy=strategy, verbose=verbose, **strategy_kwargs)

    def run(self) -> Report:
        """Execute the optimisation and return a Report."""
        t0 = time.perf_counter()
        report = self.strategy.run(self.params, self.fitness_fn, verbose=self.verbose)
        report.elapsed = time.perf_counter() - t0
        return report


# ─── Demo / self-test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # demo: find x, y that maximise -(x-7)^2 - (y+3)^2  (optimum at x=7, y=-3)
    params = [
        Parameter("x", -20, 20, step=0.5),
        Parameter("y", -20, 20, step=0.5),
    ]

    def demo_fitness(values: dict[str, float]) -> float:
        x, y = values["x"], values["y"]
        return -((x - 7) ** 2) - ((y + 3) ** 2)

    print("=== Genetic ===")
    opt = Optimizer(params, demo_fitness, strategy="genetic", population_size=40, generations=30)
    report = opt.run()
    report.show(top_n=5)

    print("=== Hill Climbing ===")
    opt = Optimizer(params, demo_fitness, strategy="hillclimb", iterations=300, restarts=5)
    report = opt.run()
    report.show(top_n=5)

    print("=== Random ===")
    opt = Optimizer(params, demo_fitness, strategy="random", n_samples=500)
    report = opt.run()
    report.show(top_n=5)
