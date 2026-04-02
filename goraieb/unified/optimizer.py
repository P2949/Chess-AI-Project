"""
Generic Parameter Optimizer / Benchmarker (improved)

Improvements over original:
  • Persistent ThreadPoolExecutor (reused across generations, no per-gen spawn)
  • Early stopping on stagnation (configurable patience)
  • Exception handling in batch evaluation (crashed evals get -inf, not crash)
  • Adaptive mutation rate (increases on stagnation, decreases on improvement)
  • Duplicate candidate detection (skip re-evaluation of identical param sets)
  • Hill climbing perturbs multiple parameters per step
  • Better genetic diversity maintenance

All strategies evaluate candidates in parallel using ThreadPoolExecutor.
On Python 3.14t (free-threaded) this gives true parallelism with no GIL.
"""

from __future__ import annotations

import copy
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable


_DEFAULT_WORKERS = os.cpu_count() or 1


def _candidate_key(values: dict) -> tuple:
    """Hashable key for duplicate detection. Rounds floats to avoid FP noise."""
    return tuple(round(v, 8) if isinstance(v, float) else v
                 for v in sorted(values.items()))


def _batch_evaluate(
    candidates: list,
    fitness_fn: Callable[[dict[str, float]], float],
    n_workers: int,
    pool: ThreadPoolExecutor | None = None,
    seen: dict | None = None,
) -> None:
    """
    Evaluate candidates with fitness=None in parallel.

    Improvements:
      • Accepts optional persistent pool (avoids per-call pool creation)
      • Duplicate detection via `seen` dict (skip identical param sets)
      • Exception handling (crashed evals get -inf instead of crashing run)
    """
    to_eval = []
    for c in candidates:
        if c.fitness is not None:
            continue
        # Duplicate detection
        if seen is not None:
            key = _candidate_key(c.values)
            if key in seen:
                c.fitness = seen[key]
                continue
        to_eval.append(c)

    if not to_eval:
        return

    if n_workers <= 1 or pool is None:
        for c in to_eval:
            try:
                c.fitness = fitness_fn(c.values)
            except Exception as e:
                print(f"  [warn] fitness eval failed: {e}", file=sys.stderr)
                c.fitness = float("-inf")
            if seen is not None:
                seen[_candidate_key(c.values)] = c.fitness
        return

    future_to_candidate = {
        pool.submit(fitness_fn, c.values): c for c in to_eval
    }
    for future in as_completed(future_to_candidate):
        c = future_to_candidate[future]
        try:
            c.fitness = future.result()
        except Exception as e:
            print(f"  [warn] fitness eval failed: {e}", file=sys.stderr)
            c.fitness = float("-inf")
        if seen is not None:
            seen[_candidate_key(c.values)] = c.fitness


@dataclass
class Parameter:
    """A single tunable parameter with bounds and step size."""

    name: str
    min_val: float
    max_val: float
    step: float = 1.0
    dtype: type = float

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
    history: list[dict[str, Any]]
    elapsed: float = 0.0
    strategy_name: str = ""

    def show(self, top_n: int = 10):
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

        if self.history:
            print("  Convergence:")
            for entry in self.history:
                label = entry.get("label", "?")
                best_f = entry.get("best_fitness", "?")
                avg_f = entry.get("avg_fitness", None)
                avg_str = f"  avg={avg_f:.4f}" if avg_f is not None else ""
                extra = entry.get("extra", "")
                print(f"    {label:.<40s} best={best_f:.4f}{avg_str}{extra}")
            print()

        ranked = sorted(self.all_evaluated, key=lambda c: c.fitness or float("-inf"), reverse=True)
        print(f"  Top {min(top_n, len(ranked))} candidates:")
        for i, c in enumerate(ranked[:top_n]):
            vals = ", ".join(f"{k}={v}" for k, v in c.values.items())
            print(f"    #{i + 1:>3d}  fitness={c.fitness:<12.4f}  {vals}")
        print("=" * 70 + "\n")

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "elapsed": self.elapsed,
            "total_evaluations": len(self.all_evaluated),
            "best_fitness": self.best.fitness,
            "best_params": dict(self.best.values),
            "history": self.history,
        }


class Strategy(ABC):
    name: str = "base"

    @abstractmethod
    def run(
        self,
        params: list[Parameter],
        fitness_fn: Callable[[dict[str, float]], float],
        *,
        verbose: bool = False,
        n_workers: int = _DEFAULT_WORKERS,
    ) -> Report: ...


class GridSearch(Strategy):
    name = "grid"

    def run(self, params, fitness_fn, *, verbose=False, n_workers=_DEFAULT_WORKERS) -> Report:
        import itertools

        grids = [p.all_values() for p in params]
        names = [p.name for p in params]
        combos = list(itertools.product(*grids))

        total = len(combos)
        if verbose:
            print(f"[grid] {total} combinations to evaluate ({n_workers} workers)")

        candidates = [Candidate(values=dict(zip(names, combo))) for combo in combos]

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            _batch_evaluate(candidates, fitness_fn, n_workers, pool=pool)

        best = max(candidates, key=lambda c: c.fitness or float("-inf"))

        if verbose:
            print(f"  [{total}/{total}] best={best.fitness:.4f}")

        history = [{"label": "full_sweep", "best_fitness": best.fitness}]
        return Report(best=best.copy(), all_evaluated=candidates,
                      history=history, strategy_name="Grid Search")


class LinearSweep(Strategy):
    name = "sweep"

    def run(self, params, fitness_fn, *, verbose=False, n_workers=_DEFAULT_WORKERS) -> Report:
        evaluated = []
        history = []
        global_best = None

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for param in params:
                midpoints = {}
                for other in params:
                    if other.name != param.name:
                        midpoints[other.name] = other.clamp(
                            (other.min_val + other.max_val) / 2)

                if verbose:
                    print(f"[sweep] {param.name}: {param.min_val} → {param.max_val}")

                sweep = [Candidate(values={**midpoints, param.name: val})
                         for val in param.all_values()]
                _batch_evaluate(sweep, fitness_fn, n_workers, pool=pool)
                evaluated.extend(sweep)

                best_p = max(sweep, key=lambda c: c.fitness or float("-inf"))
                history.append({
                    "label": f"sweep_{param.name}",
                    "best_fitness": best_p.fitness,
                    "best_value": best_p.values[param.name],
                })

                if global_best is None or best_p.fitness > global_best.fitness:
                    global_best = best_p.copy()

                if verbose:
                    print(f"  best {param.name}={best_p.values[param.name]} "
                          f"(fitness={best_p.fitness:.4f})")

        return Report(best=global_best, all_evaluated=evaluated,
                      history=history, strategy_name="Linear Sweep")


class RandomSearch(Strategy):
    name = "random"

    def __init__(self, n_samples: int = 200):
        self.n_samples = n_samples

    def run(self, params, fitness_fn, *, verbose=False, n_workers=_DEFAULT_WORKERS) -> Report:
        if verbose:
            print(f"[random] {self.n_samples} samples ({n_workers} workers)")

        candidates = [Candidate(values={p.name: p.random_value() for p in params})
                      for _ in range(self.n_samples)]

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            seen = {}
            _batch_evaluate(candidates, fitness_fn, n_workers, pool=pool, seen=seen)

        best = None
        history = []
        for i, c in enumerate(candidates):
            if best is None or (c.fitness or float("-inf")) > (best.fitness or float("-inf")):
                best = c.copy()
            if verbose and (i + 1) % max(1, self.n_samples // 10) == 0:
                history.append({"label": f"sample_{i+1}", "best_fitness": best.fitness})
                print(f"  [{i+1}/{self.n_samples}] best={best.fitness:.4f}")

        return Report(best=best, all_evaluated=candidates,
                      history=history, strategy_name="Random Search")


class HillClimb(Strategy):
    """
    Improved hill climbing:
      • Perturbs MULTIPLE parameters per step (not just one)
      • Simulated annealing: occasionally accepts worse moves early on
      • Independent restarts run in parallel
    """
    name = "hillclimb"

    def __init__(self, iterations: int = 500, restarts: int = 3,
                 perturb_fraction: float = 0.3):
        self.iterations = iterations
        self.restarts = restarts
        self.perturb_fraction = perturb_fraction  # fraction of params to perturb per step

    def _single_restart(self, params, fitness_fn):
        evaluated = []
        current_vals = {p.name: p.random_value() for p in params}

        try:
            current_fitness = fitness_fn(current_vals)
        except Exception:
            current_fitness = float("-inf")

        current = Candidate(values=current_vals, fitness=current_fitness)
        evaluated.append(current.copy())
        best = current.copy()

        n_perturb = max(1, int(len(params) * self.perturb_fraction))

        for step in range(self.iterations):
            new_vals = dict(current.values)

            # Perturb multiple random parameters
            to_perturb = random.sample(params, min(n_perturb, len(params)))
            for p in to_perturb:
                magnitude = random.randint(1, 3)
                direction = random.choice([-1, 1])
                new_vals[p.name] = p.clamp(new_vals[p.name] + direction * magnitude * p.step)

            try:
                new_fitness = fitness_fn(new_vals)
            except Exception:
                new_fitness = float("-inf")

            neighbor = Candidate(values=new_vals, fitness=new_fitness)
            evaluated.append(neighbor.copy())

            # Simulated annealing: accept worse moves early with decreasing probability
            temperature = 1.0 - (step / self.iterations)  # 1.0 → 0.0
            if new_fitness > current.fitness:
                current = neighbor
            elif temperature > 0 and random.random() < temperature * 0.1:
                current = neighbor  # occasional bad move to escape local optima

            if new_fitness > best.fitness:
                best = neighbor.copy()

        return best, evaluated

    def run(self, params, fitness_fn, *, verbose=False, n_workers=_DEFAULT_WORKERS) -> Report:
        if verbose:
            print(f"[hillclimb] {self.restarts} restarts × {self.iterations} iters "
                  f"(perturb {self.perturb_fraction:.0%} params/step, {n_workers} workers)")

        all_evaluated = []
        history = []
        global_best = None

        with ThreadPoolExecutor(max_workers=min(n_workers, self.restarts)) as pool:
            futures = [pool.submit(self._single_restart, params, fitness_fn)
                       for _ in range(self.restarts)]
            for i, future in enumerate(as_completed(futures)):
                try:
                    best, evaluated = future.result()
                except Exception as e:
                    print(f"  [warn] restart failed: {e}", file=sys.stderr)
                    continue
                all_evaluated.extend(evaluated)
                if global_best is None or best.fitness > global_best.fitness:
                    global_best = best.copy()
                history.append({
                    "label": f"restart_{i+1}",
                    "best_fitness": global_best.fitness,
                })
                if verbose:
                    print(f"  restart {i+1}/{self.restarts} done, best={global_best.fitness:.4f}")

        return Report(best=global_best, all_evaluated=all_evaluated,
                      history=history, strategy_name="Hill Climbing")


class Genetic(Strategy):
    """
    Improved genetic algorithm:
      • Persistent thread pool (no per-generation spawn/destroy)
      • Early stopping on stagnation
      • Adaptive mutation rate (increases when stagnating)
      • Duplicate candidate detection (skip re-evaluation)
      • Diversity injection when population converges
    """
    name = "genetic"

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 60,
        elite_ratio: float = 0.1,
        crossover_rate: float = 0.7,
        mutation_rate: float = 0.15,
        tournament_size: int = 3,
        patience: int = 0,  # 0 = no early stopping; N = stop after N gens no improvement
    ):
        self.population_size = population_size
        self.generations = generations
        self.elite_ratio = elite_ratio
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.patience = patience if patience > 0 else max(5, generations // 3)

    def _tournament_select(self, population: list[Candidate]) -> Candidate:
        contestants = random.sample(population, min(self.tournament_size, len(population)))
        return max(contestants, key=lambda c: c.fitness or float("-inf"))

    def _crossover(self, a: Candidate, b: Candidate, params: list[Parameter]) -> Candidate:
        child_vals = {}
        for p in params:
            if random.random() < 0.5:
                child_vals[p.name] = a.values[p.name]
            else:
                child_vals[p.name] = b.values[p.name]

            if random.random() < 0.3:
                alpha = random.random()
                blended = alpha * a.values[p.name] + (1 - alpha) * b.values[p.name]
                child_vals[p.name] = p.clamp(blended)

        return Candidate(values=child_vals)

    def _mutate(self, candidate: Candidate, params: list[Parameter],
                mutation_rate: float) -> Candidate:
        """Mutate with adaptive rate."""
        vals = dict(candidate.values)
        for p in params:
            if random.random() < mutation_rate:
                magnitude = random.randint(1, 3)
                direction = random.choice([-1, 1])
                vals[p.name] = p.clamp(vals[p.name] + direction * magnitude * p.step)
        return Candidate(values=vals)

    def _inject_diversity(self, params: list[Parameter], count: int) -> list[Candidate]:
        """Create random candidates to maintain population diversity."""
        return [Candidate(values={p.name: p.random_value() for p in params})
                for _ in range(count)]

    def run(self, params, fitness_fn, *, verbose=False, n_workers=_DEFAULT_WORKERS) -> Report:
        population = [Candidate(values={p.name: p.random_value() for p in params})
                      for _ in range(self.population_size)]

        all_evaluated = []
        history = []
        global_best = None
        n_elite = max(1, int(self.population_size * self.elite_ratio))
        seen = {}  # duplicate detection cache

        # Adaptive mutation state
        current_mutation_rate = self.mutation_rate
        gens_without_improvement = 0

        if verbose:
            print(f"[genetic] pop={self.population_size} gen={self.generations} "
                  f"patience={self.patience} ({n_workers} workers)")

        # Persistent pool — created once, reused every generation
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            for gen in range(self.generations):
                unevaluated_before = [c for c in population if c.fitness is None]
                _batch_evaluate(population, fitness_fn, n_workers,
                                pool=pool, seen=seen)
                for c in unevaluated_before:
                    if c.fitness is not None:
                        all_evaluated.append(c.copy())

                population.sort(key=lambda c: c.fitness or float("-inf"), reverse=True)

                gen_best = population[0]
                gen_avg = statistics.mean(
                    c.fitness for c in population if c.fitness is not None)

                prev_best = global_best.fitness if global_best else float("-inf")
                if global_best is None or gen_best.fitness > global_best.fitness:
                    global_best = gen_best.copy()
                    gens_without_improvement = 0
                else:
                    gens_without_improvement += 1

                # Adaptive mutation: increase when stagnating
                if gens_without_improvement >= 3:
                    current_mutation_rate = min(0.5, self.mutation_rate * (1 + gens_without_improvement * 0.1))
                else:
                    current_mutation_rate = self.mutation_rate

                extra = ""
                if gens_without_improvement > 0:
                    extra = f"  stag={gens_without_improvement} mut={current_mutation_rate:.2f}"

                history.append({
                    "label": f"gen_{gen + 1:>3d}",
                    "best_fitness": global_best.fitness,
                    "avg_fitness": round(gen_avg, 6),
                    "gen_best": gen_best.fitness,
                    "extra": extra,
                })

                if verbose:
                    stag_str = f"  STAG={gens_without_improvement}" if gens_without_improvement > 0 else ""
                    print(
                        f"  gen {gen + 1:>3d}/{self.generations}"
                        f"  best={global_best.fitness:.4f}"
                        f"  gen_best={gen_best.fitness:.4f}"
                        f"  avg={gen_avg:.4f}"
                        f"  mut={current_mutation_rate:.2f}"
                        f"{stag_str}"
                    )

                # Early stopping
                if gens_without_improvement >= self.patience:
                    if verbose:
                        print(f"  Early stop: no improvement for {self.patience} generations")
                    break

                # Build next generation
                next_gen = []

                # Elitism
                for elite in population[:n_elite]:
                    next_gen.append(elite.copy())

                # Diversity injection when stagnating heavily
                if gens_without_improvement >= self.patience // 2:
                    n_inject = max(1, self.population_size // 5)
                    next_gen.extend(self._inject_diversity(params, n_inject))
                    if verbose:
                        print(f"    Injected {n_inject} random candidates for diversity")

                # Fill rest via selection + crossover + mutation
                while len(next_gen) < self.population_size:
                    if random.random() < self.crossover_rate:
                        parent_a = self._tournament_select(population)
                        parent_b = self._tournament_select(population)
                        child = self._crossover(parent_a, parent_b, params)
                    else:
                        child = self._tournament_select(population).copy()

                    child = self._mutate(child, params, current_mutation_rate)
                    child.fitness = None
                    next_gen.append(child)

                population = next_gen

        n_dupes = len(all_evaluated) - len(seen)
        if verbose and n_dupes > 0:
            print(f"  Duplicate evals skipped: ~{max(0, len(seen) - len(all_evaluated))} "
                  f"(cache size: {len(seen)})")

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
    """

    def __init__(
        self,
        params: list[Parameter],
        fitness_fn: Callable[[dict[str, float]], float],
        strategy: str | Strategy = "genetic",
        verbose: bool = True,
        n_workers: int = _DEFAULT_WORKERS,
        **strategy_kwargs,
    ):
        self.params = params
        self.fitness_fn = fitness_fn
        self.verbose = verbose
        self.n_workers = n_workers

        if isinstance(strategy, str):
            if strategy not in STRATEGIES:
                raise ValueError(
                    f"Unknown strategy '{strategy}'. Choose from: {list(STRATEGIES.keys())}")
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
        n_workers: int = _DEFAULT_WORKERS,
        **strategy_kwargs,
    ) -> Optimizer:
        regex = re.compile(fitness_pattern)

        def fitness_fn(values: dict[str, float]) -> float:
            cmd = command.format(**values)
            try:
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return float("-inf")

            output = result.stdout + "\n" + result.stderr
            match = regex.search(output)
            if not match:
                print(f"  [warn] no fitness match: {cmd}", file=sys.stderr)
                return float("-inf")

            val = float(match.group(1))
            return -val if invert else val

        return cls(params, fitness_fn, strategy=strategy, verbose=verbose,
                   n_workers=n_workers, **strategy_kwargs)

    def run(self) -> Report:
        t0 = time.perf_counter()
        report = self.strategy.run(
            self.params, self.fitness_fn,
            verbose=self.verbose, n_workers=self.n_workers,
        )
        report.elapsed = time.perf_counter() - t0
        return report


if __name__ == "__main__":
    params = [
        Parameter("x", -20, 20, step=0.5),
        Parameter("y", -20, 20, step=0.5),
    ]

    def demo_fitness(values: dict[str, float]) -> float:
        x, y = values["x"], values["y"]
        return -((x - 7) ** 2) - ((y + 3) ** 2)

    print(f"Using {_DEFAULT_WORKERS} worker threads\n")

    print("=== Genetic ===")
    opt = Optimizer(params, demo_fitness, strategy="genetic",
                    population_size=40, generations=30)
    report = opt.run()
    report.show(top_n=5)

    print("=== Hill Climbing ===")
    opt = Optimizer(params, demo_fitness, strategy="hillclimb",
                    iterations=300, restarts=5)
    report = opt.run()
    report.show(top_n=5)

    print("=== Random ===")
    opt = Optimizer(params, demo_fitness, strategy="random", n_samples=500)
    report = opt.run()
    report.show(top_n=5)