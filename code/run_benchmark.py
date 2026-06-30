"""
Main benchmark driver.

For each dimension D in {10, 30}, each of the 10 benchmark functions, and each
of the 5 optimizers, perform 30 independent runs with fixed, reproducible
seeds. A common budget of pop=30 wolves/particles and max_iter=500 iterations
is used for all algorithms (DE uses the same population and iteration count;
its per-individual trial evaluations give it a comparable function-evaluation
budget). The best-so-far convergence curve is averaged over runs.

Outputs (written to ../results):
  - summary_D{D}.csv      : mean/std/best/median error per (func, algo)
  - convergence_D{D}.npz  : mean convergence curves per (func, algo)
  - raw_D{D}.csv          : every final error (func, algo, run, error)
"""
import numpy as np
import csv, os, time
from cec2017 import get_suite
from optimizers import ALGORITHMS

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

N_RUNS = 30
POP = 30
MAX_ITER = 500
DIMS = [10, 30]
ALGO_NAMES = ["EOGWO", "GWO", "PSO", "DE", "WOA"]


def main():
    for D in DIMS:
        suite = get_suite(D)
        n_func = len(suite)
        n_algo = len(ALGO_NAMES)
        # storage
        final = {(fn.name, a): np.empty(N_RUNS) for fn in suite for a in ALGO_NAMES}
        curves = {(fn.name, a): np.zeros(MAX_ITER) for fn in suite for a in ALGO_NAMES}
        t_start = time.time()
        for fi, fn in enumerate(suite):
            for a in ALGO_NAMES:
                alg = ALGORITHMS[a]
                for r in range(N_RUNS):
                    seed = 100000 * D + 1000 * fi + r  # fully reproducible
                    rng = np.random.default_rng(seed)
                    bv, _, curve = alg(fn, fn.lb, fn.ub, D, POP, MAX_ITER, rng)
                    final[(fn.name, a)][r] = bv - fn.bias  # error to known optimum
                    curves[(fn.name, a)] += curve - fn.bias
                curves[(fn.name, a)] /= N_RUNS
            print(f"D={D} {fn.name} done  ({time.time()-t_start:.0f}s)")

        # write summary
        with open(os.path.join(RESULTS, f"summary_D{D}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["function", "class", "algo", "mean", "std", "best", "median"])
            for fn in suite:
                for a in ALGO_NAMES:
                    e = final[(fn.name, a)]
                    w.writerow([fn.name, fn.cls, a, f"{e.mean():.6e}",
                                f"{e.std():.6e}", f"{e.min():.6e}", f"{np.median(e):.6e}"])
        # write raw
        with open(os.path.join(RESULTS, f"raw_D{D}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["function", "algo", "run", "error"])
            for fn in suite:
                for a in ALGO_NAMES:
                    for r in range(N_RUNS):
                        w.writerow([fn.name, a, r, f"{final[(fn.name,a)][r]:.6e}"])
        # write curves
        np.savez(os.path.join(DATA, f"convergence_D{D}.npz"),
                 **{f"{fn.name}__{a}": curves[(fn.name, a)]
                    for fn in suite for a in ALGO_NAMES})
        print(f"=== D={D} complete in {time.time()-t_start:.0f}s ===")


if __name__ == "__main__":
    main()
