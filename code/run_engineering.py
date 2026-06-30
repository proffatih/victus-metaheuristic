"""
Run the 5 optimizers on the 3 engineering design problems.
30 independent runs each, fixed seeds, pop=30, max_iter=500.
Writes ../results/engineering.csv and ../results/engineering_best.csv.
"""
import numpy as np, csv, os
from optimizers import ALGORITHMS
from engineering import PROBLEMS

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
ALGO_NAMES = ["EOGWO", "GWO", "PSO", "DE", "WOA"]
N_RUNS, POP, MAX_ITER = 30, 30, 500


def main():
    rows = []      # per-run
    best_rows = [] # best feasible per (problem, algo)
    for pi, (meta, fobj, feas) in enumerate(PROBLEMS):
        D = meta["D"]; lb = meta["lb"]; ub = meta["ub"]
        for a in ALGO_NAMES:
            alg = ALGORITHMS[a]
            costs = []
            best_cost = np.inf; best_x = None
            for r in range(N_RUNS):
                rng = np.random.default_rng(7000 * pi + 13 * r + 1)
                bv, bx, _ = alg(fobj, lb, ub, D, POP, MAX_ITER, rng)
                feasible = feas(bx)
                costs.append(bv if feasible else np.nan)
                rows.append([meta["name"], a, r, f"{bv:.6f}", int(feasible)])
                if feasible and bv < best_cost:
                    best_cost = bv; best_x = bx
            costs = np.array(costs, dtype=float)
            valid = costs[~np.isnan(costs)]
            mean = np.nanmean(costs) if valid.size else np.nan
            std = np.nanstd(costs) if valid.size else np.nan
            fr = valid.size / N_RUNS
            best_rows.append([meta["name"], a, f"{best_cost:.6f}",
                              f"{mean:.6f}", f"{std:.6e}", f"{fr:.2f}",
                              meta["ref"],
                              " ".join(f"{v:.5f}" for v in best_x) if best_x is not None else ""])
            print(f"{meta['name']:28s} {a:6s} best={best_cost:.6f} mean={mean:.6f} feas_rate={fr:.2f}")

    with open(os.path.join(RESULTS, "engineering.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["problem", "algo", "run", "cost", "feasible"]); w.writerows(rows)
    with open(os.path.join(RESULTS, "engineering_best.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["problem", "algo", "best_cost", "mean", "std", "feas_rate", "literature_best", "best_x"])
        w.writerows(best_rows)


if __name__ == "__main__":
    main()
