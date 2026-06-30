"""
Nonparametric statistical analysis of the benchmark results.

  - Wilcoxon rank-sum test: EOGWO vs each competitor on each function
    (per-run final errors, two-sided). Reports W/T/L counts at alpha=0.05.
  - Friedman test: average ranks of all 5 algorithms across all functions
    (per dimension and pooled), with the Friedman chi-square statistic and
    p-value.
  - Holm post-hoc: adjusted p-values for EOGWO (control) vs the others, based
    on the Friedman average-rank z-statistic.

Outputs (../results):
  - wilcoxon_D{D}.csv
  - friedman_ranks.csv
  - holm_posthoc.csv
"""
import numpy as np, csv, os
from scipy.stats import ranksums, friedmanchisquare, norm

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
ALGO_NAMES = ["EOGWO", "GWO", "PSO", "DE", "WOA"]
DIMS = [10, 30]
ALPHA = 0.05


def load_raw(D):
    data = {}
    with open(os.path.join(RESULTS, f"raw_D{D}.csv")) as f:
        for row in csv.DictReader(f):
            data.setdefault((row["function"], row["algo"]), []).append(float(row["error"]))
    funcs = sorted({k[0] for k in data}, key=lambda s: int(s[1:]))
    return data, funcs


def wilcoxon_tables():
    all_ranks = {}  # (D) -> matrix of per-function mean-based ranks
    for D in DIMS:
        data, funcs = load_raw(D)
        rows = []
        wtl = {a: [0, 0, 0] for a in ALGO_NAMES if a != "EOGWO"}  # W,T,L for EOGWO vs a
        for fn in funcs:
            base = np.array(data[(fn, "EOGWO")])
            row = [fn]
            for a in ALGO_NAMES:
                if a == "EOGWO":
                    continue
                other = np.array(data[(fn, a)])
                try:
                    stat, p = ranksums(base, other)
                except Exception:
                    p = 1.0
                sign = "="
                if p < ALPHA:
                    if base.mean() < other.mean():
                        sign = "+"; wtl[a][0] += 1   # EOGWO wins
                    else:
                        sign = "-"; wtl[a][2] += 1   # EOGWO loses
                else:
                    wtl[a][1] += 1
                row += [f"{p:.3e}", sign]
            rows.append(row)
        header = ["function"]
        for a in ALGO_NAMES:
            if a != "EOGWO":
                header += [f"p({a})", f"sig({a})"]
        with open(os.path.join(RESULTS, f"wilcoxon_D{D}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
            w.writerow([])
            w.writerow(["EOGWO vs", "Win(+)", "Tie(=)", "Loss(-)"])
            for a in ALGO_NAMES:
                if a != "EOGWO":
                    w.writerow([a] + wtl[a])
        print(f"--- Wilcoxon D={D} (EOGWO vs others, +/=/-) ---")
        for a, v in wtl.items():
            print(f"  vs {a}: W={v[0]} T={v[1]} L={v[2]}")
    return


def friedman_and_holm():
    rank_rows = []
    pooled_ranks_per_algo = {a: [] for a in ALGO_NAMES}
    holm_all = []
    for D in DIMS + ["pooled"]:
        if D == "pooled":
            # combine functions from both dims as separate problems
            all_mean = []
            for d in DIMS:
                data, funcs = load_raw(d)
                for fn in funcs:
                    all_mean.append([np.mean(data[(fn, a)]) for a in ALGO_NAMES])
            M = np.array(all_mean)
        else:
            data, funcs = load_raw(D)
            M = np.array([[np.mean(data[(fn, a)]) for a in ALGO_NAMES] for fn in funcs])

        # ranks per function (1=best=lowest error)
        ranks = np.empty_like(M)
        for i in range(M.shape[0]):
            ranks[i] = M[i].argsort().argsort() + 1
            # handle ties via average rank
            order = M[i].argsort()
            vals = M[i][order]
            rr = np.arange(1, len(vals) + 1, dtype=float)
            # average ties
            j = 0
            while j < len(vals):
                k = j
                while k + 1 < len(vals) and vals[k + 1] == vals[j]:
                    k += 1
                if k > j:
                    rr[j:k + 1] = rr[j:k + 1].mean()
                j = k + 1
            tmp = np.empty(len(vals)); tmp[order] = rr
            ranks[i] = tmp
        avg_ranks = ranks.mean(axis=0)
        N, kk = M.shape
        # Friedman statistic
        try:
            chi, p = friedmanchisquare(*[M[:, j] for j in range(kk)])
        except Exception:
            chi, p = float("nan"), float("nan")
        rank_rows.append([str(D), N] + [f"{r:.3f}" for r in avg_ranks] + [f"{chi:.3f}", f"{p:.3e}"])
        print(f"Friedman D={D}: chi2={chi:.3f} p={p:.3e}  ranks=" +
              ", ".join(f"{a}:{r:.2f}" for a, r in zip(ALGO_NAMES, avg_ranks)))

        # Holm post-hoc with EOGWO as control (only for pooled)
        if D == "pooled":
            ctrl = ALGO_NAMES.index("EOGWO")
            se = np.sqrt(kk * (kk + 1) / (6.0 * N))
            comps = []
            for j, a in enumerate(ALGO_NAMES):
                if j == ctrl:
                    continue
                z = (avg_ranks[ctrl] - avg_ranks[j]) / se
                pj = 2 * norm.cdf(-abs(z))
                comps.append([a, z, pj])
            comps.sort(key=lambda c: c[2])
            m = len(comps)
            for idx, (a, z, pj) in enumerate(comps):
                holm = min(1.0, (m - idx) * pj)
                holm_all.append([a, f"{z:.3f}", f"{pj:.3e}", f"{holm:.3e}",
                                 "reject H0" if holm < ALPHA else "retain H0"])

    with open(os.path.join(RESULTS, "friedman_ranks.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim", "n_func"] + ALGO_NAMES + ["friedman_chi2", "p_value"])
        w.writerows(rank_rows)
    with open(os.path.join(RESULTS, "holm_posthoc.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["EOGWO_vs", "z", "p_unadjusted", "p_holm", "decision"])
        w.writerows(holm_all)
    print("--- Holm post-hoc (control=EOGWO, pooled) ---")
    for r in holm_all:
        print("  ", r)


if __name__ == "__main__":
    wilcoxon_tables()
    friedman_and_holm()
