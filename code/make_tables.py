"""Generate LaTeX table fragments from result CSVs (mean+/-std, bold best)."""
import csv, os
import numpy as np

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")
OUT = os.path.join(HERE, "..", "manuscript")
ALGOS = ["EOGWO", "GWO", "PSO", "DE", "WOA"]


def fmt(v):
    return f"{v:.2e}".replace("e", "E")


def benchmark_table(D):
    rows = {}
    with open(os.path.join(RESULTS, f"summary_D{D}.csv")) as f:
        for r in csv.DictReader(f):
            rows[(r["function"], r["algo"])] = (float(r["mean"]), float(r["std"]), r["class"])
    funcs = sorted({k[0] for k in rows}, key=lambda s: int(s[1:]))
    lines = []
    for fn in funcs:
        means = {a: rows[(fn, a)][0] for a in ALGOS}
        best = min(means, key=means.get)
        cells = []
        for a in ALGOS:
            m, s, _ = rows[(fn, a)]
            cell = f"{fmt(m)} ({fmt(s)})"
            if a == best:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(f"{fn} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def eng_table():
    rows = {}
    with open(os.path.join(RESULTS, "engineering_best.csv")) as f:
        for r in csv.DictReader(f):
            rows[(r["problem"], r["algo"])] = r
    probs = []
    for r in rows:
        if r[0] not in probs:
            probs.append(r[0])
    lines = []
    for pidx, p in enumerate(probs):
        best_costs = {a: float(rows[(p, a)]["best_cost"]) for a in ALGOS}
        overall_best = min(best_costs, key=best_costs.get)
        lit = rows[(p, ALGOS[0])]["literature_best"]
        for ai, a in enumerate(ALGOS):
            rr = rows[(p, a)]
            bc = float(rr["best_cost"]); mn = float(rr["mean"]); sd = float(rr["std"])
            bcs = f"{bc:.5f}"
            if a == overall_best:
                bcs = r"\textbf{" + bcs + "}"
            short = {"Welded Beam": "Welded beam", "Pressure Vessel": "Pressure vessel",
                     "Tension/Compression Spring": "Spring"}.get(p, p)
            label = short if ai == 0 else ""
            lines.append(f"{label} & {a} & {bcs} & {mn:.5f} & {sd:.2e} & {rr['feas_rate']} & {float(lit):.5g} " + r"\\")
        if pidx < len(probs) - 1:
            lines.append(r"\midrule")
    return "\n".join(lines)


def wilcoxon_summary():
    out = {}
    for D in [10, 30]:
        with open(os.path.join(RESULTS, f"wilcoxon_D{D}.csv")) as f:
            r = list(csv.reader(f))
        # find WTL block
        for i, row in enumerate(r):
            if row[:1] == ["EOGWO vs"]:
                for rr in r[i+1:]:
                    if len(rr) >= 4 and rr[0] in ALGOS:
                        out[(D, rr[0])] = (rr[1], rr[2], rr[3])
    lines = []
    for a in ["GWO", "PSO", "DE", "WOA"]:
        w10 = out[(10, a)]; w30 = out[(30, a)]
        lines.append(f"EOGWO vs {a} & {w10[0]}/{w10[1]}/{w10[2]} & {w30[0]}/{w30[1]}/{w30[2]} " + r"\\")
    return "\n".join(lines)


def friedman_table():
    lines = []
    with open(os.path.join(RESULTS, "friedman_ranks.csv")) as f:
        for r in csv.DictReader(f):
            d = r["dim"]
            cells = " & ".join(f"{float(r[a]):.2f}" for a in ALGOS)
            lines.append(f"{d} & {r['n_func']} & {cells} & {float(r['friedman_chi2']):.2f} & {float(r['p_value']):.2e}" + r" \\")
    return "\n".join(lines)


def holm_table():
    lines = []
    with open(os.path.join(RESULTS, "holm_posthoc.csv")) as f:
        for r in csv.DictReader(f):
            lines.append(f"EOGWO vs {r['EOGWO_vs']} & {float(r['z']):.3f} & {float(r['p_unadjusted']):.3e} & {float(r['p_holm']):.3e} & {r['decision']}" + r" \\")
    return "\n".join(lines)


if __name__ == "__main__":
    def w(name, content):
        content = content.rstrip()
        if content.endswith(r"\\"):
            content = content[:-2].rstrip()  # drop trailing \\ to avoid empty row before rule
        with open(os.path.join(OUT, name), "w") as f:
            f.write(content + "\n")
    w("tab_bench_D10.tex", benchmark_table(10))
    w("tab_bench_D30.tex", benchmark_table(30))
    w("tab_eng.tex", eng_table())
    w("tab_wilcoxon.tex", wilcoxon_summary())
    w("tab_friedman.tex", friedman_table())
    w("tab_holm.tex", holm_table())
    print("tables written")
