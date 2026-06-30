"""
Generate all manuscript figures as both vector PDF and 300-dpi PNG.
Colorblind-safe palette (Wong/Okabe-Ito). Saves to ../figures.
"""
import numpy as np, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(__file__)
RESULTS = os.path.join(HERE, "..", "results")
DATA = os.path.join(HERE, "..", "data")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)

ALGOS = ["EOGWO", "GWO", "PSO", "DE", "WOA"]
# Okabe-Ito colorblind-safe palette
CMAP = {"EOGWO": "#000000", "GWO": "#E69F00", "PSO": "#56B4E9",
        "DE": "#009E73", "WOA": "#CC79A7"}
MARK = {"EOGWO": "o", "GWO": "s", "PSO": "^", "DE": "D", "WOA": "v"}
LS = {"EOGWO": "-", "GWO": "--", "PSO": "-.", "DE": ":", "WOA": (0, (3, 1, 1, 1))}

plt.rcParams.update({
    "font.size": 10, "font.family": "serif", "axes.grid": True,
    "grid.alpha": 0.3, "axes.linewidth": 0.8, "figure.dpi": 120,
    "savefig.bbox": "tight", "legend.framealpha": 0.9,
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)
    print("saved", name)


def load_raw(D):
    data = {}
    with open(os.path.join(RESULTS, f"raw_D{D}.csv")) as f:
        for row in csv.DictReader(f):
            data.setdefault((row["function"], row["algo"]), []).append(float(row["error"]))
    return data


# ---------------------------------------------------------------------------
# 1. Convergence curves (representative functions, D=30)
# ---------------------------------------------------------------------------
def fig_convergence():
    npz = np.load(os.path.join(DATA, "convergence_D30.npz"))
    funcs = ["F1", "F3", "F5", "F6", "F8", "F10"]
    titles = {"F1": "F1 Bent Cigar (uni.)", "F3": "F3 Elliptic (uni.)",
              "F5": "F5 Rastrigin (multi.)", "F6": "F6 Schaffer F7 (multi.)",
              "F8": "F8 Ackley (multi.)", "F10": "F10 Discus (uni.)"}
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    for ax, fn in zip(axes.ravel(), funcs):
        for a in ALGOS:
            y = npz[f"{fn}__{a}"]
            y = np.maximum(y, 1e-12)
            ax.semilogy(np.arange(1, len(y) + 1), y, color=CMAP[a],
                        ls=LS[a], lw=1.6, label=a)
        ax.set_title(titles[fn], fontsize=10)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Mean error (log)")
    handles = [Line2D([0], [0], color=CMAP[a], ls=LS[a], lw=2, label=a) for a in ALGOS]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, "fig_convergence")


# ---------------------------------------------------------------------------
# 2. Box plots (D=30, representative)
# ---------------------------------------------------------------------------
def fig_boxplots():
    data = load_raw(30)
    funcs = ["F1", "F5", "F6", "F8"]
    titles = {"F1": "F1 Bent Cigar", "F5": "F5 Rastrigin",
              "F6": "F6 Schaffer F7", "F8": "F8 Ackley"}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, fn in zip(axes, funcs):
        vals = [np.array(data[(fn, a)]) for a in ALGOS]
        # log-scale safe
        bp = ax.boxplot(vals, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black"), showfliers=True,
                        flierprops=dict(marker=".", markersize=3, alpha=0.4))
        for patch, a in zip(bp["boxes"], ALGOS):
            patch.set_facecolor(CMAP[a]); patch.set_alpha(0.65)
        ax.set_yscale("log")
        ax.set_xticks(range(1, len(ALGOS) + 1)); ax.set_xticklabels(ALGOS, rotation=45, fontsize=8)
        ax.set_title(titles[fn], fontsize=10); ax.set_ylabel("Final error (log)")
    fig.tight_layout()
    save(fig, "fig_boxplots")


# ---------------------------------------------------------------------------
# 3. Friedman average-rank bar chart (pooled)
# ---------------------------------------------------------------------------
def fig_friedman_ranks():
    rows = {}
    with open(os.path.join(RESULTS, "friedman_ranks.csv")) as f:
        for r in csv.DictReader(f):
            rows[r["dim"]] = {a: float(r[a]) for a in ALGOS}
    dims = ["10", "30", "pooled"]
    x = np.arange(len(ALGOS)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, d in enumerate(dims):
        vals = [rows[d][a] for a in ALGOS]
        ax.bar(x + (i - 1) * w, vals, w, label=f"D={d}" if d != "pooled" else "pooled",
               color=["#444", "#888", "#bbb"][i], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(ALGOS)
    ax.set_ylabel("Friedman average rank (lower = better)")
    ax.set_title("Friedman average ranks across benchmark functions")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_friedman_ranks")


# ---------------------------------------------------------------------------
# 4. Critical-difference (CD) diagram (pooled, Nemenyi)
# ---------------------------------------------------------------------------
def fig_cd_diagram():
    rows = {}
    nrow = {}
    with open(os.path.join(RESULTS, "friedman_ranks.csv")) as f:
        for r in csv.DictReader(f):
            rows[r["dim"]] = {a: float(r[a]) for a in ALGOS}
            nrow[r["dim"]] = int(r["n_func"])
    ranks = rows["pooled"]; N = nrow["pooled"]; k = len(ALGOS)
    # Nemenyi critical difference, q_alpha for k=5, alpha=0.05 = 2.728
    q_alpha = 2.728
    CD = q_alpha * np.sqrt(k * (k + 1) / (6.0 * N))
    order = sorted(ALGOS, key=lambda a: ranks[a])
    rmin, rmax = 1, k
    fig, ax = plt.subplots(figsize=(8, 2.8))
    ax.set_xlim(rmin - 0.5, rmax + 0.5); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.hlines(0.8, rmin, rmax, color="black", lw=1.2)
    for r in range(rmin, rmax + 1):
        ax.vlines(r, 0.78, 0.82, color="black", lw=1)
        ax.text(r, 0.86, str(r), ha="center", fontsize=9)
    # place algorithms
    for i, a in enumerate(order):
        rank = ranks[a]
        side = i < (k + 1) // 2
        ypos = 0.55 - 0.12 * (i if side else (k - 1 - i))
        xtext = rmin - 0.4 if side else rmax + 0.4
        ax.vlines(rank, 0.8, ypos, color=CMAP[a], lw=1.2)
        ax.hlines(ypos, rank, xtext, color=CMAP[a], lw=1.2)
        ax.text(xtext + (-0.05 if side else 0.05), ypos,
                f"{a} ({rank:.2f})", ha="right" if side else "left",
                va="center", fontsize=9, color=CMAP[a])
    # CD bar
    ax.hlines(0.95, rmin, rmin + CD, color="black", lw=2)
    ax.vlines([rmin, rmin + CD], 0.93, 0.97, color="black", lw=1)
    ax.text(rmin + CD / 2, 0.99, f"CD = {CD:.2f}", ha="center", fontsize=9)
    ax.set_title("Critical-difference diagram (Nemenyi, pooled, $\\alpha$=0.05)", fontsize=10)
    save(fig, "fig_cd_diagram")


# ---------------------------------------------------------------------------
# 5. Engineering-design convergence (welded beam, best run per algo)
# ---------------------------------------------------------------------------
def fig_engineering():
    # recompute one representative convergence curve per algorithm
    import sys
    sys.path.insert(0, HERE)
    from optimizers import ALGORITHMS
    from engineering import PROBLEMS
    meta, fobj, feas = PROBLEMS[0]  # welded beam
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for a in ALGOS:
        rng = np.random.default_rng(2024)
        _, _, curve = ALGORITHMS[a](fobj, meta["lb"], meta["ub"], meta["D"], 30, 500, rng)
        curve = np.maximum(curve, meta["ref"])  # clip below lit. optimum for log
        ax.plot(np.arange(1, len(curve) + 1), curve, color=CMAP[a],
                ls=LS[a], lw=1.7, label=a)
    ax.axhline(meta["ref"], color="gray", ls=":", lw=1, label="literature best")
    ax.set_ylim(1.7, 4.0)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Best feasible cost")
    ax.set_title("Convergence on the welded-beam design problem")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    save(fig, "fig_engineering")


# ---------------------------------------------------------------------------
# 6. Exploration / exploitation (population diversity) for EOGWO vs GWO
# ---------------------------------------------------------------------------
def fig_diversity():
    import sys
    sys.path.insert(0, HERE)
    from cec2017 import get_suite
    from optimizers import _clip, _levy_step
    import numpy as np

    f = get_suite(30)[4]  # F5 Rastrigin multimodal
    D = 30; pop = 30; max_iter = 500
    curves = {}
    for label, use_enh in [("EOGWO", True), ("GWO", False)]:
        rng = np.random.default_rng(11)
        X = rng.uniform(f.lb, f.ub, size=(pop, D))
        fit = f(X); order = np.argsort(fit)
        alpha, beta_, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
        div = np.empty(max_iter)
        for t in range(max_iter):
            if use_enh:
                a = 2.0 * np.cos((np.pi / 2.0) * (t / max_iter))
            else:
                a = 2.0 - 2.0 * t / max_iter
            for i in range(pop):
                A1 = a * (2 * rng.random(D) - 1); C1 = 2 * rng.random(D)
                A2 = a * (2 * rng.random(D) - 1); C2 = 2 * rng.random(D)
                A3 = a * (2 * rng.random(D) - 1); C3 = 2 * rng.random(D)
                X1 = alpha - A1 * np.abs(C1 * alpha - X[i])
                X2 = beta_ - A2 * np.abs(C2 * beta_ - X[i])
                X3 = delta - A3 * np.abs(C3 * delta - X[i])
                X[i] = _clip((X1 + X2 + X3) / 3.0, f.lb, f.ub)
            fit = f(X); order = np.argsort(fit)
            alpha, beta_, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
            # population diversity = mean distance to centroid
            c = X.mean(axis=0)
            div[t] = np.mean(np.sqrt(np.sum((X - c) ** 2, axis=1)))
        curves[label] = div
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label in ["EOGWO", "GWO"]:
        ax.plot(np.arange(1, max_iter + 1), curves[label], color=CMAP[label],
                ls=LS[label], lw=1.8, label=label)
    ax.set_xlabel("Iteration"); ax.set_ylabel("Population diversity (mean dist. to centroid)")
    ax.set_title("Exploration--exploitation balance on F5 (Rastrigin, D=30)")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_diversity")


if __name__ == "__main__":
    fig_convergence()
    fig_boxplots()
    fig_friedman_ranks()
    fig_cd_diagram()
    fig_engineering()
    fig_diversity()
    print("All figures generated.")
