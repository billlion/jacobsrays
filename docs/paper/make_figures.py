"""Regenerate the guide's figures from a schematic and recorded measurements."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
METADATA = {"CreationDate": None, "ModDate": None}


def ordered_ray():
    fig, ax = plt.subplots(figsize=(7.0, 3.0), layout="constrained")
    for x in range(5):
        ax.plot([x, x], [0, 2], color="0.75", lw=.8)
    for y in range(3):
        ax.plot([0, 4], [y, y], color="0.75", lw=.8)
    start = np.array([-.65, .23])
    end = np.array([4.65, 1.79])
    d = end-start
    cuts = [(x-start[0])/d[0] for x in range(5)]
    cuts += [(1-start[1])/d[1]]
    cuts = np.sort(cuts)
    ax.plot([start[0], end[0]], [start[1], end[1]], "--", color="0.6")
    for number, (a, b) in enumerate(zip(cuts[:-1], cuts[1:]), start=1):
        p, q = start+a*d, start+b*d
        colour = "#126E82" if number % 2 else "#BF5B3D"
        ax.plot([p[0], q[0]], [p[1], q[1]], color=colour, lw=4,
                solid_capstyle="butt")
        midpoint = (p+q)/2
        dy = .23 if number % 2 else -.26
        ax.text(midpoint[0], midpoint[1]+dy, str(number),
                ha="center", va="center", color=colour, weight="bold")
    ax.scatter(*start, color="0.2", s=20)
    ax.annotate("start", start, xytext=(-4, -16), textcoords="offset points")
    ax.annotate("end", end, xytext=(-5, 10), textcoords="offset points")
    ax.annotate("", xy=end, xytext=end-.45*d/np.linalg.norm(d),
                arrowprops={"arrowstyle": "->", "color": "0.2", "lw": 1.4})
    ax.text(2, -.65, r"Scalar: $\sum_j \ell_j f_j$     Transport: "
            r"$e^{\ell_qG_q}\cdots e^{\ell_2G_2}e^{\ell_1G_1}$",
            ha="center", va="center")
    ax.set(xlim=(-.9, 4.9), ylim=(-.9, 2.2), aspect="equal")
    ax.axis("off")
    fig.savefig(OUT / "ordered-ray.pdf", metadata=METADATA)
    plt.close(fig)


def atomic_adjoint():
    data = json.loads((HERE.parents[1] / "benchmarks/atomic_results_2026-10-27.json").read_text())
    fig, ax = plt.subplots(figsize=(6.7, 3.45), layout="constrained")
    for case, label, color, marker in [
        ("distributed", "Distributed", "#126E82", "o"),
        ("central", "Through centre", "#BF5B3D", "s"),
        ("identical", "Identical rays", "#594A9C", "^")]:
        rows = sorted((r for r in data["results"] if r["case"] == case),
                      key=lambda r: r["requested_threads"])
        ax.plot([r["requested_threads"] for r in rows],
                [1000*r["timings"]["atomic"]["median_seconds"] for r in rows],
                label=label, color=color, marker=marker, lw=1.8, ms=6)
    ax.set(xlabel="Requested OpenMP threads", ylabel="Atomic adjoint median (ms)",
           xticks=[1, 2, 4, 8], xlim=(.7, 8.3), ylim=(0, 28))
    ax.grid(axis="y", color="0.88", lw=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.savefig(OUT / "atomic-adjoint.pdf", metadata=METADATA)
    plt.close(fig)


def exact_notation():
    """Exact example from Appendix A, drawn in physical and parameter space."""
    fig = plt.figure(figsize=(7.0, 4.65), layout="constrained")
    axes = fig.subplots(2, 1, gridspec_kw={"height_ratios": [3.5, 1]})
    ax, time_ax = axes
    teal, orange = "#126E82", "#B44923"
    for c, color in [((0, 0), teal), ((1, 0), orange), ((2, 1), teal)]:
        ax.add_patch(Rectangle(c, 1, 1, facecolor=color, alpha=.10, edgecolor="none"))
    for x in [1, 2]:
        ax.plot([x, x], [0, 2], color=".72", lw=.8)
    ax.plot([0, 3], [1, 1], color=".72", lw=.8)
    ax.plot([0, 0, 3], [2, 0, 0], color=".3", lw=1.2)
    ax.plot([0, 3, 3], [2, 2, 0], color=".3", lw=1.2, ls=(0, (4, 3)))

    start, end = np.array([-1., -.5]), np.array([4., 2.])
    cuts = np.arange(1, 5)/5
    points = start + cuts[:, None]*(end-start)
    ax.plot([start[0], points[0, 0]], [start[1], points[0, 1]],
            color=".55", lw=1.2, ls="--")
    ax.plot([points[-1, 0], end[0]], [points[-1, 1], end[1]],
            color=".55", lw=1.2, ls="--")
    for j, color in enumerate([teal, orange, teal]):
        ax.plot(points[j:j+2, 0], points[j:j+2, 1], color=color, lw=4,
                solid_capstyle="butt")
    ax.scatter(points[:, 0], points[:, 1], color=".15", s=16, zorder=4)
    ax.scatter(*start, color=".25", s=16)
    ax.annotate("", xy=end, xytext=end-.4*(end-start)/np.linalg.norm(end-start),
                arrowprops={"arrowstyle": "->", "color": ".35", "lw": 1.2})
    ax.text(-1.08, -.78, r"$p=(-1,-1/2)$", ha="left", va="top")
    ax.text(4.15, 2.18, r"$e=(4,2)$", ha="right", va="bottom")
    ax.text(-.16, .17, r"$o=r(\alpha)$", ha="right", va="bottom")
    ax.text(.86, .66, r"$r(a)$", ha="center", va="bottom")
    ax.text(2.10, .84, r"$r(b)$", ha="left", va="top")
    ax.text(3.12, 1.40, r"$r(\beta)$", ha="left", va="top")
    for x in range(4):
        ax.text(x, -.16, str(x), ha="center", va="top", color=".3", fontsize=9)
    for y in [1, 2]:
        ax.text(-.13, y, str(y), ha="right", va="center", color=".3", fontsize=9)
    ax.text(3.25, -.16, r"$x\ (i=0)$", ha="left", va="top")
    ax.text(-.1, 2.33, r"$y\ (i=1)$", ha="right", va="bottom")
    ax.text(.45, .79, r"$V_{(0,0)}$", color=teal, ha="center")
    ax.text(1.5, .20, r"$V_c,\quad c=(1,0)$", color=orange, ha="center", fontsize=9)
    ax.text(2.5, 1.76, r"$V_{(2,1)}$", color=teal, ha="center")
    ax.set(xlim=(-1.3, 4.3), ylim=(-1.02, 2.65), aspect="equal")
    ax.axis("off")

    time_ax.plot([0, 1], [0, 0], color=".55", lw=1.2, ls="--")
    for j, color in enumerate([teal, orange, teal]):
        time_ax.plot(cuts[j:j+2], [0, 0], color=color, lw=4, solid_capstyle="butt")
    locations = [0, *cuts, 1]
    labels = [r"$0$", r"$\alpha=1/5$", r"$a=2/5$", r"$b=3/5$", r"$\beta=4/5$", r"$1$"]
    for t, label in zip(locations, labels):
        time_ax.plot([t, t], [-.06, .06], color=".25", lw=1)
        time_ax.text(t, .14, label, ha="center", va="bottom", fontsize=9)
    time_ax.text(.1, -.17, "outside grid", ha="center", va="top", color=".4", fontsize=9)
    time_ax.text(.5, -.17, r"interval for $V_{(1,0)}$", ha="center", va="top", color=orange, fontsize=9)
    time_ax.text(.9, -.17, "outside grid", ha="center", va="top", color=".4", fontsize=9)
    time_ax.text(1.04, 0, r"$t$", va="center")
    time_ax.set(xlim=(-.05, 1.08), ylim=(-.46, .48))
    time_ax.axis("off")
    fig.savefig(OUT / "exact-notation.pdf", metadata=METADATA)
    plt.close(fig)


if __name__ == "__main__":
    ordered_ray()
    atomic_adjoint()
    exact_notation()
