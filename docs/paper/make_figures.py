"""Regenerate the guide's figures from a schematic and recorded measurements."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


if __name__ == "__main__":
    ordered_ray()
    atomic_adjoint()
