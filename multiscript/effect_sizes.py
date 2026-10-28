"""Wilcoxon signed-rank tests (one-sided, paired by algorithm), median percent
reductions and Cliff's delta for the multi-script evaluation.

Usage: python effect_sizes.py <metrics.csv> [universe]
  metrics.csv: output of graph_metrics.py with a 'folder' and 'version' column
               (versions: expert, monolithic, modular)
  universe: 'all' (evaluation + file nodes) or 'eval' (default: all)
"""
import csv, statistics as st, sys
from scipy.stats import wilcoxon

path = sys.argv[1]
uni = sys.argv[2] if len(sys.argv) > 2 else "all"
rows = list(csv.DictReader(open(path)))
folders = sorted({r["folder"] for r in rows if not r["folder"].startswith("_")})
get = lambda f, v, k: rows[[i for i, r in enumerate(rows) if r["folder"] == f and r["version"] == v][0]][k]

METRICS = [("edges", f"edges_{uni}", "reduction"), ("nodes", f"nodes_{uni}", "reduction"),
           ("longest_path", f"longest_{uni}", "reduction"), ("width", f"width_{uni}", "reduction"),
           ("LOC", "LOC", "increase")]


def cliffs(x, y):
    return sum((a > b) - (a < b) for a in x for b in y) / (len(x) * len(y))


print(f"universe={uni}  N(folders)={len(folders)}: {', '.join(folders)}\n")
print(f"{'metric':13s} {'vs':9s} {'N':>2s} {'W':>4s} {'p(one-sided)':>13s} {'ties':>4s} {'median %chg':>11s} {'range':>14s} {'Cliff d':>8s}")
for name, col, direction in METRICS:
    for other in ("modular", "expert"):
        mono = [float(get(f, "monolithic", col)) for f in folders]
        oth = [float(get(f, other, col)) for f in folders]
        diffs = [m - o for m, o in zip(mono, oth)]
        nz = [(m, o) for m, o in zip(mono, oth) if m != o]
        ties = len(folders) - len(nz)
        if len(nz) >= 1 and len(set(d for d in diffs if d != 0)) >= 1:
            alt = "greater" if direction == "reduction" else "less"
            res = wilcoxon([a for a, b in nz], [b for a, b in nz], alternative=alt)
            W, p = res.statistic, res.pvalue
        else:
            W, p = float("nan"), float("nan")
        pct = [(m - o) / m * 100 for m, o in zip(mono, oth) if m]
        d = cliffs(mono, oth)
        print(f"{name:13s} mono×{other[:3]:4s} {len(nz):2d} {W:4.0f} {p:13.4f} {ties:4d} {st.median(pct):10.0f}% {min(pct):6.0f}..{max(pct):4.0f}% {d:+8.2f}")
