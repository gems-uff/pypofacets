"""M8 (part 2): functional-structure alignment (metric Sim of Section 4.6.2) between the expert
and the LLM-modularized versions of the six multi-script algorithms.

Usage: python sim_per_script.py <experiments-jidm clone> [out.csv]
For each algorithm folder, reference = <name>.py (expert), candidate = <name>_claude_modular.py (LLM).
Reports |G|, |AI|, TP_G->AI, TP_AI->G, functional precision, recall and F, plus the best match per function.
Note: Sim is a soft, non-bijective similarity (each function takes its best counterpart in the other
set, as in Section 4.6.2); several LLM functions may map to the same expert function. As a sensitivity
check, the one-to-one columns use the optimal one-to-one assignment (Hungarian algorithm).
"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_functions_lib import extract_functions, function_similarity

ALGOS = ["belief_propagation", "cvar", "fft", "gauss_legendre_quadrature", "heat_distribution_lu", "look_and_say"]

def soft(ref, cand):
    tp_ref = sum(max((function_similarity(lr, lc) for lc in cand.values()), default=0.0) for lr in ref.values())
    tp_cand = sum(max((function_similarity(lc, lr) for lr in ref.values()), default=0.0) for lc in cand.values())
    p = tp_cand / len(cand) if cand else 0.0
    r = tp_ref / len(ref) if ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return tp_ref, tp_cand, p, r, f

def one_to_one(ref, cand):
    """Optimal one-to-one assignment (maximum total similarity, Hungarian algorithm via scipy);
    falls back to greedy if scipy is unavailable and says so."""
    R, C = list(ref), list(cand)
    S = [[function_similarity(ref[a], cand[b]) for b in C] for a in R]
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(-np.array(S))
        total = float(sum(S[i][j] for i, j in zip(ri, ci))); how = "optimal"
    except ImportError:
        pairs = sorted(((S[i][j], i, j) for i in range(len(R)) for j in range(len(C))), reverse=True)
        ur, uc, total = set(), set(), 0.0
        for s, i, j in pairs:
            if i in ur or j in uc: continue
            ur.add(i); uc.add(j); total += s
        how = "greedy (scipy missing)"
    p = total / len(cand) if cand else 0.0; r = total / len(ref) if ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0), how

def main():
    repo = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "sim_per_script.csv"
    rows = []
    for a in ALGOS:
        ref = extract_functions(os.path.join(repo, a, f"{a}.py"))
        cand = extract_functions(os.path.join(repo, a, f"{a}_claude_modular.py"))
        tp_ref, tp_cand, p, r, f = soft(ref, cand)
        p1, r1, f1, how = one_to_one(ref, cand)
        rows.append(dict(algorithm=a, G=len(ref), AI=len(cand), TP_G_to_AI=round(tp_ref, 2), TP_AI_to_G=round(tp_cand, 2),
                         precision=round(p, 2), recall=round(r, 2), F=round(f, 2),
                         precision_1to1=round(p1, 2), recall_1to1=round(r1, 2), F_1to1=round(f1, 2)))
        print(f"\n== {a}: |G|={len(ref)} |AI|={len(cand)}  P={p:.2f} R={r:.2f} F={f:.2f}   (one-to-one {how}: P={p1:.2f} R={r1:.2f} F={f1:.2f})")
        for name_g, lines_g in ref.items():
            best = max(cand.items(), key=lambda kv: function_similarity(lines_g, kv[1]), default=(None, []))
            print(f"   {name_g:40s} -> {best[0]!s:35s} {function_similarity(lines_g, best[1]) if best[0] else 0:.2f}")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"\nwritten {out}")

if __name__ == "__main__":
    main()
