"""Behaviour-preservation check (guideline G1): for each algorithm, compare the digests
recorded by capture_results.py for the expert, monolithic and LLM-modular versions.
A version 'preserves' the expert result when the expert's key result digest appears
among its returned values or final globals (monolithic scripts keep results in globals;
modular scripts return them from functions whose names may differ).

Usage: python compare_results.py <results_dir>
"""
import glob, json, os, sys

KEY = {  # expert function whose return value is the scientific result
    "belief_propagation": "belief_propagation",
    "cvar": "get_empirical_CVaR",
    "fft": "compute_FFT",
    "gauss_legendre_quadrature": "compute_quadrature",
    "look_and_say": "look_and_say_sequence",
    "heat_distribution_lu": "Solver.solve.current_distribution",
}

d = sys.argv[1]
for folder, fn in KEY.items():
    data = {v: json.load(open(f"{d}/{folder}__{v}.json")) for v in ("expert", "monolithic", "modular")}
    keys = [r[1] for r in data["expert"]["returns"] if r[0] == fn]
    if not keys:
        print(f"{folder}: expert key result not captured"); continue
    print(f"\n{folder}: expert key result(s) from {fn}: {len(keys)} value(s)")
    for v in ("monolithic", "modular"):
        pool = [r[1] for r in data[v]["returns"]] + list(data[v]["globals"].values())
        found = [k for k in keys if k in pool]
        status = "PRESERVED" if len(found) == len(keys) else ("PARTIAL" if found else "NOT FOUND")
        print(f"  {v:10s}: {status} ({len(found)}/{len(keys)} values matched)")
        if status != "PRESERVED":
            print("     expert:", keys[:3]); print("     candidates:", sorted(set(pool))[:12])
    so = {v: [l for l in data[v]["stdout"] if not l.replace('.', '').replace('-', '').replace('e', '').isdigit()] for v in data}
    if so["expert"] or so["monolithic"] or so["modular"]:
        print("  stdout (timings removed):", {v: so[v] for v in so})
