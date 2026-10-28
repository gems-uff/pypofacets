"""Re-execute the multi-script evaluation (6 algorithms x 3 versions) and the three
PyPOFacets versions under noWorkflow 2, one argument per folder, exporting the
dataflow graph of each trial with depth 1. Writes runs.csv, environment.txt and
one .dot per execution. Also runs capture_results.py (plain Python) for the
behaviour-preservation check.

Usage: python run_all.py <repo_dir> <work_dir>
"""
import csv, os, platform, shutil, subprocess, sys, time
from importlib.metadata import version

REPO, WORK = map(os.path.abspath, sys.argv[1:3])
TOOLS = os.path.dirname(os.path.abspath(__file__))

ARGS = {  # intended inputs (arguments.txt order); FFT raised to 500 so the loop executes
    "belief_propagation": "20",
    "cvar": "50",  # 10000 (SpeeduPy workload) and even 100 exhaust 7 GB RAM in the dataflow export of the modular versions; 50 is the largest tractable input
    "fft": "500",
    "gauss_legendre_quadrature": "50",
    "heat_distribution_lu": "1",
    "look_and_say": "5",
}
VERSIONS = {"expert": "{n}.py", "monolithic": "{n}_claude.py", "modular": "{n}_claude_modular.py"}
AUX = {"heat_distribution_lu": ["model.py", "solver.py", "fatoracao_lu.py", "heat_map.py"]}
PYPO = {"monolithic": "monolithic_pypofacets.py", "expert": "modular_pypofacets.py",
        "gemini_v2": "modular_pypofacets_gemini_v2.py"}

os.makedirs(WORK, exist_ok=True)
rows = []


def sh(cmd, cwd, timeout=3600):
    t = time.time()
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return p, time.time() - t


def run_one(folder, label, script, arg, extra_files, extra_args=()):
    d = os.path.join(WORK, folder, label)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    for f in [script] + extra_files:
        src = os.path.join(REPO, folder, f)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(d, f))
        else:
            shutil.copy(src, d)
    p, dt = sh([sys.executable, "-m", "noworkflow", "run", script, arg, *extra_args], d)
    open(os.path.join(d, "stdout.txt"), "w").write(p.stdout)
    open(os.path.join(d, "stderr.txt"), "w").write(p.stderr)
    lst, _ = sh([sys.executable, "-m", "noworkflow", "list"], d)
    trial = "?"
    uuid = "?"
    for line in lst.stdout.splitlines():
        if "Sequence Key:" in line:
            trial = line.split("Sequence Key:")[1].strip()
        if "Trial " in line and ": run" in line:
            uuid = line.split("Trial ")[1].split(":")[0]
    dot = os.path.join(WORK, "dataflows", f"{folder}__{label}.dot")
    os.makedirs(os.path.dirname(dot), exist_ok=True)
    e, _ = sh([sys.executable, "-m", "noworkflow", "export", "dataflow", trial, "-d", "1"], d)
    open(dot, "w").write(e.stdout)
    rows.append({"folder": folder, "version": label, "script": script, "argument": " ".join([arg, *extra_args]),
                 "trial": trial, "trial_uuid": uuid, "returncode": p.returncode, "seconds": round(dt, 1),
                 "dot": os.path.relpath(dot, WORK)})
    # graph metrics (nodes, edges, ...) are computed afterwards by graph_metrics.py
    print(f"{folder:28s} {label:10s} trial={trial} rc={p.returncode} {dt:6.1f}s dot_bytes={len(e.stdout)}", flush=True)
    if p.returncode != 0:
        print(p.stderr[-800:])


for folder, arg in ARGS.items():
    for label, pat in VERSIONS.items():
        script = pat.format(n=folder)
        run_one(folder, label, script, arg, AUX.get(folder, []))
        # plain-Python result capture for the behaviour check
        out = os.path.join(WORK, "results", f"{folder}__{label}.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        p, _ = sh([sys.executable, os.path.join(TOOLS, "capture_results.py"),
                   os.path.join(WORK, folder, label, script), arg, out], WORK)
        if p.returncode != 0:
            print("  capture failed:", p.stderr[-400:])

for label, script in PYPO.items():
    run_one("_pypofacets", label, script, "BOX", ["BOX", "input_data_file_002.dat"], ("input_data_file_002.dat",))

with open(os.path.join(WORK, "runs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

env = [f"date: {time.strftime('%Y-%m-%d %H:%M:%S %z')}", f"python: {platform.python_version()} ({platform.python_implementation()})",
       f"platform: {platform.platform()}"]
for pkg in ["noworkflow", "numpy", "scipy", "matplotlib", "seaborn", "imageio"]:
    try:
        env.append(f"{pkg}: {version(pkg)}")
    except Exception:
        env.append(f"{pkg}: not installed")
env.append("export command: now export dataflow <trial> -d 1  (mode coarseGrain, accesses 1, evaluations 1)")
env.append("arguments: " + ", ".join(f"{k}={v}" for k, v in ARGS.items()) + ", _pypofacets=BOX input_data_file_002.dat")
open(os.path.join(WORK, "environment.txt"), "w").write("\n".join(env) + "\n")
print("\n".join(env))
