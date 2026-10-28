"""Run a script (without noWorkflow) and record digests of every value returned by
functions defined in the script, plus the final module globals. Used to verify that
the three versions of each algorithm (expert, monolithic, LLM-modular) compute the same
results for the same input (guideline G1 / threat B9).

Usage: python capture_results.py <script.py> <arg> <out.json>
"""
import hashlib, json, os, runpy, sys, io, contextlib

import numpy as np


def digest(v):
    try:
        if isinstance(v, np.ndarray):
            return "ndarray%s:" % (v.shape,) + hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16]
        if isinstance(v, (list, tuple)) and v and all(isinstance(x, np.ndarray) for x in v):
            return "[" + ",".join(digest(x) for x in v) + "]"
        if isinstance(v, (int, float, complex, str, bool)) or v is None:
            return repr(v)
        if isinstance(v, (list, tuple)):
            r = repr(v)
            return "seq:" + hashlib.sha256(r.encode()).hexdigest()[:16] if len(r) > 200 else r
    except Exception as e:  # pragma: no cover
        return "err:%s" % e
    return None


script = os.path.abspath(sys.argv[1])
arg = sys.argv[2]
out = os.path.abspath(sys.argv[3])
sys.argv = [script, arg]
sys.path.insert(0, os.path.dirname(script))
os.chdir(os.path.dirname(script))

returns = []


def prof(frame, event, value):
    if event != "return":
        return
    fn = frame.f_code.co_filename
    name = frame.f_code.co_name
    if fn == script:
        d = digest(value)
        if d is not None:
            returns.append([name, d])
    elif os.path.basename(fn) == "solver.py" and name == "solve":
        cd = frame.f_locals.get("self")
        if cd is not None and hasattr(cd, "current_distribution"):
            returns.append(["Solver.solve.current_distribution", digest(cd.current_distribution)])


buf = io.StringIO()
sys.setprofile(prof)
try:
    with contextlib.redirect_stdout(buf):
        g = runpy.run_path(script, run_name="__main__")
finally:
    sys.setprofile(None)

globals_d = {k: digest(v) for k, v in g.items() if not k.startswith("__") and digest(v) is not None}
stdout_lines = [l for l in buf.getvalue().splitlines()]
json.dump({"script": os.path.basename(script), "arg": arg, "returns": returns,
           "globals": globals_d, "stdout": stdout_lines}, open(out, "w"), indent=1)
print("ok", os.path.basename(script), len(returns), "returns")
