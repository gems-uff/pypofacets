"""Automatic recomputation of the variable-equivalence component of Table 3.

For each PyPOFacets run, the data nodes of the depth-1 dataflow graph are the variables
assigned in the top-level activation (the script body, or main() when the script has
one). Each such variable is represented by its name and the repr of its final value
recorded by noWorkflow. V_G = data nodes of the expert run (whose names map one-to-one to
the YesWorkflow data items), V_X = data nodes of the compared run.

Two matching rules are reported:
  strict  : (name, final value) must coincide
  name    : names coincide (values ignored)  -- for reference only
Run-dependent values (timestamp, program name, output directory) are listed separately.

Usage: python table3_variables.py <work_dir>
"""
import os, sqlite3, sys

WORK = sys.argv[1]
RUNS = {"expert": "expert", "gemini_v2": "gemini_v2", "monolithic": "monolithic"}
RUN_DEPENDENT = {"time", "now", "program_name", "output_dir", "r_data", "header"}


def data_nodes(run):
    db = os.path.join(WORK, "_pypofacets", run, ".noworkflow", "db.sqlite")
    c = sqlite3.connect(db)
    # activations: find the top-level script activation and a 'main' activation if any
    acts = c.execute("select id, name from activation order by id").fetchall()
    top = [a[0] for a in acts if a[1] in ("main",)] or []
    script_act = acts[0][0]
    scopes = set([script_act] + top)
    rows = c.execute("""select cc.name, e.repr, e.checkpoint from evaluation e
        join code_component cc on cc.id = e.code_component_id and cc.trial_id = e.trial_id
        where cc.type = 'name' and cc.mode = 'w' and e.activation_id in (%s)
        order by e.checkpoint""" % ",".join(str(s) for s in scopes)).fetchall()
    final = {}
    for name, rep, _ in rows:
        final[name] = rep
    return final


V = {k: data_nodes(v) for k, v in RUNS.items()}
G = V["expert"]
print(f"|V_G| (expert data nodes) = {len(G)}: {sorted(G)}\n")
for run in ("gemini_v2", "monolithic"):
    X = V[run]
    strict = {n for n in G if n in X and G[n] == X[n]}
    byname = {n for n in G if n in X}
    rd = {n for n in byname - strict if n in RUN_DEPENDENT}
    other = byname - strict - rd
    print(f"== {run}: |V_X| = {len(X)}")
    print(f"   name matches        : {len(byname)}  (FN_name = {len(G)-len(byname)}, FP_name = {len(X)-len(byname)})")
    print(f"   strict matches      : {len(strict)}  (FN = {len(G)-len(strict)}, FP = {len(X)-len(strict)})")
    print(f"   run-dependent values: {sorted(rd)}")
    print(f"   value mismatches    : {sorted(other)}")
    for n in sorted(other)[:6]:
        print(f"      {n}: expert={G[n][:50]!r} | {run}={X[n][:50]!r}")
    print(f"   expert-only (FN)    : {sorted(set(G)-set(X))}")
    print(f"   {run}-only (FP)     : {sorted(set(X)-set(G))[:40]}{' ...' if len(X)-len(byname) > 40 else ''}\n")
