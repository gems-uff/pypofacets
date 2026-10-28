"""Compute code and provenance-graph metrics for noWorkflow dataflow .dot files.

Usage: python graph_metrics.py <out.csv> <script.py>:<graph.dot> [...]

Node categories follow noWorkflow's visual encoding (fillcolor):
  #3A85B9 (dark blue box)      = function-call activation node
  #85CBD0 (light blue rounded) = expression / evaluation node (includes the
                                  function-definition reference nodes "<line> <name>")
  white                        = file artifact node (a_ ids)
Note: 'user_call' only recognizes functions defined in the analyzed script; calls to
classes or functions of auxiliary local modules (e.g., Model/Solver in Heat Distribution)
are counted as library calls. This classification is only used for PyPOFacets (Table 4).
Two universes are reported: 'all' (evaluation + file nodes, all edges, as in Table 4)
and 'eval' (evaluation nodes and edges between them only, as the repository script
behind Table 7). Longest path, width and parallelism are computed on the DAG obtained
after removing back edges (cycle-closing edges found by DFS), and the number of removed
edges is reported.
"""
import ast, csv, re, sys
from collections import defaultdict

import networkx as nx

NODE_RE = re.compile(r'^\s*([ea]_\w+)\s*\[label="(.*?)"\s+fillcolor="([^"]*)"')
EDGE_RE = re.compile(r'([ea]_\w+)\s*->\s*([ea]_\w+)')
CALL_RE = re.compile(r'^-?\d+ ([A-Za-z_][\w.]*)\((.*)\)$', re.S)
DEF_RE = re.compile(r'^-?\d+ ([A-Za-z_]\w*)$')


def user_functions(script):
    tree = ast.parse(open(script, encoding="utf-8").read())
    return {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}


def loc(script):
    return sum(1 for l in open(script, encoding="utf-8") if l.strip() and not l.strip().startswith("#"))


def parse_dot(path):
    G = nx.DiGraph()
    for line in open(path, encoding="utf-8", errors="replace"):
        m = NODE_RE.match(line)
        if m:
            G.add_node(m.group(1), label=m.group(2), color=m.group(3))
            continue
        for s, d in EDGE_RE.findall(line):
            G.add_edge(s, d)
    return G


def dag_metrics(G):
    """Longest path, width and parallelism after removing back edges."""
    H = G.copy()
    removed = 0
    while not nx.is_directed_acyclic_graph(H):
        cyc = nx.find_cycle(H)
        H.remove_edge(*cyc[-1][:2])
        removed += 1
    if H.number_of_nodes() == 0:
        return dict(longest=0, levels=0, width=0, parallelism=0, removed=removed)
    level = {}
    for n in nx.topological_sort(H):
        preds = list(H.predecessors(n))
        level[n] = 0 if not preds else 1 + max(level[p] for p in preds)
    longest = max(level.values())
    per = defaultdict(int)
    for n, l in level.items():
        per[l] += 1
    return dict(longest=longest, levels=longest + 1, width=max(per.values()),
                parallelism=round(H.number_of_nodes() / (longest + 1), 2), removed=removed)


def count_cycles(G, cap=100000):
    n = 0
    for _ in nx.simple_cycles(G):
        n += 1
        if n >= cap:
            return f">={cap}"
    return n


def metrics(script, dot):
    funcs = user_functions(script)
    G = parse_dot(dot)
    cats = dict(user_call=0, lib_call=0, def_ref=0, expr=0, file=0)

    def is_user(name):
        # labels longer than noWorkflow's name limit are shown as "prefix ... suffix"
        if " ... " in name:
            pre = name.split(" ... ")[0]
            return any(f.startswith(pre) for f in funcs)
        return name.split(".")[-1] in funcs

    for n, d in G.nodes(data=True):
        lab, col = d.get("label", ""), d.get("color", "")
        body = lab.split(" ", 1)[1] if " " in lab else lab
        if n.startswith("a_") or col == "white":
            cats["file"] += 1
        elif col == "#3A85B9":
            cats["user_call" if is_user(body.split("(")[0]) else "lib_call"] += 1
        else:
            cats["def_ref" if ("(" not in body and re.match(r'^[A-Za-z_]', body) and is_user(body)) else "expr"] += 1
    E = G.subgraph([n for n in G if n.startswith("e_")]).copy()
    da, de = dag_metrics(G), dag_metrics(E)
    return {
        "script": script.split("/")[-1], "dot": dot.split("/")[-1], "LOC": loc(script), "NFuncs": len(funcs),
        "nodes_all": G.number_of_nodes(), "edges_all": G.number_of_edges(),
        "nodes_eval": E.number_of_nodes(), "edges_eval": E.number_of_edges(),
        **{f"n_{k}": v for k, v in cats.items()},
        "is_dag_all": nx.is_directed_acyclic_graph(G), "cycles_all": count_cycles(G),
        "components_all": nx.number_weakly_connected_components(G),
        "components_eval": nx.number_weakly_connected_components(E) if E.number_of_nodes() else 0,
        "longest_all": da["longest"], "width_all": da["width"], "parallelism_all": da["parallelism"], "backedges_removed_all": da["removed"],
        "longest_eval": de["longest"], "width_eval": de["width"], "parallelism_eval": de["parallelism"], "backedges_removed_eval": de["removed"],
    }


if __name__ == "__main__":
    out, pairs = sys.argv[1], sys.argv[2:]
    rows = [metrics(*p.split(":")) for p in pairs]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['dot']:45s} LOC={r['LOC']:3d} F={r['NFuncs']} nodes={r['nodes_all']:5d}/{r['nodes_eval']:5d} edges={r['edges_all']:5d}/{r['edges_eval']:5d} "
              f"uc={r['n_user_call']:2d} lc={r['n_lib_call']:3d} def={r['n_def_ref']:2d} ex={r['n_expr']:4d} file={r['n_file']} "
              f"dag={r['is_dag_all']} cyc={r['cycles_all']} LP={r['longest_all']}/{r['longest_eval']} W={r['width_all']}/{r['width_eval']} par={r['parallelism_all']}")
