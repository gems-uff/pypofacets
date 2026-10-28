"""M7: reduced view of a noWorkflow dataflow graph for print: keeps only function-call activation
nodes (fillcolor #3A85B9) and file artifact nodes (a_ ids / white), and draws an edge between two
kept nodes when the full graph has a path between them that passes only through removed
(expression) nodes. Layout bottom-up (rankdir=BT), so that the first stage is at the top, as in the YesWorkflow graph.

Usage: python reduced_view.py <in.dot> <out.dot> [--keep-defs] [--user-only <script.py>]
  --user-only: keep only calls to functions defined in <script.py> (drops library calls)
Then: dot -Tpdf out.dot -o out.pdf
"""
import ast, re, sys
import networkx as nx

NODE_RE = re.compile(r'^\s*([ea]_\w+)\s*\[label="(.*?)"\s+fillcolor="([^"]*)"')
EDGE_RE = re.compile(r'([ea]_\w+)\s*->\s*([ea]_\w+)(\s*\[[^\]]*\])?')

def main():
    src, dst = sys.argv[1], sys.argv[2]
    user_only = sys.argv[sys.argv.index("--user-only") + 1] if "--user-only" in sys.argv else None
    funcs = {f.name for f in ast.walk(ast.parse(open(user_only, encoding="utf-8").read())) if isinstance(f, ast.FunctionDef)} if user_only else None
    G = nx.DiGraph(); attrs = {}; dashed = set()
    for line in open(src, errors="replace"):
        m = NODE_RE.match(line)
        if m:
            G.add_node(m.group(1)); attrs[m.group(1)] = (m.group(2), m.group(3)); continue
        for s, d, a in EDGE_RE.findall(line):
            G.add_edge(s, d)
            if a and "dashed" in a: dashed.add((s, d))
    def keep(n):
        lab, col = attrs.get(n, ("", ""))
        if n.startswith("a_") or col == "white": return True
        if col == "#3A85B9":
            if funcs is None: return True
            name = lab.split(" ", 1)[-1].split("(")[0]
            if " ... " in name: name = name.split(" ... ")[0]
            return any(f.startswith(name) or name.startswith(f) for f in funcs)
        return False
    kept = [n for n in G if keep(n)]
    H = nx.DiGraph(); H.add_nodes_from(kept)
    for u in kept:   # BFS through removed nodes only
        stack = [v for v in G.successors(u)]; seen = set()
        while stack:
            v = stack.pop()
            if v in seen: continue
            seen.add(v)
            if v in H:
                H.add_edge(u, v)
            else:
                stack.extend(G.successors(v))
    with open(dst, "w") as f:
        f.write("digraph reduced {\n  rankdir=BT; ranksep=0.6; nodesep=0.3;\n  node [fontname=Helvetica fontsize=11];\n  edge [arrowsize=0.6];\n")
        for n in H:
            lab, col = attrs[n]
            lab = re.sub(r'^-?\d+ ', '', lab).replace('"', '\\"')
            if n.startswith("a_") or col == "white":
                lab = lab.split("/")[-1] if "/" in lab and not lab.endswith(".m") else lab
                f.write(f'  {n} [label="{lab}" shape=box style="rounded,filled" fillcolor=white];\n')
            else:
                name = lab.split("(")[0]
                if " ... " in name and funcs:   # expand truncated labels using the script's function names
                    pre = name.split(" ... ")[0]
                    full = [fn for fn in funcs if fn.startswith(pre)]
                    name = full[0] if len(full) == 1 else name
                f.write(f'  {n} [label="{name}" shape=box style=filled fillcolor="#3A85B9" fontcolor=white];\n')
        for u, v in H.edges():
            style = ' [style=dashed]' if (u.startswith("a_") or v.startswith("a_")) else ''
            f.write(f"  {u} -> {v}{style};\n")
        f.write("}\n")
    print(f"{src}: {G.number_of_nodes()} nodes -> {H.number_of_nodes()} nodes, {H.number_of_edges()} edges written to {dst}")

if __name__ == "__main__":
    main()
