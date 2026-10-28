"""Label-matching rule shared by table3_recompute.py and edges_compare.py (audit G1-F01 to G1-F04).

A ground-truth name matches a graph node only through one of these explicit rules; anything else
(suffixes, substrings, expressions that contain the name, case changes, abbreviated paths,
renamings) is not a match and can only enter through a manifest alias with evidence
(table3_recompute --aliases).
  1. Data name: the node label, without noWorkflow's line-number prefix ("19 input_data_file"),
     equals the name exactly.
  2. File path, only for file nodes (id a_*), with "\\" read as "/": "output/.../<f>" matches
     output/<dir>/<14-digit timestamp>/<f> (one run of the script); any other name matches exactly.
  3. Function name (call nodes and definition-reference nodes): the label body up to "(" equals the
     name; for labels truncated by noWorkflow as "<prefix> ... <rest>", the name must start with
     <prefix> (and end with <rest> when <rest> is still part of the name, not the argument list),
     and exactly one function of the script may satisfy it.
Item -> node assignment (resolve_items), in sheet order, a node serving at most one item: when
several nodes match an item's name, the stage rule keeps those that depend on a call of the
block that produces the item in YesWorkflow (rows OUTPUT of sheet Edges; noWorkflow draws
dependent -> dependency, so the edge is node -> call). One survivor is the item's node ("TP-auto").
If several remain, the item still counts as one TP (the counting rule is unchanged) but the
assignment is not asserted: the status is "TP-ambiguous" and every candidate is exported.
"""
import re

CALL = "#3A85B9"

def body(label):
    return label.split(" ", 1)[1] if re.match(r'^-?\d+ ', label) else label

def snake(n):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', n).lower()

def file_match(name, label):
    lab = label.replace("\\", "/")
    if name.startswith("output/.../"):
        m = re.fullmatch(r'output/[^/]+/\d{14}/([^/]+)', lab)
        return bool(m) and m.group(1) == name[len("output/.../"):]
    return lab == name

def data_match(name, node_id, label):
    """Rules 1 and 2. The caller must already have excluded call and definition-reference nodes."""
    if node_id.startswith("a_"):
        return file_match(name, label)
    return body(label) == name

def function_of(label, funcs):
    """Rule 3: the user-defined function named by a call or definition-reference label, or None."""
    b = body(label)
    cut = b.find(" ... ")
    paren = b.find("(")
    if cut < 0 or 0 <= paren < cut:
        n = b.split("(")[0]
        return n if n in funcs else None
    pre, rest = b[:cut], b[cut + 5:]
    tail = rest.split("(")[0]
    name_tail = tail if re.fullmatch(r'\w+', tail) else ""   # otherwise <rest> is the argument list
    hits = [f for f in funcs if f.startswith(pre) and f.endswith(name_tail) and len(f) >= len(pre) + len(name_tail)]
    return hits[0] if len(hits) == 1 else None

def block_calls(nodes, funcs, fn):
    """Call node ids of the block whose mapped function is fn (YW block name: snake_case fallback).
    nodes: iterable of (id, label, color)."""
    fn = fn if fn in funcs else snake(fn)
    return sorted(n for n, l, c in nodes if c == CALL and function_of(l, funcs | {fn}) == fn)

def producers(wb, nodes, funcs):
    """Ground-truth item -> call node ids of the blocks that output it (rows OUTPUT of sheet Edges)."""
    out = {}
    for r in wb["Edges"].iter_rows(min_row=5, values_only=True):
        if r[0] == "OUTPUT" and r[3] == "YES":
            out.setdefault(r[2], set()).update(block_calls(nodes, funcs, r[4] or r[1]))
    return out

def resolve_items(items, eligible, edges, prod):
    """items: [(gt_item, mapped name or None)] in sheet order; eligible: [(id, label)] in .dot order;
    edges: set of (src, dst); prod: producers(). Returns {gt_item: (status, [node ids])} with
    status TP-auto, TP-ambiguous or FN; for TP-ambiguous the first id is the one reserved for the count."""
    trail, used = {}, set()
    for item, mapped in items:
        cands = [n for n, l in eligible if mapped is not None and data_match(mapped, n, l)]
        if len(cands) > 1:
            staged = [n for n in cands if any((n, c) in edges for c in prod.get(item, ()))]
            cands = staged if staged else cands
        free = [n for n in cands if n not in used]
        if not free:
            trail[item] = ("FN", [])
        else:   # ambiguity is judged on all candidates, not on those left free by earlier items
            used.add(free[0])
            trail[item] = ("TP-auto", free) if len(cands) == 1 else ("TP-ambiguous", [free[0]] + [n for n in cands if n != free[0]])
    return trail
