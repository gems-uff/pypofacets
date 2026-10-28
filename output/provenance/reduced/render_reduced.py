"""M7, presentation only: renders a reduced view (output of reduced_view.py) for print, breaking
function labels longer than 24 characters into two lines at the underscore closest to the middle,
with node font 13 (was 11) and horizontal node separation 0.15 (was 0.3). Nodes, edges and every
other attribute are left unchanged.

Usage: python render_reduced.py <reduced.dot> <out.dot>   (then dot -Tpdf out.dot -o out.pdf)
"""
import re, sys

def wrap(name):
    if len(name) <= 24 or "_" not in name:
        return name
    cuts = [i for i, ch in enumerate(name) if ch == "_"]
    i = min(cuts, key=lambda k: abs(k - len(name) / 2))
    return name[:i + 1] + "\\n" + name[i + 1:]

src, dst = sys.argv[1], sys.argv[2]
out = []
for line in open(src, encoding="utf-8"):
    line = line.replace("nodesep=0.3;", "nodesep=0.15;").replace("fontsize=11]", "fontsize=13]")
    m = re.match(r'(\s*e_\w+\s*\[label=")([A-Za-z_]\w*)(".*)', line)
    out.append(m.group(1) + wrap(m.group(2)) + m.group(3) + "\n" if m else line)
open(dst, "w", encoding="utf-8").writelines(out)
