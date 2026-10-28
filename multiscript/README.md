Multi-script evaluation (Section 7) and re-execution package
============================================================

Environment: noWorkflow 2.1.1, CPython 3.13, Linux (see `environment.txt`). All graphs were exported with `now export dataflow <trial> -d 1`.

* `run_all.py`: runs the 6 x 3 scripts (and the three PyPOFacets versions) under noWorkflow with the inputs used in the paper, exports the dataflow graphs, and writes `runs.csv` and `environment.txt`. Usage: `python run_all.py <experiments-jidm clone> <work dir>`.
* `dataflows/`: the 18 provenance graphs of the six algorithms (`<algorithm>__<version>.dot`, version in {expert, monolithic, modular}) and the three PyPOFacets graphs (`_pypofacets__*.dot`).
* `graph_metrics.py`: nodes, edges, node categories, cycles, components, longest path, maximum width, and average parallelism (computed after removing back edges); produces `metrics.csv` (Tables 4 and 7).
* `effect_sizes.py`: Wilcoxon signed-rank tests, median reductions, and Cliff's delta over `metrics.csv` (Table 8 and Section 7).
* `capture_results.py` and `compare_results.py`: behaviour-preservation check (guideline G1): the three versions of each algorithm produce identical results for the same input.
* `table3_variables.py`: automatic robustness check of the variable equivalence component of Table 3 (strict name and final-value intersection), reported in Section 9.
* The source of the six algorithms (expert, monolithic, and LLM-modularized versions) is in the [experiments-jidm repository](https://github.com/claytonchagas/experiments-jidm).
