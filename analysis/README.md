Comparison with the ground truth (Sections 4.6.1, 4.6.2, 7 and 9)
==================================================================

* `label_match.py`: the single matching rule (a node matches a ground-truth item only when its label equals the item's name, file nodes by path; one node per item).
* `table3_recompute.py`: Table 3 (functional component with the metric of `compare_functions_lib.py`, variable component with `label_match.py`); `aliases.csv` is the manifest of manual matches by final value, empty for the three versions.
* `edges_compare.py`: edge comparison of Section 4.6.1 (one-to-one, orientation ignored).
* `sim_per_script.py`: functional alignment per script of Section 7.
* `reduced_view.py`: derived views of Figures 6 and 7.
* `results/`: outputs used in the paper (see the main README).

First submission
----------------

* `compare_functions.py`: computes the functional-structure equivalence between the functions of the expert-modularized script (reference) and the LLM-modularized script (candidate): Jaccard similarity over the tokenized lines of each function and optimal bidirectional matching. Usage: `python compare_functions.py modular_expert_pypofacets.py modular_llm_pypofacets.py`.
* `similarity.txt`: output of `compare_functions.py` used as the functional component of Table 3. It was produced from the copies of the two scripts kept in the [experiments-jidm repository](https://github.com/claytonchagas/experiments-jidm/tree/main/_pypofacets) (`modular_pypofacets.py` and `modular_pypofacets_gemini_v2.py`), which differ from the copies in this repository only in the name of the output folder written by the first function; running the script on this repository's copies therefore changes only the similarity of that first pair (0.9749 instead of 0.9870) and the aggregate scores in the third decimal place.
* `../mappings.xlsx`: variable equivalence mapping (the 40 data items and 7 port nodes of the YesWorkflow ground truth against the data nodes of the monolithic, expert, and Gemini noWorkflow graphs) and the aggregation of both components into Soft Precision, Soft Recall, and Soft F-Measure (Table 3).
