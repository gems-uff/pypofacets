Supplementary LLM refactoring round (28 October 2026)
=====================================================

Section 9 of the paper (Table 10) reports a supplementary round run to examine whether other LLM generations would yield different function boundaries. The files of this round use the internal prefix `m5_`.

Protocol
--------

* Prompt: the block of [`../prompts/pypofacets_llm_refactoring_prompt.md`](../prompts/pypofacets_llm_refactoring_prompt.md), byte for byte (1,658 bytes), with `monolithic_pypofacets.py` attached. No other instruction was given.
* One generation per model. All three outputs are evaluated; none was selected or discarded.
* No file was edited after generation. The evaluated copies received only the two edits made to the Gemini version of Section 4.5, listed below.

Configuration of each interface, as presented at the time
---------------------------------------------------------

| Model | Interface | Configuration shown | Local time (approx.) | Interaction |
|---|---|---|---|---|
| ChatGPT | Web, Plus plan, temporary chat | Model name not shown by the interface (default model); reasoning level Medium | 15:30 | One message. The code was delivered as a downloadable file (`modular_pypofacets.py`, identical to `original/m5_chatgpt.py`). A second message, sent after the generation, only asked the name of the model; the answer did not change the code, and the self-identification is not confirmed by the interface. |
| Gemini | Web | "Gemini 3.1 Pro" in the model selector, reasoning level Low (the selection shown when the conversation was opened) | 15:45 | One message. The answer contains only the code block. Public link to the conversation: <https://share.gemini.google/FtlmNXvkC1Ew>. |
| GitHub Copilot Chat | VS Code, GitHub Copilot Chat extension | Model Auto (Balance) | 16:00 and 16:05 | Attempt 1, Ask mode: no code, only a decomposition plan. Attempt 2, Agent mode, same prompt, in an isolated folder containing a copy of the monolithic script, which the agent rewrote in place; this is the evaluated output. The second attempt was made because Ask mode does not write files. |

Statements made by the models about their own outputs (ChatGPT and the Copilot agent report having executed the script) were not verified and are not used.

Files
-----

* `original/`: the three outputs as generated (`m5_chatgpt.py`, `m5_gemini.py`, `m5_copilot.py`).
* `responses/`: the text of the answers, as transcribed: the ChatGPT answer that accompanied the file (`m5_chatgpt_resposta.md`), the plan returned by GitHub Copilot Chat in Ask mode with the prompt as sent (`m5_copilot_tentativa1_ask.md`), and the final message of the Agent-mode attempt (`m5_copilot_resposta.md`, in Portuguese, preceded by a one-line note of the authors). Gemini returned no text besides the code.
* `edited/`: the evaluated copies and their diffs against the originals. The two edits, the same as for the Gemini version of Section 4.5: (1) the body of `main()` is placed at the module level, under the existing `if __name__ == "__main__":` guard (ChatGPT, Gemini) or without a guard where the output has none (GitHub Copilot Chat); in ChatGPT, whose `main(argv)` was called with `sys.argv`, the only use of `argv` in the body becomes `sys.argv`; (2) the output directory `output/monolithic`, inherited from the monolithic script, becomes `output/m5_<model>`.
* `provenance/`: the graphs exported by noWorkflow 2.1.1 (`now run m5_<model>.py BOX input_data_file_002.dat`, then `now export dataflow 1 -d 1`, CPython 3.13); `provenance/reduced/`: the derived views (`../analysis/reduced_view.py --user-only`); `provenance/normalized/m5_copilot.dot`: see below.
* Results, in [`../analysis/results/`](../analysis/results/) with the prefix `llm_round_`: Table 10 and its trails (`llm_round_table3.txt`, `llm_round_table3_m5_<model>.csv`), the Gemini version of Table 3 under the monolithic-name rule (`llm_round_table3_gemini_monolithic_names.txt`), graph metrics (`llm_round_metrics.csv`, `.txt`), behavior checks (`llm_round_g1_<model>_article.txt`, `_multi.txt`), and the execution records (`llm_round_runs.csv`, `llm_round_environment.txt`).

Results
-------

| Version | Matching of variables | \|AI\| | Nodes | Edges | SP | SR | SF |
|---|---|---:|---:|---:|---:|---:|---:|
| Gemini, Table 3 of the paper | own correspondence (column "AI Modular v2 name") | 8 | 55 | 82 | 0.841 | 0.663 | 0.741 |
| Gemini, Table 3 of the paper | names of the monolithic script (column "Monolithic name") | 8 | 55 | 82 | 0.796 | 0.630 | 0.703 |
| ChatGPT, this round | names of the monolithic script | 24 | 79 | 118 | 0.767 | 0.803 | 0.785 |
| Gemini, this round | names of the monolithic script | 10 | 58 | 88 | 0.810 | 0.647 | 0.719 |
| GitHub Copilot Chat, this round | names of the monolithic script | 20 | 82 | 124 | 0.766 | 0.811 | 0.788 |

The paper rounds these values to two decimal places. Matching rule: no hand-built correspondence exists for the new outputs, so their variable component matches the ground-truth items through the names of the monolithic script (column "Monolithic name" of `../mappings.xlsx`), which is the code the models received. None of the new outputs has, at the module level, a new name that corresponds to a ground-truth item under another name. The Gemini version of Table 3 uses its own correspondence, which credits its renamed variables `adj_delp` and `adj_delt`; under the monolithic-name rule it scores 0.703.

Behavior (`../analysis/results/llm_round_g1_*`): for the input of the case study (`0 0 0 0 360 90`), the three outputs produce the same `R.dat`, `E0.dat`, and `plot.png` as the monolithic and expert-modularized scripts, apart from the two header lines that record the execution time and the program name. With an input containing several azimuth angles (`0 30 10 0 30 10`), all three reproduce the output of the monolithic script, including its indexing defect (Section 4.4), whereas the Gemini version of Table 3 agrees with the expert-modularized one.

Notes on the graphs
-------------------

* Multi-line labels. In the GitHub Copilot Chat graph, two calls written over several lines produce node labels with line breaks inside quotes, which the line-oriented readers of `table3_recompute.py`, `reduced_view.py`, and `graph_metrics.py` do not parse. `provenance/normalized/m5_copilot.dot` replaces each of those line breaks with a space and changes nothing else; all measures of this round are computed on it (for ChatGPT and Gemini the normalized copy is identical to the exported graph). The values of Table 10 are the same with and without this normalization.
* Edge labels. The numeric indices in the `label` attribute of the edges are not deterministic in the export of noWorkflow 2.1.1: exporting the same provenance database again can number some edges differently. A re-export of these runs is therefore equivalent to the graphs published here (same topology up to node identifiers and dated paths, same node labels, same edge colors, styles, and arrowheads), not identical to them. No measure of this round uses those indices.
