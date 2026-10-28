Prompt: automated refactoring of the monolithic PyPOFacets script (Section 4.5)
==============================================================================

Model: Gemini (web application), 19 February 2026. The prompt below is reproduced verbatim from the shared Gemini session (https://share.gemini.google/MXgG5rTzhst3), to which `monolithic_pypofacets.py` was attached. No YesWorkflow annotations, stage names, or expert-modularized script were provided.

```
I am conducting a comparative study on data provenance extraction using YesWorkflow (YW) as the ground truth (gold standard). The objective is to analyze how tools trace data lineage in a monolithic script versus a modularly refactored version. We aim for the modular structure to naturally reflect the functional units of calculation and data transformation steps, producing a provenance graph (nodes and edges) that aligns with scientific workflow best practices.

The Task:

Act as a Senior Software Architect specializing in Scientific Computing. Below is a monolithic Python script. Refactor it into a modular structure following these strict constraints:

Functional Decomposition by Calculation: Break the monolith into functions where each module represents a discrete step or "parcel" of the calculation logic. Each function should act as a clear transformation node (Input -> Process -> Output).

Alignment with Workflow Graphs: Design the granularity of the functions to be compatible with workflow modeling (like YesWorkflow). Avoid over-abstracting; instead, ensure that each major data state change in the original script corresponds to a specific function.

Data Flow Integrity: The sequence of operations and the "data lineage" must remain 100% identical to the monolith. Do not optimize, combine steps, or use "Pythonic" shortcuts that hide the intermediate stages of calculation.

Interface Clarity: Use explicit arguments and return values for all functions to make the data dependencies transparent for static analysis.

Comment Constraint: Do not add new comments to the script. Only keep the ones that were already present in the code.
```

Output and edits before execution
---------------------------------

The script returned in that session is `modular_pypofacets_gemini_v2.py` of the `experiments-jidm` repository, executed under noWorkflow as `modular_llm_pypofacets.py` (this repository). Two edits were made to the model output before execution, and nothing else was changed (the function bodies are byte-for-byte those of the session):

1. the `main()` wrapper produced by the model, which only called the eight functions in sequence, was removed and its body placed at the module level, so that the script has the same top-level structure as the monolithic and expert versions, which have no `main` function (with the wrapper, the whole execution would be summarized by a single activation node at the export depth used in the paper);
2. the name of the output directory was changed from `output/monolithic` to `output/modular_llm`.

Earlier version and other candidates
------------------------------------

- An earlier formulation of the prompt, sent to Gemini on 16 February 2026, produced a first version with seven functions (`modular_pypofacets_gemini_v1.py` in `experiments-jidm`), which was discarded; the version used in the paper (eight functions) is the one generated with the prompt above.
- The same prompt was given to ChatGPT and GitHub Copilot. Both produced modularized versions with multiple user-defined functions; these two candidates were not retained, and their sessions are not available, so this selection cannot be verified by third parties (Section 9 of the paper).
