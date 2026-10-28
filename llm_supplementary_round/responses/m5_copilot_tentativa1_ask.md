User: I am conducting a comparative study on data provenance extraction using YesWorkflow (YW) as the ground truth (gold standard). The objective is to analyze how tools trace data lineage in a monolithic script versus a modularly refactored version. We aim for the modular structure to naturally reflect the functional units of calculation and data transformation steps, producing a provenance graph (nodes and edges) that aligns with scientific workflow best practices.

The Task:

Act as a Senior Software Architect specializing in Scientific Computing. Below is a monolithic Python script. Refactor it into a modular structure following these strict constraints:

Functional Decomposition by Calculation: Break the monolith into functions where each module represents a discrete step or "parcel" of the calculation logic. Each function should act as a clear transformation node (Input -> Process -> Output).

Alignment with Workflow Graphs: Design the granularity of the functions to be compatible with workflow modeling (like YesWorkflow). Avoid over-abstracting; instead, ensure that each major data state change in the original script corresponds to a specific function.

Data Flow Integrity: The sequence of operations and the "data lineage" must remain 100% identical to the monolith. Do not optimize, combine steps, or use "Pythonic" shortcuts that hide the intermediate stages of calculation.

Interface Clarity: Use explicit arguments and return values for all functions to make the data dependencies transparent for static analysis.

Comment Constraint: Do not add new comments to the script. Only keep the ones that were already present in the code.


GitHub Copilot: I can’t edit the workspace in the current read-only Ask mode. For `monolithic_pypofacets.py`, the refactor should use a thin `main()` to orchestrate explicit transformation functions, keeping intermediate states visible:

- `parse_arguments(argv)` returns the program name, input paths, and output directory; `create_output_dir(output_dir)` performs directory creation.
- `read_parameters(input_data_file)` returns the ten parsed values; `calculate_wavelength(freq)` retains the existing `waveL` calculation, even though its result is unused.
- `select_polarization(ipol)` returns `et` and `ep`.
- `load_coordinates(input_model)` returns the coordinate table and its `xpts`, `ypts`, `zpts`, and `nverts` values. `load_facets(input_model)` returns the facet table and `node1`, `node2`, and `node3`.
- `build_vertex_list(...)` returns `r`; `build_triangle_indices(...)` returns `ntria` and `vind`.
- `save_model_plot(...)` handles the existing plotting steps.
- `normalize_angle_steps(...)` returns the adjusted `delp` and `delt`, preserving the existing `phr0` and `thr0` calculations. `calculate_scan_counts(...)` returns `it` and `ip`.
- `build_header(...)` returns the header text; `write_scan_rows(...)` handles the scan loops and file writes, delegating angle, direction-vector, and field-vector calculations to functions with explicit inputs and outputs.

For identical behavior, the refactor must preserve details that look unusual: `phi[i2]` is indexed by the inner-loop index, output rows use `i2`, plot coordinates are cast to `int`, and the two timestamps are taken at different points. The existing calculations for `waveL`, `phr0`, and `thr0` should also remain despite being unused. I don’t see existing comments in the supplied script, so the comment constraint means adding none.
