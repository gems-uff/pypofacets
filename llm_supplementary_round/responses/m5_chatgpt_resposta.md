Refactored Python script — provenance-oriented modularization
I refactored the supplied PyPOFacets monolithic script into a modular implementation organized around explicit calculation and data-transformation stages.
Download modular_pypofacets.py

The refactoring preserves the original mathematical expressions, calculation order, intermediate data structures, loop behavior, and output file formats. No new comments were introduced.
Functional decomposition
The implementation contains 22 computational and workflow functions, plus a main() entry point.
Workflow stage	Main functions	Transformation
Initialization	read_arguments, create_output_directory	CLI arguments → execution paths
Input parameters	read_parameters	Parameter file → calculation parameters
Physical calculations	calculate_wavelength, calculate_polarization	Frequency/polarization → physical quantities
Geometry loading	read_coordinates, read_facets	Model files → coordinate and facet arrays
Geometry preparation	extract_facet_nodes, assemble_vertices, assemble_triangles	Mesh arrays → vertex and triangle structures
Visualization	render_model_plot	Geometry → 3D plot
Angular discretization	prepare_angular_steps, calculate_angles	Angular parameters → sampling states
Vector calculations	calculate_trigonometric_components, calculate_direction_vector, calculate_polarization_basis, calculate_field_vector	Angles → direction and field vectors
Output generation	build_output_header, open_output_files, write_output_headers, write_direction_record, write_field_record, generate_output_records, close_output_files	Calculated data → output files

This decomposition exposes the calculation stages as separate units with explicit interfaces, making their dependencies more accessible to provenance extraction and static analysis.
Validation results
I compiled the refactored script successfully and executed both versions using the same sample input data.
Verification	Result
Python syntax	Passed
R.dat numerical records	Identical
E0.dat numerical records	Identical
Output file generation	Passed
Plot file creation	Passed
Original equations and loop order	Preserved
New comments	None

The output headers can differ in the executable filename and execution timestamp. The numerical records matched exactly in the sample test. This is not yet proof of equivalence across every possible input.
Important provenance considerations
Three original behaviors were intentionally retained:
- Angular indexing: phr = phi[i2] * RAD references the accumulated phi list using the inner-loop index. I preserved this even though it may be an indexing defect.
- Unused calculations: waveL, phr0, and thr0 remain part of the original calculation sequence, although they do not contribute to the written output.
- Output dependencies: Writing R.dat and E0.dat remains explicitly separated, with the original record ordering preserved.
One clarification to the decomposition table: the implementation has 24 functions plus main(), rather than 22.

For your comparative experiment, the modular version exposes more function-level boundaries than the monolith. Whether those boundaries improve alignment with the YesWorkflow gold standard must be established by comparing the extracted nodes and edges against the canonical YW graph; modularity alone does not guarantee provenance correctness.

The delivered file is ready for that comparative analysis.
