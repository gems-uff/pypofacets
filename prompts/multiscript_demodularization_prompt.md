Prompt: demodularization (expert script to monolithic baseline), Section 7
=========================================================================

Model: Claude Opus 4.6 (identifier `copilot/claude-opus-4.6`), accessed through GitHub Copilot Chat in Visual Studio Code, 16 February 2026 (UTC-03:00). Applied to each of the six expert scripts, one chat session per script. The texts below are reproduced verbatim from the `message.text` field of the local Copilot Chat history of the workspace (SHA-256 of each message, computed over the UTF-8 text with CRLF line endings, in the table at the end).

Prompt (final form, used for FFT, Gauss-Legendre Quadrature, Heat Distribution, and Look-and-Say)
--------------------------------------------------------------------------------------------------

```
I am conducting a comparative study on the extraction of data provenance graphs. The goal is to analyze how static and dynamic analysis tools behave when comparing original Python scripts (which we will assume are well-written and modular, as they were authored by specialists) against their functionally identical, "demodularized" (monolithic) versions.
Act as a Software Refactoring Specialist. Below, I will provide a modularized Python script. Your task is to rewrite this code into a single, linear execution block (a monolithic script), adhering to these strict constraints:
Functional Integrity: The resulting script must produce exactly the same output and behavior as the original.
Total Demodularization: Eliminate all function definitions (def) and class definitions (class). The code must flow from top to bottom.
Scope Replacement: Where there were function calls, you must inline the logic directly, ensuring that variable handling and control flow (loops, conditionals) are preserved.
Provenance Preservation: Do not simplify calculations or data logic. If a variable passed through three functions to be processed, it must continue to pass through the same logical steps in the linear script, so that the data lineage remains traceable and identical to the original.
Clean Code: Preserve original comments only. Do not add new comments or annotations to the script.
```

For these four scripts, the source code of the expert script was pasted in the same message, immediately after the prompt. The full messages, including the pasted code, are in `multiscript_demodularization_messages.txt`.

Variant used for Belief Propagation and CVaR
--------------------------------------------

For Belief Propagation and CVaR, the message contained the same prompt without its last line (the `Clean Code` constraint), and the script was not pasted in the message text (it was supplied through the editor context of Copilot Chat, which is not recorded in `message.text`):

```
I am conducting a comparative study on the extraction of data provenance graphs. The goal is to analyze how static and dynamic analysis tools behave when comparing original Python scripts (which we will assume are well-written and modular, as they were authored by specialists) against their functionally identical, "demodularized" (monolithic) versions.
Act as a Software Refactoring Specialist. Below, I will provide a modularized Python script. Your task is to rewrite this code into a single, linear execution block (a monolithic script), adhering to these strict constraints:
Functional Integrity: The resulting script must produce exactly the same output and behavior as the original.
Total Demodularization: Eliminate all function definitions (def) and class definitions (class). The code must flow from top to bottom.
Scope Replacement: Where there were function calls, you must inline the logic directly, ensuring that variable handling and control flow (loops, conditionals) are preserved.
Provenance Preservation: Do not simplify calculations or data logic. If a variable passed through three functions to be processed, it must continue to pass through the same logical steps in the linear script, so that the data lineage remains traceable and identical to the original.
```

Discarded first attempts
------------------------

For Belief Propagation and CVaR, a first message (sent a few minutes before the final one, in a separate session) asked for boundary comments instead of forbidding new comments; its last line was `Comments: Insert brief comments indicating where each original module began and ended to facilitate research verification.` The outputs of these two attempts were discarded and both scripts were regenerated with the prompt above. The discarded message text is also in `multiscript_demodularization_messages.txt`.

Auxiliary modules of Heat Distribution
--------------------------------------

The same prompt (final form, with the code pasted) was also sent for the auxiliary modules `fatoracao_lu.py` and `heat_map.py` of Heat Distribution. The resulting files (`fatoracao_lu_claude.py`, `heat_map_claude.py`) are kept in the repository but are not used by any of the three versions analyzed in the paper: the monolithic, LLM-modularized, and expert versions of Heat Distribution all import the original auxiliary modules (`model.py`, `solver.py`, `fatoracao_lu.py`, `heat_map.py`), which were kept identical across the three versions.

Record of the messages
----------------------

| Script | Message timestamp | Request ID | SHA-256 (prefix) |
|---|---|---|---|
| belief_propagation | 2026-02-16T18:20:34-03:00 | request_e732162d-ce4e-4bf2-ad96-ed9012711812 | 8ef2fd11e4d5db80... |
| belief_propagation | 2026-02-16T18:24:37-03:00 | request_ff5259d0-42cf-4920-b2c5-f84e399818b6 | 6375f27eb9d5d9ee... |
| cvar | 2026-02-16T18:22:57-03:00 | request_29bf09a9-eefd-409e-8b14-f0ddb8301849 | 8ef2fd11e4d5db80... |
| cvar | 2026-02-16T18:33:04-03:00 | request_bcde03cb-1940-4c17-a62f-56b14a2cdbc3 | 6375f27eb9d5d9ee... |
| fft | 2026-02-16T19:01:06-03:00 | request_bdc040d2-d247-4c09-ad2b-b41d6dd3c64d | 2111dfce29e56c6f... |
| gauss_legendre_quadrature | 2026-02-16T19:04:47-03:00 | request_5c71b638-860f-4089-bd4e-8a8536b4a435 | 422252093596e452... |
| heat_distribution_lu | 2026-02-16T19:31:40-03:00 | request_44278d83-0820-43b2-bfd5-cdcf6a9ff4df | 6c47a1c029d1184e... |
| look_and_say | 2026-02-16T19:08:35-03:00 | request_11b2861b-ca07-40dc-a4f1-b505d636f1df | 00bfc62fe3f857fb... |
| fatoracao_lu | 2026-02-16T19:30:49-03:00 | request_dc453160-7bc3-4c92-be69-6b09f84ff7a3 | 6446186eecc851ba... |
| heat_map | 2026-02-16T19:32:53-03:00 | request_42cebd2f-fb42-42f5-914a-2e129e0d060d | b4688076fd299936... |

The six monolithic baselines were committed on 16 February 2026 (commit `b13c271` of the `experiments-jidm` repository).
