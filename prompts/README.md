Prompts used in the paper
=========================

All prompts are reproduced verbatim, as sent to the models. Each file records the model, the date, the source of the transcript, and every follow-up message, variant, discarded attempt, and edit made to the output before execution.

| File | Used in | Model(s) | Date |
|---|---|---|---|
| `pypofacets_llm_refactoring_prompt.md` | Section 4.5 (Step 4: automated refactoring of PyPOFacets) | ChatGPT, Gemini, and GitHub Copilot (same prompt, only the monolithic script attached); the Gemini output was selected | 19 February 2026 (Gemini session) |
| `pypofacets_llm_refactoring_prompt.md` (same prompt, unchanged) | Section 9 and Table 10 (supplementary round, one generation per model, no selection) | ChatGPT, Gemini, and GitHub Copilot Chat, each in the configuration its interface presented at the time | 28 October 2026 |
| `multiscript_demodularization_prompt.md` | Section 7 (expert script to monolithic baseline) | Claude Opus 4.6 via GitHub Copilot Chat (VS Code) | 16 February 2026 |
| `multiscript_demodularization_messages.txt` | Full messages of the demodularization sessions, including the pasted scripts and the two discarded first attempts | Claude Opus 4.6 via GitHub Copilot Chat (VS Code) | 16 February 2026 |
| `multiscript_remodularization_prompt.md` | Section 7 (monolithic baseline to LLM-modularized version) | Claude Opus 4.6 via GitHub Copilot Chat (VS Code) | 26 February 2026 |

Summary of the deviations from a single, identical prompt per step (all detailed in the files):

- Demodularization: the final prompt was sent in two forms, with the `Clean Code` constraint (four scripts, code pasted in the message) and without it (Belief Propagation and CVaR, code supplied through the editor context); for these two scripts a first attempt asking for boundary comments was discarded.
- Remodularization: identical text in the six sessions; two follow-up messages, one for CVaR (write the result to a separate file) and one for Heat Distribution (modularize only the main script, not its auxiliary modules).
- PyPOFacets: the Gemini output was executed after removing its `main()` wrapper and renaming the output directory; an earlier Gemini version with seven functions was discarded; the ChatGPT and GitHub Copilot candidates were not retained.
- PyPOFacets, supplementary round (28 October 2026): the same prompt was sent again, once to each model, and all three outputs were evaluated without selection. For GitHub Copilot Chat, a first attempt in Ask mode returned a plan without code, and the evaluated output comes from a second attempt in Agent mode; for ChatGPT, a second message asked only the name of the model and did not change the code. The configuration of each interface, the outputs as generated, the two edits (the same as for the Gemini version), and the results are in [`../llm_supplementary_round/`](../llm_supplementary_round/).

The `experiments-jidm` repository also contains demodularized versions of the six scripts produced by ChatGPT (`*_chatgpt.py`) and by Gemini (`*_gemini.py`), and remodularized versions produced by Gemini (`*_gemini_modular.py`). They are not used in the paper, whose Section 7 reports only the Claude versions (`*_claude.py`, `*_claude_modular.py`).
