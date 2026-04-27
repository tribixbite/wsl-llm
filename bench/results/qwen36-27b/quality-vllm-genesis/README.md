# Qwen 3.6 27B Quality Bench on vLLM + Genesis (2026-04-27)

Hard prompts run against the production vLLM + Genesis stack (the §11 setup). Reports `decode_TPS` (excluding prefill) on real coding workloads and saves the actual generated outputs for inspection.

## Methodology

- Server: `vllm/vllm-openai 0.17.0rc1.dev126` + Sandermage Genesis v7.x patches (21 applied) + Lorbus AutoRound INT4 + fp8_e5m2 KV + MTP n=3
- Sampling: temp=0.6, top_p=0.95, top_k=20, `chat_template_kwargs:{enable_thinking:False}`
- Streaming bench: `decode_TPS = completion_tokens / (wall - TTFT)`
- Hard prompts based on prior session's hard-bench suite (Conway GoL HTML, regex Thompson NFA, Sudoku CSP+AC-3, Svelte todo, Kotlin LRU)

## Results

| Prompt | Tokens | Wall (s) | Decode TPS | Output |
|--------|-------:|---------:|-----------:|--------|
| Conway's Game of Life HTML (RLE URL hash) | 3436 | 65.55 | **53.02** | [conway_gol_rle.txt](conway_gol_rle.txt) / [conway_gol.html](conway_gol.html) |
| Regex Thompson NFA engine | 5468 | 99.37 | **55.16** | [regex_thompson_nfa.txt](regex_thompson_nfa.txt) |
| Sudoku CSP + AC-3 + backtracking | 3005 | 59.41 | **50.91** | [sudoku_csp_ac3.txt](sudoku_csp_ac3.txt) |
| Svelte 5 + TS todo app | 1891 | 35.79 | **53.14** | [svelte_todo.txt](svelte_todo.txt) |
| Kotlin generic LRU cache | 2448 | 45.51 | **54.05** | [kotlin_lru.txt](kotlin_lru.txt) |
| **Average** | | | **53.26** | |

## Average **53.3 t/s** on real production code prompts

This is consistent with the §11 envelope:
- Highly repetitive (JSON sequences) → 70 t/s
- **Real code prompts → 50-67 t/s** (this bench averages 53)
- Free-form prose (essays) → 45-50 t/s

## Conway GoL output validation

`conway_gol.html` — 9 KB self-contained single-page Conway's Game of Life with:
- 10 fps tick, play/pause/step/reset buttons
- Click-to-toggle cells when paused
- URL hash sync (load/save state via #pattern)
- Default glider when no hash given

The model produced a **standard RLE parser** (where `!` is end-of-pattern, `$` is end-of-row), even though the prompt example `3o2b1o!2b3o` used `!` as a row separator (non-standard). The output is consistent with [Conway's Game of Life RLE spec](https://conwaylife.com/wiki/Run_Length_Encoded). Open the HTML in a browser to see it work.

## Files

| File | What it is |
|------|------------|
| `*.txt` | Raw model output (markdown-formatted, includes commentary) |
| `conway_gol.html` | Extracted self-contained HTML from `conway_gol_rle.txt` |
| `results.json` | Machine-readable token/timing data |
