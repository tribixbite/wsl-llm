# Full 225-exercise aider polyglot — first run (2026-09-03)

Server: vLLM W4A16 + MTP + vision, 250 W, `MAX_SEQS=1`, via the monitor proxy.
temp 1.0, `reasoning_effort=medium`, 2 attempts, whole-file edit format.

## ⚠️ This run is INVALID for cpp and java — harness bug, now fixed

| language | n | pass@1 | pass@2 | |
|---|---:|---:|---:|---|
| javascript | 49 | 67.3% | **87.8%** | ok |
| go | 39 | 38.5% | 61.5% | ok |
| python | 34 | 26.5% | 55.9% | ok |
| rust | 30 | 30.0% | 53.3% | ok |
| **cpp** | 26 | **0.0%** | **0.0%** | ❌ runner bug |
| **java** | 47 | **0.0%** | **0.0%** | ❌ runner bug |
| TOTAL (as reported) | 225 | 29.3% | 45.3% | misleading |
| **TOTAL (working runners only)** | **152** | **41.4%** | **67.1%** | |

**Root cause:** `run_tests()` copied each exercise into a randomly-named temp dir
(`tempfile.TemporaryDirectory()`), but exercism's cpp `CMakeLists.txt` and java
gradle derive the target/project name from the **directory name**. CMake failed with
`No SOURCES given to target: <random>` for every cpp exercise. Fixed by copying to
`<tmp>/<exercise-name>/`; verified CMake now configures cleanly.

So the headline 45.3% is a **harness artifact**. The defensible figure from this run
is **67.1% pass@2 across the 152 exercises whose runners worked** — consistent with
our earlier 77.0% on the easier python+javascript subset.

Other signals from this run:
- `percent_cases_well_formed` **92.4%** (17 cases with a malformed reply) — worse than
  the 97.3% in the community reference run; whole-file format on a thinking model
  sometimes emits prose instead of a fenced block.
- `exhausted_context_windows` **21** — 21 cases hit `finish_reason=length`. Thinking
  plus a 12k cap is tight for the harder exercises; raise `--max-tokens`.
- `seconds_per_case` **58.1 s** (vs 237.7 s in the community 5090 run at xhigh effort).

## Comparison to the community reference

```
- dirname: 2026-08-27-14-04-02--gittensor-5090-Qwen3.8-27B-NVFP4-xhigh-effort
  test_cases: 225   edit_format: diff   pass_rate_1: 32.9   pass_rate_2: 77.8
  percent_cases_well_formed: 97.3   seconds_per_case: 237.7
```
Theirs is **diff format at xhigh effort on a 5090**; ours is **whole-file at medium
effort on a 3090**. Not directly comparable, but their 77.8% vs our 67.1%
(working-runner subset) is the right pairing to improve against — and the two obvious
levers are the malformed-reply rate and the 21 context exhaustions.
