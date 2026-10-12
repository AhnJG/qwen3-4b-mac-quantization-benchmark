# Lessons on how I used AI

This page collects the mistakes that came from how I used AI while building this repo. It covers wrong directions, measurement mistakes, and documentation errors. Most of the measurement code and docs were written by AI coding tools (Claude, Codex), and other AIs reviewed the work. Whichever tool wrote or reviewed something, I am responsible for everything in this repo.

The "Relation to AI use" column says only what the records support. Where the records do not say, I wrote "unclear". I did not blame AI for experiment conditions I chose myself. "AI review" in the finder column refers to private review notes that are not in the public repo. Dates are 2026 commit dates.

## At a glance

| When / version | What went wrong | Relation to AI use | Who found it, how | Outcome |
|---|---|---|---|---|
| 1. 08-03 v1 | `generate()` was called twice, so prefill time leaked into decode speed. GGUF was biased the other way by its prefix cache | Used a bug in AI-written measurement code without verifying it | AI review | Discarded in v2. The context-scaling conclusion reversed |
| 2. 08-03 v1 CUDA | "NF4 prefill is faster on CUDA" was produced by run order with no warm-up | Used AI-written measurement code that had no warm-up, without verifying | AI review | CUDA claim corrected in a commit the same day |
| 3. 08-03 v1 | Peak VRAM on MPS and CUDA measured different things, but was interpreted as the same | Used AI-written measurement code without verifying it | AI review | Peak-VRAM measurement fixed |
| 4. 08-03 design | Engine effect and quantization effect were mixed in one comparison | Unclear | AI direction review | v2 added GGUF F16 to separate them in a 2×2 |
| 5. 08-04 v2 | MPS is asynchronous, so timestamps were taken at the wrong moment. Prefill leaked in again | Used AI-written code I believed was fixed, without independent checks | AI review: HF TTFT was 550 times shorter than the physical lower bound | Fixed in v3 with `sync_device()` |
| 6. 08-04 v2 | The ballast tool rewrote the same 64MB only, so no pressure was applied. I still concluded the mmap hypothesis was falsified | Adopted the AI-written tool and its interpretation as a conclusion before checking | AI review | Tool fixed in v3. mmap hypothesis re-confirmed in v4 (falsification withdrawn) |
| 7. v5 | The computer slept for 6 of 8 hours of the run | Not AI-related (run environment) | Unclear | Everything discarded. `caffeinate -i` and per-row timestamps |
| 8. 08-09 v6 | Repeating the same setting in blocks skewed every median by −21 to 25% | Unclear | Unclear | Whole run shuffled, error reduced to ±4% |
| 9. v3 to v6 | The 2048-token "cliff" was read as a kernel effect. Four versions reproduced it, but the cause was an average-based metric | Unclear | v7 diagnostic experiments (`tools/cliff_probe.py`) | 1024→2048 went from 3.38× to 1.28×. The v6 advice to look at the kernel first was the wrong direction |
| 10. v6 | Two claims did not hold: agreement of n=3 medians was called "reproduction", and a ratio of 1.99 was used to claim "orthogonal" | Unclear | Withdrawn in the v7 docs. An external review confirmed the withdrawal | Withdrawn. The two factors do interact |
| 11. 08-10 v7 | The engine-effect table used the metric I had already discarded, and showed only input 128, the smallest value | Unclear | AI direction review | Replaced with geometric means and per-length ranges |
| 12. 08-12 v7.1 | `min-p50` (the fastest value across repeats) was biased in favor of the conclusion | Unclear (the records do not say who proposed the correction) | AI review | Four claims withdrawn. See `estimator-bias-note.ko.md` |
| 13. 08-10 v7 text | Wrote "fully monotonic" although the label said median and the value was the minimum (A2). The self-contamination story misread a time series (A3). "GGUF is clean in 20/20 cells" was refuted by my own definition (B3) | Unclear (interpretation was written into docs before the data supported it) | AI review | First edition marked as corrected. Wording toned down |
| 14. 08-10 v7 | `output_agreement.py` had two defects. "All 240 tokens identical" was never actually measured | Used an AI-written check without verifying it | AI review | Removed from conclusions |
| 15. 08-10 v7 | The procedure behind some tables was not in code. Diagnostic logs were missing, so numbers were typed by hand from console output. Quoted values differed from the logs | Unclear | AI review and the pre-release independent review | Logs kept as `results/probe_*.log`. Values replaced with the log values |
| 16. 08-12 direction | The main thesis ("averages hide stalls") was already in prior work | Unclear | AI direction review (with web literature search) | Dropped the paper and finished the repo as "tool + case record" (v8) |
| 17. Oct, v8.3 | "4.3× → 1.28×" mixed a v6 mean (n=3) with a v7 median (n=5) in one sentence | Unclear | Pre-release independent review | Same-data value is 3.38× → 1.28× |
| 18. Oct, v8.3 | "NF4 is slower at every context and both capacities" holds only at 24GB | Unclear | Pre-release independent review | Fixed by stating the scope |
| 19. Oct, v8.3 | Called a result "robust" using confidence intervals from the withdrawn min-over-repeats estimator | Unclear | Pre-release independent review | Replaced with median-based intervals. Wording lowered to "consistent with" |
| 20. Oct, v8.3 | "20 llama.cpp cells" should be 10 per capacity. A "5 min 44 s stall" could not be recomputed | Unclear | Pre-release independent review | Cell count fixed. Figure replaced |
| 21. Oct, v8.4 | Speed used `(n_new−1)/decode time`, but total time used `150/speed` | Unclear | Pre-release external feedback | Unified to `TTFT + (n_new−1)/speed`. 414 s became 412 s |
| 22. Oct, v8.4 | "About 355 s of stall" mixed medians from different runs. "Evenly slow" had no basis | Unclear | Pre-release external feedback | Re-estimated per run: 231–363 s in four of five runs, the fifth not separable |
| 23. Oct, v8.4 | "The 16GB imitation is not a rough approximation", a causal "because", "reproduces on other machines" | Left overstated wording from an AI-drafted doc as it was | Pre-release external feedback | Rewritten as observation, hypothesis, and untested prediction |
| 24. Oct, v8.4 | "Community quality loss 1–3%" had no source | Left an unsourced citation from an AI-drafted doc as it was | Pre-release external feedback | Whole section deleted |
| 25. Oct | README ratios (2.58×, 2.48×, 1.27×) had no stated basis. Dividing the table below (input 128) gives 2.82×, 2.80×, 1.16× | The AI-drafted doc left out the basis | AI assistant check while preparing the README | Stated the basis (geometric mean over 5 input lengths) and ranges |
| 26. Oct | Wrote "the same 4 bits". Q4_K_M keeps some weights at 6 bits, and NF4 keeps embedding/output layers at 16 bits | Left wording from an AI-drafted doc as it was | AI assistant check while preparing the README | Reworded to "both called 4-bit" |
| 27. Oct | 24GB and 16GB were run one after the other (15:30–16:47, 16:59–21:05), so memory effect is mixed with background-load change. Chrome and the Claude app were left open on purpose, but their load was not recorded | My own experiment choice (not AI's fault) | I raised it, recalling the measurement day | Added to README limitations. "24GB" and "16GB" are described as nominal conditions |
| 28. Oct | My first `claims.py` passed a wrong number, because it only checked that a number equaled some computed value | Used an AI-written check tool without verifying it | Found while preparing the README with an AI assistant | Now ties 32 key sentences to their own values |
| 29. Oct 11 | An AI assistant rewrote history to remove a file with a personal email. Commit hashes changed and two README commit links broke | Did not check the side effects of the AI's work | Checked right after the push | Fixed in a follow-up commit |

## Recurring patterns

**1. I trusted AI-written measurement and check code without an independent check.** If the code ran and the numbers looked plausible, I took them as right. External reviews did checks like "is this value physically possible". Items: 1, 2, 3, 5, 6, 14, 28.

**2. Interpretation became the conclusion before the data supported it.** A plausible cause went into the docs first, and the data reversed it later. The mmap hypothesis was falsified and then re-confirmed. The "cliff" was read as a kernel effect for four versions. Items: 2, 6, 9, 10, 13.

**3. Numbers were copied across versions with no single source.** One sentence could mix numbers from different versions, different summary statistics, or different formulas. Items: 11, 15, 17, 20, 21, 22, 25.

**4. Wording in the docs was stronger than the evidence.** "Robust", "not an approximation", "the same 4 bits", and an unsourced citation stayed in. One sentence relied on a value from a withdrawn method. Items: 13, 18, 19, 23, 24, 26.

**5. I trusted one check or one output, and the same kind of mistake came back somewhere new.** I believed a version was "fixed" without checking, and the prefill leak returned in a different form (5). A document that criticized biased summary values then showed only the smallest input length as its headline number (11). The check script (28) had the same problem. Items: 5, 11, 28.

**6. Cases not directly about AI use, or unclear in the records.** They include the first design that mixed engine and quantization, the sleep, the blocked repeat order, and checking prior work late. Running the memory conditions one after the other (27) was my own choice. Items: 4, 7, 8, 16, 27.

**7. I did not check the side effects of work an AI did.** After the history rewrite, README commit links broke, and they were fixed only after the push. Item: 29.

## What I do now (what changed)

**Already in place (in the repo)**

- `tools/audit.py`: an integrity gate before analysis. It checks sleep, incomplete cells, and timing mismatches. It also checks run-to-run variation.
- Every row records `started_at`, `ended_at`, `swap_used_gb`, and `ballast_gb`, and runs use `caffeinate -i`.
- Measurement order is shuffled across the whole run.
- The measurement code now also stores raw per-token times (`step_ms`). The current v7 data (200 rows) does not have them.
- Diagnostic logs are kept as `results/probe_*.log`.
- `tools/claims.py --check`: compares 32 key sentences and table rows in the README and the detail doc with the value each one states. It has limits. A correct number placed in the wrong sentence is not caught unless that sentence is a key sentence. The full report and `measurement-history.ko.md` are not checked.
- Numbers in the docs are taken only from the output of `python tools/claims.py`.
- Independent reviews come from an AI other than the one that wrote the code (review notes are private).
- Retractions and corrections are recorded in the [measurement history](measurement-history.ko.md) (Korean).
- Limitations are stated in the README: background apps, the sequential run order, and environment details I did not record.

**Planned for next time (not in the repo yet)**

- Record background load and free memory for every run.
- Interleave the 24GB and 16GB conditions.
- Decide the pass/fail criteria before looking at the results.

## See also

- [Measurement history](measurement-history.ko.md) (Korean): defects and corrections by version
- [README](../README.md): results summary and limitations
