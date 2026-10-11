# Details

[Back to README](../README.md) · [한국어](details.ko.md)

This page explains the README results in more detail. For the setup names and measurement conditions, see "What was compared" and "Setup" in the [README](../README.md). The full set of tables ([full-report.ko.md](full-report.ko.md)) and the history of fixed measurement mistakes ([measurement-history.ko.md](measurement-history.ko.md)) are in Korean.

## Terms used here

- **Normal speed**: the time it usually takes to produce one token. It is the median (p50) over 150 tokens, and conclusions use it.
- **Stall**: a token that takes several seconds instead of the usual ~0.1 s.
- **Stall factor**: how many times slower the average speed is than the normal speed because of stalls. 1.0× means stalls had no effect.
- **Time to first token (TTFT)**: the time from giving the input to the first output token.
- **Perplexity**: how unsure the model is when predicting the next word. Lower is better.
- When summarizing ratios over the 5 input lengths, I give the geometric mean (an average of ratios) and the range.

## Key findings

### 1. Two methods both called 4-bit go in opposite directions

![Both called 4-bit, opposite directions](../figures/fig1_same_4bit_opposite.png)

With 24GB of memory, compared with the same program's 16-bit version:

- llama.cpp 4-bit (Q4_K_M) became **2.58× faster** (2.35–2.84× across input lengths).
- transformers 4-bit (NF4) became **2.48× slower** (0.36–0.46× the 16-bit speed).

With 16GB, llama.cpp 4-bit was still faster at every input length (2.21–2.87×). transformers 4-bit was 1.90× slower at 16GB, but that figure includes the two combinations the checks flag because their values jumped around between repeats (transformers 16-bit at input 2048, transformers 4-bit at input 4096). At input 2048, transformers 4-bit was even faster than its 16-bit version (1.09×), because the 16-bit version stalled heavily there.

**llama.cpp 4-bit also kept quality slightly better.** The 16-bit originals score a perplexity of about 9.54 (transformers 9.5413, llama.cpp 9.5417). Shrinking to 4-bit makes it a little worse: llama.cpp 4-bit scored 9.7446, +2.13% worse than its original, and transformers 4-bit scored 9.8060, +2.77% worse. Both losses are small, but llama.cpp 4-bit lost less. So llama.cpp 4-bit is both faster and slightly better in quality. The two 4-bit methods do not store exactly the same number of bits per weight, so part of the quality difference may come from that (see Limitations in the README). transformers 4-bit was also slower than llama.cpp 4-bit in every combination at both memory sizes.

**A smaller model was not automatically faster.** transformers 4-bit (2.9GB model) was slower than llama.cpp 16-bit (8GB model) by 3.2× at 24GB and 3.3× at 16GB (2.7–3.4× across input lengths). The likely reason is that llama.cpp has dedicated fast code for 4-bit math, while NF4 was built to save memory during model *training* and has no comparable fast code for the Mac GPU. I did not profile the GPU code, so this is an educated guess.

### 2. Most of the "3.3× faster" comes from 4-bit, not the program (starting from 16-bit)

| 24GB, normal-speed ratio | input 128 | 512 | 1024 | 2048 | 4096 | **Geo. mean** |
|---|---|---|---|---|---|---|
| Program only (transformers 16-bit → llama.cpp 16-bit) | 1.16 | 1.22 | 1.30 | 1.49 | 1.19 | **1.27×** |
| Then 4-bit (llama.cpp 16-bit → llama.cpp 4-bit) | 2.84 | 2.70 | 2.60 | 2.46 | 2.35 | **2.58×** |
| 4-bit in transformers (16-bit → 4-bit) | 0.36 | 0.38 | 0.39 | 0.46 | 0.44 | **0.40×** |
| Both (transformers 16-bit → llama.cpp 4-bit) | 3.30 | 3.29 | 3.40 | 3.67 | 2.79 | **3.28×** |

A ratio above 1 means faster; below 1 means slower.

Going from transformers 16-bit to llama.cpp 4-bit makes things 3.28× faster. Split into two steps:

1. Switch only the program to llama.cpp (still 16-bit): 1.27× (1.16–1.49× across input lengths)
2. Then switch to 4-bit: 2.58×

Since 1.27 × 2.58 ≈ 3.28, starting from 16-bit most of the gain comes from the 4-bit step.

Comparing the two 4-bit setups tells a different story, though. Going from transformers 4-bit (NF4) to llama.cpp 4-bit (Q4_K_M) makes things 8.1× (6.3–9.2×) faster; this changes both the program and the 4-bit format. 4-bit actually sped things up only in llama.cpp (finding 1). So the effect of 4-bit depends strongly on the program and format, and the step-by-step ratios above are a multiplicative split, not an established independent contribution of each factor.

### 3. The sudden slowdown at 2048 tokens came mostly from rare stalls

![The 2048-token cliff came mostly from rare stalls](../figures/fig2_cliff_was_metric.png)

In several rounds of measurement (v3, v5, v6, v7), transformers 16-bit slowed down sharply when the input grew from 1024 to 2048 tokens, then sped up again at 4096. Recording the time of every single token showed why:

- The average speed (total tokens ÷ total time) drops a lot if just a few of the 150 tokens stall for seconds.
- In the current data, going from 1024 to 2048 tokens makes things **3.38×** slower by the average, but only **1.28×** slower by the normal speed.
- It does not vanish completely: even by the normal speed, input 2048 is slower than 4096 (153.7 vs 144.9 ms).

Why the stalls happen is not settled. transformers grabs new memory for its KV cache (the memory that holds information about earlier tokens) at every token. Grabbing that memory once up front (`StaticCache`) made the stalls disappear in a diagnostic test (slowest token 3,447 → 272 ms, average of 2 runs). But that setting also slowed the normal speed from 137 to 246 ms, so it changes more than one thing at once. The result fits the explanation "grabbing memory every token causes the stalls", but does not prove it. During generation the GPU's memory use also climbs to about the machine's total memory (24.90 GiB); that the stalls are the operating system freeing memory at that point is still a hypothesis.

This is not a new discovery. Server benchmarks already report percentiles instead of averages (for example, MLPerf Inference's SingleStream scenario uses the 90th-percentile latency, p90). Agrawal et al. (arXiv:2507.09019) point out that average-style metrics hide generation stalls, *Etalon* (arXiv:2407.07000) argues that existing metrics give an incomplete picture of user experience, and Mytkowicz et al. (ASPLOS '09) show that a biased measuring method can produce the same wrong answer every time. Of the two on-device tools I checked, MLPerf Client reports average tokens/s and `llama-bench` reports the mean ± standard deviation over repeats.

### 4. When memory is short, the gap grows to minutes

With 16GB, llama.cpp was barely affected: stall factor ≤1.03× in all 10 of its combinations (2 setups × 5 input lengths), and its normal speed differed from the 24GB runs by -6% to +15%, within the run-to-run variation. llama.cpp reserves the memory it needs when it starts and reads the model file from disk as needed (mmap); this is standard llama.cpp behaviour, and its memory use was not measured here.

transformers 16-bit, by contrast, took 4.4–7.7 minutes of timed generation across five runs to produce 150 tokens from a 2048-token input. In four of the five runs, an estimated 231–363 s of that was stalls. In the fifth run the normal speed (p50) itself was so high that the estimate came out negative, so stalls could not be separated this way. (Stall time is not measured directly: it is the generation time minus tokens × that run's normal speed.) llama.cpp 4-bit did the same job in 14 seconds. That transformers combination is one the checks flag as jumping around too much, so read the minutes as a range, not a single number.

Running short of memory did not cause "transformers 4-bit is slow" in finding 1; it only makes an existing gap bigger through stalls.

## How the numbers are computed

- Conclusions use the **normal speed** (median of 5 repeats). The average speed (`tokens_per_sec`) mixes in stalls, so it is not used to compare computing speed. The total time including stalls (time to first token + generation) is reported separately as what a user actually waits for.
- **Stall factor** = normal speed ÷ average speed.
- Ratios use the geometric mean over the 5 input lengths plus the range, because picking one input length could amount to picking the number you like.
- Run order was shuffled, and each run records the time, swap use and reserved memory, so problems (sleep, memory shortage) can be checked afterwards. `tools/audit.py` runs these checks before analysis and separately flags combinations whose values jump around between repeats.
- Quality (perplexity): all four setups were scored the same way on the same text (the full WikiText-2 test set, 299,078 tokens). The 16-bit results of the two programs agreed within 0.005% (9.5413 vs 9.5417).
- Every number here can be recomputed from the raw data in `results/`: `tools/analyze.py` builds the speed tables, `tools/figures.py` draws the two figures, and `tools/claims.py` prints each number in this README together with the file it was computed from.

Headline numbers at 128 input tokens, 24GB, median of 5 runs (here all four have a stall factor of 1.0×, so the average and the normal speed agree):

| Setup | Peak memory (GiB) | Speed (tokens/s) | Time to first token (s) | Perplexity |
|---|---|---|---|---|
| transformers 16-bit (FP16), baseline | 7.54 | 9.74 | 0.48 | 9.5413 |
| transformers 4-bit (NF4) | 2.88 | 3.48 | 0.67 | 9.8060 |
| llama.cpp 16-bit (F16) | –\* | 11.31 | 0.43 | 9.5417 |
| llama.cpp 4-bit (Q4_K_M) | –\* | 31.92 | 0.46 | 9.7446 |

\* llama.cpp has no PyTorch-style memory counter, so it was not measured. transformers' peak memory was sampled every 20 ms.

## Measurement record

The current data was measured in v7. Nothing was re-measured after that; v8 added saving every token's time to the code, but the 200 measurements were taken before that. Across v1–v8, six measurement mistakes were found and fixed, several of them by independent AI reviews; I decided which findings to accept and what to retract:

1. **Input processing counted twice** — `generate()` was called twice, so the time spent reading the input leaked into the generation speed.
2. **Timing at the wrong moment** — on the Mac GPU, `time.time()` recorded when the CPU handed off the work, not when the GPU finished it, which brought the same problem back.
3. **Sleep** — the computer slept for 6 hours of an 8-hour run. Now every run uses `caffeinate -i` and records timestamps.
4. **Repeat order** — repeating the same setup back-to-back skewed every median by 21–25%; shuffling the full order cut that to ±4%.
5. **The metric** — the average hid the stalls (finding 3).
6. **How the representative value was picked** — picking the fastest of the repeats to "remove contamination" skewed the very comparison it was used for; four claims based on it were withdrawn.

Which experiments ruled out each alternative explanation is in [docs/measurement-history.ko.md](measurement-history.ko.md) (Korean); every table, including the 16GB and memory-shortage experiments, is in [docs/full-report.ko.md](full-report.ko.md) (Korean); and mistake 6 is analysed in [docs/estimator-bias-note.ko.md](estimator-bias-note.ko.md) (Korean). Raw data from older versions is kept in `results/archive/` as a record.
