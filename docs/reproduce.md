# Measure again from scratch

[Back to README](../README.md) · [한국어](reproduce.ko.md)

This needs an Apple Silicon Mac. Run the commands from the repository root. Use `--out` with a new folder so the stored raw data in `results/` is not overwritten.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # versions found afterwards in the measurement venv: environment/requirements-measured.txt
CMAKE_ARGS="-DGGML_METAL=on" pip install --no-cache-dir --force-reinstall llama-cpp-python

# caffeinate -i keeps the Mac awake: sleep silently corrupts timing
caffeinate -i python benchmark_quantization.py --stage context_sweep --repeats 5 --out results/rerun                  # 24GB
caffeinate -i python benchmark_quantization.py --stage context_sweep --repeats 5 --ballast-gb 8 --out results/rerun   # 16GB
python benchmark_quantization.py --stage perplexity --out results/rerun

python tools/audit.py results/rerun/results.jsonl --repeats 5
python tools/analyze.py results/rerun/results.jsonl
```

The package versions found in the measurement environment are listed in [environment/requirements-measured.txt](../environment/requirements-measured.txt), and the GGUF model files in [environment/model-files.txt](../environment/model-files.txt).

## How long it takes

By the 2026-08-09 records, the 100 measurements at 24GB took about 1.3 hours and the 100 at 16GB about 4.1 hours. Perplexity run time was not recorded (the code notes it takes hours across the four setups).

## Quick test

With 1 repeat, I expect about 15 minutes at 24GB, a fifth of the full run (not measured).

```bash
caffeinate -i python benchmark_quantization.py --stage context_sweep --repeats 1 --out results/smoke
python benchmark_quantization.py --stage perplexity --ppl-max-tokens 20000 --out results/smoke
python tools/audit.py results/smoke/results.jsonl --repeats 1
```

## Conditions

- The original run kept Chrome and the Claude app open on purpose, but their use was not recorded (see Limitations in the README). To compare cleanly, either close other programs, or keep a fixed set open and record their memory use during the run.
- On a machine with much more memory I expect the stall results not to appear (not tested). To recreate the conditions, use `--ballast-gb` to take up memory until the usable amount matches 24GB or 16GB (on a 64GB machine, `40` and `48`).
