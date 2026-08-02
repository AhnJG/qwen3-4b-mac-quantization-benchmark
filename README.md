# Efficient AI: On-Device Quantization Benchmark (Qwen3-4B-Instruct)

> **Status: scaffold ready, numbers pending.** Code, structure, and docs are in place; the actual benchmark run happens on a GPU (Colab/local), not in this authoring environment. See [Next steps](#next-steps).

## TL;DR (to fill in after Phase 2)

> `Qwen3-4B-Instruct-2507`를 4-bit(NF4)로 양자화해 VRAM ??% 절감, 속도 ??배 향상, 품질 저하는 미미한 수준으로 유지했다. 컨텍스트 길이가 늘어날수록 ... (실제 수치로 교체)

## What this is

`Qwen/Qwen3-4B-Instruct-2507` 모델을 세 가지 방식(FP16 원본, bitsandbytes NF4 4-bit, llama.cpp GGUF Q4_K_M)으로 실행하며 다음을 정량 비교합니다.

- **메모리**: Peak VRAM 사용량 (GB)
- **속도**: Time-to-first-token(TTFT), 초당 토큰 생성 수(tokens/sec)
- **품질**: WikiText-2 (test split) sliding-window perplexity

추가로, 입력 길이가 128 → 4096 토큰으로 늘어날 때 세 방식의 속도/메모리가 어떻게 변하는지(KV 캐시 스케일링) 측정합니다 — 이 부분이 흔한 "짧은 프롬프트 한 번" 벤치마크와의 차별점입니다.

## Quickstart

**Google Colab (권장)**: `benchmark_quantization.ipynb`을 열고 안내를 따르세요. Runtime을 GPU(T4 이상)로 바꿔야 합니다.

**로컬 (CUDA GPU 필요)**:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124  # CUDA 버전에 맞게 cu121/cu124 등 선택

python benchmark_quantization.py --stage all
python benchmark_quantization.py --stage context_sweep
```

결과는 `results/results.jsonl`에 누적 기록됩니다.

## Results

*(Phase 2 실행 후 실제 수치로 교체 예정)*

| Config | Peak VRAM (GB) | Tokens/sec | TTFT (s) | Perplexity (WikiText-2) |
|---|---|---|---|---|
| FP16 (baseline) | – | – | – | – |
| NF4 4-bit (bitsandbytes) | – | – | – | – |
| GGUF Q4_K_M (llama.cpp) | – | – | – | n/a* |

\* GGUF perplexity는 llama.cpp 자체 구현으로 계산되어 FP16/NF4와 동일한 방법이 아니므로 직접 비교하지 않음 (자세한 내용은 [Limitations](#limitations) 참고).

### Context-length scaling

*(context_sweep 결과를 표/그래프로 채울 예정)*

## Related benchmarks (sanity check)

이 프로젝트와 똑같은 조합(FP16 vs bitsandbytes-NF4 vs GGUF, 이 모델 기준)을 그대로 측정한 공개 자료는 찾지 못했지만, 양자화의 일반적인 트레이드오프는 이미 여러 곳에 보고돼 있습니다. 아래 수치와 우리 결과가 같은 방향(대략적인 절감률/품질 저하폭)을 가리키는지 리포트에서 교차 검증합니다.

- Unsloth GGUF Benchmarks — Q4_K_M vs Q5_K_M perplexity 비교
- 커뮤니티 보고: Q4_K_M은 FP16 대비 용량 약 75% 감소, MMLU 등 품질 저하는 1~3% 수준

## Limitations

- **GGUF perplexity는 FP16/NF4와 다른 방식으로 계산됨** (llama.cpp 내부 구현 vs HuggingFace loss 기반). 숫자를 나란히 두되 "직접 비교 불가"임을 명시.
- **GGUF 컨텍스트 스윕의 프롬프트 길이는 근사치**입니다 (HF 토크나이저를 그대로 쓰지 못해 단어 반복으로 근사). 정확한 토큰 수가 아닌 "대략 이 정도 길이"로 이해해야 함.
- **단일 실행 기준**입니다 (반복 측정 후 평균/분산을 내지 않음). 엄밀히 하려면 각 설정을 3회 이상 반복해 평균과 표준편차를 리포트하는 게 좋음 — 시간이 허락하면 추가 예정.
- Colab 무료 티어의 GPU(T4/L4)는 세션마다 배정이 달라질 수 있어, 완전히 동일한 하드웨어 재현은 보장되지 않음. 리포트에 실제 배정된 GPU 모델을 명시할 것.

## Next steps

1. Colab에서 `--stage all` 및 `--stage context_sweep` 실행 → `results/results.jsonl` 확보
2. 수치를 이 README와 `report/technical_report_template.md`에 반영
3. GitHub에 push → Zenodo 연동 → 릴리즈 태그 → DOI 발급

## Citation

`CITATION.cff` 참고 (저자명/저장소 주소는 TODO로 표시되어 있으니 채워 넣을 것).

## License

Code: MIT (`LICENSE` 참고). Report/데이터: CC-BY-4.0 권장.
