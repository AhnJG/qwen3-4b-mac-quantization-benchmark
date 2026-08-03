# Efficient AI: On-Device Quantization Benchmark (Qwen3-4B-Instruct)

> **Status: v2 (프로토콜 수정 후 재측정).** 외부 동료 리뷰에서 v1 측정 스크립트의 치명적 버그(이중 prefill, GGUF만 챗 템플릿 미적용)와 실험 설계 결함(GGUF 행에서 양자화 방식과 추론 엔진이 동시에 바뀜)이 지적됐습니다. 코드를 고치고 GGUF F16 설정을 추가해 다시 측정했고, **그 결과 "GGUF가 컨텍스트 확장에 가장 강건하다"는 v1의 핵심 결론이 뒤집혔습니다.** v1 데이터는 `results/*_v1_double_prefill_bug.jsonl`에 감사 기록으로 보존돼 있습니다. CUDA 교차검증과 메모리 압박 실험은 v1 방식으로 측정된 것이라 재검증이 필요하며 잠정 보류 상태입니다 — 자세한 내용은 [Limitations](#limitations), [Next steps](#next-steps) 참고.

## TL;DR

> `Qwen3-4B-Instruct-2507`를 세 축(FP16/NF4 × HF/llama.cpp 엔진, 사실상 GGUF-F16 설정 추가로 2×2에 가까운 격자)으로 비교했다. **핵심 발견 1**: "GGUF가 빠르다"의 대부분은 엔진이 아니라 양자화 때문이다 — 같은 정밀도(F16)에서 llama.cpp는 HF `generate()`보다 겨우 1.13배 빠른데, 같은 엔진 안에서 Q4_K_M으로 양자화하면 2.93배 더 빨라진다. **핵심 발견 2**: bitsandbytes NF4는 반대다 — VRAM은 61% 줄지만 속도는 오히려 2.70배 느려진다. 같은 "4-bit"라는 이름 아래 정반대 결과가 나온 것은, llama.cpp의 K-quant 커널은 추론에 맞춰 손수 최적화된 반면 bitsandbytes NF4는 QLoRA 학습(메모리 절감)을 위해 설계돼 추론 시 디퀀타이즈 오버헤드를 상쇄할 커널 융합이 없기 때문으로 보인다. **핵심 발견 3 (v1 결론 반전)**: 측정 버그(이중 prefill)를 고치고 나니, 컨텍스트가 128→4096 토큰으로 늘어날 때 가장 완만하게 저하되는 건 GGUF Q4_K_M(-56.9%)이 아니라 **GGUF F16(-19.6%)**이었다 — GGUF Q4_K_M은 FP16(-55.9%)·NF4(-52.8%)와 비슷하게 저하됐다. 원인은 아직 규명하지 못했다. 자세한 내용은 [Results](#results), [Context-length scaling](#context-length-scaling) 참고.

## What this is

`Qwen/Qwen3-4B-Instruct-2507` 모델을 네 가지 설정(HF FP16, HF NF4 4-bit, llama.cpp GGUF F16, llama.cpp GGUF Q4_K_M)으로 실행하며 다음을 정량 비교합니다.

- **메모리**: Peak VRAM 사용량 (GB) — HF 경로만 (llama.cpp는 이 측정 API가 없음)
- **속도**: Time-to-first-token(TTFT), 디코딩 초당 토큰 수(tokens/sec)
- **품질**: WikiText-2 (test split) sliding-window perplexity (HF 경로만)

추가로, 입력 길이가 128 → 4096 토큰으로 늘어날 때 각 설정의 속도/메모리가 어떻게 변하는지 측정합니다.

**설계 의도**: GGUF F16을 넣은 건 "양자화 효과"와 "엔진 효과"를 분리하기 위해서입니다. GGUF Q4_K_M을 HF FP16과 직접 비교하면 양자화 방식과 추론 엔진(Python `generate()` 루프 vs C++ 전용 엔진)이 동시에 바뀌어서, 속도 차이가 어느 쪽 때문인지 알 수 없습니다. GGUF F16(같은 엔진, 같은 정밀도)을 기준점으로 추가하면 `GGUF-F16 / HF-FP16`(엔진 효과)과 `GGUF-Q4_K_M / GGUF-F16`(양자화 효과)으로 분리할 수 있습니다.

## Quickstart

**Google Colab**: `benchmark_quantization.ipynb`을 열고 안내를 따르세요. Runtime을 GPU(T4 이상)로 바꿔야 합니다. (⚠️ 현재 Colab/CUDA 수치는 v1 버그가 있던 방식으로 측정된 것이라 재검증 전까지는 참고용으로만 보세요 — [Limitations](#limitations) 참고.)

**로컬 (CUDA GPU)**:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124  # CUDA 버전에 맞게 cu121/cu124 등 선택

python benchmark_quantization.py --stage all
python benchmark_quantization.py --stage context_sweep
```

**로컬 (Apple Silicon, MPS/Metal)** — 이 README의 실측치는 이 경로(Apple M3)로 얻었습니다:
```bash
python3.12 -m venv .venv && source .venv/bin/activate  # bitsandbytes/torch 호환성 때문에 3.12 권장
pip install -r requirements.txt   # bitsandbytes는 pip가 자동으로 Apple Silicon 네이티브(MPS 지원) wheel을 설치함
CMAKE_ARGS="-DGGML_METAL=on" pip install --no-cache-dir --force-reinstall llama-cpp-python  # Metal 가속 빌드

python benchmark_quantization.py --stage all
python benchmark_quantization.py --stage context_sweep
```

개별 스테이지(`fp16`/`nf4`/`gguf`/`gguf_f16`)도 따로 실행할 수 있습니다. 결과는 `results/results.jsonl`에 누적 기록됩니다(각 줄에 `device`, `prompt_tokens`, `n_new`도 포함 — tok/s 계산을 사후 검증할 수 있도록 v2에서 추가).

## Results

측정 환경: Apple M3 (macOS, unified memory) · HF 경로(FP16/NF4)는 PyTorch MPS backend, GGUF 경로(F16/Q4_K_M)는 llama.cpp Metal backend · 각 설정 1회 실행.

| Config | Peak VRAM (GB) | Tokens/sec (decode) | TTFT (s) | Perplexity (WikiText-2) |
|---|---|---|---|---|
| HF FP16 (baseline) | 7.53 | 7.48 | 0.018 | 9.54 |
| HF NF4 4-bit (bitsandbytes) | 2.90 | 2.77 | 0.033 | 9.81 |
| GGUF F16 (llama.cpp) | –** | 8.47 | 0.307 | n/a* |
| GGUF Q4_K_M (llama.cpp) | –** | 24.85 | 0.130 | n/a* |

\* GGUF perplexity는 llama.cpp 자체 구현으로 계산되어 FP16/NF4와 동일한 방법이 아니므로 직접 비교하지 않음 (자세한 내용은 [Limitations](#limitations) 참고).

\*\* llama.cpp는 torch 스타일의 VRAM 측정 API를 노출하지 않아 미측정.

### 엔진 효과 vs 양자화 효과 분리

|  | 계산식 | 배수 | 해석 |
|---|---|---|---|
| 엔진 효과 (같은 F16, 엔진만 다름) | GGUF-F16 / HF-FP16 = 8.47 / 7.48 | **1.13배** | llama.cpp 엔진 자체의 이점은 크지 않음 |
| 양자화 효과 (같은 llama.cpp, 정밀도만 다름) | GGUF-Q4_K_M / GGUF-F16 = 24.85 / 8.47 | **2.93배** | "GGUF가 빠르다"의 실체는 대부분 이것 |
| bitsandbytes 양자화 효과 (같은 HF, 정밀도만 다름) | HF-NF4 / HF-FP16 = 2.77 / 7.48 | **0.37배 (2.70배 느림)** | 정반대 방향 |

같은 "4-bit 양자화"라는 이름 아래 llama.cpp의 K-quant(Q4_K_M)는 확실한 속도 향상을, bitsandbytes NF4는 오히려 속도 손해를 냅니다. bitsandbytes NF4는 원래 QLoRA 학습(메모리 절감)을 위해 설계됐고, 추론 시 디퀀타이즈를 행렬곱과 하나로 묶어 처리하는 커널 융합이 없어서 대역폭 절감분을 오버헤드가 상쇄하는 것으로 보입니다 — llama.cpp의 K-quant는 반대로 추론 전용으로 손수 최적화된 커널입니다(둘 다 커널 단위로 직접 프로파일링해 검증한 건 아니고, 일반적으로 알려진 아키텍처 지식에 기반한 추정입니다).

VRAM 절감은 61%(7.53GB → 2.90GB, HF NF4 기준)로 확실했고, perplexity 저하도 +2.8%(9.54 → 9.81)로 미미했습니다 — "VRAM은 확실히 줄지만 속도는 대가를 치른다"가 bitsandbytes NF4에 대한 정확한 요약입니다.

### Context-length scaling

프롬프트 길이 128 → 4096 토큰에서 각 설정의 디코딩 tokens/sec:

| Context (tokens) | HF FP16 | HF NF4 | GGUF F16 | GGUF Q4_K_M |
|---|---|---|---|---|
| 128 | 9.15 | 3.44 | 9.25 | 31.27 |
| 512 | 8.23 | 3.16 | 9.25 | 23.52 |
| 1024 | 6.73 | 2.28 | 9.87 | 22.09 |
| 2048 | 2.56 | 2.43 | 9.61 | 19.25 |
| 4096 | 4.04 | 1.62 | 7.44 | 13.48 |

저하폭(128→4096): **GGUF F16 -19.6%**, NF4 -52.8%, HF FP16 -55.9%, GGUF Q4_K_M -56.9%.

⚠️ **v1(이 리포트의 이전 버전)에서는 GGUF Q4_K_M이 가장 완만하게 저하된다(-41%)고 결론 내렸는데, 이는 측정 버그였습니다.** v1의 `measure_hf_generate`는 TTFT와 처리량을 재려고 `model.generate()`를 두 번 호출했는데, 두 번째 호출이 KV 캐시 없이 프롬프트를 처음부터 다시 prefill했습니다 — 그래서 HF 경로의 "처리량"에는 컨텍스트 길이만큼의 prefill 시간이 몰래 섞여 있었습니다. 반면 GGUF 경로는 같은 `llm` 인스턴스에 같은 프롬프트로 두 번 호출해서 llama.cpp의 접두사 캐시가 두 번째 호출의 prefill을 건너뛰었습니다 — 그래서 GGUF만 "순수 디코딩" 속도가 재졌고, 나머지는 "prefill+디코딩 뒤섞인" 속도가 재졌습니다. 외부 동료 리뷰가 TTFT(1토큰 생성 시간)가 전체 생성 시간보다 긴 경우(물리적으로 불가능)를 근거로 이 버그를 잡아냈습니다. v2에서는 단일 `generate()`/스트리밍 호출 + 토큰별 타임스탬프로 재작성했고(워밍업도 추가), 그 결과가 위 표입니다.

**v2에서 새로 드러난 점**: GGUF Q4_K_M은 더 이상 가장 강건하지 않고 FP16/NF4와 비슷하게 저하됩니다. 대신 GGUF F16이 가장 완만합니다 — 이건 예상 밖의 결과이고, 원인을 아직 설명하지 못합니다(다음 조사 대상: KV 캐시 크기 자체가 아니라 GGUF의 배치/스케줄링 방식이 F16에서 유독 컨텍스트 성장에 덜 민감한 것인지). HF FP16은 ctx=2048에서 저점(2.56 tok/s)을 찍고 4096에서 반등(4.04)하는 비단조 곡선을 보이는데, 반복 측정 없이는 노이즈인지 실제 현상인지 판단할 수 없습니다.

같은 구간의 Peak VRAM(GB, HF 경로만 — `generate()` 실행 중 20ms 간격 폴링으로 근사, 자세한 내용은 [Limitations](#limitations) 참고):

| Context (tokens) | HF FP16 | HF NF4 |
|---|---|---|
| 128 | 7.537 | 2.878 |
| 512 | 7.740 | 3.129 |
| 1024 | 7.963 | 3.168 |
| 2048 | 8.079 | 3.296 |
| 4096 | 8.330 | 3.946 |

FP16 절대 증가량 +0.79GB, NF4 +1.07GB — NF4가 더 큽니다. NF4의 compute dtype도 FP16과 같은 float16이라 KV 캐시 자체의 바이트당 크기는 이론상 비슷해야 하는데, 원인은 v1 때부터 특정하지 못한 채로 남아 있습니다.

## Related benchmarks (sanity check)

이 프로젝트와 똑같은 조합(FP16 vs bitsandbytes-NF4 vs GGUF, 이 모델 기준)을 그대로 측정한 공개 자료는 찾지 못했지만, 양자화의 일반적인 트레이드오프는 이미 여러 곳에 보고돼 있습니다.

- Unsloth GGUF Benchmarks — Q4_K_M vs Q5_K_M perplexity 비교
- 커뮤니티 보고: Q4_K_M은 FP16 대비 용량 약 75% 감소, MMLU 등 품질 저하는 1~3% 수준

우리 perplexity 저하폭(NF4 +2.8%)은 이 범위 안에 들어와 방향이 일치합니다. 다만 지표(WikiText-2 언어모델링 손실 vs MMLU 정확도)와 양자화 방식(bitsandbytes NF4 vs GGUF Q4_K_M)이 서로 달라 엄밀한 검증은 아닙니다.

## Limitations

- **GGUF perplexity는 FP16/NF4와 다른 방식으로 계산됨** (llama.cpp 내부 구현 vs HuggingFace loss 기반). 숫자를 나란히 두되 "직접 비교 불가"임을 명시.
- **단일 실행 기준**입니다 (반복 측정 후 평균/분산을 내지 않음). HF FP16의 ctx=2048/4096 간 비단조 구간처럼, 노이즈인지 실제 현상인지 구분이 안 되는 지점이 있습니다.
- **CUDA 교차검증과 메모리 압박(스왑) 실험은 v1 버그가 있던 방식으로 측정됐습니다.** `results/results_cuda_t4_colab_v1_double_prefill_bug.jsonl`에 원본 데이터가 남아있지만, 이중 prefill 버그가 있던 스크립트로 측정한 것이므로 결론(특히 컨텍스트 스케일링 관련 부분)을 그대로 신뢰할 수 없습니다. 재검증 전까지 이 README에서 제외했습니다 — [Next steps](#next-steps) 참고.
- **MPS에서의 Peak VRAM은 폴링 기반 근사치**입니다. PyTorch의 MPS backend는 CUDA의 `reset_peak_memory_stats()`/`max_memory_allocated()`에 대응하는 reset 가능한 peak 카운터를 제공하지 않습니다. `generate()` 실행 중 백그라운드 스레드가 20ms 간격으로 `torch.mps.current_allocated_memory()`를 폴링해 최댓값을 추적하는 방식(`PeakMemoryTracker`)으로 근사했으며, 폴링 간격보다 짧게 스파이크하는 할당은 놓칠 수 있습니다.
- **GGUF 경로는 매 컨텍스트 길이마다 워밍업 후 `llm.reset()`으로 캐시를 비우고 측정**합니다 — 즉 매번 진짜 prefill을 다시 겪습니다. 이건 실사용(멀티턴 대화에서 접두사가 재사용되는 상황)보다 더 보수적인(불리한) 조건입니다.
- **GGUF F16이 컨텍스트 확장에 가장 강건한 이유는 규명하지 못했습니다** — 위 [Context-length scaling](#context-length-scaling) 참고.
- **NF4의 절대 VRAM 증가량이 FP16보다 큰 이유도 v1부터 미규명 상태입니다.**

## Next steps

1. ~~Apple M3(MPS/Metal)에서 `--stage all` 및 `--stage context_sweep` 실행~~ (v1, 완료 — 버그 있는 방식이었음이 이후 드러남)
2. ~~외부 동료 리뷰 2건 수령, 이중 prefill 버그 및 실험 설계 결함(엔진×양자화 교락) 확인~~ (완료)
3. ~~측정 스크립트 재작성(단일 호출+토큰 타임스탬프, 워밍업, GGUF도 챗 템플릿 적용) + GGUF F16 설정 추가 + Apple M3에서 전면 재측정~~ (완료 — 이 README가 그 결과)
4. 메모리 압박을 "관측 변수"가 아니라 "조작 변수"로 바꾸는 실험 — ballast 프로세스로 0/4/8/12/16GB 강제 점유 + `vm_stat` 동시 로깅 + `use_mmap=False` 대조군. v1의 "Safari를 켜놨더니 스왑으로 붕괴" 관찰은 흥미롭지만 재현 불가능한 일화였음.
5. v2 프로토콜로 Colab(CUDA T4)에서 재교차검증
6. GGUF F16이 컨텍스트 확장에 가장 강건한 이유 규명 (반복 측정, KV 캐시/스케줄링 프로파일링)
7. 최소 3회 반복 측정 + 표준편차 보고
8. GitHub에 push → Zenodo 연동 → 릴리즈 태그 → DOI 발급 (위 항목들 마무리 후)

## Citation

`CITATION.cff` 참고 (저자명/저장소 주소는 TODO로 표시되어 있으니 채워 넣을 것).

## License

Code: MIT (`LICENSE` 참고). Report/데이터: CC-BY-4.0 권장.
