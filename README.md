# Efficient AI: On-Device Quantization Benchmark (Qwen3-4B-Instruct)

> **Status: v3.** 두 번의 외부 동료 리뷰를 거치며 측정 코드의 버그를 두 번 잡았습니다. v1은 이중 `generate()` 호출로 prefill이 처리량에 섞였고, v2는 그 "수정"이 다시 틀렸습니다(MPS 비동기 실행 때문에 `time.time()`이 GPU 완료가 아니라 CPU 도달 시각을 찍어, prefill이 또 decode에 누출). v3에서 `sync_device()`로 실측 검증까지 마쳐 고쳤고, 최소 3회 반복 + 무작위 순서로 재측정했습니다. **핵심 결론("GGUF F16이 컨텍스트 확장에 가장 완만")은 버그 수정 후에도 유지**됐지만, 그 해석은 바뀌었습니다(아래). 이 저장소는 실측 하드웨어(Apple M3)가 장시간 벤치마크로 스왑이 누적된 상태에서 측정됐다는 한계가 있으며, 이는 HF 경로의 장문 수치에 큰 변동으로 나타납니다 — [Limitations](#limitations) 참고. 이전 버전의 데이터는 `results/*_v1_*.jsonl`, `results/*_v2_*.jsonl`에 감사 기록으로 보존.

## TL;DR

> `Qwen3-4B-Instruct-2507`을 2×2 격자(HF/llama.cpp 엔진 × FP16/4-bit)로 비교했다. **엔진 vs 양자화 분리** (ctx=128, 3회 중앙값): 같은 정밀도에서 llama.cpp는 HF `generate()`보다 1.12배 빠른 데 그치지만(엔진 효과), 같은 llama.cpp 안에서 Q4_K_M 양자화는 2.89배를 더 낸다 — "GGUF가 빠르다"의 실체는 대부분 양자화지 엔진이 아니다. 반대로 **bitsandbytes NF4는 4-bit인데도 FP16보다 2.67배 느리다**(0.375배 속도) — VRAM은 62% 줄지만(7.53→2.88GB) 속도는 대가를 치른다. 품질 저하는 NF4 perplexity +2.8%로 미미. **컨텍스트 스케일링(128→4096)**: 저하폭 %로는 GGUF F16(-14%)이 가장 완만하지만, 이건 강건성이 아니라 착시다 — 토큰당 절대 지연 증가는 GGUF F16(+16ms)과 Q4_K_M(+18ms)이 거의 같고, F16이 baseline이 느려 같은 손해가 작은 %로 보일 뿐이다. HF 경로(FP16 -49%, NF4 -45%)는 절대 저하도 크고(각 +108ms/+249ms) 반복 간 변동이 극심한데(FP16 @4096이 1.48~4.71 tok/s), 이는 스왑 포화 상태에서 상주 메모리가 경쟁하기 때문이다 — GGUF(mmap)는 같은 조건에서 안정적이다(F16 @4096: 8.42~8.55). 자세한 내용은 [Results](#results), [Context-length scaling](#context-length-scaling) 참고.

## What this is

`Qwen/Qwen3-4B-Instruct-2507`을 네 설정으로 실행해 정량 비교합니다. GGUF F16을 넣은 이유는 "엔진 효과"와 "양자화 효과"를 분리하기 위해서입니다 — 그러지 않으면 "GGUF Q4_K_M vs HF FP16"에서 엔진과 정밀도가 동시에 바뀌어 원인을 가를 수 없습니다.

|  | HF transformers 엔진 | llama.cpp 엔진 |
|---|---|---|
| **FP16/F16 정밀도** | HF FP16 (baseline) | GGUF F16 |
| **4-bit 양자화** | HF NF4 (bitsandbytes) | GGUF Q4_K_M |

- 행 = 엔진 효과 (`GGUF-F16 / HF-FP16`), 열 = 양자화 효과 (`GGUF-Q4_K_M / GGUF-F16`)
- 지표: Peak VRAM(HF만), decode tok/s + TTFT, WikiText-2 perplexity(HF만)
- 컨텍스트 128→4096 토큰 스윕

## Quickstart

**로컬 (Apple Silicon, MPS/Metal)** — 이 README의 실측치는 이 경로(Apple M3):
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
CMAKE_ARGS="-DGGML_METAL=on" pip install --no-cache-dir --force-reinstall llama-cpp-python

python benchmark_quantization.py --stage all           --repeats 3   # 4설정 헤드라인
python benchmark_quantization.py --stage context_sweep --repeats 3   # 128~4096 스윕
python benchmark_quantization.py --stage perplexity                  # HF FP16/NF4 perplexity
```

**로컬 (CUDA GPU)**: `requirements.txt` 설치 후 `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124`. (⚠️ CUDA 교차검증은 아직 v3 프로토콜로 재측정 전입니다 — [Limitations](#limitations) 참고.)

결과는 `results/results.jsonl`에 기록됩니다(각 줄에 `device`/`prompt_tokens`/`n_new`/`repeat`/`max_new_tokens` — 사후 검증용). 메모리 압박 실험 도구는 `tools/ballast.py`, `tools/pressure_probe.py`.

## Results

측정: Apple M3 (24GB unified memory) · HF는 PyTorch MPS, GGUF는 llama.cpp Metal · 아래는 컨텍스트 128 토큰, **3회 반복의 중앙값**(단발 short-prompt는 반복 간 변동이 커서 헤드라인 기준점으로 부적합).

| Config | Peak VRAM (GB) | decode tok/s | TTFT (s) | Perplexity |
|---|---|---|---|---|
| HF FP16 (baseline) | 7.54 | 8.78 | 0.50 | 9.54 |
| HF NF4 (bitsandbytes) | 2.88 | 3.29 | 0.70 | 9.81 |
| GGUF F16 (llama.cpp) | –\* | 9.80 | 0.45 | n/a\*\* |
| GGUF Q4_K_M (llama.cpp) | –\* | 28.34 | 0.49 | n/a\*\* |

\* llama.cpp는 torch식 VRAM API가 없어 미측정. \*\* GGUF perplexity는 llama.cpp 자체 구현이라 HF loss 기반과 직접 비교 불가.

### 엔진 효과 vs 양자화 효과

|  | 계산식 | 배수 |
|---|---|---|
| 엔진 (F16 고정, 엔진만) | GGUF-F16 / HF-FP16 = 9.80 / 8.78 | **1.12배** |
| 양자화 (llama.cpp 고정, 정밀도만) | GGUF-Q4_K_M / GGUF-F16 = 28.34 / 9.80 | **2.89배** |
| bitsandbytes 양자화 (HF 고정, 정밀도만) | HF-NF4 / HF-FP16 = 3.29 / 8.78 | **0.375배 (2.67배 느림)** |

"GGUF가 FP16보다 3.23배 빠르다"(28.34/8.78)는 사실이지만 요인을 분리하면 대부분(2.89배)이 K-quant 양자화이고 엔진 자체 기여는 1.12배에 불과합니다. 같은 "4-bit"인데 llama.cpp의 K-quant는 2.89배 빨라지고 bitsandbytes NF4는 2.67배 느려지는 정반대 결과는, K-quant가 추론 전용으로 디퀀타이즈+행렬곱을 융합한 커널인 반면 NF4는 QLoRA 학습의 메모리 절감이 주목적이라 추론 시 디퀀타이즈 오버헤드가 대역폭 절감분을 상쇄하기 때문으로 보입니다(커널 단위 프로파일링으로 직접 검증한 건 아닌 아키텍처 지식 기반 추정).

VRAM 절감은 62%(7.54→2.88GB, HF NF4), perplexity 저하는 +2.8%(9.54→9.81)로 미미 — "메모리는 확실히 줄지만 속도는 대가를 치른다"가 bitsandbytes NF4의 요약입니다.

### Context-length scaling

프롬프트 128→4096 토큰의 decode tok/s. **3회 반복 중앙값 [최소–최대]**:

| ctx | HF FP16 | HF NF4 | GGUF F16 | GGUF Q4_K_M |
|---|---|---|---|---|
| 128 | 8.78 [8.72–8.81] | 3.29 [3.29–3.33] | 9.80 [9.34–11.27] | 28.34 [28.20–29.15] |
| 512 | 6.82 [5.51–8.27] | 3.25 [3.08–3.27] | 9.94 [9.62–11.07] | 27.61 [26.90–27.78] |
| 1024 | 7.84 [6.38–8.80] | 3.12 [2.71–3.17] | 9.73 [9.58–10.27] | 25.94 [25.41–25.99] |
| 2048 | 1.91 [1.57–1.99] | 3.05 [3.04–3.06] | 9.24 [9.23–9.29] | 23.22 [16.31–23.23] |
| 4096 | 4.51 [1.48–4.71] | 1.81 [0.03–2.51] | 8.44 [8.42–8.55] | 18.85 [16.91–19.26] |

TTFT(prefill)는 이제 네 설정 모두 컨텍스트에 거의 선형으로 비례해 4096에서 15~18초로 수렴합니다(C-1 수정 검증 — 버그 있던 v2는 HF TTFT를 0.03초로 보고했는데 이는 4104토큰 prefill로선 물리적으로 불가능).

**저하폭 %는 오해를 부릅니다.** 128→4096 중앙값 저하: GGUF F16 -14%, GGUF Q4_K_M -33%, NF4 -45%, FP16 -49%. 순서만 보면 "GGUF F16이 가장 강건"이지만, **토큰당 절대 지연 증가로 보면 그렇지 않습니다**:

| Config | ms/token 128→4096 | 절대 증가 |
|---|---|---|
| GGUF Q4_K_M | 35.3 → 53.0 | **+17.8 ms** |
| GGUF F16 | 102.1 → 118.5 | **+16.4 ms** |
| HF FP16 | 113.9 → 221.9 | +108.0 ms |
| HF NF4 | 303.6 → 552.1 | +248.5 ms |

GGUF F16과 Q4_K_M은 컨텍스트가 커질 때 **거의 같은 절대 비용(+16 vs +18 ms/token)**을 추가합니다 — F16이 -14%로 완만해 보이는 건 baseline이 느려서(102ms) 같은 손해가 작은 %로 나올 뿐, 실제로 더 강건한 게 아닙니다. 이건 대역폭 바운드 디코딩의 roofline으로 예상되는 결과입니다(토큰당 읽는 바이트 = 가중치 + KV 캐시; KV 증가는 정밀도와 무관하게 거의 같으므로 절대 지연 증가도 비슷). "GGUF F16이 왜 가장 강건한가"는 미스터리가 아니라 % 지표의 함정이었습니다.

**HF 경로의 큰 변동이 진짜 발견입니다.** FP16 @4096은 3회 반복에서 1.48~4.71 tok/s로 3배 넘게 흔들리고, NF4 @4096은 한 번 0.03 tok/s로 사실상 붕괴했습니다. 반면 GGUF는 안정적입니다(F16 @4096: 8.42~8.55). FP16 @2048 중앙값(1.91)이 @4096(4.51)보다 낮은 비단조도 이 변동 때문입니다. 원인은 이 머신이 측정 당시 스왑 포화 상태(8~18GB)였고, HF는 모델·활성화 텐서를 상주 메모리에 두어 다른 프로세스와 경쟁한 반면 GGUF는 mmap으로 페이지 단위 적재라 덜 취약했기 때문으로 보입니다. **즉 메모리 압박 현상이 인위적 ballast 없이 메인 스윕에서 자연스럽게 드러났습니다** — 이는 [별도 ballast 실험](#memory-pressure-ballast-실험)이 겨냥했던 바로 그 효과입니다.

같은 구간 Peak VRAM(GB, HF만, 중앙값): FP16 128:7.54 → 4096:8.22 (+0.68), NF4 128:2.88 → 4096:4.03 (+1.15). NF4 절대 증가가 FP16보다 큰 이유는 v1부터 미규명이며, MPS 폴링 근사의 노이즈일 가능성이 있습니다([Limitations](#limitations)).

## Memory pressure (ballast 실험)

v1은 "다른 앱이 켜져 있어 스왑으로 붕괴"라는 재현 불가 일화로 "GGUF는 mmap 덕분에 강건"을 주장했습니다. v2에서 이를 통제 실험으로 만들었지만(ballast 프로세스로 메모리 강제 점유 + mmap on/off 대조), 두 가지 결함이 드러났습니다: (a) `ballast.py`가 초기 fill 후 64MB 창만 재기록하는 순회 버그로 나머지 ~99%가 스왑아웃돼 실제 압박을 유지하지 못했고, (b) 0GB 기준선이 이미 스왑 30GB로 오염돼 "유휴 vs 압박"이 아니라 "압박 vs 더-압박"이었습니다.

v3에서 도구는 완전히 고쳤습니다:
- `ballast.py` 순회 버그 수정(전 페이지를 매 주기 터치해 상주 유지 — 3GB 요청 시 RSS가 축소 없이 유지됨을 확인)
- `pressure_probe.py`에 **GGUF F16 추가**(8GB 가중치로 압박에 가장 취약한 후보), **config 순서 무작위화**(순서 교락 제거), **매 측정 시점의 실제 스왑 사용량 기록**(압박이 실제로 걸렸는지 조작검증), 다단계 ballast(0/8/16GB 양성 대조), 그리고 v3의 수정된 측정 경로(`measure_one`) 사용

**그러나 통제된 재실행은 보류했습니다.** 깨끗한 0GB 기준선은 재부팅이 필요한데(이 세션은 수 시간 벤치마크로 스왑이 8~18GB 누적), 재부팅은 사용자 조작입니다. 이미 압박된 머신에서 ballast를 더 얹으면 v2의 "압박 vs 더-압박" 문제를 그대로 반복할 뿐입니다. 대신 위 [Context-length scaling](#context-length-scaling)의 반복 spread가 같은 효과(HF는 장문에서 노이즈 지배, GGUF는 안정)를 ballast 없이 더 깨끗하게 보여주므로 이를 1차 근거로 삼았습니다. 재부팅 후 `tools/pressure_probe.py --ballast-gb 0,8,16` 실행이 남은 확정 단계입니다.

## Related benchmarks (sanity check)

- Unsloth GGUF Benchmarks — Q4_K_M vs Q5_K_M perplexity
- 커뮤니티: Q4_K_M ≈ FP16 대비 용량 75% 감소, 품질 저하 1~3%(MMLU)

우리 perplexity 저하(NF4 +2.8%)는 이 범위와 방향이 일치하나, 지표(WikiText-2 언어모델링 손실 vs MMLU 정확도)·양자화 방식(NF4 vs GGUF)이 달라 엄밀한 검증은 아닙니다.

## Limitations

- **측정 머신이 스왑 포화 상태였습니다** (8~18GB). HF(상주 메모리) 장문 수치는 반복 간 변동이 극심하므로(FP16 @4096: 1.48~4.71) 중앙값+범위로 보고했습니다. 깨끗한 측정은 재부팅이 필요하며, GGUF·단문 수치와 엔진/양자화 비율은 상대적으로 안정적입니다.
- **3회 반복 기준**입니다. 변동이 큰 셀은 반복을 더 늘려야 신뢰구간이 좁아집니다.
- **CUDA 교차검증은 v1 버그 방식으로 측정된 것이라 제외**했습니다(`results/results_cuda_t4_colab_v1_double_prefill_bug.jsonl`). C-1은 CUDA도 비동기라 동일하게 발현하므로, v3 프로토콜로 재측정해야 유효합니다.
- **ballast 통제 실험은 도구만 완비**됐고 실행은 재부팅 후로 보류(위 참고).
- **MPS Peak VRAM은 20ms 폴링 근사**로 짧은 스파이크를 놓칠 수 있습니다. NF4의 절대 VRAM 증가가 FP16보다 큰 원인은 미규명이며, 폴링 노이즈일 가능성이 있습니다.
- **GGUF 경로는 매 측정 워밍업 후 `llm.reset()`으로 캐시를 비워** 매번 완전한 prefill을 강제합니다 — 접두사가 재사용되는 실사용 멀티턴 대화보다 보수적(불리한) 조건입니다.
- **속도 지표는 커널 프로파일링이 아닌 end-to-end 측정**입니다. NF4가 느린 원인(디퀀타이즈 커널 융합 부재)은 아키텍처 지식 기반 추정이며 직접 프로파일링은 안 했습니다.

## Next steps

1. ~~C-1(async TTFT 누출)·M-1(GGUF 청크 카운트)·D-8 수정~~ (완료 — 실측 검증)
2. ~~반복+무작위 순서+per-shape 워밍업 프로토콜로 context_sweep 재측정~~ (완료 — 이 README)
3. ~~ballast.py 순회 버그 수정 + pressure_probe 재설계(F16/mmap/무작위/조작검증)~~ (완료, 도구 검증됨)
4. **재부팅 후** 깨끗한 상태에서 (a) context_sweep 재측정(HF 장문 변동 제거) (b) `pressure_probe --ballast-gb 0,8,16` 실행
5. v3 프로토콜로 CUDA(Colab T4) 재교차검증
6. NF4 디퀀타이즈 오버헤드 커널 프로파일링, NF4 VRAM 증가 원인 규명(폴링 간격 축소 대조)
7. GitHub push → Zenodo → DOI (위 마무리 후)

## Citation

`CITATION.cff` 참고 (저자명/저장소 TODO).

## License

Code: MIT (`LICENSE`). Report/데이터: CC-BY-4.0 권장.
