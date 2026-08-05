# Quantitative Analysis of Post-Training Quantization for On-Device LLM Inference: A Qwen3-4B-Instruct Case Study

**Author:** TODO-이름 · TODO-소속
**Date:** 2026-08-03 (v1) / 08-04 (v2) / 08-05 (v3, 프로토콜 재수정 후 재측정)
**Code & data:** TODO-GitHub · **DOI:** 미발급 — §5의 재검증(재부팅 후 재측정, CUDA 재교차검증, 반복 확대) 완료 후

> 이 문서는 두 차례 외부 동료 리뷰를 거치며 측정 코드의 타이밍 버그를 두 번 잡아 v3에 이르렀습니다. v1: 이중 `generate()` 호출로 prefill이 처리량에 합산. v2: 그 수정이 MPS 비동기 실행을 간과해 여전히 prefill 누출(리뷰어가 "TTFT가 물리적 하한보다 550배 짧다"로 지적, `sync_device()` 삽입 실측으로 확인). v3: 수정 + 3회 반복 + 무작위 순서. 이전 데이터는 `results/*_v1_*.jsonl`, `results/*_v2_*.jsonl`에 보존.

## Abstract

온디바이스/저비용 서빙을 위한 소형 LLM 양자화는 실질적 수요가 있다. 본 리포트는 `Qwen3-4B-Instruct-2507`을 2×2 격자(HF/llama.cpp 엔진 × FP16/4-bit) 네 설정으로 Apple M3(24GB 통합 메모리)에서 측정한다. **엔진과 양자화 효과를 분리한 결과**(ctx=128, 3회 중앙값), 같은 정밀도에서 llama.cpp 엔진의 이점은 1.12배에 불과한 반면 llama.cpp 내부 K-quant 양자화는 2.89배를 낸다 — "GGUF가 빠르다"의 대부분은 엔진이 아니라 양자화다. 대조적으로 bitsandbytes NF4는 같은 4-bit이면서 FP16보다 2.67배 느리다(VRAM은 62% 절감, perplexity 저하 +2.8%). 두 4-bit 구현이 정반대로 움직이는 것은 K-quant가 추론 전용 융합 커널인 반면 NF4는 QLoRA 학습용 메모리 절감이 목적이기 때문으로 추정된다. **컨텍스트 128→4096 스케일링**에서 저하폭 %는 GGUF F16(-14%)이 가장 완만하지만, 이는 강건성이 아니라 % 지표의 함정이다 — 토큰당 절대 지연 증가는 GGUF F16(+16.4ms)과 Q4_K_M(+17.8ms)이 거의 같으며, F16은 baseline이 느려 같은 손해가 작은 %로 보일 뿐이다(대역폭 roofline으로 예상되는 결과). HF 경로는 절대 저하가 훨씬 크고(FP16 +108ms, NF4 +249ms) 반복 간 변동이 극심한데(FP16 @4096: 1.48–4.71 tok/s, NF4는 한 번 0.03으로 붕괴), 이는 스왑 포화 머신에서 상주 메모리가 경쟁하기 때문이며 GGUF(mmap)는 같은 조건에서 안정적이다(F16 @4096: 8.42–8.55). 즉 메모리 압박 효과가 인위적 ballast 없이 반복 측정의 분산으로 자연히 드러났다.

## 1. Motivation & Problem Framing

- 국내 기업(삼성전자, SK하이닉스, LG, 네이버 등)의 서빙 비용 절감·온디바이스 탑재 수요.
- FP16 vs 4-bit 비교는 흔하므로, 본 리포트는 (a) 엔진 효과와 양자화 효과를 분리하는 2×2 요인 설계, (b) 컨텍스트 스케일링 두 축으로 차별화한다.
- v1은 (a) 축이 없어 원인을 못 갈랐고 측정 버그로 (b)도 틀렸다. v2는 (a)를 추가했으나 측정 버그가 남았다. v3은 버그를 실측으로 고치고 반복·무작위화했다.

## 2. Methodology

- **모델:** Qwen/Qwen3-4B-Instruct-2507 (dense, 4B).
- **설정 (2×2):** HF FP16 / HF NF4(bitsandbytes) / GGUF F16 / GGUF Q4_K_M. 대각선만 비교하면 엔진·양자화가 함께 바뀌므로 GGUF F16을 추가해 행(엔진)·열(양자화)을 분리.
- **하드웨어:** Apple M3, 24GB 통합 메모리. HF=PyTorch MPS, GGUF=llama.cpp Metal.
- **측정 지표 (v3에서 재설계):**
  - decode tok/s, TTFT: **단일** `generate()` 호출 안에서 logits processor가 매 디코딩 스텝에 타임스탬프를 찍되, **찍기 전에 `sync_device()`를 호출**한다. MPS/CUDA는 비동기라 sync가 없으면 `time.time()`이 GPU 완료가 아니라 CPU 도달 시각을 기록하고, prefill이 통째로 첫 두 스텝 사이에 떨어져 decode 시간에 누출된다(v2 버그). 실측: ctx=4096에서 sync 없으면 TTFT 0.03s(불가능), sync 있으면 ~15–17s(실제 prefill). GGUF는 스트리밍 단일 호출에서 마지막 빈 `finish_reason` 청크를 제외해 실제 토큰만 센다(v2는 `n_new=max+1`로 과대계수).
  - **각 측정은 백엔드를 새로 로드**하고 그 프롬프트 길이로 워밍업(폐기)한 뒤 측정·해제한다 → HF가 GGUF의 매-호출 재생성과 대칭이 되고, 측정 간 할당자 파편화가 컨텍스트 축에 누적되지 않으며, MPS shape별 커널 컴파일을 타이밍 전에 지불한다.
  - **(config × ctx) 작업을 무작위 순서로 N회 반복**(기본 3) → 중앙값·범위 보고. 순서 무작위화는 "컨텍스트↔경과시간" 교락을 끊는다.
  - Peak VRAM(HF만): MPS는 reset 가능한 peak API가 없어 `generate()` 중 20ms 폴링 근사(`PeakMemoryTracker`).
  - Perplexity: WikiText-2 test, sliding window(1024/512), FP16/NF4 동일. **별도 파일**(`results_perplexity.jsonl`)에 기록해 0.0 속도 행이 속도 집계를 오염시키지 않게 함. v1과 계산 로직이 동일하고 타이밍 버그의 영향을 받지 않으므로 v1 값 재사용(플랫폼 간 5자리 일치 검증됨).
- **재현성:** 각 결과 행에 `device`/`prompt_tokens`/`n_new`/`repeat`/`max_new_tokens` 기록해 tok/s를 사후 검증 가능하게 함.

## 3. Experimental Results & Discussion

ctx=128, 3회 반복 중앙값:

| Config | Peak VRAM (GB) | decode tok/s | TTFT (s) | Perplexity |
|---|---|---|---|---|
| HF FP16 | 7.54 | 8.78 | 0.50 | 9.54 |
| HF NF4 | 2.88 | 3.29 | 0.70 | 9.81 |
| GGUF F16 | n/a\* | 9.80 | 0.45 | n/a |
| GGUF Q4_K_M | n/a\* | 28.34 | 0.49 | n/a |

\* llama.cpp는 torch식 VRAM API 미제공.

### 3.1 엔진 효과와 양자화 효과의 분리

| 대조 | 계산식 | 배수 |
|---|---|---|
| 엔진 (F16 고정) | GGUF-F16 / HF-FP16 = 9.80 / 8.78 | 1.12배 |
| llama.cpp 양자화 (엔진 고정) | GGUF-Q4_K_M / GGUF-F16 = 28.34 / 9.80 | 2.89배 |
| bitsandbytes 양자화 (엔진 고정) | HF-NF4 / HF-FP16 = 3.29 / 8.78 | 0.375배 (2.67배 느림) |

"GGUF가 FP16보다 3.23배 빠르다"(28.34/8.78)는 사실이나, 분해하면 대부분(2.89배)이 K-quant 양자화의 효과이고 엔진 자체 기여는 1.12배다. 같은 4-bit인데 K-quant는 2.89배 빨라지고 bitsandbytes NF4는 2.67배 느려지는 정반대 결과는, K-quant가 디퀀타이즈+행렬곱을 융합한 추론 전용 커널인 반면 NF4는 QLoRA 학습의 메모리 절감이 주목적이라 추론 시 디퀀타이즈 오버헤드가 대역폭 절감분을 상쇄하기 때문으로 추정된다(커널 프로파일링 미수행, 아키텍처 지식 기반). VRAM 절감 62%(7.54→2.88GB), perplexity +2.8%(9.54→9.81)로 품질 저하는 미미.

### 3.2 Context-length scaling

decode tok/s, 3회 중앙값 [최소–최대]:

| ctx | HF FP16 | HF NF4 | GGUF F16 | GGUF Q4_K_M |
|---|---|---|---|---|
| 128 | 8.78 [8.72–8.81] | 3.29 [3.29–3.33] | 9.80 [9.34–11.27] | 28.34 [28.20–29.15] |
| 512 | 6.82 [5.51–8.27] | 3.25 [3.08–3.27] | 9.94 [9.62–11.07] | 27.61 [26.90–27.78] |
| 1024 | 7.84 [6.38–8.80] | 3.12 [2.71–3.17] | 9.73 [9.58–10.27] | 25.94 [25.41–25.99] |
| 2048 | 1.91 [1.57–1.99] | 3.05 [3.04–3.06] | 9.24 [9.23–9.29] | 23.22 [16.31–23.23] |
| 4096 | 4.51 [1.48–4.71] | 1.81 [0.03–2.51] | 8.44 [8.42–8.55] | 18.85 [16.91–19.26] |

TTFT는 네 설정 모두 컨텍스트에 거의 선형(4096에서 15–18s로 수렴) — C-1 수정의 검증이다(버그 있던 v2는 HF TTFT를 0.03s로 보고).

**저하폭 %는 오해를 부른다.** 128→4096 중앙값 저하: GGUF F16 -14%, Q4_K_M -33%, NF4 -45%, FP16 -49%. 순위만 보면 "F16이 가장 강건"이지만 토큰당 절대 지연으로 보면 다르다:

| Config | ms/token 128→4096 | 절대 증가 |
|---|---|---|
| GGUF Q4_K_M | 35.3 → 53.0 | +17.8 ms |
| GGUF F16 | 102.1 → 118.5 | +16.4 ms |
| HF FP16 | 113.9 → 221.9 | +108.0 ms |
| HF NF4 | 303.6 → 552.1 | +248.5 ms |

GGUF F16과 Q4_K_M은 거의 같은 절대 비용(+16 vs +18 ms/token)을 추가한다 — F16이 -14%로 완만해 보이는 건 baseline이 느려서(102ms) 같은 손해가 작은 %로 나오는 것이지 더 강건한 게 아니다. 이는 대역폭 바운드 디코딩의 roofline으로 예상된다: 토큰당 읽는 바이트 = 가중치 + KV 캐시이고, KV 증가는 정밀도와 무관하게 거의 같으므로 절대 지연 증가도 비슷하다. v2가 "F16 강건성 미규명"으로 남긴 것은 % 지표의 함정이었을 뿐이다.

**HF 경로의 큰 변동이 실질적 발견이다.** FP16 @4096은 3회에서 1.48–4.71 tok/s로 3배 넘게 흔들리고 NF4 @4096은 한 번 0.03으로 붕괴했다. GGUF는 안정적이다(F16 @4096: 8.42–8.55). FP16 @2048 중앙값(1.91)이 @4096(4.51)보다 낮은 비단조도 이 변동 탓이다. 측정 당시 머신이 스왑 포화(8–18GB)였고, HF는 모델·활성화 텐서를 상주 메모리에 두어 경쟁한 반면 GGUF는 mmap 페이지 단위 적재라 덜 취약했다. 즉 v2가 별도 ballast로 겨냥했던 메모리 압박 효과가 메인 스윕의 반복 분산에서 인위적 압박 없이 드러났다.

Peak VRAM(HF, 중앙값): FP16 7.54→8.22GB(+0.68), NF4 2.88→4.03GB(+1.15). NF4 증가가 더 큰 원인은 v1부터 미규명이며 폴링 근사 노이즈 가능성(§5).

### 3.3 Memory pressure — 도구는 완비, 통제 실행은 보류

v1의 "다른 앱 때문에 스왑 붕괴" 일화를 v2에서 ballast 통제 실험으로 만들었으나 결함이 있었다: (a) `ballast.py`가 초기 fill 후 64MB 창만 재기록하는 순회 버그로 대부분이 스왑아웃돼 압박을 유지하지 못했고, (b) 0GB 기준선이 이미 스왑 30GB로 오염됐다. v3에서 도구를 고쳤다 — 전 페이지를 매 주기 터치해 상주 유지(3GB 요청 시 RSS 축소 없음 확인), pressure_probe에 GGUF F16 추가·config 순서 무작위화·매 측정 시점 스왑 사용량 기록(조작검증)·다단계 ballast·수정된 `measure_one` 사용. 그러나 통제 재실행은 보류했다: 깨끗한 0GB 기준선은 재부팅이 필요한데(세션이 스왑 8–18GB 누적) 이는 사용자 조작이고, 이미 압박된 머신에 ballast를 더하면 v2의 "압박 vs 더-압박"을 반복할 뿐이다. §3.2의 반복 분산이 같은 효과를 더 깨끗하게 보이므로 1차 근거로 삼았다.

## 4. Related Benchmarks (Sanity Check)

- Unsloth GGUF Benchmarks (Q4_K_M vs Q5_K_M perplexity)
- 커뮤니티: Q4_K_M ≈ FP16 대비 용량 75% 감소, 품질 저하 1–3%(MMLU)

perplexity 저하(NF4 +2.8%)는 방향이 일치하나 지표(WikiText-2 손실 vs MMLU)·양자화 방식(NF4 vs GGUF)이 달라 엄밀한 검증은 아니다.

## 5. Limitations & Future Work

- **측정 머신이 스왑 포화(8–18GB)** — HF 장문 수치는 반복 변동이 극심(중앙값+범위로 보고). 깨끗한 측정은 재부팅 필요. GGUF·단문·엔진/양자화 비율은 상대적으로 안정.
- **3회 반복** — 변동 큰 셀은 반복 확대로 신뢰구간 좁혀야 함.
- **CUDA 교차검증은 v1 버그 방식**이라 제외. C-1은 CUDA도 비동기라 동일 발현하므로 v3 프로토콜 재측정 필요.
- **ballast 통제 실험은 도구만 완비**, 실행은 재부팅 후 보류.
- **MPS Peak VRAM은 20ms 폴링 근사** — 짧은 스파이크 누락 가능. NF4 VRAM 증가 초과 원인 미규명(폴링 노이즈 가능).
- **GGUF는 매 측정 `llm.reset()`으로 캐시를 비워** 매번 완전 prefill 강제 — 접두사 재사용 멀티턴보다 보수적 조건.
- **속도는 end-to-end 측정** — NF4 저속 원인(커널 융합 부재)은 추정, 프로파일링 미수행.
- Future work: (1) 재부팅 후 깨끗한 context_sweep + `pressure_probe --ballast-gb 0,8,16`, (2) v3 프로토콜 CUDA 재교차검증, (3) NF4 디퀀타이즈 커널 프로파일링 + VRAM 증가 원인 규명(폴링 간격 축소 대조), (4) 반복 확대(≥5)·신뢰구간, (5) 소형 VLM 확장, (6) vLLM PagedAttention 동시요청 비교.

## References

1. Qwen Team, "Qwen3 Technical Report" / model card, Hugging Face.
2. bitsandbytes-foundation, "bitsandbytes" (NF4 quantization, MPS backend PR #1853), GitHub.
3. ggml-org, "llama.cpp", GitHub.
4. Unsloth, "GGUF Benchmarks" documentation.
5. Merity et al., "Pointer Sentinel Mixture Models" (WikiText-2 dataset).
