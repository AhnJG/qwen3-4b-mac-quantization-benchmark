# Quantitative Analysis of Post-Training Quantization for On-Device LLM Inference: A Qwen3-4B-Instruct Case Study

**Author:** TODO-이름 · TODO-소속(예: OO대학교 컴퓨터공학과)
**Date:** 2026-08-03 (v1 실행) / 2026-08-04 (v2, 프로토콜 수정 후 재측정)
**Code & data:** TODO-GitHub 저장소 링크 · **DOI:** 미발급 — 아래 §5의 재검증 항목(재부팅 후 ballast 재실행, CUDA 재교차검증, 반복 측정) 완료 후 발급 예정

> 이 문서는 Phase 3에서 실제 측정값을 채운 뒤 `pdf` 스킬로 5페이지 내외 PDF로 변환합니다. v1은 외부 동료 리뷰 2건에서 측정 스크립트의 치명적 결함(이중 prefill, 챗 템플릿 비대칭 적용)과 실험 설계 결함(엔진×양자화 교락)을 지적받아 v2로 갱신했습니다. v1 데이터는 `results/*_v1_double_prefill_bug.jsonl`에 감사 기록으로 남겨뒀습니다.

## Abstract

온디바이스/저비용 서빙을 위해 소형 LLM을 양자화하는 것은 산업 현장에서 실질적 수요가 있다. 본 리포트는 `Qwen3-4B-Instruct-2507`을 네 설정 — HF FP16, HF NF4 4-bit(bitsandbytes), llama.cpp GGUF F16, llama.cpp GGUF Q4_K_M — 으로 Apple M3(24GB 통합 메모리, MPS/Metal backend)에서 실행하여 VRAM, 처리량(tokens/sec), WikiText-2 perplexity를 측정하고, 컨텍스트 길이(128–4096 토큰)에 따른 속도 스케일링을 분석했다. GGUF F16을 추가한 것은 "GGUF가 빠르다"는 관찰에서 엔진 효과(llama.cpp vs HF `generate()`)와 양자화 효과를 분리하기 위해서다: 같은 정밀도(F16)에서 llama.cpp는 HF보다 1.13배 빠른 데 그치지만, 같은 llama.cpp 엔진 안에서 Q4_K_M으로 양자화하면 2.93배 더 빨라진다 — 속도 향상의 대부분은 엔진이 아니라 양자화다. 반대로 bitsandbytes NF4는 HF FP16 대비 VRAM을 61% 절감하면서도 처리량은 오히려 0.37배(2.70배 느림)로 떨어졌다. 같은 "4-bit"라는 이름의 두 구현이 정반대 방향으로 움직인 것은, llama.cpp의 K-quant 커널이 추론에 맞춰 손수 최적화된 반면 bitsandbytes NF4는 QLoRA 학습(메모리 절감)을 위해 설계돼 디퀀타이즈와 행렬곱을 묶는 커널 융합이 부족하기 때문으로 추정된다. Perplexity 저하는 NF4가 FP16 대비 +2.8%로 미미했다. 컨텍스트를 128–4096 토큰까지 늘리자 저하폭은 GGUF F16이 가장 완만했다(-19.6%) — GGUF Q4_K_M(-56.9%)은 HF FP16(-55.9%)·NF4(-52.8%)와 비슷하게 저하되어, "GGUF가 컨텍스트 확장에 가장 강건하다"던 이전(v1) 결론이 뒤집혔다. v1의 이 결론은 측정 버그(HF 경로의 두 번째 `generate()` 호출이 KV 캐시 없이 프롬프트를 처음부터 재-prefill하면서 컨텍스트 길이만큼의 지연이 "처리량"에 섞여 들어간 반면, GGUF 경로는 동일 프롬프트 재호출 시 llama.cpp의 접두사 캐시가 prefill을 건너뛰어 순수 디코딩 속도만 재진 것)에서 비롯됐으며, 외부 동료 리뷰가 "TTFT(1토큰)가 전체 생성 시간보다 길다"는 물리적으로 불가능한 관측을 근거로 확인했다. v1은 또한 GGUF가 llama.cpp의 mmap 로딩 덕분에 메모리 압박에 강건하다고 추측했는데, ballast 프로세스로 메모리 압박을 조작 변수화하고 `use_mmap=False` 대조군을 둔 통제 실험으로 직접 검증한 결과 이 가설은 반증됐다 — mmap을 꺼도 GGUF 성능은 변하지 않았다(다만 이 실험의 기준선이 완전히 유휴한 상태는 아니었다는 한계가 있다, §3.3).

## 1. Motivation & Problem Framing

- 국내 기업(삼성전자, SK하이닉스, LG, 네이버 등)이 서빙 비용 절감·온디바이스 탑재를 위해 경량화 연구 인력 수요가 높다는 배경.
- FP16 vs 4-bit 비교 자체는 흔한 벤치마크이므로(§4 참고), 본 리포트는 (a) 엔진 효과와 양자화 효과를 분리하는 2×2에 가까운 요인 설계, (b) 컨텍스트 길이에 따른 스케일링이라는 두 축으로 차별화를 시도한다.
- v1은 (a) 축이 없어 "GGUF가 빠르다"의 원인(엔진 vs 양자화)을 분리하지 못했고, 측정 버그로 (b) 축의 핵심 결론도 틀렸다. v2는 두 문제를 모두 겨냥해 설계를 다시 짰다.

## 2. Methodology

- **모델:** Qwen/Qwen3-4B-Instruct-2507 (dense, 4B params)
- **설정 (2×2에 가까운 요인 설계):**

  |  | HF transformers 엔진 | llama.cpp 엔진 |
  |---|---|---|
  | **FP16/F16 정밀도** | HF FP16 (baseline) | GGUF F16 |
  | **4-bit 양자화** | HF NF4 (bitsandbytes) | GGUF Q4_K_M |

  대각선(HF FP16 vs GGUF Q4_K_M)만 비교하면 엔진과 양자화가 동시에 바뀌어 원인을 분리할 수 없다는 점이 외부 리뷰에서 지적됐다. GGUF F16을 네 번째 칸으로 추가해 행(엔진 효과)과 열(양자화 효과)을 각각 분리했다.
- **하드웨어:** Apple M3, 24GB 통합 메모리(macOS). HF 경로(FP16/NF4)는 PyTorch MPS backend, GGUF 경로(F16/Q4_K_M)는 llama.cpp Metal backend.
- **측정 지표:**
  - Peak VRAM (HF 경로만): MPS는 CUDA의 `reset_peak_memory_stats()`에 대응하는 API가 없어 `generate()` 실행 중 20ms 간격으로 `torch.mps.current_allocated_memory()`를 폴링해 근사(`PeakMemoryTracker`). llama.cpp는 이 측정 API가 없어 미측정.
  - TTFT, tokens/sec: **v2에서 측정 방식을 재설계함.** v1은 `model.generate(max_new_tokens=1)`로 TTFT를, 이어서 *독립적인* `model.generate(max_new_tokens=150)` 호출로 처리량을 쟀다 — 두 번째 호출이 KV 캐시를 재사용하지 않아 컨텍스트 길이만큼의 prefill이 "처리량"에 중복 포함됐다. v2는 **단일** `generate()` 호출 안에서 logits processor로 매 디코딩 스텝마다 타임스탬프를 찍어 TTFT(첫 스텝)와 디코딩 처리량(`(n-1)/(마지막 스텝 시각 - 첫 스텝 시각)`)을 분리했고, 타이밍 전 워밍업 1회(CUDA/MPS 커널 초기화 비용 격리)를 추가했다. GGUF 경로도 대칭적으로 재설계했다: 이전에는 같은 `llm` 인스턴스에 같은 프롬프트로 두 번 호출해 llama.cpp의 접두사 캐시가 두 번째 호출의 prefill을 건너뛰었다 — v2는 매 컨텍스트 길이마다 워밍업 후 `llm.reset()`으로 캐시를 비우고, `stream=True`로 단일 스트리밍 호출을 하며 청크마다 타임스탬프를 찍는다. 또한 GGUF 경로도 HF와 **동일한 HF 토크나이저로 생성한 챗 템플릿 문자열**을 입력으로 쓰도록 통일했다(v1은 GGUF에만 챗 템플릿 없는 원시 텍스트를 넣어, 인스트럭션 튜닝 모델에 분포 밖(OOD) 입력을 주는 비대칭이 있었다).
  - Perplexity: WikiText-2 test split, sliding window (max_length=1024, stride=512), FP16/NF4 동일 방법 적용. GGUF는 별도 방법이라 직접 비교하지 않음(§5 한계 참고). v1과 계산 로직이 동일하고 이 버그의 영향을 받지 않으므로 v1 값을 재사용함(플랫폼 간 5자리 일치까지 이미 교차검증됨).
- **컨텍스트 스윕:** 128 / 512 / 1024 / 2048 / 4096 토큰 길이에서 각 방식의 속도(및 HF 경로는 VRAM) 재측정.
- **재현성:** 코드 전체는 GitHub 저장소에 공개 예정. `pip install -r requirements.txt` 후 `python benchmark_quantization.py --stage all`로 동일 실행 가능. 각 결과 행에 `device`/`prompt_tokens`/`n_new`를 함께 기록해 tok/s 계산을 사후 검증할 수 있게 함(v2에서 추가 — 외부 리뷰가 "데이터에 검증 수단이 없다"고 지적한 데 대한 대응).

## 3. Experimental Results & Discussion

| Config | Peak VRAM (GB) | Tokens/sec (decode) | TTFT (s) | Perplexity |
|---|---|---|---|---|
| HF FP16 | 7.53 | 7.48 | 0.018 | 9.54 |
| HF NF4 4-bit | 2.90 | 2.77 | 0.033 | 9.81 |
| GGUF F16 | n/a\* | 8.47 | 0.307 | n/a |
| GGUF Q4_K_M | n/a\* | 24.85 | 0.130 | n/a |

\* llama.cpp는 torch 스타일의 VRAM 측정 API를 노출하지 않아 미측정.

### 3.1 엔진 효과와 양자화 효과의 분리

|  | 계산식 | 배수 |
|---|---|---|
| 엔진 효과 (F16 고정, 엔진만 변경) | GGUF-F16 / HF-FP16 = 8.47 / 7.48 | 1.13배 |
| llama.cpp 내 양자화 효과 (엔진 고정, 정밀도만 변경) | GGUF-Q4_K_M / GGUF-F16 = 24.85 / 8.47 | 2.93배 |
| bitsandbytes 양자화 효과 (엔진 고정, 정밀도만 변경) | HF-NF4 / HF-FP16 = 2.77 / 7.48 | 0.37배 (2.70배 느림) |

"GGUF가 FP16보다 3.32배 빠르다"(24.85/7.48)는 관찰은 사실이지만, 요인을 분리하면 그중 대부분(2.93배)이 llama.cpp 자체의 우수함이 아니라 K-quant 양자화의 효과이고, 엔진 자체의 기여는 1.13배에 그친다. 반면 bitsandbytes NF4는 같은 "4-bit 양자화"이면서도 정반대로 작동해 VRAM은 61% 절감하지만(7.53GB → 2.90GB) 처리량은 2.70배 느려진다. 두 구현의 목적이 다르기 때문으로 추정된다 — llama.cpp의 K-quant는 추론 전용으로 설계돼 디퀀타이즈와 행렬곱을 커널 단위로 융합하는 반면, bitsandbytes NF4는 QLoRA 학습 시의 메모리 절감이 주 목적이라 추론 시 디퀀타이즈 오버헤드가 대역폭 절감분을 상쇄한다(커널 단위 프로파일링으로 직접 검증한 것은 아니며, 일반적인 아키텍처 지식에 기반한 추정).

품질 저하는 NF4 perplexity가 FP16 대비 +2.8%(9.54 → 9.81)로 미미했다.

### 3.2 Context-length scaling

프롬프트 길이 128 → 4096 토큰에서의 디코딩 tokens/sec:

| Context (tokens) | HF FP16 | HF NF4 | GGUF F16 | GGUF Q4_K_M |
|---|---|---|---|---|
| 128 | 9.15 | 3.44 | 9.25 | 31.27 |
| 512 | 8.23 | 3.16 | 9.25 | 23.52 |
| 1024 | 6.73 | 2.28 | 9.87 | 22.09 |
| 2048 | 2.56 | 2.43 | 9.61 | 19.25 |
| 4096 | 4.04 | 1.62 | 7.44 | 13.48 |

저하폭(128→4096): **GGUF F16 -19.6%**, HF NF4 -52.8%, HF FP16 -55.9%, GGUF Q4_K_M -56.9%.

**v1 결론의 반전.** v1은 GGUF Q4_K_M이 가장 완만하게 저하된다(-41%)고 결론 내리고, 이를 이 리포트의 핵심 차별화 축으로 제시했다. v2에서 측정 프로토콜을 수정한 결과 이 순위가 뒤집혔다 — GGUF Q4_K_M은 이제 가장 가파르게 저하되는 축에 속하고(HF FP16·NF4와 비슷), 오히려 GGUF F16이 가장 완만하다. 원인은 §2에서 설명한 이중 prefill 버그다: v1의 HF 측정은 컨텍스트가 길어질수록 커지는 prefill 시간이 "처리량"에 반복적으로 섞여 들어가 저하폭을 과장했고, GGUF 측정은 접두사 캐시로 이 prefill을 건너뛰어 저하폭이 실제보다 완만하게 나왔다. 두 경로가 같은 방향으로 편향돼 있었기 때문에, 버그가 있는 상태에서도 "GGUF가 가장 완만하다"는 그럴듯한 패턴이 우연히 만들어졌다.

**GGUF F16이 가장 강건한 이유는 v2 시점에서 아직 규명하지 못했다.** 같은 llama.cpp 엔진 안에서 정밀도만 다른데(F16 vs Q4_K_M) 컨텍스트 확장에 대한 강건성이 이렇게 크게 갈리는 것은 KV 캐시 크기만으로는 설명되지 않는다(두 설정 모두 KV 캐시는 동일). ggml의 배치/스케줄링 로직이 정밀도별로 다르게 동작하는지 추가 조사가 필요하다.

HF FP16은 ctx=2048에서 저점(2.56 tok/s)을 찍고 4096에서 반등(4.04)하는 비단조 곡선을 보였다 — 반복 측정 없이는 이것이 실제 현상인지 노이즈인지 판단할 수 없다(§5 한계).

같은 구간의 Peak VRAM(GB, HF 경로만):

| Context (tokens) | HF FP16 | HF NF4 |
|---|---|---|
| 128 | 7.537 | 2.878 |
| 512 | 7.740 | 3.129 |
| 1024 | 7.963 | 3.168 |
| 2048 | 8.079 | 3.296 |
| 4096 | 8.330 | 3.946 |

VRAM은 예상대로 컨텍스트 길이에 따라 증가했다(FP16 +0.79GB, NF4 +1.07GB). NF4의 절대 증가량이 FP16보다 큰 것은 이론(NF4의 compute dtype도 FP16이라 KV 캐시 바이트 크기는 양자화 방식과 무관해야 함)과 어긋나는데, v1부터 원인을 특정하지 못한 채로 남아 있다.

### 3.3 Memory pressure (ballast experiment)

v1은 "다른 애플리케이션이 켜져 있어 스왑으로 속도가 붕괴했다"는 재현 불가능한 관측을 근거로, GGUF가 llama.cpp의 mmap 로딩 덕분에 메모리 압박에 강건하다고 추측했다. v2에서는 이를 통제된 실험으로 전환했다: `tools/ballast.py`가 지정한 크기(GB)만큼 메모리를 할당하고 압축을 회피하려 주기적으로 랜덤 데이터로 재기록하며 유지하는 별도 프로세스를 띄우고, 그 압박 수준에서 `vm_stat`을 1초 간격으로 동시 기록하며 각 설정의 속도를 측정한다(`tools/pressure_probe.py`). GGUF에는 `use_mmap=True`/`False` 대조군을 둬서 mmap 가설을 직접 검증했다.

ballast 0GB vs 16GB, ctx=128과 4096에서의 디코딩 tok/s:

| Config | ctx=128 (0GB) | ctx=128 (16GB) | ctx=4096 (0GB) | ctx=4096 (16GB) |
|---|---|---|---|---|
| HF FP16 | 8.92 | 8.07 | 0.60 | 0.30 |
| HF NF4 | 3.29 | 3.25 | 0.78 | 0.87 |
| GGUF Q4_K_M (mmap) | 27.81 | 30.78 | 19.91 | 20.46 |
| GGUF Q4_K_M (mmap 끔) | 29.31 | 30.91 | 19.60 | 19.57 |

**mmap 가설은 반증됐다.** `use_mmap=False`로 mmap을 완전히 꺼도 GGUF 성능은 mmap을 켰을 때와 사실상 동일하다(ctx=4096에서 19.60 vs 19.91, 19.57 vs 20.46 — 오차 범위 내). "GGUF가 메모리 압박에 강건한 이유는 mmap 때문"이라는 v1의 설명은 이 데이터로 지지되지 않는다. 외부 리뷰 중 하나가 제안한 대안 — PyTorch 캐싱 할당자의 파편화 vs llama.cpp의 정적 KV 캐시 아레나 할당 — 이 더 그럴듯한 설명으로 남는다.

⚠️ **이 실험의 "0GB" 기준선은 깨끗한 유휴 상태가 아니었다.** 이 프로젝트를 여러 시간 연속 실행하면서 시스템 스왑이 이미 30GB 넘게 불어 있었고(실험 시작 시점 `vm_stat` 확인), HF FP16/NF4는 추가 ballast 없이도 ctx=4096에서 이미 심하게 느렸다(0.60/0.78 tok/s — §3.2의 깨끗한 측정치 4.04/1.62 tok/s보다 훨씬 나쁨). 즉 이 비교는 "유휴 vs 압박"이 아니라 "이미 압박된 상태 vs 더 압박된 상태"에 가깝다. GGUF가 mmap 유무와 무관하게 두 압박 수준 모두에서 흔들리지 않았다는 결론은 신뢰할 수 있지만, HF 경로가 정확히 몇 GB부터 무너지기 시작하는지는 이 실험으로 알 수 없다 — 재부팅한 깨끗한 상태에서의 재실행이 필요하다(§5).

## 4. Related Benchmarks (Sanity Check)

우리 결과를 다음과 교차 검증한다 (완전히 동일한 실험 세팅은 아니므로 "방향성 일치 여부"를 확인하는 용도):

- Unsloth GGUF Benchmarks (Qwen 계열 Q4_K_M vs Q5_K_M perplexity 비교)
- 커뮤니티에 보고된 일반적 트렌드: Q4_K_M ≈ FP16 대비 용량 75% 감소, 품질 저하 1–3%(MMLU 기준)

우리 결과의 perplexity 저하폭(NF4 +2.8%)은 커뮤니티에서 보고되는 1–3% 범위 안에 들어와 방향이 일치한다. 다만 지표(WikiText-2 언어모델링 손실 vs MMLU 정확도)와 양자화 방식(bitsandbytes NF4 vs GGUF Q4_K_M)이 서로 달라 엄밀한 검증은 아니다. VRAM 절감률(NF4 61%)과 "모델 파일 용량 감소율"(75%)도 다른 지표(런타임 peak VRAM, KV 캐시·activation 포함 vs 파일 크기)라서 직접 비교는 부적절하다.

## 5. Limitations & Future Work

- GGUF perplexity는 FP16/NF4와 계산 방법이 달라 직접 비교하지 않음.
- 단일 실행 기준 수치이며 반복 측정에 따른 분산은 보고하지 않음. HF FP16의 ctx=2048/4096 간 비단조 구간(§3.2)이 대표적으로, 반복 측정 없이는 노이즈와 실제 현상을 구분할 수 없다.
- **CUDA 교차검증은 v1 버그가 있던 측정 방식으로 얻은 결과이며, 재검증 전까지 이 리포트에서 제외했다.** v1에서는 이 실험이 "GGUF가 컨텍스트 확장에 강건하다"는 서사를 CUDA에서도 재현되는 것으로 뒷받침했는데, 그 서사가 측정 버그였음이 드러난 이상 재검증 없이는 신뢰할 수 없다.
- **ballast 실험(§3.3)의 "0GB" 기준선이 깨끗한 유휴 상태가 아니었다** — 여러 시간 연속 실행으로 시스템 스왑이 이미 30GB 넘게 불어 있었다. mmap 가설 반증은 신뢰할 수 있지만, HF 경로의 정확한 붕괴 임계치는 재부팅 후 재측정이 필요하다.
- MPS의 Peak VRAM은 폴링 기반 근사치다(20ms 간격). 폴링 간격보다 짧게 스파이크하는 할당은 놓칠 수 있다.
- GGUF 경로는 매 컨텍스트 길이마다 워밍업 후 `llm.reset()`으로 캐시를 비우고 측정한다 — 이는 매번 완전한 재-prefill을 강제하는 보수적(불리한) 조건으로, 접두사가 재사용되는 실사용 멀티턴 대화보다 나쁜 조건이다.
- GGUF F16이 컨텍스트 확장에 가장 강건한 이유(§3.2)는 규명하지 못했다.
- NF4의 절대 VRAM 증가량이 FP16보다 큰 이유(§3.2)도 v1부터 미규명 상태다.
- Future work: (1) 재부팅한 깨끗한 상태에서 ballast 실험 재실행(0/4/8/12/16GB로 세분화, 정확한 붕괴 임계치 규명), (2) v2 프로토콜로 CUDA(T4) 재교차검증, (3) GGUF F16의 컨텍스트 강건성 원인 규명, (4) 최소 3회 반복 측정 + 표준편차 보고, (5) 텍스트 전용 모델을 넘어 소형 VLM으로 확장, (6) vLLM PagedAttention 기반 동시 요청 비교, (7) NF4 디퀀타이즈 오버헤드의 커널 단위 프로파일링.

## References

1. Qwen Team, "Qwen3 Technical Report" / model card, Hugging Face.
2. bitsandbytes-foundation, "bitsandbytes" (NF4 quantization, MPS backend PR #1853), GitHub.
3. ggml-org, "llama.cpp", GitHub.
4. Unsloth, "GGUF Benchmarks" documentation.
5. Merity et al., "Pointer Sentinel Mixture Models" (WikiText-2 dataset).
