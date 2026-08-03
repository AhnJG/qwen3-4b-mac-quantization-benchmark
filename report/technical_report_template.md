# Quantitative Analysis of Post-Training Quantization for On-Device LLM Inference: A Qwen3-4B-Instruct Case Study

**Author:** TODO-이름 · TODO-소속(예: OO대학교 컴퓨터공학과)
**Date:** 2026-08-03 (실행 완료일)
**Code & data:** TODO-GitHub 저장소 링크 · **DOI:** TODO (Zenodo 발급 후 채움)

> 이 문서는 Phase 3에서 실제 측정값을 채운 뒤 `pdf` 스킬로 5페이지 내외 PDF로 변환합니다. 지금은 구조와 서술 방향을 잡아둔 템플릿입니다.

## Abstract

온디바이스/저비용 서빙을 위해 소형 LLM을 양자화하는 것은 산업 현장에서 실질적 수요가 있다. 본 리포트는 `Qwen3-4B-Instruct-2507`을 FP16, 4-bit NF4(bitsandbytes), GGUF Q4_K_M(llama.cpp) 세 가지 방식으로 Apple M3(24GB 통합 메모리, MPS/Metal backend)에서 실행하여 VRAM, 처리량(tokens/sec), WikiText-2 perplexity를 측정하고, 컨텍스트 길이(128–4096 토큰)에 따른 속도·메모리 스케일링을 분석했다. NF4는 VRAM을 61% 절감했으나(7.53GB → 2.90GB) MPS에서는 4-bit 연산 오버헤드로 오히려 FP16보다 약 2.7배 느렸다(9.10 → 3.35 tok/s). GGUF Q4_K_M은 FP16 대비 3.25배 빠른 29.5 tok/s를 보였다. Perplexity 저하는 NF4가 FP16 대비 +2.8%로 미미했다. 컨텍스트를 128–4096 토큰까지 늘리자 세 방식 모두 처리량이 감소했으며 저하폭은 GGUF -41%, NF4 -52%, FP16 -57% 순으로 GGUF가 가장 완만했다. 이 24GB 통합 메모리 머신을 다른 애플리케이션들로 메모리 압박이 심한 상태에서 먼저 측정했을 때는 통합 메모리가 실제 VRAM 증가분을 감당하지 못해 시스템 스왑이 발생, FP16/NF4의 4096 토큰 처리량이 각각 0.28, 0.10 tok/s까지 붕괴했으나, 다른 애플리케이션을 정리한 뒤 재측정하니 각각 3.92, 1.63 tok/s(14~17배)로 크게 회복됐다. 이는 메모리 제약 환경에서는 양자화 비트 수뿐 아니라 시스템 전체의 메모리 여유와 로딩/매핑 방식(mmap 기반 GGUF가 상대적으로 강건)이 실사용 강건성에 미치는 영향이 클 수 있음을 시사한다.

## 1. Motivation & Problem Framing

- 국내 기업(삼성전자, SK하이닉스, LG, 네이버 등)이 서빙 비용 절감·온디바이스 탑재를 위해 경량화 연구 인력 수요가 높다는 배경.
- 단, FP16 vs 4-bit 비교 자체는 이미 흔한 벤치마크이므로(§5 참고), 본 리포트는 (a) 세 방식을 동일 하드웨어·동일 모델에서 나란히 재현하고 (b) 컨텍스트 길이에 따른 스케일링이라는 덜 다뤄진 축을 추가해 차별화를 시도한다.

## 2. Methodology

- **모델:** Qwen/Qwen3-4B-Instruct-2507 (dense, 4B params)
- **양자화 방식:**
  - FP16 (baseline, `torch_dtype=torch.float16`)
  - NF4 4-bit (bitsandbytes, double quantization, compute dtype FP16)
  - GGUF Q4_K_M (llama.cpp, `unsloth/Qwen3-4B-Instruct-2507-GGUF`)
- **하드웨어:** Apple M3, 24GB 통합 메모리(macOS). FP16/NF4는 PyTorch MPS backend, GGUF는 llama.cpp Metal backend로 실행. 원래 계획한 CUDA GPU(Colab T4/L4)가 아닌 로컬 Apple Silicon 환경에서 측정함 (§5 참고).
- **측정 지표:**
  - Peak VRAM: CUDA에서는 `torch.cuda.max_memory_allocated()`, MPS에서는 이에 대응하는 API가 없어 `generate()` 실행 중 20ms 간격으로 `torch.mps.current_allocated_memory()`를 폴링해 최댓값을 추적(`PeakMemoryTracker`)
  - TTFT, tokens/sec (직접 생성 시간 측정, chat template 적용)
  - Perplexity: WikiText-2 test split, sliding window (max_length=1024, stride=512), FP16/NF4 동일 방법 적용. GGUF는 별도 방법이라 직접 비교하지 않음(§5 한계 참고).
- **컨텍스트 스윕:** 128 / 512 / 1024 / 2048 / 4096 토큰 길이에서 각 방식의 VRAM·속도 재측정.
- **재현성:** 코드 전체는 GitHub 저장소에 공개, `pip install -r requirements.txt` 후 `python benchmark_quantization.py --stage all`로 동일 실행 가능.

## 3. Experimental Results & Discussion

| Config | Peak VRAM (GB) | Tokens/sec | TTFT (s) | Perplexity |
|---|---|---|---|---|
| FP16 | 7.53 | 9.10 | 0.33 | 9.54 |
| NF4 4-bit | 2.90 | 3.35 | 0.60 | 9.81 |
| GGUF Q4_K_M | n/a\* | 29.54 | 0.31 | n/a |

\* llama.cpp는 torch 스타일의 VRAM 측정 API를 노출하지 않아 미측정.

- **VRAM 절감률:** NF4가 FP16 대비 61% 절감 (7.53GB → 2.90GB).
- **속도 변화:** NF4는 FP16 대비 0.37배(약 2.7배 느림) — MPS에서는 bitsandbytes의 4-bit 디퀀타이즈 오버헤드가 대역폭 절감분을 상쇄한 것으로 추정된다(커널 단위 프로파일링은 수행하지 않음). GGUF Q4_K_M은 FP16 대비 3.25배 빠름 — llama.cpp/Metal의 최적화된 커널 덕분으로 보인다.
- **품질 저하:** NF4 perplexity는 FP16 대비 +2.8%(9.54 → 9.81)로 미미함.

### 3.1 Context-length scaling

다른 애플리케이션을 정리해 시스템 메모리 여유를 확보한("조용한") 상태에서 측정한 프롬프트 길이 128 → 4096 토큰에서의 tokens/sec:

| Context (tokens) | FP16 | NF4 | GGUF |
|---|---|---|---|
| 128 | 9.18 | 3.43 | 25.02 |
| 512 | 8.17 | 3.28 | 22.60 |
| 1024 | 7.12 | 3.07 | 21.22 |
| 2048 | 3.98 | 2.63 | 18.29 |
| 4096 | 3.92 | 1.63 | 14.82 |

같은 구간의 Peak VRAM(GB):

| Context (tokens) | FP16 | NF4 |
|---|---|---|
| 128 | 7.54 | 2.88 |
| 512 | 7.75 | 3.11 |
| 1024 | 7.99 | 3.17 |
| 2048 | 8.10 | 3.30 |
| 4096 | 8.33 | 3.99 |

VRAM은 예상대로 컨텍스트 길이에 따라 증가했다(FP16 7.54→8.33GB, +0.79GB / NF4 2.88→3.99GB, +1.11GB) — KV 캐시가 컨텍스트 길이에 비례해 커지는 이론적 예상과 방향이 일치한다. NF4의 절대 증가량이 FP16보다 큰 것은 다소 예상 밖인데, NF4의 compute dtype도 FP16과 동일한 float16이라 KV 캐시 자체의 바이트당 크기는 이론상 비슷해야 하며 원인은 특정하지 못했다. tok/s 저하폭(128→4096)은 GGUF -41%, NF4 -52%, FP16 -57% 순으로 GGUF가 가장 완만했다.

**메모리 압박이 결과를 바꾸는 현상.** 이 컨텍스트 스윕은 원래 다른 애플리케이션들이 실행 중이던 메모리 압박 상태(스왑 23.6GB/24.5GB)에서 먼저 측정했는데, 그때는 FP16과 NF4가 2048 토큰을 넘어서며 처리량이 사실상 붕괴했다(FP16 6.99→1.87→0.28, NF4 2.84→2.11→0.10 tok/s). 다른 애플리케이션을 닫아 스왑을 줄인 뒤(측정 시작 시점 스왑 7.6GB/9.2GB) 재측정하니 4096 토큰 처리량이 FP16 0.28→3.92 tok/s(약 14배), NF4 0.10→1.63 tok/s(약 17배)로 크게 회복됐다. 이는 MPS 연산 자체의 한계가 아니라, 실제로 늘어난 VRAM 사용량이 24GB 통합 메모리에서 다른 프로세스들과 경쟁하며 유발한 디스크 스와핑이 원인이었음을 시사한다. 반대로 GGUF는 두 측정 모두에서 상대적으로 완만하게 저하됐는데, llama.cpp가 GGUF 파일을 mmap으로 매핑해 페이지 단위로만 물리 메모리에 적재하고 OS가 필요시 유연하게 페이지를 회수할 수 있는 반면, PyTorch/HF 경로는 모델과 활성화 텐서를 프로세스 메모리에 상주시켜야 하므로 같은 메모리 압박에서 더 취약했던 것으로 보인다.

다만 두 측정은 완벽히 통제된 A/B는 아니다 — 조용한 상태 재측정에서 GGUF조차 모든 컨텍스트 길이의 tok/s가 이전 측정보다 균일하게 낮았다(예: ctx=128 30.53→25.02 tok/s). 원인은 확인하지 못했다. 후보 가설로 (a) 벤치마크 스크립트에서 `run_gguf()`는 `run_nf4()`의 `del model; gc.collect(); empty_cache()` 직후 실행되므로 두 트라이얼에서 GGUF 차례의 실제 메모리 여유가 서로 달랐을 가능성, (b) GGUF 파일(~2.5GB)의 OS 페이지 캐시 warm/cold 상태 차이, (c) 열 스로틀링 등이 있으나 어느 것도 검증하지 않았다. 그럼에도 FP16/NF4의 4096 토큰 회복폭(14~17배)은 이런 후보들로 설명되는 수준을 크게 웃돌므로, "심한 메모리 압박이 FP16/NF4를 스왑으로 붕괴시켰다"는 핵심 결론 자체는 유효하다고 판단하지만, 이 역시 반복 측정 없이 단일 비교로 내린 잠정적 결론이다. "양자화 비트 수"뿐 아니라 "로딩/매핑 방식"이 메모리 제약 환경에서의 실사용 강건성에 영향을 줄 수 있다는 시사점이다.

## 4. Related Benchmarks (Sanity Check)

우리 결과를 다음과 교차 검증한다 (완전히 동일한 실험 세팅은 아니므로 "방향성 일치 여부"를 확인하는 용도):

- Unsloth GGUF Benchmarks (Qwen 계열 Q4_K_M vs Q5_K_M perplexity 비교)
- 커뮤니티에 보고된 일반적 트렌드: Q4_K_M ≈ FP16 대비 용량 75% 감소, 품질 저하 1–3%(MMLU 기준)

우리 결과의 perplexity 저하폭(NF4 +2.8%)은 커뮤니티에서 보고되는 1–3% 범위 안에 들어와 방향이 일치한다. 다만 VRAM 절감률(NF4 61%)은 "모델 파일 용량 감소율"(75%)과 다른 지표(런타임 peak VRAM, KV 캐시·activation 포함)라서 직접 비교는 부적절하다 — bitsandbytes NF4는 가중치만 4-bit로 저장하고 compute dtype(FP16)으로 연산하므로 GGUF의 파일 용량 감소율보다 런타임 절감률이 낮게 나오는 것은 예상된 결과다. 컨텍스트 스윕 및 메모리 압박 하에서의 속도 붕괴 현상(§3.1)은 우리가 확인한 범위에서는 비교 가능한 공개 벤치마크를 찾지 못했다 — 대부분의 공개 벤치마크가 전용 GPU 서버(VRAM이 시스템 메모리와 분리된 환경)에서 측정되어 통합 메모리 환경 특유의 이 현상이 보고되지 않았을 가능성이 있다.

## 5. Limitations & Future Work

- GGUF perplexity는 FP16/NF4와 계산 방법이 달라 직접 비교하지 않음.
- 단일 실행 기준 수치이며 반복 측정에 따른 분산은 보고하지 않음.
- **측정 하드웨어는 Apple M3(24GB 통합 메모리)이며, 원래 계획한 CUDA GPU(Colab T4/L4 등)가 아니다.** FP16/NF4는 PyTorch MPS backend, GGUF는 llama.cpp Metal backend로 실행했다. CUDA 환경에서의 수치는 이 결과와 다를 수 있으며, 특히 NF4가 FP16보다 느린 현상은 bitsandbytes 4-bit 커널이 CUDA에서 더 최적화되어 있어 재현되지 않을 가능성이 있다.
- **MPS의 Peak VRAM은 폴링 기반 근사치다.** PyTorch MPS backend는 CUDA의 reset 가능한 peak 카운터를 제공하지 않아, `generate()` 실행 중 20ms 간격 폴링으로 최댓값을 추적했다. 폴링 간격보다 짧게 스파이크하는 할당은 놓칠 수 있다.
- **컨텍스트 스윕은 메모리 압박 상태와 조용한 상태 두 번 측정했다** (§3.1). 본문 수치는 조용한 상태 기준이다. 두 측정 간 GGUF의 전 구간 tok/s가 균일하게 차이 나는 등(스크립트 실행 순서에 따른 트라이얼별 메모리 여유 차이, 파일 캐시 warm/cold, 열 스로틀링 등 원인 미확인) 완벽히 통제된 A/B는 아니므로, 절대 수치보다는 "메모리 압박이 심하면 FP16/NF4가 GGUF보다 훨씬 크게 무너진다"는 방향성으로 해석해야 한다.
- Future work: (1) 텍스트 전용 모델을 넘어 소형 VLM(Qwen3-VL 등)으로 확장, (2) vLLM의 PagedAttention을 적용한 동시 요청(Throughput) 비교 추가, (3) CUDA GPU에서 재측정해 이 리포트의 결과와 교차검증, (4) GGUF 단계의 tok/s 차이 원인(메모리 여유 시점, 파일 캐시 상태, 열 스로틀링)을 통제된 반복 측정으로 규명.

## References

1. Qwen Team, "Qwen3 Technical Report" / model card, Hugging Face.
2. bitsandbytes-foundation, "bitsandbytes" (NF4 quantization, MPS backend PR #1853), GitHub.
3. ggml-org, "llama.cpp", GitHub.
4. Unsloth, "GGUF Benchmarks" documentation.
5. Merity et al., "Pointer Sentinel Mixture Models" (WikiText-2 dataset).
