# Quantitative Analysis of Post-Training Quantization for On-Device LLM Inference: A Qwen3-4B-Instruct Case Study

**Author:** TODO-이름 · TODO-소속(예: OO대학교 컴퓨터공학과)
**Date:** TODO (실행 완료일)
**Code & data:** TODO-GitHub 저장소 링크 · **DOI:** TODO (Zenodo 발급 후 채움)

> 이 문서는 Phase 3에서 실제 측정값을 채운 뒤 `pdf` 스킬로 5페이지 내외 PDF로 변환합니다. 지금은 구조와 서술 방향을 잡아둔 템플릿입니다.

## Abstract

*(150–200 단어)* 온디바이스/저비용 서빙을 위해 소형 LLM을 양자화하는 것은 산업 현장에서 실질적 수요가 있다. 본 리포트는 `Qwen3-4B-Instruct-2507`을 FP16, 4-bit NF4(bitsandbytes), GGUF Q4_K_M(llama.cpp) 세 가지 방식으로 실행하여 VRAM, 처리량(tokens/sec), WikiText-2 perplexity를 측정하고, 추가로 컨텍스트 길이(128–4096 토큰)에 따른 속도·메모리 스케일링을 분석한다. 결과: TODO-핵심수치 요약.

## 1. Motivation & Problem Framing

- 국내 기업(삼성전자, SK하이닉스, LG, 네이버 등)이 서빙 비용 절감·온디바이스 탑재를 위해 경량화 연구 인력 수요가 높다는 배경.
- 단, FP16 vs 4-bit 비교 자체는 이미 흔한 벤치마크이므로(§5 참고), 본 리포트는 (a) 세 방식을 동일 하드웨어·동일 모델에서 나란히 재현하고 (b) 컨텍스트 길이에 따른 스케일링이라는 덜 다뤄진 축을 추가해 차별화를 시도한다.

## 2. Methodology

- **모델:** Qwen/Qwen3-4B-Instruct-2507 (dense, 4B params)
- **양자화 방식:**
  - FP16 (baseline, `torch_dtype=torch.float16`)
  - NF4 4-bit (bitsandbytes, double quantization, compute dtype FP16)
  - GGUF Q4_K_M (llama.cpp, `unsloth/Qwen3-4B-Instruct-2507-GGUF`)
- **하드웨어:** TODO (예: Google Colab, NVIDIA T4, 16GB VRAM / 실제 `nvidia-smi` 출력 첨부)
- **측정 지표:**
  - Peak VRAM (`torch.cuda.max_memory_allocated`)
  - TTFT, tokens/sec (직접 생성 시간 측정, chat template 적용)
  - Perplexity: WikiText-2 test split, sliding window (max_length=1024, stride=512), FP16/NF4 동일 방법 적용. GGUF는 별도 방법이라 직접 비교하지 않음(§6 한계 참고).
- **컨텍스트 스윕:** 128 / 512 / 1024 / 2048 / 4096 토큰 길이에서 각 방식의 VRAM·속도 재측정.
- **재현성:** 코드 전체는 GitHub 저장소에 공개, `pip install -r requirements.txt` 후 `python benchmark_quantization.py --stage all`로 동일 실행 가능.

## 3. Experimental Results & Discussion

*(Phase 2 수치로 채움)*

| Config | Peak VRAM (GB) | Tokens/sec | TTFT (s) | Perplexity |
|---|---|---|---|---|
| FP16 | | | | |
| NF4 4-bit | | | | |
| GGUF Q4_K_M | | | | n/a |

- VRAM 절감률: TODO%
- 속도 변화: TODO배
- 품질 저하: TODO (perplexity 차이 및 해석)

### 3.1 Context-length scaling

*(그래프: x축 컨텍스트 길이, y축 VRAM/속도, 방식별 라인 3개)*

- 컨텍스트가 늘어날 때 어느 방식이 가장 완만하게/가파르게 저하되는가?
- KV 캐시 메모리 증가 패턴이 이론적 예상(선형 증가)과 일치하는가?

## 4. Related Benchmarks (Sanity Check)

우리 결과를 다음과 교차 검증한다 (완전히 동일한 실험 세팅은 아니므로 "방향성 일치 여부"를 확인하는 용도):

- Unsloth GGUF Benchmarks (Qwen 계열 Q4_K_M vs Q5_K_M perplexity 비교)
- 커뮤니티에 보고된 일반적 트렌드: Q4_K_M ≈ FP16 대비 용량 75% 감소, 품질 저하 1–3%(MMLU 기준)

TODO: 우리 수치가 이 범위 안에 들어오는지, 벗어난다면 왜인지(모델 크기, 태스크 차이 등) 서술.

## 5. Limitations & Future Work

- GGUF perplexity는 FP16/NF4와 계산 방법이 달라 직접 비교하지 않음.
- 컨텍스트 스윕에서 GGUF 프롬프트 길이는 HF 토크나이저 기준이 아닌 근사치.
- 단일 실행 기준 수치이며 반복 측정에 따른 분산은 보고하지 않음.
- Colab 무료 GPU는 세션마다 배정이 달라질 수 있어 하드웨어 재현성에 한계가 있음.
- Future work: (1) 텍스트 전용 모델을 넘어 소형 VLM(Qwen3-VL 등)으로 확장, (2) vLLM의 PagedAttention을 적용한 동시 요청(Throughput) 비교 추가.

## References

1. Qwen Team, "Qwen3 Technical Report" / model card, Hugging Face.
2. bitsandbytes-foundation, "bitsandbytes" (NF4 quantization, MPS backend PR #1853), GitHub.
3. ggml-org, "llama.cpp", GitHub.
4. Unsloth, "GGUF Benchmarks" documentation.
5. Merity et al., "Pointer Sentinel Mixture Models" (WikiText-2 dataset).
