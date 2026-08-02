# Efficient AI: On-Device Quantization Benchmark (Qwen3-4B-Instruct)

> **Status: benchmarked.** `--stage all`, `--stage context_sweep` 실행 완료. 수치는 Apple M3(macOS, MPS/Metal) 1회 실행 기준이며, 원본 Colab/CUDA 경로는 아직 검증되지 않음 — 자세한 내용은 [Limitations](#limitations) 참고.

## TL;DR

> `Qwen3-4B-Instruct-2507`를 4-bit(NF4, bitsandbytes)로 양자화해 VRAM을 61%(7.53GB → 2.90GB) 절감했지만, Apple Silicon(MPS)에서는 생성 속도가 FP16보다 오히려 약 2.7배 느렸다(9.10 → 3.35 tok/s). GGUF Q4_K_M(llama.cpp, Metal)은 FP16 대비 약 3.2배 빠른 29.5 tok/s를 기록했다. 품질 저하(WikiText-2 perplexity)는 NF4가 FP16 대비 +2.8%(9.54 → 9.81)로 미미했다. 컨텍스트 길이를 128 → 4096 토큰으로 늘리자 VRAM은 셋 다(측정 가능한 FP16/NF4 기준) 실제로 증가했고(FP16 7.54→8.41GB, NF4 2.88→4.02GB), 이 24GB 통합 메모리 Mac에서는 2048 토큰을 넘어서며 FP16/NF4의 속도가 사실상 붕괴했다(FP16 6.99→0.28 tok/s, NF4 2.84→0.10 tok/s) — 원인은 시스템 스왑(측정 시점 스왑 사용량 23.6GB/24.5GB)이다. 반면 GGUF는 mmap 기반 로딩 덕분에 같은 구간에서도 27.3→19.9 tok/s로 완만하게만 저하됐다 — 메모리 제약 환경에서는 양자화 방식보다 로딩/매핑 방식이 실사용 강건성을 더 크게 좌우할 수 있음을 보여준다.

## What this is

`Qwen/Qwen3-4B-Instruct-2507` 모델을 세 가지 방식(FP16 원본, bitsandbytes NF4 4-bit, llama.cpp GGUF Q4_K_M)으로 실행하며 다음을 정량 비교합니다.

- **메모리**: Peak VRAM 사용량 (GB)
- **속도**: Time-to-first-token(TTFT), 초당 토큰 생성 수(tokens/sec)
- **품질**: WikiText-2 (test split) sliding-window perplexity

추가로, 입력 길이가 128 → 4096 토큰으로 늘어날 때 세 방식의 속도/메모리가 어떻게 변하는지(KV 캐시 스케일링) 측정합니다 — 이 부분이 흔한 "짧은 프롬프트 한 번" 벤치마크와의 차별점입니다.

## Quickstart

**Google Colab (권장)**: `benchmark_quantization.ipynb`을 열고 안내를 따르세요. Runtime을 GPU(T4 이상)로 바꿔야 합니다.

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

결과는 `results/results.jsonl`에 누적 기록됩니다.

## Results

측정 환경: Apple M3 (macOS, unified memory) · FP16/NF4는 PyTorch MPS backend, GGUF는 llama.cpp Metal backend · 각 설정 1회 실행.

| Config | Peak VRAM (GB) | Tokens/sec | TTFT (s) | Perplexity (WikiText-2) |
|---|---|---|---|---|
| FP16 (baseline) | 7.53 | 9.10 | 0.33 | 9.54 |
| NF4 4-bit (bitsandbytes) | 2.90 | 3.35 | 0.60 | 9.81 |
| GGUF Q4_K_M (llama.cpp) | –** | 29.54 | 0.31 | n/a* |

\* GGUF perplexity는 llama.cpp 자체 구현으로 계산되어 FP16/NF4와 동일한 방법이 아니므로 직접 비교하지 않음 (자세한 내용은 [Limitations](#limitations) 참고).

\*\* llama.cpp는 torch 스타일의 VRAM 측정 API를 노출하지 않아 미측정.

NF4는 VRAM을 61% 줄였지만(7.53GB → 2.90GB) MPS에서는 FP16보다 오히려 느렸다(0.37배 속도, 즉 약 2.7배 느림) — 커널 단위 프로파일링은 하지 않았지만, 4-bit 디퀀타이즈 연산의 오버헤드가 대역폭 절감분을 상쇄했을 가능성이 높다. GGUF Q4_K_M은 llama.cpp/Metal의 최적화된 커널 덕분에 FP16 대비 3.25배 빠른 속도를 냈다. NF4 perplexity는 FP16 대비 +2.8%(9.54 → 9.81)로 품질 저하는 미미했다.

### Context-length scaling

프롬프트 길이 128 → 4096 토큰에서 각 설정의 tokens/sec, TTFT:

| Context (tokens) | FP16 tok/s | NF4 tok/s | GGUF tok/s | FP16 TTFT (s) | NF4 TTFT (s) | GGUF TTFT (s) |
|---|---|---|---|---|---|---|
| 128 | 8.90 | 3.29 | 30.53 | 0.55 | 0.75 | 0.40 |
| 512 | 8.00 | 2.98 | 28.87 | 1.64 | 2.05 | 1.57 |
| 1024 | 6.99 | 2.84 | 27.34 | 3.86 | 4.10 | 3.46 |
| 2048 | 1.87 | 2.11 | 24.33 | 6.84 | 8.34 | 7.68 |
| 4096 | 0.28 | 0.10 | 19.95 | 14.83 | 17.72 | 17.58 |

같은 구간의 Peak VRAM(GB) — `generate()` 실행 중 20ms 간격으로 폴링해 실제 최댓값을 추적(자세한 내용은 [Limitations](#limitations) 참고):

| Context (tokens) | FP16 VRAM | NF4 VRAM |
|---|---|---|
| 128 | 7.54 | 2.88 |
| 512 | 7.75 | 3.11 |
| 1024 | 7.99 | 3.17 |
| 2048 | 8.10 | 3.30 |
| 4096 | 8.41 | 4.02 |

VRAM은 컨텍스트가 늘어날수록 예상대로 증가한다(FP16 7.54→8.41GB, NF4 2.88→4.02GB — NF4가 더 가파르게 느껴지지만 이는 baseline이 작기 때문이고 절대 증가량 자체는 FP16이 더 큼).

⚠️ **2048/4096 토큰 구간의 FP16/NF4 tok/s는 시스템 메모리 압박(스왑)의 영향을 크게 받았습니다.** 측정 당시 이 Mac(24GB 통합 메모리)에서 다른 애플리케이션들이 메모리를 많이 점유하고 있었고, 측정 도중 스왑 사용량이 23.6GB/24.5GB까지 찼습니다. 실제로 늘어난 VRAM 사용량과 맞물려 디스크 스와핑이 발생하면서 tok/s가 6.99→0.28(FP16), 2.84→0.10(NF4)로 붕괴했습니다 — MPS 연산 자체가 느려진 게 아니라 메모리 압박에 의한 병목입니다. 반대로 GGUF는 27.3→19.9 tok/s로 완만하게만 저하됐는데, llama.cpp가 GGUF 파일을 mmap으로 매핑해 페이지 단위로만 물리 메모리에 올리기 때문에 같은 메모리 압박에서도 훨씬 강건했던 것으로 보입니다. 이 결과는 측정 시점 다른 애플리케이션들의 메모리 사용량에 의존적이므로, 조용한(quiescent) 머신에서 재측정하면 FP16/NF4의 저하폭이 더 완만하게 나올 수 있습니다.

## Related benchmarks (sanity check)

이 프로젝트와 똑같은 조합(FP16 vs bitsandbytes-NF4 vs GGUF, 이 모델 기준)을 그대로 측정한 공개 자료는 찾지 못했지만, 양자화의 일반적인 트레이드오프는 이미 여러 곳에 보고돼 있습니다. 아래 수치와 우리 결과가 같은 방향(대략적인 절감률/품질 저하폭)을 가리키는지 리포트에서 교차 검증합니다.

- Unsloth GGUF Benchmarks — Q4_K_M vs Q5_K_M perplexity 비교
- 커뮤니티 보고: Q4_K_M은 FP16 대비 용량 약 75% 감소, MMLU 등 품질 저하는 1~3% 수준

## Limitations

- **GGUF perplexity는 FP16/NF4와 다른 방식으로 계산됨** (llama.cpp 내부 구현 vs HuggingFace loss 기반). 숫자를 나란히 두되 "직접 비교 불가"임을 명시.
- **단일 실행 기준**입니다 (반복 측정 후 평균/분산을 내지 않음). 엄밀히 하려면 각 설정을 3회 이상 반복해 평균과 표준편차를 리포트하는 게 좋음 — 시간이 허락하면 추가 예정.
- **실측 하드웨어는 Apple M3(macOS, unified memory)**이며 원본 README/노트북이 가정한 CUDA GPU(Colab T4/L4 등)가 아님. FP16/NF4는 PyTorch MPS backend, GGUF는 llama.cpp Metal backend로 실행. CUDA 환경에서의 수치는 이 결과와 다를 수 있음 (특히 NF4의 상대 속도 — bitsandbytes 4-bit 커널은 CUDA에서 최적화가 더 잘 되어 있어 MPS에서 관찰된 "NF4가 FP16보다 느림" 현상이 CUDA에서는 재현되지 않을 수 있음).
- **MPS에서의 Peak VRAM은 폴링 기반 근사치**입니다. PyTorch의 MPS backend는 CUDA의 `reset_peak_memory_stats()`/`max_memory_allocated()`에 대응하는 reset 가능한 peak 카운터를 제공하지 않습니다. `generate()` 실행 중 백그라운드 스레드가 20ms 간격으로 `torch.mps.current_allocated_memory()`를 폴링해 최댓값을 추적하는 방식(`PeakMemoryTracker`)으로 근사했으며, 폴링 간격보다 짧게 스파이크하는 할당은 놓칠 수 있습니다. (초기 버전은 `generate()` 종료 *후*에 한 번만 샘플링해 KV 캐시가 이미 해제된 값을 쟀는데, 이 때문에 컨텍스트 스윕에서 VRAM이 거의 변화 없이 보였습니다 — 코드 리뷰로 발견해 수정.)
- **컨텍스트 스윕의 2048/4096 토큰 구간은 시스템 메모리 압박(스왑)의 영향을 받았습니다.** 자세한 내용은 [Context-length scaling](#context-length-scaling) 참고. 조용한 머신에서 재측정하면 FP16/NF4의 저하폭이 이보다 완만할 수 있습니다.

## Next steps

1. ~~Apple M3(MPS/Metal)에서 `--stage all` 및 `--stage context_sweep` 실행 → `results/results.jsonl` 확보~~ (완료)
2. ~~수치를 이 README와 `report/technical_report_template.md`에 반영~~ (완료)
3. (선택) 다른 애플리케이션을 모두 닫은 조용한 상태에서 `--stage context_sweep` 재실행 → 스왑 영향 없는 2048/4096 tok/s 확보
4. (선택) CUDA GPU(Colab 등)에서 재실행해 MPS 대비 결과 교차검증, 특히 NF4 상대 속도
5. GitHub에 push → Zenodo 연동 → 릴리즈 태그 → DOI 발급

## Citation

`CITATION.cff` 참고 (저자명/저장소 주소는 TODO로 표시되어 있으니 채워 넣을 것).

## License

Code: MIT (`LICENSE` 참고). Report/데이터: CC-BY-4.0 권장.
