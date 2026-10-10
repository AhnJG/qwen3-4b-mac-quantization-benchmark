# 온디바이스 LLM 양자화 벤치마크 — Apple Silicon에서의 Qwen3-4B

[English](README.md)

Apple M3 MacBook Air에서 `Qwen/Qwen3-4B-Instruct-2507`을 대상으로 **추론 engine × 가중치 정밀도**를 통제된 2×2 설계로 비교했습니다. 메모리 용량은 두 가지(24GB, 그리고 8GB를 고정(pinning)하여 모사한 16GB 머신)를 사용했습니다. 현재 데이터셋은 **감사(audit)를 마친 200회 실행**(4개 config × 5개 컨텍스트 길이 × 2개 용량 × 5회 반복)으로 구성됩니다. 모든 표와 그림은 원본 raw 행으로부터 명령어 하나로 다시 생성됩니다.

|  | HF transformers (PyTorch MPS) | llama.cpp (Metal) |
|---|---|---|
| **16-bit** | HF FP16 (baseline) | GGUF F16 |
| **4-bit** | HF NF4 (bitsandbytes) | GGUF Q4_K_M (K-quant) |

GGUF F16을 포함한 것은 두 효과를 분리하기 위해서입니다. "GGUF Q4_K_M 대 HF FP16"만 비교하면 engine과 정밀도가 동시에 바뀌기 때문입니다.

## 주요 결과

### 1. 같은 4-bit이 정반대 방향으로 움직입니다

![Same 4 bits, opposite directions](figures/fig1_same_4bit_opposite.png)

각 engine의 16-bit 가중치를 기준으로 할 때, llama.cpp의 **K-quant는 2.58× 더 빠르게 동작**(컨텍스트 길이별 2.35–2.84×)하는 반면, bitsandbytes의 **NF4는 2.48× 더 느리게 동작**(0.36–0.46×)합니다. 품질 손실도 K-quant 쪽이 더 작습니다. WikiText-2 perplexity는 9.7446 대 9.8060이며, 16-bit baseline 대비 +2.13% 대 +2.77%입니다. 이 실험 격자 안에는 NF4가 더 나은 선택이 되는 추론 설정이 없습니다. 이는 이 저장소에서 가장 견고한 결과로, 모든 컨텍스트 길이와 두 용량 모두에서, 그리고 시도한 모든 추정량(estimator)에서 성립합니다.

**순위를 결정하는 것은 가중치 크기가 아니라 커널 지원입니다.** HF NF4(가중치 2.9GB)는 두 용량 모두에서 GGUF F16(가중치 8GB)보다 약 **3.3× 느립니다**(컨텍스트 길이별 2.7–3.4×). K-quant는 dequantize와 matmul을 융합한 추론 커널을 사용하는 반면, NF4는 QLoRA *학습* 시 메모리를 절약하려는 목적으로 설계되었고 이에 상응하는 Metal 경로가 없기 때문일 가능성이 있습니다. 이는 커널 프로파일링이 아니라 아키텍처로부터 한 추론입니다.

### 2. "3.3× 빠르다"는 대부분 engine이 아니라 양자화의 효과입니다

| 24GB, steady-state 속도 비율 | 128 | 512 | 1024 | 2048 | 4096 | **기하평균** |
|---|---|---|---|---|---|---|
| Engine `GGUF-F16 / HF-FP16` | 1.16 | 1.22 | 1.30 | 1.49 | 1.19 | **1.27×** |
| K-quant `Q4_K_M / GGUF-F16` | 2.84 | 2.70 | 2.60 | 2.46 | 2.35 | **2.58×** |
| NF4 `HF-NF4 / HF-FP16` | 0.36 | 0.38 | 0.39 | 0.46 | 0.44 | **0.40×** |
| 전체 `Q4_K_M / HF-FP16` | 3.30 | 3.29 | 3.40 | 3.67 | 2.79 | **3.28×** |

전체 3.28×는 engine 효과 1.27×와 양자화 효과 2.58×로 분해됩니다. 그럼에도 engine은 다른 이유에서 여전히 중요합니다. 이 격자에서 4-bit이 실제로 이득이 되는 곳은 engine이 llama.cpp인 경우뿐이기 때문입니다(결과 1).

### 3. 컨텍스트 길이의 "절벽(cliff)"은 네 개 버전에서 재현되었지만 그 원인은 지표였습니다

![The 2048-token cliff was the metric](figures/fig2_cliff_was_metric.png)

v3부터 v6까지 HF FP16의 throughput은 1024 토큰에서 2048 토큰 사이에 급락했다가 4096 토큰에서 일부 회복되었고, 이 양상은 매번 재현되었습니다. 모든 decode step을 기록해 보니 원인이 드러났습니다. `tokens_per_sec = tokens / total time`은 **평균**이므로, 150개 step 중 수 초씩 걸리는 몇 번의 stall이 값을 좌우합니다. 1024 토큰에서 2048 토큰으로 갈 때 평균 기반 비용은 **3.38×** 증가하지만, step별 median은 **1.28×** 증가합니다. KV cache를 미리 할당(`StaticCache`)하면 stall이 사라집니다. HF의 기본값인 `DynamicCache`는 매 step마다 cache를 재할당하며, HF FP16 생성은 한 번 수행할 때마다 드라이버의 메모리 사용량을 머신의 전체 24GB까지 끌어올립니다. 이 한계 부근에서 stall이 OS의 메모리 회수 때문에 발생한다는 해석은 probe 결과와 부합하지만 증명된 것은 아닙니다.

이는 새로운 통찰이 아닙니다. 서빙 벤치마크에서 percentile 보고는 표준적인 관행이고(MLPerf Inference SingleStream은 p90을 사용합니다), Agrawal et al.(arXiv:2507.09019)과 *Etalon*(arXiv:2407.07000)은 생성 중 stall을 가리는 throughput 지표를 알려진 함정으로 지적하며, Mytkowicz et al.(ASPLOS '09)은 측정 편향이 어떻게 재현 가능한 잘못된 결론을 낳는지 보였습니다. 이 저장소가 기록하는 것은 이러한 관행이 아직 온디바이스 도구에는 도달하지 못했다는 점입니다. MLPerf Client는 평균 tokens/s를 보고하고, `llama-bench`는 반복 실행에 대한 평균 ± 표준편차를 보고합니다.

### 4. 메모리 압박 아래에서는 격차가 분 단위로 벌어집니다

16GB 머신에서 llama.cpp의 steady-state 속도는 전혀 반응하지 않습니다(20개 셀 모두에서 스톨 세금(stall tax) 1.0×). KV cache를 로드 시점에 한 번만 확보하고 가중치는 memory-map하기 때문입니다. 2048 토큰 프롬프트에서 150 토큰을 생성하는 데 HF FP16은 **약 7분**이 걸리며 그중 약 5분 44초가 stall 시간인 반면, llama.cpp Q4_K_M은 **14초**가 걸립니다. 메모리 압박이 위에서 본 NF4–F16 커널 격차를 만들어 내는 것은 아니며, stall을 통해 그 격차를 증폭시킵니다. (감사 결과, 16GB의 두 셀 FP16@2048과 NF4@4096은 반복 간 변동이 너무 커서 어떤 단일 요약값으로도 결론을 뒷받침할 수 없다고 표시되었습니다. 이 수치들은 예시로만 봐 주십시오.)

## 수치 산출 방법

- **Steady state = step별 decode latency p50**이며, 5회 반복에 대한 median을 취합니다. `tokens_per_sec`는 기록은 하지만 결론에는 사용하지 않습니다. 커널 비용과 stall이 섞여 있기 때문입니다.
- **스톨 세금(stall tax)** = (1000 / step p50) ÷ 평균 tokens/s. 값이 1.0×이면 평균이 왜곡되지 않았다는 뜻입니다.
- **비율은 다섯 개 컨텍스트 길이 전체에 대한 기하평균과 그 범위로 요약합니다.** 하나의 컨텍스트만 인용하는 것 자체가 추정량의 선택이 되기 때문입니다.
- 실행 순서는 전역적으로 섞었습니다. 각 행에는 타임스탬프, swap 사용량, 고정(pinned) 메모리가 기록되므로 오염(sleep, swap 포화)을 사후에 확인할 수 있습니다. `tools/audit.py`가 분석 전에 통과해야 하는 관문입니다.
- Perplexity: 네 개 config 모두 동일한 토큰 시퀀스(WikiText-2 test set 전체, 299,078 토큰)에 대해 동일한 함수로 점수를 매겼습니다. 서로 독립적인 두 F16 경로의 결과는 0.004% 이내로 일치합니다(9.5413 대 9.5417).

ctx=128, 24GB, 5회 실행의 median 기준 주요 수치입니다(네 config 모두 스톨 세금이 1.0×이므로 여기서는 평균과 steady state가 일치합니다).

| Config | 최대 메모리 (GB) | Decode tok/s | TTFT (s) | Perplexity |
|---|---|---|---|---|
| HF FP16 (baseline) | 7.54 | 9.74 | 0.48 | 9.5413 |
| HF NF4 (bitsandbytes) | 2.88 | 3.48 | 0.67 | 9.8060 |
| GGUF F16 | –\* | 11.31 | 0.43 | 9.5417 |
| GGUF Q4_K_M | –\* | 31.92 | 0.46 | 9.7446 |

\* llama.cpp는 torch 방식의 메모리 API를 제공하지 않아 측정하지 못했습니다.

## 측정 기록

현재 결과는 여덟 번째 버전입니다. v1–v8에 걸쳐 여섯 가지 측정 결함을 발견해 수정했으며, 각각이 결론을 바꾼 바 있습니다.

1. **Prefill 이중 수행** — `generate()`가 두 번 호출되어 prefill이 decode throughput에 섞여 들어갔습니다.
2. **비동기 타이밍** — MPS에서 `time.time()`은 GPU가 작업을 끝낸 시점이 아니라 CPU가 작업을 발행한 시점을 가리켰고, 이 때문에 prefill이 다시 섞여 들어갔습니다.
3. **Sleep 오염** — 8시간짜리 실행 한 번에서 6시간이 sleep 상태였습니다. 이제 모든 실행에 `caffeinate -i`를 사용하고 타임스탬프를 기록합니다.
4. **반복 블록 순서** — 반복을 블록 단위로 실행하면 모든 median이 21–25% 편향되었고, 전역 shuffle로 이를 ±4%까지 줄였습니다.
5. **지표** — 평균이 stall을 가렸습니다(결과 3).
6. **추정량** — "오염을 제거"하려고 반복 실행의 최솟값을 취한 것이 정작 그 비교 자체를 편향시켰고, 네 개의 주장을 철회했습니다.

각 대안 설명을 배제한 probe를 포함한 전체 기록은 [docs/measurement-history.ko.md](docs/measurement-history.ko.md)에 있습니다. 16GB 및 메모리 압박 실험을 포함한 모든 표는 [docs/full-report.ko.md](docs/full-report.ko.md)에 있습니다. 추정량 편향은 [docs/estimator-bias-note.ko.md](docs/estimator-bias-note.ko.md)에 정리했습니다. 이전 버전의 raw 데이터는 감사 추적(audit trail)으로 `results/archive/`에 보관합니다.

## 재현 방법

위의 모든 수치를 측정한 경로인 Apple Silicon(MPS / Metal) 기준입니다.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
CMAKE_ARGS="-DGGML_METAL=on" pip install --no-cache-dir --force-reinstall llama-cpp-python

# caffeinate -i is required: sleep silently corrupts wall-clock timing
caffeinate -i python benchmark_quantization.py --stage context_sweep --repeats 5                  # 24GB
caffeinate -i python benchmark_quantization.py --stage context_sweep --repeats 5 --ballast-gb 8   # 16GB
python benchmark_quantization.py --stage perplexity

python tools/audit.py --repeats 5   # integrity gate
python tools/analyze.py             # regenerates every table from results/results.jsonl
python tools/figures.py             # regenerates the two figures
```

**더 큰 머신에서는 stall 결과가 나타나지 않으며, 이는 예상된 동작입니다.** stall은 HF가 메모리를 머신의 물리적 한계까지 끌어올리기 때문에 발생합니다. 64GB 이상에서는 모든 곳에서 스톨 세금이 1.0×가 되어, 결과 1과 2(커널과 정밀도)는 재현되지만 결과 3과 4(stall)는 사라집니다. 압박 조건을 재현하려면 `--ballast-gb`로 메모리를 고정하여 유효 용량이 24GB 또는 16GB가 되도록 하십시오(64GB 머신에서는 `40`과 `48`).

## 저장소 구조

```
benchmark_quantization.py   # the sweep: context_sweep and perplexity stages
tools/
  audit.py                  # integrity gate: sleep gaps, incomplete cells, token counts, per-cell variance
  analyze.py                # regenerates every table from results/results.jsonl
  figures.py                # regenerates the README figures
  cliff_probe.py            # stall diagnostics (prompt, SDPA backend, per-step, cache, allocator, residue)
  output_agreement.py       # where the four configs' greedy outputs diverge
  ballast.py, pressure_probe.py   # memory-capacity simulation and mmap on/off A/B
results/
  results.jsonl             # 200 audited rows (current data)
  results_perplexity_v7.jsonl
  results_memory_pressure.jsonl
  probe_*.log               # saved probe outputs behind the causal claims
  archive/                  # data from superseded versions, kept as an audit trail
docs/                       # full report, measurement history, estimator note (Korean)
figures/
```

## 한계

- **Apple Silicon 통합 메모리(unified memory)에 한정됩니다.** CUDA 교차 검증은 v1(결함이 있던) 프로토콜에서만 존재하며 제외했습니다. mmap의 이점이 VRAM이 분리된 GPU에서도 유지되는지는 검증하지 못했습니다.
- **모델은 하나이며, 그 크기는 설계에 의해 정해졌습니다.** 16-bit 구성이 16GB에 들어가야 하므로 4B가 사용 가능한 가장 큰 크기입니다. 이 실험은 요인 분해이며, 16GB Mac에서 무엇을 실행해야 하는지에 대한 권고가 아닙니다.
- **End-to-end 타이밍이며 커널 프로파일링이 아닙니다.** 결과 1의 NF4에 대한 설명은 추론입니다.
- **n = 5.** 컨텍스트 길이가 두 정밀도에서 토큰당 동일한 절대 비용을 추가하는지와 같은 일부 비교는 roofline 예측과 부합하지만, bootstrap 구간이 너무 넓어 두 값이 같다고 판단할 수는 없습니다. 어떤 쌍이 미결정인지는 전체 보고서에 표시했습니다.
- **16GB 머신은 모사한 것입니다.** 24GB M3에서 상주 메모리를 고정하여 만들었으며, 메모리 대역폭과 코어 수는 같지만 완전히 동일하지는 않습니다.

## 라이선스

코드: MIT ([LICENSE](LICENSE)).
