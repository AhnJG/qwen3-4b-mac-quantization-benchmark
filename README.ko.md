# 온디바이스 LLM 양자화 벤치마크 — MacBook에서 Qwen3-4B 비교

[English](README.md)

MacBook Air(M3)에서 언어 모델 Qwen3-4B를 **실행 프로그램 2가지(transformers, llama.cpp) × 숫자 정밀도 2가지(16비트, 4비트)**로 돌려, 속도와 언어 모델 품질(WikiText-2 perplexity)을 비교한 개인 연구 기록입니다. 200번 측정한 원자료와 분석 코드를 함께 공개합니다.

![Overview: design, data pipeline and key findings](figures/fig0_overview.png)

> 결과를 자세히 풀어 쓴 설명은 [docs/details.ko.md](docs/details.ko.md)에, 모든 표를 담은 보고서는 [docs/full-report.ko.md](docs/full-report.ko.md)에, 측정 실수를 찾아 고쳐 온 과정은 [docs/measurement-history.ko.md](docs/measurement-history.ko.md)에 있습니다.

## 핵심 결과

아래 속도 비율은 토큰 하나를 만드는 보통 시간(생성한 150개 토큰의 중앙값)을 비교한 것이고, 입력 길이 5가지(128–4096 토큰)의 기하평균입니다. 괄호 안은 입력 길이별 범위입니다. 「결과」의 표는 128 토큰만 보여 주므로, 표의 숫자끼리 나누면 이 범위 안의 값 하나가 나옵니다.

- **둘 다 4비트라고 부르지만 결과는 정반대였습니다.** 메모리 24GB에서 llama.cpp 4비트는 자기 16비트보다 2.58× 빨라졌고(2.35–2.84×), transformers 4비트는 오히려 2.48× 느려졌습니다(16비트 속도의 0.36–0.46×). 품질 손실도 llama.cpp 4비트가 더 작았습니다(perplexity +2.13% 대 +2.77%). 다만 두 4비트 방식은 세부 구성이 달라, 품질 차이의 일부는 그 때문으로 의심됩니다(「한계」 참고).
- **"3.3배 빨라짐"의 대부분은 4비트 덕분입니다.** transformers 16비트에서 llama.cpp 4비트로 바꾸면 3.28× 빨라집니다(2.79–3.67×). 16비트에서 출발해 두 단계로 나눠 보면 프로그램 교체가 1.27×, 4비트 전환이 2.58×입니다. 입력 길이별로는 각각 1.16–1.49×, 2.35–2.84×입니다. 다만 4비트의 효과가 프로그램에 따라 달라서, 이 분해가 각 요인의 독립적인 기여를 뜻하지는 않습니다.
- **transformers 16비트가 입력 2048 토큰에서 급격히 느려진 것은 대부분 드문 멈춤 때문이었습니다.** 메모리 24GB에서 1024 → 2048로 늘리면 평균 속도는 3.38× 느려졌지만, 토큰 하나의 보통 시간(중앙값)은 1.28×만 느려졌습니다. 150개 토큰 중 몇 개가 몇 초씩 멈췄기 때문입니다. 멈춤은 실제로 기다리는 시간이라, 평균(전체 대기시간)과 중앙값(보통 토큰의 지연)을 함께 봐야 합니다.
- **메모리가 16GB로 줄면 차이가 분 단위로 벌어집니다.** 같은 작업(입력 2048 토큰, 150 토큰 생성)이 llama.cpp 4비트는 14초, transformers 16비트는 실행에 따라 4.4–7.7분 걸렸습니다. 이 transformers 결과는 반복마다 가장 크게 흔들렸으므로 범위로 읽어야 합니다.

## 무엇을 비교했나

|  | transformers | llama.cpp |
|---|---|---|
| **16비트** | transformers 16비트 (FP16) — 기준 | llama.cpp 16비트 (F16) |
| **4비트** | transformers 4비트 (NF4) | llama.cpp 4비트 (Q4_K_M) |

- **transformers**: Hugging Face가 만든 파이썬 라이브러리입니다. 연구에서 모델을 돌릴 때 가장 흔히 씁니다.
- **llama.cpp**: C/C++로 만든 가벼운 모델 실행 프로그램입니다. 노트북과 PC에서 많이 쓰고, 모델 파일 형식으로 GGUF를 씁니다.
- **16비트(FP16 / F16)**: 숫자 하나를 16비트로 저장하는 원래 정밀도입니다. **4비트**는 가중치 대부분을 4비트로 줄여 저장하는 것(양자화)이고, transformers의 방식이 NF4, llama.cpp의 방식이 Q4_K_M입니다. 둘 다 전부 4비트는 아닙니다. Q4_K_M은 일부 가중치를 6비트로 남기고, NF4는 임베딩층과 출력층을 16비트로 둡니다.
- **llama.cpp 16비트를 넣은 이유**: "transformers 16비트"와 "llama.cpp 4비트"만 비교하면 프로그램과 비트 수가 한꺼번에 바뀌어, 빨라진 이유가 어느 쪽 덕인지 알 수 없습니다. 가운데에 llama.cpp 16비트를 넣으면 두 효과를 따로 볼 수 있습니다.
- **메모리 조건을 두 가지로 나눈 이유**: 가진 Mac은 24GB 한 대뿐인데, 많은 사람이 쓰는 16GB Mac에서는 어떻게 되는지도 보고 싶었습니다. 그래서 다른 프로그램이 8GB를 계속 붙잡게 해 모델이 쓸 수 있는 메모리를 줄였습니다. 같은 기계와 같은 측정 방법에서 의도한 차이는 8GB를 추가로 차지한 것뿐입니다. 또 Chrome과 Claude 앱은 일부러 켜 두었습니다. 작은 모델은 다른 프로그램과 함께 쓰일 때가 많다고 봤기 때문입니다. 다만 이 프로그램들을 얼마나 썼는지는 기록하지 않았습니다(「한계」 참고).

## 결과

![Both called 4-bit, opposite directions](figures/fig1_same_4bit_opposite.png)

입력 128 토큰, 메모리 24GB, 5회 측정의 중앙값입니다.

| 방식 | 최대 메모리 (GiB) | 생성 속도 (토큰/초) | 첫 토큰까지 (초) | perplexity (낮을수록 좋음) |
|---|---|---|---|---|
| transformers 16비트 (FP16), 기준 | 7.54 | 9.74 | 0.48 | 9.5413 |
| transformers 4비트 (NF4) | 2.88 | 3.48 | 0.67 | 9.8060 |
| llama.cpp 16비트 (F16) | –\* | 11.31 | 0.43 | 9.5417 |
| llama.cpp 4비트 (Q4_K_M) | –\* | 31.92 | 0.46 | 9.7446 |

\* llama.cpp는 PyTorch식 메모리 측정 기능이 없어 재지 못했습니다.

![The 2048-token cliff came mostly from rare stalls](figures/fig2_cliff_was_metric.png)

입력 길이별 비교, 메모리 16GB 결과, 멈춤이 왜 생기는지에 대한 진단 실험은 [docs/details.ko.md](docs/details.ko.md)에 있습니다.

## 실험 환경

| 항목 | 내용 |
|---|---|
| 기기 | MacBook Air (Apple M3, CPU 8코어 · GPU 10코어) |
| 메모리 | 24GB 통합 메모리(대역폭 100GB/s). 16GB 조건은 8GB를 다른 프로그램이 차지해 흉내 냄 |
| 함께 실행된 프로그램 | Chrome, Claude 앱. 일상 사용 환경에 가깝게 일부러 켜 둠. 메모리·CPU 사용량은 기록되지 않음 |
| 실행 환경 | transformers: PyTorch MPS · llama.cpp: Metal 빌드 |
| 소프트웨어 | Python 3.12, torch 2.13.0, transformers 5.14.1, bitsandbytes 0.50.0, accelerate 1.14.0, llama-cpp-python 0.3.34, datasets 5.0.1, numpy 2.5.1. 측정에 쓴 가상환경에서 나중에 읽은 버전입니다. 전체 목록: [environment/requirements-measured.txt](environment/requirements-measured.txt) |
| 모델 | transformers: `Qwen/Qwen3-4B-Instruct-2507` · llama.cpp: `unsloth/Qwen3-4B-Instruct-2507-GGUF`의 F16·Q4_K_M 파일(판본과 파일 해시: [environment/model-files.txt](environment/model-files.txt)) |
| 측정 코드 | v6 커밋([`570caaa`](https://github.com/AhnJG/qwen3-4b-mac-quantization-benchmark/commit/570caaa)) 이후의 작업본. 같은 계측 코드는 다음 날 [`f83d79e`](https://github.com/AhnJG/qwen3-4b-mac-quantization-benchmark/commit/f83d79e)로 커밋 |
| 측정 | 2026-08-09. 4가지 방식 × 입력 길이 5가지(128–4096 토큰) × 메모리 2가지 × 5회 = 200회. 측정마다 150 토큰 생성, 측정 순서는 메모리 조건마다 무작위 |
| 품질 측정 | WikiText-2 테스트 세트 전체(299,078 토큰)의 perplexity. 답변의 정확도나 지시 수행 능력은 평가하지 않았습니다 |

## 재현 방법

GPU 없이도 저장소에 있는 원자료로 속도 표와 그림 두 장을 다시 만들고, README의 핵심 숫자를 원자료와 대조할 수 있습니다. 몇 초면 끝나고 Python 3과 matplotlib만 있으면 됩니다.

```bash
python tools/audit.py --repeats 5   # check the data (sleep gaps, missing runs, variance)
python tools/analyze.py             # speed tables
python tools/figures.py             # the two figures
python tools/claims.py --check      # key README numbers, checked sentence by sentence against the raw data
```

Apple Silicon Mac에서 처음부터 다시 측정하는 방법(원래 측정은 24GB 약 1.3시간, 16GB 약 4.1시간)은 [docs/reproduce.ko.md](docs/reproduce.ko.md)에 있습니다.

## 저장소 구성

```
benchmark_quantization.py   # the benchmark: context_sweep and perplexity stages
tools/
  audit.py                  # data check: sleep gaps, missing or short runs, token counts, variance
  analyze.py                # speed tables from results/results.jsonl
  figures.py                # the README figures (overview numbers come from claims.py)
  claims.py                 # every README number with its source; --check compares it with the docs
  cliff_probe.py            # diagnostic tests for the stalls
  output_agreement.py       # where the four setups' outputs start to differ
  ballast.py, pressure_probe.py   # holding memory to imitate a smaller Mac; mmap on/off test
results/                    # raw data (see results/README.md for every field)
  results.jsonl             # 200 audited measurements (current data)
  results_perplexity_v7.jsonl
  results_memory_pressure.jsonl
  probe_*.log               # saved outputs of the diagnostic tests
  archive/                  # data from older, superseded versions
docs/
  details.md, details.ko.md # README results explained in detail
  reproduce.md, reproduce.ko.md # how to measure again from scratch
  ai-lessons.md, ai-lessons.ko.md # mistakes from how AI was used, and what changed
  full-report.ko.md         # every table (Korean)
  measurement-history.ko.md # measurement mistakes found and fixed (Korean)
  estimator-bias-note.ko.md # why "pick the fastest run" was biased (Korean)
figures/
```

원자료의 각 항목(필드)이 무엇을 뜻하는지는 [results/README.md](results/README.md)에 있습니다.

## 측정 기록

이 저장소는 v1부터 v8까지 여러 판본을 거쳤고, 지금 데이터는 v7(2026-08-09)에서 잰 것입니다. 그동안 측정 실수 여섯 가지가 발견되어 고쳐졌습니다. 그중 여러 개는 AI의 독립 검토가 찾았고, 어떤 지적을 받아들이고 무엇을 철회할지는 본인이 판단했습니다. 입력 처리 시간이 생성 속도에 섞인 것, GPU 시간을 잘못된 시점에 잰 것, 측정 중 컴퓨터가 잠든 것, 같은 설정을 몰아서 반복해 생긴 치우침, 평균이 멈춤을 가린 것, 반복 중 가장 빠른 값을 골라 생긴 치우침입니다. 각각 어떻게 찾고 고쳤는지는 [docs/measurement-history.ko.md](docs/measurement-history.ko.md)에 있습니다. AI를 쓰다가 방향을 잘못 잡거나 측정·문서에서 실수한 사례와, 그 뒤 바꾼 작업 방식은 [docs/ai-lessons.ko.md](docs/ai-lessons.ko.md)에 모았습니다.

## 역할 분담

- **본인**: 연구 질문 설정, AI가 제안한 실험 설계(엔진 × 정밀도 2×2)의 채택, 메모리 조건 선택(24GB, 그리고 8GB를 차지해 흉내 낸 16GB. 메모리를 차지하는 방식 자체는 AI 검토가 제안), 일상 앱을 일부러 켜 둔 채 본인 Mac에서 측정, 실험이 잘못된 방향으로 갈 때 중단하고 되돌리기, 근거가 부족한 주장의 철회와 공개 범위의 최종 판단
- **AI 코딩 도구(Claude, Codex)**: 실험 조사와 계획 초안, 측정·분석 코드 작성, 문서 초안 작성, 원자료 재계산
- **AI 검토**: 측정 결함과 과장된 표현을 찾는 독립 검토. 2×2 설계, 메모리 점유·mmap 대조 실험 같은 설계 변경과 주장 철회로 이어진 결함 발견도 여기서 나왔습니다. 어떤 지적을 받아들일지는 본인이 결정

## 한계

- Apple Silicon 한 기기(MacBook Air M3)에서만 쟀습니다. 메모리가 따로 있는 GPU에서도 같은지는 확인하지 않았습니다.
- 모델은 하나(Qwen3-4B)입니다. 16비트 버전이 16GB에 들어가야 해서 4B를 골랐습니다. 16GB Mac에서 무엇을 쓰라는 추천이 아니라, 원인을 나눠 보는 실험입니다.
- 전체 시간을 쟀고 GPU 코드 단위로 분석하지는 않았습니다. transformers 4비트가 느린 이유는 추정입니다.
- 반복은 5회이고 진단 실험은 설정당 1–2회입니다. 일부 비교는 이 횟수로는 판단할 수 없습니다(상세 설명에 표시).
- 두 "4비트" 방식은 가중치 하나당 비트 수가 같지 않습니다. Q4_K_M은 일부 가중치를 6비트로 남기고, NF4는 임베딩층과 출력층을 16비트로 둡니다. 가중치당 평균 비트 수는 재지 않았습니다. 이 차이는 품질 차이(+2.13% 대 +2.77%)의 일부 원인으로 의심됩니다. 다만 속도 차이의 주된 원인일 가능성은 낮습니다. transformers 4비트는 비트를 훨씬 많이 쓰는 llama.cpp 16비트보다도 느렸기 때문입니다(상세 설명 참고).
- llama.cpp 쪽은 스트리밍으로 받은 조각 하나를 토큰 하나로 셌습니다. 조각 하나에 토큰이 여러 개 들어갈 수 있어서, transformers의 토큰 단위와 정확히 같다는 보장은 없습니다(실제로 llama.cpp 측정 30회는 150개가 아니라 149개로 세어졌습니다).
- 16GB 조건은 24GB 기기에서 메모리를 일부 차지해 흉내 낸 것으로, 실제 16GB Mac과 완전히 같지는 않습니다.
- llama.cpp 4비트 파일이 품질 보정용 데이터(importance matrix)로 만들어졌는지 확인하지 못했습니다. 그렇다면 perplexity 비교에서 유리했을 수 있습니다.
- 측정 중 Chrome과 Claude 앱을 일부러 켜 두었지만, 얼마나 쓰였고 메모리를 얼마나 차지했는지는 기록하지 않았습니다. 첫 측정 직후 이미 스왑(메모리가 모자라 디스크로 밀려난 양)이 4.46GB(24GB 조건), 5.74GB(16GB 조건)였으므로, 모델이 실제로 쓸 수 있던 메모리는 24GB·16GB보다 적었을 가능성이 큽니다. 즉 두 조건은 정확한 사용 가능 메모리가 아니라 "24GB 기기 그대로"와 "거기서 8GB를 추가로 차지한 상태"입니다.
- 네 방식은 같은 시간대에 무작위 순서로 측정해 서로 비교한 결과는 비교적 안전합니다. 하지만 두 메모리 조건은 차례로 측정했습니다(24GB 15:30–16:47, 16GB 16:59–21:05). 그래서 두 조건의 차이에는 8GB를 추가로 차지한 효과와 다른 프로그램의 부하 변화가 섞여 있습니다. 멈춤의 크기와 16GB 조건의 분 단위 결과는 이 영향을 받았을 수 있습니다.
- 측정 당시 기록하지 않은 정보가 있습니다. macOS 버전, 패키지 목록(위 버전은 측정 환경에서 나중에 읽은 것), 정확한 코드 판본, transformers 모델 판본, WikiText-2 데이터셋 판본입니다.

## 라이선스

코드는 MIT([LICENSE](LICENSE))입니다. 데이터(`results/`), 문서(README, `docs/`), 그림(`figures/`)은 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)으로, 출처를 밝히면 자유롭게 쓸 수 있습니다. 문서 안의 코드 예시는 MIT를 따릅니다.
