# 처음부터 다시 측정하기

[README로 돌아가기](../README.ko.md) · [English](reproduce.md)

Apple Silicon Mac이 필요합니다. 명령은 저장소 최상위 폴더에서 실행합니다. 저장소의 원자료(`results/`)를 덮어쓰지 않도록 `--out`으로 새 폴더를 지정하세요.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # versions found afterwards in the measurement venv: environment/requirements-measured.txt
CMAKE_ARGS="-DGGML_METAL=on" pip install --no-cache-dir --force-reinstall llama-cpp-python

# caffeinate -i keeps the Mac awake: sleep silently corrupts timing
caffeinate -i python benchmark_quantization.py --stage context_sweep --repeats 5 --out results/rerun                  # 24GB
caffeinate -i python benchmark_quantization.py --stage context_sweep --repeats 5 --ballast-gb 8 --out results/rerun   # 16GB
python benchmark_quantization.py --stage perplexity --out results/rerun

python tools/audit.py results/rerun/results.jsonl --repeats 5
python tools/analyze.py results/rerun/results.jsonl
```

측정 환경에서 확인한 패키지 버전은 [environment/requirements-measured.txt](../environment/requirements-measured.txt)에, GGUF 모델 파일 정보는 [environment/model-files.txt](../environment/model-files.txt)에 있습니다.

## 걸리는 시간

2026-08-09 측정 기록 기준으로 24GB 측정 100회는 약 1.3시간, 16GB 측정 100회는 약 4.1시간 걸렸습니다. perplexity 측정 시간은 기록되지 않았습니다(코드 설명상 네 방식을 합쳐 몇 시간).

## 짧게 시험해 보기

반복 1회로 돌리면 24GB 측정의 5분의 1인 약 15분이 걸릴 것으로 예상합니다(직접 재 보지는 않음).

```bash
caffeinate -i python benchmark_quantization.py --stage context_sweep --repeats 1 --out results/smoke
python benchmark_quantization.py --stage perplexity --ppl-max-tokens 20000 --out results/smoke
python tools/audit.py results/smoke/results.jsonl --repeats 1
```

## 측정 조건

- 원래 측정에서는 Chrome과 Claude 앱을 일부러 켜 두었지만, 사용량은 기록하지 않았습니다(README의 「한계」 참고). 깨끗하게 비교하려면 다른 프로그램을 모두 닫거나, 정해진 프로그램만 켜 두고 측정 중 메모리 사용량을 기록하세요.
- 메모리가 훨씬 큰 기계에서는 멈춤 결과가 나오지 않을 것으로 예상합니다(직접 확인하지는 않음). 같은 조건을 만들려면 `--ballast-gb`로 메모리를 일부 차지해 24GB나 16GB에 맞추세요(64GB 기계라면 `40`과 `48`).
