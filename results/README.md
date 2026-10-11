# Raw data / 원자료

All numbers in the READMEs and `docs/` are computed from the files in this folder. Do not edit them by hand; new runs should go to a separate folder (`--out`, see the main README).

README와 `docs/`의 모든 숫자는 이 폴더의 파일에서 계산합니다. 손으로 고치지 마세요. 새로 돌린 결과는 `--out`으로 다른 폴더에 저장하세요(메인 README 참고).

## Files / 파일

| File | Contents | 내용 |
|---|---|---|
| `results.jsonl` | Current data: 200 speed measurements (v7, 2026-08-09), one JSON object per line | 현재 데이터: 속도 측정 200회(v7, 2026-08-09), 한 줄에 측정 하나 |
| `results_perplexity_v7.jsonl` | Perplexity of the four setups on the full WikiText-2 test set | 네 방식의 WikiText-2 테스트 전체 perplexity |
| `results_memory_pressure.jsonl` | v4 memory-shortage test with mmap on/off (30 rows, 1 run each, 40 tokens generated) | v4 메모리 부족 실험, mmap 켜고 끄기(30행, 조합당 1회, 40토큰 생성) |
| `probe_*.log`, `ppl_v7_stdout.log`, `sweep_v7_ballast8_stdout.log`, `vm_stat_*.log` | Saved text output of the diagnostic tests and runs | 진단 실험과 측정 실행의 출력 기록 |
| `archive/` | Data from older versions that had measurement mistakes, kept as a record. Not used for any current number | 측정 실수가 있던 예전 판본의 데이터. 기록으로만 보관하며 현재 숫자에는 쓰지 않음 |

## Fields in `results.jsonl` / 필드 설명

| Field | Unit | Meaning | 설명 |
|---|---|---|---|
| `config` | — | `FP16`, `NF4_4bit` (transformers), `GGUF_F16`, `GGUF_Q4_K_M` (llama.cpp) | 실행 방식 4가지 |
| `context_len` | tokens | Target input length: 128, 512, 1024, 2048, 4096 | 목표 입력 길이 |
| `prompt_tokens` | tokens | Actual input length including the chat template | 채팅 형식을 포함한 실제 입력 길이 |
| `repeat` | — | Repeat index (0–4) | 반복 번호 |
| `max_new_tokens` / `n_new` | tokens | Tokens requested (150) / actually generated | 생성 요청 토큰 수 / 실제 생성 토큰 수 |
| `ttft_s` | s | Time to first token (input processing) | 첫 토큰까지 걸린 시간(입력 처리) |
| `tokens_per_sec` | tokens/s | Average decode speed = (n_new − 1) ÷ time from first to last token. Includes stalls | 평균 생성 속도. 멈춤이 포함됨 |
| `step_ms_p50` | ms | Median time between consecutive tokens — the "normal speed" used for conclusions | 토큰 사이 시간의 중앙값 — 결론에 쓰는 "정상 속도" |
| `step_ms_p90` / `step_ms_max` | ms | 90th percentile / slowest time between consecutive tokens | 토큰 사이 시간의 90번째 백분위 / 가장 긴 값 |
| `peak_vram_gb` | GiB | transformers only: peak GPU memory allocated by PyTorch, sampled every 20 ms. `null` for llama.cpp | transformers만: PyTorch가 잡은 GPU 메모리 최댓값(20ms마다 확인). llama.cpp는 `null` |
| `swap_used_gb` | GiB | System swap in use when the measurement ended (`sysctl vm.swapusage`) | 측정이 끝난 시점의 시스템 스왑 사용량 |
| `ballast_gb` | GB | Memory held by the ballast program: `null` = 24GB condition, `8.0` = 16GB condition | 다른 프로그램이 차지한 메모리. `null` = 24GB 조건, `8.0` = 16GB 조건 |
| `started_at` / `ended_at` | local time | Start and end of the measurement (ISO 8601) | 측정 시작·끝 시각 |
| `device` | — | `mps` (Apple GPU) | 사용 장치 |
| `perplexity` | — | Always `null` in this file (see `results_perplexity_v7.jsonl`) | 이 파일에서는 항상 `null` |

These rows predate the v8 fields `step_ms` (every per-token time) and `schema`; those appear only in runs made with the current code.

이 200행은 v8에서 추가된 `step_ms`(토큰별 시간 전체)와 `schema` 필드가 생기기 전에 측정해서, 두 필드가 없습니다. 현재 코드로 새로 돌린 결과에만 들어갑니다.

## License / 라이선스

The data in this folder is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). 이 폴더의 데이터는 CC BY 4.0을 따릅니다(출처를 밝히면 자유롭게 사용 가능).
