# Experimental ASR robustness workflow — Mac M-series CLI

Phiên bản chạy thẳng trên Mac không cần Colab. Mirror của `notebooks/01-04` nhưng đóng gói thành CLI tuần tự, không phụ thuộc Google Drive.

## Khi nào dùng `local/` thay vì `notebooks/`

| Tình huống                                    | Dùng                 |
| --------------------------------------------- | -------------------- |
| Colab Free hết quota                          | `local/`             |
| Dataset cần lưu local Mac để rerun nhanh      | `local/`             |
| Cần benchmark tốc độ trên M4 Pro vs Colab CPU | `local/`             |
| Fine-tuning model lớn cần GPU thật (T4/A100)  | `notebooks/` (defer) |

## Setup một lần

Đã có repo + `.venv` từ `uv sync`. Nếu chưa:

```bash
uv sync --extra dev --extra eval
```

`silero-vad` và `torchcodec` nằm trong base deps; `pyahocorasick` chỉ có trong
extra `eval` vì BoH là tooling nghiên cứu, không phải production runtime.

## Pipeline đầy đủ — 5 CLI tuần tự

```
collect_noise   →  experimental BoH build  →                  ┐
download_fleurs →  calibrate_thresh → eval_table7 (benchmark) ┘
```

### Bước 1: Thu thập noise (~3-10 phút)

```bash
uv run python -m local.collect_noise
```

Mặc định: 500 ESC-50 (stream, loại category có voice) + 300 synthetic (silence/white/pink). Tổng 800.

Output:

```
data/noise_for_boh/wav/*.wav
data/noise_for_boh/manifest.jsonl
data/noise_for_boh/noise_collection_config.json
```

Options:

```bash
uv run python -m local.collect_noise --target 200      # smoke
uv run python -m local.collect_noise --force            # rebuild even if manifest exists
uv run python -m local.collect_noise --seed 7           # different RNG
```

### Bước 2: Build BoH (~10-25 phút trên M4 Pro với CoreML)

```bash
uv run python -m eval.experimental.asr_boh.build
```

Output:

```
data/asr/boh/phowhisper_tiny_vi_boh_v1.json     # research artifact, model-specific
notebooks/outputs/{RUN_ID}/logs/boh_runs/phowhisper_tiny/phowhisper_noise_outputs.jsonl
notebooks/outputs/{RUN_ID}/config_snapshot.json
```

Options thường dùng:

```bash
uv run python -m eval.experimental.asr_boh.build --max-files 20
uv run python -m eval.experimental.asr_boh.build --providers cpu
uv run python -m eval.experimental.asr_boh.build \
    --model phowhisper_tiny --model phowhisper_base
uv run python -m eval.experimental.asr_boh.review \
    --boh-path data/asr/boh/phowhisper_tiny_vi_boh_v1.json
```

### Bước 3: Tải FLEURS vi speech (~2-3 phút)

Speech eval set tiếng Việt để calibrate threshold + benchmark.

```bash
uv run python -m local.download_fleurs --target 200
```

Output:

```
data/fleurs_vi/wav/*.wav
data/fleurs_vi/manifest.jsonl
data/fleurs_vi/fleurs_download_config.json
```

### Bước 4: Calibrate threshold heuristics (~5 giây)

Đo phân bố `repetition_ratio`, `n_gram_repetition`, `chars_per_100ms` trên FLEURS ground truth. Recommended threshold = p99 + margin.

```bash
uv run python -m local.calibrate_thresholds
```

Output:

```
data/asr/threshold_calibration.json
```

Sau khi xong, đọc giá trị recommended từ table, paste vào defaults trong [soca/asr/hallucination_heuristics.py](../soca/asr/hallucination_heuristics.py) hoặc pass kwargs khi gọi `check_heuristics()`.

### Bước 5: Table VII benchmark (~10-30 phút trên M4 Pro)

So sánh 6 config: raw, deloop, vad, boh, deloop_boh, vad_deloop_boh. Đây là **deliverable trung tâm** của D2.5 cho CV.

```bash
uv run python -m local.eval_table7 --n-speech 50 --n-noise 20
```

Output:

```
eval/results/table7_replication.json
```

Sample output:

```
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Config              ┃    WER ┃   CER ┃ Halluc rate ┃ Lat p50 ms ┃ Lat p95 ms ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ (1) Raw ASR         │  ~23%  │ ~12%  │  ~30-50%    │  ~700 ms   │  ~1200 ms  │
│ (3) Silero VAD only │  ~25%  │ ~13%  │   ~0-3%     │  ~500 ms   │  ~1200 ms  │
│ (6) Production + experimental BoH │ ~25% │ ~13% │ ~0-2% │ ~500 ms │ ~1200 ms │
└─────────────────────┴────────┴───────┴─────────────┴────────────┴────────────┘
```

→ Đây là benchmark ablation nghiên cứu. Production `RobustASR` hiện chỉ dùng
VAD, confidence guard, de-loop và heuristics; BoH nếu bật chỉ được áp dụng
trong evaluator sau production để giữ khả năng so sánh lịch sử.

Options:

```bash
uv run python -m local.eval_table7                                # default 50+20
uv run python -m local.eval_table7 --n-speech 200 --n-noise 100  # serious run
uv run python -m local.eval_table7 \
    --configs production_no_boh,production_with_boh              # paired A/B
uv run python -m local.eval_table7 --providers cpu               # force CPU
```

## GPU trên Mac M4 Pro

ONNX Runtime trên macOS arm64 có 3 provider khả dụng:

```
['CoreMLExecutionProvider', 'AzureExecutionProvider', 'CPUExecutionProvider']
```

CLI ưu tiên `CoreMLExecutionProvider` (Apple Neural Engine + GPU), fallback `CPUExecutionProvider`. Nếu node ONNX không support CoreML, runtime tự fallback node đó về CPU — không crash.

So sánh ước lượng trên PhoWhisper-tiny (39M):

| Provider              | Latency/sample 5s noise | Tổng 800 file |
| --------------------- | ----------------------- | ------------- |
| CPU 4-thread          | ~150-200 ms             | ~30-40 phút   |
| CoreML + CPU fallback | ~60-100 ms              | ~10-25 phút   |

Silero VAD dùng PyTorch backend (CPU). Vì model VAD chỉ 1.8M params, chạy CPU ~30ms cho 5s audio — không lợi gì khi đẩy lên MPS.

## Sanity checklist sau khi build_boh chạy xong

```bash
ls data/asr/boh/

uv run python -c "
import json
data = json.load(open('data/asr/boh/phowhisper_tiny_vi_boh_v1.json'))
print(f\"Hallucination rate: {data['metadata']['hallucination_rate']:.2%}\")
print(f\"BoH size: {len(data['boh'])}\")
print('Top 10:')
for item in data['boh'][:10]:
    print(f\"  count={item['count']:3d}  '{item['phrase'][:80]}'\")
"
```

Kỳ vọng:

- Hallucination rate trên non-speech: **30-50%** (paper BoH báo 40.3% cho Whisper-large-v3 trên 301k file).
- Top 30 cover ~70-77% tổng hallucinations.
- BoH size sau filter count≥2, len≥5: **30-100 phrase** cho 800 sample.

Nếu rate <10% hoặc BoH size <5: pipeline có bug, đừng scale lên 800.

## Notebooks 05, 06 (fine-tuning, ONNX export) — deferred

Plan v3 (`zplan/asr_robustness_colab_plan.md` line 510) explicitly defers:

> Chỉ bắt đầu phần này sau khi pipeline robustness không cần training đã chạy ổn.

Lý do:

- Cần GPU thật (Colab Pro hoặc local NVIDIA), Mac M4 Pro Metal không support PyTorch training nhiều ops.
- Cần curated command-domain Vietnamese speech (chưa có).
- D2.5 robustness đã defendable mà không cần tuning — qua Table VII benchmark.

→ Khi nào D2.5 đủ deliver, mới mở 05 + 06.

## Quan hệ với `notebooks/`

`local/` và `notebooks/02` cùng config (model registry, MIN_COUNT, MIN_CHARS, normalization), output JSON cùng schema. BoH file từ `local/` và `notebooks/02` chỉ dùng cho evaluator/ablation; không được auto-load vào voice runtime. Metadata phân biệt `execution_mode` (`"local"` vs `"colab"`).

Notebooks 03/04 hiện tại là placeholder rỗng — `local/calibrate_thresholds.py` và `local/eval_table7.py` là implementation chính cho 2 deliverable đó. Khi nào cần share workflow trên Colab GPU thì port logic từ `local/` sang notebook.
