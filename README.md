# Live CCTV analytics (live RTSP camera, CPU)

Project folder: `D:\priya\Live CCTV analytics`

Watches a live CCTV camera and reports, for each use case, what it sees: counts, missing PPE, objects.
Every use case has its **own script**, so you can run one, several, or all of them.

| Use case | Script | Model | Raises an alert when |
|---|---|---|---|
| Person count | `detectors\person_count.py` | pretrained (auto-download) | people > `max_counts.person` |
| Cap | `detectors\cap_detection.py` | custom `models\cap.pt` | `no_cap` seen |
| Gloves | `detectors\gloves_detection.py` | custom `models\gloves.pt` | `no_gloves` seen |
| Button | `detectors\button_detection.py` | custom `models\button.pt` | (counts only) |
| Shoes | `detectors\shoes_detection.py` | custom `models\shoes.pt` | `no_shoes` seen |
| Mask | `detectors\mask_detection.py` | custom `models\mask.pt` | `no_mask` seen |
| Beard (unshaved face) | `detectors\beard_detection.py` | custom `models\beard.pt` | `beard` seen |
| Laptop, pen, diary, chair, mobile phone | `detectors\object_detection.py` | pretrained (laptop, chair, phone) + custom `models\pen_diary.pt` (pen, diary) | (counts only) |

Alert rules are in `config.yaml` (`alert_classes`, `max_counts`) and can be changed without touching code.

> **Important:** the pretrained model only knows person, laptop, chair and mobile phone.
> Cap, gloves, shoes, mask, beard, button, pen and diary need **custom weights**
> (download an existing one or train your own, see "Custom models"). Until the file exists,
> that detector is skipped with a clear message; the others keep working.

## How it works (step by step)

1. `common/stream.py` reads the RTSP stream in a background thread and keeps only the **newest** frame, so video never lags, and it reconnects if the camera drops.
2. On CPU, AI runs on every Nth frame (`runtime.frame_skip`). When several detectors run together they **take turns**, one per slot, so the PC is not overloaded.
3. `common/base.py` runs the YOLO model, keeps only the classes listed in `config.yaml`, and counts them.
4. An event is raised only if it is seen in `min_consecutive` AI runs in a row (fewer false alarms).
5. The event is saved to `outputs\events.csv`, a snapshot goes to `outputs\snapshots\`, and Telegram/email alerts are sent (not more often than `alerts.cooldown_seconds` per event).
6. Each detector also writes `outputs\stats\<name>.json` and `outputs\frames\<name>.jpg` about once a second; the Streamlit dashboard just reads those files.

## Install (Windows PowerShell)

Python 3.10 to 3.12 recommended.

```powershell
cd "D:\priya\Live CCTV analytics"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
copy .env.example .env
notepad .env
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

Put your camera URL in `.env` (`RTSP_URL=rtsp://user:pass@IP:554/...`). Common RTSP paths:
Hikvision `/Streaming/Channels/101`, Dahua `/cam/realmonitor?channel=1&subtype=0`, CP Plus `/cam/realmonitor?channel=1&subtype=1`. If the password has `@` or `#`, URL-encode it (`@` = `%40`).

## Run

```powershell
python tools\check_stream.py                    # 1. test the camera; saves test_frame.jpg
python detectors\person_count.py                # 2. first real test (no custom weights needed)
python detectors\object_detection.py            #    laptop / chair / mobile phone
python run_all.py                               # 3. everything that has weights
python run_all.py --only person_count,mask_detection
python run_all.py --exclude button_detection,mask_detection
python run_all.py --source 0                    # laptop webcam test
python run_all.py --source "D:\priya\Live CCTV analytics\test.mp4" --no-window
streamlit run dashboard\app.py                  # dashboard at http://localhost:8501
```

Press **q** in the video window to quit. Options on every script: `--source`, `--config`, `--no-window`, `--max-frames`.

## Alerts

Edit `.env` and set `enabled: true` in `config.yaml` under `alerts`.
- **Telegram:** create a bot with @BotFather, put the token in `TELEGRAM_BOT_TOKEN`, send the bot a message, and get your `TELEGRAM_CHAT_ID` from `https://api.telegram.org/bot<TOKEN>/getUpdates`.
- **Email:** `SMTP_*` and `ALERT_EMAIL_TO`. For Gmail use an App Password, not your normal password.
Alerts include the snapshot image.

## Custom models (cap, gloves, shoes, mask, beard, button, pen, diary)

**Option A: use existing weights.** Search Roboflow Universe or Hugging Face for a YOLOv8 model for the item, check its licence, and save it as e.g. `models\mask.pt`. Then see its class names:

```powershell
python tools\inspect_model.py models\mask.pt
```

If names differ from `config.yaml`, edit `classes:` or add a `rename:` map, for example:

```yaml
- weights: models/mask.pt
  classes: [Mask, NO-Mask]
  rename: {Mask: mask, NO-Mask: no_mask}
```

**Option B: train your own.** Collect images from your own camera (best accuracy), label them with Roboflow, CVAT or LabelImg (YOLO format), and arrange:

```
datasets\cap_detection\images\train\*.jpg   datasets\cap_detection\labels\train\*.txt
datasets\cap_detection\images\val\*.jpg     datasets\cap_detection\labels\val\*.txt
```

Class ids follow the `classes:` order in `config.yaml` (cap=0, no_cap=1). Then:

```powershell
python training\train.py --detector cap_detection --epochs 50
```

The best weights are copied to `models\cap.pt` automatically. CPU training is slow; for more than a few hundred images, run the same command on Google Colab (free GPU) and copy the `.pt` file back. Aim for 200+ labelled images per class, taken from the real camera angle.

**Training on Google Colab (free GPU, recommended for more than a few hundred images).** Use the ready-made notebook `training\colab_train.ipynb`:
1. Upload your Roboflow YOLOv11 dataset zip to Google Drive.
2. Open https://colab.research.google.com, choose File > Upload notebook, and select `training\colab_train.ipynb`.
3. Runtime > Change runtime type > T4 GPU. Edit the settings in cell 3 (zip path, output file name), then run the cells top to bottom.
4. Copy the downloaded `.pt` file into `models\` and check its class names with `python tools\inspect_model.py models\<file>.pt`.

Local training with several detectors sharing one dataset (for example gloves + shoes) uses `--dataset`, `--keep-dataset-names` and `--out`, see the top of `training\train.py`. `--fraction 0.3` uses 30% of the images to shorten CPU training.

## Files

```
config.yaml            camera, speed, alert and per-detector settings
.env.example           template for secrets (copy to .env)
run_all.py             run many detectors on one stream
detectors\*.py         one script per use case (8)
common\                stream reader, detector engine, runner, alerts, model loader
dashboard\app.py       Streamlit dashboard
training\train.py      custom model training    training\colab_train.ipynb  same on Colab (free GPU)
tools\check_stream.py  camera test        tools\inspect_model.py  list a model's classes
models\                weights (not committed)    outputs\  events, snapshots (not committed)
```

Output files: `outputs\events.csv` (time, detector, event, count, snapshot name), `outputs\snapshots\` (image of each event), `outputs\stats\` and `outputs\frames\` (live data for the dashboard).

## Tuning for CPU speed and accuracy

- Slow / laggy AI: raise `frame_skip` (e.g. 5), lower `imgsz` (480), run fewer detectors, or use the camera's sub-stream (lower resolution).
- Live picture in the dashboard: the detector plays the camera as a real video stream (about 10 pictures a second) on `http://127.0.0.1:8599/live.mjpg`, separate from the AI, so the picture stays smooth and does not flicker in full screen. Settings in `config.yaml -> runtime`: `live_stream` (false = still pictures instead), `live_interval` (0.1 = 10 per second, raise it on a slow PC), `stream_port`, `stream_host` (`127.0.0.1` = only this PC; `0.0.0.0` lets other computers on your network watch, and the stream has **no password**). If the port is busy the dashboard falls back to still pictures.
- A busy status file (Windows "Access is denied" while the dashboard reads it) no longer stops the detector; the write is retried or skipped.
- Alerts: an event is confirmed after `min_consecutive` AI checks, and a few missed checks in between (`runtime.streak_gap`, default 2) do not restart that count.
- Small objects (buttons, pens, gloves) missed: raise `imgsz` to 960 and place the camera closer; this costs speed.
- Too many false alarms: raise `conf` or `min_consecutive`. Missed detections: lower `conf`.
- Faster on Intel CPUs: `pip install openvino`, then `python training\train.py ... --export-openvino`, or export manually with `yolo export format=openvino`.

## Known limitations

- Tested here only with a stand-in model and a test video; it has not been run against a real camera or real custom weights. Run `tools\check_stream.py` first.
- Accuracy depends entirely on the custom weights. Free public models are often weak on CCTV footage; fine-tune on your own frames.
- "no_cap / no_gloves / no_shoes / no_mask" rely on the model having a "no_" class. A model that only detects the item (e.g. just `mask`) cannot tell you it is missing; use one trained with both classes.
- The detectors do not identify people; they only report what is visible per frame.

## Privacy

Snapshots and footage contain people's faces. `outputs\`, `datasets\`, videos and weights are git-ignored. Use a private repository, inform staff as your local rules require, and delete old snapshots regularly.

## Troubleshooting

| Problem | Fix |
|---|---|
| `cannot open source` / no frame | Check URL, credentials, IP, port 554; `ping` the camera; test the URL in VLC; encode special characters in the password |
| `weights not found ... Skipping` | That detector needs its `.pt` in `models\` (see Custom models) |
| `no model weights available` | Same as above; the detector was skipped |
| Window does not open | Install `opencv-python` (not `opencv-python-headless`); or use `--no-window` plus the dashboard |
| Image smeared / green blocks | Network packet loss; use the camera's sub-stream or a wired connection (TCP is already forced) |
| Class names do not match | Run `tools\inspect_model.py`, fix `classes:` or add `rename:` |
| `ModuleNotFoundError` | Activate `.venv` and run `pip install -r requirements.txt` |
