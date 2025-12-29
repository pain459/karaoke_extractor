# 🎤 Karaoke Extractor

A reliable, GPU-accelerated CLI tool to extract **vocals** and **instrumental (karaoke-style)** tracks from **any audio or video file**.

Built on top of **Demucs** (state-of-the-art music source separation), with a robust pipeline that avoids common `torchaudio` / codec issues and works cleanly on modern Python, CUDA, and Linux systems.

---

## ✨ Features

- ✅ Accepts **any media format** supported by `ffmpeg`
  - mp3, wav, flac, m4a, aac, ogg, mp4, mkv, webm, etc.
- ✅ Outputs **separate vocals and instrumental tracks**
- ✅ **MP3 output** (karaoke-ready)
- ✅ **Automatic GPU usage (CUDA)** with CPU fallback
- ✅ Deterministic, clean filenames:
  ```
  <original_name_snake>_yyyymmdd_vocals.mp3
  <original_name_snake>_yyyymmdd_instrumental.mp3
  ```
- ✅ Uses **Demucs as a library** (not CLI saving)
- ✅ Packaged as a **proper Python wheel**
- ✅ Clean temp handling and clear error messages

---

## 🧠 Why this exists

Demucs CLI works well interactively, but in automated pipelines it can break due to:
- `torchaudio` → `torchcodec` save-path changes
- silent failures when saving stems
- inconsistent output directory layouts

This tool:
- uses Demucs **only for separation**
- handles **audio I/O explicitly**
- produces predictable, reproducible outputs

Result: **boring, reliable, production-safe behavior**.

---

## 🏗 Architecture Overview

```
Input Media
   │
   ▼
ffmpeg (decode / normalize)
   │
   ▼
WAV (stereo, 44.1kHz)
   │
   ▼
Demucs (library mode, CUDA/CPU)
   │
   ▼
Torch tensors (vocals / other)
   │
   ▼
soundfile (write WAV stems)
   │
   ▼
ffmpeg (encode MP3)
```

### Key design decisions

- **Demucs library mode**
  - avoids torchaudio save-path instability
- **soundfile for WAV output**
  - stable, minimal dependency surface
- **ffmpeg for decode/encode**
  - widest format support
- **explicit device selection**
  - CUDA when available, CPU fallback

---

## 📦 Requirements

### System
- Linux (tested on Ubuntu)
- NVIDIA GPU (optional, recommended)
- `ffmpeg` installed and available in `PATH`

```bash
sudo apt update
sudo apt install -y ffmpeg
```

### Python
- Python **3.10+**
- Virtual environment strongly recommended

---

## 🚀 Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

This installs the CLI command:

```bash
karaoke-extract
```

---

## ▶️ Usage

### Basic

```bash
karaoke-extract "song.flac" --outdir outputs
```

### Output

```
outputs/
├── song_20251229_vocals.mp3
└── song_20251229_instrumental.mp3
```

### Video input

```bash
karaoke-extract "music_video.mp4"
```

---

## ⚙️ CLI Options

| Option | Description |
|------|-------------|
| `--outdir DIR` | Output directory (default: `outputs`) |
| `--model NAME` | Demucs model (default: `htdemucs`) |
| `--device auto|cpu|cuda|mps` | Compute device |
| `--bitrate 192k` | MP3 bitrate |
| `--keep-temp` | Preserve temp files for debugging |

---

## 🧪 GPU / CUDA Check

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📦 Build Wheel

```bash
pip install -r requirements-dev.txt
python -m build
```

Install elsewhere:

```bash
pip install karaoke_extractor-0.1.0-py3-none-any.whl
```

---

## 🧹 Temp Files

Temporary files are created under:

```
/tmp/karaoke_extract_<random>/
```

Automatically cleaned unless `--keep-temp` is used.

---

## ⚠️ Known Limitations

- Karaoke quality depends on the original mix
- ML separation is not perfect
- Runtime depends on track length and GPU availability

---

## 📄 License

MIT

---

## 🙌 Credits

- Demucs – Facebook AI Research
- ffmpeg
- soundfile
- PyTorch

---

## 🧠 Philosophy

> Make it boring. Make it reliable. Make it obvious.

This tool favors explicit behavior over clever shortcuts so it keeps working as dependencies evolve.
