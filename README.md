# 🎤 Karaoke Extractor

**Karaoke Extractor** is a GPU‑accelerated CLI tool that separates **vocals** and **instrumental (karaoke-style)** tracks from **any audio or video file**.

It is built on top of **Demucs** (state‑of‑the‑art music source separation) with a production‑safe architecture that avoids common `torchaudio` / codec issues and works reliably on modern Python, CUDA, and Linux systems.

---

## 🚀 Quick Start

```bash
# system dependency
sudo apt install -y ffmpeg

# python setup
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .

# extract karaoke tracks
karaoke-extract "song.flac" --outdir outputs
```

Result:
```
outputs/
├── song_20251229_vocals.mp3
└── song_20251229_instrumental.mp3
```

---

## ✨ Features

- 🎧 Works with **any media format** supported by `ffmpeg`
  - mp3, wav, flac, m4a, aac, ogg, mp4, mkv, webm, etc.
- 🎤 **Vocal + instrumental separation**
- 🎶 Karaoke‑ready **MP3 output**
- ⚡ **Automatic CUDA usage** with CPU fallback
- 🧼 Deterministic, clean filenames:
  ```
  <original_name_snake>_yyyymmdd_vocals.mp3
  <original_name_snake>_yyyymmdd_instrumental.mp3
  ```
- 🧠 Uses **Demucs as a library**, not CLI saving
- 📦 Distributed as a **Python wheel**
- 🧹 Automatic temp cleanup

---

## 🧠 Motivation

While Demucs CLI is excellent for manual use, it can be fragile in automated pipelines due to:

- `torchaudio` → `torchcodec` save‑path changes
- silent stem‑save failures
- inconsistent output directory layouts

This project intentionally:
- uses Demucs **only for separation**
- manages **audio I/O explicitly**
- prioritizes **predictability over cleverness**

Result: a boring, reliable tool suitable for scripts, cron jobs, and pipelines.

---

## 🏗 Architecture

```
Input media (audio / video)
        │
        ▼
ffmpeg ── decode & normalize
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
soundfile ── write WAV stems
        │
        ▼
ffmpeg ── encode MP3 outputs
```

### Design choices

- **Demucs library mode**
  - avoids torchaudio save instability
- **soundfile**
  - explicit, stable audio writes
- **ffmpeg**
  - widest input/output support
- **explicit device selection**
  - CUDA when available, CPU otherwise

---

## 📦 Requirements

### System
- Linux (tested on Ubuntu)
- NVIDIA GPU (optional but recommended)
- `ffmpeg` in `PATH`

```bash
sudo apt update
sudo apt install -y ffmpeg
```

### Python
- Python **3.10+**
- Virtual environment recommended

---

## 📥 Installation

### Development / local use

```bash
pip install -r requirements.txt
pip install -e .
```

### From wheel

```bash
pip install karaoke_extractor-0.1.0-py3-none-any.whl
```

---

## ▶️ Usage

```bash
karaoke-extract INPUT [options]
```

### Common options

| Option | Description |
|------|-------------|
| `--outdir DIR` | Output directory (default: `outputs`) |
| `--model NAME` | Demucs model (default: `htdemucs`) |
| `--device auto|cpu|cuda|mps` | Compute device |
| `--bitrate 192k` | MP3 bitrate |
| `--keep-temp` | Preserve temp files for debugging |

### Examples

Force CPU:
```bash
karaoke-extract song.mp3 --device cpu
```

High‑quality output:
```bash
karaoke-extract song.wav --bitrate 320k
```

Debug temp artifacts:
```bash
karaoke-extract song.flac --keep-temp
```

---

## 🧪 GPU / CUDA Check

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Expected output:
```
True
```

---

## 📦 Building the Wheel

```bash
pip install -r requirements-dev.txt
python -m build
```

Artifacts:
```
dist/
└── karaoke_extractor-0.1.0-py3-none-any.whl
```

---

## 🧹 Temporary Files

Temporary working directories are created under:

```
/tmp/karaoke_extract_<random>/
```

They are automatically removed unless `--keep-temp` is specified.

---

## ⚠️ Limitations

- Separation quality depends on the original mix
- Vocals embedded in instruments may leave artifacts
- ML‑based separation is not perfect by design

---

## 📄 License

MIT

---

## 🙌 Credits

- **Demucs** — Facebook AI Research
- **ffmpeg**
- **soundfile**
- **PyTorch**

---

## 🧠 Philosophy

> Make it boring. Make it reliable. Make it obvious.

This project favors explicit behavior and predictable outputs so it keeps working as dependencies evolve.
