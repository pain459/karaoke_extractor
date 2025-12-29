# karaoke-extractor

Extract karaoke-style stems (vocals + instrumental) from any media file supported by ffmpeg.

## Requirements
- Python 3.10+
- ffmpeg installed and available in PATH

## Install (editable for dev)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
