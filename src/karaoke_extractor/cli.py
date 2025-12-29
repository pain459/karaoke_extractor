from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf


@dataclass
class AppError(Exception):
    message: str
    exit_code: int = 2

    def __str__(self) -> str:
        return self.message


def which_or_fail(bin_name: str) -> str:
    p = shutil.which(bin_name)
    if not p:
        raise AppError(
            f"Missing dependency: '{bin_name}'. Please install it and ensure it's in PATH.",
            3,
        )
    return p


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise AppError(f"Command failed ({e.returncode}): {' '.join(cmd)}", 10)


def to_snake_case(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = name.lower().strip("_")
    return name or "track"


def stamp_yyyymmdd() -> str:
    return datetime.now().strftime("%Y%m%d")


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested

    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def validate_input_file(inp: Path) -> None:
    if not inp.exists():
        raise AppError(f"Input file not found: {inp}", 4)
    if inp.is_dir():
        raise AppError(f"Input path is a directory, expected a media file: {inp}", 4)
    try:
        if inp.stat().st_size < 1024:
            raise AppError(f"Input file looks too small to be a valid media file: {inp}", 4)
    except OSError:
        raise AppError(f"Unable to read input file: {inp}", 4)


def ffmpeg_convert_to_wav(ffmpeg: str, inp: Path, out_wav: Path) -> None:
    cmd = [
        ffmpeg,
        "-y",
        "-vn",
        "-i",
        str(inp),
        "-ac",
        "2",
        "-ar",
        "44100",
        str(out_wav),
    ]
    run(cmd)


def audio_to_mp3(ffmpeg: str, inp_audio: Path, out_mp3: Path, bitrate: str) -> None:
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(inp_audio),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        bitrate,
        str(out_mp3),
    ]
    run(cmd)


def separate_with_demucs_library(
    wav_path: Path,
    model_name: str,
    device: str,
    out_dir: Path,
) -> tuple[Path, Path]:
    """
    Uses Demucs as a library and writes stems using soundfile (not torchaudio).
    Output: out_dir/vocals.wav and out_dir/other.wav
    """
    try:
        import torch  # type: ignore
        from demucs.apply import apply_model  # type: ignore
        from demucs.pretrained import get_model  # type: ignore
        from demucs.audio import AudioFile  # type: ignore
    except Exception as e:
        raise AppError(f"Failed to import Demucs dependencies: {e}", 20)

    out_dir.mkdir(parents=True, exist_ok=True)

    model = get_model(model_name)
    model.to(device)
    model.eval()

    # Load audio using Demucs' AudioFile helper; returns torch tensor [C, T]
    af = AudioFile(str(wav_path))
    wav = af.read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    # Add batch dim -> [1, C, T]
    wav = wav.unsqueeze(0)

    with torch.no_grad():
        # sources: [1, S, C, T]
        sources = apply_model(model, wav, device=device, progress=True)

    # Map sources to names
    # For htdemucs and most models, model.sources includes 'vocals', 'drums', 'bass', 'other', etc.
    source_names = list(getattr(model, "sources", []))
    if "vocals" not in source_names or "other" not in source_names:
        raise AppError(
            f"Model '{model_name}' does not provide expected sources. Found: {source_names}",
            21,
        )

    vocals_idx = source_names.index("vocals")
    other_idx = source_names.index("other")

    vocals = sources[0, vocals_idx]  # [C, T]
    other = sources[0, other_idx]    # [C, T]

    # Convert to numpy float32 for soundfile
    vocals_np = vocals.detach().cpu().numpy().T.astype(np.float32)  # [T, C]
    other_np = other.detach().cpu().numpy().T.astype(np.float32)    # [T, C]

    vocals_wav = out_dir / "vocals.wav"
    other_wav = out_dir / "other.wav"

    sf.write(str(vocals_wav), vocals_np, model.samplerate)
    sf.write(str(other_wav), other_np, model.samplerate)

    return vocals_wav, other_wav


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Extract vocals + instrumental (karaoke-style) from any media file and output MP3 stems."
    )
    ap.add_argument("input", help="Input audio/video file (any format supported by ffmpeg).")
    ap.add_argument("--outdir", default="outputs", help="Output directory.")
    ap.add_argument("--model", default="htdemucs", help="Demucs model (default: htdemucs).")
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Compute device for Demucs. 'auto' uses CUDA if available else CPU.",
    )
    ap.add_argument("--bitrate", default="192k", help="MP3 bitrate (e.g., 128k/192k/256k/320k).")
    ap.add_argument("--keep-temp", action="store_true", help="Keep temporary working directory.")
    args = ap.parse_args(argv)

    try:
        ffmpeg = which_or_fail("ffmpeg")

        inp = Path(args.input).expanduser().resolve()
        validate_input_file(inp)

        outdir = Path(args.outdir).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        base = to_snake_case(inp.stem)
        date_stamp = stamp_yyyymmdd()

        vocals_mp3 = outdir / f"{base}_{date_stamp}_vocals.mp3"
        inst_mp3 = outdir / f"{base}_{date_stamp}_instrumental.mp3"

        device = pick_device(args.device)
        print(f"Using device for Demucs: {device}")

        tmp_root = Path(tempfile.mkdtemp(prefix="karaoke_extract_"))
        try:
            tmp_wav = tmp_root / f"{base}.wav"
            stems_dir = tmp_root / "stems"

            print(f"Temp workspace: {tmp_root}")
            print("1) Converting input -> WAV via ffmpeg...")
            ffmpeg_convert_to_wav(ffmpeg, inp, tmp_wav)

            print("2) Separating stems via Demucs (library mode)...")
            vocals_wav, other_wav = separate_with_demucs_library(tmp_wav, args.model, device, stems_dir)

            print("3) Encoding stems to MP3...")
            audio_to_mp3(ffmpeg, vocals_wav, vocals_mp3, args.bitrate)
            audio_to_mp3(ffmpeg, other_wav, inst_mp3, args.bitrate)

            print("\n✅ Done.")
            print(f"Vocals:       {vocals_mp3}")
            print(f"Instrumental: {inst_mp3}")

        finally:
            if args.keep_temp:
                print(f"🧪 Kept temp directory: {tmp_root}")
            else:
                shutil.rmtree(tmp_root, ignore_errors=True)

    except AppError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(e.exit_code)
    except Exception as e:
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        raise SystemExit(99)


if __name__ == "__main__":
    main()
