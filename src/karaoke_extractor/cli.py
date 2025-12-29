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


@dataclass
class AppError(Exception):
    message: str
    exit_code: int = 2

    def __str__(self) -> str:
        return self.message


def which_or_fail(bin_name: str) -> str:
    p = shutil.which(bin_name)
    if not p:
        raise AppError(f"Missing dependency: '{bin_name}'. Please install it and ensure it's in PATH.", 3)
    return p


def run(cmd: list[str]) -> None:
    # Show commands for debuggability
    print(">>", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise AppError(f"Command failed ({e.returncode}): {' '.join(cmd)}", 10)


def to_snake_case(name: str) -> str:
    """
    Convert filename stem to snake_case.
    Keeps alnum, converts runs of non-alnum to underscores, trims.
    """
    name = name.strip()
    # Replace non-alphanumeric with underscore
    name = re.sub(r"[^A-Za-z0-9]+", "_", name)
    # Split camel-ish boundaries minimally
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = name.lower().strip("_")
    return name or "track"


def stamp_yyyymmdd() -> str:
    # Uses local machine date
    return datetime.now().strftime("%Y%m%d")


def pick_device(requested: str) -> str:
    """
    Demucs accepts: cpu | cuda | mps
    If requested is 'auto', choose cuda if torch.cuda.is_available() else cpu.
    """
    if requested != "auto":
        return requested

    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        # If torch isn't importable here for some reason, fall back
        pass

    return "cpu"


def ffmpeg_convert_to_wav(ffmpeg: str, inp: Path, out_wav: Path) -> None:
    # Make a consistent WAV for demucs
    cmd = [
        ffmpeg, "-y",
        "-vn",
        "-i", str(inp),
        "-ac", "2",
        "-ar", "44100",
        str(out_wav),
    ]
    run(cmd)


def demucs_separate(
    wav_path: Path,
    model: str,
    device: str,
    out_root: Path,
) -> tuple[Path, Path]:
    """
    Output layout from demucs:
      out_root/<model>/<track_stem>/{vocals.wav,other.wav}
    """
    track_name = wav_path.stem

    cmd = [
        sys.executable, "-m", "demucs",
        "-n", model,
        "--two-stems", "vocals",
        "--device", device,
        "-o", str(out_root),
        str(wav_path),
    ]
    run(cmd)

    stems_dir = out_root / model / track_name
    vocals = stems_dir / "vocals.wav"
    other = stems_dir / "other.wav"

    if not vocals.exists() or not other.exists():
        raise AppError(
            "Separation completed but expected stem files were not found. "
            "Try --keep-temp to inspect outputs.",
            11,
        )
    return vocals, other


def wav_to_mp3(ffmpeg: str, inp_wav: Path, out_mp3: Path, bitrate: str) -> None:
    cmd = [
        ffmpeg, "-y",
        "-i", str(inp_wav),
        "-vn",
        "-codec:a", "libmp3lame",
        "-b:a", bitrate,
        str(out_mp3),
    ]
    run(cmd)


def validate_input_file(inp: Path) -> None:
    # Basic file validation
    if not inp.exists():
        raise AppError(f"Input file not found: {inp}", 4)
    if inp.is_dir():
        raise AppError(f"Input path is a directory, expected a media file: {inp}", 4)
    try:
        if inp.stat().st_size < 1024:
            raise AppError(f"Input file looks too small to be a valid media file: {inp}", 4)
    except OSError:
        raise AppError(f"Unable to read input file: {inp}", 4)


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Extract vocals + instrumental (karaoke-style) from any media file and output MP3 stems."
    )
    ap.add_argument("input", help="Input audio/video file (any format supported by ffmpeg).")
    ap.add_argument("--outdir", default="outputs", help="Output directory.")
    ap.add_argument("--model", default="htdemucs", help="Demucs model (default: htdemucs).")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                    help="Compute device for Demucs. 'auto' uses CUDA if available else CPU.")
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
            demucs_root = tmp_root / "separated"

            print(f"Temp workspace: {tmp_root}")
            print("1) Converting input -> WAV via ffmpeg...")
            ffmpeg_convert_to_wav(ffmpeg, inp, tmp_wav)

            print("2) Running Demucs separation (vocals vs other)...")
            vocals_wav, other_wav = demucs_separate(tmp_wav, args.model, device, demucs_root)

            print("3) Encoding stems to MP3...")
            wav_to_mp3(ffmpeg, vocals_wav, vocals_mp3, args.bitrate)
            wav_to_mp3(ffmpeg, other_wav, inst_mp3, args.bitrate)

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
        # Catch-all for unexpected cases
        print(f"UNEXPECTED ERROR: {e}", file=sys.stderr)
        raise SystemExit(99)


if __name__ == "__main__":
    main()
