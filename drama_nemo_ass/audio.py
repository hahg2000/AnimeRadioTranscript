from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .io_utils import ensure_dir


def prepared_audio_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "prepared.wav"


def prepared_mono_16k_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "prepared_mono_16k.wav"


def prepared_source_path(out_dir: str | Path) -> Path:
    return Path(out_dir) / "prepared_source.wav"


def prepare_audio(
    input_path: str | Path,
    out_dir: str | Path,
    sample_rate: int = 16000,
    *,
    preserve_source: bool = True,
) -> Path:
    """Create the ASR mono WAV and retain a decoded source-channel master."""

    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH; install ffmpeg before running prepare.")

    out = ensure_dir(out_dir)
    target = prepared_audio_path(out)
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-acodec",
        "pcm_s16le",
        str(target),
    ]
    subprocess.run(command, check=True)
    shutil.copyfile(target, prepared_mono_16k_path(out))

    if preserve_source:
        source_command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-map",
            "0:a:0",
            "-acodec",
            "pcm_s16le",
            str(prepared_source_path(out)),
        ]
        subprocess.run(source_command, check=True)
    return target
