from __future__ import annotations

from pathlib import Path
from typing import Any

from .audio import prepared_audio_path
from .io_utils import write_asr_segments, write_words
from .models import AsrSegment, Word
from .model_paths import resolve_faster_whisper_model


def _object_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def run_faster_whisper_asr(
    out_dir: str | Path,
    model_name: str = "large-v3",
    language: str = "ja",
    device: str = "cuda",
    compute_type: str = "float16",
    beam_size: int = 5,
    vad_filter: bool = False,
    condition_on_previous_text: bool = False,
    models_root: str | Path | None = None,
    audio_path: str | Path | None = None,
    words_filename: str = "asr_words.jsonl",
    segments_filename: str = "asr_segments.json",
) -> tuple[Path, Path]:
    """Run faster-whisper and write asr_words.jsonl plus asr_segments.json."""

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. Install the WSL dependencies from requirements-wsl.txt."
        ) from exc

    out = Path(out_dir)
    audio = Path(audio_path) if audio_path is not None else prepared_audio_path(out)
    if not audio.exists():
        raise FileNotFoundError(f"Missing prepared audio: {audio}")

    resolved_model = resolve_faster_whisper_model(model_name, models_root)
    try:
        model = WhisperModel(resolved_model, device=device, compute_type=compute_type)
    except Exception as exc:
        message = str(exc)
        if exc.__class__.__name__ == "LocalEntryNotFoundError" or "cannot find the appropriate snapshot" in message:
            raise RuntimeError(
                "Unable to download or locate the faster-whisper model. Check WSL network access to Hugging Face, "
                "set HF_ENDPOINT if you need a mirror, or pre-download Systran/faster-whisper-large-v3 and pass "
                "that local directory with --model."
            ) from exc
        raise
    raw_segments, _info = model.transcribe(
        str(audio),
        language=language,
        word_timestamps=True,
        beam_size=beam_size,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
    )

    words: list[Word] = []
    segments: list[AsrSegment] = []
    for index, raw_segment in enumerate(raw_segments):
        chunk_id = f"asr_{index:05d}"
        text = str(_object_value(raw_segment, "text", "")).strip()
        start = float(_object_value(raw_segment, "start", 0.0) or 0.0)
        end = float(_object_value(raw_segment, "end", start) or start)
        segments.append(AsrSegment(text=text, start=start, end=end, language=language, chunk_id=chunk_id))

        for raw_word in _object_value(raw_segment, "words", []) or []:
            word_text = str(_object_value(raw_word, "word", _object_value(raw_word, "text", ""))).strip()
            if not word_text:
                continue
            words.append(
                Word(
                    text=word_text,
                    start=float(_object_value(raw_word, "start", start) or start),
                    end=float(_object_value(raw_word, "end", end) or end),
                    language=language,
                    probability=(
                        None
                        if _object_value(raw_word, "probability", None) is None
                        else float(_object_value(raw_word, "probability"))
                    ),
                    chunk_id=chunk_id,
                )
            )

    words_path = out / words_filename
    segments_path = out / segments_filename
    write_words(words_path, words)
    write_asr_segments(segments_path, segments)
    return words_path, segments_path
