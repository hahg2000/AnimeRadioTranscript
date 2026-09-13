from __future__ import annotations

from pathlib import Path
from typing import Any

from .audio import prepared_audio_path
from .io_utils import write_asr_segments, write_words
from .models import AsrSegment, Word

KOTOBA_MODEL_ID = "kotoba-tech/kotoba-whisper-v2.2"
KOTOBA_LOCAL_DIR = "/mnt/e/models/kotoba-whisper-v2.2"
MAX_SEGMENT_SECONDS = 28.0
SEGMENT_GAP_SECONDS = 0.5
SEGMENT_MAX_SECONDS = 15.0


def _load_kotoba_model(device: str | None = None) -> tuple[Any, Any]:
    import torch

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    source = KOTOBA_LOCAL_DIR if Path(KOTOBA_LOCAL_DIR).exists() else KOTOBA_MODEL_ID
    model = AutoModelForSpeechSeq2Seq.from_pretrained(source, torch_dtype=dtype).to(device)
    processor = AutoProcessor.from_pretrained(source)
    return model, processor


def _vad_segments(samples: Any, rate: int, max_seconds: float = MAX_SEGMENT_SECONDS) -> list[tuple[int, int]]:
    """Split audio into speech segments at silence, capping at max_seconds."""

    import librosa

    intervals = librosa.effects.split(samples, top_db=30, frame_length=2048, hop_length=512)
    if intervals.size == 0:
        return [(0, len(samples))]

    raw: list[tuple[int, int]] = []
    for start, end in intervals:
        raw.append((int(start), int(end)))

    merged: list[tuple[int, int]] = []
    for start, end in raw:
        if merged and start - merged[-1][1] < int(0.3 * rate):
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    max_samples = int(max_seconds * rate)
    segments: list[tuple[int, int]] = []
    for start, end in merged:
        position = start
        while end - position > max_samples:
            segments.append((position, position + max_samples))
            position += max_samples
        segments.append((position, end))
    return segments


def _transcribe_segment(model: Any, processor: Any, samples: Any, device: str, dtype: Any) -> list[dict]:
    import numpy as np
    import torch

    inputs = processor(np.asarray(samples, dtype=np.float32), sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device).to(dtype)
    with torch.inference_mode():
        generated = model.generate(
            input_features,
            language="ja",
            task="transcribe",
            return_timestamps=True,
        )
    result = processor.decode(generated[0], skip_special_tokens=True, output_offsets=True)
    offsets = result.get("offsets", []) if isinstance(result, dict) else []
    return [
        {
            "text": str(item.get("text") or "").strip(),
            "start": float(item.get("timestamp", (0.0, 0.0))[0]),
            "end": float(item.get("timestamp", (0.0, 0.0))[1]),
        }
        for item in offsets
        if str(item.get("text") or "").strip()
    ]


def _words_to_segments(words: list[Word], chunk_id: str) -> list[AsrSegment]:
    segments: list[AsrSegment] = []
    current: list[Word] = []
    for word in words:
        if (
            current
            and (
                word.start - current[-1].end > SEGMENT_GAP_SECONDS
                or word.end - current[0].start > SEGMENT_MAX_SECONDS
            )
        ):
            segments.append(
                AsrSegment(
                    text="".join(item.text for item in current),
                    start=current[0].start,
                    end=current[-1].end,
                    language="ja",
                    chunk_id=chunk_id,
                )
            )
            current = []
        current.append(word)
    if current:
        segments.append(
            AsrSegment(
                text="".join(item.text for item in current),
                start=current[0].start,
                end=current[-1].end,
                language="ja",
                chunk_id=chunk_id,
            )
        )
    return segments


def run_kotoba_asr(
    out_dir: str | Path,
    *,
    device: str | None = None,
) -> tuple[Path, Path]:
    out = Path(out_dir)
    audio = prepared_audio_path(out)
    if not audio.exists():
        raise FileNotFoundError(f"Missing prepared audio: {audio}")

    import librosa
    import torch

    model, processor = _load_kotoba_model(device=device)
    effective_device = str(next(model.parameters()).device)
    dtype = torch.float16 if effective_device.startswith("cuda") else torch.float32

    waveform, rate = librosa.load(str(audio), sr=16000)
    vad_intervals = _vad_segments(waveform, rate)

    all_words: list[Word] = []
    all_segments: list[AsrSegment] = []
    for segment_index, (start_sample, end_sample) in enumerate(vad_intervals):
        chunk = waveform[start_sample:end_sample]
        if chunk.size == 0:
            continue
        offset = start_sample / rate
        offsets = _transcribe_segment(model, processor, chunk, effective_device, dtype)
        words: list[Word] = []
        for item in offsets:
            text = item["text"]
            if not text:
                continue
            words.append(
                Word(
                    text=text,
                    start=round(offset + item["start"], 3),
                    end=round(offset + item["end"], 3),
                    language="ja",
                    probability=None,
                )
            )
        chunk_id = f"kotoba_{segment_index:05d}"
        for word in words:
            word.chunk_id = chunk_id
        all_words.extend(words)
        all_segments.extend(_words_to_segments(words, chunk_id))

    all_words.sort(key=lambda item: (item.start, item.end))
    words_path = out / "asr_words.jsonl"
    segments_path = out / "asr_segments.json"
    write_words(words_path, all_words)
    write_asr_segments(segments_path, all_segments)
    return words_path, segments_path
