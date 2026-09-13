from __future__ import annotations

from pathlib import Path
from typing import Any

from .audio import prepared_audio_path
from .io_utils import write_asr_segments, write_words
from .model_paths import resolve_qwen_model
from .models import AsrSegment, Word

QWEN_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
QWEN_ANIME_MODEL_ID = "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf"
QWEN_MODEL_CHOICES = {QWEN_MODEL_ID, QWEN_ANIME_MODEL_ID}
MAX_SEGMENT_SECONDS = 28.0
SENTENCE_END_CHARS = set("。！？!?…")

PUNCTUATION_CHARS = set("。？！、，,.!?｡｢｣「」『』（）()・…—")


def _tokenize_words(text: str) -> list[str]:
    try:
        from sudachipy import dictionary
        from sudachipy import tokenizer
    except ImportError as exc:
        raise RuntimeError("sudachipy is required for word segmentation.") from exc
    tokenizer_obj = dictionary.Dictionary().create(mode=tokenizer.Tokenizer.SplitMode.C)
    return [morpheme.surface() for morpheme in tokenizer_obj.tokenize(text)]


def _merge_trailing_punctuation(words: list[Word]) -> list[Word]:
    merged: list[Word] = []
    for word in words:
        if merged and word.text and all(char in PUNCTUATION_CHARS for char in word.text):
            previous = merged[-1]
            merged[-1] = Word(
                text=previous.text + word.text,
                start=previous.start,
                end=max(previous.end, word.end),
                language=previous.language,
                probability=previous.probability,
                chunk_id=previous.chunk_id,
            )
        else:
            merged.append(word)
    return merged

try:
    from transformers import AutoModelForMultimodalLM, AutoProcessor  # type: ignore

    _QWEN_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - depends on environment
    AutoModelForMultimodalLM = None  # type: ignore
    AutoProcessor = None  # type: ignore
    _QWEN_IMPORT_ERROR = exc


def _load_qwen_model(model_name: str = QWEN_MODEL_ID, device: str | None = None) -> tuple[Any, Any]:
    if AutoModelForMultimodalLM is None or AutoProcessor is None:
        raise RuntimeError(
            "Qwen3-ASR requires transformers>=5.13.0. "
            f"Import failed: {_QWEN_IMPORT_ERROR}"
        )

    source = resolve_qwen_model(model_name)
    model = AutoModelForMultimodalLM.from_pretrained(source, device_map="auto")
    processor = AutoProcessor.from_pretrained(source)
    return model, processor


def _vad_segments(samples: Any, rate: int, max_seconds: float = MAX_SEGMENT_SECONDS) -> list[tuple[int, int]]:
    import librosa

    intervals = librosa.effects.split(samples, top_db=30, frame_length=2048, hop_length=512)
    if intervals.size == 0:
        return [(0, len(samples))]

    raw: list[tuple[int, int]] = [(int(start), int(end)) for start, end in intervals]

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


def _write_pcm_wav(path: Path, samples: Any, rate: int) -> None:
    import wave

    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _transcribe_chunk(model: Any, processor: Any, wav_path: Path, language: str = "ja") -> str:
    import torch

    inputs = processor.apply_transcription_request(audio=str(wav_path), language=language)
    inputs = inputs.to(model.device, model.dtype)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    transcription = processor.decode(generated_ids, return_format="transcription_only")[0]
    return str(transcription).strip()


def _split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    current: list[str] = []
    for char in text:
        current.append(char)
        if char in SENTENCE_END_CHARS:
            sentences.append("".join(current))
            current = []
    if current:
        sentences.append("".join(current))
    return [sentence for sentence in sentences if sentence.strip()]


def _sentence_words(text: str, start: float, end: float) -> list[Word]:
    words = _tokenize_words(text)
    if not words:
        return []
    total_chars = max(1, sum(len(word) for word in words))
    duration = max(0.001, end - start)
    output: list[Word] = []
    position = 0
    for word in words:
        word_start = start + position * duration / total_chars
        position += len(word)
        word_end = start + position * duration / total_chars
        output.append(
            Word(
                text=word,
                start=round(word_start, 3),
                end=round(min(end, word_end), 3),
                language="ja",
                probability=None,
            )
        )
    return _merge_trailing_punctuation(output)


def run_qwen_asr(
    out_dir: str | Path,
    *,
    device: str | None = None,
    model_name: str = QWEN_MODEL_ID,
    chunk_seconds: float = MAX_SEGMENT_SECONDS,
) -> tuple[Path, Path]:
    out = Path(out_dir)
    audio = prepared_audio_path(out)
    if not audio.exists():
        raise FileNotFoundError(f"Missing prepared audio: {audio}")

    import librosa
    import tempfile

    model, processor = _load_qwen_model(model_name=model_name, device=device)

    waveform, rate = librosa.load(str(audio), sr=16000)
    vad_intervals = _vad_segments(waveform, rate, max_seconds=chunk_seconds)

    all_words: list[Word] = []
    all_segments: list[AsrSegment] = []
    sentence_index = 0
    with tempfile.TemporaryDirectory() as tmp:
        for segment_index, (start_sample, end_sample) in enumerate(vad_intervals):
            chunk = waveform[start_sample:end_sample]
            if chunk.size == 0:
                continue
            wav_path = Path(tmp) / f"{segment_index:05d}.wav"
            _write_pcm_wav(wav_path, chunk, rate)
            text = _transcribe_chunk(model, processor, wav_path, language="ja")

            start = start_sample / rate
            end = end_sample / rate
            sentence_texts = _split_sentences(text)
            if not sentence_texts:
                continue

            sentence_chars = [len(sentence) for sentence in sentence_texts]
            total_chars = max(1, sum(sentence_chars))
            position = start
            for sentence in sentence_texts:
                chunk_id = f"qwen_{sentence_index:05d}"
                sentence_index += 1
                sentence_duration = (end - start) * len(sentence) / total_chars
                sentence_end = min(end, position + sentence_duration)
                words = _sentence_words(sentence, position, sentence_end)
                for word in words:
                    word.chunk_id = chunk_id
                all_words.extend(words)
                all_segments.append(
                    AsrSegment(
                        text=sentence,
                        start=round(position, 3),
                        end=round(sentence_end, 3),
                        language="ja",
                        chunk_id=chunk_id,
                    )
                )
                position = sentence_end

    all_words.sort(key=lambda item: (item.start, item.end))
    words_path = out / "asr_words.jsonl"
    segments_path = out / "asr_segments.json"
    write_words(words_path, all_words)
    write_asr_segments(segments_path, all_segments)
    return words_path, segments_path
