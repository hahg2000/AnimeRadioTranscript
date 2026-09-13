from __future__ import annotations

from pathlib import Path

from .assignment import (
    assign_words_to_turns,
    build_segments,
    segments_from_assigned_words,
    smooth_short_speaker_islands,
)
from .io_utils import read_speaker_activity, read_turns, read_words, write_review_tsv, write_segments

SENTENCE_CHUNK_PREFIXES = ("qwen_",)


def relabel_out_dir(
    out_dir: str | Path,
    smooth_short_islands: bool = False,
) -> tuple[Path, Path]:
    out = Path(out_dir)
    turns = read_turns(out / "nemo_diarization.json")

    words = read_words(out / "asr_words.jsonl")
    activity_path = out / "speaker_activity.jsonl"
    activity = read_speaker_activity(activity_path) if activity_path.exists() else None
    assigned = assign_words_to_turns(words, turns, activity=activity)
    if smooth_short_islands:
        assigned = smooth_short_speaker_islands(assigned)

    has_sentence_chunks = any(
        word.chunk_id and str(word.chunk_id).startswith(SENTENCE_CHUNK_PREFIXES) for word in assigned
    )
    if has_sentence_chunks:
        segments = segments_from_assigned_words(assigned, split_on_chunk=True)
    else:
        segments = build_segments(assigned)

    segments_path = out / "segments.json"
    review_path = out / "review.tsv"
    write_segments(segments_path, segments)
    write_review_tsv(review_path, segments)
    return segments_path, review_path
