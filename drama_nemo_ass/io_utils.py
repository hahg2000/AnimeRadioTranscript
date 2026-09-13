from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, TypeVar

from .models import AsrSegment, Segment, SpeakerActivityFrame, Turn, Word

T = TypeVar("T")


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, data: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return target


def _read_jsonl(path: str | Path, factory: type[T]) -> list[T]:
    items: list[T] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(factory.from_dict(json.loads(line)))  # type: ignore[attr-defined]
    return items


def _write_jsonl(path: str | Path, items: Iterable[Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            data = item.to_dict() if hasattr(item, "to_dict") else item
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")
    return target


def read_words(path: str | Path) -> list[Word]:
    return _read_jsonl(path, Word)


def write_words(path: str | Path, words: Iterable[Word]) -> Path:
    return _write_jsonl(path, words)


def read_speaker_activity(path: str | Path) -> list[SpeakerActivityFrame]:
    return _read_jsonl(path, SpeakerActivityFrame)


def write_speaker_activity(path: str | Path, frames: Iterable[SpeakerActivityFrame]) -> Path:
    return _write_jsonl(path, frames)


def write_asr_segments(path: str | Path, segments: Iterable[AsrSegment]) -> Path:
    return write_json(path, [item.to_dict() for item in segments])


def read_turns(path: str | Path) -> list[Turn]:
    return [Turn.from_dict(item) for item in read_json(path)]


def write_turns(path: str | Path, turns: Iterable[Turn]) -> Path:
    return write_json(path, [item.to_dict() for item in turns])


def read_segments(path: str | Path) -> list[Segment]:
    return [Segment.from_dict(item) for item in read_json(path)]


def write_segments(path: str | Path, segments: Iterable[Segment]) -> Path:
    return write_json(path, [item.to_dict() for item in segments])


REVIEW_FIELDS = ["segment_id", "start", "end", "speaker", "text", "flags", "confidence"]


def write_review_tsv(path: str | Path, segments: Iterable[Segment]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for index, segment in enumerate(segments):
            writer.writerow(
                {
                    "segment_id": segment.segment_id or f"seg_{index:05d}",
                    "start": f"{segment.start:.3f}",
                    "end": f"{segment.end:.3f}",
                    "speaker": segment.speaker,
                    "text": segment.text,
                    "flags": "|".join(segment.flags),
                    "confidence": "" if segment.confidence is None else f"{segment.confidence:.4f}",
                }
            )
    return target


def read_review_tsv(path: str | Path) -> list[Segment]:
    segments: list[Segment] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            confidence = row.get("confidence") or None
            segments.append(
                Segment(
                    segment_id=row.get("segment_id") or None,
                    start=float(row.get("start") or 0.0),
                    end=float(row.get("end") or 0.0),
                    speaker=row.get("speaker") or "UNKNOWN",
                    text=row.get("text") or "",
                    flags=[item for item in (row.get("flags") or "").split("|") if item],
                    confidence=None if confidence is None else float(confidence),
                )
            )
    return segments
