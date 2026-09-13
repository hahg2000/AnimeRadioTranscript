from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .render import plain_ass_text


@dataclass(slots=True)
class AssEvent:
    start: float
    end: float
    speaker: str
    text: str


def parse_ass_time(value: str) -> float:
    hour_text, minute_text, second_text = value.strip().split(":")
    return int(hour_text) * 3600 + int(minute_text) * 60 + float(second_text)


def parse_ass_events(path: str | Path) -> list[AssEvent]:
    events: list[AssEvent] = []
    in_events = False
    fields: list[str] = []
    for raw_line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower() == "[events]":
            in_events = True
            continue
        if line.startswith("[") and line.endswith("]"):
            in_events = False
            continue
        if not in_events:
            continue
        if line.startswith("Format:"):
            fields = [field.strip() for field in line.split(":", 1)[1].split(",")]
            continue
        if not line.startswith("Dialogue:") or not fields:
            continue
        values = line.split(":", 1)[1].lstrip().split(",", len(fields) - 1)
        if len(values) != len(fields):
            continue
        row = dict(zip(fields, values))
        speaker = row.get("Name") or row.get("Style") or "UNKNOWN"
        events.append(
            AssEvent(
                start=parse_ass_time(row["Start"]),
                end=parse_ass_time(row["End"]),
                speaker=speaker,
                text=plain_ass_text(row.get("Text", "")),
            )
        )
    return sorted(events, key=lambda item: (item.start, item.end, item.speaker))


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row_index, left_char in enumerate(left, start=1):
        current = [row_index]
        for col_index, right_char in enumerate(right, start=1):
            insert = current[col_index - 1] + 1
            delete = previous[col_index] + 1
            substitute = previous[col_index - 1] + (left_char != right_char)
            current.append(min(insert, delete, substitute))
        previous = current
    return previous[-1]


def _overlap(left: AssEvent, right: AssEvent) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def speaker_confusion(reference: list[AssEvent], hypothesis: list[AssEvent]) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for ref in reference:
        best: AssEvent | None = None
        best_overlap = 0.0
        for hyp in hypothesis:
            overlap = _overlap(ref, hyp)
            same_speaker_tie = overlap == best_overlap and best is not None and hyp.speaker == ref.speaker
            if overlap > best_overlap or same_speaker_tie:
                best_overlap = overlap
                best = hyp
        if best is None or best_overlap <= 0.0:
            matrix[ref.speaker]["<missing>"] += 1
        else:
            matrix[ref.speaker][best.speaker] += 1
    return {speaker: dict(counts) for speaker, counts in matrix.items()}


def compare_ass(reference_path: str | Path, hypothesis_path: str | Path) -> dict[str, object]:
    reference = parse_ass_events(reference_path)
    hypothesis = parse_ass_events(hypothesis_path)
    reference_text = "".join(event.text for event in reference)
    hypothesis_text = "".join(event.text for event in hypothesis)
    distance = levenshtein(reference_text, hypothesis_text)
    return {
        "reference_events": len(reference),
        "hypothesis_events": len(hypothesis),
        "char_errors": distance,
        "reference_chars": len(reference_text),
        "cer": 0.0 if not reference_text else distance / len(reference_text),
        "speaker_confusion": speaker_confusion(reference, hypothesis),
    }


def compare_and_write(
    reference_path: str | Path,
    hypothesis_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    metrics = compare_ass(reference_path, hypothesis_path)
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics
