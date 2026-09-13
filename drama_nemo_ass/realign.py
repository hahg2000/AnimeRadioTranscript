from __future__ import annotations

import difflib
import json
import unicodedata
from pathlib import Path

from .io_utils import read_json, write_json

MIN_WORD_SECONDS = 0.05


def normalize_for_compare(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    kept: list[str] = []
    for char in value:
        if char.isspace() or unicodedata.category(char)[0] in {"P", "S"}:
            continue
        kept.append(char)
    return "".join(kept)


def build_reference_timeline(reference_words: list[dict]) -> tuple[str, list[tuple[float, float]]]:
    """Return (concatenated normalized text, per-char (start, end) timeline)."""

    pieces: list[str] = []
    timeline: list[tuple[float, float]] = []
    for word in reference_words:
        text = normalize_for_compare(str(word.get("text") or ""))
        if not text:
            continue
        start = float(word["start"])
        end = float(word["end"])
        for char in text:
            pieces.append(char)
            timeline.append((start, end))
    return "".join(pieces), timeline


def char_mapping(hyp_text: str, reference_text: str) -> dict[int, int]:
    """Map hypothesis char index -> reference char index via SequenceMatcher.

    Unmatched hypothesis chars are interpolated between surrounding matches.
    """

    matcher = difflib.SequenceMatcher(None, reference_text, hyp_text, autojunk=False)
    blocks = matcher.get_matching_blocks()
    mapping: dict[int, int] = {}
    for ref_start, hyp_start, size in blocks:
        if size == 0:
            continue
        for offset in range(size):
            mapping[hyp_start + offset] = ref_start + offset
    if not mapping:
        return mapping
    ordered = sorted(mapping.items())
    first_hyp, first_ref = ordered[0]
    last_hyp, last_ref = ordered[-1]
    for index in range(0, first_hyp):
        mapping[index] = max(0, first_ref - (first_hyp - index))
    for index in range(first_hyp, last_hyp + 1):
        if index not in mapping:
            before = max((e for e, r in ordered if e < index), default=None)
            after = min((e for e, r in ordered if e > index), default=None)
            if before is not None and after is not None:
                frac = (index - before) / max(1, after - before)
                mapping[index] = int(mapping[before] + frac * (mapping[after] - mapping[before]))
            elif before is not None:
                mapping[index] = mapping[before]
            else:
                mapping[index] = mapping[after]
    for index in range(last_hyp + 1, len(hyp_text)):
        mapping[index] = min(len(reference_text) - 1, last_ref + (index - last_hyp))
    return mapping


def align_words_to_reference(
    words: list[dict],
    reference_words: list[dict],
) -> list[dict]:
    """Character-level alignment: each hypothesis word adopts the reference
    times of the reference words covering its mapped characters."""

    reference_text, timeline = build_reference_timeline(reference_words)
    hyp_text = "".join(normalize_for_compare(str(word.get("text") or "")) for word in words)
    if not reference_text or not hyp_text:
        return [{**word, "source_time": "hypothesis"} for word in words]

    mapping = char_mapping(hyp_text, reference_text)

    output: list[dict] = []
    position = 0
    for word in words:
        text = normalize_for_compare(str(word.get("text") or ""))
        if not text or position >= len(mapping):
            output.append({**word, "source_time": "hypothesis"})
            position += len(text)
            continue
        char_indices = [mapping.get(position + offset, 0) for offset in range(len(text))]
        char_indices = [max(0, min(len(reference_text) - 1, value)) for value in char_indices]
        spans = [timeline[value] for value in char_indices]
        new_start = min(span[0] for span in spans)
        new_end = max(span[1] for span in spans)
        if new_end <= new_start:
            new_end = new_start + 0.001
        output.append(
            {
                **word,
                "start": round(new_start, 3),
                "end": round(new_end, 3),
                "source_time": "whisper",
            }
        )
        position += len(text)
    return output


def align_words_to_segments(words: list[dict], segments: list[dict]) -> list[dict]:
    """Clamp whisper-mapped word spans inside their containing segment."""

    output: list[dict] = []
    for word in words:
        if word.get("source_time") != "whisper":
            output.append(word)
            continue
        containing = [
            segment
            for segment in segments
            if float(segment["start"]) <= (float(word["start"]) + float(word["end"])) / 2 <= float(segment["end"])
        ]
        if containing:
            segment = containing[0]
            word["start"] = round(max(float(segment["start"]), float(word["start"])), 3)
            word["end"] = round(min(float(segment["end"]), float(word["end"])), 3)
        output.append(word)
    return output


def enforce_monotonic_words(words: list[dict]) -> list[dict]:
    """Clamp overlapping adjacent word spans so word[i].start >= word[i-1].end,
    and give every word a minimum duration."""

    output: list[dict] = []
    previous_end = -1.0
    for word in words:
        start = max(float(word["start"]), previous_end)
        end = max(float(word["end"]), start + MIN_WORD_SECONDS)
        previous_end = end
        output.append({**word, "start": round(start, 3), "end": round(end, 3)})
    return output


def realign_out_dir(
    out_dir: str | Path,
    reference_dir: str | Path,
    *,
    apply: bool = False,
) -> tuple[Path, Path, int, int]:
    out = Path(out_dir)
    reference = Path(reference_dir)

    segments = [dict(item) for item in read_json(out / "asr_segments.json")]

    words: list[dict] = []
    words_path = out / "asr_words.jsonl"
    if words_path.exists():
        for line in words_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                words.append(json.loads(line))

    reference_words: list[dict] = []
    ref_words_path = reference / "asr_words.jsonl"
    if ref_words_path.exists():
        for line in ref_words_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                reference_words.append(json.loads(line))
    if not reference_words:
        raise FileNotFoundError(f"Missing reference words: {ref_words_path}")

    aligned_words = align_words_to_reference(words, reference_words)
    aligned_words = align_words_to_segments(aligned_words, segments)
    aligned_words = enforce_monotonic_words(aligned_words)
    anchored = sum(1 for word in aligned_words if word.get("source_time") == "whisper")

    if apply:
        words_target = out / "asr_words.jsonl"
        with words_target.open("w", encoding="utf-8", newline="\n") as handle:
            for word in aligned_words:
                handle.write(json.dumps(word, ensure_ascii=False) + "\n")
        return words_target, words_target, anchored, len(aligned_words)

    words_target = out / "asr_words.aligned.jsonl"
    with words_target.open("w", encoding="utf-8", newline="\n") as handle:
        for word in aligned_words:
            handle.write(json.dumps(word, ensure_ascii=False) + "\n")
    return words_target, words_target, anchored, len(aligned_words)
