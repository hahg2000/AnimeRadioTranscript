from __future__ import annotations

import hashlib
import itertools
import json
import math
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .compare import AssEvent, levenshtein, parse_ass_events
from .diarization import write_rttm
from .io_utils import ensure_dir, write_json
from .models import Turn


def normalize_japanese_text(text: str, *, fold_kana: bool = False) -> str:
    """Normalize Japanese transcript text for character-error evaluation.

    NFKC normalizes width variants. Whitespace, punctuation and control
    characters are ignored, while Kanji and Kana are otherwise preserved.
    ``fold_kana`` additionally maps Katakana to Hiragana for a pronunciation-
    oriented secondary metric.
    """

    normalized = unicodedata.normalize("NFKC", text)
    kept: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category[0] in {"P", "Z", "C"}:
            continue
        if fold_kana and "ァ" <= char <= "ヶ":
            char = chr(ord(char) - 0x60)
        kept.append(char)
    return "".join(kept)


def _speaker_texts(events: Iterable[AssEvent], *, fold_kana: bool = False) -> dict[str, str]:
    grouped: dict[str, list[AssEvent]] = defaultdict(list)
    for event in events:
        grouped[event.speaker].append(event)
    return {
        speaker: normalize_japanese_text(
            "".join(item.text for item in sorted(items, key=lambda value: (value.start, value.end))),
            fold_kana=fold_kana,
        )
        for speaker, items in grouped.items()
    }


def _minimum_permutation_distance(reference: dict[str, str], hypothesis: dict[str, str]) -> tuple[int, dict[str, str]]:
    ref_names = sorted(reference)
    hyp_names = sorted(hypothesis)
    size = max(len(ref_names), len(hyp_names), 1)
    padded_refs: list[str | None] = ref_names + [None] * (size - len(ref_names))
    padded_hyps: list[str | None] = hyp_names + [None] * (size - len(hyp_names))

    best_distance = math.inf
    best_mapping: dict[str, str] = {}
    for permutation in itertools.permutations(range(size)):
        distance = 0
        mapping: dict[str, str] = {}
        for ref_index, hyp_index in enumerate(permutation):
            ref_name = padded_refs[ref_index]
            hyp_name = padded_hyps[hyp_index]
            ref_text = "" if ref_name is None else reference[ref_name]
            hyp_text = "" if hyp_name is None else hypothesis[hyp_name]
            distance += levenshtein(ref_text, hyp_text)
            if ref_name is not None and hyp_name is not None:
                mapping[hyp_name] = ref_name
        if distance < best_distance:
            best_distance = distance
            best_mapping = mapping
    return int(best_distance), best_mapping


def _frame_speaker_sets(events: Iterable[AssEvent], duration: float, frame_step: float) -> list[set[str]]:
    frame_count = max(1, int(math.ceil(duration / frame_step)))
    frames = [set() for _ in range(frame_count)]
    for event in events:
        start_index = max(0, int(math.floor(event.start / frame_step)))
        end_index = min(frame_count, int(math.ceil(event.end / frame_step)))
        for index in range(start_index, end_index):
            frames[index].add(event.speaker)
    return frames


def _best_frame_mapping(
    reference_frames: list[set[str]], hypothesis_frames: list[set[str]]
) -> dict[str, str]:
    ref_names = sorted(set().union(*reference_frames)) if reference_frames else []
    hyp_names = sorted(set().union(*hypothesis_frames)) if hypothesis_frames else []
    size = max(len(ref_names), len(hyp_names), 1)
    padded_refs: list[str | None] = ref_names + [None] * (size - len(ref_names))
    padded_hyps: list[str | None] = hyp_names + [None] * (size - len(hyp_names))

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for ref_active, hyp_active in zip(reference_frames, hypothesis_frames):
        for ref_name in ref_active:
            for hyp_name in hyp_active:
                pair_counts[(ref_name, hyp_name)] += 1

    best_score = -1
    best_mapping: dict[str, str] = {}
    for permutation in itertools.permutations(range(size)):
        score = 0
        mapping: dict[str, str] = {}
        for hyp_index, ref_index in enumerate(permutation):
            hyp_name = padded_hyps[hyp_index]
            ref_name = padded_refs[ref_index]
            if hyp_name is not None and ref_name is not None:
                mapping[hyp_name] = ref_name
                score += pair_counts[(ref_name, hyp_name)]
        if score > best_score:
            best_score = score
            best_mapping = mapping
    return best_mapping


def _safe_ratio(numerator: float, denominator: float, *, empty: float = 0.0) -> float:
    return empty if denominator == 0 else numerator / denominator


def evaluate_ass(
    reference_path: str | Path,
    hypothesis_path: str | Path,
    *,
    frame_step: float = 0.01,
) -> dict[str, object]:
    """Evaluate text, speaker attribution and overlap from two ASS files."""

    if frame_step <= 0:
        raise ValueError("frame_step must be positive")
    reference = parse_ass_events(reference_path)
    hypothesis = parse_ass_events(hypothesis_path)
    duration = max([0.0] + [item.end for item in reference] + [item.end for item in hypothesis])

    ref_text = normalize_japanese_text("".join(item.text for item in reference))
    hyp_text = normalize_japanese_text("".join(item.text for item in hypothesis))
    char_errors = levenshtein(ref_text, hyp_text)
    ref_kana = normalize_japanese_text("".join(item.text for item in reference), fold_kana=True)
    hyp_kana = normalize_japanese_text("".join(item.text for item in hypothesis), fold_kana=True)
    kana_errors = levenshtein(ref_kana, hyp_kana)

    ref_speaker_text = _speaker_texts(reference)
    hyp_speaker_text = _speaker_texts(hypothesis)
    cp_errors, text_mapping = _minimum_permutation_distance(ref_speaker_text, hyp_speaker_text)
    reference_characters = sum(len(value) for value in ref_speaker_text.values())

    reference_frames = _frame_speaker_sets(reference, duration, frame_step)
    hypothesis_frames = _frame_speaker_sets(hypothesis, duration, frame_step)
    frame_mapping = _best_frame_mapping(reference_frames, hypothesis_frames)

    miss = false_alarm = confusion = correct = reference_speaker_frames = 0
    osd_tp = osd_fp = osd_fn = 0
    reference_speech_frames = hypothesis_speech_frames = 0
    reference_overlap_frames = hypothesis_overlap_frames = 0
    for ref_active, hyp_active in zip(reference_frames, hypothesis_frames):
        mapped_hyp = {frame_mapping.get(name, f"<unmapped:{name}>") for name in hyp_active}
        ref_count = len(ref_active)
        hyp_count = len(mapped_hyp)
        matched = len(ref_active & mapped_hyp)
        reference_speaker_frames += ref_count
        correct += matched
        miss += max(0, ref_count - hyp_count)
        false_alarm += max(0, hyp_count - ref_count)
        confusion += min(ref_count, hyp_count) - matched

        reference_speech_frames += int(ref_count > 0)
        hypothesis_speech_frames += int(hyp_count > 0)
        ref_overlap = ref_count >= 2
        hyp_overlap = hyp_count >= 2
        reference_overlap_frames += int(ref_overlap)
        hypothesis_overlap_frames += int(hyp_overlap)
        osd_tp += int(ref_overlap and hyp_overlap)
        osd_fp += int(not ref_overlap and hyp_overlap)
        osd_fn += int(ref_overlap and not hyp_overlap)

    precision = _safe_ratio(osd_tp, osd_tp + osd_fp, empty=1.0 if osd_fn == 0 else 0.0)
    recall = _safe_ratio(osd_tp, osd_tp + osd_fn, empty=1.0)
    f1 = _safe_ratio(2 * precision * recall, precision + recall, empty=0.0)
    der_numerator = miss + false_alarm + confusion
    ref_speaker_names = sorted(set().union(*reference_frames)) if reference_frames else []
    jer_values: list[float] = []
    for ref_speaker in ref_speaker_names:
        intersection = union = 0
        for ref_active, hyp_active in zip(reference_frames, hypothesis_frames):
            ref_present = ref_speaker in ref_active
            hyp_present = any(frame_mapping.get(name) == ref_speaker for name in hyp_active)
            intersection += int(ref_present and hyp_present)
            union += int(ref_present or hyp_present)
        jer_values.append(1.0 - _safe_ratio(intersection, union, empty=1.0))

    return {
        "schema_version": 1,
        "frame_step_seconds": frame_step,
        "duration_seconds": duration,
        "events": {"reference": len(reference), "hypothesis": len(hypothesis)},
        "text": {
            "normalization": "NFKC + ignore punctuation/whitespace/control",
            "reference_characters": len(ref_text),
            "hypothesis_characters": len(hyp_text),
            "character_errors": char_errors,
            "cer": _safe_ratio(char_errors, len(ref_text)),
            "kana_folded_character_errors": kana_errors,
            "kana_folded_cer": _safe_ratio(kana_errors, len(ref_kana)),
            "cp_character_errors": cp_errors,
            "cpcer": _safe_ratio(cp_errors, reference_characters),
            "cpcer_speaker_mapping": text_mapping,
        },
        "diarization": {
            "reference_speaker_seconds": reference_speaker_frames * frame_step,
            "miss_seconds": miss * frame_step,
            "false_alarm_seconds": false_alarm * frame_step,
            "confusion_seconds": confusion * frame_step,
            "correct_seconds": correct * frame_step,
            "der": _safe_ratio(der_numerator, reference_speaker_frames),
            "jer": sum(jer_values) / len(jer_values) if jer_values else 0.0,
            "speaker_mapping": frame_mapping,
        },
        "overlap": {
            "reference_speech_seconds": reference_speech_frames * frame_step,
            "hypothesis_speech_seconds": hypothesis_speech_frames * frame_step,
            "reference_overlap_seconds": reference_overlap_frames * frame_step,
            "hypothesis_overlap_seconds": hypothesis_overlap_frames * frame_step,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
    }


def evaluate_and_write(
    reference_path: str | Path,
    hypothesis_path: str | Path,
    output_path: str | Path,
    *,
    frame_step: float = 0.01,
) -> dict[str, object]:
    metrics = evaluate_ass(reference_path, hypothesis_path, frame_step=frame_step)
    write_json(output_path, metrics)
    return metrics


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def init_evaluation(
    reference_ass: str | Path,
    output_dir: str | Path,
    *,
    audio_path: str | Path | None = None,
) -> dict[str, Path]:
    """Create reproducible reference artifacts from a hand-authored ASS file."""

    reference_path = Path(reference_ass)
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    audio = None if audio_path is None else Path(audio_path)
    if audio is not None and not audio.exists():
        raise FileNotFoundError(audio)

    output = ensure_dir(output_dir)
    events = parse_ass_events(reference_path)
    segments_path = write_json(
        output / "reference_segments.json",
        [
            {
                "segment_id": f"ref_{index:05d}",
                "speaker": event.speaker,
                "start": event.start,
                "end": event.end,
                "text": event.text,
                "normalized_text": normalize_japanese_text(event.text),
                "overlap": any(
                    other_index != index
                    and other.speaker != event.speaker
                    and min(event.end, other.end) > max(event.start, other.start)
                    for other_index, other in enumerate(events)
                ),
            }
            for index, event in enumerate(events)
        ],
    )
    turns = [Turn(event.speaker, event.start, event.end, source="reference_ass") for event in events]
    rttm_path = write_rttm(output / "reference.rttm", turns, file_id=reference_path.stem)
    manifest = {
        "schema_version": 1,
        "reference_ass": str(reference_path.resolve()),
        "reference_ass_sha256": _sha256(reference_path),
        "audio": None if audio is None else str(audio.resolve()),
        "audio_sha256": None if audio is None else _sha256(audio),
        "reference_segments": segments_path.name,
        "reference_rttm": rttm_path.name,
        "event_count": len(events),
        "speakers": sorted({event.speaker for event in events}),
        "text_normalization": "NFKC + ignore punctuation/whitespace/control",
    }
    manifest_path = write_json(output / "manifest.json", manifest)
    return {"manifest": manifest_path, "reference_segments": segments_path, "reference_rttm": rttm_path}
