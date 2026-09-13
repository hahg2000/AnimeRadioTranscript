from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any, Iterable

from .audio import prepared_audio_path
from .io_utils import write_json, write_speaker_activity, write_turns
from .models import SpeakerActivityFrame, Turn
from .model_paths import resolve_sortformer_model


def parse_rttm_line(line: str, source: str = "rttm") -> Turn | None:
    parts = line.strip().split()
    if not parts or parts[0].upper() != "SPEAKER":
        return None
    if len(parts) < 8:
        raise ValueError(f"Invalid RTTM line: {line!r}")
    start = float(parts[3])
    duration = float(parts[4])
    return Turn(speaker=parts[7], start=start, end=start + duration, source=source)


def read_rttm(path: str | Path, source: str = "rttm") -> list[Turn]:
    turns: list[Turn] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed = parse_rttm_line(line, source=source)
            if parsed is not None:
                turns.append(parsed)
    return sorted(turns, key=lambda item: (item.start, item.end, item.speaker))


def write_rttm(path: str | Path, turns: Iterable[Turn], file_id: str = "prepared") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for turn in turns:
            handle.write(
                "SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n".format(
                    file_id=file_id,
                    start=turn.start,
                    duration=max(0.0, turn.end - turn.start),
                    speaker=turn.speaker,
                )
            )
    return target


def _speaker_durations(turns: Iterable[Turn]) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    for turn in turns:
        totals[turn.speaker] += turn.duration
    return dict(totals)


def converge_speakers(turns: Iterable[Turn], speaker_count: int | None) -> list[Turn]:
    """Map arbitrary speaker labels to SPEAKER_XX and collapse extras deterministically."""

    ordered_turns = sorted(list(turns), key=lambda item: (item.start, item.end, item.speaker))
    if not ordered_turns:
        return []
    if speaker_count is not None and speaker_count < 1:
        raise ValueError("speaker_count must be positive")

    totals = _speaker_durations(ordered_turns)
    first_seen: dict[str, float] = {}
    for turn in ordered_turns:
        first_seen.setdefault(turn.speaker, turn.start)

    all_speakers = sorted(totals, key=lambda name: (-totals[name], first_seen[name], name))
    kept = all_speakers if speaker_count is None else all_speakers[:speaker_count]
    kept = sorted(kept, key=lambda name: (first_seen[name], name))
    numeric_speakers: dict[str, int] = {}
    for speaker in kept:
        match = re.search(r"speaker[_-]?(\d+)$", speaker, flags=re.IGNORECASE)
        if match is not None:
            numeric_speakers[speaker] = int(match.group(1))
    if len(numeric_speakers) == len(kept) and len(set(numeric_speakers.values())) == len(kept):
        # Activity tensor channel N corresponds to Sortformer's speaker_N.
        speaker_map = {speaker: f"SPEAKER_{index:02d}" for speaker, index in numeric_speakers.items()}
    else:
        speaker_map = {speaker: f"SPEAKER_{index:02d}" for index, speaker in enumerate(kept)}

    def nearest_kept_speaker(turn: Turn) -> str:
        best_name = kept[0]
        best_distance = float("inf")
        for candidate in ordered_turns:
            if candidate.speaker not in speaker_map:
                continue
            if candidate.end < turn.start:
                distance = turn.start - candidate.end
            elif turn.end < candidate.start:
                distance = candidate.start - turn.end
            else:
                distance = 0.0
            if distance < best_distance:
                best_distance = distance
                best_name = candidate.speaker
        return best_name

    normalized: list[Turn] = []
    for turn in ordered_turns:
        mapped_source = turn.speaker if turn.speaker in speaker_map else nearest_kept_speaker(turn)
        flags = list(turn.flags)
        if mapped_source != turn.speaker:
            flags.append("speaker_collapsed")
        normalized.append(
            Turn(
                speaker=speaker_map[mapped_source],
                start=turn.start,
                end=turn.end,
                confidence=turn.confidence,
                source=turn.source,
                flags=sorted(set(flags)),
            )
        )
    return normalized


def _flatten_result(value: Any) -> Iterable[Any]:
    if value is None:
        return
    if isinstance(value, dict):
        yield value
        return
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_result(item)
        return
    yield value


def _turn_from_sortformer_item(item: Any, source: str) -> Turn | None:
    if isinstance(item, str):
        rttm_turn = parse_rttm_line(item, source=source)
        if rttm_turn is not None:
            return rttm_turn
        parts = item.replace(",", " ").split()
        floats: list[float] = []
        speaker = None
        for part in parts:
            try:
                floats.append(float(part))
            except ValueError:
                speaker = part
        if len(floats) >= 2:
            return Turn(speaker=speaker or "SPEAKER_00", start=floats[0], end=floats[1], source=source)
        return None

    def value(*names: str, default: Any = None) -> Any:
        for name in names:
            if isinstance(item, dict) and name in item:
                return item[name]
            if hasattr(item, name):
                return getattr(item, name)
        return default

    start = value("start", "start_time", "begin")
    end = value("end", "end_time")
    duration = value("duration")
    speaker = value("speaker", "speaker_label", "label", default="SPEAKER_00")
    if start is None:
        return None
    start_f = float(start)
    end_f = float(end) if end is not None else start_f + float(duration or 0.0)
    confidence = value("confidence", "score", default=None)
    return Turn(
        speaker=str(speaker),
        start=start_f,
        end=end_f,
        confidence=None if confidence is None else float(confidence),
        source=source,
    )


def turns_from_sortformer_result(result: Any, source: str) -> list[Turn]:
    turns: list[Turn] = []
    for item in _flatten_result(result):
        parsed = _turn_from_sortformer_item(item, source=source)
        if parsed is not None and parsed.end > parsed.start:
            turns.append(parsed)
    return sorted(turns, key=lambda item: (item.start, item.end, item.speaker))


def _split_sortformer_result(result: Any) -> tuple[Any, Any | None]:
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, None


def _to_nested_list(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def activity_from_sortformer_tensor(
    tensor_outputs: Any,
    *,
    source: str,
    frame_step: float = 0.08,
) -> list[SpeakerActivityFrame]:
    """Convert NeMo's T x S probabilities to portable JSONL frames."""

    values = tensor_outputs
    if isinstance(values, (list, tuple)) and len(values) == 1:
        values = values[0]
    values = _to_nested_list(values)
    while (
        isinstance(values, (list, tuple))
        and len(values) == 1
        and isinstance(values[0], (list, tuple))
        and values[0]
        and isinstance(values[0][0], (list, tuple))
    ):
        values = values[0]
    if not isinstance(values, (list, tuple)):
        return []
    frames: list[SpeakerActivityFrame] = []
    for index, row in enumerate(values):
        row_values = _to_nested_list(row)
        if not isinstance(row_values, (list, tuple)):
            continue
        frames.append(
            SpeakerActivityFrame(
                start=index * frame_step,
                end=(index + 1) * frame_step,
                scores={f"SPEAKER_{speaker_index:02d}": float(score) for speaker_index, score in enumerate(row_values)},
                source=source,
            )
        )
    return frames


def activity_from_turns(turns: Iterable[Turn], *, source: str = "rttm") -> list[SpeakerActivityFrame]:
    ordered = list(turns)
    boundaries = sorted({value for turn in ordered for value in (turn.start, turn.end)})
    frames: list[SpeakerActivityFrame] = []
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        scores = {
            turn.speaker: 1.0
            for turn in ordered
            if min(end, turn.end) > max(start, turn.start)
        }
        if scores:
            frames.append(SpeakerActivityFrame(start=start, end=end, scores=scores, source=source))
    return frames


def exclusive_turns_from_activity(
    activity: Iterable[SpeakerActivityFrame], *, threshold: float = 0.5
) -> list[Turn]:
    output: list[Turn] = []
    for frame in activity:
        if not frame.scores:
            continue
        speaker, score = max(frame.scores.items(), key=lambda item: (item[1], item[0]))
        if score < threshold:
            continue
        if output and output[-1].speaker == speaker and abs(output[-1].end - frame.start) < 1e-6:
            output[-1].end = frame.end
            output[-1].confidence = max(output[-1].confidence or 0.0, score)
        else:
            output.append(Turn(speaker, frame.start, frame.end, confidence=score, source="exclusive_activity"))
    return output


def overlap_regions_from_activity(
    activity: Iterable[SpeakerActivityFrame], *, threshold: float = 0.5
) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for frame in activity:
        speakers = sorted(name for name, score in frame.scores.items() if score >= threshold)
        if len(speakers) < 2:
            continue
        if regions and regions[-1]["speakers"] == speakers and abs(float(regions[-1]["end"]) - frame.start) < 1e-6:
            regions[-1]["end"] = frame.end
        else:
            regions.append(
                {
                    "region_id": f"overlap_{len(regions):05d}",
                    "start": frame.start,
                    "end": frame.end,
                    "speakers": speakers,
                    "flags": ["needs_overlap_separation"],
                }
            )
    return regions


def run_nemo_sortformer_with_activity(
    out_dir: str | Path,
    diar_model: str,
    models_root: str | Path | None = None,
    device: str | None = None,
) -> tuple[list[Turn], list[SpeakerActivityFrame]]:
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ImportError as exc:
        message = str(exc)
        if "huggingface-hub" in message or "transformers" in message:
            raise RuntimeError(
                "NeMo could not be imported because the Hugging Face/Transformers dependency set is incompatible. "
                "Install a Transformers-compatible hub client with: "
                'python -m pip install --force-reinstall "huggingface-hub>=0.34.0,<1.0"'
            ) from exc
        raise RuntimeError(
            "NVIDIA NeMo is not installed. Install the WSL dependencies from requirements-wsl.txt, "
            "or pass --rttm to import an existing diarization file."
        ) from exc

    audio = prepared_audio_path(out_dir)
    if not audio.exists():
        raise FileNotFoundError(f"Missing prepared audio: {audio}")

    resolved_model = resolve_sortformer_model(diar_model, models_root)
    if Path(resolved_model).is_file() and Path(resolved_model).suffix.lower() == ".nemo":
        model = SortformerEncLabelModel.restore_from(resolved_model)
    else:
        model = SortformerEncLabelModel.from_pretrained(resolved_model)
    if device is not None and hasattr(model, "to"):
        model = model.to(device)
    try:
        result = model.diarize(audio=[str(audio)], batch_size=1, include_tensor_outputs=True)
    except TypeError:
        try:
            result = model.diarize(audio=str(audio), batch_size=1, include_tensor_outputs=True)
        except TypeError:
            result = model.diarize(str(audio))
    segments_result, tensor_result = _split_sortformer_result(result)
    turns = turns_from_sortformer_result(segments_result, source=resolved_model)
    if not turns:
        raise RuntimeError("NeMo Sortformer returned no parseable diarization turns.")
    activity = activity_from_sortformer_tensor(tensor_result, source=resolved_model) if tensor_result is not None else []
    return turns, activity


def diarize_out_dir(
    out_dir: str | Path,
    diar_model: str = "nvidia/diar_sortformer_4spk-v1",
    speakers: int | None = None,
    rttm: str | Path | None = None,
    models_root: str | Path | None = None,
    device: str | None = None,
) -> tuple[Path, Path]:
    out = Path(out_dir)
    if rttm is not None:
        turns = read_rttm(rttm, source="rttm")
        activity: list[SpeakerActivityFrame] = []
    else:
        turns, activity = run_nemo_sortformer_with_activity(out, diar_model, models_root, device)
    turns = converge_speakers(turns, speakers)
    if not activity:
        activity = activity_from_turns(turns, source="rttm" if rttm is not None else diar_model)
    else:
        allowed_speakers = {turn.speaker for turn in turns}
        activity = [
            SpeakerActivityFrame(
                start=frame.start,
                end=frame.end,
                scores={name: score for name, score in frame.scores.items() if name in allowed_speakers},
                source=frame.source,
            )
            for frame in activity
        ]
    json_path = out / "nemo_diarization.json"
    rttm_path = out / "nemo_diarization.rttm"
    write_turns(json_path, turns)
    write_rttm(rttm_path, turns)
    write_speaker_activity(out / "speaker_activity.jsonl", activity)
    write_turns(out / "exclusive_turns.json", exclusive_turns_from_activity(activity))
    write_json(out / "overlap_regions.json", overlap_regions_from_activity(activity))
    return json_path, rttm_path
