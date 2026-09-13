from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from .models import Segment, SpeakerActivityFrame, Turn, Word


@dataclass(slots=True)
class AssignedWord:
    text: str
    start: float
    end: float
    speaker: str
    confidence: float | None = None
    flags: list[str] = field(default_factory=list)
    chunk_id: str | None = None


def _center(word: Word) -> float:
    return (word.start + word.end) / 2.0


def _interval_overlap(start: float, end: float, other_start: float, other_end: float) -> float:
    return max(0.0, min(end, other_end) - max(start, other_start))


def _turn_coverage(word: Word, turns: Iterable[Turn]) -> tuple[dict[str, float], dict[str, Turn]]:
    coverage: dict[str, float] = {}
    representatives: dict[str, Turn] = {}
    for turn in turns:
        overlap = _interval_overlap(word.start, word.end, turn.start, turn.end)
        if overlap <= 0:
            continue
        coverage[turn.speaker] = coverage.get(turn.speaker, 0.0) + overlap
        current = representatives.get(turn.speaker)
        if current is None or (turn.confidence or 0.0) > (current.confidence or 0.0):
            representatives[turn.speaker] = turn
    duration = max(0.001, word.end - word.start)
    return {speaker: min(duration, value) for speaker, value in coverage.items()}, representatives


def _activity_scores(
    word: Word,
    activity: Iterable[SpeakerActivityFrame],
    allowed_speakers: set[str],
) -> tuple[dict[str, float], float]:
    totals: dict[str, float] = defaultdict(float)
    covered = 0.0
    for frame in activity:
        overlap = _interval_overlap(word.start, word.end, frame.start, frame.end)
        if overlap <= 0:
            continue
        covered += overlap
        for speaker, score in frame.scores.items():
            if not allowed_speakers or speaker in allowed_speakers:
                totals[speaker] += max(0.0, min(1.0, score)) * overlap
    if covered <= 0:
        return {}, 0.0
    return {speaker: total / covered for speaker, total in totals.items()}, covered


def _nearest_turn(word: Word, turns: list[Turn], max_gap: float) -> Turn | None:
    center = _center(word)
    best_turn: Turn | None = None
    best_gap = float("inf")
    for turn in turns:
        if center < turn.start:
            gap = turn.start - center
        elif center > turn.end:
            gap = center - turn.end
        else:
            gap = 0.0
        if gap < best_gap:
            best_gap = gap
            best_turn = turn
    return best_turn if best_gap <= max_gap else None


def assign_words_to_turns(
    words: Iterable[Word],
    turns: Iterable[Turn],
    max_unassigned_gap: float = 1.0,
    *,
    activity: Iterable[SpeakerActivityFrame] | None = None,
    activity_threshold: float = 0.5,
    turn_overlap_ratio: float = 0.2,
) -> list[AssignedWord]:
    ordered_turns = sorted(list(turns), key=lambda item: (item.start, item.end, item.speaker))
    activity_frames = sorted(list(activity or []), key=lambda item: (item.start, item.end))
    allowed_speakers = {turn.speaker for turn in ordered_turns}
    assigned: list[AssignedWord] = []
    previous_speaker: str | None = None

    for word in sorted(list(words), key=lambda item: (item.start, item.end)):
        flags: list[str] = []
        chosen: Turn | None = None
        chosen_score: float | None = None

        scores, activity_coverage = _activity_scores(word, activity_frames, allowed_speakers)
        if scores and activity_coverage > 0:
            ranked_scores = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            chosen_speaker, chosen_score = ranked_scores[0]
            chosen = next((turn for turn in ordered_turns if turn.speaker == chosen_speaker), None)
            active_speakers = [speaker for speaker, score in ranked_scores if score >= activity_threshold]
            if len(active_speakers) > 1:
                flags.extend(["overlap", "needs_overlap_separation"])
            flags.append("speaker_activity")
        else:
            coverage, representatives = _turn_coverage(word, ordered_turns)
            if coverage:
                duration = max(0.001, word.end - word.start)
                meaningful = [speaker for speaker, value in coverage.items() if value / duration >= turn_overlap_ratio]
                if len(meaningful) > 1:
                    flags.extend(["overlap", "needs_overlap_separation"])
                chosen_speaker = sorted(
                    coverage,
                    key=lambda speaker: (
                        -coverage[speaker],
                        -(representatives[speaker].confidence or 0.0),
                        speaker,
                    ),
                )[0]
                chosen = representatives[chosen_speaker]

        if chosen is None:
            chosen = _nearest_turn(word, ordered_turns, max_gap=max_unassigned_gap)
            if chosen is None:
                assigned.append(
                    AssignedWord(
                        text=word.text,
                        start=word.start,
                        end=word.end,
                        speaker=previous_speaker or "UNKNOWN",
                        confidence=word.probability,
                        flags=["no_speaker"],
                        chunk_id=word.chunk_id,
                    )
                )
                continue
            flags.append("nearest_speaker")

        flags.extend(chosen.flags)
        previous_speaker = chosen.speaker
        confidence_values = [value for value in [word.probability, chosen_score, chosen.confidence] if value is not None]
        confidence = sum(confidence_values) / len(confidence_values) if confidence_values else None
        assigned.append(
            AssignedWord(
                text=word.text,
                start=word.start,
                end=word.end,
                speaker=chosen.speaker,
                confidence=confidence,
                flags=sorted(set(flags)),
                chunk_id=word.chunk_id,
            )
        )
    return assigned


def _speaker_groups(words: list[AssignedWord]) -> list[tuple[int, int]]:
    if not words:
        return []
    groups: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(words)):
        if words[index].speaker != words[index - 1].speaker:
            groups.append((start, index))
            start = index
    groups.append((start, len(words)))
    return groups


def smooth_short_speaker_islands(
    assigned_words: Iterable[AssignedWord],
    max_duration: float = 0.65,
    max_chars: int = 3,
) -> list[AssignedWord]:
    """Reassign tiny A-B-A speaker islands to A to reduce word-level diarization jitter."""

    words = list(assigned_words)
    changed = True
    while changed:
        changed = False
        groups = _speaker_groups(words)
        for group_index in range(1, len(groups) - 1):
            left_start, left_end = groups[group_index - 1]
            mid_start, mid_end = groups[group_index]
            right_start, right_end = groups[group_index + 1]
            left_speaker = words[left_start].speaker
            mid_speaker = words[mid_start].speaker
            right_speaker = words[right_start].speaker
            if left_speaker != right_speaker or mid_speaker == left_speaker:
                continue
            middle = words[mid_start:mid_end]
            if any({"overlap", "needs_overlap_separation"} & set(word.flags) for word in middle):
                continue
            duration = middle[-1].end - middle[0].start
            char_count = len(join_word_texts(word.text for word in middle))
            if duration <= max_duration and char_count <= max_chars:
                for index in range(mid_start, mid_end):
                    flags = sorted(set(words[index].flags + ["speaker_smoothed"]))
                    words[index] = AssignedWord(
                        text=words[index].text,
                        start=words[index].start,
                        end=words[index].end,
                        speaker=left_speaker,
                        confidence=words[index].confidence,
                        flags=flags,
                        chunk_id=words[index].chunk_id,
                    )
                changed = True
                break
    return words


def _needs_space(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if right[0].isspace() or left[-1].isspace():
        return False
    return left[-1].isascii() and left[-1].isalnum() and right[0].isascii() and right[0].isalnum()


def join_word_texts(words: Iterable[str]) -> str:
    text = ""
    for word in words:
        if not word:
            continue
        if _needs_space(text, word):
            text += " "
        text += word.strip()
    return text.strip()


def _has_safe_soft_boundary(text: str) -> bool:
    return text.endswith(("。", "？", "！", "?", "!", "」", "』", "）", ")", "…", "、", ","))


def _segment_from_words(words: list[AssignedWord], segment_id: str) -> Segment:
    flags = sorted({flag for word in words for flag in word.flags})
    confidences = [word.confidence for word in words if word.confidence is not None]
    confidence = sum(confidences) / len(confidences) if confidences else None
    return Segment(
        speaker=words[0].speaker,
        start=words[0].start,
        end=max(word.end for word in words),
        text=join_word_texts(word.text for word in words),
        flags=flags,
        confidence=confidence,
        segment_id=segment_id,
    )


def build_segments(
    assigned_words: Iterable[AssignedWord],
    gap: float = 0.20,
    soft_duration: float = 3.0,
    hard_duration: float = 4.2,
    soft_chars: int = 22,
    hard_chars: int = 34,
) -> list[Segment]:
    ordered = sorted(list(assigned_words), key=lambda item: (item.start, item.end))
    if not ordered:
        return []

    segments: list[Segment] = []
    current: list[AssignedWord] = []

    def flush() -> None:
        if current:
            segments.append(_segment_from_words(current, f"seg_{len(segments):05d}"))
            current.clear()

    for word in ordered:
        if not current:
            current.append(word)
            continue

        current_text = join_word_texts(item.text for item in current)
        candidate_text = join_word_texts([current_text, word.text])
        candidate_duration = word.end - current[0].start
        gap_from_previous = word.start - current[-1].end
        speaker_changed = word.speaker != current[-1].speaker
        chunk_changed = word.chunk_id is not None and current[-1].chunk_id is not None and word.chunk_id != current[-1].chunk_id
        exceeds_hard = candidate_duration > hard_duration or len(candidate_text) > hard_chars
        exceeds_soft = candidate_duration >= soft_duration and len(candidate_text) >= soft_chars
        prefer_chunk_boundary = chunk_changed and (candidate_duration >= soft_duration or len(candidate_text) >= soft_chars)
        split = (
            speaker_changed
            or gap_from_previous > gap
            or prefer_chunk_boundary
            or exceeds_hard
            or (exceeds_soft and _has_safe_soft_boundary(current_text))
        )
        if split:
            flush()
        current.append(word)
    flush()

    return [segment for segment in segments if segment.end > segment.start and segment.text]


def words_and_turns_to_segments(
    words: Iterable[Word],
    turns: Iterable[Turn],
    *,
    activity: Iterable[SpeakerActivityFrame] | None = None,
    smooth_short_islands: bool = False,
) -> list[Segment]:
    assigned = assign_words_to_turns(words, turns, activity=activity)
    if smooth_short_islands:
        assigned = smooth_short_speaker_islands(assigned)
    return build_segments(assigned)


def segments_from_assigned_words(
    assigned_words: Iterable[AssignedWord],
    *,
    split_on_chunk: bool = False,
) -> list[Segment]:
    """Build subtitle segments from assigned words.

    With ``split_on_chunk`` the ASR sentence boundaries (chunk_id changes,
    e.g. qwen sentence chunks) are preserved as subtitle boundaries, and
    segments are only further split at speaker changes. Without it, words
    are grouped by speaker with no duration-based splitting.
    """

    ordered = sorted(list(assigned_words), key=lambda item: (item.start, item.end))
    if not ordered:
        return []

    segments: list[Segment] = []
    current: list[AssignedWord] = []

    def flush() -> None:
        if current:
            segments.append(_segment_from_words(current, f"seg_{len(segments):05d}"))
            current.clear()

    for word in ordered:
        if current:
            speaker_changed = word.speaker != current[-1].speaker
            chunk_changed = (
                split_on_chunk
                and word.chunk_id is not None
                and current[-1].chunk_id is not None
                and word.chunk_id != current[-1].chunk_id
            )
            if speaker_changed or chunk_changed:
                flush()
        current.append(word)
    flush()

    return [segment for segment in segments if segment.end > segment.start and segment.text]
