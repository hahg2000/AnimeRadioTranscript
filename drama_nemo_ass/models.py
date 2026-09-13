from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item for item in value.split("|") if item]
    return [str(item) for item in value]


@dataclass(slots=True)
class Word:
    text: str
    start: float
    end: float
    language: str = "ja"
    probability: float | None = None
    chunk_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Word":
        return cls(
            text=str(data.get("text", "")),
            start=_float(data.get("start")),
            end=_float(data.get("end")),
            language=str(data.get("language", "ja")),
            probability=None if data.get("probability") is None else float(data["probability"]),
            chunk_id=None if data.get("chunk_id") is None else str(data["chunk_id"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AsrSegment:
    text: str
    start: float
    end: float
    language: str = "ja"
    chunk_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AsrSegment":
        return cls(
            text=str(data.get("text", "")),
            start=_float(data.get("start")),
            end=_float(data.get("end")),
            language=str(data.get("language", "ja")),
            chunk_id=None if data.get("chunk_id") is None else str(data["chunk_id"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Turn:
    speaker: str
    start: float
    end: float
    confidence: float | None = None
    source: str = "nemo"
    flags: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Turn":
        return cls(
            speaker=str(data.get("speaker", "UNKNOWN")),
            start=_float(data.get("start")),
            end=_float(data.get("end")),
            confidence=None if data.get("confidence") is None else float(data["confidence"]),
            source=str(data.get("source", "nemo")),
            flags=_str_list(data.get("flags")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SpeakerActivityFrame:
    start: float
    end: float
    scores: dict[str, float]
    source: str = "nemo"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpeakerActivityFrame":
        return cls(
            start=_float(data.get("start")),
            end=_float(data.get("end")),
            scores={str(name): float(score) for name, score in dict(data.get("scores", {})).items()},
            source=str(data.get("source", "nemo")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Segment:
    speaker: str
    start: float
    end: float
    text: str
    flags: list[str] = field(default_factory=list)
    confidence: float | None = None
    segment_id: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Segment":
        return cls(
            speaker=str(data.get("speaker", "UNKNOWN")),
            start=_float(data.get("start")),
            end=_float(data.get("end")),
            text=str(data.get("text", "")),
            flags=_str_list(data.get("flags")),
            confidence=None if data.get("confidence") is None else float(data["confidence"]),
            segment_id=None if data.get("segment_id") is None else str(data["segment_id"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
