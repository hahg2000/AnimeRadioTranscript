from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from .asr import run_faster_whisper_asr
from .asr_kotoba import run_kotoba_asr
from .asr_qwen import run_qwen_asr
from .audio import prepared_audio_path
from .evaluation import normalize_japanese_text
from .io_utils import write_json


def _engine_runner(engine: str) -> Callable[..., tuple[Path, Path]]:
    if engine == "kotoba":
        return lambda out_dir: run_kotoba_asr(out_dir, device=None)
    if engine == "qwen":
        return lambda out_dir: run_qwen_asr(out_dir, device=None)
    if engine == "whisper":
        return lambda out_dir: run_faster_whisper_asr(
            out_dir,
            model_name="large-v3",
            language="ja",
            device="cuda",
            compute_type="float16",
            beam_size=5,
        )
    raise ValueError(f"Unknown engine: {engine}")


def _reference_text(evaluation_dir: Path) -> str:
    segments_path = evaluation_dir / "reference_segments.json"
    if not segments_path.exists():
        raise FileNotFoundError(segments_path)
    raw = json.loads(segments_path.read_text(encoding="utf-8"))
    return "".join(
        str(item.get("text") or item.get("normalized_text") or "")
        for item in sorted(raw, key=lambda value: (float(value.get("start", 0.0)), float(value.get("end", 0.0))))
    )


def _asr_text(out_dir: Path) -> str:
    segments_path = out_dir / "asr_segments.json"
    if not segments_path.exists():
        return ""
    raw = json.loads(segments_path.read_text(encoding="utf-8"))
    return "".join(str(item.get("text") or "") for item in raw)


def _cer(reference: str, hypothesis: str) -> tuple[int, int, float]:
    ref_norm = normalize_japanese_text(reference)
    hyp_norm = normalize_japanese_text(hypothesis)
    distance = _levenshtein(ref_norm, hyp_norm)
    return distance, len(ref_norm), (0.0 if not ref_norm else distance / len(ref_norm))


def _levenshtein(left: str, right: str) -> int:
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


def benchmark_asr(
    out_dir: str | Path,
    evaluation_dir: str | Path,
    engines: list[str] | None = None,
    *,
    output: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(out_dir)
    evaluation = Path(evaluation_dir)
    if not prepared_audio_path(out).exists():
        raise FileNotFoundError(prepared_audio_path(out))
    selected = engines or ["whisper", "kotoba", "qwen"]
    reference = _reference_text(evaluation)
    results: dict[str, Any] = {
        "schema_version": 1,
        "reference_chars": len(normalize_japanese_text(reference)),
        "engines": {},
    }
    for engine in selected:
        started = time.monotonic()
        try:
            _engine_runner(engine)(out)
            status = "ok"
        except Exception as exc:
            status = f"error: {type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - started
        hypothesis = _asr_text(out)
        distance, ref_chars, cer = _cer(reference, hypothesis)
        results["engines"][engine] = {
            "status": status,
            "elapsed_seconds": round(elapsed, 2),
            "char_errors": distance,
            "cer": round(cer, 4),
        }
        print(f"{engine}: status={status} elapsed={elapsed:.1f}s cer={cer:.4f}")
    target = Path(output) if output is not None else out / "benchmark_asr.json"
    write_json(target, results)
    return results
