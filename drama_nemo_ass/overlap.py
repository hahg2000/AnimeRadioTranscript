from __future__ import annotations

import subprocess
import wave
from pathlib import Path
from typing import Any, Sequence

from .audio import prepared_audio_path
from .io_utils import read_json, read_segments, write_json
from .models import Segment

CONTEXT_SECONDS = 0.7
MIN_REGION_SECONDS = 0.15


def _clip_wav(source: Path, target: Path, start: float, end: float) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        rate = reader.getframerate()
        start_frame = max(0, int(start * rate))
        end_frame = max(start_frame + 1, int(end * rate))
        reader.setpos(min(start_frame, reader.getnframes()))
        frames = reader.readframes(max(0, min(end_frame, reader.getnframes()) - start_frame))
    with wave.open(str(target), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)
    return target


def read_overlap_regions(out_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(out_dir) / "overlap_regions.json"
    if not path.exists():
        return []
    return [dict(item) for item in read_json(path)]


def plan_overlap_jobs(
    out_dir: str | Path,
    context_seconds: float = CONTEXT_SECONDS,
    min_region_seconds: float = MIN_REGION_SECONDS,
) -> Path:
    out = Path(out_dir)
    audio = prepared_audio_path(out)
    if not audio.exists():
        raise FileNotFoundError(f"Missing prepared audio: {audio}")
    regions = read_overlap_regions(out)
    jobs: list[dict[str, Any]] = []
    job_dir = out / "overlap_jobs"
    for index, region in enumerate(regions):
        start = float(region.get("start", 0.0))
        end = float(region.get("end", 0.0))
        if end - start < min_region_seconds:
            continue
        speakers = sorted(str(speaker) for speaker in region.get("speakers", []))
        if len(speakers) < 2:
            continue
        context_start = max(0.0, start - context_seconds)
        context_end = end + context_seconds
        job_id = f"job_{index:05d}"
        clip_path = job_dir / f"{job_id}.wav"
        _clip_wav(audio, clip_path, context_start, context_end)
        jobs.append(
            {
                "job_id": job_id,
                "region_start": round(start, 3),
                "region_end": round(end, 3),
                "clip_start": round(context_start, 3),
                "clip_end": round(context_end, 3),
                "speakers": speakers,
                "clip_path": str(clip_path),
            }
        )
    return write_json(job_dir / "overlap_jobs.json", jobs)


def _worker_stub(job_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    job = read_json(job_path)
    out = Path(output_dir)
    stems: list[dict[str, Any]] = []
    for speaker in job.get("speakers", []):
        stems.append(
            {
                "speaker": speaker,
                "stem_path": None,
                "status": "needs_review",
                "reason": "separator_not_configured",
            }
        )
    result = {
        "job_id": job.get("job_id"),
        "status": "needs_review",
        "reason": "separator_not_configured",
        "stems": stems,
    }
    write_json(out / f"{job.get('job_id')}_result.json", result)
    return result


def run_overlap_separation(
    out_dir: str | Path,
    separator_cmd: str | None = None,
) -> Path:
    out = Path(out_dir)
    jobs_path = out / "overlap_jobs" / "overlap_jobs.json"
    if not jobs_path.exists():
        raise FileNotFoundError(f"Missing overlap jobs: {jobs_path}")
    jobs = read_json(jobs_path)
    results: list[dict[str, Any]] = []
    for job in jobs:
        job_path = out / "overlap_jobs" / f"{job['job_id']}.json"
        write_json(job_path, job)
        result_dir = out / "overlap_results"
        result_dir.mkdir(parents=True, exist_ok=True)
        if separator_cmd is None:
            result = _worker_stub(job_path, result_dir)
        else:
            command = [item for item in separator_cmd.split() if item]
            completed = subprocess.run(
                command + [str(job_path), str(result_dir)],
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                result = {
                    "job_id": job["job_id"],
                    "status": "error",
                    "reason": completed.stderr.strip()[:500] or "separator_failed",
                    "stems": [
                        {"speaker": speaker, "stem_path": None, "status": "error"}
                        for speaker in job["speakers"]
                    ],
                }
            else:
                result_path = result_dir / f"{job['job_id']}_result.json"
                result = read_json(result_path) if result_path.exists() else {"job_id": job["job_id"], "status": "error", "reason": "no_result_file"}
        results.append(result)
    return write_json(out / "overlap_results" / "overlap_results.json", results)


def stem_quality(
    stem_text: str,
    sibling_text: str,
    *,
    duplicate_threshold: float = 0.6,
) -> dict[str, Any]:
    stem_norm = _normalize_for_compare(stem_text)
    sibling_norm = _normalize_for_compare(sibling_text)
    empty = not stem_norm
    if empty or not sibling_norm:
        similarity = 0.0
    else:
        similarity = _similarity(stem_norm, sibling_norm)
    return {
        "empty": empty,
        "duplicate": similarity > duplicate_threshold,
        "sibling_similarity": round(similarity, 3),
        "pass": bool(stem_norm) and similarity <= duplicate_threshold,
    }


def _normalize_for_compare(text: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKC", text)
    return "".join(
        char
        for char in normalized
        if not char.isspace() and unicodedata.category(char)[0] not in {"P", "S"}
    )


def _similarity(left: str, right: str) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    matches = 0
    for char in left:
        if char in right:
            matches += 1
    shorter = min(len(left), len(right))
    return matches / max(1, shorter)


def merge_overlap_segments(
    out_dir: str | Path,
    stem_segments: Sequence[dict[str, Any]],
) -> Path:
    out = Path(out_dir)
    segments = read_segments(out / "segments.json")
    merged: list[Segment] = []
    for item in stem_segments:
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        merged.append(
            Segment(
                speaker=str(item.get("speaker") or "UNKNOWN"),
                start=float(item.get("start", 0.0)),
                end=float(item.get("end", 0.0)),
                text=text,
                flags=sorted(set(item.get("flags", []))),
                confidence=None,
                segment_id=f"overlap_{item.get('job_id', '')}_{item.get('speaker', '')}",
            )
        )
    merged.sort(key=lambda item: (item.start, item.end, item.speaker))
    all_segments = sorted(segments + merged, key=lambda item: (item.start, item.end, item.speaker))
    return write_json(out / "segments_overlap.json", [segment.to_dict() for segment in all_segments])


def assess_overlap_results(
    stem_texts_by_job: dict[str, dict[str, str]],
    results: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for result in results:
        job_id = str(result.get("job_id", ""))
        texts = stem_texts_by_job.get(job_id, {})
        stems: list[dict[str, Any]] = []
        for stem in result.get("stems", []):
            speaker = str(stem.get("speaker", ""))
            sibling_texts = [text for other, text in texts.items() if other != speaker]
            quality = stem_quality(
                texts.get(speaker, ""),
                sibling_texts[0] if sibling_texts else "",
            )
            stems.append({**stem, **quality})
        output.append({**result, "stems": stems})
    return output


def overlap_out_dir(
    out_dir: str | Path,
    context_seconds: float = CONTEXT_SECONDS,
    separator_cmd: str | None = None,
    asr_engine: str = "whisper",
) -> Path:
    out = Path(out_dir)
    plan_overlap_jobs(out, context_seconds=context_seconds)
    results_path = run_overlap_separation(out, separator_cmd=separator_cmd)
    results = read_json(results_path)

    jobs = read_json(out / "overlap_jobs" / "overlap_jobs.json")
    jobs_by_id = {str(job["job_id"]): job for job in jobs}

    texts_by_job: dict[str, dict[str, str]] = {}
    for result in results:
        job_id = str(result.get("job_id", ""))
        stem_texts: dict[str, str] = {}
        for stem in result.get("stems", []):
            stem_path = stem.get("stem_path")
            if not stem_path or not Path(stem_path).exists():
                continue
            _, segments_path = _transcribe_stem(out, Path(stem_path), asr_engine)
            stem_texts[str(stem.get("speaker", ""))] = _segments_text(segments_path)
        texts_by_job[job_id] = stem_texts

    assessed = assess_overlap_results(texts_by_job, results)

    stem_segments: list[dict[str, Any]] = []
    for result in assessed:
        job_id = str(result.get("job_id", ""))
        job = jobs_by_id.get(job_id, {})
        for stem in result.get("stems", []):
            text = texts_by_job.get(job_id, {}).get(str(stem.get("speaker", "")), "")
            if not text.strip():
                continue
            flags = ["overlap", "multi_lane"]
            if not stem.get("pass", False):
                flags.append("needs_review")
            stem_segments.append(
                {
                    "job_id": job_id,
                    "speaker": str(stem.get("speaker") or "UNKNOWN"),
                    "start": float(job.get("clip_start", 0.0)),
                    "end": float(job.get("clip_end", 0.0)),
                    "text": text,
                    "flags": flags,
                }
            )
    merge_path = merge_overlap_segments(out, stem_segments)
    write_json(out / "overlap_results" / "overlap_stem_segments.json", stem_segments)
    return merge_path


def _transcribe_stem(out_dir: Path, stem_wav: Path, engine: str) -> tuple[Path, Path]:
    stem_out = stem_wav.parent / (stem_wav.stem + "_asr")
    stem_out.mkdir(parents=True, exist_ok=True)
    from .asr import run_faster_whisper_asr

    return run_faster_whisper_asr(
        stem_out,
        language="ja",
        device="cuda",
        compute_type="float16",
        audio_path=stem_wav,
    )


def _segments_text(segments_path: Path) -> str:
    if not segments_path.exists():
        return ""
    return "".join(str(item.get("text") or "") for item in read_json(segments_path))


def fuse_overlap_transcripts(
    out_dir: str | Path,
    *,
    engines: Sequence[str] = ("whisper", "kotoba"),
    coverage_threshold: float = 0.6,
    min_chars: int = 3,
) -> Path:
    """Fuse per-clip dual-engine transcripts into extra multi-lane segments.

    Reads ``overlap_results/fusion_<engine>/<job_id>.json`` (written by
    ``scripts/exp_fusion_*.py``), keeps sentences not already covered by the
    main-line segments, and attributes each to the region speaker whose
    main-line text is least similar (the "other" speaker).

    Writes ``segments_overlap.json`` = main segments + fused extras.
    """

    out = Path(out_dir)
    jobs = read_json(out / "overlap_jobs" / "overlap_jobs.json")
    segments = read_segments(out / "segments.json")

    fused_segments: list[Segment] = []
    for job in jobs:
        job_id = str(job.get("job_id", ""))
        clip_start = float(job.get("clip_start", 0.0))
        clip_end = float(job.get("clip_end", 0.0))
        speakers = [str(speaker) for speaker in job.get("speakers", [])]

        region_baseline = [
            segment
            for segment in segments
            if segment.end > clip_start and segment.start < clip_end
        ]
        baseline_by_speaker: dict[str, list[Segment]] = {}
        for segment in region_baseline:
            baseline_by_speaker.setdefault(segment.speaker, []).append(segment)

        engine_texts: list[tuple[str, float, float]] = []
        for engine in engines:
            path = out / "overlap_results" / f"fusion_{engine}" / f"{job_id}.json"
            if not path.exists():
                continue
            data = read_json(path)
            if isinstance(data, dict):
                continue
            for item in data:
                text = str(item.get("text") or "").strip()
                if not text:
                    continue
                start = clip_start + float(item.get("start", 0.0))
                end = clip_start + float(item.get("end", float(item.get("start", 0.0)) + 1.0))
                engine_texts.append((text, min(start, clip_end), min(clip_end, max(start + 0.1, end))))

        candidates: list[tuple[str, float, float]] = []
        for text, start, end in engine_texts:
            norm = _normalize_for_compare(text)
            if not norm:
                continue
            merged = False
            for index, (existing, existing_start, existing_end) in enumerate(candidates):
                existing_norm = _normalize_for_compare(existing)
                shorter = min(len(norm), len(existing_norm))
                if shorter and _similarity(norm, existing_norm) >= 0.5:
                    if len(norm) > len(existing_norm):
                        candidates[index] = (text, start, end)
                    merged = True
                    break
            if not merged:
                candidates.append((text, start, end))

        baseline_texts = [segment.text for segment in region_baseline]
        viable: list[tuple[str, float, float]] = []
        for text, start, end in candidates:
            norm = _normalize_for_compare(text)
            if not norm or len(norm) < min_chars:
                continue
            if any(_similarity(norm, _normalize_for_compare(base)) >= coverage_threshold for base in baseline_texts):
                continue
            viable.append((text, start, end))

        max_extras = max(1, len(speakers))
        extras: list[tuple[str, float, float, str]] = []
        used_speakers: set[str] = set()
        for text, start, end in sorted(viable, key=lambda item: -len(_normalize_for_compare(item[0]))):
            if len(extras) >= max_extras:
                break
            norm = _normalize_for_compare(text)
            speaker_scores: list[tuple[float, str]] = []
            for speaker in speakers:
                speaker_texts = [segment.text for segment in baseline_by_speaker.get(speaker, [])]
                similarity = max(
                    (_similarity(norm, _normalize_for_compare(base)) for base in speaker_texts),
                    default=0.0,
                )
                speaker_scores.append((similarity, speaker))
            speaker_scores.sort(key=lambda item: (item[0], item[1]))
            chosen = next((score for score in speaker_scores if score[1] not in used_speakers), speaker_scores[0])[1] if speaker_scores else (speakers[0] if speakers else "UNKNOWN")
            used_speakers.add(chosen)
            extras.append((text, start, end, chosen))

        for text, start, end, speaker in extras:
            fused_segments.append(
                Segment(
                    speaker=speaker,
                    start=start,
                    end=end,
                    text=text,
                    flags=["overlap", "multi_lane", "fused"],
                    confidence=None,
                    segment_id=f"fused_{job_id}_{speaker}",
                )
            )

    all_segments = sorted(
        segments + fused_segments,
        key=lambda item: (item.start, item.end, item.speaker),
    )
    return write_json(out / "segments_overlap.json", [segment.to_dict() for segment in all_segments])
