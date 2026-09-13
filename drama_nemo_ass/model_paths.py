from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


def candidate_model_roots(models_root: str | Path | None = None) -> list[Path]:
    values: list[str | Path] = []
    if models_root is not None:
        values.append(models_root)
    if os.environ.get("DRAMA_MODELS_ROOT"):
        values.append(os.environ["DRAMA_MODELS_ROOT"])
    values.extend([Path("/mnt/e/models"), Path("E:/models")])
    roots: list[Path] = []
    for value in values:
        path = Path(value).expanduser()
        if path not in roots and path.exists():
            roots.append(path)
    return roots


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists() or os.path.lexists(candidate):
            return candidate
    return None


def _snapshot_files(root: Path, repo_cache_name: str, filename: str | None = None) -> Iterable[Path]:
    snapshots = root / "huggingface" / "hub" / repo_cache_name / "snapshots"
    if not snapshots.exists():
        return []
    if filename is None:
        return sorted(path for path in snapshots.iterdir() if path.is_dir())
    return sorted(snapshots.glob(f"*/{filename}"))


def resolve_faster_whisper_model(model_name: str, models_root: str | Path | None = None) -> str:
    explicit = Path(model_name).expanduser()
    if explicit.exists():
        return str(explicit)
    if model_name not in {"large-v3", "Systran/faster-whisper-large-v3"}:
        return model_name
    for root in candidate_model_roots(models_root):
        found = _first_existing(
            [root / "faster-whisper-large-v3", *_snapshot_files(root, "models--Systran--faster-whisper-large-v3")]
        )
        if found is not None:
            return str(found)
    return model_name


def resolve_qwen_model(model_name: str, models_root: str | Path | None = None) -> str:
    explicit = Path(model_name).expanduser()
    if explicit.exists():
        return str(explicit)
    if model_name not in {
        "Qwen/Qwen3-ASR-1.7B-hf",
        "Qwen/Qwen3-ASR-0.6B-hf",
        "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf",
    }:
        return model_name
    repo_cache_name = "models--" + model_name.replace("/", "--")
    for root in candidate_model_roots(models_root):
        found = _first_existing(
            [root / model_name.split("/")[-1], *_snapshot_files(root, repo_cache_name)]
        )
        if found is not None:
            return str(found)
    return model_name


def resolve_sortformer_model(model_name: str, models_root: str | Path | None = None) -> str:
    explicit = Path(model_name).expanduser()
    if explicit.exists():
        if explicit.is_dir():
            nemo_files = sorted(explicit.glob("*.nemo"))
            return str(nemo_files[0]) if nemo_files else str(explicit)
        return str(explicit)
    if model_name not in {"nvidia/diar_sortformer_4spk-v1", "diar_sortformer_4spk-v1"}:
        return model_name
    for root in candidate_model_roots(models_root):
        found = _first_existing(
            [
                root / "diar_sortformer_4spk-v1.nemo",
                *_snapshot_files(
                    root,
                    "models--nvidia--diar_sortformer_4spk-v1",
                    "diar_sortformer_4spk-v1.nemo",
                ),
            ]
        )
        if found is not None:
            return str(found)
    return model_name


def inspect_local_models(models_root: str | Path | None = None) -> list[dict[str, object]]:
    roots = candidate_model_roots(models_root)
    whisper = resolve_faster_whisper_model("large-v3", models_root)
    sortformer = resolve_sortformer_model("nvidia/diar_sortformer_4spk-v1", models_root)
    qwen = resolve_qwen_model("Qwen/Qwen3-ASR-1.7B-hf", models_root)

    def item(stage: str, requested: str, resolved: str, required: list[str]) -> dict[str, object]:
        path = Path(resolved)
        local = path.exists() or os.path.lexists(path)
        return {
            "stage": stage,
            "requested": requested,
            "resolved": resolved,
            "local": local,
            "required_python_packages": required,
        }

    return [
        {"stage": "model_roots", "roots": [str(root) for root in roots], "local": bool(roots)},
        item("asr", "large-v3", whisper, ["faster-whisper"]),
        item("asr", "Qwen/Qwen3-ASR-1.7B-hf", qwen, ["transformers>=5.13.0"]),
        item("diarization", "nvidia/diar_sortformer_4spk-v1", sortformer, ["nemo_toolkit[asr]"]),
    ]
