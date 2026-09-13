from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .asr import run_faster_whisper_asr
from .asr_kotoba import run_kotoba_asr
from .asr_qwen import QWEN_MODEL_ID, run_qwen_asr
from .audio import prepare_audio
from .benchmark import benchmark_asr
from .compare import compare_and_write
from .diarization import diarize_out_dir
from .evaluation import evaluate_and_write, init_evaluation
from .model_paths import inspect_local_models
from .overlap import fuse_overlap_transcripts, overlap_out_dir
from .realign import realign_out_dir
from .relabel import relabel_out_dir
from .render import render_out_dir


def _print_path(label: str, path: str | Path) -> None:
    print(f"{label}: {path}")


def cmd_prepare(args: argparse.Namespace) -> int:
    target = prepare_audio(args.input, args.out, sample_rate=args.sample_rate, preserve_source=not args.no_preserve_source)
    _print_path("prepared", target)
    return 0


def cmd_asr(args: argparse.Namespace) -> int:
    words_path, segments_path = run_asr_engine(
        args.engine,
        args.out_dir,
        model_name=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        condition_on_previous_text=args.condition_on_previous_text,
        models_root=args.models_root,
        precision=args.precision,
    )
    _print_path("asr_words", words_path)
    _print_path("asr_segments", segments_path)
    return 0


def run_asr_engine(
    engine: str,
    out_dir: str | Path,
    **kwargs: object,
) -> tuple[Path, Path]:
    if engine == "kotoba":
        return run_kotoba_asr(
            out_dir,
            device=str(kwargs["device"]) if kwargs.get("device") else None,
        )
    if engine == "qwen":
        return run_qwen_asr(
            out_dir,
            device=str(kwargs["device"]) if kwargs.get("device") else None,
            model_name=str(kwargs.get("model_name") or QWEN_MODEL_ID),
        )
    return run_faster_whisper_asr(
        out_dir,
        model_name=str(kwargs.get("model_name") or "large-v3"),
        language=str(kwargs.get("language") or "ja"),
        device=str(kwargs.get("device") or "cuda"),
        compute_type=str(kwargs.get("compute_type") or "float16"),
        beam_size=int(kwargs.get("beam_size") or 5),
        vad_filter=bool(kwargs.get("vad_filter")),
        condition_on_previous_text=bool(kwargs.get("condition_on_previous_text")),
        models_root=kwargs.get("models_root"),
    )


def cmd_diarize(args: argparse.Namespace) -> int:
    json_path, rttm_path = diarize_out_dir(
        args.out_dir,
        diar_model=args.diar_model,
        speakers=args.speakers,
        rttm=args.rttm,
        models_root=args.models_root,
        device=args.diar_device,
    )
    _print_path("diarization_json", json_path)
    _print_path("diarization_rttm", rttm_path)
    return 0


def cmd_relabel(args: argparse.Namespace) -> int:
    segments_path, review_path = relabel_out_dir(
        args.out_dir,
        smooth_short_islands=args.smooth_short_islands,
    )
    _print_path("segments", segments_path)
    _print_path("review", review_path)
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    output = render_out_dir(
        args.out_dir,
        speaker_map_path=args.speaker_map,
        source=args.source,
        prefer_pysubs2=not args.no_pysubs2,
    )
    _print_path("ass", output)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    output = args.out
    if output is None:
        hyp_parent = Path(args.hypothesis).parent
        output = hyp_parent / "metrics.json"
    metrics = compare_and_write(args.reference, args.hypothesis, output)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    _print_path("metrics", output)
    return 0


def cmd_init_eval(args: argparse.Namespace) -> int:
    outputs = init_evaluation(args.reference, args.out, audio_path=args.audio)
    for label, path in outputs.items():
        _print_path(label, path)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    output = Path(args.out) if args.out else Path(args.hypothesis).parent / "evaluation_metrics.json"
    metrics = evaluate_and_write(args.reference, args.hypothesis, output, frame_step=args.frame_step)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    _print_path("evaluation_metrics", output)
    return 0


def cmd_inspect_models(args: argparse.Namespace) -> int:
    inventory = inspect_local_models(args.models_root)
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    return 0


def cmd_benchmark_asr(args: argparse.Namespace) -> int:
    metrics = benchmark_asr(
        args.out_dir,
        args.evaluation,
        engines=[item.strip() for item in args.engines.split(",") if item.strip()] if args.engines else None,
        output=args.out,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def cmd_overlap(args: argparse.Namespace) -> int:
    output = overlap_out_dir(
        args.out_dir,
        context_seconds=args.context,
        separator_cmd=args.separator_cmd,
        asr_engine=args.engine,
    )
    _print_path("segments_overlap", output)
    if args.fuse:
        fused = fuse_overlap_transcripts(args.out_dir, engines=["whisper", "kotoba"])
        _print_path("segments_overlap_fused", fused)
    return 0


def cmd_realign(args: argparse.Namespace) -> int:
    segments_path, words_path, anchored, total = realign_out_dir(
        args.out_dir,
        args.reference,
        apply=args.apply,
    )
    print(f"anchored {anchored}/{total} segments")
    _print_path("segments", segments_path)
    _print_path("words", words_path)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    prepare_audio(args.input, args.out, sample_rate=args.sample_rate, preserve_source=not args.no_preserve_source)
    run_asr_engine(
        args.engine,
        args.out,
        model_name=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        beam_size=args.beam_size,
        vad_filter=args.vad_filter,
        condition_on_previous_text=args.condition_on_previous_text,
        models_root=args.models_root,
        precision=args.precision,
    )
    diarize_out_dir(
        args.out,
        diar_model=args.diar_model,
        speakers=args.speakers,
        rttm=args.rttm,
        models_root=args.models_root,
        device=args.diar_device,
    )
    relabel_out_dir(
        args.out,
        smooth_short_islands=args.smooth_short_islands,
    )
    output = render_out_dir(args.out, speaker_map_path=args.speaker_map, prefer_pysubs2=not args.no_pysubs2)
    _print_path("ass", output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="drama-nemo-ass")
    parser.add_argument("--version", action="version", version="drama-nemo-ass 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Convert media to prepared.wav")
    prepare.add_argument("input")
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--sample-rate", type=int, default=16000)
    prepare.add_argument("--no-preserve-source", action="store_true")
    prepare.set_defaults(func=cmd_prepare)

    asr = subparsers.add_parser("asr", help="Run ASR (whisper, kotoba, or qwen) for an output directory")
    asr.add_argument("out_dir")
    asr.add_argument("--engine", choices=["whisper", "kotoba", "qwen"], default="whisper")
    asr.add_argument("--model", default="large-v3")
    asr.add_argument("--language", default="ja")
    asr.add_argument("--device", default="cuda")
    asr.add_argument("--compute-type", default="float16")
    asr.add_argument("--beam-size", type=int, default=5)
    asr.add_argument("--vad-filter", action="store_true")
    asr.add_argument("--condition-on-previous-text", action="store_true")
    asr.add_argument("--precision", default="fp32", choices=["fp32", "int8", "int8-fp32"])
    asr.add_argument("--models-root")
    asr.set_defaults(func=cmd_asr)

    diarize = subparsers.add_parser("diarize", help="Run NeMo Sortformer or import RTTM")
    diarize.add_argument("out_dir")
    diarize.add_argument("--diar-model", default="nvidia/diar_sortformer_4spk-v1")
    diarize.add_argument("--speakers", type=int, choices=[2, 3], required=True)
    diarize.add_argument("--rttm")
    diarize.add_argument("--models-root")
    diarize.add_argument("--diar-device", default="cuda")
    diarize.set_defaults(func=cmd_diarize)

    relabel = subparsers.add_parser("relabel", help="Assign ASR words to speakers and write review.tsv")
    relabel.add_argument("out_dir")
    relabel.add_argument("--smooth-short-islands", action="store_true")
    relabel.set_defaults(func=cmd_relabel)

    render = subparsers.add_parser("render", help="Render segments or review.tsv to output.ass")
    render.add_argument("out_dir")
    render.add_argument("--speaker-map")
    render.add_argument("--source", choices=["auto", "review", "segments", "overlap"], default="auto")
    render.add_argument("--no-pysubs2", action="store_true")
    render.set_defaults(func=cmd_render)

    compare = subparsers.add_parser("compare", help="Compare reference ASS and hypothesis ASS")
    compare.add_argument("reference")
    compare.add_argument("hypothesis")
    compare.add_argument("--out")
    compare.set_defaults(func=cmd_compare)

    init_eval = subparsers.add_parser("init-eval", help="Create evaluation reference artifacts from an ASS file")
    init_eval.add_argument("reference")
    init_eval.add_argument("--out", required=True)
    init_eval.add_argument("--audio")
    init_eval.set_defaults(func=cmd_init_eval)

    evaluate = subparsers.add_parser("evaluate", help="Run overlap-aware ASS evaluation")
    evaluate.add_argument("reference")
    evaluate.add_argument("hypothesis")
    evaluate.add_argument("--out")
    evaluate.add_argument("--frame-step", type=float, default=0.01)
    evaluate.set_defaults(func=cmd_evaluate)

    inspect_models = subparsers.add_parser("inspect-models", help="Resolve locally downloaded model paths")
    inspect_models.add_argument("--models-root")
    inspect_models.set_defaults(func=cmd_inspect_models)

    benchmark = subparsers.add_parser("benchmark-asr", help="Compare ASR engines (whisper, kotoba, qwen) against reference")
    benchmark.add_argument("out_dir")
    benchmark.add_argument("evaluation")
    benchmark.add_argument("--engines", default="whisper,kotoba,qwen")
    benchmark.add_argument("--out")
    benchmark.set_defaults(func=cmd_benchmark_asr)

    overlap = subparsers.add_parser("overlap", help="Detect and process overlapping speech regions")
    overlap.add_argument("out_dir")
    overlap.add_argument("--context", type=float, default=0.7)
    overlap.add_argument("--separator-cmd")
    overlap.add_argument("--engine", choices=["whisper", "kotoba", "qwen"], default="whisper")
    overlap.add_argument("--fuse", action="store_true", help="Fuse dual-engine clip transcripts into extra lines")
    overlap.set_defaults(func=cmd_overlap)

    realign = subparsers.add_parser("realign", help="Realign ASR timestamps to a reference (whisper) transcript")
    realign.add_argument("out_dir")
    realign.add_argument("--reference", required=True, help="Directory containing reference asr_segments.json")
    realign.add_argument("--apply", action="store_true", help="Overwrite asr_words.jsonl/asr_segments.json in place")
    realign.set_defaults(func=cmd_realign)

    run = subparsers.add_parser("run", help="Run prepare, ASR, diarization, relabel, and render")
    run.add_argument("input")
    run.add_argument("--out", required=True)
    run.add_argument("--language", default="ja")
    run.add_argument("--speakers", type=int, choices=[2, 3], required=True)
    run.add_argument("--speaker-map")
    run.add_argument("--sample-rate", type=int, default=16000)
    run.add_argument("--no-preserve-source", action="store_true")
    run.add_argument("--engine", choices=["whisper", "kotoba", "qwen"], default="whisper")
    run.add_argument("--precision", default="fp32", choices=["fp32", "int8", "int8-fp32"])
    run.add_argument("--model", default="large-v3")
    run.add_argument("--device", default="cuda")
    run.add_argument("--compute-type", default="float16")
    run.add_argument("--beam-size", type=int, default=5)
    run.add_argument("--vad-filter", action="store_true")
    run.add_argument("--condition-on-previous-text", action="store_true")
    run.add_argument("--models-root")
    run.add_argument("--diar-model", default="nvidia/diar_sortformer_4spk-v1")
    run.add_argument("--diar-device", default="cuda")
    run.add_argument("--rttm")
    run.add_argument("--smooth-short-islands", action="store_true")
    run.add_argument("--no-pysubs2", action="store_true")
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
