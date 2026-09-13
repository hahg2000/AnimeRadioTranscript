# 操作速查（WORKFLOW）

日常跑字幕的常用命令，按实际工作流排序。完整背景见 [RUNBOOK.md](RUNBOOK.md)（详细手册）、[README.md](README.md)（项目说明）、[spike_notes.md](spike_notes.md)（模型评测结论）。

## 0. 环境与路径

两个 venv（按引擎分，不能混用）：

| venv | 用途 |
| --- | --- |
| `~/.venvs/drama-nemo-ass` | whisper、kotoba、NeMo diarize（transformers 4.57） |
| `~/.venvs/drama-qwen` | qwen 引擎（transformers 5.15） |

```bash
cd /mnt/e/gitCode/drama-nemo-ass
export PYTHONPATH=/mnt/e/gitCode/drama-nemo-ass
```

模型目录：`/mnt/e/models/`（whisper、kotoba、qwen、Sortformer 都在这）。

## 1. 视频转码（AV1 → H.264 给 PR 用）

PR 不支持 AV1，先转 H.264：

```bash
# NVENC 硬件编码，快（13×实时），720p 质量好
ffmpeg -y -i "video/马戏团/01.mp4" \
  -c:v h264_nvenc -preset p4 -rc vbr -cq 19 -b:v 0 \
  -c:a aac -b:a 320k -movflags +faststart \
  "video/马戏团/01_h264.mp4"

# 确认输出编码
ffprobe -v error -show_entries stream=codec_name,codec_type,width,height -of default=noprint_wrappers=1 "video/马戏团/01_h264.mp4"
```

H.264 + AAC 是 PR 最稳的组合。原 AV1 文件保留不删。

## 1.5 截取片段（从 H.264 文件）

从已转好的 H.264 文件流复制截取（`-c copy` 无损、秒级完成，`-ss` 起点 `-to` 终点）：

```bash
# 截取 6:19 ~ 24:30
ffmpeg -y -ss 6:19 -to 24:30 -i "video/马戏团/02_h264.mp4" \
  -c copy -avoid_negative_ts make_zero \
  "video/马戏团/02_h264_cut.mp4"

# 确认结果时长（应 ≈ 24:30 - 6:19 = 18:11，即约 1095s）
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1 "video/马戏团/02_h264_cut.mp4"
```

要点：
- `-c copy` 直接复制编码流，不重新编码，快且不损画质。
- `-ss` 放 `-i` 之前是快速 seek，配合 `-c copy` 会落在最近的关键帧上；如需**精确到帧**，把 `-ss` 移到 `-i` 之后（会重新编码或慢速精确定位）。
- `-avoid_negative_ts make_zero` 让输出时间轴从 0 开始，后续 ASR 时间戳才对齐。
- 原文件保留不删。

## 2. 完整流水线（视频 → ASS）

```bash
source ~/.venvs/drama-nemo-ass/bin/activate
cd /mnt/e/gitCode/drama-nemo-ass
export PYTHONPATH=/mnt/e/gitCode/drama-nemo-ass

OUT="outputs/马戏团/01-cut"

# 2.1 提音频
python -m drama_nemo_ass prepare "video/马戏团/01_h264_12min.mp4" --out "$OUT"

# 2.2 ASR（whisper 主线）
python -m drama_nemo_ass asr "$OUT" --engine whisper --device cuda

# 2.3 说话人分离（>7min 需分块，见第 3 节；≤7min 直接跑）
python -m drama_nemo_ass diarize "$OUT" --speakers 2 --models-root /mnt/e/models

# 2.4 合并文本+说话人
python -m drama_nemo_ass relabel "$OUT"

# 2.5 渲染 ASS
python -m drama_nemo_ass render "$OUT"
```

产物：`$OUT/output.ass`（最终字幕）、`$OUT/review.tsv`（人工复核表）。

## 3. 长音频分块 diarize（>7 分钟）

Sortformer 在 >7min 音频上会报 `CUDA driver error: device not ready`，需切 180s 分块跑：

```bash
source ~/.venvs/drama-nemo-ass/bin/activate
cd /mnt/e/gitCode/drama-nemo-ass
export PYTHONPATH=/mnt/e/gitCode/drama-nemo-ass

OUT="outputs/马戏团/01-cut"

# 3.1 切 180s 块
mkdir -p "$OUT/chunks"
ffmpeg -y -v error -i "$OUT/prepared.wav" -f segment -segment_time 180 -c copy "$OUT/chunks/chunk_%03d.wav"

# 3.2 分块 diarize（自动合并回全局时间轴）
python scripts/diarize_chunked.py "$OUT" --speakers 2

# 3.3 继续流水线
python -m drama_nemo_ass relabel "$OUT"
python -m drama_nemo_ass render "$OUT"
```

## 4. 三引擎对照（whisper + kotoba + qwen）

用途：日语不熟时，交叉参考三个模型判断文本正确性。

原则：kotoba（日语微调 whisper）最准，qwen 有标点可辅助断句；三者一致基本可信；专名三家都不可靠，只能查原作设定。

**为什么要 VAD 窗口（默认）**：whisper 漏掉的句子不会出现在任何对照行里（whisper 独自定义时间轴）。改用 Silero VAD 独立检测发声区间作为切片基准后，whisper 漏掉的地方 kotoba/qwen 仍会转写，缺句会暴露出来。实测 03 集：whisper 在 714.56→744.64 有整 30 秒空洞，VAD 窗口版把这段找回（22 个窗口 whisper 为空但其他引擎有内容）。

**为什么要逐窗重跑 whisper**：whisper 长音频解码会整段漏切（同一模型、同一音频，孤立窗口能转对，30 秒空洞复测 25/28 找回）。逐窗重跑后，03 集 whisper 空窗口 22→2、部分漏 6→2。

```bash
# 4.1 先跑 whisper 全片（提供长音频基线；对照 whisper 列会优先用逐窗版）
python -m drama_nemo_ass asr "$OUT" --engine whisper --device cuda

# 4.2 生成 VAD 窗口（nemo venv，需 silero-vad）
python scripts/vad_segments.py "$OUT"
# 调参：--threshold 0.35 --merge-gap 0.35 --pad 0.20 --max-seconds 28 --energy-union

# 4.3 whisper 按 VAD 窗口重跑（找回长音频漏句；nemo venv）
python scripts/transcribe_segments.py "$OUT" --engine whisper

# 4.4 kotoba 按 VAD 窗口转写（nemo venv；默认 --boundaries vad）
python scripts/transcribe_segments.py "$OUT" --engine kotoba

# 4.5 qwen 按 VAD 窗口转写（切到 qwen venv）
source ~/.venvs/drama-qwen/bin/activate
python scripts/transcribe_segments.py "$OUT" --engine qwen

# 4.6 合并成对照表（默认 VAD 模式）
python scripts/merge_comparison.py "$OUT"
```

产物：`$OUT/comparison.tsv`（制表符分隔，5 列：start / end / whisper / kotoba / qwen）。
whisper 列优先用逐窗结果（`asr_segments_whisper_vad.json`），缺的窗口回退长音频时间重叠映射；whisper 为空但 kotoba/qwen 有内容的行，就是漏掉的句子。

用 Excel/WPS 打开看最方便，三列并排逐句对照。

旧版 whisper 边界模式（复现历史对照，供回归）：

```bash
python scripts/transcribe_segments.py "$OUT" --engine kotoba --boundaries whisper
python scripts/transcribe_segments.py "$OUT" --engine qwen --boundaries whisper
python scripts/merge_comparison.py "$OUT" --boundaries whisper
```

### 4.7 人工校对后从对照表生成 ASS

在 `comparison_reviewed.tsv`（三引擎对照 + 人工校对后的"最终原文"+"中文意思"列）基础上，直接按"最终原文"列渲染 ASS：

```bash
# 用「最终原文」列生成 ASS（单说话人）
python scripts/ass_from_reviewed.py "$OUT"

# 可选：指定输出文件名
python scripts/ass_from_reviewed.py "$OUT" --output output_reviewed.ass
```

行为：
- 若 `$OUT/nemo_diarization.json` 存在，按时间中点自动分配说话人（双色字幕）；否则全部单说话人（speaker0）。
- 产物：`$OUT/output_reviewed.ass`（默认）。

生成中文版 ASS（一句日文替换一句中文）需要额外脚本或手动替换，见当时的处理。

## 5. 常用场景速查

| 想做什么 | 命令 |
| --- | --- |
| 只改 review.tsv 后重出字幕 | `python -m drama_nemo_ass render "$OUT" --source review` |
| 换 ASR 模型 | 从 `asr` 重跑 |
| 换说话人数 | 从 `diarize`（或分块 diarize）重跑 |
| 换视频 | 从 `prepare` 完整重跑 |
| 看本地有哪些模型 | `python -m drama_nemo_ass inspect-models --models-root /mnt/e/models` |

## 6. 主要产物索引

| 文件 | 作用 |
| --- | --- |
| `prepared.wav` | 16k 单声道模型输入 |
| `asr_segments.json` | whisper 分段结果（对照的 whisper 列） |
| `asr_words.jsonl` | whisper 词级时间戳 |
| `vad_windows.json` | Silero VAD 发声窗口（对照切片基准） |
| `asr_segments_whisper_vad.json` | whisper 逐窗重跑结果（找回长音频漏句） |
| `asr_segments_kotoba_vad.json` / `asr_segments_qwen_vad.json` | VAD 窗口对照结果（默认） |
| `asr_segments_kotoba.json` / `asr_segments_qwen.json` | whisper 边界对照结果（旧模式） |
| `comparison.tsv` | 三引擎对照表 |
| `comparison_reviewed.tsv` | 三引擎对照 + 人工校对的"最终原文/中文意思" |
| `output_reviewed.ass` | 由对照表"最终原文"列生成的 ASS |
| `nemo_diarization.json` | 说话人时间线 |
| `segments.json` / `review.tsv` | 合并结果 / 人工复核表 |
| `output.ass` | 最终 ASS 字幕 |
