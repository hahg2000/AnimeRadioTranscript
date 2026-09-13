# WSL 完整运行手册

本文档说明如何在 WSL 中从视频生成 ASS 字幕。

默认示例：

```bash
cd /mnt/e/gitCode/drama-nemo-ass
INPUT=video/02.mp4
OUT=outputs/02
```

完整流程如下：

```text
视频 -> 准备音频 -> 本地 ASR -> 说话人分离 -> 合并字幕
     -> 第一版 ASS -> 人工检查 -> 最终 ASS
```

## 1. 首次配置 WSL 环境

创建并进入 Python 虚拟环境：

```bash
cd /mnt/e/gitCode/drama-nemo-ass
python3.10 -m venv ~/.venvs/drama-nemo-ass
source ~/.venvs/drama-nemo-ass/bin/activate
python -m pip install -U pip setuptools wheel
```

作用：把本项目的 Python 和模型依赖隔离在独立环境中，避免与 WSL 系统 Python 冲突。

安装匹配的 CUDA 版 PyTorch 和项目依赖：

```bash
pip install --no-cache-dir \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements-wsl.txt
```

作用：安装 faster-whisper、NeMo、PyTorch 及音频处理依赖。本地 ASR 和说话人分离会使用 NVIDIA GPU。

检查环境：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
ffmpeg -version
python -m drama_nemo_ass --help
```

预期 `torch.cuda.is_available()` 为 `True`。更完整的 WSL、CUDA 和 Hugging Face 故障处理见 [README.md](README.md)。

当前机器的既有 venv 能真实运行 Whisper 和 Sortformer，但 `pip check` 仍报告 NeMo 2.7.3 对 Torch `>=2.6` / fsspec `==2024.12.0`，以及 Lightning 对 packaging/fsspec 的元数据约束不一致。不要在这个共享 venv 中单独升级某一个核心包；准备升级 NeMo/PyTorch 时应新建 venv，按同一 CUDA wheel 系列重装并重跑本手册的模型冒烟测试。此次 Sortformer 推理成功不等于依赖锁已经完全干净。

每次新开 WSL 终端后，只需要重新激活虚拟环境：

```bash
cd /mnt/e/gitCode/drama-nemo-ass
source ~/.venvs/drama-nemo-ass/bin/activate
```

## 2. 设置本次任务路径

```bash
cd /mnt/e/gitCode/drama-nemo-ass
source ~/.venvs/drama-nemo-ass/bin/activate

INPUT=video/02.mp4
OUT=outputs/02
```

作用：后续命令统一使用 `$INPUT` 和 `$OUT`，换视频时只需要修改这两个变量。

确认输入文件存在：

```bash
ls -lh "$INPUT"
```

## 3. 准备音频

```bash
python -m drama_nemo_ass prepare "$INPUT" --out "$OUT"
```

作用：使用 FFmpeg 从视频提取音轨，并转换为模型需要的 WAV 格式。

主要产物：

```text
outputs/02/prepared.wav
outputs/02/prepared_mono_16k.wav
outputs/02/prepared_source.wav
```

`prepared.wav` 与 `prepared_mono_16k.wav` 是 16 kHz 单声道模型输入；`prepared_source.wav` 保留源采样率和声道，供后续声像、分离和目标说话人提取实验使用。只有磁盘空间明确受限时才传 `--no-preserve-source`。

## 4. 本地语音识别

```bash
python -m drama_nemo_ass asr "$OUT" \
  --model large-v3 \
  --models-root /mnt/e/models \
  --language ja
```

作用：使用 faster-whisper `large-v3` 在本地识别日语，生成文本、分段时间和词级时间戳。

主要产物：

```text
outputs/02/asr_segments.json
outputs/02/asr_words.jsonl
```

第一次运行可能从 Hugging Face 下载模型。已有本地模型时可以改为：

```bash
python -m drama_nemo_ass asr "$OUT" \
  --model /mnt/e/models/faster-whisper-large-v3 \
  --language ja
```

该步骤会占用本地 GPU 和较长运行时间。

## 5. 本地说话人分离

```bash
python -m drama_nemo_ass diarize "$OUT" --speakers 3
```

作用：使用 NeMo 判断每个时间段由哪位说话人发言。`--speakers 3` 表示预计有 3 位说话人；如果音频只有 2 人，应改为 `--speakers 2`。

主要产物：

```text
outputs/02/nemo_diarization.json
outputs/02/nemo_diarization.rttm
outputs/02/speaker_activity.jsonl
outputs/02/exclusive_turns.json
outputs/02/overlap_regions.json
```

已有 `E:\models` 时推荐：

```bash
python -m drama_nemo_ass inspect-models --models-root /mnt/e/models
python -m drama_nemo_ass diarize "$OUT" \
  --speakers 3 \
  --models-root /mnt/e/models \
  --diar-device cuda
```

`speaker_activity.jsonl` 保存约 0.08 秒一帧的每位 speaker 活动概率。`exclusive_turns.json` 供普通 ASR 对齐，`overlap_regions.json` 保留需要送入后续分离/TSE 分支的多人区间。

该步骤会使用本地 GPU。

## 6. 合并文本和说话人

```bash
python -m drama_nemo_ass relabel "$OUT"
```

作用：把 ASR 词级时间戳分配到 NeMo 的说话人时间线上，生成可以渲染的字幕段，并标记重叠、低置信度和说话人边界风险。

主要产物：

```text
outputs/02/segments.json
outputs/02/review.tsv
```

`segments.json` 是本地主线结果；`review.tsv` 是便于人工修改的表格文件。

词到说话人的归属会优先积分 `speaker_activity.jsonl`；多人同时超过阈值时写入 `overlap|needs_overlap_separation`。短插话默认不会再被 A-B-A 平滑吞掉，如需兼容旧行为才显式增加 `--smooth-short-islands`。

### 6.1 建立重叠感知评测基线

```bash
python -m drama_nemo_ass init-eval video/sample.ass \
  --audio video/02.mp4 \
  --out evaluation/02

python -m drama_nemo_ass evaluate video/sample.ass "$OUT/output.ass" \
  --out evaluation/02/current_metrics.json
```

主要产物为 `manifest.json`、`reference.rttm`、`reference_segments.json` 和 `current_metrics.json`。模型替换至少同时比较 CER/cpCER、DER/JER 与 OSD F1，不能只看普通 CER。

## 7. 生成第一版 ASS

```bash
python -m drama_nemo_ass render "$OUT"
```

作用：把本地字幕段渲染为 Aegisub、mpv 等工具可读取的 ASS 字幕。

主要产物：

```text
outputs/02/output.ass
```

以上第 3 至第 7 步也可以合并为一条命令：

```bash
python -m drama_nemo_ass run "$INPUT" \
  --out "$OUT" \
  --language ja \
  --speakers 3
```

## 8. 人工检查 review.tsv

可以在 Windows 侧使用表格编辑器打开：

```text
E:\gitCode\drama-nemo-ass\outputs\02\review.tsv
```

重点检查带有以下标记的行：

- `overlap`：本地模型检测到说话重叠。
- `nearest_speaker`：词没有直接落入说话人区间，使用了最近说话人。
- `low_confidence`：本地 ASR 置信度较低。

可直接修改 `speaker` 和 `text` 列。不要修改 `segment_id`、`start`、`end` 的格式。

## 9. 生成最终 ASS

```bash
python -m drama_nemo_ass render "$OUT" --source review
```

作用：明确使用人工检查后的 `review.tsv` 生成最终字幕。

最终产物：

```text
outputs/02/output.ass
```

简单检查字幕条数：

```bash
grep -c '^Dialogue:' "$OUT/output.ass"
```

查看最终几条字幕：

```bash
grep '^Dialogue:' "$OUT/output.ass" | tail -n 5
```

## 10. 常用完整命令序列

```bash
cd /mnt/e/gitCode/drama-nemo-ass
source ~/.venvs/drama-nemo-ass/bin/activate

INPUT=video/02.mp4
OUT=outputs/02

python -m drama_nemo_ass prepare "$INPUT" --out "$OUT"
python -m drama_nemo_ass asr "$OUT" --model large-v3 --language ja
python -m drama_nemo_ass diarize "$OUT" --speakers 3
python -m drama_nemo_ass relabel "$OUT"
python -m drama_nemo_ass render "$OUT"
```

## 11. 哪些情况需要重跑哪些步骤

| 情况 | 需要执行 |
| --- | --- |
| 只修改了 `review.tsv` | `render --source review` |
| 修改 ASR 模型或语言 | 从 `asr` 开始重跑 |
| 修改说话人数 | 从 `diarize` 开始重跑 |
| 更换输入视频 | 从 `prepare` 开始完整重跑 |
| 只想重新生成 ASS 样式 | 只运行 `render` |

## 12. 主要产物索引

| 文件 | 作用 |
| --- | --- |
| `prepared.wav` | 标准化后的模型输入音频 |
| `asr_segments.json` | 本地 ASR 分段结果 |
| `asr_words.jsonl` | 本地 ASR 词级时间戳 |
| `nemo_diarization.json` | NeMo 说话人时间线 |
| `nemo_diarization.rttm` | 标准 RTTM 说话人结果 |
| `segments.json` | 本地文本与说话人合并结果 |
| `review.tsv` | 人工复核入口 |
| `output.ass` | 最终 ASS 字幕 |

## 13. 运行原则

1. 本地主线始终是基础产出。
2. 人工修改 `review.tsv` 后，用 `render --source review` 生成最终 ASS。
3. 修改 ASR、说话人数或输入视频时，按第 11 节从对应步骤重跑。
