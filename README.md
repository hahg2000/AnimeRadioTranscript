# drama-nemo-ass

完整的 WSL 安装、逐步运行、失败重试和产物说明见 [RUNBOOK.md](RUNBOOK.md)。

按 `PLAN_NeMo.md` 搭出的独立项目：用 faster-whisper 做日语 ASR，用 NVIDIA NeMo Sortformer 做说话人分离，最后输出可被 Aegisub/mpv 打开的 ASS。

重型依赖都延迟导入。没有安装 NeMo/faster-whisper 时，纯 Python 的数据转换、RTTM 导入、词到说话人归属、TSV 复核、ASS 渲染和 ASS 对比仍然可以测试或使用。

## 阶段总结（2026-08，ReazonSpeech 重构轮）

> 完整过程记录见 [PLAN_ReazonSpeech_v3.md](PLAN_ReazonSpeech_v3.md)（计划）、
> [spike_notes.md](spike_notes.md)（实测数据）、[INSTALL_ReazonSpeech.md](INSTALL_ReazonSpeech.md)（环境安装）。

### 已落地

| 改进 | 内容 | 结果 |
| --- | --- | --- |
| ASR 主线换代 | `reazonspeech-espnet-v2` 成为默认引擎（`asr`/`run --engine espnet`） | 自带标点+分句，字幕行数 96 vs whisper 66（参考 85） |
| 词级 CTC 对齐 | espnet `get_timings` 逐字符对齐 → SudachiPy 并词，替换段内均分 | cpCER 41.5%→39.6%，DER 41.6%→41.2% |
| 多引擎并存 | `--engine espnet|whisper|k2` + `benchmark-asr` 统一对比 | whisper 25.5% / espnet 30.1% / k2 47.4% |
| 重叠分支框架 | `overlap` 命令：区域检测+上下文切片 → 可插拔分离器 → 质量闸门 → 双行 ASS（lane 垂直排布） | 无分离器时全部 `needs_review` 回退，不硬编 |
| 双引擎融合 | `overlap --fuse`：espnet+whisper 各转写 clip，去重后把基线未覆盖的句子作为第二行 | 重叠区句子恢复 129→**141**，OSD F1 0.001→**0.103** |
| 多集回归 | `evaluation/01`（忍杀 01 集，428s，2 人无重叠）加入评测基线 | CER 21.6%，验证主线泛化 |

### 已关闭的路线

- **ReazonSpeech v3**：HF 上不存在（404），官方最新是 v2 系列。
- **WeSep 分离**：官方没有发布预训练 TSE checkpoint。
- **pyannote PixIT 零样本**：AMI 英语会议域，动漫日语上恢复率 60.6%→27.7%，负收益。

### 当前指标（151s 样本，evaluation/02）

| 指标 | whisper 基线 | espnet 主线 | espnet+融合 |
| --- | --- | --- | --- |
| CER | 25.5% | 30.1% | 38.3% |
| cpCER | 38.7% | 39.6% | 50.0% |
| DER | 37.0% | 41.2% | 43.2% |
| OSD F1 | 0.000 | 0.001 | **0.103** |

「句子正确优先」用 espnet 主线（默认）；「尽可能转录全」再加 `overlap --fuse`。

### 剩余方向

1. 域内自训练 TSE/分离模型（唯一能真正恢复第二人台词的路，计划见 [PLAN_TSE_Training.md](PLAN_TSE_Training.md)）
2. 融合行幻觉抑制（提高融合候选门槛，压低 CER 代价）
3. 更激进的 ASR 候选（Qwen3-ASR 已接入 `--engine qwen`，151s 样本 CER 32.2%，待更长样本对比）

## Upgrade Plan: 动漫广播剧密集交叉说话

> 计划更新时间：2026-07-13。当前主线仍可使用；本节描述下一阶段的评测基线、目标架构、模型候选和分阶段实施顺序。

### 当前实施状态

本轮已把计划中的 Phase 0、Phase 1 核心改造和本地模型适配落地：

- 新增 `init-eval` / `evaluate`：从人工 ASS 生成 manifest、重叠 RTTM 和参考片段，并报告 NFKC 去标点 CER、假名折叠 CER、cpCER、重叠感知 DER/JER、OSD precision/recall/F1。
- `prepare` 现在同时保留 `prepared_source.wav`（源采样率/声道）和 `prepared_mono_16k.wav`；`prepared.wav` 继续作为兼容入口。
- Sortformer 会请求原始 `T x S` speaker activity probability，并新增 `speaker_activity.jsonl`、`exclusive_turns.json`、`overlap_regions.json`。
- 词归属已从中心点/前一 speaker 偏好改为词区间覆盖率或活动概率积分；多人活动区写入 `overlap|needs_overlap_separation`。
- 短 speaker island 平滑默认关闭，需要时显式传 `--smooth-short-islands`；重叠词不会被平滑。
- `--models-root /mnt/e/models` 与 `DRAMA_MODELS_ROOT` 可自动解析本地 Whisper 和 Sortformer，避免重复下载。

尚未完成的是 Phase 2 的 Qwen/Kotoba adapter 与强制对齐、Phase 3 的分类器校准，以及 Phase 4 真正的 TSE/盲源分离。因此当前版本能正确发现并保留交叉说话风险，但不会假装已经恢复混音中第二个人缺失的文本。

### 结论

当前质量上限主要不在 `Whisper large-v3`，而在于流水线把重叠语音处理成“从多个候选说话人中选一个”。单说话人 ASR 无法从单声道混合中恢复第二个人同时说出的完整文本，因此下一版必须把普通对白与真正重叠对白拆成两条路径：

1. 非重叠区继续执行 ASR、强制对齐和角色识别。
2. 重叠区先做 Overlapped Speech Detection（OSD），再按角色做目标说话人提取或语音分离，最后对每个输出声道独立 ASR。
3. 分离或角色判断置信度不足时保留 `needs_review`，不自动伪造确定结果。

### 当前样本基线

`video/02.mp4` 与 `video/sample.ass` 的现有结果可作为第一份回归样本：

| 指标 | 人工参考 | 当前结果/检测 |
| --- | ---: | ---: |
| 时长 | 151.05 秒 | 150.56 秒 |
| 字幕事件 | 85 | 66 |
| 字符数 | 1130 | 895 |
| 当前粗略 CER | - | 30.0% |
| 多人同时说话时长 | 27.14 秒（18.0%） | 50.73 秒（33.7%） |
| 最大同时说话人数 | 3 | 3 |
| 标记为 `overlap` 的结果段 | - | 48 / 66 |

上述 CER 只是方向性指标。现有 `compare.py` 会按开始时间拼接重叠事件，而同时发生的两句台词没有唯一文本顺序；后续需要加入 permutation-invariant 的重叠文本指标。

现有实现还存在以下限制：

- `prepare_audio` 立即把原始 44.1 kHz 立体声压成 16 kHz 单声道。样本的 Side 能量比 Mid 低约 20.5 dB，空间信息有限，但仍应保留原始立体声供声像和分离实验。
- 一个词同时命中多个 speaker turn 时，当前代码优先延续前一个说话人，第二个说话人的文字不会被恢复。
- `0.65s / 3 字` 的 A-B-A speaker island 平滑会误删广播剧常见的「え？」「うん」「そう」等短插话。
- 多余 speaker 当前按时间最近关系折叠，而不是按全局最优映射折叠。
- 当前 RTTM 产物缺少 Sortformer 原始帧级活动概率，重叠决策无法使用置信度。

### 目标架构

```text
原始音频（保留 stereo/master）
  |
  +-- 可选 BGM/人声预处理（必须通过 A/B 评测，默认关闭）
  |
  +-- VAD + SCD + OSD + diarization frame probabilities
        |
        +-- 非重叠区
        |     +-- ASR 候选
        |     +-- 强制对齐
        |     +-- 单行 ASS
        |
        +-- 重叠区（前后保留 0.5-1.0s 上下文）
              +-- 目标说话人提取 TSE / 盲源分离
              +-- 每个角色声道独立 ASR + 对齐
              +-- 分离质量校验
              +-- 多行同屏 ASS 或 needs_review
```

普通 ASR 对齐可以使用 exclusive diarization 时间线，但重叠分支必须保留非 exclusive 的多人活动结果，不能把它强制压成单 speaker。

### 候选模型与用途

| 环节 | 优先候选 | 用途与限制 |
| --- | --- | --- |
| 日语 ASR | [`Qwen3-ASR-1.7B`](https://github.com/QwenLM/Qwen3-ASR) | 2026 年发布，支持日语、长音频、歌声和带 BGM 歌曲。作为新的精度候选，不假定它能直接恢复两个人的重叠文字。建议在独立环境中运行，避免与当前 NeMo 依赖冲突。 |
| 时间戳 | `Qwen3-ForcedAligner-0.6B` | 支持日语词/字级强制对齐；先确定文本，再重建稳定时间戳。长音频按不超过 5 分钟的窗口处理。 |
| 日语 ASR 基线 | [`kotoba-whisper-v2.2`](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2) | 日语对话域候选。只使用其 ASR/标点能力；其集成 diarization 基于较旧的 pyannote 3.1，不作为新主线。 |
| 现有 ASR 基线 | `faster-whisper large-v3` | 保留为速度、显存和质量对照组。不要在完成同一测试集 A/B 前直接移除。 |
| 本地 diarization | [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) | 支持固定 speaker 数、重叠时间线和 exclusive 时间线；作为 Sortformer 的独立对照。 |
| NeMo diarization | [`nvidia/diar_streaming_sortformer_4spk-v2.1`](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2.1) | 相比当前 offline v1 值得测试，支持 0.08 秒帧级多人活动概率；官方说明训练数据以英语为主，日语动漫域可能退化，因此必须使用本项目标注评测。 |
| 目标说话人提取 | [`WeSep`](https://github.com/wenet-e2e/wesep) | 角色固定时优先；每次按指定角色从混合音频中提取目标语音，可避免盲源分离的声道置换问题。 |
| 分离/增强实验 | [`ClearerVoice-Studio`](https://github.com/modelscope/ClearerVoice-Studio) | 提供 MossFormer、语音增强、分离与目标说话人提取训练方案。预训练域不等于动漫广播剧，必须检查伪影和 ASR 增益。 |
| 快速联合基线 | [`pyannote/speech-separation-ami-1.0`](https://huggingface.co/pyannote/speech-separation-ami-1.0) | PixIT 可同时输出 diarization 和分离源；模型训练于英语 AMI 会议数据，只作为研究基线。 |

### `E:\models` 实测可用性

以下结论来自 WSL venv 与 RTX 4080 SUPER 的真实加载，而不是仅检查文件名：

| 本地模型 | 解析路径（WSL） | 状态 | 当前用途 |
| --- | --- | --- | --- |
| faster-whisper large-v3 | `/mnt/e/models/faster-whisper-large-v3` | 已成功加载 | 当前日语 ASR 基线，可直接用 |
| NeMo Sortformer 4spk v1 | `/mnt/e/models/huggingface/hub/models--nvidia--diar_sortformer_4spk-v1/.../diar_sortformer_4spk-v1.nemo` | 已成功 `restore_from`；151 秒样本得到 114 turns / 1890 activity frames | diarization、OSD 与帧概率输出 |

Windows 中 Sortformer cache 文件可能显示为 0 字节 reparse point，这是 Hugging Face cache 的 Linux 符号链接；本项目重模型本来就在 WSL 运行，WSL 中链接和 blob 已验证有效。

检查解析结果：

```bash
python -m drama_nemo_ass inspect-models --models-root /mnt/e/models
# 或永久设置
export DRAMA_MODELS_ROOT=/mnt/e/models
```

### 分阶段实施

#### Phase 0：建立可信评测

- 从人工 ASS 的 `Name`/`Style` 导出角色时间线、重叠 RTTM 和规范化文本。
- 固定 train/dev/test 文件清单；同一集不能同时用于阈值调整和最终报告。
- 日语文本至少报告 NFKC 规范化、标点忽略和假名/汉字保留两套 CER。
- 增加 `DER`、`JER`、OSD precision/recall/F1、speaker-attributed CER、cpCER、漏字率、角色混淆矩阵、复核段比例、RTF 和峰值显存。
- 保留当前 151 秒样本做快速回归，但正式选型至少覆盖多集、不同 BGM、不同情绪和不同重叠强度。

建议新增产物：

```text
evaluation/
  manifest.json
  reference.rttm
  reference_segments.json
  asr_benchmark.json
  diarization_benchmark.json
  overlap_benchmark.json
  report.md
```

验收门槛：任何模型晋级主线前，必须在固定 test 上同时报告文本、speaker、overlap 和资源指标，不能只根据单一 CER 或主观试听替换。

#### Phase 1：修正当前融合逻辑

- `prepare` 同时输出原始声道副本与 `prepared_mono_16k.wav`，不要永久丢弃 stereo 信息。
- 保存 diarization 的 `T x S` 帧级活动概率，而不只保存 RTTM hard turns。
- 词归属从“中心点命中”升级为“词时间区间与 speaker frame probability 的积分/覆盖率”。
- 命中多人时不再默认延续前一个 speaker；将该区间送入 overlap 分支。
- 默认关闭短 speaker island 平滑；只有 OSD 判定非重叠且短岛置信度低时才允许平滑。
- 多余 speaker 使用 profile embedding、全局 Hungarian 匹配或聚类中心合并，禁止仅按时间最近关系折叠。
- 分开保存 `speaker_activity.jsonl`、`exclusive_turns.json` 和 `overlap_regions.json`。

#### Phase 2：ASR 与时间戳 A/B

- 为 `large-v3`、`kotoba-whisper-v2.2`、`Qwen3-ASR-1.7B` 建立统一 adapter，输出同一 `Word`/`AsrSegment` 数据结构。
- 同一音频、同一切片和同一文本规范化规则下比较三个模型。
- 优先用 Qwen ForcedAligner 对最终候选文本重新对齐；保留 Whisper 原始 word timestamps 作为对照。
- 增加角色名、作品名、招式、口癖等 glossary；区分 ASR prompt、受约束后处理和人工修订，禁止 LLM 无记录地改写原文。
- 普通区间采用带上下文切片：识别时保留相邻上下文，只接收中心有效区的 token，减少快速换人边界漏字。

建议新增 CLI：

```bash
python -m drama_nemo_ass benchmark-asr evaluation/manifest.json --models large-v3,kotoba-v2.2,qwen3-asr-1.7b
python -m drama_nemo_ass align outputs/02 --model qwen3-forced-aligner-0.6b
```

#### Phase 3：真正的重叠语音分支

- 只对 OSD 区间运行分离/TSE，前后增加 0.5-1.0 秒上下文，避免对整集分离造成大量伪影。
- 固定已知角色场景优先使用 TSE：同一混合片段分别以每个角色提取，再独立 ASR。
- 未知角色时，使用二源/三源盲分离作为回退。
- 每个分离结果检查目标角色相似度、非目标泄漏、有效语音比例、ASR 置信度和文本重复率。
- 两个声道输出高度重复、目标角色相似度不足或分离后 ASR 明显恶化时，不自动合入正文，写入 `needs_review`。
- 成功时输出多条时间互相重叠的 ASS Dialogue，使用固定角色颜色和垂直 lane。

建议新增 CLI 与产物：

```bash
python -m drama_nemo_ass detect-overlap outputs/02
python -m drama_nemo_ass separate-overlap outputs/02 --backend wesep
python -m drama_nemo_ass transcribe-stems outputs/02 --model qwen3-asr-1.7b
python -m drama_nemo_ass render outputs/02 --layout multilane
```

```text
overlap_regions.json
overlap_stems/<region>/<character>.wav
overlap_transcripts.jsonl
overlap_quality.jsonl
```

#### Phase 4：集成与回归

- 所有重模型保留延迟导入，并为 Qwen、pyannote、NeMo、分离模型建立互相隔离的可选依赖或独立 worker 环境。
- 单元测试覆盖帧概率聚合、重叠区切片、speaker permutation、短插话保留、双 lane ASS 和失败回退。
- 集成测试覆盖无重叠、快速轮流、双人同时、三人同时、BGM、笑声和未知角色。
- 每次模型或阈值变更自动生成 benchmark diff；质量提升但显存/RTF 不可接受时保留为高质量离线模式。

### Agent Skills

当前 Codex 环境中已经存在、可直接用于本项目的技能：

| Skill | 用途 | 建议 |
| --- | --- | --- |
| `hugging-face:hf-cli` | 下载和管理 Qwen、Kotoba、pyannote、NeMo 模型仓库 | 直接使用，适合模型缓存和版本固定。 |
| `hugging-face:huggingface-datasets` | 查询 Hugging Face Dataset Viewer、split、样本和 Parquet | 用于检查日语 ASR/说话人数据集元数据；大规模训练数据仍需单独的数据许可审查。 |
| `hugging-face:huggingface-jobs` | 在 Hugging Face Jobs 上运行通用 GPU 任务 | 本地 RTX 4080 不够或需要并行 benchmark 时再用，涉及云端费用。 |
| `skill-creator` | 为本项目创建可重复执行的专用 skill | 当 benchmark、模型下载和报告流程稳定后，可创建 `anime-drama-asr-eval` skill 固化工作流。 |

通过 `find-skills` 检索并验证后，以下外部技能有实际参考价值；本轮仅安装了与当前 Phase 0/6 直接相关的一项：

| Skill | 质量信号 | 适用范围 | 安装命令 |
| --- | --- | --- | --- |
| [`openai/skills@transcribe`](https://skills.sh/openai/skills/transcribe) | OpenAI 官方；约 2.2K installs；仓库约 23.4K stars | 用 `gpt-4o-transcribe-diarize` 建立额外云端转写/diarization 基线；需要 API key，不能替代本地 overlap separation。 | `npx skills add https://github.com/openai/skills --skill transcribe` |
| [`elevenlabs/skills@speech-to-text`](https://skills.sh/elevenlabs/skills/speech-to-text) | ElevenLabs 官方；约 5.5K installs；安全审计含 Snyk warning | 用 Scribe v2 做 90+ 语言、词级时间戳和 diarization 对照；安装前检查 warning 和数据上传要求。 | `npx skills add https://github.com/elevenlabs/skills --skill speech-to-text` |
| [`wshobson/agents@python-testing-patterns`](https://skills.sh/wshobson/agents/python-testing-patterns) | 约 26.7K installs、37K+ stars，三项安全审计通过 | 适合 Phase 0/6 的 pytest、fixture、参数化、mock 和集成测试建设；**已通过 Codex skill-installer 安装**。 | 已安装到 `$CODEX_HOME/skills/python-testing-patterns` |

暂不建议安装：

- `theplasmak/faster-whisper@faster-whisper` 虽有约 1.5K installs，但来源仓库 stars 很少，且 skills.sh 显示多项安全审计失败；项目已有 faster-whisper 实现，不需要承担额外风险。
- 以 `speech separation`、`speaker diarization` 和 `ffmpeg audio processing` 检索到的直接候选安装量普遍只有几十到一百左右，没有发现能覆盖日语动漫重叠语音并且质量信号充分的成熟 skill。
- 通用 Whisper/转写 skill 主要封装调用流程，不会解决目标说话人提取、角色域适配、重叠评测和多 lane ASS，这些仍属于本项目核心实现。

其余外部 skill 在安装前仍应先阅读 `SKILL.md`、脚本、依赖、网络请求和安全审计。云端转写 skill 不解决本项目最关键的 overlap separation，因此本轮没有安装。

## Quick Start

```bash
python -m drama_nemo_ass --help
python -m drama_nemo_ass inspect-models --models-root /mnt/e/models
python -m drama_nemo_ass prepare video/02.mp4 --out outputs/02
python -m drama_nemo_ass asr outputs/02 --engine whisper
python -m drama_nemo_ass diarize outputs/02 --speakers 3 --models-root /mnt/e/models
python -m drama_nemo_ass relabel outputs/02
python -m drama_nemo_ass render outputs/02
python -m drama_nemo_ass init-eval video/sample.ass --audio video/02.mp4 --out evaluation/02
python -m drama_nemo_ass evaluate video/sample.ass outputs/02/output.ass --out evaluation/02/current_metrics.json
```

本轮生成的 [evaluation/02/current_metrics.json](evaluation/02/current_metrics.json) 显示：规范化 CER `25.50%`、cpCER `38.73%`、DER `37.03%`、JER `47.66%`，OSD recall/F1 都为 `0`。这里 OSD 为 0 是因为当前最终 ASS 几乎没有同时存在的 Dialogue；它准确暴露出旧版渲染链把检测到的重叠风险压回单行，而不是代表参考字幕没有重叠。

如果已经有外部工具生成的 RTTM，可以先跳过 NeMo：

```bash
python -m drama_nemo_ass diarize outputs/02 --rttm external.rttm --speakers 3
```

多引擎 ASR（环境安装见 [INSTALL_ReazonSpeech.md](INSTALL_ReazonSpeech.md) 与 [spike_notes.md](spike_notes.md)）：

```bash
python -m drama_nemo_ass asr outputs/02 --engine whisper  # 默认：句子正确性最优，CER 25.5%
python -m drama_nemo_ass asr outputs/02 --engine kotoba   # 短音频最优，长音频有重复 bug
python -m drama_nemo_ass asr outputs/02 --engine qwen     # Qwen3-ASR（独立 venv：~/.venvs/drama-qwen）
python -m drama_nemo_ass benchmark-asr outputs/02 evaluation/02 --engines whisper,kotoba,qwen
```

默认引擎 whisper：`asr`/`run` 的 `--engine` 默认值均为 `whisper`（广播剧句子正确性最优，时间戳经人工对照验证）。
kotoba/qwen 输出带标点的句子边界会被 `relabel` 保留为字幕行边界，仅按说话人切换进一步切分。

**时间轴注意**：qwen/kotoba 逐段转写的时间戳为段内按字符数均分，长音频上可能偏晚。
如需精确时间轴，先 whisper 转写到 `outputs/XX-whisper` 作为时间基准，再执行：

```bash
python -m drama_nemo_ass realign outputs/XX --reference outputs/XX-whisper --apply
python -m drama_nemo_ass relabel outputs/XX
```

`realign` 把当前 ASR 词序列与 whisper 词序列做字符级对齐，每个词直接采用
whisper 词时间（当前 ASR 文本 + whisper 时间）。

完整流水线：

```bash
python -m drama_nemo_ass run video/02.mp4 --out outputs/02 --language ja --speakers 3
```

## WSL2 Environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-wsl.txt
```

NeMo 和 faster-whisper 默认按计划走 WSL2 + NVIDIA GPU。Windows 侧适合保存视频、查看 ASS、编辑 `review.tsv`。

如果 NeMo/torchvision 报 `RuntimeError: operator torchvision::nms does not exist`，说明 `torch`、`torchvision`、`torchaudio` 版本或 CPU/CUDA wheel 混装了。重新安装同一组 CUDA wheel：

```bash
source ~/.venvs/drama-nemo-ass/bin/activate
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import torchvision; print(torchvision.__version__)"
```

## Hugging Face Model Downloads

第一次运行 `asr` 或 `run` 会从 Hugging Face 下载模型。如果 WSL 不能直接访问 `huggingface.co`，`faster-whisper` 可能报 `LocalEntryNotFoundError`。可以先设置缓存和镜像，再预下载模型：

```bash
cd /mnt/e/projects/drama-nemo-ass
source ~/.venvs/drama-nemo-ass/bin/activate

export HF_HOME=/mnt/e/models/huggingface
export HF_HUB_CACHE=/mnt/e/models/huggingface/hub
export HUGGINGFACE_HUB_CACHE=/mnt/e/models/huggingface/hub
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1

python -m pip install --force-reinstall "huggingface-hub>=0.34.0,<1.0"
hf download Systran/faster-whisper-large-v3 --local-dir /mnt/e/models/faster-whisper-large-v3
```

然后 ASR 使用本地模型目录：

```bash
python -m drama_nemo_ass asr outputs/02 \
  --model /mnt/e/models/faster-whisper-large-v3 \
  --language ja
```

如果网络可以直接访问 Hugging Face，把 `HF_ENDPOINT` 那行去掉即可。NeMo 的 Sortformer 下载也会受这些 Hugging Face 环境变量影响。

如果浏览器或 `urllib` 能访问镜像，但 `hf download` 仍报 `Network is unreachable`，先在同一个 WSL shell 里确认环境变量，并禁用 Xet 下载后端：

```bash
echo "$HF_ENDPOINT"
echo "$HF_HOME"
echo "$HF_HUB_CACHE"
export HF_HUB_DISABLE_XET=1
python -m huggingface_hub.commands.huggingface_cli env
```

也可以不用 `hf` 命令，直接走 Python API 下载同一个模型：

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-large-v3', local_dir='/mnt/e/models/faster-whisper-large-v3')"
```

如果 `python3.10 -m venv .venv` 在 WSL 里报 `ensurepip` 相关错误，通常是 Ubuntu 的 venv 包没有装完整，或虚拟环境建在 `/mnt/e` 这类 Windows 挂载盘时触发了权限/链接问题。先重装基础包并删除失败残留：

```bash
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip python3.10-dev
rm -rf .venv
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

不要用系统 Python 直接运行 `python3.10 -m ensurepip` 作为诊断；Ubuntu/Debian 会故意禁用系统级 ensurepip。如果在 `/mnt/e` 下仍然失败，建议把 venv 建到 WSL Linux 文件系统里，再从项目目录引用它：

```bash
mkdir -p ~/.venvs
python3.10 -m venv ~/.venvs/drama-nemo-ass
source ~/.venvs/drama-nemo-ass/bin/activate
python -m pip install -U pip setuptools wheel
```

## Dependency Compatibility Notes

当前 NeMo/Transformers 组合需要 `huggingface-hub>=0.34.0,<1.0`。不要直接运行 `python -m pip install -U huggingface_hub`，否则可能升级到 1.x 并触发：

```text
ImportError: huggingface-hub>=0.34.0,<1.0 is required ... found huggingface-hub==1.x
```

修复：

```bash
source ~/.venvs/drama-nemo-ass/bin/activate
python -m pip install --force-reinstall "huggingface-hub>=0.34.0,<1.0"
python -m pip check
```
