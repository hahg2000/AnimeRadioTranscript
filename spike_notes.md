# Phase 0 Spike 结论（实测）

> 记录日期：2026-08-21。执行人：用户手动安装 `~/.venvs/drama-reazon` + `scripts/spike_smoke.py`。

## 关键事实

1. **ReazonSpeech v3 模型不存在**（`reazonspeech-espnet-v3` / `reazonspeech-k2-v3` 均为 `RepositoryNotFoundError`）。
   官方公开的最新 ASR 模型是 **v2 系列**：
   - `reazonspeech-k2-v2`（Zipformer transducer，ONNX，159M，Apache-2.0）
   - `reazonspeech-espnet-v2`（Conformer-Transducer，ESPnet）
   - `reazonspeech-nemo-v2`（NeMo，本项目正在移除 NeMo，不用）
2. **官方包没有 diarization 功能**。`reazonspeech.k2.asr` / `reazonspeech.espnet.asr` 都只做 ASR，
   没有 speaker diarization、也没有 BGM/SE 音事件检测。"v3 自带 diarization" 的前提不成立。
3. **k2-v2 实测可用且很快**：30 秒片段 CPU 推理 1.9s（RTF 0.063），文本正确
   （「そうキリオくんしりとり…コーナーでございます」）。
4. **k2-v2 官方基准优于其余日语 ASR**（CER）：JSUT 6.45 / CommonVoice 7.85 / TEDxJP-10K 9.09，
   对比 Whisper large-v3 7.18 / 8.18 / 9.96，espnet-v2 6.89 / 8.27 / 9.28。
5. **k2-v2 限制**：
   - 长音频（>30s）内存大，官方 transcribe 会警告（icefall#1680），需要分窗解码
   - subword 时间戳是单点（每个 token 一个 seconds），且头部有 0.9s padding 伪影（第一个 token 0.0）
   - 无标点输出（与 whisper 日语行为类似）
6. **espnet-asr 包未安装**（`No module named 'reazonspeech.espnet'`）。
   其依赖 `espnet` + `espnet_model_zoo`（git 安装），较重；且官方基准中 espnet-v2 精度略低于 k2-v2。
7. WSL 网络：huggingface.co 与 hf-mirror.com 均不通，但模型下载实际成功
   （k2-v2 已入 `/mnt/e/models/huggingface/hub`），说明当时 HF_ENDPOINT 生效或网络已恢复。

## 对计划的影响（PLAN_ReazonSpeech_v3.md 需修订）

| 原计划 | 修订 |
| --- | --- |
| ASR: espnet-v3 | **k2-v2 为主**（更快更准、无重依赖）；espnet-v2 可选作对照（重依赖，收益小） |
| 对齐: espnet-v3 char 级时间戳 | k2-v2 subword 单点时间戳 → 相邻 token 中点/下一边界插值出字符区间；分窗解码（20-25s + 0.9s pad） |
| diarization: 全换 ReazonSpeech v3 | **不可行**（官方无 diarization）。二选一：保留 Sortformer，或换 pyannote community-1 |
| 重叠分支: WeSep TSE | 不变（短 clip per-stem ASR 用 k2-v2 很合适） |

## espnet-v2 实测补充（2026-08-21）

安装修复后 espnet-v2 跑通（GPU，RTF 0.48）。与 k2-v2 对比：

| 维度 | k2-v2 | espnet-v2 |
| --- | --- | --- |
| RTF | 0.064（CPU） | 0.48（GPU，含模型加载） |
| 标点 | 无 | **有**（。？！、） |
| 分句 | 无（整段文本） | **有**（Segments 带 start/end，按标点+停顿切分） |
| 时间戳 | subword 单点（首个 token 有 padding 伪影：`そ@0.0` 实际语音在 10.7s） | **可信**：首段 0.34-1.53s 与人工参考 0.0-3.4s 吻合 |
| 文本 | 正确（无标点） | 正确且接近参考风格（带标点） |

**对字幕生成的启示**：espnet-v2 的输出天然接近 ASS 字幕形态（有标点+分句），
其 CTC 对齐的字符时间戳质量更高；k2-v2 只快 8 倍但无标点、时间戳需后处理修正。
Phase 2 adapter 两者都实现，`benchmark-asr` 以实测 CER/cpCER 决定主线；
若 espnet 文本质量优势明显，主线可用 espnet-v2、k2-v2 只用于重叠 stem 的快速 ASR。

## 安装修复记录

1. numpy 冲突：`espnet 202511` 要求 numpy>=2.0.0，但 `ctc_segmentation` wheel 是 numpy 1.x 编译的。
   修复：numpy 保持 2.2.6 + 从源码重编译 ctc_segmentation（删掉 sdist 里的预生成 .c，用 cython 重新生成）。
2. setuptools 冲突：espnet 要求 setuptools<74，pip 装包时被升到 84 → `pip install "setuptools<74"`。
3. espnet-v2 模型下载：WSL 直连 huggingface.co 不通，`export HF_ENDPOINT=https://hf-mirror.com` 后
   用 `espnet_model_zoo.downloader.ModelDownloader.download_and_unpack()` 预下载（镜像偶发断连，加重试循环）。

## benchmark-asr 实测（151s 样本，evaluation/02，2026-08-21）

| 引擎 | CER | 耗时 | 说明 |
| --- | --- | --- | --- |
| whisper large-v3 (GPU) | **25.5%** | 46.8s | 现状基线 |
| espnet-v2 (GPU) | 30.1% | 222.2s（含模型加载） | 有标点+分句，时间戳与参考吻合 |
| k2-v2 (CPU) | 47.4% | 12.4s | **窗口开头丢语音**（见下） |

k2 丢词原因：sherpa-onnx 流式解码在每个窗口开头存在 warm-up 丢失，
0-25s 窗口丢了前 ~9.9s 语音（「霧尾くんしりとりしながら帰ろうよ」整段），
前置 padding 无法修复。换 stride/warmup 丢弃策略只能缓解不能消除。

**结论**：按验收标准「k2 ≥ whisper」，k2 主线不成立。
建议：主线保留 whisper（文本 CER 最优）或换 espnet（字幕形态最优，标点+分句+可信时间戳）；
k2 只作为极快粗转写备选，或在 Phase 5 重叠 stem 短音频（<10s 语音）上试点。

## Phase 3 完成：espnet 词级 CTC 对齐（2026-08-21）

`asr_reazon.py` 的 `run_espnet_asr` 重写为窗口循环 + `get_timings()` 逐字符 CTC 对齐，
词时间戳不再段内均分。端到端指标（151s 样本，evaluation/02）：

| 指标 | whisper 基线 | espnet 段内均分 | espnet CTC 对齐 |
| --- | --- | --- | --- |
| CER | 25.5% | 30.1% | 30.1% |
| cpCER | 38.7% | 41.5% | **39.6%** |
| DER | 37.0% | 41.6% | **41.2%** |
| 字幕行数 | 66 | 94 | 96（参考 85） |

CTC 对齐带来 cpCER -1.9pp、DER -0.4pp 改善。
剩余 DER 差距主要来自 miss（espnet 语音覆盖 124s vs whisper 137s vs 参考 150s），
是 espnet 转录覆盖率问题而非时间戳问题，留待双引擎融合或 Phase 4 重叠分支解决。


## pyannote 零样本分离实验结论（2026-08-22）

93 个重叠 job（178.6s）经 pyannote/speech-separation-ami-1.0（PixIT，AMI 英语会议域）分离 +
espnet stem ASR，与基线（原单行 ASR）对比参考恢复率：

| 指标 | 基线（无分离） | pyannote stems |
| --- | --- | --- |
| 恢复的参考句子（共 213 个） | **129** | 59 |
| 分离后改善的 job | — | 6 |
| 分离后恶化的 job | — | 32 |

**结论：pyannote PixIT 零样本在动漫日语上为负收益**（恢复率 60.6% → 27.7%）。
stem 文本质量差（常见「はい」「ピッ」幻觉、句子截断），符合 AMI 域差大的预期。

依赖修复记录（drama-pyannote venv）：
- huggingface-hub 锁 <1.0（pyannote 3.3.2 用旧 API use_auth_token）
- transformers 锁 >=4.44,<4.50（4.57+ 要求 torch>=2.6；5.x 要求 hub>=1.5）
- speechbrain 锁 ==1.0.0（1.1.0 移除了 use_auth_token）
- 补 matplotlib

后续方向：零样本路线关闭。域内自训练 WeSep 是唯一能救重叠区的路；
否则接受重叠区 needs_review 回退。

## 双引擎融合实验结论（2026-08-23）

对 93 个重叠 job 分别用 espnet 和 whisper 转写 clip，融合（相似度去重，每 job 最多
2 条、每 speaker 1 条、>=3 字符、基线未覆盖才保留）后评估：

| 指标 | espnet 基线 | espnet+whisper 融合 |
| --- | --- | --- |
| 重叠区参考句子恢复（213） | 129 (60.6%) | **141 (66.2%)** |
| 单一引擎 clip 恢复 | espnet 63 / whisper 65 | 融合去重后 81 |
| OSD F1 | 0.001 | **0.103** |
| CER | 30.1% | 38.3%（融合行增加幻觉文本，CER 变差） |
| 字幕行数 | 96 | 119（参考 85） |

**结论**：融合恢复率 +12 句、OSD F1 显著提升，但 CER 因幻觉行升高 8pp。
已实现为可选开关：overlap --fuse（读 fusion_espnet/fusion_whisper 目录，写 segments_overlap.json）。
用法：先跑 scripts/exp_fusion_espnet.py（reazon venv）和 exp_fusion_whisper.py（nemo venv），再 overlap --fuse。

## espnet 时间轴偏晚问题与修复（2026-08-23）

马戏团 01（18 分钟）实测发现 espnet CTC 对齐时间戳系统性偏晚：
30 对同文本句子对比能量起音，espnet median +0.78s（max +2.2s），whisper median +0.40s。
用户人工对照视频确认 whisper 对得上、espnet 不对。

修复：新增 ealign 命令（drama_nemo_ass/realign.py）：
- 每个 espnet 段在 whisper 参考里找同文本锚点（归一化后子串匹配，±12s 窗口）
- 有锚点 → 采用 whisper 边界；无锚点 → 用相邻锚点偏移插值
- 词级时间戳在新边界内按字符数比例重分布
- 用法：python -m drama_nemo_ass realign outputs/XX --reference outputs/XX-whisper [--apply]
- 马戏团 01：165/374 段锚定，渲染后首条字幕 0.00s 起（修复前 0.96s）

流程变为：espnet 转写（文本+分句） → whisper 转写（时间基准） → realign → relabel → render。

## realign 升级为词级对齐（2026-08-23 晚）

段落级锚定+插值对 espnet 时间轴修复不彻底（偏差波动 0.4-2.2s，插值救不了）。
改为**词级字符对齐**：espnet 词序列全文与 whisper 词序列全文做 SequenceMatcher
字符映射，每个 espnet 词直接采用覆盖其字符的 whisper 词时间区间；再单调化
（相邻词不重叠、最小 50ms）。马戏团 01 实测 1900/1900 词全部映射到 whisper 时间，
渲染首行 0.00s 起（espnet 原始 0.96s、段落级 realign 0.00s 但中段仍有偏差）。

结论：espnet 文本 + whisper 词级时间的组合是当前最优字幕管线。

## hubert 引擎实测（2026-08-23）

接入 japanese-hubert-base-k2-rs35kh-bpe（98M，CTC，transformers，Apache-2.0）为新引擎：

- 帧率 50fps（20ms/帧），可一次跑 300s 长音频（RTF 0.004），CTC 帧级时间戳天然可靠
- 时间戳正确（首段 0.02s 起，与 whisper 0.00 一致），**不依赖 realign**
- 但文本质量差：马戏团 01 上 hubert vs whisper CER = **35.2%**（大量重复/漏字：
  「頑張るいさま」「ヒまわり」「金血の末」「振り替えエリー」），
  劣于 espnet（~30%）和 whisper（25.5%）
- 官方 benchmark 的低 CER（11%）是干净朗读集（JSUT/CommonVoice），广播剧多人自然对话退化严重

结论：hubert 时间戳好但文本不如 whisper，不宜作主线。whisper 仍是文本最优。

## kotoba-whisper-v2.2 接入结论（2026-08-23）

- 接入 --engine kotoba（官方 transformers 版，模型预下载到 /mnt/e/models/kotoba-whisper-v2.2）。
- **短音频质量优秀**：按 whisper 段边界逐段转写，kotoba 对 whisper 的 CER 仅 **9.88%**，
  显著优于原生 whisper 的日语表现。
- **长音频是硬伤**：transformers 4.57 的 whisper generate 对 >30s 长音频存在
  重复幻觉 bug（greedy/beam/long-form 参数均无法消除，会陷入「楽しかったね」循环）；
  pipeline 的 chunk_length_s 也会出现窗口内容串扰。
- 社区 ctranslate2 转换（faster-whisper 格式）均不可用：RoachLin segfault、
  Vinxscribe MemoryError(bad_alloc)、jctv 下载失败。
- 自己 VAD 分窗（librosa）在 BGM 下切点碎（1158 个 interval，median 0.45s），
  合并到 28s 后 CER 32%，仍劣于 whisper 25.5%。

结论：kotoba 文本质量确实是 whisper 系列里最好的，但**官方只有 transformers 格式，
而 transformers 长音频生成有 bug**，导致完整一集（18 分钟）无法可靠转写。
短片段/重剪片段场景可用；整集流水线暂不建议切 kotoba 主线。

## espnet 移除 + Qwen3-ASR 接入（2026-08-25）

按用户要求移除 espnet 相关代码（`run_espnet_asr`、`_load_espnet_model`、
`_espnet_transcribe_with_timings`、`_segment_words`/`_split_by_timings`/`_words_from_char_timings`、
`scripts/exp_fusion_espnet.py`、`scripts/exp_stem_asr_all.py`、`scripts/realign_espnet.py` 等），
并接入 Qwen3-ASR 作为新引擎。

- 模型 `Qwen/Qwen3-ASR-1.7B-hf`（transformers 原生版，52 语言，支持歌声+BGM，Apache-2.0）
- **环境隔离**：transformers 5.15.1 强制 `huggingface-hub>=1.5`，与 NeMo 的 `hub<1.0` 冲突，
  故建独立 venv `~/.venvs/drama-qwen`（torch 2.5.1 cu121 从 drama-nemo-ass 复制 + transformers 5.15.1 + hub 1.28）
- **模型下载**：WSL 内 hf-mirror.com 本轮持续超时，改由 Windows 侧直连 huggingface.co 下载到
  `/mnt/e/models/Qwen3-ASR-1.7B-hf`（9 文件，model.safetensors 3.9GB）
- **接入**：`asr_qwen.py` 用 `AutoModelForMultimodalLM` + `apply_transcription_request`，
  VAD 分窗（≤28s）逐段转写，`return_format="transcription_only"`，句子按句末标点切分，
  词时间段内按字符数均分
- **151s 样本（evaluation/02）实测**：CER **32.2%**（fold_kana 31.8%），31s 跑完（RTF ~0.2）
- 首段文本样例：`きりおくん尻取りしながら帰ろうよ。なんだそれ。…`（参考「霧尾くんしりとりしながら帰ろうよ」）

**结论**：Qwen3-ASR 接入成功、能跑、产出带标点分句，但 151s 样本 CER 32.2% 劣于 whisper 25.5%，
也不如 kotoba 的短音频 9.88%。「霧尾」→「きりお」、「しりとり」→「尻取り」等词形差异是 CER 虚高主因
（词形正确但用字/假名不同）。时间戳为段内均分，长音频精确时间轴仍需 `realign` 到 whisper。
是否优于现有引擎、是否切主线，需在更长/更多样本上进一步对比，并考虑 fold_kana 口径下的真实差距。

## k2/hubert 移除 + 动漫微调版实测（2026-08-25）

按用户要求删除 k2（`asr_reazon.py`）与 hubert（`asr_hubert.py`）两个引擎及对应测试，
engine 收敛为 whisper / kotoba / qwen 三个。共享的 `_tokenize_words`/`_merge_trailing_punctuation`
移入 `asr_qwen.py`。顺带修复 `--model` 参数键名不匹配 bug（`cmd_asr` 传 `model_name` 但
`run_asr_engine` 读 `model`，导致 `--model` 从未生效）。

下载 `jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf`（2B，日语动漫/galgame 微调，Apache-2.0）
到 `/mnt/e/models/`，并修复其缺失的 `chat_template.jinja`（该版模板缺 assistant 渲染块，
`apply_transcription_request` 的 `continue_final_message=True` 会报错；用官方 1.7B 模板覆盖后正常）。

马戏团 01（18 分钟）完整对照（CER vs whisper(orig)，raw / fold_kana）：

| 模型 | CER | fold_kana |
| --- | --- | --- |
| whisper(增强) | 20.3% | 19.8% |
| qwen 官方 1.7B(原始) | 30.0% | 29.4% |
| qwen 官方 1.7B(增强) | 25.8% | 25.2% |
| **anime 微调(原始)** | **34.2%** | 33.2% |
| **anime 微调(增强)** | **29.5%** | 28.5% |

**结论**：动漫微调版在完整广播剧上**反而劣于官方 1.7B**（34.2% vs 30.0%）。冒烟测试里它对
「霧尾くん」「しりとり」等专名更准，但全片 VAD 分窗后仍把「キムスター大サーカス」听成
「木下大相撲」、「喋っていきたい」错成「食べていきたい」，且微调语料偏 galgame/动漫 NSFW 域，
在带 BGM 的多人对话上更不稳定。微调版无增益。

## 云服务转录评测（2026-08-25）

用户提供腾讯云「通用语音识别」与「通译听悟」两家对马戏团 01 增强音频的转录结果，
与 whisper 参考对照（两家都用增强音频转，故以 whisper(enh) 为公平参考）：

| 服务 | vs whisper(orig) | vs whisper(enh) | fold_kana |
| --- | --- | --- | --- |
| 腾讯云 通用语音识别 | 33.6% | 30.8% | 29.9% |
| 通译听悟 | 37.0% | 32.9% | 31.8% |

**结论**：两者均明显劣于 whisper。通用云 ASR 针对会议/通话域，共性短板：

1. **填词噪声**：把「うん」「ね」「そうだね」等口语 filler 当真实文本堆进字幕
   （`からうんうんうんとね`、`サーカスのねうんうんうんこと`）。
2. **专名崩坏**：「キムスター大サーカス」→ 腾讯云「キムスタを大サーカス」/ 通译听悟「木下大騒がせ」。
3. **断句混乱**：腾讯云 45s 压一个块无标点；通译听悟按呼吸切得过碎。
4. **错字**：腾讯云「金月の末」（金欠の末）、「鶴巻水」。

腾讯云（30.8%）略优于通译听悟（32.9%），但都不如 whisper。三家外部方案（云 ×2 + Qwen）都卡在
**专有名词**上，而 whisper 对该剧专名反而更稳。

## 火山引擎录音文件识别评测（2026-08-25）

接入豆包大模型录音文件识别（`volc.seedasr.auc`，异步 submit + query 两阶段），
独立脚本 `scripts/volcengine_transcribe.py`（未进 engine，云服务生命周期与本地引擎不符）。

- 接口走增强音频 URL，故以 whisper(enh) 为公平参考：**CER 28.4%**（raw）/ 27.8%（fold_kana）。
- vs whisper(orig)：34.0%。

| 服务/引擎 | vs whisper(orig) | vs whisper(enh) |
| --- | --- | --- |
| whisper(enh) 基线 | 20.3% | — |
| qwen 官方 1.7B(enh) | 25.8% | — |
| anime 微调(enh) | 29.5% | — |
| 腾讯云 | 33.6% | 30.8% |
| 火山引擎 | 34.0% | **28.4%** |
| 通译听悟 | 37.0% | 32.9% |

**结论**：火山（28.4%）与腾讯云（30.8%）基本同档，仍明显劣于 whisper。`enable_ddc` 语义顺滑
未起作用，填词噪声仍在（`反省したから、うん、うん、ちょっと、ねこの生配信では…`），
并出现新的**长串重复幻觉**（`ひからひから…` 连写 30+ 次），专名同一个人名拼出 4 种写法
（みずか / 水香 / ミズカ / 瑞季）。唯一亮点是逐字符 `words[]` 毫秒级时间戳。

**三家云服务 + Qwen 系列全部落入 28-37% 区间，无一家达到 whisper 的 20%。**
通用云 ASR 的域不适配（填词、专名、重复幻觉）在所有外部方案中普遍存在，whisper 仍是广播剧
句子正确性最优解。

### 接入备忘（火山录音文件识别）

- 鉴权：新版控制台 `X-Api-Key` + `X-Api-Resource-Id: volc.seedasr.auc` + `X-Api-Request-Id`（UUID）。
- 任务状态在 **response header 的 `X-Api-Status-Code`**（20000000=成功 / 20000001=处理中 /
  20000002=排队），不在 body，轮询判断必须读 header。
- 音频必须国内可达 URL；`raw.githubusercontent.com` 火山服务器 fetch 失败（`45000006 audio download failed`），
  需改用 TOS/OSS/COS 等国内对象存储。
- `enable_speaker_info` 仅在不指定 language 或 `zh-CN` 时生效，日语无效。

## 主线切换（2026-08-25）

综合评测结论，**ASR 主线切回 whisper**（`asr`/`run` 的 `--engine` 默认值由 kotoba 改为 whisper）。
whisper large-v3 在广播剧场景句子正确性最优，且时间戳经过用户人工对照视频验证可信。

## 缺句问题修复：VAD 窗口 + 逐窗重跑（2026-09-13）

### 起因

03 集（44.5 分钟）用户反馈 11:54~12:24 大量漏句。核查 `asr_segments.json` 发现
whisper 在 **714.56→744.64 有整 30 秒空洞**（另有 2511.14→2541.14 的 30 秒空洞、
1815.92→1828.26 的 12 秒空洞）。根因不是音频不可识别，而是：

1. **whisper 独自定义时间轴**：三引擎对照以 whisper 段边界切片，whisper 跳过的区间
   在旧 `comparison.tsv` 里连一行都不存在，kotoba/qwen 也没有机会转写（"Whisper 单点
   时间轴绑架"）。
2. **whisper 长音频解码漏切**：把 28 个漏/半漏窗口单独切出来、用相同参数重跑 whisper，
   **25/28 完全找回**（含 30 秒空洞内的「メガネ…睦美様の登場シーン…」）。剩余 3 个失败
   全是 0.6~1.2 秒的极短语气词（「フェッ」「ごめん」「はい」）。证明模型能力没问题，
   是长音频连续解码（BGM、低能量段、no-speech 抑制）把整段漏掉了。

### 改动

| 文件 | 改动 |
| --- | --- |
| `scripts/vad_segments.py` | 新增。Silero VAD 检测发声窗口 → `vad_windows.json`；merge/pad/28s cap/最短过滤，`--energy-union` 可并合能量检测补抓笑声 |
| `scripts/transcribe_segments.py` | `--boundaries vad|whisper`（默认 vad）；新增 `--engine whisper` 逐窗重跑（faster-whisper 同参）；VAD 模式写 `*_vad.json` |
| `scripts/merge_comparison.py` | VAD 模式以窗口为行；whisper 列优先用 `asr_segments_whisper_vad.json`，缺失回退长音频重叠映射；旧 whisper 边界模式保留 |
| 环境 | `silero-vad 6.2.1` 装入 `~/.venvs/drama-nemo-ass`（torch 未动） |
| `WORKFLOW.md` | 第 4 节改为 VAD 五步流程 + 旧模式回归命令 |

### 结果（03 集，163 个 VAD 窗口）

| 指标 | 修复前 | 修复后 |
| --- | --- | --- |
| whisper 完全空窗口 | 22（47.7s） | **2** |
| whisper 部分漏窗口 | 6（69.7s） | **2** |
| 30 秒空洞 714-744s | 整段缺失 | whisper/kotoba/qwen 均有内容 |

- 逐窗 whisper 找回 20 个长音频漏掉的窗口。
- VAD 窗口独立于 whisper，kotoba/qwen 对 22 个漏句窗口全部有内容（两独立模型互证，非幻觉）。
- 残留风险：<1 秒的极短窗口 whisper 可能空或乱猜（如 0.6s 的「はい」），仍需三方对照。

结论：whisper 仍是句子质量基线，但**时间轴必须交给独立 VAD**，且 **whisper 应按 VAD 窗口
逐窗重跑**，否则长音频漏句会静默丢失。
