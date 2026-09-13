# PLAN: ReazonSpeech v2 重构转录管线

> 修订于 Phase 0 spike 之后（2026-08-21）。前提变化：
> ReazonSpeech **v3 模型不存在**（HF 404），官方最新为 **v2 系列**；
> 官方包**无 diarization**。实测结论见 `spike_notes.md`。

## 0. 目标与决策摘要

| 改进点 | 方案 |
| --- | --- |
| 1. 词级时间戳强制对齐 | k2-v2 subword 单点时间戳 → 边界插值出字符区间 + SudachiPy 并词；分窗解码（≤25s + pad） |
| 2. ASR 模型换代 | **`reazonspeech-k2-v2` 为主**（Zipformer ONNX，RTF 0.063，官方基准 CER 优于 whisper large-v3 与 espnet-v2）；**espnet-v2 作 A/B 对照** |
| 3. 重叠说话恢复 | 保留 Sortformer diarization/OSD → WeSep TSE 分离（独立 venv）→ 每 stem 用 k2-v2 独立 ASR → 双行 ASS |
| Diarization | **保留 NeMo Sortformer**（官方无 diarization 能力；帧级 activity 概率继续用于词归属） |

架构变化：

```text
现在:  prepare → whisper ASR → Sortformer diarize → relabel → render
之后:  prepare → k2-v2 ASR(+对齐) → Sortformer diarize
            → overlap 分支: WeSep TSE → per-stem k2-v2 ASR → 双行合并
            → relabel → render(multi-lane)
```

## Phase 0：API spike ✅ 已完成

见 `spike_notes.md`。关键结论：

- k2-v2 实测 RTF 0.063（CPU）、文本正确；官方 CER：JSUT 6.45 / CV 7.85 / TEDxJP 9.09
- k2 subword 为单点时间戳，头部有 0.9s padding 伪影，>30s 长音频需分窗
- espnet-v2 精度略低于 k2-v2，仅作 A/B 对照
- WSL 下模型下载正常（HF_ENDPOINT 已生效）

## Phase 1：环境（进行中）

- 主 venv `~/.venvs/drama-reazon`：sherpa-onnx + k2-asr ✅ 已装；**espnet-asr 待装**（重依赖）
- 现有 `~/.venvs/drama-nemo-ass` 继续承担 Sortformer diarization（NeMo 保留）
- 分离 venv `~/.venvs/drama-wesep`：Phase 4 再建
- 两个 venv 之间靠 CLI + 文件产物通信，无 import 依赖

## Phase 2：多引擎 ASR adapter ✅ 已完成

新模块 `asr_reazon.py` + `benchmark.py`：

- `run_k2_asr()`：分窗解码（25s 窗 / 15s stride），subword 单点时间戳 → 中点边界 → SudachiPy 并词
- `run_espnet_asr()`：官方 transcribe（20s 分窗 + CTC 分段），segment → SudachiPy 按字符数均分词时间
- CLI：`asr --engine whisper|k2|espnet`、`run --engine ...`、`benchmark-asr`
- 单测 `tests/test_asr_reazon.py`（25 个测试全通过）

**实测 CER（151s 样本，evaluation/02）**：whisper 25.5%（GPU 47s）/ espnet 30.1%（GPU 222s，带标点分句）/ k2 47.4%（CPU 12s，窗口开头丢词）。
k2 未达「≥ whisper」验收标准，主线维持 whisper；espnet 作为字幕形态最优候选；
k2 保留用于 Phase 5 重叠 stem 短音频试点。

## Phase 3：对齐集成 ✅ 已完成

- espnet 词级时间戳改用 CTC 真实对齐（`get_timings` 逐字符 → SudachiPy 并词），不再段内均分
- cpCER 41.5% → 39.6%，DER 41.6% → 41.2%
- 剩余 DER 差距来自 espnet 语音覆盖率（miss 55s），留待 Phase 4 / 双引擎融合

## Phase 4：重叠分支（改进点 3）⏸ 框架完成，分离器待接入

已完成：

- `overlap.py`：`plan_overlap_jobs`（重叠区+上下文切片）/ `run_overlap_separation`（subprocess worker）/
  `assess_overlap_results`（质量闸门：空文本、双 stem 文本重复）/ `merge_overlap_segments`（双行合并）
- `scripts/separate_worker.py`：WeSep 侧 worker 契约（job.json → stems → result.json）
- `render.py`：`_assign_lanes` 车道分配（重叠段不同 MarginV 垂直排布）+ `--source overlap`
- CLI：`overlap outputs/XX --context 0.7 [--separator-cmd "..."]`，40 个测试全过
- 端到端（outputs/12）：无分离器时全部 job 正确标记 `needs_review`，渲染链路正常

**阻塞项**：WeSep 目前没有发布预训练 TSE checkpoint（其 README 中 Pretrained models 仍在 To Do）。
pyannote PixIT 零样本实验已做：恢复率 60.6% → 27.7%，**负收益**，路线关闭。

## Phase 4.5：双引擎融合 ✅ 已完成（可选开关）

- `scripts/exp_fusion_espnet.py` / `exp_fusion_whisper.py`：每 overlap clip 双引擎转录（各一次模型加载）
- `overlap.py: fuse_overlap_transcripts()`：相似度去重 + 每 job 最多 2 条 + 基线未覆盖才保留 + 归给"另一说话人"
- CLI：`overlap --fuse`；评估：`scripts/eval_fusion.py`
- 实测：重叠区参考句子恢复 **129→141（+12 句）**，OSD F1 0.001→**0.103**；
  代价 CER 30.1%→38.3%（融合行幻觉文本），按需启用

## Phase 5：评测、回归与文档

- 评测扩展：stem 质量指标；OSD P/R/F1 已有
- 测试：fake model 单测覆盖 adapter 转换、时间戳插值、重叠区生成、multi-lane render
- 文档：RUNBOOK 更新环境与命令，README 架构图更新
- 全量回归：`evaluation/02` 上 whisper 基线 vs 新管线全指标对比表

## 验收标准

| 指标 | 标准 | 现状 |
| --- | --- | --- |
| CER/cpCER | 新引擎 ≥ whisper large-v3 | k2 ❌ 47.4%；espnet ❌ 30.1%（但有标点分句）；whisper 25.5% 为基线 |
| DER/JER | 对齐词戳后 ≥ 现有 whisper 词戳基线 | 待测 |
| OSD F1 | 参考重叠区出现双行 ASS 且第二人文本被恢复 | 待做（Phase 4） |
| 资源 | RTF、峰值显存记录并纳入报告 | k2 RTF 0.064 / espnet 0.48 |

## 主要风险

1. k2-v2 分窗解码窗口开头丢词 → 已量化（CER 47.4%），仅用于短音频场景
2. WeSep 训练域为中英，动漫日语 TSE 效果未知 → 02 样本先验，效果差则 `needs_review` 回退
3. espnet 重依赖与 torch 2.5.1 兼容性 → 已解决（numpy 2.x + setuptools<74 + ctc_segmentation 源码编译）
4. 词级时间戳变动影响下游 → 格式契约不变 + 全量回归兜底
