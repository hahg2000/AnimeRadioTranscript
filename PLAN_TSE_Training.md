# PLAN: 域内自训练 TSE 分离模型

> 创建：2026-08-23。目标：训练一个能在动漫广播剧重叠区做目标说话人提取（TSE）
> 的模型，替换当前「重叠区 needs_review 回退」的现状。
> 前置背景：WeSep 无预训练模型、pyannote PixIT 零样本负收益（见 spike_notes.md）。

## 0. 数据盘点（现有素材）

| 视频 | 时长 | 说话人（标注时长） | 重叠情况 |
| --- | --- | --- | --- |
| video/sample（drama 02） | 151s | speaker0 65s / speaker1 38s / speaker2 76s | **27.2s 真重叠（3 人）** |
| Nin koro/01 | 428s | 忍者 203s / 杀手 224s | 无 |
| Nin koro/02 | 347s | 明日 155s / 下毒 96s / 富美 96s | 1 对，~0s |
| Nin koro/03 | 371s | 老大 197s / 百合 173s | 2 对，2.7s |

合计约 **22 分钟**干净单人语音、11 个说话人标签。

结论：素材量距"训练完整模型"（建议每说话人数小时）还差一个量级，
但**足够跑可行性探针**；且 Nin koro 系列还在持续添加，素材会累积。

## 1. 思路

TSE 模型（如 WeSep SpatialNet_TSE）训练需要成对数据：
「双人混合音频 + 两个干净单人音轨」。

我们只有混合音频，所以用**数据模拟**生成训练样本：
从说话人 A 的独白池和说话人 B 的独白池各随机取一段 → 按随机音量比相加
（可选时间偏移、BGM/噪声增强）→ 得到混合样本 + 两条标准答案音轨。
WeSep 的 on-the-fly data simulation 原生支持，无需预生成文件。

训练后，推理时给模型「重叠区混合音频 + 目标说话人的参考语音（enrollment）」，
它输出只含目标说话人的音轨 → espnet 独立 ASR → 双行 ASS。

## 2. 步骤

### Phase A：数据准备（1-2 天）

1. **切独白段**：用各集 ASS 的非重叠段切出每个说话人的 1-8s 干净 wav 片段
   （`overlap=true` 的段和 <1s 的段排除；可辅以 Sortformer exclusive turns）。
2. **池划分**：train/dev/test 按说话人隔离（Nin koro 01/02/03 做 train；
   video/sample 的三个 speaker 做 dev，其中重叠区做最终 eval——同集不得混用）。
3. **模拟合成**：WeSep 配置数据管道（A/B 随机段、0.2-1.0 随机音量、可选 reverb/noise）。

### Phase B：可行性探针（1 天，先做）

- 用 Nin koro 3 个说话人（下毒/富美/明日）+ 合成 5000 条 2 人混合样本
- 训练 2-4 小时小模型（WeSep voxceleb1 配方改小）
- 判定：dev 集 SISNR > 0 且 dev ASR 增益为正 → 值得继续；
  否则先攒素材（继续加集数），不投入完整训练。

### Phase C：完整训练（GPU 数天，探针通过后）

- 全部说话人 + 增强（BGM 混合、音量扰动、1-3 人场景）
- 监控 SISNR/SI-SNRi，警惕恒等映射退化（输出=输入）

### Phase D：接入管线（1-2 天）

- 导出 checkpoint → 实现 worker（契约已有：`scripts/separate_worker.py`）
- enrollment 来源：每个说话人在 overlap 区外最近的 exclusive turn 音频
- 每 stem 用 espnet 转写 → `merge_overlap_segments` 双行合并 → 渲染 lane
- 验收：evaluation/02 的 OSD F1、参考句子恢复率、CER 三者同时报告

## 3. 验收标准

| 指标 | 门槛 |
| --- | --- |
| dev SISNR | > 0（正增益，拒绝恒等映射） |
| OSD F1（video/sample） | ≥ 0.10（超过当前融合的 0.103，或与融合叠加更高） |
| 重叠区参考句子恢复 | ≥ 当前基线 129/213 且不引入明显 CER 退化 |
| 人工抽检 | 重叠区双行字幕，第二人台词可读 |

## 4. 风险

1. **素材不足**（最大风险）：22 分钟训练出的模型可能严重欠拟合 → 探针先验
2. 恒等映射退化：SISNR 为正但输出原样 → SI-SNRi 与 ASR 增益双重校验
3. 说话人身份混淆：A 的样本被抽成 B → enrollment 用 reference 语音约束
4. 数据泄漏：train 说话人出现在 dev → 池划分隔离

## 5. 与现状的关系

- 不阻塞现有主线（espnet + overlap 框架已可用）
- 成功后替换 `overlap --separator-cmd` 的 stub；失败则维持 needs_review 回退
- 与「双引擎融合」正交：可叠加使用（分离 stem + 融合去重）
