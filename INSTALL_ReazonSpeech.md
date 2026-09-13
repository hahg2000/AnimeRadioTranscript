# ReazonSpeech 环境安装步骤（Phase 0 spike / Phase 1 主环境）

> 计划文档：`PLAN_ReazonSpeech_v3.md`（已修订为 v2 方案）。本文按用户手动执行编写，命令全部在 WSL2 Ubuntu 中运行。

## 0. 已确认结论（2026-08-21 实测）

1. **v3 模型不存在**：`reazonspeech-espnet-v3` / `reazonspeech-k2-v3` 在 HF 上 404。
   官方最新为 v2 系列：`reazonspeech-k2-v2`（Zipformer ONNX）与 `reazonspeech-espnet-v2`（Conformer-Transducer）。
2. **官方包没有 diarization**：说话人分离保留现有 NeMo Sortformer（`~/.venvs/drama-nemo-ass`）。
3. k2-v2 已实测可用：CPU RTF 0.063，30s 片段文本正确；官方基准 CER 优于 whisper large-v3 和 espnet-v2。
4. k2-v2 限制：>30s 长音频需分窗；subword 时间戳为单点且头部有 0.9s padding 伪影；无标点。

## 当前进度

| 步骤 | 状态 |
| --- | --- |
| venv `~/.venvs/drama-reazon` + torch + sherpa-onnx + k2-asr | ✅ 完成，k2 冒烟通过 |
| espnet-asr + espnet-v2 模型 | ✅ 完成（numpy 2.x + ctc_segmentation 源码重编译 + setuptools<74） |
| v3 模型检查 | ✅ 已确认不存在 |
| sudachipy + sudachidict_core（k2 adapter 分词依赖） | ✅ 已装 |
| benchmark-asr | ✅ 跑通：whisper 25.5% / espnet 30.1% / k2 47.4%（详见 spike_notes.md） |
| WeSep venv | 未开始（Phase 5 再做） |

## 1. 修复 numpy 冲突 + 下载 espnet-v2 模型（当前唯一待办）

问题：`espnet 202511` 要求 `numpy>=2.0.0`，但 `ctc_segmentation` 的预编译 wheel 是
针对 numpy 1.x 编译的。解决办法：numpy 回到 2.x，并从源码重编译 `ctc_segmentation`
（WSL 已有 gcc）。

```bash
source ~/.venvs/drama-reazon/bin/activate

# 1. numpy 回到 2.x（espnet 202511 的要求）
pip install "numpy>=2.0.0"

# 2. 从源码重编译 ctc_segmentation（需要 cython + gcc）
pip install cython
pip install --no-binary ctc-segmentation --force-reinstall ctc-segmentation

# 3. 验证 import 链
python -c "import ctc_segmentation, numpy; print('ctc ok, numpy', numpy.__version__)"
python -c "from reazonspeech.espnet.asr import load_model; print('espnet-asr ok')"
```

常见问题：

- 若 `--no-binary` 编译失败（缺 python3-dev），执行
  `sudo apt install -y python3-dev python3.10-dev` 后重试。
- 若第 3 步第二句报 `HF_ENDPOINT` 相关错误，见下一步网络设置。

## 2. 下载 espnet-v2 模型（在设置 HF_ENDPOINT 的 shell 里执行）

WSL 直连 huggingface.co 不通，必须走 hf-mirror（k2 模型当时能下载成功就是因为
HF_ENDPOINT 已生效）。**注意：`bash -lc` 不读 .bashrc，每个新 shell 都要先 export。**

```bash
source ~/.venvs/drama-reazon/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

# 用 espnet_model_zoo 的下载器预下载模型（走 HF_ENDPOINT 镜像）
python - <<'EOF'
from espnet_model_zoo.downloader import ModelDownloader
d = ModelDownloader()
path = d.download_and_unpack("https://huggingface.co/reazon-research/reazonspeech-espnet-v2")
print("downloaded to:", path)
EOF
```

输出一个目录路径即成功。若报 `LocalEntryNotFoundError`，先确认
`curl -sI https://hf-mirror.com | head -1` 返回 200，再重试。

## 3. 冒烟测试

```bash
cd /mnt/e/gitCode/drama-nemo-ass
python scripts/spike_smoke.py
```

这次应输出 espnet engine 的结果（TranscribeResult: text + segments，segments 含 start_seconds/end_seconds/text）。
注意 espnet 推理在 CPU 上较慢（120M Conformer），25s 片段可能需要几十秒；
GPU 可用时 `load_model()` 会自动选 cuda。

## 4. WeSep venv（Phase 4 分离器，可选）

```bash
python3.10 -m venv ~/.venvs/drama-wesep
source ~/.venvs/drama-wesep/bin/activate
python -m pip install -U pip
git clone https://github.com/wenet-e2e/wesep.git ~/wesep
# 按 ~/wesep/README.md 的 Installation 章节安装 espnet 依赖和 TSE 模型
```

注意：

- **WeSep 目前没有发布预训练 TSE checkpoint**（README 的 Pretrained models 仍在 To Do）。
  分离器是可插拔接口：`overlap --separator-cmd "python /path/to/worker.py"`，
  worker 契约见 `scripts/separate_worker.py` 的 docstring。
- WeSep 依赖 ESPnet，与主 venv 的 espnet 版本可能冲突，**必须保持两个 venv 隔离**。
- 没有分离器时，`overlap` 命令仍可运行：所有重叠 job 标记 `needs_review`，不硬编错误结果。

## 5. pyannote 零样本分离实验（已关闭，结论：负收益）

实验脚本保留：`scripts/separate_worker_pyannote.py`、`scripts/exp_separate_all.py`。
结论见 `spike_notes.md`：恢复率 60.6% → 27.7%，负收益。

## 6. 双引擎融合（可选开关，overlap --fuse）

```bash
# 1) 两个引擎分别转写所有 overlap clip（各一次模型加载）
~/.venvs/drama-reazon/bin/python scripts/exp_fusion_espnet.py outputs/02
~/.venvs/drama-nemo-ass/bin/python scripts/exp_fusion_whisper.py outputs/02

# 2) 主流程：融合进 segments_overlap.json + 渲染
python -m drama_nemo_ass overlap outputs/02 --context 0.7 --fuse
python -m drama_nemo_ass render outputs/02 --source overlap

# 3) 评估（恢复率 + CER 双视角）
~/.venvs/drama-nemo-ass/bin/python scripts/eval_fusion.py outputs/02 evaluation/02
```

实测：重叠区参考句子恢复 129→141（+12 句），OSD F1 0.001→0.103；
代价是 CER 30.1%→38.3%（融合行带幻觉文本）。按需启用。

## 7. 完成后

把 espnet 冒烟输出贴回来，即进入 Phase 2（k2-v2 ASR adapter 开发）。
