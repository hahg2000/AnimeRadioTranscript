# PLAN_NeMo.md: Windows + WSL2 的 faster-whisper large-v3 + NVIDIA NeMo 广播剧 ASS 方案

## Summary
- 从零新建独立项目 `drama-nemo-ass`，不依赖当前 `broadcast_ass` 或其他旧代码。
- 主线组合：`faster-whisper large-v3` 做日语 ASR，`NVIDIA NeMo Sortformer` 做说话人分离。
- 目标场景：Windows 主机、本地 NVIDIA GPU、几十分钟日语动漫广播剧、固定 2 或 3 个主要说话人、输出可在 Aegisub/mpv 打开的 `.ass`。
- 默认运行环境：Windows + WSL2 Ubuntu。Docker 可选，Windows 原生运行 NeMo 不作为默认路线。

## WSL2 安装位置策略
- `wsl --install -d Ubuntu-22.04` 默认通常会把发行版数据放到系统盘用户目录下，C 盘空间少时不推荐直接这样装大型 ML 环境。
- 优先方案：使用 Microsoft WSL 新版支持的 `--location` 把 Ubuntu 安装到 E 盘：
  ```powershell
  wsl --install -d Ubuntu-22.04 --location E:\WSL\Ubuntu-22.04
  ```
- 如果当前 Windows/WSL 版本不支持 `--location`，使用导入法安装到 E 盘：
  ```powershell
  wsl --list --online
  wsl --install --no-distribution
  mkdir E:\WSL
  ```
  然后下载 Ubuntu rootfs tar，并导入：
  ```powershell
  wsl --import Ubuntu-22.04-NeMo E:\WSL\Ubuntu-22.04-NeMo E:\Downloads\ubuntu-22.04-rootfs.tar --version 2
  wsl --set-default Ubuntu-22.04-NeMo
  ```
- 如果已经误装到 C 盘，可迁移：
  ```powershell
  wsl --shutdown
  wsl --export Ubuntu E:\WSL\ubuntu-backup.tar
  wsl --unregister Ubuntu
  wsl --import Ubuntu-22.04-NeMo E:\WSL\Ubuntu-22.04-NeMo E:\WSL\ubuntu-backup.tar --version 2
  wsl --set-default Ubuntu-22.04-NeMo
  ```
- 项目、虚拟环境、模型缓存都放在 WSL 的 E 盘位置或 WSL Linux 文件系统中，避免 C 盘增长：
  ```bash
  mkdir -p /mnt/e/projects/drama-nemo-ass
  mkdir -p /mnt/e/models/huggingface
  ```
- 在 WSL 内设置缓存路径：
  ```bash
  echo 'export HF_HOME=/mnt/e/models/huggingface' >> ~/.bashrc
  echo 'export HUGGINGFACE_HUB_CACHE=/mnt/e/models/huggingface/hub' >> ~/.bashrc
  echo 'export TRANSFORMERS_CACHE=/mnt/e/models/huggingface/transformers' >> ~/.bashrc
  ```

## Key Interfaces
- CLI：
  ```bash
  python -m drama_nemo_ass run INPUT --out OUT_DIR --language ja --speakers 2|3
  python -m drama_nemo_ass prepare INPUT --out OUT_DIR
  python -m drama_nemo_ass asr OUT_DIR --model large-v3 --language ja
  python -m drama_nemo_ass diarize OUT_DIR --diar-model nvidia/diar_sortformer_4spk-v1 --speakers 2|3
  python -m drama_nemo_ass relabel OUT_DIR
  python -m drama_nemo_ass render OUT_DIR --speaker-map speakers.json
  python -m drama_nemo_ass compare reference.ass OUT_DIR/output.ass
  ```
- 输出文件：
  `prepared.wav`、`asr_words.jsonl`、`asr_segments.json`、`nemo_diarization.json`、`nemo_diarization.rttm`、`segments.json`、`review.tsv`、`output.ass`、`metrics.json`。
- 核心数据结构：
  - `asr_words.jsonl`: `text/start/end/language/probability/chunk_id`
  - `nemo_diarization.json`: `speaker/start/end/confidence/source`
  - `segments.json`: `speaker/start/end/text/flags/confidence`

## Implementation
- WSL 基础环境：
  ```bash
  sudo apt update
  sudo apt install -y ffmpeg git build-essential python3.10 python3.10-venv python3-pip
  cd /mnt/e/projects
  mkdir -p drama-nemo-ass
  cd drama-nemo-ass
  python3.10 -m venv .venv
  source .venv/bin/activate
  python -m pip install -U pip setuptools wheel
  ```
- Python 依赖：
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  pip install "nemo_toolkit[asr]" faster-whisper pysubs2 soundfile librosa scikit-learn pydantic
  pip install sudachipy sudachidict_core pytest
  ```
- 音频预处理：
  - 用 FFmpeg 统一转为 `16kHz mono wav`。
  - 默认不做强降噪、不做人声分离。
  - BGM/音效特别重时，再单独加入可选的 vocal separation 实验，不进入 v1 默认流程。
- ASR：
  - 使用 `faster-whisper` 的 `WhisperModel("large-v3", device="cuda", compute_type="float16")`。
  - 默认参数：`language="ja"`、`word_timestamps=True`、`beam_size=5`、`vad_filter=False`、`condition_on_previous_text=False`。
  - 保存词级时间戳到 `asr_words.jsonl`，ASR 阶段不决定说话人。
- NeMo diarization：
  - 默认使用离线 `nvidia/diar_sortformer_4spk-v1`。
  - Sortformer 输出最多 4 个 speaker，后处理按 `--speakers 2|3` 收敛到固定人数，得到 `SPEAKER_00/01/02`。
- 词到说话人归属：
  - 默认用词时间区间与 speaker 活动概率的积分/覆盖率归属。
  - 若同一时间命中多个 speaker，仍冲突则延续上一 speaker。
  - 重叠说话 v1 只输出主 speaker，保留 `overlap` flag 供人工复核。
- 分句与 ASS：
  - 用 SudachiPy mode C 做日语词边界，避免切开 `コーナー`、`小学校`、连续片假名/汉字词。
  - 默认分句参数：gap `0.20s`、soft duration `3.0s`、hard duration `4.2s`、soft chars `22`、hard chars `34`。
  - speaker 切换是强切分信号；若切点落在明显词内，延迟到下一个安全词边界。
  - 用 `pysubs2` 输出 ASS；每个 speaker 一个 style，Dialogue `Name` 写角色名或 `SPEAKER_00`。

## Test Plan
- 单元测试：
  - FFmpeg 输出为 16k mono wav。
  - faster-whisper 输出正确转换为 `asr_words.jsonl`。
  - NeMo RTTM/turns 正确转换为统一 JSON。
  - `--speakers 2|3` 后 speaker 数收敛正确。
  - 日语词边界分句不切开长音、复合词、连续片假名/汉字。
- 集成测试：
  - 30 秒双人样例、3 人样例、5 分钟以上长音频。
  - 用人工 `sample.ass` 跑 compare，输出 event count、CER、speaker confusion 简表。
- 验收标准：
  - `output.ass` 可被 Aegisub/mpv 打开。
  - 字幕时间递增、无负时长、单段不超过 hard duration。
  - `review.tsv` 可人工修改 speaker/text 后重新 render。
  - 前 1-2 分钟人工检查不出现明显词内切分和大规模角色互换。

## Assumptions
- 默认使用本地 NVIDIA GPU，RTX 4080 16GB 作为目标机器。
- 用途为个人本地非商业；`nvidia/diar_sortformer_4spk-v1` 是 CC BY-NC 4.0，需要避免商业发布。
- 主要语言是日语；中文/英文先走通用分句回退逻辑。
- Docker 不进入 v1 默认流程；只有需要复现环境或打包时再补 Dockerfile。
- Windows 只负责保存视频、查看 ASS、运行编辑器；NeMo、Whisper、模型缓存和 Python 环境默认在 WSL2 Ubuntu 内完成。

## References
- Microsoft WSL install docs: https://learn.microsoft.com/en-us/windows/wsl/install
- Microsoft WSL commands, including `--location`, `--export`, `--import`: https://learn.microsoft.com/en-us/windows/wsl/basic-commands
- NeMo diarization intro: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/intro.html
- NeMo diarization models: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/models.html
- NeMo speaker recognition: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_recognition/results.html
- NVIDIA Sortformer model: https://huggingface.co/nvidia/diar_sortformer_4spk-v1
- Whisper large-v3: https://huggingface.co/openai/whisper-large-v3
