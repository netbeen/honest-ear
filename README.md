# honest-ear

AI 英语口语陪练原型。

## 当前状态

当前仓库已经具备一套可运行的 Phase 1 验证原型，重点验证下面这条链路：

- 同一段音频并行跑双通道 ASR
- 本地做 diff 和置信度融合
- 调用 Ark 官方 Python SDK 或本地 LM Studio 输出结构化纠错结果
- 展示忠实文案、理解文案和回复文案

产品目标仍然是 macOS 原生应用，但在 Tauri 壳子尚未搭建完成前，当前版本先提供一个浏览器界面的录音验证页，方便快速测试完整链路。

## 你需要知道

- 这是一个新项目，不做旧版本兼容
- Python 版本直接要求 `3.11+`
- 如果你本机还是 `Python 3.9`，请直接升级，不需要为了旧环境降级项目实现
- 当前浏览器页面只是原生应用前的过渡验证形态，后续可以直接封装进 `Tauri`

## 已实现能力

- 忠实通道：`wav2vec2`，不接语言模型，尽量保留原始表达
- 理解通道：`faster-whisper`
- 融合层：输出 `faithful_text`、`intended_text`、`diff_spans`、`should_correct`
- LLM 层：支持 Ark 官方 Python SDK 与本地 LM Studio，输出结构化 JSON
- 录音页面：长按麦克风开始录音，松开后自动上传分析
- 页面展示：忠实文案、理解文案、回复文案
- 回复语音：分析完成后自动生成并播放本地 TTS，可手动重播
- 样本集：内置 `30` 条测试样本

## 目录结构

```text
src/honest_ear/
  api.py           FastAPI 接口与浏览器录音页入口
  asr.py           双通道 ASR provider
  cli.py           CLI 入口
  config.py        环境配置
  fusion.py        本地 diff 和置信度融合
  llm.py           Ark SDK / LM Studio LLM 调用
  pipeline.py      端到端编排
  samples.py       样本集加载
  schemas.py       结构化数据模型
  tts.py           macOS 本地 TTS 封装
  web/
    index.html     录音验证页面
    recorder.js    浏览器录音与上传逻辑
data/samples/
  phase1_eval_samples.jsonl
tests/
  test_api.py
  test_fusion.py
models/
  .gitkeep
scripts/
  download-asr-models.sh
```

## 环境要求

- macOS
- Python `3.11+`
- 支持 Ark 官方 Python SDK 或本地 LM Studio
- 如需真实本地 ASR，需要安装：
  - `faster-whisper`
  - `transformers`
  - `torch`
  - `soundfile`

## 安装

如果当前 `python3 --version` 小于 `3.11`，请先直接安装新版 Python。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .[dev]
pip install -e .[asr]
cp .env.example .env
```

常用 LLM 配置：

```env
LLM_BACKEND=ark_sdk
LLM_REASONING_EFFORT=low
ARK_BASE_URL=https://ark-cn-beijing.bytedance.net/api/v3
ARK_API_KEY=your-api-key
ARK_MODEL=your-endpoint-id
LM_STUDIO_BASE_URL=http://127.0.0.1:1234/v1
LM_STUDIO_API_KEY=
LM_STUDIO_MODEL=qwen/qwen3.5-35b-a3b
HONEST_EAR_TTS_VOICE=Tessa
HONEST_EAR_TTS_RATE=180
```

后端切换说明：

- `LLM_BACKEND=ark_sdk`：使用 Ark 官方 Python SDK 调用 `chat.completions.create(...)`
- `LLM_BACKEND=lm_studio`：使用本地 `LM Studio` 的 OpenAI-compatible 接口

Ark SDK 说明：

- `ARK_MODEL` 应填写真实接入点 ID（如 `ep-...`），而不是内部 `bots` 路径里的模型别名
- 在当前验证过的接入点上，`LLM_REASONING_EFFORT=low` 可正常工作，`none` 会被 Ark 直连接口拒绝

LM Studio 说明：

- `LM_STUDIO_MODEL` 填写本地加载的模型名，例如 `qwen/qwen3.5-35b-a3b`
- 当前 `LM Studio` 不接受 `response_format.type=json_object`，项目已自动使用更兼容的请求形态

TTS 说明：

- 当前在 macOS `say` 音色里，`Tessa` 的试听效果明显好于 `Flo`，也比 `Karen` 更自然，当前作为默认推荐音色
- `Karen` 可作为备选，整体可用，但英文断句和连读感略重
- 如需调整语速，可修改 `HONEST_EAR_TTS_RATE`，当前验证值为 `180`

## 模型目录约定

- 项目内约定使用 `models/` 存放本地 ASR 模型
- 该目录会保留在仓库中，但真实模型文件不会上传 Git
- 仓库中只保留 `models/.gitkeep`，用于提示后续开发者模型应放在这里

推荐目录结构：

```text
models/
  .gitkeep
  whisper/
    small.en/
    medium.en/
  wav2vec2/
    facebook--wav2vec2-large-960h-lv60-self/
```

## 下载本地 ASR 模型

你可以通过 shell 触发模型下载，方便切换不同模型做效果验证。

下载默认模型：

```bash
./scripts/download-asr-models.sh
```

只下载 Whisper：

```bash
./scripts/download-asr-models.sh --whisper-only --whisper-model small.en
```

只下载 wav2vec2：

```bash
./scripts/download-asr-models.sh \
  --wav2vec2-only \
  --wav2vec2-model facebook/wav2vec2-large-960h-lv60-self
```

同时下载指定组合：

```bash
./scripts/download-asr-models.sh \
  --whisper-model medium.en \
  --wav2vec2-model facebook/wav2vec2-base-960h
```

也可以直接用 Python 命令：

```bash
python -m honest_ear.download_models \
  --whisper-model small.en \
  --wav2vec2-model facebook/wav2vec2-large-960h-lv60-self
```

下载后，脚本会把模型放到 `models/whisper/...` 和 `models/wav2vec2/...` 下。

## 使用本地模型路径

下载完成后，需要将 `.env` 改成本地目录路径，而不是远程模型名：

```env
WHISPER_MODEL_SIZE=/Users/bytedance/Documents/github/honest-ear/models/whisper/small.en
WAV2VEC2_MODEL_NAME=/Users/bytedance/Documents/github/honest-ear/models/wav2vec2/facebook--wav2vec2-large-960h-lv60-self
```

服务只允许读取已经存在的本地模型目录。

- 如果目录不存在，服务会在启动预热阶段直接报错
- 不会在启动后自动下载
- 也不会在收到请求后自动下载

## 启动

启动本地服务：

```bash
./scripts/start-service.sh
```

如需自定义端口或追加 `uvicorn` 参数：

```bash
HONEST_EAR_PORT=8010 ./scripts/start-service.sh
./scripts/start-service.sh --reload-dir src
```

启动后直接打开：

```text
http://127.0.0.1:8000
```

页面会显示一个居中的麦克风按钮：

- 长按开始录音
- 松开结束录音
- 自动上传音频到本地服务分析
- 页面展示忠实文案、理解文案和回复文案
- 自动播放回复语音，并支持手动重播

## 命令行方式

分析本地音频文件：

```bash
honest-ear process /absolute/path/to/input.wav --mode accuracy
```

查看内置样本：

```bash
honest-ear list-samples
```

## API

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

查看样本集：

```bash
curl http://127.0.0.1:8000/v1/samples
```

直接传文件分析：

```bash
curl -X POST http://127.0.0.1:8000/v1/process-upload \
  -F "audio=@/absolute/path/to/input.wav" \
  -F "mode=accuracy" \
  -F "speak_reply=false"
```

## 输出说明

融合层的中间结果形如：

```json
{
  "faithful_text": "he dont like the coffee what you make",
  "intended_text": "he doesn't like the coffee you made",
  "faithful_confidence": 0.78,
  "intended_confidence": 0.92,
  "diff_spans": [
    {
      "faithful": "dont",
      "intended": "doesn't",
      "start_ms": 100,
      "end_ms": 200,
      "confidence": 0.84,
      "reason": "likely_grammar_inflection"
    }
  ],
  "should_correct": true,
  "gating_reason": "stable_diff_detected"
}
```

LLM 返回结果形如：

```json
{
  "reply": "I understood you. Here is a cleaner way to say it.",
  "should_show_correction": true,
  "corrections": [
    {
      "wrong": "he dont like",
      "right": "he doesn't like",
      "why": "Third-person singular verbs in the present simple need does not.",
      "confidence": 0.83
    }
  ],
  "faithful_text": "he dont like the coffee what you make",
  "intended_text": "he doesn't like the coffee you made",
  "naturalness_score": 80,
  "mode": "accuracy",
  "meta": {
    "decision_reason": "stable_diff_detected"
  }
}
```

## 测试

运行自动化测试：

```bash
pytest
```

你认可的“可接受测试状态”在当前实现中对应为：

- 启动程序后有可见 UI
- 页面中间有麦克风按钮
- 长按开始录音，放手结束
- 音频上传到 app 分析
- 页面展示忠实文案、理解文案和回复文案

## 下一步

如果要继续朝原生应用推进，推荐下一步直接做：

- `Tauri 2` 壳
- React 前端
- 当前 Python 服务作为本地推理后端

这样当前录音页和接口设计可以基本保留，不会浪费这轮实现。
