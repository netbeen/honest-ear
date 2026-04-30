# Project Brief: **HonestEar** — AI English Speaking Coach

> 取名寓意：**Honest** = 忠实转录、不美化用户语法错误；**Ear** = 像真人老师一样的"耳朵"，听得懂意图又听得清错误。
> 备选名：`TrueParrot` / `RawTongue` / `CandidCoach` / `MirrorTalk` / `PlainSpeak`

---

## 一、项目目标

打造一款可运行在 **macOS** 上的 AI 英语口语陪练 App。核心体验是像真人外教一样自然对话，但语音能力尽量 **本地运行**，只把文本送给你提供的 **OpenAI 兼容 chat LLM** 做理解、回复和纠错决策。

### 目标拆解
- **对话自然**：像真人外教一样连续交流，不过度打断。
- **纠错诚实**：尽量保留学习者原始表达，不把语法错误自动美化掉。
- **本地优先**：录音、VAD、ASR、TTS 优先本地执行，降低隐私风险和调用成本。
- **可验证**：先验证"双通道是否真的能暴露错误"，再做完整产品化。

---

## 二、核心问题与破局思路

### 用户痛点
主流 ASR（Whisper / Google STT / 讯飞等）为了"好用"，经常会自动平滑用户的语法错误。例如把 `he dont like` 修成 `he doesn't like`，从而**掩盖学习者真实的语言问题**。

### 破局方案：本地双通道 ASR + 置信度融合 + 远程文本 LLM

```text
用户语音
   ├─→ 本地 VAD 切段
   ├─→ 忠实通道（本地 wav2vec2 / CTC，无语言模型）
   │        → "HE DONT LIKE THE COFFEE WHAT YOU MAKE"
   └─→ 理解通道（本地 Whisper）
            → "He doesn't like the coffee you made"
                             ↓
                  本地对齐 / 置信度融合层
                             ↓
               OpenAI-compatible chat LLM
                             ↓
       ┌─────────────────────┴─────────────────────┐
       ↓                                           ↓
 本地 TTS 语音回复                        结构化纠错卡片（语法 / 用词）
```

**关键洞察**：
- 忠实通道 = 学生"更接近实际说了什么"
- 理解通道 = 学生"更可能想表达什么"
- 两路差异 + 置信度 = 候选教学点，而不是直接等于错误
- 远程 LLM 只处理文本，不直接接触原始音频

### 为什么要加"对齐 / 置信度融合层"
仅靠 `wav2vec2` 与 `Whisper` 的差异还不够。因为差异可能来自：
- 学生真实语法错误
- 口音导致的 ASR 偏差
- 背景噪声导致的识别错误
- 连读、吞音、停顿带来的边界切分误差

因此在把结果交给 LLM 前，必须先做一层本地规则判断：
- 忠实通道置信度过低时，不要强行纠错
- 两通道差异很大但都不稳定时，只做自然回复，不展示明确纠错
- 只有差异稳定、且能映射到具体词组或语法点时，才输出纠错卡片

---

## 三、技术选型（三档推荐）

| 方案 | App 形态 | 本地语音链路 | 远程能力 | 延迟目标 | 推荐度 |
|---|---|---|---|---|---|
| **A. 本地 MVP** | Tauri + React | Silero VAD + wav2vec2 + faster-whisper + 本地 TTS | OpenAI-compatible chat LLM | 2–5s / 轮 | ★★★★★ |
| **B. 本地增强** | Tauri + React | 在 A 基础上补充量化、缓存、分段并行、纠错回放 | OpenAI-compatible chat LLM | 1.5–4s / 轮 | ★★★★ |
| **C. 可选云增强** | 同上 | 本地链路保留 | 后续再接云端发音评估或 Realtime 能力 | 视供应商而定 | ★★ |

**MVP 建议**：先用 **方案 A**。不要一开始追求"真人级实时通话"，先做 **turn-based** 的自然对话，把双通道纠错准确性跑通。

### 为什么不把云语音放主路径
- 你当前的资源条件是：**语音模型本地跑**，远程只提供 **chat 类型 LLM**。
- 这意味着 `GPT-4o Realtime`、`Gemini Live`、`Azure Pronunciation Assessment` 都不应作为 MVP 主路径。
- 它们可以保留在长期路线图里，但不能写进当前最小可行方案的核心依赖。

---

## 四、关键工程要点（交付给 code agent 必看）

### 1. App 形态必须按 macOS 桌面应用设计
不要把项目默认成纯网页。建议：
- UI：`React`
- 桌面壳：`Tauri 2`
- 本地推理服务：`Python + FastAPI`

这样可以兼顾：
- 更方便访问本地模型与本地文件系统
- 更容易分发为 macOS App
- 前端仍可复用现代 Web UI 技术栈

### 2. Whisper 只承担"理解通道"，不要让它决定纠错结果
Whisper 的优势是语义恢复强，但它容易把错误说法平滑成标准表达。建议把它定位为：
- 用于理解用户意图
- 用于生成较自然的 intended text
- 不直接作为"学生实际说了什么"的最终依据

建议参数原则：

```python
model.transcribe(
    audio,
    temperature=0.0,
    condition_on_previous_text=False,
    initial_prompt=None,
)
```

说明：
- `temperature=0.0` 减少随机性
- `condition_on_previous_text=False` 避免上下文脑补
- 即便如此，Whisper 仍然可能自动纠正语法，所以它只能是"理解通道"

### 3. 忠实通道必须坚持无语言模型解码
忠实通道建议使用 `facebook/wav2vec2-large-960h-lv60-self` 作为首个实验基线，并明确：
- **禁止接 KenLM / n-gram LM**
- 优先保留原始 CTC 风格输出
- 输出 token / word 级置信度，供后续融合层使用

注意：
- `wav2vec2` 是否能成为最终生产方案，需要通过你的测试集验证
- 若中式口音下误差过大，可以再评估更适合非母语英语场景的本地模型，但"无 LM、少脑补"这个原则不变

### 4. 必须增加本地对齐 / 置信度融合层
不要直接把两路文本丢给 LLM。中间先做一层轻量逻辑，输出统一结构：

```json
{
  "faithful_text": "he dont like the coffee what you make",
  "intended_text": "he doesn't like the coffee you made",
  "faithful_confidence": 0.78,
  "diff_spans": [
    {
      "faithful": "dont",
      "intended": "doesn't",
      "start_ms": 540,
      "end_ms": 890
    }
  ],
  "should_correct": true
}
```

最小规则建议：
- 忠实通道置信度低于阈值时，`should_correct = false`
- 只保留词级或短语级差异，不做整句级拍脑袋纠错
- 一轮最多上报 1 到 2 个候选纠错点

### 5. 裁决层 Prompt 要面向"文本 LLM"，并输出更完整的 JSON
因为你接的是 **OpenAI 兼容 chat 接口**，Prompt 应该明确告诉模型：
- 它拿到的是两份文本和置信度，不是原始音频
- 它不能把低置信度片段当成确定性错误
- 它的任务是"自然回复 + 适度纠错"，不是语法批改器

推荐输出格式：

```json
{
  "reply": "I see what you mean. He probably didn't like the coffee very much.",
  "should_show_correction": true,
  "corrections": [
    {
      "wrong": "he dont like",
      "right": "he doesn't like",
      "why": "Third-person singular verbs in the present simple need 'doesn't'.",
      "confidence": 0.83
    }
  ],
  "faithful_text": "he dont like the coffee what you make",
  "intended_text": "he doesn't like the coffee you made",
  "naturalness_score": 82
}
```

### 6. 避免过度纠错
- 单轮最多指出 **1 到 2 个** 错误
- 提供两种模式：
  - **Fluency Mode**：只纠影响理解或高频核心错误
  - **Accuracy Mode**：允许更细粒度纠错
- 当置信度不足时，优先给自然回应，不展示明显纠错
- 错误应累计到 session 结束后做复盘，而不是每句都打断

### 7. 发音评估不要放进 MVP 主链路
语法错误不等于发音错误。当前阶段不建议把发音评估做成主依赖，原因是：
- 你当前的前提是本地语音优先
- 高质量音素级评分通常依赖外部服务或额外复杂模型
- 如果主链路还没稳定，发音评估会明显分散工程精力

MVP 阶段建议：
- 先只做语法 / 用词纠错
- 发音能力留到 Phase 3 再接入

### 8. 前端交互建议
- 主界面保持干净，以语音输入和 AI 回复为主
- 纠错信息以侧边栏卡片或折叠面板展示
- 每轮显示"原始表达 / 推荐表达 / 解释"
- Session 结束生成错误模式分析报告，突出重复犯错的点

---

## 五、推荐技术栈

| 层 | 技术建议 | 备注 |
|---|---|---|
| macOS App 壳 | `Tauri 2` | 比 Electron 更轻，更适合本地工具型 App |
| 前端 UI | `React` + `TypeScript` | 负责录音、状态管理、纠错卡片展示 |
| 本地服务 | `Python + FastAPI` | 统一承接模型加载与音频处理 |
| VAD | `Silero VAD` | 负责切分用户语音段 |
| 忠实 ASR | `transformers` + `facebook/wav2vec2-large-960h-lv60-self` | 无 LM 解码，输出置信度 |
| 理解 ASR | `faster-whisper` 或 Apple Silicon 上的 `mlx-whisper` | 优先本地运行 |
| 对齐层 | Python 自定义规则 + 文本 diff | 负责候选错误筛选 |
| 裁决 LLM | 你提供的 OpenAI-compatible chat API | 只传文本，不传原始音频 |
| 本地 TTS | `Piper` / `MeloTTS` / `Kokoro` | 先选易落地、速度够用的方案 |
| 数据存储 | SQLite / 本地 JSON | 用于 session 历史和错误复盘 |

### 模型建议
- `理解通道`：优先尝试 `faster-whisper small.en` 或 `medium.en`
- `Apple Silicon` 优先：可评估 `mlx-whisper`
- `忠实通道`：先用 `wav2vec2-large-960h-lv60-self` 做基线验证
- `TTS`：优先本地可部署方案，不依赖外部云 TTS

---

## 六、性能与产品预期

### MVP 不追求绝对实时
当前阶段建议把体验定义为：
- 用户说完一句
- 本地切段并并行跑双通道 ASR
- 本地融合
- 远程 chat LLM 返回文本
- 本地 TTS 播放回复

这是一个 **turn-based conversation**，不是严格意义上的实时打断式通话。

### 更现实的指标
- 单轮首版目标：**2 到 5 秒**
- 本地模型常驻内存：尽量控制在 **可接受的 Apple Silicon 机器范围内**
- 先保证稳定性，再压缩时延

---

## 七、落地路线图

### Phase 1：Core Validation（1 到 2 周）
- [ ] 搭建离线双通道 demo：同一段录音并行跑 `Whisper` 和 `wav2vec2`
- [ ] 实现本地 diff + 置信度融合层，输出结构化中间结果
- [ ] 接入 OpenAI-compatible chat LLM，输出结构化纠错 JSON
- [ ] 接入本地 TTS，形成最小闭环：`音频 -> 纠错 -> 语音回复`
- [ ] 建立 `30 到 50` 条测试样本，覆盖典型中式英语错误与不同口音

### Phase 1 验证指标
- [ ] 忠实通道对目标错误的暴露率
- [ ] 双通道差异中的误报率
- [ ] 单轮总耗时
- [ ] Apple Silicon macOS 设备上的内存占用
- [ ] LLM 纠错卡片的可读性和教学价值

### Phase 2：MVP（2 到 4 周）
- [ ] 做出 macOS 桌面原型：录音、播放、纠错卡片、历史记录
- [ ] 支持 Fluency / Accuracy 模式切换
- [ ] 增加 session 结束复盘报告
- [ ] 增加错误模式统计，例如第三人称单数、时态、冠词
- [ ] 对本地模型加载、缓存和并行调度做优化

### Phase 3：Polish（持续）
- [ ] 增加更好的本地模型选择与量化策略
- [ ] 研究是否接入发音评估
- [ ] 研究是否需要云端增强能力作为可选模式
- [ ] 增加场景化课程（面试 / 点餐 / 旅行 / 商务）
- [ ] 增加个性化纠错优先级与长期学习画像

---

## 八、风险与备选

| 风险 | 说明 | 缓解方案 |
|---|---|---|
| `wav2vec2` 对非母语口音不稳 | 可能把口音偏差当成语法错误 | 增加置信度阈值；先只纠高确定性错误 |
| 本地双模型延迟偏高 | 一台 Mac 同时跑多模型会有压力 | 并行执行、量化模型、缩小模型尺寸 |
| Whisper 仍然过度平滑 | 理解通道会美化语法 | 明确它只用于 intended text，不直接定错 |
| 过度纠错打击积极性 | 用户可能感觉被频繁打断 | 默认 Fluency Mode，每轮最多 1 到 2 条 |
| TTS 音质不够自然 | 本地 TTS 通常不如云端 | MVP 先保证速度和可用性，后续再替换更优方案 |

---

## 九、MVP 结论

**项目名：HonestEar**

**核心卖点一句话**：*The only English coach that hears what you actually said — not what you meant to say.*

交付给 code agent 时，建议优先完成 **Phase 1 的 Core Validation**，但这里的重点已经不是"接一个云端实时语音 API"，而是：

- 在 `macOS` 上把本地双通道跑通
- 证明"双通道 + 置信度融合"能稳定暴露真实错误
- 证明远程 `OpenAI-compatible chat LLM` 足以承担自然回复与纠错裁决
- 形成最小闭环：`本地音频 -> 本地 ASR -> 文本 LLM -> 本地 TTS`

只要这个闭环成立，再扩展到完整 App 就是工程放大问题，而不是方向风险。
