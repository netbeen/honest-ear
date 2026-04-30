const micButton = document.getElementById("micButton");
const statusText = document.getElementById("status");
const faithfulText = document.getElementById("faithfulText");
const faithfulMeta = document.getElementById("faithfulMeta");
const intendedText = document.getElementById("intendedText");
const intendedMeta = document.getElementById("intendedMeta");
const correctionText = document.getElementById("correctionText");
const correctionMeta = document.getElementById("correctionMeta");
const replyText = document.getElementById("replyText");
const replyMeta = document.getElementById("replyMeta");
const replayButton = document.getElementById("replayButton");
const replyAudio = document.getElementById("replyAudio");

let mediaStream = null;
let audioContext = null;
let processor = null;
let source = null;
let recording = false;
let pcmChunks = [];
let sampleRate = 44100;
let latestTtsAudioUrl = "";

function setStatus(message) {
  statusText.textContent = message;
}

function setResult(result) {
  faithfulText.textContent = result.faithful_asr?.text || "暂无结果";
  faithfulMeta.textContent = `置信度：${formatConfidence(result.faithful_asr?.confidence)}`;

  intendedText.textContent = result.intended_asr?.text || "暂无结果";
  intendedMeta.textContent = `置信度：${formatConfidence(result.intended_asr?.confidence)}`;

  const renderedCorrections = formatCorrections(result.llm?.corrections || []);
  correctionText.textContent = renderedCorrections.text;
  correctionMeta.textContent = renderedCorrections.meta;

  replyText.textContent = result.llm?.reply || "暂无结果";
  const ttsReady = result.tts_audio_url ? "已生成" : "未生成";
  replyMeta.textContent = `是否建议纠错：${result.fusion?.should_correct ? "是" : "否"} | 回复语音：${ttsReady}`;
}

function formatConfidence(value) {
  if (typeof value !== "number") {
    return "-";
  }
  return value.toFixed(2);
}

/**
 * Formats structured correction items into readable UI text and metadata.
 */
function formatCorrections(corrections) {
  if (!Array.isArray(corrections) || corrections.length === 0) {
    return {
      text: "本轮没有明确的语法纠错。",
      meta: "纠错条目：0",
    };
  }

  const lines = corrections.map((item, index) => {
    const wrong = item?.wrong || "-";
    const right = item?.right || "-";
    const why = item?.why || "未提供原因";
    return `${index + 1}. ${wrong} -> ${right}\n原因：${why}`;
  });

  return {
    text: lines.join("\n\n"),
    meta: `纠错条目：${corrections.length}`,
  };
}

function mergeChunks(chunks) {
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const merged = new Float32Array(totalLength);
  let offset = 0;
  chunks.forEach((chunk) => {
    merged.set(chunk, offset);
    offset += chunk.length;
  });
  return merged;
}

function downsampleBuffer(buffer, inputRate, outputRate) {
  if (outputRate === inputRate) {
    return buffer;
  }
  const sampleRateRatio = inputRate / outputRate;
  const newLength = Math.round(buffer.length / sampleRateRatio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * sampleRateRatio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i += 1) {
      accum += buffer[i];
      count += 1;
    }
    result[offsetResult] = accum / count;
    offsetResult += 1;
    offsetBuffer = nextOffsetBuffer;
  }
  return result;
}

async function readErrorMessage(response) {
  try {
    const payload = await response.json();
    if (payload && typeof payload.detail === "string" && payload.detail.trim()) {
      return payload.detail.trim();
    }
  } catch (_error) {
    // Ignore JSON parse errors and fall back to plain text below.
  }

  const fallbackText = await response.text();
  return fallbackText.trim() || `请求失败，HTTP ${response.status}`;
}

/**
 * Updates the current reply audio source and toggles replay availability.
 */
function setReplyAudio(url) {
  latestTtsAudioUrl = typeof url === "string" ? url : "";
  replayButton.disabled = !latestTtsAudioUrl;
  replyAudio.src = latestTtsAudioUrl;
  replyAudio.load();
}

/**
 * Attempts to play the latest synthesized reply audio and surfaces autoplay failures.
 */
async function playReplyAudio() {
  if (!latestTtsAudioUrl) {
    return;
  }

  try {
    await replyAudio.play();
  } catch (error) {
    console.warn(error);
    setStatus("分析完成，语音已生成；如未自动播放，请点击“重播回复语音”");
  }
}

function encodeWav(samples, outputRate) {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const buffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
  const view = new DataView(buffer);

  writeString(view, 0, "RIFF");
  view.setUint32(4, 36 + samples.length * bytesPerSample, true);
  writeString(view, 8, "WAVE");
  writeString(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, outputRate, true);
  view.setUint32(28, outputRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, "data");
  view.setUint32(40, samples.length * bytesPerSample, true);

  floatTo16BitPCM(view, 44, samples);
  return new Blob([view], { type: "audio/wav" });
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i += 1) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

function floatTo16BitPCM(view, offset, input) {
  for (let i = 0; i < input.length; i += 1, offset += 2) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
}

async function startRecording() {
  if (recording) {
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioContext = new (window.AudioContext || window.webkitAudioContext)();
    sampleRate = audioContext.sampleRate;
    source = audioContext.createMediaStreamSource(mediaStream);
    processor = audioContext.createScriptProcessor(4096, 1, 1);
    pcmChunks = [];

    processor.onaudioprocess = (event) => {
      if (!recording) {
        return;
      }
      const inputData = event.inputBuffer.getChannelData(0);
      pcmChunks.push(new Float32Array(inputData));
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
    recording = true;
    micButton.classList.add("recording");
    setStatus("录音中，松开按钮后开始分析");
  } catch (error) {
    console.error(error);
    setStatus("无法访问麦克风，请检查浏览器权限");
  }
}

async function stopRecording() {
  if (!recording) {
    return;
  }

  recording = false;
  micButton.classList.remove("recording");
  setStatus("正在上传音频并分析...");

  processor.disconnect();
  source.disconnect();
  mediaStream.getTracks().forEach((track) => track.stop());
  await audioContext.close();

  const merged = mergeChunks(pcmChunks);
  const downsampled = downsampleBuffer(merged, sampleRate, 16000);
  const wavBlob = encodeWav(downsampled, 16000);

  try {
    const formData = new FormData();
    formData.append("audio", wavBlob, "recording.wav");
    formData.append("mode", "accuracy");
    formData.append("speak_reply", "true");

    const response = await fetch("/v1/process-upload", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(await readErrorMessage(response));
    }

    const result = await response.json();
    setResult(result);
    setReplyAudio(result.tts_audio_url || "");
    setStatus(result.tts_audio_url ? "分析完成，正在播放回复语音" : "分析完成，可以继续长按录音");
    await playReplyAudio();
  } catch (error) {
    console.error(error);
    setReplyAudio("");
    setStatus(`分析失败：${error.message}`);
  } finally {
    pcmChunks = [];
    mediaStream = null;
    audioContext = null;
    processor = null;
    source = null;
  }
}

micButton.addEventListener("pointerdown", async (event) => {
  event.preventDefault();
  await startRecording();
});

micButton.addEventListener("pointerup", async (event) => {
  event.preventDefault();
  await stopRecording();
});

micButton.addEventListener("pointerleave", async () => {
  if (recording) {
    await stopRecording();
  }
});

micButton.addEventListener("touchstart", async (event) => {
  event.preventDefault();
  await startRecording();
});

micButton.addEventListener("touchend", async (event) => {
  event.preventDefault();
  await stopRecording();
});

replayButton.addEventListener("click", async () => {
  await playReplyAudio();
});
