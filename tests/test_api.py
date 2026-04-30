"""API tests for the HonestEar demo UI and upload endpoint."""

from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile
import wave

from fastapi.testclient import TestClient

from honest_ear.api import app
from honest_ear.schemas import ASRResult, CorrectionResponse, FusionResult, PipelineResult


client = TestClient(app)


def _build_pipeline_result(audio_path: Path) -> PipelineResult:
    """Builds a deterministic pipeline result used to isolate API tests from model runtime."""

    faithful = ASRResult(channel="faithful", text="he dont like coffee", confidence=0.81, tokens=[], model_name="mock")
    intended = ASRResult(
        channel="intended",
        text="he doesn't like coffee",
        confidence=0.93,
        tokens=[],
        model_name="mock",
    )
    fusion = FusionResult(
        faithful_text=faithful.text,
        intended_text=intended.text,
        faithful_confidence=faithful.confidence,
        intended_confidence=intended.confidence,
        diff_spans=[],
        should_correct=True,
        gating_reason="stable_diff_detected",
    )
    llm = CorrectionResponse(
        reply="He probably doesn't like the coffee.",
        should_show_correction=True,
        corrections=[],
        faithful_text=faithful.text,
        intended_text=intended.text,
        naturalness_score=82,
        mode="accuracy",
        meta={"decision_reason": "mock"},
    )
    return PipelineResult(
        audio_path=audio_path,
        faithful_asr=faithful,
        intended_asr=intended,
        fusion=fusion,
        llm=llm,
        tts_output=None,
    )


def _build_wav_bytes() -> bytes:
    """Builds a tiny valid wav payload so the upload route receives realistic audio input."""

    with NamedTemporaryFile(suffix=".wav") as temp_file:
        with wave.open(temp_file.name, "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(b"\x00\x00" * 1600)
        return Path(temp_file.name).read_bytes()


def test_index_page_is_served() -> None:
    """Ensures the demo recording UI is reachable from the root path."""

    response = client.get("/")

    assert response.status_code == 200
    assert "HonestEar 录音验证原型" in response.text


def test_process_upload_returns_pipeline_result(monkeypatch) -> None:
    """Ensures uploaded wav audio is accepted and transformed into structured output."""

    def _fake_run_pipeline(audio_path: Path, mode: str, speak_reply: bool, settings):
        """Replaces runtime inference with a deterministic fixture."""

        assert mode == "accuracy"
        assert speak_reply is False
        return _build_pipeline_result(audio_path)

    monkeypatch.setattr("honest_ear.api.run_pipeline", _fake_run_pipeline)

    response = client.post(
        "/v1/process-upload",
        files={"audio": ("recording.wav", _build_wav_bytes(), "audio/wav")},
        data={"mode": "accuracy", "speak_reply": "false"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["faithful_asr"]["text"] == "he dont like coffee"
    assert body["intended_asr"]["text"] == "he doesn't like coffee"
    assert body["llm"]["reply"] == "He probably doesn't like the coffee."

