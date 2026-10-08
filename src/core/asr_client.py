"""云端语音识别（ASR）：用于判定演唱语种并获取歌词文本

采用 OpenAI 兼容的 /audio/transcriptions 接口（默认 SiliconFlow SenseVoice）。
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import requests

from .settings import Settings

_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_ASR_TIMEOUT_PAD = 30

LANG_ZH = "zh"          # 含中文发音
LANG_EN = "en"          # 纯英文
LANG_UNKNOWN = "unknown"  # 无人声或转写失败


class AsrError(RuntimeError):
    pass


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str          # zh / en / unknown
    ok: bool = True
    error: str | None = None

    @property
    def has_vocal(self) -> bool:
        return bool(self.text.strip())


def _probe_duration(path: Path) -> float:
    try:
        proc = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return float(proc.stdout.decode("utf-8", "replace").strip() or 0.0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def _extract_sample(path: Path, start: float, seconds: int) -> bytes:
    """截取一段音频（转 mp3 压缩）用于上传，控制流量与耗时。"""
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error",
        "-ss", f"{max(0.0, start):.2f}", "-t", str(seconds),
        "-i", str(path),
        "-ac", "1", "-ar", "16000", "-b:a", "64k",
        "-f", "mp3", "-",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=120,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise AsrError(f"音频截取失败: {e}") from e
    if proc.returncode != 0 or not proc.stdout:
        raise AsrError("音频截取失败（ffmpeg 无输出）")
    return proc.stdout


def detect_language(text: str) -> str:
    """依据转写文本判定语种。"""
    if _CJK_RE.search(text):
        return LANG_ZH
    if _LATIN_RE.search(text):
        return LANG_EN
    return LANG_UNKNOWN


def probe_connection(settings: Settings) -> Transcript:
    """用 1 秒 440Hz 正弦音探测 ASR 连通性；只验证 Key/URL/模型可达。"""
    import math
    import struct

    sr = 16000
    pcm = b"".join(
        struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / sr)))
        for i in range(sr)
    )
    url = f"{settings.asr_base_url.rstrip('/')}/audio/transcriptions"
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {settings.asr_api_key}"},
            files={"file": ("probe.wav", _wav_bytes(pcm, sr), "audio/wav")},
            data={"model": settings.asr_model},
            timeout=settings.timeout_sec + _ASR_TIMEOUT_PAD,
        )
    except requests.RequestException as e:
        return Transcript(text="", language=LANG_UNKNOWN, ok=False, error=f"网络失败: {e}")

    if resp.status_code == 401:
        return Transcript(text="", language=LANG_UNKNOWN, ok=False, error="Key 无效")
    if resp.status_code >= 400:
        return Transcript(
            text="", language=LANG_UNKNOWN, ok=False,
            error=f"返回 {resp.status_code}: {resp.text[:150]}",
        )
    try:
        resp.json()
    except ValueError:
        return Transcript(text="", language=LANG_UNKNOWN, ok=False, error="响应非 JSON")
    return Transcript(text="", language=LANG_ZH, ok=True)


def _wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """把 16bit 单声道 PCM 包成 WAV 容器。"""
    import struct

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(pcm), b"WAVE", b"fmt ", 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16, b"data", len(pcm),
    )
    return header + pcm


def transcribe(path: Path, settings: Settings) -> Transcript:
    """转写音频片段；失败时返回 ok=False 的未知语种结果（不抛异常）。"""
    if not settings.asr_enabled:
        return Transcript(text="", language=LANG_UNKNOWN, ok=False, error="ASR 未启用")
    if not settings.asr_api_key.strip():
        return Transcript(text="", language=LANG_UNKNOWN, ok=False, error="未配置 ASR Key")

    duration = _probe_duration(path)
    start = min(30.0, duration * 0.2) if duration > 0 else 0.0
    try:
        audio = _extract_sample(path, start, settings.asr_sample_sec)
    except AsrError as e:
        return Transcript(text="", language=LANG_UNKNOWN, ok=False, error=str(e))

    url = f"{settings.asr_base_url.rstrip('/')}/audio/transcriptions"
    try:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {settings.asr_api_key}"},
            files={"file": ("sample.mp3", audio, "audio/mpeg")},
            data={"model": settings.asr_model},
            timeout=settings.timeout_sec + _ASR_TIMEOUT_PAD,
        )
    except requests.RequestException as e:
        return Transcript(text="", language=LANG_UNKNOWN, ok=False, error=f"ASR 网络失败: {e}")

    if resp.status_code == 401:
        return Transcript(text="", language=LANG_UNKNOWN, ok=False, error="ASR Key 无效")
    if resp.status_code >= 400:
        return Transcript(
            text="", language=LANG_UNKNOWN, ok=False,
            error=f"ASR 返回 {resp.status_code}: {resp.text[:150]}",
        )
    try:
        text = str(resp.json().get("text", "")).strip()
    except ValueError:
        return Transcript(text="", language=LANG_UNKNOWN, ok=False, error="ASR 响应非 JSON")

    return Transcript(text=text, language=detect_language(text), ok=True)


def sample_to_tempfile(path: Path, start: float, seconds: int) -> Path:
    """调试辅助：截取片段写入临时文件。"""
    data = _extract_sample(path, start, seconds)
    tmp = Path(tempfile.mkstemp(suffix=".mp3")[1])
    tmp.write_bytes(data)
    return tmp
