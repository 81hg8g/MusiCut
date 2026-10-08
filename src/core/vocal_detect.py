"""本地人声检测（Silero VAD，ONNX 推理）

用于在调用云端 ASR 之前，快速判断音频是否含有人声演唱，
避免对纯器乐曲目产生无谓的识别费用。
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SR = 16000
_CHUNK = 512                     # 每个推理块的新增样本数
_CONTEXT = 64                    # 16kHz 下需拼接的历史上下文样本数
_STATE_SHAPE = (2, 1, 128)
_MODEL_FILENAME = "silero_vad.onnx"
_WINDOW_SEC = 20.0               # 单个分析窗口长度
_WINDOW_COUNT = 4                # 全曲均匀取样的窗口数
_PROB_THRESHOLD = 0.5            # 单帧判定为语音的概率阈值

_lock = threading.Lock()
_session = None
_load_error: str | None = None


@dataclass(frozen=True)
class VocalVerdict:
    has_vocal: bool          # 最终结论（模型不可用时保守为 True）
    speech_ratio: float      # 语音帧占比 0~1
    analyzed_sec: float      # 实际分析的音频时长
    available: bool = True
    error: str | None = None


def model_path() -> Path:
    """定位模型文件：开发态在项目 assets/，打包后在资源目录。"""
    if getattr(sys, "frozen", False):
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base = Path(__file__).resolve().parents[2]
    return base / "assets" / _MODEL_FILENAME


def _get_session():
    """延迟加载 ONNX 会话（单例，线程安全）。"""
    global _session, _load_error
    if _session is not None or _load_error is not None:
        return _session
    with _lock:
        if _session is not None or _load_error is not None:
            return _session
        try:
            import onnxruntime as ort

            path = model_path()
            if not path.exists():
                raise FileNotFoundError(f"模型文件缺失: {path}")
            options = ort.SessionOptions()
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = 1
            options.log_severity_level = 3
            _session = ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
        except Exception as e:  # noqa: BLE001 - 推理后端缺失不应中断流程
            _load_error = f"{type(e).__name__}: {e}"
    return _session


def is_available() -> bool:
    return _get_session() is not None


def load_error() -> str | None:
    return _load_error


def _window_starts(duration: float) -> list[float]:
    """在全曲均匀取若干窗口的起点，避开最前与最后的边角。"""
    if duration <= _WINDOW_SEC:
        return [0.0]
    starts: list[float] = []
    for i in range(_WINDOW_COUNT):
        center = duration * (i + 0.5) / _WINDOW_COUNT
        start = center - _WINDOW_SEC / 2
        starts.append(max(0.0, min(start, duration - _WINDOW_SEC)))
    return starts


def _run_window(session, audio: np.ndarray) -> list[float]:
    """对单段音频逐块推理，返回每块的语音概率。

    注意：Silero VAD v5 在 16kHz 下要求输入为 576 样本
    （末尾 64 个历史上下文样本 + 512 个新样本）。
    """
    state = np.zeros(_STATE_SHAPE, dtype=np.float32)
    context = np.zeros(_CONTEXT, dtype=np.float32)
    sr = np.array(_SR, dtype=np.int64)
    probs: list[float] = []
    usable = (audio.size // _CHUNK) * _CHUNK
    for i in range(0, usable, _CHUNK):
        chunk = audio[i:i + _CHUNK].astype(np.float32)
        model_input = np.concatenate([context, chunk]).reshape(1, _CHUNK + _CONTEXT)
        out, state = session.run(None, {"input": model_input, "state": state, "sr": sr})
        context = chunk[-_CONTEXT:]
        probs.append(float(np.asarray(out).ravel()[0]))
    return probs


def detect(pcm: np.ndarray, min_ratio: float = 0.05) -> VocalVerdict:
    """检测 16kHz 单声道 PCM 是否含人声。

    min_ratio：语音帧占比达到该值即判定为“有人声”。
    """
    session = _get_session()
    if session is None:
        return VocalVerdict(
            has_vocal=True, speech_ratio=0.0, analyzed_sec=0.0,
            available=False, error=_load_error,
        )
    if pcm.size < _CHUNK * 8:
        return VocalVerdict(
            has_vocal=True, speech_ratio=0.0, analyzed_sec=0.0,
            available=True, error="音频过短，无法判定",
        )

    duration = pcm.size / _SR
    probs: list[float] = []
    for start in _window_starts(duration):
        a = int(start * _SR)
        b = min(pcm.size, a + int(_WINDOW_SEC * _SR))
        if b - a < _CHUNK * 4:
            continue
        probs.extend(_run_window(session, pcm[a:b]))

    if not probs:
        return VocalVerdict(
            has_vocal=True, speech_ratio=0.0, analyzed_sec=0.0,
            available=True, error="无有效分析窗口",
        )

    ratio = sum(1 for p in probs if p >= _PROB_THRESHOLD) / len(probs)
    return VocalVerdict(
        has_vocal=ratio >= min_ratio,
        speech_ratio=float(ratio),
        analyzed_sec=len(probs) * _CHUNK / _SR,
    )
