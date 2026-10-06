"""波形数据提取与多级缓存"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_PCM_SAMPLE_RATE = 8000  # 8kHz 单声道足够波形显示
_PCM_TIMEOUT = 120


class WaveformError(RuntimeError):
    pass


def _extract_pcm(path: Path, timeout: int = _PCM_TIMEOUT) -> np.ndarray:
    """用 ffmpeg 提取音频为 8kHz 16-bit 单声道 PCM。"""
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error",
        "-i", str(path),
        "-ac", "1",
        "-ar", str(_PCM_SAMPLE_RATE),
        "-f", "s16le",
        "-",
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        timeout=timeout,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    if proc.returncode != 0:
        raise WaveformError(f"ffmpeg 提取PCM失败: {proc.stderr.decode('utf-8', 'replace')[-300:]}")
    data = np.frombuffer(proc.stdout, dtype=np.int16)
    if data.size == 0:
        raise WaveformError("提取的PCM数据为空")
    return data


@dataclass
class WaveformData:
    """波形数据：保存低采样PCM，按需计算峰值。"""
    duration: float          # 秒
    sample_rate: int         # PCM 采样率
    pcm: np.ndarray          # int16 一维

    @classmethod
    def from_file(cls, path: Path, duration_hint: float = 0.0) -> "WaveformData":
        pcm = _extract_pcm(path)
        duration = pcm.size / _PCM_SAMPLE_RATE
        # 若与提示时长差距大，以提示为准（ffmpeg -c copy 文件可能时间戳不准）
        if duration_hint > 0 and abs(duration - duration_hint) > 2.0:
            duration = duration_hint
        return cls(duration=duration, sample_rate=_PCM_SAMPLE_RATE, pcm=pcm)

    def peaks(self, start_sec: float, end_sec: float, max_points: int) -> tuple[np.ndarray, np.ndarray, float]:
        """返回 (mins, maxes, bucket_sec)。区间均分为 max_points 桶，每桶取 min/max。

        mins/maxes 归一化到 [-1, 1]。
        """
        start_sec = max(0.0, start_sec)
        end_sec = min(self.duration, end_sec)
        if end_sec <= start_sec or max_points <= 0:
            return np.zeros(0), np.zeros(0), 0.0

        s0 = int(start_sec * self.sample_rate)
        s1 = min(int(end_sec * self.sample_rate), self.pcm.size)
        if s1 <= s0:
            return np.zeros(0), np.zeros(0), 0.0

        region = self.pcm[s0:s1]
        bucket_sec = (end_sec - start_sec) / max_points
        samples_per_bucket = max(1, region.size // max_points)
        n_buckets = min(max_points, region.size // samples_per_bucket + 1)
        trimmed = region[: n_buckets * samples_per_bucket]
        reshaped = trimmed.reshape(n_buckets, samples_per_bucket)
        mins = reshaped.min(axis=1).astype(np.float32) / 32768.0
        maxs = reshaped.max(axis=1).astype(np.float32) / 32768.0
        return mins, maxs, bucket_sec
