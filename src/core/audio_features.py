"""轻量音频特征分析（基于 numpy，不依赖 librosa）

从音频提取客观特征并转成中文描述，供大模型推断曲风与氛围。
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_SR = 16000        # 采样率
_FRAME = 2048      # 帧长
_HOP = 1024        # 帧移
_TIMEOUT = 180
_EPS = 1e-9

_LOW_HZ = 250.0
_HIGH_HZ = 2000.0


class AudioFeatureError(RuntimeError):
    pass


@dataclass(frozen=True)
class AudioFeatures:
    duration_sec: float
    loudness_db: float
    dynamic_range_db: float
    tempo_bpm: float
    percussiveness: float      # 0~1
    brightness_hz: float       # 频谱质心均值
    band_low: float            # 低频能量占比
    band_mid: float
    band_high: float
    zcr: float                 # 过零率
    onset_density: float       # 每秒起音数

    def describe(self) -> str:
        """转换为中文描述，供 prompt 使用。"""
        parts = [
            f"时长 {self._fmt_dur()}",
            f"响度{self._loud()}({self.loudness_db:.0f} dB)",
            f"动态范围{self._dyn()}({self.dynamic_range_db:.0f} dB)",
            f"节奏约 {self.tempo_bpm:.0f} BPM（{self._tempo_word()}）",
            f"音色{self._bright_word()}(质心 {self.brightness_hz:.0f} Hz)",
            f"频谱分布：低频{self._pct(self.band_low)}、中频{self._pct(self.band_mid)}、高频{self._pct(self.band_high)}",
            f"打击感{self._perc_word()}",
            f"起音密度 {self.onset_density:.1f} 次/秒",
        ]
        return "；".join(parts)

    def to_dict(self) -> dict:
        return {
            "duration_sec": round(self.duration_sec, 1),
            "loudness_db": round(self.loudness_db, 1),
            "dynamic_range_db": round(self.dynamic_range_db, 1),
            "tempo_bpm": round(self.tempo_bpm, 1),
            "percussiveness": round(self.percussiveness, 2),
            "brightness_hz": round(self.brightness_hz, 0),
            "band_low": round(self.band_low, 3),
            "band_mid": round(self.band_mid, 3),
            "band_high": round(self.band_high, 3),
            "zcr": round(self.zcr, 4),
            "onset_density": round(self.onset_density, 2),
        }

    # --- 描述词 ---
    def _fmt_dur(self) -> str:
        m = int(self.duration_sec // 60)
        s = int(self.duration_sec % 60)
        return f"{m}分{s}秒"

    def _loud(self) -> str:
        if self.loudness_db > -14:
            return "偏响"
        if self.loudness_db > -22:
            return "中等"
        return "偏轻"

    def _dyn(self) -> str:
        if self.dynamic_range_db > 14:
            return "很大"
        if self.dynamic_range_db > 8:
            return "较大"
        if self.dynamic_range_db > 4:
            return "适中"
        return "较小"

    def _tempo_word(self) -> str:
        b = self.tempo_bpm
        if b < 70:
            return "缓慢抒情"
        if b < 95:
            return "中速"
        if b < 125:
            return "明快"
        return "快速强烈"

    def _bright_word(self) -> str:
        hz = self.brightness_hz
        if hz < 1200:
            return "偏暗沉温暖"
        if hz < 2200:
            return "适中"
        return "明亮通透"

    def _perc_word(self) -> str:
        if self.percussiveness > 0.66:
            return "强"
        if self.percussiveness > 0.33:
            return "中等"
        return "弱（偏氛围/旋律）"

    @staticmethod
    def _pct(v: float) -> str:
        return f"{v * 100:.0f}%"


def _extract_pcm(path: Path) -> np.ndarray:
    cmd = [
        "ffmpeg", "-hide_banner", "-v", "error",
        "-i", str(path), "-ac", "1", "-ar", str(_SR),
        "-f", "s16le", "-",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=_TIMEOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as e:
        raise AudioFeatureError(f"ffmpeg 调用失败: {e}") from e
    if proc.returncode != 0:
        raise AudioFeatureError(
            f"提取 PCM 失败: {proc.stderr.decode('utf-8', 'replace')[-200:]}"
        )
    data = np.frombuffer(proc.stdout, dtype=np.int16)
    if data.size < _FRAME * 2:
        raise AudioFeatureError("音频过短，无法分析")
    return data.astype(np.float32) / 32768.0


def analyze(path: Path) -> AudioFeatures:
    """分析音频文件，返回客观特征。"""
    pcm = _extract_pcm(path)
    duration = pcm.size / _SR

    win = sliding_window_view(pcm, _FRAME)[::_HOP].copy()
    rms = np.sqrt(np.mean(win ** 2, axis=1) + _EPS)

    hann = np.hanning(_FRAME).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(win * hann, axis=1))
    freqs = np.fft.rfftfreq(_FRAME, 1.0 / _SR)

    loudness_db = 20 * np.log10(float(np.mean(rms)) + _EPS)
    p95, p10 = np.percentile(rms, 95), np.percentile(rms, 10)
    dynamic_db = 20 * np.log10(p95 + _EPS) - 20 * np.log10(p10 + _EPS)

    centroid = _spectral_centroid(spectrum, freqs)
    band_low, band_mid, band_high = _band_ratios(spectrum, freqs)
    tempo, onset_density, percussive = _rhythm(spectrum)

    return AudioFeatures(
        duration_sec=duration,
        loudness_db=float(loudness_db),
        dynamic_range_db=float(max(0.0, dynamic_db)),
        tempo_bpm=tempo,
        percussiveness=percussive,
        brightness_hz=centroid,
        band_low=band_low,
        band_mid=band_mid,
        band_high=band_high,
        zcr=float(np.mean(np.abs(np.diff(np.sign(win), axis=1)) > 0)),
        onset_density=onset_density,
    )


def _spectral_centroid(spectrum: np.ndarray, freqs: np.ndarray) -> float:
    total = spectrum.sum(axis=1) + _EPS
    return float(np.mean((spectrum * freqs).sum(axis=1) / total))


def _band_ratios(spectrum: np.ndarray, freqs: np.ndarray) -> tuple[float, float, float]:
    energy = (spectrum ** 2).sum(axis=0)   # 按频率轴聚合 → shape (n_freqs,)
    total = float(energy.sum()) + _EPS
    low = float(energy[freqs < _LOW_HZ].sum())
    mid = float(energy[(freqs >= _LOW_HZ) & (freqs < _HIGH_HZ)].sum())
    high = float(energy[freqs >= _HIGH_HZ].sum())
    return low / total, mid / total, high / total


def _rhythm(spectrum: np.ndarray) -> tuple[float, float, float]:
    """基于频谱通量估计 BPM、起音密度与打击感。"""
    flux = np.sum(np.diff(spectrum, axis=0).clip(min=0), axis=1)
    if flux.size < 8:
        return 0.0, 0.0, 0.0

    fps = _SR / _HOP
    norm = flux - flux.mean()
    autocorr = np.correlate(norm, norm, mode="full")[norm.size - 1:]
    lag_min = max(1, int(60.0 / 180 * fps))
    lag_max = min(autocorr.size - 1, int(60.0 / 60 * fps))
    if lag_max <= lag_min:
        return 0.0, 0.0, 0.0
    segment = autocorr[lag_min:lag_max]
    best_lag = lag_min + int(np.argmax(segment))
    tempo = 60.0 * fps / best_lag

    threshold = flux.mean() + flux.std()
    onsets = int(np.sum(flux > threshold))
    onset_density = onsets / max(1.0, flux.size / fps)
    percussive = float(min(1.0, onset_density / 6.0))
    return float(tempo), float(onset_density), percussive
