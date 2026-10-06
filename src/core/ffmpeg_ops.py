"""ffmpeg 调用封装"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SilenceInterval:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):([\d.]+)")
_SILENCE_START_RE = re.compile(r"silence_start:\s*([\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([\d.]+)")


def run_ffmpeg(args: list[str], timeout: int = 600) -> tuple[int, str]:
    """运行 ffmpeg，返回 (returncode, stderr)。ffmpeg 的信息都输出到 stderr。"""
    cmd = ["ffmpeg", "-hide_banner", "-y"] + args
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    return proc.returncode, proc.stderr


def get_duration(path: Path, timeout: int = 60) -> float:
    """获取音频时长（秒）。失败抛异常。"""
    code, err = run_ffmpeg(["-i", str(path), "-f", "null", "-"], timeout=timeout)
    m = _DURATION_RE.search(err)
    if not m:
        raise RuntimeError(f"无法解析时长: {path.name}")
    h, mnt, sec = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mnt * 60 + sec


def detect_silences(
    path: Path,
    noise_db: float = -28.0,
    min_duration: float = 1.2,
    timeout: int = 600,
) -> tuple[float, list[SilenceInterval]]:
    """检测静音区间。返回 (总时长, 静音区间列表)。"""
    code, err = run_ffmpeg(
        [
            "-i", str(path),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
            "-f", "null", "-",
        ],
        timeout=timeout,
    )
    duration_m = _DURATION_RE.search(err)
    if not duration_m:
        raise RuntimeError(f"无法获取时长: {path.name}")
    h, mnt, sec = int(duration_m.group(1)), int(duration_m.group(2)), float(duration_m.group(3))
    total = h * 3600 + mnt * 60 + sec

    starts = [float(m.group(1)) for m in _SILENCE_START_RE.finditer(err)]
    ends = [float(m.group(1)) for m in _SILENCE_END_RE.finditer(err)]
    intervals = [SilenceInterval(start=s, end=e) for s, e in zip(starts, ends) if e > s]
    return total, intervals


def split_track(
    src: Path,
    dst: Path,
    start: float,
    end: float,
    timeout: int = 300,
) -> None:
    """直切（不重编码）切出 [start, end) 区间到 dst。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, end - start)
    if duration <= 0:
        raise ValueError(f"切分区间非法: start={start}, end={end}")
    code, err = run_ffmpeg(
        [
            "-ss", f"{start:.3f}",
            "-i", str(src),
            "-t", f"{duration:.3f}",
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(dst),
        ],
        timeout=timeout,
    )
    if code != 0:
        raise RuntimeError(f"ffmpeg 切分失败 ({dst.name}): {err[-500:]}")
