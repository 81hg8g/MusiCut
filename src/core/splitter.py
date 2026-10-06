"""切分执行器"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .analyzer import FileAnalysis, TrackSegment, short_title_for_file
from .ffmpeg_ops import split_track


@dataclass(frozen=True)
class SplitTask:
    src_path: Path
    dst_path: Path
    segment: TrackSegment


@dataclass(frozen=True)
class SplitResult:
    task: SplitTask
    success: bool
    error: str | None = None


def plan_output_paths(
    analyses: list[FileAnalysis],
    output_dir: Path,
) -> list[SplitTask]:
    """根据分析结果规划所有切分任务的输出路径。

    输出文件名：`{源文件短标题} - NN.mp3`，编号两位数。
    文件名冲突时自动追加后缀 `_2`, `_3`...
    """
    tasks: list[SplitTask] = []
    used: set[str] = set()
    for fa in analyses:
        if fa.error is not None:
            continue
        prefix = short_title_for_file(fa.src_path)
        for seg in fa.segments:
            base_name = f"{prefix} - {seg.index:02d}.mp3"
            candidate = base_name
            suffix_n = 2
            while candidate in used:
                stem = base_name[:-4]
                candidate = f"{stem}_{suffix_n}.mp3"
                suffix_n += 1
            used.add(candidate)
            tasks.append(SplitTask(
                src_path=fa.src_path,
                dst_path=output_dir / candidate,
                segment=seg,
            ))
    return tasks


def execute_task(task: SplitTask, timeout: int = 300) -> SplitResult:
    """执行单个切分任务。"""
    try:
        split_track(
            src=task.src_path,
            dst=task.dst_path,
            start=task.segment.start,
            end=task.segment.end,
            timeout=timeout,
        )
        return SplitResult(task=task, success=True)
    except Exception as e:
        return SplitResult(task=task, success=False, error=str(e))
