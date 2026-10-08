"""批量重命名执行器（原地重命名 + CSV 回滚日志）"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .tag_writer import TagWriteError, write_title

_LOG_HEADER = ["时间", "目录", "原文件名", "新文件名", "标签", "结果", "说明"]


@dataclass(frozen=True)
class RenamePlan:
    src: Path
    new_stem: str

    @property
    def dst(self) -> Path:
        return self.src.with_name(f"{self.new_stem}{self.src.suffix}")


@dataclass(frozen=True)
class RenameResult:
    src: Path
    dst: Path
    success: bool
    skipped: bool = False
    error: str | None = None
    tagged: bool = False
    tag_error: str | None = None


def sanitize_stem(stem: str, max_len: int = 60) -> str:
    """净化文件名主体：去非法字符、首尾空白与点。"""
    cleaned = stem.strip()
    for ch in '\\/:*?"<>|\n\r\t':
        cleaned = cleaned.replace(ch, "")
    cleaned = cleaned.strip().strip(".")
    return cleaned[:max_len] or "未命名"


def plan_renames(items: list[tuple[Path, str]]) -> list[RenamePlan]:
    """规划重命名：净化 + 目录内去重（重名追加 _2、_3）。"""
    used: set[str] = set()
    plans: list[RenamePlan] = []
    for src, raw_title in items:
        stem = sanitize_stem(raw_title)
        candidate = stem
        n = 2
        key = candidate.lower()
        while key in used or _target_exists_excluding(src, candidate):
            candidate = f"{stem}_{n}"
            key = candidate.lower()
            n += 1
        used.add(key)
        plans.append(RenamePlan(src=src, new_stem=candidate))
    return plans


def _target_exists_excluding(src: Path, stem: str) -> bool:
    target = src.with_name(f"{stem}{src.suffix}")
    return target.exists() and target != src


def execute_renames(
    plans: list[RenamePlan],
    log_path: Path | None = None,
    dry_run: bool = False,
    write_metadata: bool = False,
) -> list[RenameResult]:
    """执行原地重命名（可选写入 ID3 标题），并写 CSV 日志便于回滚。"""
    results: list[RenameResult] = []
    for plan in plans:
        results.append(_rename_one(plan, dry_run=dry_run, write_metadata=write_metadata))
    if log_path is not None:
        write_log(log_path, results)
    return results


def _rename_one(plan: RenamePlan, dry_run: bool, write_metadata: bool) -> RenameResult:
    src, dst = plan.src, plan.dst
    if not src.exists():
        return RenameResult(src, dst, success=False, error="源文件不存在")
    if dst.exists() and src != dst:
        return RenameResult(src, dst, success=False, error="目标文件已存在")
    if dry_run:
        return RenameResult(src, dst, success=True, skipped=True, error="试运行")

    tagged, tag_error = False, None
    if write_metadata:
        try:
            write_title(src, plan.new_stem)
            tagged = True
        except TagWriteError as e:
            tag_error = str(e)

    if src == dst:
        return RenameResult(src, dst, success=True, skipped=True,
                            error="名称未变化", tagged=tagged, tag_error=tag_error)
    try:
        src.rename(dst)
    except OSError as e:
        return RenameResult(src, dst, success=False, error=f"重命名失败: {e}",
                            tagged=tagged, tag_error=tag_error)
    return RenameResult(src, dst, success=True, tagged=tagged, tag_error=tag_error)


def default_log_path(directory: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"重命名记录_{stamp}.csv"


def write_log(log_path: Path, results: list[RenameResult]) -> None:
    """写 CSV 日志，UTF-8-SIG 便于 Excel 打开。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(_LOG_HEADER)
        for r in results:
            status = "成功" if r.success else "失败"
            if r.skipped:
                status = "跳过"
            if r.tag_error:
                tag = f"失败({r.tag_error[:40]})"
            elif r.tagged:
                tag = "已写入"
            else:
                tag = "-"
            writer.writerow([
                now,
                str(r.src.parent),
                r.src.name,
                r.dst.name,
                tag,
                status,
                r.error or "",
            ])


def rollback_from_log(log_path: Path) -> tuple[int, int]:
    """依据日志回滚重命名，返回 (成功数, 失败数)。"""
    if not log_path.exists():
        raise FileNotFoundError(f"日志不存在: {log_path}")

    ok = fail = 0
    with log_path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if row.get("结果") != "成功":
                continue
            directory = Path(row["目录"])
            current = directory / row["新文件名"]
            original = directory / row["原文件名"]
            if not current.exists() or original.exists():
                fail += 1
                continue
            try:
                current.rename(original)
                ok += 1
            except OSError:
                fail += 1
    return ok, fail
