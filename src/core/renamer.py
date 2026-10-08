"""批量重命名执行器（原地重命名 + CSV 回滚日志）"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..utils.paths import app_dir
from .tag_writer import TagWriteError, write_tags

_LOG_HEADER = ["时间", "目录", "原文件名", "新文件名", "标签", "结果", "说明"]
_LOG_NAME = "重命名记录.csv"


@dataclass(frozen=True)
class RenamePlan:
    src: Path
    new_stem: str
    cover: Path | None = None

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
    covered: bool = False
    cover_error: str | None = None


def sanitize_stem(stem: str, max_len: int = 60) -> str:
    """净化文件名主体：去非法字符、首尾空白与点。"""
    cleaned = stem.strip()
    for ch in '\\/:*?"<>|\n\r\t':
        cleaned = cleaned.replace(ch, "")
    cleaned = cleaned.strip().strip(".")
    return cleaned[:max_len] or "未命名"


def plan_renames(
    items: list[tuple[Path, str, Path | None]],
) -> list[RenamePlan]:
    """规划重命名：净化 + 目录内去重（重名追加 _2、_3）。"""
    used: set[str] = set()
    plans: list[RenamePlan] = []
    for src, raw_title, cover in items:
        stem = sanitize_stem(raw_title)
        candidate = stem
        n = 2
        key = candidate.lower()
        while key in used or _target_exists_excluding(src, candidate):
            candidate = f"{stem}_{n}"
            key = candidate.lower()
            n += 1
        used.add(key)
        plans.append(RenamePlan(src=src, new_stem=candidate, cover=cover))
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
    covered, cover_error = False, None
    if write_metadata:
        try:
            write_tags(src, plan.new_stem, plan.cover)
            tagged = True
            if plan.cover is not None:
                covered = True
        except TagWriteError as e:
            tag_error = str(e)
            if plan.cover is not None:
                cover_error = str(e)

    if src == dst:
        return RenameResult(src, dst, success=True, skipped=True,
                            error="名称未变化", tagged=tagged, tag_error=tag_error,
                            covered=covered, cover_error=cover_error)
    try:
        src.rename(dst)
    except OSError as e:
        return RenameResult(src, dst, success=False, error=f"重命名失败: {e}",
                            tagged=tagged, tag_error=tag_error,
                            covered=covered, cover_error=cover_error)
    return RenameResult(src, dst, success=True, tagged=tagged, tag_error=tag_error,
                        covered=covered, cover_error=cover_error)


def default_log_path() -> Path:
    """记录文件固定放在软件所在目录，单一文件累加历史。"""
    return app_dir() / _LOG_NAME


def write_log(log_path: Path, results: list[RenameResult]) -> None:
    """追加写 CSV 日志，UTF-8-SIG 便于 Excel 打开。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_new = not log_path.exists() or log_path.stat().st_size == 0
    encoding = "utf-8-sig" if is_new else "utf-8"
    with log_path.open("a", newline="", encoding=encoding) as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(_LOG_HEADER)
        for r in results:
            status = "成功" if r.success else "失败"
            if r.skipped:
                status = "跳过"
            has_cover = r.covered or r.cover_error is not None
            tag = _tag_label(r, has_cover)
            writer.writerow([
                now,
                str(r.src.parent),
                r.src.name,
                r.dst.name,
                tag,
                status,
                r.error or "",
            ])


def _tag_label(result: RenameResult, has_cover: bool) -> str:
    """组装"标签"列文本：标题部分 + 可选封面部分；未写入则 "-"。"""
    parts: list[str] = []
    if result.tagged:
        parts.append("标题已写入")
    elif result.tag_error:
        parts.append(f"标题失败({result.tag_error[:30]})")
    if has_cover:
        if result.covered:
            parts.append("封面已写入")
        elif result.cover_error:
            parts.append(f"封面失败({result.cover_error[:30]})")
    return "；".join(parts) or "-"


def rollback_from_log(log_path: Path) -> tuple[int, int]:
    """回滚日志中最近一批重命名，返回 (成功数, 失败数)。"""
    if not log_path.exists():
        raise FileNotFoundError(f"日志不存在: {log_path}")

    with log_path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0, 0

    last_batch = rows[-1].get("时间", "")
    ok = fail = 0
    for row in rows:
        if row.get("时间") != last_batch or row.get("结果") != "成功":
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
