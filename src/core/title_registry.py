"""全局歌名库：保证跨批次、跨目录、跨会话不重名"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .settings import settings_dir

_REGISTRY_FILE = "used_titles.json"
_PROMPT_SAMPLE = 20   # 传给模型避让的最近歌名数量（过多会使模型模仿旧名风格，抑制新意）


def registry_path() -> Path:
    return settings_dir() / _REGISTRY_FILE


def normalize(title: str) -> str:
    """归一化用于比较：全角转半角、去空白与标点、转小写。"""
    text = unicodedata.normalize("NFKC", title).strip().lower()
    return re.sub(r"[\s\W_]+", "", text, flags=re.UNICODE)


@dataclass(frozen=True)
class TitleRegistry:
    """不可变的歌名库视图；写入返回新实例。"""
    titles: tuple[str, ...] = ()

    @property
    def keys(self) -> frozenset[str]:
        return frozenset(normalize(t) for t in self.titles)

    def contains(self, title: str) -> bool:
        return normalize(title) in self.keys

    def recent(self, count: int = _PROMPT_SAMPLE) -> tuple[str, ...]:
        """最近使用的歌名（用于提示模型避让）。"""
        return self.titles[-count:]

    def with_titles(self, new_titles: tuple[str, ...]) -> "TitleRegistry":
        existed = set(self.keys)
        merged = list(self.titles)
        for t in new_titles:
            key = normalize(t)
            if key and key not in existed:
                merged.append(t)
                existed.add(key)
        return TitleRegistry(titles=tuple(merged))


def load_registry() -> TitleRegistry:
    path = registry_path()
    if not path.exists():
        return TitleRegistry()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return TitleRegistry()
    titles = raw.get("titles") if isinstance(raw, dict) else raw
    if not isinstance(titles, list):
        return TitleRegistry()
    clean = tuple(str(t).strip() for t in titles if str(t).strip())
    return TitleRegistry(titles=clean)


def save_registry(registry: TitleRegistry) -> None:
    d = settings_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = registry_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps({"titles": list(registry.titles)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def register_titles(registry: TitleRegistry, titles: tuple[str, ...]) -> TitleRegistry:
    """登记歌名并持久化，返回更新后的库。"""
    updated = registry.with_titles(titles)
    save_registry(updated)
    return updated
