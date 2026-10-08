"""本地配置读写（API Key 等，保存在用户目录，不入版本库）"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

_APP_DIR_NAME = "MusiCut"
_SETTINGS_FILE = "settings.json"

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_ASR_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_ASR_MODEL = "FunAudioLLM/SenseVoiceSmall"


def settings_dir() -> Path:
    """用户配置目录：%APPDATA%\\MusiCut（Windows）。"""
    base = os.environ.get("APPDATA")
    if base:
        return Path(base) / _APP_DIR_NAME
    return Path.home() / f".{_APP_DIR_NAME.lower()}"


def settings_path() -> Path:
    return settings_dir() / _SETTINGS_FILE


@dataclass(frozen=True)
class Settings:
    """应用配置（不可变）。"""
    # 大模型（起名）
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_sec: int = 60
    concurrency: int = 4
    language: str = "zh"   # 无人声时的起名语言：zh / en

    # 语音识别（语种判定）
    asr_enabled: bool = True
    asr_base_url: str = DEFAULT_ASR_BASE_URL
    asr_api_key: str = ""
    asr_model: str = DEFAULT_ASR_MODEL
    asr_sample_sec: int = 90

    # 输出行为
    write_metadata: bool = True   # 将歌名写入 ID3 Title

    def with_updates(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key.strip()) and bool(self.base_url.strip())

    @property
    def asr_configured(self) -> bool:
        return bool(self.asr_api_key.strip()) and bool(self.asr_base_url.strip())


def load_settings() -> Settings:
    """读取配置；文件不存在或损坏时返回默认配置。"""
    path = settings_path()
    if not path.exists():
        return Settings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return Settings()
    if not isinstance(raw, dict):
        return Settings()
    allowed = {f for f in Settings.__dataclass_fields__}
    clean = {k: v for k, v in raw.items() if k in allowed}
    try:
        return Settings(**clean)
    except TypeError:
        return Settings()


def save_settings(settings: Settings) -> None:
    """保存配置（目录不存在则创建）。"""
    d = settings_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = settings_path()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(settings.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
