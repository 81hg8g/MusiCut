"""基于云端大模型（DeepSeek，OpenAI 兼容）为音频起名"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

import requests

from .settings import Settings

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_MAX_LYRICS_CHARS = 1200

_SYSTEM_ZH = (
    "你是一位资深音乐编辑与作词人。任务：为一段音频起一个凝练、有画面感的歌名。\n"
    "规则：\n"
    "1) 若提供了歌词/演唱文本，优先依据其主题、意象与情绪命名；\n"
    "2) 若没有歌词，则依据音频的曲风与氛围描述来命名；\n"
    "3) 语言要求以用户说明为准，务必严格遵守；\n"
    "4) 歌名需简洁（中文 2~10 字；英文 2~5 个单词），不含标点、引号，"
    "不含“歌名”“Untitled”“Song”等字样；\n"
    "5) 不得与用户列出的已用歌名重复或高度近似；\n"
    "6) 只输出 JSON 对象，格式为 {\"title\": \"...\", \"reason\": \"...\"}，"
    "reason 用一句话说明命名依据。"
)

_LANG_RULE_ZH = "这首歌包含中文演唱。歌名以中文为主，也可以使用英文。"
_LANG_RULE_EN = "这首歌为纯英文演唱。歌名必须全部使用英文，不得出现任何中文汉字。"
_LANG_RULE_ZH_ONLY = "未检测到演唱信息。请使用中文命名。"
_LANG_RULE_EN_ONLY = "未检测到演唱信息。请使用英文命名。"


class AiNamerError(RuntimeError):
    pass


@dataclass(frozen=True)
class NamingResult:
    title: str
    reason: str


@dataclass(frozen=True)
class NamingInput:
    """单曲起名所需信息。"""
    file_name: str
    duration_sec: float
    lyrics: str = ""
    lyrics_source: str = "none"
    features_desc: str = ""
    meta_hint: str = ""
    detected_language: str = "unknown"   # zh / en / unknown


def suggest_title(
    settings: Settings,
    data: NamingInput,
    avoid: Sequence[str] = (),
) -> NamingResult:
    """调用大模型为单曲起名。avoid 为需避让的已用歌名。"""
    if not settings.is_configured:
        raise AiNamerError("未配置 API Key，请先在设置中填写")

    payload = {
        "model": settings.model,
        "messages": _build_messages(settings, data, avoid),
        "temperature": 1.0,
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    try:
        resp = requests.post(
            f"{settings.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=settings.timeout_sec,
        )
    except requests.RequestException as e:
        raise AiNamerError(f"网络请求失败: {e}") from e

    if resp.status_code == 401:
        raise AiNamerError("API Key 无效或已过期")
    if resp.status_code == 429:
        raise AiNamerError("请求过于频繁，请降低并发或稍后重试")
    if resp.status_code >= 400:
        raise AiNamerError(f"接口返回 {resp.status_code}: {resp.text[:200]}")

    return _parse_response(resp)


def _build_messages(
    settings: Settings,
    data: NamingInput,
    avoid: Sequence[str] = (),
) -> list[dict]:
    lines = [
        f"【语言要求】{_language_rule(settings, data.detected_language)}",
        f"原文件名：{data.file_name}",
        f"时长：{_fmt_dur(data.duration_sec)}",
    ]
    if data.meta_hint:
        lines.append(f"元数据：{data.meta_hint}")

    if data.lyrics.strip():
        lyric_text = data.lyrics.strip()[:_MAX_LYRICS_CHARS]
        lines.append(f"歌词/演唱文本（来源：{data.lyrics_source}）：\n{lyric_text}")
    else:
        lines.append("歌词：无（请依据音频特征推断曲风与氛围来命名）")

    if data.features_desc:
        lines.append(f"音频特征：{data.features_desc}")

    if avoid:
        lines.append("以下歌名已被使用，请务必避开，不得重复或高度近似：\n"
                     + "、".join(avoid))

    lines.append("请输出 JSON。")
    return [
        {"role": "system", "content": _SYSTEM_ZH},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _language_rule(settings: Settings, detected: str) -> str:
    """按检测语种给出命名语言约束。"""
    if detected == "zh":
        return _LANG_RULE_ZH
    if detected == "en":
        return _LANG_RULE_EN
    return _LANG_RULE_ZH_ONLY if settings.language == "zh" else _LANG_RULE_EN_ONLY


def _parse_response(resp: requests.Response) -> NamingResult:
    try:
        body = resp.json()
    except ValueError as e:
        raise AiNamerError(f"响应非 JSON: {resp.text[:200]}") from e

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise AiNamerError(f"响应结构异常: {str(body)[:200]}") from e

    data = _extract_json(content)
    title = str(data.get("title", "")).strip()
    if not title:
        raise AiNamerError(f"模型未返回歌名: {content[:200]}")
    return NamingResult(title=_sanitize_title(title), reason=str(data.get("reason", "")).strip())


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    # 兜底：整段文本当作歌名
    return {"title": text.splitlines()[0] if text else "", "reason": ""}


def _sanitize_title(title: str) -> str:
    """去除引号、换行与文件系统非法字符。"""
    title = title.strip().strip('"').strip("'").strip("《》").strip()
    title = title.splitlines()[0].strip() if title else ""
    for ch in '\\/:*?"<>|':
        title = title.replace(ch, "")
    return title.strip()[:50]


def _fmt_dur(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}分{s}秒"
