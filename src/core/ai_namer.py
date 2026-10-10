"""基于云端大模型（DeepSeek，OpenAI 兼容）为音频起名"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

import requests

from .settings import Settings

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_MAX_LYRICS_CHARS = 1200

_SYSTEM_ZH = (
    "你是一位独立音乐厂牌的资深 A&R，为氛围、后摇、爵士、Chill/Lofi 等曲风命名。\n"
    "好歌名的标准：有画面、有故事、有情绪，读起来像一个真实存在的歌名。\n"
    "正例：Air Forgiven Gently（风轻恕）、Bread Past the Bakery（余香过巷）、"
    "Dust Learns the Ledger（尘识账）、Light Learns to Wait（光学会等待）、"
    "Greenness Undertows（绿意暗涌）；\n"
    "反例：Basalt Learns to Linger（玄武岩习留连）、Cork Absorbs the Hour（软木吞时）"
    "——把生僻材质词当主语、中文硬凑文言，都是错的。\n"
    "任务：为一段音频起一个凝练、有画面感且不生硬的歌名。\n"
    "规则：\n"
    "1) 若提供了歌词/演唱文本，优先依据其主题、意象与情绪命名；\n"
    "2) 若没有歌词，则依据原文件名透露的场景，以及音频的曲风与氛围描述来命名；\n"
    "3) 语言要求以用户说明为准，务必严格遵守；\n"
    "4) 歌名需简洁：中文不超过 8 个字，英文不超过 5 个单词；"
    "不含标点、引号，不含“歌名”“Untitled”“Song”等字样；\n"
    "5) 允许直接从演唱文本中摘取一句凝练的短语作为歌名"
    "（例如副歌里最有画面感的一小句），但必须是简短短语，"
    "不得照抄整句长句或连续多个分句；\n"
    "6) 歌名的中心词优先用可感知的情绪、自然现象、时间与光、身体动作、"
    "日常场景；避开已被用滥的现成词组（如 时光、永远、孤独、寂寞；"
    "forever、alone、soul），但 光、尘、风、雾、雨、暮、霜、影 等"
    "元素性单字是好的，可以用；\n"
    "7) 严禁把具体材质、器物或工业品名词当作歌名的中心词或主语"
    "（如 玄武岩、软木、黄铜、石墨、羊毛、蜡、丹宁、树脂、糖蜜、羊皮纸、"
    "板岩、淤泥、沉积物、棉絮 等）——这类词让听者无法入画；"
    "若确实想表达其质感，只能用感受（如 玄武岩→冷硬、永恒；软木→轻、软、缓）；\n"
    "8) 不要与已用歌名做近义词替换或词序调换，必须是新的意象；\n"
    "9) 不得与用户列出的已用歌名重复或高度近似；\n"
    "10) 若最终歌名为英文（不含任何汉字），必须额外给出对应中文译名并填入 title_zh："
    "译名要“信达雅”——忠实原意（信）、顺畅自然（达）、凝练有韵味（雅），"
    "如 Song of Bamboo → 竹之曲，不得逐字硬译，"
    "更不得生造文言词（如“习留连”“吞时”“承拍”这类生硬组合）；"
    "译名用词须准确，不得有歧义或贬义（如 soft/drift 不可译作“轻浮”）；"
    "若歌名为中文，title_zh 填空字符串；\n"
    "11) 只输出 JSON 对象，格式为 "
    "{\"title\": \"...\", \"title_zh\": \"...\", \"reason\": \"...\"}，"
    "reason 用一句话说明命名依据。"
)

_LANG_RULE_ZH = "这首歌包含中文演唱。title 以中文为主，也可以使用英文；title_zh 留空。"
_LANG_RULE_EN = (
    "这首歌为纯英文演唱。title 必须全部使用英文，不得出现任何中文汉字；"
    "中文译名只能写在 title_zh。"
)
_LANG_RULE_ZH_ONLY = "未检测到演唱信息。请使用中文命名，title_zh 留空。"
_LANG_RULE_EN_ONLY = "未检测到演唱信息。请使用英文命名，并在 title_zh 给出中文译名。"


class AiNamerError(RuntimeError):
    pass


class AiContentError(AiNamerError):
    """模型输出内容畸形（瞬时错误，可重试）。"""


@dataclass(frozen=True)
class NamingResult:
    title: str
    reason: str
    title_zh: str = ""   # 英文歌名的中文译名；中文歌名时为空


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
    temperature: float = 1.0,
) -> NamingResult:
    """调用大模型为单曲起名。avoid 为需避让的已用歌名。

    temperature 控制采样温度：高温易触发 response_format 回声等畸形输出，
    重试末轮可降至 0.3。
    """
    if not settings.is_configured:
        raise AiNamerError("未配置 API Key，请先在设置中填写")

    payload = {
        "model": settings.model,
        "messages": _build_messages(settings, data, avoid),
        "temperature": temperature,
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
    # title 键缺失（如回声 {"type": "json_object"}）或为空白，均属瞬时内容畸形
    title = str(data.get("title", "")).strip()
    if not title:
        raise AiContentError(f"模型未返回歌名: {content[:200]}")
    title = _sanitize_title(title)
    _validate_title(title)

    title_zh = _sanitize_title(str(data.get("title_zh", "")).strip())
    if title_zh:
        _validate_title(title_zh)
    # 英文歌名必须带中文译名，缺失视为瞬时内容畸形以便重试
    if not has_chinese(title) and not title_zh:
        raise AiContentError(f"模型未返回歌名的中文译名: {content[:200]}")

    return NamingResult(
        title=title, reason=str(data.get("reason", "")).strip(), title_zh=title_zh
    )


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
    # 接口已强制 response_format=json_object；纯文本/畸形输出视为内容错误
    raise AiContentError("模型未返回合法 JSON")


def _sanitize_title(title: str) -> str:
    """去除引号、换行与文件系统非法字符。"""
    title = title.strip().strip('"').strip("'").strip("《》").strip()
    title = title.splitlines()[0].strip() if title else ""
    for ch in '\\/:*?"<>|':
        title = title.replace(ch, "")
    return title.strip()[:50]


def _validate_title(title: str) -> None:
    """拦截 response_format 回声（json_object / JSON）与残留花括号等畸形标题。"""
    if "{" in title or "}" in title:
        raise AiContentError("歌名包含非法字符: 花括号")
    lowered = title.strip().lower()
    if "json_object" in lowered or lowered == "json":
        raise AiContentError("模型回显了 JSON 模式标记")


def _fmt_dur(sec: float) -> str:
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m}分{s}秒"


def has_chinese(text: str) -> bool:
    """是否含汉字（用于判定歌名语言）。"""
    return bool(_CJK_RE.search(text))


def compose_title(title: str, title_zh: str) -> str:
    """英文歌名拼接中文译名。

    ('Song of Bamboo', '竹之曲') → 'Song of Bamboo 竹之曲'
    中文歌名或译名为空时原样返回。
    """
    zh = title_zh.strip()
    if not zh or has_chinese(title):
        return title
    return f"{title} {zh}"
