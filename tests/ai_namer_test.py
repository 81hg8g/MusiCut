"""ai_namer / rename_pipeline 测试：畸形 JSON（response_format 回声）不应被当成歌名。

所有网络请求均打桩，不联网。
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core import ai_namer as an
from src.core import rename_pipeline as rp
from src.core.ai_namer import (
    AiContentError,
    AiNamerError,
    NamingInput,
    NamingResult,
)
from src.core.audio_features import AudioFeatureError
from src.core.settings import Settings
from src.core.title_registry import TitleRegistry


# ---- 打桩辅助 ----

class _FakeResp:
    def __init__(self, payload=None, text: str = "", status_code: int = 200):
        self._payload = payload
        self.text = text
        self.status_code = status_code

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def _chat_resp(content: str) -> _FakeResp:
    return _FakeResp({"choices": [{"message": {"content": content}}]})


def _settings(**kwargs) -> Settings:
    params = {"api_key": "k", "asr_enabled": False}
    params.update(kwargs)
    return Settings(**params)


def _patch_prereqs(monkeypatch, feats=None):
    """打桩 naming_one 的本地依赖；feats=None 表示特征分析失败。"""
    monkeypatch.setattr(
        rp, "read_lyrics",
        lambda path: SimpleNamespace(has_lyrics=False, text="", source="none"),
    )
    if feats is None:
        def _raise(path, vocal_threshold):
            raise AudioFeatureError("boom")
        monkeypatch.setattr(rp, "analyze", _raise)
    else:
        monkeypatch.setattr(rp, "analyze", lambda path, vocal_threshold: feats)


# ---- _extract_json ----

def test_extract_json_rejects_echo_garbage():
    with pytest.raises(AiContentError):
        an._extract_json("{type json_object}")


def test_extract_json_plain_object():
    data = an._extract_json('{"title":"正常名","reason":"r"}')
    assert data["title"] == "正常名"
    assert data["reason"] == "r"


def test_extract_json_embedded_in_prose():
    text = '好的，结果如下：\n{"title": "夜曲", "reason": "副歌意象"}\n希望你喜欢'
    data = an._extract_json(text)
    assert data["title"] == "夜曲"


# ---- _parse_response ----

def test_parse_response_malformed_echo_raises():
    with pytest.raises(AiContentError):
        an._parse_response(_chat_resp('{"type": json_object}'))


def test_parse_response_braces_in_title_raises():
    with pytest.raises(AiContentError):
        an._parse_response(_chat_resp('{"title":"{x}"}'))


def test_parse_response_json_object_title_raises():
    with pytest.raises(AiContentError):
        an._parse_response(_chat_resp('{"title":"json_object","reason":""}'))


def test_parse_response_json_title_raises():
    with pytest.raises(AiContentError):
        an._parse_response(_chat_resp('{"title":"JSON","reason":""}'))


def test_parse_response_empty_title_is_content_error():
    # title 为空白：瞬时内容畸形，可重试（不再是硬错误 AiNamerError）
    with pytest.raises(AiContentError) as exc:
        an._parse_response(_chat_resp('{"title":"  ","reason":""}'))
    assert "未返回歌名" in str(exc.value)


def test_parse_response_echo_without_title_is_content_error():
    # DeepSeek 高频回声：合法 JSON 但无 title 键
    with pytest.raises(AiContentError) as exc:
        an._parse_response(_chat_resp('{"type":"json_object"}'))
    assert "未返回歌名" in str(exc.value)


def test_parse_response_ok():
    result = an._parse_response(_chat_resp('{"title":"夜曲","reason":"意象"}'))
    assert result == NamingResult(title="夜曲", reason="意象")


def test_parse_response_bad_envelope():
    with pytest.raises(AiNamerError):
        an._parse_response(_FakeResp({"nope": []}))


def test_parse_response_non_json_body():
    with pytest.raises(AiNamerError):
        an._parse_response(_FakeResp(None, text="oops"))


# ---- suggest_title（requests 打桩）----

def test_suggest_title_requires_config():
    with pytest.raises(AiNamerError):
        an.suggest_title(Settings(api_key=""), NamingInput("a.mp3", 10.0))


def test_suggest_title_success_and_payload(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return _chat_resp('{"title":"Night Drive","reason":"synth vibe"}')

    monkeypatch.setattr(an.requests, "post", fake_post)
    data = NamingInput(
        "a.mp3", 95.0, lyrics="city lights", lyrics_source="asr",
        features_desc="electronic, 120bpm", meta_hint="artist hint",
        detected_language="en",
    )
    result = an.suggest_title(_settings(language="en"), data, avoid=("旧名",))

    assert result.title == "Night Drive"
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["json"]["temperature"] == 1.0
    assert captured["json"]["messages"][0]["role"] == "system"


def test_suggest_title_passes_temperature(monkeypatch):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["json"] = json
        return _chat_resp('{"title":"Night Drive","reason":"synth vibe"}')

    monkeypatch.setattr(an.requests, "post", fake_post)
    an.suggest_title(
        _settings(), NamingInput("a.mp3", 10.0), avoid=(), temperature=0.3
    )

    assert captured["json"]["temperature"] == 0.3


def test_suggest_title_401(monkeypatch):
    monkeypatch.setattr(an.requests, "post", lambda *a, **k: _FakeResp(status_code=401, text="x"))
    with pytest.raises(AiNamerError):
        an.suggest_title(_settings(), NamingInput("a.mp3", 1.0))


def test_suggest_title_network_error(monkeypatch):
    def fake_post(*a, **k):
        raise an.requests.RequestException("dns down")

    monkeypatch.setattr(an.requests, "post", fake_post)
    with pytest.raises(AiNamerError):
        an.suggest_title(_settings(), NamingInput("a.mp3", 1.0))


# ---- naming_one：畸形内容重试 ----

def test_naming_one_retries_content_error_then_ok(tmp_path, monkeypatch):
    _patch_prereqs(monkeypatch)
    temps: list[float] = []

    def fake_suggest(settings, data, avoid=(), temperature=1.0):
        temps.append(temperature)
        if len(temps) < 3:
            # 前两次（temperature=1.0）返回回声式内容畸形
            raise AiContentError("模型未返回歌名: {\"type\": \"json_object\"}")
        # 最后一次降温 0.3：调用参数必须为 0.3
        assert temperature == 0.3
        return NamingResult(title="夜曲", reason="副歌")

    monkeypatch.setattr(rp, "suggest_title", fake_suggest)
    record = rp.naming_one(
        _settings(), tmp_path / "a.mp3",
        rp._UsedTitles(frozenset()), TitleRegistry(),
    )

    assert record.ok
    assert record.title == "夜曲"
    assert record.attempts == 3
    assert record.error is None
    assert temps == [1.0, 1.0, 0.3]


def test_naming_one_content_error_all_attempts_fail(tmp_path, monkeypatch):
    _patch_prereqs(monkeypatch)
    temps: list[float] = []

    def fake_suggest(settings, data, avoid=(), temperature=1.0):
        temps.append(temperature)
        raise AiContentError("模型回显了 JSON 模式标记")

    monkeypatch.setattr(rp, "suggest_title", fake_suggest)
    record = rp.naming_one(
        _settings(), tmp_path / "a.mp3",
        rp._UsedTitles(frozenset()), TitleRegistry(),
    )

    assert not record.ok
    assert record.attempts == 3
    assert record.error == "模型回显了 JSON 模式标记"
    assert record.lyrics_source == "none"
    assert record.features_desc == ""
    assert record.local_vocal == "未知"
    assert temps == [1.0, 1.0, 0.3]


def test_naming_one_hard_error_fails_fast(tmp_path, monkeypatch):
    _patch_prereqs(monkeypatch)

    def fake_suggest(settings, data, avoid=(), temperature=1.0):
        raise AiNamerError("API Key 无效或已过期")

    monkeypatch.setattr(rp, "suggest_title", fake_suggest)
    record = rp.naming_one(
        _settings(), tmp_path / "a.mp3",
        rp._UsedTitles(frozenset()), TitleRegistry(),
    )

    assert not record.ok
    assert record.attempts == 1
    assert record.error == "API Key 无效或已过期"


def test_naming_one_collision_falls_back_to_uniquify(tmp_path, monkeypatch):
    _patch_prereqs(monkeypatch)
    used = rp._UsedTitles(frozenset())
    assert used.try_claim("夜曲")  # 预占，模型返回的名字始终冲突

    monkeypatch.setattr(
        rp, "suggest_title",
        lambda settings, data, avoid=(), temperature=1.0:
            NamingResult(title="夜曲", reason="r"),
    )
    record = rp.naming_one(_settings(), tmp_path / "a.mp3", used, TitleRegistry())

    assert record.ok
    assert record.attempts == 3
    assert record.title != "夜曲"


def test_naming_one_asr_path_ok(tmp_path, monkeypatch):
    feats = SimpleNamespace(describe=lambda: "pop, vocal", duration_sec=12.0, has_vocal=True)
    _patch_prereqs(monkeypatch, feats=feats)
    transcript = SimpleNamespace(
        has_vocal=True, ok=True, language="zh", text="城市灯火", error="",
    )
    monkeypatch.setattr(rp, "transcribe", lambda path, settings: transcript)
    monkeypatch.setattr(
        rp, "suggest_title",
        lambda settings, data, avoid=(), temperature=1.0:
            NamingResult(title="灯火", reason="asr 歌词"),
    )
    record = rp.naming_one(
        _settings(asr_enabled=True, asr_api_key="ak"),
        tmp_path / "a.mp3", rp._UsedTitles(frozenset()), TitleRegistry(),
    )

    assert record.ok
    assert record.lyrics_source == "asr"
    assert record.detected_language == "zh"
    assert record.local_vocal == "有"
    assert record.asr_note == "已调用(zh)"


# ---- run_batch 冒烟 ----

def test_run_batch_order_and_progress(tmp_path, monkeypatch):
    _patch_prereqs(monkeypatch)
    monkeypatch.setattr(
        rp, "suggest_title",
        lambda settings, data, avoid=(), temperature=1.0:
            NamingResult(title=f"歌_{data.file_name[0]}", reason="r"),
    )
    files = [tmp_path / "a.mp3", tmp_path / "b.mp3"]
    seen: list[int] = []
    records = rp.run_batch(
        _settings(concurrency=1), files, TitleRegistry(),
        progress=lambda done, total, rec: seen.append(done),
    )

    assert [r.path for r in records] == files
    assert all(r.ok for r in records)
    assert seen == [1, 2]
