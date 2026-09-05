import dataclasses
import logging

import pytest

from aether_scientist.core.profiles import (
    PROFILES,
    ModelProfile,
    get_profile,
    list_profiles,
    resolve,
)


def test_profile_registry_completeness():
    expected = {"offline", "fast", "balanced", "quality"}
    assert set(PROFILES.keys()) == expected
    for name in expected:
        p = PROFILES[name]
        assert isinstance(p, ModelProfile)
        assert p.name == name
        assert isinstance(p.model, str)
        assert isinstance(p.chat, bool)
        assert p.max_new_tokens > 0
        assert 0.0 <= p.temperature <= 1.0
        assert len(p.description) > 0


def test_profile_frozen_immutability():
    prof = PROFILES["offline"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        prof.temperature = 0.1  # type: ignore[misc]


def test_get_profile_valid():
    prof = get_profile("fast")
    assert prof.name == "fast"
    assert prof.chat is True
    assert prof.max_new_tokens == 256


def test_get_profile_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unknown profile 'unknown_model'"):
        get_profile("unknown_model")


def test_resolve_default():
    prof = resolve()
    assert prof.name == "offline"
    assert prof.model == "distilgpt2"


def test_resolve_by_name():
    prof = resolve("balanced")
    assert prof.name == "balanced"
    assert prof.model == "Qwen/Qwen2.5-0.5B-Instruct"


def test_resolve_unknown_fallback_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        prof = resolve("non_existent_preset")
    assert prof.name == "offline"
    assert "non_existent_preset" in caplog.text


def test_resolve_env_priority(monkeypatch):
    monkeypatch.setenv("AETHER_MODEL", "quality")
    # env var beats explicit argument
    prof = resolve("fast")
    assert prof.name == "quality"


def test_list_profiles():
    items = list_profiles()
    assert len(items) == 4
    names = [item["name"] for item in items]
    assert "offline" in names
    assert "quality" in names
    for item in items:
        assert "name" in item
        assert "model" in item
        assert "max_new_tokens" in item
