"""Tests for degenerate output detection, loop cutting, and fallback guard."""

from aether_scientist.core.format import (
    clean_output,
    detect_degenerate,
)


def test_clean_text_untouched():
    text = (
        "Photosynthesis is a biological process used by plants to convert light energy "
        "into chemical energy that can be utilized to fuel cellular activities."
    )
    assert not detect_degenerate(text)
    assert clean_output(text, profile_name="balanced") == text


def test_degenerate_loop_cut_preserves_prefix():
    prefix = (
        "The authors scaled the dot products by 1/sqrt(d_k) to avoid gradient vanishing. "
    )
    repeated = "scaled by 1/sqrt(d_k) " * 4
    dirty = prefix + repeated
    assert detect_degenerate(dirty)

    cleaned = clean_output(dirty, profile_name="fast")
    assert "scaled the dot products" in cleaned
    assert cleaned.count("scaled by 1/sqrt(d_k)") <= 1
    assert "degenerate output" not in cleaned


def test_degenerate_short_fallback_message():
    loop_only = "repeat this phrase " * 5
    assert detect_degenerate(loop_only)

    cleaned = clean_output(loop_only, profile_name="offline")
    assert "[Model produced degenerate output on profile 'offline'." in cleaned
    assert "Use a stronger profile (--profile balanced|quality)." in cleaned


def test_charset_collapse_detection_and_fallback():
    collapse = "x _ 1 % $ # @ ! ^ & * ~ ` + = [ ] { } ; : < > ,"
    assert detect_degenerate(collapse)

    cleaned = clean_output(collapse, profile_name="fast")
    assert "[Model produced degenerate output on profile 'fast'." in cleaned
