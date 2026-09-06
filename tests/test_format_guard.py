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


def test_url_guard_strips_fabricated_urls():
    """URLs not present in source snippets should be stripped."""
    text = "The answer is here http://fake.example.com/paper and also http://real.org/data"
    snippets = ["Some context mentioning http://real.org/data in text"]

    result = clean_output(text, source_snippets=snippets)
    assert "http://real.org/data" in result
    assert "http://fake.example.com" not in result


def test_url_guard_replaces_when_majority_fabricated():
    """If stripping removes >50% of text, return insufficient-evidence message."""
    text = "http://fake1.com http://fake2.com http://fake3.com http://fake4.com"
    snippets = ["No URLs here at all"]

    result = clean_output(text, source_snippets=snippets)
    assert "Insufficient evidence" in result
    assert "fabricate" in result


def test_dangling_list_stub_trim():
    """Dangling list stubs like ': 1.' or ' 1. **' should be trimmed cleanly."""
    assert clean_output("This highlights key challenges: 1.") == "This highlights key challenges."
    assert (
        clean_output("This highlights key challenges: 1. **") == "This highlights key challenges."
    )
    assert clean_output("Summary of findings: 2.") == "Summary of findings."
    assert clean_output("Recent developments:\n1.") == "Recent developments."
    assert (
        clean_output("A complete sentence. Next points: 1.")
        == "A complete sentence."
    )

