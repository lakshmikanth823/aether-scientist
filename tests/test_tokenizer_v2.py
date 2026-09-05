
from aether_scientist.core.tokenizer import ScientificTokenizer


def test_tokenizer_latex_patterns():
    tok = ScientificTokenizer()
    text = r"\frac{a}{b} + \sqrt{x} = \alpha \beta"
    tokens = tok.encode(text)
    decoded = tok.decode(tokens)
    assert r"\frac{a}{b}" in decoded
    assert r"\sqrt{x}" in decoded
    assert r"\alpha" in decoded
    assert r"\beta" in decoded


def test_tokenizer_chemical_formulas():
    tok = ScientificTokenizer()
    text = "The reaction of H2SO4 with NaCl yields HCl and Na2SO4."
    tokens = tok.encode(text)
    decoded = tok.decode(tokens)
    assert "H2SO4" in decoded
    assert "NaCl" in decoded
    assert "HCl" in decoded
    assert "Na2SO4" in decoded


def test_tokenizer_scientific_notation():
    tok = ScientificTokenizer()
    text = "Planck constant is 6.626e-34 and Avogadro is 6.022e23."
    tokens = tok.encode(text)
    decoded = tok.decode(tokens)
    assert "6.626e-34" in decoded
    assert "6.022e23" in decoded


def test_tokenizer_subscripts_superscripts():
    tok = ScientificTokenizer()
    text = "Let x^2 + y_1 = z_{max}"
    tokens = tok.encode(text)
    decoded = tok.decode(tokens)
    assert "x^2" in decoded
    assert "y_1" in decoded
    assert "z_{max}" in decoded


def test_tokenizer_citation_detection():
    tok = ScientificTokenizer()
    sample = (
        "Previous studies [1] and [2, 3] demonstrated this effect. "
        "As established by (Smith et al., 2023) and \\cite{hawking1974}, "
        "black holes emit radiation."
    )
    citations = tok.detect_citations(sample)
    assert "[1]" in citations
    assert "[2, 3]" in citations
    assert "(Smith et al., 2023)" in citations
    assert r"\cite{hawking1974}" in citations


def test_tokenizer_vocab_and_batch():
    tok = ScientificTokenizer()
    assert tok.vocab_size == 5
    batch = tok.encode_batch(["CO2 emission", "E = mc^2"])
    assert len(batch) == 2
    assert tok.vocab_size > 5
