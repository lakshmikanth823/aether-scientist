import re

# Module-level compiled patterns (ordered by specificity)
LATEX_CMD = re.compile(r"\\[a-zA-Z]+\{[^}]*\}(?:\{[^}]*\})?|\\[a-zA-Z]+")
CHEMISTRY = re.compile(r"[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+")
CITATION_BRACKET = re.compile(r"\[\d+(?:,\s*\d+)*\]")
CITATION_PARENS = re.compile(r"\([A-Z][a-z]+(?:\s+et\s+al\.)?(?:,|\s)\s*\d{4}\)")
CITATION_LATEX = re.compile(r"\\cite\{[^}]+\}")
SCIENTIFIC_NUM = re.compile(r"\d+\.\d+[eE][-+]?\d+")
SUBSUPER = re.compile(r"\w+[\^_]\{?\w+\}?")

# Unified tokenization pattern
TOKEN_PATTERN = re.compile(
    r"("
    r"\\[a-zA-Z]+\{[^}]*\}(?:\{[^}]*\})?|"  # LaTeX commands with args
    r"\\cite\{[^}]+\}|"                     # \cite{...}
    r"\[\d+(?:,\s*\d+)*\]|"                 # [1] or [1,2,3]
    r"\([A-Z][a-z]+(?:\s+et\s+al\.)?[,\s]\s*\d{4}\)|"  # (Author, Year)
    r"\d+\.\d+[eE][-+]?\d+|"                # Scientific notation
    r"\w+[\^_]\{?\w+\}?|"                   # Subscripts/superscripts
    r"[A-Z][a-z]?\d*(?:[A-Z][a-z]?\d*)+|"   # Chemical formulas
    r"\\[a-zA-Z]+|"                          # LaTeX symbols
    r"\w+|"                                  # Standard words
    r"[^\w\s]"                               # Punctuation
    r")"
)

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[EQ]"]


class ScientificTokenizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.inverse_vocab: dict[int, str] = {}
        for token in SPECIAL_TOKENS:
            self._get_or_add_token(token)

    def _get_or_add_token(self, token: str) -> int:
        if token not in self.vocab:
            token_id = len(self.vocab)
            self.vocab[token] = token_id
            self.inverse_vocab[token_id] = token
        return self.vocab[token]

    def encode(self, text: str) -> list[int]:
        tokens = [m.group(1) for m in TOKEN_PATTERN.finditer(text)]
        return [self._get_or_add_token(t) for t in tokens]

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self.inverse_vocab.get(tid, "[UNK]") for tid in token_ids)

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(text) for text in texts]

    def detect_citations(self, text: str) -> list[str]:
        """Return all citation strings found in text."""
        results: list[str] = []
        results.extend(CITATION_BRACKET.findall(text))
        results.extend(CITATION_PARENS.findall(text))
        results.extend(CITATION_LATEX.findall(text))
        return results

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)
