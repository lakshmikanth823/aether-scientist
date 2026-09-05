import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    confidence_score: float = 1.0


class OutputValidator:
    def __init__(self) -> None:
        self.citation_pattern = re.compile(r"\([A-Za-z]+ et al\., \d{4}\)|\([A-Za-z]+, \d{4}\)")
        self.p_value_pattern = re.compile(r"p\s*[<>=]\s*0\.\d+")
        self.unit_pattern = re.compile(r"\b\d+\s*(kg|g|mg|L|mL|m|cm|mm|s|ms)\b")

    def validate(self, output: dict[str, Any]) -> ValidationResult:
        text = str(output.get("text", ""))

        errors: list[str] = []
        warnings: list[str] = []

        warnings.extend(self._check_citations(text))
        errors.extend(self._check_units(text))
        warnings.extend(self._check_statistical_claims(text))
        errors.extend(self._check_numerical_consistency(text))

        is_valid = len(errors) == 0
        confidence_score = 1.0 - (len(errors) * 0.2) - (len(warnings) * 0.1)
        confidence_score = max(0.0, min(1.0, confidence_score))

        return ValidationResult(
            is_valid=is_valid, errors=errors, warnings=warnings, confidence_score=confidence_score
        )

    def _check_citations(self, text: str) -> list[str]:
        citations = self.citation_pattern.findall(text)
        if not citations and len(text) > 500:
            return ["No citations found in long text output."]
        return []

    def _check_units(self, text: str) -> list[str]:
        units_found = list(self.unit_pattern.findall(text))
        if "kg" in units_found and "g" in units_found:
            return ["Inconsistent mass units detected (both kg and g)."]
        return []

    def _check_statistical_claims(self, text: str) -> list[str]:
        p_values = self.p_value_pattern.findall(text)
        if p_values and "significant" not in text.lower():
            return [f"Statistical values found ({p_values[0]}) but no significance claimed."]
        return []

    def _check_numerical_consistency(self, text: str) -> list[str]:
        nums = [int(n) for n in re.findall(r"\b\d+\b", text) if n.isdigit()]
        if len(nums) >= 2 and max(nums) > sum(nums) * 2:
            return ["Potential numerical contradiction detected."]
        return []
