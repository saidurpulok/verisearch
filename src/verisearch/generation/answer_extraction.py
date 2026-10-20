"""Conservative exact numeric answers, not substring or last-number matching.

Priority: last explicit #### marker; last balanced boxed expression; last
'final answer'/'answer is' line. A bare numeric completion is accepted.
Unmarked arithmetic text is deliberately not guessed.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

_NUMBER = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUMERIC = re.compile(rf"^({_NUMBER})(?:\s*/\s*({_NUMBER}))?$")


def normalize_number(text: str) -> str | None:
    value = text.strip().strip("$").strip()
    value = re.sub(r"\\(?:,|!|;|\s)", "", value)
    value = value.replace("−", "-")
    # LaTeX fractions are useful for malformed/nonstandard model output.
    value = re.sub(r"\\(?:dfrac|tfrac|frac)\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", value)
    match = _NUMERIC.fullmatch(value)
    if not match:
        return None
    try:
        numerator = Fraction(Decimal(match[1].replace(",", "")))
        denominator = Fraction(Decimal(match[2].replace(",", ""))) if match[2] else Fraction(1)
        number = numerator / denominator
    except (InvalidOperation, ZeroDivisionError, ValueError):
        return None
    return str(number.numerator) if number.denominator == 1 else str(number)


def _boxed(text: str) -> str | None:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    start += len("\\boxed{")
    depth = 1
    for end in range(start, len(text)):
        depth += (text[end] == "{") - (text[end] == "}")
        if depth == 0:
            return text[start:end]
    return None


def extract_answer(text: str) -> str | None:
    if "####" in text:
        return normalize_number(text.rsplit("####", 1)[1].split("\n", 1)[0])
    if "\\boxed{" in text:
        boxed = _boxed(text)
        return normalize_number(boxed) if boxed is not None else None
    markers = list(re.finditer(r"(?:final answer\s*[:=]|answer is)\s*([^\n]+)", text, re.I))
    if markers:
        # Permit terminal sentence punctuation and a currency symbol, not units.
        return normalize_number(markers[-1][1].strip().rstrip("."))
    return normalize_number(text)


def is_correct(prediction: str | None, gold: str) -> bool:
    target = normalize_number(gold)
    if target is None:
        raise ValueError(f"Invalid gold numeric answer: {gold!r}")
    return prediction is not None and normalize_number(prediction) == target
