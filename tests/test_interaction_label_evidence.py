"""Fail-closed evidence matching; no engine or accelerator required."""
import runpy
from pathlib import Path

import pytest


api = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/audit_interaction_labels.py"))


def test_anchor_is_one_based_and_bounded():
    assert api["evidence"](["hidden", "public", "public"], "public", 2, 2) == {
        "line": 2, "quote": "public"}


@pytest.mark.parametrize("lines", [[], ["other"], ["public", "public"]])
def test_missing_or_ambiguous_anchor_fails(lines):
    with pytest.raises(ValueError):
        api["evidence"](lines, "public")


def test_unknown_is_not_a_negative_label():
    assert api["unknown"]("no proof") == {
        "value": None, "status": "unknown", "reason": "no proof"}


def test_fact_requires_evidence():
    with pytest.raises(ValueError):
        api["fact"](True)
