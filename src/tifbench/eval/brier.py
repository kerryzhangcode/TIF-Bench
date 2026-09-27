"""Brier score helpers."""

from __future__ import annotations


def brier_score(probability_yes: float, outcome: int) -> float:
    return (float(probability_yes) - float(outcome)) ** 2


def mean_brier(scores: list[float]) -> float:
    if not scores:
        return float("nan")
    return sum(scores) / len(scores)
