from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, stdev
from typing import Dict, Iterable, List


@dataclass
class BenchmarkResult:
    model_name: str
    macro_f1_scores: List[float]
    fit_times_sec: List[float]
    predict_times_sec: List[float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _safe_stdev(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return stdev(values)


def summarize_results(results: Iterable[BenchmarkResult]) -> List[Dict[str, float]]:
    summary: List[Dict[str, float]] = []
    for result in results:
        summary.append(
            {
                "model_name": result.model_name,
                "macro_f1_mean": mean(result.macro_f1_scores),
                "macro_f1_std": _safe_stdev(result.macro_f1_scores),
                "fit_time_mean_sec": mean(result.fit_times_sec),
                "predict_time_mean_sec": mean(result.predict_times_sec),
            }
        )

    summary.sort(key=lambda row: row["macro_f1_mean"], reverse=True)
    return summary
