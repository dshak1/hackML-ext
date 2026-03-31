from src.benchmarking.metrics import BenchmarkResult, summarize_results


def test_summarize_results_ranks_by_macro_f1() -> None:
    low = BenchmarkResult(
        model_name="low",
        macro_f1_scores=[0.4, 0.5],
        fit_times_sec=[1.0, 1.2],
        predict_times_sec=[0.2, 0.2],
    )
    high = BenchmarkResult(
        model_name="high",
        macro_f1_scores=[0.8, 0.7],
        fit_times_sec=[1.5, 1.7],
        predict_times_sec=[0.3, 0.3],
    )

    summary = summarize_results([low, high])

    assert summary[0]["model_name"] == "high"
    assert summary[1]["model_name"] == "low"
    assert summary[0]["macro_f1_mean"] > summary[1]["macro_f1_mean"]


def test_summarize_results_handles_single_fold_std() -> None:
    single = BenchmarkResult(
        model_name="single",
        macro_f1_scores=[0.6],
        fit_times_sec=[0.8],
        predict_times_sec=[0.1],
    )

    summary = summarize_results([single])

    assert summary[0]["macro_f1_std"] == 0.0
