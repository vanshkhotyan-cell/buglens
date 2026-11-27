from pathlib import Path

from buglens.pipeline import run_evaluate, run_prepare


def test_end_to_end_pipeline_stages(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    summary = run_prepare(seed=42)
    assert summary["train"] > 0 and summary["cal"] > 0 and summary["test"] > 0
    report = run_evaluate(log_mlflow=False)
    assert report["teach_precision"] >= 0.90
    assert (tmp_path / "reports" / "metrics.json").exists()
