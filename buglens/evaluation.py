from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from buglens.modeling.fusion import (
    ALL_CLASSES,
    compute_ece,
    selective_decision_policy,
    sweep_teach_threshold,
)


def compute_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 6) -> float:
    """Compute unweighted Macro-F1 across all 6 classes in pure NumPy."""
    f1s: list[float] = []
    for c in range(num_classes):
        tp = float(np.sum((y_true == c) & (y_pred == c)))
        fp = float(np.sum((y_true != c) & (y_pred == c)))
        fn = float(np.sum((y_true == c) & (y_pred != c)))
        denom = 2.0 * tp + fp + fn
        f1s.append((2.0 * tp / denom) if denom > 0.0 else 0.0)
    return float(np.mean(f1s))


def compute_top2_accuracy(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of items where true class is in the top-2 posterior probabilities."""
    top2 = np.argsort(probs, axis=1)[:, -2:]
    hits = [int(y_true[i]) in top2[i] for i in range(len(y_true))]
    return float(np.mean(hits)) if len(hits) > 0 else 0.0


def evaluate_selective_risk_coverage(
    posteriors: np.ndarray,
    y_true: np.ndarray,
    tau_teach: float,
    max_eigs: np.ndarray | None = None,
    min_eig_bits: float = 0.15,
) -> dict[str, float]:
    """Evaluate TEACH precision, TEACH coverage, PROBE rate, and ABSTAIN rate on test."""
    n = len(y_true)
    if max_eigs is None:
        max_eigs = np.zeros(n, dtype=float)

    taught_correct = 0
    taught_total = 0
    probe_total = 0
    abstain_total = 0

    for i in range(n):
        decision = selective_decision_policy(
            posterior=posteriors[i],
            replay_outputs=[],
            tau_teach=tau_teach,
            probes_used=0,
            max_probes=2,
            max_eig_bits=float(max_eigs[i]),
            min_eig_bits=min_eig_bits,
        )
        if decision.action == "TEACH" and decision.target_bug is not None:
            taught_total += 1
            pred_idx = ALL_CLASSES.index(decision.target_bug)
            if pred_idx == int(y_true[i]):
                taught_correct += 1
        elif decision.action == "PROBE":
            probe_total += 1
        else:
            abstain_total += 1

    teach_precision = (taught_correct / taught_total) if taught_total > 0 else 1.0
    teach_coverage = (taught_total / n) if n > 0 else 0.0
    return {
        "tau_teach": round(float(tau_teach), 4),
        "teach_precision": round(float(teach_precision), 4),
        "teach_coverage": round(float(teach_coverage), 4),
        "probe_rate": round(float(probe_total / n) if n > 0 else 0.0, 4),
        "abstain_rate": round(float(abstain_total / n) if n > 0 else 0.0, 4),
    }


def evaluate_benchmark_split(
    cal_posteriors: np.ndarray,
    cal_labels: np.ndarray,
    test_posteriors: np.ndarray,
    test_labels: np.ndarray,
    target_precision: float = 0.90,
    output_metrics_path: Path | None = Path("reports/metrics.json"),
) -> dict[str, Any]:
    """Run full Step 3 evaluation: tune tau_teach on cal, report all metrics on test."""
    tau_teach = sweep_teach_threshold(cal_posteriors, cal_labels, target_precision=target_precision)
    test_preds = np.argmax(test_posteriors, axis=1)

    metrics: dict[str, Any] = {
        "macro_f1": round(
            compute_macro_f1(test_labels, test_preds, num_classes=len(ALL_CLASSES)), 4
        ),
        "top2_accuracy": round(compute_top2_accuracy(test_posteriors, test_labels), 4),
        "ece_10bin": round(compute_ece(test_posteriors, test_labels, n_bins=10), 4),
        **evaluate_selective_risk_coverage(test_posteriors, test_labels, tau_teach=tau_teach),
    }

    if output_metrics_path is not None:
        output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        output_metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    return metrics
