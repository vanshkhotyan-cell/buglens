from pathlib import Path

import numpy as np
import pandas as pd

from buglens.evaluation import (
    compute_macro_f1,
    compute_top2_accuracy,
    evaluate_benchmark_split,
)
from buglens.modeling.train import (
    augment_with_bug_plus_slip,
    compute_class_weights,
    format_deberta_sequence,
    prepare_training_records,
)


def test_evaluation_metrics_and_risk_coverage(tmp_path: Path):
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 4, 5, 5])
    probs = np.full((10, 6), 0.01)
    for i, c in enumerate(y_true):
        probs[i, c] = 0.95

    assert compute_macro_f1(y_true, y_true) == 1.0
    assert compute_top2_accuracy(probs, y_true) == 1.0

    metrics = evaluate_benchmark_split(
        cal_posteriors=probs,
        cal_labels=y_true,
        test_posteriors=probs,
        test_labels=y_true,
        target_precision=0.90,
        output_metrics_path=tmp_path / "metrics.json",
    )
    assert metrics["macro_f1"] == 1.0
    assert metrics["teach_precision"] >= 0.90
    assert metrics["teach_coverage"] == 0.80


def test_deberta_formatting_and_bug_plus_slip():
    seq = format_deberta_sequence("1/2 + 1/3", "5/6", "2/5")
    assert seq == "[CLS] Question: 1/2 + 1/3 [SEP] Correct: 5/6 [SEP] Student: 2/5 [SEP]"

    df = pd.DataFrame(
        [
            {
                "QuestionText": "(x+3)^2",
                "CorrectAnswerText": "x^2+6x+9",
                "StudentAnswerText": "x^2+9",
                "bug_id": "binomial_expand",
            },
            {
                "QuestionText": "1/2+1/3",
                "CorrectAnswerText": "5/6",
                "StudentAnswerText": "2/5",
                "bug_id": "fraction_add",
            },
        ]
    )
    aug = augment_with_bug_plus_slip(df, slip_fraction=0.5, seed=42)
    assert len(aug) == 3
    texts, labels = prepare_training_records(aug)
    assert len(texts) == 3
    weights = compute_class_weights(labels)
    assert len(weights) == 6
