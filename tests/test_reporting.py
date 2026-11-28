from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from buglens.modeling.train import fit_post_training_calibration
from buglens.reporting import (
    assemble_final_report_bundle,
    compute_cohens_kappa,
    export_offline_demo_bundle,
    generate_error_slices,
    generate_risk_coverage_curve,
    run_all_baselines_and_ablations,
)


def test_cohens_kappa():
    ann_vansh = ["fraction_add", "binomial_expand", "slip", "fraction_add", "operator_precedence"]
    ann_farman = [
        "fraction_add",
        "binomial_expand",
        "slip",
        "operator_precedence",
        "operator_precedence",
    ]
    kappa = compute_cohens_kappa(ann_vansh, ann_farman)
    assert kappa == pytest.approx(0.7368, abs=1e-3)


def test_post_training_calibration_and_reporting(tmp_path: Path):
    rng = np.random.default_rng(11)
    n = 60
    labels = np.arange(n) % 6
    logits = rng.normal(0.0, 0.5, size=(n, 6))
    logits[np.arange(n), labels] += 3.0

    p_sym = np.full((n, 6), 0.02)
    p_sym[np.arange(n), labels] = 0.90

    art = fit_post_training_calibration(logits, p_sym, labels, target_precision=0.90)
    assert art.cal_ece_after <= 0.08
    assert art.tau_teach >= 0.30
    art.save(tmp_path / "calibration.json")
    assert (tmp_path / "calibration.json").exists()

    bugs = [
        "fraction_add",
        "binomial_expand",
        "operator_precedence",
        "unary_minus_distrib",
        "transposition_sign",
        "slip",
    ]
    df = pd.DataFrame(
        [
            {
                "QuestionText": f"q {i} {bugs[i % 6]}",
                "CorrectAnswerText": "1",
                "StudentAnswerText": "0",
                "bug_id": bugs[i % 6],
            }
            for i in range(n)
        ]
    )

    ablation_df = run_all_baselines_and_ablations(
        train_df=df,
        test_df=df,
        p_sym_test=p_sym,
        p_nn_test=p_sym,
        alphas=art.alphas,
        tau_teach=art.tau_teach,
        output_csv=tmp_path / "ablation.csv",
    )
    assert len(ablation_df) == 5

    slices_df = generate_error_slices(
        p_sym, labels, tau_teach=art.tau_teach, output_csv=tmp_path / "slices.csv"
    )
    assert len(slices_df) == 6

    rc_df = generate_risk_coverage_curve(p_sym, labels, output_csv=tmp_path / "rc.csv")
    assert len(rc_df) == 36

    bundle_path = export_offline_demo_bundle(
        df, p_sym, ablation_df, slices_df, rc_df, output_json=tmp_path / "bundle.json"
    )
    assert bundle_path.exists()

    md_path = assemble_final_report_bundle(
        ablation_df, slices_df, gold_kappa=0.84, output_md=tmp_path / "final_summary.md"
    )
    assert md_path.exists()
