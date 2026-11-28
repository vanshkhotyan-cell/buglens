from pathlib import Path

import numpy as np
import pandas as pd

from buglens.run_ablations import (
    audit_intake_gate,
    benchmark_probing_and_generation,
    compute_aurc_naurc_augrc,
    evaluate_dual_gold_annotations,
    run_diagnosis_ablations,
)


def test_step4_full_ablations_and_audits(tmp_path: Path):
    gold_df = pd.DataFrame(
        [
            {
                "vansh_label": "fraction_add",
                "farman_label": "fraction_add",
                "adjudicated_label": "fraction_add",
            },
            {
                "vansh_label": "binomial_expand",
                "farman_label": "binomial_expand",
                "adjudicated_label": "binomial_expand",
            },
            {
                "vansh_label": "slip",
                "farman_label": "operator_precedence",
                "adjudicated_label": "slip",
            },
            {
                "vansh_label": "unary_minus_distrib",
                "farman_label": "unary_minus_distrib",
                "adjudicated_label": "unary_minus_distrib",
            },
        ]
    )
    gold_summary = evaluate_dual_gold_annotations(gold_df, output_json=tmp_path / "gold_kappa.json")
    assert gold_summary["n_items"] == 4
    assert gold_summary["disagreements_adjudicated"] == 1

    n = 36
    bugs = [
        "fraction_add",
        "binomial_expand",
        "operator_precedence",
        "unary_minus_distrib",
        "transposition_sign",
        "slip",
    ]
    labels = np.arange(n) % 6
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
    p_sym = np.full((n, 6), 0.02)
    p_sym[np.arange(n), labels] = 0.90

    curves = compute_aurc_naurc_augrc(p_sym, labels)
    assert 0.0 <= curves["naurc"] <= 1.0
    assert 0.0 <= curves["augrc"] <= curves["aurc"] + 1e-6

    abl_df = run_diagnosis_ablations(
        train_df=df,
        eval_df=df,
        p_sym=p_sym,
        p_nn=p_sym,
        p_llm_zero_shot=p_sym,
        split_name="gold_150",
    )
    assert len(abl_df) == 6
    assert "augrc" in abl_df.columns

    probe_df = pd.DataFrame(
        [
            {"strategy": "max_eig", "probes_used": 1, "correct_diagnosis": 1},
            {"strategy": "random_same_topic", "probes_used": 2, "correct_diagnosis": 0},
        ]
    )
    gen_df = pd.DataFrame(
        [
            {
                "generator": "sympy_verified",
                "sympy_valid": 1,
                "verified_distinct": 1,
                "human_relevance": 4.8,
                "human_difficulty": 3.5,
            },
            {
                "generator": "raw_llm_only",
                "sympy_valid": 0,
                "verified_distinct": 0,
                "human_relevance": 3.9,
                "human_difficulty": 3.1,
            },
        ]
    )
    bench = benchmark_probing_and_generation(probe_df, gen_df)
    assert (
        bench["probing_comparison"]["max_eig"]["mean_probes"]
        < bench["probing_comparison"]["random_same_topic"]["mean_probes"]
    )

    intake_df = pd.DataFrame(
        [
            {
                "input_slice": "clean_latex",
                "ast_parsed": 1,
                "expected_in_scope": True,
                "predicted_in_scope": True,
            },
            {
                "input_slice": "messy_algebra",
                "ast_parsed": 1,
                "expected_in_scope": True,
                "predicted_in_scope": True,
            },
            {
                "input_slice": "out_of_scope",
                "ast_parsed": 1,
                "expected_in_scope": False,
                "predicted_in_scope": False,
            },
        ]
    )
    audit = audit_intake_gate(intake_df)
    assert audit["scope_gate_f1"] == 1.0
