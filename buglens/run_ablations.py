from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buglens.evaluation import (
    compute_macro_f1,
    compute_top2_accuracy,
    evaluate_selective_risk_coverage,
)
from buglens.modeling.fusion import (
    ALL_CLASSES,
    CLASS_TO_IDX,
    compute_ece,
    log_linear_pool,
)
from buglens.reporting import NumpyTfidfBaseline, compute_cohens_kappa


def compute_aurc_naurc_augrc(
    posteriors: np.ndarray,
    y_true: np.ndarray,
) -> dict[str, float]:
    """Compute AURC, Normalized AURC (NAURC), and Generalized Risk AUGRC (NeurIPS 2024)."""
    bug_probs = posteriors[:, :5]
    kappas = np.max(bug_probs, axis=1)
    preds = np.argmax(bug_probs, axis=1)
    errors = (preds != y_true).astype(float)
    n = len(y_true)

    order = np.argsort(-kappas)
    sorted_errors = errors[order]
    coverages = np.arange(1, n + 1, dtype=float) / n
    cum_errors = np.cumsum(sorted_errors)

    selective_risks = cum_errors / np.arange(1, n + 1, dtype=float)
    aurc = float(np.trapezoid(selective_risks, coverages))

    total_errors = int(np.sum(errors))
    oracle_errors = np.concatenate([np.zeros(n - total_errors), np.ones(total_errors)])
    oracle_risks = np.cumsum(oracle_errors) / np.arange(1, n + 1, dtype=float)
    oracle_aurc = float(np.trapezoid(oracle_risks, coverages))
    r_full = float(np.mean(errors))
    denom = max(1e-8, r_full - oracle_aurc)
    naurc = float(np.clip((aurc - oracle_aurc) / denom, 0.0, 1.0))

    generalized_risks = cum_errors / n
    augrc = float(np.trapezoid(generalized_risks, coverages))

    return {
        "aurc": round(aurc, 4),
        "naurc": round(naurc, 4),
        "augrc": round(augrc, 4),
    }


def evaluate_dual_gold_annotations(
    gold_df: pd.DataFrame,
    output_json: Path = Path("reports/gold_kappa.json"),
) -> dict[str, Any]:
    """Step 4.1: Compute pre-adjudication Cohen's kappa between Vansh and Farman."""
    kappa = compute_cohens_kappa(
        gold_df["vansh_label"].astype(str).tolist(),
        gold_df["farman_label"].astype(str).tolist(),
    )
    disagreements = int((gold_df["vansh_label"] != gold_df["farman_label"]).sum())
    summary = {
        "n_items": int(len(gold_df)),
        "pre_adjudication_cohens_kappa": round(kappa, 4),
        "disagreements_adjudicated": disagreements,
        "agreement_rate": round(1.0 - (disagreements / max(1, len(gold_df))), 4),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_diagnosis_ablations(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    p_sym: np.ndarray,
    p_nn: np.ndarray,
    p_llm_zero_shot: np.ndarray | None = None,
    alphas: tuple[float, float, float] = (0.1, 1.0, 1.0),
    tau_teach: float = 0.80,
    split_name: str = "unseen_templates",
) -> pd.DataFrame:
    """Step 4.2: Benchmark all 6 systems across a split with Macro-F1, ECE, AURC, NAURC, AUGRC."""
    train_labels = np.array([CLASS_TO_IDX[str(b)] for b in train_df["bug_id"]], dtype=int)
    eval_labels = np.array([CLASS_TO_IDX[str(b)] for b in eval_df["bug_id"]], dtype=int)
    pi_0 = np.full(len(ALL_CLASSES), 1.0 / len(ALL_CLASSES))

    maj_cls = int(np.bincount(train_labels).argmax())
    maj_probs = np.full((len(eval_labels), len(ALL_CLASSES)), 0.01)
    maj_probs[:, maj_cls] = 0.95

    tfidf = NumpyTfidfBaseline(max_features=512)
    train_texts = (
        train_df["QuestionText"].astype(str)
        + " | "
        + train_df["CorrectAnswerText"].astype(str)
        + " | "
        + train_df["StudentAnswerText"].astype(str)
    ).tolist()
    eval_texts = (
        eval_df["QuestionText"].astype(str)
        + " | "
        + eval_df["CorrectAnswerText"].astype(str)
        + " | "
        + eval_df["StudentAnswerText"].astype(str)
    ).tolist()
    tfidf.fit(train_texts, train_labels, num_classes=len(ALL_CLASSES))
    tfidf_probs = tfidf.predict_proba(eval_texts)

    sym_only = np.vstack(
        [
            log_linear_pool(pi_0, p_sym[i], None, (alphas, alphas, 0.0))
            for i in range(len(eval_labels))
        ]
    )
    nn_only = np.vstack(
        [
            log_linear_pool(pi_0, p_sym[i], p_nn[i], (alphas, 0.0, alphas))
            for i in range(len(eval_labels))
        ]
    )
    fused = np.vstack(
        [log_linear_pool(pi_0, p_sym[i], p_nn[i], alphas) for i in range(len(eval_labels))]
    )

    systems: list[tuple[str, np.ndarray]] = [
        ("majority_class", maj_probs),
        ("replay_only_rules", sym_only),
        ("tfidf_logistic_baseline", tfidf_probs),
    ]
    if p_llm_zero_shot is not None:
        systems.append(("zero_shot_llm", p_llm_zero_shot))
    systems.extend(
        [
            ("neural_only_deberta_t", nn_only),
            ("fused_log_linear_pool", fused),
        ]
    )

    rows: list[dict[str, Any]] = []
    for sys_name, probs in systems:
        preds = np.argmax(probs, axis=1)
        rc = evaluate_selective_risk_coverage(probs, eval_labels, tau_teach=tau_teach)
        curves = compute_aurc_naurc_augrc(probs, eval_labels)
        rows.append(
            {
                "split": split_name,
                "system": sys_name,
                "macro_f1": round(compute_macro_f1(eval_labels, preds, len(ALL_CLASSES)), 4),
                "top2_accuracy": round(compute_top2_accuracy(probs, eval_labels), 4),
                "ece_10bin": round(compute_ece(probs, eval_labels, n_bins=10), 4),
                "teach_precision": rc["teach_precision"],
                "teach_coverage": rc["teach_coverage"],
                **curves,
            }
        )
    return pd.DataFrame(rows)


def benchmark_probing_and_generation(
    probe_df: pd.DataFrame,
    gen_df: pd.DataFrame,
) -> dict[str, Any]:
    """Step 4.3: Compare Max-EIG vs random probing & SymPy-verified vs raw LLM generation."""
    probing_summary = {}
    for strategy in ("max_eig", "random_same_topic"):
        sub = probe_df[probe_df["strategy"] == strategy]
        probing_summary[strategy] = {
            "mean_probes": round(float(sub["probes_used"].mean()), 4) if len(sub) > 0 else 0.0,
            "final_accuracy": round(float(sub["correct_diagnosis"].mean()), 4)
            if len(sub) > 0
            else 0.0,
        }

    gen_summary = {}
    for generator in ("sympy_verified", "raw_llm_only"):
        sub = gen_df[gen_df["generator"] == generator]
        gen_summary[generator] = {
            "sympy_validity_rate": round(float(sub["sympy_valid"].mean()), 4)
            if len(sub) > 0
            else 0.0,
            "rival_discrimination_rate": round(float(sub["verified_distinct"].mean()), 4)
            if len(sub) > 0
            else 0.0,
            "mean_relevance_1_5": round(float(sub["human_relevance"].mean()), 4)
            if len(sub) > 0
            else 0.0,
            "mean_difficulty_1_5": round(float(sub["human_difficulty"].mean()), 4)
            if len(sub) > 0
            else 0.0,
        }

    return {"probing_comparison": probing_summary, "generation_comparison": gen_summary}


def audit_intake_gate(intake_df: pd.DataFrame) -> dict[str, Any]:
    """Step 4.4: Measure AST parse success rate and scope-gate precision/recall."""
    by_slice: dict[str, float] = {}
    for slice_name, grp in intake_df.groupby("input_slice"):
        by_slice[f"parse_success_{slice_name}"] = round(float(grp["ast_parsed"].mean()), 4)

    y_true = intake_df["expected_in_scope"].astype(bool).to_numpy()
    y_pred = intake_df["predicted_in_scope"].astype(bool).to_numpy()
    tp = float(np.sum(y_true & y_pred))
    fp = float(np.sum((~y_true) & y_pred))
    fn = float(np.sum(y_true & (~y_pred)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "overall_ast_parse_rate": round(float(intake_df["ast_parsed"].mean()), 4),
        "scope_gate_precision": round(prec, 4),
        "scope_gate_recall": round(rec, 4),
        "scope_gate_f1": round(f1, 4),
        **by_slice,
    }
