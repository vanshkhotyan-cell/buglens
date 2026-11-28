from __future__ import annotations

import json
import re
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
    softmax,
)


def compute_cohens_kappa(
    labels_a: list[str] | np.ndarray, labels_b: list[str] | np.ndarray
) -> float:
    """Compute unweighted Cohen's kappa for Vansh + Farman ~150-item dual gold annotation."""
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    if len(a) != len(b) or len(a) == 0:
        raise ValueError("Annotation arrays must be non-empty and have identical length.")

    categories = sorted(set(a.tolist()) | set(b.tolist()))
    n = float(len(a))
    p_o = float(np.mean(a == b))
    p_e = 0.0
    for cat in categories:
        p_e += (float(np.sum(a == cat)) / n) * (float(np.sum(b == cat)) / n)

    if p_e >= 1.0:
        return 1.0
    return float((p_o - p_e) / (1.0 - p_e))


class NumpyTfidfBaseline:
    """Pure-NumPy TF-IDF + L2-regularized ridge classifier baseline (zero scipy DLL risk)."""

    def __init__(self, max_features: int = 512, reg: float = 1.0) -> None:
        self.max_features = max_features
        self.reg = reg
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.weights: np.ndarray | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+|[+\-*/=^()]", text.lower())

    def _featurize(self, texts: list[str], fit: bool = False) -> np.ndarray:
        tokenized = [self._tokenize(t) for t in texts]
        if fit:
            df_counts: dict[str, int] = {}
            for toks in tokenized:
                for tok in set(toks):
                    df_counts[tok] = df_counts.get(tok, 0) + 1
            top_toks = sorted(df_counts.items(), key=lambda kv: (-kv[1], kv[0]))[
                : self.max_features
            ]
            self.vocab = {tok: i for i, (tok, _) in enumerate(top_toks)}
            n_docs = len(texts)
            self.idf = np.array(
                [np.log((1.0 + n_docs) / (1.0 + df_counts[tok])) + 1.0 for tok, _ in top_toks],
                dtype=float,
            )

        assert self.idf is not None
        x = np.zeros((len(texts), len(self.vocab)), dtype=float)
        for i, toks in enumerate(tokenized):
            for tok in toks:
                j = self.vocab.get(tok)
                if j is not None:
                    x[i, j] += 1.0
        x = x * self.idf
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return x / norms

    def fit(self, texts: list[str], labels: np.ndarray, num_classes: int = 6) -> None:
        x = self._featurize(texts, fit=True)
        y_onehot = np.eye(num_classes, dtype=float)[labels]
        gram = x.T @ x + self.reg * np.eye(x.shape[1], dtype=float)
        self.weights = np.linalg.solve(gram, x.T @ y_onehot)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert self.weights is not None
        x = self._featurize(texts, fit=False)
        return softmax(x @ self.weights)


def generate_error_slices(
    test_posteriors: np.ndarray,
    test_labels: np.ndarray,
    tau_teach: float,
    output_csv: Path = Path("reports/error_slices.csv"),
) -> pd.DataFrame:
    """Break down per-class F1, precision, recall, and teach rate across all 6 classes."""
    preds = np.argmax(test_posteriors, axis=1)
    kappas = np.max(test_posteriors[:, :5], axis=1)
    rows: list[dict[str, object]] = []

    for c, name in enumerate(ALL_CLASSES):
        mask = test_labels == c
        support = int(np.sum(mask))
        tp = float(np.sum((test_labels == c) & (preds == c)))
        fp = float(np.sum((test_labels != c) & (preds == c)))
        fn = float(np.sum((test_labels == c) & (preds != c)))
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        taught_rate = float(np.mean(kappas[mask] >= tau_teach)) if support > 0 else 0.0
        rows.append(
            {
                "bug_id": name,
                "support": support,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "teach_rate_at_tau": round(taught_rate, 4),
            }
        )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def generate_risk_coverage_curve(
    test_posteriors: np.ndarray,
    test_labels: np.ndarray,
    output_csv: Path = Path("reports/risk_coverage_curve.csv"),
) -> pd.DataFrame:
    """Sweep tau_teach from 0.30 to 0.99 to generate full AURC risk-coverage curve points."""
    rows: list[dict[str, float]] = []
    for tau in np.linspace(0.30, 0.99, num=36):
        stats = evaluate_selective_risk_coverage(test_posteriors, test_labels, tau_teach=float(tau))
        risk = round(1.0 - stats["teach_precision"], 4) if stats["teach_coverage"] > 0 else 0.0
        rows.append(
            {
                "tau_teach": stats["tau_teach"],
                "coverage": stats["teach_coverage"],
                "precision": stats["teach_precision"],
                "selective_risk": risk,
                "abstain_rate": stats["abstain_rate"],
            }
        )
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def run_all_baselines_and_ablations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    p_sym_test: np.ndarray,
    p_nn_test: np.ndarray,
    alphas: tuple[float, float, float] = (0.1, 1.0, 1.0),
    tau_teach: float = 0.80,
    output_csv: Path = Path("reports/ablation_table.csv"),
) -> pd.DataFrame:
    """Compare Majority, TF-IDF, Symbolic-Only, Neural-Only, and Fused Log-Linear Pool."""
    train_labels = np.array([CLASS_TO_IDX[str(b)] for b in train_df["bug_id"]], dtype=int)
    test_labels = np.array([CLASS_TO_IDX[str(b)] for b in test_df["bug_id"]], dtype=int)
    pi_0 = np.full(len(ALL_CLASSES), 1.0 / len(ALL_CLASSES))

    maj_cls = int(np.bincount(train_labels).argmax())
    maj_probs = np.full((len(test_labels), len(ALL_CLASSES)), 0.01)
    maj_probs[:, maj_cls] = 0.95

    tfidf = NumpyTfidfBaseline(max_features=512)
    train_texts = (
        train_df["QuestionText"].astype(str)
        + " | "
        + train_df["CorrectAnswerText"].astype(str)
        + " | "
        + train_df["StudentAnswerText"].astype(str)
    ).tolist()
    test_texts = (
        test_df["QuestionText"].astype(str)
        + " | "
        + test_df["CorrectAnswerText"].astype(str)
        + " | "
        + test_df["StudentAnswerText"].astype(str)
    ).tolist()
    tfidf.fit(train_texts, train_labels, num_classes=len(ALL_CLASSES))
    tfidf_probs = tfidf.predict_proba(test_texts)

    sym_only = np.vstack(
        [
            log_linear_pool(pi_0, p_sym_test[i], None, (alphas[0], alphas[1], 0.0))
            for i in range(len(test_labels))
        ]
    )
    nn_only = np.vstack(
        [
            log_linear_pool(pi_0, p_sym_test[i], p_nn_test[i], (alphas[0], 0.0, alphas[2]))
            for i in range(len(test_labels))
        ]
    )
    fused = np.vstack(
        [
            log_linear_pool(pi_0, p_sym_test[i], p_nn_test[i], alphas)
            for i in range(len(test_labels))
        ]
    )

    models: list[tuple[str, np.ndarray]] = [
        ("majority_baseline", maj_probs),
        ("tfidf_linear_baseline", tfidf_probs),
        ("symbolic_replay_only", sym_only),
        ("neural_deberta_only", nn_only),
        ("fused_log_linear_pool", fused),
    ]

    rows: list[dict[str, Any]] = []
    for name, probs in models:
        preds = np.argmax(probs, axis=1)
        rc = evaluate_selective_risk_coverage(probs, test_labels, tau_teach=tau_teach)
        rows.append(
            {
                "system": name,
                "macro_f1": round(compute_macro_f1(test_labels, preds, len(ALL_CLASSES)), 4),
                "top2_accuracy": round(compute_top2_accuracy(probs, test_labels), 4),
                "ece_10bin": round(compute_ece(probs, test_labels, n_bins=10), 4),
                "teach_precision": rc["teach_precision"],
                "teach_coverage": rc["teach_coverage"],
            }
        )

    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def export_offline_demo_bundle(
    test_df: pd.DataFrame,
    test_posteriors: np.ndarray,
    ablation_df: pd.DataFrame,
    slices_df: pd.DataFrame,
    rc_df: pd.DataFrame,
    output_json: Path = Path("reports/offline_demo_bundle.json"),
) -> Path:
    """Create a self-contained offline demo & metrics bundle for Round 2 venue safety."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    sample_cases = []
    for i in range(min(15, len(test_df))):
        row = test_df.iloc[i]
        probs = {
            ALL_CLASSES[c]: round(float(test_posteriors[i, c]), 4) for c in range(len(ALL_CLASSES))
        }
        sample_cases.append(
            {
                "question": str(row["QuestionText"]),
                "correct_answer": str(row["CorrectAnswerText"]),
                "student_answer": str(row["StudentAnswerText"]),
                "true_bug": str(row["bug_id"]),
                "posterior": probs,
            }
        )
    bundle = {
        "python_version_locked": "3.12",
        "offline_ready": True,
        "sample_cases": sample_cases,
        "ablation_table": ablation_df.to_dict(orient="records"),
        "error_slices": slices_df.to_dict(orient="records"),
        "risk_coverage_curve": rc_df.to_dict(orient="records"),
    }
    output_json.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return output_json


def assemble_final_report_bundle(
    ablation_df: pd.DataFrame,
    slices_df: pd.DataFrame,
    gold_kappa: float | None = None,
    output_md: Path = Path("reports/final_summary.md"),
) -> Path:
    """Compile Markdown metrics tables for final presentation deck & offline demo check."""
    output_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# BugLens — Final Evaluation & Ablation Summary",
        "",
        f"**Dual-Annotation Gold Set Cohen's κ:** `{gold_kappa:.4f}`"
        if gold_kappa is not None
        else "**Dual-Annotation Gold Set Cohen's κ:** Pending ~150-item annotation",
        "",
        "## 1. Baselines & Ablation Comparison",
        ablation_df.to_markdown(index=False),
        "",
        "## 2. Per-Misconception Error Slices",
        slices_df.to_markdown(index=False),
        "",
    ]
    output_md.write_text("\n".join(lines), encoding="utf-8")
    return output_md
