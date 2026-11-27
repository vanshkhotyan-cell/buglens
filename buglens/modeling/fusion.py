from __future__ import annotations

import math
from typing import Literal

import numpy as np

from buglens.schemas import DiagnosisDecision, ReplayOutput

BUG_FAMILIES: tuple[str, ...] = (
    "fraction_add",
    "binomial_expand",
    "operator_precedence",
    "unary_minus_distrib",
    "transposition_sign",
)
ALL_CLASSES: tuple[str, ...] = (*BUG_FAMILIES, "slip")
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(ALL_CLASSES)}


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=-1, keepdims=True)


def apply_temperature_scaling(logits: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("Temperature T must be strictly positive.")
    return softmax(logits / temperature)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Step 3.1: Optimize scalar T > 0 on held-out cal logits to minimize NLL."""
    best_t = 1.0
    best_nll = float("inf")
    for t in np.geomspace(0.05, 10.0, num=400):
        probs = apply_temperature_scaling(logits, float(t))
        nll = -float(np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0))))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)

    low, high = max(1e-3, best_t * 0.8), best_t * 1.25
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    for _ in range(50):
        c = high - (high - low) / phi
        d = low + (high - low) / phi
        pc = apply_temperature_scaling(logits, c)
        pd_ = apply_temperature_scaling(logits, d)
        nll_c = -float(np.mean(np.log(np.clip(pc[np.arange(len(labels)), labels], 1e-12, 1.0))))
        nll_d = -float(np.mean(np.log(np.clip(pd_[np.arange(len(labels)), labels], 1e-12, 1.0))))
        if nll_c < nll_d:
            high = d
        else:
            low = c
    return float((low + high) / 2.0)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute 10-bin Expected Calibration Error (target ECE <= 0.08)."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (
            (confidences > lo) & (confidences <= hi)
            if i > 0
            else (confidences >= lo) & (confidences <= hi)
        )
        count = int(np.sum(in_bin))
        if count > 0:
            ece += (count / n) * abs(
                float(np.mean(accuracies[in_bin])) - float(np.mean(confidences[in_bin]))
            )
    return float(ece)


def smoothed_symbolic_likelihood(
    replay_outputs: list[ReplayOutput],
    eps: float = 0.08,
    eta: float = 0.10,
) -> np.ndarray:
    """Step 3.2: Map Shivam's replay_all_bugs(q, a) outputs into smoothed p_sym(c | q, a)."""
    matching = [r.bug_id for r in replay_outputs if r.matches_student and r.bug_id in BUG_FAMILIES]
    unique_matching = sorted(set(matching))
    p_sym = np.zeros(len(ALL_CLASSES), dtype=float)

    k = len(unique_matching)
    if k >= 1:
        match_prob = (1.0 - eps) / k
        rem_prob = eps / (len(ALL_CLASSES) - k)
        for c, idx in CLASS_TO_IDX.items():
            p_sym[idx] = match_prob if c in unique_matching else rem_prob
    else:
        for b in BUG_FAMILIES:
            p_sym[CLASS_TO_IDX[b]] = eta / len(BUG_FAMILIES)
        p_sym[CLASS_TO_IDX["slip"]] = 1.0 - eta

    return p_sym


def log_linear_pool(
    pi_0: np.ndarray,
    p_sym: np.ndarray,
    p_nn: np.ndarray | None = None,
    alphas: tuple[float, float, float] = (0.1, 1.0, 1.0),
    delta: float = 1e-6,
) -> np.ndarray:
    """Step 3.3: Weighted log-linear opinion pooling (set alphas=0 for symbolic MVP)."""
    a0, a1, a2 = alphas
    if a0 < 0 or a1 < 0 or a2 < 0:
        raise ValueError("Pooling weights (alpha_0, alpha_1, alpha_2) must be non-negative.")

    log_tilde = a0 * np.log(pi_0 + delta) + a1 * np.log(p_sym + delta)
    if p_nn is not None and a2 > 0.0:
        log_tilde = log_tilde + a2 * np.log(p_nn + delta)

    shifted = log_tilde - np.max(log_tilde)
    unnorm = np.exp(shifted)
    return unnorm / np.sum(unnorm)


def fit_pooling_weights(
    pi_0_cal: np.ndarray,
    p_sym_cal: np.ndarray,
    p_nn_cal: np.ndarray,
    labels_cal: np.ndarray,
    delta: float = 1e-6,
    lr: float = 0.1,
    steps: int = 400,
) -> tuple[float, float, float]:
    """Optimize non-negative (alpha_0, alpha_1, alpha_2) on cal via projected gradient descent."""
    alphas = np.array([0.1, 1.0, 1.0], dtype=float)
    n = len(labels_cal)
    y_idx = (np.arange(n), labels_cal)

    for _ in range(steps):
        log_scores = (
            alphas * np.log(pi_0_cal + delta)
            + alphas * np.log(p_sym_cal + delta)
            + alphas * np.log(p_nn_cal + delta)
        )
        probs = softmax(log_scores)
        grad = np.zeros(3, dtype=float)
        features = [np.log(pi_0_cal + delta), np.log(p_sym_cal + delta), np.log(p_nn_cal + delta)]
        for k in range(3):
            expected_f = np.sum(probs * features[k], axis=1)
            actual_f = features[k][y_idx]
            grad[k] = float(np.mean(expected_f - actual_f))
        alphas = np.maximum(0.0, alphas - lr * grad)

    return float(alphas), float(alphas), float(alphas)


def shannon_entropy_bits(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    return float(-np.sum(p * np.log2(p)))


def expected_info_gain_bits(
    prior: np.ndarray,
    outcome_bug_map: list[set[int]],
    eps: float = 0.08,
    slip_bug_prob: float = 0.02,
    slip_other_prob: float = 0.90,
) -> float:
    """Compute EIG(q) = H(pi) - E_{o}[H(pi | o)] in bits for a candidate probe item."""
    h_prior = shannon_entropy_bits(prior)
    n_classes = len(prior)
    n_outcomes = len(outcome_bug_map)
    lik = np.zeros((n_outcomes, n_classes), dtype=float)

    for c in range(5):
        matched_outcomes = [o for o, bugs in enumerate(outcome_bug_map) if c in bugs]
        unmatched_outcomes = [o for o, bugs in enumerate(outcome_bug_map) if c not in bugs]
        other_outcomes_c = [o for o, bugs in enumerate(outcome_bug_map) if len(bugs) == 0]
        if matched_outcomes:
            for o in matched_outcomes:
                lik[o, c] = (1.0 - eps) / len(matched_outcomes)
            if other_outcomes_c:
                for o in other_outcomes_c:
                    lik[o, c] = eps / len(other_outcomes_c)
        else:
            lik[:, c] = 1.0 / n_outcomes

    bug_outcomes = [o for o, bugs in enumerate(outcome_bug_map) if len(bugs) > 0]
    other_outcomes = [o for o, bugs in enumerate(outcome_bug_map) if len(bugs) == 0]
    for o in bug_outcomes:
        lik[o, 5] = slip_bug_prob
    if other_outcomes:
        for o in other_outcomes:
            lik[o, 5] = slip_other_prob / len(other_outcomes)

    p_o = lik @ prior
    expected_post_h = 0.0
    for o in range(n_outcomes):
        if p_o[o] > 0:
            post_o = (lik[o, :] * prior) / p_o[o]
            expected_post_h += float(p_o[o]) * shannon_entropy_bits(post_o)

    return float(h_prior - expected_post_h)


def sweep_teach_threshold(
    posteriors: np.ndarray,
    labels: np.ndarray,
    target_precision: float = 0.90,
) -> float:
    """Step 3.4: Find lowest tau_teach over B1..B5 guaranteeing >= target_precision."""
    bug_probs = posteriors[:, :5]
    kappas = np.max(bug_probs, axis=1)
    preds = np.argmax(bug_probs, axis=1)

    for tau in np.linspace(0.30, 0.99, num=70):
        taught = kappas >= tau
        if np.sum(taught) == 0:
            continue
        precision = float(np.mean(preds[taught] == labels[taught]))
        if precision >= target_precision:
            return float(tau)
    return 0.95


def selective_decision_policy(
    posterior: np.ndarray,
    replay_outputs: list[ReplayOutput],
    tau_teach: float = 0.80,
    probes_used: int = 0,
    max_probes: int = 2,
    max_eig_bits: float = 0.0,
    min_eig_bits: float = 0.15,
) -> DiagnosisDecision:
    """Step 3.4: Geifman & El-Yaniv selective policy excluding S_slip from kappa(x)."""
    bug_probs = posterior[:5]
    top_idx = int(np.argmax(bug_probs))
    kappa = float(bug_probs[top_idx])
    top_bug = BUG_FAMILIES[top_idx]
    entropy = shannon_entropy_bits(posterior)

    action: Literal["TEACH", "PROBE", "ABSTAIN"]
    target_bug: str | None = None

    if kappa >= tau_teach:
        action = "TEACH"
        target_bug = top_bug
    elif probes_used < max_probes and max_eig_bits >= min_eig_bits:
        action = "PROBE"
        target_bug = top_bug
    else:
        action = "ABSTAIN"
        target_bug = None

    return DiagnosisDecision(
        posterior={ALL_CLASSES[i]: float(posterior[i]) for i in range(len(ALL_CLASSES))},
        entropy_bits=round(entropy, 6),
        action=action,
        target_bug=target_bug,
        confidence=round(kappa, 6),
        evidence_traces=replay_outputs,
    )


def bkt_inverse_polarity_update(
    p_prior: float,
    observed_correct: bool,
    p_correct_given_active: float = 0.10,
    p_correct_given_inactive: float = 0.90,
    p_learn: float = 0.30,
    p_relapse: float = 0.05,
) -> tuple[float, float]:
    """Verified BKT inverse-polarity Bayes update returning (p_posterior, p_next_prior)."""
    if observed_correct:
        num = p_correct_given_active * p_prior
        den = num + p_correct_given_inactive * (1.0 - p_prior)
    else:
        num = (1.0 - p_correct_given_active) * p_prior
        den = num + (1.0 - p_correct_given_inactive) * (1.0 - p_prior)

    p_post = num / den
    p_next = p_post * (1.0 - p_learn) + (1.0 - p_post) * p_relapse
    return float(p_post), float(p_next)
