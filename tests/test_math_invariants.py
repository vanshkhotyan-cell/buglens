import numpy as np
import pytest

from buglens.modeling.fusion import (
    bkt_inverse_polarity_update,
    compute_ece,
    expected_info_gain_bits,
    fit_temperature,
    log_linear_pool,
    selective_decision_policy,
    shannon_entropy_bits,
    smoothed_symbolic_likelihood,
)
from buglens.schemas import ReplayOutput


def test_eig_separation_invariant():
    pi = np.array([0.45, 0.05, 0.05, 0.05, 0.05, 0.35], dtype=float)
    assert shannon_entropy_bits(pi) == pytest.approx(1.9129, abs=1e-4)

    separated_outcomes = [{0}, {1}, {2}, {3}, {4}, set()]
    eig_verified = expected_info_gain_bits(
        pi, separated_outcomes, eps=0.08, slip_bug_prob=0.02, slip_other_prob=0.90
    )
    assert eig_verified == pytest.approx(1.4478, abs=1e-4)

    collided_outcomes = [{0, 2}, {1}, {3}, {4}, set()]
    eig_collided = expected_info_gain_bits(
        pi, collided_outcomes, eps=0.08, slip_bug_prob=0.02, slip_other_prob=0.90
    )
    assert eig_collided == pytest.approx(1.2608, abs=1e-4)
    assert eig_verified > eig_collided


def test_bkt_inverse_polarity_invariant():
    p_post, p_next = bkt_inverse_polarity_update(
        p_prior=0.85,
        observed_correct=True,
        p_correct_given_active=0.10,
        p_correct_given_inactive=0.90,
        p_learn=0.30,
        p_relapse=0.05,
    )
    assert p_post == pytest.approx(0.3864, abs=1e-4)
    assert p_next == pytest.approx(0.3012, abs=1e-4)


def test_symbolic_mvp_pooling_and_slip_exclusion():
    slip_post = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.95], dtype=float)
    decision = selective_decision_policy(
        slip_post, replay_outputs=[], tau_teach=0.80, probes_used=0, max_eig_bits=0.0
    )
    assert decision.action == "ABSTAIN"
    assert decision.confidence == pytest.approx(0.01)

    trace = ReplayOutput(
        bug_id="fraction_add",
        applicable=True,
        matches_student=True,
        matches_correct=False,
        predicted_latex=r"\frac{2}{5}",
        trace_steps=["(1+1)/(2+3)=2/5"],
    )
    p_sym = smoothed_symbolic_likelihood([trace], eps=0.08)
    pi_0 = np.full(6, 1.0 / 6.0)
    post = log_linear_pool(pi_0, p_sym, p_nn=None, alphas=(0.1, 1.0, 0.0))
    mvp_decision = selective_decision_policy(post, replay_outputs=[trace], tau_teach=0.80)
    assert mvp_decision.action == "TEACH"
    assert mvp_decision.target_bug == "fraction_add"


def test_temperature_scaling_and_ece():
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 6, size=120)
    logits = rng.normal(0, 1.0, size=(120, 6))
    logits[np.arange(120), labels] += 3.0
    overconf_logits = logits * 5.0
    t_opt = fit_temperature(overconf_logits, labels)
    assert t_opt > 1.4
    cal_probs = np.exp(overconf_logits / t_opt)
    cal_probs /= np.sum(cal_probs, axis=1, keepdims=True)
    assert compute_ece(cal_probs, labels) <= 0.12
