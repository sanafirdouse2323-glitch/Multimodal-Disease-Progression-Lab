"""Synthetic causal-effect demonstration with known ground truth.

These simulations teach confounding and estimators. They are not evidence that the
same assumptions hold in real observational healthcare data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.special import expit


def simulate_treatment_effect(
    n: int = 5000,
    seed: int = 42,
    true_effect: float = -0.8,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if n < 200:
        raise ValueError("Use at least 200 synthetic individuals for this demonstration")
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    age = np.clip(rng.normal(62, 10, size=n), 30, 88)
    propensity_true = expit(-0.25 + 0.95 * risk + 0.025 * (age - 62))
    treatment = rng.binomial(1, propensity_true, size=n)
    outcome = (
        1.0 + 1.25 * risk + 0.035 * (age - 62)
        + true_effect * treatment + rng.normal(0, 0.8, size=n)
    )
    frame = pd.DataFrame({
        "synthetic_risk": risk,
        "synthetic_age": age,
        "treatment": treatment,
        "outcome": outcome,
        "true_propensity": propensity_true,
    })

    treated_mean = frame.loc[frame.treatment == 1, "outcome"].mean()
    control_mean = frame.loc[frame.treatment == 0, "outcome"].mean()
    naive_difference = float(treated_mean - control_mean)

    design = np.column_stack([np.ones(n), treatment, risk, age - 62])
    adjusted_beta = np.linalg.lstsq(design, outcome, rcond=None)[0]
    adjusted_effect = float(adjusted_beta[1])

    propensity_model = LogisticRegression(max_iter=1000, random_state=seed)
    propensity_model.fit(frame[["synthetic_risk", "synthetic_age"]], treatment)
    propensity = np.clip(
        propensity_model.predict_proba(frame[["synthetic_risk", "synthetic_age"]])[:, 1],
        0.03,
        0.97,
    )
    weights = np.where(treatment == 1, 1.0 / propensity, 1.0 / (1.0 - propensity))
    treated_mask = treatment == 1
    weighted_treated = float(np.average(outcome[treated_mask], weights=weights[treated_mask]))
    weighted_control = float(np.average(outcome[~treated_mask], weights=weights[~treated_mask]))
    ipw_effect = weighted_treated - weighted_control

    estimates = {
        "known_simulated_effect": float(true_effect),
        "naive_difference": float(naive_difference),
        "adjusted_regression": adjusted_effect,
        "inverse_probability_weighting": float(ipw_effect),
        "propensity_min": float(propensity.min()),
        "propensity_max": float(propensity.max()),
        "n": int(n),
    }
    return frame, estimates
