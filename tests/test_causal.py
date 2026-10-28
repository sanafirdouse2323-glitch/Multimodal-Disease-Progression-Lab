from medprogress.causal import simulate_treatment_effect


def test_adjusted_estimator_recovers_known_simulated_effect_reasonably_well():
    _, estimates = simulate_treatment_effect(n=6000, seed=42, true_effect=-0.8)
    assert abs(estimates["adjusted_regression"] - estimates["known_simulated_effect"]) < 0.12
    assert estimates["n"] == 6000
    assert estimates["propensity_min"] > 0
    assert estimates["propensity_max"] < 1


def test_causal_simulation_rejects_too_small_sample():
    try:
        simulate_treatment_effect(n=50)
    except ValueError as exc:
        assert "at least 200" in str(exc)
    else:
        raise AssertionError("Expected invalid causal simulation size to be rejected")
