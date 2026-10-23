from shopmini.smoke import cpu_smoke


def test_actual_sampling_gradient_update_and_reload(tmp_path):
    result = cpu_smoke(tmp_path)
    assert result["updates"] > 0
    assert result["final_probabilities"][0] > result["initial_probabilities"][0]
    assert result["checkpoint_reload"]
