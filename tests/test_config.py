from amr_unbind.config import SimulationConfig


def test_pull_steps_are_positive_and_expected():
    cfg = SimulationConfig(pull_distance_nm=0.25, pull_velocity_nm_per_ps=0.005)
    assert cfg.pull_steps == 25000
    assert cfg.pull_time_ps == 50.0


def test_config_roundtrip():
    cfg = SimulationConfig(seed=123)
    clone = SimulationConfig.from_dict(cfg.to_dict())
    assert clone == cfg
