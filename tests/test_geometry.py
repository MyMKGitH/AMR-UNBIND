import numpy as np


def test_normalized_direction():
    vector = np.array([3.0, 4.0, 0.0])
    unit = vector / np.linalg.norm(vector)
    assert np.isclose(np.linalg.norm(unit), 1.0)
