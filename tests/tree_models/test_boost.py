import numpy as np
import pandas as pd
from entrax.tree_models.boost import BinaryBoost, MultiBoost

def test_sigmoid():
    bb = BinaryBoost.__new__(BinaryBoost)
    # Test extreme values using clipping
    z = np.array([-1000, 0, 1000], dtype=np.float64)
    s = bb.sigmoid(z)
    expected = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    assert np.allclose(s, expected, atol=1e-6)

def test_log_loss():
    bb = BinaryBoost.__new__(BinaryBoost)
    # For y = [0,1] and predictions p = [0.1, 0.9]
    p = np.array([0.1, 0.9], dtype=np.float64)
    y = np.array([0, 1], dtype=np.float64)
    loss = bb.loss(p, y)
    # Expected loss: -log(0.9)
    expected = -np.log(0.9)
    assert np.isclose(loss, expected)

def test_softmax():
    mb = MultiBoost.__new__(MultiBoost)
    # Test softmax function
    z = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    s = mb.softmax(z)
    expected = np.array([0.09003057, 0.24472847, 0.66524096], dtype=np.float64)
    assert np.allclose(s, expected, atol=1e-6)