import pandas as pd
import numpy as np
from entrax.tree_models.cart import Cart

def test_gini():
    # Create a DataFrame with balanced binary classes: gini = 0.5
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": [0, 0, 1, 1]
    })
    # Bypass __init__ via __new__
    cart = Cart.__new__(Cart)
    cart.class_column_name = "target"
    gini_val = cart.gini(df, "target")
    assert np.isclose(gini_val, 0.5)

def test_compute_split_gain():
    # This test uses a perfect split:
    # For feature with values [1,2,3,4] and target [0,0,1,1]:
    # Splitting at 2.5 gives left pure (0) and right pure (1); parent's gini = 0.5 so gain = 0.5.
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "target": [0, 0, 1, 1]
    })
    cart = Cart.__new__(Cart)
    cart.class_column_name = "target"
    parent_impurity = cart.gini(df, "target")  # Expected 0.5
    gain = cart.compute_split_gain(df, 2.5, "feature")
    assert np.isclose(gain, parent_impurity)