import numpy as np

def expected_value_discrete(x, p):
    x_arr = np.array(x)
    p_arr = np.array(p)

    if x_arr.shape != p_arr.shape:
        raise ValueError("Shapes of x and p must match.")

    if not np.isclose(np.sum(p_arr), 1.0, atol=1e-6):
        raise ValueError("Probabilities must sum to 1.")

    return float(np.sum(x_arr * p_arr))
