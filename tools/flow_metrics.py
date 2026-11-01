import numpy as np


def metrics(expected, actual, absolute=1e-3, relative=1e-4):
    if expected.shape != actual.shape or not np.isfinite(actual).all():
        return dict(passed=False, reason="shape mismatch or nonfinite output")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    max_abs = float(np.abs(delta).max(initial=0))
    rel_l2 = float(np.linalg.norm(delta) / max(np.linalg.norm(expected.astype(np.float64)), 1e-30))
    return dict(max_abs=max_abs, relative_l2=rel_l2, passed=max_abs <= absolute and rel_l2 <= relative)
