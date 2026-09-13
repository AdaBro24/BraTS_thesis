import numpy as np

def z_score_normalise(volume: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    mask = volume > 0
    normalised = np.zeros_like(volume, dtype=np.float32)

    mean = volume[mask].mean()
    std = volume[mask].std()

    normalised[mask] = (volume[mask] - mean)/(std + epsilon)

    return normalised

 