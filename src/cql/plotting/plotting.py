
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_heatmap(paths: list[list[tuple[float, float]]], heatmap_dim: float, bins: int = 25, log_scale: bool = True):
    trajectories = np.zeros((bins, bins))

    for path in paths:
        if path is None:
            continue
        for (x, y) in path:
            x_int = int((x / heatmap_dim) * bins)
            y_int = int((y / heatmap_dim) * bins)
            if 0 <= x_int < bins and 0 <= y_int < bins:
                trajectories[y_int, x_int] += 1

    plt.figure(figsize=(6,6))
    norm = LogNorm() if log_scale else None
    plt.imshow(trajectories, cmap='hot', interpolation='nearest', origin='lower',
               extent=[0, heatmap_dim, 0, heatmap_dim], norm=norm)
    plt.colorbar(label='Visit count')
    plt.title(f"Heatmap ({bins}x{bins} grid)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
