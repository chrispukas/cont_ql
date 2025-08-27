
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_heatmap(paths: list[list[tuple[float, float]]], 
                 heatmap_dim: float,
                 bins: int = 25, 
                 interpolation_step: float=0.2,
                 log_scale: bool = True) -> None:
    trajectories = np.ones((bins, bins))

    for path in paths:
        if path is None or len(path) < 2:
            continue
        for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
            steps = np.linspace(0, 1, num=int(1 / interpolation_step))
            for t in steps:
                x = x0 + (x1 - x0) * t
                y = y0 + (y1 - y0) * t
                x_int = int((x / heatmap_dim) * bins)
                y_int = int((y / heatmap_dim) * bins)
                if 0 <= x_int < bins and 0 <= y_int < bins:
                    trajectories[y_int, x_int] += 1

    print(trajectories)
    plt.figure(figsize=(6,6))
    norm = LogNorm() if log_scale else None
    plt.imshow(trajectories, cmap='hot', interpolation='nearest', origin='lower',
               extent=[0, heatmap_dim, 0, heatmap_dim], norm=norm)
    plt.colorbar(label='Visit count')
    plt.title(f"Heatmap ({bins}x{bins} grid)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

def plot_epochs(x_vals, y_vals):
    plt.plot(x_vals, np.log(np.array(y_vals)))

    # Labels and title
    plt.xlabel("# of Epochs")
    plt.ylabel("# of Steps")
    plt.title("Epoch-Steps Plot")
    plt.legend()

    # Show the graph
    plt.grid(True)
    plt.show()