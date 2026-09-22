import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
import pandas as pd

def Henon_Graph(results):
    df = pd.DataFrame(results)

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="#965151")  # NaN entries will be dark red

    pivot_table = df.pivot(index="a", columns="b", values="l1")

    plt.figure(figsize=(20, 20))
    ax = sns.heatmap(pivot_table, cmap=cmap, center=0, cbar_kws={'label': 'Largest Lyapunov Exponent λ1'})
    cbar = ax.collections[0].colorbar
    cbar.set_label('Largest Lyapunov Exponent λ1', fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel("b parameter", labelpad=10, fontsize=20)
    plt.ylabel("a parameter", labelpad=10, fontsize=20)
    plt.title("Lyapunov Exponent Map with Diverging Regions in Dark Red", fontsize=24)
    plt.savefig("Plots/lyapunov_map_largest.png", dpi=500)
    plt.show()

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="#4A4A8F")  # NaN entries will be dark blue

    pivot_table = df.pivot(index="a", columns="b", values="l2")

    plt.figure(figsize=(20, 20))
    # Assuming 'ax' and 'cbar' from sns.heatmap
    ax = sns.heatmap(pivot_table, cmap=cmap, center=0, cbar_kws={'label': 'Largest Lyapunov Exponent λ2'})
    cbar = ax.collections[0].colorbar
    # Increase label font size
    cbar.set_label('Smallest Lyapunov Exponent λ2', fontsize=20)
    # Increase tick label font size
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel("b parameter", labelpad=10, fontsize=20)
    plt.ylabel("a parameter", labelpad=10, fontsize=20)
    plt.title("Lyapunov Exponent Map with Diverging Regions in Dark Blue", fontsize=24)
    plt.savefig("Plots/lyapunov_map_smallest.png", dpi=500)
    plt.show()

def Map_Simulation(Map, iterations, number_points, name, title, bounds=(-1.5, 1.5, -0.5, 0.5)):
    np.random.seed(67)
    x_min, x_max, y_min, y_max = bounds
    initial_points = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(number_points, 2))
    time_series = [initial_points.copy()]
    points = initial_points.copy()
    for _ in range(iterations):
        points = np.array([Map(pt) for pt in points])
        time_series.append(points.copy())

    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter(time_series[0][:, 0], time_series[0][:, 1], s=2, c='blue')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Create a "legend" text object, adjust transform for upper right
    legend_text = ax.text(
        x_max-0.05, y_max-0.025, "", va='top', ha='right', fontsize=12, color="darkblue")

    def update(frame):
        pts = time_series[frame]
        scat.set_offsets(pts)
        # Count points within bounds
        mask = (
                (pts[:, 0] >= x_min) & (pts[:, 0] <= x_max) &
                (pts[:, 1] >= y_min) & (pts[:, 1] <= y_max)
        )
        num_in_bounds = np.sum(mask)
        legend_text.set_text(f"Points in bounds: {num_in_bounds}")
        ax.set_title(f'{title}', fontsize=16)
        return scat, legend_text

    ani = FuncAnimation(fig, update, frames=len(time_series), interval=300, blit=True)

    ani.save(f'Plots/{name}.gif', writer='pillow', fps=2)  # Try changing fps

    plt.close(fig)

def Peter_Attractor(Map, iterations, name, title):
    coords = np.array([0.2, 0.2])
    trajectory = np.empty((iterations, 2))
    for i in range(iterations):
        coords = Map(coords)
        trajectory[i] = coords

    plt.figure(figsize=(8, 8))
    plt.scatter(trajectory[:, 0], trajectory[:, 1], s=0.1, c='black', marker='.')
    plt.title(title, fontsize=16)
    plt.savefig(f"Plots/{name}.png", dpi=500)
    plt.close()