import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def Henon_Graph(results):
    df = pd.DataFrame(results)

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="#8B0000")  # NaN entries will be dark red

    pivot_table = df.pivot(index="a", columns="b", values="l1")

    plt.figure(figsize=(20, 20))
    ax = sns.heatmap(pivot_table, cmap=cmap, center=0, cbar_kws={'label': 'Largest Lyapunov Exponent λ1'})
    cbar = ax.collections[0].colorbar
    cbar.set_label('Largest Lyapunov Exponent λ1', fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel("a parameter", labelpad=10, fontsize=20)
    plt.ylabel("b parameter", labelpad=10, fontsize=20)
    plt.title("Lyapunov Exponent Map with Diverging Regions in Dark Red", fontsize=24)
    plt.savefig("Plots/lyapunov_map_largest.png", dpi=500)
    plt.show()

    cmap = plt.cm.coolwarm.copy()
    cmap.set_bad(color="#191970")  # NaN entries will be dark blue

    pivot_table = df.pivot(index="a", columns="b", values="l2")

    plt.figure(figsize=(20, 20))
    # Assuming 'ax' and 'cbar' from sns.heatmap
    ax = sns.heatmap(pivot_table, cmap=cmap, center=0, cbar_kws={'label': 'Largest Lyapunov Exponent λ2'})
    cbar = ax.collections[0].colorbar
    # Increase label font size
    cbar.set_label('Smallest Lyapunov Exponent λ2', fontsize=20)
    # Increase tick label font size
    cbar.ax.tick_params(labelsize=16)
    plt.xlabel("a parameter", labelpad=10, fontsize=20)
    plt.ylabel("b parameter", labelpad=10, fontsize=20)
    plt.title("Lyapunov Exponent Map with Diverging Regions in Dark Blue", fontsize=24)
    plt.savefig("Plots/lyapunov_map_smallest.png", dpi=500)
    plt.show()