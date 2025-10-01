
import numpy as np
from Lyapunov import Lyapunov_Exponent_Henon
from graphing import Henon_Graph
from tqdm import tqdm

a_vals = np.linspace(1.2, 1.5, 61)
b_vals = np.linspace(0.15, 0.35, 41)

results = []

for a in tqdm(a_vals):
    for b in b_vals:
        lyap = Lyapunov_Exponent_Henon(a, b)
        results.append({"a": np.round(a,3), "b": np.round(b,3), "l1": lyap[0], "l2": lyap[1]})

Henon_Graph(results)

