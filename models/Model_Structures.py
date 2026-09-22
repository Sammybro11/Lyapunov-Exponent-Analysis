import numpy as np
import torch

# Discrete Maps

def Create_Henon_Map(a, b, batched = False):
    # Creates a discrete function of a Henon Map for a given parameters

    def Henon_Map(coords):
        if abs(coords[0]) > 1e100 or abs(coords[1]) > 1e100:
            return np.array([np.nan, np.nan])
        x_next = 1 - a * np.power(coords[0], 2) + coords[1]
        y_next = b*coords[0]

        return np.array([x_next, y_next])

    def Batched_Map(coords, a_batch, b_batch):
        x = coords[:, 0]
        y = coords[:, 1]
        x_next = 1.0 - a_batch * x * x + y
        y_next = b_batch * y
        return torch.stack([x_next, y_next], dim = 1)

    if batched:
        return Batched_Map
    else:
        return Henon_Map

def Create_Peter_Map(a, b, c, d):
    # Canonical Peter de Jong map:
    # x' = sin(a*y) - cos(b*x), y' = sin(c*x) - cos(d*y)

    def Peter_Jong(coords):
        x_next = np.sin(a * coords[1]) - np.cos(b * coords[0])
        y_next = np.sin(c * coords[0]) - np.cos(d * coords[1])

        return np.array([x_next, y_next])
    return Peter_Jong
