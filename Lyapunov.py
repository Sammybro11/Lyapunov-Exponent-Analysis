import numpy as np
from Model_Structures import DoublePendulum, Create_Henon_Map
from Helper_Functions import rk4_step, Gram_Schmidt_Orthonormalization
from tqdm import tqdm

def Lyapunov_Exponent_Henon(a, b):
    Map = Create_Henon_Map(a,b)

    init_coordinates = np.array([0.2, 0.2])

    perturbation_vectors = np.eye(2, dtype=float)
    # Each Column is an ortho unit vector in the 2 x 2 id matrix

    sum_log = 0
    transient = 3000
    n_iterations = 5000
    coords = init_coordinates

    for _ in range(transient):
        coords = Map(coords)
        if np.any(np.isnan(coords)):
            return np.array([np.nan, np.nan])

    for _ in range(n_iterations):
        coords = Map(coords)
        if np.any(np.isnan(coords)):
            return np.array([np.nan, np.nan])

        Jacobian = np.array([[-2 * a * coords[0], 1],
                    [b, 0]], dtype=float)
        perturbation = Jacobian @ perturbation_vectors

        Q, R = Gram_Schmidt_Orthonormalization(perturbation)

        diag = np.abs(np.diag(R)) # Norm of each vector before normalization, the "stretch"
        diag[diag < 1e-14] = 1e-14 # Incase vector becomes tiny
        sum_log += np.log(diag)

        perturbation_vectors = Q

    return sum_log/n_iterations





