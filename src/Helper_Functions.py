import numpy as np

def rk4_step(function, state: np.ndarray, time: float, dt: float = 0.01, structure: np.ndarray = None) -> np.ndarray:
    """
    :param function: Pendulum ODE Equation
    :param state: Pendulum State Space
    :param time: Current Time
    :param dt: Step
    :param structure: Pendulum Structure Variables
    :return: next state array
    """
    k1 = function(state, time, structure)
    k2 = function(state + 0.5 * dt * k1, time + 0.5 * dt, structure)
    k3 = function(state + 0.5 * dt * k2, time + 0.5 * dt, structure)
    k4 = function(state + dt * k3, time + dt, structure)
    return state + (dt/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)

def Gram_Schmidt_Orthonormalization(vector_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns the Gram Schmidt Orthonormalization Matrix and Projection Matrix using Modified Gram Schmidt Method.
    :param vector_matrix: Column Vector Matrix m x n
    :return: tuple(Q, R): Q contains the orthonormal vectors
    and R is such that A = QR, basically upper triangle matrix containing all projection combinations.
    """
    vector_matrix = np.asarray(vector_matrix, dtype=float)
    ortho_matrix = np.zeros_like(vector_matrix, dtype=float)
    projection_matrix = np.zeros((vector_matrix.shape[1], vector_matrix.shape[1]), dtype=float)
    working_matrix = np.copy(vector_matrix)

    for j in range(vector_matrix.shape[1]): # for each column vector
        for i in range(j): # for each column vector before v_j
            projection_matrix[i, j] = np.dot(ortho_matrix[:, i], working_matrix[:, j]) # < q_i , v_j >
            working_matrix[:, j] -= projection_matrix[i, j] * ortho_matrix[:, i] # v_j <- v_j - r_ij * q_i

        # Run a second pass to remove possible floating point errors in the first pass
        for i in range(j):
            correction = np.dot(ortho_matrix[:, i], working_matrix[:, j]) # correction values in projection
            working_matrix[:, j] -= correction * ortho_matrix[:, i]
            projection_matrix[i, j] += correction

        projection_matrix[j, j] = np.linalg.norm(working_matrix[:, j])
        if projection_matrix[j, j] < 1e-14:
            # raise ValueError(f"Column {j} had a linear dependent vector making the remaining vector zero")
            projection_matrix[j, j] = 1e-14
        ortho_matrix[:, j] = working_matrix[:, j]/projection_matrix[j, j]

    return ortho_matrix, projection_matrix



