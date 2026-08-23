"""
Photonic Matrix Multiplier Simulator (MZI Mesh & SVD Decomposition)  [ORIGINAL / 原始版本]
"""
import numpy as np

def mzi_transfer_matrix(theta: float, phi: float) -> np.ndarray:
    return np.array([
        [-np.exp(1j * phi) * np.sin(theta), np.cos(theta)],
        [ np.exp(1j * phi) * np.cos(theta), np.sin(theta)]
    ], dtype=complex)

def embed_mzi(n_modes: int, m: int, theta: float, phi: float) -> np.ndarray:
    T = np.eye(n_modes, dtype=complex)
    T[m:m+2, m:m+2] = mzi_transfer_matrix(theta, phi)
    return T

def decompose_unitary_to_mzi_mesh(U: np.ndarray):
    N = U.shape[0]
    U_curr = np.copy(U).astype(complex)
    mzi_list = []
    for col in range(N - 1):
        for row in range(N - 1, col, -1):
            m = row - 1
            x = U_curr[m, col]
            y = U_curr[row, col]
            if np.abs(y) < 1e-15:
                theta = 0.0
                phi = 0.0
            elif np.abs(x) < 1e-15:
                theta = np.pi / 2.0
                phi = 0.0
            else:
                r = np.abs(x / y)
                alpha = np.angle(-x / y)
                phi = -alpha
                theta = np.arctan(r)
            T_mat = embed_mzi(N, m, theta, phi)
            U_curr = T_mat @ U_curr
            mzi_list.append((m, theta, phi))
    diag_phases = np.diag(U_curr)
    return mzi_list, diag_phases

class PhotonicMatrixProcessor:
    def __init__(self, M: np.ndarray):
        self.M = M.astype(float)
        self.N = M.shape[0]
        U, S, Vt = np.linalg.svd(self.M)
        self.U_target = U
        self.S_target = S
        self.Vt_target = Vt
        self.vt_mzis, self.vt_phases = decompose_unitary_to_mzi_mesh(self.Vt_target)
        self.u_mzis, self.u_phases = decompose_unitary_to_mzi_mesh(self.U_target)

    def forward_optical_simulation(self, x, add_noise=False, noise_std=0.02):
        E = x.astype(complex).copy()
        E = E * self.vt_phases
        for (m, theta, phi) in reversed(self.vt_mzis):
            if add_noise:
                theta += np.random.normal(0, noise_std)
                phi += np.random.normal(0, noise_std)
            T = embed_mzi(self.N, m, theta, phi)
            E = T.conj().T @ E
        E = E * self.S_target
        E = E * self.u_phases
        for (m, theta, phi) in reversed(self.u_mzis):
            if add_noise:
                theta += np.random.normal(0, noise_std)
                phi += np.random.normal(0, noise_std)
            T = embed_mzi(self.N, m, theta, phi)
            E = T.conj().T @ E
        return E.real
