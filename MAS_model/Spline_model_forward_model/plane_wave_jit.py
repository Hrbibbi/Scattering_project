import numpy as np
from numba import njit, prange

# -----------------------------------------------------------------------------
#  Numba‐jitted core routine
# -----------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _evaluate_plane_wave(pvs, pols, epsilon, mu, omega, X):
    """
    Jitted evaluation of multiple plane waves at evaluation points.
    
    Parameters:
    pvs:    (R,3) float64 - unit propagation vectors
    pols:   (R,)  float64 - polarization angles [rad], in [0, π/2]
    epsilon, mu, omega: scalars
    X:      (N,3) float64 - evaluation points

    Returns:
    E_fields, H_fields: (R, N, 3) complex128 - electric and magnetic field values
    """
    M = pvs.shape[0]
    N = X.shape[0]

    wavenumber = omega * np.sqrt(epsilon * mu)
    eta = np.sqrt(mu / epsilon)

    E_fields = np.zeros((M, N, 3), dtype=np.complex128)
    H_fields = np.zeros((M, N, 3), dtype=np.complex128)

    for i in prange(M):
        k = pvs[i]
        pol = pols[i]
        kx, ky, kz = k[0], k[1], k[2]
        kxyx, kxyy = -kx, -ky

        # Handle edge case where kx = ky = 0
        if kxyx == 0.0 and kxyy == 0.0:
            phi = 0.0
        else:
            phi = np.arctan2(kxyy, kxyx)

        cphi = np.cos(phi)
        sphi = np.sin(phi)

        # Rotation matrix R_z
        R_z = np.array([
            [ cphi,  sphi, 0.0],
            [-sphi,  cphi, 0.0],
            [ 0.0,   0.0,  1.0]
        ])
        R_inv = R_z.T

        # Rotate k
        k_rot = R_z @ k
        theta = np.arccos(-k_rot[2])
        ctheta = np.cos(theta)
        stheta = np.sin(theta)

        # Loop over evaluation points
        for j in range(N):
            Xx, Xy, Xz = X[j]
            # Rotate point
            x_rot = cphi * Xx + sphi * Xy
            z_rot = Xz  # y component irrelevant after rotation

            phase = np.exp(-1j * wavenumber * (x_rot * stheta - z_rot * ctheta))

            # Basis fields
            E_perp = np.array([0.0, phase, 0.0], dtype=np.complex128)
            E_par  = np.array([phase * ctheta, 0.0, phase * stheta], dtype=np.complex128)

            H_perp = np.array([-phase * ctheta, 0.0, -phase * stheta], dtype=np.complex128) / eta
            H_par  = np.array([0.0, phase, 0.0], dtype=np.complex128) / eta

            E_lab = np.cos(pol) * E_perp + np.sin(pol) * E_par
            H_lab = np.sin(pol) * H_perp + np.cos(pol) * H_par

            # Rotate back
            E_fields[i, j] = R_inv.astype(np.complex128) @ E_lab
            H_fields[i, j] = R_inv.astype(np.complex128) @ H_lab

    return E_fields, H_fields


# -----------------------------------------------------------------------------
#  Thin Python wrapper with checks, same class name
# -----------------------------------------------------------------------------
class Plane_wave:
    def __init__(self, propagation_vectors, polarizations, epsilon, mu, omega):
        propagation_vectors = np.asarray(propagation_vectors, dtype=np.float64)
        polarizations       = np.asarray(polarizations,       dtype=np.float64)

        # shape checks
        if propagation_vectors.ndim != 2 or propagation_vectors.shape[1] != 3:
            raise ValueError(f"propagation_vectors must be Mx3. Got {propagation_vectors.shape}")
        if polarizations.ndim != 1 or polarizations.shape[0] != propagation_vectors.shape[0]:
            raise ValueError("polarizations must be 1D, same length as propagation_vectors")

        # ranges and norms
        if not np.all((polarizations >= 0) & (polarizations <= np.pi/2)):
            raise ValueError("polarizations must lie in [0, π/2]")
        norms = np.linalg.norm(propagation_vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise ValueError("All propagation_vectors must be unit‐length")

        # scalar checks
        for name, val in (("epsilon", epsilon), ("mu", mu), ("omega", omega)):
            if not np.isscalar(val):
                raise TypeError(f"{name} must be a scalar")

        # store
        self.propagation_vectors = propagation_vectors
        self.polarizations       = polarizations
        self.epsilon             = float(epsilon)
        self.mu                  = float(mu)
        self.omega               = float(omega)
        self.wavenumber          = self.omega * np.sqrt(self.epsilon * self.mu)
        self.eta                 = np.sqrt(self.mu / self.epsilon)

    def evaluate_at_points(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"X must be N×3. Got {X.shape}")
        return _evaluate_plane_wave(
            self.propagation_vectors,
            self.polarizations,
            self.epsilon,
            self.mu,
            self.omega,
            X
        )
