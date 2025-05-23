import numpy as np
from numba import njit, prange

# -----------------------------------------------------------------------------
#  Numba‐jitted core routine for plane wave evaluation
# -----------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _evaluate_plane_wave(pvs, pols, epsilon, mu, omega_array, X):
    M = pvs.shape[0]
    N = X.shape[0]

    # allocate outputs
    E_fields = np.empty((M, N, 3), dtype=np.complex128)
    H_fields = np.empty((M, N, 3), dtype=np.complex128)

    for i in prange(M):
        omega = omega_array[i]
        wavenumber = omega * np.sqrt(epsilon * mu)
        eta = np.sqrt(mu / epsilon)

        kx, ky, kz = pvs[i, 0], pvs[i, 1], pvs[i, 2]
        pol = pols[i]

        phi = np.arctan2(-ky, -kx)
        cphi = np.cos(phi)
        sphi = np.sin(phi)

        theta = np.arccos(-kz)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)

        for j in range(N):
            Xx, Xy, Xz = X[j, 0], X[j, 1], X[j, 2]

            x_rot = cphi * Xx - sphi * Xy
            z_rot = Xz

            phase = np.exp(-1j * wavenumber * (x_rot * stheta - z_rot * ctheta))

            Exr = np.sin(pol) * (phase * ctheta)
            Eyr = np.cos(pol) * phase
            Ezr = np.sin(pol) * (phase * stheta)

            Hxr = np.cos(pol) * (-phase * ctheta / eta)
            Hyr = np.sin(pol) * (phase / eta)
            Hzr = np.cos(pol) * (-phase * stheta / eta)

            E_fields[i, j, 0] = cphi * Exr + sphi * Eyr
            E_fields[i, j, 1] = -sphi * Exr + cphi * Eyr
            E_fields[i, j, 2] = Ezr

            H_fields[i, j, 0] = cphi * Hxr + sphi * Hyr
            H_fields[i, j, 1] = -sphi * Hxr + cphi * Hyr
            H_fields[i, j, 2] = Hzr

    return E_fields, H_fields

@njit(parallel=True, fastmath=True)
def _evaluate_reflected_field_core(pvs, pols, wavenumbers, eta1, eta2, n1, n2, X):
    M = pvs.shape[0]
    N = X.shape[0]
    E_ref = np.zeros((M, N, 3), dtype=np.complex128)
    H_ref = np.zeros((M, N, 3), dtype=np.complex128)

    for i in prange(M):
        wavenumber = wavenumbers[i]

        kx, ky, kz = pvs[i]
        pol = pols[i]

        theta_i = np.arccos(-kz)
        sin_theta_i = np.sqrt(kx**2 + ky**2)
        sin_theta_t = (n1 / n2) * sin_theta_i
        cos_theta_t = np.sqrt(1 - sin_theta_t**2) if sin_theta_t <= 1.0 else 0.0

        rs = (eta2 * np.cos(theta_i) - eta1 * cos_theta_t) / (eta2 * np.cos(theta_i) + eta1 * cos_theta_t)
        rp = (eta1 * np.cos(theta_i) - eta2 * cos_theta_t) / (eta1 * np.cos(theta_i) + eta2 * cos_theta_t)

        kz_r = -kz
        phi = np.arctan2(-ky, -kx)
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        theta = np.arccos(-kz_r)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)

        for j in range(N):
            Xx, Xy, Xz = X[j]

            x_rot = cphi * Xx - sphi * Xy
            z_rot = Xz
            phase = np.exp(-1j * wavenumber * (x_rot * stheta - z_rot * ctheta))

            Exr = np.sin(pol) * rp * (phase * ctheta)
            Eyr = np.cos(pol) * rs * phase
            Ezr = np.sin(pol) * rp * (phase * stheta)

            Hxr = np.cos(pol) * (-rs * phase * ctheta / eta1)
            Hyr = np.sin(pol) * (rp * phase / eta1)
            Hzr = np.cos(pol) * (-rs * phase * stheta / eta1)

            E_ref[i, j, 0] = cphi * Exr + sphi * Eyr
            E_ref[i, j, 1] = -sphi * Exr + cphi * Eyr
            E_ref[i, j, 2] = Ezr

            H_ref[i, j, 0] = cphi * Hxr + sphi * Hyr
            H_ref[i, j, 1] = -sphi * Hxr + cphi * Hyr
            H_ref[i, j, 2] = Hzr

    return E_ref, H_ref

class Plane_wave:
    def __init__(self, propagation_vectors, polarizations, epsilon, mu, omega):
        propagation_vectors = np.asarray(propagation_vectors, dtype=np.float64)
        polarizations = np.asarray(polarizations, dtype=np.float64)
        omega = np.asarray(omega, dtype=np.float64)

        if propagation_vectors.ndim != 2 or propagation_vectors.shape[1] != 3:
            raise ValueError(f"propagation_vectors must be Mx3. Got {propagation_vectors.shape}")
        if polarizations.ndim != 1 or polarizations.shape[0] != propagation_vectors.shape[0]:
            raise ValueError("polarizations must be 1D, same length as propagation_vectors")
        if omega.ndim != 1 or omega.shape[0] != propagation_vectors.shape[0]:
            raise ValueError("omega must be a 1D array with one value per plane wave")

        if not np.all((polarizations >= 0) & (polarizations <= np.pi / 2)):
            raise ValueError("polarizations must lie in [0, π/2]")
        norms = np.linalg.norm(propagation_vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise ValueError("All propagation_vectors must be unit-length")

        if not np.isscalar(epsilon) or not np.isscalar(mu):
            raise TypeError("epsilon and mu must be scalars")

        self.propagation_vectors = propagation_vectors
        self.polarizations = polarizations
        self.epsilon = float(epsilon)
        self.mu = float(mu)
        self.omega = omega
        self.wavenumber = self.omega * np.sqrt(self.epsilon * self.mu)
        self.eta = np.sqrt(self.mu / self.epsilon)

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

def evaluate_incident_plus_reflected(plane_wave, epsilon_substrate, X):
    epsilon1 = plane_wave.epsilon
    mu1 = plane_wave.mu
    eta1 = plane_wave.eta
    n1 = np.sqrt(epsilon1 * mu1)

    epsilon2 = epsilon_substrate
    mu2 = mu1
    eta2 = np.sqrt(mu2 / epsilon2)
    n2 = np.sqrt(epsilon2 * mu2)

    E_inc, H_inc = plane_wave.evaluate_at_points(X)

    E_ref, H_ref = _evaluate_reflected_field_core(
        plane_wave.propagation_vectors,
        plane_wave.polarizations,
        plane_wave.wavenumber,
        eta1,
        eta2,
        n1,
        n2,
        X
    )

    return E_inc + E_ref, H_inc + H_ref
