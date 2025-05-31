import numpy as np
from numba import njit, prange

# -----------------------------------------------------------------------------
#  Numba‐jitted core routine
# -----------------------------------------------------------------------------

@njit(parallel=True)
def _evaluate_plane_wave(pvs, pols, epsilon_air, epsilon_substrate, mu, omega, X):
    """
    Jitted evaluation of multiple plane waves at evaluation points,
    including reflected fields at a dielectric interface z = 0.

    Parameters:
    pvs:    (R,3) float64 - unit propagation vectors
    pols:   (R,)  float64 - polarization angles [rad], in [0, π/2]
    epsilon_air, epsilon_substrate: scalars
    mu:     scalar
    omega:  (R,) float64 - frequency of each planewave
    X:      (N,3) float64 - evaluation points

    Returns:
    E_fields, H_fields: (R, N, 3) complex128 - total (incident + reflected) fields
    """
    R = pvs.shape[0]
    N = X.shape[0]

    eta = np.sqrt(mu / epsilon_air)

    E_fields = np.zeros((R, N, 3), dtype=np.complex128)
    H_fields = np.zeros((R, N, 3), dtype=np.complex128)

    for r in prange(R):
        k = pvs[r]
        pol = pols[r]
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
        ],dtype=np.complex128)
        R_inv = R_z.T

        # Rotate k to xz plane
        k_rot = R_z @ k.astype(np.complex128)
        theta = np.arccos(-k_rot[2])  # incidence angle
        ctheta = np.cos(theta)
        stheta = np.sin(theta)

        wavenumber = omega[r] * np.sqrt(epsilon_air * mu)

        # Fresnel reflection coefficients
        n1 = np.sqrt(epsilon_air * mu)
        n2 = np.sqrt(epsilon_substrate * mu)

        sin_theta = stheta
        sin_theta_t = n1 / n2 * sin_theta
        cos_theta_t_sq = 1.0 - sin_theta_t**2

        cos_theta_t = np.sqrt(cos_theta_t_sq)
        cos_theta_i = ctheta
        r_perp = (n1 * cos_theta_i - n2 * cos_theta_t) / (n1 * cos_theta_i + n2 * cos_theta_t)
        r_par  = (n2 * cos_theta_i - n1 * cos_theta_t) / (n2 * cos_theta_i + n1 * cos_theta_t)

        for i in range(N):
            Xx, Xy, Xz = X[i]
            # Rotate point into wave frame
            x_rot = cphi * Xx + sphi * Xy
            z_rot = Xz

            # Incident phase
            phase_inc = np.exp(-1j * wavenumber * (x_rot * stheta - z_rot * ctheta))

            # Incident basis fields
            E_perp = np.array([0.0, phase_inc, 0.0], dtype=np.complex128)
            E_par  = np.array([-phase_inc * ctheta, 0.0, -phase_inc * stheta], dtype=np.complex128)

            H_perp = np.array([phase_inc * ctheta, 0.0, phase_inc * stheta], dtype=np.complex128) / eta
            H_par  = np.array([0.0, phase_inc, 0.0], dtype=np.complex128) / eta

            # Combine using polarization
            E_lab = np.cos(pol) * E_perp + np.sin(pol) * E_par
            H_lab = np.cos(pol) * H_perp + np.sin(pol) * H_par

            # Reflected phase
            phase_ref = np.exp(-1j * wavenumber * (x_rot * stheta + z_rot * ctheta))

            # Reflected basis fields (z-component flips sign)
            E_perp_ref = np.array([0.0, phase_ref * r_perp, 0.0], dtype=np.complex128)
            E_par_ref  = np.array([-phase_ref * ctheta * r_par, 0.0, phase_ref * stheta * r_par], dtype=np.complex128)

            H_perp_ref = np.array([phase_ref * ctheta * r_perp, 0.0, -phase_ref * stheta * r_perp], dtype=np.complex128) / eta
            H_par_ref  = np.array([0.0, phase_ref * r_par, 0.0], dtype=np.complex128) / eta

            # Combine using polarization
            E_lab_ref = np.cos(pol) * E_perp_ref + np.sin(pol) * E_par_ref
            H_lab_ref = np.cos(pol) * H_perp_ref + np.sin(pol) * H_par_ref

            # Rotate back to lab frame and add
            E_fields[r, i] = R_inv @ (E_lab + E_lab_ref)
            H_fields[r, i] = R_inv @ (H_lab + H_lab_ref)

    return E_fields, H_fields


# -----------------------------------------------------------------------------
#  Thin Python wrapper with checks, same class name
# -----------------------------------------------------------------------------
class Plane_wave:
    def __init__(self, propagation_vectors, polarizations, epsilon, epsilon_substrate, mu, omega):
        propagation_vectors = np.asarray(propagation_vectors, dtype=np.float64)  # shape (R, 3)
        polarizations       = np.asarray(polarizations, dtype=np.float64)        # shape (R,)
        omega               = np.asarray(omega, dtype=np.float64)                # shape (R,)

        R = propagation_vectors.shape[0]

        # --- shape checks ---
        if propagation_vectors.ndim != 2 or propagation_vectors.shape[1] != 3:
            raise ValueError(f"propagation_vectors must be (R, 3). Got {propagation_vectors.shape}")
        if polarizations.shape != (R,):
            raise ValueError(f"polarizations must be shape (R,). Got {polarizations.shape}")
        if omega.shape != (R,):
            raise ValueError(f"omega must be shape (R,). Got {omega.shape}")

        # --- value checks ---
        if not np.all((polarizations >= 0) & (polarizations <= np.pi / 2)):
            raise ValueError("polarizations must lie in [0, π/2]")

        norms = np.linalg.norm(propagation_vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise ValueError("All propagation_vectors must be unit-length")

        for name, val in [("epsilon", epsilon), ("epsilon_substrate", epsilon_substrate), ("mu", mu)]:
            if not np.isscalar(val):
                raise TypeError(f"{name} must be a scalar")

        # --- store values ---
        self.propagation_vectors = propagation_vectors  # (R, 3)
        self.polarizations       = polarizations        # (R,)
        self.omega               = omega                # (R,)
        self.epsilon             = float(epsilon)
        self.epsilon_substrate   = float(epsilon_substrate)
        self.mu                  = float(mu)

        # --- derived quantities ---
        self.wavenumber = omega * np.sqrt(epsilon * mu)  # (R,)
        self.eta        = np.sqrt(mu / epsilon)          # scalar (non-dispersive)


    def evaluate_at_points_reflection(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"X must be N×3. Got {X.shape}")
        return _evaluate_plane_wave(
            self.propagation_vectors,
            self.polarizations,
            self.epsilon,
            self.epsilon_substrate,
            self.mu,
            self.omega,
            X
        )

