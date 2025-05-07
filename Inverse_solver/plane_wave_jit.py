import numpy as np
from numba import njit, prange

# -----------------------------------------------------------------------------
#  Numba‐jitted core routine
# -----------------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _evaluate_plane_wave(pvs, pols, epsilon, mu, omega, X):
    """
    pvs:   (M,3)  float64  unit propagation vectors
    pols:  (M,)   float64  polarization angles [rad]
    epsilon, mu, omega: scalars
    X:     (N,3)  float64  eval points

    returns: E_fields, H_fields of shape (M, N, 3), complex128
    """
    M = pvs.shape[0]
    N = X.shape[0]

    # precompute
    wavenumber = omega * np.sqrt(epsilon * mu)
    eta        = np.sqrt(mu / epsilon)

    # allocate outputs
    E_fields = np.empty((M, N, 3), dtype=np.complex128)
    H_fields = np.empty((M, N, 3), dtype=np.complex128)

    # loop over plane waves (parallelized)
    for i in prange(M):
        kx, ky, kz = pvs[i, 0], pvs[i, 1], pvs[i, 2]
        pol        = pols[i]

        # rotation to align k in x–z plane
        # original code did: phi = arctan2(-ky, -kx)
        phi  = np.arctan2(-ky, -kx)
        cphi = np.cos(phi)
        sphi = np.sin(phi)

        # after rotation, kz stays the same
        # theta = arccos(–kz)
        theta = np.arccos(-kz)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)

        # inner loop over evaluation points
        for j in range(N):
            Xx, Xy, Xz = X[j, 0], X[j, 1], X[j, 2]

            # rotate the point into k–aligned frame
            x_rot =  cphi * Xx - sphi * Xy
            z_rot =  Xz

            # phase factor
            phase = np.exp(-1j * wavenumber * (x_rot * stheta - z_rot * ctheta))

            # build the two basis fields in that frame
            # E_perp = [0, ϕ, 0],   E_par = [ϕ·ctheta, 0, ϕ·stheta]
            # H_perp = [–ϕ·ctheta, 0, –ϕ·stheta]/η,   H_par = [0, ϕ, 0]/η
            # then superpose with polarization angle
            Exr = np.sin(pol) * (phase * ctheta)
            Eyr = np.cos(pol) * phase
            Ezr = np.sin(pol) * (phase * stheta)

            Hxr = np.cos(pol) * (-phase * ctheta / eta)
            Hyr = np.sin(pol) * ( phase       / eta)
            Hzr = np.cos(pol) * (-phase * stheta / eta)

            # rotate back into lab frame (R_z @ [Exr,Eyr,Ezr])
            E_fields[i, j, 0] =  cphi * Exr + sphi * Eyr
            E_fields[i, j, 1] = -sphi * Exr + cphi * Eyr
            E_fields[i, j, 2] =  Ezr

            H_fields[i, j, 0] =  cphi * Hxr + sphi * Hyr
            H_fields[i, j, 1] = -sphi * Hxr + cphi * Hyr
            H_fields[i, j, 2] =  Hzr

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
