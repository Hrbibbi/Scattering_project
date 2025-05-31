import plane_wave_jit as PW
import Domains as DS
import numpy as np
from numba import njit, prange
from scipy.spatial.distance import pdist

@njit(parallel=True)
def _poynting_far_field_jit(direction, dipole_moments, points, k, omega, epsilon):
    R, M, _ = dipole_moments.shape

    # Constant factor
    constant_factors = -k**3 / (16 * np.pi**2 * omega * epsilon)

    # Project points
    point_project = np.zeros(M)
    for m in range(M):
        for i in range(3):
            point_project[m] += direction[i] * points[m, i]

    # Phase terms
    phase_terms = np.zeros((R, M, M), dtype=np.complex128)
    for r in range(R):
        for m in range(M):
            for n in range(M):
                phase = k[r] * (point_project[n] - point_project[m])
                phase_terms[r, m, n] = np.exp(1j * phase)

    # Scalar terms
    scalar_term = np.zeros((R, M, M), dtype=np.complex128)
    for r in range(R):
        for m in range(M):
            dot_m = 0.0
            dot_n = 0.0
            for i in range(3):
                dot_m += direction[i] * dipole_moments[r, m, i]
                dot_n += direction[i] * dipole_moments[r, n, i]
            outer = dot_m * np.conj(dot_n)

            inner = 0.0
            for i in range(3):
                inner += dipole_moments[r, m, i] * np.conj(dipole_moments[r, n, i])

            scalar_term[r, m, n] = outer - inner

    # Total sum
    total_sum = np.zeros(R, dtype=np.complex128)
    for r in range(R):
        for m in range(M):
            for n in range(M):
                total_sum[r] += scalar_term[r, m, n] * phase_terms[r, m, n]

    # Final Poynting vector
    S = np.zeros((R, 3), dtype=np.complex128)
    for r in range(R):
        for i in range(3):
            S[r, i] = constant_factors[r] * total_sum[r] * direction[i]

    return S

def poynting_far_field(direction, domain: DS.Domain, Incident_waves: PW.Plane_wave):
    points = domain.points
    E_new, _ = Incident_waves.evaluate_at_points_reflection(points)
    alpha = domain.alpha_tensor
    dipole_moments = np.einsum('mij,rmj->rmi', alpha, E_new)

    return _poynting_far_field_jit(
        direction.astype(np.float64),
        dipole_moments.astype(np.complex128),
        points.astype(np.float64),
        Incident_waves.wavenumber.astype(np.float64),
        Incident_waves.omega.astype(np.float64),
        float(Incident_waves.epsilon),
    )
 
def testing():
    import time
    X,Y,Z=DS.cylinder_cartesian_grid(1,2)
    factor=DS.matern_covariance_matrix(X,Y,Z)
    R=100
    propagation_vector = np.tile([0, 0, -1], (R, 1))
    beta=np.tile(0,(R))
    omega=np.tile(1,(R))
    incident_wave=PW.Plane_wave(propagation_vector,beta,1,2.56,1,omega)
    direction=np.array([0,0,1])
    tot_time=time.time()
    for i in range(100):
        iter_time=time.time()
        alpha,angles=DS.generate_rotation_tensor_sample(factor)
        domain=DS.Domain(X,Y,Z,alpha,angles)
        poynting_far_field(direction,domain,incident_wave)
        print(f"iter_time {time.time()-iter_time}")
    print(f"total_time {time.time()-tot_time}")
testing()