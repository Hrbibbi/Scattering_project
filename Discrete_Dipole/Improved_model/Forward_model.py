import numpy as np
from numba import njit, prange
import numpy as np

@njit(parallel=True)
def evaluate_plane_wave(pv, pol, epsilon_air, mu, omega, X):
    """
    Evaluate incident plane waves (no reflection) at evaluation points.

    Parameters:
    pv:     (3,) float64 - common unit propagation vector (should be a unit vector)
    pol:    float64 - polarization angle [rad], in [0, π/2]
    epsilon_air: float - permittivity of air
    mu:     float - permeability
    omega:  (R,) float64 - frequency of each plane wave
    X:      (N,3) float64 - evaluation points

    Returns:
    E_fields, H_fields: (R, N, 3) complex128 - incident fields only
    """
    R = omega.shape[0]
    N = X.shape[0]

    eta = np.sqrt(mu / epsilon_air)

    E_fields = np.zeros((R, N, 3), dtype=np.complex128)
    H_fields = np.zeros((R, N, 3), dtype=np.complex128)

    # Normalize and extract propagation vector
    k = pv / np.sqrt(np.sum(pv**2))
    kx, ky, kz = k[0], k[1], k[2]
    kxyx, kxyy = -kx, -ky

    if kxyx == 0.0 and kxyy == 0.0:
        phi = 0.0
    else:
        phi = np.arctan2(kxyy, kxyx)

    cphi = np.cos(phi)
    sphi = np.sin(phi)

    R_z = np.array([
        [ cphi,  sphi, 0.0],
        [-sphi,  cphi, 0.0],
        [ 0.0,   0.0,  1.0]
    ], dtype=np.complex128)
    R_inv = R_z.T

    # Rotate k to xz plane
    k_rot = R_z @ k.astype(np.complex128)
    theta = np.arccos(-k_rot[2])
    ctheta = np.cos(theta)
    stheta = np.sin(theta)

    for r in prange(R):
        wavenumber = omega[r] * np.sqrt(epsilon_air * mu)

        for i in range(N):
            Xx, Xy, Xz = X[i]
            x_rot = cphi * Xx + sphi * Xy
            z_rot = Xz

            # Incident phase
            phase_inc = np.exp(-1j * wavenumber * (x_rot * stheta - z_rot * ctheta))

            # Basis incident fields (in rotated frame)
            E_perp = np.array([0.0, phase_inc, 0.0], dtype=np.complex128)
            E_par  = np.array([-phase_inc * ctheta, 0.0, -phase_inc * stheta], dtype=np.complex128)

            H_perp = np.array([phase_inc * ctheta, 0.0, phase_inc * stheta], dtype=np.complex128) / eta
            H_par  = np.array([0.0, phase_inc, 0.0], dtype=np.complex128) / eta

            # Combine using polarization
            E_lab = np.cos(pol) * E_perp + np.sin(pol) * E_par
            H_lab = np.cos(pol) * H_perp + np.sin(pol) * H_par

            # Rotate to lab frame
            E_fields[r, i] = R_inv @ E_lab
            H_fields[r, i] = R_inv @ H_lab

    return E_fields, H_fields


def poynting_far_field_polarized_light(direction,alpha_tensor,points,
                                       prop_vec, pol, epsilon,mu, omegas):
    
    E_new,_=evaluate_plane_wave(prop_vec,pol,epsilon,mu,omegas,points) #shape R,M,3

    z_hat=np.array([0,0,1]) 
    directions=np.einsum('mij,j->mi', alpha_tensor, z_hat) #shape M,3
    excitations=np.einsum('rmk,mk->rm', E_new, directions)
    dipole_moments=excitations[:,:,None] * directions[None,:,:] #shape R,M,3

    #-----------------------------------------
    # Phase computation
    #-----------------------------------------

    k=omegas*np.sqrt(epsilon*mu)
    point_project=np.einsum("k,mk->m",direction,points)  #shape M
    pairwise_difference=np.einsum("m,n->mn",point_project,-point_project) #shape (M,M)
    phase_terms=np.exp(1j*k[:,None,None]*pairwise_difference) #shape (R,M,M)
    constant_factors = -k**3 / (16*np.pi**2*omegas*epsilon) #shape (R,)
    #-----------------------------------------
    # moment terms
    #-----------------------------------------

    # Dipole moment projections
    moment_direction_project = np.einsum("k,rmk->rm", direction, dipole_moments)  # (R, M)
    moment_moment_project = np.einsum("rmi, rnj -> rmn", dipole_moments, np.conj(dipole_moments))  # (R, M, M)
    outer_product = moment_direction_project[:, :, None] * np.conj(moment_direction_project[:, None, :])  # (R, M, M)
    scalar_term = outer_product - moment_moment_project  # (R, M, M)

    # Sum over i, j
    total_sum = constant_factors * np.einsum("rij,rij->r", scalar_term, phase_terms)  # (R,)

    # Final vector
    S = total_sum[:, None] * direction[None, :]  # (R, 3)
    S = 1/2*np.real(np.einsum("rk,k->r",S,direction)) #(R)
    return S

def poynting_far_field_unpolarized_light(direction,alpha_tensor,points,prop_vec,epsilon,mu,omegas,numangles):

    polarizations=np.linspace(0,np.pi/2,numangles)
    poyntings=np.array([poynting_far_field_polarized_light(direction,alpha_tensor,points,prop_vec,pol,epsilon,mu,omegas) 
    for pol in polarizations]) # numangles,R
    mean_poynting=np.mean(poyntings,axis=0) # R
    if np.sum(mean_poynting)==0:
        return mean_poynting
    else:
        mean_poynting /= np.linalg.norm(mean_poynting)
        return mean_poynting

def protusion_gen(height=50, width=250, voxel_size=7):
    """
    Generate a 3D voxel-based ellipsoidal protrusion extending from the z=0 plane.

    Parameters:
    - height: float, height of the ellipsoid in z-direction.
    - width: float, full width of the base (x and y).
    - voxel_size: float, spacing between voxel grid points.

    Returns:
    - points: (N, 3) array of XYZ coordinates inside the ellipsoid.
    """

    # Semi-axes
    a = width / 2
    c = height

    # Grid ranges
    x = np.arange(-a, a + voxel_size, voxel_size)
    y = np.arange(-a, a + voxel_size, voxel_size)
    z = np.arange(0, c + voxel_size, voxel_size)

    # Create 3D meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Ellipsoid mask
    ellipsoid_mask = (X / a) ** 2 + (Y / a) ** 2 + (Z / c) ** 2 <= 1

    # Select points inside the ellipsoid
    X_inside = X[ellipsoid_mask]
    Y_inside = Y[ellipsoid_mask]
    Z_inside = Z[ellipsoid_mask]

    # Stack into (N, 3) array
    points = np.stack([X_inside, Y_inside, Z_inside], axis=1)

    return points
