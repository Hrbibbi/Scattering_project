import plane_wave_jit as PW
import Domains as DS
import numpy as np
from numba import njit, prange
from scipy.spatial.distance import pdist

def poynting_far_field(direction,alpha_tensor,domain: DS.Domain,Incident_waves: PW.Plane_wave):
    '''
    Computes the normalized poynting vector in the far field in some direction

    input:
        direction: (3,) numpy array containing unit direction we compute the vector in
        domain: domain class storing the placement of the dipoles
        dipole_moments: (R,M,3) array containing R different instances of dipole momements for the M dipoles
    
    output:
        (R,3) array with the poynting vector calculation for the R different instances of dipole moments
    '''
    #-----------------------------------------
    # Compute dipole momements
    #-----------------------------------------
    points = domain.points  # shape (M, 3)
    E_new, _ = Incident_waves.evaluate_at_points_reflection(points)  # shape (R, M, 3)
    alpha = alpha_tensor  # shape (M, 3, 3)

    # Step 1: Compute rotated direction vector: v_i = alpha_i @ z
    z_hat = np.array([0.0, 0.0, 1.0])
    rotated_directions = np.einsum('mij,j->mi', alpha, z_hat)  # shape (M, 3)

    # Step 2: Project E onto the rotated direction
    # This gives a scalar for each R and M
    dot_products = np.einsum('rmk,mk->rm', E_new, rotated_directions)  # shape (R, M)

    # Step 3: Multiply the scalar projection with the rotated direction vector
    dipole_moments = dot_products[:, :, None] * rotated_directions[None, :, :]  # shape (R, M, 3)

    M,_=np.shape(points)
    R,_,_=np.shape(dipole_moments)
    #print(f"R: {R}, M: {M}")
    #-----------------------------------------
    # constant factor in poynting
    #-----------------------------------------
    k=Incident_waves.wavenumber
    epsilon=Incident_waves.epsilon
    omega=Incident_waves.omega
    constant_factors = -k**3 / (16*np.pi**2*omega*epsilon) #shape (R,)
    #print(f"constant factors: {np.shape(constant_factors)}")
    #-----------------------------------------
    # Phase computation
    #-----------------------------------------

    point_project=np.einsum("k,mk->m",direction,points)  #shape M (x_hat \cdot x_i)
    pairwise_difference=np.einsum("m,n->mn",point_project,-point_project) #shape (M,M)
    phase_terms=np.exp(1j*k[:,None,None]*pairwise_difference) #shape (R,M,M)
    #print(f"phase term: {np.shape(phase_terms)}")
    #-----------------------------------------
    # moment terms
    #-----------------------------------------
    # Dipole moment projections
    moment_direction_project = np.einsum("k,rmk->rm", direction, dipole_moments)  # (R, M)
    #print(f"moment_direction_project: {np.shape(moment_direction_project)}")
    moment_moment_project = np.einsum("rmi, rnj -> rmn", dipole_moments, np.conj(dipole_moments))  # (R, M, M)
    #print(f"moment_moment_project: {np.shape(moment_moment_project)}")
    # Scalar term
    outer_product = moment_direction_project[:, :, None] * np.conj(moment_direction_project[:, None, :])  # (R, M, M)
    scalar_term = outer_product - moment_moment_project  # (R, M, M)

    # Sum over i, j
    total_sum = np.einsum("rij,rij->r", scalar_term, phase_terms)  # (R,)

    # Final vector
    S = constant_factors[:,None]*(total_sum[:, None] * direction[None, :])  # (R, 3)
    S = 1/2*np.real(np.einsum("rk,k->r",S,direction))
    return S / np.linalg.norm(S)