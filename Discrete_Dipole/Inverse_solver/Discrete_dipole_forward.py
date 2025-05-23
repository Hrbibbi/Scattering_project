import plane_wave_jit_multifreq as PW
import Domains as DS
import numpy as np
from numba import njit, prange

@njit(parallel=True)
def batched_cross(a, b):
    """
    Compute cross product between a (1, N, M, 3) and b (R, 1, M, 3) complex arrays.
    Returns an array of shape (R, N, M, 3) with complex values.
    """
    R = b.shape[0]
    N = a.shape[1]
    M = a.shape[2]

    result = np.empty((R, N, M, 3), dtype=np.complex128)

    for r in prange(R):
        for n in range(N):
            for m in range(M):
                ax, ay, az = a[0, n, m, 0], a[0, n, m, 1], a[0, n, m, 2]
                bx, by, bz = b[r, 0, m, 0], b[r, 0, m, 1], b[r, 0, m, 2]

                result[r, n, m, 0] = ay * bz - az * by
                result[r, n, m, 1] = az * bx - ax * bz
                result[r, n, m, 2] = ax * by - ay * bx

    return result





def compute_scattered_field_at_points(points,Incident_waves,domain,epsilon_substrate):
    '''
    Computes the scattered fields from multiple Incident waves using the discrete dipole method

    Input:
        points: a Nx3 array with the points to evaluate the scattered field at
        Incident waves: Instance of plane_wave class storing information about the incident waves
        domain: Instance of domain class storing information about the position of dipoles, and related alpha_tensors

    Output:
        A 2xMxNx3 array with the with F[0,i,j,k]: k'th component of E^{scat} for incident wave i at point j 
                                      F[1,i,j,k]: '---------------' H^{scat} '----------------------------'                              
    '''
    
    #-----------------------------------------------------------------
    # Distance calculations
    #-----------------------------------------------------------------

    #compute r_xi vector with distances between point located x=[x_x,x_y,x_z] and the dipole number i's position
    #resulting size is NxMx3, N points, M dipoles, 3 spatial components
    # Also in this calculation we get the scalar values |r_xi| which is a NxM array of the distances
    
    print(f"Computing distance vectors")
    r_vec = points[:, None, :] - domain.points[None, :, :]  # (N, M, 3)
    r_mag = np.linalg.norm(r_vec, axis=2)  # (N, M)
    r_hat = r_vec / r_mag[...,None] # (N,M,3)
    print(f"size of distance vectors: {np.shape(r_hat)}")
    N,M,_=np.shape(r_hat)
    #-----------------------------------------------------------------
    # E,H field calculation
    #-----------------------------------------------------------------

    #Resulting arrays are size (R_incident,M_dipoles,3)
    print(f"Computing incident fields")
    E_new, H_new = PW.evaluate_incident_plus_reflected(Incident_waves,epsilon_substrate,domain.points)
    print(f"size of incident fields: {np.shape(E_new)}")
    #-----------------------------------------------------------------
    # polarization calculation
    #-----------------------------------------------------------------

    alpha=domain.alpha_tensor # a (M,3,3) numpy array 

    #Compute the matrix vector product P_{r,i} = alpha_i @ E_new_{r,i} 
    #resulting in R,M,3 array polarization vector for dipole i given incident wave R
    print(f"Compution polarization tensors")
    Pol_tensor = np.einsum('mij,rmj->rmi',alpha,E_new) #R,M,3
    print(f"size of polarization tensor: {np.shape(Pol_tensor)}")
    R,_,_=np.shape(Pol_tensor)
    #-----------------------
    
    #compute phase part
    print(f"Computing phase and cross products")
    print(f"expected size: {(R,N,M,3)} resulting memory use: {16*R*N*M*3/(1024**2)} MB")
    wave_numbers=Incident_waves.wavenumber #(R) array
    phase=np.exp(1j*wave_numbers[:,None,None]*r_mag[None,:,:]) #resulting in R,N,M
    phase /= r_mag[None,:,:]

    # Expand dimensions to align shapes
    r_hat_exp = r_hat[None, :, :, :]       # shape: (1, N, M, 3)
    P_exp = Pol_tensor[:, None, :, :]      # shape: (R, 1, M, 3)

    hat_cross_p = batched_cross(r_hat_exp, P_exp)
    hat_hat_cross_p = batched_cross(r_hat_exp, hat_cross_p)

    print(f"size of hat_hat_cross_p: {np.shape(hat_hat_cross_p)}")
    print(f"Computing total fields")
    fields=wave_numbers[:,None,None,None]**2 *hat_hat_cross_p*phase[...,None] #R,N,M,3 array
    print(f"Contracting fields")
    E_scat=np.sum(fields,axis=2) #contract along the dipoles to get sum shape: (R,N,3)


def scattered_field_measurement(Incident_waves,domain,epsilon_substrate):
    """
    Computes the scattered field measurements to be used in the forward solver.

    returns a array 
    """

    #Compute the scattered field
    E_scat,H_scat=compute_scattered_field_at_points

    #do measurement calculation


def test():
    #create domain
    X,Y,Z=DS.cylinder_cartesian_grid(1,1,numpoints_xy=10,numpoints_z=10)
    chol_fact=DS.matern_covariance_matrix(X,Y,Z)
    alpha=DS.generate_tensor_sample(chol_fact)
    domain=DS.Domain(X,Y,Z,alpha)
    
    k=np.array([0,0,-1])
    number=100
    propagation_vector = np.tile(k, (number, 1))
    betas=np.zeros(number)
    PW1=PW.Plane_wave(propagation_vector,betas,1,1,np.ones_like(betas))
    x0=np.linspace(-1,1,10)
    y0=np.linspace(-1,1,10)
    X,Y=np.meshgrid(x0,y0)
    Z=2*np.ones_like(X)
    test_points=np.column_stack([X.ravel(),Y.ravel(),Z.ravel()])
    for i in range(100):
        compute_scattered_field_at_points(test_points,PW1,domain,2)


test()