#-----------------------------------------------------------------------------
#                                   Imports
#-----------------------------------------------------------------------------
#import Hertzian_dipole_speedup as HD
import Hertzian_dipole_jit as HD
import C2_surface
import plane_wave_jit as PW
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from numba import njit, prange
from os import environ
#Computer only has 4 cores
environ['OMP_NUM_THREADS']='4'
environ['MPI_NUM_THREADS']='4'
environ['MKL_NUM_THREADS']='4'
environ['OPENBLAS_NUM_THREADS']='4'

#This could be done where we dont construct the 2xMxNx3 array but imediatly construct the matrix from planewaves:
def construct_RHSs(Surface, propagation_vectors, polarizations, epsilon_air, mu, omegas):
    '''
    Constructs RHS matrix for multiple incident plane waves, ready for least squares solvers.

    Output:
        rhs_matrix: (4*N) x M array
                    - Each column corresponds to the RHS for one plane wave.
                    - Each block of N rows corresponds to:
                        [0:N]   → -E ⋅ tau1
                        [N:2N]  → -E ⋅ tau2
                        [2N:3N] → -H ⋅ tau1
                        [3N:4N] → -H ⋅ tau2

    Input:
        Surface: object with attributes
            - points: Nx3 array
            - tau1: Nx3 array (tangent vector 1)
            - tau2: Nx3 array (tangent vector 2)
        propagation_vectors: Mx3 array
        polarizations: M array
        epsilon_air: scalar
        mu: scalar
        omegas: M array
    '''
    points = Surface.points
    tau1 = Surface.tau1
    tau2 = Surface.tau2
    
    # Evaluate fields for all M plane waves
    planewaves = PW.Plane_wave(propagation_vectors, polarizations, epsilon_air, mu, omegas)
    E_all, H_all = planewaves.evaluate_at_points(points)  # (2, M, N, 3)

    # Compute tangential components → shape (M, N)
    b1 = -np.einsum("mnj,nj->mn", E_all, tau1)
    b2 = -np.einsum("mnj,nj->mn", E_all, tau2)
    b3 = -np.einsum("mnj,nj->mn", H_all, tau1)
    b4 = -np.einsum("mnj,nj->mn", H_all, tau2)

    b=np.hstack([b1,b2,b3,b4]) #Stack columns to get a (M_planewaves,4*N points matrix)
    rhs_matrix=b.T # Transpose it to get the desired (4*N, M) matrix

    return rhs_matrix

#Similarly this could avoid constructing Hertizan dipole array
def construct_sub_column(dipoles, Surface):
    '''
    Computes a column block of the MAS matrix corresponding to a set of dipoles.

    Input:
        dipoles
        Surface: Surface object with:
            - points: (N, 3)
            - tau1, tau2: (N, 3) tangent vectors
            - N: number of test points

    Output:
        (4*N, M) matrix — tangential components of E and H fields from each dipole
    '''
    points = Surface.points
    tau1 = Surface.tau1
    tau2 = Surface.tau2
    N = points.shape[0]

    # Evaluate E and H fields
    E_all, H_all = dipoles.evaluate_at_points(points)  # (M_dipoles, N_points, 3)

    # Compute projections using einsum
    E_tau1 = np.einsum('nmj,mj->nm', E_all, tau1)  # (M_dipoles, N_points)
    E_tau2 = np.einsum('nmj,mj->nm', E_all, tau2)
    H_tau1 = np.einsum('nmj,mj->nm', H_all, tau1)
    H_tau2 = np.einsum('nmj,mj->nm', H_all, tau2)

    # Stack the projections: shape (4*N, M_dipoles)
    sub_column = np.vstack([
        E_tau1.T,
        E_tau2.T,
        H_tau1.T,
        H_tau2.T
    ])

    return sub_column

def construct_matrix(Surface, inneraux, outeraux, mu, air_epsilon, scatter_epsilon, omega):
    '''
    Constructs the full 4x4 block MAS matrix using HertzianDipole class instances.

    Returns:
        MAS matrix: shape (4*M, 4*N)
        Dipole instances: intDP1, intDP2, extDP1, extDP2
    '''
    #----------------------------------------
    # Extract geometry
    #----------------------------------------
    inner_points = inneraux.points
    inner_tau1 = inneraux.tau1
    inner_tau2 = inneraux.tau2
    outer_points = outeraux.points
    outer_tau1 = outeraux.tau1
    outer_tau2 = outeraux.tau2

    #----------------------------------------
    # Create Hertzian Dipole instances
    #----------------------------------------
    intDP1 = HD.Hertzian_Dipole(inner_points, inner_tau1, mu, air_epsilon, omega)
    intDP2 = HD.Hertzian_Dipole(inner_points, inner_tau2, mu, air_epsilon, omega)
    extDP1 = HD.Hertzian_Dipole(outer_points, outer_tau1, mu, scatter_epsilon, omega)
    extDP2 = HD.Hertzian_Dipole(outer_points, outer_tau2, mu, scatter_epsilon, omega)
    #----------------------------------------
    # Construct block columns
    #----------------------------------------
    
    Col1 = construct_sub_column(intDP1, Surface)
    Col2 = construct_sub_column(intDP2, Surface)
    Col3 = construct_sub_column(extDP1, Surface)
    Col4 = construct_sub_column(extDP2, Surface)

    #----------------------------------------
    # Assemble MAS matrix
    #----------------------------------------
    MAS = np.column_stack((Col1, Col2, Col3, Col4))

    return MAS, intDP1, intDP2, extDP1, extDP2
    
def Construct_solve_MAS_system(Scatter_information, Incident_information, plot=False, reduce_grid=True,print_solution_times=False):
    '''
    Solves the MAS system for multiple plane wave excitations (same omega, mu, epsilon).

    Output:
        int_coeffs: [C1, C2] arrays of shape (M, R) for interior dipoles in tau1 and tau2 (M dipoles, R planewaves)
        ext_coeffs: [C3, C4] arrays of shape (N, M) for exterior dipoles in tau1 and tau2 (M dipoles, R planewaves)
        InteriorDipoles: [intDP1, intDP2] — input dipole data
        ExteriorDipoles: [extDP1, extDP2] — input dipole data

    Input:
        Scatter_information: dict containing surface + permittivity/permeability info
        Incident_information: dict containing *lists* of propagation_vectors and polarizations
                              and shared epsilon, mu, omega
    '''
    #-------------------------------------------------------------
    # Unpack scattering info
    #-------------------------------------------------------------
    Surface = Scatter_information['Surface']
    inneraux = Scatter_information['inneraux']
    outeraux = Scatter_information['outeraux']
    scatter_epsilon = Scatter_information['epsilon']
    scatter_mu = Scatter_information['mu']

    #-------------------------------------------------------------
    # Unpack incident info
    #-------------------------------------------------------------
    propagation_vectors = Incident_information['propagation_vectors']  # shape (R, 3)
    polarizations = Incident_information['polarizations']              # shape (R,)
    epsilon_air = Incident_information['epsilon']
    incident_mu = Incident_information['mu']
    omega = Incident_information['omega']
    lam = Incident_information['lambda']

    #-------------------------------------------------------------
    # Reduce surface information to necessary wavelength
    #-------------------------------------------------------------

    if reduce_grid:
        Surface, inneraux, outeraux = C2_surface.Set_dipoles_pr_WL(Surface,inneraux,outeraux,lam)
    M=np.shape(inneraux.points)[0]
    N=np.shape(Surface.points)[0]
    R=len(polarizations)
    if print_solution_times:
        print(f"M: {M}, N: {N}, R: {R}")
    #-------------------------------------------------------------
    # Construct RHS matrix and MAS matrix
    #-------------------------------------------------------------
    con_time = time.time()
    RHS_matrix = construct_RHSs(Surface, propagation_vectors, polarizations, epsilon_air, incident_mu, omega)
    construct_matrix(Surface,inneraux,outeraux,scatter_mu,epsilon_air,scatter_epsilon,omega)
    MAS_matrix, intDP1, intDP2, extDP1, extDP2 = construct_matrix(
        Surface, inneraux, outeraux, scatter_mu, epsilon_air, scatter_epsilon, omega
    )
    if print_solution_times:
        print(f"Number of RHS: {np.shape(RHS_matrix)[1]}")
        print(f"Construction time: {time.time() - con_time:.3f} s")
    #print(f"Matrix shape: {MAS_matrix.shape}, RHS shape: {RHS_matrix.shape}")

    #-------------------------------------------------------------
    # Solve system for all RHS
    #-------------------------------------------------------------

    sol_start = time.time()
    C_matrix, *_ = np.linalg.lstsq(MAS_matrix, RHS_matrix, rcond=None) #solution size 4MxR
    C1 = C_matrix[:M]
    C2 = C_matrix[M:2*M]
    C3 = C_matrix[2*M:3*M]
    C4 = C_matrix[3*M:]
    solution_time=time.time()-sol_start
    if print_solution_times:
        print(f"Solution time: {solution_time:.3f} s")

    #-------------------------------------------------------------
    # Optional plotting
    #-------------------------------------------------------------
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        im0 = axs[0].imshow(np.abs(MAS_matrix), aspect='auto', cmap='viridis')
        axs[0].set_title('abs(MAS_matrix)')
        plt.colorbar(im0, ax=axs[0])
        
        axs[1].imshow(np.abs(RHS_matrix), cmap='viridis')
        axs[1].set_title('abs(RHS matrix)')
        axs[1].set_xlabel('Incident Wave Index')
        
        axs[2].imshow(np.abs(C_matrix), cmap='viridis')
        axs[2].set_title('abs(Solution C)')
        axs[2].set_xlabel('Incident Wave Index')
        
        plt.tight_layout()
        plt.show()

    #-------------------------------------------------------------
    # Return values
    #-------------------------------------------------------------
    int_coeffs = [C1, C2]  # each shape (N, M)
    ext_coeffs = [C3, C4]
    InteriorDipoles = [intDP1, intDP2]
    ExteriorDipoles = [extDP1, extDP2]

    return int_coeffs, ext_coeffs, InteriorDipoles, ExteriorDipoles, solution_time


def blas_einsum4(amnd, mr):
    # amnd: (A, M, N, D) → reshape to (A*M, N*D), mr: (M, R)
    A, M, N, D = amnd.shape
    # Step 1: for each a, stack M rows of length N*D
    flat = amnd.transpose(0,2,3,1).reshape(A*N*D, M)   # (A*N*D, M)
    # Step 2: matmul → (A*N*D, M) @ (M, R) = (A*N*D, R)
    out_flat = flat.dot(mr)                            # BLAS call
    # Step 3: reshape back to (A, R, N, D)
    return out_flat.reshape(A, N, D, -1).transpose(0,3,1,2)

def compute_scattered_field_at_point_jit(points, int_coeff, InteriorDipoles,print_solution_times=False):
    """
    Jitted version of compute_scattered_field_at_point that returns
    total_field of shape (2, R, N, 3), separating E & H.
    """
    C_1, C_2 = int_coeff                  # each shape (M, R)
    intDP1, intDP2 = InteriorDipoles

    # Evaluate dipole fields: each is (2, M, N, 3)
    evals1 = intDP1.evaluate_at_points(points)
    evals2 = intDP2.evaluate_at_points(points)

    # JIT’d contractions:
    einsum_time=time.time()
    total_1 = blas_einsum4(evals1, C_1)  # (2, R, N, 3)
    total_2 = blas_einsum4(evals2, C_2)  # (2, R, N, 3)
    if print_solution_times:
        print(f"einsum time: {time.time()-einsum_time}")
    # Sum contributions from both dipole sets, per field type:
    total_field = total_1 + total_2       # (2, R, N, 3)

    return total_field

def compute_flux_integral_scattered_field(plane, int_coeff, InteriorDipoles,plot_first_integrand=False,print_solution_times=False):
    '''
    Computes the average power (flux) integral for the scattered field for multiple RHSs.

    Input:
        plane: A C2_object (with .points and .normals)
        int_coeff: List of dipole weights [C_1, C_2], each (M_dipoles, R_incident configs)
        dipoles: List of dipole classes [intDP1, intDP2]

    Output:
        flux_values: Array of shape (R,) — power flux per RHS
    '''
    if print_solution_times:
        print("\n Integration start")
    int_start=time.time()
    #---------------------------------------------------------------------
    # Extract geometry
    #---------------------------------------------------------------------
    points = plane.points             # (N, 3)
    normals = plane.normals           # (N, 3)

    dx = np.linalg.norm(points[1] - points[0])
    dA = dx * dx                      # Scalar area element (uniform)

    #---------------------------------------------------------------------
    # Evaluate scattered fields: (2, M, N, 3)
    #---------------------------------------------------------------------
    eval_time=time.time()
    E, H = compute_scattered_field_at_point_jit(points, int_coeff, InteriorDipoles) #E and H (R,N,3) each
    if print_solution_times:
        print(f"evalutation time in integral: {time.time()-eval_time}")

    integrand_time=time.time()
    R, N , _ = np.shape(E)
    Cross = 0.5 * np.cross(E, np.conj(H)) # (R,N,3)
    integrands = np.einsum("rnk,nk->rn", Cross,normals) # (R,N)
    if print_solution_times:
        print(f"Integrand construction: {time.time()-integrand_time}")
    if plot_first_integrand:
        first_integrand=integrands[0,:]
        N=int(np.sqrt(N))
        plt.imshow(np.abs(np.reshape(first_integrand,(N,N))))
        plt.show()
    #Integral calculation
    integration_time=time.time()
    integrals = np.einsum("rn -> r", integrands*dA)    # (M,)
    if print_solution_times:
        print(f"integration_time: {time.time()-integration_time}")
        print(f"Total_time: {time.time()-int_start}")
    return integrals  # Return real-valued power flux


'''
def compute_scattered_field_at_point(points, int_coeff, InteriorDipoles):
    """
    Compute scattered EM fields at given points due to interior dipoles
    for multiple sets of dipole coefficients (i.e., multiple incident conditions).

    Parameters:
        points : (N, 3) array — evaluation points, (N points)
        int_coeff : [C_1, C_2] — each of shape (M, R), (M dipoles, R incident coefficients)
        InteriorDipoles : [intDP1, intDP2] — dipole objects 

    Returns:
        (2, R, N, 3) array — scattered E and H fields
    """
    C_1, C_2 = int_coeff                      # Shape: (M, R)
    intDP1, intDP2 = InteriorDipoles

    # Evaluate dipole fields: shape (2, M, N, 3)
    evals1 = intDP1.evaluate_at_points(points)  # shape (2, M, N, 3)
    evals2 = intDP2.evaluate_at_points(points)  # shape (2, M, N, 3)

    # Use einsum to perform summation over dipoles with coefficients
    # 'a m n d, m r -> a r n d' where:
    # a = 2 (E, H), m = M (dipoles), n = N (points), d = 3 (vector components), r = R (inputs)
    total_1 = np.einsum('amnd,mr->arnd', evals1, C_1)
    total_2 = np.einsum('amnd,mr->arnd', evals2, C_2)

    # Add contributions from both dipole groups
    total_field = total_1 + total_2  # shape: (2, R, N, 3)
    return total_field
'''
"""
def compute_flux_integral_scattered_field(plane, int_coeff, InteriorDipoles,plot_first_integrand=False,print_solution_times=False):
    '''
    Computes the average power (flux) integral for the scattered field for multiple RHSs.

    Input:
        plane: A C2_object (with .points and .normals)
        int_coeff: List of dipole weights [C_1, C_2], each (M_dipoles, R_incident configs)
        dipoles: List of dipole classes [intDP1, intDP2]

    Output:
        flux_values: Array of shape (R,) — power flux per RHS
    '''
    int_start=time.time()
    #---------------------------------------------------------------------
    # Extract geometry
    #---------------------------------------------------------------------
    points = plane.points             # (N, 3)
    normals = plane.normals           # (N, 3)

    dx = np.linalg.norm(points[1] - points[0])
    dA = dx * dx                      # Scalar area element (uniform)

    #---------------------------------------------------------------------
    # Evaluate scattered fields: (2, M, N, 3)
    #---------------------------------------------------------------------
    E, H = compute_scattered_field_at_point(points, int_coeff, InteriorDipoles) #E and H (R,N,3) each
    
    R, N , _ = np.shape(E)
    Cross = 0.5 * np.cross(E, np.conj(H)) # (R,N,3)
    integrands = np.einsum("rnk,nk->rn", Cross,normals) # (R,N)

    if plot_first_integrand:
        first_integrand=integrands[0,:]
        N=int(np.sqrt(N))
        plt.imshow(np.abs(np.reshape(first_integrand,(N,N))))
        plt.show()
    #Integral calculation
    integrals = np.einsum("rn -> r", integrands*dA)    # (M,)
    if print_solution_times:
        print(f"integration_time: {time.time()-int_start}")
    return integrals  # Return real-valued power flux
"""
def Single_scatter_solver(Scatter_information, Incident_configurations, options):
    """
    Solves the forward problem for a specified scatterer and multiple sets of incident waves sorted by wavelength.

    Returns
    -------
    flux_integrals : np.ndarray
        Real part of the flux integrals for each plane wave configuration (flattened).
    """
    show_MAS         = options.get('show_MAS', False)
    show_power_curve = options.get('Show_power_curve', False)
    plane_z          = options.get('plane_location', None)
    print_solution_times = options.get('print_solution_times',False)

    Surface  = Scatter_information['Surface']
    pts      = Surface.points
    x        = pts[:,0]
    a, b     = np.min(x), np.max(x)
    N_grid   = int(np.sqrt(pts.shape[0]))  # assume N×N grid

    if plane_z is None:
        plane_z = 5 * pts[:,2].max()

    Plane = C2_surface.generate_plane_xy(plane_z, a, b, 50)

    all_flux = []
    total_sol_time=0
    for idx, inc_lam in enumerate(Incident_configurations):
        if print_solution_times:
            print(f"\nComputing incident information number {idx+1}/{len(Incident_configurations)}, wavelength: {inc_lam['lambda']:.4f}")
        total_time = time.time()

        int_coeffs, _, InteriorDipoles, _, solution_time = Construct_solve_MAS_system(
            Scatter_information,
            inc_lam,
            plot=show_MAS,
            print_solution_times=print_solution_times
        )
        total_sol_time +=solution_time

        power_ints = compute_flux_integral_scattered_field(
            plane=Plane,
            InteriorDipoles=InteriorDipoles,
            int_coeff=int_coeffs,
            print_solution_times=False
        )  # shape: (M_block,)

        all_flux.append(np.real(power_ints))  # collect real parts

        if show_power_curve:
            plt.figure()
            plt.plot(np.abs(all_flux), marker='o')
            plt.xlabel('Incident index')
            plt.ylabel('Power integral')
            plt.title('Scattered Power vs. Incident Wave')
            plt.tight_layout()
            plt.show()
        if print_solution_times:
            print(f"total time: {time.time() - total_time:.2f} seconds")
    if print_solution_times:
        print(f"total time used on solving: {total_sol_time}")
    return np.concatenate(all_flux)  # shape: (total_M_block,)

def Multiple_scatter_solver(Scatter_configurations, Incident_configurations, options):
    """
    Solves the forward problem for a set of scatterers and incident wave configurations.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing averaged flux integrals across all scatterers.
    """
    all_fluxes = []
    completion_time=time.time()
    for i,Scatter in enumerate(Scatter_configurations):
        print(f"\n\nSolving realization {i+1}/{len(Scatter_configurations)}")
        flux = Single_scatter_solver(Scatter, Incident_configurations, options)
        all_fluxes.append(flux)
    # Stack into shape (n_scatterers, total_M_block)
    flux_array = np.stack(all_fluxes, axis=0)  # shape: (S, T)
    mean_flux = np.mean(flux_array, axis=0)    # shape: (T,)
    # Build metadata from the first scatterer config
    first_config = Incident_configurations
    records = []
    for inc in first_config:
        props = inc['propagation_vectors']
        pols  = inc['polarizations']
        omega = inc['omega']
        lam   = inc['lambda']
        freq  = omega / (2*np.pi)

        for j in range(props.shape[0]):
            records.append({
                'propagation_vector': props[j,:],
                'polarization'      : pols[j],
                'wavelength'        : lam,
                'frequency'         : freq,
                'mean_simulated_fluxintegral': mean_flux[len(records)]
            })
    config=0
    for inc in Incident_configurations:
        config+=len(inc['polarizations'])
    completion_time=time.time()-completion_time
    print(f"\n\nCompletion time: {completion_time}")
    print(f"Number of forward problems solved: {len(Scatter_configurations)*config}")
    print(f"Average time pr forward solve: {completion_time/(len(Scatter_configurations)*config)}")
    return pd.DataFrame.from_records(records)