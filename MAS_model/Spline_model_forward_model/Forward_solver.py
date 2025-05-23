#-----------------------------------------------------------------------------
#                                   Imports
#-----------------------------------------------------------------------------
import Hertzian_dipole_jit as HD
import plane_wave_jit as PW
import Spline_function as SP
import numpy as np
import time
import matplotlib.pyplot as plt
import json
import pandas as pd
from os import environ
#Computer only has 4 cores
environ['OMP_NUM_THREADS']='4'
environ['MPI_NUM_THREADS']='4'
environ['MKL_NUM_THREADS']='4'
environ['OPENBLAS_NUM_THREADS']='4'

def construct_RHSs(Surface, propagation_vectors, polarizations, epsilon_air, mu, omega):
    '''
    Constructs RHS matrix for multiple incident plane waves, ready for least squares solvers.

    Output:
        rhs_matrix: (4*N) x R array
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
        propagation_vectors: Rx3 array
        polarizations: R array
        epsilon_air: scalar
        mu: scalar
        omega: scalar
    '''

    # Extract geometry
    points = Surface.points
    tau1 = Surface.tau1
    tau2 = Surface.tau2
    
    # Evaluate fields for all R plane waves
    planewaves = PW.Plane_wave(propagation_vectors, polarizations, epsilon_air, mu, omega)
    E_all, H_all = planewaves.evaluate_at_points(points)  # 2 (R, N, 3) arrays

    # Compute tangential components → shape (R,N)
    b1 = -np.einsum("rnk,nk->rn", E_all, tau1)
    b2 = -np.einsum("rnk,nk->rn", E_all, tau2)
    b3 = -np.einsum("rnk,nk->rn", H_all, tau1)
    b4 = -np.einsum("rnk,nk->rn", H_all, tau2)

    b=np.hstack([b1,b2,b3,b4]) #Stack horizontal to get a (R_planewaves,4*N points matrix)
    rhs_matrix=b.T # Transpose it to get the desired (4*N, R) matrix

    return rhs_matrix

def construct_sub_column(dipoles, Surface):
    '''
    Computes a column block of the MAS matrix corresponding to a set of dipoles.

    Input:
        dipoles: A dipole class corresponding to M hertzian dipoles
        Surface: Surface object with:
            - points: (N, 3)
            - tau1, tau2: (N, 3) tangent vectors
            - N: number of test points

    Output:
        (4*N, M) matrix — tangential components of E and H fields from each dipole
    '''
    
    #extract geometry
    points = Surface.points
    tau1 = Surface.tau1
    tau2 = Surface.tau2

    # Evaluate E and H fields
    E_all, H_all = dipoles.evaluate_at_points(points)  # 2 (M_dipoles, N_points, 3) arrays

    # Compute projections using einsum
    E_tau1 = np.einsum('mnk,nk->mn', E_all, tau1)  # (M_dipoles, N_points)
    E_tau2 = np.einsum('mnk,nk->mn', E_all, tau2)
    H_tau1 = np.einsum('mnk,nk->mn', H_all, tau1)
    H_tau2 = np.einsum('mnk,nk->mn', H_all, tau2)

    # Stack the projections: shape (4*N, M_dipoles)
    sub_column = np.vstack([
        E_tau1.T, # (M,N) transpose to get (N,M)
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
    MAS = np.column_stack((Col1, Col2, -Col3, -Col4))

    return MAS, intDP1, intDP2, extDP1, extDP2

def Construct_solve_MAS_system(Scatter_information, Incident_information):
    '''
    Solves the MAS system for multiple plane wave excitations (same omega, lambda, mu, epsilon). for all planewaves

    Input:
        Scatter_information: dict continain information on the scattering object contains
            'SPSurface' : Instance of the spline surface class
            'epsilon'   : Permetivity of the scatterer
        Incident_information: dict containing information about incident waves assumed to have same wavelength
            'propagation_vectors': (R,3) array of propagation vectors
            'polarizations'      : (R) array of polarizations angles
            'epsilon'            : permetivity of the air
            'mu'                 : permeability of the air
            'omega'              : angular frequency of the incident waves
            'lambda'             : wavelength of the the incident waves
          of propagation_vectors and polarizations
                              and shared epsilon, mu, omega
    Output:
        int_coeffs: [C1, C2] arrays of shape (M, R) for interior dipoles in tau1 and tau2 (M dipoles, R planewaves)
        ext_coeffs: [C3, C4] arrays of shape (N, M) for exterior dipoles in tau1 and tau2 (M dipoles, R planewaves)
        InteriorDipoles: [intDP1, intDP2] — input dipole data
        ExteriorDipoles: [extDP1, extDP2] — input dipole data
    '''
    #-------------------------------------------------------------
    # Unpack scattering info
    #-------------------------------------------------------------
    SPSurface = Scatter_information['SPSurface'] #Spline surface class
    scatter_epsilon = Scatter_information['epsilon'] #Scatter epsilon

    #-------------------------------------------------------------
    # Unpack incident info
    #-------------------------------------------------------------
    propagation_vectors = Incident_information['propagation_vectors']  # shape (R, 3)
    polarizations = Incident_information['polarizations']              # shape (R,)
    epsilon_air = Incident_information['epsilon']
    mu = Incident_information['mu']
    omega = Incident_information['omega']
    lam = Incident_information['lambda']

    #-------------------------------------------------------------
    # Construct testpoints and auxilary surfaces
    #-------------------------------------------------------------
    Surface,inneraux,outeraux=SPSurface.sample_surface_MAS(lam)
    M=np.shape(inneraux.points)[0]
    N=np.shape(Surface.points)[0]
    R=len(polarizations)
    print(f"M: {M}, N: {N}, R: {R}")

    #-------------------------------------------------------------
    # Construct RHS matrix and MAS matrix
    #-------------------------------------------------------------
    con_time = time.time()
    RHS_matrix = construct_RHSs(Surface, propagation_vectors, polarizations, epsilon_air, mu, omega)
    MAS_matrix, intDP1, intDP2, extDP1, extDP2 = construct_matrix(Surface,inneraux,outeraux,mu,epsilon_air,scatter_epsilon,omega)


    print(f"Construction time: {time.time() - con_time:.3f} s")
    print(f"Number of RHS: {np.shape(RHS_matrix)[1]}")
    print(f"Matrix shape: {MAS_matrix.shape}, RHS shape: {RHS_matrix.shape}")

    #-------------------------------------------------------------
    # Solve system for all RHS
    #-------------------------------------------------------------

    sol_start = time.time()
    C_matrix, *_ = np.linalg.lstsq(MAS_matrix, RHS_matrix, rcond=None) #solution size 4MxR
    C1 = C_matrix[:M,:] #C1 is the first M rows
    C2 = C_matrix[M:2*M,:] #C2 is the next M rows
    C3 = C_matrix[2*M:3*M,:] #C3 is next M rows
    C4 = C_matrix[3*M:,:] #C3 is the next M rows
    print(f"Solution time: {time.time() - sol_start:.3f} s")

    #-------------------------------------------------------------
    # Optional plotting
    #-------------------------------------------------------------
        
    #-------------------------------------------------------------
    # Return values
    #-------------------------------------------------------------
    int_coeffs = [C1, C2]  # each shape (M, R)
    ext_coeffs = [C3, C4]
    InteriorDipoles = [intDP1, intDP2]
    ExteriorDipoles = [extDP1, extDP2]

    return int_coeffs, ext_coeffs, InteriorDipoles, ExteriorDipoles

def compute_linear_combinations(points, coeffs, dipoles):
    """
    compute the EM fields at given points using the found coeffs and dipoles
    for multiple sets of dipole coefficients (i.e., multiple incident conditions).

    Parameters:
        points : (N, 3) array — evaluation points, (N points)
        coeffs : [C1, C2] — each of shape (M, R), (M dipoles, R incident coefficients)
        dipoles : [DP1, DP2] — dipole objects 

    Returns:
        (2, R, N, 3) array — E and H fields
    """
    C_1, C_2 = coeffs                      # Shape: (M, R)
    DP1, DP2 = dipoles

    # Evaluate dipole fields: shape (2, M, N, 3)
    evals1 = DP1.evaluate_at_points(points)  # shape (2, M, N, 3)
    evals2 = DP2.evaluate_at_points(points)  # shape (2, M, N, 3)

    # Use einsum to perform summation over dipoles with coefficients
    # 'a m n d, m r -> a r n d' where:
    # a = 2 (E, H), m = M (dipoles), n = N (points), d = 3 (vector components), r = R (inputs)
    total_1 = np.einsum('amnk,mr->arnk', evals1, C_1)
    total_2 = np.einsum('amnk,mr->arnk', evals2, C_2)

    # Add contributions from both dipole groups
    total_field = total_1 + total_2  # shape: (2, R, N, 3)
    return total_field

def compute_flux_integral_scattered_field(plane, int_coeff, InteriorDipoles):
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
    E, H = compute_linear_combinations(points, int_coeff, InteriorDipoles) #E and H (R,N,3) each
    
    R, N , _ = np.shape(E)
    Cross = 0.5 * np.cross(E, np.conjugate(H)) # (R,N,3)

    integrands = np.einsum("rnk,nk->rn", Cross,normals) # (R,N)
    #Integral calculation
    integrals = np.einsum("rn -> r", integrands*dA)    # (R,)
    print(f"integration_time: {time.time()-int_start}")
    return integrals 

def Single_scatter_solver(Scatter_information, Incident_configurations):
    """
    Solves the forward problem for a specified scatterer and multiple sets of incident waves sorted by wavelength.

    Returns:
        DataFrame with columns:
            - wavelength: float
            - propagation_vector: (3,) ndarray
            - polarization: float
            - flux: float
    """
    SPSurface = Scatter_information['SPSurface']
    all_data = []

    z_height = 10 * np.max(SPSurface.z_fine) + 1
    Plane = SP.generate_plane_xy(z_height, SPSurface.a, SPSurface.b, 30)

    for idx, inc_lam in enumerate(Incident_configurations):
        print(f"\nComputing incident information number {idx+1}/{len(Incident_configurations)}, wavelength: {inc_lam['lambda']:.4f}")
        total_time = time.time()

        # Solve MAS system
        int_coeffs, _, InteriorDipoles, _ = Construct_solve_MAS_system(Scatter_information, inc_lam)

        # Compute power flux for each incident wave
        flux_values = compute_flux_integral_scattered_field(Plane, int_coeffs, InteriorDipoles)  # shape: (R,)

        # Store data for each plane wave (R incident waves per config)
        wavelength = inc_lam['lambda']
        propagation_vectors = inc_lam['propagation_vectors']  # shape: (R, 3)
        polarizations = inc_lam['polarizations']              # shape: (R,)

        for i in range(len(flux_values)):
            all_data.append({
                'wavelength': wavelength,
                'propagation_vector': propagation_vectors[i],
                'polarization': polarizations[i],
                'flux': flux_values[i].real  # drop imaginary part if negligible
            })

        print(f"Total time: {time.time() - total_time:.3f} s")

    # Create dataframe
    df = pd.DataFrame(all_data, columns=['wavelength', 'propagation_vector', 'polarization', 'flux'])
    return df       
        
    


def check_transmission_conditions(Scatter_information,Incident_information):
    int_coeffs, ext_coeffs, InteriorDipoles, ExteriorDipoles=Construct_solve_MAS_system(Scatter_information,Incident_information)
    SPSurface=Scatter_information['SPSurface']
    testpoints, tau_1, tau_2, _=SPSurface.construct_auxiliary_points(10,0)

    propagation_vectors = Incident_information['propagation_vectors']  # shape (R, 3)
    polarizations = Incident_information['polarizations']              # shape (R,)
    epsilon_air = Incident_information['epsilon']
    mu = Incident_information['mu']
    omega = Incident_information['omega']
    lam = Incident_information['lambda']
    planewaves = PW.Plane_wave(propagation_vectors, polarizations, epsilon_air, mu, omega)

    E_scat,H_scat=compute_linear_combinations(testpoints,int_coeffs,InteriorDipoles) #(R,N,3)
    E_tot,H_tot  =compute_linear_combinations(testpoints,ext_coeffs,ExteriorDipoles) #(R,N,3)   
    E_inc,H_inc  =planewaves.evaluate_at_points(testpoints)                          #(R,N,3)
    M1=np.einsum("rnk,nk->rn",E_scat-E_tot+E_inc,tau_1) # (R,N)
    M2=np.einsum("rnk,nk->rn",E_scat-E_tot+E_inc,tau_2) # (R,N)
    M3=np.einsum("rnk,nk->rn",H_scat-H_tot+H_inc,tau_1) # (R,N)
    M4=np.einsum("rnk,nk->rn",H_scat-H_tot+H_inc,tau_2) # (R,N)
    M=np.hstack([M1,M2,M3,M4])[0] #(R,4N) each row is now the error for incident wave r
    I1=np.einsum("rnk,nk->rn",-E_inc,tau_1) # (R,N) #changed order of tau
    I2=np.einsum("rnk,nk->rn",-E_inc,tau_2) # (R,N)
    I3=np.einsum("rnk,nk->rn",-H_inc,tau_1) # (R,N)
    I4=np.einsum("rnk,nk->rn",-H_inc,tau_2) # (R,N)
    I=np.hstack([I1,I2,I3,I4])[0]

    ST1=np.einsum("rnk,nk->rn",E_scat-E_tot,tau_1) # (R,N)
    ST2=np.einsum("rnk,nk->rn",E_scat-E_tot,tau_2) # (R,N)
    ST3=np.einsum("rnk,nk->rn",H_scat-H_tot,tau_1) # (R,N)
    ST4=np.einsum("rnk,nk->rn",H_scat-H_tot,tau_2) # (R,N)
    ST=np.hstack([ST1,ST2,ST3,ST4])[0] #(R,4N) each row is now the error for incident wave r
    #plt.plot(ST,'-o',label='Scat-tot',)
    #plt.plot(I,'-o', label='inc')
    #plt.legend()
    #plt.show()
    return np.linalg.norm(M,2) /np.linalg.norm(I,2)

def create_surface_and_scattering_info_from_json(json_path):
    with open(json_path, 'r') as f:
        params = json.load(f)

    width = params['halfWidth_x']
    resol = params['resolution']
    alpha = params['alpha']
    bump_params = params['bumpData']
    scatter_epsilon = params['epsilon1']
    mu = 1  # Assumed constant

    # -------------------------
    # Surface creation
    # -------------------------
    a, b = -width, width
    X0 = np.linspace(a, b, resol)
    Y0 = np.linspace(a, b, resol)
    X, Y = np.meshgrid(X0, Y0)

    def bump(x, y, x0, y0, height, sigma):
        return height * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def surface_function(x, y):
        return sum(
            bump(x, y, b['x0'], b['y0'], b['height'], b['sigma'])
            for b in bump_params
        )

    Z = surface_function(X, Y)
    Z = np.zeros_like(X)
    for s in np.linspace(10,0.5,10):
        Surface=SP.SplineSurface(X,Y,Z,smoothness=s)
        if False:
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            cont1 = axs[0].contourf(X, Y, Z, levels=100, cmap='viridis')
            axs[0].set_title("Original Surface")
            axs[0].set_aspect('equal')

            #Surface.plot_surface_with_vectors(resolution=8)
            points,_,_,_=Surface.construct_auxiliary_points(100,0)
            X_aux,Y_aux,Z_aux=points[:,0],points[:,1],points[:,2]
            X_aux,Y_aux,Z_aux=np.reshape(X_aux,[100,100]),np.reshape(Y_aux,[100,100]),np.reshape(Z_aux,[100,100])

            cont2 = axs[1].contourf(X_aux, Y_aux, Z_aux, levels=100, cmap='viridis')
            axs[1].set_title("Spline Surface (Auxiliary Points)")
            axs[1].set_aspect('equal')
            cbar = fig.colorbar(cont2, ax=axs, orientation='vertical', shrink=0.9, pad=0.05)
            cbar.set_label("Z-value")
            plt.show()
        Scatterinformation = {
            'SPSurface': Surface,
            'epsilon': scatter_epsilon,
            'mu': mu
        }

        # -------------------------
        # Incident information
        # -------------------------
        max_errors=[]
        wavelengths=[]
        compute_times=[]
        for omega in np.linspace(np.pi,4*np.pi,10):
            k = np.array(params['k'])
            betas = np.array(params['betas'])
            betas= np.linspace(0,np.pi/2,1)
            epsilon_air = 1
            number = len(betas)

            propagation_vector = np.tile(k, (number, 1))
            polarization = betas
            #omega = params['omega']
            wavelength = 2 * np.pi / omega
            wavelengths.append(2/wavelength)
            Incidentinformation = {
                'propagation_vectors': propagation_vector,
                'polarizations': polarization,
                'epsilon': epsilon_air,
                'mu': mu,
                'lambda': wavelength,
                'omega': omega
            }
            start=time.time()
            relative_norm=check_transmission_conditions(Scatterinformation,Incidentinformation)
            compute_time=time.time()-start
            max_errors.append(relative_norm)
            compute_times.append(compute_time)
        plt.plot(wavelengths, max_errors, '-o', label=f"s={s:.1f}")
        plt.savefig(f'Transmission_error_{s:.1f}.png')
    plt.xlabel("length of scattering object / wavelength")
    plt.ylabel("||E^tot||_2/||E^inc||_2")
    plt.legend()
    plt.title("Transmission Error vs Wavelength for Different Smoothness")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("multiple_error.png")
    plt.show()

    #check_transmission_conditions(Scatterinformation,Incidentinformation)
    #df=Single_scatter_solver(Scatterinformation,[Incidentinformation])
    #flux=df['flux'].values
    #plt.plot(betas,flux/np.linalg.norm(flux,2))
    #plt.show()
create_surface_and_scattering_info_from_json('surfaceParamsNormal-4.json')