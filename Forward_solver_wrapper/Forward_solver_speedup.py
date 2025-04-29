#-----------------------------------------------------------------------------
#                                   Imports
#-----------------------------------------------------------------------------
import Hertzian_dipole_speedup as HD
import C2_surface
import plane_wave_speedup as PW
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


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
    
def Construct_solve_MAS_system(Scatter_information, Incident_information, plot=False, reduce_grid=True):
    '''
    Solves the MAS system for multiple plane wave excitations (same omega, mu, epsilon).

    Output:
        int_coeffs: [C1, C2] arrays of shape (N, M) for interior dipoles in tau1 and tau2
        ext_coeffs: [C3, C4] arrays of shape (N, M) for exterior dipoles in tau1 and tau2
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
    propagation_vectors = Incident_information['propagation_vectors']  # shape (M, 3)
    polarizations = Incident_information['polarizations']              # shape (M,)
    epsilon_air = Incident_information['epsilon']
    incident_mu = Incident_information['mu']
    omega = Incident_information['omega']
    lam = Incident_information['lambda']
    
    #-------------------------------------------------------------
    # Reduce surface information to necessary wavelength
    #-------------------------------------------------------------

    if reduce_grid:
        Surface, inneraux, outeraux = C2_surface.Set_dipoles_pr_WL(Surface,inneraux,outeraux,lam)
    
    #-------------------------------------------------------------
    # Construct RHS matrix and MAS matrix
    #-------------------------------------------------------------
    con_time = time.time()
    RHS_matrix = construct_RHSs(Surface, propagation_vectors, polarizations, epsilon_air, incident_mu, omega)
    construct_matrix(Surface,inneraux,outeraux,scatter_mu,epsilon_air,scatter_epsilon,omega)
    MAS_matrix, intDP1, intDP2, extDP1, extDP2 = construct_matrix(
        Surface, inneraux, outeraux, scatter_mu, epsilon_air, scatter_epsilon, omega
    )
    print(f"Number of RHS: {np.shape(RHS_matrix)[1]}")
    print(f"Construction time: {time.time() - con_time:.3f} s")
    #print(f"Matrix shape: {MAS_matrix.shape}, RHS shape: {RHS_matrix.shape}")

    #-------------------------------------------------------------
    # Solve system for all RHS
    #-------------------------------------------------------------
    sol_start = time.time()
    C_matrix, *_ = np.linalg.lstsq(MAS_matrix, RHS_matrix, rcond=None)  # shape (4*N, M)
    N = len(intDP1.positions)
    C1 = C_matrix[:N]
    C2 = C_matrix[N:2*N]
    C3 = C_matrix[2*N:3*N]
    C4 = C_matrix[3*N:]
    print(f"Solution time: {time.time() - sol_start:.3f} s")

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

    return int_coeffs, ext_coeffs, InteriorDipoles, ExteriorDipoles

def compute_scattered_field_at_point(points, int_coeff, InteriorDipoles):
    """
    Compute scattered EM fields at given points due to interior dipoles
    for multiple sets of dipole coefficients (i.e., multiple incident conditions).

    Parameters:
        points : (N, 3) array — evaluation points
        int_coeff : [C_1, C_2] — each of shape (M, R)
        InteriorDipoles : [intDP1, intDP2] — dipole objects

    Returns:
        (2, R, N, 3) array — scattered E and H fields
    """
    C_1, C_2 = int_coeff                      # Shape: (M, R)
    intDP1, intDP2 = InteriorDipoles

    # Evaluate dipole fields: shape (2, M, N, 3)
    evals1 = intDP1.evaluate_at_points(points)
    evals2 = intDP2.evaluate_at_points(points)

    # Reshape for broadcasting:
    # evals: (2, M, N, 3) → (2, M, 1, N, 3)
    # coeffs: (M, R) → (1, M, R, 1, 1)
    evals1 = evals1[:, :, None, :, :]        # (2, M, 1, N, 3)
    evals2 = evals2[:, :, None, :, :]        # (2, M, 1, N, 3)
    C_1 = C_1[None, :, :, None, None]        # (1, M, R, 1, 1)
    C_2 = C_2[None, :, :, None, None]        # (1, M, R, 1, 1)

    # Weighted sum across dipoles (axis=1) → result shape: (2, R, N, 3)
    field1 = np.sum(C_1 * evals1, axis=1)
    field2 = np.sum(C_2 * evals2, axis=1)

    total_fields = field1 + field2           # shape: (2, R, N, 3)
    return total_fields

def compute_flux_integral_scattered_field(plane, dipoles, coefficients):
    '''
    Computes the average power (flux) integral for the scattered field for multiple RHSs.

    Input:
        plane: A C2_object (with .points and .normals)
        dipoles: List of dipole dictionaries [intDP1, intDP2]
        coefficients: List of dipole weights [C_1, C_2], each (N_dipoles, M_rhs)

    Output:
        flux_values: Array of shape (M,) — power flux per RHS
    '''
    int_start=time.time()
    #---------------------------------------------------------------------
    # Extract geometry
    #---------------------------------------------------------------------
    points = plane.points             # (N_points, 3)
    normals = plane.normals           # (N_points, 3)

    dx = np.linalg.norm(points[1] - points[0])
    dA = dx * dx                      # Scalar area element (uniform)

    #---------------------------------------------------------------------
    # Evaluate scattered fields: (2, M, N, 3)
    #---------------------------------------------------------------------
    E, H = compute_scattered_field_at_point(points, coefficients, dipoles)

    Cross = 0.5 * np.cross(E, np.conj(H))           # shape: (M, N, 3)
    Cross_dot_n = np.einsum("mnj,nj->mn", Cross, normals)  # (M, N)

    # Integrate over surface → sum over points
    integrals = np.sum(Cross_dot_n, axis=1) * dA    # (M,)
    print(f"integration_time: {time.time()-int_start}")
    return integrals  # Return real-valued power flux

def Forward_solver(Scatter_information, Incident_informations, options):
    """
    Solves the forward problem for a specified scatterer and multiple sets of incident waves.

    Returns
    -------
    df : pandas.DataFrame
        Columns: propagation_vector, polarization, wavelength, frequency, power_integral
    """
    # Unpack options (with defaults)
    show_MAS         = options.get('show_MAS', False)
    show_power_curve = options.get('Show_power_curve', False)
    plane_z          = options.get('plane_location', None)

    # Extract geometry & domain
    Surface  = Scatter_information['Surface']
    pts      = Surface.points
    M_total  = pts.shape[0]
    N_grid   = int(np.sqrt(M_total))     # assume N×N grid
    a, b     = pts[:,0].min(), pts[:,0].max()

    # default plane height = 5× max z if not specified
    if plane_z is None:
        plane_z = 5 * pts[:,2].max()

    # Build the evaluation plane *once*
    Plane = C2_surface.generate_plane_xy(plane_z, a, b, 10) #CHANGE DEPENDING ON REQUIRED PRECISION
    # Accumulate results here
    records = []
    for idx, inc in enumerate(Incident_informations):
        # ---- NEW PRINT HERE ----
        print(f"\nComputing incident information number {idx+1}/{len(Incident_informations)}, wavelength: {inc['lambda']:.4f}")
        total_time=time.time()
        # Solve MAS for this block of plane waves
        int_coeffs, _, InteriorDipoles, _ = Construct_solve_MAS_system(
            Scatter_information,
            inc,
            plot=show_MAS
        )

        # Incident metadata
        props = inc['propagation_vectors']  # (M_block,3)
        pols  = inc['polarizations']        # (M_block,)
        omega = inc['omega']
        lam   = inc['lambda']
        freq  = omega / (2*np.pi)

        power_int = compute_flux_integral_scattered_field(
            Plane,
            InteriorDipoles,
            int_coeffs
        )  # shape (M_block)

        # Record one row per wave
        for j in range(props.shape[0]):
            records.append({
                'propagation_vector': props[j],
                'polarization'      : pols[j],
                'wavelength'        : lam,
                'frequency'         : freq,
                'power_integral'    : power_int[j]
            })

        # Optional plotting of this block’s power curve
        if show_power_curve:
            plt.figure()
            plt.plot(np.abs(power_int), marker='o')
            plt.xlabel('Incident index')
            plt.ylabel('Power integral')
            plt.title('Scattered Power vs. Incident Wave')
            plt.tight_layout()
            plt.show()
        print(f"total time: {time.time()-total_time}")
    # Build DataFrame and return
    df = pd.DataFrame.from_records(records,
        columns=['propagation_vector','polarization','wavelength','frequency','power_integral'])
    return df

def average_forward_response(Scattering_Informations, Incident_Informations, options):
    """
    Averages the power integral from forward solves across multiple surface realizations.

    Parameters
    ----------
    Scattering_Informations : list of dict
        Each dict describes one surface realization + its physical parameters.
    Incident_Informations : list of dict
        Each dict contains incident wave configurations. All are used for each surface.
    options : dict
        Dictionary of options to pass to Forward_solver (e.g., show_MAS, plane_location, Show_power_curve).

    Returns
    -------
    avg_power : np.ndarray
        The average power integral across surface realizations.
    all_power_curves : list of np.ndarray
        The power integral arrays for each realization (for optional plotting).
    """
    assert len(Scattering_Informations) > 0, "Need at least one surface realization"
    assert isinstance(Incident_Informations, list) and len(Incident_Informations) > 0, "Incident_Informations must be a non-empty list"
    completion_time=time.time()
    all_power_curves = []

    for i, scatter_info in enumerate(Scattering_Informations):
        print(f"\n\nSolving realization {i+1}/{len(Scattering_Informations)}")
        df = Forward_solver(scatter_info, Incident_Informations, options)
        power_curve = np.array(df['power_integral'].values)
        all_power_curves.append(power_curve)

    # Stack and compute mean over realizations
    all_power_array = np.stack(all_power_curves, axis=0)  # shape (n_realizations, n_incidents_total)
    avg_power = np.mean(all_power_array, axis=0)          # shape (n_incidents_total,)
    completion_time=time.time()-completion_time
    print(f"\ncompletion time: {completion_time}")
    config=0
    for inc in Incident_Informations:
        config+=len(inc['polarizations'])
    print(f"Number of forward problems solved: {len(Scattering_Informations)*config}")
    print(f"Average time pr forward solve: {completion_time/(len(Scattering_Informations)*config)}")
    return avg_power, all_power_curves


def test_instance():
    f = lambda x,y: (1+1/2+1/4)+np.cos(np.sqrt(x**2+y**2))+1/4*np.cos(2*np.sqrt(x**2+y**2))
    a,b=-np.pi,np.pi
    N=20
    x0,y0=np.linspace(a,b,N),np.linspace(a,b,N)
    x,y=np.meshgrid(x0,y0)
    z=f(x,y)
    point_cloud,tau1,tau2,normals,mean_curvature=C2.compute_geometric_data(x,y,z,(b-a)/N)
    inner_cloud=C2.generate_curvature_scaled_offset(point_cloud,normals,mean_curvature,-0.86)
    outer_cloud=C2.generate_curvature_scaled_offset(point_cloud,normals,mean_curvature,0.86)
    Surface=C2.C2_surface(point_cloud,normals,tau1,tau2)
    inneraux=C2.C2_surface(inner_cloud,normals,tau1,tau2)
    outeraux=C2.C2_surface(outer_cloud,normals,tau1,tau2)
    scatter_epsilon=2
    mu=1
    Scatterinformation={'Surface': Surface,'inneraux': inneraux, 'outeraux': outeraux,'epsilon': scatter_epsilon,'mu': mu}
    iter=1
    iter_time=time.time()
    propagation_vector = np.tile([0, 0, -1], (100*iter, 1))
    polarization=np.linspace(0,np.pi/2,100*iter)
    epsilon_air=1
    omega=1
    Incidentinformation={'propagation_vectors': propagation_vector, 'polarizations': polarization, 'epsilon': epsilon_air, 'mu': mu, 'omega':omega}
    int_coeff,ext_coeff, InteriorDipoles, ExteriorDipoles=Construct_solve_MAS_system(Scatterinformation,Incidentinformation,False)
    Plane=C2.generate_plane_xy(100,a,b,20)
    power_int=compute_flux_integral_scattered_field(Plane,InteriorDipoles,int_coeff)
    plt.plot(np.abs(power_int))
    plt.show()
    print(f"RHS shape: {100*iter}, iteration_time: {time.time()-iter_time}")

def bump_test(width=1,resol=100): 
    #----------------------------------------
    #       Surface creation
    # ---------------------------------------  
    a,b=-width,width
    X0=np.linspace(a,b,resol)
    Y0=np.linspace(a,b,resol)
    X,Y=np.meshgrid(X0,Y0)
    bump = lambda x,y,x0,y0,height,sigma: height*np.exp(
        -( (x-x0)**2 + (y-y0)**2 ) / (2*sigma**2)
    )
    f = lambda x,y: (
                    bump(x,y,-0.20073580984422001,0.7211428902558659,0.31959818254342154,0.49932924209851826)+
                    bump(x,y,-0.5503701752921016,-0.5504087674620758,0.11742508365045984,0.6330880728874675) +
                    bump(x,y,0.16178401878913407,0.3329161244736727,0.10617534828874074,0.6849549260809971) 
                     )
    Z=f(X,Y)
    point_cloud,tau1,tau2,normals,mean_curvature=C2_surface.compute_geometric_data(X,Y,Z,(width-(-width))/resol)
    inner_cloud=C2_surface.generate_curvature_scaled_offset(point_cloud,normals,mean_curvature,-0.86)
    outer_cloud=C2_surface.generate_curvature_scaled_offset(point_cloud,normals,mean_curvature,0.86)
        
    Surface=C2_surface.C2_surface(point_cloud,normals,tau1,tau2)
    inneraux=C2_surface.C2_surface(inner_cloud,normals,tau1,tau2)
    outeraux=C2_surface.C2_surface(outer_cloud,normals,tau1,tau2)

    #---------------------------------------------
    #           Scattering information
    #---------------------------------------------
    scatter_epsilon=2.56
    mu=1
    Scatterinformation={'Surface': Surface,'inneraux': inneraux, 'outeraux': outeraux,'epsilon': scatter_epsilon,'mu': mu}
    
    #---------------------------------------------
    #           Incident information
    #---------------------------------------------
    Incidentinformations=[]
    for i in range(2):
        number=np.random.randint(low=1000,high=2000)
        propagation_vector = np.tile([0, 0, -1], (number, 1))
        polarization=np.linspace(0,np.pi/2,number)
        wavelength=np.random.rand(1)[0]+1
        epsilon_air=1
        #wavelength=1.5
        omega=2*np.pi/wavelength
        Incidentinformations.append(
            {'propagation_vectors': propagation_vector, 'polarizations': polarization, 'epsilon': epsilon_air, 'mu': mu,'lambda': wavelength, 'omega':omega}
        )
    options = {
    'show_MAS'         : False,
    'plane_location'   : None,   # auto = 5×max height
    'Show_power_curve' : False
    }
    #solution_time=time.time()
    avg_power, all_power_curves=average_forward_response([Scatterinformation],Incidentinformations,options)
    #print(f"solution time {time.time()-solution_time}")
#bump_test()