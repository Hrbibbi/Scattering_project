#-----------------------------------------------------------------------------
#                                   Imports
#-----------------------------------------------------------------------------
import Hertzian_dipole_speedup as HD
import C2_surface as C2
import plane_wave_speedup as PW
import numpy as np
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
    N = points.shape[0]

    # Evaluate fields for all M plane waves
    planewaves = PW.Plane_wave(propagation_vectors, polarizations, epsilon_air, mu, omegas)
    E_all, H_all = planewaves.evaluate_at_points(points)  # (2, M, N, 3)

    # Compute tangential components → shape (M, N)
    b1 = -np.einsum("mnj,nj->mn", E_all, tau1)
    b2 = -np.einsum("mnj,nj->mn", E_all, tau2)
    b3 = -np.einsum("mnj,nj->mn", H_all, tau1)
    b4 = -np.einsum("mnj,nj->mn", H_all, tau2)

    # Stack → shape (4, M, N), then permute → (M, 4, N), then reshape → (4*N, M)
    rhs_matrix = np.stack([b1, b2, b3, b4], axis=0)    # (4, M, N)
    rhs_matrix = np.transpose(rhs_matrix, (1, 0, 2))   # (M, 4, N)
    rhs_matrix = rhs_matrix.reshape(len(propagation_vectors), 4*N).T  # (4*N, M)

    return rhs_matrix


def construct_sub_column(dipoles, Surface):
    '''
    Computes a column block of the MAS matrix corresponding to a set of dipoles.

    Input:
        dipoles
        Surface: Surface object with:
            - points: (M, 3)
            - tau1, tau2: (M, 3) tangent vectors
            - M: number of test points

    Output:
        (4*M, N) matrix — tangential components of E and H fields from each dipole
    '''
    points = Surface.points
    tau1 = Surface.tau1
    tau2 = Surface.tau2
    M = Surface.M

    E_all, H_all = dipoles.evaluate_at_points(points)

    # Extract field blocks
    E = E_all  # shape (N, M, 3)
    H = H_all

    # Project fields onto tau vectors → shape (N, M)
    E_tau1 = np.einsum("nmj,mj->nm", E, tau1)
    E_tau2 = np.einsum("nmj,mj->nm", E, tau2)
    H_tau1 = np.einsum("nmj,mj->nm", H, tau1)
    H_tau2 = np.einsum("nmj,mj->nm", H, tau2)

    # Stack and reshape to (4*M, N)
    block = np.stack([E_tau1, E_tau2, H_tau1, H_tau2], axis=0)  # (4, N, M)
    block = block.transpose(1, 0, 2).reshape(len(dipoles.positions), 4*M).T  # (4*M, N)

    return block

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
    # Create HertzianDipole instances
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
    
def Construct_solve_MAS_system(Scatter_information, Incident_information, plot=False):
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

    #-------------------------------------------------------------
    # Construct RHS matrix and MAS matrix
    #-------------------------------------------------------------
    con_time = time.time()
    RHS_matrix = construct_RHSs(Surface, propagation_vectors, polarizations, epsilon_air, incident_mu, omega)
    construct_matrix(Surface,inneraux,outeraux,scatter_mu,epsilon_air,scatter_epsilon,omega)
    MAS_matrix, intDP1, intDP2, extDP1, extDP2 = construct_matrix(
        Surface, inneraux, outeraux, scatter_mu, epsilon_air, scatter_epsilon, omega
    )
    
    print(f"Construction time: {time.time() - con_time:.3f} s")
    print(f"Matrix shape: {MAS_matrix.shape}, RHS shape: {RHS_matrix.shape}")

    #-------------------------------------------------------------
    # Solve system for all RHS
    #-------------------------------------------------------------
    sol_start = time.time()
    C_matrix, *_ = np.linalg.lstsq(MAS_matrix, RHS_matrix, rcond=None)  # shape (4*N, M)
    print(np.shape(C_matrix))
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

def compute_scattered_field_at_point(points,int_coeff,InteriorDipoles):
    C_1,C_2=int_coeff
    intDP1,intDP2=InteriorDipoles
    evaluations1=np.array(HD.evaluate_Hertzian_Dipoles_at_points_parallel(points,intDP1))
    evaluations2=np.array(HD.evaluate_Hertzian_Dipoles_at_points_parallel(points,intDP2))

    weights1=np.sum( C_1[:,None,None,None]*evaluations1, axis=0)
    weights2=np.sum( C_2[:,None,None,None]*evaluations2, axis=0)
    E_scat,H_scat=weights1+weights2
    
    return E_scat,H_scat

def compute_flux_integral_scattered_field(plane,dipoles,coefficents):
    '''
    Function that computes the average power integral of for the scattered field

    input:
        Plane: A C2_object from which we integrate over 
        Dipoles: list of lists of dipoles used, assumed to be in order
            DP1, DP2, refDP1, refDP2 
    output:
        THe flux integral over tthe plane
    '''

    #-----------------------------------------------------------------------------
    #                        Extract plane information
    #-----------------------------------------------------------------------------
    points=plane.points
    
    #We assume uniform grid space for the plane
    dx=np.linalg.norm(points[1]-points[0])
    dA=dx*dx
    normals=plane.normals

    #-----------------------------------------------------------------------------
    #                         Compute integrand information
    #-----------------------------------------------------------------------------
    E,H=compute_scattered_field_at_point(points,coefficents,dipoles)
    Cross=1/2*np.cross(E, np.conj(H))
    integrand= np.sum(Cross * normals, axis=1)
    
    integral=np.sum(integrand*dA)
    return integral



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

    propagation_vector=np.array([[0,0,-1]])
    polarization=np.array([np.pi/2])
    epsilon_air=1
    omega=1
    Incidentinformation={'propagation_vectors': propagation_vector, 'polarizations': polarization, 'epsilon': epsilon_air, 'mu': mu, 'omega':omega}
    int_coeff,ext_coeff, InteriorDipoles, ExteriorDipoles=Construct_solve_MAS_system(Scatterinformation,Incidentinformation,True)
    #Construct_solve_MAS_system(Scatterinformation,Incidentinformation,False)
    #Plane=C2.generate_plane_xy(100,a,b,20)
    #print(compute_flux_integral_scattered_field(Plane,InteriorDipoles,int_coeff))

def bump_test():
    f = lambda x,y,x0,y0,height,sigma: height * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    g = lambda x,y: (f(x,y,3.2049716441173697,-2.6184373514244346,0.3566862845006943,0.5045178007249862)+
                     f(x,y,0.7804496417098932,-2.8682866110638114,0.16735177712044924,0.549210117954695)+
                     f(x,y,3.2254281899208737,1.0637048351396867,0.10154774456698834,0.48617824563009)+
                     f(x,y,-1.1080292473196582,0.8330555059815783,0.2175298584938063,0.6059738740659731)
                    ) 
    a,b=-5,5
    N=200
    x0,y0=np.linspace(a,b,N),np.linspace(a,b,N)
    x,y=np.meshgrid(x0,y0)
    z=g(x,y)
    point_cloud,tau1,tau2,normals,mean_curvature=C2.compute_geometric_data(x,y,z,(b-a)/N)
    inner_cloud=C2.generate_curvature_scaled_offset(point_cloud,normals,mean_curvature,-0.86)
    outer_cloud=C2.generate_curvature_scaled_offset(point_cloud,normals,mean_curvature,0.86)
    Surface=C2.C2_surface(point_cloud,normals,tau1,tau2)
    inneraux=C2.C2_surface(inner_cloud,normals,tau1,tau2)
    outeraux=C2.C2_surface(outer_cloud,normals,tau1,tau2)
    inneraux=C2.take_5_points_per_WL(inneraux,a,b,N)
    outeraux=C2.take_5_points_per_WL(outeraux,a,b,N)



    scatter_epsilon=2
    mu=1
    Scatterinformation={'Surface': Surface,'inneraux': inneraux, 'outeraux': outeraux,'epsilon': scatter_epsilon,'mu': mu}

    propagation_vector=np.array([0,0,-1])
    polarization=0
    epsilon_air=1
    omega=1
    Incidentinformation={'propagation_vector': propagation_vector, 'polarization': polarization, 'epsilon': epsilon_air, 'mu': mu, 'omega':omega}
    int_coeff,ext_coeff, InteriorDipoles, ExteriorDipoles=Construct_solve_MAS_system(Scatterinformation,Incidentinformation,False)
    Plane=C2.generate_plane_xy(100,a,b,20)
    print(compute_flux_integral_scattered_field(Plane,InteriorDipoles,int_coeff))   

test_instance()