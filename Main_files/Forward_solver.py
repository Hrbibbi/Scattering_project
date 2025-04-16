#-----------------------------------------------------------------------------
#                                   Imports
#-----------------------------------------------------------------------------
import Hertzian_dipole as HD
import C2_surface as C2
import plane_wave as PW
import numpy as np
import matplotlib.pyplot as plt
import time


def construct_RHS(Surface,propagation_vector,polarization,epsilon_air,mu,omega):
    '''
    Function to construct RHS of MAS system, in this it also returns the reflection coefficents to be used in construction of MAS matrix
    
    output:
        RHS of MAS system and reflection coefficents
    Input:
        Surface: Instance of C2 surface storing the information of points and normal vectors
        propagation_vector: the propagation_vector for the incident wave
        polarization: polarization angle of incident wave
        epsilon_air: permetivity of the air
        mu: magnetic permeability assumed to be the same in both air and substrate
        omega: frequency for incident wave
    
    '''

    #-----------------------------------------------------------------------------
    #                       store inforamtion for ease of use
    #-----------------------------------------------------------------------------
    points=Surface.points
    tau1=Surface.tau1
    tau2=Surface.tau2

    #-----------------------------------------------------------------------------
    #                construct incident wave and calculate reflection
    #-----------------------------------------------------------------------------
    planewave=PW.Plane_wave(propagation_vector,polarization,epsilon_air,mu,omega)
    E,H=planewave.evaluate_at_points(points)
    #-----------------------------------------------------------------------------
    #                               construct vector
    #-----------------------------------------------------------------------------
    b1=-np.sum(tau1*E,axis=1)
    b2=-np.sum(tau2*E,axis=1)
    b3=-np.sum(tau1*H,axis=1)
    b4=-np.sum(tau2*H,axis=1)

    return np.concatenate((b1,b2,b3,b4))

def construct_sub_column(Dipoles,Surface):
    '''
    function to calculate a coulmn in the block MAS matrix.
    
    Input:
        Dipoles: A list of dipoles which correspond to the ones in the block MAS
        Surface: The Surface containing the test points
    Output:
        Returns a 4MxN matrix with M number of testpoints and N number of dipoles
    '''
    
    #-----------------------------------------------------------------------------
    #                                   Setup
    #-----------------------------------------------------------------------------
    points=Surface.points
    tau1=Surface.tau1
    tau2=Surface.tau2
    N=len(Dipoles)
    M=Surface.M

    E_tau1=np.zeros([M,N],dtype=complex)
    E_tau2=np.zeros([M,N],dtype=complex)
    H_tau1=np.zeros([M,N],dtype=complex)
    H_tau2=np.zeros([M,N],dtype=complex)

    #-----------------------------------------------------------------------------
    #                          Dipole evaluation step
    #-----------------------------------------------------------------------------
    
    #Major bottleneck in code: See if it can be made faster
    evaluations=HD.evaluate_Hertzian_Dipoles_at_points_parallel(points,Dipoles)

    #-----------------------------------------------------------------------------
    #                             Matrix create step
    #-----------------------------------------------------------------------------

    for index,Dieval in enumerate(evaluations):
        E,H=Dieval
        E_tau1[:,index]=np.sum(tau1*E,axis=1)
        E_tau2[:,index]=np.sum(tau2*E,axis=1)
        H_tau1[:,index]=np.sum(tau1*H,axis=1)
        H_tau2[:,index]=np.sum(tau2*H,axis=1)

    return np.vstack((E_tau1,E_tau2,H_tau1,H_tau2))

def construct_matrix(Surface,inneraux,outeraux,mu,air_epsilon,scatter_epsilon,omega):
    '''
    Function to create the MAS matrix, it uses the subcolumn function to create each coulmn in the 4x4 block matrix

    input:
        Surface: instance of Surface class containing the geometric information
        innneraux: Surface placed inside of Surface to represent scattered field
        outeraux: Surface placed outside of Surface to represent total field
        mu: Magnetic permeability, assumed to be the same in all of R^3
        air_epsilon: electric permetivity of the sorrounding medium used for the dipoles placed inside the surface
        substrate_epsilon: electric permetivity outside the substrate
        scatter_epsilon: electric permetivity inside the scattering object, used for the dipoles placed outside the surface
        omega: frequency term assumed to constant
    '''
    #-----------------------------------------------------------------------------
    #                                   Precompute
    #-----------------------------------------------------------------------------

    Minner,Mouter=inneraux.M,outeraux.M
    innerarr,outerarr=np.ones([Minner]),np.ones([Mouter])
    innerpoints=inneraux.points
    innertau1=inneraux.tau1
    innertau2=inneraux.tau2
    outerpoints=outeraux.points
    outertau1=outeraux.tau1
    outertau2=outeraux.tau2
    
    #-----------------------------------------------------------------------------
    #                               Dipole construction 
    #-----------------------------------------------------------------------------

    #dipoles placed on the inside to aproximate the scattered field outside
    intDP1=HD.construct_Hertzian_Dipoles(innerpoints,innertau1,mu*innerarr,air_epsilon*innerarr,omega*innerarr)
    intDP2=HD.construct_Hertzian_Dipoles(innerpoints,innertau2,mu*innerarr,air_epsilon*innerarr,omega*innerarr)

    #dipoles placed on the outside to approximate the total field inside
    extDP1=HD.construct_Hertzian_Dipoles(outerpoints,outertau1,mu*outerarr,scatter_epsilon*outerarr,omega*outerarr)
    extDP2=HD.construct_Hertzian_Dipoles(outerpoints,outertau2,mu*outerarr,scatter_epsilon*outerarr,omega*outerarr)
    
    #-----------------------------------------------------------------------------
    #                           Block matrix columns
    #-----------------------------------------------------------------------------

    #using sub_column function we constuct the blocks of the matrix
    #scattering part
    Col1=construct_sub_column(intDP1,Surface)
    Col2=construct_sub_column(intDP2,Surface)
    #total part
    Col3=construct_sub_column(extDP1,Surface)
    Col4=construct_sub_column(extDP2,Surface)

    return np.column_stack((Col1,Col2,Col3,Col4)),intDP1,intDP2,extDP1,extDP2

def Construct_solve_MAS_system(Scatter_information,Incident_information,plot=False):
    '''
    Function that constructs the MAS_system based on information given about the scatterer,incident wave and substrate

    Output:
        int_coeff:
            -> C_1 coefficents for interior dipoles placed in tau1 orientation
            -> C_2 coefficents for interior dipoles placed in tau2 orientation
        ext_coeff:
            -> C_3 coefficents for exterior dipoles placed in tau1 orientation
            -> C_4 coefficents for exterior dipoles placed in tau2 orientation
        InteriorDipoles: list of dipoles placed in the interior
            -> intDP1: interior dipoles placed in tau1 orientation 
            -> intDP2: interior dipoles placed in tau2 orientation
        ExteriorDipoles: list of dipoles placed in the exterior
            -> extDP1: exterior dipoles placed in tau1 orientation
            -> extDP2: exterior dipoles placed in tau2 orientation

    Input:
        Scatter_information: Directory storing the required information about the scattering object
            -> Surface: A C2 object Storing the geometric information of the object
            -> inneraux: A C2 object Storing the geometric information for the inner aux surface
            -> outeraux: A C2 object Storing the geometric information for the outer aux surface
            -> epsilon: a float with the electric permetivity of the scattering material
            -> mu: a float with the magnetic permeability of the scattering object

        Incident_information: Directory storing the required information about the incident wave:
            -> propagation_vector: a unit normal vector descriping the propagation vector
            -> polarization: polarization angle of incident wave
            -> epsilon: a float with the electric permetivity of the medium the incident wave is in
            -> mu: a float with the magnetic permeability of the medium the incident wave is in
            -> omega: a float with the frequency of the incident wave
    '''

    #-----------------------------------------------------------------------------
    #                           Scattering information
    #-----------------------------------------------------------------------------

    Surface = Scatter_information['Surface']
    inneraux = Scatter_information['inneraux']
    outeraux = Scatter_information['outeraux']
    scatter_epsilon = Scatter_information['epsilon']
    scatter_mu = Scatter_information['mu']
    
    #-----------------------------------------------------------------------------
    #                           incident wave information
    #-----------------------------------------------------------------------------
    
    propagation_vector = Incident_information['propagation_vector']
    polarization = Incident_information['polarization']
    epsilon_air = Incident_information['epsilon']
    incident_mu = Incident_information['mu']
    omega = Incident_information['omega']
    
    #-----------------------------------------------------------------------------
    #                                Construction
    #-----------------------------------------------------------------------------
    # Construct the RHS and reflection coefficients
    con_time=time.time()
    RHS = construct_RHS(Surface,propagation_vector,polarization,epsilon_air,incident_mu,omega)
    
    # Construct the MAS matrix
    MAS_matrix,intDP1,intDP2,extDP1,extDP2 = construct_matrix(Surface,inneraux,outeraux,scatter_mu,epsilon_air,scatter_epsilon,omega)
    print(f"construction time {time.time()-con_time}")
    print(f"Matrix size {np.shape(MAS_matrix)}")
    #-----------------------------------------------------------------------------
    #                                 Solution
    #-----------------------------------------------------------------------------

    sol_start=time.time()
    N=len(intDP1)
    C=np.linalg.lstsq(MAS_matrix,RHS,rcond=-1)[0]
    C_int,C_ext=C[:2*N],C[2*N:]
    C_1,C_2=np.split(C_int,2)
    C_3,C_4=np.split(C_ext,2)
    print(f"solution time {time.time()-sol_start}")
    #-----------------------------------------------------------------------------
    #                              Optional plotting
    #-----------------------------------------------------------------------------

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. MAS matrix plot
        im0 = axs[0].imshow(np.abs(MAS_matrix), aspect='auto', cmap='viridis')
        axs[0].set_title('abs(MAS_matrix)')
        plt.colorbar(im0, ax=axs[0])
        
        # 2. RHS vector plot
        axs[1].plot(np.abs(RHS),'.')
        axs[1].set_title('abs(RHS vector)')
        axs[1].set_xlabel('Index')
        
        # 3. Solution vector C plot
        axs[2].plot(np.abs(C),'.')
        axs[2].set_title('abs(Solution vector C)')
        axs[2].set_xlabel('Index')

        plt.tight_layout()
        plt.show()
    
    #-----------------------------------------------------------------------------
    #                                 Return part
    #-----------------------------------------------------------------------------

    int_coeff=[C_1,C_2]
    ext_coeff=[C_3,C_4]
    InteriorDipoles=[intDP1,intDP2]
    ExteriorDipoles=[extDP1,extDP2]
    
    return int_coeff,ext_coeff, InteriorDipoles, ExteriorDipoles

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

    propagation_vector=np.array([0,0,-1])
    polarization=np.pi/2
    epsilon_air=1
    omega=1
    Incidentinformation={'propagation_vector': propagation_vector, 'polarization': polarization, 'epsilon': epsilon_air, 'mu': mu, 'omega':omega}
    int_coeff,ext_coeff, InteriorDipoles, ExteriorDipoles=Construct_solve_MAS_system(Scatterinformation,Incidentinformation,True)
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