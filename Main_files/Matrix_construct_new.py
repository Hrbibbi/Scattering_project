#-----------------------------------------------------------------------------
#                                   Imports
#-----------------------------------------------------------------------------
import Hertzian_dipole as HD
import C2_surface as C2
import plane_wave as PW
import numpy as np
import matplotlib.pyplot as plt
import time

def construct_RHS(Surface,propagation_vector,polarization,epsilon_air,epsilon_substrate,mu,omega):
    '''
    Function to construct RHS of MAS system, in this it also returns the reflection coefficents to be used in construction of MAS matrix
    
    output:
        RHS of MAS system and reflection coefficents
    Input:
        Surface: Instance of C2 surface storing the information of points and normal vectors
        propagation_vector: the propagation_vector for the incident wave
        polarization: polarization angle of incident wave
        epsilon_air: permetivity of the air
        epsilon_substrate: permetivity of substrate
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
    E_ref,H_ref,r_perp,r_par=PW.get_reflected_field_at_points(points,planewave,mu,epsilon_substrate,epsilon_air)
    E_inc,H_inc=planewave.evaluate_at_points(Surface.points)
    E_new=E_inc+E_ref
    H_new=H_inc+H_ref

    #-----------------------------------------------------------------------------
    #                               construct vector
    #-----------------------------------------------------------------------------
    b1=np.sum(tau1*E_new,axis=1)
    b2=np.sum(tau2*E_new,axis=1)
    b3=np.sum(tau1*H_new,axis=1)
    b4=np.sum(tau2*H_new,axis=1)

    return np.concatenate((b1,b2,b3,b4)),r_perp,r_par

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

def construct_matrix(Surface,inneraux,outeraux,mu,air_epsilon,substrate_epsilon,scatter_epsilon,omega,r_perp,r_par):
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
        r_perp: reflection coefficent for perpendicular part of incident wave
        r_par: reflection coefficent for parallel par of incident wave
    '''
    #-----------------------------------------------------------------------------
    #                                   Precompute
    #-----------------------------------------------------------------------------

    Minner,Mouter=inneraux.M,outeraux.M
    innerarr,outerarr=np.ones([Minner]),np.ones([Mouter])
    innerpoints=inneraux.points
    innerpoints_reflected=np.copy(inneraux.points)
    innerpoints_reflected[:,2]*=-1
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

    #dipoles placed at the reflected points to approximate the scattered field outside
    refDP1=HD.construct_Hertzian_Dipoles(innerpoints_reflected,innertau1,mu*innerarr,air_epsilon*innerarr,omega*innerarr)
    refDP2=HD.construct_Hertzian_Dipoles(innerpoints_reflected,innertau2,mu*innerarr,air_epsilon*innerarr,omega*innerarr)

    #dipoles placed on the outside to approximate the total field inside
    extDP1=HD.construct_Hertzian_Dipoles(outerpoints,outertau1,mu*outerarr,scatter_epsilon*outerarr,omega*outerarr)
    extDP2=HD.construct_Hertzian_Dipoles(outerpoints,outertau2,mu*outerarr,scatter_epsilon*outerarr,omega*outerarr)
    
    #-----------------------------------------------------------------------------
    #                           Block matrix columns
    #-----------------------------------------------------------------------------

    #using sub_column function we constuct the blocks of the matrix
    Col1=construct_sub_column(intDP1,Surface)
    Col2=construct_sub_column(intDP2,Surface)
    
    ref1=construct_sub_column(refDP1,Surface)
    Col1_ref=r_perp*ref1+r_par*ref1
    
    ref2=construct_sub_column(refDP2,Surface)
    Col2_ref=r_perp*ref2+r_par*ref2

    Col3=construct_sub_column(extDP1,Surface)
    Col4=construct_sub_column(extDP2,Surface)

    return np.column_stack((Col1+Col1_ref,Col2+Col2_ref,Col3,Col4)),intDP1,intDP2,refDP1,refDP2,extDP1,extDP2

def Construct_MAS_system(Scatter_information,Incident_information,Substrate_information):
    '''
    Function that constructs the MAS_system based on information given about the scatterer,incident wave and substrate

    Output:
        Returns the MAS matrix, RHS and the dipoles that can be used to reconstruct fields

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

        Substrate_information: Directory storing information about the substrate
            -> mu: magnetic permeability of the substrate
            -> epsilon: electric permetivity of the substrate
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
    incident_epsilon = Incident_information['epsilon']
    incident_mu = Incident_information['mu']
    omega = Incident_information['omega']
    
    #-----------------------------------------------------------------------------
    #                           substrate information
    #-----------------------------------------------------------------------------
    
    substrate_mu = Substrate_information['mu']
    substrate_epsilon = Substrate_information['epsilon']
    
    #-----------------------------------------------------------------------------
    #                           Construction
    #-----------------------------------------------------------------------------
    # Construct the RHS and reflection coefficients
    RHS, r_perp, r_par = construct_RHS(Surface, propagation_vector, polarization, 
                                        incident_epsilon, substrate_epsilon, incident_mu, omega)
    
    # Construct the MAS matrix
    MAS_matrix,intDP1,intDP2,refDP1,refDP2,extDP1,extDP2 = construct_matrix(Surface, inneraux, outeraux, scatter_mu, 
                                  incident_epsilon, substrate_epsilon, scatter_epsilon, omega, 
                                  r_perp, r_par)
    
    return MAS_matrix, RHS

'''
Surface=C2.sphere(1,np.array([0,0,1]),30)
inneraux=C2.sphere(0.8,np.array([0,0,1]),30)
outeraux=C2.sphere(1.2,np.array([0,0,1]),30)
scatter_epsilon=2
mu=1
Scatterinformation={'Surface': Surface,'inneraux': inneraux, 'outeraux': outeraux,'epsilon': scatter_epsilon,'mu': mu}

propagation_vector=np.array([0,-1,0])
polarization=0
epsilon_air=1
omega=1
Incidentinformation={'propagation_vector': propagation_vector, 'polarization': polarization, 'epsilon': epsilon_air, 'mu': mu, 'omega':omega}

Substrateinformation={'mu': mu,'epsilon': 10}
A,b=Construct_MAS_system(Scatterinformation,Incidentinformation,Substrateinformation)

C=np.linalg.lstsq(A,b,rcond=-1)[0]
plt.plot(np.abs(C))
plt.show()

'''