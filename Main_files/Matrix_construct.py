import Hertzian_dipole as HD
import C2_surface as C2
import numpy as np
import matplotlib.pyplot as plt
import time
import plane_wave as PW
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

def construct_matrix(Surface,inneraux,outeraux,mu,int_epsilon,out_epsilon,omega):
    '''
    Function to create the MAS matrix, it uses the subcolumn function to create each coulmn in the 4x4 block matrix

    input:
        Surface: instance of Surface class containing the geometric information
        innneraux: Surface placed inside of Surface to represent scattered field
        outeraux: Surface placed outside of Surface to represent total field
        mu: Magnetic permeability, assumed to be the same in all of R^3
        int_epsilon: electric permeability inside the surface used for the dipoles placed on outeraux
        out_epsilon: electric permeability outside the surface used for the dipoles placed on inneraux
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
    intDP1=HD.construct_Hertzian_Dipoles(innerpoints,innertau1,mu*innerarr,out_epsilon*innerarr,omega*innerarr)
    intDP2=HD.construct_Hertzian_Dipoles(innerpoints,innertau2,mu*innerarr,out_epsilon*innerarr,omega*innerarr)

    #dipoles placed on the outside to approximate the total field inside
    extDP1=HD.construct_Hertzian_Dipoles(outerpoints,outertau1,mu*outerarr,int_epsilon*outerarr,omega*outerarr)
    extDP2=HD.construct_Hertzian_Dipoles(outerpoints,outertau2,mu*outerarr,int_epsilon*outerarr,omega*outerarr)

    #Missing reflected dipoles
     
     
    #-----------------------------------------------------------------------------
    #                           Block matrix columns
    #-----------------------------------------------------------------------------

    #using sub_column function we constuct the blocks of the matrix
    Col1=construct_sub_column(intDP1,Surface)
    Col2=construct_sub_column(intDP2,Surface)
    Col3=construct_sub_column(extDP1,Surface)
    Col4=construct_sub_column(extDP2,Surface)

    return np.column_stack((Col1,Col2,Col3,Col4)),intDP1,intDP2

def construct_RHS(Surface,planewave):
    '''
    Function to create right hand side of MAS solver
    This needs to be remade as we a not taking into account the reflected-transmission part
    '''

    E,H=planewave.evaluate_at_points(Surface.points)
    b1=np.sum(Surface.tau1*E,axis=1)
    b2=np.sum(Surface.tau2*E,axis=1)
    b3=np.sum(Surface.tau1*H,axis=1)
    b4=np.sum(Surface.tau2*H,axis=1)
    return np.concatenate((b1,b2,b3,b4))


#-----------------------------------------------------------------------------
#                                   Unit tests 
#-----------------------------------------------------------------------------
'''
S=C2.sphere(1,np.array([0,0,0]),3)
DP1=HD.Hertzian_Dipole(np.array([0,0,0]),np.array([1,0,0]),1,1,1)
DP2=HD.Hertzian_Dipole(np.array([0,0,0]),np.array([0,1,0]),1,1,1)
DP3=HD.Hertzian_Dipole(np.array([0,0,0]),np.array([0,0,1]),1,1,1)
col=construct_sub_column([DP1,DP2,DP3],S)
plt.imshow(np.abs(col))
plt.show()
'''