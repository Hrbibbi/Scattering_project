import Hertzian_dipole as HD
import C2_surface as C2
import numpy as np
import matplotlib.pyplot as plt
import time
import plane_wave as PW
def construct_sub_column(Dipoles,Surface):
    #Major bottleneck in code:
    evaluations=HD.evaluate_Hertzian_Dipoles_at_points_parallel(Surface.points,Dipoles)

    N=len(Dipoles)
    M=np.shape(Surface.points)[0]

    E_tau1=np.zeros([M,N],dtype=complex)
    E_tau2=np.zeros([M,N],dtype=complex)
    H_tau1=np.zeros([M,N],dtype=complex)
    H_tau2=np.zeros([M,N],dtype=complex)

    for index,Dieval in enumerate(evaluations):
        E,H=Dieval
        E_tau1[:,index]=np.sum(Surface.tau1*E,axis=1)
        E_tau2[:,index]=np.sum(Surface.tau2*E,axis=1)
        H_tau1[:,index]=np.sum(Surface.tau1*H,axis=1)
        H_tau2[:,index]=np.sum(Surface.tau2*H,axis=1)
    
    return np.vstack((E_tau1,E_tau2,H_tau1,H_tau2))

def construct_matrix(Surface,auxsurface1,auxsurface2,mu,int_epsilon,out_epsilon,omega):
    M1,M2=auxsurface1.M,auxsurface2.M

    intDP1=HD.construct_Hertzian_Dipoles(auxsurface1.points,auxsurface1.tau1,mu*np.ones([M1]),int_epsilon*np.ones([M1]),omega*np.ones([M1]))
    intDP2=HD.construct_Hertzian_Dipoles(auxsurface1.points,auxsurface1.tau2,mu*np.ones([M1]),int_epsilon*np.ones([M1]),omega*np.ones([M1]))
    extDP1=HD.construct_Hertzian_Dipoles(auxsurface2.points,auxsurface2.tau1,mu*np.ones([M2]),out_epsilon*np.ones([M2]),omega*np.ones([M2]))
    extDP2=HD.construct_Hertzian_Dipoles(auxsurface2.points,auxsurface2.tau2,mu*np.ones([M2]),out_epsilon*np.ones([M2]),omega*np.ones([M2]))

    Col1=construct_sub_column(intDP1,Surface)
    Col2=construct_sub_column(intDP2,Surface)
    Col3=construct_sub_column(extDP1,Surface)
    Col4=construct_sub_column(extDP2,Surface)

    return np.column_stack((Col1,Col2,Col3,Col4))

def construct_RHS(Surface,planewave):
    E,H=planewave.evaluate_at_points(Surface.points)
    b1=np.sum(Surface.tau1*E,axis=1)
    b2=np.sum(Surface.tau2*E,axis=1)
    b3=np.sum(Surface.tau1*H,axis=1)
    b4=np.sum(Surface.tau2*H,axis=1)
    return np.concatenate((b1,b2,b3,b4))

'''
N=10*2
N_prime=10**2
M=10**2
S1=C2.sphere(1,np.array([0,0,0]),int(np.sqrt(M)))
#Auxiliiary_surfaces
S2=C2.sphere(0.8,np.array([0,0,0]),int(np.sqrt(N)))
S3=C2.sphere(1.2,np.array([0,0,0]),int(np.sqrt(N_prime)))
#plt.plot(np.linalg.norm(S3.tau2,2,axis=1))
#plt.show()
PW1=PW.Plane_wave(np.array([0,1,0]),0,1)
print(np.shape(construct_RHS(S1,PW1)))
print(np.shape(construct_matrix(S1,S2,S3,1,1,2,1)))
'''
'''
DP1=HD.construct_Hertzian_Dipoles(S2.points,S2.tau1,np.ones([N]),np.ones([N]),np.ones([N]))
DP2=HD.construct_Hertzian_Dipoles(S2.points,S2.tau2,np.ones([N]),np.ones([N]),np.ones([N]))
DP1_prime=HD.construct_Hertzian_Dipoles(S3.points,S3.tau1,np.ones([N_prime]),2*np.ones([N_prime]),np.ones([N_prime]))
DP2_prime=HD.construct_Hertzian_Dipoles(S3.points,S3.tau2,np.ones([N_prime]),2*np.ones([N_prime]),np.ones([N_prime]))
evaluations=HD.evaluate_Hertzian_Dipoles_at_points_parallel(S1.points,DP1)
E1=np.zeros([M,N],dtype=complex)
E2=np.zeros([M,N],dtype=complex)
for index,Dieval in enumerate(evaluations):
    E,H=Dieval
    test=np.sum(S1.tau1*E,axis=1)
    #print(test)
    E1[:,index]=test
print(E1)'
''
'''