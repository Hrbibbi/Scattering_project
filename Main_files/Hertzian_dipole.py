'''
Hertzian dipole class used for for approximation of the scattered electric and magnetic field
'''

#Import of modules
import numpy as np
import warnings
import multiprocessing
from numba import jit

class Hertzian_Dipole():
    def __init__(self,position,direction,mu,epsilon,omega):
        #----------------------------------------------------------------------------
        #Parameter check
        #----------------------------------------------------------------------------
        for param, name in zip([mu, epsilon, omega], ["mu", "epsilon", "omega"]):
            if not isinstance(param, (int, float, np.number)):
                raise TypeError(f"{name} must be a numerical value (int, float, or numpy number), got {type(param)} instead.")
        
        for vec, name in zip([position, direction], ["position", "direction"]):
            if not isinstance(vec, np.ndarray):
                raise TypeError(f"{name} must be a numpy array, got {type(vec)} instead.")
            if vec.shape != (3,):
                raise ValueError(f"{name} must have exactly 3 elements, but got shape {vec.shape}.")
        
        # Check if direction is a unit vector
        direction_norm = np.linalg.norm(direction)
        if not np.isclose(direction_norm, 1, atol=1e-6):
            raise ValueError(f"Direction vector must be a unit vector (norm = 1), but got norm = {direction_norm:.6f}.")

        #constant values
        self.mu=mu
        self.epsilon=epsilon
        self.omega=omega
        self.wavenumber=omega*np.sqrt(epsilon*mu)
        #vector values
        self.position = position
        self.direction = direction
    #--------------------------------------------------------------------------------
    # Methods
    #--------------------------------------------------------------------------------
    def evaluate_at_points(self,X):
        mu=self.mu
        epsilon=self.epsilon
        k=self.wavenumber
        dx,dy,dz=self.direction[0],self.direction[1],self.direction[2]
        omega=self.omega
        xi=1j*omega*mu / (4*np.pi)
        
        X_trans=X-self.position
        x,y,z=X_trans[:,0],X_trans[:,1],X_trans[:,2]
        r=np.sqrt(np.sum(X_trans**2,axis=1))
        dotted=dx*x+dy*y+dz*z

        R=1/(r**3)+1j*k/(r**2)
        Phi = lambda p: 3*p/(r**5)+3j*k*p/(r**4)-k**2*p/(r**3)
        phase=np.exp(-1j*k*r)
        
        E_x = ( dx*(xi/(k**2)*R-xi/r) - xi/(k**2)*Phi(x)*dotted ) * phase
        E_y = ( dy*(xi/(k**2)*R-xi/r) - xi/(k**2)*Phi(y)*dotted ) * phase
        E_z = ( dz*(xi/(k**2)*R-xi/r) - xi/(k**2)*Phi(z)*dotted ) * phase

        E=np.column_stack( (E_x,E_y,E_z) )

        H_x = 1/(4*np.pi)*(dy*z-dz*y)*R*phase
        H_y = -1/(4*np.pi)*(dx*z-dz*x)*R*phase
        H_z = 1/(4*np.pi)*(dx*y-dy*x)*R*phase

        H=np.column_stack( (H_x,H_y,H_z) )
        return E,H

#--------------------------------------------------------------------------------
#
#
#                            Dipole related functions
#
#
#--------------------------------------------------------------------------------


def construct_Hertzian_Dipoles(positions,directions,mus,epsilons,omegas):
    '''
    Returns list of Hertzian dipoles
    
    input:
        positions: Nx3 numpy array with positions of each dipole
        directions: Nx3 numpy array with unit direction of each dipole
        mus: N numpy array with magnetic permeability for each dipole
        epsilons: N numpy array with electric permitivity for each dipole
        omegas: N numpy array with the frequency for each dipole
    '''
    return [Hertzian_Dipole(positions[idx,:],directions[idx,:],mus[idx],epsilons[idx],omegas[idx]) for idx in range(len(mus))]

def evaluate_dipole(args):
    '''
    Wrapper function for evaluation Hertzian dipoles in parallel
    '''
    dipole, points = args
    return dipole.evaluate_at_points(points)

def evaluate_Hertzian_Dipoles_at_points_parallel(points, Dipoles):
    '''
    Returns a Nx2xMx3 numpy array with the evaluations of each each dipole in the M points

    input:
        points: Mx3 numpy array of points to evaluate
        dipoles N list if the Hertzian dipoles
    '''
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        evaluations = pool.map(evaluate_dipole, [(dipole, points) for dipole in Dipoles])
    return evaluations
