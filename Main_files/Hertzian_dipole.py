'''
Hertzian dipole class used for for approximation of the scattered electric and magnetic field
'''

#Import of modules
import numpy as np
import warnings
import multiprocessing


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
        #precomputes
        p=X-self.position
        r=np.sqrt(np.sum(p**2,axis=1))
        exponential_term=np.exp(-1j*self.wavenumber*r)
        k=self.wavenumber
        x,y,z=p[:,0],p[:,1],p[:,2]
        dx,dy,dz=self.direction
        omega=self.omega
        mu=self.mu

        front_term1=-1j*omega*mu/(4*np.pi*r)*exponential_term
        front_term2=exponential_term/(4*np.pi*omega*self.epsilon*r**5)
        term1=1j*k**2*r**2+3*k*r-3j
        term2=1j-k*r
        E_x=front_term1*dx+front_term2*( x**2*dx*term1+x*(y*dy+z*dz)*term1+r**2*dx*term2 )
        E_y=front_term1*dy+front_term2*( y**2*dy*term1+y*(x*dx+z*dz)*term1+r**2*dy*term2 )
        E_z=front_term1*dz+front_term2*( z**2*dz*term1+z*(x*dx+y*dy)*term1+r**2*dz*term2 )
        E=np.column_stack((E_x,E_y,E_z))
        
        term3=exponential_term*(1+1j*k*r)/(4*np.pi*r**3)
        H_x=-(y*dz-z*dy)*term3
        H_y=(x*dz-z*dx)*term3
        H_z=-(x*dy-y*dx)*term3
        H=np.column_stack((H_x,H_y,H_z))
        return [E,H]


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

def evaluate_linear_combination(points,Dipoles,coefficents):
    evaluations=evaluate_Hertzian_Dipoles_at_points_parallel(points,Dipoles)
    E_tot,H_tot=np.zeros(np.shape(points),dtype=complex),np.zeros(np.shape(points),dtype=complex)
    for index,Dieval in enumerate(evaluations):
        E,H=Dieval
        E_tot+=coefficents[index]*E
        H_tot+=coefficents[index]*H
    return E_tot,H_tot
        
