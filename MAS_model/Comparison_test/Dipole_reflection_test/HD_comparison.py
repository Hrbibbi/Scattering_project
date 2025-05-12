'''
Hertzian dipole class used for for approximation of the scattered electric and magnetic field
'''

#Import of modules
import numpy as np
import warnings
import multiprocessing
from numba import jit
import pandas as pd

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

    def compute_reflected_field_at_points(self,points,mu,epsilon_substrate,epsilon_air):
        '''
        Computes the reflected field at points assuming that the dipole is placed under the xy-plane.
        output: returns the reflected field from a dipole 
        '''
        #---------------------------------------------------------------
        #                     Calculate the angles
        #---------------------------------------------------------------
        nu=np.array([0,0,1])
        eta_substrate=np.sqrt(mu/epsilon_substrate) #dipole is assumed to be placed within the substrate
        eta_air=np.sqrt(mu/epsilon_air)  
        direction=self.direction
        theta_inc = np.mod( np.arccos( np.dot( direction , nu ) ) , np.pi )
        print(theta_inc)
        theta_trans=np.emath.arcsin(epsilon_substrate/epsilon_air*np.sin(theta_inc))
        #---------------------------------------------------------------
        #                       Calculate the fields
        #---------------------------------------------------------------
        
        E_inc,H_inc=self.evaluate_at_points(points)
        E_perp = np.cos(theta_inc)*E_inc
        H_perp=np.cos(theta_inc)*H_inc
        E_par=np.sin(theta_inc)*E_inc
        H_par=np.sin(theta_inc)*H_inc

        #---------------------------------------------------------------
        #                reflection and transmission coeff
        #---------------------------------------------------------------

        r_perp=(eta_air*np.cos(theta_inc)-eta_substrate*np.cos(theta_trans) ) / ( eta_air*np.cos(theta_inc)+eta_substrate*np.cos(theta_trans) )

        r_par=(eta_air*np.cos(theta_trans)-eta_substrate*np.cos(theta_inc) ) / ( eta_air*np.cos(theta_trans)+eta_substrate*np.cos(theta_inc) )
        E_ref = r_perp * E_perp + r_par * E_par
        H_ref = r_perp * H_perp + r_par * H_par
        print(f"perpendicular_coefficent: {r_perp}, parallel_coefficent: {r_par}")
        return E_ref, H_ref
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

def compute_fields_from_csv(param_file, testpoints_file, output_file):
    # Read parameters
    param_df = pd.read_csv(param_file)
    params = dict(zip(param_df["Parameter"], param_df["Value"]))

    # Convert values properly
    mu = float(params["mu"])
    epsilon_air = float(params["epsilon_air"])
    epsilon_substrate =float(params['epsilon_substrate'])
    omega = float(params["omega"])
    position = np.array([params["position_x"], params["position_y"], params["position_z"]])
    direction = np.array([params["direction_x"], params["direction_y"], params["direction_z"]])

    # Read test points
    testpoints_df = pd.read_csv(testpoints_file)
    testpoints = testpoints_df.to_numpy()  # Convert DataFrame to NumPy array

    # Debugging: Check types
    print(f"mu: {mu}, epsilon_air: {epsilon_air}, epsilon_substrate: {epsilon_substrate}, omega: {omega}")
    print(f"Position: {position}, Direction: {direction}")
    print(f"Testpoints shape: {testpoints.shape}")

    # Compute fields (assuming Hertzian_Dipole is defined)
    DP = Hertzian_Dipole(position, direction, mu, epsilon_substrate, omega) #wheter the reflected dipoles should be epsilon_air or substrate?
    E,H=DP.compute_reflected_field_at_points(testpoints,mu,epsilon_substrate,epsilon_air)

    # Convert complex values into real & imaginary parts for saving
    data = {
        "Ex_Re": E[:, 0].real, "Ex_Im": E[:, 0].imag,
        "Ey_Re": E[:, 1].real, "Ey_Im": E[:, 1].imag,
        "Ez_Re": E[:, 2].real, "Ez_Im": E[:, 2].imag,
        "Hx_Re": H[:, 0].real, "Hx_Im": H[:, 0].imag,
        "Hy_Re": H[:, 1].real, "Hy_Im": H[:, 1].imag,
        "Hz_Re": H[:, 2].real, "Hz_Im": H[:, 2].imag,
    }

    output_df = pd.DataFrame(data)
    output_df.to_csv(output_file, index=False)

    print(f"Computed field data saved to {output_file}")