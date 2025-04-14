'''
Hertzian dipole class used for for approximation of the scattered electric and magnetic field
'''

#Import of modules
import numpy as np
import warnings
import multiprocessing
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

        H_x = 1/(4*np.pi)*(dy-dz)*R*phase
        H_y = 1/(4*np.pi)*(dz-dx)*R*phase
        H_z = 1/(4*np.pi)*(dx-dy)*R*phase

        H=np.column_stack( (H_x,H_y,H_z) )
        return E,H       
    

import ast  # For safely evaluating string representations of lists

def compute_fields_from_csv(param_file, testpoints_file, output_file):
    # Read parameters
    param_df = pd.read_csv(param_file)
    params = dict(zip(param_df["Parameter"], param_df["Value"]))

    # Convert values properly
    mu = float(params["mu"])
    epsilon = float(params["epsilon"])
    omega = float(params["omega"])
    position = np.array([params["position_x"], params["position_y"], params["position_z"]])
    direction = np.array([params["direction_x"], params["direction_y"], params["direction_z"]])

    # Read test points
    testpoints_df = pd.read_csv(testpoints_file)
    testpoints = testpoints_df.to_numpy()  # Convert DataFrame to NumPy array

    # Debugging: Check types
    print(f"mu: {mu}, epsilon: {epsilon}, omega: {omega}")
    print(f"Position: {position}, Direction: {direction}")
    print(f"Testpoints shape: {testpoints.shape}")

    # Compute fields (assuming Hertzian_Dipole is defined)
    DP = Hertzian_Dipole(position, direction, mu, epsilon, omega)
    E, H = DP.evaluate_at_points(testpoints)

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
    