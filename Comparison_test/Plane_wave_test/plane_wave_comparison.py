import numpy as np
import ast  # For safely evaluating string representations of lists
import pandas as pd
class Plane_wave():
    def __init__(self,propagation_vector,polarization,epsilon,mu,omega):
        '''
        Check input
        '''
        for param, name in zip([polarization,epsilon,mu,omega], ["polarization","epsilon","mu","omega"]):
            if not isinstance(param, (int, float, np.number)):
                raise TypeError(f"{name} must be a numerical value (int, float, or numpy number), got {type(param)} instead.")
        if (polarization<0 and np.pi/2<polarization):
            raise ValueError(f"polarization angle not in range {(0,np.pi/2)}, value found {polarization}")
        
        for vec, name in zip([propagation_vector], ["propagation_vector"]):
            if not isinstance(vec, np.ndarray):
                raise TypeError(f"{name} must be a numpy array, got {type(vec)} instead.")
            if vec.shape != (3,):
                raise ValueError(f"{name} must have exactly 3 elements, but got shape {vec.shape}.")

        # Check if direction is a unit vector
        prop_norm = np.linalg.norm(propagation_vector)
        if not np.isclose(prop_norm, 1, atol=1e-6):
            raise ValueError(f"propagation vector must be a unit vector (norm = 1), but got norm = {prop_norm:.6f}.")

        self.propagation_vector=propagation_vector
        self.polarization=polarization
        self.wavenumber=omega*np.sqrt(epsilon*mu)
        self.mu=mu
        self.omega=omega
        self.eta=np.sqrt(mu/epsilon)

    def evaluate_at_points(self,X):
        kx,ky,kz=self.propagation_vector

        phi=np.arctan2(-kx,-ky)
        
        R_z=np.array([[np.cos(phi),np.sin(phi),0],
                      [np.sin(phi),np.cos(phi),0],
                      [   0       ,  0        ,1]])
        rotated_points = (R_z @ X.T).T
        
        theta_i=np.arccos(np.dot(np.array([0,0,-1]),self.propagation_vector))

        beta=self.polarization
        eta=self.eta

        x,y,z=rotated_points[:,0],rotated_points[:,1],rotated_points[:,2]

        exp_term=np.exp( -1j*self.wavenumber*(x*np.sin(theta_i))+z*np.cos(theta_i))
        
        E_perp=np.column_stack( (np.zeros_like(x),np.ones_like(x),np.zeros_like(x)) )*exp_term[:,None]
        H_perp=np.column_stack( (-np.cos(theta_i)*np.ones_like(x) ,np.zeros_like(x) ,np.sin(theta_i)*np.ones_like(x) ))*exp_term[:,None]/eta
        
        E_par=np.column_stack( (np.cos(theta_i)*np.ones_like(x) ,np.zeros_like(x) ,-np.sin(theta_i)*np.ones_like(x) ))*exp_term[:,None]
        H_par=np.column_stack( (np.zeros_like(x),np.ones_like(x),np.zeros_like(x)) )*exp_term[:,None]/eta
        E=np.cos(beta)*E_perp+np.sin(beta)*E_par
        H=np.cos(beta)*H_perp+np.sin(beta)*H_par
        R_inv=R_z.T
        E=(R_inv @ E.T).T
        H=(R_inv @ H.T).T
        return E,H

def get_reflected_field_at_points(points,PW,mu,epsilon_substrate,epsilon_air):
    #---------------------------------------------------------------
    #                     Calculate the angles
    #---------------------------------------------------------------
    nu=np.array([0,1,0])
    eta_substrate=np.sqrt(mu/epsilon_substrate)
    eta_air=np.sqrt(mu/epsilon_air)
    theta_inc=np.abs( np.mod( np.arccos(np.dot(PW.propagation_vector,nu)), np.pi) ) 
    #theta_ref=theta_inc
    theta_trans=np.arcsin(eta_air/eta_substrate*np.sin(theta_inc))

    #---------------------------------------------------------------
    #                       Calculate the fields
    #---------------------------------------------------------------
    
    E_inc,H_inc=PW.evaluate_at_points(points)
    E_perp = np.cos(theta_inc)*E_inc
    H_perp=np.cos(theta_inc)*H_inc
    E_par=np.sin(theta_inc)*E_inc
    H_par=np.sin(theta_inc)*H_inc
    #---------------------------------------------------------------
    #                reflection and transmission coeff
    #---------------------------------------------------------------

    r_perp=(eta_substrate*np.cos(theta_inc)-eta_air*np.cos(theta_trans) ) / ( eta_substrate*np.cos(theta_inc)+eta_air*np.cos(theta_trans) )

    r_par=(eta_substrate*np.cos(theta_trans)-eta_air*np.cos(theta_inc) ) / ( eta_substrate*np.cos(theta_trans)+eta_air*np.cos(theta_inc) )
    E_ref = r_perp * E_perp + r_par * E_par
    # Similarly for the magnetic field:
    H_ref = r_perp * H_perp + r_par * H_par
    return E_ref, H_ref, r_perp,r_par


def compute_fields_from_csv(param_file, testpoints_file, output_file):
    import numpy as np
    import pandas as pd
    # Read parameters
    param_df = pd.read_csv(param_file)
    params = dict(zip(param_df["Parameter"], param_df["Value"]))
    
    # Convert scalar parameters
    mu = float(params["mu"])
    epsilon = float(params["epsilon"])
    omega = float(params["omega"])
    
    # Read propagation vector from the parameters
    propagation_vector = np.array([
        float(params["propagation_x"]),
        float(params["propagation_y"]),
        float(params["propagation_z"])
    ])
    
    # Interpret the polarization parameter as an index: 0->x, 1->y, 2->z.
    polarization = float(params["polarization"])
    # Read test points from CSV (assumes columns are ordered x, y, z)
    testpoints_df = pd.read_csv(testpoints_file)
    testpoints = testpoints_df.to_numpy()  # Convert DataFrame to NumPy array

    # Debug: Print out parameter values and shapes
    print(f"mu: {mu}, epsilon: {epsilon}, omega: {omega}")
    print(f"Propagation vector: {propagation_vector}, Polarization: {polarization}")
    print(f"Testpoints shape: {testpoints.shape}")
    
    # Compute fields (assuming Plane_wave is defined elsewhere)
    PW = Plane_wave(propagation_vector, polarization, epsilon, mu, omega)
    E, H = PW.evaluate_at_points(testpoints)
    
    # Prepare data for saving: split real and imaginary parts
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