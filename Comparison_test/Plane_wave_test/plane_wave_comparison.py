import numpy as np
import ast  # For safely evaluating string representations of lists
import pandas as pd
class Plane_wave:
    def __init__(self, propagation_vectors, polarizations, epsilon, mu, omega):
        """
        propagation_vectors: Mx3 array of unit propagation directions
        polarizations: M array of polarization angles (in radians)
        epsilon, mu: scalar material constants
        omega: scalar angular frequency
        """
        propagation_vectors = np.asarray(propagation_vectors)
        polarizations = np.asarray(polarizations)

        if propagation_vectors.ndim != 2 or propagation_vectors.shape[1] != 3:
            raise ValueError(f"propagation_vectors must be Mx3 array. Got shape {propagation_vectors.shape}")
        if polarizations.ndim != 1 or polarizations.shape[0] != propagation_vectors.shape[0]:
            raise ValueError("polarizations must be a 1D array of the same length as propagation_vectors.")

        if not np.all((polarizations >= 0) & (polarizations <= np.pi/2)):
            raise ValueError("Polarization angles must be in range [0, π/2].")

        norms = np.linalg.norm(propagation_vectors, axis=1)
        if not np.allclose(norms, 1, atol=1e-6):
            raise ValueError("All propagation vectors must be unit vectors.")

        self.propagation_vectors = propagation_vectors  # Mx3
        self.polarizations = polarizations              # M
        self.omega = omega
        self.mu = mu
        self.epsilon = epsilon
        self.wavenumber = omega * np.sqrt(epsilon * mu)  # scalar
        self.eta = np.sqrt(mu / epsilon)

    def evaluate_at_points(self,X):
        
        M=self.propagation_vectors.shape[0]
        N=X.shape[0]
        
        E_fields = np.zeros((M, N, 3), dtype=complex)
        H_fields = np.zeros((M, N, 3), dtype=complex)

        for PW_index in range(M):
            k=self.propagation_vectors[PW_index,:]
            polarization=self.polarizations[PW_index]
            kxy=-k[:2]
            phi=np.arctan2(kxy[1],kxy[0])
            R_z=np.array([
                [np.cos(phi),np.sin(phi),0],
                [-np.sin(phi),np.cos(phi), 0],
                [0,0,1]
                        ])
            
            R_inv=R_z.T
            k_rot= R_inv@k
            X_rot=(R_inv @ X.T).T

            x,y,z=X_rot[:,0],X_rot[:,1],X_rot[:,2]
            theta=np.arccos(-k_rot[2])

            exp_term=np.exp( -1j*self.wavenumber* ( x*np.sin(theta)-z*np.cos(theta) ) )
            oner,zoer=np.ones_like(x),np.zeros_like(x)

            E_perp=np.column_stack( (zoer,oner*exp_term,zoer) )
            E_par =np.column_stack( (oner*np.cos(theta)*exp_term,zoer,oner*np.sin(theta)*exp_term) )

            H_perp=np.column_stack( (-oner*np.cos(theta)*exp_term,zoer,-oner*np.sin(theta)*exp_term) )/self.eta
            H_par =np.column_stack( (zoer,oner*exp_term,zoer))/self.eta
            
            E=np.cos(polarization)*E_perp+np.sin(polarization)*E_par
            H=np.sin(polarization)*H_perp+np.cos(polarization)*H_par
        
            E=(R_z @ E.T).T
            H=(R_z @ H.T).T
            E_fields[PW_index,:]=E
            H_fields[PW_index,:]=H
        return  E_fields, H_fields

def compute_flux_integral(plane, planewave):
    '''
    Computes the average power (flux) integral for the scattered field for multiple RHSs.

    Input:
        plane: A C2_object (with .points and .normals)
        planewave

    Output:
        flux_values: Array of shape (R,) — power flux per RHS
    '''
    #---------------------------------------------------------------------
    # Extract geometry
    #---------------------------------------------------------------------
    points = plane.points             # (N, 3)
    normals = plane.normals           # (N, 3)

    dx = np.linalg.norm(points[1] - points[0])
    dA = dx * dx                      # Scalar area element (uniform)

    #---------------------------------------------------------------------
    # Evaluate scattered fields: (2, M, N, 3)
    #---------------------------------------------------------------------
    E, H = planewave.evaluate_at_points(points) #E and H (R,N,3) each
    
    R, N , _ = np.shape(E)
    Cross = 0.5 * np.cross(E, np.conj(H)) # (R,N,3)
    integrands = np.einsum("rnk,nk->rn", Cross,normals) # (R,N)

    integrals = np.einsum("rn -> r", integrands*dA)    # (M,)
    return integrals  # Return real-valued power flux




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
    propagation_vector = np.array([[
        float(params["propagation_x"]),
        float(params["propagation_y"]),
        float(params["propagation_z"])
    ]])
    
    # Interpret the polarization parameter as an index: 0->x, 1->y, 2->z.
    polarization = np.array([float(params["polarization"])])
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
    E, H = E[0], H[0]
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