import numpy as np

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
            H=np.cos(polarization)*H_perp+np.sin(polarization)*H_par
        
            E=(R_z @ E.T).T
            H=(R_z @ H.T).T
            E_fields[PW_index,:]=E
            H_fields[PW_index,:]=H
        return  E_fields, H_fields



