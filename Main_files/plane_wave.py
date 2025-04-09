import numpy as np
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

    def evaluate_at_points(self, X):
        kx, ky, kz = self.propagation_vector
        phi = np.arctan2(-kx, -ky)

        R_z = np.array([
            [np.cos(phi),  np.sin(phi), 0],
            [-np.sin(phi), np.cos(phi), 0],
            [0,            0,           1]
        ])

        X_rot = (R_z @ X.T).T
        k_rot = R_z @ self.propagation_vector
        theta_i = np.arccos(np.dot([0, 0, -1], k_rot))

        # Unit propagation vector (in rotated frame)
        k_hat = np.array([np.sin(theta_i), 0, -np.cos(theta_i)])

        # Polarization decomposition
        beta = self.polarization
        e_perp = np.array([0, 1, 0])
        e_par = np.cross(k_hat, e_perp)
        e_par /= np.linalg.norm(e_par)

        E_hat = np.cos(beta) * e_perp + np.sin(beta) * e_par
        H_hat = np.cross(k_hat, E_hat)

        # Wavenumber and impedance
        k_mag = self.wavenumber
        eta = np.sqrt(self.mu / (self.wavenumber**2 / self.omega**2 * self.mu))

        # Dot product k ⋅ r for phase
        phase = np.dot(X_rot, k_mag * k_hat)
        exp_phase = np.exp(-1j * phase)

        # Fields
        E = E_hat * exp_phase[:, np.newaxis]
        H = H_hat / eta * exp_phase[:, np.newaxis]

        # Rotate fields back to original coordinates
        R_z_inv = R_z.T
        E_global = E @ R_z_inv
        H_global = H @ R_z_inv

        return E_global, H_global
def get_reflected_field_at_points(points,PW,mu,epsilon_substrate,epsilon_air):
    #---------------------------------------------------------------
    #                     Calculate the angles
    #---------------------------------------------------------------
    nu=np.array([0,0,-1])
    eta_substrate=np.sqrt(mu/epsilon_substrate)
    eta_air=np.sqrt(mu/epsilon_air)
    prop_vec=PW.propagation_vector
    theta_inc=np.arccos(np.dot(prop_vec,nu))
    theta_trans=np.emath.arcsin(epsilon_air/epsilon_substrate*np.sin(theta_inc))
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
    H_ref = r_perp * H_perp + r_par * H_par
    return E_ref, H_ref, r_perp,r_par


mu=1
epsilon_air=1
epsilon_substrate=11.6964
omega=1
polarization=0
prop_vec=np.array([0, 1,-1])
prop_vec=prop_vec/np.linalg.norm(prop_vec)
PW1=Plane_wave(prop_vec,polarization,epsilon_air,mu,omega)
print(PW1.evaluate_at_points(np.array([[1,0,1],
                                 [0,1,0]])))