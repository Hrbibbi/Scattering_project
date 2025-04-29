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
        self.eta=np.sqrt(mu/epsilon)

    def evaluate_at_points(self,X):
        k=self.propagation_vector
        kxy=-k[:2]
        phi=np.arctan2(kxy[1],kxy[0])
        R_z=np.array([
            [np.cos(phi),-np.sin(phi),0],
            [np.sin(phi),np.cos(phi), 0],
            [0,0,1]
        ])
        k_rot= R_z@k
        X_rot=(R_z @ X.T).T
        x,y,z=X_rot[:,0],X_rot[:,1],X_rot[:,2]
        theta=np.arccos(-k_rot[2])

        exp_term=np.exp( -1j*self.wavenumber* ( x*np.sin(theta)-z*np.cos(theta) ) )
        oner,zoer=np.ones_like(x),np.zeros_like(x)

        E_perp=np.column_stack( (zoer,oner*exp_term,zoer) )
        E_par =np.column_stack( (-oner*np.cos(theta)*exp_term,zoer,-oner*np.sin(theta)*exp_term) )

        H_perp=np.column_stack( (-oner*np.cos(theta)*exp_term,zoer,-oner*np.sin(theta)*exp_term) )/self.eta
        H_par =np.column_stack( (zoer,oner*exp_term,zoer))/self.eta
        E=np.cos(self.polarization)*E_perp+np.sin(self.polarization)*E_par
        H=np.cos(self.polarization)*H_perp+np.sin(self.polarization)*H_par
        R_inv=R_z.T
        E=(R_inv @ E.T).T
        H=(R_inv @ H.T).T
        return E,H

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
testpotins=np.random.rand(100,3)
PW1.evaluate_at_points(testpotins)
