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

        if np.array_equal(self.propagation_vector,np.array([0,0,-1])):
            self.rot_matrix=np.eye(3)
        elif np.array_equal(self.propagation_vector,np.array([0,0,1])):
            self.rot_matrix=np.array([
                [1,0,0],
                [0,-1,0],
                [0,0,-1]
            ])
        else:
            N=np.sqrt(self.propagation_vector[0]**2+self.propagation_vector[1]**2)
            self.rot_matrix=-1*np.array([
                [ self.propagation_vector[1]/N , self.propagation_vector[0]*self.propagation_vector[2]/N, self.propagation_vector[0]  ],
                [ -self.propagation_vector[0]/N, self.propagation_vector[1]*self.propagation_vector[2]/N, self.propagation_vector[1]  ],
                [         0           ,                 -N                   , self.propagation_vector[2]  ]
            ])
        self.rot_matrix_inv=np.linalg.inv(self.rot_matrix)
    def evaluate_at_points(self,X):
        #---------------------------------------------------------------------------
        #                           Rotate points and precompute
        #---------------------------------------------------------------------------

        rotated_points=(self.rot_matrix_inv @ X.T).T
        exponential_term=np.exp(1j*self.wavenumber*rotated_points[:,2])
        eta=self.omega*self.mu/self.wavenumber


        #---------------------------------------------------------------------------
        #                           Electric field computation
        #---------------------------------------------------------------------------

        Ex=np.sin(self.polarization)*exponential_term
        Ey=np.cos(self.polarization)*exponential_term
        Ez=np.zeros_like(Ex)
        E=np.column_stack((Ex,Ey,Ez))
        #---------------------------------------------------------------------------
        #                           Magnetic field computation
        #---------------------------------------------------------------------------
        Hx=-np.cos(self.polarization)*exponential_term/eta
        Hy=np.sin(self.polarization)*exponential_term/eta
        Hz=np.zeros_like(Hx)
        H=np.column_stack((Hx,Hy,Hz))

        #---------------------------------------------------------------------------
        #                           rotate fields back
        #---------------------------------------------------------------------------
        
        E_rotated = (self.rot_matrix @ E.T).T
        H_rotated = (self.rot_matrix @ H.T).T
        return [E_rotated, H_rotated]

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

'''
mu=1
epsilon_air=1
epsilon_substrate=11.6964
omega=1
polarization=0
PW1=Plane_wave(np.array([0,-1,0]),polarization,epsilon_air,mu,omega)
points=np.array([
    [0,1,0],
    [0,2,0],
    [1,1,1]
])
E_ref,H_ref,r_perp,r_par=get_reflected_field_at_points(points,PW1,mu,epsilon_substrate,epsilon_air)
print(E_ref)
'''