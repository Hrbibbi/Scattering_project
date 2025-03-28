import numpy as np
class Plane_wave():
    def __init__(self,propagation_vector,polarization,wavenumber,mu,omega):
        '''
        Check input
        '''
        for param, name in zip([wavenumber,polarization], ["wavenumber","polarization"]):
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
        self.wavenumber=wavenumber
        self.mu=mu
        self.omega=omega

        if np.array_equal(self.propagation_vector,np.array([0,0,1])):
            self.rot_matrix=np.eye(3)
        elif np.array_equal(self.propagation_vector,np.array([0,0,-1])):
            self.rot_matrix=np.array([
                [1,0,0],
                [0,-1,0],
                [0,0,-1]
            ])
            print("hej")
        else:
            N=np.sqrt(self.propagation_vector[0]**2+self.propagation_vector[1]**2)
            self.rot_matrix=np.array([
                [ self.propagation_vector[1]/N , self.propagation_vector[0]*self.propagation_vector[2]/N, self.propagation_vector[0]  ],
                [ -self.propagation_vector[0]/N, self.propagation_vector[1]*self.propagation_vector[2]/N, self.propagation_vector[1]  ],
                [         0           ,                 -N                   , self.propagation_vector[2]  ]
            ])
        self.rot_matrix_inv=np.linalg.inv(self.rot_matrix)
        print(self.rot_matrix)
    def evaluate_at_points(self,X):
        #---------------------------------------------------------------------------
        #                           Rotate points and precompute
        #---------------------------------------------------------------------------

        rotated_points=(self.rot_matrix_inv @ X.T).T
        exponential_term=np.exp(1j*self.wavenumber*rotated_points[:,2])
        eta=self.wavenumber/(self.omega*self.mu)


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
        Hx=eta*np.cos(self.polarization)*exponential_term
        Hy=-eta*np.sin(self.polarization)*exponential_term
        Hz=np.zeros_like(Hx)
        H=np.column_stack((Hx,Hy,Hz))

        #---------------------------------------------------------------------------
        #                           rotate fields back
        #---------------------------------------------------------------------------
        
        E_rotated = (self.rot_matrix @ E.T).T
        H_rotated = (self.rot_matrix @ H.T).T
        return [E_rotated, H_rotated]

def get_reflected_field_at_points(points,PW,eta1,eta2):
    #---------------------------------------------------------------
    #                       Calculate the fields
    #---------------------------------------------------------------

    nu=np.array([0,1,0])
    E_inc,H_inc=PW.evaluate_at_points(points)
    E_perp=np.sum(E_inc*nu,axis=1)
    E_perp = E_perp[:, None] * nu
    E_par=E_inc-E_perp

    H_perp = np.sum(H_inc * nu, axis=1)[:, None] * nu
    H_par = H_inc - H_perp

    #---------------------------------------------------------------
    #                     Calculate the angles
    #---------------------------------------------------------------
    
    theta_inc=np.arccos(np.dot(PW.propagation_vector,nu))
    theta_ref=theta_inc
    theta_trans=np.arcsin(eta1/eta2*np.sin(theta_inc))
    #---------------------------------------------------------------
    #                reflection and transmission coeff
    #---------------------------------------------------------------

    r_perp = (eta2 * np.cos(theta_inc) - eta1 * np.cos(theta_trans)) / \
             (eta2 * np.cos(theta_inc) + eta1 * np.cos(theta_trans))
    
    r_par = (eta1 * np.cos(theta_inc) - eta2 * np.cos(theta_trans)) / \
            (eta1 * np.cos(theta_inc) + eta2 * np.cos(theta_trans))

    t_perp= 2*eta2*np.cos(theta_inc) / (eta2*np.cos(theta_inc)+eta1*np.cos(theta_trans))
    t_per= 2*eta2*np.cos(theta_inc) / (eta2*np.cos(theta_trans)+eta1*np.cos(theta_inc))
    
    E_ref = r_perp * E_perp + r_par * E_par
    
    # Similarly for the magnetic field:
    H_ref = r_perp * H_perp + r_par * H_par
    return E_ref, H_ref, r_perp,r_par

'''
PW1=Plane_wave(np.array([0,1,0]),0,1,1,1)
points=np.array([ [1,0,0] , [0,1,0], [0,0,1]])
E,H=PW1.evaluate_at_points(points)
#print(E)
E_ref,H_ref,r_perp,r_par=get_reflected_field_at_points(points,PW1,1,2)
print(r_perp)
'''