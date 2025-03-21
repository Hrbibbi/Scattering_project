import numpy as np
class Plane_wave():
    def __init__(self,propagation_vector,polarization,wavenumber):
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

        if np.array_equal(self.propagation_vector,np.array([0,0,1])):
            self.rot_matrix=np.eye(3)
        else:
            N=np.sqrt(self.propagation_vector[0]**2+self.propagation_vector[1]**2)
            self.rot_matrix=np.array([
                [ self.propagation_vector[1]/N , self.propagation_vector[0]*self.propagation_vector[2]/N, self.propagation_vector[0]  ],
                [ -self.propagation_vector[0]/N, self.propagation_vector[1]*self.propagation_vector[2]/N, self.propagation_vector[1]  ],
                [         0           ,                 -N                   , self.propagation_vector[2]  ]
            ])
    def evaluate_at_points(self,X):
        exponential_term=np.exp(1j*self.wavenumber*X[:,1])
        Ex=np.sin(self.polarization)*exponential_term
        Ez=np.zeros_like(Ex)
        Ey=np.cos(self.polarization)*exponential_term
        E_H=np.column_stack((Ex,Ey,Ez))
        Hx=1j*self.wavenumber*np.cos(self.polarization)*exponential_term
        Hz=np.zeros_like(Hx)
        Hy=-1j*self.wavenumber*np.sin(self.polarization)*exponential_term
        H_H=np.column_stack((Hx,Hy,Hz))
        E_rotated = (self.rot_matrix @ E_H.T).T
        H_rotated = (self.rot_matrix @ H_H.T).T
        return [E_rotated,H_rotated]



        