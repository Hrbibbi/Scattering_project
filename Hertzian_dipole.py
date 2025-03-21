import numpy as np
import warnings
import multiprocessing

class Hertzian_Dipole():
    def __init__(self,position,direction,mu,epsilon,omega):
        '''
        Check input
        '''
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

        self.mu=mu
        self.epsilon=epsilon
        self.omega=omega
        self.wavenumber=omega*np.sqrt(epsilon*mu)

        #vector values
        self.position = position
        self.direction = direction
        if np.array_equal(self.direction,np.array([0,0,1])):
            self.rot_matrix=np.eye(3)
        else:
            N=np.sqrt(self.direction[0]**2+self.direction[1]**2)
            self.rot_matrix=np.array([
                [ self.direction[1]/N , self.direction[0]*self.direction[2]/N, self.direction[0]  ],
                [ -self.direction[0]/N, self.direction[1]*self.direction[2]/N, self.direction[1]  ],
                [         0           ,                 -N                   , self.direction[2]  ]
            ])
    
    def evaluate_at_points(self,X):
        p=X-self.position
        r=np.sqrt(np.sum(p**2,axis=1))
        exponential_term=np.exp(-1j*self.wavenumber*r)
        electric_constant=1/(4*np.pi*self.omega*self.epsilon)
        E_x=electric_constant*p[:,0]*p[:,2]*( (1j*self.wavenumber**2)/(r**3) + (3*self.wavenumber)/(r**4)- (3j)/(r**5) )*exponential_term
        E_y=electric_constant*p[:,1]*p[:,2]*( (1j*self.wavenumber**2)/(r**3) + (3*self.wavenumber)/(r**4)- (3j)/(r**5) )*exponential_term
        E_z=electric_constant*( (-3j*p[:,2]**2)/(r**5) + (3*self.wavenumber*p[:,2])/(r**4) + 
                               (1j*(self.wavenumber**2*p[:,2]**2+1))/(r**3) - (self.wavenumber)/(r**2) )*exponential_term -(
                                1j*self.omega*self.mu/(4*np.pi*r)*exponential_term
                               )
        E_H = np.column_stack((E_x, E_y, E_z))
        H_x=-p[:,1]/(4*np.pi)*( 1j*self.wavenumber/r**2 + 1/r**3)*exponential_term
        H_y=-p[:,0]/(4*np.pi)*( 1j*self.wavenumber/r**2 + 1/r**3)*exponential_term
        H_z=np.zeros(np.shape(H_x))
        H_H=np.column_stack((H_x,H_y,H_z)) 
        E_rotated = (self.rot_matrix @ E_H.T).T
        H_rotated = (self.rot_matrix @ H_H.T).T
        return [E_rotated,H_rotated]
    
    def crossprod(self,points,vectors):
        E_H,H_H=self.evaluate_at_points(points)
        E_C=np.cross(E_H,vectors)
        H_C=np.cross(H_H,vectors)
        return [E_C,H_C]

def construct_Hertzian_Dipoles(positions,directions,mus,epsilons,omegas):
    return [Hertzian_Dipole(positions[idx,:],directions[idx,:],mus[idx],epsilons[idx],omegas[idx]) for idx in range(len(mus))]

def evaluate_Hertzian_Dipoles_at_points(points,Dipoles):
    evaluations=[]
    for Dipole in Dipoles:
        evaluations.append(Dipole.evaluate_at_points(points))
    return evaluations

def evaluate_dipole(args):
    dipole, points = args
    return dipole.evaluate_at_points(points)

def evaluate_Hertzian_Dipoles_at_points_parallel(points, Dipoles):
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        evaluations = pool.map(evaluate_dipole, [(dipole, points) for dipole in Dipoles])
    return evaluations



'''
N=150
mus=np.ones(N)
epsilons=np.ones(N)
omegas=np.ones(N)
positions=1+np.random.rand(N,3)
directions=np.zeros([N,3])
directions[:,2]=1
import time
Dipoles=construct_Hertzian_Dipoles(positions,directions,mus,epsilons,omegas)
n=250
grid_param=np.linspace(-1,1,n)
x,y=np.meshgrid(grid_param,grid_param)
points=np.column_stack( (x.ravel(),y.ravel(),np.ones_like(x.ravel()) ))
print(np.shape(points))
#E,H=H1.evaluate_at_points(points)


print(f"Number of dipoles {N}")
start=time.time()
evaluate_Hertzian_Dipoles_at_points(points,Dipoles)
print(f"time for sequential {time.time()-start}")
start=time.time()
evaluate_Hertzian_Dipoles_at_points_parallel(points,Dipoles)
print(f"time for parallel {time.time()-start}")
'''