import Hertzian_dipole as HD
import numpy as np
import time
k=1
p=np.random.rand(k,3)
vals=np.random.rand(k,3)
dir=vals/np.linalg.norm(vals,axis=1,keepdims=True)
mus=np.ones(k)
eps=np.ones(k)
om=np.ones(k)
dipoles=HD.construct_Hertzian_Dipoles(p,dir,mus,eps,om)
points=1+np.random.rand(10**6,3)
#HD.evaluate_Hertzian_Dipoles_at_points_parallel(points,dipoles)
DP1=dipoles[0]
start=time.time()
DP1.evaluate_at_points(points)
print(f"{time.time()-start}")