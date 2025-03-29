import numpy as np
import matplotlib.pyplot as plt
import time
def squared_exponential(points,tau,ell):

    mat=points[:,None]-points 
    return tau**2*np.exp(  -np.sum(mat**2,axis=-1) / (2*ell**2) ) 

def squared_exponential_2(a, b, tau = 1, ell = 1):
    return tau**2*np.exp(-((a-b).T @ (a-b))/(2*ell**2))
n=10
grid_param=np.linspace(-1,1,n)
x,y=np.meshgrid(grid_param,grid_param)
points=np.column_stack((x.ravel(),y.ravel()))
mu=np.zeros(n**2)
cov=squared_exponential(points,tau=1,ell=1)

def covariance_python(X,Y,tau,ell):
    n = len(X);
    K = np.zeros((n,n));

    for i in (range(n)):
        for j in range(i,n):
            a = np.array([X[i], Y[i]])
            b = np.array([X[j], Y[j]])
            k = squared_exponential_2(a,b,tau,ell)
            K[i,j] = k
            K[j,i] = k

    return K

#sprint(np.linalg.det(covariance_python(x.flatten(),y.flatten(),1,1)))
#print(np.linalg.det(cov))
plt.imshow(cov)
plt.show()
"""
mu=np.zeros(n**2)
cov=squared_exponential(points,tau=0.5,ell=0.1)
sample=mu + np.linalg.cholesky(cov) @ np.random.standard_normal(mu.size)
plt.imshow( np.reshape(sample,(n,n)) )
plt.show()
"""


""" averagetime_cov=[]
averagetime_sample=[]
for i in range(2,11):
    n=5*i
    grid_param=np.linspace(-1,1,n)
    x,y=np.meshgrid(grid_param,grid_param)
    points=np.column_stack((x.ravel(),y.ravel()))
    mu=np.zeros(n**2)

    start=time.time()
    for iter in range(10):
        squared_exponential(points,1,1)
    averagetime_cov.append((time.time()-start)/10)

    cov=squared_exponential(points,1,0.1)
    #plt.imshow(cov)
    #plt.show()
    print(np.linalg.det(cov))
    start=time.time()
    for iter in range(10):
        #np.random.multivariate_normal(mu,cov)
        mu + np.linalg.cholesky(cov) @ np.random.standard_normal(mu.size)
    averagetime_sample.append((time.time()-start)/10)
    #print(f"start {time.time()-start}") """

#plt.plot(np.arange(2,11)*3,np.array(averagetime_sample)/np.array(averagetime_cov))
#plt.show()

#start=time.time()
#sample=np.random.multivariate_normal(mu,cov)
#print(f"start {time.time()-start}")
#sample=np.reshape(sample,(n,n))
#plt.imshow(sample)
#plt.show()