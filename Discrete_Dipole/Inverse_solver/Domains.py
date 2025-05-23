import numpy as np
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt

def cylinder_cartesian_grid(height, radius,offset=[0,0], numpoints_xy=10, numpoints_z=10):
    """
    Generate a 3D Cartesian grid of points uniformly filling the volume of a cylinder.

    Parameters:
        height (float): Height of the cylinder along the z-axis.
        radius (float): Radius of the cylinder in the xy-plane.
        numpoints_xy (int): Approximate number of points across the diameter (x and y).
        numpoints_z (int): Number of points along the height (z).

    Returns:
        X, Y, Z (ndarrays): 1D arrays of Cartesian coordinates for points inside the cylinder.
    """
    # Uniform grid in the cube enclosing the cylinder
    x = np.linspace(-radius, radius, numpoints_xy)
    y = np.linspace(-radius, radius, numpoints_xy)
    z = np.linspace(0, height, numpoints_z)
    x0,y0=offset
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten and filter points inside the cylinder
    Xf = X.flatten()
    Yf = Y.flatten()
    Zf = Z.flatten()
    
    mask = Xf**2 + Yf**2 <= radius**2
    
    return Xf[mask]-x0, Yf[mask]-y0, Zf[mask]

X, Y, Z = cylinder_cartesian_grid(height=1.0, radius=0.5, numpoints_xy=20, numpoints_z=20)

def matern_covariance_matrix(X, Y, Z, kernel=None):
    """
    Compute the Matern covariance matrix for 3D points inside a cylinder.

    Parameters:
        X, Y, Z (1D arrays): Coordinates of the points (same length N).
        kernel (sklearn.gaussian_process.kernels.Matern or None): 
            If None, a default Matern kernel is used (nu=1.5, length_scale=1.0).

    Returns:
        chol_fact (ndarray): cholesky factorization of  NxN covariance matrix.
    """
    points = np.column_stack((X, Y, Z))  # shape (N, 3)

    # Use default Matern kernel if none provided
    if kernel is None:
        kernel = Matern(length_scale=1.0, nu=1.5)

    # Compute covariance matrix
    K = kernel(points)
    chol_fact=np.linalg.cholesky(K)
    return chol_fact

def generate_sample(GP_factorization):
    '''
    Function that generates a sample from the Gaussian process

    Input:
        The cholesky factorization of the covariance matrix for the GP
    
    Output:
        sample ready for for the forward solver wrapper
    '''
    M,N=np.shape(GP_factorization)
    Normal_sample=np.random.normal(size=N)
    
    return GP_factorization @ Normal_sample

def generate_tensor_sample(GP_factorization):
    '''
    Generate a sample of a symmetric 3x3 tensor field from a Gaussian Process.

    Input:
        GP_factorization: (N, N) Cholesky factor of the covariance matrix
    
    Output:
        alpha: (N, 3, 3) array representing the full tensor at each of N points
    '''
    N = GP_factorization.shape[0]

    # Sample 9 independent GP coefficient vectors (one for each tensor entry)
    alpha = np.empty((N, 3, 3), dtype=np.float64)

    for i in range(3):
        for j in range(3):
            normal_sample = np.random.normal(size=N)
            alpha[:, i, j] = GP_factorization @ normal_sample
    
    return alpha

class Domain:
    def __init__(self,X,Y,Z,alpha):
        self.X=X
        self.Y=Y
        self.Z=Z
        X_flat=X.ravel()
        Y_flat=Y.ravel()
        Z_flat=Z.ravel()

        self.points=np.column_stack([X_flat,Y_flat,Z_flat])
        self.alpha_tensor=alpha

