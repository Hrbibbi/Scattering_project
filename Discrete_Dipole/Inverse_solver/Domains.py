import numpy as np
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

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

def matern_covariance_matrix(X_samples, Y_samples, Z_samples, length_scale=1, nu=0.5, jitter=1e-6):
    """
    Compute the Matern covariance matrix using strided sampling.

    Returns:
        chol_fact: (M, M) Cholesky factor of covariance matrix
        X_samples, Y_samples, Z_samples: strided arrays of sampled coordinates
    """

    points = np.column_stack((X_samples, Y_samples, Z_samples))  # shape (M, 3)
    kernel = Matern(length_scale=length_scale, nu=nu)
    K = kernel(points) + jitter * np.eye(len(points))

    chol_fact = np.linalg.cholesky(K)
    return chol_fact

def sample_gp_field(chol_fact, num_fields=3):
    """
    Sample raw GP values given Cholesky factor.

    Returns:
        raw_samples: (M, num_fields) raw GP values at sampled locations
    """
    M = chol_fact.shape[0]
    xi = np.random.randn(M, num_fields)
    return chol_fact @ xi

def gp_to_euler_angles(gp_samples, X, a=1, b=1):
    """
    Interpolate GP samples in Euler angle domain over full (X,Y,Z).

    Returns:
        (N, 3): interpolated Euler angles over full domain
    """
    M, D = gp_samples.shape
    assert D == 3
    N_full = len(X)

    def sigmoid(x, scale):
        return 1 / (1 + np.exp(-scale * x))

    # Map sampled GPs to Euler angles
    theta = np.pi * sigmoid(gp_samples[:, 0], a)
    phi   = (np.pi / 2) * sigmoid(gp_samples[:, 1], b)
    psi   = (np.pi / 2) * sigmoid(gp_samples[:, 2], b)

    x_samples=np.linspace(0,1,M)
    x_full   =np.linspace(0,1,N_full)
    theta_interp=np.interp(x_full,x_samples,theta)
    phi_interp=np.interp(x_full,x_samples,phi)
    psi_interp=np.interp(x_full,x_samples,psi)

    return np.stack([theta_interp, phi_interp, psi_interp], axis=-1)

class Domain:
    def __init__(self,X,Y,Z):
        self.X=X
        self.Y=Y
        self.Z=Z
        X_flat=X.ravel()
        Y_flat=Y.ravel()
        Z_flat=Z.ravel()
        
        self.points=np.column_stack([X_flat,Y_flat,Z_flat])

def check_bounds():
    X, Y, Z = cylinder_cartesian_grid(1, 1, numpoints_xy=4, numpoints_z=4)
    Chol_fact=matern_covariance_matrix(X,Y,Z)

    for i in range(50):
        sample=sample_gp_field(Chol_fact,3)
        angles=gp_to_euler_angles(sample,X)
        plt.plot(angles[:,0],'.')
    plt.show()
