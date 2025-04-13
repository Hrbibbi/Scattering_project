import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D

class C2_surface:
    def __init__(self,points,normals,tau1,tau2):
        self.points=points
        self.normals=normals
        self.tau1=tau1
        self.tau2=tau2
        self.M=np.shape(self.points)[0]

    def plot_tangents(self, scale=0.1):
        """Plots the surface as a solid and overlays tangent vectors tau1, tau2, and normal vectors."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        N = int(np.sqrt(self.points.shape[0]))  # Assuming the points are N x N grid
        X = self.points[:, 0].reshape((N, N))
        Y = self.points[:, 1].reshape((N, N))
        Z = self.points[:, 2].reshape((N, N))

        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

        # Plot tangent vectors (tau1 and tau2)
        for i in range(self.points.shape[0]):
            p = self.points[i]
            t1 = self.tau1[i] * scale
            t2 = self.tau2[i] * scale
            normal = self.normals[i] * scale
            # Tangent vector tau1
            ax.quiver(p[0], p[1], p[2], t1[0], t1[1], t1[2], color='r', length=np.linalg.norm(t1), normalize=True, linewidth=1)
            # Tangent vector tau2
            ax.quiver(p[0], p[1], p[2], t2[0], t2[1], t2[2], color='b', length=np.linalg.norm(t2), normalize=True, linewidth=1)

            # Normal vector (calculated as cross product of tau1 and tau2)
            ax.quiver(p[0], p[1], p[2], normal[0], normal[1], normal[2], color='g', length=scale, normalize=True, linewidth=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Surface with Tangent and Normal Vectors')

        plt.tight_layout()
        plt.show()

def compute_geometric_data(x,y,z,h):
    '''
    We start by assuming that x,y,z are all 2d arrays for ease of computation
    we also assume that x,y is sampled from the same so h=max(x[1,0]-x[0,0]
    '''
    f_y,f_x=np.gradient(z,h,h)
    f_yx,f_xx=np.gradient(f_x,h,h)
    f_yy,f_xy=np.gradient(f_y,h,h)
    x,y,z=x.ravel(),y.ravel(),z.ravel()
    point_cloud=np.column_stack((x,y,z))
    tau1=np.column_stack( (np.ones_like(x),np.zeros_like(x),f_x.ravel()) )
    tau2=np.column_stack( (np.zeros_like(x),np.ones_like(x),f_y.ravel()) )
    
    numerator=(1+f_x**2)*f_yy-2*f_x*f_y*f_xy+(1+f_y**2)*f_xx
    denom=(1+f_x**2+f_y**2)**(3/2)
    mean_curvature=np.abs(numerator/denom)
    mean_curvature=mean_curvature.ravel()
    
    normals=np.cross(tau1,tau2)
    tau2=np.cross(tau1,normals)
    tau1=tau1 / np.linalg.norm(tau1,axis=1,keepdims=True)
    tau2=tau2 / np.linalg.norm(tau2,axis=1,keepdims=True)
    normals=normals / np.linalg.norm(normals,axis=1,keepdims=True)
    
    return point_cloud,tau1,tau2,normals,mean_curvature


def generate_curvature_scaled_offset(points, normals, mean_curvature,scaling):
    safe_c = scaling/np.max(mean_curvature)
    offset_points = points + safe_c * normals
    return offset_points



def plot_surface_with_offset(original_points, offset_points, N):
    """
    Plots both the original surface and the offset surface in the same 3D plot.
    
    Args:
        original_points (Nx3 array): The original surface points.
        offset_points (Nx3 array): The offset surface points (same shape).
        N (int): Grid resolution (assumes square N x N grid).
    """
    X_orig = original_points[:, 0].reshape((N, N))
    Y_orig = original_points[:, 1].reshape((N, N))
    Z_orig = original_points[:, 2].reshape((N, N))

    X_off = offset_points[:, 0].reshape((N, N))
    Y_off = offset_points[:, 1].reshape((N, N))
    Z_off = offset_points[:, 2].reshape((N, N))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot original surface
    surf_orig = ax.plot_surface(X_orig, Y_orig, Z_orig, cmap='viridis', alpha=0.8)
    # Plot offset surface
    surf_off = ax.plot_surface(X_off, Y_off, Z_off, cmap='plasma', alpha=0.6)

    # Create proxy handles for the legend
    legend_orig = mlines.Line2D([], [], color='yellow', label='Original Surface')
    legend_off = mlines.Line2D([], [], color='red', label='Offset Surface')

    # Add legend manually
    ax.legend(handles=[legend_orig, legend_off])

    ax.set_title("Original and Offset Surfaces")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.tight_layout()
    plt.show()


f = lambda x,y: (1+1/2+1/4)+np.cos(x)+1/2*np.sin(4*x)+1/4*np.cos(8*x)
a,b=-np.pi,np.pi
N=50
x0,y0=np.linspace(a,b,N),np.linspace(a,b,N)
x,y=np.meshgrid(x0,y0)
z=f(x,y)
point_cloud,tau1,tau2,normals,mean_curvature=compute_geometric_data(x,y,z,(b-a)/N)
new_points=generate_curvature_scaled_offset(point_cloud,normals,mean_curvature,1)
plot_surface_with_offset(point_cloud,new_points,N)