import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

    def plot_tangents(self, scale=1):
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

def Set_dipoles_pr_WL(surface, inneraux, outeraux, lam, points_per_wavelength_surface=10, points_per_wavelength_aux=5):
    '''
    Reduces surface to have points_per_wavelength_surface samples per wavelength,
    and inneraux/outeraux to have points_per_wavelength_aux samples per wavelength.
    Assumes x-y grids are square and regularly spaced.
    
    Input:
        surface, inneraux, outeraux: C2_surface objects (must be same size initially)
        lam: wavelength (scalar)
        points_per_wavelength_surface: for true surface (default 10)
        points_per_wavelength_aux: for auxiliary surfaces (default 5)

    Output:
        reduced_surface, reduced_inneraux, reduced_outeraux: reduced C2_surface objects
    '''
    # -------------------------------
    # 1. Compute scaling from main surface
    # -------------------------------
    x = surface.points[:, 0]
    surface_size = np.max(x) - np.min(x)
    scale = surface_size / lam
    print(f"Wavelength scale: {scale}")
    N_points = surface.points.shape[0]
    N_side = int(np.sqrt(N_points))
    if N_side**2 != N_points:
        raise ValueError("Surface points do not form a perfect square grid!")

    # -------------------------------
    # 2. Define internal helper
    # -------------------------------
    def reduce_surface(surf, step):
        points_grid = surf.points.reshape((N_side, N_side, 3))
        tau1_grid = surf.tau1.reshape((N_side, N_side, 3))
        tau2_grid = surf.tau2.reshape((N_side, N_side, 3))
        normals_grid = surf.normals.reshape((N_side, N_side, 3))

        reduced_points = points_grid[::step, ::step].reshape(-1, 3)
        reduced_tau1 = tau1_grid[::step, ::step].reshape(-1, 3)
        reduced_tau2 = tau2_grid[::step, ::step].reshape(-1, 3)
        reduced_normals = normals_grid[::step, ::step].reshape(-1, 3)

        return C2_surface(
            points=reduced_points,
            tau1=reduced_tau1,
            tau2=reduced_tau2,
            normals=reduced_normals
        )

    # -------------------------------
    # 3. Compute different step sizes
    # -------------------------------
    total_points_surface = int(np.ceil(points_per_wavelength_surface * scale))
    step_surface = max(1, N_side // total_points_surface)

    total_points_aux = int(np.ceil(points_per_wavelength_aux * scale))
    step_aux = max(1, N_side // total_points_aux)

    # -------------------------------
    # 4. Reduce with appropriate steps
    # -------------------------------
    reduced_surface = reduce_surface(surface, step_surface)
    reduced_inneraux = reduce_surface(inneraux, step_aux)
    reduced_outeraux = reduce_surface(outeraux, step_aux)

    print(f"Resulting matrix size: {4*np.shape(reduced_surface.points)[0]}x {4*np.shape(reduced_inneraux.points)[0]}")
    return reduced_surface, reduced_inneraux, reduced_outeraux

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

def generate_plane_xy(height,a,b,numpoints):
    x0,y0=np.linspace(a,b,numpoints),np.linspace(a,b,numpoints)
    x,y=np.meshgrid(x0,y0)
    x,y=x.ravel(),y.ravel()
    points=np.column_stack( (x,y,height*np.ones_like(x)) )
    normals=np.zeros_like(points)
    normals[:,2]=1
    tau1=normals
    tau2=normals
    return C2_surface(points,normals,tau1,tau2)

def cylinder(radius,height,num_points):
    r=radius
    theta0=np.linspace(0,2*np.pi,num_points)
    y0=np.linspace(0,height,num_points)
    theta,y=np.meshgrid(theta0,y0)
    x=r*np.cos(theta)
    z=r*np.sin(theta)
    points=np.column_stack((x.ravel(),y.ravel(),z.ravel()))
    normals=np.column_stack( ( x.ravel()/r, np.zeros_like( y.ravel() ), z.ravel()/r ) )
    tau1=np.zeros_like(points)
    tau1[:,0],tau1[:,2]=-np.sin(theta.ravel()),np.cos(theta.ravel())
    tau2=np.zeros_like(points)
    tau2[:,1]=1

    return C2_surface(points,normals,tau1,tau2)














'''
def sphere(radius,center,num_points):
    r=radius
    x0,y0,z0=center
    theta0=np.linspace(0,np.pi,num_points+2)[1:-1]
    phi0=np.linspace(0,2*np.pi,num_points+2)[1:-1]
    theta,phi=np.meshgrid(theta0,phi0)
    x=x0+r*np.sin(theta)*np.cos(phi)
    y=y0+r*np.sin(theta)*np.sin(phi)
    z=z0+r*np.cos(theta)
    points=np.column_stack((x.ravel(),y.ravel(),z.ravel()))
    normals=points/ np.linalg.norm(points,axis=1,keepdims=True)
    tau1 = np.column_stack((
        r * np.cos(theta).ravel() * np.cos(phi).ravel(),
        r * np.cos(theta).ravel() * np.sin(phi).ravel(),
        -r * np.sin(theta).ravel()
    ))
    tau1=tau1 / np.linalg.norm(tau1,axis=1,keepdims=True)

    tau2 = np.column_stack((
        -r * np.sin(theta).ravel() * np.sin(phi).ravel(),
        r * np.sin(theta).ravel() * np.cos(phi).ravel(),
        np.zeros_like(tau1[:, 0])
    ))
    tau2=tau2 / np.linalg.norm(tau2,axis=1,keepdims=True)
    return C2_surface(points,normals,tau1,tau2)


def inverted_parabola(numpoints):
    r0=np.linspace(0,1,numpoints+1)[1:]
    theta0=np.linspace(0,2*np.pi,numpoints)
    r,theta=np.meshgrid(r0,theta0)
    z=1-r**2
    x=r*np.cos(theta)
    y=r*np.sin(theta)
    points=np.column_stack((x.ravel(),y.ravel(),z.ravel()))
    normals=points
    tau1=np.zeros_like(points)
    tau1[:,0],tau1[:,1]=-np.sin(theta.ravel()),np.cos(theta.ravel())
    tau2=np.zeros_like(points)
    tau2[:,0],tau2[:,1],tau2[:,2]=x.ravel(),y.ravel(),-2*r.ravel()
    tau1=tau1 / np.linalg.norm(tau1,axis=1,keepdims=True)
    tau2=tau2 / np.linalg.norm(tau2,axis=1,keepdims=True)
    return C2_surface(points,normals,tau1,tau2)

def bump_function(center, radius,height):
    def f(x,y):
        r=np.sqrt((x-center[0])**2+(y-center[1])**2)
        inside=r<radius
        result=np.zeros_like(x)
        result[inside]=height*np.exp( 1/((r[inside]-radius)*(radius-r[inside])) )
        
        
    return f

def nipples(centers, radii, heights, grid_length, length_number, grid_width, width_number):
    x0 = np.linspace(0, grid_length, length_number)
    y0 = np.linspace(0, grid_width, width_number)
    dx = x0[1] - x0[0]
    dy = y0[1] - y0[0]
    x, y = np.meshgrid(x0, y0)
    
    # Compute z(x, y)
    z = np.zeros_like(x)
    for center, radius, height in zip(centers, radii, heights):
        bf = bump_function(center, radius, height)
        z += bf(x, y)

    # Compute gradients
    dz_dx, dz_dy = np.gradient(z, dx, dy)

    # Flatten everything
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    dz_dx = dz_dx.ravel()
    dz_dy = dz_dy.ravel()

    # Points array
    points = np.column_stack((x, y, z))

    # Normal vectors
    normals = np.column_stack((-dz_dx, -dz_dy, np.ones_like(z)))
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]

    # Tangent vectors
    tau1 = np.column_stack((np.ones_like(z), np.zeros_like(z), dz_dx))
    tau1 /= np.linalg.norm(tau1, axis=1)[:, np.newaxis]

    tau2 = np.column_stack((np.zeros_like(z), np.ones_like(z), dz_dy))
    tau2 /= np.linalg.norm(tau2, axis=1)[:, np.newaxis]

    return C2_surface(points, normals, tau1, tau2)
'''