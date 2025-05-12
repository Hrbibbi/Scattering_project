import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator

class C2_surface:
    def __init__(self,points,normals,tau1,tau2):
        self.points=points
        self.normals=normals
        self.tau1=tau1
        self.tau2=tau2
        self.M=np.shape(self.points)[0]

def compute_geometric_data(x,y,z):
    '''
    We start by assuming that x,y,z are all 2d arrays for ease of computation
    we also assume that x,y is sampled from the same so h=max(x[1,0]-x[0,0]
    '''
    h=x[0,1]-x[0,0]
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

def build_closed_surface_from_graph(x, y, f):
    
    # Ensure x, y, f are 2D arrays
    assert x.ndim == y.ndim == f.ndim == 2

    # Compute top surface
    z_top = f
    top_pts, top_tau1, top_tau2, top_normals, _ = compute_geometric_data(x, y, z_top)

    # Compute bottom surface (flat at z=0)
    z_bot = np.zeros_like(f)
    bot_pts, bot_tau1, bot_tau2, bot_normals, _ = compute_geometric_data(x, y, z_bot)
    bot_normals *= -1  # Invert normals to point outward

    # Initialize lists to accumulate all surface patches
    all_pts = [top_pts, bot_pts]
    all_tau1 = [top_tau1, bot_tau1]
    all_tau2 = [top_tau2, bot_tau2]
    all_normals = [top_normals, bot_normals]

    def add_side_patch(x_edge, y_edge, f_edge,sign, axis='x'):
        """
        Build a dense side patch (with full vertical resolution) for the closed surface.
        
        Parameters:
        - x_edge, y_edge, f_edge: 1D arrays (N,) of the edge of the top surface
        - axis: 'x' or 'y' indicating orientation of the edge
        
        Returns:
        - pts, tau1, tau2, normals for the side patch
        """
        N = len(x_edge)
        Z = np.linspace(0, 1, N)[:, None]  # vertical linspace shape (N, 1)

        # Prepare vertical stacking
        x_side = np.tile(x_edge[None, :], (N, 1))
        y_side = np.tile(y_edge[None, :], (N, 1))
        f_side = np.tile(f_edge[None, :], (N, 1))
        z_vals = z_vals = Z * f_side

        pts = np.column_stack((x_side.ravel(), y_side.ravel(), z_vals.ravel()))

        # Tangents
        if axis == 'x':
            # Varying in x direction (left/right walls)
            tau1 = np.column_stack((sign*np.ones(N*N), np.zeros(N*N), np.zeros(N*N)))
        else:
            # Varying in y direction (top/bottom walls)
            tau1 = np.column_stack((np.zeros(N*N), sign*np.ones(N*N), np.zeros(N*N)))

        # Vertical tangents (z-direction)
        dz = np.gradient(z_vals, axis=0)
        tau2 = np.column_stack((
            np.zeros_like(dz).ravel(), 
            np.zeros_like(dz).ravel(), 
            dz.ravel()
        ))

        # Normalize tangents
        tau1 = tau1 / np.linalg.norm(tau1, axis=1, keepdims=True)
        tau2 = tau2 / np.linalg.norm(tau2, axis=1, keepdims=True)

        # Normals
        normals = np.cross(tau1, tau2)
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        return pts, tau1, tau2, normals

    # Add sides (loop over 4 boundaries)
    Ny, Nx = f.shape

    edges = [
        (x[:, 0], y[:, 0], f[:, 0],-1, 'y'),   # x = a (left edge, vertical in y)
        (x[:, -1], y[:, -1], f[:, -1],1, 'y'), # x = b (right edge)
        (x[0, :], y[0, :], f[0, :],1, 'x',),   # y = a (bottom edge, vertical in x)
        (x[-1, :], y[-1, :], f[-1, :],-1, 'x'), # y = b (top edge)
    ]

    for x_edge, y_edge, f_edge, axis, sign in edges:
        pts, tau1, tau2, normals = add_side_patch(x_edge, y_edge, f_edge, axis, sign)
        all_pts.append(pts)
        all_tau1.append(tau1)
        all_tau2.append(tau2)
        all_normals.append(normals)

    # Concatenate all
    points = np.vstack(all_pts)
    tau1 = np.vstack(all_tau1)
    tau2 = np.vstack(all_tau2)
    normals = np.vstack(all_normals)

    return C2_surface(points, normals, tau1, tau2)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_surface(surface, show_normals=False, normal_scale=0.1, stride=10):
    """
    Visualizes a C2_surface object.
    
    Parameters:
    - surface: C2_surface object
    - show_normals: whether to plot the normals
    - normal_scale: length of normal vectors when plotted
    - stride: step size for downsampling normals (for readability)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    pts = surface.points
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1, color='lightblue', alpha=0.7)

    if show_normals:
        idx = np.arange(0, surface.M, stride)
        ax.quiver(
            pts[idx, 0], pts[idx, 1], pts[idx, 2], 
            surface.normals[idx, 0], surface.normals[idx, 1], surface.normals[idx, 2], 
            length=normal_scale, color='red', normalize=True
        )

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('C2 Surface Visualization')
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.show()


def generate_curvature_scaled_offset(points, normals, mean_curvature,scaling):
    if np.max(mean_curvature)==0:
        safe_c = scaling
    else:
        safe_c = scaling/np.max(mean_curvature)
    offset_points = points + safe_c * normals
    return offset_points

def Set_dipoles_pr_WL(surface, inneraux, outeraux, lam, points_per_wavelength_surface=10, points_per_wavelength_aux=5):
    '''
    Reduces surface to have points_per_wavelength_surface samples per wavelength,
    and inneraux/outeraux to have points_per_wavelength_aux samples per wavelength.
    Uses bilinear interpolation to ensure accurate point counts.
    Assumes x-y grids are square and regularly spaced.
    '''
    def interpolate_surface(surf, new_side):
        N_points = surf.points.shape[0]
        N_side = int(np.sqrt(N_points))
        if N_side ** 2 != N_points:
            raise ValueError("Surface points do not form a perfect square grid!")

        # Reshape surface components to 2D grid form
        grid_shape = (N_side, N_side, 3)
        pts = surf.points.reshape(grid_shape)
        tau1 = surf.tau1.reshape(grid_shape)
        tau2 = surf.tau2.reshape(grid_shape)
        normals = surf.normals.reshape(grid_shape)

        # Create original parameter grid (assumed regular)
        x = np.linspace(0, 1, N_side)
        y = np.linspace(0, 1, N_side)
        new_x = np.linspace(0, 1, new_side)
        new_y = np.linspace(0, 1, new_side)
        mesh_x, mesh_y = np.meshgrid(new_x, new_y, indexing='ij')
        interp_points = np.stack([mesh_x.ravel(), mesh_y.ravel()], axis=-1)

        # Helper to interpolate each vector field
        def interp_field(field):
            out = np.zeros((new_side * new_side, 3))
            for i in range(3):
                interp = RegularGridInterpolator((x, y), field[:, :, i])
                out[:, i] = interp(interp_points)
            return out

        return C2_surface(
            points=interp_field(pts),
            tau1=interp_field(tau1),
            tau2=interp_field(tau2),
            normals=interp_field(normals)
        )

    # Estimate scale
    x = surface.points[:, 0]
    surface_size = np.max(x) - np.min(x)
    scale = surface_size / lam
    #print(f"Wavelength scale: {scale:.2f}")

    # Compute target number of points per side
    side_surface = int(np.ceil(np.sqrt(points_per_wavelength_surface**2 * scale**2)))
    side_aux = int(np.ceil(np.sqrt(points_per_wavelength_aux**2 * scale**2)))

    #print(f"Target side (surface): {side_surface}, total: {side_surface**2}")
    #print(f"Target side (aux): {side_aux}, total: {side_aux**2}")

    # Interpolate to new grids
    reduced_surface = interpolate_surface(surface, side_surface)
    reduced_inneraux = interpolate_surface(inneraux, side_aux)
    reduced_outeraux = interpolate_surface(outeraux, side_aux)

    #print(f"Final matrix sizes: surface {reduced_surface.points.shape[0]}, aux {reduced_inneraux.points.shape[0]}")
    return reduced_surface, reduced_inneraux, reduced_outeraux

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

def testing():
    x=np.linspace(-1,1,50)
    y=np.linspace(-1,1,50)
    X,Y=np.meshgrid(x,y)
    Z=np.exp(-(X**2+Y**2))
    surface=build_closed_surface_from_graph(X,Y,Z)
    visualize_surface(surface,show_normals=False,stride=5)
testing()