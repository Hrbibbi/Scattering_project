import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import bisplrep, bisplev

class C2_surface:
    def __init__(self,points,normals,tau1,tau2):
        self.points=points
        self.normals=normals
        self.tau1=tau1
        self.tau2=tau2
        self.M=np.shape(self.points)[0]

class SplineSurface:
    def __init__(self, x_fine, y_fine, z_fine,smoothness=0.5):
        # Ensure x_fine and y_fine are 2D and meshgridded
        self.x_fine = x_fine
        self.y_fine = y_fine
        self.z_fine = z_fine
        self.a=np.min(x_fine)
        self.b=np.max(x_fine)
        self.tck = bisplrep(x_fine.ravel(), y_fine.ravel(), z_fine.ravel(),s=smoothness)
        self.size=self.x_fine.max()-self.x_fine.min()
        self.max_mean_curvature = self._compute_max_mean_curvature()
        print(self.max_mean_curvature)
    def _evaluate_spline(self, x, y, dx=0, dy=0):
        # Accept 1D input arrays and use meshgrid internally
        return np.array(bisplev(x, y, self.tck, dx=dx, dy=dy))

    def _compute_max_mean_curvature(self, resolution=None): #change to do on finest resultion
        # Uniform grid in parameter space
        if resolution==None:
            resolution=len(self.x_fine)
        x = np.linspace(self.x_fine.min(), self.x_fine.max(), resolution)
        y = np.linspace(self.y_fine.min(), self.y_fine.max(), resolution)

        fx = self._evaluate_spline(x, y, dx=1, dy=0)
        fy = self._evaluate_spline(x, y, dx=0, dy=1)
        fxx = self._evaluate_spline(x, y, dx=2, dy=0)
        fxy = self._evaluate_spline(x, y, dx=1, dy=1)
        fyy = self._evaluate_spline(x, y, dx=0, dy=2)

        fx = np.array(fx)
        fy = np.array(fy)
        fxx = np.array(fxx)
        fxy = np.array(fxy)
        fyy = np.array(fyy)

        denom = (1 + fx**2 + fy**2) ** 1.5
        numer = ((1 + fy**2) * fxx - 2 * fx * fy * fxy + (1 + fx**2) * fyy)
        H = numer / (2 * denom)
        return np.max(np.abs(H))

    def construct_auxiliary_points(self, resolution, scale,fixed_offset=0):
        x = np.linspace(self.x_fine.min(), self.x_fine.max(), resolution)
        y = np.linspace(self.y_fine.min(), self.y_fine.max(), resolution)
        X, Y = np.meshgrid(x, y, indexing='ij')
        Z = self._evaluate_spline(x, y)

        fx = self._evaluate_spline(x, y, dx=1, dy=0)
        fy = self._evaluate_spline(x, y, dx=0, dy=1)

        tau_x = np.stack([np.ones_like(fx), np.zeros_like(fx), fx], axis=-1)
        tau_y = np.stack([np.zeros_like(fy), np.ones_like(fy), fy], axis=-1)

        normals = np.cross(tau_x, tau_y)
        tau_y = np.cross(normals, tau_x)  # Ensure right-handed orthonormal basis

        # Normalize
        tau_x /= np.linalg.norm(tau_x, axis=-1, keepdims=True)
        tau_y /= np.linalg.norm(tau_y, axis=-1, keepdims=True)
        normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

        points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
        
        #----------------------------------------------------------------------
        # offset changes set to fixed number if specified
        #----------------------------------------------------------------------
        if fixed_offset==0:
            print("not fixed")
            if self.max_mean_curvature==0:
                offset = scale * normals.reshape(-1, 3)
            else:
                offset = (scale / self.max_mean_curvature) * normals.reshape(-1, 3) #scale/max_man = 0.14 og 0.014
        else:
            print("fixed")
            offset = fixed_offset * normals.reshape(-1, 3)
            
        
        auxiliary_points = points + offset

        return auxiliary_points, tau_x.reshape(-1, 3), tau_y.reshape(-1, 3), normals.reshape(-1, 3)

    def sample_surface_MAS(self,wavelength,fixed_offset=0):
        scale=self.size/wavelength
        surface_resol=int(np.ceil(np.sqrt(2)*5*scale)) #maybe increase
        auxiliary_resol=int(np.ceil(5*scale)) #maybe increase
        inner_points  , tau1_inner, tau2_inner, normals_inner=self.construct_auxiliary_points(auxiliary_resol, -0.14,fixed_offset=-fixed_offset)
        outer_points  , tau1_outer, tau2_outer, normals_outer=self.construct_auxiliary_points(auxiliary_resol,  0.14,fixed_offset=fixed_offset)
        surface_points, tau1_surf , tau2_surf , normals_surf =self.construct_auxiliary_points(surface_resol  ,  0.00)

        inneraux=C2_surface(inner_points,normals_inner,tau1_inner,tau2_inner)
        outeraux=C2_surface(outer_points,normals_outer,tau1_outer,tau2_outer)
        surface =C2_surface(surface_points,normals_surf,tau1_surf,tau2_surf)
        return surface, inneraux, outeraux

    def plot_surface_with_vectors(self, resolution=20, quiver_scale=0.2):
        # High-res mesh for smooth surface
        dense_res = 4 * resolution
        x_dense = np.linspace(self.x_fine.min(), self.x_fine.max(), dense_res)
        y_dense = np.linspace(self.y_fine.min(), self.y_fine.max(), dense_res)
        X_dense, Y_dense = np.meshgrid(x_dense, y_dense, indexing='ij')
        Z_dense = self._evaluate_spline(x_dense, y_dense)

        # Tangents and normals from auxiliary surface at scale=0
        points, tau_x, tau_y, normals = self.construct_auxiliary_points(resolution, scale=0.0)

        # Plot
        fig = plt.figure(figsize=(18, 5))

        # Surface only
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot_surface(X_dense, Y_dense, Z_dense, cmap='viridis', alpha=0.9)
        ax1.set_title("Spline Surface")

        # Surface + tangents + normals
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot_surface(X_dense, Y_dense, Z_dense, cmap='viridis', alpha=0.5)
        ax2.quiver(points[:, 0], points[:, 1], points[:, 2],
                   tau_x[:, 0], tau_x[:, 1], tau_x[:, 2],
                   color='r', length=quiver_scale, normalize=True)
        ax2.quiver(points[:, 0], points[:, 1], points[:, 2],
                   tau_y[:, 0], tau_y[:, 1], tau_y[:, 2],
                   color='g', length=quiver_scale, normalize=True)
        ax2.quiver(points[:, 0], points[:, 1], points[:, 2],
                   normals[:, 0], normals[:, 1], normals[:, 2],
                   color='b', length=quiver_scale, normalize=True)
        ax2.set_title("Tangents and Normals")

        # Surface + auxiliary ±0.5
        aux_above, _, _, _ = self.construct_auxiliary_points(resolution, scale=0.5)
        aux_below, _, _, _ = self.construct_auxiliary_points(resolution, scale=-0.5)

        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot_surface(X_dense, Y_dense, Z_dense, cmap='gray', alpha=0.3)
        ax3.scatter(aux_above[:, 0], aux_above[:, 1], aux_above[:, 2],
                    c='magenta', s=10, label='Aux +')
        ax3.scatter(aux_below[:, 0], aux_below[:, 1], aux_below[:, 2],
                    c='cyan', s=10, label='Aux -')
        ax3.set_title("Auxiliary Surfaces ±0.5")

        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")

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

def surface_from_json(json_path, output_prefix='surface_data',fixed_offset=0):
    import json
    import numpy as np
    import pandas as pd

    with open(json_path, 'r') as f:
        params = json.load(f)

    width = params['halfWidth_x']
    resol = params['resolution']
    alpha = params['alpha']
    bump_params = params['bumpData']
    scatter_epsilon = params['epsilon1']
    mu = 1  # Assumed constant
    lam = params['minLambda'] #lam=0.7

    # Surface creation
    a, b = -width, width
    X0 = np.linspace(a, b, resol)
    Y0 = np.linspace(a, b, resol)
    X, Y = np.meshgrid(X0, Y0)

    def bump(x, y, x0, y0, height, sigma):
        return height * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def surface_function(x, y):
        return sum(
            bump(x, y, b['x0'], b['y0'], b['height'], b['sigma'])
            for b in bump_params
        )

    Z = surface_function(X, Y)
    #Z = np.zeros_like(X)
    SPSurface = SplineSurface(X, Y, Z)

    surface, inneraux, outeraux = SPSurface.sample_surface_MAS(lam,fixed_offset=fixed_offset)

    def save_surface_to_csv(surf_obj, filename):
        data = np.hstack([surf_obj.points, surf_obj.tau1, surf_obj.tau2, surf_obj.normals])
        columns = ['x', 'y', 'z',
                   'tau1_x', 'tau1_y', 'tau1_z',
                   'tau2_x', 'tau2_y', 'tau2_z',
                   'normal_x', 'normal_y', 'normal_z']
        df = pd.DataFrame(data, columns=columns)
        df.to_csv(filename, index=False)

    save_surface_to_csv(surface, f'{output_prefix}.csv')
    save_surface_to_csv(inneraux, f'{output_prefix}_inneraux.csv')
    save_surface_to_csv(outeraux, f'{output_prefix}_outeraux.csv')

#surface_from_json('surfaceParamsTen.json',output_prefix='surfaceTen_0014',fixed_offset=0.014)