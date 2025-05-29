import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import bisplrep, bisplev


#----------------------------------------------------
# Classes used for further computation
#----------------------------------------------------
class C2_surface:
    """
    Simple wrapper class that stores necessary information about a surface to be used in other calculations
    """
    def __init__(self,points,normals,tau1,tau2):
        self.points=points
        self.normals=normals
        self.tau1=tau1
        self.tau2=tau2
        self.M=np.shape(self.points)[0]

class SplineSurface:
    '''
    Spline surface class creates a spline representation of the surface which can be sampled from
    '''
    def __init__(self, x_fine, y_fine, z_fine,smoothness=0.5,check_curvature=False):
        #2D arrays of points
        self.x_fine = x_fine
        self.y_fine = y_fine
        self.z_fine = z_fine

        #size information
        self.a=self.x_fine.min()
        self.b=self.x_fine.max()
        self.size=self.b-self.a
        self.fine_resol=len(x_fine)

        #Spline representation
        self.tck = bisplrep(x_fine.ravel(), y_fine.ravel(), z_fine.ravel(),s=smoothness)
        
        #Curvature calculation
        self.max_mean_curvature = self._compute_max_mean_curvature(check_via_plot=check_curvature)

    def _evaluate_spline(self, x, y, dx=0, dy=0):
        '''
        Wrapper to compute the spline model
        '''
        return np.array(bisplev(x, y, self.tck, dx=dx, dy=dy))

    def sample_surface_MAS(self, wavelength: float, fixed_offset: float=0, dipoles_per_wl: float=5, scaling: float=0.14
                          )-> tuple[C2_surface,C2_surface,C2_surface]:
        '''
        Construct necessary surface data for the main and inner/outer auxiliary surfaces based on the given wavelength.

        Parameters:
        ----------
        wavelength : float
            The wavelength of the incident wave used to determine the surface resolution.
        fixed_offset: float, optional
            a fixed offset of the auxiliary points if not used it is based on a curvature penalty function
        dipoles_per_wl: float, optional
            number of dipole pairs per wavlength, default is 5 pairs
        
        Returns:
        ----------
        surface: C2_surface:
            C2 class of surface
        inneraux: C2_surface:
            C2 class of inner auxiliary surface
        outeraux: C2_surface:
            C2 class of outer auxiliary surface 
        '''

        #----------------------------------------------------------------
        # resolution calculations to specify points on the surfaces
        #----------------------------------------------------------------

        scale=self.size/wavelength
        M=dipoles_per_wl
        print(f"dipoles in spline function {M}")
        surface_resol=int(np.ceil(np.sqrt(2)*M*scale))
        auxiliary_resol=int(np.ceil(M*scale))

        #----------------------------------------------------------------
        # Surface construction using internal constructer function
        #----------------------------------------------------------------
        inner_points  , tau1_inner, tau2_inner, normals_inner=self.construct_auxiliary_points(auxiliary_resol, -scaling,fixed_offset=-fixed_offset)
        outer_points  , tau1_outer, tau2_outer, normals_outer=self.construct_auxiliary_points(auxiliary_resol,  scaling,fixed_offset=fixed_offset)
        surface_points, tau1_surf , tau2_surf , normals_surf =self.construct_auxiliary_points(surface_resol  ,  0.00)

        #----------------------------------------------------------------
        # Saving data as C2 surfaces
        #----------------------------------------------------------------
        inneraux=C2_surface(inner_points,normals_inner,tau1_inner,tau2_inner)
        outeraux=C2_surface(outer_points,normals_outer,tau1_outer,tau2_outer)
        surface =C2_surface(surface_points,normals_surf,tau1_surf,tau2_surf)
        return surface, inneraux, outeraux

    def plot_surface_with_vectors(self, resolution: int=20, quiver_scale: float=0.2):
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
    
    #----------------------------------------------------------------------------------
    # Internal helper functions
    #----------------------------------------------------------------------------------
    def _compute_max_mean_curvature(self,check_via_plot=False):
        #fine resultion
        resolution=self.fine_resol
        x = np.linspace(self.x_fine.min(), self.x_fine.max(), resolution)
        y = np.linspace(self.y_fine.min(), self.y_fine.max(), resolution)

        fx  = np.array(self._evaluate_spline(x, y, dx=1, dy=0))
        fy  = np.array(self._evaluate_spline(x, y, dx=0, dy=1))
        fxx = np.array(self._evaluate_spline(x, y, dx=2, dy=0))
        fxy = np.array(self._evaluate_spline(x, y, dx=1, dy=1))
        fyy = np.array(self._evaluate_spline(x, y, dx=0, dy=2))

        denom = ( 1+fx**2+fy**2 )**(1.5)
        num   = ( 1+fx**2)*fyy
        num  += ( 1+fy**2)*fxx
        num  -= ( 2*fx*fy*fxy)
        
        H=np.abs(num/(2*denom))
        H_max=np.max(H)
        if check_via_plot:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Contour of surface height
                x0=np.linspace(self.a,self.b,self.fine_resol)
                y0=np.linspace(self.a,self.b,self.fine_resol)
                z=self._evaluate_spline(x0,y0)
                cs1 = axes[0].contourf(self.x_fine, self.y_fine, z, cmap='viridis')
                axes[0].set_title("Surface Height (z)")
                axes[0].set_xlabel("x")
                axes[0].set_ylabel("y")
                fig.colorbar(cs1, ax=axes[0], orientation='vertical')

                # Contour of mean curvature
                cs2 = axes[1].contourf(self.x_fine, self.y_fine, H, cmap='coolwarm')
                axes[1].set_title("Mean Curvature (H)")
                axes[1].set_xlabel("x")
                axes[1].set_ylabel("y")
                fig.colorbar(cs2, ax=axes[1], orientation='vertical')

                plt.tight_layout()
                plt.show()

        return H_max
    
    def construct_auxiliary_points(self, resolution: float, scale: float,fixed_offset: float=0)->tuple[
                                    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Constructs auxiliary points displaced from the surface by the surface normal.

        Parameters:
        -----------
        resolution : float
            Number of points along each axis to sample the surface.
        scale : float
            Scale factor for displacement based on curvature.
        fixed_offset : float, optional
            If nonzero, overrides curvature-based displacement with a fixed offset.

        Returns:
        --------
        auxiliary_points : (N, 3) np.ndarray
            The displaced auxiliary surface points.
        tau_x : (N, 3) np.ndarray
            The first tangent vector at each point.
        tau_y : (N, 3) np.ndarray
            The second tangent vector at each point.
        normals : (N, 3) np.ndarray
            The normal vector at each point.
        """
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
            #Case that zero curvature
            penalty_term = max(1,self.max_mean_curvature)
            print(f"normal scaling: {scale/penalty_term}")
            offset = (scale / penalty_term) * normals.reshape(-1, 3) #scale/max_man = 0.14 og 0.014
        else:
            print(f"fixed scaling {fixed_offset}")
            offset = fixed_offset * normals.reshape(-1, 3)
            
        
        auxiliary_points = points + offset

        return auxiliary_points, tau_x.reshape(-1, 3), tau_y.reshape(-1, 3), normals.reshape(-1, 3)

#----------------------------------------------------
# C2_surface related functions
#----------------------------------------------------
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

#----------------------------------------------------
# SplineSurface functions
#----------------------------------------------------

#----------------------------------------------------
# data handling functions
#----------------------------------------------------
def surface_from_json(json_path, output_prefix='surface_data',fixed_offset=0,output_dir='surface_csvs'):
    import json
    import numpy as np
    import pandas as pd
    import os

    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, 'r') as f:
        params = json.load(f)

    width = params['halfWidth_x']
    resol=200
    bump_params = params['bumpData']
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

    Z = np.zeros_like(X)
    Z += surface_function(X, Y)
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

    save_surface_to_csv(surface, os.path.join(output_dir, f'{output_prefix}.csv'))
    save_surface_to_csv(inneraux, os.path.join(output_dir, f'{output_prefix}_inneraux.csv'))
    save_surface_to_csv(outeraux, os.path.join(output_dir, f'{output_prefix}_outeraux.csv'))

#----------------------------------------------------
# testing 
#----------------------------------------------------

def test_mean_curvature_calculation(json_path):
    import json
    import numpy as np
    import pandas as pd

    with open(json_path, 'r') as f:
        params = json.load(f)

    width = params['halfWidth_x']
    resol = 100
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
    Z  = np.zeros_like(X)
    Z += surface_function(X, Y)
    SPSurface=SplineSurface(X,Y,Z,smoothness=0.5,check_curvature=True)
    #SPSurface.construct_auxiliary_points(100,0.14)
    SPSurface.plot_surface_with_vectors(resolution=10,quiver_scale=0.0)
#test_mean_curvature_calculation("Json_files/surfaceParamsOne.json")
'''
surface_from_json("Json_files/surfaceParamsZero.json",output_prefix="surfaceZero_0014",fixed_offset=0.014)
surface_from_json("Json_files/surfaceParamsZero.json",output_prefix="surfaceZero_014",fixed_offset=0.14)
surface_from_json("Json_files/surfaceParamsOne.json",output_prefix="surfaceOne_0014",fixed_offset=0.014)
surface_from_json("Json_files/surfaceParamsOne.json",output_prefix="surfaceOne_014",fixed_offset=0.14)
surface_from_json("Json_files/surfaceParamsTen.json",output_prefix="surfaceTen_0014",fixed_offset=0.014)
surface_from_json("Json_files/surfaceParamsTen.json",output_prefix="surfaceTen_014",fixed_offset=0.14)
'''