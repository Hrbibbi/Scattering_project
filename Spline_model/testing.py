import numpy as np
from scipy.interpolate import bisplrep, bisplev

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

class SplineSurface:
    def __init__(self, x_fine, y_fine, z_fine):
        self.x_fine = x_fine
        self.y_fine = y_fine
        self.z_fine = z_fine
        self.tck = bisplrep(x_fine.ravel(), y_fine.ravel(), z_fine.ravel())
        self.max_mean_curvature = self._compute_max_mean_curvature()

    def _evaluate_spline(self, x, y, dx=0, dy=0):
        return np.array(bisplev(x, y, self.tck, dx=dx, dy=dy)).T

    def _compute_max_mean_curvature(self):
        x=1
    def construct_test_points(self, resolution):
        x = np.linspace(self.x_fine.min(), self.x_fine.max(), resolution) #resolution x 1
        y = np.linspace(self.y_fine.min(), self.y_fine.max(), resolution) #resolution x 1

        Z=self._evaluate_spline(x,y) # resolution x resolution both have same order
        X,Y=np.meshgrid(x,y) # resolution x resolution 

        return np.column_stack((X.ravel(), Y.ravel(), Z.ravel())) # resolution^2 x 3 array of points

    def construct_auxiliary_points(self, resolution, scale):
        x = np.linspace(self.x_fine.min(), self.x_fine.max(), resolution) #resolution x 1
        y = np.linspace(self.y_fine.min(), self.y_fine.max(), resolution) #resolution x 1

        Z = self._evaluate_spline(x, y) #resolution x resolution array stored in meshgrid order
        
        fx = self._evaluate_spline(x, y, dx=1, dy=0).ravel() #resolution^2 stored like meshgrid.ravel()
        fy = self._evaluate_spline(x, y, dx=0, dy=1).ravel() #resolution^2 stored like meshgrid.ravel()

        tau1=np.column_stack( (np.ones_like(fx),np.zeros_like(fx),fx) ) #resolution^2x3
        tau2=np.column_stack( (np.zeros_like(fx),np.ones_like(fy),fy) ) #resolution^2x3
        
        normals=np.column_stack( (np.zeros_like(fx),np.zeros_like(fx),np.ones_like(fx)) )  #resolution^2x3
        tau1=tau1 / np.linalg.norm(tau1,axis=1,keepdims=True)
        tau2=tau2 / np.linalg.norm(tau2,axis=1,keepdims=True)
        normals=normals / np.linalg.norm(normals,axis=1,keepdims=True)
        
        points = np.column_stack( (X.ravel(),Y.ravel(),Z.ravel()) )
        if self.max_mean_curvature==0:
            offset = scale * normals
        else:
            offset = (scale / self.max_mean_curvature) * normals
        auxiliary_points = points + offset
        return auxiliary_points, tau1 , tau2 , normals  


# Example usage:
x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)
Z = 1-X**2

spline_surf = SplineSurface(X, Y, Z)
points=spline_surf.construct_test_points(10)