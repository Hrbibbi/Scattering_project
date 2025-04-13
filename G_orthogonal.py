import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class C2_surface:
    def __init__(self,points,normals,tau1,tau2):
        self.points=points
        self.normals=normals
        self.tau1=tau1
        self.tau2=tau2
        self.M=np.shape(self.points)[0]

    def plot_tangents(self, scale=0.1):
        """Plots the surface as a solid and overlays tangent vectors tau1 and tau2."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        N = int(np.sqrt(self.points.shape[0]))
        X = self.points[:, 0].reshape((N, N))
        Y = self.points[:, 1].reshape((N, N))
        Z = self.points[:, 2].reshape((N, N))

        # Plot the surface
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')

        # Plot tangent vectors
        for i in range(self.points.shape[0]):
            p = self.points[i]
            t1 = self.tau1[i] * scale
            t2 = self.tau2[i] * scale

            ax.quiver(p[0], p[1], p[2], t1[0], t1[1], t1[2],
                    color='r', length=np.linalg.norm(t1), normalize=True, linewidth=1)
            ax.quiver(p[0], p[1], p[2], t2[0], t2[1], t2[2],
                    color='b', length=np.linalg.norm(t2), normalize=True, linewidth=1)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Surface with Tangent Vectors')
        plt.tight_layout()
        plt.show()

def compute_tangent_vectors(pointcloud, N, a, b):
    f_vals = pointcloud[:, 2].reshape((N, N))
    h = (b - a) / (N - 1)

    f_x = np.zeros_like(f_vals)
    f_y = np.zeros_like(f_vals)

    f_x[1:-1, :] = (f_vals[2:, :] - f_vals[:-2, :]) / (2 * h)
    f_y[:, 1:-1] = (f_vals[:, 2:] - f_vals[:, :-2]) / (2 * h)

    f_x[0, :] = (f_vals[1, :] - f_vals[0, :]) / h
    f_x[-1, :] = (f_vals[-1, :] - f_vals[-2, :]) / h

    f_y[:, 0] = (f_vals[:, 1] - f_vals[:, 0]) / h
    f_y[:, -1] = (f_vals[:, -1] - f_vals[:, -2]) / h

    t_x = np.stack([np.ones(N*N), np.zeros(N*N), f_x.ravel()], axis=1)
    t_y = np.stack([np.zeros(N*N), np.ones(N*N), f_y.ravel()], axis=1)
    t_x = t_x / np.linalg.norm(t_x, axis=1, keepdims=True)
    t_y = t_y / np.linalg.norm(t_y, axis=1, keepdims=True)
    return t_x, t_y

def compute_normals(tangent1, tangent2):
    normals = np.cross(tangent1, tangent2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals_normalized = normals / norms
    return normals_normalized
    

f = lambda x,y: x**2+y**2
a,b=-1,1
N=20
x0,y0=np.linspace(a,b,N),np.linspace(a,b,N)
x,y=np.meshgrid(x0,y0)
z=f(x,y).ravel()
pointcloud=np.column_stack( (x.ravel() , y.ravel() , z) )

t_x,t_y=compute_tangent_vectors(pointcloud,N,a,b)
normals = compute_normals(t_x,t_y)
C=C2_surface(pointcloud,normals,t_x,t_y)
#C.plot_surface()
C.plot_tangents()
plt.show()