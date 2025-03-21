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

    def construct_conformal_surface(self,radius):
        points=radius*self.points
        return C2_surface(points,self.normals,self.tau1,self.tau2)
    
    def plot_surface(self):
        x, y, z = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='c', marker='o', alpha=0.6)  # 3D scatter plot

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Scatter Plot of Surface Points')
        return ax
    
    def plot_tangents(self, scale=0.1):
        """Plots the tangent vectors tau1 and tau2 at each surface point."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of the points
        ax.scatter(self.points[:, 0], self.points[:, 1], self.points[:, 2], c='c', marker='o', alpha=0.6, label="Surface Points")

        # Plot tangent vectors
        for i in range(self.M):
            p = self.points[i]
            t1 = self.tau1[i] * scale
            t2 = self.tau2[i] * scale

            # Quiver (arrow) for tau1
            ax.quiver(p[0], p[1], p[2], t1[0], t1[1], t1[2], color='r', length=np.linalg.norm(t1), normalize=True, label="Tau1" if i == 0 else "")
            # Quiver (arrow) for tau2
            ax.quiver(p[0], p[1], p[2], t2[0], t2[1], t2[2], color='b', length=np.linalg.norm(t2), normalize=True, label="Tau2" if i == 0 else "")

        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Tangent Vectors on Surface')

        # Legend
        ax.legend()
        plt.show()

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

def cylinder(radius,height,num_points):
    r=radius
    theta0=np.linspace(0,2*np.pi,num_points+1)[:-1]
    z0=np.linspace(0,height,num_points)
    theta,z=np.meshgrid(theta0,z0)
    x=r*np.cos(theta)
    y=r*np.sin(theta)
    points=np.column_stack((x.ravel(),y.ravel(),z.ravel()))
    normals=np.column_stack( ( x.ravel()/r ,y.ravel()/r , np.zeros_like( x.ravel() ) ) )
    tau1=np.zeros_like(points)
    tau1[:,0],tau1[:,1]=-np.sin(theta.ravel()),np.cos(theta.ravel())
    tau2=np.zeros_like(points)
    tau2[:,2]=1

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