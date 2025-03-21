import numpy as np
import matplotlib.pyplot as plt
def plot_hertzian_dipole(direction):
    '''
    Creates a plot of a hertzian dipole located at (1.2,0,0) (outside the test area to avoid singularity)
    It takes as input a direction vector which is the orientation of the dipole (needs to be unit vector)
    '''
    import Hertzian_dipole as HD # my Dipole calcualtions
    
    DP1 = HD.Hertzian_Dipole(np.array([1.2, 0, 0]), direction, 1, 2, 3)

    '''
    I work by setting points up in a N x 3 array where each row is a point in R^3
    note here i take the points [-1,1] x [-1,1] x {1}
    '''
    x0, y0 = np.linspace(-1, 1, 100), np.linspace(-1, 1, 100)
    x, y = np.meshgrid(x0, y0)
    points = np.column_stack((x.ravel(), y.ravel(), np.ones_like(x.ravel())))
    

    '''
    This calculates the electric and magnetic fields at each point.
    it returns an array [E,H] where E and H are Nx3 with the values of the fields at each point
    '''
    E, H = DP1.evaluate_at_points(points)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    E_components = [E[:, 0], E[:, 1], E[:, 2]]
    H_components = [H[:, 0], H[:, 1], H[:, 2]]
    titles = ["E_x", "E_y", "E_z", "H_x", "H_y", "H_z"]
    
    for i in range(3):
        im = axes[0, i].imshow(np.reshape(E_components[i], (100, 100)).real, extent=(-1, 1, -1, 1), origin='lower')
        axes[0, i].set_title(titles[i])
        axes[0, i].set_xlabel("x")
        axes[0, i].set_ylabel("y")
        fig.colorbar(im, ax=axes[0, i])
    
        im = axes[1, i].imshow(np.reshape(H_components[i], (100, 100)).real, extent=(-1, 1, -1, 1), origin='lower')
        axes[1, i].set_title(titles[i + 3])
        axes[1, i].set_xlabel("x")
        axes[1, i].set_ylabel("y")
        fig.colorbar(im, ax=axes[1, i])
    
    plt.tight_layout()
    plt.show()

def plot_field_on_surface(points):
    '''
    Similar to the other function, however here the points are whatever surface i work on
    '''
    import Hertzian_dipole as HD
    
    DP1 = HD.Hertzian_Dipole(np.array([0, 0, 0]), np.array([1,0, 0]), 1, 2, 3)
    E, H = DP1.evaluate_at_points(points)
    
    fig = plt.figure(figsize=(18, 12))
    titles = ["E_x", "E_y", "E_z", "H_x", "H_y", "H_z"]
    E_components = [E[:, 0], E[:, 1], E[:, 2]]
    H_components = [H[:, 0], H[:, 1], H[:, 2]]
    
    for i in range(3):
        ax = fig.add_subplot(2, 3, i+1, projection='3d')
        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=E_components[i].real, cmap='viridis')
        ax.set_title(titles[i])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        fig.colorbar(sc, ax=ax)
    
        ax = fig.add_subplot(2, 3, i+4, projection='3d')
        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=H_components[i].real, cmap='viridis')
        ax.set_title(titles[i + 3])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        fig.colorbar(sc, ax=ax)
    
    plt.tight_layout()
    plt.show()

def solve_and_plot():
    import Matrix_construct
    import plane_wave as PW
    import C2_surface as C2

    '''the differen parameters i set the interiror epsilon to 2 and the outside to 1'''
    mu = 1
    omega = 1
    epsilon_int = 2
    epsilon_ext = 1
    k = omega * np.sqrt(epsilon_ext * mu)
    
    '''
    Here I initate an instace of my C2_surface object. This basically just stores the information about the object. it has
    S.points: a Nx3 array storing the points of my surface
    S.normal: a Nx3 array of the normal vector at each point
    S.tau1: a Nx3 array storing the first orthonomal basis element to the tangent plane
    S.tau2: a Nx3 array storing the second orthonomal basis element to the tangent plane
    
    S.construct_conformal_surface: a function that just takes all points an scale them be a factor r and returns a new surface.
      This works for the sphere as the tangent and normal doesnt change, but maybe for more complex surfaces it needs to be more shopisticated
    '''
    num_points=5 
    S = C2.sphere(1, np.array([0, 0, 0]), num_points) # construct a C2 of a sphere with radius 1 at 0,0,0 which has 5^2 test points
    #S=C2.in(5) 
    '''
    Construction of the matrix as input i give it the surface S and the inner and outer auxiliary surface plus the constants
    It then places dipoles at each point on the conformal surface, so we have 2*2*numpoints**2 dipoles that is 2 at each point
    and 2 times that since we have two surfaces. The dipoles have orientation tau_1 and tau_2.

    '''
    A = Matrix_construct.construct_matrix(S, S.construct_conformal_surface(0.8), S.construct_conformal_surface(1.2), mu, epsilon_int, epsilon_ext, omega)
    
    '''
    The right hand side is just a plane wave with polarization notation as Nikoline and propagation direction (x,y,z)
    '''
    PW1 = PW.Plane_wave(np.array([1, 0, 0]), 0, k)
    rhs = Matrix_construct.construct_RHS(S, PW1)
    
    C = np.linalg.solve(A, rhs)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].plot(np.abs(C),'.')
    axes[0].set_title("Absolute Values of C")
    
    im = axes[1].imshow(np.abs(A))
    axes[1].set_title("Absolute Values of A")
    fig.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.show()

'''
Test how it looks on the surface
'''
import C2_surface as C2
#S=C2.sphere(1,np.array([0,0,0]),30)
#S=C2.inverted_parabola(30)
#plot_field_on_surface(S.points)
'''
Test matrix setup
'''
#solve_and_plot()

'''
Test just a single dipole
'''
#plot_hertzian_dipole(np.array([0,0,1]))

S=C2.sphere(1,np.array([0,0,0]),10)
points=S.points
normals=S.normals
tau1=S.tau1
tau2=S.tau2
import pandas as pd

# Assuming S.points, S.normals, S.tau1, S.tau2 are Nx3 numpy arrays
data = np.hstack([points, normals, tau1, tau2])  # Concatenating arrays horizontally

# Creating column names
columns = ['px', 'py', 'pz', 'nx', 'ny', 'nz', 't1x', 't1y', 't1z', 't2x', 't2y', 't2z']

# Convert to DataFrame
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv("sphere_data.csv", index=False)
