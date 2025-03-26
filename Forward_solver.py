#initial stuff
import Hertzian_dipole as HD
import Matrix_construct
import plane_wave as PW
import C2_surface as C2
import numpy as np
import matplotlib.pyplot as plt

def forward_solver(Surface,planewave,inner_radius,outer_radius,mu,epsilon_int,epsilon_ext,omega):
    S=Surface
    A,Dp1,Dp2=Matrix_construct.construct_matrix(
    S,S.construct_conformal_surface(inner_radius),S.construct_conformal_surface(outer_radius),mu,epsilon_int,epsilon_ext,omega
    )
    plt.imshow(np.abs(A))
    plt.show()
    rhs=Matrix_construct.construct_RHS(S,planewave)
    C=np.linalg.solve(A,rhs)
    M=np.shape(Surface.points)[0]
    C1=C[0:M]
    C2=C[M+1:2*M+1]
    out_function = lambda points : np.add(HD.evaluate_linear_combination(points, Dp1, C1), HD.evaluate_linear_combination(points, Dp2, C2))
    #np.sum([HD.evaluate_linear_combination(points,Dp1,C1),HD.evaluate_linear_combination(points,Dp2,C2)],axis=0)
    return out_function

'''test setup'''
mu = 1
omega = 1
epsilon_int = 2
epsilon_ext = 1
k = omega * np.sqrt(epsilon_ext * mu)
PW1 = PW.Plane_wave(np.array([1,0,0]), np.pi, k)
num_points=20 
S = C2.sphere(1, np.array([0, 0, 0]), num_points)
#points=S.construct_conformal_surface(2).points
points=C2.sphere(2,np.array([0,0,0]),num_points).points
forward_function=forward_solver(S,PW1,0.8,1.2,mu,epsilon_int,epsilon_ext,omega)
#print(np.shape(forward_function(points)))
E,H=forward_function(points)
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