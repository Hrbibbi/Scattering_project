import C2_surface as C2
import plane_wave as PW
import numpy as np
import matplotlib.pyplot as plt
import Matrix_construct
mu=1
omega=1
epsilon_int=2
epsilon_ext=1
k=omega*np.sqrt(epsilon_ext*mu)

def check_rhs():
    S=C2.sphere(1,np.array([0,0,1]),5)
    PW1=PW.Plane_wave(np.array([0,0,1]),0,k,mu,omega)
    rhs=np.abs(Matrix_construct.construct_RHS(S,PW1))

    file_path = "vector_b_simple.csv"

    # Read file and replace 'i' with 'j'
    with open(file_path, 'r') as f:
        content = f.read().replace('i', 'j')

    # Convert to numpy array
    from io import StringIO

    rhs_2 = np.abs(np.loadtxt(StringIO(content), delimiter=",", dtype=complex))
    plt.plot(np.sort(rhs), label="A code (rhs) sorted")
    plt.plot(np.sort(rhs_2), label="P and N code (rhs_2) sorted")
    plt.legend()
    plt.show()

    plt.plot((rhs), label="A code (rhs)")
    plt.plot((rhs_2), label="P and N code (rhs_2)")
    plt.legend()
    plt.show()

def check_A():
    S=C2.sphere(1,np.array([0,0,1]),5)
    inneraux=C2.sphere(0.8,np.array([0,0,1]),5)
    outeraux=C2.sphere(1.2,np.array([0,0,1]),5)
    A,DP1,DP2=Matrix_construct.construct_matrix(S,inneraux,outeraux,mu,epsilon_int,epsilon_ext,omega)
    A=np.abs(A)
    file_path = "matrix_A_simple.csv"

    # Read file and replace 'i' with 'j'
    with open(file_path, 'r') as f:
        content = f.read().replace('i', 'j')

    # Convert to numpy array
    from io import StringIO

    A_NP = np.abs(np.loadtxt(StringIO(content), delimiter=",", dtype=complex))
    plt.imshow(np.sort(A))
    plt.show()
    plt.imshow(np.sort(A_NP))
    plt.show()
check_A()