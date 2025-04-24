import numpy as np
from sklearn.gaussian_process.kernels import Matern
from numpy.random import default_rng
import matplotlib.pyplot as plt
import time
from scipy.linalg import cholesky
import C2_surface as C2

def generate_matern_surface(
    x: np.ndarray,
    y: np.ndarray,
    length_scale: float = 1.0,
    nu: float = 1.5,
    sigma_f: float = 1.0,
    n_samples: int = 1,
    random_state: int | None = None,
    use_cholesky: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate z = f(x, y) where f ~ GP(0, k) with Matérn kernel.

    Parameters
    ----------
    ...
    use_cholesky
        If True, use Cholesky factorization for sampling instead of NumPy's multivariate sampler.

    Returns
    -------
    X, Y, Z : meshgrid and sampled values as described earlier
    """
    construct_start=time.time()
    # Meshgrid
    X, Y = np.meshgrid(x, y)
    coords = np.vstack([X.ravel(), Y.ravel()]).T
    N = coords.shape[0]

    # Kernel and covariance matrix
    kernel = sigma_f**2 * Matern(length_scale=length_scale, nu=nu)
    K = kernel(coords)
    K += 1e-8 * np.eye(N)  # Jitter
    print(f"cov matrix shape {np.shape(K)} ")
    print(f"Construction time: {time.time()-construct_start}")

    sample_start=time.time()
    rng = default_rng(random_state)

    if use_cholesky:
        sample_method="cholesky"
        # Cholesky factorization: K = L @ L.T
        L = cholesky(K, lower=True)
        # Draw N(0, I) and multiply: f ~ L @ p
        P = rng.normal(size=(N, n_samples))  # shape (N, n_samples)
        Z_samples = L @ P                    # shape (N, n_samples)
        Z_samples = Z_samples.T              # shape (n_samples, N)
    else:
        sample_method="numpy"
        Z_samples = rng.multivariate_normal(
            mean=np.zeros(N), cov=K, size=n_samples
        )
    print(f"\nnumber of samples: {n_samples}")
    print(f"sample method: {sample_method}")
    print(f"Sample time: {time.time()-sample_start}")
    # Reshape
    if n_samples == 1:
        Z = Z_samples.reshape(X.shape)
    else:
        Z = Z_samples.reshape(n_samples, *X.shape)

    return X, Y, Z

def plot_matern_gp_realizations(
    a: float,
    b: float,
    N: int,
    length_scale: float = 1.0,
    nu: float = 1.5,
    sigma_f: float = 1.0,
    seed: int = 42,
    use_cholesky: bool = False
):
    """
    Generate and plot 5 independent GP realizations with a Matérn kernel.

    Parameters
    ----------
    a, b : float
        Grid bounds in x and y (square domain [a, b] x [a, b])
    N : int
        Number of points along each axis (grid resolution)
    length_scale, nu, sigma_f : float
        Parameters of the Matérn kernel
    seed : int
        Random seed for reproducibility
    """
    from numpy.random import default_rng
    rng = default_rng(seed)

    # Create the grid
    x = np.linspace(a, b, N)
    y = np.linspace(a, b, N)

    # Generate 5 samples
    _, _, Z_samples = generate_matern_surface(
        x, y,
        length_scale=length_scale,
        nu=nu,
        sigma_f=sigma_f,
        n_samples=5,
        random_state=rng,
        use_cholesky=use_cholesky
    )

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), constrained_layout=True)
    for i, ax in enumerate(axes):
        contour = ax.contourf(x, y, Z_samples[i], cmap='viridis')
        ax.set_title(f'Sample {i+1}')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(contour, ax=ax, orientation='vertical', shrink=0.7)

    fig.suptitle('Matérn GP Surface Samples', fontsize=16)
    plt.show()

def generate_C2_surfaces_with_offsets(
    a: float,
    b: float,
    N: int,
    length_scale: float = 1.0,
    nu: float = 1.5,
    sigma_f: float = 1.0,
    n_instances: int = 1,
    seed: int = 42
) -> list[tuple[C2.C2_surface, C2.C2_surface, C2.C2_surface]]:
    """
    Generate multiple C2_surface objects from Gaussian process samples with Matérn kernel,
    including inner and outer offset surfaces scaled by curvature.

    Returns
    -------
    List of tuples: (surface, inner_surface, outer_surface)
    """
    # 1D grid
    x = np.linspace(a, b, N)
    y = np.linspace(a, b, N)
    h = (b - a) / (N - 1)

    # Generate GP samples
    X, Y, Z_all = generate_matern_surface(
        x, y,
        length_scale=length_scale,
        nu=nu,
        sigma_f=sigma_f,
        n_samples=n_instances,
        random_state=seed,
        use_cholesky=True
    )

    results = []

    for i in range(n_instances):
        Z = Z_all if n_instances == 1 else Z_all[i]

        # Geometric data
        point_cloud, tau1, tau2, normals, mean_curvature = C2.compute_geometric_data(X, Y, Z, h)
        # Main surface
        surface = C2.C2_surface(points=point_cloud, normals=normals, tau1=tau1, tau2=tau2)

        # Offset points (inner and outer)
        inner_points = C2.generate_curvature_scaled_offset(point_cloud, normals, mean_curvature, scaling=-0.86)
        outer_points = C2.generate_curvature_scaled_offset(point_cloud, normals, mean_curvature, scaling=0.86)

        inner_surface = C2.C2_surface(points=inner_points, normals=normals, tau1=tau1, tau2=tau2)
        outer_surface = C2.C2_surface(points=outer_points, normals=normals, tau1=tau1, tau2=tau2)

        results.append((surface, inner_surface, outer_surface))

    return results

#test=generate_C2_surfaces_with_offsets(a=-1,b=1,N=70,n_instances=1000)
#print(len(test))
plot_matern_gp_realizations(-1,1,50,use_cholesky=True)
