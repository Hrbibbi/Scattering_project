import numpy as np
import warnings
import multiprocessing
from numba import jit

class Hertzian_Dipole():
    def __init__(self, positions, directions, mu, epsilon, omega):
        """
        Initialize a set of Hertzian dipoles.

        Parameters:
            positions : numpy.ndarray
                An M×3 array containing the positions of each dipole.
            directions : numpy.ndarray
                An M×3 array containing the unit direction vectors for each dipole.
            mu : scalar (int, float, or numpy number)
                Magnetic permeability (common to all dipoles).
            epsilon : scalar (int, float, or numpy number)
                Electric permittivity (common to all dipoles).
            omega : scalar (int, float, or numpy number)
                Angular frequency (common to all dipoles).

        Raises:
            TypeError or ValueError if inputs are not in expected formats.
        """
        # Check that mu, epsilon, and omega are numerical scalars
        for param, name in zip([mu, epsilon, omega], ["mu", "epsilon", "omega"]):
            if not isinstance(param, (int, float, np.number)):
                raise TypeError(f"{name} must be a numerical value (int, float, or numpy number), got {type(param)} instead.")
        
        # Validate positions and directions as 2D numpy arrays with shape (M,3)
        for arr, name in zip([positions, directions], ["positions", "directions"]):
            if not isinstance(arr, np.ndarray):
                raise TypeError(f"{name} must be a numpy array, got {type(arr)} instead.")
            if arr.ndim != 2 or arr.shape[1] != 3:
                raise ValueError(f"{name} must be of shape (M, 3), but got shape {arr.shape}.")
        
        # Verify that each direction vector is a unit vector
        norms = np.linalg.norm(directions, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-6):
            raise ValueError("All direction vectors must be unit vectors (norm = 1).")
        
        # Store scalar parameters and compute common quantities
        self.mu = mu
        self.epsilon = epsilon
        self.omega = omega
        self.wavenumber = omega * np.sqrt(epsilon * mu)
        self.positions = positions  # Shape: (M, 3)
        self.directions = directions  # Shape: (M, 3)

    def evaluate_at_points(self, X):
        """
        Evaluate the scattered electric and magnetic fields of all dipoles 
        at the given evaluation points.

        Parameters:
            X : numpy.ndarray
                An N×3 array representing the coordinates of evaluation points.

        Returns:
            fields : numpy.ndarray
                A complex array of shape 2×M×N×3 containing the field evaluations.
                fields[0, i, j, :] corresponds to the electric field from dipole i at point j.
                fields[1, i, j, :] corresponds to the magnetic field from dipole i at point j.
        """
        # Check that X is an N×3 numpy array
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Evaluation points X must be a numpy array, got {type(X)} instead.")
        if X.ndim != 2 or X.shape[1] != 3:
            raise ValueError(f"X must have shape (N, 3), but got shape {X.shape}.")
        
        mu = self.mu
        epsilon = self.epsilon
        k = self.wavenumber
        omega = self.omega
        xi = 1j * omega * mu / (4 * np.pi)  # constant factor

        # Number of dipoles (M) and evaluation points (N)
        M = self.positions.shape[0]
        N = X.shape[0]
        
        # Reshape dipoles for broadcasting:
        # positions: (M, 1, 3) and directions: (M, 1, 3)
        positions = self.positions[:, None, :]  # shape: (M, 1, 3)
        directions = self.directions[:, None, :]  # shape: (M, 1, 3)

        # Calculate displacement from each dipole to each point: (M, N, 3)
        X_trans = X[None, :, :] - positions

        # Compute distances r for each dipole-point pair: shape (M, N)
        r = np.linalg.norm(X_trans, axis=2)
        
        # Avoid division by zero (if a point is exactly at a dipole location)
        # Here one may want to set a small minimum r value.
        r = np.where(r == 0, 1e-16, r)

        # Extract x, y, z components; shape (M, N)
        x = X_trans[:, :, 0]
        y = X_trans[:, :, 1]
        z = X_trans[:, :, 2]

        # Components of dipole directions; shape (M, 1) which broadcasts with (M, N)
        dx = directions[:, :, 0]
        dy = directions[:, :, 1]
        dz = directions[:, :, 2]

        # Dot product of the dipole direction with the displacement vectors; shape (M, N)
        dotted = dx * x + dy * y + dz * z

        # Calculate R and Phi factors
        R = 1/(r**3) + 1j * k/(r**2)  # shape (M, N)

        # Define Phi for each coordinate component
        Phi_x = 3*x/(r**5) + 3j*k*x/(r**4) - (k**2)*x/(r**3)
        Phi_y = 3*y/(r**5) + 3j*k*y/(r**4) - (k**2)*y/(r**3)
        Phi_z = 3*z/(r**5) + 3j*k*z/(r**4) - (k**2)*z/(r**3)

        # Phase term for spherical wave propagation; shape (M, N)
        phase = np.exp(-1j * k * r)

        # Electric field components (each with shape (M, N))
        # For electric fields, note: xi/(k^2)*R and xi/r are broadcast over dipoles and points.
        E_x = (dx * (xi/(k**2) * R - xi/r) - xi/(k**2) * Phi_x * dotted) * phase
        E_y = (dy * (xi/(k**2) * R - xi/r) - xi/(k**2) * Phi_y * dotted) * phase
        E_z = (dz * (xi/(k**2) * R - xi/r) - xi/(k**2) * Phi_z * dotted) * phase
        
        # Stack the electric components; shape becomes (M, N, 3)
        E = np.stack((E_x, E_y, E_z), axis=2)
        
        # Magnetic field components
        # Here we use the original approximate formulation; note that they only depend on dipole geometry.
        factor = 1/(4*np.pi)
        H_x = factor * ((dy*z - dz*y) * R) * phase
        H_y = factor * ((dz*x - dx*z) * R) * phase
        H_z = factor * ((dx*y - dy*z) * R) * phase
        
        # Stack the magnetic components; shape (M, N, 3)
        H = np.stack((H_x, H_y, H_z), axis=2)
        
        # Pack the two fields into a single array with shape 2×M×N×3
        fields = np.empty((2, M, N, 3), dtype=complex)
        fields[0, :, :, :] = E
        fields[1, :, :, :] = H

        return fields
