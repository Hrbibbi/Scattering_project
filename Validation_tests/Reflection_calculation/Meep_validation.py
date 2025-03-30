import meep as mp
import numpy as np
import matplotlib.pyplot as plt

def substrate_sim():
    # Simulation parameters
    resolution = 10  # pixels per wavelength
    dpml = 1.0  # PML thickness
    sx = 16.0  # Size of simulation domain (x-direction)
    sy = 10.0  # Size of simulation domain (y-direction)
    sz = 16.0  # Size of simulation domain (z-direction)
    cell_size = mp.Vector3(sx, sy, sz)
    
    # Define materials
    air = mp.Medium(epsilon=1.0)
    silicon = mp.Medium(epsilon=11.7)  # Silicon substrate
    
    # Define geometry (extending substrate fully to the bottom)
    geometry = [mp.Block(size=mp.Vector3(mp.inf, sy/2 + dpml, mp.inf),
                         center=mp.Vector3(0, -sy/4 - dpml/2, 0),
                         material=silicon)]
    
    # Define boundary conditions (use absorber at bottom instead of PML)
    pml_layers = [mp.PML(dpml, direction=mp.Y, side=mp.High),
                  mp.Absorber(dpml, direction=mp.Y, side=mp.Low)]
    
    # Define source (plane wave propagating in -y direction, polarized along z)
    fcen = 1.0  # Frequency (arbitrary units)
    k_vector = mp.Vector3(0, -fcen, 0)  # Propagation in -y direction
    source = [mp.EigenModeSource(mp.ContinuousSource(fcen),
                                  component=mp.Ez,
                                  center=mp.Vector3(0, sy/2 - dpml, 0),
                                  size=mp.Vector3(sx, 0, sz),
                                  direction=mp.Y,
                                  eig_kpoint=k_vector,
                                  eig_parity=mp.ODD_Z,
                                  eig_match_freq=True)]
    
    # Create simulation object
    sim = mp.Simulation(cell_size=cell_size,
                         boundary_layers=pml_layers,
                         geometry=geometry,
                         sources=source,
                         resolution=resolution)
    
    # Add Fourier transforms to extract phasor representation
    dft = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez], fcen, 0, 1, center=mp.Vector3(0, 1, 0), size=mp.Vector3(sx, 0, sz))
    
    # Run simulation
    sim.run(until=200)
    
    # Extract complex field data in the (x,z) plane at y=1
    Ex_data = sim.get_dft_array(dft, mp.Ex, 0)
    Ey_data = sim.get_dft_array(dft, mp.Ey, 0)
    Ez_data = sim.get_dft_array(dft, mp.Ez, 0)
    
    # Compute magnitude and phase
    Ex_mag, Ex_phase = np.abs(Ex_data), np.angle(Ex_data)
    Ey_mag, Ey_phase = np.abs(Ey_data), np.angle(Ey_data)
    Ez_mag, Ez_phase = np.abs(Ez_data), np.angle(Ez_data)
    
    # Plot Magnitude of Ez
    plt.figure(figsize=(8, 6))
    plt.imshow(Ez_mag.T, cmap='inferno', interpolation='spline36', extent=[-sx/2, sx/2, -sz/2, sz/2])
    plt.colorbar(label='|Ez|')
    plt.title('Magnitude of Ez in (x,z) plane at y=1')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()
    
    # Plot Phase of Ez
    plt.figure(figsize=(8, 6))
    plt.imshow(Ez_phase.T, cmap='twilight', interpolation='spline36', extent=[-sx/2, sx/2, -sz/2, sz/2])
    plt.colorbar(label='Phase of Ez (radians)')
    plt.title('Phase of Ez in (x,z) plane at y=1')
    plt.xlabel('x')
    plt.ylabel('z')
    plt.show()
    
if __name__ == "__main__":
    substrate_sim()