import meep as mp
import numpy as np
import matplotlib.pyplot as plt

def run_sim(rot_angle=0):

    resolution = 25  # pixels/μm

    sx = 12
    sy = 16
    cell_size = mp.Vector3(sx, sy, 0)  # for 2D


    pml_layers = [mp.PML(thickness=2, direction=mp.Y)]

    fsrc = 1.0  # frequency of planewave (wavelength = 1/fsrc)

    n = 1  # refractive index of homogeneous material
    default_material = mp.Medium(index=n)

    k_point = mp.Vector3(y=-fsrc * n).rotate(mp.Vector3(z=1), rot_angle)

    geometry1 = [
        mp.Block(
            center=mp.Vector3(0, -sy/4),              # centered halfway down
            size=mp.Vector3(mp.inf, sy/2, mp.inf),    # fill y < 0
            material=mp.Medium(epsilon=2.56)
        )
    ]


    geometry2 = [
    mp.Cylinder(
        radius=1.0,
        height=mp.inf,                     # for 2D: extend infinitely in z
        center=mp.Vector3(0, 1),           # center at (0, 1)
        material=mp.Medium(epsilon=2.56)    # dielectric with ε=4 (or use mp.metal for PEC)
    )
    ]



    sources = [
        mp.EigenModeSource(
            src=mp.ContinuousSource(fsrc),
            center=mp.Vector3(0,6),
            size=mp.Vector3(x=12),
            direction=mp.AUTOMATIC if rot_angle == 0 else mp.NO_DIRECTION,
            eig_kpoint=k_point,
            eig_band=1,
            eig_parity=mp.EVEN_Y + mp.ODD_Z if rot_angle == 0 else mp.ODD_Z,
            eig_match_freq=True,
        )
    ]

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        geometry = geometry1 + geometry2,
        boundary_layers=pml_layers,
        sources=sources,
        k_point=k_point,
        default_material=default_material,
        symmetries=[mp.Mirror(mp.Y)] if rot_angle == 0 else [],
    )

    sim.run(until=100)

    plt.figure(dpi=100)
    sim.plot2D(fields=mp.Ex)
    plt.show()
for rot_angle in np.radians([0.1]):
    run_sim(rot_angle)