import Hertzian_dipole as HD
import C2_surface as C2
import numpy as np
import matplotlib.pyplot as plt
#S=C2.sphere(1,np.array([0,0,0]),20)
S=C2.cylinder(1,1,20)
S1=S.construct_conformal_surface(0.8)
S1.plot_tangents()
plt.show()
'''
S=C2.sphere(1,np.array([0,0,0]),20)
S1=S.construct_conformal_surface(0.8)
S.plot_surface()
S1.plot_surface()
plt.show()
'''