import numpy as np
import matplotlib.pyplot as plt

# Load data
scales = np.load("scales.npy").flatten()
e1     = np.load("e1.npy")[0,0,:]  # Shape (N,)
e2     = np.load("e2.npy")[0,0,:]
e3     = np.load("e3.npy")[0,0,:]
e4     = np.load("e4.npy")[0,0,:]

e1_heu = np.load("e1_heu.npy")[0,0,:]
e2_heu = np.load("e2_heu.npy")[0,0,:]
e3_heu = np.load("e3_heu.npy")[0,0,:]
e4_heu = np.load("e4_heu.npy")[0,0,:]

errors = [(e1, e1_heu), (e2, e2_heu), (e3, e3_heu), (e4, e4_heu)]
titles = [
    r'$||\tau_1 \cdot (E^{\mathrm{scat}} + E^{\mathrm{inc}} - E^{\mathrm{tot}})|| / ||\tau_1 \cdot E^{\mathrm{inc}}||$',
    r'$||\tau_2 \cdot (E^{\mathrm{scat}} + E^{\mathrm{inc}} - E^{\mathrm{tot}})|| / ||\tau_2 \cdot E^{\mathrm{inc}}||$',
    r'$||\tau_1 \cdot (H^{\mathrm{scat}} + H^{\mathrm{inc}} - H^{\mathrm{tot}})|| / ||\tau_1 \cdot H^{\mathrm{inc}}||$',
    r'$||\tau_2 \cdot (H^{\mathrm{scat}} + H^{\mathrm{inc}} - H^{\mathrm{tot}})|| / ||\tau_2 \cdot H^{\mathrm{inc}}||$',
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)

for i, ax in enumerate(axes.flat):
    err_std, err_heu = errors[i]
    ax.plot(scales, err_std, '-o', label='Standard')
    ax.plot(scales, err_heu, '-o', label='Heuristic')
    ax.set_title(titles[i], fontsize=10)
    ax.grid(True)

axes[1, 0].set_xlabel('Wavelength Scale (Surface size / λ)')
axes[1, 1].set_xlabel('Wavelength Scale (Surface size / λ)')
axes[0, 0].set_ylabel('Relative Error')
axes[1, 0].set_ylabel('Relative Error')

axes[0, 1].legend()

plt.tight_layout()
plt.savefig("heumethod.png")
plt.show()
