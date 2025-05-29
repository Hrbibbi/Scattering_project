import Forward_solver as FW
import plane_wave_jit as PW
import numpy as np
import matplotlib.pyplot as plt
import Spline_function as SP
import json
import os

def bump_function_wrapper(json_path):
    with open(json_path, 'r') as f:
        params = json.load(f)

    width = params['halfWidth_x']
    resol = 200
    alpha = params['alpha']
    bump_params = params['bumpData']
    scatter_epsilon = params['epsilon1']
    mu = 1  # Assumed constant

    # -------------------------
    # Surface creation
    # -------------------------
    a, b = -width, width
    X0 = np.linspace(a, b, resol)
    Y0 = np.linspace(a, b, resol)
    X, Y = np.meshgrid(X0, Y0)

    def bump(x, y, x0, y0, height, sigma):
        return height * np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    def surface_function(x, y):
        return sum(
            bump(x, y, b['x0'], b['y0'], b['height'], b['sigma'])
            for b in bump_params
        )

    Z = np.zeros_like(X)

    Z += surface_function(X, Y)
    print("there")
    return X,Y,Z,scatter_epsilon,mu

def check_transmission_conditions(Scatter_information,Incident_information,plotting=False,dipoles_per_wl=5,scaling=0.86):
    #surface stuff
    int_coeffs, ext_coeffs, InteriorDipoles, ExteriorDipoles=FW.Construct_solve_MAS_system(Scatter_information,Incident_information,dipoles_per_wl=dipoles_per_wl,scaling=scaling)
    SPSurface=Scatter_information['SPSurface']
    testpoints, tau_1, tau_2, _=SPSurface.construct_auxiliary_points(100,0)

    #incident stuff
    propagation_vectors = Incident_information['propagation_vectors']  # shape (R, 3)
    polarizations = Incident_information['polarizations']              # shape (R,)
    epsilon_air = Incident_information['epsilon']
    mu = Incident_information['mu']
    omega = Incident_information['omega']
    lam = Incident_information['lambda']
    planewaves = PW.Plane_wave(propagation_vectors, polarizations, epsilon_air, mu, omega)

    E_scat,H_scat=FW.compute_linear_combinations(testpoints,int_coeffs,InteriorDipoles) #(R,N,3)
    E_tot,H_tot  =FW.compute_linear_combinations(testpoints,ext_coeffs,ExteriorDipoles) #(R,N,3)   
    E_inc,H_inc  =planewaves.evaluate_at_points(testpoints)                             #(R,N,3)
    def einsumming(E_field,H_field):
        STI1=np.einsum("rnk,nk->rn",E_field,tau_1) # (R,N)
        STI2=np.einsum("rnk,nk->rn",E_field,tau_2)  # (R,N)
        STI3=np.einsum("rnk,nk->rn",H_field,tau_1)  # (R,N)
        STI4=np.einsum("rnk,nk->rn",H_field,tau_2)  # (R,N)
        return np.hstack([STI1,STI2,STI3,STI4])[0]
   
    STinc=einsumming(E_scat-E_tot+E_inc,H_scat-H_tot+H_inc)
    ST=einsumming(E_scat-E_tot,H_scat-H_tot)
    inc=einsumming(-E_inc,-H_inc)
    E_scat=E_scat[0]
    E_tot=E_tot[0]
    if plotting:
        fig, axes = plt.subplots(2, 2, figsize=(12, 5))

        # Top-left: Difference in transmission conditions (E_tot⁺ - E_tot⁻)
        axes[0, 0].plot(np.abs(STinc))
        axes[0, 0].set_title("|E_tot⁺ - E_tot⁻| (Transmission Condition)")
        axes[0, 0].set_xlabel("Transmission Condition Index")
        axes[0, 0].set_ylabel("Magnitude")

        # Top-right: Energy conservation comparison
        axes[0, 1].plot(np.abs(ST), label="|E_MAS|")
        axes[0, 1].plot(np.abs(inc), label="|E_inc|")
        axes[0, 1].set_title("Energy Balance: MAS vs Incident")
        axes[0, 1].set_xlabel("Transmission Condition Index")
        axes[0, 1].set_ylabel("Magnitude")
        axes[0, 1].legend()

        # Bottom-left: Norm of approximated scattered field
        axes[1, 0].plot(np.linalg.norm(E_scat, axis=1))
        axes[1, 0].set_title("‖E_scat‖ (Approximated Scattered Field)")
        axes[1, 0].set_xlabel("Test Point Index")
        axes[1, 0].set_ylabel("Field Norm")

        # Bottom-right: Norm of approximated total field
        axes[1, 1].plot(np.linalg.norm(E_tot, axis=1))
        axes[1, 1].set_title("‖E_tot‖ (Approximated Total Field)")
        axes[1, 1].set_xlabel("Test Point Index")
        axes[1, 1].set_ylabel("Field Norm")

        plt.tight_layout()
        plt.savefig(f"transmission_error_scaling_{scaling}.png")
        plt.show()
    return np.linalg.norm(STinc,2) /np.linalg.norm(inc,2)

def plot_transmission_scale_dipoles(json_path,output_folder="transmission_plots",output_name="bump_test",scaling=1.0):
    X,Y,Z,scatter_epsilon,mu=bump_function_wrapper(json_path)
    print("here")
    Surface=SP.SplineSurface(X,Y,Z,smoothness=0.5)
    print("surface created")
    Scatterinformation = {
    'SPSurface': Surface,
    'epsilon': scatter_epsilon,
    'mu': mu
    }
    dipole_range = [5,6,7]
    wavelength_range = np.linspace(1, 0.5, 10)

    all_scales = []
    all_errors = []

    for dipoles_pr_wl in dipole_range:
        scales = []
        relative_errors = []

        for wavelength in wavelength_range:
            propagation_vector = np.array([[0, 0, -1]])
            polarization = np.array([0])
            epsilon_air = 1
            omega = 2 * np.pi / wavelength
            wavelength_scale = Surface.size / wavelength

            Incidentinformation = {
                'propagation_vectors': propagation_vector,
                'polarizations': polarization,
                'epsilon': epsilon_air,
                'mu': mu,
                'lambda': wavelength,
                'omega': omega
            }

            rel_error = check_transmission_conditions(
                Scatterinformation,
                Incidentinformation,
                plotting=False,
                dipoles_per_wl=dipoles_pr_wl,
                scaling=scaling
            )

            scales.append(wavelength_scale)
            relative_errors.append(rel_error)

        all_scales.append(scales)
        all_errors.append(relative_errors)

    # Evaluate surface height for contour plot
    x0 = np.linspace(Surface.a, Surface.b, Surface.fine_resol)
    z_eval = Surface._evaluate_spline(x0, x0)

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Surface height contour
    cf = axes[0].contourf(Surface.x_fine, Surface.y_fine, z_eval, cmap='viridis')
    fig.colorbar(cf, ax=axes[0])
    axes[0].set_title('Surface Height')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')

    # Subplot 2: Multiple relative error curves
    for scales, errors, dpw in zip(all_scales, all_errors, dipole_range):
        axes[1].plot(scales, errors, marker='o', label=f'{dpw} dip./λ')

    axes[1].set_xlabel('Wavelength Scale (Surface size / λ)')
    axes[1].set_ylabel('||E^scat+E^inc-E^tot||/||E^inc||')
    axes[1].set_title('Error vs. Wavelength Scale')
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()

    # Save figure
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f'{output_name}_scaling{scaling}.png')
    plt.savefig(output_path)
    plt.close(fig)

def plot_MAS_system(json_path,output_folder="MAS_system_plots",output_name="bump_test",scaling=1):
    X,Y,Z,scatter_epsilon,mu=bump_function_wrapper(json_path)
    SPSurface=SP.SplineSurface(X,Y,Z)
    propagation_vector = np.array([[0, 0, -1]])
    polarization = np.array([0])
    epsilon_air = 1
    wavelength = 0.5
    omega = 2 * np.pi / wavelength
    Surface,inneraux,outeraux=SPSurface.sample_surface_MAS(wavelength,scaling=scaling,dipoles_per_wl=5)
    RHS_matrix = FW.construct_RHSs(Surface, propagation_vector, polarization, epsilon_air, mu, omega)
    MAS_matrix, intDP1, intDP2, extDP1, extDP2 = FW.construct_matrix(Surface,inneraux,outeraux,mu,epsilon_air,scatter_epsilon,omega)
    C_matrix, *_ = np.linalg.lstsq(MAS_matrix, RHS_matrix, rcond=None) #solution size 4MxR
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10),
                         gridspec_kw={'height_ratios': [2, 1, 1]})  # First row is 2x taller

    # Row 1: MAS_matrix
    im0 = axes[0, 0].imshow(np.abs(MAS_matrix))
    axes[0, 0].set_title('Abs(MAS_matrix)')
    fig.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(np.log(np.abs(MAS_matrix)+1))
    axes[0, 1].set_title('log of MAS')
    fig.colorbar(im1, ax=axes[0, 1])

    # Row 2: RHS_matrix (assumed shape Nx1 or N)
    axes[1, 0].plot(np.abs(RHS_matrix).squeeze())
    axes[1, 0].set_title('Abs(RHS_matrix)')

    axes[1, 1].plot(np.real(RHS_matrix).squeeze())
    axes[1, 1].set_title('Re(RHS_matrix)')

    # Row 3: C_matrix (assumed shape Nx1 or N)
    axes[2, 0].plot(np.abs(C_matrix).squeeze())
    axes[2, 0].set_title('Abs(C_matrix)')

    axes[2, 1].plot(np.real(C_matrix).squeeze())
    axes[2, 1].set_title('Re(C_matrix)')

    # Layout and display
    for ax in axes.flat:
        ax.grid(True)

    # Save figure
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f'{output_name}_scaling{scaling}.png')
    plt.savefig(output_path)
    plt.close(fig)


def plot_distance_matrix(json_path, output_folder="Distance_matrix_plots", output_name="bump_test", scaling=1):
    # Load surface and auxiliary data
    from scipy.spatial.distance import cdist
    X, Y, Z, scatter_epsilon, mu = bump_function_wrapper(json_path)
    SPSurface = SP.SplineSurface(X, Y, Z)

    propagation_vector = np.array([[0, 0, -1]])
    polarization = np.array([0])
    epsilon_air = 1
    wavelength = 0.5
    omega = 2 * np.pi / wavelength

    # Sample surface and auxiliary layers
    Surface, inneraux, outeraux = SPSurface.sample_surface_MAS(wavelength, scaling=scaling)
    Surface_points = Surface.points
    inneraux_points = inneraux.points
    outeraux_points = outeraux.points

    # Compute distance matrices
    dist_inner_to_surface = cdist(inneraux_points, Surface_points)
    dist_outer_to_surface = cdist(outeraux_points, Surface_points)

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    im1 = axs[0].imshow(1 / dist_inner_to_surface, aspect='auto', cmap='viridis')
    axs[0].set_title("1 / Distance: Inner Aux → Surface")
    axs[0].set_xlabel("Surface Point Index")
    axs[0].set_ylabel("Inner Aux Point Index")
    fig.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(1 / dist_outer_to_surface, aspect='auto', cmap='viridis')
    axs[1].set_title("1 / Distance: Outer Aux → Surface")
    axs[1].set_xlabel("Surface Point Index")
    axs[1].set_ylabel("Outer Aux Point Index")
    fig.colorbar(im2, ax=axs[1])

    plt.tight_layout()

    # Save figure to file
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f'{output_name}_scaling{scaling}.png')
    plt.savefig(output_path)
    plt.close(fig)

def plot_single_transmission_conditions(json_path,output_folder="transmission_plots",output_name="bump_test_single",scaling=1):
    X,Y,Z,scatter_epsilon,mu=bump_function_wrapper(json_path)
    SPSurface=SP.SplineSurface(X,Y,Z)
    propagation_vector = np.array([[0, 0, -1]])
    polarization = np.array([np.pi/4])
    epsilon_air = 1
    wavelength = 0.5
    omega = 2 * np.pi / wavelength
    Scatterinformation = {
    'SPSurface': SPSurface,
    'epsilon': scatter_epsilon,
    'mu': mu
        }
    Incidentinformation = {
        'propagation_vectors': propagation_vector,
        'polarizations': polarization,
        'epsilon': epsilon_air,
        'mu': mu,
        'lambda': wavelength,
        'omega': omega
        }
    print(Incidentinformation)
    check_transmission_conditions(Scatterinformation,Incidentinformation,plotting=True,dipoles_per_wl=5,scaling=scaling)

def plot_surface(json_path):
    X,Y,Z,scatter_epsilon,mu=bump_function_wrapper(json_path)
    SPSurface=SP.SplineSurface(X,Y,Z)
    x0=np.linspace(SPSurface.a,SPSurface.b,SPSurface.fine_resol)
    Z_spline=SPSurface._evaluate_spline(x0,x0)
    plt.contourf(X,Y,Z)
    plt.show()
    plt.contourf(X,Y,Z_spline)
    plt.colorbar()
    plt.show()
if True:
    scale=2
    param="20"
    plot_transmission_scale_dipoles(f"Json_files/surfaceParams{param}.json",output_name=f"{param}_bump",scaling=scale)
    plot_MAS_system(f"Json_files/surfaceParams{param}.json",output_name=f"{param}_bump",scaling=scale)
    plot_distance_matrix(f"Json_files/surfaceParams{param}.json",output_name=f"{param}_bump",scaling=scale)
    plot_single_transmission_conditions(f"Json_files/surfaceParams{param}.json",output_name=f"{param}_bump",scaling=scale)

#plot_surface("Json_files/surfaceParams200.json")
#plot_single_transmission_conditions("Json_files/surfaceParamsTen.json",scaling=0.14)
#plot_single_transmission_conditions("Json_files/surfaceParamsTen.json",scaling=1.00)
#plot_single_transmission_conditions("Json_files/surfaceParamsTen.json",scaling=2.00)
#plot_MAS_system("Json_files/surfaceParamsOne.json",output_name="one_bump",scaling=0.14)
#plot_MAS_system("Json_files/surfaceParamsOne.json",output_name="one_bump",scaling=1)
#plot_MAS_system("Json_files/surfaceParamsTen.json",output_name="MASTen_bump",scaling=0.14)
#plot_MAS_system("Json_files/surfaceParamsTen.json",output_name="MASTen_bump",scaling=1)

#plot_distance_matrix("Json_files/surfaceParamsOne.json",output_name="one_bump",scaling=0.14)
#plot_distance_matrix("Json_files/surfaceParamsOne.json",output_name="one_bump",scaling=1)
#plot_distance_matrix("Json_files/surfaceParamsTen.json",output_name="Ten_bump",scaling=0.14)
#plot_distance_matrix("Json_files/surfaceParamsTen.json",output_name="Ten_bump",scaling=1)






#plot_MAS_system("Json_files/surfaceParamsOne.json",output_name="one_bump_0.14")
#plot_MAS_system("Json_files/surfaceParamsZero.json",output_name="zero_bump_0.14")
#plot_MAS_system("Json_files/surfaceParamsTen.json",output_name="ten_bump",scaling=1)
#plot_MAS_system("Json_files/surfaceParamsTen.json",output_name="ten_bump",scaling=0.14)

#plot_distance_matrix("Json_files/surfaceParamsTen.json",scaling=1)
#plot_transmission_scale_dipoles("Json_files/surfaceParamsOne.json",output_name="One_bump",scaling=0.14)
#plot_transmission_scale_dipoles("Json_files/surfaceParamsZero.json",output_name="Zero_bump",scaling=0.14)
#plot_transmission_scale_dipoles("Json_files/surfaceParamsTen.json",output_name="Ten_bump",scaling=2)
#plot_transmission_scale_dipoles("Json_files/surfaceParamsOne.json",output_name="One_bump",scaling=1)
#plot_transmission_scale_dipoles("Json_files/surfaceParamsZero.json",output_name="Zero_bump",scaling=1)
#plot_transmission_scale_dipoles("Json_files/surfaceParamsTen.json",output_name="Ten_bump",scaling=1)
#plot_single_transmission_conditions("Json_files/surfaceParamsTen.json")