import Domains
import plane_wave_jit as PW
import Discrete_dipole_forward as FW
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot
import plotly.graph_objects as go
import os

def log_likelihood(sample,domain: Domains.Domain,planewaves,measurement_values,beta):
    X=domain.X
    X=X.ravel()
    angles=Domains.gp_to_euler_angles(sample,X)
    alpha_tensor=Rot.from_euler('ZYX', angles, degrees=False).as_matrix()
    if False:
        simulated_measurements=np.hstack([FW.poynting_far_field(np.array([0,0,1]),alpha_tensor,domain,planewaves),
                                        FW.poynting_far_field(np.array([0,1,0]),alpha_tensor,domain,planewaves),
                                        FW.poynting_far_field(np.array([1,0,0]),alpha_tensor,domain,planewaves)])
    if True:
        thetas = np.linspace(0, np.pi/2, 3)
        phis = np.linspace(0, 2*np.pi, 3, endpoint=False)
        directions = np.array([[np.sin(theta) * np.cos(phi),np.sin(theta) * np.sin(phi),np.cos(theta)]
                    for theta in thetas for phi in phis])

        simulated_measurements = np.hstack([FW.poynting_far_field(direction, alpha_tensor, domain, planewaves)
                for direction in directions])
    #simulated_measurements=FW.poynting_far_field(np.array([0,0,1]),alpha_tensor,domain,planewaves)
    normdiff = np.linalg.norm(measurement_values - simulated_measurements)**2
    loglike = -0.5 * beta * normdiff
    return loglike

def pCN(GP_factorization,domain: Domains.Domain,planewaves: PW.Plane_wave,measurement_values,
        stepsize, beta,num_iter,acceptance_rate,warmup_period,start=None):
    '''
    Preconditioned Crank-Nicolson scheme to give proposals for the permetivity tensor

    Input:
        GP_factorization: cholesky factorization for the GP covariance matrix
        true_measurements: dataframe of true measurements to compare the proposals to
        step_size: stepsize in pCN
        beta: likelihood precision
        num_iter: maximal number of iterations
        warmup_period: after what percent of num_iter to start accepting proposals
        convergence_tol: tolerance for difference in log-likelihood after warmup period
    '''
    #----------------------------------------------------
    # Initalization
    #----------------------------------------------------
    delta = stepsize
    if start is None:
        proposal = Domains.sample_gp_field(GP_factorization)
    else:
        proposal = start.copy()
    L_old = log_likelihood(proposal,domain,planewaves,measurement_values,beta)
    accepted_proposals=[]
    accepted_likelihood=[]
    num_acceptance=0
    for k in range(num_iter):
        iter_start=time.time()
        #----------------------------------------------------
        # Generate new proposal and new likelihood
        #----------------------------------------------------

        new_sample = Domains.sample_gp_field(GP_factorization)
        new_proposal = np.sqrt(1 - 2 * delta) * proposal + np.sqrt(2 * delta) * new_sample

        L_new = log_likelihood(new_proposal,domain,planewaves,measurement_values,beta)
        
        Ak = min(1, np.exp(L_new - L_old))
        uk = np.random.uniform()

        #----------------------------------------------------
        # Update step
        #----------------------------------------------------

        if uk < Ak:
            
            if k>num_iter*warmup_period:
                accepted_proposals.append(new_proposal)
                accepted_likelihood.append(L_new)
            
            proposal = new_proposal
            L_old = L_new 
            num_acceptance+=1
        #printing
        if k%100==0:
            print(f"Iteration: {k}, Iteration time: {time.time()-iter_start}, loglike: {L_old} acceptance rate: {num_acceptance}, beta: {beta}")
            if k<num_iter*warmup_period:
                if num_acceptance<(acceptance_rate*100):
                    beta*=0.8
                else:
                    beta*=1.2
            
            num_acceptance=0
    return np.array(accepted_proposals),np.array(accepted_likelihood)  # return accepted_proposals

def MC_pCN(
    GP_factorization, domain, planewaves, measurement_values,
    delta_explore=0.3, beta_explore=1e2, num_iter_explore=10000,
    delta_refine=0.01, beta_refine=1e3, num_iter_refine=5000,
    n_phase2=20,top_n_per_chain=10,true_values=None
):
    # -------------------------------------------
    # Phase 1: Broad exploration
    # -------------------------------------------
    print("Phase 1: Exploration")
    props_explore, likes_explore = pCN(
        GP_factorization, domain, planewaves, measurement_values,
        stepsize=delta_explore, beta=beta_explore,
        num_iter=num_iter_explore, warmup_period=0.5,acceptance_rate=0.5)
    props_explore = np.array(props_explore)
    likes_explore = np.array(likes_explore)

    # Select top n_phase2 samples for Phase 2
    top_idxs_1 = np.argsort(likes_explore)[-n_phase2:][::-1]
    starts_refine1 = [props_explore[i] for i in top_idxs_1]

    if true_values is not None:
        true_angle,true_alpha,output_folder=true_values
        plot_alpha_components(np.array(starts_refine1), true_alpha,domain,output_folder=output_folder,filename_prefix="alpha_phase1_explore")
        plot_euler_angles(np.array(starts_refine1),true_angle,domain,output_folder=output_folder,filename_prefix="angles_phase1_explore")

    print("Phase 2: refinement from top samples")
    final_proposals = []
    final_loglikes = []

    for i, start in enumerate(starts_refine1):
        print(f"Phase 2 chain {i+1}/{n_phase2}")
        p3_props, p3_lls = pCN(
            GP_factorization, domain, planewaves, measurement_values,
            stepsize=delta_refine, beta=beta_refine,
            num_iter=num_iter_refine, warmup_period=0.5, start=start,acceptance_rate=0.3
        )
        p3_lls = np.array(p3_lls)
        top_idxs = np.argsort(p3_lls)[-top_n_per_chain:][::-1]
        final_proposals.extend(p3_props[i] for i in top_idxs)
        final_loglikes.extend(p3_lls[i] for i in top_idxs)

    return np.array(final_proposals), np.array(final_loglikes)

if True:
    def plot_alpha_components(accepted_proposals, alpha_true,domain,output_folder, filename_prefix="alpha_components"):
        """
        Plot all 9 components of the alpha tensor comparing accepted GP samples with ground truth.

        Parameters:
            accepted_proposals: (K, N, 3) array of GP samples (not yet Euler angles)
            alpha_true: (N, 3, 3) array of ground truth rotation matrices
        """
        X=domain.X
        X=X.ravel()
        K, _, _ = accepted_proposals.shape
        N= X.shape[0]
        alpha_tensor_samples = np.empty((K, N, 3, 3))

        for k in range(K):
            angles_k = Domains.gp_to_euler_angles(accepted_proposals[k],X)  # (N, 3)
            alpha_tensor_samples[k] = Rot.from_euler('ZYX', angles_k, degrees=False).as_matrix()  # (N, 3, 3)

        mean_alpha = np.mean(alpha_tensor_samples, axis=0)  # (N, 3, 3)

        fig, axes = plt.subplots(3, 3, figsize=(15, 10))

        for i in range(3):
            for j in range(3):
                ax = axes[i, j]
                for k in range(K):
                    ax.plot(alpha_tensor_samples[k, :, i, j], color='gray', alpha=0.3)
                ax.plot(alpha_true[:, i, j], color='blue', label='True')
                ax.plot(mean_alpha[:, i, j], color='red', label='Mean Proposal')
                ax.set_title(f'alpha[{i},{j}]')
                ax.set_ylim(-1, 1)
                if i == 0 and j == 0:
                    ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{filename_prefix}.png"))
        plt.close()
    def plot_euler_angles(accepted_proposals, angles_true,domain,output_folder, filename_prefix="euler_angles"):
        """
        Plot Euler angles (theta, phi, psi) comparing accepted GP samples with ground truth.

        Parameters:
            accepted_proposals: (K, N, 3) array of GP samples (not yet Euler angles)
            angles_true: (N, 3) array of ground truth Euler angles
            filename_prefix: prefix for saved plot filename
        """
        X=domain.X
        X=X.ravel()
        K, _, _ = accepted_proposals.shape
        N= X.shape[0]
        euler_angle_samples = np.empty((K, N, 3))

        for k in range(K):
            euler_angle_samples[k] = Domains.gp_to_euler_angles(accepted_proposals[k],X)  # (N, 3)

        mean_angles = np.mean(euler_angle_samples, axis=0)  # (N, 3)
        angle_labels = ['Yaw (θ)', 'Pitch (φ)', 'Roll (ψ)']

        fig, axes = plt.subplots(3, 1, figsize=(12, 8))

        angle_limits = [np.pi, np.pi / 2, np.pi / 2]

        for i in range(3):
            ax = axes[i]
            for k in range(K):
                ax.plot(euler_angle_samples[k, :, i], color='gray', alpha=0.3)
            ax.plot(angles_true[:, i], color='blue', label='True')
            ax.plot(mean_angles[:, i], color='red', label='Mean Proposal')
            ax.set_title(angle_labels[i])
            ax.set_ylabel("Angle (radians)")
            ax.set_xlabel("Point index")
            ax.set_ylim(0, angle_limits[i])  # <- Set axis range per subplot
            if i == 0:
                ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{filename_prefix}.png"))
        plt.close()
    def plot_comparison_dipole_moments_from_ensemble(alpha_true, accepted_proposals, domain, Incident_waves, output_folder, title):
        """
        Save an interactive 3D HTML plot comparing dipole moments between true and mean proposal from an ensemble.
        All dipole arrows are normalized to the same fixed length (0.2) to emphasize direction.
        """
        X=domain.X
        X=X.ravel()
        K, _, _ = accepted_proposals.shape
        N= X.shape[0]
        alpha_tensor_samples = np.empty((K, N, 3, 3))
        for k in range(K):
            angles_k = Domains.gp_to_euler_angles(accepted_proposals[k],X)  # (N, 3)
            alpha_tensor_samples[k] = Rot.from_euler('ZYX', angles_k, degrees=False).as_matrix()  # (N, 3, 3)

        points = domain.points  # (N, 3)
        E_new, _ = Incident_waves.evaluate_at_points_reflection(points)  # (R, N, 3)

        dipoles_true = np.einsum('mij,rmj->rmi', alpha_true, np.real(E_new))[0]       # (N, 3)
        dipoles_prop_all = np.einsum('kmij,rmj->krmi', alpha_tensor_samples, np.real(E_new))  # (K, R, N, 3)
        dipoles_prop_mean = np.mean(dipoles_prop_all[:, 0], axis=0)  # mean over K at r=0 → (N, 3)

        X, Y, Z = points[:, 0], points[:, 1], points[:, 2]

        fig = go.Figure()

        # Add true dipoles (rescaled to fixed length)
        for i in range(N):
            u, v, w = dipoles_true[i]
            norm = np.linalg.norm([u, v, w])
            if norm > 0:
                u, v, w = 0.2 * np.array([u, v, w]) / norm
            fig.add_trace(go.Scatter3d(
                x=[X[i] - 0.5 * u, X[i] + 0.5 * u],
                y=[Y[i] - 0.5 * v, Y[i] + 0.5 * v],
                z=[Z[i] - 0.5 * w, Z[i] + 0.5 * w],
                mode='lines',
                line=dict(color='blue'),
                name='True' if i == 0 else None,
                showlegend=(i == 0)
            ))

        # Add proposal dipoles (rescaled to fixed length)
        for i in range(N):
            u, v, w = dipoles_prop_mean[i]
            norm = np.linalg.norm([u, v, w])
            if norm > 0:
                u, v, w = 0.2 * np.array([u, v, w]) / norm
            fig.add_trace(go.Scatter3d(
                x=[X[i] - 0.5 * u, X[i] + 0.5 * u],
                y=[Y[i] - 0.5 * v, Y[i] + 0.5 * v],
                z=[Z[i] - 0.5 * w, Z[i] + 0.5 * w],
                mode='lines',
                line=dict(color='orange'),
                name='Mean Proposal' if i == 0 else None,
                showlegend=(i == 0)
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            legend=dict(itemsizing='constant')
        )

        output_path = os.path.join(output_folder, f"{title.replace(' ', '_')}.html")
        fig.write_html(output_path)
    def plot_dipole_angle_discrepancy_from_ensemble(alpha_true, accepted_proposals, domain, Incident_waves,output_folder, filename_prefix="dipole_angle_discrepancy"):
        """
        Plot average angular error (mod π) between true polarization and *mean dipoles* from an ensemble of proposals.

        Parameters:
            alpha_true: (N, 3, 3)
            alpha_proposals: (K, N, 3, 3)
            domain: Domain object
            Incident_waves: Plane_wave object
            title: filename prefix for plot
        """
        X=domain.X
        X=X.ravel()
        K, _, _ = accepted_proposals.shape
        N= X.shape[0]
        alpha_tensor_samples = np.empty((K, N, 3, 3))
        for k in range(K):

            angles_k = Domains.gp_to_euler_angles(accepted_proposals[k],X)  # (N, 3)
            alpha_tensor_samples[k] = Rot.from_euler('ZYX', angles_k, degrees=False).as_matrix()  # (N, 3, 3)
        points = domain.points  # (N, 3)
        E_new, _ = Incident_waves.evaluate_at_points_reflection(points)  # (R, N, 3)

        dipoles_true = np.einsum('mij,rmj->rmi', alpha_true, E_new)        # (R, N, 3)
        dipoles_prop_all = np.einsum('kmij,rmj->krmi', alpha_tensor_samples, E_new)  # (K, R, N, 3)
        dipoles_prop_mean = np.mean(dipoles_prop_all, axis=0)  # (R, N, 3)

        # Normalize dipoles
        norm_true = np.linalg.norm(dipoles_true, axis=2, keepdims=True)
        norm_prop = np.linalg.norm(dipoles_prop_mean, axis=2, keepdims=True)
        unit_true = dipoles_true / np.maximum(norm_true, 1e-12)
        unit_prop = dipoles_prop_mean / np.maximum(norm_prop, 1e-12)

        # Compute angle between dipole directions (mod π)
        cos_theta = np.einsum("rni,rni->rn", unit_true, unit_prop)  # (R, N)
        cos_theta_clipped = np.clip(np.abs(cos_theta), -1.0, 1.0)   # take |cos| to remove ± ambiguity
        angles = np.arccos(cos_theta_clipped)  # radians

        mean_angle_deg = np.rad2deg(np.mean(angles, axis=0))  # degrees per point (N,)

        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(mean_angle_deg, marker='o')
        plt.title("Mean dipole angle discrepancy per point (mod 180°)")
        plt.ylabel("Angle [degrees]")
        plt.xlabel("Dipole index")
        plt.ylim(0, 90)  # <- Set y-axis range
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f"{filename_prefix}.png"))
        plt.close()

    def run_clean_pCN_analysis(runname="test"):
        
        output_folder = os.path.join("results", runname)
        os.makedirs(output_folder, exist_ok=True)
        
        # SETUP: Test object (unit quaternions)
        X, Y, Z = Domains.cylinder_cartesian_grid(1, 1, numpoints_xy=5, numpoints_z=5)
        N = X.shape[0]
        Dom = Domains.Domain(X, Y, Z)
        #print(np.shape(Dom))
        #setup alpha_tensor
        R=np.sqrt(X**2+Y**2)#np.linspace(0,np.pi,N)
        polar_angle= np.arctan2(Y, X) 
        yaw=np.pi/2*np.ones_like(X)#R/R
        pitch=np.zeros_like(X)
        roll=np.pi/4 *np.ones_like(X) #Z
        true_angles=np.column_stack([yaw,pitch,roll])
        alpha_true=Rot.from_euler('ZYX', true_angles, degrees=False).as_matrix()

        # SETUP: Incident fields
        R = 50
        propagation_vectors = np.tile([0, 0, -1], (R, 1))
        betas = np.tile(np.pi / 4, R)
        omegas = np.linspace(0.01, 10, R)
        PW1 = PW.Plane_wave(propagation_vectors, betas, 1, 2.56, 1, omegas)

        if False:
            thetas = np.linspace(0, np.pi, 20)
            true_measurements=np.hstack([FW.poynting_far_field(np.array([0,0,1]),alpha_true,Dom,PW1),
                                        FW.poynting_far_field(np.array([0,1,0]),alpha_true,Dom,PW1),
                                        FW.poynting_far_field(np.array([1,0,0]),alpha_true,Dom,PW1)])
        if True:
            thetas = np.linspace(0, np.pi/2, 3)
            phis = np.linspace(0, 2*np.pi, 3, endpoint=False)
            directions = np.array([[np.sin(theta) * np.cos(phi),np.sin(theta) * np.sin(phi),np.cos(theta)]
                        for theta in thetas for phi in phis])

            true_measurements = np.hstack([
                FW.poynting_far_field(direction, alpha_true, Dom, PW1)
                for direction in directions
            ])
        # GP Prior and pCN sampling
        X_sample,Y_sample,Z_sample=X.ravel(),Y.ravel(),Z.ravel()
        X_sample,Y_sample,Z_sample=X_sample[::2],Y_sample[::2],Z_sample[::2]
        GP_factor = Domains.matern_covariance_matrix(X_sample, Y_sample, Z_sample)
        final_proposals,final_loglikes=MC_pCN(GP_factor,Dom,PW1,true_measurements,
                                num_iter_explore=10000,num_iter_refine=2000,n_phase2=15,beta_explore=1,beta_refine=100,
                                true_values=[true_angles,alpha_true,output_folder])
        np.save(os.path.join(output_folder, "final_proposals"),final_proposals)
        np.save(os.path.join(output_folder, "final_loglikes"),final_loglikes)
        # Save mean GP proposal plot
        plt.plot(np.mean(final_proposals, axis=0))
        plt.savefig(os.path.join(output_folder, "final_GP.png"))
        plt.close()

        # Plot and save figures in the output folder
        plot_alpha_components(final_proposals, alpha_true,Dom,output_folder, "alpha_components")
        plot_euler_angles(final_proposals, true_angles,Dom,output_folder, "euler_angles")
        plot_comparison_dipole_moments_from_ensemble(alpha_true, final_proposals, Dom, PW1,output_folder, "dipole_plot")
        plot_dipole_angle_discrepancy_from_ensemble(alpha_true, final_proposals, Dom, PW1,output_folder, "angle_plot")
#run_clean_pCN_analysis(runname="bigger_testing")

final_proposals=np.load("results/test_test/final_proposals.npy")
print(np.shape(final_proposals))