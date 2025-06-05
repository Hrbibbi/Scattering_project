import Forward_model as FW
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
from sklearn.gaussian_process.kernels import Matern 
import numpy as np
import os
import time
class pCN_module:
    def __init__(self,measure_direction,domain_points,light_direction,epsilon,mu,omegas,true_measurements,
                 control_size,k_nearest,length_scale,nu):
        """
        pCN method for inverse problem

        Experiment parameters
        ----------
        measure_direction : array-like, shape (3,)
            Direction along which measurements are taken.

        domain_points : array-like, shape (M, 3)
            All points in the domain (e.g., ellipsoid voxel centers), voxel spacing ~7.

        light_direction : array-like, shape (3,)
            Propagation vector for the incident light.

        epsilon : float
            Permittivity of the medium.

        mu : float
            Permeability of the medium.

        omegas : array-like, shape (R,)
            Angular frequencies for plane waves.

        true_measurements : array-like
            Observed measurements (used in Metropolis step, etc.).
        
        inverse parameters
        ----------
        control_spacing : float, default=14.0
            Spacing between control points in x, y, z. Should be >= domain voxel size.

        k_nearest : int, default=3
            Number of nearest control points used to interpolate each domain point.

        length_scale : float, default=1.0
            Length scale for the Matern kernel on control points.

        nu : float, default=1.5
            Smoothness parameter for the Matern kernel.
        """
        self.measure_direction=measure_direction
        self.domain_points=domain_points
        self.prop_vec=light_direction
        self.epsilon=epsilon
        self.mu=mu
        self.omegas=omegas
        self.true_measurements=true_measurements

        mins = np.min(self.domain_points, axis=0)
        maxs = np.max(self.domain_points, axis=0)
        self.box_min = mins
        self.box_max = maxs
        cX,cY,cZ=control_size
        nX,nY,nZ=int(np.ceil((maxs[0]-mins[0])/cX)),int(np.ceil((maxs[1]-mins[1])/cY)),int(np.ceil((maxs[2]-mins[2])/cZ))
        x_coords = np.linspace(mins[0], maxs[0], nX)
        y_coords = np.linspace(mins[1], maxs[1], nY)
        z_coords = np.linspace(mins[2], maxs[2], nZ)
        Xc, Yc, Zc = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        self.control_points = np.vstack([Xc.ravel(), Yc.ravel(), Zc.ravel()]).T

        self.n_controls = self.control_points.shape[0]
        self.k_nearest = k_nearest

        # 3) Build KD-tree on control points for interpolation
        self.tree = cKDTree(self.control_points)
        print(f"num control_points: {np.shape(self.control_points)[0]}")
        #print(self.control_points)
        # 4) Precompute Cholesky factor for Matern covariance on control points
        self.chol_fact = self.matern_covariance_matrix(self.control_points, length_scale=length_scale, nu=nu)
    def matern_covariance_matrix(self,points,length_scale=1,nu=1.5):
        '''
        returns the cholesky factorization for the matern kernel to sample
        '''
        kernel=Matern(length_scale=length_scale,nu=nu)
        K=kernel(points)
        chol_fact = np.linalg.cholesky(K)
        return chol_fact
    def sample_gp_field(self):
        '''
        returns a sample of the gp_field
        '''
        M= self.chol_fact.shape[0]
        xi=np.random.randn(M,3)
        return self.chol_fact @ xi
    
    def gp_to_euler_angles(self, gp_control_samples, a=1.0, b=1.0):
        """
        Map GP samples at control points to (yaw, pitch, roll) for each domain point.

        Uses k-NN on the control grid and circular mean over k_nearest neighbors.
        """
        def sigmoid(x, scale):
            return 1.0 / (1.0 + np.exp(-scale * x))

        # Map control GP to angles
        theta_ctrl = 2 * np.pi * sigmoid(gp_control_samples[:, 0], a) - np.pi
        phi_ctrl   = np.pi * sigmoid(gp_control_samples[:, 1], b) - (np.pi / 2)
        psi_ctrl   = np.pi * sigmoid(gp_control_samples[:, 2], b) - (np.pi / 2)
        control_angles = np.column_stack([theta_ctrl, phi_ctrl, psi_ctrl])  # (n_ctrl, 3)

        # For each domain point, find k nearest control indices
        _, idxs = self.tree.query(self.domain_points, k=self.k_nearest)

        M = self.domain_points.shape[0]
        domain_angles = np.zeros((M, 3), dtype=float)

        # Circular average per component
        for i in range(M):
            neighbors = idxs[i]
            for comp in range(3):
                vals = control_angles[neighbors, comp]
                cpx = np.exp(1j * vals)
                domain_angles[i, comp] = np.angle(np.mean(cpx))

        return domain_angles
    def euler_angles_to_alpha_tensor(self,euler_angles):
        alpha_tensor=Rot.from_euler('ZYX',euler_angles,degrees=False).as_matrix()
        return alpha_tensor
    def log_likelihood(self,sample,numangles,beta):
        angles=self.gp_to_euler_angles(sample)
        alpha_tensor=self.euler_angles_to_alpha_tensor(angles)
        
        sim=FW.poynting_far_field_unpolarized_light(self.measure_direction,alpha_tensor,self.domain_points,
                                                self.prop_vec,self.epsilon,self.mu,self.omegas,numangles)
        import matplotlib.pyplot as plt
        loglike=-0.5*beta*np.linalg.norm(self.true_measurements-sim)
        return loglike
    def pCN_explore(self,output_folder,delta=0.3,beta=10,num_iter=20000,warup_period=0.5,
                    top_index=10,numangles=2):
        
        block=[]
        block_likelihood=[]
        os.makedirs(output_folder, exist_ok=True)
        iter_start=time.time()
        proposal=self.sample_gp_field()

        L_old=self.log_likelihood(proposal,numangles,beta)
        num_acceptance=0
        for k in range(num_iter+1):
            new_sample=self.sample_gp_field()
            new_proposal=np.sqrt(1 - 2 * delta) * proposal + np.sqrt(2 * delta) * new_sample
            L_new=self.log_likelihood(new_proposal,numangles,beta)
            Ak=min(1,np.exp(L_new-L_old))
            if np.random.uniform() < Ak:
                num_acceptance+=1
                proposal = new_proposal
                L_old = L_new

                if k>= int(num_iter*warup_period):
                    block.append(proposal.copy())
                    block_likelihood.append(L_old)
            if (k % 1000)==0 and k>=int(num_iter*warup_period):
                idx_sorted = np.argsort(block_likelihood)[-top_index:][::-1]
                top_proposals = [block[i] for i in idx_sorted]
                top_likelihoods = [block_likelihood[i] for i in idx_sorted]

                # Save to disk
                np.save(os.path.join(output_folder, f"top_proposals_iter_{k}.npy"), top_proposals)
                np.save(os.path.join(output_folder, f"top_likelihoods_iter_{k}.npy"), top_likelihoods)
                block=[]
                block_likelihood=[]
        
            if k % 100 == 0:
                print(f"Iteration: {k}, Time: {time.time() - iter_start:.2f}s, Log-like: {L_old:.3f}, Accepted: {num_acceptance}, Beta: {beta}")
                print(len(block))
                iter_start=time.time()
                num_acceptance = 0
    def pCN_refine(self,output_folder,output_name,start,warup_period=0.5,delta=0.01,beta=1000,num_iter=5000,top_index=10,numangles=2):
        os.makedirs(output_folder, exist_ok=True)
        block=[]
        block_likelihood=[]
            
        iter_start=time.time()
        proposal=start
        L_old=self.log_likelihood(proposal,numangles,beta)
        num_acceptance=0
        for k in range(num_iter+1):
            new_sample=self.sample_gp_field()
            new_proposal=np.sqrt(1 - 2 * delta) * proposal + np.sqrt(2 * delta) * new_sample
            L_new=self.log_likelihood(new_proposal,numangles,beta)
            Ak=min(1,np.exp(L_new-L_old))
            if np.random.uniform() < Ak:
                num_acceptance+=1
                proposal = new_proposal
                L_old = L_new

                if k>= int(num_iter*warup_period):
                    block.append(proposal.copy())
                    block_likelihood.append(L_old)
            
            if k % 100 == 0:
                print(f"Iteration: {k}, Time: {time.time() - iter_start:.2f}s, Log-like: {L_old:.3f}, Accepted: {num_acceptance}, Beta: {beta}")
                print(len(block))
                iter_start=time.time()
                num_acceptance = 0
        idx_sorted = np.argsort(block_likelihood)[-top_index:][::-1]
        top_proposals = [block[i] for i in idx_sorted]
        top_likelihoods = [block_likelihood[i] for i in idx_sorted]

        # Save to disk
        np.save(os.path.join(output_folder, f"{output_name}_proposals.npy"), top_proposals)
        np.save(os.path.join(output_folder, f"{output_name}_likelihoods.npy"), top_likelihoods)