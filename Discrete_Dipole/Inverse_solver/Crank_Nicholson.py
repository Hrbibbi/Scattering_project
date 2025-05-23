import Domains
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

def forward_solver_setup(true_measurements: pd.DataFrame,
                         surface_data: np.ndarray,
                         surface_size: float
                         ):
                         
    '''
    Forward solver setup: creates necessary scattering and incident info plus measurement arrays.

    Parameters
    ----------
    true_measurements : pd.DataFrame
        Must have columns ['wavelength', 'propagation_vector', 'polarization', 'measurement_value']
    surface_data : np.ndarray
        NxN height map of the surface.
    surface_size : float
        Physical size of the NxN surface (meters).

    Returns
    -------
    Scatter_information : dict
        Contains 'Surface', 'inneraux', 'outeraux', 'epsilon'(placeholder), 'mu'.
    incident_configs : list of dict
        Each with keys ['lambda','omega','propagation_vectors','polarizations','epsilon','mu'].
    measurement_values : np.ndarray
        1D array of all measurement_value entries concatenated across wavelengths, matching the order of incidents.
    '''
    # Validate DataFrame
    if not isinstance(true_measurements, pd.DataFrame):
        raise TypeError("true_measurements must be a pandas DataFrame")
    required_cols = {'wavelength','propagation_vector','polarization','measurement_value'}
    if not required_cols.issubset(true_measurements.columns):
        raise ValueError(f"true_measurements must contain columns: {required_cols}")

    # Surface geometry
    Z = surface_data
    N = Z.shape[0]
    x = np.linspace(-surface_size/2, surface_size/2, N)
    y = x
    stepsize = surface_size / N
    X,Y=np.meshgrid(x,y)
    # Compute point clouds and offsets
    point_cloud, tau1, tau2, normals, mean_curvature = C2_surface.compute_geometric_data(X,Y,Z, stepsize)
    inner_cloud = C2_surface.generate_curvature_scaled_offset(point_cloud, normals, mean_curvature, -0.86)
    outer_cloud = C2_surface.generate_curvature_scaled_offset(point_cloud, normals, mean_curvature,  0.86)

    Surface  = C2_surface.C2_surface(point_cloud, normals, tau1, tau2)
    inneraux = C2_surface.C2_surface(inner_cloud, normals, tau1, tau2)
    outeraux = C2_surface.C2_surface(outer_cloud, normals, tau1, tau2)

    Scatter_information = {
        'Surface': Surface,
        'inneraux': inneraux,
        'outeraux': outeraux,
        'epsilon': None,      # placeholder: to be set per-sample
        'mu': 1.0
    }

    # Build incident configurations and collect measurement values
    incident_configs = []
    meas_list = []
    for lam, group in true_measurements.groupby('wavelength'):
        prop_vecs_str = np.vstack(group['propagation_vector'].values)
        N=np.shape(prop_vecs_str)[0]
        prop_vecs = np.zeros( (N,3))
        for i in range(len(prop_vecs)):
            prop_vecs[i]= np.fromstring( prop_vecs_str[i][0][1:-1], dtype=float, sep=" " )
        pols_str     = np.vstack(group['polarization'].values)
        pols = np.zeros(N)
        for i in range(len(pols_str)):
            pols[i]=pols_str[i,:]
        meas     = group['measurement_value'].to_numpy()
        omega = 2 * np.pi / lam

        incident_configs.append({
            'lambda': lam,
            'omega': omega,
            'propagation_vectors': prop_vecs,
            'polarizations': pols,
            'epsilon': 1.0,
            'mu': 1.0
        })
        meas_list.append(meas)
        measurement_values = np.concatenate(meas_list)
    return Scatter_information, incident_configs, measurement_values

# Placeholder for GP-based sample generation
def generate_sample(GP_factorization):
    '''
    Function that generates a sample from the Gaussian process

    Input:
        The cholesky factorization of the covariance matrix for the GP
    
    Output:
        sample ready for for the forward solver wrapper
    '''
    M,N=np.shape(GP_factorization)
    Normal_sample=np.random.normal(size=N)
    
    return GP_factorization @ Normal_sample

def generate_tensor_sample(GP_factorization):
    '''
    Generate a sample of a symmetric 3x3 tensor field from a Gaussian Process.

    Input:
        GP_factorization: (N, N) Cholesky factor of the covariance matrix
    
    Output:
        alpha: (N, 3, 3) array representing the full tensor at each of N points
    '''
    N = GP_factorization.shape[0]

    # Sample 9 independent GP coefficient vectors (one for each tensor entry)
    alpha = np.empty((N, 3, 3), dtype=np.float64)

    for i in range(3):
        for j in range(3):
            normal_sample = np.random.normal(size=N)
            alpha[:, i, j] = GP_factorization @ normal_sample
    
    return alpha

def log_likelihood(sample,Scatter_information,Incident_configs,measurement_values, beta):
    '''
    Function that returns the log likelihood of a sample

    input:
        sample: sample from the GP
        true_measurements: data_frame containing the information for the true measurements
        beta: likelihood precision

    Output:
        The loglikelihood of the sample
    '''
    Scatter_information['epsilon']=sample
    options = {
    'show_MAS'         : False,
    'plane_location'   : None,   # auto = 5×max height
    'Show_power_curve' : False,
    'print_solution_times': False
    }
    simulated_measurements = Forward_solver_fast.Single_scatter_solver(Scatter_information,Incident_configs,options)
    normdiff = np.linalg.norm(measurement_values - simulated_measurements)**2
    loglike = -0.5 * beta * normdiff
    return loglike

def pCN(GP_factorization,Scatter_information,Incident_configs,measurement_values, stepsize, beta,num_iter,warmup_period,convergence_tol):
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

    proposal = generate_sample(GP_factorization)+4  # requires actual implementation
    L_old = log_likelihood(proposal,Scatter_information,Incident_configs,measurement_values, beta)
    accepted_proposals=[]
    
    for k in range(num_iter):
        iter_start=time.time()
        #----------------------------------------------------
        # Generate new proposal and new likelihood
        #----------------------------------------------------

        new_sample = generate_sample(GP_factorization)  # requires actual implementation
        new_proposal = np.sqrt(1 - 2 * delta) * proposal + np.sqrt(2 * delta) * new_sample
        #print(f"old proposal: {proposal}, new proposal {new_proposal}")
        #print(f"new propsal {new_proposal}")

        L_new = log_likelihood(new_proposal,Scatter_information,Incident_configs,measurement_values, beta)
        #print(f"old likelihood: {L_old}, new likelihood {L_new}")
        Ak = min(1, np.exp(L_new - L_old))
        uk = np.random.uniform()
        #print(f"Ak: {Ak}, uk {uk}")
        #----------------------------------------------------
        # Update step
        #----------------------------------------------------
        if uk < Ak:
            #only store proposals after warmup period
            if k>num_iter*warmup_period:
                #print("here")
                accepted_proposals.append(proposal)
                #Check if we are converged
                if np.abs(L_old-L_new)<convergence_tol:
                    break  
            proposal = new_proposal
            L_old = L_new 
            #print(f"updated proposal {proposal}")
            #print(f"accepted propsals {accepted_proposals}")
        print(f"Iteration: {k}, Iteration time: {time.time()-iter_start}")
        if k%10 == 0:
            print(f"current proposal {proposal}")
    return accepted_proposals  # return accepted_proposals

X,Y,Z=Domains.cylinder_cartesian_grid(1,1)
chol_fact=Domains.matern_covariance_matrix(X,Y,Z)
alpha=generate_tensor_sample(chol_fact)
print(np.shape(alpha))

def testing():
    width=1
    resol=100
    a,b=-width,width
    X0=np.linspace(a,b,resol)
    Y0=np.linspace(a,b,resol)
    X,Y=np.meshgrid(X0,Y0)
    bump = lambda x,y,x0,y0,height,sigma: height*np.exp(
        -( (x-x0)**2 + (y-y0)**2 ) / (2*sigma**2)
    )
    f = lambda x,y: (
                    bump(x,y,-0.20073580984422001,0.7211428902558659,0.31959818254342154,0.49932924209851826)+
                    bump(x,y,-0.5503701752921016,-0.5504087674620758,0.11742508365045984,0.6330880728874675) +
                    bump(x,y,0.16178401878913407,0.3329161244736727,0.10617534828874074,0.6849549260809971) 
                        )
    Z=f(X,Y)
    df=pd.read_csv("true_testing.csv")
    Scatter_information, incident_configs, measurement_values=forward_solver_setup(df,Z,1)
    '''
    epsilons=np.linspace(0.5,4,30)
    L=[]
    normdiffs=[]
    for i,epsilon in enumerate(epsilons):
        print(i)
        l,normdiff=log_likelihood(epsilon,Scatter_information,incident_configs,measurement_values,100)
        L.append(l)
        normdiffs.append(normdiff)
    plt.plot(epsilons,L)
    plt.title("loglikelihood")
    plt.show()
    plt.plot(epsilons,normdiffs)
    plt.title("normdiff")
    plt.show()
    '''
    aP=pCN(1,Scatter_information,incident_configs,measurement_values,stepsize=0.05,beta=1e6,num_iter=500,convergence_tol=1e-8,warmup_period=0.5)
    plt.hist(aP, bins=30, edgecolor='black')
    plt.title("Histogram of accepted permittivity proposals")
    plt.xlabel("Permittivity")
    plt.ylabel("Frequency")
    plt.show()
    print(np.mean(aP))


