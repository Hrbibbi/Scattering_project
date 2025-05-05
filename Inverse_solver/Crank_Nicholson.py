import GP_constructer
import numpy as np

# Placeholder for the forward solver, which should simulate measurements from a sample
def forward_solver_wrapper(sample,true_measurements):
    '''
    Forward solver wrapper, solves the forward problem for the specific sample

    Input:
        A sample from the specified Gaussian process
    
    Output:
        Simulated measurement data setup to be compatible with the true measurements
    '''
    # TODO: Implement the actual forward solver that takes a sample and returns simulated measurements
    raise NotImplementedError("forward_solver_wrapper is not yet implemented")

# Placeholder for GP-based sample generation
def generate_sample(GP_factorization):
    '''
    Function that generates a sample from the Gaussian process

    Input:
        The cholesky factorization of the covariance matrix for the GP
    
    Output:
        sample ready for for the forward solver wrapper
    '''
    # TODO: Implement sampling from the GP using the factorized covariance (e.g., Cholesky)
    raise NotImplementedError("generate_sample is not yet implemented")

def log_likelihood(sample, true_measurements, beta):
    '''
    Function that returns the log likelihood of a sample

    input:
        sample: sample from the GP
        true_measurements: data_frame containing the information for the true measurements
        beta: likelihood precision

    Output:
        The loglikelihood of the sample
    '''
    simulated_measurements = forward_solver_wrapper(sample)  # requires actual implementation
    normdiff = np.linalg.norm(true_measurements - simulated_measurements)**2
    loglike = -0.5 * beta * normdiff
    return loglike

def pCN(GP_factorization,true_measurements, stepsize, beta,num_iter,warmup_period,convergence_tol):
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

    proposal = generate_sample(GP_factorization)  # requires actual implementation
    L_old = log_likelihood(proposal, true_measurements, beta)

    accepted_proposals=[]
    
    for k in range(num_iter):
        #----------------------------------------------------
        # Generate new proposal and new likelihood
        #----------------------------------------------------
        
        new_sample = generate_sample(GP_factorization)  # requires actual implementation
        new_proposal = np.sqrt(1 - 2 * delta) * proposal + np.sqrt(2 * delta) * new_sample
        L_new = log_likelihood(new_proposal, true_measurements, beta)
        Ak = min(1, np.exp(L_new - L_old))
        uk = np.random.uniform()

        #----------------------------------------------------
        # Update step
        #----------------------------------------------------
        if uk < Ak:
            #only store proposals after warmup period
            if k>num_iter*warmup_period:
                #Check if we are converged
                if np.abs(L_old-L_new)<convergence_tol:
                    break

                accepted_proposals.append(proposal)
            proposal = new_proposal
            L_old = L_new  # update log-likelihood if proposal is accepted

    return accepted_proposals  # return accepted_proposals
