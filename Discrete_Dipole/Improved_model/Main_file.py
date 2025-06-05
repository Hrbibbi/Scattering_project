import pCN as pCN
import numpy as np
import Forward_model as FW
import matplotlib.pyplot as plt
import plotting_tools as plotting

def simple_test():
    measure_direction=np.array([0,0,1])
    domain_points = FW.protusion_gen(50,100,28)
    N,_=np.shape(domain_points)
    light_direction=np.array([0,0,-1])
    omegas=np.linspace(1,1000,200)
    alpha_tensor=np.tile([[1,0,0],[0,0,1],[0,1,0]],(N,1,1))
    prop_vec=np.array([0,0,-1])
    numangles=10
    true_measurements=FW.poynting_far_field_unpolarized_light(measure_direction,alpha_tensor,domain_points,prop_vec,1,1,omegas,numangles)
    experiment=pCN.pCN_module(measure_direction,domain_points,light_direction,1,1,omegas,true_measurements,
                              [200,200,200],1,1,1.5)
    experiment.pCN_explore("pCN-testing",numangles=numangles,beta=100,num_iter=1500,warup_period=0)
    index=np.load("pCN-testing/top_likelihoods_iter_1000.npy").argmin()
    proposal=np.load("pCN-testing/top_proposals_iter_1000.npy")[index]
    print("refinement")
    experiment.pCN_refine("pCN-testing","starting_point",proposal,beta=1e4)
    refine=np.mean(np.load("pCN-testing/starting_point_proposals.npy"),axis=0)
    angles=experiment.gp_to_euler_angles(refine)
    alpha_tensor_prop=experiment.euler_angles_to_alpha_tensor(angles)
    #measurement=FW.poynting_far_field_unpolarized_light(measure_direction,alpha_tensor_prop,domain_points,prop_vec,1,1,omegas,numangles)
    plotting.plot_dipole_orientations(domain_points,alpha_tensor,"pCN-testing","true")
    plotting.plot_dipole_orientations(domain_points,alpha_tensor_prop,"pCN-testing","proposal")
simple_test()

