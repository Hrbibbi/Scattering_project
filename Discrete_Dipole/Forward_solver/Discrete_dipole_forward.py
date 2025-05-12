import plane_wave_jit as PW
import Domains as DS

def compute_scattered_field_at_points(points,Incident_waves,domain):
    '''
    Computes the scattered fields from multiple Incident waves using the discrete dipole method

    Input:
        points: a Nx3 array with the points to evaluate the scattered field at
        Incident waves: Instance o plane_wave class storing information about the incident waves
        domain: Instance of domain class storing information about the position of dipoles, and related alpha_tensors

    Output:
        A 2xMxNx3 array with the with F[0,i,j,k]: k'th component of E^{scat} for incident wave i at point j 
                                      F[1,i,j,k]: '---------------' H^{scat} '----------------------------'                              
    '''
    
    #-----------------------------------------------------------------
    #  
    #-----------------------------------------------------------------

    #compute r_xi vector with distances between point located x=[x_x,x_y,x_z] and the dipole number i's position
    #resulting size is NxMx3, N points, M dipoles, 3 spatial components
    # Also in this calculation we get the scalar values |r_xi| which is a NxM array of the distances
    
    #------------------------

    #compute the polarization vectors P_{r,i}=\alpha_i E_r^{new}
    #resulting size is a RxMx3 array, R incident waves, M dipoles, 3 spatial components

    #-----------------------
    
    #compute scattered field E_r^{scat}(x)=\sum_{i=1}^M k_r^2/epsilon_0 * r_{xi} \times r_{xi} \times P_{r,i} e^{jk|r_{xi}|}/|r_{xi}|
    #resulting size is a 2xNxRx3 scattered field, N evalutation points, R different incident waves, 3 spatial components to each field


def scattered_field_measurement(Incident_waves,domain):
    """
    Computes the scattered field measurements to be used in the forward solver.
    """

