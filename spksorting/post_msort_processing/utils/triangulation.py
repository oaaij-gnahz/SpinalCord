"""
Monopolar triangulation of putative units

Code snippet from
https://github.com/SpikeInterface/spikeinterface/blob/master/spikeinterface/toolkit/postprocessing/unit_localization.py

Implementing method proposed in
https://www.biorxiv.org/content/10.1101/2021.11.05.467503v1

Modified 20220321 by jz103
"""

import numpy as np
import scipy
from scipy import optimize

def estimate_distance_error(vec, wf_ptp, local_contact_locations):
    # vec dims ar (x, y, z amplitude_factor)
    # given that for contact_location x=dim0 + z=dim1 and y is orthogonal to probe
    dist = np.sqrt(((local_contact_locations - vec[np.newaxis, :2])**2).sum(axis=1) + vec[2]**2)
    ptp_estimated = vec[3] / dist
    err = wf_ptp - ptp_estimated
    return err


def make_initial_guess_and_bounds(wf_ptp, local_contact_locations, max_distance_um):
    # constant for initial guess and bounds
    initial_z = 20

    ind_max = np.argmax(wf_ptp)
    max_ptp = wf_ptp[ind_max]
    max_alpha = max_ptp * max_distance_um

    # initial guess is the center of mass
    com = np.sum(wf_ptp[:, np.newaxis] * local_contact_locations, axis=0) / np.sum(wf_ptp)
    x0 = np.zeros(4, dtype='float32')
    x0[:2] = com
    x0[2] = initial_z
    initial_alpha = np.sqrt(np.sum((com - local_contact_locations[ind_max, :])**2) + initial_z**2) * max_ptp
    x0[3] = initial_alpha

    # bounds depend on initial guess
    bounds = ([x0[0] - max_distance_um, x0[1] - max_distance_um, 1, 0],
              [x0[0] + max_distance_um,  x0[1] + max_distance_um, max_distance_um*10, max_alpha])

    return x0, bounds

def compute_monopolar_triangulation(templates, chan_geom, radius_um=50, max_distance_um=1000, return_alpha=False):
    '''
    Localize unit with monopolar triangulation.
    This method is from Julien Boussard
    https://www.biorxiv.org/content/10.1101/2021.11.05.467503v1
    Important note about axis:
      * x/y are dimmension on the probe plane (dim0, dim1)
      * y is the depth by convention
      * z it the orthogonal axis to the probe plan
    Parameters
    ----------
    templates: (n_ch, n_samples, n_clus)
    chan_geom: (n_ch, 2)
    radius_um: float
        For channel sparsity
    max_distance_um: float
        to make bounddary in x, y, z and also for alpha
    Returns
    -------
    unit_location: np.array
        (n_clus, 3) or (n_clus, 4), {x, y, z, (alpha)}
        alpha is the amplitude at source estimation
    '''
    n_clus = templates.shape[2]
    unit_location = np.zeros((n_clus, 4), dtype='float64')
    for i in range(n_clus):

        # get neighborhood channels as a binary mask
        wf_ptp_allchs = templates[:,:,i].ptp(axis=1) # (n_ch,) : template peak-to-peak values
        prim_ch_ind = np.argmax(wf_ptp_allchs)
        prim_x, prim_y = chan_geom[prim_ch_ind, :]
        neighbor_mask = ( ((chan_geom[:,0]-prim_x)**2+(chan_geom[:,1]-prim_y)**2) < radius_um**2 )
        # chan_inds = np.where(neighbor_mask)[0]
        
        local_contact_locations = chan_geom[neighbor_mask, :]
        wf_ptp = wf_ptp_allchs[neighbor_mask]

        x0, bounds = make_initial_guess_and_bounds(wf_ptp, local_contact_locations, max_distance_um)

        # run optimization
        args = (wf_ptp, local_contact_locations)
        output = optimize.least_squares(estimate_distance_error, x0=x0, bounds=bounds, args = args)

        unit_location[i] = tuple(output['x'])

    if not return_alpha:
        unit_location = unit_location[:, :3]

    return unit_location


# if __name__=='__main__':
#     pass