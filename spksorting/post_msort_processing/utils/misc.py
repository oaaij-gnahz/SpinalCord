import numpy as np
import os
import matplotlib

def binwidth2binedges(binwidth: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    nbins = int(np.ceil((fmax-fmin)/binwidth))
    return fmin + np.arange(nbins+1)*binwidth


def construct_firings_from_spktrains(spk_trains, prim_channels=None):
    """ Construct MountainSort-like firings from list of spike trains
    spk_trains: a list of ndarray. The list has size n_clus. Each ndarray has variable size (n_spikes,)
    prim_channels: a (n_clus,) array; elements are in range [1,n_ch]. prim_channels[0] denote the primary channel of cluster 1 (first cluster)
    """
    event_times_all = np.concatenate(spk_trains)
    labels_all = np.concatenate([np.ones(spktrain.shape[0], dtype=int)*int(i_st+1) for (i_st, spktrain) in enumerate(spk_trains)])
    idx_sorted = np.argsort(event_times_all)
    event_times_all = event_times_all[idx_sorted]
    labels_all = labels_all[idx_sorted]
    if prim_channels is None:
        prim_channels_all = np.ones(labels_all.shape) 
    else:
        prim_channels_all = prim_channels[labels_all-1]
    mda_array = np.stack([prim_channels_all, event_times_all, labels_all], axis=0).astype(np.int64)
    print("Shape of constructed firings:", mda_array.shape)
    return mda_array

def recursively_empty_dir(dirpath):
    path_dirs = []
    path_files = []
    for rootpath, dirs, files in os.walk(dirpath):
        for name in files:
            path_files.append(os.path.join(rootpath, name))
        for name in dirs:
            path_dirs.append(os.path.join(rootpath, name))
    path_dirs = sorted(path_dirs, key=lambda x: len(x))[::-1]
    for path_file in path_files:
        os.remove(path_file)
    for path_dir in path_dirs:
        os.rmdir(path_dir)

def plt3d_set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
    ax: a matplotlib axis, e.g., as output from plt.gca().
    https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])