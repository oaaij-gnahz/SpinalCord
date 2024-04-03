import os 
import gc

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import scipy.signal as signal
import pandas as pd

from utils.read_mda import readmda


F_SAMPLE = 30e3
WINDOW_LEN_IN_SEC = 30e-3
SMOOTHING_SIZE = 11
session_folder = "/media/luanlab/Data_Processing/Jim-Zhang/SpinalCordSpikeSort/msort_out_gait/Corolla/Corolla_211206_001936/seg3"
PLOT_SCALE_Y = False

RESULT_PATH = os.path.join(session_folder, "firing_rate_by_clusters")
if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)


window_in_samples = int(WINDOW_LEN_IN_SEC*F_SAMPLE)

# read firing.mda
firings = readmda(os.path.join(session_folder, "firings.mda")).astype(np.int64)
filt_mda = readmda(os.path.join(session_folder, "filt.mda"))
n_samples = filt_mda.shape[1]
print("n_samples=",n_samples)
del(filt_mda)
gc.collect()
# get spike stamp for all clusters (in SAMPLEs not seconds)
spike_times_all = firings[1,:]
spike_labels = firings[2,:]
n_clus = np.max(spike_labels)
print(n_clus)
spike_times_by_clus =[[] for i in range(n_clus)]
spike_count_by_clus = np.zeros((n_clus,))
for spk_time, spk_lbl in zip(spike_times_all, spike_labels):
    spike_times_by_clus[spk_lbl-1].append(spk_time)
for i in range(n_clus):
    spike_times_by_clus[i] = np.array(spike_times_by_clus[i])
    spike_count_by_clus[i] = spike_times_by_clus[i].shape[0]

final_firing_timesample = firings[1, -1]

n_windows = int(np.ceil(n_samples/window_in_samples))

def single_cluster_firing_rate_series(firing_stamp, window_in_samples):
    n_windows = int(np.ceil(n_samples/window_in_samples))
    bin_edges = np.arange(0, window_in_samples*n_windows+1, step=window_in_samples)
    tmp_hist, _ = np.histogram(firing_stamp, bin_edges)
    tmp_hist = tmp_hist / WINDOW_LEN_IN_SEC
    smoother = signal.windows.hamming(SMOOTHING_SIZE)
    smoother = smoother / np.sum(smoother)
    firing_rate_series = signal.convolve(tmp_hist, smoother, mode='same')
    return firing_rate_series

for i_clus in range(n_clus):
    firing_stamp = spike_times_by_clus[i_clus]
    firing_rate_series = single_cluster_firing_rate_series(firing_stamp, window_in_samples)
    plt.figure(figsize=(6,2))
    plt.plot(np.arange(firing_rate_series.shape[0])*WINDOW_LEN_IN_SEC, firing_rate_series, color='k')
    plt.xlabel("Time (seconds)")
    plt.ylabel("Firing Rate (spikes/sec)")
    plt.title("cluster#%d"%(i_clus+1))
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, "cluster%d.svg"%(i_clus+1)))
    plt.close()
    # print(i_clus)
