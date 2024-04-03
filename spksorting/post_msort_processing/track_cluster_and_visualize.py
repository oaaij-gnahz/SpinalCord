import os
from time import time
from copy import deepcopy
import gc
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
from natsort import natsorted

from utils.read_mda import readmda

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['font.size']=20

# TODO: read accepted clusters; plot them in the same cluster and link with lines
F_SAMPLE = 30000
T_BEG = 800
T_END = 860
# MAP_PATH="../geom_channel_maps/map_corolla24ch.csv"
# geom = pd.read_csv(MAP_PATH, header=None).values
def get_pairwise_L2_distance_square_matrix(data_a, data_b):
    """
    assumes data_a is (n_1, n_features) and data_b is (n_2, n_features)
    returns (n_1, n_2) distance matrix
    where n_1 and n_2 could be cluster counts from 2 sessions
    """
    return np.sum((data_a[:,None,:]-data_b[None,:,:])**2, axis=2) / data_a.shape[1]

cont_root_path = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/MustangContinuous/Mustang_220126_125248/dt375Segged"
session_folders = ["%s/seg%d"%(cont_root_path, i+1) for i in range(6)]
# session_folders = ["%s/%s"%(cont_root_path, k) for k in natsorted(os.listdir(cont_root_path))]
store_path = "%s/cluster_track20200310" % (cont_root_path)
if not os.path.exists(store_path):
    os.makedirs(store_path)
data = []
fdata = []
single_masks = []
for i in range(len(session_folders)):
    data1 = readmda(os.path.join(session_folders[i], "templates.mda")).astype(np.float64)
    n_ch = data1.shape[0] # should stay constant across all segs
    n_sample = data1.shape[1] # should stay constant across all segs
    fdata1 = data1.reshape((n_ch*n_sample, data1.shape[2])).T
    data.append(data1) # data1 is of shape (n_ch, n_samples, n_clus)
    fdata.append(fdata1) # fdata is of shape (n_clus, n_ch*n_samples)
    # print(data1.shape)
    tmp_mask = np.load(os.path.join(session_folders[i], "cluster_rejection_mask.npz"))
    # unit_mask = np.logical_or(tmp_mask['single_unit_mask'], tmp_mask['multi_unit_mask'])
    unit_mask = tmp_mask['single_unit_mask']
    single_masks.append(unit_mask)
    
    # print(single_masks[-1].shape)
    
    # raster plot
    PLOT_RASTER = False
    if PLOT_RASTER:
        firings = readmda(os.path.join(session_folders[i], "firings.mda")).astype(int)
        # extract one small time chunk (1 minute)
        chunkstamp = firings[:, np.logical_and(firings[1,:]>T_BEG*F_SAMPLE, firings[1,:]<T_END*F_SAMPLE)]
        # print(chunkstamp.shape)    # keep only stamps of accepted units
        chunk_mask = unit_mask[(chunkstamp[2,:]-1).astype(int)]
        chunkstamp = chunkstamp[:, chunk_mask]
        # re-order by #unit
        n_clus = unit_mask.shape[0]
        spike_times_by_clus =[[] for i in range(n_clus)]
        for spk_time, spk_lbl in zip(chunkstamp[1,:], chunkstamp[2,:]):
            spike_times_by_clus[spk_lbl-1].append(spk_time-1)
        for i_this_clus in range(n_clus):
            spike_times_by_clus[i_this_clus] = np.array(spike_times_by_clus[i_this_clus])
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cnt=0
        for i_this_clus, stamp in enumerate(spike_times_by_clus):
            if unit_mask[i_this_clus]==False:
                continue
            ax.scatter(stamp/F_SAMPLE-T_BEG, -cnt*np.ones(stamp.shape[0]), color='k', s=0.01)
            cnt += 1
        ax.set_xlim(0, T_END-T_BEG)
        ax.axis("off")
        plt.savefig("tmp_Mustang_cont_rasters/seg%d.png"%(i))
        plt.savefig("tmp_Mustang_cont_rasters/seg%d.svg"%(i))
# exit(0)


ts = time()

# initialize cluster tracking marker
# NOTE do not deal with cases of one cluster having multiple nearest neighbors
clus_track_strs = ["%d"%(i+1) for i in range(fdata[0].shape[0])]
# print(clus_track_strs)
clus_track_l2dists = [[] for _ in range(fdata[0].shape[0])]
for i_seg in range(1, len(session_folders)):
    print("i_seg=",i_seg)
    clus_new_strs = ["%d"%(i+1) for i in range(fdata[i_seg].shape[0])]
    # n_clus_new = min(len(clus_track_strs, clus_new_strs))
    l2dist_square_mat = get_pairwise_L2_distance_square_matrix(fdata[i_seg-1], fdata[i_seg])
    # single_mask_square = np.ones(l2dist_square_mat.shape, dtype=bool)
    # single_mask_square[single_masks[i_seg-1], :] = False
    # single_mask_square[:, single_masks[i_seg]] = False
    # l2dist_square_mat = np.ma.array(l2dist_square_mat, mask=single_mask_square)
    clus_track_strs_new = []
    clus_track_l2dists_new = []
    # if len(clus_track_strs) <= len(clus_new_strs):
    #     print("len(clus_track_strs) <= len(clus_new_strs)")
    # else:
    #     print("len(clus_track_strs) > len(clus_new_strs)")
    if len(clus_track_strs) <= len(clus_new_strs):
        for i_clus_track, (clus_track_str, clus_track_l2dist_list) in enumerate(zip(clus_track_strs, clus_track_l2dists)):
            # get current cluster index
            this_clus_ind = int(clus_track_str.split("-")[-1])-1
            # check if it is accepted single unit
            # if single_masks[i_seg][this_clus_ind]==False:
            #     continue
            # find nearest neibor
            candidate_new_ind = np.argmin(l2dist_square_mat[this_clus_ind, :])
            # check if the two are mutual nearest neighbors
            if np.argmin(l2dist_square_mat[:, candidate_new_ind])==this_clus_ind:
                # print("haha")
                clus_track_str_new = "-".join([clus_track_str, "%d"%(candidate_new_ind+1)])
                clus_track_l2dist_list_new = deepcopy(clus_track_l2dist_list)
                clus_track_l2dist_list_new.append(l2dist_square_mat[this_clus_ind, candidate_new_ind])
                # update cluster tracking chain
                clus_track_strs_new.append(clus_track_str_new)
                clus_track_l2dists_new.append(clus_track_l2dist_list_new)
    else:
        for i_clus_new, clus_new_str in enumerate(clus_new_strs):
            # get current cluster index
            this_clus_ind = int(clus_new_str.split("-")[-1])-1
            # if single_masks[i_seg][this_clus_ind]==False:
            #     continue
            # find nearest neibor
            candidate_track_ind = np.argmin(l2dist_square_mat[:, this_clus_ind])
            # check if the two are mutual nearest neighbors
            if np.argmin(l2dist_square_mat[candidate_track_ind, :])==this_clus_ind:
                # find the corresponding tracking string
                clus_track_str = list(filter(lambda x: int(x.split('-')[-1])-1==candidate_track_ind, clus_track_strs))
                assert len(clus_track_str)==1
                clus_track_str = clus_track_str[0]
                clus_track_str_idx = clus_track_strs.index(clus_track_str)
                # create new cluster tracking chain element to replace the old one
                clus_track_l2dist_list_new = deepcopy(clus_track_l2dists[clus_track_str_idx])
                clus_track_l2dist_list_new.append(l2dist_square_mat[candidate_track_ind, this_clus_ind])
                clus_track_str_new = "-".join([clus_track_str, "%d"%(this_clus_ind+1)])
                # update cluster tracking chain
                clus_track_strs_new.append(clus_track_str_new)
                clus_track_l2dists_new.append(clus_track_l2dist_list_new)
    clus_track_strs = deepcopy(clus_track_strs_new)
    clus_track_l2dists = deepcopy(clus_track_l2dists_new)
    gc.collect()
    # for (clus_track_str, clus_track_l2dist) in zip(clus_track_strs, clus_track_l2dists):
    #     print(clus_track_str)
    #     print(clus_track_l2dist)
    #     print('-------')

# read clus_locations and clus_rejection_mask
clus_coordinates = []
firing_rates = []
peak_amplitudes =[]
for i in range(len(session_folders)):
    seg_locs = pd.read_csv(os.path.join(session_folders[i], "clus_locations.csv"), header=None).values
    clus_coordinates.append(seg_locs)
    with open(os.path.join(session_folders[i], "combine_metrics_new.json"), 'r') as f:
        x = json.load(f)
    clus_metrics_list = x['clusters']
    firing_rates.append(np.array([k['metrics']['firing_rate'] for k in clus_metrics_list]))
    peak_amplitudes.append(np.array([k['metrics']['peak_amplitude'] for k in clus_metrics_list]))

MAP_PATH="../geom_channel_maps/map.csv"
geom = pd.read_csv(MAP_PATH, header=None).values
ELECTRODE_RADIUS = 12.5
SEG_SPACING = 140

# VIZUALIZATION CODE
VIZ=False
if VIZ:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # first plot n_seg layouts
    for i_seg in range(len(session_folders)):
        n_clus = firing_rates[i_seg].shape[0]
        # get the ranking of clusters by peak amplitude
        peak_amplitudes_argsort = np.argsort(peak_amplitudes[i_seg])
        peak_amplitude_ranks = np.zeros(n_clus)
        peak_amplitude_ranks[peak_amplitudes_argsort] = np.arange(n_clus) # rank from low to high
        peak_amplitude_ranks = peak_amplitude_ranks.astype(int)

        smap = np.logspace(np.log10(2), np.log10(40), num=n_clus) * 10
        # layout
        for i in range(geom.shape[0]):
            ax.add_patch(plt.Circle((geom[i,0]+i_seg*SEG_SPACING, geom[i,1]), ELECTRODE_RADIUS, edgecolor='k', fill=False))
        ax.add_patch(plt.Rectangle((i_seg*SEG_SPACING-15, -20), 60, 700, edgecolor='k', fill=False))
        # clusters
        ax.scatter(\
            clus_coordinates[i_seg][single_masks[i_seg]][:, 0]+i_seg*SEG_SPACING, clus_coordinates[i_seg][single_masks[i_seg]][:, 1], \
            marker='.', \
            # c=firing_rates[i_seg][single_masks[i_seg]], \
            #cmap="seismic", vmin=firing_rates[i_seg][single_masks[i_seg]].min(), vmax=firing_rates[i_seg][single_masks[i_seg]].max(), \
            c=peak_amplitude_ranks[single_masks[i_seg]],\
            cmap="viridis", 
            s=smap[peak_amplitude_ranks[single_masks[i_seg]]], alpha=.4\
        )
        print("XXX%dXXX"%(np.sum(single_masks[i_seg])))
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    plt.savefig("tmp.png")
    colors = [(0,1,1), (1,1,.5), (1, .5, .5), (0, 0, 0), (.5, .5, 1), (1,0,1), (1, .5, 1)]
    for i_track, (clus_track_str, clus_track_l2dist) in enumerate(zip(clus_track_strs, clus_track_l2dists)):
        print(clus_track_str)
        clus_inds = [int(indclus)-1 for indclus in clus_track_str.split("-")]
        for i_seg, (clus_ind0, clus_ind1) in enumerate(zip(clus_inds[:-1], clus_inds[1:])):
            if single_masks[i_seg][clus_ind0] and single_masks[i_seg+1][clus_ind1]:
                plt_x0 = clus_coordinates[i_seg][clus_ind0, 0]+i_seg*SEG_SPACING
                plt_x1 = clus_coordinates[i_seg+1][clus_ind1, 0]+(i_seg+1)*SEG_SPACING
                plt_y0 = clus_coordinates[i_seg][clus_ind0, 1]
                plt_y1 = clus_coordinates[i_seg+1][clus_ind1, 1]
                ax.plot([plt_x0, plt_x1], [plt_y0, plt_y1], color=colors[i_track%len(colors)], alpha=0.8, linewidth=0.6)
    plt.savefig("tmp1.png")
    plt.savefig("tmp1.svg")


drifts = []
for i_track, (clus_track_str, clus_track_l2dist) in enumerate(zip(clus_track_strs, clus_track_l2dists)):
    print(clus_track_str)
    clus_inds = [int(indclus)-1 for indclus in clus_track_str.split("-")]
    for i_seg, (clus_ind0, clus_ind1) in enumerate(zip(clus_inds[:-1], clus_inds[1:])):
        if single_masks[i_seg][clus_ind0] and single_masks[i_seg+1][clus_ind1]:
            plt_y0 = clus_coordinates[i_seg][clus_ind0, 1]
            plt_y1 = clus_coordinates[i_seg+1][clus_ind1, 1]
            drifts.append((plt_y0-plt_y1)/4) # um/hr
print(len(drifts))
hist_drift, binedges = np.histogram(drifts, bins=30)
x_scale = np.max(np.abs(binedges))
binwidth = binedges[1]-binedges[0]
plt.figure(); plt.bar((binedges[1:]+binedges[:-1])/2, hist_drift, width=binwidth*.95, color='k'); 
plt.xlim([-x_scale-binwidth/2, x_scale+binwidth/2])
plt.xlabel("Drift (um/hr)"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig("tmp00.png"); plt.savefig("tmp00.svg"); plt.show()




#     # print(clus_track_l2dist)
#     print('-------')
#     for i_seg in range(0, len(session_folders)):
#         tmp_figpath = "%s/figs_allclus_waveforms_new/waveform_clus%d.png"%(session_folders[i_seg], clus_inds[i_seg])
#         if not os.path.exists(tmp_figpath):
#             tmp_figpath = "%s/figs_allclus_waveforms_new/z_multiunit_waveform_clus%d.png"%(session_folders[i_seg], clus_inds[i_seg])
#         if not os.path.exists(tmp_figpath):
#             tmp_figpath = "%s/figs_allclus_waveforms_new/z_failed_waveform_clus%d.png"%(session_folders[i_seg], clus_inds[i_seg])
#         # tmp_figpath = "%s/track_clus/clus%d.svg"%(session_folders[i_seg], clus_inds[i_seg])
#         new_figpath = os.path.join(track_folder, "seg%d_clus%d.png"%(i_seg+1, clus_inds[i_seg]))
#         cmd_str = "cp %s %s" % (tmp_figpath, new_figpath)
#         print("Execute:", cmd_str)
#         os.system(cmd_str)




# print(time()-ts)

# inds_ranked = np.argsort(l2dist_square_mat, axis=None) # argsort on flattend array
# pairs_ranked_ind1, pairs_ranked_ind2 = np.unravel_index(inds_ranked, l2dist_square_mat.shape)

# for i in range(5):
#     clus_ind1 = pairs_ranked_ind1[i]
#     clus_ind2 = pairs_ranked_ind2[i]
#     print(clus_ind1, clus_ind2, l2dist_square_mat[clus_ind1, clus_ind2])
#     fig = plt.figure(figsize=(10,16))
#     gs_ovr = gridspec.GridSpec(16, 10, figure=fig)
#     y_scale = np.max([np.max(np.abs(data1[:,:,clus_ind1])), np.max(np.abs(data2[:,:,clus_ind2]))])
#     for i_ch in range(N_CH):
#         x, y = geom[i_ch,:]
#         plot_row, plot_col = (15-int(y/40)), (int(x/25))
#         ax = fig.add_subplot(gs_ovr[plot_row, plot_col*2:2+plot_col*2])# plt.subplot(16,2,plot_row*2+plot_col+1)
#         ax.plot(\
#             np.arange(n_sample)/F_SAMPLE*1000, \
#             data1[i_ch, :, clus_ind1], \
#             # label="Coordinate (%d,%d)" % (x, y),\
#             # color=cmap(peak_amplitude_ranks[i_clus_plot]) \
#             )
#         ax.set_ylim(-1*y_scale, y_scale)
#         if plot_col==1:
#             ax.set_yticks([])
#         if plot_row!=15:
#             ax.set_xticks([])
#         else:
#             ax.set_xlabel("Time (ms)")
#     for i_ch in range(N_CH):
#         x, y = geom[i_ch,:]
#         plot_row, plot_col = (15-int(y/40)), (int(x/25))
#         ax = fig.add_subplot(gs_ovr[plot_row, 6+plot_col*2:8+plot_col*2])# plt.subplot(16,2,plot_row*2+plot_col+1)
#         ax.plot(\
#             np.arange(n_sample)/F_SAMPLE*1000, \
#             data2[i_ch, :, clus_ind2], \
#             # label="Coordinate (%d,%d)" % (x, y),\
#             # color=cmap(peak_amplitude_ranks[i_clus_plot]) \
#             )
#         ax.set_ylim(-1*y_scale, y_scale)
#         if plot_col==1:
#             ax.set_yticks([])
#         if plot_row!=15:
#             ax.set_xticks([])
#         else:
#             ax.set_xlabel("Time (ms)")
#     plt.show()