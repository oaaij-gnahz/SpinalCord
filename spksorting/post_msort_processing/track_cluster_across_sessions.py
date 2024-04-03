import os
from time import time
from copy import deepcopy
import gc

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.gridspec as gridspec
from natsort import natsorted

from utils.read_mda import readmda

N_CH = 24
F_SAMPLE = 30000
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
store_path = "%s/cluster_track20200228" % (cont_root_path)
if not os.path.exists(store_path):
    os.makedirs(store_path)
data = []
fdata = []
for i in range(len(session_folders)):
    data1 = readmda(os.path.join(session_folders[i], "templates.mda")).astype(np.float64)
    n_ch = data1.shape[0] # should stay constant across all segs
    n_sample = data1.shape[1] # should stay constant across all segs
    fdata1 = data1.reshape((n_ch*n_sample, data1.shape[2])).T
    data.append(data1) # data1 is of shape (n_ch, n_samples, n_clus)
    fdata.append(fdata1) # fdata is of shape (n_clus, n_ch*n_samples)
    print(data1.shape)
    # data2 = readmda(os.path.join(session_folders[1], "templates.mda")).astype(np.float64)
    # fdata2 = data2.reshape((n_ch*n_sample, data2.shape[2])).T
    # print(data2.shape)

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


for i_track, (clus_track_str, clus_track_l2dist) in enumerate(zip(clus_track_strs, clus_track_l2dists)):
    print(clus_track_str)
    clus_inds = [int(indclus) for indclus in clus_track_str.split("-")]
    print(clus_track_l2dist)
    print('-------')
    track_folder = os.path.join(store_path, "clus_track%d"%(i_track+1))
    if not os.path.exists(track_folder):
        os.mkdir(track_folder)
    for i_seg in range(0, len(session_folders)):
        tmp_figpath = "%s/figs_allclus_waveforms_new/waveform_clus%d.png"%(session_folders[i_seg], clus_inds[i_seg])
        if not os.path.exists(tmp_figpath):
            tmp_figpath = "%s/figs_allclus_waveforms_new/z_multiunit_waveform_clus%d.png"%(session_folders[i_seg], clus_inds[i_seg])
        if not os.path.exists(tmp_figpath):
            tmp_figpath = "%s/figs_allclus_waveforms_new/z_failed_waveform_clus%d.png"%(session_folders[i_seg], clus_inds[i_seg])
        # tmp_figpath = "%s/track_clus/clus%d.svg"%(session_folders[i_seg], clus_inds[i_seg])
        new_figpath = os.path.join(track_folder, "seg%d_clus%d.png"%(i_seg+1, clus_inds[i_seg]))
        cmd_str = "cp %s %s" % (tmp_figpath, new_figpath)
        print("Execute:", cmd_str)
        os.system(cmd_str)




print(time()-ts)

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