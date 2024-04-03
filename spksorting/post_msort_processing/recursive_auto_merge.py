import json
import os
from collections import OrderedDict
import typing
import traceback
# from time import time
# from copy import deepcopy
# import gc

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.gridspec as gridspec

from utils.misc import construct_firings_from_spktrains
from utils.read_mda import readmda
from utils.mdaio import writemda64

POSTPROC_SUBDIRNAME = "postproc_230611"
MERGE_SUBDIRNAME    = "merge_230614"
USE_MANUAL_UNITMASK = True

F_SAMPLE = None #30000
TRANSIENT_AMPLITUDE_VALID_DURATION = 10e-4 # seconds (duration of data before and after each spike that we consider when deciding the transient amplitude)
TAVD_NSAMPLE = None #int(np.ceil(TRANSIENT_AMPLITUDE_VALID_DURATION*F_SAMPLE))
MAX_GEOM_DIST = 25 # um
ADJACENCY_RADIUS_SQUARED = 140**2
# MAP_PATH0="../geom_channel_maps/map.csv"
# MAP_PATH1="../geom_channel_maps/map_corolla24ch.csv"
MAP_PATH0="../geom_channel_maps/ChannelMap_Ben.csv"
MAP_PATH1="../geom_channel_maps/ChannelMap_Ben.csv"



def get_peak_amp_ratio_matrix(data_a, data_b=None):
    """
    assumes data_a is (n_1, ) and data_b is (n_2, )
    returns (n_1, n_2) distance matrix
    where n_1 and n_2 could be cluster counts from 2 sessions
    """
    if data_b is None:
        data_b = data_a
    data_min = np.minimum(data_a[:,None], data_b[None,:])
    data_max = np.maximum(data_a[:,None], data_b[None,:])
    return data_max/data_min - 1.

def read_firings_and_templates_msort(session_folder_msort: str, session_folder_custom: str) -> typing.Tuple[np.ndarray, np.ndarray, OrderedDict, np.ndarray]:
    global F_SAMPLE
    global TAVD_NSAMPLE
    with open(os.path.join(session_folder_msort, "session_rhd_info.json"), "r") as f:
        F_SAMPLE = json.load(f)['sample_freq']
    TAVD_NSAMPLE = int(np.ceil(TRANSIENT_AMPLITUDE_VALID_DURATION*F_SAMPLE))
    # read "raw" mountainsort result and discard rejected units
    firings = readmda(os.path.join(session_folder_msort, "firings.mda")).astype(np.int64)
    templates_raw = readmda(os.path.join(session_folder_msort, "templates.mda"))
    
    if templates_raw.shape[0]==32:
        geom = pd.read_csv(MAP_PATH0, header=None).values
    else:
        geom = pd.read_csv(MAP_PATH1, header=None).values
    
    if USE_MANUAL_UNITMASK:
        accept_mask = pd.read_csv(
            os.path.join(session_folder_custom, "single_unit_mask_human.csv"), 
            header=None
        ).values.squeeze().astype(bool)
        # print(accept_mask)
        # input()
    else:
        masks = np.load(os.path.join(session_folder_custom, "cluster_rejection_mask.npz"))
        accept_mask = masks['single_unit_mask'] # np.logical_or(masks['single_unit_mask'], masks['multi_unit_mask'])
    
    n_clus = templates_raw.shape[2]
    n_clus_accepted = np.sum(accept_mask)
    labels_map = -1*np.ones(n_clus)
    labels_map[accept_mask] = np.arange(n_clus_accepted)+1
    labels_map = labels_map.astype(int)
    if np.any(labels_map==0):
        raise ValueError
    # print(labels_map)
    # (1) reorganize firings.mda
    print(np.unique(firings[2,:]))
    tmp_labels_old = firings[2,:]
    if not np.all(tmp_labels_old>0):
        raise ValueError
    tmp_labels_new = labels_map[tmp_labels_old-1]
    firings[2,:] = tmp_labels_new
    spikes_keep_mask = firings[2,:]!=-1
    print(np.unique(firings[2,:]))
    firings_curated = firings[:, spikes_keep_mask]
    # (2) reorganize template.mda
    templates_curated = templates_raw[:,:, accept_mask]
    # (3) get reordering dict
    map_raw2curated = OrderedDict()
    for clus_label_curated in np.arange(1, n_clus_accepted+1):
        labels_raw = np.where(labels_map==clus_label_curated)[0]+1
        map_raw2curated[str(clus_label_curated)] = labels_raw.tolist()
    return firings_curated, templates_curated, map_raw2curated, geom
    

def calc_key_metrics(templates_full, firings, geom, radius_squared=ADJACENCY_RADIUS_SQUARED):
    """Calculate key cluster metrics used by recursive automerging"""

    n_chs, waveform_len, n_clus = templates_full.shape
    
    # get primary channels
    pri_ch_lut = -1 * np.ones(n_clus, dtype=int)
    n_pri_ch_known = 0
    for (spk_ch, spk_lbl) in zip(firings[0,:], firings[2,:]):
        if pri_ch_lut[spk_lbl-1]==-1:
            pri_ch_lut[spk_lbl-1] = spk_ch-1
            n_pri_ch_known += 1
            if n_pri_ch_known==n_clus:
                break
    
    # slice templates
    my_slice = slice(int(waveform_len//2-TAVD_NSAMPLE), int(waveform_len//2+TAVD_NSAMPLE), 1)
    templates = templates_full[:,my_slice,:]
    # waveform_len_sliced = templates.shape[1]

    # get template peaks and p2ps
    template_peaks = np.max(np.abs(templates), axis=1)
    print("template_peaks shape <should be (n_ch,n_clus)>:", template_peaks.shape)
    peak_amplitudes = template_peaks[pri_ch_lut, np.arange(n_clus)] # (n_clus,)
    template_peaks = np.max(templates, axis=1) # use full to calculate p2p
    template_troughs = np.min(templates, axis=1)
    template_p2ps = template_peaks - template_troughs

    # estimate locations by center-of-mass
    clus_coordinates = np.zeros((n_clus, 2))
    for i_clus in range(n_clus):
        prim_ch = pri_ch_lut[i_clus]
        prim_x, prim_y = geom[prim_ch, :]
        non_neighbor_mask = ((geom[:,0]-prim_x)**2 + (geom[:,1]-prim_y)**2 >= radius_squared)
        weights = template_p2ps[:, i_clus]
        weights[non_neighbor_mask] = 0
        weights = weights / np.sum(weights)
        clus_coordinates[i_clus, :] = np.sum(weights[:,None] * geom, axis=0)
    
    return templates, pri_ch_lut, peak_amplitudes, clus_coordinates

def calc_merge_candidates(templates, locations, peak_amplitudes):
    """
    calculate merging candidates by distance & waveform similarity\n
    returns a 3-column matrix of (n_pairs), the columns would be (src_unit, snk_unit, cand?)\n
    Please Use sliced templates and clean firings only (noise cluster must be rejected and the clean ones reordered)
    """

    n_ch, waveform_len, n_clus = templates.shape
    template_features = templates.reshape((n_ch*waveform_len, n_clus)).T

    pairs_all = []
    pairs_cand = []
    for i_clus in range(n_clus):
        neighborhood_mask = np.sum((locations-locations[i_clus,:])**2, axis=1) < MAX_GEOM_DIST**2
        neighborhood_mask[i_clus: ] = False # Force non-directed graph for merging; also no comparison with self
        n_neighborhood = np.sum(neighborhood_mask)
        if n_neighborhood<1:
            continue
        neighborhood_clus_ids = np.where(neighborhood_mask)[0] + 1 # cluster id starts from 1
        current_clus_id = i_clus + 1
        dist_mat = np.array([np.corrcoef(template_features[i_clus,:], template_features[k-1, :])[1,0] for k in neighborhood_clus_ids])
        corr_mask = dist_mat > 0.7 # actually a vector
        amp_ratio_mat = get_peak_amp_ratio_matrix(peak_amplitudes[:,None][i_clus,:], peak_amplitudes[neighborhood_mask]).squeeze()
        amp_ratio_mask = amp_ratio_mat < 0.5 # np.logical_and(amp_ratio_mat>0.8, amp_ratio_mat<1.25)
        merge_cand_mask = np.logical_and(amp_ratio_mask, corr_mask) # actually a vector
        n_cands = np.sum(merge_cand_mask)
        
        clus_id_paired_prev_all, clus_id_paired_post_all = np.zeros(n_neighborhood, dtype=int)+current_clus_id, neighborhood_clus_ids
        clus_id_paired_prev_cand, clus_id_paired_post_cand = np.zeros(n_cands, dtype=int)+current_clus_id, neighborhood_clus_ids[merge_cand_mask]

        pairs_all.extend(list(zip(clus_id_paired_prev_all, clus_id_paired_post_all))) # list of tuples
        pairs_cand.extend(list(zip(clus_id_paired_prev_cand, clus_id_paired_post_cand))) # list of tuples

        # plt.figure(figsize=(12,4)); 
        # plt.subplot(131); plt.imshow(merge_cand_mask, cmap='gray'); plt.colorbar(); 
        # plt.xticks(np.arange(n_neighborhood), neighborhood_clus_ids+1)
        # plt.yticks(np.arange(n_neighborhood), neighborhood_clus_ids+1)
        # plt.subplot(132); plt.imshow(dist_mat, cmap='gray', vmin=0, vmax=1); plt.colorbar(); plt.title("Corr")
        # plt.xticks(np.arange(n_neighborhood), neighborhood_clus_ids+1)
        # plt.yticks(np.arange(n_neighborhood), neighborhood_clus_ids+1)
        # plt.subplot(133); plt.imshow(amp_ratio_mat, cmap='gray'); plt.colorbar(); plt.title("ampRatio")
        # plt.xticks(np.arange(n_neighborhood), neighborhood_clus_ids+1)
        # plt.yticks(np.arange(n_neighborhood), neighborhood_clus_ids+1)
        # plt.show()
    if len(pairs_all) > 0: 
        cand_mask_1d = np.array([(pair in pairs_cand) for pair in pairs_all], dtype=bool) # (n_pairs,)
        assert(np.sum(cand_mask_1d)==len(pairs_cand))
        arr_pairs_all = np.array(pairs_all) # (n_pairs,2)
        print(arr_pairs_all.shape,cand_mask_1d.shape)
        arr_ret = np.concatenate([arr_pairs_all, cand_mask_1d[:,None]], axis=1)
    else:
        arr_ret = None
    return arr_ret


def merge_on_isi(merge_cand_arr, firings, templates_full) -> typing.Tuple[np.ndarray, np.ndarray, OrderedDict, int]:
    """
    Determine whether to merge the candidates by ISI criterion; written by Haad R, modified by Jiaao Z
    dataframe manipulation by Haad R

    Parameters
    ----------
    merge_cand_arr: (n_pairs, 3) array returned by _calc_merge_candidates()_
    firings: must be noise-rejected and reordered
    templates_full: must be noise-rejected and reordered

    Returns
    ----------
    firings_merged: merged firings
    templates_full_merged: merged full templates
    map_prev2new: a dict with {key: new label} and {value: list of src units from which it is merged}
    n_merges: number of merges that occurred
    """

    n_clus = np.max(firings[2,:])

    if merge_cand_arr is None:
        # no candidates to check for merging
        map_prev2new = OrderedDict()
        for i in range(n_clus):
            map_prev2new[str(i+1)] = tuple([i+1])
        return firings, templates_full, map_prev2new, 0

    # Read spike times and separate by clusters:
    df_spikes_by_clusters = pd.DataFrame(columns=['prim_chan', 'spike_stamps','cell_id'])
    for iter_local in range(1,n_clus+1,1):
        # extracting time stamps of each cluster
        mask_bin_cluster = (firings[2,:] == iter_local)
        temp_indx = mask_bin_cluster.nonzero()[0][0]
        df_spikes_by_clusters.at[iter_local,'prim_chan'] =  firings[0,temp_indx]
        df_spikes_by_clusters.at[iter_local,'cell_id'] =  firings[2,temp_indx]
        mask_bin_cluster = firings[1, mask_bin_cluster]
        # mask_bin_cluster = firings[1,:][mask_bin_cluster]
        # mask_bin_cluster = mask_bin_cluster.to_numpy()
        df_spikes_by_clusters.at[iter_local,'spike_stamps'] = mask_bin_cluster
    spike_count_by_clus = np.array([df_spikes_by_clusters.at[iter_local,'spike_stamps'].shape[0] for iter_local in range(1, n_clus+1)])
    arr_merge_cluster = merge_cand_arr.copy()
    cluster_pairs = arr_merge_cluster[:,[0,1]]
    distance_mask = arr_merge_cluster[:,2]

    # now determine whether to merge a candidate pair by ISI
    cluster_pairs = cluster_pairs - 1 # force the indices to start from 0
    merged_mask = np.zeros(n_clus,dtype=bool)
    final_merged_trains = []
    final_pairs_to_merge_base0 = []
    for local_iter, (clus_indx_i, clus_indx_j) in enumerate(zip(cluster_pairs[:,0],cluster_pairs[:,1])):
        if local_iter%100==0:
            print(local_iter, '/', cluster_pairs.shape[0]) 
        if distance_mask[local_iter]==1 and clus_indx_i!=clus_indx_j and merged_mask[clus_indx_i]==False and merged_mask[clus_indx_j]==False:
            # caclulate ISI
            spk_times_if_merged = np.concatenate(
                [df_spikes_by_clusters.loc[clus_indx_i+1,'spike_stamps'], 
                df_spikes_by_clusters.loc[clus_indx_j+1,'spike_stamps']
                ])
            spk_times_if_merged = np.sort(spk_times_if_merged) # TODO faster sorting because 2 stamps are sorted already?
            n_bins=100
            isi_vis_max=100 # each bin is 1ms wide
            isi_bin_edges = np.linspace(0, isi_vis_max, n_bins+1) # in millisec; 1ms per bin
            refrac_violation_ratio = np.full((n_clus,), -1.0)
            isi = np.diff(spk_times_if_merged) / F_SAMPLE * 1000 # ISI series in millisec
            isi_hist_this, _ = np.histogram(isi, bins=isi_bin_edges)
            refrac_violation_ratio = (isi_hist_this[0]+isi_hist_this[1]) / isi.shape[0]
            if refrac_violation_ratio < 0.07:
                merged_mask[clus_indx_i] = True
                merged_mask[clus_indx_j] = True
                final_pairs_to_merge_base0.append((clus_indx_i, clus_indx_j))
                final_merged_trains.append(spk_times_if_merged)
    n_merges = np.sum(merged_mask)
    print("# merged=", n_merges)
    
    
    # generate finally merged stuff
    map_prev2new = OrderedDict() # key: merged label; value: tuple of original labels. Both start from 1
    spk_trains_merged = []
    templates_full_merged = []

    id_final = 1
    for idx0 in np.where(merged_mask==False)[0]:
        map_prev2new[str(id_final)] = tuple([int(idx0+1)])
        id_final += 1
        spk_trains_merged.append(df_spikes_by_clusters.loc[idx0+1,'spike_stamps'])
        # print("SpikeCount", spk_trains_merged[-1].shape)
        templates_full_merged.append(templates_full[:,:,idx0])
    for i_pair, (idx_i0, idx_j0) in enumerate(final_pairs_to_merge_base0):
        map_prev2new[str(id_final)] = (int(idx_i0+1), int(idx_j0+1))
        id_final += 1
        spk_trains_merged.append(final_merged_trains[i_pair])
        # print(idx_i0+1, idx_j0+1, "MergedSpikeCount", spk_trains_merged[-1].shape)
        # get new template
        total_spike_cnt = spike_count_by_clus[idx_i0] + spike_count_by_clus[idx_j0]
        weight_i, weight_j = spike_count_by_clus[idx_i0]/total_spike_cnt, spike_count_by_clus[idx_j0]/total_spike_cnt
        this_merged_template = templates_full[:,:,idx_i0]*weight_i + templates_full[:,:,idx_j0]*weight_j
        templates_full_merged.append(this_merged_template)

    templates_full_merged = np.stack(templates_full_merged, axis=2)
    print("Shape of merged templates (n_ch, n_sample, n_clus_after_merging):", templates_full_merged.shape)

    # data structure that keeps track of the merging: array alternative to a ordered dict
    # reorg_map = np.zeros(1+n_clus) # reorg_map[0] is not used
    # for tar, srcs in map_prev2new.items():
    #     for src in srcs:
    #         reorg_map[src] = int(tar)
            # print(src, tar)
    
    # calculate primary channels
    template_peaks_merged = np.max(np.abs(templates_full_merged), axis=1)
    print("template_peaks_merged shape <should be (n_ch,n_clus_after_merging)>:", template_peaks_merged.shape)
    prim_channels_base0 = np.argmax(template_peaks_merged, axis=0)
    prim_channels = prim_channels_base0 + 1
    print("prim_channels shape <should be (n_clus_after_merging)>:", prim_channels.shape)

    firings_merged = construct_firings_from_spktrains(spk_trains_merged)
    return firings_merged, templates_full_merged, map_prev2new, n_merges

def get_map_raw2new(map_raw2prev, map_prev2new):
    """For all the dicts are involved; a key is a merged cluster label and a value is a list of merged units"""
    map_raw2new = OrderedDict()
    for label_new, list_src_prev in map_prev2new.items():
        list_src_raw = []
        for src_prev in list_src_prev:
            list_src_raw.extend(map_raw2prev[str(src_prev)])
        map_raw2new[label_new] = list_src_raw
    return map_raw2new

def recursive_merge(firings, templates_full, map_raw2curated, geom):
    firings_this = firings
    templates_full_this = templates_full
    map_raw2new_this = map_raw2curated
    flag = True
    num_iter = 0
    while flag:
        # calculate key metrics
        templates_sliced, pri_ch_lut, peak_amplitudes, clus_coordinates = calc_key_metrics(templates_full_this, firings_this, geom)
        # merge candidates
        merge_cand_arr = calc_merge_candidates(templates_sliced, clus_coordinates, peak_amplitudes)
        # mege on ISI criterion
        firings_merged, templates_full_merged, map_prev2new, n_merges = merge_on_isi(merge_cand_arr, firings_this, templates_full_this)
        # update mapping
        map_raw2new_this = get_map_raw2new(map_raw2new_this, map_prev2new)
        print("num_iter:", num_iter)
        if n_merges == 0:
            # stop
            flag = False
        else:
            # update firings and template
            firings_this = firings_merged
            templates_full_this = templates_full_merged
    
    return firings_merged, templates_full_merged, map_raw2new_this

def merge_one_session(session_folder, session_postproc_subdir, session_merge_dir):
    postproc_folder = os.path.join(session_folder, session_postproc_subdir)
    session_folder_merge = os.path.join(session_folder, session_postproc_subdir, session_merge_dir)
    os.makedirs(session_folder_merge, exist_ok=True)
    try:
        firings_curated, templates_curated, map_raw2curated, geom = read_firings_and_templates_msort(session_folder, postproc_folder)
        firings_merged, templates_full_merged, map_raw2new = recursive_merge(firings_curated, templates_curated, map_raw2curated, geom)
        print("#single units after curation:", templates_curated.shape[2])
        print("#single units after merging :", templates_full_merged.shape[2])
        print("Saving merged results to disk")
        writemda64(firings_merged, os.path.join(session_folder_merge, "firings_merged.mda"))
        writemda64(templates_full_merged, os.path.join(session_folder_merge, "templates_full_merged.mda"))
        with open(os.path.join(session_folder_merge, "map_srclut.json"), 'w') as f:
            json.dump(map_raw2new, f)
    except Exception as e:
        print("ERROR in", session_folder)
        traceback.print_exc()
        return -1


def main_one_animal(result_folder, postproc_subdirname, merge_subdirname):
    relevant_subpaths = list(filter(lambda x: (('_' in x) and ('.' not in x) and ('__' not in x) and ('_bad' not in x)), os.listdir(result_folder)))
    relevant_fullpaths = list(map(lambda x: os.path.join(result_folder, x), relevant_subpaths))
    error_session_folders = []
    for session_folder in filter(lambda x: os.path.isdir(x), relevant_fullpaths):
        ret = merge_one_session(session_folder, postproc_subdirname, merge_subdirname)
        if ret==-1:
            error_session_folders.append(session_folder)
    print("ERROR_SESSIONS:")
    print(error_session_folders)
    return error_session_folders

if __name__ == '__main__':
    # TODO iterate over all animals
    result_folders = [
        # "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/corolla_chronic/",
        # "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/dorito_chronic/",
        # "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/mustang_chronic/",
        # "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/nacho_chronic/",
        # "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/nora_chronic/",
        # "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/tacoma_chronic/",
        # "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/yogurt_chronic/",
        "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/BenMouse0/",
        "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/BenMouse1/"
    ]

    for res_folder in result_folders:
        main_one_animal(res_folder, POSTPROC_SUBDIRNAME, MERGE_SUBDIRNAME)