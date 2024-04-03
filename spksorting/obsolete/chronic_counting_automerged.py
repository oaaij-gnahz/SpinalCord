'''
Count chronic stats of a mouse
8/30/2022
'''
import os
import json
import re
import datetime
import traceback
from collections import OrderedDict
# import typing

import shutil
import numpy as np
import matplotlib.pyplot as plt

from utils.read_mda import readmda

POSTPROC_NAME = "postproc_220913"
MERGE_NAME = "merge_220930"
STATFIGS_NAME = "statfigs_220913__" + POSTPROC_NAME + "__" + MERGE_NAME
F_SAMPLE = 30000
ORIGINAL_FORMAT = "OPENEPHYS"
if ORIGINAL_FORMAT=="RHD":
    SESSION_NAMING_PATTERN = r"[A-Za-z_ ]+_([0-9]+_[0-9]+)"
    DATETIME_STR_PATTERN = "%y%m%d_%H%M%S"
elif ORIGINAL_FORMAT=="OPENEPHYS":
    SESSION_NAMING_PATTERN = r"[0-9]+_([0-9]+-[0-9]+-[0-9]+_[0-9]+-[0-9]+-[0-9]+)"
    DATETIME_STR_PATTERN = "%Y-%m-%d_%H-%M-%S"
else:
    raise ValueError("Unknown ORIGINAL_FORMAT:", ORIGINAL_FORMAT)

def read_firings_and_templates_msort(session_folder_msort: str, session_folder_custom: str):
    # read "raw" mountainsort result and discard rejected units
    firings = readmda(os.path.join(session_folder_msort, "firings.mda")).astype(np.int64)
    templates_raw = readmda(os.path.join(session_folder_msort, "templates.mda")).astype(np.float64)
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
    return firings_curated, templates_curated, map_raw2curated

def get_session_stat(session_folder):
    
    with open(os.path.join(session_folder, POSTPROC_NAME, "log_postproc_params.json"), "r") as f:
        params = json.load(f)
    try:
        firings = readmda(os.path.join(session_folder, POSTPROC_NAME, MERGE_NAME, "firings_merged.mda")).astype(np.int64)
        template_waveforms = readmda(os.path.join(session_folder, POSTPROC_NAME, MERGE_NAME, "templates_full_merged.mda")).astype(np.float64)
    except: 
        # in case no merging available
        firings, template_waveforms, _ = read_firings_and_templates_msort(session_folder, os.path.join(session_folder, POSTPROC_NAME))
    n_ch = template_waveforms.shape[0]
    n_clus = template_waveforms.shape[2]
    spike_labels = firings[2,:]
    n_pri_ch_known = 0
    pri_ch_lut = -1 * np.ones(n_clus, dtype=int)
    for (spk_ch, spk_lbl) in zip(firings[0,:], spike_labels):
        if pri_ch_lut[spk_lbl-1]==-1:
            pri_ch_lut[spk_lbl-1] = spk_ch-1
            n_pri_ch_known += 1
            if n_pri_ch_known==n_clus:
                break
    template_peaks = np.max(template_waveforms, axis=1) # (n_ch, n_clus)
    template_troughs = np.min(template_waveforms, axis=1)
    template_p2ps = template_peaks - template_troughs
    p2p_amplitudes = np.max(template_p2ps, axis=0)# [pri_ch_lut, np.arange(n_clus)] # (n_clus, )
    n_ch = template_waveforms.shape[0]
    # count spikes one by one
    spike_times_all = firings[1,:]
    spike_labels = firings[2,:]
    spike_times_by_clus =[[] for i in range(n_clus)]
    spike_count_by_clus = np.zeros((n_clus,))
    for spk_time, spk_lbl in zip(spike_times_all, spike_labels):
        spike_times_by_clus[spk_lbl-1].append(spk_time-1)
    for i in range(n_clus):
        spike_times_by_clus[i] = np.array(spike_times_by_clus[i])
        spike_count_by_clus[i] = spike_times_by_clus[i].shape[0]
    # calculate isi violations
    n_bins=100
    isi_vis_max=100 # each bin is 1ms wide
    isi_bin_edges = np.linspace(0, isi_vis_max, n_bins+1) # in millisec; 1ms per bin
    isi_hists = np.zeros((n_clus, isi_bin_edges.shape[0]-1))
    refrac_violation_ratio = np.full((n_clus,), -1.0)
    for i_clus in range(n_clus):
        isi = np.diff(spike_times_by_clus[i_clus]) / F_SAMPLE * 1000 # ISI series in millisec
        isi_hist_this, _ = np.histogram(isi, bins=isi_bin_edges)
        isi_hists[i_clus, :] =isi_hist_this
        refrac_violation_ratio[i_clus] = (isi_hist_this[0]+isi_hist_this[1]) / isi.shape[0]
    multi_unit_mask = (refrac_violation_ratio > params['REFRAC_VIOLATION_RATIO_THRESH'])
    n_single_units = n_clus - np.sum(multi_unit_mask)
    n_sing_or_mult = n_clus
    stat_dict = {}
    stat_dict['n_single_units'] = n_single_units
    stat_dict['n_sing_or_mult'] = n_sing_or_mult
    stat_dict['template_peaks'] = template_peaks
    stat_dict['n_ch'] = n_ch
    stat_dict['p2p_amplitudes'] = p2p_amplitudes
    return stat_dict

def process_one_animal(animal_folder, animal_name):
    result_folder = os.path.join(animal_folder, STATFIGS_NAME)
    if os.path.exists(result_folder):
        print("RESULT_FOLDER is exisiting:\n%s\nPress Enter To delete and continue; ^Z to terminate" % result_folder)
        input()
        if not os.path.isdir(result_folder):
            os.unlink(result_folder)
        else:
            shutil.rmtree(result_folder)
    os.makedirs(result_folder)
    sessions_subfolders = os.listdir(animal_folder)
    if ORIGINAL_FORMAT=="RHD":
        sessions_subfolders = list(filter(
            lambda x: os.path.isdir(os.path.join(animal_folder, x)) and x.lower().startswith(animal_name.lower()), 
            sessions_subfolders
        ))
    elif ORIGINAL_FORMAT=="OPENEPHYS":
        sessions_subfolders = list(filter(
            lambda x: os.path.isdir(os.path.join(animal_folder, x)) and (re.match(SESSION_NAMING_PATTERN, x) is not None), 
            sessions_subfolders
        ))
    else:
        raise ValueError("Unknown ORIGINAL_FORMAT:", ORIGINAL_FORMAT)
    sessions_subfolders = sorted(
        sessions_subfolders, 
        key=lambda x: datetime.datetime.strptime(re.match(SESSION_NAMING_PATTERN, x)[1], DATETIME_STR_PATTERN) 
    )
    session_datetimes = []
    session_stats = {}
    session_stats['n_single_units'] = []
    session_stats['n_sing_or_mult'] = []
    session_stats['n_ch'] = []
    session_stats['p2p_amplitudes'] = []
    for session_subfolder in sessions_subfolders:
        try:
            datetimestr = re.match(SESSION_NAMING_PATTERN, session_subfolder)[1]
            session_datetime = datetime.datetime.strptime(datetimestr, DATETIME_STR_PATTERN)
            stat_dict = get_session_stat(os.path.join(animal_folder, session_subfolder))
            session_stats['n_single_units'].append(stat_dict['n_single_units'])
            session_stats['n_sing_or_mult'].append(stat_dict['n_sing_or_mult'])
            session_stats['n_ch'].append(stat_dict['n_ch'])
            session_stats['p2p_amplitudes'].append(stat_dict['p2p_amplitudes'])
            session_datetimes.append(session_datetime)
            print(session_subfolder, stat_dict['n_single_units'], stat_dict['n_sing_or_mult'])
        except Exception as e:
            traceback.print_exc()
            pass

    # timedelta relative to first session.
    timedelta_floats = np.array([(sdt - session_datetimes[0]).total_seconds() for sdt in session_datetimes])
    print("N_sessions_total", timedelta_floats.shape)
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.plot(session_datetimes, np.array(session_stats['n_sing_or_mult'])/np.array(session_stats['n_ch']), label='# single- and multi-units')
    # ax.plot(session_datetimes, np.array(session_stats['n_single_units'])/np.array(session_stats['n_ch']), label='# single-units')
    # ax.legend()
    # plt.xticks(session_datetimes, rotation=45)
    plt.xticks(session_datetimes, np.round(timedelta_floats/(24*3600)).astype(int))
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(result_folder, "n_units.png"))
    plt.close()

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    # timedelta_floats = timedelta_floats/timedelta_floats[-1]
    valid_session_ids = np.where(np.array(session_stats['n_sing_or_mult'])>0)[0]
    p2p_amplitudes = [session_stats['p2p_amplitudes'][sid] for sid in valid_session_ids]
    valid_timedelta_floats = timedelta_floats[valid_session_ids]
    ax.violinplot(p2p_amplitudes, positions=valid_timedelta_floats, showmeans=True, widths=timedelta_floats[-1]/timedelta_floats.shape[0]/2, points=500)
    # ax.plot(session_datetimes, np.array(session_stats['n_single_units'])/np.array(session_stats['n_ch']), label='# single-units')
    # ax.legend()
    plt.xticks(timedelta_floats, np.round(timedelta_floats/(24*3600)).astype(int))#, rotation=45)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(result_folder, "amplitudes.png"))
    plt.close()

if __name__=="__main__":
    process_one_animal("/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/BenMouse1", "dorito")