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
import matplotlib
import pandas as pd
# font = {# 'family' : 'monospace',
#         # 'weight' : 'bold',
#         'size'   : 26}

# matplotlib.rc('font', **font)

from utils.read_mda import readmda


USE_MANUAL_UNITMASK = True
POSTPROC_NAME_PREFIX = "postproc_"
MERGE_NAME_PREFIX = "merge_"


# fixed storage structure for spinal cord project chronic data
SESSION_NAMING_PATTERN = {}
DATETIME_STR_PATTERN = {}
SESSION_NAMING_PATTERN["RHD"] = r"[A-Za-z_ ]+_([0-9]+_[0-9]+)"
DATETIME_STR_PATTERN["RHD"] = "%y%m%d_%H%M%S"
SESSION_NAMING_PATTERN["OPENEPHYS"] = r"[0-9]+_([0-9]+-[0-9]+-[0-9]+_[0-9]+-[0-9]+-[0-9]+)"
DATETIME_STR_PATTERN["OPENEPHYS"] = "%Y-%m-%d_%H-%M-%S"

DATA_SAVE_FOLDER = "../_pls_ignore_chronic_data_230614"


def get_surgery_date(animalfolder):
    with open(os.path.join(animalfolder, "general__info", "surgerydate.txt"), "r") as f:
        x=f.readline().split('\n')[0]
        print(x)
    return datetime.datetime.strptime(x, "%Y%m%d")

def find_most_recent_postproc(session_folder_msort):
    list_postproc_names = list(filter(lambda x: x.startswith(POSTPROC_NAME_PREFIX), os.listdir(session_folder_msort)))
    recent_postproc_name = list_postproc_names[-1] # largest YYMMDD number
    print('\t', list_postproc_names, '\n\t    Most Recent:', recent_postproc_name)
    list_merge_names = list(filter(lambda x: x.startswith(MERGE_NAME_PREFIX), os.listdir(os.path.join(session_folder_msort, recent_postproc_name))))
    if len(list_merge_names)>0:
        recent_merge_name = list_merge_names[-1] # largest YYMMDD number
    else:
        recent_merge_name = None
    print('\t', list_merge_names, '\n\t    Most Recent:', recent_merge_name)
    return recent_postproc_name, recent_merge_name


def read_firings_and_templates_msort(session_folder_msort: str, session_folder_custom: str):
    # read "raw" mountainsort result and discard rejected units
    firings = readmda(os.path.join(session_folder_msort, "firings.mda")).astype(np.int64)
    templates_raw = readmda(os.path.join(session_folder_msort, "templates.mda")).astype(np.float64)
    if USE_MANUAL_UNITMASK:
        accept_mask = pd.read_csv(
            os.path.join(session_folder_custom, "single_unit_mask_human.csv"), 
            header=None
        ).values.squeeze().astype(bool)
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
    return firings_curated, templates_curated, map_raw2curated

def get_session_stat(session_folder):
    with open(os.path.join(session_folder, "session_rhd_info.json"), "r") as f:
        rhd_info_ = json.load(f)
    f_sample = rhd_info_["sample_freq"]
    postproc_name, merge_name = find_most_recent_postproc(session_folder)
    with open(os.path.join(session_folder, postproc_name, "log_postproc_params.json"), "r") as f:
        params = json.load(f)
    try:
        firings = readmda(os.path.join(session_folder, postproc_name, merge_name, "firings_merged.mda")).astype(np.int64)
        template_waveforms = readmda(os.path.join(session_folder, postproc_name, merge_name, "templates_full_merged.mda")).astype(np.float64)
    except: 
        # in case no merging available
        firings, template_waveforms, _ = read_firings_and_templates_msort(session_folder, os.path.join(session_folder, postproc_name))
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
        isi = np.diff(spike_times_by_clus[i_clus]) / f_sample * 1000 # ISI series in millisec
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

def process_one_animal(animal_folder, animal_name, original_format, plot_axes):
    surgery_date = get_surgery_date(animal_folder)
    sessions_subfolders = os.listdir(animal_folder)
    if original_format=="RHD":
        sessions_subfolders = list(filter(
            lambda x: os.path.isdir(os.path.join(animal_folder, x)) and x.lower().startswith(animal_name.lower()), 
            sessions_subfolders
        ))
    elif original_format=="OPENEPHYS":
        sessions_subfolders = list(filter(
            lambda x: os.path.isdir(os.path.join(animal_folder, x)) and (re.match(SESSION_NAMING_PATTERN[original_format], x) is not None), 
            sessions_subfolders
        ))
    else:
        raise ValueError("Unknown ORIGINAL_FORMAT:", original_format)
    sessions_subfolders = sorted(
        sessions_subfolders, 
        key=lambda x: datetime.datetime.strptime(re.match(SESSION_NAMING_PATTERN[original_format], x)[1], DATETIME_STR_PATTERN[original_format]) 
    )
    session_datetimes = []
    session_stats = {}
    session_stats['n_single_units'] = []
    session_stats['n_sing_or_mult'] = []
    session_stats['n_ch'] = []
    session_stats['p2p_amplitudes'] = []
    session_datetimestrs = []
    for session_subfolder in sessions_subfolders:
        try:
            datetimestr = re.match(SESSION_NAMING_PATTERN[original_format], session_subfolder)[1]
            session_datetime = datetime.datetime.strptime(datetimestr, DATETIME_STR_PATTERN[original_format])
            stat_dict = get_session_stat(os.path.join(animal_folder, session_subfolder))
            session_stats['n_single_units'].append(stat_dict['n_single_units'])
            session_stats['n_sing_or_mult'].append(stat_dict['n_sing_or_mult'])
            session_stats['n_ch'].append(stat_dict['n_ch'])
            session_stats['p2p_amplitudes'].append(stat_dict['p2p_amplitudes'])
            session_datetimes.append(session_datetime)
            session_datetimestrs.append(datetimestr)
            print(session_subfolder, stat_dict['n_single_units'], stat_dict['n_sing_or_mult'])
        except Exception as e:
            traceback.print_exc()
            print("See above exception")
            # break
            pass

    # timedelta relative to first session.
    timedelta_floats = np.array([(sdt - surgery_date).total_seconds() for sdt in session_datetimes])
    print("N_sessions_total", timedelta_floats.shape)
    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(111)
    units_per_channel = np.array(session_stats['n_sing_or_mult'])/np.array(session_stats['n_ch'])
    plot_axes[0].plot(timedelta_floats, units_per_channel, label=animal_name, marker='.')
    # ax.plot(session_datetimes, np.array(session_stats['n_single_units'])/np.array(session_stats['n_ch']), label='# single-units')
    # ax.legend()
    # plt.xticks(session_datetimes, rotation=45)
    

    # fig = plt.figure(figsize=(10,5))
    # ax = fig.add_subplot(111)
    valid_session_ids = np.where(np.array(session_stats['n_sing_or_mult'])>0)[0]
    p2p_amplitudes_mean = [np.mean(session_stats['p2p_amplitudes'][sid]) for sid in valid_session_ids]
    valid_timedelta_floats = timedelta_floats[valid_session_ids]
    plot_axes[1].plot(valid_timedelta_floats, p2p_amplitudes_mean, label=animal_name, marker='.')
    p2p_amplitudes_all = session_stats['p2p_amplitudes']
    ret = {
        "timedelta_floats": timedelta_floats,
        "units_per_channel": units_per_channel,
        "valid_timedelta_floats": valid_timedelta_floats,
        "p2p_amplitudes_mean": p2p_amplitudes_mean,
        "p2p_amplitudes_all": p2p_amplitudes_all,
        "session_datetimestrs": session_datetimestrs,
        "n_ch": session_stats['n_ch']
    }
    return ret
    # return timedelta_floats, units_per_channel, valid_timedelta_floats, p2p_amplitudes_mean, p2p_amplitudes_all, session_datetimestrs, session_stats['n_ch']
    
    # # ax.plot(session_datetimes, np.array(session_stats['n_single_units'])/np.array(session_stats['n_ch']), label='# single-units')
    # # ax.legend()
    # plt.xticks(timedelta_floats, np.round(timedelta_floats/(24*3600)).astype(int))#, rotation=45)
    # plt.subplots_adjust(bottom=0.2)
    # plt.savefig(os.path.join(result_folder, "amplitudes.png"))
    # plt.close()

def process_multi_animals(animal_list, recording_types, save):
    fig1 = plt.figure(figsize=(10,5))
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure(figsize=(10,5))
    ax2 = fig2.add_subplot(111)
    for i_animal, (animal_folder, recording_type) in enumerate(zip(animal_list, recording_types)):
        animal_name = animal_folder.split('/')[-1].split('_')[0]
        print(i_animal, animal_folder, animal_name)
        # timedelta_floats, units_per_channel, valid_timedelta_floats, p2p_amplitudes_mean, p2p_amplitudes_all, session_datetimestrs
        save_dict = process_one_animal(animal_folder, animal_name, recording_type, [ax1, ax2])
        if save:
            np.savez(
                os.path.join(DATA_SAVE_FOLDER, animal_name+"_firings.npz"),
                timedelta_floats=save_dict['timedelta_floats'],
                units_per_channel=save_dict['units_per_channel'],
                valid_timedelta_floats=save_dict['valid_timedelta_floats'],
                p2p_amplitudes_mean=save_dict['p2p_amplitudes_mean']
                )
            df_save = pd.DataFrame()
            df_save['dayAfterSurgery'] = (save_dict['timedelta_floats']/(24*3600)).astype(int)
            df_save['datetime'] = save_dict['session_datetimestrs']
            df_save['n_units'] = [len(p2p_amplitudes) for p2p_amplitudes in save_dict['p2p_amplitudes_all']]
            df_save['n_channels'] = save_dict['n_ch']
            df_save['amplitudes_accepted_units'] = save_dict['p2p_amplitudes_all']
            df_save.to_csv(os.path.join(DATA_SAVE_FOLDER, animal_name+"_firings.csv"), index=False)

    for ax in [ax1, ax2]:
        ax.legend()
        # ax.set_xticks(np.arange(0, 61, 10)*24*3600)
        # ax.set_xticklabels(np.arange(0, 61, 10))
        # ax.set_xticklabels(np.round(ax.get_xticks()/(24*3600)).astype(int))
        xtickmax = int(ax.get_xticks()[-1] / (24*3600))
        # print(xtickmax)
        ax.set_xticks(np.arange(0, xtickmax, 20)*24*3600)
        ax.set_xticklabels(np.arange(0, xtickmax, 20))
        ax.set_xlabel("Days after Implantation")
        # ax.set_xlim([-1*(10*3600), 60*(24*3600)])
    ax1.set_ylabel("#Units")
    ax2.set_ylabel(r"Avg Amplitude ($\mu V$)")
    ax2.set_ylim([0, 220])
    # for ax in [ax1,ax2]:
    #     ax.set_xlim([-1*(10*3600), ax.get_xlim()[1]+40*24*3600])
    # plt.subplots_adjust(bottom=0.1)
    fig1.savefig("../_pls_ignore_n_units.eps")
    fig1.savefig("../_pls_ignore_n_units.png")
    fig2.savefig("../_pls_ignore_ampltud.png")
    fig2.savefig("../_pls_ignore_ampltud.eps")
    for ax in [ax1,ax2]:
        ax.set_xlim([-1*(10*3600), 120*(24*3600)])
    fig1.savefig("../_pls_ignore_n_units_120days.eps")
    fig1.savefig("../_pls_ignore_n_units_120days.png")
    fig2.savefig("../_pls_ignore_ampltud_120days.png")
    fig2.savefig("../_pls_ignore_ampltud_120days.eps")
    plt.close()
    plt.close()


if __name__=="__main__":
    animals_list = [
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/BenMouse0", "OPENEPHYS"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/BenMouse1", "OPENEPHYS"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/dorito_chronic", "RHD"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/mustang_chronic", "RHD"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/nacho_chronic", "RHD"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/tacoma_chronic", "RHD"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/corolla_chronic", "RHD"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/yogurt_chronic", "RHD"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/nora_chronic", "RHD")
    ]
    print("STARTING")
    if not os.path.exists(DATA_SAVE_FOLDER):
        os.makedirs(DATA_SAVE_FOLDER)
    process_multi_animals([k[0] for k in animals_list], [t[1] for t in animals_list], save=True)