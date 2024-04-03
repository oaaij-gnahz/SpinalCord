'''
Count chronic stats of a mouse
8/30/2022
'''
import os
import json
import re
import datetime

import shutil
import numpy as np
import matplotlib.pyplot as plt

from utils.read_mda import readmda

POSTPROC_NAME = "postproc_220913"
STATFIGS_NAME = "statfigs_220913_from__" + POSTPROC_NAME.replace('/', '-') + "__"

def get_session_stat(session_folder):
    with open(os.path.join(session_folder, "combine_metrics_new.json"), 'r') as f:
        x = json.load(f)
    clus_metrics_list = x['clusters']
    n_clus = len(clus_metrics_list)
    
    firings = readmda(os.path.join(session_folder, "firings.mda")).astype(np.int64)
    spike_labels = firings[2,:]
    n_pri_ch_known = 0
    pri_ch_lut = -1 * np.ones(n_clus, dtype=int)
    for (spk_ch, spk_lbl) in zip(firings[0,:], spike_labels):
        if pri_ch_lut[spk_lbl-1]==-1:
            pri_ch_lut[spk_lbl-1] = spk_ch-1
            n_pri_ch_known += 1
            if n_pri_ch_known==n_clus:
                break
    template_waveforms = readmda(os.path.join(session_folder, "templates.mda")).astype(np.float64)
    template_peaks = np.max(template_waveforms, axis=1)
    template_troughs = np.min(template_waveforms, axis=1)
    template_p2ps = template_peaks - template_troughs
    peak_amplitudes = template_p2ps[pri_ch_lut, np.arange(n_clus)] # (n_clus, )
    n_ch = template_waveforms.shape[0]
    postprocfolder = os.path.join(session_folder, POSTPROC_NAME)
    rejection_maskz = np.load(os.path.join(postprocfolder, "cluster_rejection_mask.npz"))
    n_single_units = np.sum(rejection_maskz['single_unit_mask'])
    n_sing_or_mult = np.sum(np.logical_or(rejection_maskz['single_unit_mask'], rejection_maskz['multi_unit_mask']))
    stat_dict = {}
    stat_dict['n_single_units'] = n_single_units
    stat_dict['n_sing_or_mult'] = n_sing_or_mult
    stat_dict['template_peaks'] = template_peaks
    stat_dict['n_ch'] = n_ch
    return stat_dict

def process_one_animal(animal_folder, animal_name):
    result_folder = os.path.join(animal_folder, STATFIGS_NAME)
    if os.path.exists(result_folder):
        if not os.path.isdir(result_folder):
            os.unlink(result_folder)
        else:
            shutil.rmtree(result_folder)
    os.makedirs(result_folder)
    sessions_subfolders = os.listdir(animal_folder)
    sessions_subfolders = list(filter(
        lambda x: os.path.isdir(os.path.join(animal_folder, x)) and x.startswith(animal_name), 
        sessions_subfolders
    ))
    session_datetimes = []
    session_stats = {}
    session_stats['n_single_units'] = []
    session_stats['n_sing_or_mult'] = []
    session_stats['n_ch'] = []
    for session_subfolder in sessions_subfolders:
        
        datetimestr = re.match(r"[A-Za-z_]+_([0-9]+_[0-9]+)", session_subfolder)[1]
        session_datetime = datetime.datetime.strptime(datetimestr, "%y%m%d_%H%M%S")
        stat_dict = get_session_stat(os.path.join(animal_folder, session_subfolder))
        session_stats['n_single_units'].append(stat_dict['n_single_units'])
        session_stats['n_sing_or_mult'].append(stat_dict['n_sing_or_mult'])
        session_stats['n_ch'].append(stat_dict['n_ch'])
        session_datetimes.append(session_datetime)
        print(session_subfolder, stat_dict['n_single_units'], stat_dict['n_sing_or_mult'])

    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    ax.plot(session_datetimes, np.array(session_stats['n_sing_or_mult'])/np.array(session_stats['n_ch']), label='# single- and multi-units')
    ax.plot(session_datetimes, np.array(session_stats['n_single_units'])/np.array(session_stats['n_ch']), label='# single-units')
    ax.legend()
    plt.xticks(session_datetimes, rotation=45)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(result_folder, "n_units_both.png"))
    plt.close()

if __name__=="__main__":
    process_one_animal("/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/yogurt_chronic", "yogurt")