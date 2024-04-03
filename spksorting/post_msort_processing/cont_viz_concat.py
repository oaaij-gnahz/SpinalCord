import os
from time import time
from copy import deepcopy
import gc
import json
import re

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from utils.read_mda import readmda

plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams['font.size']=30

MAP_PATH="../geom_channel_maps/map.csv"
ELECTRODE_RADIUS = 12.5
SEG_SPACING = 250
cont_root_path = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/MustangContinuous/Mustang_220126_125248/onePiece/onePieceSegged_1hours"

def get_segment_index(segment_name: str) -> int:
    return int(re.search("seg([0-9]+)", segment_name)[1])


def read_postproc_data(msort_path : str, postproc_path : str) -> dict:
    """ read post processed data
    Returns dict: 
    ret['firing_rates']
    ret['peak_amplitudes']
    ret['accpet_mask'] : single units
    ret['locations'] : unit locations
    ret['prim_chs']
    !!! Different from the case of processing entire session,
    In this case,  one segment may miss entirely the firing of a neuron, causing 
        (1) the metrics.json to be shorter than the true #units
        (2) corresponding position at template.mda to be NaN
    So we keep track of the "clus_labels" from metrics.json
    And reconstruct the metrics of full length. The missing units will be marked -1 isolation and 999 noise overlap
    And corresponding template to be Zero

    """
    
    ### read clustering metrics file 
    const_SEGMENT_LEN = 3600 # seconds
    with open(os.path.join(msort_path, "combine_metrics_new.json"), 'r') as f:
        x = json.load(f)
    ret = {}
    # read firing stamps, template and continuous waveforms from MountainSort outputs and some processing
    firings = readmda(os.path.join(msort_path, "firings_seg.mda")).astype(np.int64)
    template_waveforms = readmda(os.path.join(msort_path, "templates.mda")).astype(np.float64)
    n_clus = template_waveforms.shape[2]
    # set nan to zero just in case some units don't fire during the segment resulting in nan 
    template_waveforms = np.nan_to_num(template_waveforms, nan=0.0)
    clus_metrics_list = x['clusters']
    clus_labels = np.array([k['label'] for k in clus_metrics_list])
    isolation_score_short = np.array([k['metrics']['isolation'] for k in clus_metrics_list])
    noise_overlap_score_short = np.array([k['metrics']['noise_overlap'] for k in clus_metrics_list])
    peak_snr_short = np.array([k['metrics']['peak_snr'] for k in clus_metrics_list])
    isolation_score = np.ones(n_clus, dtype=float)*(-1)
    isolation_score[clus_labels-1] = isolation_score_short
    noise_overlap_score = np.ones(n_clus, dtype=float)*999
    noise_overlap_score[clus_labels-1] = noise_overlap_score_short
    peak_snr = np.ones(n_clus, dtype=float)*(-1)
    peak_snr[clus_labels-1] = peak_snr_short
    ret['peak_snr'] = peak_snr
    ret['isolation_score'] = isolation_score
    ret['noise_overlap_score'] = noise_overlap_score
    # template_waveforms = template_waveforms[:,20:-20,:]
    n_ch = template_waveforms.shape[0]
    waveform_len = template_waveforms.shape[1]
    # print(waveform_len); exit(0)
    template_peaks = np.max(template_waveforms, axis=1)
    template_troughs = np.min(template_waveforms, axis=1)
    template_p2ps = template_peaks - template_troughs
    print("template_p2ps shape <should be (n_ch,n_clus)>:", template_p2ps.shape)
    # peak_amplitudes = np.max(template_p2ps, axis=0)

    # get spike stamp for all clusters (in SAMPLEs not seconds)
    spike_times_all = firings[1,:]
    spike_labels = firings[2,:]
    spike_times_by_clus =[[] for _ in range(n_clus)]
    spike_count_by_clus = np.zeros((n_clus,))
    for spk_time, spk_lbl in zip(spike_times_all, spike_labels):
        spike_times_by_clus[spk_lbl-1].append(spk_time-1)
    for i in range(n_clus):
        spike_times_by_clus[i] = np.array(spike_times_by_clus[i])
        spike_count_by_clus[i] = spike_times_by_clus[i].shape[0]
    firing_rates = spike_count_by_clus/const_SEGMENT_LEN
    ret['firing_rates'] = firing_rates

    # get primayr channel; channel index starts from 0 here
    pri_ch_lut = -1 * np.ones(n_clus, dtype=int)
    template_peaks_single_sided = np.max(np.abs(template_waveforms), axis=1) # (n_ch, n_clus)
    pri_ch_lut = np.argmax(template_peaks_single_sided, axis=0) # (n_clus)
    peak_amplitudes = template_p2ps[pri_ch_lut, np.arange(n_clus)]
    ret['peak_amplitudes'] = peak_amplitudes
    ret['prim_chs'] = pri_ch_lut
    ret['accept_mask'] = np.load(os.path.join(postproc_path, "cluster_rejection_mask.npz"))['single_unit_mask']
    ret['locations'] = pd.read_csv(os.path.join(postproc_path, "clus_locations.csv"), header=None).values 
    
    return ret



def add_seg_to_axis(ax : matplotlib.axes.Axes, ix_offset : int, pp_dict : dict, geom : np.ndarray, draw_rejected : bool):
    
    # read data from dict and only keep units that have spikes during the segment
    peak_snr = pp_dict['peak_snr']
    n_clus = peak_snr.shape[0]
    spiking_mask = (peak_snr>=0)
    n_spiking = np.sum(spiking_mask)
    locations = pp_dict['locations'][spiking_mask]
    accept_mask = pp_dict['accept_mask'][spiking_mask]
    firing_rates = pp_dict['firing_rates'][spiking_mask]
    peak_amplitudes = pp_dict['peak_amplitudes'][spiking_mask]
    # clus_labels_spiking = np.arange(1, n_clus+1)[spiking_mask]
    reject_mask = (accept_mask==False)
    print("#accepted / #spiking = %d/%d  " % (np.sum(accept_mask), n_spiking))
    # add channels and a bounding box
    for i_ch in range(geom.shape[0]):
        ax.add_patch(plt.Circle((geom[i_ch,0]+ix_offset*SEG_SPACING, geom[i_ch,1]), ELECTRODE_RADIUS, edgecolor='k', fill=False))
    ax.add_patch(plt.Rectangle((ix_offset*SEG_SPACING-80, -60), 180, 760, edgecolor='k', fill=False))
    ax.text(ix_offset*SEG_SPACING, 720, str(np.sum(accept_mask)), va="center", ha="center")
    # prepare to plot accepted single-units
    const_amp_min = 0 # peak_amplitudes[cluster_accept_mask].min()
    const_amp_max = 250 # peak_amplitudes[cluster_accept_mask].max()
    # print(np.argmax(peak_amplitudes)+1, np.max(peak_amplitudes))
    # print(np.where(peak_amplitudes==const_amp_max))
    lambda_smap = lambda x: 200 + (x-const_amp_min)/(const_amp_max-const_amp_min)*1000
    # now plot
    scatter_accepted = ax.scatter(
        locations[accept_mask][:, 0]+ix_offset*SEG_SPACING, locations[accept_mask][:, 1], 
        marker='.', alpha=.4, 
        c=firing_rates[accept_mask], vmin=0, vmax=15, cmap='jet',
        s=lambda_smap(peak_amplitudes[accept_mask])
    )

    # plot rejected units
    if draw_rejected:
        ax.scatter(
            locations[reject_mask][:, 0]+ix_offset*SEG_SPACING, locations[reject_mask][:, 1], 
            marker='x', color='k'
        )

    return spiking_mask, lambda_smap, scatter_accepted


def plot_main(postproc_folder : str, draw_tracking : bool, draw_rejected : bool):

    segment_folders = os.listdir(postproc_folder)
    segment_folders = sorted(segment_folders, key=get_segment_index)
    segment_folders = [os.path.join(postproc_folder, seg_folder) for seg_folder in segment_folders]
    geom = pd.read_csv(MAP_PATH, header=None).values
    
    locations_across_segs = []
    spiking_masks_across_segs = []
    accept_masks_across_segs = []
    peak_amplitudes_across_segs = [] # p2p

    fig = plt.figure(figsize=(70, 10))
    gs_ovr = gridspec.GridSpec(10, 70)
    ax = fig.add_subplot(gs_ovr[:, :64])
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    for i_seg, foldername in enumerate(segment_folders[:-1]):
        print(i_seg)
        pp_dict = read_postproc_data(foldername, os.path.join(foldername, "customPostproc220509"))
        spiking_mask, lambda_smap, scatter_accepted = add_seg_to_axis(ax, i_seg, pp_dict, geom, draw_rejected)
        locations_across_segs.append(pp_dict['locations'])
        spiking_masks_across_segs.append(spiking_mask)
        accept_masks_across_segs.append(pp_dict['accept_mask'])
        # peak_amplitudes_across_segs.append(pp_dict['peak_amplitudes'][np.logical_and(pp_dict['accept_mask'], pp_dict['peak_snr']>0)])
        peak_amplitudes_across_segs.append(pp_dict['peak_amplitudes'][pp_dict['peak_snr']>0])
    
    # color bar
    ax_cbar = fig.add_subplot(gs_ovr[:,64])
    cbar = fig.colorbar(scatter_accepted, ax=ax_cbar, cax=ax_cbar)
    cbar.ax.tick_params(labelsize=30)

    # size legends
    ax_legend = fig.add_subplot(gs_ovr[:, 67:])
    amps4leg = np.array([50, 100, 150, 200])
    labels = ["${:.0f}\\mu V$".format(amp) for amp in amps4leg]
    handles = [ax.scatter([],[],s=lambda_smap(amp), marker='.', c='gray', alpha=0.5, edgecolor='k') for amp in amps4leg]
    ax_legend.legend(handles=handles, labels=labels, labelspacing=2)
    ax_legend.axis("off")

    # stack list of ndarrays (n_clus,) into (n_clus, n_segs)
    locations_across_segs = np.stack(locations_across_segs, axis=-1) # (n_clus, 2, n_segs)
    spiking_masks_across_segs = np.stack(spiking_masks_across_segs, axis=-1) # (n_clus, n_segs)
    accept_masks_across_segs = np.stack(accept_masks_across_segs, axis=-1) # (n_clus, n_segs)
    n_clus, n_segs = accept_masks_across_segs.shape

    if draw_rejected:
        draw_masks = spiking_masks_across_segs
    else:
        draw_masks = np.logical_and(accept_masks_across_segs, spiking_masks_across_segs)

    # draw lines
    if draw_tracking:
        colors = [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,.5,.5), (.5,.5,1), (1,0,1), (1,.5,1)]
        for i_clus in np.arange(n_clus):
            this_locations = locations_across_segs[i_clus, :, :]
            this_draw_mask = draw_masks[i_clus, :]
            for i_seg in range(n_segs-1):
                if this_draw_mask[i_seg] and this_draw_mask[i_seg+1]:
                    plt_x0 = this_locations[0, i_seg]+i_seg*SEG_SPACING
                    plt_x1 = this_locations[0, i_seg+1]+(i_seg+1)*SEG_SPACING
                    plt_y0 = this_locations[1, i_seg]
                    plt_y1 = this_locations[1, i_seg+1]
                    ax.plot([plt_x0, plt_x1], [plt_y0, plt_y1], color=colors[i_clus%len(colors)], alpha=0.8, linewidth=1.2)

    # plt.tight_layout()
    plt.subplots_adjust(left=0.02, right=0.98)
    plt.savefig("tmp.png")
    plt.close()

    plt.rcParams['font.size']=12
    plt.figure()
    peak_amps_all = np.concatenate(peak_amplitudes_across_segs)
    plt.hist(peak_amps_all, bins=np.linspace(0, 250, 26), color='k')
    plt.xticks(np.linspace(0, 250, 13), [str(int(x)) for x in np.linspace(0,250,13)])
    plt.savefig("tmp_amp_hist_before_curation.png")
    plt.close()
plot_main(cont_root_path, draw_tracking=False, draw_rejected=False)
