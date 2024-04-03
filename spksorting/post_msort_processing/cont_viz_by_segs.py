"""
Visualize each individual segment of unit metrics chopped from the entire sorted data 
"""
# (1) read filt.mda for how many samples there are
# (2) for each defined segment, create new filt_seg.mda and firings_seg.mda and calculate template and metrics
#     and save template and metrics to a file

import json
import os
import re
from time import time
from copy import deepcopy
import gc
from collections import OrderedDict
import multiprocessing

import numpy as np
# from scipy.io import loadmat
import matplotlib
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import scipy.signal as signal


from utils.mdaio import readmda
from utils.triangulation import compute_monopolar_triangulation
from utils.misc import plt3d_set_axes_equal

# settings
FIGDIRNAME = "postproc_figs"
MAP_PATH="../geom_channel_maps/map.csv"
NO_READING_FILTMDA = False
N_PROCESSES = 8 # multiprocessing option

# rejection parameters
F_SAMPLE = 30e3
ADJACENCY_RADIUS_SQUARED = 140**2 # um^2, consistent with mountainsort shell script
SNR_THRESH = 1.3 # 2
AMP_THRESH = 25 # 50 # uV
FIRING_RATE_THRESH = 0.1# 1 # Hz
NOISE_OVERLAP_THRESH = 0.25
ISOLATION_THRESH = 0.6 # 0.
WINDOW_LEN_IN_SEC = 5 # 30e-3
SMOOTHING_SIZE = 1 # no smoothing
TRANSIENT_AMPLITUDE_VALID_DURATION = 1.5e-3 # seconds (duration of data before and after each spike that we consider when deciding the transient amplitude)
P2P_PROPORTION_THRESH = 0.4
PARAMS = {}
PARAMS['F_SAMPLE'] = F_SAMPLE
PARAMS['ADJACENCY_RADIUS_SQUARED'] = ADJACENCY_RADIUS_SQUARED
PARAMS['SNR_THRESH'] = SNR_THRESH
PARAMS['AMP_THRESH'] = AMP_THRESH
PARAMS['FIRING_RATE_THRESH'] = FIRING_RATE_THRESH
PARAMS['P2P_PROPORTION_THRESH'] = P2P_PROPORTION_THRESH
PARAMS['FIRING_RATE_WINDOW_LEN_IN_SEC'] = WINDOW_LEN_IN_SEC
PARAMS['FIRING_RATE_SMOOTHING_SIZE'] = SMOOTHING_SIZE
PARAMS['TRANSIENT_AMPLITUDE_VALID_DURATION'] = TRANSIENT_AMPLITUDE_VALID_DURATION
PARAMS['NOISE_OVERLAP_THRESH'] = NOISE_OVERLAP_THRESH
PARAMS['ISOLATION_THRESH'] = ISOLATION_THRESH


WINDOW_IN_SAMPLES = int(WINDOW_LEN_IN_SEC*F_SAMPLE)

TAVD_NSAMPLE = int(np.ceil(TRANSIENT_AMPLITUDE_VALID_DURATION*F_SAMPLE))
matplotlib.rcParams['font.size'] = 22


def single_cluster_firing_rate_series(firing_stamp, smooth=True):
    n_samples = int(3600 * F_SAMPLE) # hard set for one hour segments# firing_stamp[-1]+10
    n_windows = int(np.ceil(n_samples/WINDOW_IN_SAMPLES))
    bin_edges = np.arange(0, WINDOW_IN_SAMPLES*n_windows+1, step=WINDOW_IN_SAMPLES)
    tmp_hist, _ = np.histogram(firing_stamp, bin_edges)
    tmp_hist = tmp_hist / WINDOW_LEN_IN_SEC
    # smoother = signal.windows.hamming(SMOOTHING_SIZE)
    if smooth:
        smoother = np.ones(SMOOTHING_SIZE)
        smoother = smoother / np.sum(smoother)
        firing_rate_series = signal.convolve(tmp_hist, smoother, mode='same')
    else:
        firing_rate_series = tmp_hist
    return firing_rate_series

def get_segment_index(segment_name: str) -> int:
    return int(re.search("seg([0-9]+)", segment_name)[1])


def plot_one_unit(
    i_clus_plot, figpath, template_waveforms, pri_ch_lut, geom, clus_coordinates2d, peak_amplitudes, this_mask, firing_rates,
    isi_hists, isi_bin_edges, isi_vis_max, waveforms_all, spike_count_by_clus, isolation_score, noise_overlap_score, 
    refrac_violation_ratio, cluster_accept_mask, multi_unit_mask, frate_series_all
    ):
    """
    A shitty plotting function.
    The plotting code is too long and I want to separated it into a function
    But it still requires a lot of data computed/read from disk early on
    And now here we are
    """
    matplotlib.font_manager._get_font.cache_clear()
    matplotlib.rcParams['font.size'] = 16
    fig_size_scale = 1
    n_ch = template_waveforms.shape[0]
    waveform_len = template_waveforms.shape[1]
    y_scale = np.max(np.abs(template_waveforms[:, :, i_clus_plot]))+1
    fig2 = plt.figure(figsize=(18*fig_size_scale,18*fig_size_scale))
    prim_ch = pri_ch_lut[i_clus_plot] # look up primary channel
    gs_ovr = gridspec.GridSpec(16, 16, figure=fig2)
    const_amp_min = 0 
    const_amp_max = 500 
    # print(np.argmax(peak_amplitudes)+1, np.max(peak_amplitudes))
    lambda_smap = lambda x: 200 + (x-const_amp_min)/(const_amp_max-const_amp_min)*1000
    probe_ELECTRODE_RADIUS = 12.5
    
    # plot locations
    ax_loc = fig2.add_subplot(gs_ovr[:, :4])
    for i_ch in range(geom.shape[0]):
        if pri_ch_lut[i_clus_plot]==i_ch:
            ax_loc.add_patch(plt.Circle((geom[i_ch,0], geom[i_ch,1]), probe_ELECTRODE_RADIUS, edgecolor='orange', fill=False))
            # ax_loc.text(geom[i,0], geom[i,1], str(i))
        else:
            ax_loc.add_patch(plt.Circle((geom[i_ch,0], geom[i_ch,1]), probe_ELECTRODE_RADIUS, edgecolor='k', fill=False))
    sizes_amp = lambda_smap(peak_amplitudes[this_mask])
    sc_plot = ax_loc.scatter(\
        clus_coordinates2d[this_mask, 0], clus_coordinates2d[this_mask, 1], \
        marker='.', c=firing_rates[this_mask], \
        # cmap="seismic", vmin=firing_rates[this_mask].min(), vmax=firing_rates[this_mask].max(), \
        cmap="jet", vmin=0, vmax=30, \
        s=sizes_amp, alpha=.4\
        )
    ax_loc.scatter([clus_coordinates2d[i_clus_plot, 0]], [clus_coordinates2d[i_clus_plot,1]], marker='x', color='orange') # mark current unit
    ax_loc.set_aspect("equal")
    ax_loc.set_xlim(-20, 50) # ax_loc.set_xlim(-13, 38)
    ax_loc.set_ylim(-20, 640)
    ax_loc.set_xlabel("$x-coordinate\ (\\mu m)$")
    ax_loc.set_ylabel("$y-coordinate\ (\\mu m)$")
    ax_loc.invert_yaxis()

    # plot waveforms
    for i_ch in range(n_ch):
        x, y = geom[i_ch,:]
        plot_row, plot_col = int(y/40), (int(x/25))
        ax = fig2.add_subplot(gs_ovr[plot_row, 5+plot_col*2:7+plot_col*2])
        ax.plot(\
            np.arange(waveform_len)/F_SAMPLE*1000, \
            template_waveforms[i_ch, :, i_clus_plot], \
            # label="Coordinate (%d,%d)" % (x, y),\
            # color=cmap(peak_amplitude_ranks[i_clus_plot]) \
            )
        ax.set_ylim(-1*y_scale, y_scale)
        if plot_col==1:
            ax.set_yticks([])
        if plot_row!=15:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time (ms)")

        # ax.legend(fontsize=13)
        # ax.set_title("Coordinate (%d,%d)" % (x, y), fontsize=10)
    
    # plot ISI histogram
    ax_isihist = fig2.add_subplot(gs_ovr[:3, 10:])
    ax_isihist.bar(0.5+np.arange(isi_hists.shape[1]), isi_hists[i_clus_plot,:], width=1.)
    ax_isihist.set_xticks(isi_bin_edges[::10])
    ax_isihist.set_xticklabels(isi_bin_edges[::10])
    ax_isihist.set_xlabel("ISI (ms)")
    ax_isihist.set_ylabel("Count")
    ax_isihist.set_xlim(0, isi_vis_max)
    ax_isihist.set_title("ISI histogram")
    
    # plot Amplitude series
    ax_frate = fig2.add_subplot(gs_ovr[4:7, 10:])
    ax_frate.plot(
        np.arange(frate_series_all[i_clus_plot].shape[0])*WINDOW_LEN_IN_SEC, 
        frate_series_all[i_clus_plot],
        color='k', linewidth=0.5
        )
    ax_frate.set_xlabel("Time (sec)")
    ax_frate.set_ylabel("Firing Rate (Hz)")
    
    # waveforms at primary channel for most events
    if waveforms_all[i_clus_plot].shape[0]>0:
        if waveforms_all[i_clus_plot].shape[0] > 300:
            ids_spikes_to_plot = np.linspace(0, waveforms_all[i_clus_plot].shape[0]-1, 300).astype(int)
        else:
            ids_spikes_to_plot = np.arange(waveforms_all[i_clus_plot].shape[0])
        ax_template = fig2.add_subplot(gs_ovr[8:11, 10:])
        ax_template.plot(\
            np.arange(waveform_len)/F_SAMPLE*1000, \
            waveforms_all[i_clus_plot][ids_spikes_to_plot, :].T, \
            color='g', alpha=0.3\
            )
        ax_template.plot(np.arange(waveform_len)/F_SAMPLE*1000, \
            template_waveforms[prim_ch, :, i_clus_plot], \
            color='k'
            )

    # print annotations
    ax_text = fig2.add_subplot(gs_ovr[12:, 10:])
    str_annot  = "Cluster label: %d\n" % (1+i_clus_plot)
    str_annot += "Average firing rate: %.2f (Total spike count: %d)\n" % (firing_rates[i_clus_plot], spike_count_by_clus[i_clus_plot])
    str_annot += "Isolation score: %.4f\n" % (isolation_score[i_clus_plot])
    str_annot += "Noise overlap score: %.4f\n" % (noise_overlap_score[i_clus_plot])
    str_annot += "Refractory 2ms violation ratio: %.4f\n" % (refrac_violation_ratio[i_clus_plot])
    str_annot += "Automatic screening: %s\n" % ("passed" if cluster_accept_mask[i_clus_plot] else ("multi-u" if multi_unit_mask[i_clus_plot] else "failed"))
    ax_text.text(0.5, 0.5, str_annot, va="center", ha="center", fontsize=13)

    # plt.suptitle("Cluster %d, kept=%d" % (i_clus_plot+1, clus_keep_mask[i_clus_plot]), fontsize=25)
    # plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(os.path.join(figpath, "waveform_clus%d.png"%(1+i_clus_plot)))
    plt.close()
    

def process_one_segment(segment_folder_msort, segment_folder_custom, curation_params, savefig=True, plot_all_units=False):
    
    """ main function for post processing and visualization
    !!! Different from the case of processing entire session,
    In this case,  one segment may miss entirely the firing of a neuron, causing 
        (1) the metrics.json to be shorter than the true #units
        (2) corresponding position at template.mda to be NaN
    So we keep track of the "clus_labels" from metrics.json
    And reconstruct the metrics of full length. The missing units will be marked -1 isolation and 999 noise overlap
    And corresponding template to be Zero
    """
    
    ### read clustering metrics file and perform rejection 
    const_SEGMENT_LEN = 3600 # seconds
    figpath = os.path.join(segment_folder_custom, FIGDIRNAME)
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    with open(os.path.join(segment_folder_custom, "params_log.json"), 'w') as f:
        json.dump(curation_params, f)
    with open(os.path.join(segment_folder_msort, "combine_metrics_new.json"), 'r') as f:
        x = json.load(f)
    geom = pd.read_csv(MAP_PATH, header=None).values# [:, [1,0]]
    

    # read firing stamps, template and continuous waveforms from MountainSort outputs and some processing
    firings = readmda(os.path.join(segment_folder_msort, "firings_seg.mda")).astype(np.int64)
    template_waveforms = readmda(os.path.join(segment_folder_msort, "templates.mda")).astype(np.float64)
    n_clus = template_waveforms.shape[2]
    # set nan to zero
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

    # get primayr channel; channel index starts from 0 here
    pri_ch_lut = -1 * np.ones(n_clus, dtype=int)
    template_peaks_single_sided = np.max(np.abs(template_waveforms), axis=1) # (n_ch, n_clus)
    pri_ch_lut = np.argmax(template_peaks_single_sided, axis=0) # (n_clus)
    # # get primary channel for each label; safely assumes each cluster has only one primary channel
    # pri_ch_lut = -1 * np.ones(n_clus, dtype=int)
    # n_pri_ch_known = 0
    # for (spk_ch, spk_lbl) in zip(firings[0,:], spike_labels):
    #     if pri_ch_lut[spk_lbl-1]==-1:
    #         pri_ch_lut[spk_lbl-1] = spk_ch-1
    #         n_pri_ch_known += 1
    #         if n_pri_ch_known==n_clus:
    #             break
    
    peak_amplitudes = template_p2ps[pri_ch_lut, np.arange(n_clus)] # (n_clus,)
    
    # get ISI histogram for each cluster
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

    

    # reject clusters by average amplitude, ISI violation ratio, and spatial spread(from template amplitude at each channel)
    cluster_accept_mask = np.ones((n_clus,), dtype=bool)
    # reject by peak snr
    snr_thresh = curation_params['SNR_THRESH']
    cluster_accept_mask[peak_snr<snr_thresh] = False
    print("%d/%d clusters kept after peak SNR screening"%(int(np.sum(cluster_accept_mask)), n_clus))
    # reject by spike amplitude
    amp_thresh = curation_params['AMP_THRESH'] # in uV
    cluster_accept_mask[peak_amplitudes < amp_thresh] = False
    print("%d/%d clusters kept after amplitude screening"%(int(np.sum(cluster_accept_mask)), n_clus))
    # reject by firing rate
    firing_rate_thresh = curation_params['FIRING_RATE_THRESH']
    cluster_accept_mask[firing_rates < firing_rate_thresh] = False
    print("%d/%d clusters kept after firing-rate screening"%(int(np.sum(cluster_accept_mask)), n_clus))
    # reject by nosie overlap
    cluster_accept_mask[noise_overlap_score > curation_params['NOISE_OVERLAP_THRESH']] = False
    print("%d/%d clusters kept after noise-overlap screening"%(int(np.sum(cluster_accept_mask)), n_clus))
    # reject by isolation
    cluster_accept_mask[isolation_score < curation_params['ISOLATION_THRESH']] = False
    print("%d/%d clusters kept after isolation screening"%(int(np.sum(cluster_accept_mask)), n_clus))
    # reject by spatial spread of less than the designated ADJACENT_RADIUS_SQUARED
    
    tmp_clus_ids = np.arange(n_clus)[cluster_accept_mask]
    for i_clus in tmp_clus_ids:
        prim_ch = pri_ch_lut[i_clus]
        prim_x, prim_y = geom[prim_ch, :]
        p2p_by_channel = template_p2ps[:, i_clus]
        p2p_prim = np.max(p2p_by_channel)
        p2p_near = p2p_by_channel > p2p_prim * curation_params['P2P_PROPORTION_THRESH']
        if np.any((geom[p2p_near,0]-prim_x)**2 + (geom[p2p_near,1]-prim_y)**2 >= curation_params['ADJACENCY_RADIUS_SQUARED']):
            cluster_accept_mask[i_clus] = False
    print("%d/%d clusters kept after spatial-spread screening"%(int(np.sum(cluster_accept_mask)), n_clus))
    # reject by 2ms-ISI violation ratio of 1%
    multi_unit_mask = np.logical_and(cluster_accept_mask, refrac_violation_ratio > 0.01)
    cluster_accept_mask[multi_unit_mask] = False # cluster_accept_mask indicates single unit clusters
    print("%d/%d clusters kept after ISI screening"%(int(np.sum(cluster_accept_mask)), n_clus))

    np.savez(os.path.join(segment_folder_custom, "cluster_rejection_mask.npz"),\
        single_unit_mask=cluster_accept_mask,
        multi_unit_mask=multi_unit_mask
        )

    # pd.DataFrame(data=cluster_accept_mask.reshape((1, n_clus)).astype(int)).to_csv("curation_mask.csv", index=False, header=False)
    ### for plotting only - count multi-units
    # cluster_accept_mask = np.logical_or(cluster_accept_mask, multi_unit_mask)
    # estimate cluster locations by center-of-mass in neighborhood electrodes
    ts = time()
    print("Triangulation started ...")
    clus_locs3d = np.ones((n_clus,4), dtype=float)*(-1)
    clus_locs3d[clus_labels-1, :] = compute_monopolar_triangulation(template_waveforms[:,:,clus_labels-1], geom, return_alpha=True)
    print("Triangulation done in %.2f" %(time()-ts))
    clus_coordinates2d = clus_locs3d[:,:2]
    # ptpest = clus_locs3d[:,3] # estimated p2p values of each unit [uV]
    clus_locs3d = clus_locs3d[:,:3]
    pd.DataFrame(data=clus_locs3d).to_csv(os.path.join(segment_folder_custom, "clus_locations.csv"), index=False, header=False)
    
    ###############################################################
    #### processing has finished now prepare for viz
    ###############################################################

    #### FIG 1 & 2: location & its colorbar
    # scatter-size code cluster amplitudes
    const_amp_min = 0 # peak_amplitudes[cluster_accept_mask].min()
    const_amp_max = 500 # peak_amplitudes[cluster_accept_mask].max()
    # print(np.argmax(peak_amplitudes)+1, np.max(peak_amplitudes))
    # print(np.where(peak_amplitudes==const_amp_max))
    lambda_smap = lambda x: 200 + (x-const_amp_min)/(const_amp_max-const_amp_min)*1000
    # lambda_smapback = lambda y: const_amp_min + (y-200)/1000*(const_amp_max-const_amp_min)
    probe_ELECTRODE_RADIUS = 12.5
    fig1 = plt.figure(figsize=(6, 12))
    # ax_loc = fig1.add_subplot(111)
    gs_ovr = gridspec.GridSpec(12,6)
    ax_loc = fig1.add_subplot(gs_ovr[:, :4])
    for i in range(geom.shape[0]):
        ax_loc.add_patch(plt.Circle((geom[i,0], geom[i,1]), probe_ELECTRODE_RADIUS, edgecolor='k', fill=False))
        # ax_loc.text(geom[i,0], geom[i,1], str(i))
    this_mask = cluster_accept_mask
    sizes_amp = lambda_smap(peak_amplitudes[this_mask])
    sc_plot = ax_loc.scatter(\
        clus_coordinates2d[this_mask, 0], clus_coordinates2d[this_mask, 1], \
        marker='.', c=firing_rates[this_mask], \
        # cmap="seismic", vmin=firing_rates[this_mask].min(), vmax=firing_rates[this_mask].max(), \
        cmap="jet", vmin=0, vmax=15, \
        s=sizes_amp, alpha=.4\
        )
    ax_loc.set_aspect("equal")
    ax_loc.set_xlim(-80, 100) # ax_loc.set_xlim(-13, 38)
    ax_loc.set_ylim(-60, 700)
    ax_loc.set_xlabel("$x-coordinate\ (\\mu m)$")
    ax_loc.set_ylabel("$y-coordinate\ (\\mu m)$")
    ax_loc.invert_yaxis()
    ax_legend = fig1.add_subplot(gs_ovr[:5,4:])
    amps4leg = np.array([100, 200, 300, 400])
    labels = ["${:.0f}\\mu V$".format(amp) for amp in amps4leg]
    handles = [ax_loc.scatter([],[],s=lambda_smap(amp), marker='.', c='gray', alpha=0.5, edgecolor='k') for amp in amps4leg]
    ax_legend.legend(handles=handles, labels=labels, labelspacing=2)
    ax_legend.axis("off")
    # if savefig:
    #     fig1.savefig("location.png")
    #     fig1.savefig("location.svg")
    #     plt.close()
    # separate figure for colorbar
    # fig1_cbar = plt.figure(figsize=(4,20))
    # gs_cbar = gridspec.GridSpec(20,4)
    # ax_cbar = fig1_cbar.add_subplot(gs_cbar[:,2])
    # cbar = fig1_cbar.colorbar(sc_plot, ax=ax_cbar, cax=ax_cbar)
    # cbar.ax.tick_params(labelsize=30)
    ax_cbar = fig1.add_subplot(gs_ovr[7:,4:])
    cbar = fig1.colorbar(sc_plot, ax=ax_cbar, cax=ax_cbar)
    cbar.ax.tick_params(labelsize=30)
    # fig1_cbar.subplots_adjust()
    if savefig:
        fig1.savefig(os.path.join(figpath, "location.png"))
        fig1.savefig(os.path.join(figpath, "location.svg"))
        plt.close()

    #### FIGs 3,4,5: plot some waveforms
    plot_edge_cut_samples = 15
    def plot5(i_chs, figname):
        fig = plt.figure(figsize=(2,3))
        gs = gridspec.GridSpec(6, 4)
        axes = [
            fig.add_subplot(gs[:2, 2:]), 
            fig.add_subplot(gs[1:3, :2]),
            fig.add_subplot(gs[2:4, 2:]),
            fig.add_subplot(gs[3:5, :2]),
            fig.add_subplot(gs[4:, 2:])
            ][::-1]
        idx_clusters = list(filter(lambda x: pri_ch_lut[x] in i_chs, np.arange(n_clus)[cluster_accept_mask]))
        print("---", idx_clusters)
        # idx_clusters = idx_clusters[::int(len(idx_clusters)/3)]
        # print("-------", idx_clusters)
        ylim = np.max(abs(template_waveforms[:,:,idx_clusters]))
        cmap = ['#85a832', '#464d40', '#7cfca0', '#063d4f']*5
        for i_ch, ax in zip(i_chs, axes):
            for j, i_clus_plot in enumerate(idx_clusters):
                ax.plot(
                    np.arange(waveform_len-2*plot_edge_cut_samples)/F_SAMPLE*1000, 
                    template_waveforms[i_ch, plot_edge_cut_samples:-plot_edge_cut_samples, i_clus_plot],
                    # color=cmap(peak_amplitude_ranks[i_clus_plot]),
                    color=cmap[j],
                    alpha=.9
                    )
            xmax = ((waveform_len-2*plot_edge_cut_samples)/F_SAMPLE*1000)
            # ylim = 120
            ax.set_xlim(0, xmax)
            ax.set_ylim(-ylim, ylim)
            ax.axis("off")
        axes[-2].axhline(y=ylim, xmin=0, xmax=1/xmax, color='k')
        # ax.text(0,85,"1 ms")
        # axes[1].axvline(x=0, ymin=0.5, ymax=.5+(50/ylim/2), color='k')
        axes[-2].axvline(x=0, ymin=1-(100/ylim/2), ymax=1, color='k') # 50 uV scalebar
        # ax.text(0, 55, "20 uV")
        # plt.tight_layout()
        if savefig:
            plt.savefig(os.path.join(figpath, figname+'.png'))
            plt.savefig(os.path.join(figpath, figname+'.svg'))
            plt.close()
        else: 
            plt.show()
        
    # for spinal cord channel map these are fixed
    # plot5([2, 28, 15, 17, 13], "bot-5clus")
    # plot5([9, 23, 7, 25, 5], "mid-5clus")
    # plot5([8, 22, 6, 24, 4], "top-5clus")

    if not plot_all_units:
        return
    
    ###### Visualize indiviual units
    frate_series_all = [single_cluster_firing_rate_series(spike_times_by_clus[idx_clus]) for idx_clus in range(n_clus)]

    ts_read_all_waveforms = time()
    print("Reading all waveforms...")
    if NO_READING_FILTMDA==False and os.path.exists(os.path.join(segment_folder_msort, "filt_seg.mda")):
        spk_amp_series = []
        waveforms_all = [] # only store the real-time waveforms at primary channel for each cluster
        proper_spike_times_by_clus = []
        filt_signal = readmda(os.path.join(segment_folder_msort, "filt_seg.mda")) # heck of a big file
        
        for i_clus in range(n_clus):
            prim_ch = pri_ch_lut[i_clus]
            # print(spike_times_by_clus[i_clus].shape)
            tmp_spk_stamp = spike_times_by_clus[i_clus].astype(int)
            tmp_spk_stamp = tmp_spk_stamp[(tmp_spk_stamp>=int((waveform_len-1)/2)) & (tmp_spk_stamp<=filt_signal.shape[1]-1-int(waveform_len/2))]
            tmp_spk_start = tmp_spk_stamp - int((waveform_len-1)/2)
            waveforms_this_cluster = deepcopy(filt_signal[prim_ch, tmp_spk_start[:,None]+np.arange(waveform_len)]) # (n_events, n_sample)
            waveforms_all.append(waveforms_this_cluster)
            waveform_peaks = np.max(waveforms_this_cluster, axis=1) 
            waveform_troughs = np.min(waveforms_this_cluster, axis=1)
            tmp_amp_series = (waveform_peaks-waveform_troughs) * (1-2*(waveform_peaks<0))
            # peak-to-peak value of each event
            spk_amp_series.append(tmp_amp_series)
            proper_spike_times_by_clus.append(tmp_spk_stamp)
        n_samples_in_signal = filt_signal.shape[1]
        final_stamp_time = n_samples_in_signal / F_SAMPLE
        del(filt_signal)
        gc.collect()
        print("Saving all waveforms across time for all clusters...")
        waveforms_all_dict = OrderedDict()
        waveforms_all_dict["n_samples_in_signal"] = n_samples_in_signal
        for i_clus in range(n_clus):
            waveforms_all_dict['clus%d'%(i_clus+1)] = waveforms_all[i_clus]
        np.savez(os.path.join(segment_folder_msort, "all_waveforms_by_cluster.npz"), **waveforms_all_dict)
    else:
        waveforms_all = []
        spk_amp_series = []
        proper_spike_times_by_clus = []
        tmp = np.load(os.path.join(segment_folder_msort, "all_waveforms_by_cluster.npz"))
        n_samples_in_signal = tmp['n_samples_in_signal']
        for i_clus in range(n_clus):
            waveforms_this_cluster = tmp['clus%d'%(i_clus+1)]
            prim_ch = pri_ch_lut[i_clus]
            tmp_spk_stamp = spike_times_by_clus[i_clus].astype(int)
            tmp_spk_stamp = tmp_spk_stamp[(tmp_spk_stamp>=int((waveform_len-1)/2))]
            tmp_spk_stamp = tmp_spk_stamp[:waveforms_this_cluster.shape[0]]
            waveforms_all.append(waveforms_this_cluster)
            waveform_peaks = np.max(waveforms_this_cluster, axis=1) 
            waveform_troughs = np.min(waveforms_this_cluster, axis=1)
            tmp_amp_series = (waveform_peaks-waveform_troughs) * (1-2*(waveform_peaks<0))
            # peak-to-peak value of each event
            spk_amp_series.append(tmp_amp_series)
            proper_spike_times_by_clus.append(tmp_spk_stamp)
    
    ts_plot = time()
    print("All waveforms read in %.2f seconds." % (ts_plot-ts_read_all_waveforms))

    def single_process_plot_func(i_clus_begin, i_clus_end):
        """plot [i_clus_begin, i_clus_end) in a for loop"""
        for i_clus_plot in range(i_clus_begin, i_clus_end):
            try:
                plot_one_unit(
                    i_clus_plot, figpath, template_waveforms, pri_ch_lut, geom, clus_coordinates2d, 
                    peak_amplitudes, this_mask, firing_rates, isi_hists, isi_bin_edges, isi_vis_max, 
                    waveforms_all, spike_count_by_clus, isolation_score, noise_overlap_score, 
                    refrac_violation_ratio, cluster_accept_mask, multi_unit_mask, frate_series_all
                    )
                print(i_clus_plot+1)
            except:
                print(i_clus_plot+1, "FAILED PLOTTING")
    if N_PROCESSES==1:
        single_process_plot_func(0, n_clus)
    else:
        n_clus_per_process = int(np.ceil(n_clus/N_PROCESSES))
        processes = []
        i_clus_beg = 0
        for i_proc in range(N_PROCESSES-1):
            processes.append(multiprocessing.Process(target=single_process_plot_func, args=(i_clus_beg, i_clus_beg+n_clus_per_process)))
            i_clus_beg += n_clus_per_process
        # the last process probably has fewer clusters to process
        processes.append(multiprocessing.Process(target=single_process_plot_func, args=(i_clus_beg, n_clus)))
        for plot_proc in processes:
            plot_proc.start()
        for plot_proc in processes:
            plot_proc.join()
    print("All units plotted in %.2f seconds." % (time()-ts_plot))
    
if __name__ == '__main__':
    postproc_folder = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/MustangContinuous/Mustang_220126_125248/onePiece/onePieceSegged_1hours"
    segment_folders = os.listdir(postproc_folder)
    segment_folders = sorted(segment_folders, key=get_segment_index)
    segment_folders = [os.path.join(postproc_folder, seg_folder) for seg_folder in segment_folders]
    for segment_folder in segment_folders[:-1]:
        print(segment_folder)
        segment_folder_custom = os.path.join(segment_folder, "customPostproc220509")
        os.makedirs(segment_folder_custom, exist_ok=True)
        process_one_segment(segment_folder, segment_folder_custom, PARAMS, savefig=True, plot_all_units=False)
        # break
    # segment_folder = os.path.join(postproc_folder, "seg0")
    # print(segment_folder)
    # segment_folder_custom = os.path.join(segment_folder, "customPostproc220506")
    # os.makedirs(segment_folder_custom, exist_ok=True)
    # process_one_segment(segment_folder, segment_folder_custom, PARAMS, savefig=True)