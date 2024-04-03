""" Automatically discard noise clusters conservatively (by amplitude, spatial spread, ISI violation ratio) and viz"""
#%%
import json
import os
from time import time
from copy import deepcopy
import gc
from collections import OrderedDict

import numpy as np
# from scipy.io import loadmat
import matplotlib
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import scipy.signal as signal

# from scipy.special import softmax

from utils.read_mda import readmda

# settings
# N_CH = 24
F_SAMPLE = 30e3
ADJACENCY_RADIUS_SQUARED = 140**2 # um^2, consistent with mountainsort shell script
RASTER_PLOT_AMPLITUDE = False
FIGDIRNAME = "figs_allclus_waveforms_new"
MAP_PATH="../geom_channel_maps/channelmapBEN_notfinal.csv"
# MAP_PATH="../geom_channel_maps/map_corolla24ch.csv"
session_folder = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/DataFromBenPurple003"
WINDOW_LEN_IN_SEC = 10 # 30e-3
SMOOTHING_SIZE = 1
WINDOW_IN_SAMPLES = int(WINDOW_LEN_IN_SEC*F_SAMPLE)
NO_READING_FILTMDA = True

def single_cluster_firing_rate_series(firing_stamp):
    n_samples = firing_stamp[-1]+10
    n_windows = int(np.ceil(n_samples/WINDOW_IN_SAMPLES))
    bin_edges = np.arange(0, WINDOW_IN_SAMPLES*n_windows+1, step=WINDOW_IN_SAMPLES)
    tmp_hist, _ = np.histogram(firing_stamp, bin_edges)
    tmp_hist = tmp_hist / WINDOW_LEN_IN_SEC
    smoother = signal.windows.hamming(SMOOTHING_SIZE)
    smoother = smoother / np.sum(smoother)
    firing_rate_series = signal.convolve(tmp_hist, smoother, mode='same')
    return firing_rate_series

def postprocess_one_session(session_folder):
    
    """ main function for post processing and visualization"""
    
    ### read clustering metrics file and perform rejection TODO improve rejection method; current version SUCKS
    figpath = os.path.join(session_folder, FIGDIRNAME)
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    with open(os.path.join(session_folder, "combine_metrics_new.json"), 'r') as f:
        x = json.load(f)

    clus_metrics_list = x['clusters']
    n_clus = len(clus_metrics_list)
    clus_labels = 1 + np.arange(n_clus)


    firing_rates = np.array([k['metrics']['firing_rate'] for k in clus_metrics_list])
    isolation_score = np.array([k['metrics']['isolation'] for k in clus_metrics_list])
    noise_overlap_score = np.array([k['metrics']['noise_overlap'] for k in clus_metrics_list])
    peak_snr = np.array([k['metrics']['peak_snr'] for k in clus_metrics_list])
    # peak_amplitudes = np.array([k['metrics']['peak_amplitude'] for k in clus_metrics_list])

    # read firing stamps, template and continuous waveforms from MountainSort outputs and some processing
    print("reading firings.mda")
    firings = readmda(os.path.join(session_folder, "firings.mda")).astype(np.int64)
    print("reading templates.mda")
    template_waveforms = readmda(os.path.join(session_folder, "templates.mda")).astype(np.float64)
    N_CH = template_waveforms.shape[0]
    waveform_len = template_waveforms.shape[1]
    template_peaks = np.max(template_waveforms, axis=1)
    template_troughs = np.min(template_waveforms, axis=1)
    template_p2ps = template_peaks - template_troughs
    print("template_p2ps shape <should be (n_ch,n_clus)>:", template_p2ps.shape)
    # peak_amplitudes = np.max(template_p2ps, axis=0)

    # get spike stamp for all clusters (in SAMPLEs not seconds)
    spike_times_all = firings[1,:]
    spike_labels = firings[2,:]
    spike_times_by_clus =[[] for i in range(n_clus)]
    spike_count_by_clus = np.zeros((n_clus,))
    for spk_time, spk_lbl in zip(spike_times_all, spike_labels):
        spike_times_by_clus[spk_lbl-1].append(spk_time-1)
    for i in range(n_clus):
        spike_times_by_clus[i] = np.array(spike_times_by_clus[i])
        spike_count_by_clus[i] = spike_times_by_clus[i].shape[0]

    # get primary channel for each label; safely assumes each cluster has only one primary channel
    pri_ch_lut = -1 * np.ones(n_clus, dtype=int)
    n_pri_ch_known = 0
    for (spk_ch, spk_lbl) in zip(firings[0,:], spike_labels):
        if pri_ch_lut[spk_lbl-1]==-1:
            pri_ch_lut[spk_lbl-1] = spk_ch-1
            n_pri_ch_known += 1
            if n_pri_ch_known==n_clus:
                break
    
    peak_amplitudes = template_p2ps[pri_ch_lut, np.arange(n_clus)] # (n_clus,)
    # get the ranking of clusters by peak amplitude
    peak_amplitudes_argsort = np.argsort(peak_amplitudes)
    peak_amplitude_ranks = np.zeros(n_clus)
    peak_amplitude_ranks[peak_amplitudes_argsort] = np.arange(n_clus) # rank from low to high
    peak_amplitude_ranks = peak_amplitude_ranks.astype(int)
    
    # get ISI histogram for each cluster
    print("Calculating ISI hist for each cluster")
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

    # calculate firing rates
    print("Calculating firing rates for each cluster")
    firing_rate_series_by_clus = [single_cluster_firing_rate_series(spike_times) for spike_times in spike_times_by_clus]


    # reject clusters by average amplitude, ISI violation ratio, and spatial spread(from template amplitude at each channel)
    cluster_accept_mask = np.ones((n_clus,), dtype=bool)
    # reject by peak snr
    snr_thresh = 1.5
    cluster_accept_mask[peak_snr<snr_thresh] = False
    print("%d/%d clusters kept after peak SNR screening"%(np.sum(cluster_accept_mask), n_clus))
    # reject by spike amplitude
    amp_thresh = 50 # in uV
    cluster_accept_mask[peak_amplitudes < amp_thresh] = False
    print("%d/%d clusters kept after amplitude screening"%(np.sum(cluster_accept_mask), n_clus))
    # reject by spatial spread of less than the designated ADJACENT_RADIUS_SQUARED
    geom = pd.read_csv(MAP_PATH, header=None).values
    
    # for Ben's data
    geom[:,1] = -1*geom[:,1]

    tmp_clus_ids = np.arange(n_clus)[cluster_accept_mask]
    for i_clus in tmp_clus_ids:
        prim_ch = pri_ch_lut[i_clus]
        prim_x, prim_y = geom[prim_ch, :]
        p2p_by_channel = template_p2ps[:, i_clus]
        p2p_prim = np.max(p2p_by_channel)
        p2p_near = p2p_by_channel>0.4*p2p_prim
        if np.any((geom[p2p_near,0]-prim_x)**2 + (geom[p2p_near,1]-prim_y)**2 >= ADJACENCY_RADIUS_SQUARED):
            cluster_accept_mask[i_clus] = False
    print("%d/%d clusters kept after spatial-spread screening"%(np.sum(cluster_accept_mask), n_clus))
    # reject by 2ms-ISI violation ratio of 1%
    multi_unit_mask = np.logical_and(cluster_accept_mask, refrac_violation_ratio > 0.01)
    cluster_accept_mask[multi_unit_mask] = False # cluster_accept_mask indicates single unit clusters
    print("%d/%d clusters kept after ISI screening"%(np.sum(cluster_accept_mask), n_clus))
    np.savez(os.path.join(session_folder, "cluster_rejection_mask.npz"),\
        single_unit_mask=cluster_accept_mask,
        multi_unit_mask=multi_unit_mask
        )
    # estimate cluster locations by center-of-mass in neighborhood electrodes
    clus_coordinates = np.zeros((n_clus, 2))
    for i_clus in range(n_clus):
        # if cluster_accept_mask[i_clus]==False:
        #     continue
        prim_ch = pri_ch_lut[i_clus]
        prim_x, prim_y = geom[prim_ch, :]
        non_neighbor_mask = ((geom[:,0]-prim_x)**2 + (geom[:,1]-prim_y)**2 >= ADJACENCY_RADIUS_SQUARED)
        weights = template_p2ps[:, i_clus]
        weights[non_neighbor_mask] = 0
        weights = weights / np.sum(weights)
        clus_coordinates[i_clus, :] = np.sum(weights[:,None] * geom, axis=0)
    
    # get spike waveforms and amplitudes with time
    print("Getting spike waveforms")
    if NO_READING_FILTMDA==False and os.path.exists(os.path.join(session_folder, "filt.mda")):
        spk_amp_series = []
        waveforms_all = [] # only store the real-time waveforms at primary channel for each cluster
        proper_spike_times_by_clus = []
        filt_signal = readmda(os.path.join(session_folder, "filt.mda")) # heck of a big file
        
        for i_clus in range(n_clus):
            # if cluster_accept_mask[i_clus]==False:
            #     continue
            # n_spikes = spike_count_by_clus[i_clus]
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
        np.savez(os.path.join(session_folder, "all_waveforms_by_cluster.npz"), **waveforms_all_dict)
    else:
        final_stamp_time = firings[1,-1] / F_SAMPLE
        waveforms_all = []
        spk_amp_series = []
        proper_spike_times_by_clus = []
        tmp = np.load(os.path.join(session_folder, "all_waveforms_by_cluster.npz"))
        for i_clus in range(n_clus):
            if i_clus%10==0:
                print(i_clus, "/", n_clus)
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
    
    # calculate spike amplitude histogram for each cluster
    # TODO parallelize
    spk_amp_hists = []
    spk_amp_hist_bin_edges = []
    spk_amp_mins = []
    spk_amp_maxs = []
    spk_amp_means = []
    spk_amp_stds = []
    n_bins_amphist_sameclus = 50
    ts_amphist = time()
    for i_clus in range(n_clus):
        peak_amp_hist, amphist_bin_edges = np.histogram(spk_amp_series[i_clus], bins=n_bins_amphist_sameclus)
        spk_amp_hists.append(peak_amp_hist)
        spk_amp_hist_bin_edges.append(amphist_bin_edges)
        spk_amp_mins.append(np.min(spk_amp_series[i_clus]))
        spk_amp_maxs.append(np.max(spk_amp_series[i_clus]))
        spk_amp_means.append(np.mean(spk_amp_series[i_clus]))
        spk_amp_stds.append(np.std(spk_amp_series[i_clus]))
    print("Amplitude statistics computation time", time()-ts_amphist)


    
    #%% viz
    ################################## VISUALIZATION
    # plt.figure()
    # plt.hist(peak_amplitudes, rwidth=0.9)
    # plt.xlabel("Amplitude (uV)")
    # plt.ylabel("Neuron count")
    # plt.savefig(os.path.join(figpath, "amplitude_hist.png"))
    # plt.close()
    final_stamp_time = firings[1,-1] / F_SAMPLE
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    nbins_amphist = 20
    peak_amp_hist, amphist_bin_edges = np.histogram(peak_amplitudes, bins=nbins_amphist)
    amphist_binwidth = amphist_bin_edges[1]-amphist_bin_edges[0]
    barplot_x_coordinates = (amphist_bin_edges[:-1] + amphist_bin_edges[1:])/2
    ax.bar(np.arange(nbins_amphist)+0.5, peak_amp_hist, width=1)
    ax.axvline((amp_thresh-amphist_bin_edges[0])/amphist_binwidth, color='red')
    ax.set_xticks(np.arange(nbins_amphist)[::2]+0.5)
    ax.set_xticklabels(barplot_x_coordinates.astype(int)[::2], fontsize=7)
    # ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.set_xlabel("Amplitude (uV)", fontsize=7)
    ax.set_ylabel("Neuron count", fontsize=7)
    text_str = "Mean=%.2fuV\nMin=%.2fuV\nMax=%.2fuV\nBinwidth=%.2fuV" % ( \
        np.mean(peak_amplitudes), np.min(peak_amplitudes), np.max(peak_amplitudes), amphist_binwidth)
    ax.text(ax.get_xlim()[1]*0.7, ax.get_ylim()[1]*0.7, text_str, fontsize=10)
    plt.savefig(os.path.join(figpath, "amplitude_hist.png"))
    plt.close()
    
    # color code cluster amplitudes
    cmap = get_cmap("viridis", n_clus)

    # scatter-size code cluster amplitudes
    smap = np.logspace(np.log10(10), np.log10(80), num=n_clus) * 20

    #### viz cluster locations
    fig1 = plt.figure(figsize=(14, 16))
    gs_ovr = gridspec.GridSpec(16,14, figure=fig1)
    ax_loc = fig1.add_subplot(gs_ovr[:, :2])
    ax_loc.scatter(geom[:,0], geom[:,1], s=24, color='blue')
    ax_loc.scatter(\
        clus_coordinates[cluster_accept_mask, 0], clus_coordinates[cluster_accept_mask, 1], \
        marker='.', c=peak_amplitude_ranks[cluster_accept_mask], \
        cmap=cmap, vmin=0, vmax=n_clus, \
        s=smap[peak_amplitude_ranks[cluster_accept_mask]], alpha=.5\
        )
    ax_loc.set_aspect("equal")
    ax_loc.set_xlim(-10, 35)
    ax_loc.set_ylim(-20, 640)
    ax_loc.set_xlabel("x-coordinate (um)")
    ax_loc.set_ylabel("y-coordinate (um)")
    # raster plot
    for i_ch in range(N_CH):
        x, y = geom[i_ch,:]
        plot_row, plot_col = (15-int(y/40)), (int(x/25))
        ax = fig1.add_subplot(gs_ovr[plot_row, 3+plot_col*2:5+plot_col*2])
        idx_clusters = np.where(pri_ch_lut==i_ch)[0]# list(filter(lambda x: pri_ch_lut[x]==i_ch), np.arange(n_clus))
        for (i_clus_this_ch, idx_clus) in enumerate(idx_clusters):
            if cluster_accept_mask[idx_clus]==False:
                continue
            tmp_lineconst = np.ones(spike_times_by_clus[idx_clus].shape)
            if RASTER_PLOT_AMPLITUDE:
                ax.scatter(spike_times_by_clus[idx_clus]/F_SAMPLE, spk_amp_series[idx_clus], \
                    c=tmp_lineconst*peak_amplitude_ranks[idx_clus], cmap=cmap, vmin=0, vmax=n_clus, \
                    s=1 \
                    )
            else:
                ax.scatter(spike_times_by_clus[idx_clus]/F_SAMPLE, np.ones(spike_times_by_clus[idx_clus].shape[0])+i_clus_this_ch, \
                    c=tmp_lineconst*peak_amplitude_ranks[idx_clus], cmap=cmap, vmin=0, vmax=n_clus, \
                    s=0.5 \
                    )
        ax.set_xlim(0, final_stamp_time)
        if RASTER_PLOT_AMPLITUDE==False:
            ax.set_ylim(0, idx_clusters.shape[0]+1)
        if plot_col==1 or RASTER_PLOT_AMPLITUDE==False:
            ax.set_yticks([])
        if plot_row!=15:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time (sec)")
    # firing rate plot
    for i_ch in range(N_CH):
        x, y = geom[i_ch,:]
        plot_row, plot_col = (15-int(y/40)), (int(x/25))
        ax = fig1.add_subplot(gs_ovr[plot_row, 9+plot_col*2:11+plot_col*2])
        idx_clusters = np.where(pri_ch_lut==i_ch)[0]# list(filter(lambda x: pri_ch_lut[x]==i_ch), np.arange(n_clus))
        for (i_clus_this_ch, idx_clus) in enumerate(idx_clusters):
            if cluster_accept_mask[idx_clus]==False:
                continue
            firing_rate = single_cluster_firing_rate_series(spike_times_by_clus[idx_clus])
            tmp_lineconst = np.ones(firing_rate.shape[0])
            ax.plot(np.arange(firing_rate.shape[0])*WINDOW_LEN_IN_SEC, firing_rate, \
                color=cmap(peak_amplitude_ranks[idx_clus]), \
                linewidth=0.5, alpha=0.7 \
                )
        ax.set_xlim(0, final_stamp_time)
        if plot_col==1:
            ax.yaxis.set_ticks_position("right")
        if plot_row!=15:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time (sec)")
    
    # plt.savefig(os.path.join(figpath, "location.svg"))
    plt.savefig(os.path.join(figpath, "location1.png"))
    plt.close()
    # exit()
    #### viz comprehensive plots (including template waveform across channels) for all clusters
    print("Beginning to make plots")
    ts = time()
    fig_size_scale = 1
    for i_clus_plot in range(n_clus):
        # if cluster_accept_mask[i_clus_plot]==False:
        #     continue
        y_scale = np.max(np.abs(template_waveforms[:, :, i_clus_plot]))
        fig2 = plt.figure(figsize=(18*fig_size_scale,18*fig_size_scale))
        prim_ch = pri_ch_lut[i_clus_plot] # look up primary channel
        gs_ovr = gridspec.GridSpec(16, 16, figure=fig2)
        
        # plot channel & cluster location viz
        ax_loc_viz = fig2.add_subplot(gs_ovr[:, :4])
        # all channels
        ax_loc_viz.scatter(geom[:,0], geom[:,1], s=24, color='blue')
        # highlight primary channel
        ax_loc_viz.scatter( \
            [geom[prim_ch,0]], [geom[prim_ch,1]], \
            marker='s', s=24, color='orange' \
            )
        # location of current cluster
        ax_loc_viz.scatter(\
            [clus_coordinates[i_clus_plot,0]], [clus_coordinates[i_clus_plot,1]], \
            marker="x", s=smap[int(peak_amplitude_ranks[i_clus_plot])], color='orange', \
            ) 
        # all clusters
        # ax_loc_viz.scatter(\
        #     clus_coordinates[:, 0], clus_coordinates[:, 1], \
        #     marker='.', c=peak_amplitude_ranks, \
        #     cmap=cmap, vmin=0, vmax=n_clus, \
        #     s=smap[peak_amplitude_ranks], alpha=.5\
        #     )
        ax_loc_viz.set_aspect("equal")
        ax_loc_viz.set_xlim(-10, 35)
        ax_loc_viz.set_ylim(-20, 640)
        ax_loc_viz.set_xlabel("x-coordinate (um)")
        ax_loc_viz.set_ylabel("y-coordinate (um)")
        
        # plot waveforms
        # gs_waveforms = gridspec.GridSpecFromSubplotSpec(16, 2, subplot_spec=gs_ovr[:, 4:8]) # syntactically correct?
        for i_ch in range(N_CH):
            x, y = geom[i_ch,:]
            plot_row, plot_col = (15-int(y/40)), (int(x/25))
            # print(plot_row, plot_col)
            ax = fig2.add_subplot(gs_ovr[plot_row, 5+plot_col*2:7+plot_col*2])# plt.subplot(16,2,plot_row*2+plot_col+1)
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
        ax_isihist = fig2.add_subplot(gs_ovr[:2, 10:])
        ax_isihist.bar(0.5+np.arange(n_bins), isi_hists[i_clus_plot,:], width=1.)
        ax_isihist.set_xticks(isi_bin_edges[::10])
        ax_isihist.set_xticklabels(isi_bin_edges[::10])
        ax_isihist.set_xlabel("ISI (ms)")
        ax_isihist.set_ylabel("Count")
        ax_isihist.set_xlim(0, isi_vis_max)
        ax_isihist.set_title("ISI histogram")
        
        # plot Amplitude series
        # ax_ampstamp = fig2.add_subplot(gs_ovr[4:7, 10:])
        # this_stamp = np.array(proper_spike_times_by_clus[i_clus_plot])
        # this_amps = np.array(spk_amp_series[i_clus_plot])
        # for i_time in range(0, this_stamp[-1], 60):
        #     mask_ids_to_plot = np.logical_and(this_stamp>i_time, this_stamp<i_time+60)
        #     current_minute_mean = np.mean(this_amps[mask_ids_to_plot])
        #     ax_ampstamp.scatter(this_stamp[mask_ids_to_plot]/F_SAMPLE, np.ones_like(this_stamp[mask_ids_to_plot])*current_minute_mean, s=1)
        # ax_ampstamp.set_xlim(0, final_stamp_time)
        # ax_ampstamp.set_xlabel("Time (sec)")
        # ax_ampstamp.set_ylabel("Transient amplitude (uV)")
        
        # plot firing rate
        ax_ampstamp = fig2.add_subplot(gs_ovr[3:5, 10:])
        # this_stamp = np.array(proper_spike_times_by_clus[i_clus_plot])
        ax_ampstamp.plot(
            np.arange(firing_rate_series_by_clus[i_clus_plot].shape[0])*WINDOW_LEN_IN_SEC, 
            firing_rate_series_by_clus[i_clus_plot], 
            linewidth=0.5,
            linestyle='-.',
            color='k',
            )
        ax_ampstamp.set_xlim(0, final_stamp_time)
        ax_ampstamp.set_xlabel("Time (sec)")
        ax_ampstamp.set_ylabel("Firing Rate (spikes/sec)")

        # ax_ampstamp.set_ylabel("Transient amplitude (uV)")
        
        # waveforms at primary channel for most events
        if waveforms_all[i_clus_plot].shape[0] > 300:
            ids_spikes_to_plot = np.linspace(0, waveforms_all[i_clus_plot].shape[0]-1, 300).astype(int)
        else:
            ids_spikes_to_plot = np.arange(waveforms_all[i_clus_plot].shape[0])
        ax_template = fig2.add_subplot(gs_ovr[6:9, 10:])
        ax_template.plot(\
            np.arange(waveform_len)/F_SAMPLE*1000, \
            waveforms_all[i_clus_plot][ids_spikes_to_plot, :].T, \
            color='g', alpha=0.3\
            )
        ax_template.plot(np.arange(waveform_len)/F_SAMPLE*1000, \
            template_waveforms[prim_ch, :, i_clus_plot], \
            color='k'
            )
        
        ax_amphist = fig2.add_subplot(gs_ovr[10:13, 10:])
        peak_amp_hist = spk_amp_hists[i_clus_plot]
        amphist_bin_edges = spk_amp_hist_bin_edges[i_clus_plot]
        nbins_amphist = amphist_bin_edges.shape[0]-1
        amp_min = spk_amp_mins[i_clus_plot]
        amp_max = spk_amp_maxs[i_clus_plot]
        amp_mean = spk_amp_means[i_clus_plot]
        amp_std = spk_amp_stds[i_clus_plot]
        amphist_binwidth = amphist_bin_edges[1]-amphist_bin_edges[0]
        barplot_x_coordinates = (amphist_bin_edges[:-1] + amphist_bin_edges[1:])/2
        ax_amphist.bar(np.arange(nbins_amphist)+0.5, peak_amp_hist, width=1)
        ax_amphist.set_xticks(np.arange(nbins_amphist)[::2]+0.5)
        ax_amphist.set_xticklabels(barplot_x_coordinates.astype(int)[::2], fontsize=7)
        # ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        ax_amphist.tick_params(axis='both', which='major', labelsize=8)
        ax_amphist.set_xlabel("Amplitude (uV)", fontsize=7)
        ax_amphist.set_ylabel("Neuron count", fontsize=7)
        text_str = "Mean=%.2fuV\nStd=%.2fuV\nMin=%.2fuV\nMax=%.2fuV\nBinwidth=%.2fuV" % ( \
            amp_mean, amp_std, amp_min, amp_max, amphist_binwidth)
        ax_amphist.text(ax_amphist.get_xlim()[1]*0.7, ax_amphist.get_ylim()[1]*0.7, text_str, fontsize=28)

        # print annotations
        ax_text = fig2.add_subplot(gs_ovr[14:, 10:])
        str_annot  = "Cluster label: %d\n" % (clus_labels[i_clus_plot])
        str_annot += "Average firing rate: %.2f (Total spike count: %d)\n" % (firing_rates[i_clus_plot], spike_count_by_clus[i_clus_plot])
        str_annot += "Isolation score: %.4f\n" % (isolation_score[i_clus_plot])
        str_annot += "Noise overlap score: %.4f\n" % (noise_overlap_score[i_clus_plot])
        str_annot += "Peak SNR: %.4f\n" % (peak_snr[i_clus_plot])
        str_annot += "Refractory 2ms violation ratio: %.4f\n" % (refrac_violation_ratio[i_clus_plot])
        str_annot += "Automatic screening: %s\n" % ("passed" if cluster_accept_mask[i_clus_plot] else ("multi-unit" if multi_unit_mask[i_clus_plot] else "failed"))
        ax_text.text(0.5, 0.5, str_annot, va="center", ha="center", fontsize=13)

        # plt.suptitle("Cluster %d, kept=%d" % (i_clus_plot+1, clus_keep_mask[i_clus_plot]), fontsize=25)
        # plt.tight_layout()
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig(os.path.join(figpath, "waveform_clus%d.png"%(clus_labels[i_clus_plot])))
        plt.close()
        print(i_clus_plot+1)
    print("Plotting done in %f seconds" % (time()-ts))


if __name__ == '__main__':
    postprocess_one_session(session_folder)
    # session_subfolders = list(os.listdir(animal_folder))
    # session_subfolders = list(filter(lambda x: ('.json' not in x) and ('.ignore' not in x) , session_subfolders) )
    # error_sessions = []
    # # session_subfolders = [
    # #     'nacho_awake_210907_204305', \
    # #     'nacho_awake_210908_212657', \
    # #     'nacho_awake_210909_191046', \
    # #     'nacho_awake_210910_211048', \
    # #     'nacho_awake_210915_110424', \
    # #     'nacho_awake_210917_162917', \
    # #     'nacho_awake_210922_130346', \
    # #     'nacho_awake_210928_190207', \
    # #     'nacho_awake_211006_121226', \
    # #     'nacho_awake_211102_154449', \
    # #     'nacho_awake_211109_131436', \
    # #     'nacho_awake_211126_151258', \
    # #     ]
    # for session_subfolder in session_subfolders:
    #     session_folder = os.path.join(animal_folder, session_subfolder)
    #     print("Processing session %s" % (session_folder))
    #     # main_postprocess_and_viz(session_folder)
    #     try:
    #         main_postprocess_and_viz(session_folder)
    #     except Exception as e:
    #         print("---------------EXCEPTION MESSAGE")
    #         print(e)
    #         error_sessions.append({
    #             "session_folder": session_folder,
    #             "error_msg": str(e),
    #         })
    
    # print("----#ERROR SESSIONS:", len(error_sessions))
    # for error_session in error_sessions:
    #     print("%s: %s" % (error_session['session_folder'], error_session['error_msg']))

    # with open(os.path.join(animal_folder, "msg.json"), 'w') as f:
    #     json.dump(error_sessions, f)
    




# %%

