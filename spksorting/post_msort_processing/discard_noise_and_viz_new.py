""" Automatically discard noise clusters conservatively (by amplitude, spatial spread, ISI violation ratio) and viz"""
#%%
import json
import os
from time import time
from copy import deepcopy
import gc
from collections import OrderedDict
import multiprocessing

import shutil # rmtree
import numpy as np
# from scipy.io import loadmat
import matplotlib; matplotlib.use("agg")
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import scipy.signal as signal


from utils.read_mda import readmda

# settings
ADJACENCY_RADIUS_SQUARED = 140**2 # um^2, consistent with mountainsort shell script
RASTER_PLOT_AMPLITUDE = False
FIGDIRNAME = "figs_allclus_waveforms"
POSTPROC_FOLDER = "postproc_230611"
# MAP_PATH0="../geom_channel_maps/map.csv"
# MAP_PATH1="../geom_channel_maps/map_corolla24ch.csv"
MAP_PATH0="../geom_channel_maps/ChannelMap_Ben.csv"
# MAP_PATH1="../geom_channel_maps/ChannelMap_Ben.csv"

WINDOW_LEN_IN_SEC = 10 # 30e-3
SMOOTHING_SIZE = 1
NO_READING_FILTMDA = False
N_PROCESSES = 8

PARAMS = {}
PARAMS['ADJACENCY_RADIUS_SQUARED'] = ADJACENCY_RADIUS_SQUARED
PARAMS['SNR_THRESH'] = 1.5
PARAMS['AMP_THRESH'] = 50
PARAMS['REFRAC_VIOLATION_RATIO_THRESH']=0.07#0.01
PARAMS['FIRING_RATE_THRESH'] = 0.05
PARAMS['NOISE_OVERLAP_THRESH'] = 0.3
PARAMS['P2P_SPREAD_RATIO'] = 0.4
PARAMS['P2P_SPREAD_CHS_ALLOWED'] = 0
PARAMS['FIRING_RATE_WINDOW_LEN_IN_SEC'] = WINDOW_LEN_IN_SEC
PARAMS['FIRING_RATE_SMOOTHING_SIZE'] = SMOOTHING_SIZE
# PARAMS['NATIVE_ORDERS'] = native_orders.tolist()
PARAMS['NOTES'] = "Reject bursting children and postive spikes"


ELECTRODE_RADIUS = 12.5
GW = 25
GH = 40

def single_cluster_firing_rate_series(firing_stamp, f_sample):
    n_samples = firing_stamp[-1]+10
    window_in_samples = int(WINDOW_LEN_IN_SEC*f_sample)
    n_windows = int(np.ceil(n_samples/window_in_samples))
    bin_edges = np.arange(0, window_in_samples*n_windows+1, step=window_in_samples)
    tmp_hist, _ = np.histogram(firing_stamp, bin_edges)
    tmp_hist = tmp_hist / WINDOW_LEN_IN_SEC
    smoother = signal.windows.hamming(SMOOTHING_SIZE)
    smoother = smoother / np.sum(smoother)
    firing_rate_series = signal.convolve(tmp_hist, smoother, mode='same')
    return firing_rate_series

def postprocess_one_session(session_folder):
    
    """ main function for post processing and visualization"""
    
    ### read clustering metrics file and perform rejection TODO improve rejection method; current version SUCKS
    print(session_folder)
    
    postprocpath = os.path.join(session_folder, POSTPROC_FOLDER)
    if os.path.exists(postprocpath):
        print("Deleting previous rejection results")
        if os.path.isfile(postprocpath) or os.path.islink(postprocpath):
            os.unlink(postprocpath)
        else:
            shutil.rmtree(postprocpath)
    os.makedirs(postprocpath)
    figpath = os.path.join(postprocpath, FIGDIRNAME)
    if not os.path.exists(figpath):
        os.makedirs(figpath)

    # record postproc parameters
    with open(os.path.join(session_folder, "session_rhd_info.json"), 'r') as f:
        f_sample = json.load(f)['sample_freq']

    PARAMS['F_SAMPLE'] = f_sample

    # record postproc parameters
    with open(os.path.join(postprocpath, "log_postproc_params.json"), 'w') as f:
        json.dump(PARAMS, f)

    # load mountainsort metrics
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
    n_ch = template_waveforms.shape[0]
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
        isi = np.diff(spike_times_by_clus[i_clus]) / f_sample * 1000 # ISI series in millisec
        isi_hist_this, _ = np.histogram(isi, bins=isi_bin_edges)
        isi_hists[i_clus, :] =isi_hist_this
        refrac_violation_ratio[i_clus] = (isi_hist_this[0]+isi_hist_this[1]) / isi.shape[0]

    # calculate firing rates
    print("Calculating firing rates for each cluster")
    firing_rate_series_by_clus = [single_cluster_firing_rate_series(spike_times, f_sample) for spike_times in spike_times_by_clus]


    # reject clusters by average amplitude, ISI violation ratio, and spatial spread(from template amplitude at each channel)
    cluster_accept_mask = np.ones((n_clus,), dtype=bool)
    # reject by peak snr
    snr_thresh = PARAMS['SNR_THRESH']
    cluster_accept_mask[peak_snr<snr_thresh] = False
    print("%d/%d clusters kept after peak SNR screening"%(np.sum(cluster_accept_mask), n_clus))
    # reject by spike amplitude
    amp_thresh = PARAMS['AMP_THRESH'] # in uV
    cluster_accept_mask[peak_amplitudes < amp_thresh] = False
    # reject by firing rate
    cluster_accept_mask[firing_rates < PARAMS['FIRING_RATE_THRESH']] = False
    # reject by noise overlap
    cluster_accept_mask[noise_overlap_score > PARAMS['NOISE_OVERLAP_THRESH']] = False
    print("%d/%d clusters kept after amplitude screening"%(np.sum(cluster_accept_mask), n_clus))
    # reject by spatial spread of less than the designated ADJACENT_RADIUS_SQUARED
    if n_ch==32:
        geom = pd.read_csv(MAP_PATH0, header=None).values
    else:
        geom = pd.read_csv(MAP_PATH1, header=None).values
    tmp_clus_ids = np.arange(n_clus)[cluster_accept_mask]
    for i_clus in tmp_clus_ids:
        prim_ch = pri_ch_lut[i_clus]
        prim_x, prim_y = geom[prim_ch, :]
        p2p_by_channel = template_p2ps[:, i_clus]
        p2p_prim = np.max(p2p_by_channel)
        p2p_near = p2p_by_channel>PARAMS['P2P_SPREAD_RATIO']*p2p_prim
        if np.sum( ( ((geom[p2p_near,0]-prim_x)**2+(geom[p2p_near,1]-prim_y)**2)>=PARAMS['ADJACENCY_RADIUS_SQUARED'] ) ) > PARAMS['P2P_SPREAD_CHS_ALLOWED']:
            cluster_accept_mask[i_clus] = False
    print("%d/%d clusters kept after spatial-spread screening"%(np.sum(cluster_accept_mask), n_clus))
    # reject by 2ms-ISI violation ratio of 1%
    multi_unit_mask = np.logical_and(cluster_accept_mask, refrac_violation_ratio > PARAMS['REFRAC_VIOLATION_RATIO_THRESH'])
    cluster_accept_mask[multi_unit_mask] = False # cluster_accept_mask indicates single unit clusters
    print("%d/%d clusters kept after ISI screening"%(np.sum(cluster_accept_mask), n_clus))
    np.savez(os.path.join(postprocpath, "cluster_rejection_mask.npz"),\
        single_unit_mask=cluster_accept_mask,
        multi_unit_mask=multi_unit_mask
        )
    
    #!!!!!!!!!!!!!!!!!!!
    ###cluster_accept_mask = np.logical_or(multi_unit_mask, cluster_accept_mask)
    #!!!!!!!!!!!!!!!!!!!

    # estimate cluster locations by center-of-mass in neighborhood electrodes
    clus_coordinates = np.zeros((n_clus, 2))
    for i_clus in range(n_clus):
        # if cluster_accept_mask[i_clus]==False:
        #     continue
        prim_ch = pri_ch_lut[i_clus]
        prim_x, prim_y = geom[prim_ch, :]
        non_neighbor_mask = ((geom[:,0]-prim_x)**2 + (geom[:,1]-prim_y)**2 >= PARAMS['ADJACENCY_RADIUS_SQUARED'])
        weights = template_p2ps[:, i_clus].copy()
        weights[non_neighbor_mask] = 0
        weights = weights / np.sum(weights)
        clus_coordinates[i_clus, :] = np.sum(weights[:,None] * geom, axis=0)
    pd.DataFrame(data=clus_coordinates).to_csv(os.path.join(postprocpath, "clus_locations.csv"), index=False, header=False)
    # get spike waveforms and amplitudes with time
    print("Getting spike waveforms")
    if NO_READING_FILTMDA==False and os.path.exists(os.path.join(session_folder, "filt.mda")):
        spk_amp_series = []
        waveforms_all = [] # only store the real-time waveforms at primary channel for each cluster
        proper_spike_times_by_clus = []
        print("Reading filt.mda ... ", end="")
        filt_signal = readmda(os.path.join(session_folder, "filt.mda")) # heck of a big file
        print("Done")
        filt_signal = filt_signal - np.mean(filt_signal, axis=0) # common average
        print("Subtracted common average potential")
        
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
        final_stamp_time = n_samples_in_signal / f_sample
        del(filt_signal)
        gc.collect()
        print("Saving all waveforms across time for all clusters...")
        waveforms_all_dict = OrderedDict()
        waveforms_all_dict["n_samples_in_signal"] = n_samples_in_signal
        for i_clus in range(n_clus):
            waveforms_all_dict['clus%d'%(i_clus+1)] = waveforms_all[i_clus]
        np.savez(os.path.join(session_folder, "all_waveforms_by_cluster.npz"), **waveforms_all_dict)
    else:
        #final_stamp_time = firings[1,-1] / f_sample
        waveforms_all = []
        spk_amp_series = []
        proper_spike_times_by_clus = []
        tmp = np.load(os.path.join(session_folder, "all_waveforms_by_cluster.npz"))
        final_stamp_time = tmp['n_samples_in_signal'] /f_sample
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
    # exit(0)
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
    # final_stamp_time = firings[1,-1] / f_sample
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
    fig1 = plt.figure(figsize=(9, 16))
    gs_ovr = gridspec.GridSpec(16,9, figure=fig1)
    ax_loc = fig1.add_subplot(gs_ovr[:, :2])
    # ax_loc.scatter(geom[:,0], geom[:,1], s=24, color='blue')
    for i in range(geom.shape[0]):
        ax_loc.add_patch(plt.Circle((geom[i,0], geom[i,1]), ELECTRODE_RADIUS, edgecolor='k', fill=False))
    ax_loc.scatter(\
        clus_coordinates[cluster_accept_mask, 0], clus_coordinates[cluster_accept_mask, 1], \
        marker='.', c=peak_amplitude_ranks[cluster_accept_mask], \
        cmap=cmap, vmin=0, vmax=n_clus, \
        s=smap[peak_amplitude_ranks[cluster_accept_mask]], alpha=.5\
        )
    ax_loc.set_aspect("equal")
    ax_loc.set_xlim(-13, GW+13)
    ax_loc.set_ylim(-20, 640)
    ax_loc.set_xlabel("x-coordinate (um)")
    ax_loc.set_ylabel("y-coordinate (um)")
    ax_loc.invert_yaxis()
    
    # firing rate plot
    for i_ch in range(n_ch):
        x, y = geom[i_ch,:]
        plot_row, plot_col = (int(y/GH)), (int(x/GW))
        ax = fig1.add_subplot(gs_ovr[plot_row, 3+plot_col*2:5+plot_col*2])
        idx_clusters = np.where(pri_ch_lut==i_ch)[0]# list(filter(lambda x: pri_ch_lut[x]==i_ch), np.arange(n_clus))
        for (i_clus_this_ch, idx_clus) in enumerate(idx_clusters):
            if cluster_accept_mask[idx_clus]==False:
                continue
            firing_rate = firing_rate_series_by_clus[idx_clus]
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
    # exit(0)
    #### viz comprehensive plots (including template waveform across channels) for all clusters
    print("Beginning to make plots")
    ts = time()
    fig_size_scale = 1
    def plot_single_cluster(i_clus_plot):
        import matplotlib; matplotlib.use("agg") # use a non-interactive backend
        matplotlib.font_manager._get_font.cache_clear() # clear cache for multi-processing correctness
        y_scale = np.max(np.abs(template_waveforms[:, :, i_clus_plot]))
        fig2 = plt.figure(figsize=(18*fig_size_scale,18*fig_size_scale))
        prim_ch = pri_ch_lut[i_clus_plot] # look up primary channel
        gs_ovr = gridspec.GridSpec(16, 16, figure=fig2)
        
        # plot channel & cluster location viz
        ax_loc_viz = fig2.add_subplot(gs_ovr[:, :4])
        # all channels
        for i in range(geom.shape[0]):
            if i != prim_ch:
                ax_loc_viz.add_patch(plt.Circle((geom[i,0], geom[i,1]), ELECTRODE_RADIUS, edgecolor='k', fill=False))
            else:
                ax_loc_viz.add_patch(plt.Circle((geom[i,0], geom[i,1]), ELECTRODE_RADIUS, edgecolor='orange', fill=False))
        # location of current cluster
        ax_loc_viz.scatter(\
            [clus_coordinates[i_clus_plot,0]], [clus_coordinates[i_clus_plot,1]], \
            marker="x", s=smap[int(peak_amplitude_ranks[i_clus_plot])], color='orange', \
            ) 
        # all clusters
        ax_loc_viz.scatter(\
            clus_coordinates[cluster_accept_mask, 0], clus_coordinates[cluster_accept_mask, 1], \
            marker='.', c=peak_amplitude_ranks[cluster_accept_mask], \
            cmap=cmap, vmin=0, vmax=n_clus, \
            s=smap[peak_amplitude_ranks[cluster_accept_mask]], alpha=.5\
            )
        ax_loc_viz.set_aspect("equal")
        ax_loc_viz.set_xlim(-13, GW+13)
        ax_loc_viz.set_ylim(-20, 640)
        ax_loc_viz.set_xlabel("x-coordinate (um)")
        ax_loc_viz.set_ylabel("y-coordinate (um)")
        ax_loc_viz.invert_yaxis()
        
        # plot waveforms
        # gs_waveforms = gridspec.GridSpecFromSubplotSpec(16, 2, subplot_spec=gs_ovr[:, 4:8]) # syntactically correct?
        for i_ch in range(n_ch):
            x, y = geom[i_ch,:]
            plot_row, plot_col = (int(y/GH)), (int(x/GW))
            ax = fig2.add_subplot(gs_ovr[plot_row, 5+plot_col*2:7+plot_col*2])# plt.subplot(16,2,plot_row*2+plot_col+1)
            ax.plot(\
                np.arange(waveform_len)/f_sample*1000, \
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
        
        # plot spike stamp
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
            np.arange(waveform_len)/f_sample*1000, \
            waveforms_all[i_clus_plot][ids_spikes_to_plot, :].T, \
            color='g', alpha=0.3\
            )
        ax_template.plot(np.arange(waveform_len)/f_sample*1000, \
            template_waveforms[prim_ch, :, i_clus_plot], \
            color='k'
            )
        ax_template.set_ylim(-2.5*y_scale, 2.5*y_scale)
        
        # amplitude histogram
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
        if cluster_accept_mask[i_clus_plot]:
            plt.savefig(os.path.join(figpath, "waveform_clus%d.png"%(clus_labels[i_clus_plot])))
        elif multi_unit_mask[i_clus_plot]:
            plt.savefig(os.path.join(figpath, "z_multiunit_waveform_clus%d.png"%(clus_labels[i_clus_plot])))
        else:
            plt.savefig(os.path.join(figpath, "z_failed_waveform_clus%d.png"%(clus_labels[i_clus_plot])))
        plt.close()
        print(i_clus_plot+1)

    def single_process_plot_func(i_clus_begin, i_clus_end):
        """plot [i_clus_begin, i_clus_end) in a for loop"""
        for i_clus in range(i_clus_begin, i_clus_end):
            plot_single_cluster(i_clus)
    
    n_clus_per_process = int(np.ceil(n_clus/N_PROCESSES))
    processes = []
    i_clus_beg = 0
    for i_proc in range(N_PROCESSES-1):
        i_clus_end = min([i_clus_beg+n_clus_per_process, n_clus])
        processes.append(multiprocessing.Process(target=single_process_plot_func, args=(i_clus_beg, i_clus_end)))
        i_clus_beg += n_clus_per_process
        if i_clus_beg >= n_clus:
            break
    # the last process probably has fewer clusters to process
    if i_clus_beg < n_clus:
        processes.append(multiprocessing.Process(target=single_process_plot_func, args=(i_clus_beg, n_clus)))
    for plot_proc in processes:
        plot_proc.start()
    for plot_proc in processes:
        plot_proc.join()

    print("Plotting done in %f seconds" % (time()-ts))



def one_animal(result_folder):
    error_sessions = []
    relevant_subpaths = list(filter(lambda x: (('_' in x) and ('.' not in x) and ('__' not in x)), os.listdir(result_folder)))
    relevant_fullpaths = list(map(lambda x: os.path.join(result_folder, x), relevant_subpaths))
    for session_folder in filter(lambda x: os.path.isdir(x), relevant_fullpaths):
        try:
            postprocess_one_session(session_folder)
        except Exception as e:
            print("---------------EXCEPTION MESSAGE")
            print(e)
            error_sessions.append({
                "session_folder": session_folder,
                "error_msg": str(e),
            })
    
    print("----#ERROR SESSIONS:", len(error_sessions))
    for error_session in error_sessions:
        print("%s: %s" % (error_session['session_folder'], error_session['error_msg']))

    with open(os.path.join(result_folder, "msg.json"), 'w') as f:
        json.dump(error_sessions, f)



if __name__ == '__main__':
    result_folders = [ 
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
        one_animal(res_folder)