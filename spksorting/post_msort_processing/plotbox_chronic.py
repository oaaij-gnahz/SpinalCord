import os
from copy import deepcopy
from collections import OrderedDict
from itertools import groupby
import warnings

import numpy as np
import matplotlib.pyplot as plt

def read_chronic_animal_npz(npzpath, keep_valid_only=True):
    # TODO process keep_valid_only argument
    npzdict = np.load(npzpath, allow_pickle=True)
    ddict = dict(npzdict.items())

    if keep_valid_only:
        assert np.all(np.diff(ddict["valid_timedelta_floats"])>0) # assert that the sessions are already sorted by datetime
        valid_session_ids = np.where(np.array(ddict['units_per_channel'])>0)[0]
        ddict["day_after_surgery"] = (ddict['valid_timedelta_floats']/(24*3600)).astype(int)
        for kn, v in ddict.items():
            if len(v) > len(valid_session_ids):
                ddict[kn] = v[valid_session_ids]
    else: 
        assert np.all(np.diff(ddict["timedelta_floats"])>0) # assert that the sessions are already sorted by datetime
        ddict["day_after_surgery"] = (ddict['timedelta_floats']/(24*3600)).astype(int)
    # if "units_per_channel" in ddict:
        # warnings.warn()
    # ddict["units_per_channel"] = ddict["n_sing_or_mult"]/ddict["n_ch"] 
    return ddict

def truncate_timeseries_in_dict(animal_dict, first_day, last_day):
    daysstamp = animal_dict["day_after_surgery"]
    mask_inds = (daysstamp>=first_day) & (daysstamp<=last_day)
    newdict = {}
    for kname, timeseries in animal_dict.items():
        newdict[kname] = timeseries[mask_inds]
    return newdict

def custom_curation(animal_name, animal_dict):
    # just switch (animal_name) and pour in shit code for curation
    pass

def group_by_week(animal_dict, duration_in_weeks):
    if duration_in_weeks is not None:
        animal_dict = truncate_timeseries_in_dict(animal_dict, 0, duration_in_weeks*7-1)
    daysstamp = animal_dict["day_after_surgery"]
    n_sessions = len(daysstamp)
    sess_inds = list(range(n_sessions))
    sess_inds_grouped = [[] for _ in range(duration_in_weeks)]
    for week, inds in groupby(sess_inds, key=lambda k: (daysstamp[k]//7)):
        sess_inds_grouped[week] = list(inds)
    # print("DBG sess_inds_grouped", sess_inds_grouped)
    # print("DBG days_grouped", [[daysstamp[i] for i in inds] for inds in sess_inds_grouped])
    # n_weeks_actual = len(sess_inds_grouped)
    # if (duration_in_weeks is not None) and (n_weeks_actual<duration_in_weeks):
    #     sess_inds_grouped.extend([[] for _ in range(duration_in_weeks-n_weeks_actual)])
    animal_dict_grouped = {}
    for kname, timeseries in animal_dict.items():
        # print("DBG", n_sessions, timeseries.shape)
        animal_dict_grouped[kname] = [timeseries[inds] for inds in sess_inds_grouped]
    return animal_dict_grouped

def box_plot_weekly(dict_animals, duration_in_weeks, ax=None):
    """make a boxplot of unit yield for given animals; x-axis it in days """
    # first group the sessions by week for each animal
    dict_animals_gbw = {}
    animal_names = list(dict_animals.keys())
    for animal_name, animal_dict in dict_animals.items():
        dict_animals_gbw[animal_name] = group_by_week(animal_dict, duration_in_weeks)
    datasets = []
    for i_week in range(duration_in_weeks):
        datasets.append(np.concatenate([ad["units_per_channel"][i_week] for ad in dict_animals_gbw.values()]))
        # print("DBG datasets[-1].shape", datasets[-1].shape)
    # now plot the datasets
    if ax is None:
        _, ax = plt.subplots()
    ax.boxplot(datasets, positions=3.5+np.arange(duration_in_weeks)*7, widths=4, showfliers=False)
    return ax

def scatter_datapoints(dict_animals, duration_in_days, ax=None):
    dict_animals_t = {}
    for animal_name, animal_dict in dict_animals.items():
        dict_animals_t[animal_name] = truncate_timeseries_in_dict(animal_dict, 0, duration_in_days-1)
    if ax is None:
        _, ax = plt.subplots()
    for animal_name, animal_dict in dict_animals_t.items():
        ax.scatter(animal_dict["day_after_surgery"], animal_dict["units_per_channel"], label=animal_name, s=5)
    # plt.legend()
    return ax


if __name__ == "__main__":
    PLOTFOLDER = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/codes/_pls_ignore_niceplots_230614"
    FOLDER = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/codes/_pls_ignore_chronic_data_230614"
    N_WEEKS = 12

    if not os.path.exists(PLOTFOLDER):
        os.makedirs(PLOTFOLDER)
    
    npznames = list(filter(lambda x: x.endswith("_firings.npz"), os.listdir(FOLDER)))
    dict_animals = {}
    for npzname in npznames:
        aname = npzname.split("_")[0]
        npzfullpath = os.path.join(FOLDER, npzname)
        dict_animals[aname] = read_chronic_animal_npz(npzfullpath)

    group_nice = ["BenMouse0", "BenMouse1", "nora", "mustang", "nacho"]
    group_meeh = list(set(dict_animals.keys()) - set(group_nice))
    animal_groups = [
        group_nice,
        group_meeh
    ]
    group_names = ["gud", "meh"]

    # all animals
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(111)
    box_plot_weekly(dict_animals, duration_in_weeks=N_WEEKS, ax=ax)
    scatter_datapoints(dict_animals, duration_in_days=N_WEEKS*7, ax=ax)
    ax.set_xlim([-1, N_WEEKS*7])
    ax.set_ylim([0, 2.5])
    ax.set_xticks(np.arange(N_WEEKS)*7+3.5)
    ax.set_xticklabels(np.arange(N_WEEKS)+1)
    ax.set_xlabel("Week")
    ax.set_ylabel("#Units per channel")
    plt.savefig(os.path.join(PLOTFOLDER, "unit_yield_all.png"))
    plt.savefig(os.path.join(PLOTFOLDER, "unit_yield_all.svg"))
    plt.show()

    # for each group of animals
    for group_name, animal_group in zip(group_names, animal_groups):
        dict_animals_subset = dict((k,dict_animals[k]) for k in dict_animals.keys() if k in animal_group)
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111)
        box_plot_weekly(dict_animals_subset, duration_in_weeks=N_WEEKS, ax=ax)
        scatter_datapoints(dict_animals_subset, duration_in_days=N_WEEKS*7, ax=ax)
        ax.set_xlim([-1, N_WEEKS*7])
        ax.set_ylim([0, 2.5])
        ax.set_xticks(np.arange(N_WEEKS)*7+3.5)
        ax.set_xticklabels(np.arange(N_WEEKS)+1)
        ax.set_xlabel("Week")
        ax.set_ylabel("#Units per channel")
        plt.savefig(os.path.join(PLOTFOLDER, "unit_yield_%s.png"%(group_name)))
        plt.savefig(os.path.join(PLOTFOLDER, "unit_yield_%s.svg"%(group_name)))
        plt.show()