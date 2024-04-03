import os
from copy import deepcopy
from collections import OrderedDict
from itertools import groupby
import warnings

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def read_chronic_animal_csv(csvpath, keep_valid_only=True):
    # TODO process keep_valid_only argument
    # npzdict = np.load(npzpath, allow_pickle=True)
    df = pd.read_csv(csvpath)
    ddict = {}
    ddict["day_after_surgery"] = df['dayAfterSurgery'].values
    ddict["units_per_channel"] = df['n_units'].values / df['n_channels'].values
    
    return ddict


def truncate_timeseries_in_dict(animal_dict, first_day, last_day):
    daysstamp = animal_dict["day_after_surgery"]
    mask_inds = (daysstamp>=first_day) & (daysstamp<=last_day)
    newdict = {}
    for kname, timeseries in animal_dict.items():
        newdict[kname] = timeseries[mask_inds]
    return newdict

def custom_curation(animal_name, animal_dict):
    # just switch-case (animal_name) and pour in shit code for curation
    pass

def group_by_period(animal_dict, duration_in_days, period_in_days):
    animal_dict = truncate_timeseries_in_dict(animal_dict, 0, duration_in_days-1)
    daysstamp = animal_dict["day_after_surgery"]
    n_sessions = len(daysstamp)
    sess_inds = list(range(n_sessions))
    sess_inds_grouped = [[] for _ in range(int(np.ceil(duration_in_days/period_in_days)))]
    for period, inds in groupby(sess_inds, key=lambda k: (daysstamp[k]//period_in_days)):
        sess_inds_grouped[period] = list(inds)
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

def stat_datapoints_by_period(dict_animals, duration_in_days, period_in_days):
    """return mean and standar error (SE) of the datapoints of all given animals by `period_in_days` for the first `duration_in_days`"""
    # first group the sessions by week for each animal
    dict_animals_gbw = {}
    animal_names = list(dict_animals.keys())
    for animal_name, animal_dict in dict_animals.items():
        dict_animals_gbw[animal_name] = group_by_period(animal_dict, duration_in_days, period_in_days)
    n_periods = int(np.ceil(duration_in_days/period_in_days))
    datasets = []
    for i_period in range(n_periods):
        datasets.append(np.concatenate([ad["units_per_channel"][i_period] for ad in dict_animals_gbw.values()]))
        print("DBG datasets[-1].shape", datasets[-1].shape)
    means = np.array([(np.mean(dataset) if dataset.shape[0]>0 else np.nan) for dataset in datasets ])
    stdes = np.array([(np.std(dataset)/np.sqrt(dataset.shape[0]) if dataset.shape[0]>0 else np.nan) for dataset in datasets ])
    nvals = np.array([dataset.shape[0] for dataset in datasets ])
    days_approx = np.arange(0, duration_in_days, period_in_days)#[np.array([dataset.shape[0]>0 for dataset in datasets], dtype=bool)] # a rough count of days as a potential x-axis when plotting
    # ax.boxplot(datasets, positions=3.5+np.arange(duration_in_weeks)*7, widths=4, showfliers=False)
    return means, stdes, days_approx, nvals

def plot_shaded_datapoints(dict_animals, duration_in_days, ax=None):
    dict_animals_t = {}
    for animal_name, animal_dict in dict_animals.items():
        dict_animals_t[animal_name] = truncate_timeseries_in_dict(animal_dict, 0, duration_in_days-1)
    if ax is None:
        _, ax = plt.subplots()
    for animal_name, animal_dict in dict_animals_t.items():
        ax.plot(animal_dict["day_after_surgery"], animal_dict["units_per_channel"], label=animal_name, linewidth=0.3, alpha=0.3, marker='s')
    # plt.legend()
    return ax


if __name__ == "__main__":
    PLOTFOLDER = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/codes/_pls_ignore_niceplots_230614"
        #r"C:\Users\zhang\Rice-Courses\neuroeng_lab\20230523\spinalprobe_chronic_summary"
    FOLDER = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/codes/_pls_ignore_chronic_data_230614"
        #r"C:\Users\zhang\Rice-Courses\neuroeng_lab\20230523\spinalprobe_chronic_csvs_counted"
    N_DAYS = 85 # 15 # 85
    PERIOD_IN_DAYS = 7 # 3 # 7

    if not os.path.exists(PLOTFOLDER):
        os.makedirs(PLOTFOLDER)
    
    csvnames = list(filter(lambda x: x.endswith("_firings.csv"), os.listdir(FOLDER)))
    dict_animals = {}
    for csvname in csvnames:
        aname = csvname.split("_")[0]
        csvfullpath = os.path.join(FOLDER, csvname)
        dict_animals[aname] = read_chronic_animal_csv(csvfullpath)

    group_nice = ["BenMouse0", "BenMouse1", "nora", "mustang", "nacho"]
    group_meeh = list(set(dict_animals.keys()) - set(group_nice))
    animal_groups = [
        group_nice,
        group_meeh
    ]
    group_names = ["gud", "meh"]

    means, stdes, days_approx, nvals = stat_datapoints_by_period(dict_animals, N_DAYS, PERIOD_IN_DAYS)
    tmp_datadict = {
        "day": days_approx,
        "mean": means,
        "Standard Error (STD/sqrt(N))": stdes,
        "N value": nvals
    }
    pd.DataFrame(
        data=tmp_datadict
    ).to_csv(os.path.join(PLOTFOLDER, "unit_yield_all.csv"))
    # all animals
    # fig = plt.figure(figsize=(12,6))
    # ax = fig.add_subplot(111)
    # ax.plot(days_approx+(PERIOD_IN_DAYS-1)/2, means, color='k', marker='s', linewidth=1.5, alpha=1)
    # ax.errorbar(days_approx+(PERIOD_IN_DAYS-1)/2, means, yerr=stdes, fmt='', color='k', capsize=5)
    # plot_shaded_datapoints(dict_animals, duration_in_days=N_DAYS, ax=ax)
    # ax.set_xlim([-1, N_DAYS])
    # ax.set_ylim([0, 2.5])
    # ax.set_xticks(np.arange(N_DAYS))
    # ax.set_xticklabels(np.arange(N_DAYS))
    # ax.set_xlabel("Day")
    # ax.set_ylabel("#Units per channel")
    # plt.savefig(os.path.join(PLOTFOLDER, "unit_yield_daily_all.png"))
    # plt.savefig(os.path.join(PLOTFOLDER, "unit_yield_daily_all.eps"))
    # plt.show()

    # for each group of animals
    for group_name, animal_group in zip(group_names, animal_groups):
        dict_animals_subset = dict((k,dict_animals[k]) for k in dict_animals.keys() if k in animal_group)
        means_s, stdes_s, days_approx_s, nvals_s = stat_datapoints_by_period(dict_animals_subset, N_DAYS, PERIOD_IN_DAYS)
        tmp_datadict = {
            "day": days_approx_s,
            "mean": means_s,
            "Standard Error (STD/sqrt(N))": stdes_s,
            "N value": nvals_s
        }
        pd.DataFrame(
            data=tmp_datadict
        ).to_csv(os.path.join(PLOTFOLDER, "unit_yield_%s.csv"%(group_name)))
        # fig = plt.figure(figsize=(12,6))
        # ax = fig.add_subplot(111)
        # # box_plot_weekly(dict_animals_subset, duration_in_weeks=N_DAYS, ax=ax)
        # # scatter_datapoints(dict_animals_subset, duration_in_days=N_DAYS*7, ax=ax)
        # ax.plot(days_approx_s+(PERIOD_IN_DAYS-1)/2, means_s, color='k', marker='s', linewidth=1.5, alpha=1)
        # ax.errorbar(days_approx_s+(PERIOD_IN_DAYS-1)/2, means_s, yerr=stdes_s, fmt='', color='k', capsize=5)
        # plot_shaded_datapoints(dict_animals_subset, duration_in_days=N_DAYS, ax=ax)
        # ax.set_xlim([-1, N_DAYS])
        # ax.set_ylim([0, 2.5])
        # ax.set_xticks(np.arange(N_DAYS))
        # ax.set_xticklabels(np.arange(N_DAYS))
        # ax.set_xlabel("Day")
        # ax.set_ylabel("#Units per channel")
        # plt.savefig(os.path.join(PLOTFOLDER, "unit_yield_daily_%s.png"%(group_name)))
        # plt.savefig(os.path.join(PLOTFOLDER, "unit_yield_daily_%s.eps"%(group_name)))
        # plt.show()
