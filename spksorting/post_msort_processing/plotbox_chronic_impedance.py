import os
from copy import deepcopy
from collections import OrderedDict
from itertools import groupby
import warnings

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches as patches

print("- matplotlib version:", matplotlib.__version__)
# matplotlib.font_manager.get_font_names()
# flist = matplotlib.font_manager.get_fontconfig_fonts()
# print(flist)
# names = [matplotlib.font_manager.FontProperties(fname=fname, size=22).get_name() for fname in flist]
# print(names)
font = {'style' : 'normal',
        'weight' : 'normal',
        'size'   : 18}

matplotlib.rc('font', **font)

def read_chronic_animal_npz(npzpath):
    # returns impedance records in kOhms.
    npzdict = np.load(npzpath, allow_pickle=True)
    ddict = dict(npzdict.items())

    assert np.all(np.diff(ddict["time_in_seconds"])>0) # assert that the sessions are already sorted by datetime
    ddict["day_after_surgery"] = (ddict['time_in_seconds']/(24*3600)).astype(int)
    impedances_by_sess = [ddict["session%d"%(i)]/1E3 for i in range(len(ddict["day_after_surgery"]))] # list of ndarrays
    for i in range(len(ddict["day_after_surgery"])):
        ddict.pop("session%d"%(i))
    ddict["impedances_all"] = np.array(impedances_by_sess, dtype=np.ndarray)
    ddict["impedances_mean"] = np.array([np.mean(k) for k in impedances_by_sess])

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

def remove_datapoints(animal_dict, inds_to_remove):
    n_sessions = len(animal_dict["day_after_surgery"])
    inds_mask = np.ones(n_sessions).astype(bool)
    inds_mask[inds_to_remove] = 0
    new_dict = {}
    for kname, val in animal_dict.items():
        if isinstance(val, np.ndarray):
            new_dict[kname] = val[inds_mask]
        else:
            new_dict[kname] = [v for i, v in enumerate(val) if inds_mask[i]]
    return new_dict


def custom_curation(animal_name, animal_dict):
    # just switch (animal_name) and pour in shit code for curation
    if animal_name.lower()=="tacoma":
        animal_dict = remove_datapoints(animal_dict, [-1])
    elif animal_name.lower()=="yogurt":
        animal_dict = remove_datapoints(animal_dict, [-1,-2,-3])
    return animal_dict

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
        print("DBG boxplot animal_name", animal_name)
        dict_animals_gbw[animal_name] = group_by_week(animal_dict, duration_in_weeks)
    datasets = []
    for i_week in range(duration_in_weeks):
        tmp_datasets_all_animals = [np.array([])]# [np.concatenate(ad["impedances_all"][i_week].tolist()) for ad in dict_animals_gbw.values()]
        for adict in dict_animals_gbw.values():
            datasets_this_animal = adict["impedances_all"][i_week].tolist() # list of ndarrays
            if len(datasets_this_animal)>0:
                tmp_datasets_all_animals.extend(datasets_this_animal)
        datasets.append(np.concatenate(tmp_datasets_all_animals))
        # print("DBG datasets[-1].shape", datasets[-1].shape)
    # now plot the datasets
    if ax is None:
        _, ax = plt.subplots()
    bp = ax.boxplot(datasets, positions=3.5+np.arange(duration_in_weeks)*7, widths=4, showfliers=False)
    [item.set_linewidth(1.5) for item in bp['boxes'] ]
    [(item.set_linewidth(1.5), item.set_color("k")) for item in bp['medians'] ]
    [(item.set_linewidth(1.5), item.set_color("k")) for item in bp['means'] ]
    [item.set_linewidth(1.5) for item in bp['whiskers'] ]
    [item.set_linewidth(1.5) for item in bp['caps'] ]
    # https://matplotlib.org/stable/gallery/statistics/boxplot_demo.html#sphx-glr-gallery-statistics-boxplot-demo-py
    # num_boxes = len(datasets)
    # for i in range(num_boxes):
    #     box = bp['boxes'][i]
    #     box_x = []
    #     box_y = []
    #     for j in range(5):
    #         box_x.append(box.get_xdata()[j])
    #         box_y.append(box.get_ydata()[j])
    #     box_coords = np.column_stack([box_x, box_y])
    #     # Alternate between Dark Khaki and Royal Blue
    #     ax.add_patch(patches.Polygon(box_coords, facecolor="darkkhaki"))
    #     # Now draw the median lines back over what we just filled in
    #     med = bp['medians'][i]
    #     median_x = []
    #     median_y = []
    #     for j in range(2):
    #         median_x.append(med.get_xdata()[j])
    #         median_y.append(med.get_ydata()[j])
    #         ax.plot(median_x, median_y, 'k')
    return ax

def scatter_datapoints(dict_animals, duration_in_days, ax=None):
    dict_animals_t = {}
    for animal_name, animal_dict in dict_animals.items():
        dict_animals_t[animal_name] = truncate_timeseries_in_dict(animal_dict, 0, duration_in_days-1)
    if ax is None:
        _, ax = plt.subplots()
    for animal_name, animal_dict in dict_animals_t.items():
        # ax.scatter(animal_dict["day_after_surgery"], animal_dict["impedances_mean"], label=animal_name, s=5)
        for k in range(len(animal_dict["day_after_surgery"])):
            this_day=animal_dict["time_in_seconds"][k]/(24*3600)
            ax.scatter([this_day]*len(animal_dict["impedances_all"][k]), animal_dict["impedances_all"][k], alpha=0.3, marker='s', s=2)
     # plt.legend()
    return ax


if __name__ == "__main__":
    PLOTFOLDER = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/codes/_pls_ignore_niceplots_230614"
    FOLDER = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/codes/_pls_ignore_chronic_data_230614"
    N_WEEKS = 12

    if not os.path.exists(PLOTFOLDER):
        os.makedirs(PLOTFOLDER)
    
    npznames = list(filter(lambda x: x.endswith("_impedances.npz"), os.listdir(FOLDER)))
    dict_animals = {}
    for npzname in npznames:
        aname = npzname.split("_")[0]
        npzfullpath = os.path.join(FOLDER, npzname)
        adict = read_chronic_animal_npz(npzfullpath)
        dict_animals[aname] = custom_curation(aname, adict)

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
    ax.set_xticks(np.arange(N_WEEKS)*7+3.5)
    ax.set_xticklabels(np.arange(N_WEEKS)+1)
    ax.set_xlabel("Week")
    ax.set_ylabel("#Impedance (kOhms)")
    plt.savefig(os.path.join(PLOTFOLDER, "impedances_all.png"))
    plt.savefig(os.path.join(PLOTFOLDER, "impedances_all.svg"))
    plt.show()

    # for each group of animals
    for group_name, animal_group in zip(group_names, animal_groups):
        dict_animals_subset = dict((k,dict_animals[k]) for k in dict_animals.keys() if k in animal_group)
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111)
        box_plot_weekly(dict_animals_subset, duration_in_weeks=N_WEEKS, ax=ax)
        scatter_datapoints(dict_animals_subset, duration_in_days=N_WEEKS*7, ax=ax)
        ax.set_xlim([-1, N_WEEKS*7])
        # ax.set_ylim([)
        ax.set_xticks(np.arange(N_WEEKS)*7+3.5)
        ax.set_xticklabels(np.arange(N_WEEKS)+1)
        ax.set_xlabel("Week")
        ax.set_ylabel("#Impedance (kOhms)")
        plt.savefig(os.path.join(PLOTFOLDER, "impedances_%s.png"%(group_name)))
        plt.savefig(os.path.join(PLOTFOLDER, "impedances_%s.svg"%(group_name)))
        plt.show()