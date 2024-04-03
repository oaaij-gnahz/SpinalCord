'''
read channel info
8/31/2022 jz103
'''

import os
import gc
import warnings
from copy import deepcopy
from time import time
import re
import datetime
import json

import numpy as np
import matplotlib.pyplot as plt

SESSION_NAMING_PATTERN = r"[A-Za-z_ ]+_([0-9]+_[0-9]+)"
VALID_SESSION_LAMBDA = lambda x: (('_' in x) and ('.' not in x) and ('statfigs' not in x))

def get_datetimestr_from_filename(filename):
    """
        Assume the rhd filename takes the form of either
        ParentPath/Animalname_YYMMDD_HHMMSS.rhd
        or 
        Animialname_YYMMDD_HHMMSS.rhd
    """
    tmp = filename.split("/")[-1].split("_")
    return tmp[1]+'_'+tmp[2].split('.')[0]



def procfunc_prepdata(
    list_of_res_folders : list, 
    resfolder : str
    ):
    ch_impedances = []
    session_datetimes = []
    for i_work, (resfoldername) in enumerate(list_of_res_folders):
        print(i_work)
        # info_dict={}
        tmp_split = resfoldername.split("/")
        ressubfoldername = tmp_split[-1] if tmp_split[-1]!="" else tmp_split[-2]
        datetimestr = re.match(SESSION_NAMING_PATTERN, ressubfoldername)[1]
        print(ressubfoldername, datetimestr)
        session_datetime = datetime.datetime.strptime(datetimestr, "%y%m%d_%H%M%S")
        # info_dict = read_one_session(resfoldername)
        with open(os.path.join(resfoldername, 'session_rhd_info.json'), "r") as f_read:
            info_dict = json.load(f_read)
        if info_dict is None:
            continue # empty session
        chs_info = info_dict['chs_info']
        chs_impedance = np.array([e['electrode_impedance_magnitude'] for e in chs_info])
        chs_impedance_valid = chs_impedance[chs_impedance<3e6]
        session_datetimes.append(session_datetime)
        ch_impedances.append(chs_impedance_valid)
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    timedelta_floats = np.array([(sdt - session_datetimes[0]).total_seconds() for sdt in session_datetimes])
    # timedelta_floats = timedelta_floats/timedelta_floats[-1]
    violin_parts = ax.violinplot(ch_impedances, positions=timedelta_floats, showmeans=True, widths=timedelta_floats[-1]/timedelta_floats.shape[0]/2, points=500)
    for pc in violin_parts['bodies']:
        pc.set_facecolor('gray')
        pc.set_edgecolor('gray')
    ch_impedances_mean = [np.mean(k) for k in ch_impedances]
    ax.plot(timedelta_floats, ch_impedances_mean, color='k')
    # for i_session in range(timedelta_floats.shape[0]):
    #     ax.text(timedelta_floats[i_session], 2.6e6, str(ch_impedances[i_session].shape[0]))
    # ax.plot(session_datetimes, np.array(session_stats['n_single_units'])/np.array(session_stats['n_ch']), label='# single-units')
    # ax.legend()
    plt.xticks(timedelta_floats, np.round(timedelta_floats/24/3600).astype(int))
    # plt.xticks(timedelta_floats[::3], [t.strftime('%Y-%m-%d') for t in session_datetimes][::3], rotation=45)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(resfolder, "impedances.svg"))
    plt.close()
    fig2 = plt.figure(figsize=(10,5))
    plt.plot(timedelta_floats, [ch_impedances[i_session].shape[0] for i_session in range(timedelta_floats.shape[0])], c='k', marker='.')
    # plt.xticks(timedelta_floats[::3], [t.strftime('%Y-%m-%d') for t in session_datetimes][::3], rotation=45)
    plt.xticks(timedelta_floats, np.round(timedelta_floats/24/3600).astype(int))
    plt.ylim([0,32.5])
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(resfolder, "valid_ch_cnts.svg"))
    plt.close()

if __name__=="__main__":
    # rawfolder_root = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/data/corolla_chronic"
    # mdafolder_root = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/data_converted/corolla_chronic"
    resfolder_root = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/tacoma_chronic"
    # if not os.path.exists(mdafolder_root):
        # os.makedirs(mdafolder_root)
    # if not os.path.exists(resfolder_root):
        # os.makedirs(resfolder_root)
    sessions_list = list(filter(VALID_SESSION_LAMBDA, os.listdir(resfolder_root)))
    sessions_list = sorted(
        sessions_list, 
        key=lambda x: datetime.datetime.strptime(re.match(SESSION_NAMING_PATTERN, x)[1], "%y%m%d_%H%M%S") 
    )
    # rhd_folders_list = list(map(lambda x: os.path.join(rawfolder_root, x), sessions_list))
    # mda_folders_list = list(map(lambda x: os.path.join(mdafolder_root, x), sessions_list))
    res_folders_list = list(map(lambda x: os.path.join(resfolder_root, x), sessions_list))
    procfunc_prepdata(res_folders_list, resfolder_root)