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
# import pandas as pd
# from scipy.io import loadmat, savemat

from load_intan_rhd_format import read_data, read_header, get_n_samples_in_data
# from .utils.write_mda import writemda16i
# from .utils.filtering import notch_filter
# from .utils.mdaio import DiskWriteMda

def get_datetimestr_from_filename(filename):
    """
        Assume the rhd filename takes the form of either
        ParentPath/Animalname_YYMMDD_HHMMSS.rhd
        or 
        Animialname_YYMMDD_HHMMSS.rhd
    """
    tmp = filename.split("/")[-1].split("_")
    return tmp[1]+'_'+tmp[2].split('.')[0]

def check_header_consistency(hA, hB):
    if len(hA)!=len(hB): 
        return False
    for a, b in zip(hA, hB):
        if a!=b:
            return False
    return True

def read_one_session(session_folder_raw):
    ts_session = time()
    filenames = os.listdir(session_folder_raw)
    filenames = list(filter(lambda x: x.endswith(".rhd"), filenames))
    if len(filenames)==0:
        print("****Empty Session:", os.listdir(session_folder_raw))
        return
    filenames = sorted(filenames, key=get_datetimestr_from_filename) # sort by time
    print("  Starting session: %s" % (session_folder_raw))

    ### REMEMBER native order starts from 0 ###
    # data_rhd_list = []
    chs_native_order = None
    chs_impedance = None
    chs_info = None
    notch_freq = None
    sample_freq = None

    # first get the number of data in each .rhd file, ASSUMING all files have the same #channels
    n_samples_cumsum_by_file = [0]
    n_samples = 0
    for filename in filenames:
        n_ch, n_samples_this_file = get_n_samples_in_data(os.path.join(session_folder_raw, filename))
        n_samples += n_samples_this_file
        n_samples_cumsum_by_file.append(n_samples)

    
    # load data from intan and use writer to append data to one single .mda file
    for i_file, filename in enumerate(filenames):
        print("    %s"%(os.path.join(session_folder_raw, filename)))
        with open(os.path.join(session_folder_raw, filename), "rb") as fh:
            head_dict = read_header(fh)
        data_dict = read_data(os.path.join(session_folder_raw, filename))
        chs_info = deepcopy(data_dict['amplifier_channels'])
        # record and check key information
        if chs_native_order is None:
            chs_native_order = [e['native_order'] for e in chs_info]
            chs_impedance = [e['electrode_impedance_magnitude'] for e in chs_info]
            print("        #Chans with >= 3MOhm impedance:", np.sum(np.array(chs_impedance)>=3e6))
            notch_freq = head_dict['notch_filter_frequency']
            sample_freq = head_dict['sample_rate']
            print("        sampleFreq=",sample_freq, " NotchFreq=", notch_freq)
        else:
            tmp_native_order = [e['native_order'] for e in chs_info]
            print("        #Chans with >= 3MOhm impedance:", np.sum(np.array([e['electrode_impedance_magnitude'] for e in chs_info])>=3e6))
            if not check_header_consistency(tmp_native_order, chs_native_order):
                warnings.warn("        WARNING in preprocess_rhd: native ordering of channels inconsistent within one session\n")
            if notch_freq != head_dict['notch_filter_frequency']:
                warnings.warn("        WARNING in preprocess_rhd: notch frequency inconsistent within one session\n")
            if sample_freq != head_dict['sample_rate']:
                warnings.warn("        WARNING in preprocess_rhd: sampling frequency inconsistent within one session\n")
        del data_dict
        gc.collect()
    print("  Session preprocessed in %.2f sec" % (time()-ts_session))
    info_struct = {}
    info_struct['sample_freq'] = sample_freq
    info_struct['notch_freq'] = notch_freq
    info_struct['chs_info'] = chs_info
    # TODO save info_struct to disk for future reference
    return info_struct


def procfunc_prepdata(
    list_of_rhd_folders : list, 
    resfolder : str
    ):
    ch_impedances = []
    session_datetimes = []
    for i_work, (rhdfoldername) in enumerate(list_of_rhd_folders):
        print(i_work)
        # info_dict={}
        tmp_split = rhdfoldername.split("/")
        rhdsubfoldername = tmp_split[-1] if tmp_split[-1]!="" else tmp_split[-2]
        datetimestr = re.match(r"[A-Z][a-z]+_([0-9]+_[0-9]+)", rhdsubfoldername)[1]
        session_datetime = datetime.datetime.strptime(datetimestr, "%y%m%d_%H%M%S")
        info_dict = read_one_session(rhdfoldername)
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
    ax.violinplot(ch_impedances, positions=timedelta_floats, showmeans=True, widths=timedelta_floats[-1]/timedelta_floats.shape[0]/2, points=500)
    ch_impedances_mean = [np.mean(k) for k in ch_impedances]
    ax.plot(timedelta_floats, ch_impedances_mean, color='k')
    for i_session in range(timedelta_floats.shape[0]):
        ax.text(timedelta_floats[i_session], 0, str(ch_impedances[i_session].shape[0]))
    # ax.plot(session_datetimes, np.array(session_stats['n_single_units'])/np.array(session_stats['n_ch']), label='# single-units')
    # ax.legend()
    plt.xticks(timedelta_floats, [t.strftime('%Y-%m-%d') for t in session_datetimes], rotation=45)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(resfolder, "impedances.png"))
    plt.close()


if __name__=="__main__":
    rawfolder_root = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/data/corolla_chronic"
    mdafolder_root = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/data_converted/corolla_chronic"
    resfolder_root = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/corolla_chronic"
    if not os.path.exists(mdafolder_root):
        os.makedirs(mdafolder_root)
    if not os.path.exists(resfolder_root):
        os.makedirs(resfolder_root)
    sessions_list = list(filter(lambda x: x.startswith("Corolla"), os.listdir(rawfolder_root)))
    rhd_folders_list = list(map(lambda x: os.path.join(rawfolder_root, x), sessions_list))
    mda_folders_list = list(map(lambda x: os.path.join(mdafolder_root, x), sessions_list))
    res_folders_list = list(map(lambda x: os.path.join(resfolder_root, x), sessions_list))
    procfunc_prepdata(rhd_folders_list, resfolder_root)