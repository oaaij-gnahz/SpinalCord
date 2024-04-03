'''
preprocess rhd files into mda files;
called from <repo_dir>/main_runnables/preprop_and_sort.py
8/31/2022 jz103
'''

import os
import gc
import warnings
from copy import deepcopy
from time import time

import numpy as np
# import pandas as pd
# from scipy.io import loadmat, savemat

from .load_intan_rhd_format import read_data, read_header, get_n_samples_in_data
# from .utils.write_mda import writemda16i
from .utils.filtering import notch_filter
from .utils.mdaio import DiskWriteMda

def get_datetimestr_from_filename(filename):
    """
        Assume the rhd filename takes the form of either
        ParentPath/Animalname_YYMMDD_HHMMSS.rhd or ParentPath/Animalname_moredescirptions_YYMMDD_HHMMSS.rhd
        or 
        Animialname_YYMMDD_HHMMSS.rhd
    """
    fname = filename.split("/")[-1]
    print(fname)
    tmp = fname.split("_")
    return tmp[-2]+'_'+tmp[-1].split('.')[0]

def check_header_consistency(hA, hB):
    if len(hA)!=len(hB): 
        return False
    for a, b in zip(hA, hB):
        if a!=b:
            return False
    return True

def preprocess_one_session(session_folder_raw, session_folder_mda, verbose=True):
    '''
    preprocess one session of rhd files -> store in mda format.
    Automatically creates folder for mda file if it does not already exist
    '''
    ts_session = time()
    filenames = os.listdir(session_folder_raw)
    filenames = list(filter(lambda x: x.endswith(".rhd"), filenames))
    if len(filenames)==0:
        print("****Empty Session:", os.listdir(session_folder_raw))
        return
    filenames = sorted(filenames, key=get_datetimestr_from_filename) # sort by time
    print("  Starting session: %s" % (session_folder_raw))
    if not os.path.exists(session_folder_mda):
        os.makedirs(session_folder_mda)

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
    n_ch = 999999
    for filename in filenames:
        n_ch_, n_samples_this_file = get_n_samples_in_data(os.path.join(session_folder_raw, filename))
        n_ch = min(n_ch, n_ch_)
        print(n_ch)
        n_samples += n_samples_this_file
        n_samples_cumsum_by_file.append(n_samples)

    # TODO directly ERROR OUT if n_ch>32
    n_ch = min(n_ch, 32)
    if verbose:
        print("#channels:", n_ch)

    # prepare writer
    mdapath = os.path.join(session_folder_mda, "converted_data.mda")
    writer = DiskWriteMda(mdapath, (n_ch, n_samples), dt="int16")
    
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
        ephys_data = data_dict['amplifier_data']
        ####------------------------------------------------------
        #### TO ACCOMODATE NORA'S INTAN RECORDING SETTINGS; sometimes there are fewer channels; sometimes there are 64 channels (only 16-47 in PORT A are good).
        if ephys_data.shape[0] > n_ch:
            ephys_data = ephys_data[16:16+n_ch, :]
            warnstr  = "WARNING IN preprocess_rhd: this rhd file has "
            warnstr +=  "%d channels, which is more than the %d present throughout session."%(ephys_data.shape[0], n_ch)
            warnstr += " CLIPPING extras! Keeping only data[%d:%d, :]" % (16, 16+n_ch)
            warnings.warn(warnstr)
            if ephys_data.shape[0] < n_ch:
                warnings.warn("After clipping, the #channels kept is %d < %d"%(ephys_data.shape[0], n_ch))
        ####------------------------------------------------------
        ts_notch = time()
        if notch_freq>0:
            print("    Applying notch")
            ephys_data = notch_filter(ephys_data, sample_freq, notch_freq, Q=20)
        ephys_data = ephys_data.astype(np.int16)
        print("    Notching done: time elapsed: %.2f sec" % (time()-ts_notch))
        print("    Appending to disk at: %s; data shape=(%d,%d)" % (mdapath, ephys_data.shape[0], ephys_data.shape[1])) # (n_ch, n_samples)
        entry_offset = n_samples_cumsum_by_file[i_file]*n_ch
        writer.writeChunk(ephys_data, i1=0, i2=entry_offset)
        del ephys_data
        gc.collect()
    print("  Session preprocessed in %.2f sec" % (time()-ts_session))
    info_struct = {}
    info_struct['sample_freq'] = sample_freq
    info_struct['notch_freq'] = notch_freq
    info_struct['chs_info'] = chs_info
    info_struct['n_samples'] = n_samples
    info_struct['tmp_mda_path'] = mdapath
    info_struct['tmp_mda_folder'] = session_folder_mda
    # #info_struct is returned so that the calling context can write it to disk
    return info_struct
