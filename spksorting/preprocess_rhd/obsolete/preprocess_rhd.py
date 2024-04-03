#%%
import os
import gc
import warnings
from copy import deepcopy
from time import time

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from load_intan_rhd_format import read_data, read_header
from utils.write_mda import writemda16i
from utils.filtering import notch_filter

# channel map .mat file
# BC6 is rigid
# BC7, BC8 is flex
# CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/chan_map_1x32_128ch_rigid.mat" # rigid
# CHANNEL_MAP_FPATH = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/channel_maps/128chMap_flex.mat" # flex 
# channel spacing
# GW = 300 # micron
# GH = 25 # micron

# given a session
DATA_ROOTPATH = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/data/"
MDA_ROOTPATH  = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/data_converted/mda_files/"
# GEOM_ROOTPATH  = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out"
SESSION_REL_PATH = "Tundra21h_220110" # 07, 09(pre stroke), 12(post stroke)
SESSION_FOLDER_RAW = os.path.join(DATA_ROOTPATH, SESSION_REL_PATH)
SESSION_FOLDER_MDA = os.path.join(MDA_ROOTPATH, SESSION_REL_PATH)
# SESSION_FOLDER_CSV = os.path.join(GEOM_ROOTPATH, SESSION_REL_PATH)

def get_datetimestr_from_filename(filename):
    """
        Assume the rhd filename takes the form of either
        ParentPath/Animalname_YYMMDD_HHMMSS.rhd
        or 
        Animialname_YYMMDD_HHMMSS.rhd
    """
    tmp = filename.split("/")[-1].split("_")
    return tmp[1]+'_'+tmp[2].split('.')[0]


filenames = os.listdir(SESSION_FOLDER_RAW)
filenames = list(filter(lambda x: x.endswith(".rhd"), filenames))
filenames = sorted(filenames, key=get_datetimestr_from_filename) # sort by time

if not os.path.exists(SESSION_FOLDER_MDA):
    os.makedirs(SESSION_FOLDER_MDA)

# if not os.path.exists(SESSION_FOLDER_CSV):
#     os.makedirs(SESSION_FOLDER_CSV)
#%%
def check_header_consistency(hA, hB):
    if len(hA)!=len(hB): 
        return False
    for a, b in zip(hA, hB):
        if a!=b:
            return False
    return True

# read rhd files and append data in list 
### REMEMBER native order starts from 0 ###
# data_rhd_list = []
ts = time()
ephys_data_whole = None
chs_native_order = None
chs_impedance = None
notch_freq = None
sample_freq = None
for filename in filenames:
    print("----%s----"%(os.path.join(SESSION_FOLDER_RAW, filename)))
    with open(os.path.join(SESSION_FOLDER_RAW, filename), "rb") as fh:
        head_dict = read_header(fh)
    data_dict = read_data(os.path.join(SESSION_FOLDER_RAW, filename))
    chs_info = deepcopy(data_dict['amplifier_channels'])
    
    # record and check key information
    if chs_native_order is None:
        chs_native_order = [e['native_order'] for e in chs_info]
        chs_impedance = [e['electrode_impedance_magnitude'] for e in chs_info]
        print("#Chans with >= 3MOhm impedance:", np.sum(np.array(chs_impedance)>=3e6))
        notch_freq = head_dict['notch_filter_frequency']
        sample_freq = head_dict['sample_rate']
        print("sampleFreq=",sample_freq, " NotchFreq=", notch_freq)
    else:
        tmp_native_order = [e['native_order'] for e in chs_info]
        print("#Chans with >= 3MOhm impedance:", np.sum(np.array([e['electrode_impedance_magnitude'] for e in chs_info])>=3e6))
        if not check_header_consistency(tmp_native_order, chs_native_order):
            warnings.warn("WARNING in preprocess_rhd: native ordering of channels inconsistent within one session\n")
        if notch_freq != head_dict['notch_filter_frequency']:
            warnings.warn("WARNING in preprocess_rhd: notch frequency inconsistent within one session\n")
        if sample_freq != head_dict['sample_rate']:
            warnings.warn("WARNING in preprocess_rhd: sampling frequency inconsistent within one session\n")
    
    
    ephys_data = data_dict['amplifier_data']
    if notch_freq>0:
        print("Applying notch")
        ephys_data = notch_filter(ephys_data, sample_freq, notch_freq, Q=20)
    ephys_data = ephys_data.astype(np.int16)
    print("Concatenating")
    if ephys_data_whole is None:
        ephys_data_whole = ephys_data
    else:
        ephys_data_whole = np.concatenate([ephys_data_whole, ephys_data], axis=1)
        del(ephys_data)
        del(data_dict)
        gc.collect()
print("Time elapsed for reading array into RAM:", time()-ts)
print("Ephys data shape:", ephys_data_whole.shape)
print("Saving mda...")
# save to mda
ts = time()
writemda16i(ephys_data_whole, os.path.join(SESSION_FOLDER_MDA, "converted_data.mda"))
print("MDA file saved to %s" % (os.path.join(SESSION_FOLDER_MDA, "converted_data.mda")))
print("Time elapsed for writing mda:", time()-ts)