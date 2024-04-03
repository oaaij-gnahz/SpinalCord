'''
divide a long continuous recording session (10's of hours) into several segments (e.g. 4h)
each segment is stored into a .mda file
then in downstream processing these mda files will be spike-sorted separately and then the tracking of clusters will be needed 
'''

import os
import gc
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from load_intan_rhd_format import read_data, read_header, get_n_samples_in_data
from utils.write_mda import writemda16i
from utils.filtering import notch_filter
from utils.mdaio import DiskWriteMda

# given a session
DATA_ROOTPATH = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/data/"
MDA_ROOTPATH  = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/data_converted/mda_files/"

SESSION_REL_PATH = "MustangContinuous/Mustang_220126_125248" 
SESSION_FOLDER_RAW = os.path.join(DATA_ROOTPATH, SESSION_REL_PATH)
SESSION_FOLDER_MDA = os.path.join(MDA_ROOTPATH, SESSION_REL_PATH)

### GEOM is not necessary in preprocessing step for SPINAL CORD ELECTRODES
# GEOM_ROOTPATH  = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out"
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

segment_size = 48 # number of .rhd files (5min per file) for each segment
n_segments = int(np.ceil(len(filenames)/segment_size))

segment_files_list = [filenames[i*segment_size:(i+1)*segment_size] for i in range(n_segments)]

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

### REMEMBER native order starts from 0 ###
# data_rhd_list = []
chs_native_order = None
chs_impedance = None
notch_freq = None
sample_freq = None
for i_seg, seg_filenames in enumerate(segment_files_list):
    # save all data in seg_filenames into a single .mda

    # first get the number of data in each .rhd file, ASSUMING all files have the same #channels
    n_samples_cumsum_by_file = [0]
    n_samples = 0
    for filename in seg_filenames:
        n_ch, n_samples_this_file =get_n_samples_in_data(os.path.join(SESSION_FOLDER_RAW, filename))
        n_samples += n_samples_this_file
        n_samples_cumsum_by_file.append(n_samples)
    
    # prepare writer
    seg_mdapath = os.path.join(SESSION_FOLDER_MDA, "converted_data_seg%d.mda"%(i_seg+1))
    print("Creating Writer for %d"%(i_seg))
    writer = DiskWriteMda(seg_mdapath, (n_ch, n_samples), dt="int16")
    
    # load data from intan and use writer to append data to one single .mda file
    for i_file, filename in enumerate(seg_filenames):
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
        # if notch_freq>0:
        #     print("Applying notch")
        #     ephys_data = notch_filter(ephys_data, sample_freq, notch_freq, Q=20)
        ephys_data = ephys_data.astype(np.int16)
        print("Appending to disk at: %s" % (seg_mdapath))
        entry_offset = n_samples_cumsum_by_file[i_file]*n_ch
        writer.writeChunk(ephys_data, i1=0, i2=entry_offset)

