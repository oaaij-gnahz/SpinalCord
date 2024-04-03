#%%
import os
import gc
import warnings
from copy import deepcopy
from time import time
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

from load_intan_rhd_format import read_data, read_header
# from utils.read_mda import readmda
# from utils.write_mda import writemda16i
from utils.mdaio import readmda, writemda16i_large
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
MDA_ROOTPATH  = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/data_converted/mda_files/"
# GEOM_ROOTPATH  = "/media/luanlab/Data_Processing/Jim-Zhang/Spike-Sort/spikesort_out"
SESSION_REL_PATH = "Tundra21h_220110" # 07, 09(pre stroke), 12(post stroke)
SESSION_FOLDER_MDA0 = os.path.join(MDA_ROOTPATH, SESSION_REL_PATH)
SESSION_FOLDER_MDA = os.path.join(SESSION_FOLDER_MDA0, "12hr_segs")
SESSION_FOLDER_MDA_NEW = os.path.join(SESSION_FOLDER_MDA0, "whole_20hour")
# SESSION_FOLDER_CSV = os.path.join(GEOM_ROOTPATH, SESSION_REL_PATH)


filenames = os.listdir(SESSION_FOLDER_MDA)
filenames = list(filter(lambda x: bool(re.match("converted_data_seg[0-9]+.mda", x)), filenames))
filenames = sorted(filenames, key=lambda x: int(re.match("converted_data_seg([0-9]+).mda", x)[1])) # sort by time

print(filenames)

if not os.path.exists(SESSION_FOLDER_MDA_NEW):
    os.makedirs(SESSION_FOLDER_MDA_NEW)

segment_size = 3
n_segments = int(np.ceil(len(filenames)/segment_size))
segment_files_list = [filenames[i*segment_size:(i+1)*segment_size] for i in range(n_segments)]



for i_seg, seg_filenames in enumerate(segment_files_list):
    ephys_data_whole = None
    for filename in seg_filenames:
        print("----%s----"%(os.path.join(SESSION_FOLDER_MDA, filename)))
        ts=time()
        ephys_data = readmda(os.path.join(SESSION_FOLDER_MDA, filename))
        print("MDA reading time:", time()-ts)
        print(ephys_data.shape)
        if ephys_data_whole is None:
            ephys_data_whole = ephys_data
        else:
            ephys_data_whole = np.concatenate([ephys_data_whole, ephys_data], axis=1)
            del(ephys_data)
            gc.collect()
    print(ephys_data_whole.shape)
    ts = time()
    writemda16i_large(ephys_data_whole, os.path.join(SESSION_FOLDER_MDA_NEW, "converted_data_seg%d.mda"%(i_seg+1)))
    print("MDA file saved to %s" % (os.path.join(SESSION_FOLDER_MDA_NEW, "converted_data_seg%d.mda"%(i_seg+1))))
    print("Time elapsed for writing mda:", time()-ts)