import os
import gc
import warnings
from copy import deepcopy
from time import time

import numpy as np

from .openEphys import Binary
from .utils.mdaio import DiskWriteMda



N_CH = 32

# def walk_dict(dict_node, depth=0):
#     for k in dict_node.keys():
#         print("  "*depth + k)
#         if isinstance(dict_node[k], dict):
#             # print('haha')
#             walk_dict(dict_node[k], depth=depth+1)
#         else:
#             print("  "*(depth+1), type(dict_node[k]))

def get_data(dict_node):
    """Assume the dict has a linear structure with random key names and arbitrary depth, and eventually only one piece of data inside it"""
    _, first_value = list(dict_node.items())[0]
    if isinstance(first_value, dict):
        return get_data(first_value)
    return first_value

def preprocess_one_session(session_folder_raw, session_folder_mda):
    ts_session = time()
    print("  Starting session: %s" % (session_folder_raw))
    if not os.path.exists(session_folder_mda):
        os.makedirs(session_folder_mda)
    # read data
    data_dict, fs_dict = Binary.Load(session_folder_raw)#, Experiment=0, Recording=1)
    data_float = get_data(data_dict)
    data_ephys_short = data_float[:,:N_CH].T.astype(np.int16) # (N_channels, n_samples)
    sample_freq = get_data(fs_dict)
    print("    data.shape=", data_ephys_short.shape, "F_SAMPLE=", sample_freq)
    # write to mda
    n_ch= data_ephys_short.shape[0]
    n_samples = data_ephys_short.shape[1]
    mdapath = os.path.join(session_folder_mda, "converted_data.mda")
    writer = DiskWriteMda(mdapath, (n_ch, n_samples), dt="int16")
    writer.writeChunk(data_ephys_short, i1=0, i2=0)
    del data_ephys_short
    gc.collect()
    print("  Session preprocessed in %.2f sec" % (time()-ts_session))
    info_struct = {}
    info_struct['sample_freq'] = sample_freq
    info_struct['notch_freq'] = None
    info_struct['chs_info'] = {"OpenEphys": "O"}
    info_struct['n_samples'] = n_samples
    info_struct['tmp_mda_path'] = mdapath
    info_struct['tmp_mda_folder'] = session_folder_mda
    # info_struct is returned so that the calling context can write it to disk
    return info_struct


