""" Automatically discard noise clusters conservatively (by amplitude, spatial spread, ISI violation ratio) and viz"""
#%%
import json
import os
from time import time
from copy import deepcopy
import gc
from collections import OrderedDict
import multiprocessing

import shutil # rmtree
import numpy as np
# from scipy.io import loadmat
import matplotlib; matplotlib.use("agg")
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import scipy.signal as signal


from utils.read_mda import readmda

# settings
POSTPROC_FOLDER = "postproc_230611"



def postprocess_one_session(session_folder):
    
    """ main function for post processing and visualization"""
    
    ### read clustering metrics file and perform rejection TODO improve rejection method; current version SUCKS
    print(session_folder)
    
    postprocpath = os.path.join(session_folder, POSTPROC_FOLDER)

    # np.savez(os.path.join(postprocpath, "cluster_rejection_mask.npz"),\
    #     single_unit_mask=cluster_accept_mask,
    #     multi_unit_mask=multi_unit_mask
    #     )
    mask_npz = np.load(os.path.join(postprocpath, "cluster_rejection_mask.npz"))
    single_unit_mask = mask_npz["single_unit_mask"]
    multi_unit_mask  = mask_npz["multi_unit_mask"]
    pd.DataFrame(data=single_unit_mask.astype(int)).to_csv(os.path.join(postprocpath, "single_unit_mask.csv"), index=False, header=False)
    pd.DataFrame(data=single_unit_mask.astype(int)).to_csv(os.path.join(postprocpath, "single_unit_mask_human.csv"), index=False, header=False)
    pd.DataFrame(data=multi_unit_mask.astype(int)).to_csv(os.path.join(postprocpath, "multi_unit_mask.csv"), index=False, header=False)



def one_animal(result_folder):
    error_sessions = []
    relevant_subpaths = list(filter(lambda x: (('_' in x) and ('.' not in x) and ('__' not in x)), os.listdir(result_folder)))
    relevant_fullpaths = list(map(lambda x: os.path.join(result_folder, x), relevant_subpaths))
    for session_folder in filter(lambda x: os.path.isdir(x), relevant_fullpaths):
        try:
            postprocess_one_session(session_folder)
        except Exception as e:
            print("---------------EXCEPTION MESSAGE")
            print(e)
            error_sessions.append({
                "session_folder": session_folder,
                "error_msg": str(e),
            })
    
    print("----#ERROR SESSIONS:", len(error_sessions))
    for error_session in error_sessions:
        print("%s: %s" % (error_session['session_folder'], error_session['error_msg']))

    with open(os.path.join(result_folder, "msg.json"), 'w') as f:
        json.dump(error_sessions, f)



if __name__ == '__main__':
    result_folders = [ 
        # "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/corolla_chronic/",
        # "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/dorito_chronic/",
        # "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/mustang_chronic/",
        # "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/nacho_chronic/",
        # "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/nora_chronic/",
        # "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/tacoma_chronic/",
        # "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/yogurt_chronic/",
        "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/BenMouse0/",
        "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/BenMouse1/"
    ]

    for res_folder in result_folders:
        one_animal(res_folder)