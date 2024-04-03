"""Purpose of this script: try sorting those single sessions separately 
that failed when batch run together with other sessions"""

import os
import sys
from time import time
import multiprocessing
from queue import Empty as qEmptyException
import traceback
import json

import numpy as np

sys.path.insert(0, "../")
from preprocess_rhd.preprocess_rhd import preprocess_one_session
from utils_here import clear_folder

BAD_SESSIONS_RHD = []
def procfunc_prepdata(
    list_of_rhd_folders : list, 
    list_of_mda_folders : list, 
    list_of_result_folders, 
    mpq : multiprocessing.Queue
    ):
    for i_work, (rhdfoldername, mdafoldername, resultfoldername) in enumerate(zip(list_of_rhd_folders, list_of_mda_folders, list_of_result_folders)):
        print(i_work)
        # info_dict={}
        # if os.path.exists(os.path.join(mdafoldername, "converted_data.mda")):
            # with open(os.path.join(resultfoldername, "session_rhd_info.json"), "r") as f:
            #     # mdafoldername/converted_data.mda and resultfoldername/session_rhd_info.json should be created together.
            #     info_dict = json.load(f)
        # else:
        try:
            info_dict = preprocess_one_session(rhdfoldername, mdafoldername)
        except Exception:
            traceback.print_exc()
            BAD_SESSIONS_RHD.append(rhdfoldername)
            continue
        if not os.path.exists(resultfoldername):
            os.makedirs(resultfoldername)
        # end if-else block

        if info_dict is None:
            continue # empty session
        info_dict['tmp_mda_folder'] = mdafoldername
        info_dict['tmp_result_folder']=resultfoldername

        # the following block that stores the session info is added after sorting mustang and corolla
        json_fpath = os.path.join(resultfoldername, "session_rhd_info.json")
        if not os.path.exists(json_fpath):
            with open(json_fpath, 'w') as f:
                json.dump(info_dict, f)

        mpq.put(info_dict)
    mpq.put("EOQ") # End of Queue

def procfunc_sortdata(info_dict):
    sample_freq = info_dict['sample_freq']
    # mdapath = info_dict['tmp_mda_path']
    mdafoldername = info_dict['tmp_mda_folder']
    resultfoldername = info_dict['tmp_result_folder']
    ml_temporary_directory = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/ml_temp"
    if len(info_dict['chs_info'])<32:
        mapcsvpath = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/codes/geom_channel_maps/map_corolla24ch.csv"
    else:
        mapcsvpath = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/codes/geom_channel_maps/map.csv"
    if not os.path.exists(resultfoldername):
        os.makedirs(resultfoldername)
    # set temporary directory
    os.environ['ML_TEMPORARY_DIRECTORY'] = ml_temporary_directory
    if not os.path.exists(ml_temporary_directory):
        os.makedirs(ml_temporary_directory)
    # run sorting script
    shell_command = "bash /media/hanlin/Liuyang_10T_backup/jiaaoZ/codes/current_shell_scripts/sort_single_yu.sh %s %s %d %s" % (
        mdafoldername, resultfoldername, sample_freq, mapcsvpath
    )
    os.system(shell_command)
    # clear_folder(ml_temporary_directory) # clear temporary directory
        

if __name__=="__main__":
    # rawfolder_root = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/data/dorito_chronic"
    # mdafolder = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/data_converted/tacoma_chronic/Tacoma_211213_153805"
    resfolder = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/tacoma_chronic/Tacoma_211213_153805"
    print("Start")
    info_path = os.path.join(resfolder, "session_rhd_info.json")
    with open(info_path, "r") as f:
        info_dict = json.load(f)
    
    # procfunc_prepdata(rhd_folders_list, mda_folders_list, res_folders_list, my_queue)
    procfunc_sortdata(info_dict)