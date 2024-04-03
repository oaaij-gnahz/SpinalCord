import os
import sys
from time import time
import multiprocessing
import subprocess
from queue import Empty as qEmptyException
import traceback
import json
import warnings

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
        
        if os.path.exists(os.path.join(mdafoldername, "converted_data.mda")):
            with open(os.path.join(resultfoldername, "session_rhd_info.json"), "r") as f:
                # mdafoldername/converted_data.mda and resultfoldername/session_rhd_info.json should be created together.
                info_dict = json.load(f)

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
            warnings.warn("Empty Session: %s"%(rhdfoldername.split('/')[-1]))
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

def procfunc_sortdata(mpq : multiprocessing.Queue):
    while True:
        try:
            info_dict = mpq.get()
            if isinstance(info_dict, str) and info_dict=="EOQ":
                print("=*=*=Sentinel Reached")
                break 
            sample_freq = info_dict['sample_freq']
            # mdapath = info_dict['tmp_mda_path']
            mdafoldername = info_dict['tmp_mda_folder']
            resultfoldername = info_dict['tmp_result_folder']
            ml_temporary_directory = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/ml_temp"
            if len(info_dict['chs_info'])<32:
                mapcsvpath = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/codes/geom_channel_maps/map_corolla24ch.csv"
            else:
                mapcsvpath = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/codes/geom_channel_maps/map.csv"
            if not os.path.exists(resultfoldername):
                os.makedirs(resultfoldername)
            # set temporary directory
            os.environ['ML_TEMPORARY_DIRECTORY'] = ml_temporary_directory
            if not os.path.exists(ml_temporary_directory):
                os.makedirs(ml_temporary_directory)
            # run sorting script
            subproc_args = [
                "/bin/bash", "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/codes/current_shell_scripts/sort_single_yu.sh", 
                str(mdafoldername), str(resultfoldername), str(sample_freq), mapcsvpath
            ]
            retcode = subprocess.call(subproc_args, shell=False) # call shell script and wait for return code
            if (retcode!=0):
                BAD_SESSIONS_RHD.append(resultfoldername + ':MS_retcode=%d'%(retcode))
            clear_folder(ml_temporary_directory) # clear temporary directory
            
        except Exception as e:
            if isinstance(e, qEmptyException):
                continue
            traceback.print_exc()
            print("See Error Above")
            break

if __name__=="__main__":
    rawfolder_root = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/data/nora_february"
    mdafolder_root = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/data_converted/nora_february"
    resfolder_root = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/nora_february"
    print("Start")
    if not os.path.exists(mdafolder_root):
        os.makedirs(mdafolder_root)
    if not os.path.exists(resfolder_root):
        os.makedirs(resfolder_root)
    sessions_list = os.listdir(rawfolder_root)
    rhd_folders_list = list(map(lambda x: os.path.join(rawfolder_root, x), sessions_list))
    mda_folders_list = list(map(lambda x: os.path.join(mdafolder_root, x), sessions_list))
    res_folders_list = list(map(lambda x: os.path.join(resfolder_root, x), sessions_list))
    my_queue = multiprocessing.Queue()

    procfunc_prepdata(rhd_folders_list, mda_folders_list, res_folders_list, my_queue)
    procfunc_sortdata(my_queue)
    print("BAD_SESSIONS:")
    print("\n".join(BAD_SESSIONS_RHD))
    print()
    # proc_prepdata = multiprocessing.Process(target=procfunc_prepdata, args=(rhd_folders_list, mda_folders_list, res_folders_list, my_queue))
    # proc_prepdata.daemon = True
    # proc_sortdata = multiprocessing.Process(target=procfunc_sortdata, args=(my_queue,))
    # proc_sortdata.daemon = True
    # proc_prepdata.start()
    # proc_sortdata.start()
    # proc_prepdata.join()
    # proc_sortdata.join()
