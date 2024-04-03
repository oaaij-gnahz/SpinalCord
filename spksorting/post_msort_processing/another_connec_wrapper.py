""" 
Calls CellExplorer connectivity function from Python. The input format fits Yu's curated data format.
2/27/2023 jiaaoZ
"""
import os
import re
import shutil
import subprocess

import numpy as np

from utils.mdaio import readmda, writemda64

def get_segment_index(segment_name: str) -> int:
    return int(re.search("seg([0-9]+)", segment_name)[1])

def connec_wrapper_main(postproc_folder, fs=30000):
    
    # prepare inputs for each given segment 
    for i_seg in range(7):
        seg_folder_new = os.path.join(postproc_folder, "seg%d"%(i_seg+1))
        os.makedirs(seg_folder_new, exist_ok=True)
        # shutil.copy2(os.path.join(postproc_folder, "firings%d_merged_v1.mda"%(i_seg+1)), os.path.join(seg_folder_new, "firings_clean.mda"))
        f = readmda(os.path.join(postproc_folder, "firings%d_merged_v1.mda"%(i_seg+1)))
        f = f.astype(np.int64)
        writemda64(f, os.path.join(seg_folder_new, "firings_clean.mda"))
        x = readmda(os.path.join(postproc_folder, "templates%d_merged_v1.mda"%(i_seg+1)))
        y = np.zeros_like(x)
        writemda64(y, os.path.join(seg_folder_new, "templates_clean.mda"))

    # set MATLAB command line arguments
    matlab_argin1 = "{"
    for i_seg in range(7):
        matlab_argin1 += "\'%s\'"%(os.path.join(postproc_folder, "seg%d"%(i_seg+1)))
        matlab_argin1 += ", "
    matlab_argin1 += "}"
    matlab_argin2 = str(fs)
    
    matlab_args = ["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r"]
    matlab_cmd = "cd utils_matlab; "
    matlab_cmd += "process_segs_batch(%s, %s); exit;" % (matlab_argin1, matlab_argin2)
    matlab_args.append(matlab_cmd)
    print(matlab_args)
    # run MATLAB
    cproc = subprocess.run(matlab_args)
    print("Return code:", cproc.returncode)

    # bring the output back
    for i_seg in range(7):
        seg_folder_new = os.path.join(postproc_folder, "seg%d"%(i_seg+1))
        assert os.path.exists(seg_folder_new)
        shutil.copy2(os.path.join(seg_folder_new, "connecs.csv"), os.path.join(postproc_folder, "connecs%d_merged_v1.csv"%(i_seg+1)))
        shutil.copy2(os.path.join(seg_folder_new, "mono_res.cellinfo.mat"), os.path.join(postproc_folder, "mono_res%d_merged_v1.cellinfo.mat"%(i_seg+1)))

postproc_folder = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/SilveradoCurated"
connec_wrapper_main(postproc_folder)