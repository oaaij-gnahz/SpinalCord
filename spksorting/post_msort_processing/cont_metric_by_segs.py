"""
This script assumes a 20-some-hour continuous recording is already sorted in one piece
Then it chops it into segments, and calculate unit metrics separately for each segment
"""
# (1) read filt.mda for how many samples there are
# (2) for each defined segment, create new filt_seg.mda and firings_seg.mda and calculate template and metrics
#     and save template and metrics to a file

import os
from time import time
import gc

import numpy as np
from mountainlab_pytools import mlproc as mlp

from utils.misc import recursively_empty_dir
from utils import mdaio
from utils.read_mda import readmda

# temporary directory is usually fixed to this path
ML_TEMP_DIR = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/ml_temp"
os.environ['ML_TEMPORARY_DIRECTORY']=ML_TEMP_DIR

# fixed for spinal cord recordings
F_SAMPLE = 30000

# BEG USER SETTINGS
SESSION_PATH = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/MustangContinuous/Mustang_220126_125248/onePiece/"
SEGS_PATH = os.path.join(SESSION_PATH, "onePieceSegged_1hours")
SEG_LEN_SECONDS = 3600 # 1 hour
# END USER SETTINGS


def eval_single_segment(filt_file_obj, firings_arr, seg_start_sample, seg_n_samples, target_folder, f_sample):
    """
    Evaluate one segment: 
    First excerpt filtered mda and firing
    Then go through the Mountainlab stuff
    * target_folder is the direct parent folder of the .mda and metric files *
    * seg_start_sample is 0-based *
    """
    # file path strings
    seg_filt_mda_path = os.path.join(target_folder, "filt_seg.mda")
    seg_firings_mda_path = os.path.join(target_folder, "firings_seg.mda")
    seg_templates_mda_path = os.path.join(target_folder, "templates.mda")
    
    # prepare filt_seg.mda
    print("  Reading chunk from filt.mda")
    n_chs = filt_file_obj.N1()
    ts_ovr = time()
    tmp_filtdata = filt_file_obj.readChunk(i1=0, N1=n_chs, i2=seg_start_sample, N2=seg_n_samples)
    t_read = time()
    print("  Reading done in %.2f seconds. Writing chunk to filt_seg.mda" % (t_read-ts_ovr))
    writer = mdaio.DiskWriteMda(seg_filt_mda_path, (n_chs, seg_n_samples), dt="int16")
    writer.writeChunk(tmp_filtdata, i1=0, i2=0)
    t_write = time()
    print("  Written. Done in %.2f seconds" % (t_write-t_read))
    del(tmp_filtdata)
    gc.collect()

    # prepare firings_seg.mda
    print("  Preparing firings_seg.mda")
    firings_mask = np.logical_and(firings_arr[1,:]>=1+seg_start_sample, firings_arr[1,:]<=seg_start_sample+seg_n_samples) # NOTE firings stamp is 1-based
    firings_seg = firings_arr[:, firings_mask]
    firings_seg[1, :] = firings_seg[1, :] - seg_start_sample # manually align seg_start_sample to sample#1 in the generated firings_seg.mda
    mdaio.writemda64(firings_seg, seg_firings_mda_path)
    
    # here come mounainlab pytools 
    # mlp.runProcess(proc_name_str, input_dict, output_dict, params_dict, options_dict)
    # ts = time()
    # compute templates
    mlp.runProcess(
        "ephys.compute_templates", 
        dict(timeseries=seg_filt_mda_path, firings=seg_firings_mda_path),
        dict(templates_out=seg_templates_mda_path),
        {}, {}
    )

    # Now, to compute metrics we gotta re-do the whitening and mask_out_artifacts,
    # However since the sorting was done on the complete continuous data, 
    # and we are computing metrics on this segmented data,
    # I am not sure how much the results hold up
    # phuc me
    mlp.runProcess(
        "ephys.whiten",
        dict(timeseries=seg_filt_mda_path),
        dict(timeseries_out=os.path.join(target_folder, "pre1.mda.prv")),
        {}, {}
    )
    mlp.runProcess(
        "ephys.mask_out_artifacts",
        dict(timeseries=os.path.join(target_folder, "pre1.mda.prv")),
        dict(timeseries_out=os.path.join(target_folder, "pre.mda.prv"))
    )
    mlp.runProcess(
        "ephys.compute_cluster_metrics",
        {
            "timeseries": os.path.join(target_folder, "pre.mda.prv"), 
            "firings": seg_firings_mda_path
        },
        {"metrics_out": os.path.join(target_folder, "cluster_metrics.json")},
        {
            "samplerate":str(f_sample), 
            "clip_size":"100", 
            "refrac_msec": 2
        },
        {}
    )
    mlp.runProcess(
        "ms3.isolation_metrics",
        {
            "timeseries": os.path.join(target_folder, "pre.mda.prv"), 
            "firings": seg_firings_mda_path
        },
        {
            "metrics_out": os.path.join(target_folder, "isolation_metrics_out.json"),
            "pair_metrics_out": os.path.join(target_folder, "pair_metrics_out.json")
        },
        {"compute_bursting_parents": "true"}, # TODO check if string or bool is needed here
        {}
    )
    mlp.runProcess(
        "ms3.combine_cluster_metrics",
        {
            "metrics_list": os.path.join(target_folder, "cluster_metrics.json"),
            "metrics_list": os.path.join(target_folder, "isolation_metrics_out.json")
        },
        {"metrics_out": os.path.join(target_folder, "combine_metrics_new.json")}
    )
    
    print("  Emptying $ML_TEMPORARY_DIRECTORY :", os.environ['ML_TEMPORARY_DIRECTORY'])
    recursively_empty_dir(os.environ['ML_TEMPORARY_DIRECTORY'])
    print("--Segment finished. Elapsed time: %.2f seconds" % (time()-ts_ovr))
    


def eval_session_in_segs(session_srcdata_path, session_segs_path, seg_len_seconds, f_sample):
    print("Hi")
    raw_filt_path = os.path.join(session_srcdata_path, "filt.mda") # large file we can only read by chunk
    firings_path = os.path.join(session_srcdata_path, "firings.mda") # firings.mda is usually several hundred MB, can afford to read all into RAM
    
    filt_mda_reader = mdaio.DiskReadMda(raw_filt_path)
    total_n_samples = filt_mda_reader.N2()
    print("Total duration of data: %.2f seconds" % (total_n_samples/f_sample))
    seg_n_samples = int(seg_len_seconds*f_sample)
    n_segs = int(np.ceil(filt_mda_reader.N2()/seg_n_samples))
    print("# segments: %d" % (n_segs))

    firings_arr = readmda(firings_path).astype(np.int64)
    seg_start_sample = 0
    ts_session = time()
    for i_seg in range(n_segs):
        target_folder = os.path.join(session_segs_path, "seg%d"%(i_seg))
        print("\n\n\n----------------\n--------I_SEG=%d--------\n----------------"%(i_seg))
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
            print("  CREATED FOLDER: %s" % (target_folder))
        if seg_start_sample+seg_n_samples <= total_n_samples:
            eval_single_segment(filt_mda_reader, firings_arr, seg_start_sample, seg_n_samples, target_folder, f_sample)
            seg_start_sample += seg_n_samples
        else:
            # last shortened segment
            this_seg_n_samples = total_n_samples - seg_start_sample
            eval_single_segment(filt_mda_reader, firings_arr, seg_start_sample, this_seg_n_samples, target_folder, f_sample)
    print("Entire session done. Elapsed time: %.2f seconds." % (time()-ts_session))


eval_session_in_segs(SESSION_PATH, SEGS_PATH, SEG_LEN_SECONDS, F_SAMPLE)
