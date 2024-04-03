import os
import re

def get_segment_index(segment_name: str) -> int:
    return int(re.search("seg([0-9]+)", segment_name)[1])

def connec_wrapper_main(postproc_folder, fs=30000):
    segment_folders = os.listdir(postproc_folder)
    segment_folders = sorted(segment_folders, key=get_segment_index)
    segment_folders = [os.path.join(postproc_folder, seg_folder) for seg_folder in segment_folders]
    
    matlab_argin1 = "{"
    for i_seg, foldername in enumerate(segment_folders[:-1]):
        matlab_argin1 += "\'%s\'"%(os.path.join(foldername, "customPostproc220509", "mdas_screened"))
        matlab_argin1 += ", "
    matlab_argin1 += "}"
    matlab_argin2 = str(fs)
    
    matlab_cmd = "matlab -nodisplay -nosplash -nodesktop -r "
    matlab_cmd += "\"cd utils_matlab; "
    matlab_cmd += "process_segs_batch(%s, %s); exit;\"" % (matlab_argin1, matlab_argin2)
    print(matlab_cmd)
    os.system(matlab_cmd)

postproc_folder = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/MustangContinuous/Mustang_220126_125248/onePiece/onePieceSegged_1hours"
connec_wrapper_main(postproc_folder)
