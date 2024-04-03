import re
import os
from time import time
import utils.fpdf as fpdf


def get_segment_index(segment_name: str) -> int:
    return int(re.search("seg([0-9]+)", segment_name)[1])

def get_cluslabel_from_figname(figname: str) -> int:
    return int(re.search("waveform_clus([0-9]+).png", figname)[1])

os.makedirs("./tmp_pdfs", exist_ok=True)
postproc_folder = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/MustangContinuous/Mustang_220126_125248/onePiece/onePieceSegged_1hours"
segment_folders = os.listdir(postproc_folder)
segment_folders = sorted(segment_folders, key=get_segment_index)
segment_folders = [os.path.join(postproc_folder, seg_folder) for seg_folder in segment_folders]
ts_ovr = time()
pdf = fpdf.FPDF()
for i_seg, segment_folder in enumerate(segment_folders[:-1]):
    print(segment_folder)
    # ts = time()
    fig_path = os.path.join(segment_folder, "customPostproc220509", "postproc_figs", "location.png")
    pdf.add_page(orientation='P', format=(400, 450))
    pdf.image(fig_path, 0, 0, 200)
    # fig_names = list(filter(lambda x: x.startswith("waveform_clus"), os.listdir(segment_folder_figs)))
    # fig_names = sorted(fig_names, key=get_cluslabel_from_figname)
    # fig_names = [os.path.join(segment_folder_figs, figname) for figname in fig_names]
    # pdf = fpdf.FPDF()
    # for fig_path in fig_names:
    #     pdf.add_page(orientation='P', format=(400, 450))
    #     pdf.image(fig_path, 0, 0, 400)
    #     # break
    # pdf.output("tmp_pdfs/seg%d.pdf"%(i_seg))
    # print("Elapsed time:", time()-ts)
    # break
pdf.output("220509.pdf")
print("Overall time:", time()-ts_ovr)