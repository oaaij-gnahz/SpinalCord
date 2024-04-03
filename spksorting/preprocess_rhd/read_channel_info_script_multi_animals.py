'''
read channel info
8/31/2022 jz103
'''

import os
import gc
import warnings
from copy import deepcopy
from time import time
import re
import datetime
import json
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

VALID_SESSION_LAMBDA = lambda x: (('_' in x) and ('.' not in x) and ('__' not in x) and ('bad' not in x))
DATA_SAVE_FOLDER = "../_pls_ignore_chronic_data"

# RHD PARAMETERS
SESSION_NAMING_PATTERN = r"[A-Za-z_ ]+_([0-9]+_[0-9]+)"
# OPENEPHYS PARAMETERS
SESSION_XML_NAMING_PATTERN = r"([0-9]+_[0-9]+)_Impedances.xml"
DATETIME_STR_PATTERN = "(%Y)%m_%d"
HEADSTAGE_NAME = 'B1'
IMPEDANCE_MEASUREMENT_OFFSET_DAY = 2

# def get_datetimestr_from_filename(filename):
#     """
#         Assume the rhd filename takes the form of either
#         ParentPath/Animalname_YYMMDD_HHMMSS.rhd
#         or 
#         Animialname_YYMMDD_HHMMSS.rhd
#     """
#     tmp = filename.split("/")[-1].split("_")
#     return tmp[1]+'_'+tmp[2].split('.')[0]

def get_surgery_date(animalfolder):
    with open(os.path.join(animalfolder, "general__info", "surgerydate.txt"), "r") as f:
        x=f.readline().split('\n')[0]
        print(x)
    return datetime.datetime.strptime(x, "%Y%m%d")

def procfunc_prepdata_rhd(list_of_res_folders : list, animal_info : dict, plot_axes : list):
    ch_impedances = []
    session_datetimes = []
    animal_name = list_of_res_folders[0].split('/')[-2].split('_')[0]
    surgery_date = animal_info['surgery_date']
    session_datetimestrs = []
    for i_work, (resfoldername) in enumerate(list_of_res_folders):
        # print(i_work)
        # info_dict={}
        relevant_fnames = []
        relevant_fnames.append(os.path.join(resfoldername, "firings.mda"))
        relevant_fnames.append(os.path.join(resfoldername, "templates.mda"))
        if not np.all([os.path.exists(f) for f in relevant_fnames]):
            # count impedance measurements only when valid recording & sorting exist.
            continue
        tmp_split = resfoldername.split("/")
        ressubfoldername = tmp_split[-1] if tmp_split[-1]!="" else tmp_split[-2]
        datetimestr = re.match(SESSION_NAMING_PATTERN, ressubfoldername)[1]
        print('    ', ressubfoldername, datetimestr)
        session_datetime = datetime.datetime.strptime(datetimestr, "%y%m%d_%H%M%S")
        # info_dict = read_one_session(resfoldername)
        with open(os.path.join(resfoldername, 'session_rhd_info.json'), "r") as f_read:
            info_dict = json.load(f_read)
        if info_dict is None:
            continue # empty session
        chs_info = info_dict['chs_info']
        # TO ACCOMODATE SOME NORA SESSIONS
        if len(chs_info) > 32:
            chs_info = chs_info[16:48]
        chs_impedance = np.array([e['electrode_impedance_magnitude'] for e in chs_info])
        chs_impedance_valid = chs_impedance[chs_impedance<3e6]
        session_datetimes.append(session_datetime)
        ch_impedances.append(chs_impedance_valid)
        session_datetimestrs.append(datetimestr)
    timedelta_floats = np.array([(sdt - surgery_date).total_seconds() for sdt in session_datetimes])
    ch_impedances_mean = np.array([np.mean(k) for k in ch_impedances])
    valid_ch_counts = np.array([ch_impedances[i_session].shape[0] for i_session in range(timedelta_floats.shape[0])])
    idx_sorted = np.argsort(timedelta_floats)
    plot_axes[0].plot(timedelta_floats[idx_sorted], ch_impedances_mean[idx_sorted], label=animal_name, marker='.')
    plot_axes[1].plot(timedelta_floats[idx_sorted], valid_ch_counts[idx_sorted], label=animal_name, marker='.')
    # return data
    return timedelta_floats[idx_sorted], [ch_impedances[idx] for idx in idx_sorted], [session_datetimestrs[idx] for idx in idx_sorted]

def procfunc_prepdata_oe(resfolder : str, animal_info : dict, plot_axes : list):
    animal_name = resfolder.split('/')[-1].split('_')[0]
    xml_folder = animal_info['xml_folder']
    surgery_date = animal_info['surgery_date']
    list_of_xmls = os.listdir(os.path.join(resfolder, xml_folder)) 
    ch_impedances = []
    session_datetimes = []
    session_datetimestrs = []
    for i_work, (xmlname) in enumerate(list_of_xmls):
        # print(i_work)
        # info_dict={}
        datetimestr = re.match(SESSION_XML_NAMING_PATTERN, xmlname)[1]
        print('    ', xmlname, datetimestr)
        session_datetime = datetime.datetime.strptime("(2022)"+datetimestr, DATETIME_STR_PATTERN)
        # info_dict = read_one_session(resfoldername)
        root = ET.parse(os.path.join(resfolder, xml_folder, xmlname)).getroot()
        for headstage in root:
            if headstage.tag != "HEADSTAGE" or headstage.attrib['name'] != HEADSTAGE_NAME:
                continue
            chs_numbers0 = [ch.attrib['number'] for ch in headstage]
            chs_impedance = np.array([float(ch.attrib['magnitude']) for ch in headstage])
            break
        chs_impedance_valid = chs_impedance[chs_impedance<3e6]
        session_datetimes.append(session_datetime)
        ch_impedances.append(chs_impedance_valid)
        session_datetimestrs.append(datetimestr)
    timedelta_floats = np.array([(sdt - surgery_date).total_seconds() for sdt in session_datetimes])
    ch_impedances_mean = np.array([np.mean(k) for k in ch_impedances])
    valid_ch_counts = np.array([ch_impedances[i_session].shape[0] for i_session in range(timedelta_floats.shape[0])])
    idx_sorted = np.argsort(timedelta_floats)
    plot_axes[0].plot(timedelta_floats[idx_sorted], ch_impedances_mean[idx_sorted], label=animal_name, marker='.')
    plot_axes[1].plot(timedelta_floats[idx_sorted], valid_ch_counts[idx_sorted], label=animal_name, marker='.')
    # return data
    return timedelta_floats[idx_sorted], [ch_impedances[idx] for idx in idx_sorted], [session_datetimestrs[idx] for idx in idx_sorted]


def process_all_animals(animal_folders, recording_formats, save):
    fig1 = plt.figure(figsize=(10,5))
    ax1 = fig1.add_subplot(111)
    fig2 = plt.figure(figsize=(10,5))
    ax2 = fig2.add_subplot(111)
    for i_animal, (animal_folder, recording_format) in enumerate(zip(animal_folders, recording_formats)):
        print(i_animal, animal_folder)
        sessions_list = list(filter(VALID_SESSION_LAMBDA, os.listdir(animal_folder)))
        if recording_format=="RHD":
            sessions_list = sorted(
                sessions_list, 
                key=lambda x: datetime.datetime.strptime(re.match(SESSION_NAMING_PATTERN, x)[1], "%y%m%d_%H%M%S") 
            )
            res_folders_list = list(map(lambda x: os.path.join(animal_folder, x), sessions_list))
            surgery_date = get_surgery_date(animal_folder)
            animal_info = {"surgery_date": surgery_date}
            print("  Surgery Date:", surgery_date)
            timedelta_floats, ch_impedances, datetimestrs = procfunc_prepdata_rhd(res_folders_list, animal_info, [ax1,ax2])
            ch_impedance_dict = dict(zip(["session%d"%(d) for d in range(len(ch_impedances))], ch_impedances))
            animal_name = res_folders_list[0].split('/')[-2].split('_')[0]
        elif recording_format=="OPENEPHYS":
            surgery_date = get_surgery_date(animal_folder)
            animal_info = {"surgery_date": surgery_date, "xml_folder": "impedances"}
            print("  Surgery Date:", surgery_date)
            timedelta_floats, ch_impedances, datetimestrs = procfunc_prepdata_oe(animal_folder, animal_info, [ax1,ax2])
            ch_impedance_dict = dict(zip(["session%d"%(d) for d in range(len(ch_impedances))], ch_impedances))
            animal_name = animal_folder.split('/')[-1]
        if save:
            np.savez(
                os.path.join(DATA_SAVE_FOLDER, animal_name+'_impedances.npz'),
                time_in_seconds=timedelta_floats,
                **ch_impedance_dict
            )
            df_save = pd.DataFrame()
            df_save['dayAfterSurgery'] = (timedelta_floats/(24*3600)).astype(int)
            df_save['datetime'] = datetimestrs
            df_save['impedances'] = ch_impedances
            df_save.to_csv(os.path.join(DATA_SAVE_FOLDER, animal_name+"_impedances.csv"), index=False)

    print("Data plotted...")
    for ax in [ax1, ax2]:
        ax.legend()
        xtickmax = int(ax.get_xticks()[-1] / (24*3600))
        print(xtickmax)
        ax.set_xticks(np.arange(0, xtickmax, 10)*24*3600)
        ax.set_xticklabels(np.arange(0, xtickmax, 10))
        # ax.set_xticklabels(np.round(ax.get_xticks()/(24*3600)).astype(int))
        ax.set_xlabel("Days after Implantation")
    print("Setting Axis Labels")
    ax1.set_ylabel(r"Avg Impedance ($\Omega$)")
    ax2.set_ylabel(r"# Channels with Impedance < 3M$\Omega$")
    ax2.set_ylim([0, 32.5])
    print("Saving figures")
    # plt.subplots_adjust(bottom=0.1)
    fig1.savefig("../_pls_ignore_impedances.eps")
    fig1.savefig("../_pls_ignore_impedances.png")
    fig2.savefig("../_pls_ignore_valid_ch_cnts.png")
    fig2.savefig("../_pls_ignore_valid_ch_cnts.eps")
    for ax in [ax1,ax2]:
        ax.set_xlim([-1*(10*3600), 120*(24*3600)])
    fig1.savefig("../_pls_ignore_impedances_120days.eps")
    fig1.savefig("../_pls_ignore_impedances_120days.png")
    fig2.savefig("../_pls_ignore_valid_ch_cnts_120days.png")
    fig2.savefig("../_pls_ignore_valid_ch_cnts_120days.eps")
    plt.close()
    plt.close()

if __name__=="__main__":
    animals_list = [
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/BenMouse0", "OPENEPHYS"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/BenMouse1", "OPENEPHYS"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/dorito_chronic", "RHD"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/mustang_chronic", "RHD"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/nacho_chronic", "RHD"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/tacoma_chronic", "RHD"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/corolla_chronic", "RHD"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/yogurt_chronic", "RHD"),
        ("/media/hanlin/Liuyang_10T_backup/jiaaoZ/spinalcord/msort_results/nora_chronic", "RHD")
    ]
    if not os.path.exists(DATA_SAVE_FOLDER):
        os.makedirs(DATA_SAVE_FOLDER)
    process_all_animals([k[0] for k in animals_list], [t[1] for t in animals_list], save=True)