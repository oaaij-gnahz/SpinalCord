'''
read channel info for OpenEphys data recorded by BT
9/20/2022 jz103
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

VALID_SESSION_LAMBDA = lambda x: (('_' in x) and ('.' not in x) and ('statfigs' not in x))
SESSION_XML_NAMING_PATTERN = r"([0-9]+_[0-9]+)_Impedances.xml"
DATETIME_STR_PATTERN = "%m_%d"
HEADSTAGE_NAME = 'B1'
IMPEDANCE_MEASUREMENT_OFFSET_DAY = 2

def get_datetimestr_from_filename(filename):
    """
        Assume the rhd filename takes the form of either
        ParentPath/Animalname_YYMMDD_HHMMSS.rhd
        or 
        Animialname_YYMMDD_HHMMSS.rhd
    """
    tmp = filename.split("/")[-1].split("_")
    return tmp[1]+'_'+tmp[2].split('.')[0]



def procfunc_prepdata(resfolder : str, xml_folder : str):
    list_of_xmls = os.listdir(os.path.join(resfolder, xml_folder)) 
    ch_impedances = []
    session_datetimes = []
    for i_work, (xmlname) in enumerate(list_of_xmls):
        print(i_work)
        # info_dict={}
        datetimestr = re.match(SESSION_XML_NAMING_PATTERN, xmlname)[1]
        print(xmlname, datetimestr)
        session_datetime = datetime.datetime.strptime(datetimestr, DATETIME_STR_PATTERN)
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
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    timedelta_floats = np.array([(sdt - session_datetimes[0]).total_seconds() for sdt in session_datetimes]) + IMPEDANCE_MEASUREMENT_OFFSET_DAY*24*3600
    # timedelta_floats = timedelta_floats/timedelta_floats[-1]
    violin_parts = ax.violinplot(ch_impedances, positions=timedelta_floats, showmeans=True, widths=timedelta_floats[-1]/timedelta_floats.shape[0]/2, points=500)
    for pc in violin_parts['bodies']:
        pc.set_facecolor('gray')
        pc.set_edgecolor('gray')
    ch_impedances_mean = [np.mean(k) for k in ch_impedances]
    ax.plot(timedelta_floats, ch_impedances_mean, color='k')
    # for i_session in range(timedelta_floats.shape[0]):
    #     ax.text(timedelta_floats[i_session], 2.6e6, str(ch_impedances[i_session].shape[0]))
    # ax.plot(session_datetimes, np.array(session_stats['n_single_units'])/np.array(session_stats['n_ch']), label='# single-units')
    # ax.legend()
    plt.xticks(timedelta_floats, np.round(timedelta_floats/24/3600).astype(int))
    # plt.xticks(timedelta_floats[::3], [t.strftime('%Y-%m-%d') for t in session_datetimes][::3], rotation=45)
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(resfolder, "impedances.svg"))
    plt.close()
    fig2 = plt.figure(figsize=(10,5))
    plt.plot(timedelta_floats, [ch_impedances[i_session].shape[0] for i_session in range(timedelta_floats.shape[0])], c='k', marker='.')
    # plt.xticks(timedelta_floats[::3], [t.strftime('%Y-%m-%d') for t in session_datetimes][::3], rotation=45)
    plt.xticks(timedelta_floats, np.round(timedelta_floats/24/3600).astype(int))
    plt.ylim([0,32.5])
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(resfolder, "valid_ch_cnts.svg"))
    plt.close()

if __name__=="__main__":
    # rawfolder_root = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/data/corolla_chronic"
    # mdafolder_root = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/data_converted/corolla_chronic"
    resfolder_root = "/media/hanlin/Liuyang_10T_backup/jiaaoZ/msort_results/BenMouse1"
    xml_folder = "impedances"
    procfunc_prepdata(resfolder_root, xml_folder)