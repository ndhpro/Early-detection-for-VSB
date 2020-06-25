import os
import sys
import json
import numpy as np
import pandas as pd


# python3 gen_per.py <report_dir> <file_list.csv> <per.csv>
features = [
    "num_total_running", "num_total_sleeping", "num_total_zombie", "num_total_stopped",
    "cpu_%_us", "cpu_%_sy", "cpu_%_ni", "cpu_%_id",
    "cpu_%_wa", "cpu_%_hi", "cpu_%_si", "cpu_%_st",
    "mem_total", "mem_used", "mem_free", "mem_buffers",
    "swap_total", "swap_used", "swap_free", "swap_cache"
]
full_report_path = sys.argv[1]
with open(sys.argv[2], 'r') as f:
    flist = f.readlines()

per = list()
for dir_ in flist:
    report_path = full_report_path + dir_[:-1] + '/top.json'
    vt = dict()
    vt['name'] = dir_[:-1]
    with open(report_path, 'r') as f:
        data = json.load(f)
    
    for ft in features:
        arr = list()
        for i, step in enumerate(data):
            arr.append(step[ft])
            if i >= 20:
                break
        arr = np.array(arr, dtype=np.float)
        vt[ft + '_mean'] = np.mean(arr)
        vt[ft + '_std'] = np.std(arr)
        vt[ft + '_max'] = np.max(arr)
        vt[ft + '_min'] = np.min(arr)
    
    if 'malware' in sys.argv[2]:
        vt['label'] = 1
    else:
        vt['label'] = -1
    per.append(vt)

header = ['name']
for ft in features:
    for att in ['_mean', '_std', '_max', '_min']:
        header.append(ft + att)
header.append('label')
pd.DataFrame(per)[header].to_csv(sys.argv[3], index=None)
