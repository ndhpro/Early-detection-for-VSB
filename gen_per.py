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
    vt = list()
    with open(report_path, 'r') as f:
        data = json.load(f)
    if len(data) <= 20:
        for step in data:
            for ft in features:
                vt.append(float(step[ft]))
        while len(vt) < 20 * 20:
            vt.append(0)
    else:
        d = len(data) // 20
        for i, step in enumerate(data):
            if i / d == 20:
                break
            if i % d == 0:
                for ft in features:
                    vt.append(float(step[ft]))

    vt.insert(0, dir_[:-1])
    if sys.argv[2].startswith('list_malware'):
        vt.append(0)
    else:
        vt.append(1)
    per.append(vt)

header = ['name']
for i in range(20):
    header.extend([ft + str(i) for ft in features])
header.append('label')
pd.DataFrame(per).to_csv(sys.argv[3], header=header, index=None)
