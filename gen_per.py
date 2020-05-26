import os
import sys
import json


features = [
    "num_user_login",
    "num_total_running", "num_total_sleeping", "num_total_zombie", "num_total_stopped",
    "cpu_%_us", "cpu_%_sy", "cpu_%_ni", "cpu_%_id",
    "cpu_%_wa", "cpu_%_hi", "cpu_%_si", "cpu_%_st", 
    "mem_total", "mem_used", "mem_free", "mem_buffers",
    "swap_total", "swap_used", "swap_free", "swap_cache"
]
full_report_path = sys.argv[1]
dst_path = 'syscall/'
with open('file_list.csv', 'r') as f:
    flist = f.readlines()

with open('perf.csv', 'w') as g:
    for dir_ in flist:
        report_path = full_report_path + dir_[:-1] + '/top.json'
        vt = list()
        with open(report_path, 'r') as f:
            data = json.load(f)
        for step in data:
            for ft in features:
                g.write(str(float(step[ft])) + ',')
        g.write('\n')
            
                    
            

