import os
import sys
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scapy.utils import RawPcapReader


def process_pcap(file_name):
    count = 0
    for (pkt_data, pkt_metadata,) in RawPcapReader(file_name):
        count += 1
    return count

data = list()
with open('list_benign.csv', 'r') as f:
    flist = f.readlines()
# for dir_ in flist:
#     dir_ = dir_[:-1]
#     dir_path = '/media/ais/data/final_report_benign/' + dir_
#     obj = dict()
#     obj['name'] = dir_
#     obj['sys'] = 0
#     for _, _, files in os.walk(dir_path):
#         for fname in files:
#             if fname.startswith('strace'):
#                 fpath = dir_path + '/' + fname
#                 with open(fpath, 'r') as f:
#                     js = json.load(f)
#                 obj['sys'] += len(js)
#     data.append(obj)
# pd.DataFrame(data).to_csv('stat_sys_benign.csv', index=None)

# for dir_ in flist:
#     dir_ = dir_[:-1]
#     path = '/media/ais/data/final_report_benign/' + dir_ + '/tcpdump.pcap'
#     obj = dict()
#     obj['name'] = dir_
#     obj['net'] = process_pcap(path)
#     data.append(obj)

# pd.DataFrame(data).to_csv('stat/net_benign.csv', index=None)
eda = pd.read_csv('stat/net_benign.csv')
print(eda.describe())
bin_values = np.arange(start=0, stop=400, step=4)
eda['net'].hist(bins=bin_values)
plt.xlabel('Number of packages in each pcap')
plt.ylabel('Frequency')
plt.show()