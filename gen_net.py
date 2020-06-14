import subprocess
import sys
import pandas as pd
import os
from scapy.all import *


# python3 gen_net.py <report_dir> <file_list.csv> <net.csv>
with open(sys.argv[2], 'r') as f:
    flist = f.readlines()
for dir_ in flist:
    if os.path.exists('net/' + dir_[:-1] + '.csv'):
        continue
    dir_path = sys.argv[1] + dir_[:-1] + '/tcpdump.pcap'
    pcap = rdpcap(dir_path)
    print(dir_path)
    wrpcap('temp.pcap','')
    for i, pkt in enumerate(pcap):
        wrpcap('temp.pcap', pkt, append=True)
        if i == 49:
            break
    p = subprocess.call('cd CICFlowMeter-4.0/bin/ && ./cfm ../../temp.pcap ../../net/', shell=True)
    os.rename('net/temp.pcap_Flow.csv', 'net/' + dir_[:-1] + '.csv')
    os.remove('temp.pcap')

attributes = ['Sum ', 'Max ']
features = ['Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
            'TotLen Fwd Pkts', 'TotLen Bwd Pkts', 'Fwd IAT Tot', 'Bwd IAT Tot',
            'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
            'Fwd Header Len', 'Bwd Header Len',
            'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt', 'PSH Flag Cnt',
            'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count', 'ECE Flag Cnt',
            'Init Fwd Win Byts', 'Init Bwd Win Byts',
            'Fwd Act Data Pkts']
headers = [att + ft for ft in features for att in attributes]
headers.insert(0, 'Num of flow')
headers.insert(0, 'name')
headers.append('label')

data = dict()
for header in headers:
    data[header] = list()
with open(sys.argv[2], 'r') as f:
    flist = f.readlines()
for dir_ in flist:
    flow_path = 'net/' + dir_[:-1] + '.csv'
    data['name'].append(dir_[:-1])
    try:
        flow = pd.read_csv(flow_path)
        data['Num of flow'].append(len(flow.index))
        for feature in features:
            data['Sum ' + feature].append(flow[feature].sum())
            data['Max ' + feature].append(max(flow[feature], default=0))
    except Exception as e:
        print(e)
        data['Num of flow'].append(0)
        for feature in features:
            data['Sum ' + feature].append(0)
            data['Max ' + feature].append(0)

if sys.argv[2].startswith('list_malware'):
    data['label'] = 0
else:
    data['label'] = 1

dp = pd.DataFrame.from_dict(data)
dp.to_csv(sys.argv[3], index=None, columns=headers)
