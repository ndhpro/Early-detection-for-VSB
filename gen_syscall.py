import os
import sys
import json
import networkx as nx
import karateclub


# python3 gen_syscall.py <report_dir> <file_list.csv>
full_report_path = sys.argv[1]
dst_path = 'syscall/'
with open(sys.argv[2], 'r') as f:
    flist = f.readlines()

for dir_ in flist:
    strace_list = list()
    report_path = full_report_path + dir_[:-1] + '/'
    if os.path.exists(dst_path + dir_ + '.json'):
        continue
    print(dir_[:-1])
    for _, _, files in os.walk(report_path):
        G = dict()
        G['edges'] = list()
        node = set()
        files.sort()
        for file_name in files:
            if file_name.startswith('strace'):
                with open(report_path + file_name, 'r') as f:
                    data = json.load(f)
                for syscall in data:
                    node.add(syscall['name'])
                if len(node) == 300:
                    break
        node = list(node)
        for file_name in files:
            if file_name.startswith('strace'):
                with open(report_path + file_name, 'r') as f:
                    data = json.load(f)
                u = -1
                for syscall in data:
                    v = node.index(syscall['name'])
                    if u >= 0:
                        G['edges'].append([u, v])
                    u = v
        with open(dst_path + dir_ + '.json', 'w') as f:
            json.dump(G, f)
