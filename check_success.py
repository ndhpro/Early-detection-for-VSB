import os
import sys
import json


# python3 check_success.py <report_dir> <file_list.csv>
def check_success(g, report_dir):
    for _, dirs, _ in os.walk(report_dir):
        for dir in dirs:
            flag = True
            ldd_path = report_dir + dir + '/ldd.txt'
            with open(ldd_path, 'r') as f:
                data = f.read()
                if 'not found' in data:
                    flag = False

            if not flag:
                continue

            for _, _, files in os.walk(report_dir + dir):
                count = 0
                strace_file = ''
                for file in files:
                    if 'strace' in file:
                        count += 1
                        strace_file = file
                if count == 1:
                    strace_path = report_dir + dir + '/' + strace_file
                    with open(strace_path, 'r') as f:
                        data = json.load(f)
                        if data[0]['return'] != "0":
                            flag = False

            if flag:
                g.write(dir + '\n')


if __name__ == "__main__":
    with open(sys.argv[2], 'w') as g:
        check_success(g, sys.argv[1])
