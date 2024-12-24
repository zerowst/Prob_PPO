import numpy as np
import matplotlib as plt
import glob
import os
import json

currect_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))

files = glob.glob(script_dir + "/ckpt*")
print(files)

target_file = 'log.json'

for file in files:
    root_dir = file + '/gsm8k_test'
    for root, dirs, files_ in os.walk(root_dir):
        if target_file in files_:
            full_path = os.path.join(root, target_file)
            print(f"Found: {full_path}")
            break
    else: 
        print('file not found')
    
    with open(full_path, 'r') as f:

        res = json.load(f)
    correct_acc = res['correct_frac']
    print(correct_acc)











