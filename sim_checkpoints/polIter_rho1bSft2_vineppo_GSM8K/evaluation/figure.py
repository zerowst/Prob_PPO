import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import json

currect_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))

files = glob.glob(script_dir + "/ckpt*")
# print(files)
target_file = 'log.json'


def get_acc():
    correct_acc = []
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
        correct_acc.append(res['correct_frac'])
        print(correct_acc)
    return correct_acc


def figure(points, axis):
    plt.plot(axis, points, label='similarity', marker='o')

    plt.xlabel('Training steps')
    plt.ylabel('Correct_accuracy')
    plt.title('Training performance of similarity method')
    plt.legend()
    plt.grid(True)

    plt.show()
    plt.savefig('acc.pdf')


if __name__ == '__main__':
    axis = range(10, 161, 10)
    acc_points = get_acc()

    figure(points=acc_points, axis=axis)








