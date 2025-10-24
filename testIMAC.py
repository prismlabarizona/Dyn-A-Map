import re
import os
import time
import math
import random
import mapIMAC
import mapWB
import numpy as np
import pandas as pd
#import swap_subarrays
#import detect_and_swap_error_subarray
import detect_and_swap_error_subarray_with_reference
from pandas import ExcelWriter



start = time.time()

n_runs = 3
flip_subarray = [None] + [i+1 for i in range(52)]

# Inputs
data_dir = 'data'
spice_dir = 'spice'
dataset_file = 'test_data.csv'
label_file = 'test_labels.csv'
weight_var = 0.0
testnum = 10
testnum_per_batch = 5
firstimage = 0
vdd = 0.8
vss = -0.8
tsampling = 1
nodes = [400, 120, 84, 10]
xbar = [32, 32]
gain = [30, 30, 10]
tech_node = 9e-9
metal = 3 * tech_node
T = 22e-9
H = 20e-9
L = 15 * tech_node
W = 12 * tech_node
D = 5 * tech_node
eps = 20 * 8.854e-12
rho = 1.9e-8
rlow = 5e3
rhigh = 15e3

hpar = [math.ceil((x+1)/xbar[0]) for x in nodes]
hpar.pop()
vpar = [math.ceil(x/xbar[1]) for x in nodes]
vpar.pop(0)

print(f'Rlow={rlow}')
print(f'Rhigh={rhigh}')
print(f'Horizontal partitions = {hpar}')
print(f'Vertical partitions = {vpar}')

# Helpers
def update_neuron(rlow, rhigh):
    with open(spice_dir+'/neuron.sp', "r+") as ff:
        data = ff.readlines()
        for i, line in enumerate(data):
            if 'Rlow' in line:
                data[i] = f'Rlow in2 input {rlow}\n'
            if 'Rhigh' in line:
                data[i] = f'Rhigh input out {rhigh}\n'
        ff.seek(0)
        ff.truncate()
        ff.writelines(data)

def update_diff(gain, LayerNUM):
    with open(spice_dir+f'/diff{LayerNUM}.sp', "r+") as ff:
        data = ff.readlines()
        for i, line in enumerate(data):
            if 'Gain' in line:
                data[i] = f'*Differential Amplifier with Gain={gain}\n'
            if 'R3' in line:
                data[i] = f'R3 n1 out {gain}k\n'
            if 'R4' in line:
                data[i] = f'R4 n2 0 {gain}k\n'
        ff.seek(0)
        ff.truncate()
        ff.writelines(data)

def findavg(line):
    i, m = 0, 0
    while m == 0:
        i += 1
        if line[i] == '=': s1 = i+1
        if line[i] == 'f': s2 = i-1; m = 1
        if line[i] == '\n': s2 = i; m =1
    volt = line[s1:s2].replace(" ", "")
    for k,v in zip("munpfkxtg", ["e-3","e-6","e-9","e-12","e-15","e3","e6","e9","e12"]):
        volt = volt.replace(k,v)
    return volt

def findat(line):
    i, m = 0, 0
    while m == 0:
        i += 1
        if line[i] == '=': s1 = i+1
        if line[i] == 'w': s2 = i-1; m = 1
        if line[i] == '\n': s2 = i; m =1
    volt = line[s1:s2].replace(" ", "")
    for k,v in zip("munpfakxtg", ["e-3","e-6","e-9","e-12","e-15","e-18","e3","e6","e9","e12"]):
        volt = volt.replace(k,v)
    return volt

# Dataset preprocessing
dataset = np.genfromtxt(f'{data_dir}/{dataset_file}', delimiter=',').flatten()
dataset_bin = np.sign(dataset)
with open(f'{data_dir}/testinput.txt', "w") as f:
    for val in dataset_bin:
        f.write(f"{val}\n")

label = np.genfromtxt(f'{data_dir}/{label_file}', delimiter=',').flatten()
with open(f'{data_dir}/testlabel.txt', "w") as f:
    for val in label:
        f.write(f"{val}\n")

with open(f'{data_dir}/testinput.txt', "r") as data_r, open(f'{data_dir}/testlabel.txt', "r") as label_r:
    data_all = data_r.readlines()
    label_all = label_r.readlines()

results = []
summary_rows = []

for run in range(1, n_runs+1):
    print(f"\n========== Run {run} ==========")

    for i in range(len(nodes)-1):
        update_diff(gain[i], i+1)

    mapWB.mapWB(len(nodes), rlow, rhigh, nodes, data_dir, weight_var)

    
    #import dynamic_remap
    #dynamic_remap.main()

    # --------- SWAP STEP: For Layer 3, before HSPICE -----------
    # Add at the top of testIMAC.py
    

    '''detect_and_swap_error_subarray.detect_and_swap_error_subarray(
    posweight_file='data/posweight3.txt',
    negweight_file='data/negweight3.txt',
    posweight_out='data/posweight3_remapped.txt',
    negweight_out='data/negweight3_remapped.txt',
    num_rows=10, col_blocks=[32, 32, 20]
)'''

    '''swap_subarrays.swap_high_low_subarray(
        posweight_file='data/posweight3.txt',
        negweight_file='data/negweight3.txt',
        excel_file='results.xlsx',
        posweight_out='data/posweight3_remapped.txt',
        negweight_out='data/negweight3_remapped.txt',
        num_rows=10, num_cols=84,
        col_blocks=[32, 32, 20], row_blocks=[10]
    )'''



    detect_and_swap_error_subarray_with_reference.detect_and_swap_error_subarray_with_reference(
    posweight_file='data/posweight3.txt',
    negweight_file='data/negweight3.txt',
    posweight_ref='data/posweight3_reference.txt',
    negweight_ref='data/negweight3_reference.txt',
    posweight_out='data/posweight3_remapped.txt',
    negweight_out='data/negweight3_remapped.txt',
    num_rows=10, col_blocks=[32,32,20]
)

    # -----------------------------------------------------------    
    batch = testnum // testnum_per_batch
    image_num = 0
    testimage = firstimage

    for b in range(batch):
        out_list, label_list, pwr_list = [], [], []

        data_sim = data_all[int(testimage*nodes[0]):int((testimage+testnum_per_batch)*nodes[0])]
        label_sim = label_all[int(testimage*nodes[-1]):int((testimage+testnum_per_batch)*nodes[-1])]

        for value in label_sim:
            label_list.append(float(value))

        with open(f'{data_dir}/data_sim.txt', "w") as sim_w:
            for j in range(int(testnum_per_batch*nodes[0])):
                sim_w.write(f"{float(data_sim[j])*vdd} ")

        mapIMAC.mapIMAC(nodes, len(nodes), hpar, vpar, metal, T, H, L, W, D, eps, rho, weight_var,
                        testnum_per_batch, data_dir, spice_dir, vdd, vss, tsampling, run, flip_subarray = run)

        os.chdir(spice_dir)
        os.system('hspice classifier.sp > output.txt')
        os.chdir('..')

        with open(f'{spice_dir}/output.txt', "r") as out_r:
            for line in out_r:
                if 'vout' in line:
                    out_list.append(float(findat(line)))
                if 'pwr' in line:
                    pwr_list.append(float(findavg(line)))

        for j in range(testnum_per_batch):
            pred, err_flag = [], 0
            list_max = max(out_list[nodes[-1]*j:nodes[-1]*(j+1)])

            for k in range(nodes[-1]):
                pred.append(1.0 if out_list[nodes[-1]*j+k] == list_max else 0.0)
                if pred[k] != label_list[nodes[-1]*j+k]:
                    err_flag = 1

            results.append({
                'Run': run,
                'Image': j+image_num+1,
                'Actual': label_list[nodes[-1]*j:nodes[-1]*(j+1)],
                'Predicted': pred,
                'Correct': 'No' if err_flag else 'Yes',
                'Power': pwr_list[j]
            })

            print(f"Run {run} Image {j+image_num+1}")
            print(f"Actual label: {label_list[nodes[-1]*j:nodes[-1]*(j+1)]}")
            print(f"Predicted label: {pred}")
            print("Wrong prediction!" if err_flag else "Correct prediction")
            print(f"Power consumption = {pwr_list[j]}")

        image_num += testnum_per_batch
        testimage += testnum_per_batch

    # === Area and summary after each run ===
    xbar_num = sum(np.multiply(hpar, vpar))
    xbar_area = W * L * xbar[0] * xbar[1] * xbar_num * 1e12
    switch_area = 0.56 * xbar_num * (xbar[0] + xbar[1])
    area = xbar_area + switch_area

    total_images = len([r for r in results if r['Run'] == run])
    wrong = sum(1 for r in results if r['Run'] == run and r['Correct'] == 'No')
    error_rate = (wrong / float(total_images) * 100)
    accuracy = 100 - error_rate
    avg_power = sum(r['Power'] for r in results if r['Run'] == run) / float(total_images)

    print(f"[Run {run}] Area = {area} µm²")
    print(f"[Run {run}] Total error = {wrong}")
    print(f"[Run {run}] Error rate = {error_rate}%")
    print(f"[Run {run}] Accuracy = {accuracy}%")
    print(f"[Run {run}] Average power = {avg_power}")

    summary_rows.append({
        'Run': run,
        'Total Area (µm²)': area,
        'Total Error': wrong,
        'Error Rate (%)': error_rate,
        'Accuracy (%)': accuracy,
        'Average Power': avg_power
    })

end = time.time()
seconds = math.floor(end-start)
minutes = math.floor(seconds/60)
hours = math.floor(minutes/60)
tmin = minutes - (60*hours)
tsec = seconds - (hours*3600) - (tmin*60)

print(f"Program Execution Time = {hours}h {tmin}m {tsec}s")

# Save Excel
with ExcelWriter("test.xlsx", engine="openpyxl") as writer:
    pd.DataFrame(results).to_excel(writer, sheet_name='Per-Image Results', index=False)
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)

print("All runs complete. Results saved to results.xlsx")
