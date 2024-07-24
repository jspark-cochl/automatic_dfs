#############################################
# core.py                                   #
# - This file contains functions that are   #
#   frequently used in auto_dfs.py.         #
#              Jeongsoo Park, Cochl         #
#############################################
import csv
import numpy as np
import scipy
import os
import time
import copy
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy import interpolate
from scipy.optimize import least_squares

from get_params import *


def moving_average(x, tabs=11):  # Moving average
    x_out = np.zeros(x.shape)
    for i in range(tabs, len(x)):
        x_out[i] = np.mean(x[i-tabs:i])
    return x_out


def speed_parsing(fileName):
    speed = 1
    if '1um' in fileName:
        speed = 1
    elif '5um' in fileName:
        speed = 5
    elif '10um' in fileName:
        speed = 10
    elif '20um' in fileName:
        speed = 20
    elif '100nm' in fileName:
        speed = 0.1
    elif '200nm' in fileName:
        speed = 0.2
    else:
        print('Speed not indicated in the filename!!!')
        raise ValueError
    return speed


# def time_axis_parsing(y_length, speed):
#     if speed == 1:
#         t_rt = np.arange((y_length))*0.9461548828125*512/y_length
#     elif speed == 5:
#         t_rt = np.arange((y_length))*0.1908802734375*512/y_length
#     elif speed == 10:
#         t_rt = np.arange((y_length))*0.0949671875000001*512/y_length
#     elif speed == 20:
#         t_rt = np.arange((y_length))*0.047019140625*512/y_length
#     elif speed == 0.1:
#         t_rt = np.arange((y_length))*9.721661328125*512/y_length
#     elif speed == 0.2:
#         t_rt = np.arange((y_length))*4.878356640625*512/y_length
#     else:
#         print('Not supported speed!!!')
#         raise ValueError
#     return t_rt


def baseline_corr1(x_rt, y_rt):
    analysis_window = int(50/512*len(y_rt))
    sd = np.zeros((10))
    for r in range(10):
        sd[r] = np.std(y_rt[r*analysis_window:(r+1)*analysis_window])

    sd_flat = np.min(sd)
    r_flat = np.argmin(sd)
    flat_idx = np.arange(r_flat*analysis_window, (r_flat+1)*analysis_window)
    y_flat = y_rt[flat_idx]
    m_flat = np.mean(y_flat)

    y_rt = y_rt-m_flat  # 0 mean
    x_flat = x_rt[flat_idx]

    # minimum value
    min_val = np.min(y_rt)
    min_idx = np.argmin(y_rt)
    return (y_rt, x_flat, sd_flat, min_val, min_idx, flat_idx)


def baseline_corr2(y_rt, baseline_order, allowed_deviation):
    t_train_tmp = np.where(  np.abs(y_rt)<allowed_deviation  )[0]
    t_train = np.array([t for t in t_train_tmp if t>1])
    y_train = y_rt[t_train]

    x0 = np.ones(baseline_order+1)
    t_test = np.linspace(0,len(y_rt)-1,len(y_rt))
    if baseline_order == 1:
        res_robust = least_squares(fun1, x0, loss='soft_l1', f_scale=allowed_deviation, args=(t_train,y_train))
        y_robust = generate_data1(t_test, *res_robust.x)
    elif baseline_order == 2:
        res_robust = least_squares(fun2, x0, loss='soft_l1', f_scale=allowed_deviation, args=(t_train,y_train))
        y_robust = generate_data2(t_test, *res_robust.x)
    elif baseline_order == 3:
        res_robust = least_squares(fun3, x0, loss='soft_l1', f_scale=allowed_deviation, args=(t_train,y_train))
        y_robust = generate_data3(t_test, *res_robust.x)
    return y_rt-y_robust, y_robust


def is_minima(y_rt, y_idx):
    if y_idx == 0 or y_idx == len(y_rt)-1:
        return 0
    if y_rt[y_idx] < y_rt[y_idx-1] and y_rt[y_idx] < y_rt[y_idx+1]:
        return 1
    else:
        return 0


def fun1(x, t, y):
    return x[0]+x[1]*t-y


def fun2(x, t, y):
    return x[0]+x[1]*t+x[2]*t**2-y


def fun3(x, t, y):
    return x[0]+x[1]*t+x[2]*t**2+x[3]*t**3-y


def generate_data1(t, x_0, x_1):
    return x_0+x_1*t


def generate_data2(t, x_0, x_1, x_2):
    return x_0+x_1*t+x_2*t**2


def generate_data3(t, x_0, x_1, x_2, x_3):
    return x_0+x_1*t+x_2*t**2+x_3*t**3


def read_list(params=None):
    file_list = []
    for dirpath, dirnames, filenames in os.walk(params['data_folder']):
        filenames.sort()
        for filename in [f for f in filenames if f.endswith(params['data_type'])]:
            file_list.append(os.path.join(dirpath,filename))
    return file_list


def parse_units(line):
    """Extract units from the # units: line."""
    parts = line.split(":")[1].strip().split()
    return parts


def read_data(filename=None, params=None):
    # Speed parsing from filename
    speed = speed_parsing(filename)
    if params['data_type'] == 'txt' or params['data_type'] == '.txt':
        """Parse the data from the file."""
        units = []
        data = []

        unit_parsing_flag = False
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith("# units:"):
                    units = parse_units(line)
                    unit_parsing_flag = True
                elif (not line.startswith("#")) and (unit_parsing_flag == True):
                    # Parse the data line
                    values = list(map(float, line.split()))
                    data.append(values)

        # Convert to numpy array for further processing
        data_array = np.array(data)
        ## return units, data_array

        t_rt = data_array[:, 3] # 4th column
        x_rt = t_rt # x axis is time
        y_rt = data_array[:, 1] # 2nd column
    
    elif params['data_type'] == 'npy' or params['data_type'] == '.npy':
        X = np.load(os.path.join(dirpath,filename))
        x_rt = X[0:511]
        y_rt = X[511:1022]

    return x_rt, y_rt, t_rt, speed


def validitycheck1(y_rt=None):
    first_crossing_idx = 0
    for i in range(len(y_rt)):
        if i==0:
            continue
        if y_rt[i] > 0:
            pass
        elif y_rt[i] <= 0:
            first_crossing_idx = i
            break
    five_nm_index = int(len(y_rt)/100)  # 500nm/100=5nm, 500nm/150=3.33nm

    # First peak
    twenty_nm_index = int(len(y_rt)/10)  # 500nm/25=20nm
    first_peak_idx = np.argmin(y_rt[:twenty_nm_index])  # minima before 20nm
    return (first_peak_idx-first_crossing_idx >= five_nm_index), (first_crossing_idx+five_nm_index)


def validitycheck2(min_val, sd_flat, params=None):
    return (min_val <= -1*sd_flat*params['relative_deviation'])


def binding_selection(y_rt, min_val, min_idx, sd_flat, nonspecific_boundary, speed, params=None):
    ### 1. Check the values that are small enough
    condition1 = (params['r_min_val']*min_val) > y_rt  # compared to the global min value
    if speed==0.1 or speed==0.2: 
        condition2 = params['binding_threshold_lowspeed'] > y_rt
    elif speed==1 or speed==5: 
        condition2 = params['binding_threshold_midspeed'] > y_rt
    elif speed==10 or speed==20: 
        condition2 = params['binding_threshold_highspeed'] > y_rt
    condition3 = params['r_sd_flat']*sd_flat > y_rt  # compared to the basic noise
    local_minima_idx = list(np.where(condition1 * condition2 * condition3)[0])    # many bindings

    # 1.1. Remove minima before the nonspecific boundary
    local_minima_idx = [lll for lll in local_minima_idx if lll >= nonspecific_boundary]

    ### 2. Remove the non-minima
    lmi_tobe_removed = list()
    for lmi in local_minima_idx:
        if is_minima(y_rt, lmi):
            pass
        else:
            lmi_tobe_removed.append(lmi)
    # Remove
    for ltr in lmi_tobe_removed:
        local_minima_idx.remove(ltr)

    ### 3. Check if those are the smallest values in a small interval
    local_minima_idx2 = list()
    for lmi in local_minima_idx:
        if y_rt[lmi] == np.min(y_rt[lmi+params['neighbor_boundary'][0]:lmi+params['neighbor_boundary'][1]]):
            local_minima_idx2.append(lmi)
    local_minima_idx = np.array(local_minima_idx2)

    ### 4. Check the distance from the global minima to the local minima
    # Those which are farther than a threshold are considered as different peaks.
    if np.sum(np.abs(local_minima_idx-min_idx) > params['peak_interval']):  # at least one of the lmi
        min_idx = np.max(local_minima_idx)
        min_val = y_rt[min_idx]
    else:  # input min_val, min_idx (=global minima) are returned
        pass
    return min_val, min_idx


def ROI_selection(x_rt, y_rt, min_idx, params=None):
    # get zero-crossing index
    y_tmp = y_rt[0:np.minimum(min_idx+1,len(y_rt))]
    zero_crossing_idx = min_idx
    try:
        while y_tmp[zero_crossing_idx] <= 0:
            zero_crossing_idx -= 1
    except:
        zero_crossing_idx += 1

    total_dist = min_idx-zero_crossing_idx

    # ROI
    left_boundary = zero_crossing_idx + int(total_dist*(params['n_divide']-1)/params['n_divide'])
    right_boundary = min_idx
    x_ROI = x_rt[left_boundary:right_boundary+1]
    y_ROI = y_rt[left_boundary:right_boundary+1]
    return left_boundary, right_boundary, x_ROI, y_ROI


def trendline_calculation(x_ROI, y_ROI):
    ##### Get the trendline & slope of it
    p = np.polyfit(x_ROI,y_ROI,1)
    trendline = p[1]+p[0]*x_ROI

    slope = p[0]
    slope = round(1000*slope,4)
    return trendline, slope


def plot_invalid(x_rt, y_rt_original, y_rt_ma, y_rt, flat_idx, filename, params=None):
    first_crossing_idx = 0
    for i in range(len(y_rt)):
        if i==0:
            continue
        if y_rt[i] > 0:
            pass
        elif y_rt[i] <= 0:
            first_crossing_idx = i
            break
    five_nm_index = int(len(y_rt)/100)  # 500nm/100=5nm, 500nm/150=3.33nm

    target_folder = os.path.join(params['data_folder'],params['invalid_png_folder'])
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    plt.figure(figsize=(8,6)) # Figure size
    plt.plot(x_rt, np.zeros((len(x_rt))), 'lightskyblue')
    plt.plot(x_rt, y_rt_original, 'lightgray')
    plt.plot(x_rt, y_rt, 'black')
    plt.plot(x_rt, y_rt_ma, 'gray')
    plt.plot(x_rt[flat_idx], y_rt_ma[flat_idx], 'lime')
    plt.plot([x_rt[first_crossing_idx+five_nm_index], x_rt[first_crossing_idx+five_nm_index]], [np.min(y_rt), np.max(y_rt)], 'chocolate')
    plt.xlabel("time (ms)")
    plt.ylabel("Force (pN)")
    plt.legend(['x axis', 'Before baseline correction', 'After baseline correction', 'Moving average', 'Flat region', '5 nm'])
    plt.title(os.path.split(filename)[-1][:30]+'\n'+os.path.split(filename)[-1][30:])
    plt.savefig(os.path.join(target_folder, os.path.split(filename)[-1][:-4]+'.png'))
    plt.close()
    return 


def plot_valid(x_rt, y_rt_original, y_rt_ma, y_rt, flat_idx, 
    left_boundary, right_boundary, trendline, x_ROI, filename, slope, min_val, params=None):
    target_folder = os.path.join(params['data_folder'], params['valid_png_folder'])
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)
    plt.figure(figsize=(8,6)) # Figure size
    plt.plot(x_rt, np.zeros((len(x_rt))), 'lightskyblue')
    plt.plot(x_rt, y_rt_original, 'lightgray', linewidth=0.5)
    plt.plot(x_rt, y_rt_ma, 'gray')
    plt.plot(x_rt, y_rt, 'black', linewidth=0.5)
    plt.plot(x_rt[flat_idx], y_rt[flat_idx], 'lime')
    plt.plot([x_rt[left_boundary], x_rt[right_boundary]], [trendline[0], trendline[-1]], 'bo')
    plt.plot(x_ROI, trendline, 'red', linewidth=0.2)
    plt.xlabel("time (ms)")
    plt.ylabel("Force (pN)")
    plt.legend(['x axis', 'Before baseline correction', 'Moving average', 'After baseline correction', 
                'Flat region', 'Trend line boundary', 'Trend line'])
    plt.title(os.path.split(filename)[-1][:30]+'\n'+os.path.split(filename)[-1][30:])
    plt.text(np.min(x_rt), (np.max(y_rt)+np.min(y_rt))/2, 'slope : '+str(slope)+', '+'min value : '+str(min_val) )
    plt.savefig( os.path.join(target_folder, os.path.split(filename)[-1][:-4]+'.png') )
    plt.close()
    return 


