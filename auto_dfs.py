#############################################
# auto_dfs.py                               #
# - The code below can be used to estimate  #
#     the binding slopes.                   #
#             Jeongsoo Park, Cochl          #
#############################################
import csv
import numpy as np
import scipy
import os
import time
import copy
import matplotlib.pyplot as plt
from core import *
from get_params import *
plt.ion()

experiment_name = os.path.split(os.path.split(os.getcwd())[0])[-1]+'_'+os.path.split(os.getcwd())[-1]
params = get_params()
my_foldernames = list()
my_filenames = list()
my_slopes = list()
my_min_vals = list()


# # # Load list of files
file_list = read_list(params)
for n_file, filename in enumerate(file_list):
    print('({}/{}) {}'.format(n_file+1, len(file_list), filename))
    # # # Load data
    x_rt, y_rt, t_rt, speed = read_data(filename, params)

    # # # Baseline correction 1 
    # Flat region -> normalize to have 0-mean
    y_rt, x_flat, sd_flat, min_val, min_idx, flat_idx = baseline_corr1(x_rt,y_rt)
    y_rt_original = copy.deepcopy(y_rt)

    # # # Baseline correction 2
    # Robust nonlinear regression
    y_rt, y_robust = baseline_corr2(y_rt, baseline_order=params['baseline_order'], allowed_deviation=params['baseline_deviation']*sd_flat)

    # # # Baseline correction 3
    y_rt, x_flat, sd_flat, min_val, min_idx, flat_idx = baseline_corr1(x_rt,y_rt)

    # # # Data validity check
    y_rt_ma = moving_average(y_rt, tabs=params['n_tabs'])  # Moving average
    y_rt_ma, _, sd_flat_ma, min_val_ma, min_idx_ma, flat_idx_ma = baseline_corr1(x_rt, y_rt_ma)
    v1, nonspecific_boundary = validitycheck1(y_rt)
    v2 = validitycheck2(min_val_ma, sd_flat_ma, params)
    

    # # # Multiple binding handling
    # Check if there are other bindings or not
    min_val, min_idx = binding_selection(y_rt, min_val, min_idx, sd_flat, nonspecific_boundary, speed, params)

    # # # Data validity check2
    if v2 == False:
        validity = False
    else:
        if v1 == True:
            validity = True
        elif min_idx >= nonspecific_boundary:
            validity = True
        else:
            validity = False

    if not validity:
        # Save info
        my_foldernames.append(os.path.split(filename)[0])
        my_filenames.append(filename)
        my_slopes.append(1)
        my_min_vals.append(1)
        # Plot
        if params['whether2plot']:
            plot_invalid(x_rt, y_rt_original, y_rt_ma, y_rt, flat_idx, filename, params)
        continue

    # # # ROI (region of interest) selection
    left_boundary, right_boundary, x_ROI, y_ROI = ROI_selection(x_rt, y_rt, min_idx, params)

    # # # Trendline
    trendline, slope = trendline_calculation(x_ROI, y_ROI)

    # # # Finish
    # Save info
    my_foldernames.append(os.path.split(filename)[0])
    my_filenames.append(filename)
    my_slopes.append(slope)
    my_min_vals.append(np.abs(round(min_val,4)))
    # Plot
    if params['whether2plot']:
        plot_valid(x_rt, y_rt_original, y_rt_ma, y_rt, flat_idx, 
            left_boundary, right_boundary, trendline, x_ROI, filename, slope, min_val, params)


# # # Save to csv file
with open('result_'+experiment_name+'.csv','w', newline='') as csvfile:
    mywriter = csv.writer(csvfile, delimiter=',')
    for ii in range(len(my_filenames)):
        file_id = my_filenames[ii][my_filenames[ii].index('ForceCurveIndex_')+16:my_filenames[ii].index('.spm.txt')]
        mywriter.writerow([file_id,str(-my_slopes[ii]),str(my_min_vals[ii])])
