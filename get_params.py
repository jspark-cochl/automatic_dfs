#############################################
# get_params.py                             #
# - This file contains parameters that are  #
#   used in auto_dfs.py.                    #
#              Jeongsoo Park, Cochl         #
#############################################
def get_params():
    # # # # # PARAMETERS # # # # #
    params = {}
    # Whether to extract hash?
    params['wheter2plot'] = True
    params['speeds'] = ['100nm', '200nm', '1um', '5um', '10um', '20um']

    params['baseline_order'] = 1  # 1: linear, 2: quadratic, ..
    params['baseline_deviation'] = 5  # 5 times of sd_flat

    params['data_folder'] = './txt' # path of the spm data exported to txt
    params['data_type'] = 'txt' # either txt or npy

    # Variables related to meaningful data filtering
    params['relative_deviation'] = 6

    # Variables related to multiple binding selection
    # -> "or" condition: don't need to satify all the conditions below.
    params['r_min_val'] = 0.75 # if r_min_val = -90, peaks under -45 are selected
    params['binding_threshold_lowspeed'] = -20   # peaks under this value are selected for 0.1, 0.2ums
    params['binding_threshold_midspeed'] = -40 # peaks under this value are selected for 1, 5ums
    params['binding_threshold_highspeed'] = -50 # peaks under this value are selected for 10, 20ums
    params['r_sd_flat'] = -3    # if r_sd_flat=8, peaks under -24 are selected

    # Multiple binding validity check
    params['neighbor_boundary'] = [-8,4]

    # Peak inverval (from the global min val)
    params['peak_interval'] = 20

    # 1/3 or 1/4 or 1/6 ...
    params['n_divide'] = 6.0

    # Folders to save figures
    params['valid_png_folder'] = 'png_valid'
    params['invalid_png_folder'] = 'png_invalid'

    params['n_tabs'] = 5

    return params



