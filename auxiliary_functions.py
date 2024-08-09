import os as os
import geopandas as gpd

from datetime import datetime

# this is required to set the timer properly
time_last_call = None


def chapter_to_logfile(str_def, log_file_name_def, overwrite_file=False):
    """
    Function to write a chapter to the logfile
    """
    check = f'\n\n****************************************\n {str_def} \n start at: {datetime.now()} \n****************************************\n'
    print(check)
    
    log_file_exists_TF = os.path.exists(f'{log_file_name_def}')
    if not log_file_exists_TF or overwrite_file:
        with open(f'{log_file_name_def}', 'w') as log_file:
            log_file.write(f'{check}\n')
    else: 
        with open(f'{log_file_name_def}', 'a') as log_file:
            log_file.write(f'{check}\n')


def subchapter_to_logfile(str_def, log_file_name_def):
    """
    Function to write a subchapter to the logfile
    """
    check = f'\n----------------------------------------\n {str_def} > start at: {datetime.now()} \n----------------------------------------\n'
    print(check)
    
    if not os.path.exists(f'{log_file_name_def}'):
        with open(f'{log_file_name_def}', 'w') as log_file:
            log_file.write(f'{check}\n')
    elif os.path.exists(f'{log_file_name_def}'): 
        with open(f'{log_file_name_def}', 'a') as log_file:
            log_file.write(f'{check}\n')


def print_to_logfile(str_def, log_file_name_def):
    """
    Function to write just plain text to logfile
    """
    check = f'{str_def}'
    print(check)

    with open(f'{log_file_name_def}', 'a') as log_file:
        log_file.write(f"{check}\n")


def checkpoint_to_logfile(str_def, log_file_name_def, n_tabs_def = 0, show_debug_prints_def = True):
    """
    Function to write a checkpoint to the logfile, Mainly used for debugging because prints can be switched off
    """ 
    global time_last_call
    
    if show_debug_prints_def:
        time_now = datetime.now()
        if time_last_call:
            total_seconds = int((time_now - time_last_call).total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            runtime_str = f"{hours} hr {minutes} min {seconds} sec"
        else:
            runtime_str = 'N/A'
        
        n_tabs_str = '\t' * n_tabs_def
        check = f'* {str_def}{n_tabs_str} > runtime: {runtime_str};   (stamp: {datetime.now()})'
        print(check)

        with open(f'{log_file_name_def}', 'a') as log_file:
            log_file.write(f"{check}\n")
        
        time_last_call = time_now


def get_bfs_from_ktnr(ktnr_list, data_path, log_name):
    """
    Function to get a list of the BFS numbers for all municipalites within a list of canton numbers
    """
    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')

    if (isinstance(ktnr_list, list)) and (not not ktnr_list):
        gm_shp_sub = gm_shp[gm_shp['KANTONSNUM'].isin(ktnr_list)]
        bfs_list = gm_shp_sub['BFS_NUMMER'].unique().tolist()
    else: 
        print_to_logfile(f' > ERROR: no canton or bfs selection applicables; NOT used any municipality selection', log_name)

    return bfs_list


def format_MASTER_settings(settings_dict):
    formatted_settings = {}
    for key, value in settings_dict.items():
        if isinstance(value, list):
            formatted_settings[key] = ', '.join(map(str, value))
        elif isinstance(value, dict):
            formatted_settings[key] = format_MASTER_settings(value)
        else:
            formatted_settings[key] = value
    return formatted_settings





# ----------------------------------------------------------------------
# book mark ------------------------------------------------------------
# 
# ----------------------------------------------------------------------
    
def test_functions(text):
    print(f'return: {text}')