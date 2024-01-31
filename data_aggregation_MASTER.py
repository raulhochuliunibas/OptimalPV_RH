# -----------------------------------------------------------------------------
# data_aggreation_MASTER.py 
# -----------------------------------------------------------------------------
# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024
# > description: This file is the master file for data aggregation. It calls all
#   necessary functions to get data from API sources and aggregate it together with
#   other locally stored spatial data sources. It does so by converting all data to 
#   parquet files (faster imports) and creating mappings for fast lookups. 

# TO-DOs:
# TODO: change code such that prepred_data is on the same directory level than output



# SETTIGNS --------------------------------------------------------------------
agg_settings = {
        'script_run_on_server': False,      # F: run on private computer, T: run on server
        'recreate_preprep_data': True,     # F: use existing parquet files, T: recreate parquet files in data prep
        'show_debug_prints': True,          # F: certain print statements are omitted, T: includes print statements that help with debugging
        'smaller_import': False,             # F: import all data, T: import only a small subset of data for debugging

        'bfs_numbers_OR_shape': [3851,],       # 3851: Davos (GR)
        'gwr_house_type_class': [0,], 
        'solkat_house_type_class': [0,], 
        }


# PACKAGES --------------------------------------------------------------------
import sys
sys.path.append('D:/RaulHochuli_inuse/OptimalPV_RH') if agg_settings['script_run_on_server'] else sys.path.append('C:/Models/OptimalPV_RH')

# external packages
import os as os
import pandas as pd
import geopandas as gpd
import glob
import shutil
import winsound
import subprocess
from datetime import datetime
from pprint import pprint, pformat

# own packages and functions
from functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile
from data_aggregation.api_electricity_prices import api_electricity_prices
from data_aggregation.sql_gwr import sql_gwr_data
from data_aggregation.preprepare_data import solkat_spatial_toparquet, gwr_spatial_toparquet, heat_spatial_toparquet, pv_spatial_toparquet, create_spatial_mappings


# SETUP -----------------------------------------------------------------------
# set working directory
wd_path = "D:\\RaulHochuli_inuse\\OptimalPV_RH"  if agg_settings['script_run_on_server'] else "C:\Models\OptimalPV_RH"
data_path = f'{wd_path}_data'
    
# create directory + log file
# if not os.path.exists(f'{data_path}/output/{agg_settings["name_dir_export"]}'):
#     os.makedirs(f'{data_path}/output/{agg_settings["name_dir_export"]}')

log_name = f'{data_path}/output/prepre_data_log.txt'
chapter_to_logfile(f'start data_aggregation_MASTER', log_name, overwrite_file=True)
print_to_logfile(f' > settings: \n{pformat(agg_settings)}', log_name)


# PRE PREP DATA ---------------------------------------------------------------

# download possible API data to local directory
subchapter_to_logfile('pre-prep data: API ELECTRICITY PRICES', log_name)
file_exists_TF = os.path.exists(f'{data_path}/elecpri.parquet')
preprep_data_rerun = agg_settings['recreate_preprep_data']
year_range = [2021, 2021] if agg_settings['smaller_import'] else [2009, 2023]

if not file_exists_TF or preprep_data_rerun:
    api_electricity_prices(script_run_on_server_def = agg_settings['script_run_on_server'],
                           recreate_parquet_files_def=agg_settings['recreate_preprep_data'],
                           smaller_import_def=agg_settings['smaller_import'],
                           log_file_name_def=log_name,
                           wd_path_def=wd_path,
                           data_path_def=data_path, 
                           show_debug_prints_def=agg_settings['show_debug_prints'], 
                            year_range_def = year_range)
    sql_gwr_data(script_run_on_server_def = agg_settings['script_run_on_server'],
                 recreate_parquet_files_def=agg_settings['recreate_preprep_data'],
                 smaller_import_def=agg_settings['smaller_import'],
                 log_file_name_def=log_name,
                 wd_path_def=wd_path,
                 data_path_def=data_path, 
                 show_debug_prints_def=agg_settings['show_debug_prints'])

else:
    checkpoint_to_logfile('use electricity prices that are downloaded already', log_name)


# transform spatial data to parquet files for faster import and transformation
pq_dir_exists_TF = os.path.exists(f'{data_path}/output/prepred_data')
pq_files_rerun = agg_settings['recreate_preprep_data']

if not pq_dir_exists_TF or pq_files_rerun:
    subchapter_to_logfile('pre-prep data: SPATIAL MAPPINGS', log_name)
    create_spatial_mappings(script_run_on_server_def= agg_settings['script_run_on_server'], 
                            smaller_import_def=agg_settings['smaller_import'], 
                            log_file_name_def=log_name,
                            wd_path_def=wd_path, 
                            data_path_def=data_path, 
                            show_debug_prints_def=agg_settings['show_debug_prints'])
    
    subchapter_to_logfile('pre-prep data: SPATIAL DATA to PARQUET', log_name)
    solkat_spatial_toparquet(agg_settings['script_run_on_server'], agg_settings['smaller_import'], 
                             log_name, wd_path, data_path, agg_settings['show_debug_prints'])
    # gwr_spatial_toparquet(agg_settings['script_run_on_server'], agg_settings['smaller_import'],
    #                         log_name, wd_path, data_path, agg_settings['show_debug_prints'])
    heat_spatial_toparquet(agg_settings['script_run_on_server'], agg_settings['smaller_import'],
                            log_name, wd_path, data_path, agg_settings['show_debug_prints'])
    pv_spatial_toparquet(agg_settings['script_run_on_server'], agg_settings['smaller_import'],
                            log_name, wd_path, data_path, agg_settings['show_debug_prints'])
    

    
else: 
    checkpoint_to_logfile('use parquet files and mappings that exist already', log_name)

chapter_to_logfile(f'END data_aggregation_MASTER', log_name, overwrite_file=False)
# MOVE AGGREGATED DATA TO DICT not to overwrite it while debugging
today = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
dirs_preprep_data_DATE = f'{data_path}/output/preprep_data_{today.split("-")[0]}{today.split("-")[1]}{today.split("-")[2]}_{today.split("-")[3]}h'
if not os.path.exists(dirs_preprep_data_DATE):
    os.makedirs(dirs_preprep_data_DATE)
file_to_move = glob.glob(f'{data_path}/output/preprep_data/*')
for f in file_to_move:
    shutil.copy(f, dirs_preprep_data_DATE)
shutil.copy(glob.glob(f'{data_path}/output/prepre*_log.txt')[0], dirs_preprep_data_DATE)


###############################
# BOOKMARK
###############################


# DATA AGGREGATION FOR MA student Lupien --------------------------------------
def move_Lupien_agg_to_dict(dict_name):
    if not os.path.exists(f'{data_path}/{dict_name}'):
        os.makedirs(f'{data_path}/{dict_name}')
    f_to_move = glob.glob(f'{data_path}/Lupien_aggregation/*')
    for f in f_to_move: 
        shutil.copy(f, f'{data_path}/{dict_name}/')
# Lupien_aggregation(script_run_on_server_def = script_run_on_server, check_vs_raw_input=True, union_vs_hull_shape = 'union')


# buffer False -----------------------------------------------------------------
if False:

    for n, kt_i in enumerate(kt_list):
        name_run = 'agg_solkat_pv_gm_gwr_heat_buffNO_KT'
        gm_number_aggdef = list(gm_shp.loc[gm_shp['KANTONSNUM'] == kt_i, 'BFS_NUMMER'].unique())
        import_aggregate_data(
            name_aggdef = f'{name_run}{str(int(kt_i))}', 
            script_run_on_server = script_run_on_server , 
            gm_number_aggdef = gm_number_aggdef, 
            data_source= 'parquet', 
            set_buffer = False)
        
        print(f'canton {kt_i} aggregated, {n+1} of {len(kt_list)} completed')
        
    # copy all subfolders to one folder
    name_dir_export ='agg_solkat_pv_gm_gwr_heat_buffNO_2_BY_KT'
    if not os.path.exists(f'{data_path}/{name_dir_export}'):
        os.makedirs(f'{data_path}/{name_dir_export}')
        os.makedirs(f'{data_path}/{name_dir_export}_to_delete')
    
    # add parquet and log files + move unncecessary folders
    files_copy = glob.glob(f'{data_path}/{name_run}*/*{name_run}*')
    for f in files_copy:
        shutil.move(f, f'{data_path}/{name_dir_export}')    

    files_del = glob.glob(f'{data_path}/{name_run}*')
    for f in files_del:
        shutil.move(f, f'{data_path}/{name_dir_export}_to_delete')


# buffer 10 ----------------------------------------------------------------
if False:
 
    for n, kt_i in enumerate(kt_list):
        name_run = 'agg_solkat_pv_gm_gwr_heat_buff10KT'
        gm_number_aggdef = list(gm_shp.loc[gm_shp['KANTONSNUM'] == kt_i, 'BFS_NUMMER'].unique())
        import_aggregate_data(
            name_aggdef = f'{name_run}{str(int(kt_i))}', 
            script_run_on_server = script_run_on_server , 
            gm_number_aggdef = gm_number_aggdef, 
            data_source= 'parquet', 
            set_buffer = 10)
        
        print(f'canton {kt_i} aggregated, {n+1} of {len(kt_list)} completed')

    # copy all subfolders to one folder
    name_dir_export ='agg_solkat_pv_gm_gwr_heat_buff10_BY_KT'
    if not os.path.exists(f'{data_path}/{name_dir_export}'):
        os.makedirs(f'{data_path}/{name_dir_export}')
        os.makedirs(f'{data_path}/{name_dir_export}_to_delete')

    # add parquet and log files + move unncecessary folders
    files_copy = glob.glob(f'{data_path}/{name_run}*/*{name_run}*')
    for f in files_copy:
        shutil.move(f, f'{data_path}/{name_dir_export}')

    files_del = glob.glob(f'{data_path}/{name_run}*')
    for f in files_del:
        shutil.move(f, f'{data_path}/{name_dir_export}_to_delete')



        
    

    
