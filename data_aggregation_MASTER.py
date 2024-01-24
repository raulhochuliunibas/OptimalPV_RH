# -----------------------------------------------------------------------------
# data_aggreation_MASTER.py 
# -----------------------------------------------------------------------------
# Preamble: 
# > 


# SETTIGNS --------------------------------------------------------------------
agg_settings = {
        'name_dir_export': 'all_CH',        # name of the directory for the aggregated data, and maybe file extension
        'script_run_on_server': False,      # F: run on private computer, T: run on server
        'recreate_parquet_files': True,     # F: use existing parquet files, T: recreate parquet files in data prep
        'show_debug_prints': True,          # F: certain print statements are omitted, T: includes print statements that help with debugging
        'smaller_import': False,             # F: import all data, T: import only a small subset of data for debugging

        'bfs_numbers_OR_shape': [3851,],       # 3851: Davos (GR)
        'gwr_house_type_class': [0,], 
        'solkat_house_type_class': [0,], 
        }


# PACKAGES --------------------------------------------------------------------
import sys
if agg_settings['script_run_on_server']:
    sys.path.append('C:/Models/OptimalPV_RH') 
elif agg_settings['script_run_on_server']:
    sys.path.append('D:/RaulHochuli_inuse/OptimalPV_RH')

# external packages
import os as os
import pandas as pd
import geopandas as gpd
import glob
import shutil
import winsound
import subprocess

# own packages and functions
import functions
from functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile

from data_aggregation.api_electricity_prices import api_electricity_prices
from data_aggregation.preprepare_data import solkat_spatial_toparquet, gwr_spatial_toparquet, heat_spatial_toparquet, pv_spatial_toparquet, create_spatial_mappings


# import OptimalPV_RH.data_aggregation.preprepare_data as spd_to_pq

# from data_aggregation.local_data_import_aggregation import import_aggregate_data
# from data_aggregation.Lupien_aggregation_roofkat_pv_munic_V2 import Lupien_aggregation
# from data_aggregation.spatial_data_toparquet_by_gm import spatal_topiarquet


# SETUP -----------------------------------------------------------------------
# set working directory
if not agg_settings['script_run_on_server']:
    wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
    data_path = f'{wd_path}_data'

    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
elif agg_settings['script_run_on_server']:
    wd_path = "D:/RaulHochuli_inuse/OptimalPV_RH"
    data_path = f'{wd_path}_data'

    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    
# create directory + log file
if not os.path.exists(f'{data_path}/output/{agg_settings["name_dir_export"]}'):
    os.makedirs(f'{data_path}/output/{agg_settings["name_dir_export"]}')

log_name = f'{data_path}/output/{agg_settings["name_dir_export"]}_log.txt'
chapter_to_logfile(f'start data_aggregation_MASTER for: {agg_settings["name_dir_export"]}', log_name, overwrite_file=True)
print_to_logfile(f'> agg_settings: \n\t name_dir_export: {agg_settings["name_dir_export"]} \n\t script_run_on_server: {agg_settings["script_run_on_server"]} \n\t recreate_parquet_files: {agg_settings["recreate_parquet_files"]} \n\t debug_prints: {agg_settings["show_debug_prints"]} \n\t smaller_import: {agg_settings["smaller_import"]}', log_name)


# PRE PREP DATA ---------------------------------------------------------------

# download possible API data to local directory
subchapter_to_logfile('pre-prep data: API ELECTRICITY PRICES', log_name)
if agg_settings['recreate_parquet_files']:
    api_electricity_prices(agg_settings['script_run_on_server'], agg_settings['recreate_parquet_files'], agg_settings['smaller_import'], 
                           log_name, wd_path, data_path, agg_settings['show_debug_prints'],
                           year_range = [2009, 2023])
else:
    checkpoint_to_logfile('use electricity prices that are downloaded already', log_name)


# transform spatial data to parquet files for faster import and transformation
pq_dir_exists_TF = os.path.exists(f'{data_path}/output/{agg_settings["name_dir_export"]}_preprep_sdtopq')
pq_files_rerun = agg_settings['recreate_parquet_files']

if not pq_dir_exists_TF or pq_files_rerun:
    subchapter_to_logfile('pre-prep data: SPATIAL DATA to PARQUET', log_name)
    solkat_spatial_toparquet(agg_settings['script_run_on_server'], agg_settings['smaller_import'], 
                             log_name, wd_path, data_path, agg_settings['show_debug_prints'])
    gwr_spatial_toparquet(agg_settings['script_run_on_server'], agg_settings['smaller_import'],
                            log_name, wd_path, data_path, agg_settings['show_debug_prints'])
    heat_spatial_toparquet(agg_settings['script_run_on_server'], agg_settings['smaller_import'],
                            log_name, wd_path, data_path, agg_settings['show_debug_prints'])
    pv_spatial_toparquet(agg_settings['script_run_on_server'], agg_settings['smaller_import'],
                            log_name, wd_path, data_path, agg_settings['show_debug_prints'])
    
    subchapter_to_logfile('pre-prep data: SPATIAL MAPPINGS', log_name)
    create_spatial_mappings(script_run_on_server_def= agg_settings['script_run_on_server'], 
                            smaller_import_def=agg_settings['smaller_import'], 
                            log_file_name_def=log_name,
                            wd_path_def=wd_path, 
                            data_path_def=data_path, 
                            show_debug_prints_def=agg_settings['show_debug_prints'])
    
else: 
    checkpoint_to_logfile('use parquet files and mappings that exist already', log_name)


# DATA AGGREGATION FOR MA student Lupien --------------------------------------

def move_Lupien_agg_to_dict(dict_name):
    if not os.path.exists(f'{data_path}/{dict_name}'):
        os.makedirs(f'{data_path}/{dict_name}')
    f_to_move = glob.glob(f'{data_path}/Lupien_aggregation/*')
    for f in f_to_move: 
        shutil.copy(f, f'{data_path}/{dict_name}/')
# Lupien_aggregation(script_run_on_server_def = script_run_on_server, check_vs_raw_input=True, union_vs_hull_shape = 'union')


###############################
# BOOKMARK
###############################
# INITIALIZE PV TOPOLOGY --------------------------------------



# AGGREGATIONS -----------------------------------------------------------------
# aggregation solkat, pv, munic, gwr, heatcool by cantons 
kt_list = list(gm_shp['KANTONSNUM'].dropna().unique())  


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



        
    

    
