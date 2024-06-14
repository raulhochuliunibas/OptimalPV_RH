# -----------------------------------------------------------------------------
# data_aggreation_MASTER.py 
# -----------------------------------------------------------------------------
# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024
# > description: This file is the master file for data aggregation. It calls all
#   necessary functions to get data from API sources and aggregate it together with
#   other locally stored spatial data sources. It does so by converting all data to 
#   parquet files (faster imports) and creating mappings for fast lookups (no single data base). 

# TO-DOs:

# TODO: create a GWR>GM and SOLKAT>GM mapping for the spatial data
# TODO: Add many more variables for GWR extraction (heating, living area etc.), WAREA not found in SQL data base

# TODO: facade data inculde 
# TODO: change code such that prepred_data is on the same directory level than output

# NOTE: ADJUST ALL MAPPINGS so that the data type is a string, not an int
# NOTE: Map_egroof_sbroof carries an unnecessary index in the export file. remove that in the preppred_data function
# NOTE: Change the GWR aggregation to take GBAUP not GBAUJ -> see email MADD, Mauro Nanini
# NOTE: Remove MSTRAHLUNG from Cummulative summation => unnecessary and not true anyway (summed up an average)



# SETTIGNS --------------------------------------------------------------------
agg_settings = {
        'script_run_on_server': False,      # F: run on private computer, T: run on server
        'smaller_import': True,             # F: import all data, T: import only a small subset of data (smaller range of years) for debugging
        'reimport_api_data': True,          # F: use existing parquet files, T: recreate parquet files in data prep
        'rerun_spatial_mappings': True,     # F: use existing parquet files, T: recreate parquet files in data prep
        'reextend_fixed_data': True,          # F: use existing exentions, T: recalculate extensions (e.g. pv installation costs per partition)
        'show_debug_prints': True,          # F: certain print statements are omitted, T: includes print statements that help with debugging

        'bfs_numbers_OR_shape': 
        [2761, 2763, 2842, 2787],           # small list for debugging
                                            # BL: als BFS lists from            
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
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile
from data_aggregation.api_electricity_prices import api_electricity_prices
from data_aggregation.sql_gwr import sql_gwr_data
from data_aggregation.api_pvtarif import api_pvtarif, api_pvtarif_gm_Mapping
from data_aggregation.preprepare_data import solkat_spatial_toparquet, gwr_spatial_toparquet, heat_spatial_toparquet, pv_spatial_toparquet, create_spatial_mappings
from data_aggregation.installation_cost import attach_pv_cost



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



# IMPORT API DATA ---------------------------------------------------------------

# download possible API data to local directory
file_exists_TF = os.path.exists(f'{data_path}/output/preprep_data/elecpri.parquet')
reimport_api_data = agg_settings['reimport_api_data']

year_range_gwr = [2020, 2021] if agg_settings['smaller_import'] else [2009, 2023]
year_range_pvtarif = [2020, 2021] if agg_settings['smaller_import'] else [2015, 2023]

if not file_exists_TF or reimport_api_data:
    # subchapter_to_logfile('pre-prep data: API ELECTRICITY PRICES', log_name)
    # api_electricity_prices(script_run_on_server_def = agg_settings['script_run_on_server'], smaller_import_def=agg_settings['smaller_import'], log_file_name_def=log_name, wd_path_def=wd_path, data_path_def=data_path, show_debug_prints_def=agg_settings['show_debug_prints'], year_range_def = year_range_gwr)
    
    # subchapter_to_logfile('pre-prep data: SQL GWR DATA', log_name)
    # sql_gwr_data(script_run_on_server_def = agg_settings['script_run_on_server'], smaller_import_def=agg_settings['smaller_import'], log_file_name_def=log_name, wd_path_def=wd_path, data_path_def=data_path, show_debug_prints_def=agg_settings['show_debug_prints'])

    subchapter_to_logfile('pre-prep data: API PVTARIF', log_name)
    api_pvtarif(script_run_on_server_def = agg_settings['script_run_on_server'], smaller_import_def=agg_settings['smaller_import'], log_file_name_def=log_name, wd_path_def=wd_path, data_path_def=data_path, show_debug_prints_def=agg_settings['show_debug_prints'], year_range_def=year_range_pvtarif )

    api_pvtarif_gm_Mapping(script_run_on_server_def = agg_settings['script_run_on_server'], smaller_import_def=agg_settings['smaller_import'], log_file_name_def=log_name, wd_path_def=wd_path, data_path_def=data_path, show_debug_prints_def=agg_settings['show_debug_prints'], year_range_def=year_range_pvtarif )

else:
    print_to_logfile('\n\n', log_name)
    checkpoint_to_logfile('use electricity prices that are downloaded already', log_name)



# SPATIAL MAPPINGS ---------------------------------------------------------------

# transform spatial data to parquet files for faster import and transformation
pq_dir_exists_TF = os.path.exists(f'{data_path}/output/preprep_data')
pq_files_rerun = agg_settings['rerun_spatial_mappings']

if not pq_dir_exists_TF or pq_files_rerun:
    subchapter_to_logfile('pre-prep data: SPATIAL MAPPINGS', log_name)
    create_spatial_mappings(script_run_on_server_def= agg_settings['script_run_on_server'], smaller_import_def=agg_settings['smaller_import'], log_file_name_def=log_name, wd_path_def=wd_path, data_path_def=data_path, show_debug_prints_def=agg_settings['show_debug_prints'])    

    subchapter_to_logfile('pre-prep data: SPATIAL DATA to PARQUET', log_name)
    solkat_spatial_toparquet(agg_settings['script_run_on_server'], agg_settings['smaller_import'], log_name, wd_path, data_path, agg_settings['show_debug_prints'])
    heat_spatial_toparquet(agg_settings['script_run_on_server'], agg_settings['smaller_import'], log_name, wd_path, data_path, agg_settings['show_debug_prints'])
    pv_spatial_toparquet(agg_settings['script_run_on_server'], agg_settings['smaller_import'], log_name, wd_path, data_path, agg_settings['show_debug_prints'])

else: 
    print_to_logfile('\n', log_name)
    checkpoint_to_logfile('use parquet files and mappings that exist already', log_name)



# EXTEND WITH TIME FIXED DATA ---------------------------------------------------------------
cost_df_exists_TF = os.path.exists(f'{data_path}/output/preprep_data/pvinstcost.parquet')
extend_data_rerun = agg_settings['reextend_fixed_data']

if not cost_df_exists_TF or extend_data_rerun:
    subchapter_to_logfile('extend data: PV INSTALLTION COST', log_name)
    attach_pv_cost(script_run_on_server_def= agg_settings['script_run_on_server'],  
                     log_file_name_def=log_name,
                     wd_path_def=wd_path, 
                     smaller_import_def=agg_settings['smaller_import'],
                     data_path_def=data_path, 
                     show_debug_prints_def=agg_settings['show_debug_prints'])
    
    subchapter_to_logfile('extend data: WEIGHTS FOR ELECTRICITY DEMAND', log_name)


   
# COPY AGGREGATED DATA ---------------------------------------------------------------
# > not to overwrite it while debugging 
chapter_to_logfile(f'END data_aggregation_MASTER', log_name, overwrite_file=False)
today = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
dirs_preprep_data_DATE = f'{data_path}/output/preprep_data_{today.split("-")[0]}{today.split("-")[1]}{today.split("-")[2]}_{today.split("-")[3]}h'
if not os.path.exists(dirs_preprep_data_DATE):
    os.makedirs(dirs_preprep_data_DATE)
file_to_move = glob.glob(f'{data_path}/output/preprep_data/*')
for f in file_to_move:
    shutil.copy(f, dirs_preprep_data_DATE)
shutil.copy(glob.glob(f'{data_path}/output/prepre*_log.txt')[0], dirs_preprep_data_DATE)


# -----------------------------------------------------------------------------
# END 
# -----------------------------------------------------------------------------






###############################
###############################
# BOOKMARK > delete in March 2024


# DATA AGGREGATION FOR MA student Lupien --------------------------------------
def move_Lupien_agg_to_dict(dict_name):
    if not os.path.exists(f'{data_path}/{dict_name}'):
        os.makedirs(f'{data_path}/{dict_name}')
    f_to_move = glob.glob(f'{data_path}/Lupien_aggregation/*')
    for f in f_to_move: 
        shutil.copy(f, f'{data_path}/{dict_name}/')
# Lupien_aggregation(script_run_on_server_def = script_run_on_server, check_vs_raw_input=True, union_vs_hull_shape = 'union')

###############################
###############################


###############################
###############################
# BOOKMARK > delete in February 2024

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

###############################
###############################
