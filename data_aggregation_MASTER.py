# -----------------------------------------------------------------------------
# data_aggreation_MASTER.py 
# -----------------------------------------------------------------------------
# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024
# > description: This file is the master file for data aggregation. It calls all
#   necessary functions to get data from API sources and aggregate it together with
#   other locally stored spatial data sources. It does so by converting all data to 
#   parquet files (faster imports) and creating mappings for fast lookups (no single data base). 


# SETTIGNS --------------------------------------------------------------------
dataagg_settings = {
        'name_dir_export': None,            # name of the directory where the data is exported to (name to replace/ extend the name of the folder "preprep_data" in the end)
        'script_run_on_server': False,      # F: run on private computer, T: run on server
        'smaller_import': False,             # F: import all data, T: import only a small subset of data (smaller range of years) for debugging
        'show_debug_prints': True,          # F: certain print statements are omitted, T: includes print statements that help with debugging

        'kt_numbers': [12,13], #[1,2,3],#[12, 13,],                       # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain individual municipalities
        'bfs_numbers': [],  # list of municipalites to select for allocation (only used if kt_numbers == 0)
        'year_range': [2018, 2023],             # range of years to import

        'reimport_api_data': True,         # F: use existing parquet files, T: recreate parquet files in data prep        
        'rerun_spatial_mappings': True,     # F: use existing parquet files, T: recreate parquet files in data prep
        'reextend_fixed_data': True,        # F: use existing exentions calculated beforehand, T: recalculate extensions (e.g. pv installation costs per partition) again       
        }


# PACKAGES --------------------------------------------------------------------
import sys
sys.path.append('D:/RaulHochuli_inuse/OptimalPV_RH') if dataagg_settings['script_run_on_server'] else sys.path.append('C:/Models/OptimalPV_RH')

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
import auxiliary_functions
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile

from data_aggregation.api_electricity_prices import api_electricity_prices_data
from data_aggregation.sql_gwr import sql_gwr_data
from data_aggregation.api_pvtarif import api_pvtarif_data, api_pvtarif_gm_ewr_Mapping
from data_aggregation.api_entsoe import api_entsoe_ahead_elecpri_data
from data_aggregation.preprepare_data import local_data_to_parquet_AND_create_spatial_mappings
from data_aggregation.extend_data import attach_pv_cost



# SETUP -----------------------------------------------------------------------
# set working directory
wd_path = "D:\\RaulHochuli_inuse\\OptimalPV_RH"  if dataagg_settings['script_run_on_server'] else "C:\Models\OptimalPV_RH"
data_path = f'{wd_path}_data'
    
# create directory + log file
preprepd_path = f'{data_path}/output/preprep_data' 
if not os.path.exists(preprepd_path):
    os.makedirs(preprepd_path)

log_name = f'{data_path}/output/prepre_data_log.txt'
chapter_to_logfile(f'start data_aggregation_MASTER', log_name, overwrite_file=True)
print_to_logfile(f' > settings: \n{pformat(dataagg_settings)}', log_name)

# get bfs numbers from canton selection if applicable
if not not dataagg_settings['kt_numbers']: 
    dataagg_settings['bfs_numbers'] = auxiliary_functions.get_bfs_from_ktnr(dataagg_settings['kt_numbers'], data_path, log_name)
    print_to_logfile(f' > no. of kt  numbers in selection: {len(dataagg_settings["kt_numbers"])}', log_name)
    print_to_logfile(f' > no. of bfs numbers in selection: {len(dataagg_settings["bfs_numbers"])}', log_name) 

# add information to dataagg_settings that's relevant for further functions
dataagg_settings['log_file_name'] = log_name
dataagg_settings['wd_path'] = wd_path
dataagg_settings['data_path'] = data_path



# IMPORT API DATA ---------------------------------------------------------------
# download API data and store it local directory for needed time and range
file_exists_TF = os.path.exists(f'{data_path}/output/preprep_data/elecpri.parquet')  # conditions that determine if the data should be reimported
reimport_api_data = dataagg_settings['reimport_api_data']

year_range_gwr = dataagg_settings['year_range'] if not not dataagg_settings['year_range'] else [2009, 2023] # else statement shows max range of years available
year_range_pvtarif = dataagg_settings['year_range'] if not not dataagg_settings['year_range'] else [2015, 2023]

if not file_exists_TF or reimport_api_data:
    subchapter_to_logfile('pre-prep data: API ELECTRICITY PRICES', log_name)
    api_electricity_prices_data(dataagg_settings_def = dataagg_settings)
        
    subchapter_to_logfile('pre-prep data: SQL GWR DATA', log_name)
    sql_gwr_data(dataagg_settings_def=dataagg_settings)

    subchapter_to_logfile('pre-prep data: API PVTARIF', log_name)
    api_pvtarif_data(dataagg_settings_def=dataagg_settings)

    subchapter_to_logfile('pre-prep data: API GM by EWR MAPPING', log_name)
    api_pvtarif_gm_ewr_Mapping(dataagg_settings_def = dataagg_settings)

    subchapter_to_logfile('pre-prep data: API ENTSOE DayAhead', log_name)
    api_entsoe_ahead_elecpri_data(dataagg_settings_def = dataagg_settings)




else:
    print_to_logfile('\n\n', log_name)
    checkpoint_to_logfile('use already downloaded data on electricity prices, GWR, PV Tarifs', log_name)



# IMPORT LOCAL DATA + SPATIAL MAPPINGS ------------------------------------------



# transform spatial data to parquet files for faster import and transformation
pq_dir_exists_TF = os.path.exists(f'{data_path}/output/preprep_data')
pq_files_rerun = dataagg_settings['rerun_spatial_mappings']

if not pq_dir_exists_TF or pq_files_rerun:
    subchapter_to_logfile('pre-prep data: IMPORT LOCAL DATA + create SPATIAL MAPPINGS', log_name)
    local_data_to_parquet_AND_create_spatial_mappings(dataagg_settings_def = dataagg_settings)
    
    # create_spatial_mappings(script_run_on_server_def= dataagg_settings['script_run_on_server'], smaller_import_def=dataagg_settings['smaller_import'], log_file_name_def=log_name, wd_path_def=wd_path, data_path_def=data_path, show_debug_prints_def=dataagg_settings['show_debug_prints'])    
    # subchapter_to_logfile('pre-prep data: SPATIAL DATA to PARQUET', log_name) # extend all spatial data sources with the gm_id and export it to parquet files for easier handling later on
    # solkat_spatial_toparquet(dataagg_settings['script_run_on_server'], dataagg_settings['smaller_import'], log_name, wd_path, data_path, dataagg_settings['show_debug_prints'])
    # heat_spatial_toparquet(dataagg_settings['script_run_on_server'], dataagg_settings['smaller_import'], log_name, wd_path, data_path, dataagg_settings['show_debug_prints'])
    # pv_spatial_toparquet(dataagg_settings['script_run_on_server'], dataagg_settings['smaller_import'], log_name, wd_path, data_path, dataagg_settings['show_debug_prints'])

else: 
    print_to_logfile('\n', log_name)
    checkpoint_to_logfile('use parquet files and mappings that exist already', log_name)



# EXTEND WITH TIME FIXED DATA ---------------------------------------------------------------
cost_df_exists_TF = os.path.exists(f'{data_path}/output/preprep_data/pvinstcost.parquet')
reextend_fixed_data = dataagg_settings['reextend_fixed_data']

if not cost_df_exists_TF or reextend_fixed_data:
    subchapter_to_logfile('extend data: ESTIM PV INSTALLTION COST FUNCTION,  ', log_name)
    attach_pv_cost(dataagg_settings_def = dataagg_settings)
    


   
# COPY AGGREGATED DATA ---------------------------------------------------------------
# > not to overwrite it while debugging 
chapter_to_logfile(f'END data_aggregation_MASTER', log_name, overwrite_file=False)

if dataagg_settings['name_dir_export'] is None:    

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



###############################
###############################


###############################
###############################
# BOOKMARK > delete in February 2024



###############################
###############################
# BOOKMARK > delete in July 2024
###############################
###############################
