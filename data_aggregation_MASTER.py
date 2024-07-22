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
        'name_dir_export': None,            # name of the directory where the data is exported to (name to replace the name of the folder "preprep_data" in the end)
        'script_run_on_server': False,      # F: run on private computer, T: run on server
        'smaller_import': True,             # F: import all data, T: import only a small subset of data (smaller range of years) for debugging
        'reimport_api_data': False,          # F: use existing parquet files, T: recreate parquet files in data prep
        'rerun_spatial_mappings': True,     # F: use existing parquet files, T: recreate parquet files in data prep
        'reextend_fixed_data': True,        # F: use existing exentions calculated beforehand, T: recalculate extensions (e.g. pv installation costs per partition) again 
        'show_debug_prints': True,          # F: certain print statements are omitted, T: includes print statements that help with debugging

        'bfs_numbers_OR_shape': 
        # [2761, 2763, 2842, 2787],           # small list for debugging   # BL: als BFS lists from  
        [2829, 2770, 2888, 2788, 2787, 2885, 2858, 2823, 2831, 2791, 2821, 2846, 2884, 2782, 2893, 2861, 2762, 2844, 2895, 2852, 2868, 2771, 2834, 2775, 2761, 2883, 2889, 2769, 2855, 2781, 2773, 2866, 2856, 2763, 2869, 2784, 2790, 2882, 2768, 2892, 2886, 2865, 2785, 2828, 2792, 2853, 2860, 2772, 2863, 2825, 2793, 2824, 2765, 2891, 2764, 2887, 2847, 2841, 2894, 2789, 2833, 2881, 2848, 2786, 2867, 2849, 2830, 2767, 2857, 2783, 2766, 2862, 2842, 2859, 2864, 2832, 2843, 2890, 2854, 2822, 2827, 2851, 2850, 2845, 2774, 2826, 2827],      
        'gwr_house_type_class': [0,], 
        'solkat_house_type_class': [0,], 
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
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile
from data_aggregation.api_electricity_prices import api_electricity_prices
from data_aggregation.sql_gwr import sql_gwr_data
from data_aggregation.api_pvtarif import api_pvtarif_data, api_pvtarif_gm_ewr_Mapping
from data_aggregation.preprepare_data import solkat_spatial_toparquet, gwr_spatial_toparquet, heat_spatial_toparquet, pv_spatial_toparquet, create_spatial_mappings
from data_aggregation.installation_cost import attach_pv_cost



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



# IMPORT API DATA ---------------------------------------------------------------

# download possible API data to local directory
file_exists_TF = os.path.exists(f'{data_path}/output/preprep_data/elecpri.parquet')  # conditions that determine if the data should be reimported
reimport_api_data = dataagg_settings['reimport_api_data']

year_range_gwr = [2020, 2021] if dataagg_settings['smaller_import'] else [2009, 2023]  # range of years to import, smaller range for debugging
year_range_pvtarif = [2020, 2021] if dataagg_settings['smaller_import'] else [2015, 2023] 

if not file_exists_TF or reimport_api_data:
    subchapter_to_logfile('pre-prep data: API ELECTRICITY PRICES', log_name)
    api_electricity_prices(script_run_on_server_def = dataagg_settings['script_run_on_server'], smaller_import_def=dataagg_settings['smaller_import'], log_file_name_def=log_name, wd_path_def=wd_path, data_path_def=data_path, show_debug_prints_def=dataagg_settings['show_debug_prints'], year_range_def = year_range_gwr)
    
    subchapter_to_logfile('pre-prep data: SQL GWR DATA', log_name)
    sql_gwr_data(script_run_on_server_def = dataagg_settings['script_run_on_server'], smaller_import_def=dataagg_settings['smaller_import'], log_file_name_def=log_name, wd_path_def=wd_path, data_path_def=data_path, show_debug_prints_def=dataagg_settings['show_debug_prints'])

    subchapter_to_logfile('pre-prep data: API PVTARIF', log_name)
    api_pvtarif_data(script_run_on_server_def = dataagg_settings['script_run_on_server'], smaller_import_def=dataagg_settings['smaller_import'], log_file_name_def=log_name, wd_path_def=wd_path, data_path_def=data_path, show_debug_prints_def=dataagg_settings['show_debug_prints'], year_range_def=year_range_pvtarif )

    subchapter_to_logfile('pre-prep data: API PVTARIF to GM MAPPING', log_name)
    api_pvtarif_gm_ewr_Mapping(script_run_on_server_def = dataagg_settings['script_run_on_server'], smaller_import_def=dataagg_settings['smaller_import'], log_file_name_def=log_name, wd_path_def=wd_path, data_path_def=data_path, show_debug_prints_def=dataagg_settings['show_debug_prints'], year_range_def=year_range_pvtarif )

else:
    print_to_logfile('\n\n', log_name)
    checkpoint_to_logfile('use already downloaded data on electricity prices, GWR, PV Tarifs', log_name)



# SPATIAL MAPPINGS ---------------------------------------------------------------

# transform spatial data to parquet files for faster import and transformation
pq_dir_exists_TF = os.path.exists(f'{data_path}/output/preprep_data')
pq_files_rerun = dataagg_settings['rerun_spatial_mappings']

if not pq_dir_exists_TF or pq_files_rerun:
    subchapter_to_logfile('pre-prep data: SPATIAL MAPPINGS', log_name)
    create_spatial_mappings(script_run_on_server_def= dataagg_settings['script_run_on_server'], smaller_import_def=dataagg_settings['smaller_import'], log_file_name_def=log_name, wd_path_def=wd_path, data_path_def=data_path, show_debug_prints_def=dataagg_settings['show_debug_prints'])    

    subchapter_to_logfile('pre-prep data: SPATIAL DATA to PARQUET', log_name) # extend all spatial data sources with the gm_id and export it to parquet files for easier handling later on
    solkat_spatial_toparquet(dataagg_settings['script_run_on_server'], dataagg_settings['smaller_import'], log_name, wd_path, data_path, dataagg_settings['show_debug_prints'])
    heat_spatial_toparquet(dataagg_settings['script_run_on_server'], dataagg_settings['smaller_import'], log_name, wd_path, data_path, dataagg_settings['show_debug_prints'])
    pv_spatial_toparquet(dataagg_settings['script_run_on_server'], dataagg_settings['smaller_import'], log_name, wd_path, data_path, dataagg_settings['show_debug_prints'])

else: 
    print_to_logfile('\n', log_name)
    checkpoint_to_logfile('use parquet files and mappings that exist already', log_name)



# EXTEND WITH TIME FIXED DATA ---------------------------------------------------------------
cost_df_exists_TF = os.path.exists(f'{data_path}/output/preprep_data/pvinstcost.parquet')
extend_data_rerun = dataagg_settings['reextend_fixed_data']

if not cost_df_exists_TF or extend_data_rerun:
    subchapter_to_logfile('extend data: PV INSTALLTION COST', log_name)
    attach_pv_cost(script_run_on_server_def= dataagg_settings['script_run_on_server'],  
                     log_file_name_def=log_name,
                     wd_path_def=wd_path, 
                     smaller_import_def=dataagg_settings['smaller_import'],
                     data_path_def=data_path, 
                     show_debug_prints_def=dataagg_settings['show_debug_prints'])
    
    subchapter_to_logfile('extend data: WEIGHTS FOR ELECTRICITY DEMAND', log_name)


   
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
