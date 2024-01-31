# -----------------------------------------------------------------------------
# pv_allocation_MASTER.py 
# -----------------------------------------------------------------------------
# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024
# > description: 


# TO-DOs:
# TODO: Adjust GBAUJ to other variable


# SETTIGNS --------------------------------------------------------------------
pvalloc_settings = {
        'name_dir_export': 'test_OW',             # name of the directory for the aggregated data, and maybe file extension
        'name_dir_import': 'preprep_data_20240126_01h',
        'script_run_on_server': False,              # F: run on private computer, T: run on server
        'show_debug_prints': True,                  # F: certain print statements are omitted, T: includes print statements that help with debugging

        'bfs_numbers': [1404,], #'kt_OW',                     # list of municipalites; certain Kantons are pre-defined (incl all municipalities, e.g. kt_LU)
        'topology_year_range':[1900, 2022],
        'gwr_house_type_class': ['1110',],               # list of house type classes to be considered

        'prediction_year_range':[2024, 2025],
        'solkat_house_type_class': [0,],            # list of house type classes to be considered
        'rate_operation_cost': 0.01,                # assumed rate of operation cost (of investment cost)

        'NPV_include_wealth_tax': False,            # F: exclude wealth tax from NPV calculation, T: include wealth tax in NPV calculation
        'smaller_import': True,                     # F: import all data, T: import only a small subset of data for debugging
         }


# PACKAGES --------------------------------------------------------------------
import sys
if pvalloc_settings['script_run_on_server']:
    sys.path.append('C:/Models/OptimalPV_RH') 
elif pvalloc_settings['script_run_on_server']:
    sys.path.append('D:/RaulHochuli_inuse/OptimalPV_RH')

# external packages
import os as os
import pandas as pd
import geopandas as gpd
import numpy as np
import dask.dataframe as dd
from datetime import datetime
from pprint import pformat

import glob
import shutil
import winsound
import subprocess
import pprint


# own packages and functions
from functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile
from data_aggregation.api_electricity_prices import api_electricity_prices
from data_aggregation.preprepare_data import solkat_spatial_toparquet, gwr_spatial_toparquet, heat_spatial_toparquet, pv_spatial_toparquet, create_spatial_mappings



# SETUP -----------------------------------------------------------------------
# set working directory
wd_path = "D:\\RaulHochuli_inuse\\OptimalPV_RH"  if pvalloc_settings['script_run_on_server'] else "C:\Models\OptimalPV_RH"
data_path = f'{wd_path}_data'

# create directory + log file
if not os.path.exists(f'{data_path}/output/{pvalloc_settings["name_dir_export"]}'):
    os.makedirs(f'{data_path}/output/{pvalloc_settings["name_dir_export"]}') 

log_name = f'{data_path}/output/{pvalloc_settings["name_dir_export"]}_log_file.txt'
chapter_to_logfile(f'start data_aggregation_MASTER for: {pvalloc_settings["name_dir_export"]}', log_name, overwrite_file=True)
print_to_logfile(f' > settings: \n{pformat(pvalloc_settings)}', log_name)

# adjust "bfs_number" for large, pre-defined municicpaltie groups
gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')

if pvalloc_settings['bfs_numbers'] == 'kt_LU':
    gm_shp_sub = gm_shp[gm_shp['KANTONSNUM'] == 3]
    pvalloc_settings['bfs_numbers'] = gm_shp_sub['BFS_NUMMER'].unique().tolist()

elif pvalloc_settings['bfs_numbers'] == 'kt_OW':
    gm_shp_sub = gm_shp[gm_shp['KANTONSNUM'] == 6]
    pvalloc_settings['bfs_numbers'] = gm_shp_sub['BFS_NUMMER'].unique().tolist()


print_to_logfile(f' > list bfs_numbers: {pvalloc_settings["bfs_numbers"]}', log_name)


# ----------------------------------------------------------------------------------------------------------------------------------

# plan:
#-- stil before pv_allocation
#   > create a py file that creates all assumptions, cost etc. 

#-- prepare all data computations

# --- cost computation
#   > compute cost per roof partition
#   >> "downwoard" computation -> compute NPV for best partition, second best and best partition, etc.
#   >> include ratio of self consumption

#   > (compute elec demand by heating squrare)
#   > 

#-- subset selection


#-- initiate topology
#   > create dict with gwr id and year
#   >> subsetable by bfs number and building type

#   > define thresholds for installations per year

#   > assign all partitions to dict
#   > assign production of installed pv to dict
#   > (assign grid connection to dict)

#-- calculate NPV
#   > select free gwrs 
#   > calculate NPV by partition (possible with switch to only consider KLASSE 3+)
#   > select best NPV
# ----------------------------------------------------------------------------------------------------------------------------------


# IMPORT ----------------------------------------------------------------
bfs_list = pvalloc_settings['bfs_numbers']
bfs_list_str = [str(bfs) for bfs in bfs_list]

# mappings
Map_egroof_sbroof = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/Map_egroof_sbroof.parquet')
Map_egroof_pv = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/Map_egroof_pv.parquet')

checkpoint_to_logfile(f'start import with DD', log_name, n_tabs_def=2, show_debug_prints_def=pvalloc_settings['show_debug_prints'])
# solkat
solkat_dd = dd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/solkat_by_gm.parquet')
solkat = solkat_dd[solkat_dd['BFS_NUMMER'].isin(bfs_list)].compute()

# gwr
gwr_dd = dd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/gwr.parquet')
gwr = gwr_dd[gwr_dd['GGDENR'].isin(bfs_list_str)].compute()

# pv
pv_dd = dd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/pv_by_gm.parquet')
pv = pv_dd[pv_dd['BFS_NUMMER'].isin(bfs_list)].compute()
checkpoint_to_logfile(f'end import with DD', log_name, n_tabs_def=2, show_debug_prints_def=pvalloc_settings['show_debug_prints'])


checkpoint_to_logfile(f'start import ALL in PQ', log_name, n_tabs_def=2, show_debug_prints_def=pvalloc_settings['show_debug_prints'])
solkat_pq = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/solkat_by_gm.parquet')
gwr_pq = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/gwr.parquet')
pv_pq = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/pv_by_gm.parquet')
checkpoint_to_logfile(f'end import ALL in PQ', log_name, n_tabs_def=2, show_debug_prints_def=pvalloc_settings['show_debug_prints'])




# INITIATE TOPOLOGY ----------------------------------------------------------------
# convert relevant data types 
gwr['GBAUJ'] = pd.to_numeric(gwr['GBAUJ'], errors='coerce')
gwr['GBAUJ'] = gwr['GBAUJ'].fillna(0)  # replace NaN with 0
gwr['GBAUJ'] = gwr['GBAUJ'].astype(int)

pv['xtf_id'] = pv['xtf_id'].astype(str)
# pv['BeginningOfOperation'] = pv['BeginningOfOperation'].replace('nan', np.nan)

Map_egroof_pv['EGID'] = Map_egroof_pv['EGID'].fillna(-1).astype(int).astype(str)    # Fill NaN values with -1, convert to integers, and then to strings
Map_egroof_pv['xtf_id'] = Map_egroof_pv['xtf_id'].fillna(-1).astype(int).astype(str)
Map_egroof_pv['EGID'] = Map_egroof_pv['EGID'].replace('-1', np.nan)                  # Replace '-1' with 'nan'
Map_egroof_pv['xtf_id'] = Map_egroof_pv['xtf_id'].replace('-1', np.nan)


# create dict with gwr id and year
gwr_sub = gwr[
    (gwr['GBAUJ'] >= pvalloc_settings['topology_year_range'][0]) &
    (gwr['GBAUJ'] <= pvalloc_settings['topology_year_range'][1]) &
    (gwr['GKLAS'].isin(pvalloc_settings['gwr_house_type_class']))]

pvtopo_df = gwr_sub[['EGID', 'GBAUJ', 'GKLAS']].copy()
pvtopo_df.set_index('EGID', inplace=True)
pvtopo = pvtopo_df.to_dict('index')
for key, value in pvtopo.items():
    value['EGID'] = key


# attach pv xtf_id to dict
pvtopo = {k: {**v, 'pv': np.nan} for k, v in pvtopo.items()} # create new key with value np.nan
Map_egroof_pv_dict = Map_egroof_pv.set_index('EGID')['xtf_id'].to_dict() # Convert the Map_egroof_pv DataFrame to a dictionary

Map_pv_BegOfOp = pv[['xtf_id','BeginningOfOperation']].copy()




for key, value in pvtopo.items():
    # print(key)
    # if key in Map_egroof_pv_dict :
    if (key in Map_egroof_pv['EGID'].tolist()) & (pd.isna(Map_egroof_pv.loc[Map_egroof_pv['EGID'] == key, 'xtf_id'].values[0])):
        # Set the 'xtf_id' and 'pv' values in the pvtopo dictionary
        value['pv'] = 1
        value['xtf_id'] = Map_egroof_pv_dict[key]
        value['BegOfOp'] = Map_pv_BegOfOp.loc[Map_pv_BegOfOp['xtf_id'] == value['xtf_id'], 'BeginningOfOperation'].iloc[0]
        



def print_5_items(dict_def):
    """
    Function to print the first 5 items of a dictionary
    """
    first_5_items = dict(list(dict_def.items())[:5])
    print(first_5_items)

pvtopo_subset = {k: v for k, v in pvtopo.items() if 'xtf_id' in v}
print_5_items(pvtopo)
print_5_items(pvtopo_subset)


print("End of script")
# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------





