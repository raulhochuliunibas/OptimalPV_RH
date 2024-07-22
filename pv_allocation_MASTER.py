# -----------------------------------------------------------------------------
# pv_allocation_MASTER.py 
# -----------------------------------------------------------------------------
# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024
# > description: 


# TO-DOs:
# TODO: Adjust GBAUJ to other variable
# TODO: Include WNART to only consider residential buildings for "primary living"
# TODO: check in QGIS if GKLAS == 1273 are really also denkmalgeschützt buildings or just monuments



# SETTIGNS --------------------------------------------------------------------
pvalloc_settings = {
        'name_dir_export': 'test_BSBL',             # name of the directory for the aggregated data, and maybe file extension
        'name_dir_import': 'preprep_data_20240207_04h_W_COST',
        'script_run_on_server': False,              # F: run on private computer, T: run on server
        'show_debug_prints': True,                  # F: certain print statements are omitted, T: includes print statements that help with debugging

        'bfs_numbers': [2763,], #'kt_OW',                     # list of municipalites; certain Kantons are pre-defined (incl all municipalities, e.g. kt_LU)
        'topology_year_range':[2019, 2022],
        'gwr_house_type_class': ['1110',],               # list of house type classes to be considered

        'prediction_year_range':[2023, 2025],
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
chapter_to_logfile(f'start pv_allocation_MASTER for: {pvalloc_settings["name_dir_export"]}', log_name, overwrite_file=True)
print_to_logfile(f' > settings: \n{pformat(pvalloc_settings)}', log_name)

# adjust "bfs_number" for large, pre-defined municicpaltie groups
gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')

if pvalloc_settings['bfs_numbers'] == 'kt_LU':
    gm_shp_sub = gm_shp[gm_shp['KANTONSNUM'] == 3]
    pvalloc_settings['bfs_numbers'] = gm_shp_sub['BFS_NUMMER'].unique().tolist()

elif pvalloc_settings['bfs_numbers'] == 'kt_OW':
    gm_shp_sub = gm_shp[gm_shp['KANTONSNUM'] == 6]
    pvalloc_settings['bfs_numbers'] = gm_shp_sub['BFS_NUMMER'].unique().tolist()

elif pvalloc_settings['bfs_numbers'] == 'kt_BSBL':
    gm_shp_sub = gm_shp[gm_shp['KANTONSNUM'].isin([12, 13])]
    pvalloc_settings['bfs_numbers'] = gm_shp_sub['BFS_NUMMER'].unique().tolist()


print_to_logfile(f' > list bfs_numbers: {pvalloc_settings["bfs_numbers"]}', log_name)



# IMPORT ----------------------------------------------------------------
bfs_list = pvalloc_settings['bfs_numbers']
bfs_list_str = [str(bfs) for bfs in bfs_list]

# mappings
Map_egroof_sbroof = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/Map_egroof_sbroof.parquet')
Map_egroof_pv = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/Map_egroof_pv.parquet')
Map_gm_ewr = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/Map_gm_ewr.parquet')

checkpoint_to_logfile(f'start import with DD', log_name, n_tabs_def=2, show_debug_prints_def=pvalloc_settings['show_debug_prints'])
# gwr
gwr_dd = dd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/gwr.parquet')
gwr = gwr_dd[gwr_dd['GGDENR'].isin(bfs_list_str)].compute()

# solkat
# solkat_dd = dd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/solkat_by_gm.parquet')
# solkat = solkat_dd[solkat_dd['BFS_NUMMER'].isin(bfs_list)].compute()
solkat_cumm_dd = dd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/solkatcost_cumm.parquet')
solkat_cumm = solkat_cumm_dd[solkat_cumm_dd['GWR_EGID'].isin(gwr['EGID'].unique())].compute()
solkat = solkat_cumm.copy()

# pv
pv_dd = dd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/pv_by_gm.parquet')
pv = pv_dd[pv_dd['BFS_NUMMER'].isin(bfs_list)].compute()
checkpoint_to_logfile(f'end import with DD', log_name, n_tabs_def=2, show_debug_prints_def=pvalloc_settings['show_debug_prints'])

# pvtarif
pvtarif = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/pvtarif.parquet')

# electricity prices
elecpri = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/elecpri.parquet')

# also import parquet files for comparison
checkpoint_to_logfile(f'start import ALL in PQ', log_name, n_tabs_def=2, show_debug_prints_def=pvalloc_settings['show_debug_prints'])
solkat_pq = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/solkat_by_gm.parquet')
gwr_pq = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/gwr.parquet')
pv_pq = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/pv_by_gm.parquet')
checkpoint_to_logfile(f'end import ALL in PQ', log_name, n_tabs_def=2, show_debug_prints_def=pvalloc_settings['show_debug_prints'])

# convert all ID columns to string
def convert_srs_to_str(df, col_name):
    df[col_name] = df[col_name].fillna(-1).astype(int).astype(str)          # Fill NaN values with -1, convert to integers, and then to stringsMap_egroof_sbroof['EGID'] = Map_egroof_sbroof['EGID'].fillna(-1).astype(int).astype(str)    # Fill NaN values with -1, convert to integers, and then to strings
    df[col_name] = df[col_name].replace('-1', np.nan)                       # Replace '-1' with 'nan'
    return df

Map_egroof_sbroof = convert_srs_to_str(Map_egroof_sbroof, 'EGID')
Map_egroof_pv = convert_srs_to_str(Map_egroof_pv, 'EGID')
Map_egroof_pv = convert_srs_to_str(Map_egroof_pv, 'xtf_id')
Map_gm_ewr = convert_srs_to_str(Map_gm_ewr, 'bfs')

solkat = convert_srs_to_str(solkat, 'GWR_EGID')

pv = convert_srs_to_str(pv, 'xtf_id')
pv = convert_srs_to_str(pv, 'BFS_NUMMER')

# converte other data types
gwr['GBAUJ'] = pd.to_numeric(gwr['GBAUJ'], errors='coerce')
gwr['GBAUJ'] = gwr['GBAUJ'].fillna(0)  # replace NaN with 0
gwr['GBAUJ'] = gwr['GBAUJ'].astype(int)

# transform to date
pv_capa = pv.copy()
pv_capa['BeginningOfOperation'] = pd.to_datetime(pv_capa['BeginningOfOperation'])


# ----------------------------------------------------------------------------------------------------------------------------------

# plan:
#-- stil before pv_allocation
#   > create a py file that creates all assumptions, cost etc. 

#-- prepare all data computations

# --- cost computation
#   > compute cost per roof partition - CHECK
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




# INITIATE TOPOLOGY ----------------------------------------------------------------


# create topo with EGID for certain GWR filter criteria
gwr_sub = gwr[
    # (gwr['GBAUJ'] >= pvalloc_settings['topology_year_range'][0]) &
    (gwr['GBAUJ'] <= pvalloc_settings['topology_year_range'][1]) &
    (gwr['GKLAS'].isin(pvalloc_settings['gwr_house_type_class']))&
    (gwr['GSTAT'] == '1004')].copy() # only consider buildings that are standing

pvtopo_df = gwr_sub[['EGID', 'GBAUJ', 'GKLAS']].copy()
pvtopo = pvtopo_df

# pvtopo_df.set_index('EGID', inplace=True)
# pvtopo = pvtopo_df.to_dict('index')
# for key, value in pvtopo.items():
#     value['EGID'] = key


# attach pv xtf_id to topo
# pvtopo = {k: {**v, 'pv': np.nan} for k, v in pvtopo.items()} # create new key with value np.nan
# Map_egroof_pv_dict = Map_egroof_pv.set_index('EGID')['xtf_id'].to_dict() # Convert the Map_egroof_pv DataFrame to a dictionary
Map_pv_BegOfOp = pv[['xtf_id','BeginningOfOperation']].copy()
Map_pv_BegOfOp.rename(columns={'BeginningOfOperation': 'BegOfOp'}, inplace=True)


pvtopo = pvtopo.merge(Map_egroof_pv[['EGID', 'xtf_id']], on='EGID', how='left', suffixes=('_pvtopo', '_Map_egroof_pv'))
pvtopo[['pv',]] = np.nan
pvtopo.loc[pd.notna(pvtopo['xtf_id']), 'pv'] = 1
pvtopo = pvtopo.merge(Map_pv_BegOfOp, on='xtf_id', how='left', suffixes=('_pvtopo', '_Map_pv_BegOfOp'))
pvtopo['BegOfOp'] = pvtopo['BegOfOp'].replace({'<NA>': np.nan})


# COMPUTE NPV ----------------------------------------------------------------

# ASSUMPTION / PARAMETERs
interest_rate = 0.01
inflation_rate = 0.018  
maturity = 25
disc_rate = (interest_rate+ inflation_rate + interest_rate * inflation_rate) # NOTE: Source?
disc_denominator = np.sum((1+disc_rate)** np.arange(1, maturity+1))
capa_years = [2019, 2023]
ann_inst_capa = pv_capa.loc[pv_capa['BeginningOfOperation'].dt.year.isin(capa_years), 'InitialPower'].sum() / len(capa_years)




year = pvalloc_settings['topology_year_range'][1]
year2d = str(year % 100).zfill(2)
solkat_t = solkat.copy()

pvtarif_t = pvtarif.loc[pvtarif['year'] == year2d, :].copy()
pvtarif_bygm_t = Map_gm_ewr.copy().merge(pvtarif_t, on='nrElcom', how='left', suffixes=('_pvtarif_bygm_t', '_pvtarif'))
# pvtarif_bygm_t['energy1'].replace({np.nan: 0}, inplace=True)    # distorts the mean with a value of 0
pvtarif_bygm_t['energy1'] = pvtarif_bygm_t['energy1'].astype(float) 
pvtarif_bygm_t = pvtarif_bygm_t.dropna(subset=['energy1'])

# group over bfs and average power1           
pvtarif_bygm_t = pvtarif_bygm_t.groupby('bfs').agg({'energy1': 'mean'}).reset_index()
pvtarif_bygm_t.rename(columns={'bfs': 'BFS_NUMMER', 'energy1': 'avg_pvrate_RpkWh'}, inplace=True)

# attach BFS_NUMMER to solkat
attach_BFS = gwr[['EGID', 'GGDENR']].copy()
attach_BFS.rename(columns={'EGID': 'GWR_EGID', 'GGDENR': 'BFS_NUMMER'}, inplace=True)
solkat_t = solkat_t.merge(attach_BFS, on='GWR_EGID', how='left', suffixes=('_solkat_t', '_attach_BFS'))

# attach avg_pvrate_RpkWh to solkat
solkat_t = solkat_t.merge(pvtarif_bygm_t, on='BFS_NUMMER', how='left', suffixes=('_solkat_t', '_pvtarif_bygm_t'))

# NPV CALCULATION -----------
solkat_t['pv_gain_1y'] = solkat_t['STROME_cumm'] * solkat_t['avg_pvrate_RpkWh']
solkat_t['NPV'] = np.nan
solkat_t['NPV'] = (maturity * solkat_t['pv_gain_1y'] /disc_denominator )-solkat_t['partition_pv_cost_chf']

# solkat_t.head(400).to_csv(f'{data_path}/output/solkat_t400.csv')
# solkat_t.head(400).to_excel(f'{data_path}/output/solkat_t400.xlsx')
# BOOKMARK: 

import matplotlib.pyplot as plt

# Plot the PDF of 'NPV'
solkat_t['NPV'].hist(density=True, bins=100)
plt.xlabel('NPV')
plt.ylabel('PDF')
plt.title('PDF of NPV')
plt.show()
# Plot the CDF of 'NPV'
solkat_t['NPV'].hist(cumulative=True, density=1, bins=100)
plt.xlabel('NPV')
plt.ylabel('CDF')
plt.title('CDF of NPV')
plt.show()

egid_noinst_t = pvtopo.loc[pvtopo['pv'] != 1, :]





print("End of script")
# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------




import pandas as pd

for id in pvtopo['EGID']:
    filtered_data = Map_egroof_pv.loc[Map_egroof_pv['EGID'] == id, 'xtf_id'].values

    if len(filtered_data) > 0:
        xtf_id_value = filtered_data[0]
        pvtopo.loc[pvtopo['EGID'] == id, 'pv'] = 1


