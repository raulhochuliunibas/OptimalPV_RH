import os as os
import pandas as pd
import geopandas as gpd
import glob
import shutil
import winsound
import functions
import datetime

from functions import chapter_to_logfile, checkpoint_to_logfile
from data_aggregation.local_data_import_aggregation import import_aggregate_data
from data_aggregation.spatial_data_toparquet_by_gm import spatial_toparquet
from datetime import datetime

# SETTIGNS --------------------------------------------------------------------
script_run_on_server = 0
# recreate_parquet_files = 1


# SETUP -----------------------------------------------------------------------
if script_run_on_server == 0:
    wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
    data_path = f'{wd_path}_data'
    
    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp',
                           layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')

elif script_run_on_server == 1:
    wd_path = "D:/RaulHochuli_inuse/OptimalPV_RH"
    data_path = f'{wd_path}_data'

    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp',
                           layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    

# ------------------------------------------------------------------------------
# --- TRY Map_roof_pv from Server ----------------------------------------------
# ------------------------------------------------------------------------------
if True: 
     print('asdf')

chunk_name = 'try_map_roof_pv_from_server'

# log file
export_txt_name = f'{wd_path}/try_out_{chunk_name}_log.txt'
with open(export_txt_name, 'w') as export_txt:
    export_txt.write(f'\n {chunk_name}, time: {datetime.now()} \n {10*"*"} \n')
     
#import     
Map_roof_pv = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm_SERVER/Map_roof_pv.parquet')
roof_kat = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm_SERVER/roof_kat_by_gm.parquet')
pv = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm_SERVER/pv_by_gm.parquet')

with open(export_txt_name, 'a') as export_txt:
    export_txt.write(f'\n roof_kat.shape: {roof_kat.shape}')
    export_txt.write(f'\n len(roof_kat[SB_UUID].unique()): {len(roof_kat["SB_UUID"].unique())}')
    
    export_txt.write(f'\n\n Map_roof_pv.shape: {Map_roof_pv.shape}')
    export_txt.write(f'\n len(Map_roof_pv[SB_UUID].unique()): {len(Map_roof_pv["SB_UUID"].unique())}')
    export_txt.write(f'\n in percent: {(len(Map_roof_pv["SB_UUID"].unique()) - Map_roof_pv.shape[0])/ Map_roof_pv.shape[0]}')
    
    export_txt.write(f'\n\n len(roof_kat[SB_UUID].unique()): {len(roof_kat["SB_UUID"].unique())}')
    export_txt.write(f'\n len(Map_roof_pv[SB_UUID].unique()): {len(Map_roof_pv["SB_UUID"].unique())}')

    export_txt.write(f'\n\n pv.shape: {pv.shape[0]}')
    export_txt.write(f'\n len(pv[xtf_id].unique()): {len(pv["xtf_id"].unique())}')
    export_txt.write(f'\n Map_roof_pv[xtf_id]: {Map_roof_pv["xtf_id"].nunique()}')
    export_txt.write(f'\n in percent: {(len(Map_roof_pv["xtf_id"].unique()) - pv["xtf_id"].nunique())/ pv["xtf_id"].nunique()}')


roof_kat.shape[0]
len(roof_kat['SB_UUID'].unique())
roof_kat.shape[0] / len(roof_kat['SB_UUID'].unique())

Map_roof_pv.shape[0]
len(Map_roof_pv['SB_UUID'].unique())
(len(Map_roof_pv['SB_UUID'].unique()) - Map_roof_pv.shape[0])/ Map_roof_pv.shape[0]

len(Map_roof_pv['SB_UUID'].unique()) 
len(roof_kat['SB_UUID'].unique())


pv.shape[0]
len(pv['xtf_id'].unique())

Map_roof_pv.shape[0] 
Map_roof_pv['xtf_id'].nunique()
(len(Map_roof_pv["xtf_id"].unique()) - pv["xtf_id"].nunique())/ pv["xtf_id"].nunique()