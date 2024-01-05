import os as os
import pandas as pd
import geopandas as gpd
import glob
import shutil
import winsound
import functions

from functions import chapter_to_logfile, checkpoint_to_logfile
from data_aggregation.local_data_import_aggregation import import_aggregate_data
from data_aggregation.spatial_data_toparquet_by_gm import spatial_toparquet

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
    Map_roof_pv = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/Map_roof_pv.parquet')
