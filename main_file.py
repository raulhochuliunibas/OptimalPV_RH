import os as os
import functions
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pyogrio
import winsound

from functions import chapter_to_logfile, checkpoint_to_logfile
from datetime import datetime
from shapely.ops import unary_union

# still uncertain if this is needed
import warnings


# pre run settings -----------------------------------------------------------------------------------------------
script_run_on_server = 0          # 0 = script is running on laptop, 1 = script is running on server
subsample_faster_run = 0          # 0 = run on all data, 1 = run on subset of data for faster run
create_data_subsample = 1         # 0 = do not create data subsample, 1 = create data subsample


# ----------------------------------------------------------------------------------------------------------------
# Setup + Import 
# ----------------------------------------------------------------------------------------------------------------


# pre setup + working directory ----------------------------------------------------------------------------------
if script_run_on_server == 0:
     winsound.Beep(840,  100)

if script_run_on_server == 0:
     wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
elif script_run_on_server == 1:
     wd_path = "D:\OptimalPV_RH"         # path for server directory

data_path = f'{wd_path}_data'
os.chdir(wd_path)

# create log file for checkpoint comments
timer = datetime.now()
with open(f'log_file.txt', 'w') as log_file:
        log_file.write(f' \n')
chapter_to_logfile('started running main_file.py')


# import geo referenced data -------------------------------------------------------------------------------------

# load administrative shapes
kt_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_KANTONSGEBIET')
#kt_shp.set_crs("EPSG:4326", allow_override=True, inplace=True)
gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
checkpoint_to_logfile(f'finished loading administrative shp', n_tabs = 2)

if subsample_faster_run == 0:
    # load solar kataster shapes
    roof_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
    checkpoint_to_logfile(f'finished loading roof solar kataster shp', n_tabs = 1)
    faca_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-fassaden_2056.gdb/SOLKAT_FASS_20230221.gdb', layer ='SOLKAT_CH_FASS')
    checkpoint_to_logfile(f'finished loading facade solar kataster shp', n_tabs = 1)

    # load building register indicating residential or industrial use
    bldng_reg = gpd.read_file(f'{data_path}/input/GebWohnRegister.CH/buildings.geojson')
    checkpoint_to_logfile(f'finished loading building register pt', n_tabs = 2)

    # load heating / cooling demand raster 150x150m
    heatcool_dem = gpd.read_file(f'{data_path}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES')
    checkpoint_to_logfile(f'finished loading heat & cool demand pt', n_tabs = 1)

    # load pv installation points
    pv = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen', layer = 'subcat_2_pv')
    checkpoint_to_logfile(f'finished loading pv installation pt', n_tabs = 2) 
    
    # check if all CRS are compatible
    kt_shp.crs == gm_shp.crs == roof_kat.crs == faca_kat.crs == bldng_reg.crs == heatcool_dem.crs == pv.crs
    gm_shp.set_crs(kt_shp.crs, allow_override=True, inplace=True)
    roof_kat.set_crs(kt_shp.crs, allow_override=True, inplace=True)
    faca_kat.set_crs(kt_shp.crs, allow_override=True, inplace=True)
    bldng_reg.set_crs(kt_shp.crs, allow_override=True, inplace=True)
    heatcool_dem.set_crs(kt_shp.crs, allow_override=True, inplace=True)
    pv.set_crs(kt_shp.crs, allow_override=True, inplace=True)
    
    # export subsamples for faster run --------------------------------------------------------------------------------
    if create_data_subsample == 1:
          kt_number_sub = [16,]
          gm_number_sub = [3851, 3901, 4761, 1083, 4001, 1061, 2829, 4042 ]
          kt_shp.loc[kt_shp['KANTONSNUM'].isin(kt_number_sub), ['NAME', 'KANTONSNUM']] 
          gm_shp.loc[gm_shp['BFS_NUMMER'].isin(gm_number_sub), ['NAME', 'BFS_NUMMER']]

          # create folder for subsample shapes selected, remove old files if they exist
          if not os.path.exists(f'{data_path}/subsample_faster_run'):
               os.makedirs(f'{data_path}/subsample_faster_run')
          elif os.path.exists(f'{data_path}/subsample_faster_run'):
               for file in os.listdir(f'{data_path}/subsample_faster_run'):
                    os.remove(f'{data_path}/subsample_faster_run/{file}')

          # create subsample shapes selected
          checkpoint_to_logfile(f'\tstart creating subsamples', n_tabs = 3)
          kt_shp_sub = kt_shp.loc[kt_shp["KANTONSNUM"].isin(kt_number_sub),].copy()
          gm_shp_sub = gm_shp.loc[gm_shp['BFS_NUMMER'].isin(gm_number_sub),].copy()
          checkpoint_to_logfile(f'\t * finished subsetting admin shp', n_tabs = 2)
          
          roof_kat_sub = gpd.sjoin(roof_kat, gm_shp_sub, how="inner", op="within")
          checkpoint_to_logfile(f'\t * finished subsetting roof_kat', n_tabs = 3)
          
          faca_kat_sub =      gpd.sjoin(faca_kat, gm_shp_sub, how="inner", op="within")
          checkpoint_to_logfile(f'\t * finished subsetting faca_kat', n_tabs = 3)
          bldng_reg_sub =     gpd.sjoin(bldng_reg, gm_shp_sub, how="inner", op="within")
          checkpoint_to_logfile(f'\t * finished subsetting bldng_reg', n_tabs = 2)
          heatcool_dem_sub =  gpd.sjoin(heatcool_dem, gm_shp_sub, how="inner", op="within")
          checkpoint_to_logfile(f'\t * finished subsetting heatcool_dem', n_tabs = 2)
          pv_sub =             gpd.sjoin(pv, gm_shp_sub, how="inner", op="within")   
          checkpoint_to_logfile(f'\t * finished subsetting pv', n_tabs = 4)

          # export subsample to shape files
          checkpoint_to_logfile(f'\tstart exporting subsample shapes', n_tabs = 2)
          gm_shp_sub.to_file(f'{data_path}/subsample_faster_run/gm_shp_sub.shp')
          checkpoint_to_logfile(f'\t\t * finished exporting gm_shp_sub', n_tabs = 3)
          roof_kat_sub.to_file(f'{data_path}/subsample_faster_run/roof_kat_sub.shp')
          checkpoint_to_logfile(f'\t\t * finished exporting roof_kat_sub', n_tabs = 3)
          faca_kat_sub.to_file(f'{data_path}/subsample_faster_run/faca_kat_sub.shp')
          checkpoint_to_logfile(f'\t\t * finished exporting faca_kat_sub', n_tabs = 3)
          bldng_reg_sub.to_file(f'{data_path}/subsample_faster_run/bldng_reg_sub.shp')
          checkpoint_to_logfile(f'\t\t * finished exporting bldng_reg_sub', n_tabs = 3)
          heatcool_dem_sub.to_file(f'{data_path}/subsample_faster_run/heatcool_dem_sub.shp')
          checkpoint_to_logfile(f'\t\t * finished exporting heatcool_dem_sub', n_tabs = 3)
          pv_sub.to_file(f'{data_path}/subsample_faster_run/pv_sub.shp')
          checkpoint_to_logfile(f'\t\t * finished exporting pv_sub', n_tabs = 3)

          # export subsample to geopackage
          gm_shp_sub.to_file(f'{data_path}/subsample_faster_run/0_Subset_OptPV_RH_data.gpkg', layer='gm_shp', driver="GPKG")
          # roof_kat_sub.to_file(f'{data_path}/subsample_faster_run/0_Subset_OptPV_RH_data.gpkg', layer='roof_kat', driver="GPKG", mode = 'a')
          # faca_kat_sub.to_file(f'{data_path}/subsample_faster_run/0_Subset_OptPV_RH_data.gpkg', layer='faca_kat', driver="GPKG", mode = 'a')
          # bldng_reg_sub.to_file(f'{data_path}/subsample_faster_run/0_Subset_OptPV_RH_data.gpkg', layer='bldng_reg', driver="GPKG", mode = 'a')
          # heatcool_dem_sub.to_file(f'{data_path}/subsample_faster_run/0_Subset_OptPV_RH_data.gpkg', layer='heatcool_dem', driver="GPKG", mode = 'a')
          # pv_sub.to_file(f'{data_path}/subsample_faster_run/0_Subset_OptPV_RH_data.gpkg', layer='pv', driver="GPKG", mode = 'a')
      


elif subsample_faster_run == 1:
     checkpoint_to_logfile(f'using SUBSAMPLE for faster run', n_tabs = 1)

     #load subset shapes
     os.listdir(f'{data_path}/subsample_faster_run')   
     roof_kat = gpd.read_file(f'{data_path}/subsample_faster_run/roof_kat_sub.shp')
     checkpoint_to_logfile(f'finished loading roof solar kataster shp', n_tabs = 1)
     faca_kat = gpd.read_file(f'{data_path}/subsample_faster_run/faca_kat_sub.shp')
     checkpoint_to_logfile(f'finished loading facade solar kataster shp', n_tabs = 1)
     bldng_reg = gpd.read_file(f'{data_path}/subsample_faster_run/bldng_reg_sub.shp')
     checkpoint_to_logfile(f'finished loading building register pt', n_tabs = 2)
     heatcool_dem = gpd.read_file(f'{data_path}/subsample_faster_run/heatcool_dem_sub.shp')
     checkpoint_to_logfile(f'finished loading heat & cool demand pt', n_tabs = 1)
     pv = gpd.read_file(f'{data_path}/subsample_faster_run/pv_sub.shp')
     checkpoint_to_logfile(f'finished loading pv installation pt', n_tabs = 2)

     
# import regular, nonGIS data --------------------------------------------------------------------------------------------
# dict_elec_prod_dispatch = {'Week':['05.01.2022', '12.01.2022', '19.01.2022', '26.01.2022', '02.02.2022', '09.02.2022', '16.02.2022', '23.02.2022', '02.03.2022', '09.03.2022', '16.03.2022', '23.03.2022', '30.03.2022', '06.04.2022', '13.04.2022', '20.04.2022', '27.04.2022', '04.05.2022', '11.05.2022', '18.05.2022', '25.05.2022', '01.06.2022', '08.06.2022', '15.06.2022', '22.06.2022', '29.06.2022','06.07.2022', '13.07.2022', '20.07.2022', '27.07.2022', '03.08.2022', '10.08.2022', '17.08.2022', '24.08.2022', '31.08.2022', '07.09.2022', '14.09.2022', '21.09.2022', '28.09.2022', '05.10.2022', '12.10.2022', '19.10.2022', '26.10.2022', '02.11.2022', '09.11.2022', '16.11.2022', '23.11.2022', '30.11.2022', '07.12.2022', '14.12.2022', '21.12.2022', '28.12.2022'],
#                          'consumption_Gwh':[195.2, 236.1, 216.7, 214.7, 218.6, 205.4, 213.2, 198.0, 196.9, 207.3, 193.0, 194.1, 191.4, 191.6, 170.2, 159.3, 167.3, 173.2, 164.4, 150.7, 158.1, 163.3, 161.3, 161.4, 175.0, 159.3, 150.1, 158.7, 144.4, 152.8, 149.6, 156.8, 144.2, 158.8, 171.6, 162.0, 172.0, 164.6, 180.7, 162.7, 173.5, 168.9, 171.1, 173.3, 190.3, 182.4, 199.0, 204.6, 220.4, 201.1, 208.0, 171.7]} 
# elec_dem_2022 = pd.DataFrame(dict_elec_prod_dispatch)
checkpoint_to_logfile(f'finished loading electricity demand 2022(non-standardized for other years)', n_tabs = 1)



# ----------------------------------------------------------------------------------------------------------------
# Create Roof Based Dataframe - Aggregate Roof Parts at House Level 
# ----------------------------------------------------------------------------------------------------------------
chapter_to_logfile('aggregate roof parts at house level')
if script_run_on_server == 0: 
     winsound.Beep(840,  100)
     winsound.Beep(840,  100)



# ----------------------------------------------------------------------------------------------------------------
# END 
# ----------------------------------------------------------------------------------------------------------------
chapter_to_logfile('END of main_file.py')
if script_run_on_server == 0:
     winsound.Beep(400, 100)
     winsound.Beep(400, 100)
     winsound.Beep(400, 500)



