import os as os
import pandas as pd
import numpy as np
import geopandas as gpd
import datetime
import winsound

from datetime import datetime
# from ..functions import chapter_to_logfile, checkpoint_to_logfile

# SETTIGNS --------------------------------------------------------------------
def spatial_toparquet(script_run_on_server_def = 0):

    # SETUP --------------------------------------------------------------------
    if script_run_on_server_def == 0:
        wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
        data_path = f'{wd_path}_data'
    elif script_run_on_server_def == 1:
        wd_path = "D:/RaulHochuli_inuse/OptimalPV_RH"
        data_path = f'{wd_path}_data'

    # set directory if necessary
    if not os.path.exists(f'{data_path}/spatial_intersection_by_gm'):
        os.makedirs(f'{data_path}/spatial_intersection_by_gm')

    # create log file for checkpoint comments
    log_file_name = f'{data_path}/spatial_intersection_by_gm/spatial_data_toparquet_by_gm_log.txt'
    with open(log_file_name, 'w') as log_file:
        log_file.write(f'\n')
    chapter_to_logfile('start spatial_data_toparquet_by_gm.py', log_file_name = log_file_name)


    # import shapes
    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    checkpoint_to_logfile('import gm_shp', log_file_name = log_file_name, n_tabs = 1)
    roof_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
    checkpoint_to_logfile('import roof_kat', log_file_name = log_file_name, n_tabs = 1)
    # faca_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-fassaden_2056.gdb/SOLKAT_FASS_20230221.gdb', layer ='SOLKAT_CH_FASS')
    # checkpoint_to_logfile('import faca_kat', log_file_name = log_file_name, n_tabs = 1)
    bldng_reg = gpd.read_file(f'{data_path}/input/GebWohnRegister.CH/buildings.geojson')
    checkpoint_to_logfile('import bldng_reg', log_file_name = log_file_name, n_tabs = 1)
    heatcool_dem = gpd.read_file(f'{data_path}/input/heating_cooling_demand.gpkg/fernwaerme-nachfrage_wohn_dienstleistungsgebaeude_2056.gpkg', layer= 'HOMEANDSERVICES')
    checkpoint_to_logfile('import heatcool_dem', log_file_name = log_file_name, n_tabs = 1)
    elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
    checkpoint_to_logfile('import elec_prod', log_file_name = log_file_name, n_tabs = 1)
    pv = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()
    checkpoint_to_logfile('subset for pv', log_file_name = log_file_name, n_tabs = 1)

    # drop unnecessary columns

    checkpoint_to_logfile('\n\n', log_file_name = log_file_name, n_tabs = 10)
    # sjoin to gm_shp
    roof_kat.set_crs(gm_shp.crs, allow_override=True, inplace=True)
    # faca_kat.set_crs(gm_shp.crs, allow_override=True, inplace=True)
    bldng_reg.set_crs(gm_shp.crs, allow_override=True, inplace=True)
    heatcool_dem.set_crs(gm_shp.crs, allow_override=True, inplace=True)
    pv.set_crs(gm_shp.crs, allow_override=True, inplace=True)


    roof_kat = gpd.sjoin(roof_kat, gm_shp, how="left", predicate="within")
    checkpoint_to_logfile('sjoin roof_kat', log_file_name = log_file_name, n_tabs = 1)
    # faca_kat = gpd.sjoin(faca_kat, gm_shp, how="left", predicate="within")
    # checkpoint_to_logfile('sjoin faca_kat', log_file_name = log_file_name, n_tabs = 1)
    bldng_reg = gpd.sjoin(bldng_reg, gm_shp, how="left", predicate="within")
    checkpoint_to_logfile('sjoin bldng_reg', log_file_name = log_file_name, n_tabs = 1)
    heatcool_dem = gpd.sjoin(heatcool_dem, gm_shp, how="left", predicate="within")
    checkpoint_to_logfile('sjoin heatcool_dem', log_file_name = log_file_name, n_tabs = 1)
    pv = gpd.sjoin(pv, gm_shp, how="left", predicate="within")
    checkpoint_to_logfile('sjoin pv', log_file_name = log_file_name, n_tabs = 1)

    roof_kat.info()
    bldng_reg.info()
    heatcool_dem.info()
    pv.info()

    # # drop unnecessary columns
    # drop_gm_cols = ['index_right', 'UUID', 'DATUM_AEND', 'DATUM_ERST', 'ERSTELL_J', 'ERSTELL_M', 'REVISION_J', 'REVISION_M',
    #                 'GRUND_AEND', 'HERKUNFT', 'HERKUNFT_J', 'HERKUNFT_M', 'OBJEKTART', 'BEZIRKSNUM', 'SEE_FLAECH', 'REVISION_Q', 
    #                 'NAME', 'KANTONSNUM', 'ICC', 'EINWOHNERZ', 'HIST_NR', 'GEM_TEIL', 'GEM_FLAECH', 'SHN']
    # roof_kat.drop(columns = drop_gm_cols, axis = 1, inplace = True)
    # bldng_reg.drop(columns = drop_gm_cols, axis = 1, inplace = True)
    # heatcool_dem.drop(columns = drop_gm_cols, axis = 1, inplace = True)
    # pv.drop(columns = drop_gm_cols, axis = 1, inplace = True)
    # checkpoint_to_logfile('drop unnecessary columns', log_file_name = log_file_name, n_tabs = 1)

    # export to parquet
    roof_kat.to_parquet(f'{data_path}/spatial_intersection_by_gm/roof_kat_by_gm.parquet')
    checkpoint_to_logfile('export roof_kat.parquet', log_file_name = log_file_name, n_tabs = 1)
    # faca_kat.to_parquet(f'{data_path}/spatial_intersection_by_gm/faca_kat_by_gm.parquet')
    # checkpoint_to_logfile('export faca_kat.parquet', log_file_name = log_file_name, n_tabs = 1)
    bldng_reg.to_parquet(f'{data_path}/spatial_intersection_by_gm/bldng_reg_by_gm.parquet')
    checkpoint_to_logfile('export bldng_reg.parquet', log_file_name = log_file_name, n_tabs = 1)
    heatcool_dem.to_parquet(f'{data_path}/spatial_intersection_by_gm/heatcool_dem_by_gm.parquet')
    checkpoint_to_logfile('export heatcool_dem.parquet', log_file_name = log_file_name, n_tabs = 1)
    pv.to_parquet(f'{data_path}/spatial_intersection_by_gm/pv_by_gm.parquet')
    checkpoint_to_logfile('export pv.parquet', log_file_name = log_file_name, n_tabs = 1)
    gm_shp.to_parquet(f'{data_path}/spatial_intersection_by_gm/gm_shp.parquet')
    checkpoint_to_logfile('export gm_shp.parquet', log_file_name = log_file_name, n_tabs = 1)

    chapter_to_logfile('end spatial_data_toparquet_by_gm.py', log_file_name = log_file_name)