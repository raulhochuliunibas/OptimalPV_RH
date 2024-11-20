import sys
import os as os
import numpy as np
import pandas as pd
import geopandas as gpd
import winsound
import json
import plotly.express as px
import glob
import shutil

from datetime import datetime
from shapely.geometry import Point

sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile, get_bfs_from_ktnr



# ------------------------------------------------------------------------------------------------------
# SPLIT DATA AND GEOMETRY
# ------------------------------------------------------------------------------------------------------
def split_data_and_geometry(
        dataagg_settings_def,):
    """
    Split data and geometry for all geo data frames for faster importing later on
    """
    
    # import settings + setup -------------------
    script_run_on_server_def = dataagg_settings_def['script_run_on_server']
    bfs_number_def = dataagg_settings_def['bfs_numbers']
    year_range_def = dataagg_settings_def['year_range']
    smaller_import_def = dataagg_settings_def['smaller_import']
    show_debug_prints_def = dataagg_settings_def['show_debug_prints']
    log_file_name_def = dataagg_settings_def['log_file_name']
    wd_path_def = dataagg_settings_def['wd_path']
    data_path_def = dataagg_settings_def['data_path']

    gwr_selection_specs_def = dataagg_settings_def['gwr_selection_specs']
    solkat_selection_specs_def = dataagg_settings_def['solkat_selection_specs']
    print_to_logfile('run function: split_data_and_geometry.py', log_file_name_def)

    # create folder if not exists
    if not os.path.exists(f'{data_path_def}/split_data_geometry'):
        os.makedirs(f'{data_path_def}/split_data_geometry')


    # IMPORT DATA --------------------------------------
    gm_shp_gdf = gpd.read_file(f'{data_path_def}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    
    # Function: Merge GM BFS numbers to spatial data sources
    def attach_bfs_to_spatial_data(gdf, gm_shp_gdf, keep_cols = ['BFS_NUMMER', 'geometry' ]):
        """
        Function to attach BFS numbers to spatial data sources
        """
        gdf.set_crs(gm_shp_gdf.crs, allow_override=True, inplace=True)
        gdf = gpd.sjoin(gdf, gm_shp_gdf, how="left", predicate="within")
        checkpoint_to_logfile('sjoin complete', log_file_name_def = log_file_name_def, n_tabs_def = 6, show_debug_prints_def = show_debug_prints_def)
        dele_cols = ['index_right'] + [col for col in gm_shp_gdf.columns if col not in keep_cols]
        gdf.drop(columns = dele_cols, inplace = True)
        if 'BFS_NUMMER' in gdf.columns:
            # transform BFS_NUMMER to str, np.nan to ''
            gdf['BFS_NUMMER'] = gdf['BFS_NUMMER'].apply(lambda x: '' if pd.isna(x) else str(int(x)))

        return gdf 
    
    # PV -------------------

    # BOOKMARK!!
    """
        run function: split_data_and_geometry.py
        c:\Models\OptimalPV_RH\.venv\Lib\site-packages\pyogrio\geopandas.py:261: UserWarning:

        More than one layer found in 'ch.bfe.elektrizitaetsproduktionsanlagen.gpkg': 'ElectricityProductionPlant' (default), 'MainCategoryCatalogue', 'SubCategoryCatalogue', 'PlantCategoryCatalogue', 'OrientationCatalogue', 'PlantDetail'. Specify layer parameter to avoid this warning.
    """
    if not smaller_import_def:
        elec_prod_gdf = gpd.read_file(f'{data_path_def}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
        pv_all_gdf = elec_prod_gdf[elec_prod_gdf['SubCategory'] == 'subcat_2'].copy()
    elif smaller_import_def:
        elec_prod_gdf = gpd.read_file(f'{data_path_def}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg', rows = 7000)
        pv_all_gdf = elec_prod_gdf[elec_prod_gdf['SubCategory'] == 'subcat_2'].copy()
    checkpoint_to_logfile(f'import pv, {pv_all_gdf.shape[0]} rows (smaller_import: {smaller_import_def})', log_file_name_def = log_file_name_def, n_tabs_def = 1, show_debug_prints_def = show_debug_prints_def)

    pv_all_gdf = attach_bfs_to_spatial_data(pv_all_gdf, gm_shp_gdf)
    pv_all_gdf.set_crs("EPSG:2056", allow_override=True, inplace=True)

    # split + export
    checkpoint_to_logfile(f'check unique identifier pv: {pv_all_gdf["xtf_id"].nunique()} xtf unique, {pv_all_gdf.shape[0]} rows', log_file_name_def = log_file_name_def, n_tabs_def = 0, show_debug_prints_def = show_debug_prints_def)
    pv_pq = pv_all_gdf.loc[:,pv_all_gdf.columns !='geometry'].copy()
    pv_geo = pv_all_gdf.loc[:,['xtf_id', 'BFS_NUMMER', 'geometry']].copy()

    pv_pq.to_parquet(f'{data_path_def}/split_data_geometry/pv_pq.parquet')
    checkpoint_to_logfile(f'exported pv_pq.parquet', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

    with open(f'{data_path_def}/split_data_geometry/pv_geo.geojson', 'w') as f:
        f.write(pv_geo.to_json())
    checkpoint_to_logfile(f'exported pv_geo.geojson', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)



    # SOLKAT -------------------
    if not smaller_import_def:  
        solkat_all_gdf = gpd.read_file(f'{data_path_def}/input\solarenergie-eignung-daecher_2056.gpkg\SOLKAT_DACH.gpkg', layer ='SOLKAT_CH_DACH')
    elif smaller_import_def:
        solkat_all_gdf = gpd.read_file(f'{data_path_def}/input\solarenergie-eignung-daecher_2056.gpkg\SOLKAT_DACH.gpkg', layer ='SOLKAT_CH_DACH', rows = 1000)
    checkpoint_to_logfile(f'import solkat, {solkat_all_gdf.shape[0]} rows (smaller_import: {smaller_import_def})', log_file_name_def, 2, show_debug_prints_def)

    solkat_all_gdf = attach_bfs_to_spatial_data(solkat_all_gdf, gm_shp_gdf)
    solkat_all_gdf.set_crs("EPSG:2056", allow_override=True, inplace=True)

    # split + export
    checkpoint_to_logfile(f'check unique identifier solkat: {solkat_all_gdf["DF_UID"].nunique()} DF_UID unique, {solkat_all_gdf.shape[0]} rows', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
    solkat_pq = solkat_all_gdf.loc[:,solkat_all_gdf.columns !='geometry'].copy()
    solkat_geo = solkat_all_gdf.loc[:,['DF_UID', 'BFS_NUMMER', 'geometry']].copy()

    solkat_pq.to_parquet(f'{data_path_def}/split_data_geometry/solkat_pq.parquet')
    checkpoint_to_logfile(f'exported solkat_pq.parquet', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)

    with open(f'{data_path_def}/split_data_geometry/solkat_geo.geojson', 'w') as f:
        f.write(solkat_geo.to_json())
    checkpoint_to_logfile(f'exported solkat_geo.geojson', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)


    # subset for BSBLSO case -------------------
    bsblso_bfs_numbers = get_bfs_from_ktnr([11, 12, 13], data_path_def, log_file_name_def)

    pv_bsblso_geo = pv_geo.loc[pv_geo['BFS_NUMMER'].isin(bsblso_bfs_numbers)].copy()
    if pv_bsblso_geo.shape[0] > 0:
        with open (f'{data_path_def}/split_data_geometry/pv_bsblso_geo.geojson', 'w') as f:
            f.write(pv_bsblso_geo.to_json())
        checkpoint_to_logfile(f'exported pv_bsblso_geo.geojson', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)
                    

    solkat_bsblso_geo = solkat_geo.loc[solkat_geo['BFS_NUMMER'].isin(bsblso_bfs_numbers)].copy()
    if solkat_bsblso_geo.shape[0] > 0:
        with open (f'{data_path_def}/split_data_geometry/solkat_bsblso_geo.geojson', 'w') as f:
            f.write(solkat_bsblso_geo.to_json())
        checkpoint_to_logfile(f'exported solkat_bsblso_geo.geojson', log_file_name_def = log_file_name_def, n_tabs_def = 5, show_debug_prints_def = show_debug_prints_def)


    # Copy Log File to split_data_geometry folder
    if os.path.exists(log_file_name_def):
        shutil.copy(log_file_name_def, f'{data_path_def}/split_data_geometry/split_data_geometry_logfile.txt')