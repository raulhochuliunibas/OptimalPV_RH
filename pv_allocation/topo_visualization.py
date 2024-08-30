import sys
import os as os
import numpy as np
import pandas as pd
import geopandas as gpd
import json
import itertools
import math
import glob
import plotly.graph_objs as go
import plotly.offline as pyo


from pyarrow.parquet import ParquetFile
import pyarrow as pa

# own functions 
sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile


# ------------------------------------------------------------------------------------------------------
# visualization of PV topology
# ------------------------------------------------------------------------------------------------------
def create_map_of_topology(
        pvalloc_settings, 
        topo_df_func ):
    
    # setup -----------------------------------------------------
    name_dir_import_def = pvalloc_settings['name_dir_import']
    data_path_def = pvalloc_settings['data_path']
    log_file_name_def = pvalloc_settings['log_file_name']

    topo_df= topo_df_func 


    # import geo data -----------------------------------------------------
    if pvalloc_settings['fast_debug_run']:
        solkat_gdf = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/solkat_gdf.geojson', rows=50)
        gwr_gdf = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/gwr_gdf.geojson', rows = 50)
        pv_gdf = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/pv_gdf.geojson', rows = 50)
    else:
        solkat_gdf = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/solkat_gdf.geojson')
        gwr_gdf = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/gwr_gdf.geojson')
        pv_gdf = gpd.read_file(f'{data_path_def}/output/{name_dir_import_def}/pv_gdf.geojson')

    # transformations
    pv_gdf['xtf_id'] = pv_gdf['xtf_id'].astype(int).replace(np.nan, "").astype(str)
    solkat_gdf['DF_UID'] = solkat_gdf['DF_UID'].astype(int).replace(np.nan, "").astype(str)
    solkat_gdf.rename(columns={'DF_UID': 'df_uid'}, inplace=True)


    # subset gwr + pv -----------------------------------------------------
    gwr_gdf_in_topo = gwr_gdf[gwr_gdf['EGID'].isin(topo_df['EGID'].unique())].copy()
    pv_gdf_in_topo = pv_gdf[pv_gdf['xtf_id'].isin(topo_df['pvid'].unique())].copy()

    topo_gdf = topo_df.merge(solkat_gdf[['df_uid', 'geometry']], on='df_uid', how='left')
    topo_gdf = gpd.GeoDataFrame(topo_gdf, crs='EPSG:2056', geometry='geometry')

    # export to shp -----------------------------------------------------
    if not os.path.exists(f'{data_path_def}/output/pvalloc_run/topo_spatial_data'):
        os.makedirs(f'{data_path_def}/output/pvalloc_run/topo_spatial_data')

    gwr_gdf.to_file(f'{data_path_def}/output/pvalloc_run/topo_spatial_data/gwr_gdf.shp')
    pv_gdf.to_file(f'{data_path_def}/output/pvalloc_run/topo_spatial_data/pv_gdf.shp')
    topo_gdf.to_file(f'{data_path_def}/output/pvalloc_run/topo_spatial_data/topo_gdf.shp')


    # subset to > max n partitions -----------------------------------------------------
    max_partitions = pvalloc_settings['gwr_selection_specs']['solkat_max_n_partitions']
    topo_above_npart_gdf = topo_gdf.copy()
    counts = topo_above_npart_gdf['EGID'].value_counts()
    topo_above_npart_gdf['EGID_count'] = topo_above_npart_gdf['EGID'].map(counts)
    topo_above_npart_gdf = topo_above_npart_gdf[topo_above_npart_gdf['EGID_count'] > max_partitions]

    gwr_above_npart_gdf = gwr_gdf[gwr_gdf['EGID'].isin(topo_above_npart_gdf['EGID'].unique())].copy()

    # export to shp -----------------------------------------------------
    topo_above_npart_gdf.to_file(f'{data_path_def}/output/pvalloc_run/topo_spatial_data/topo_above_{max_partitions}_npart_gdf.shp')
    gwr_above_npart_gdf.to_file(f'{data_path_def}/output/pvalloc_run/topo_spatial_data/gwr_above_{max_partitions}_npart_gdf.shp')

