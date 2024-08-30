# -----------------------------------------------------------------------------
# visualizations_MASTER.py 
# -----------------------------------------------------------------------------
# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024
# > description: 

visual_settings = {
    'wd_path_laptop': 'C:/Models/OptimalPV_RH',

    'general_specs': {
    }, 
    'plot_TOPOinterim': {
        'plot_show': True, 
        'plot_save': True,
        'import_path': 'C:\Models\OptimalPV_RH_data\output\pvalloc_BSBLSO_wrkn_prgrss_20240826_9h_BREAK',
        'preprep_path': 'C:/Models/OptimalPV_RH_data/output/preprep_BSBLSO_15to23_20240821_02h',
        'title': 'TOPO interim development',

    }, 
    'plot_SOLKAT_max_Xpartitions': {
        'plot_show': True, 
        'plot_save': True,
        'import_path': 'preprep_BSBLSO_18to22_20240826_22h',
        'preprep_path': '',
        'title': 'TOPO interim development',

    },
}

# PACKAGES =============================================================
if True:
    import sys
    sys.path.append(visual_settings['wd_path_laptop']) 


    # external packages
    import os as os
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import dask.dataframe as dd
    import glob
    import shutil
    import winsound
    import subprocess
    import pprint
    import json 
    
    from datetime import datetime
    from pprint import pformat


    
    import plotly.express as px
    import plotly.graph_objects as go


    # SETUP ================================================================
    # set working directory
    wd_path = visual_settings['wd_path_laptop']
    data_path = f'{wd_path}_data'

    # create directory + log file
    visual_path = f'{data_path}/output/visualizations'
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

    log_name = f'{data_path}/output/visual_log.txt'


    # general plot settings


# plot: TOPO interm development ========================================
def plot_TOPOinterim(general_specs, plot_specs):
    # setup
    title = plot_specs['title']
    import_path = plot_specs['import_path']
    preprep_path = plot_specs['preprep_path']

    # import data
    topo = json.load(open(f'{import_path}/topo_egid.json', 'r'))

    keys = topo.keys()
    
    # gwr = pd.read_parquet(f'{prepre}'                        
# plot_TOPOinterim(visual_settings['general_specs'], visual_settings['plot_TOPOinterim'])

# plot: MAP of SOLKAT > 100 partitions ================================
def plot_SOLKAT_max_Xpartitions(general_specs, plot_specs):
    # setup
    title = plot_specs['title']
    import_path = plot_specs['import_path']

    # import data
    solkat_pq = pd.read_parquet(f'{data_path}/split_data_geometry/solkat_pq.parquet')
    solkat_geo = gpd.read_file(f'{data_path}/split_data_geometry/solkat_bsblso_geo.geojson', rows = 50)
    
    solkat_gdf = gpd.GeoDataFrame(solkat_pq.merge(solkat_geo[['DF_UID', 'geometry']], on='DF_UID'), geometry='geometry')

    solkat_gdf['EGID'] = solkat_gdf['GWR_EGID'].fillna(0).astype(int).astype(str)
    solkat_gdf.drop(columns=[col for col in solkat_gdf.columns if 'DATUM' in col], inplace=True)

    egid_counts = solkat_gdf['EGID'].value_counts()
    solkat_gdf['EGID_frequency'] = solkat_gdf['EGID'].map(egid_counts)

    def assign_color(frequency):
        if frequency > 400:
            return 'purple'
        elif frequency > 300:
            return 'red'
        elif frequency > 200:
            return 'orange'
        elif frequency > 100:
            return 'yellow'
        else:
            return 'blue'

    solkat_gdf['color'] = solkat_gdf['EGID_frequency'].apply(assign_color)

    # Convert the GeoDataFrame to GeoJSON
    geojson = solkat_gdf.to_json()

    # Plot the GeoDataFrame using Plotly
    fig = px.choropleth_mapbox(
        solkat_gdf,
        geojson=geojson,
        locations=solkat_gdf.index,
        color='EGID_frequency',
        color_continuous_scale=["blue", "yellow", "orange", "red", "purple"],
        mapbox_style="carto-positron",
        zoom=10,
        center={"lat": solkat_gdf.geometry.centroid.y.mean(), "lon": solkat_gdf.geometry.centroid.x.mean()},
        opacity=0.5,
        labels={'EGID_frequency': 'Number of Partitions'}
    )

    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()
plot_SOLKAT_max_Xpartitions(visual_settings['general_specs'], visual_settings['plot_SOLKAT_max_Xpartitions'])


# END OF SCRIPT ========================================================
winsound.Beep(2000, 100)
winsound.Beep(2000, 100)

