import os as os
import sys

import os as os
import pandas as pd
import geopandas as gpd
import numpy as np
import json 
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import copy
import glob
import matplotlib.pyplot as plt
import winsound
import itertools
import shutil
import scipy.stats as stats

from datetime import datetime
from pprint import pformat
from shapely.geometry import Polygon, MultiPolygon
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

sys.path.append('..')
from auxiliary_functions import *
from .plot_auxiliary_functions import *


# ------------------------------------------------------------------------------------------------------
# PLOT INDIVIDUAL MAP of TOPOLOGY with EGID
# ------------------------------------------------------------------------------------------------------

def plot(pvalloc_scen_list, 
         visual_settings, 
         wd_path, 
         data_path, 
         log_name,
         ):
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]
    plot_show = visual_settings['plot_show']

    default_map_zoom = visual_settings['default_map_zoom']
    default_map_center = visual_settings['default_map_center']


    if visual_settings['plot_ind_map_topo_egid'][0]:
        map_topo_egid_specs = visual_settings['plot_ind_map_topo_egid_specs']
        checkpoint_to_logfile(f'plot_ind_map_topo_egid', log_name)

        for i_scen, scen in enumerate(scen_dir_export_list):
            pvalloc_scen = pvalloc_scen_list[i_scen]
            
            # get pvinst_gdf ----------------
            if True: 
                mc_data_path = glob.glob(f'{data_path}/output/{scen}/{visual_settings["MC_subdir_for_plot"]}')[0] # take first path if multiple apply, so code can still run properly
                
                # import 
                gwr_gdf = gpd.read_file(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/gwr_gdf.geojson')
                gm_gdf = gpd.read_file(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/gm_shp_gdf.geojson')                                         

                topo  = json.load(open(f'{mc_data_path}/topo_egid.json', 'r'))
                egid_list, inst_TF_list, info_source_list, BeginOp_list, TotalPower_list, bfs_list= [], [], [], [], [], []
                gklas_list, node_list, demand_type_list, pvtarif_list, elecpri_list, elecpri_info_list = [], [], [], [], [], []

                for k,v, in topo.items():
                    egid_list.append(k)
                    inst_TF_list.append(v['pv_inst']['inst_TF'])
                    info_source_list.append(v['pv_inst']['info_source'])
                    BeginOp_list.append(v['pv_inst']['BeginOp'])
                    TotalPower_list.append(v['pv_inst']['TotalPower'])
                    bfs_list.append(v['gwr_info']['bfs'])

                    gklas_list.append(v['gwr_info']['gklas'])
                    node_list.append(v['grid_node'])
                    demand_type_list.append(v['demand_type'])
                    pvtarif_list.append(v['pvtarif_Rp_kWh'])
                    elecpri_list.append(v['elecpri_Rp_kWh'])
                    elecpri_info_list.append(v['elecpri_info'])

                pvinst_df = pd.DataFrame({'EGID': egid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list,
                                        'BeginOp': BeginOp_list, 'TotalPower': TotalPower_list, 'bfs': bfs_list, 
                                        'gklas': gklas_list, 'node': node_list, 'demand_type': demand_type_list,
                                        'pvtarif': pvtarif_list, 'elecpri': elecpri_list, 'elecpri_info': elecpri_info_list })
                
                pvinst_df = pvinst_df.merge(gwr_gdf[['geometry', 'EGID']], on='EGID', how='left')
                pvinst_gdf = gpd.GeoDataFrame(pvinst_df, crs='EPSG:2056', geometry='geometry')
                firstkey_topo = topo[list(topo.keys())[0]]

            # base map ----------------
            if True: 
                # setup
                scen_data_path = f'{data_path}/output/{scen}'
                T0_prediction = pvalloc_scen['T0_prediction']
                
                # transformations
                gm_gdf['BFS_NUMMER'] = gm_gdf['BFS_NUMMER'].astype(str)
                gm_gdf = gm_gdf.loc[gm_gdf['BFS_NUMMER'].isin(pvinst_df['bfs'].unique())].copy()
                date_cols = [col for col in gm_gdf.columns if (gm_gdf[col].dtype == 'datetime64[ns]') or (gm_gdf[col].dtype == 'datetime64[ms]')]
                gm_gdf.drop(columns=date_cols, inplace=True)
                
                # add map relevant columns
                gm_gdf['hover_text'] = gm_gdf.apply(lambda row: f"{row['NAME']}<br>BFS_NUMMER: {row['BFS_NUMMER']}", axis=1)

                # geo transformations
                gm_gdf = gm_gdf.to_crs('EPSG:4326')
                gm_gdf['geometry'] = gm_gdf['geometry'].apply(flatten_geometry)

                # geojson = gm_gdf.__geo_interface__
                geojson = json.loads(gm_gdf.to_json())

                # Plot using Plotly Express
                fig_topobase = px.choropleth_mapbox(
                    gm_gdf,
                    geojson=geojson,
                    locations="BFS_NUMMER",  # Link BFS_NUMMER for color and location
                    featureidkey="properties.BFS_NUMMER",  # This must match the GeoJSON's property for BFS_NUMMER
                    color_discrete_sequence=[map_topo_egid_specs['uniform_municip_color']],  # Apply the single color to all shapes
                    hover_name="hover_text",  # Use the new column for hover text
                    mapbox_style="carto-positron",  # Basemap style
                    center={"lat": default_map_center[0], "lon": default_map_center[1]},  # Center the map on the region
                    zoom=default_map_zoom,  # Adjust zoom as needed
                    opacity=map_topo_egid_specs['shape_opacity'],   # Opacity to make shapes and basemap visible    
                )
                # Update layout for borders and title
                fig_topobase.update_layout(
                    mapbox=dict(
                        layers=[{
                            'source': geojson,
                            'type': 'line',
                            'color': 'black',  # Set border color for polygons
                            'opacity': 0.25,
                        }]
                    ),
                    title=f"Map of PV topology (scen: {scen})", 
                    legend=dict(
                        itemsizing='constant',
                        title='Legend',
                        traceorder='normal'
                    ),
                )

                # Show the map
                # fig_topobase.show()

            # topo egid map: highlight EGIDs selected for summary ----------------
            if True:
                fig_topoegid = copy.deepcopy(fig_topobase)
                pvinst_gdf = pvinst_gdf.to_crs('EPSG:4326')
                pvinst_gdf['geometry'] = pvinst_gdf['geometry'].apply(flatten_geometry)

                if len(glob.glob(f'{data_path}/output/{scen}/sanity_check_byEGID/summary*.csv')) > 1:
                    files_sanity_check = glob.glob(f'{data_path}/output/{scen}/sanity_check_byEGID/summary*.csv')
                    file = files_sanity_check[0]
                    egid_sanity_check = [file.split('summary_')[-1].split('.csv')[0] for file in files_sanity_check]

                    subinst4_gdf = pvinst_gdf.copy()
                    subinst4_gdf = subinst4_gdf.loc[subinst4_gdf['EGID'].isin(egid_sanity_check)]

                    # Add the points using Scattermapbox
                    fig_topoegid.add_trace(go.Scattermapbox(lat=subinst4_gdf.geometry.y,lon=subinst4_gdf.geometry.x, mode='markers',
                        marker=dict(
                            size=map_topo_egid_specs['point_size_sanity_check'],
                            color=map_topo_egid_specs['point_color_sanity_check'],
                            opacity=map_topo_egid_specs['point_opacity_sanity_check']
                        ),
                        name = 'EGIDs in sanity check xlsx',
                    ))
                
            # topo egid map: all buildings ----------------
            if True:
                # subset inst_gdf for different traces in map plot
                pvinst_gdf['hover_text'] = pvinst_gdf.apply(lambda row: f"EGID: {row['EGID']}<br>BeginOp: {row['BeginOp']}<br>TotalPower: {row['TotalPower']}<br>gklas: {row['gklas']}<br>node: {row['node']}<br>pvtarif: {row['pvtarif']}<br>elecpri: {row['elecpri']}<br>elecpri_info: {row['elecpri_info']}", axis=1)

                subinst1_gdf, subinst2_gdf, subinst3_gdf  = pvinst_gdf.copy(), pvinst_gdf.copy(), pvinst_gdf.copy()
                subinst1_gdf, subinst2_gdf, subinst3_gdf = subinst1_gdf.loc[(subinst1_gdf['inst_TF'] == True) & (subinst1_gdf['info_source'] == 'pv_df')], subinst2_gdf.loc[(subinst2_gdf['inst_TF'] == True) & (subinst2_gdf['info_source'] == 'alloc_algorithm')], subinst3_gdf.loc[(subinst3_gdf['inst_TF'] == False)]

                # Add the points using Scattermapbox
                fig_topoegid.add_trace(go.Scattermapbox(lat=subinst1_gdf.geometry.y,lon=subinst1_gdf.geometry.x, mode='markers',
                    marker=dict(
                        size=map_topo_egid_specs['point_size_pv'],
                        color=map_topo_egid_specs['point_color_pv_df'],
                        opacity=map_topo_egid_specs['point_opacity']
                    ),
                    name = 'house w pv (real)',
                    text=subinst1_gdf['hover_text'],
                    hoverinfo='text'
                ))
                fig_topoegid.add_trace(go.Scattermapbox(lat=subinst2_gdf.geometry.y,lon=subinst2_gdf.geometry.x, mode='markers',
                    marker=dict(
                        size=map_topo_egid_specs['point_size_pv'],
                        color=map_topo_egid_specs['point_color_alloc_algo'],
                        opacity=map_topo_egid_specs['point_opacity']
                    ),
                    name = 'house w pv (predicted)',
                    text=subinst2_gdf['hover_text'],
                    hoverinfo='text'
                ))
                fig_topoegid.add_trace(go.Scattermapbox(lat=subinst3_gdf.geometry.y,lon=subinst3_gdf.geometry.x, mode='markers',
                    marker=dict(
                        size=map_topo_egid_specs['point_size_rest'],
                        color=map_topo_egid_specs['point_color_rest'],
                        opacity=map_topo_egid_specs['point_opacity']
                    ),
                    name = 'house w/o pv',
                    text=subinst3_gdf['hover_text'],
                    hoverinfo='text'
                ))


            # Update layout ----------------
            fig_topoegid.update_layout(
                    title=f"Map of model PV Topology ({scen})",
                    mapbox=dict(
                        style="carto-positron",
                        center={"lat": default_map_center[0], "lon": default_map_center[1]},  # Center the map on the region
                        zoom=default_map_zoom
                    ))

            if plot_show and visual_settings['plot_ind_map_topo_egid'][1]:
                if visual_settings['plot_ind_map_topo_egid'][2]:
                    fig_topoegid.show()
                elif not visual_settings['plot_ind_map_topo_egid'][2]:
                    fig_topoegid.show() if i_scen == 0 else None
            fig_topoegid.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_map_topo_egid.html')

