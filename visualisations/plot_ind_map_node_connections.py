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
# PLOT INDIVIDUAL MAP of DSO NODE CONNECTIONS 
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

    if visual_settings['plot_ind_map_node_connections'][0]:
        map_topo_egid_specs = visual_settings['plot_ind_map_topo_egid_specs']
        map_node_connections_specs = visual_settings['plot_ind_map_node_connections_specs']
        checkpoint_to_logfile(f'plot_ind_map_node_connections', log_name)
        
        for i_scen, scen in enumerate(scen_dir_export_list):
            scen_data_path = f'{data_path}/output/{scen}'
            pvalloc_scen = pvalloc_scen_list[i_scen]            

            # import
            gwr_gdf = gpd.read_file(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/gwr_gdf.geojson')
            gm_gdf = gpd.read_file(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/gm_shp_gdf.geojson')   
            dsonodes_gdf = gpd.read_file(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/dsonodes_gdf.geojson')                                      
            
            Map_egid_dsonode = pd.read_parquet(f'{scen_data_path}/Map_egid_dsonode.parquet')
            topo  = json.load(open(f'{scen_data_path}/topo_egid.json', 'r'))
           


            # transformations
            egid_in_topo = [k for k in topo.keys()]
            gwr_gdf = copy.deepcopy(gwr_gdf.loc[gwr_gdf['EGID'].isin(egid_in_topo)])
            Map_egid_dsonode.reset_index(drop=True, inplace=True)

            gwr_gdf = gwr_gdf.merge(Map_egid_dsonode, on='EGID', how='left')


            # pv_instdf_creation for base map
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


            # dsonode map ----------
            fig_dsonodes = copy.deepcopy(fig_topobase)
            gwr_gdf = gwr_gdf.set_crs('EPSG:2056', allow_override=True)
            gwr_gdf = gwr_gdf.to_crs('EPSG:4326')
            gwr_gdf['geometry'] = gwr_gdf['geometry'].apply(flatten_geometry)

            dsonodes_gdf = dsonodes_gdf.set_crs('EPSG:2056', allow_override=True)
            dsonodes_gdf = dsonodes_gdf.to_crs('EPSG:4326')
            dsonodes_gdf['geometry'] = dsonodes_gdf['geometry'].apply(flatten_geometry)

            # define point coloring
            unique_nodes = gwr_gdf['grid_node'].unique()
            colors = pc.sample_colorscale(map_node_connections_specs['point_color_palette'], [n/(len(unique_nodes)) for n in range(len(unique_nodes))])
            node_colors = [colors[c] for c in range(len(unique_nodes))]
            colors_df = pd.DataFrame({'grid_node': unique_nodes, 'node_color': node_colors})
            
            gwr_gdf = gwr_gdf.merge(colors_df, on='grid_node', how='left')
            dsonodes_gdf = dsonodes_gdf.merge(colors_df, on='grid_node', how='left')

            # plot points as Scattermapbox
            gwr_gdf['hover_text'] = gwr_gdf['EGID'].apply(lambda egid: f'EGID: {egid}')

            fig_dsonodes.add_trace(go.Scattermapbox(lat=gwr_gdf.geometry.y,lon=gwr_gdf.geometry.x, mode='markers',
                marker=dict(
                    size=map_node_connections_specs['point_size_all'],
                    color=map_node_connections_specs['point_color_all'],
                    opacity=map_node_connections_specs['point_opacity_all']
                    ),
                    text=gwr_gdf['hover_text'],
                    hoverinfo='text',
                    showlegend=False
                    ))

            for un in unique_nodes:
                # node center / trafo location
                dsonodes_gdf_node = dsonodes_gdf.loc[dsonodes_gdf['grid_node'] == un]
                fig_dsonodes.add_trace(go.Scattermapbox(lat=dsonodes_gdf_node.geometry.y,lon=dsonodes_gdf_node.geometry.x, mode='markers',
                    # marker_symbol = 'cross', 
                    marker=dict(
                        size=map_node_connections_specs['point_size_dsonode_loc'],
                        color=dsonodes_gdf_node['node_color'],
                        opacity=map_node_connections_specs['point_opacity_dsonode_loc']
                        ),
                        name= f'trafo: {un}',
                        text=f'node: {un}, kVA_thres: {dsonodes_gdf_node["kVA_threshold"].sum()}',
                        hoverinfo='text',
                        legendgroup='trafo',
                        legendgrouptitle=dict(text='Trafo Locations'),
                        showlegend=True
                        ))

                # all buildings
                gwr_gdf_node = gwr_gdf.loc[gwr_gdf['grid_node'] == un]
                fig_dsonodes.add_trace(go.Scattermapbox(lat=gwr_gdf_node.geometry.y,lon=gwr_gdf_node.geometry.x, mode='markers',
                    marker=dict(
                        size=map_node_connections_specs['point_size_bynode'],
                        color=gwr_gdf_node['node_color'],
                        opacity=map_node_connections_specs['point_opacity_bynode']
                        ),
                        name= f'{un}',
                        text=gwr_gdf_node['grid_node'],
                        hoverinfo='text',
                        showlegend=True
                        ))
            if plot_show and visual_settings['plot_ind_map_node_connections'][1]:
                if visual_settings['plot_ind_map_node_connections'][2]:
                    fig_dsonodes.show()
                elif not visual_settings['plot_ind_map_node_connections'][2]:
                    fig_dsonodes.show() if i_scen == 0 else None
            fig_dsonodes.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_map_node_connections.html')

