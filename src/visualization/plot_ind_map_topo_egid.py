# python integrated packages
import sys
import os as os
import copy
import glob
import json

# external packages
import geopandas as gpd
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndMapTopoEgid(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_map_topo_egid(self, plot_show: bool = True, plot_export: bool = True):

        map_topo_egid_specs = self.visual_sett.plot_ind_map_topo_egid_specs
        checkpoint_to_logfile('plot_ind_map_topo_egid', self.visual_sett.log_name)

        trace_color_dict = {
            'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
            'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
            'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Plotly3': pc.sequential.Plotly3,
            'Viridis': pc.sequential.Viridis, 'Turbo': pc.sequential.Turbo, 'Blackbody': pc.sequential.Blackbody, 
            'Bluered': pc.sequential.Bluered, 'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
            'Rainbow': pc.sequential.Rainbow, 'Burg': pc.sequential.Burg, 'Cividis': pc.sequential.Cividis,
            'Inferno': pc.sequential.Inferno, 'Magma': pc.sequential.Magma, 'Plasma': pc.sequential.Plasma,
        }     

        fig_agg_color_palettes = ['Mint', 'Oranges', 'Greys',  'Burg',  'Blues', 'Greens', 'Reds',  ]


        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):

            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            # get pvinst_gdf ----------------
            if True: 
                if not self._set_mc_data_path_or_skip(scen):
                    continue

                # import
                gwr_gdf =       gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_gdf.geojson')
                gm_gdf =        gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gm_shp_gdf.geojson')

                topo = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
                # gwr_all_building_df  = pd.read_parquet(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_all_building_df.parquet')
                gwr_all_building_gdf = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_all_building_gdf.geojson')
                outtopo_subdf_paths = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/outtopo_time_subdf/outtopo_subdf_*.parquet')
                Map_egid_dsonode = pd.read_parquet(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/Map_egid_dsonode.parquet')

                egid_list, inst_TF_list, info_source_list, BeginOp_list, TotalPower_list, bfs_list = [], [], [], [], [], []
                gklas_list, heating_system_list, gwaerzh1, gwaerzh2, node_list, demand_type_list   = [], [], [], [], [], []
                pvtarif_list, elecpri_list, elecpri_info_list = [], [], []

                for k,v, in topo.items():
                    egid_list.append(k)
                    inst_TF_list.append(v['pv_inst']['inst_TF'])
                    info_source_list.append(v['pv_inst']['info_source'])
                    BeginOp_list.append(v['pv_inst']['BeginOp'])
                    TotalPower_list.append(v['pv_inst']['TotalPower'])
                    bfs_list.append(v['gwr_info']['bfs'])
                    gklas_list.append(v['gwr_info']['gklas'])
                    heating_system_list.append(v['gwr_info']['heating_system'])
                    gwaerzh1.append(v['gwr_info']['gwaerzh1'])
                    gwaerzh2.append(v['gwr_info']['gwaerzh2'])
                    node_list.append(v['grid_node'])
                    demand_type_list.append(v['demand_arch_typ'])
                    pvtarif_list.append(v['pvtarif_Rp_kWh'])
                    elecpri_list.append(v['elecpri_Rp_kWh'])
                    elecpri_info_list.append(v['elecpri_info'])

                pvinst_df = pd.DataFrame({'EGID': egid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list,
                                        'BeginOp': BeginOp_list, 'TotalPower': TotalPower_list, 'bfs': bfs_list, 
                                        'gklas': gklas_list, 'heating_system': heating_system_list,
                                        'gwaerzh1': gwaerzh1, 'gwaerzh2': gwaerzh2,
                                        'grid_node': node_list, 'demand_type': demand_type_list,
                                        'pvtarif': pvtarif_list, 'elecpri': elecpri_list, 'elecpri_info': elecpri_info_list })
                pvinst_df = pvinst_df.merge(gwr_gdf[['geometry', 'EGID']], on='EGID', how='left')
                pvinst_gdf = gpd.GeoDataFrame(pvinst_df, crs='EPSG:2056', geometry='geometry')

            # base map ----------------
            if True:
                # transformations
                gm_gdf['BFS_NUMMER'] = gm_gdf['BFS_NUMMER'].astype(str)
                gm_gdf = gm_gdf.loc[gm_gdf['BFS_NUMMER'].isin(pvinst_df['bfs'].unique())].copy()
                date_cols = [col for col in gm_gdf.columns if (gm_gdf[col].dtype == 'datetime64[ns]') or (gm_gdf[col].dtype == 'datetime64[ms]')]
                gm_gdf.drop(columns=date_cols, inplace=True)

                # add map relevant columns
                gm_gdf['hover_text'] = gm_gdf.apply(lambda row: f"{row['NAME']}<br>BFS_NUMMER: {row['BFS_NUMMER']}", axis=1)

                # geo transformations
                gm_gdf = gm_gdf.to_crs('EPSG:4326')
                gm_gdf['geometry'] = gm_gdf['geometry'].apply(self.flatten_geometry)

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
                    center={"lat": self.visual_sett.default_map_center[0], "lon": self.visual_sett.default_map_center[1]},  # Center the map on the region
                    zoom=self.visual_sett.default_map_zoom,  # Adjust zoom as needed
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


            # topo egid map: highlight EGIDs selected for summary ----------------
            if True:
                fig_topoegid = copy.deepcopy(fig_topobase)
                pvinst_gdf = pvinst_gdf.to_crs('EPSG:4326')
                pvinst_gdf['geometry'] = pvinst_gdf['geometry'].apply(self.flatten_geometry)

                if len(glob.glob(f'{self.visual_sett.mc_data_path}/topo_egid_summary_byEGID/*.csv')) > 1:
                    files_sanity_check = glob.glob(f'{self.visual_sett.mc_data_path}/topo_egid_summary_byEGID/*.csv')
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
                if not self.visual_sett.reduce_information_content:
                    pvinst_gdf['hover_text'] = pvinst_gdf.apply(lambda row: f"EGID: {row['EGID']}<br>BeginOp: {row['BeginOp']}<br>TotalPower: {row['TotalPower']}<br>gklas: {row['gklas']}<br>heating_system: {row['heating_system']}<br>gwaerzh1/2: {row['gwaerzh1']}, {row['gwaerzh2']}<br>node: {row['grid_node']}<br>pvtarif: {row['pvtarif']}<br>elecpri: {row['elecpri']}<br>elecpri_info: {row['elecpri_info']}", axis=1)
                elif self.visual_sett.reduce_information_content:
                    pvinst_gdf['hover_text'] = pvinst_gdf.apply(lambda row: f"EGID: {row['EGID']}<br>BeginOp: {row['BeginOp']}<br>TotalPower: {row['TotalPower']}<br>gklas: {row['gklas']}<br>heating_system: {row['heating_system']}<br>gwaerzh1/2: {row['gwaerzh1']}, {row['gwaerzh2']}<br>pvtarif: {row['pvtarif']}<br>elecpri: {row['elecpri']}<br>elecpri_info: {row['elecpri_info']}", axis=1)

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

            # topo egid map: outsample buildings ----------------
            if True: 
                outtopo_egid_list = []
                for i_path, path in enumerate(outtopo_subdf_paths):
                    outtopo_subdf = pl.read_parquet(path)
                    unique_egids_in_subdf = outtopo_subdf.select(pl.col("EGID").unique()).to_series().to_list()
                    outtopo_egid_list = outtopo_egid_list + unique_egids_in_subdf

                outtopo_gdf = gwr_all_building_gdf.loc[gwr_all_building_gdf['EGID'].isin(outtopo_egid_list)].copy()
                outtopo_gdf = outtopo_gdf.to_crs('EPSG:4326')
                outtopo_gdf['geometry'] = outtopo_gdf['geometry'].apply(self.flatten_geometry)
                outtopo_gdf = outtopo_gdf.merge(Map_egid_dsonode[['EGID', 'grid_node']], on='EGID', how='left')

                outtopo_gdf['hover_text'] = outtopo_gdf.apply(lambda row: f"EGID: {row['EGID']}<br>grid_node: {row['grid_node']}<br>gklas: {row['GKLAS']}<br>gbauj: {row['GBAUJ']}<br>gstat: {row['GSTAT']}<br>gkat: {row['GKAT']}<br>gwaerzh1/2: {row['GWAERZH1']}, -", axis=1)


                # Add out sample points to map
                fig_topoegid.add_trace(go.Scattermapbox(lat=outtopo_gdf.geometry.y,lon=outtopo_gdf.geometry.x, mode='markers',
                        marker=dict(
                            size    = map_topo_egid_specs['point_size_outsample'], 
                            color   = map_topo_egid_specs['point_color_outsample'], 
                            opacity = map_topo_egid_specs['point_opacity'], 
                        ), 
                        name = 'EGIDs out of sample',
                        text=outtopo_gdf['hover_text'],
                        hoverinfo='text'
                ))


            # Update layout ----------------
            fig_topoegid.update_layout(
                    title=f"Map of Model PV Topology ({scen})",
                    mapbox=dict(
                        style="carto-positron",
                        center={"lat": self.visual_sett.default_map_center[0], "lon": self.visual_sett.default_map_center[1]},  # Center the map on the region
                        zoom=self.visual_sett.default_map_zoom
                    ))


            if plot_show:
                fig_topoegid.show()
            if plot_export:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig_topoegid.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_ind_map_topo_egid.html')
                else:
                    fig_topoegid.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_map_topo_egid.html')
