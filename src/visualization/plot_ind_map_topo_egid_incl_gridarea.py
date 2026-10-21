# python integrated packages
import sys
import os as os
import copy
import glob
import json

# external packages
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from shapely import MultiPoint

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndMapTopoEgidInclGridarea(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_map_topo_egid_incl_gridarea(self, plot_show: bool = True, plot_export: bool = True):

        map_topo_egid_specs = self.visual_sett.plot_ind_map_topo_egid_specs
        checkpoint_to_logfile('plot_ind_map_topo_egid_incl_gridarea', self.visual_sett.log_name)

        trace_color_dict = {
            'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
            'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
            'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Plotly3': pc.sequential.Plotly3,
            'Viridis': pc.sequential.Viridis, 'Turbo': pc.sequential.Turbo, 'Blackbody': pc.sequential.Blackbody, 
            'Bluered': pc.sequential.Bluered, 'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
            'Rainbow': pc.sequential.Rainbow, 'Burg': pc.sequential.Burg, 'Cividis': pc.sequential.Cividis,
            'Inferno': pc.sequential.Inferno, 'Magma': pc.sequential.Magma, 'Plasma': pc.sequential.Plasma,
        }     

        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):

            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            # get pvinst_gdf ----------------
            if True: 
                if not self._set_mc_data_path_or_skip(scen):
                    continue

                # import
                gwr_gdf         = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gwr_gdf.geojson')
                gm_gdf          = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/gm_shp_gdf.geojson')
                dsonodes_gdf    = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/dsonodes_gdf.geojson')
                pred_inst_df    = pl.read_parquet( f'{self.visual_sett.mc_data_path}/pred_inst_df.parquet')
                topo            = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))

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
                    demand_type_list.append(v['demand_arch_typ'])
                    pvtarif_list.append(v['pvtarif_Rp_kWh'])
                    elecpri_list.append(v['elecpri_Rp_kWh'])
                    elecpri_info_list.append(v['elecpri_info'])

                pvinst_df = pd.DataFrame({'EGID': egid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list,
                                        'BeginOp': BeginOp_list, 'TotalPower': TotalPower_list, 'bfs': bfs_list, 
                                        'gklas': gklas_list, 'grid_node': node_list, 'demand_type': demand_type_list,
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
                    pvinst_gdf['hover_text'] = pvinst_gdf.apply(lambda row: f"EGID: {row['EGID']}<br>BeginOp: {row['BeginOp']}<br>TotalPower: {row['TotalPower']}<br>gklas: {row['gklas']}<br>node: {row['grid_node']}<br>pvtarif: {row['pvtarif']}<br>elecpri: {row['elecpri']}<br>elecpri_info: {row['elecpri_info']}", axis=1)
                elif self.visual_sett.reduce_information_content:
                    pvinst_gdf['hover_text'] = pvinst_gdf.apply(lambda row: f"EGID: {row['EGID']}<br>BeginOp: {row['BeginOp']}<br>TotalPower: {row['TotalPower']}<br>gklas: {row['gklas']}<br>pvtarif: {row['pvtarif']}<br>elecpri: {row['elecpri']}<br>elecpri_info: {row['elecpri_info']}", axis=1)

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


            # topo egid map: add grid nodes ----------------
            if True:
                dsonodes_in_topo_gdf = dsonodes_gdf.loc[dsonodes_gdf['grid_node'].isin(pvinst_gdf['grid_node'].unique())].copy()
                dsonodes_in_topo_gdf.to_crs('EPSG:4326', inplace=True)
                dsonodes_in_topo_gdf['kW_threshold'] = dsonodes_in_topo_gdf['kVA_threshold'] * self.pvalloc_scen.GRIDspec_perf_factor_1kVA_to_XkW

                # add color 
                area_palette= trace_color_dict[map_topo_egid_specs['gridnode_area_palette']]
                unique_nodes = sorted(pvinst_gdf['grid_node'].unique())
                n_nodes = len(unique_nodes)
                n_palette = len(area_palette)
                palette_indices = np.linspace(0, n_palette - 1, n_nodes, dtype=int)

                node_color_map = {node: area_palette[i] for node, i in zip(unique_nodes, palette_indices)}
                pvinst_gdf['color'] = pvinst_gdf['grid_node'].map(node_color_map)
                dsonodes_in_topo_gdf['color'] = dsonodes_in_topo_gdf['grid_node'].map(node_color_map)

                if not self.visual_sett.reduce_information_content:
                    fig_topoegid.add_trace(
                        go.Scattermapbox(
                            lat=dsonodes_in_topo_gdf.geometry.y,
                            lon=dsonodes_in_topo_gdf.geometry.x,
                            mode='markers+text',  # Add text next to markers
                            marker=dict(
                                size=map_topo_egid_specs['gridnode_point_size'],
                                color=dsonodes_in_topo_gdf['color'],
                                opacity=map_topo_egid_specs['gridnode_point_opacity']
                            ),
                            text=dsonodes_in_topo_gdf['grid_node'],  # Or any other label (e.g. 'id')
                            textposition='top right',  # Position relative to marker
                            textfont=dict(
                                size=10,  # Adjust font size as needed
                                color='black'
                            ),
                            customdata=dsonodes_in_topo_gdf[['grid_node', 'kW_threshold']],
                            hovertemplate=(
                                "Grid Node: %{customdata[0]}<br>" +
                                "kW Limit: %{customdata[1]}<extra></extra>" 
                                ), 
                            name='Grid Nodes (Centroid)',
                            showlegend=True
                        )
                    )
                elif self.visual_sett.reduce_information_content:
                    fig_topoegid.add_trace(
                        go.Scattermapbox(
                            lat=dsonodes_in_topo_gdf.geometry.y,
                            lon=dsonodes_in_topo_gdf.geometry.x,
                            mode='markers+text',  # Add text next to markers
                            marker=dict(
                                size=map_topo_egid_specs['gridnode_point_size'],
                                color=dsonodes_in_topo_gdf['color'],
                                opacity=map_topo_egid_specs['gridnode_point_opacity']
                            ),
                            text=dsonodes_in_topo_gdf['grid_node'],  # Or any other label (e.g. 'id')
                            textposition='top right',  # Position relative to marker
                            textfont=dict(
                                size=10,  # Adjust font size as needed
                                color='black'
                            ),
                            customdata=dsonodes_in_topo_gdf[['grid_node', 'kW_threshold']],
                            hovertemplate=(
                                "Grid Node: %{customdata[0]}<br><extra></extra>" 
                                ), 
                            name='Grid Nodes (Centroid)',
                            showlegend=True
                        )
                    )

                first_node_hull = True
                spatial_gdf_list = []
                for node in pvinst_gdf['grid_node'].unique():
                    subdf_node_egid = pvinst_gdf.loc[pvinst_gdf['grid_node'] == node].copy()
                    hull_subdf = MultiPoint(subdf_node_egid.geometry.tolist()).convex_hull
                    hull_subdf_gdf = gpd.GeoDataFrame(geometry=[hull_subdf], crs='EPSG:4326')
                    hull_subdf_geojson = json.loads(hull_subdf_gdf.to_json())
                    color = subdf_node_egid["color"].iloc[0]

                    hull_trace = px.choropleth_mapbox(
                        hull_subdf_gdf,
                        geojson=hull_subdf_geojson,
                        locations=hull_subdf_gdf.index.astype(str),
                        color_discrete_sequence=[color], 
                        opacity=map_topo_egid_specs['girdnode_egid_opacity'],
                    ).data[0]
                    hull_trace.update(hoverinfo="skip", hovertemplate=None)
                    hull_trace.name = "Grid Node (EGID hull)"
                    hull_trace.legendgroup = "grid_node_area"  # <== GROUPING KEY
                    hull_trace.showlegend = first_node_hull     # Show only on first
                    # hull_trace.visible = True                   # Ensures all toggle together
                    first_node_hull = False

                    fig_topoegid.add_trace(hull_trace)      


                    # export spatial hull data
                    row = {
                        'grid_node': node,
                        'color': color,
                        'geometry': hull_subdf
                    }
                    spatial_gdf_list.append(row)

            # topo egid map: add iter_round hulls ----------------
            if True:
                pred_inst_df = pred_inst_df.with_columns([
                    pl.col("iter_round").cast(pl.Int64).alias("iter_round")
                ])
                pred_inst_df = pred_inst_df.to_pandas()
                subinst2_gdf_iter = subinst2_gdf.merge(pred_inst_df[['EGID', 'iter_round']], on='EGID', how='left')
                unique_iter_rounds = sorted(subinst2_gdf_iter['iter_round'].dropna().unique())

                for iter in unique_iter_rounds:
                    gdf_iter = subinst2_gdf_iter.loc[subinst2_gdf_iter['iter_round'] == iter].copy()
                    fig_topoegid.add_trace(go.Scattermapbox(lat=gdf_iter.geometry.y,lon=gdf_iter.geometry.x, mode='markers',
                        marker=dict(
                            size=map_topo_egid_specs['point_size_pv'],
                            opacity=0.9,
                        ),
                        name = f'pvinst iter {iter}',
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
            if plot_export and not self.visual_sett.export_only_agg_comparison_plots:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig_topoegid.write_html(f'{self.visual_sett.visual_path}/{scen}/plot_ind_map_topo_egid_incl_gridarea___{scen}.html')
                else:
                    fig_topoegid.write_html(f'{self.visual_sett.visual_path}/plot_ind_map_topo_egid_incl_gridarea___{scen}.html')
                print_to_logfile(f'\texport: plot_ind_map_topo_egid_incl_gridarea (for: {scen})', self.visual_sett.log_name)


            # export spatial hull data
            spatial_hulls_gdf = gpd.GeoDataFrame(spatial_gdf_list, crs='EPSG:4326', geometry='geometry')
            spatial_path = os.path.join(self.visual_sett.data_path, 'pvalloc',self.pvalloc_scen.name_dir_export, 'topo_spatial_data')
            if plot_export:
                if os.path.exists(spatial_path):


                    with open (os.path.join(self.visual_sett.data_path, 'pvalloc',self.pvalloc_scen.name_dir_export, 'topo_spatial_data', f'topo_egid_gridnode_hulls___{scen}.geojson'), 'w') as f:
                        f.write(spatial_hulls_gdf.to_json())

                    # spatial_hulls_gdf.to_file(os.path.join(self.visual_sett.data_path, self.pvalloc_scen.name_dir_export, 'topo_spatial_data', f'topo_egid_gridnode_hulls___{scen}.geojson'), driver='GeoJSON')
                    print_to_logfile(f'\texport: topo_egid_gridnode_hulls geojson (for: {scen})', self.visual_sett.log_name)
