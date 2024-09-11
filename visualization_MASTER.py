# -----------------------------------------------------------------------------
# visualizations_MASTER.py 
# -----------------------------------------------------------------------------
# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024
# > description: 



# PACKAGES --------------------------------------------------------------------
if True:
    import os as os
    import sys
    sys.path.append(os.getcwd())
    # sys.path.append(pvalloc_settings['wd_path_laptop']) if pvalloc_settings['script_run_on_server'] else sys.path.append(pvalloc_settings['wd_path_server'])

    # external packages
    import os as os
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import json 
    import plotly.express as px
    import plotly.graph_objects as go
    import shapely

    from datetime import datetime
    from pprint import pformat
    from shapely.geometry import Polygon, MultiPolygon


    import auxiliary_functions
    from auxiliary_functions import chapter_to_logfile

pvalloc_scenarios_local={
    'pvalloc_smallBL_SLCTN_npv_weighted': {
            'name_dir_import': 'preprep_BSBLSO_18to22_20240826_22h',
  
            'recreate_topology':            True, 
            'recalc_economics_topo_df':     True,
            'run_allocation_loop':          True,

            'algorithm_specs': {'inst_selection_method': 'prob_weighted_npv',},
    },
    'pvalloc_smallBL_SLCTN_random': {
            'name_dir_import': 'preprep_BSBLSO_18to22_20240826_22h',

            'recreate_topology':            True, 
            'recalc_economics_topo_df':     True,
            'run_allocation_loop':          True,

            'algorithm_specs': {'inst_selection_method': 'random',},
        },

}

visual_settings_local = {
        'plot_show': True,

        'plot_ind_line_productionHOY_per_node': False,
        'plot_ind_line_installedCap_per_month': False,
        'plot_ind_line_installedCap_per_BFS': False,
        'map_ind_topo_egid': True,
        
        'default_zoom_year': [2012, 2030],
    }

def visualization_MASTER(pvalloc_scenarios_func, visual_settings_func):
    # SETTINGS --------------------------------------------------------------------
    if not isinstance(pvalloc_scenarios_func, dict):
        print(' USE LOCAL SETTINGS - DICT  ')
        pvalloc_scenarios = pvalloc_scenarios_local
    else:
        pvalloc_scenarios = pvalloc_scenarios_func

    if not isinstance(visual_settings_func, dict) or visual_settings_func == {}:
        print(' USE LOCAL SETTINGS - DICT  ')
        visual_settings = visual_settings_local

    
    # SETUP =====================================================================
    
    # general setup for paths etc.
    first_alloc_sett = pvalloc_scenarios[list(pvalloc_scenarios.keys())[0]]
    wd_path = first_alloc_sett['wd_path_laptop']
    data_path = f'{wd_path}_data'

    # create directory + log file
    visual_path = f'{data_path}/output/visualizations'
    if not os.path.exists(visual_path):
        os.makedirs(visual_path)

    log_name = f'{data_path}/output/visual_log.txt'


    chapter_to_logfile(f'start run_visualisations MASTER ', log_name, overwrite_file=True)


    # SETTINGS LATER ADDED TO visual_settings ------------------------
    plot_show = visual_settings_local['plot_show']
    default_zoom_year = visual_settings_local['default_zoom_year']

    plot_ind_line_productionHOY_per_node = visual_settings_local['plot_ind_line_productionHOY_per_node']
    plot_ind_line_installedCap_per_month = visual_settings_local['plot_ind_line_installedCap_per_month']
    plot_ind_line_installedCap_per_BFS = visual_settings_local['plot_ind_line_installedCap_per_BFS']
    map_ind_topo_egid = visual_settings_local['map_ind_topo_egid']


    plot_show = True
    zoom_window_HOY = [0, 8760]
    export_png = False
    # plot_width_cm = 20 THIS WILL ALL COME FROM GENERAL SETTIGNS!
    # plot_height_cm = 20
    
    # ---

    # EXTRACT SCENARIO INFORMATION + IMPORT --------------------------------------------------------------------

    scen_dir_export_list, scen_dir_import_list, T0_prediction_list, months_prediction_list = [], [], [], []
    T0_prediction_list, months_lookback_list, months_prediction_list = [], [], []
    pvalloc_scen_list = []
    for key, val in pvalloc_scenarios.items():
        scen_dir_export_list.append(val['name_dir_export'])
        scen_dir_import_list.append(val['name_dir_import'])
        T0_prediction_list.append(val['T0_prediction'])
        months_prediction_list.append(val['months_prediction'])
        pvalloc_scen_list.append(val)

    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp/swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET.shp')
    gm_shp['BFS_NUMMER'] = gm_shp['BFS_NUMMER'].astype(int).astype(str) 



    # universal funcs --------------------------------------------------------------------
    # universal func for plot adjustments -----
    def add_scen_name_to_plot(fig_func, scen, pvalloc_scen):
        # add scenario name
        fig_func.add_annotation(
            text=f'Scen: {scen}, (start T0: {pvalloc_scen["T0_prediction"].split(" ")[0]}, {pvalloc_scen["months_prediction"]} prediction months)',
            xref="paper", yref="paper",
            x=0.5, y=1.05, showarrow=False,
            font=dict(size=12)
        )
        return fig_func
    
    def add_multip_scen_name_to_plot(fig, scen_list, pvalloc_scen_list):
        # add scenario name
        for i, scen in enumerate(scen_list):
            pvalloc_scen = pvalloc_scen_list[i]
            fig.add_annotation(
                text=f'Scen: {scen}, (start T0: {pvalloc_scen["T0_prediction"].split(" ")[0]}, {pvalloc_scen["months_prediction"]} prediction months)',
                xref="paper", yref="paper",
                x=0.5, y=1.05 + 0.05*i, showarrow=False,
                font=dict(size=12)
            )
        return fig
    
    # universal func for plot T0 tick -----
    def add_T0_tick_to_plot(fig, T0_prediction):
        fig.add_shape(
            # Line Vertical
            dict(
                type="line",
                x0=T0_prediction,
                y0=0,
                x1=T0_prediction,
                y1=1,  # Dynamic height
                line=dict(color="black", width=1, dash="dot"),
            )
        )
        fig.add_annotation(
            x=  T0_prediction,
            y=1,
            text="T0 Prediction",
            showarrow=False,
            yshift=10
        )
        return fig

    # universial func to set default plot zoom -----
    def set_default_fig_zoom(fig, zoom_window, df, datecol):
        start_zoom = pd.to_datetime(f'{zoom_window[0]}-01-01')
        max_date = df[datecol].max() + pd.DateOffset(years=1)
        if pd.to_datetime(f'{zoom_window[1]}-01-01') > max_date:
            end_zoom = max_date
        else:
            end_zoom = pd.to_datetime(f'{zoom_window[1]}-01-01')
        fig.update_layout(
            xaxis = dict(range=[start_zoom, end_zoom])
        )
        return fig 
    


    # PLOT INDIVIDUAL SCEN --------------------------------------------------------------------
       
    # plot ind - line: Production HOY per Node ============================
    if plot_ind_line_productionHOY_per_node:
        i, scen = 0, scen_dir_export_list[0]
        for i, scen in enumerate(scen_dir_export_list):
            # setup + import ----------
            scen_data_path = f'{data_path}/output/{scen}'
            pvalloc_scen = pvalloc_scen_list[i]

            gridnode_df = pd.read_parquet(f'{scen_data_path}/gridnode_df.parquet') 
            gridnode_df['t_int'] = gridnode_df['t'].str.extract('t_(\d+)').astype(int)
            gridnode_df.sort_values(by=['t_int'], inplace=True)

            # plot ----------------
            fig = px.line(gridnode_df, x='t', y='pvprod_kW', color = 'grid_node' )
    
            fig.update_layout(
                xaxis_title='Hour of Year',
                yaxis_title='Production (kWh)',
                legend_title='Node ID',
                title = f'Production per node (kWh, weather year: {pvalloc_scen["weather_specs"]["weather_year"]})'
            )

            fig = add_scen_name_to_plot(fig, scen, pvalloc_scen_list[i])
            if plot_show:
                fig.show() 

            fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_line_productionHOY_per_node.html')


    # plot ind - line:
    #     plot ind - line: Installed Capacity per Month ===========================
    #     plot ind - line: Installed Capacity per BFS   ===========================
    if plot_ind_line_installedCap_per_month or plot_ind_line_installedCap_per_BFS:
        i, scen = 0, scen_dir_export_list[0]
        for i, scen in enumerate(scen_dir_export_list):
            scen_data_path = f'{data_path}/output/{scen}'
            T0_prediction = T0_prediction_list[0]
            months_prediction = months_prediction_list[0]
            pvalloc_scen = pvalloc_scen_list[i]

            topo = json.load(open(f'{scen_data_path}/topo_egid.json', 'r'))
            egid_list, inst_TF_list, info_source_list, BeginOp_list, TotalPower_list, bfs_list= [], [], [], [], [], []

            for k,v, in topo.items():
                egid_list.append(k)
                inst_TF_list.append(v['pv_inst']['inst_TF'])
                info_source_list.append(v['pv_inst']['info_source'])
                BeginOp_list.append(v['pv_inst']['BeginOp'])
                TotalPower_list.append(v['pv_inst']['TotalPower'])
                bfs_list.append(v['gwr_info']['bfs'])

            pvinst_df = pd.DataFrame({'EGID': egid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list, 
                                      'BeginOp': BeginOp_list, 'TotalPower': TotalPower_list, 'bfs': bfs_list})
            pvinst_df = pvinst_df.loc[pvinst_df['inst_TF'] == True]

            pvinst_df['TotalPower'] = pd.to_numeric(pvinst_df['TotalPower'], errors='coerce')
            pvinst_df['BeginOp'] = pvinst_df['BeginOp'].apply(lambda x: x if len(x) == 10 else x + '-01')
            pvinst_df['BeginOp'] = pd.to_datetime(pvinst_df['BeginOp'], format='%Y-%m-%d')
            pvinst_df['bfs'] = pvinst_df['bfs'].astype(str)

            # REMOVE some old historic data, because not interesting on Plot
            # pvinst_df = pvinst_df.loc[pvinst_df['BeginOp'] > pd.to_datetime('2010-01-01')]


            # plot ind - line: Installed Capacity per Month ===========================
            if plot_ind_line_installedCap_per_month: 
                capa_month_df = pvinst_df.copy()
                capa_month_df['BeginOp_month'] = capa_month_df['BeginOp'].dt.to_period('M')
                capa_month_df = capa_month_df.groupby(['BeginOp_month', 'info_source'])['TotalPower'].sum().reset_index().copy()
                capa_month_df['BeginOp_month'] = capa_month_df['BeginOp_month'].dt.to_timestamp()
                capa_month_built = capa_month_df.loc[capa_month_df['info_source'] == 'pv_df'].copy()
                capa_month_predicted = capa_month_df.loc[capa_month_df['info_source'] == 'alloc_algorithm'].copy()

                capa_year_df = pvinst_df.copy()
                capa_year_df['BeginOp_year'] = capa_year_df['BeginOp'].dt.to_period('Y')
                capa_year_df = capa_year_df.groupby(['BeginOp_year', 'info_source'])['TotalPower'].sum().reset_index().copy()
                capa_year_df['BeginOp_year'] = capa_year_df['BeginOp_year'].dt.to_timestamp()
                capa_year_built = capa_year_df.loc[capa_year_df['info_source'] == 'pv_df'].copy()
                capa_year_predicted = capa_year_df.loc[capa_year_df['info_source'] == 'alloc_algorithm'].copy()

                # plot ----------------
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'], line = dict(color = 'navy'),name='built + predicted (month)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=capa_month_built['BeginOp_month'], y=capa_month_built['TotalPower'], line = dict(color = 'deepskyblue'), name='built (month)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=capa_month_predicted['BeginOp_month'], y=capa_month_predicted['TotalPower'], line = dict(color = 'cornflowerblue'), name='predicted (month)', mode='lines+markers'))

                fig1.add_trace(go.Scatter(x=capa_year_df['BeginOp_year'], y=capa_year_df['TotalPower'], line = dict(color = 'forestgreen'), name='built + predicted (year)', mode='lines+markers',))
                fig1.add_trace(go.Scatter(x=capa_year_built['BeginOp_year'], y=capa_year_built['TotalPower'], line = dict(color = 'lightgreen'), name='built (year)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=capa_year_predicted['BeginOp_year'], y=capa_year_predicted['TotalPower'], line = dict(color = 'limegreen'), name='predicted (year)', mode='lines+markers'))

                fig1.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Installed Capacity (kW)',
                    legend_title='Time steps',
                    title = f'Installed Capacity per Month (weather year: {pvalloc_scen["weather_specs"]["weather_year"]})'
                )

                # add T0 prediction
                date = '2008-01-01 00:00:00'
                fig1.add_shape(
                    # Line Vertical
                    dict(
                        type="line",
                        x0=T0_prediction,
                        y0=0,
                        x1=T0_prediction,
                        y1=max(capa_year_df['TotalPower'].max(), capa_year_df['TotalPower'].max()),  # Dynamic height
                        line=dict(color="black", width=1, dash="dot"),
                    )
                )
                fig1.add_annotation(
                    x=  T0_prediction,
                    y=max(capa_year_df['TotalPower'].max(), capa_year_df['TotalPower'].max()),
                    text="T0 Prediction",
                    showarrow=False,
                    yshift=10
                )

                fig1 = add_scen_name_to_plot(fig1, scen, pvalloc_scen_list[i])
                fig1 = set_default_fig_zoom(fig1, default_zoom_year, capa_year_df, 'BeginOp_year')
                if plot_show:
                    fig1.show()
                fig1.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_line_installedCap_per_month.html')

            # plot ind - line: Installed Capacity per BFS ===========================
            if plot_ind_line_installedCap_per_BFS: 
                capa_bfs_df = pvinst_df.copy()
                gm_shp.rename(columns={'BFS_NUMMER': 'bfs'}, inplace=True)
                capa_bfs_df = capa_bfs_df.merge(gm_shp[['bfs', 'NAME']], on='bfs', how = 'left' )
                capa_bfs_df['BeginOp_month'] = capa_bfs_df['BeginOp'].dt.to_period('M')
                capa_bfs_month_df = capa_bfs_df.groupby(['BeginOp_month', 'bfs'])['TotalPower'].sum().reset_index().copy()
                capa_bfs_month_df['BeginOp_month'] = capa_bfs_month_df['BeginOp_month'].dt.to_timestamp()

                capa_bfs_df['BeginOp_year'] = capa_bfs_df['BeginOp'].dt.to_period('Y')
                capa_bfs_year_df = capa_bfs_df.groupby(['BeginOp_year', 'bfs'])['TotalPower'].sum().reset_index().copy()
                capa_bfs_year_df['BeginOp_year'] = capa_bfs_year_df['BeginOp_year'].dt.to_timestamp()

                # plot ----------------
                fig2 = go.Figure()
                for bfs in capa_bfs_month_df['bfs'].unique():
                    name = gm_shp.loc[gm_shp['bfs'] == bfs, 'NAME'].values[0]
                    subdf = capa_bfs_month_df.loc[capa_bfs_month_df['bfs'] == bfs].copy()
                    fig2.add_trace(go.Scatter(x=subdf['BeginOp_month'], y=subdf['TotalPower'], name=f'{name}', legendgroup = 'By Month',  mode = 'lines'))

                for bfs in capa_bfs_year_df['bfs'].unique():
                    name = gm_shp.loc[gm_shp['bfs'] == bfs, 'NAME'].values[0]
                    subdf = capa_bfs_year_df.loc[capa_bfs_year_df['bfs'] == bfs].copy()
                    fig2.add_trace(go.Scatter(x=subdf['BeginOp_year'], y=subdf['TotalPower'], name=f'{name}', legendgroup = 'By Year', mode = 'lines'))

                fig2.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Installed Capacity (kW)',
                    legend_title='BFS',
                    title = f'Installed Capacity per Municipality (BFS) (weather year: {pvalloc_scen["weather_specs"]["weather_year"]})'
                )

                fig2.add_shape(
                    # Line Vertical
                    dict(
                        type="line",
                        x0=T0_prediction,
                        y0=0,
                        x1=T0_prediction,
                        y1=capa_bfs_year_df['TotalPower'],  # Dynamic height
                        line=dict(color="black", width=1, dash="dot"),
                    )
                )
                fig2.add_annotation(
                    x=  T0_prediction,
                    y=1,
                    text="T0 Prediction",
                    showarrow=False,
                    yshift=10
                )
                
                fig2 = add_scen_name_to_plot(fig2, scen, pvalloc_scen_list[i])
                fig2 = set_default_fig_zoom(fig2, default_zoom_year, capa_bfs_year_df, 'BeginOp_year')
                if plot_show:
                    fig2.show()
                fig2.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_line_installedCap_per_BFS.html')


    # plot ind - map:  Covered Area of Allocation Model ========================
    i = 0
    if map_ind_topo_egid:
        i, scen = 0, scen_dir_export_list[0]

        # setup
        scen_data_path = f'{data_path}/output/{scen}'
        T0_prediction = T0_prediction_list[0]
        months_prediction = months_prediction_list[0]
        pvalloc_scen = pvalloc_scen_list[i]

        # general import 
        gwr_gdf = gpd.read_file(f'{data_path}/output/{pvalloc_scen['name_dir_import']}/gwr_gdf.geojson')
        gm_gdf = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp/swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET.shp')

        topo  = json.load(open(f'{scen_data_path}/topo_egid.json', 'r'))
        egid_list, inst_TF_list, info_source_list, BeginOp_list, TotalPower_list, bfs_list= [], [], [], [], [], []

        for k,v, in topo.items():
            egid_list.append(k)
            inst_TF_list.append(v['pv_inst']['inst_TF'])
            info_source_list.append(v['pv_inst']['info_source'])
            BeginOp_list.append(v['pv_inst']['BeginOp'])
            TotalPower_list.append(v['pv_inst']['TotalPower'])
            bfs_list.append(v['gwr_info']['bfs'])

        pvinst_df = pd.DataFrame({'EGID': egid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list,
                                  'BeginOp': BeginOp_list, 'TotalPower': TotalPower_list, 'bfs': bfs_list})
        

        # plot GM map ============================
        if True: 
            # transformations
            gm_gdf['BFS_NUMMER'] = gm_gdf['BFS_NUMMER'].astype(str)
            gm_gdf = gm_gdf.loc[gm_gdf['BFS_NUMMER'].isin(pvinst_df['bfs'].unique())].copy()
            date_cols = [col for col in gm_gdf.columns if (gm_gdf[col].dtype == 'datetime64[ns]') or (gm_gdf[col].dtype == 'datetime64[ms]')]
            gm_gdf.drop(columns=date_cols, inplace=True)
            
            gm_gdf['uniform_color'] = "#EF553B"
            uniform_color = "#EF553B"
            gm_gdf['hover_text'] = gm_gdf.apply(lambda row: f"{row['NAME']}<br>BFS_NUMMER: {row['BFS_NUMMER']}", axis=1)

            # geo transformations
            gm_gdf = gm_gdf.to_crs('EPSG:4326')
            # Function to flatten geometries to 2D (ignoring Z-dimension)
            def flatten_geometry(geom):
                if geom.has_z:
                    if geom.geom_type == 'Polygon':
                        exterior = [(x, y) for x, y, z in geom.exterior.coords]
                        interiors = [[(x, y) for x, y, z in interior.coords] for interior in geom.interiors]
                        return Polygon(exterior, interiors)
                    elif geom.geom_type == 'MultiPolygon':
                        return MultiPolygon([flatten_geometry(poly) for poly in geom.geoms])
                return geom
            gm_gdf['geometry'] = gm_gdf['geometry'].apply(flatten_geometry)

            geojson = gm_gdf.__geo_interface__

            # Plot using Plotly Express
            fig = px.choropleth_mapbox(
                gm_gdf,
                geojson=geojson,
                locations="BFS_NUMMER",  # Link BFS_NUMMER for color and location
                featureidkey="properties.BFS_NUMMER",  # This must match the GeoJSON's property for BFS_NUMMER
                # color="uniform_color",  # Column to use for coloring the shapes
                color_discrete_sequence=[uniform_color],  # Apply the single color to all shapes
                hover_name="hover_text",  # Use the new column for hover text
                mapbox_style="carto-positron",  # Basemap style
                center={"lat": 47.41, "lon": 7.49},  # Center the map on the region
                zoom=10,  # Adjust zoom as needed
                opacity=0.25  # Opacity to make shapes and basemap visible
            )
            # Update layout for borders and title
            fig.update_layout(
                mapbox=dict(
                    layers=[{
                        'source': geojson,
                        'type': 'line',
                        'color': 'black',  # Set border color for polygons
                        'opacity': 0.25,
                    }]
                ),
                title=f"Map of model range ({scen})"
            )

            # Show the map
            # fig.show()

        # plot GWR map ============================
        if True:
            pvinst_df.dtypes
            gwr_gdf.dtypes
            pvinst_df = pd.DataFrame({'EGID': egid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list,'BeginOp': BeginOp_list, 'TotalPower': TotalPower_list, 'bfs': bfs_list})

            pvinst_df = pvinst_df.merge(gwr_gdf[['geometry', 'EGID']], on='EGID', how='left')
            pvinst_gdf = gpd.GeoDataFrame(pvinst_df, crs='EPSG:2056', geometry='geometry')

            pvinst_gdf = pvinst_gdf.to_crs('EPSG:4326')
            # Function to flatten geometries to 2D (ignoring Z-dimension)
            def flatten_geometry(geom):
                if geom.has_z:
                    if geom.geom_type == 'Polygon':
                        exterior = [(x, y) for x, y, z in geom.exterior.coords]
                        interiors = [[(x, y) for x, y, z in interior.coords] for interior in geom.interiors]
                        return Polygon(exterior, interiors)
                    elif geom.geom_type == 'MultiPolygon':
                        return MultiPolygon([flatten_geometry(poly) for poly in geom.geoms])
                return geom
            pvinst_gdf['geometry'] = pvinst_gdf['geometry'].apply(flatten_geometry)

            # set coloring
            def determine_color(row):
                if row['inst_TF'] and row['info_source'] == 'pv_df':
                    return 'darkgreen'
                elif row['inst_TF'] and row['info_source'] == 'alloc_algorithm':
                    return 'yellow'
                else:
                    return 'grey'
            pvinst_gdf['color'] = pvinst_gdf.apply(determine_color, axis=1)

            pvinst_gdf['hover_text'] = pvinst_gdf.apply(lambda row: f"EGID: {row['EGID']}<br>TotalPower: {row['TotalPower']}", axis=1)
            
            geojson = pvinst_gdf.__geo_interface__
            
            # Add the points using Scattermapbox
            fig.add_trace(go.Scattermapbox(
                lat=pvinst_gdf.geometry.y,
                lon=pvinst_gdf.geometry.x,
                mode='markers',
                marker=dict(
                    size=10,
                    color=pvinst_gdf['color'],
                    opacity=0.7
                ),
                text=pvinst_gdf['hover_text'],
                hoverinfo='text'
            ))

            # Update layout
            fig.update_layout(
                title=f"Map of model range with additional points ({scen})",
                mapbox=dict(
                    style="carto-positron",
                    center={"lat": 47.41, "lon": 7.49},
                    zoom=10
                )
            )

        if plot_show:
            fig.show()
        fig.write_html(f'{data_path}/output/visualizations/{scen}__map_ind_topo_egid.html')




    # PLOT AGGREGATED SCEN --------------------------------------------------------------------
    if len(list(set(T0_prediction_list))) ==1:
        T0_pred_agg = T0_prediction_list[0]


    # plot agg - line: Installed Capacity per Month ============================
    fig = go.Figure()
    i, scen = 0, scen_dir_export_list[0]
    # i, scen = 1, scen_dir_export_list[1]
    for i, scen in enumerate(scen_dir_export_list):
        scen_data_path = f'{data_path}/output/{scen}'
        T0_prediction = T0_prediction_list[0]
        months_prediction = months_prediction_list[0]
        pvalloc_scen = pvalloc_scen_list[i]

        topo = json.load(open(f'{scen_data_path}/topo_egid.json', 'r'))
        egid_list, inst_TF_list, info_source_list, BeginOp_list, TotalPower_list = [], [], [], [], []

        for k,v, in topo.items():
            egid_list.append(k)
            inst_TF_list.append(v['pv_inst']['inst_TF'])
            info_source_list.append(v['pv_inst']['info_source'])
            BeginOp_list.append(v['pv_inst']['BeginOp'])
            TotalPower_list.append(v['pv_inst']['TotalPower'])

        pvinst_df = pd.DataFrame({'EGID': egid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list, 'BeginOp': BeginOp_list, 'TotalPower': TotalPower_list})
        pvinst_df = pvinst_df.loc[pvinst_df['inst_TF'] == True]

        pvinst_df['BeginOp'] = pvinst_df['BeginOp'].apply(lambda x: x if len(x) == 10 else x + '-01')
        pvinst_df['BeginOp'] = pd.to_datetime(pvinst_df['BeginOp'], format='%Y-%m-%d')
        pvinst_df['TotalPower'] = pd.to_numeric(pvinst_df['TotalPower'], errors='coerce')

        def agg_pvinst_df(df, freq, new_datecol, datecol, infocol, valcol):
            df_agg = df.copy()
            df_agg[new_datecol] = df_agg[datecol].dt.to_period(freq)
            df_agg = df_agg.groupby([new_datecol, infocol])[valcol].sum().reset_index().copy()
            df_agg[new_datecol] = df_agg[new_datecol].dt.to_timestamp()
            return df_agg
                
        pvinst_month_df = agg_pvinst_df(pvinst_df, 'M', 'BeginOp_month', 'BeginOp', 'info_source', 'TotalPower')
        pvinst_month_built = pvinst_month_df.loc[pvinst_month_df['info_source'] == 'pv_df'].copy()
        capa_month_predicted = pvinst_month_df.loc[pvinst_month_df['info_source'] == 'alloc_algorithm'].copy()

        pvinst_year_df = agg_pvinst_df(pvinst_df, 'Y', 'BeginOp_year', 'BeginOp', 'info_source', 'TotalPower')
        pvinst_year_built = pvinst_year_df.loc[pvinst_year_df['info_source'] == 'pv_df'].copy()
        pvinst_year_predicted = pvinst_year_df.loc[pvinst_year_df['info_source'] == 'alloc_algorithm'].copy()


        # plot ----------------
        fig.add_trace(go.Scatter(x=pvinst_month_df['BeginOp_month'], y=pvinst_month_df['TotalPower'], line = dict(color = 'cornflowerblue'),name=f'built + predicted (month)',  mode='lines+markers', legendgroup = scen, legendgrouptitle_text=scen))
        # fig.add_trace(go.Scatter(x=pvinst_month_built['BeginOp_month'], y=pvinst_month_built['TotalPower'], line = dict(color = 'deepskyblue'), name=f'built (month)',  mode='lines+markers', legendgroup = scen, legendgrouptitle_text=scen))
        # fig.add_trace(go.Scatter(x=capa_month_predicted['BeginOp_month'], y=capa_month_predicted['TotalPower'], line = dict(color = 'navy'), name=f'predicted (month)',  mode='lines+markers', legendgroup = scen, legendgrouptitle_text=scen))

        fig.add_trace(go.Scatter(x=pvinst_year_df['BeginOp_year'], y=pvinst_year_df['TotalPower'], line = dict(color = 'limegreen'), name=f'built + predicted (year)',  mode='lines+markers', legendgroup = scen, legendgrouptitle_text=scen))
        # fig.add_trace(go.Scatter(x=pvinst_year_built['BeginOp_year'], y=pvinst_year_built['TotalPower'], line = dict(color = 'lightgreen'), name=f'built (year)',  mode='lines+markers', legendgroup = scen, legendgrouptitle_text=scen))
        # fig.add_trace(go.Scatter(x=pvinst_year_predicted['BeginOp_year'], y=pvinst_year_predicted['TotalPower'], line = dict(color = 'forestgreen'), name=f'predicted (year)',  mode='lines+markers', legendgroup = scen, legendgrouptitle_text=scen))

    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Installed Capacity (kW)',
        legend_title='Scenarios',
        title = f'Installed Capacity per Month (weather year: {pvalloc_scen["weather_specs"]["weather_year"]})'
    )

    fig = add_T0_tick_to_plot(fig, T0_pred_agg)
    fig = set_default_fig_zoom(fig, [2010, 2030], pvinst_year_df, 'BeginOp_year')
    if plot_show:
        fig.show()

    fig.write_html(f'{data_path}/output/visualizations/plot_agg_line_installedCap_per_month.html')



    # plot agg - line: PV Production per Hour of Year ============================
    fig = go.Figure()
    for i, scen in enumerate(scen_dir_export_list):
        # setup + import ----------
        scen_data_path = f'{data_path}/output/{scen}'
        pvalloc_scen = pvalloc_scen_list[i]

        gridnode_df = pd.read_parquet(f'{scen_data_path}/gridnode_df.parquet') 
        gridnode_df['t_int'] = gridnode_df['t'].str.extract('t_(\d+)').astype(int)
        gridnode_df.sort_values(by=['t_int'], inplace=True)

        # plot ----------------
        fig = px.line(gridnode_df, x='t', y='pvprod_kW', color = 'grid_node',labels= f'{scen}, wy: {pvalloc_scen["weather_specs"]["weather_year"]})' )
        
        
    fig.update_layout(
        xaxis_title='Hour of Year',
        yaxis_title='Production (kWh)',
        legend_title='Node ID',
        title = f'Production per Hour of Year, by Scenario (kW)'
    )
    if plot_show:
        fig.show()

    fig.write_html(f'{data_path}/output/visualizations/plot_agg_line_productionHOY_per_node.html')



    # plot agg - line: Grid Premium per Hour of Year ============================
    fig = go.Figure()
    for i, scen in enumerate(scen_dir_export_list):
        # setup + import ----------
        scen_data_path = f'{data_path}/output/{scen}'
        pvalloc_scen = pvalloc_scen_list[i]

        gridprem_ts = pd.read_parquet(f'{scen_data_path}/gridprem_ts.parquet') 
        gridprem_ts['t_int'] = gridprem_ts['t'].str.extract('t_(\d+)').astype(int)
        gridprem_ts.sort_values(by=['t_int'], inplace=True)

        # plot ----------------
        fig = px.line(gridprem_ts, x='t', y='prem_Rp_kWh', color = 'grid_node',labels = f'{scen}, wy: {pvalloc_scen["weather_specs"]["weather_year"]})' )
    
    fig.update_layout(
        xaxis_title='Hour of Year',
        yaxis_title='Grid Premium (CHF)',
        legend_title='Node ID',
        title = f'Grid Premium per Hour of Year, by Scenario (CHF)'
    )
    if plot_show:
        fig.show()

    fig.write_html(f'{data_path}/output/visualizations/plot_agg_line_gridPremiumHOY_per_node.html')









# plot ind - ???: Buliding Charachteristics ===========================







