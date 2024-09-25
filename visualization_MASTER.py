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

    # external packages
    import os as os
    import pandas as pd
    import geopandas as gpd
    import numpy as np
    import json 
    import plotly.express as px
    import plotly.graph_objects as go
    import shapely
    import copy
    import glob

    from datetime import datetime
    from pprint import pformat
    from shapely.geometry import Polygon, MultiPolygon

    # own packages and functions
    import pv_allocation.default_settings as pvalloc_default_sett
    import visualisations.defaults_settings as visual_default_sett

    from auxiliary_functions import chapter_to_logfile, checkpoint_to_logfile
    from pv_allocation.default_settings import *
    from visualisations.defaults_settings import *


def visualization_MASTER(pvalloc_scenarios_func, visual_settings_func):
    # SETTINGS ------------------------------------------------------------------------------------------------------
    if not isinstance(pvalloc_scenarios_func, dict):
        print(' USE LOCAL SETTINGS - DICT  ')
        pvalloc_scenarios = pvalloc_default_sett.get_default_pvalloc_settings()
    else:
        pvalloc_scenarios = pvalloc_scenarios_func

    if not isinstance(visual_settings_func, dict) or visual_settings_func == {}:
        visual_settings = visual_default_sett.get_default_visual_settings()
    else:
        visual_settings = visual_settings_func

    
    # SETUP ------------------------------------------------------------------------------------------------------
    
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
    plot_show = visual_settings['plot_show']
    default_zoom_year = visual_settings['default_zoom_year']



    # EXTRACT SCENARIO INFORMATION + IMPORT -----------------------------------------------------------------------------------

    # scen settings ----------------
    scen_dir_export_list, scen_dir_import_list, T0_prediction_list, months_prediction_list = [], [], [], []
    T0_prediction_list, months_lookback_list, months_prediction_list = [], [], []
    pvalloc_scen_list = []
    for key, val in pvalloc_scenarios.items():
        scen_dir_export_list.append(val['name_dir_export'])
        scen_dir_import_list.append(val['name_dir_import'])
        T0_prediction_list.append(val['T0_prediction'])
        months_prediction_list.append(val['months_prediction'])
        pvalloc_scen_list.append(val)

    # visual settings ----------------  
    plot_show = visual_settings['plot_show']
    default_zoom_year = visual_settings['default_zoom_year']
    default_zoom_hour = visual_settings['default_zoom_hour']


    # UNIVERSIAL FUNCTIONS ------------------------------------------------------------------------------------------------------
    if True:
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
        
        # universal func for plot T0 tick -----
        def add_T0_tick_to_plot(fig, T0_prediction, df, df_col):
            fig.add_shape(
                # Line Vertical
                dict(
                    type="line",
                    x0=T0_prediction,
                    y0=0,
                    x1=T0_prediction,
                    y1= df[df_col].max(),  # Dynamic height
                    line=dict(color="black", width=1, dash="dot"),
                )
            )
            fig.add_annotation(
                x=  T0_prediction,
                y= df[df_col].max(),
                text="T0 Prediction",
                showarrow=False,
                yshift=10
            )
            return fig

        # universial func to set default plot zoom -----
        def set_default_fig_zoom_year(fig, zoom_window, df, datecol):
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
        
        def set_default_fig_zoom_hour(fig, zoom_window):
            start_zoom, end_zoom = zoom_window[0], zoom_window[1]
            fig.update_layout(
                xaxis_range=[start_zoom, end_zoom])
            return fig

        # Function to flatten geometries to 2D (ignoring Z-dimension) -----
        def flatten_geometry(geom):
            if geom.has_z:
                if geom.geom_type == 'Polygon':
                    exterior = [(x, y) for x, y, z in geom.exterior.coords]
                    interiors = [[(x, y) for x, y, z in interior.coords] for interior in geom.interiors]
                    return Polygon(exterior, interiors)
                elif geom.geom_type == 'MultiPolygon':
                    return MultiPolygon([flatten_geometry(poly) for poly in geom.geoms])
            return geom


    # PLOT INDIVIDUAL SCEN ------------------------------------------------------------------------------------------------------
       
    # plot ind - line: Production + Feedin HOY per Node ============================
    if visual_settings['plot_ind_line_productionHOY_per_node']:
        checkpoint_to_logfile(f'plot_ind_line_productionHOY_per_node', log_name)
        i, scen = 0, scen_dir_export_list[0]
        for i, scen in enumerate(scen_dir_export_list):
            # setup + import ----------
            scen_data_path = f'{data_path}/output/{scen}'
            pvalloc_scen = pvalloc_scen_list[i]

            gridnode_df = pd.read_parquet(f'{scen_data_path}/gridnode_df.parquet')
            # gridnode_df = pd.read_parquet("C:\Models\OptimalPV_RH_data\output\pvalloc_run\gridnode_df.parquet")
            gridnode_df['t_int'] = gridnode_df['t'].str.extract(r't_(\d+)').astype(int)
            gridnode_df.sort_values(by=['t_int'], inplace=True)

            # plot ----------------
            if 'info_source' in gridnode_df.columns:
                nodes = gridnode_df['grid_node'].unique()
                pvsources = gridnode_df['info_source'].unique()
                fig = go.Figure()

                for node in nodes:
                    for source in pvsources:
                        if source != '':
                        # if True:
                            filter_df = gridnode_df.loc[
                                (gridnode_df['grid_node'] == node) & (gridnode_df['info_source'] == source)].copy()
                            
                            # fig.add_trace(go.Scatter(x=filter_df['t'], y=filter_df['pvprod_kW'], name=f'Prod Node: {node}, Source: {source}'))
                            fig.add_trace(go.Scatter(x=filter_df['t'], y=filter_df['feedin_kW'], name=f'{node} - feedin (all),  Source: {source}'))
                            fig.add_trace(go.Scatter(x=filter_df['t'], y=filter_df['feedin_kW_taken'], name= f'{node} - feedin_taken, Source: {source}'))
                            fig.add_trace(go.Scatter(x=filter_df['t'], y=filter_df['feedin_kW_loss'], name=f'{node} - feedin_loss, Source: {source}'))

                
                # gridnode_total_df = gridnode_df.groupby(['t', 't_int'])['feedin_kW'].sum().reset_index()
                gridnode_total_df = gridnode_df.groupby(['t', 't_int']).agg({'pvprod_kW': 'sum', 'feedin_kW': 'sum','feedin_kW_taken': 'sum','feedin_kW_loss': 'sum'}).reset_index()
                gridnode_total_df.sort_values(by=['t_int'], inplace=True)
                fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['pvprod_kW'], name='Total production', line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['feedin_kW'], name='Total feedin', line=dict(color='black', width=2)))
                fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['feedin_kW_taken'], name='Total feedin_taken', line=dict(color='green', width=2)))
                fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['feedin_kW_loss'], name='Total feedin_loss', line=dict(color='red', width=2)))
            

            fig.update_layout(
                xaxis_title='Hour of Year',
                yaxis_title='Production / Feedin (kW)',
                legend_title='Node ID',
                title = f'Production per node (kW, weather year: {pvalloc_scen["weather_specs"]["weather_year"]}, self consum. rate{pvalloc_scen["tech_economic_specs"]["self_consumption_ifapplicable"]})'
            )


            fig = add_scen_name_to_plot(fig, scen, pvalloc_scen_list[i])
            fig = set_default_fig_zoom_hour(fig, default_zoom_hour)

            if plot_show:
                fig.show() 

            fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_line_productionHOY_per_node.html')


    # plot ind - line:
    #     plot ind - line: Installed Capacity per Month ===========================
    #     plot ind - line: Installed Capacity per BFS   ===========================
    if visual_settings['plot_ind_line_installedCap_per_month'] or visual_settings['plot_ind_line_installedCap_per_BFS']:
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
            if visual_settings['plot_ind_line_installedCap_per_month']: 
                checkpoint_to_logfile(f'plot_ind_line_installedCap_per_month', log_name)
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
                fig1 = set_default_fig_zoom_year(fig1, default_zoom_year, capa_year_df, 'BeginOp_year')
                if plot_show:
                    fig1.show()
                fig1.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_line_installedCap_per_month.html')

            # plot ind - line: Installed Capacity per BFS ===========================
            if visual_settings['plot_ind_line_installedCap_per_BFS']: 
                checkpoint_to_logfile(f'plot_ind_line_installedCap_per_BFS', log_name)
                capa_bfs_df = pvinst_df.copy()
                gm_gdf = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp/swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET.shp')
                gm_gdf.rename(columns={'BFS_NUMMER': 'bfs'}, inplace=True)
                capa_bfs_df = capa_bfs_df.merge(gm_gdf[['bfs', 'NAME']], on='bfs', how = 'left' )
                capa_bfs_df['BeginOp_month'] = capa_bfs_df['BeginOp'].dt.to_period('M')
                capa_bfs_month_df = capa_bfs_df.groupby(['BeginOp_month', 'bfs'])['TotalPower'].sum().reset_index().copy()
                capa_bfs_month_df['BeginOp_month'] = capa_bfs_month_df['BeginOp_month'].dt.to_timestamp()

                capa_bfs_df['BeginOp_year'] = capa_bfs_df['BeginOp'].dt.to_period('Y')
                capa_bfs_year_df = capa_bfs_df.groupby(['BeginOp_year', 'bfs'])['TotalPower'].sum().reset_index().copy()
                capa_bfs_year_df['BeginOp_year'] = capa_bfs_year_df['BeginOp_year'].dt.to_timestamp()

                # plot ----------------
                fig2 = go.Figure()
                for bfs in capa_bfs_month_df['bfs'].unique():
                    name = gm_gdf.loc[gm_gdf['bfs'] == bfs, 'NAME'].values[0]
                    subdf = capa_bfs_month_df.loc[capa_bfs_month_df['bfs'] == bfs].copy()
                    fig2.add_trace(go.Scatter(x=subdf['BeginOp_month'], y=subdf['TotalPower'], name=f'{name}', legendgroup = 'By Month',  mode = 'lines'))

                for bfs in capa_bfs_year_df['bfs'].unique():
                    name = gm_gdf.loc[gm_gdf['bfs'] == bfs, 'NAME'].values[0]
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
                fig2 = set_default_fig_zoom_year(fig2, default_zoom_year, capa_bfs_year_df, 'BeginOp_year')
                if plot_show:
                    fig2.show()
                fig2.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_line_installedCap_per_BFS.html')


    # plot ind - hist: NPV possible PV inst before / after ============================
    if visual_settings['plot_ind_hist_NPV_freepartitions']:
        checkpoint_to_logfile(f'plot_ind_hist_NPV_freepartitions', log_name)
        i, scen = 0, scen_dir_export_list[0]
        for i, scen in enumerate(scen_dir_export_list):
            # setup + import ----------
            scen_data_path = f'{data_path}/output/{scen}'
            pvalloc_scen = pvalloc_scen_list[i]

            npv_df_paths = glob.glob(f'{scen_data_path}/pred_npv_inst_by_M/npv_df_*.parquet')
            periods_list = [pd.to_datetime(path.split('npv_df_')[-1].split('.parquet')[0]) for path in npv_df_paths]
            before_period, after_period = min(periods_list), max(periods_list)

            npv_df_before = pd.read_parquet(f'{scen_data_path}/pred_npv_inst_by_M/npv_df_{before_period.to_period("M")}.parquet')
            npv_df_after  = pd.read_parquet(f'{scen_data_path}/pred_npv_inst_by_M/npv_df_{after_period.to_period("M")}.parquet')

            # plot ----------------
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=npv_df_before['NPV_uid'], name='Before Allocation Algorithm', opacity=0.75))
            fig.add_trace(go.Histogram(x=npv_df_after['NPV_uid'], name='After Allocation Algorithm', opacity=0.75))

            fig.update_layout(
                xaxis_title=f'Net Present Value (NPV, interest rate: {pvalloc_scen["tech_economic_specs"]["interest_rate"]}, maturity: {pvalloc_scen["tech_economic_specs"]["invst_maturity"]} yr)',
                yaxis_title='Frequency',
                title = f'NPV Distribution of possible PV installations, first / last year (weather year: {pvalloc_scen["weather_specs"]["weather_year"]})',
                barmode = 'overlay')
            fig.update_traces(bingroup=1, opacity=0.75)

            fig = add_scen_name_to_plot(fig, scen, pvalloc_scen_list[i])
            if plot_show:
                fig.show()
            fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_hist_NPV_freepartitions.html')


    # plot ind - map:  Model PV topology ========================
    default_map_zoom = visual_settings['default_map_zoom']
    uniform_municip_color = visual_settings['map_topo_specs']['uniform_municip_color']
    map_topo_specs = visual_settings['map_topo_specs']


    # map ind - topo_egid ============================
    if visual_settings['plot_ind_map_topo_egid']:
        checkpoint_to_logfile(f'plot_ind_map_topo_egid', log_name)

        for i, scen in enumerate(scen_dir_export_list):
            
            # get pvinst_gdf ----------------
            if True: 
                scen_data_path = f'{data_path}/output/{scen}'
                pvalloc_scen = pvalloc_scen_list[i]
                
                # import 
                gwr_gdf = gpd.read_file(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/gwr_gdf.geojson')
                gm_gdf = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp/swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET.shp')

                topo  = json.load(open(f'{scen_data_path}/topo_egid.json', 'r'))
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
                T0_prediction = T0_prediction_list[0]
                months_prediction = months_prediction_list[0]
                pvalloc_scen = pvalloc_scen_list[i]

                # transformations
                gm_gdf['BFS_NUMMER'] = gm_gdf['BFS_NUMMER'].astype(str)
                gm_gdf = gm_gdf.loc[gm_gdf['BFS_NUMMER'].isin(pvinst_df['bfs'].unique())].copy()
                date_cols = [col for col in gm_gdf.columns if (gm_gdf[col].dtype == 'datetime64[ns]') or (gm_gdf[col].dtype == 'datetime64[ms]')]
                gm_gdf.drop(columns=date_cols, inplace=True)
                
                # add map relevant columns
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
                fig0 = px.choropleth_mapbox(
                    gm_gdf,
                    geojson=geojson,
                    locations="BFS_NUMMER",  # Link BFS_NUMMER for color and location
                    featureidkey="properties.BFS_NUMMER",  # This must match the GeoJSON's property for BFS_NUMMER
                    color_discrete_sequence=[uniform_municip_color],  # Apply the single color to all shapes
                    hover_name="hover_text",  # Use the new column for hover text
                    mapbox_style="carto-positron",  # Basemap style
                    center={"lat": 47.41, "lon": 7.49},  # Center the map on the region
                    zoom=default_map_zoom,  # Adjust zoom as needed
                    opacity=map_topo_specs['shape_opacity'],   # Opacity to make shapes and basemap visible    
                )
                # Update layout for borders and title
                fig0.update_layout(
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
                # fig.show()

            # topo map ----------------
            if True:
                fig1 = copy.deepcopy(fig0)
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

                
                # subset inst_gdf for different traces in map plot
                pvinst_gdf['hover_text'] = pvinst_gdf.apply(lambda row: f"EGID: {row['EGID']}<br>BeginOp: {row['BeginOp']}<br>TotalPower: {row['TotalPower']}<br>gklas: {row['gklas']}<br>node: {row['node']}<br>pvtarif: {row['pvtarif']}<br>elecpri: {row['elecpri']}<br>elecpri_info: {row['elecpri_info']}", axis=1)

                subinst1_gdf, subinst2_gdf, subinst3_gdf  = pvinst_gdf.copy(), pvinst_gdf.copy(), pvinst_gdf.copy()
                subinst1_gdf, subinst2_gdf, subinst3_gdf = subinst1_gdf.loc[(subinst1_gdf['inst_TF'] == True) & (subinst1_gdf['info_source'] == 'pv_df')], subinst2_gdf.loc[(subinst2_gdf['inst_TF'] == True) & (subinst2_gdf['info_source'] == 'alloc_algorithm')], subinst3_gdf.loc[(subinst3_gdf['inst_TF'] == False)]

                # Add the points using Scattermapbox
                fig1.add_trace(go.Scattermapbox(lat=subinst1_gdf.geometry.y,lon=subinst1_gdf.geometry.x, mode='markers',
                    marker=dict(
                        size=map_topo_specs['point_size_pv'],
                        color=map_topo_specs['point_color_pv_df'],
                        opacity=map_topo_specs['point_opacity']
                    ),
                    name = 'house w pv (real)',
                    text=subinst1_gdf['hover_text'],
                    hoverinfo='text'
                ))
                fig1.add_trace(go.Scattermapbox(lat=subinst2_gdf.geometry.y,lon=subinst2_gdf.geometry.x, mode='markers',
                    marker=dict(
                        size=map_topo_specs['point_size_pv'],
                        color=map_topo_specs['point_color_alloc_algo'],
                        opacity=map_topo_specs['point_opacity']
                    ),
                    name = 'house w pv (predicted)',
                    text=subinst2_gdf['hover_text'],
                    hoverinfo='text'
                ))
                fig1.add_trace(go.Scattermapbox(lat=subinst3_gdf.geometry.y,lon=subinst3_gdf.geometry.x, mode='markers',
                    marker=dict(
                        size=map_topo_specs['point_size_rest'],
                        color=map_topo_specs['point_color_rest'],
                        opacity=map_topo_specs['point_opacity']
                    ),
                    name = 'house w/o pv',
                    text=subinst3_gdf['hover_text'],
                    hoverinfo='text'
                ))
                
                # Update layout
                fig1.update_layout(
                        title=f"Map of model PV Topology ({scen})",
                        mapbox=dict(
                            style="carto-positron",
                            center={"lat": 47.41, "lon": 7.49},
                            zoom=default_map_zoom
                        )
                    )

                if plot_show:
                    fig1.show()
                fig1.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_map_topo_egid.html')


    # V - NOT WORKING YET - V
    # map_ind_production ============================ 
    if visual_settings['map_ind_production']:
   
        # import
        solkat_gdf = gpd.read_file(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/solkat_gdf.geojson', rows =1000)
        solkat_gdf['DF_UID'] = solkat_gdf['DF_UID'].astype(int).astype(str)
        solkat_gdf.dtypes
        solkat_gdf.rename(columns={'DF_UID': 'df_uid'}, inplace=True)

        topo_subdf_paths = glob.glob(f'{scen_data_path}/topo_time_subdf/*.parquet')
        subdf_dfuid_list = []

        path = topo_subdf_paths[0]
        for i, path in enumerate(topo_subdf_paths):
            subdf = pd.read_parquet(path)
            subdf_dfuid = subdf.groupby('df_uid').agg({'pvprod_kW': 'sum'}).reset_index()
            subdf_dfuid_list.append(subdf_dfuid)

        aggdf_dfuid = pd.concat(subdf_dfuid_list)

        # merge
        solkat_gdf = solkat_gdf.merge(aggdf_dfuid, on='df_uid', how='left')

        # plot ----------

        solkat_gdf['hover_text'] = solkat_gdf.apply(lambda row: f"DF_UID: {row['df_uid']}<br>EGID: {row['EGID']}<br>pvprod_kW: {row['pvprod_kW']}<br>FLAECHE: {row['FLAECHE']}<br>AUSRICHTUNG: {row['AUSRICHTUNG']}<br>NEIGUNG: {row['NEIGUNG']}", axis=1)
        
        solkat_gdf['geometry'] = solkat_gdf['geometry'].apply(flatten_geometry)
        solkat_gdf = solkat_gdf.to_crs('EPSG:4326')
        solkat_gdf = solkat_gdf[solkat_gdf.is_valid]
        geojson = solkat_gdf.__geo_interface__

        # Add the shapes to figX
        fig2 = px.choropleth_mapbox(
            solkat_gdf, 
            geojson=geojson,
            locations="df_uid",
            featureidkey="properties.df_uid",
            color="pvprod_kW",
            color_continuous_scale="Viridis",
            range_color=(0, 100),
            mapbox_style="carto-positron",
            center={"lat": 47.41, "lon": 7.49},
            zoom=default_map_zoom,
            opacity=0.5,
            hover_name="hover_text",
            title=f"Map of production per DF_UID ({scen})",
        )
        if plot_show:
            fig2.show()
        fig2.write_html(f'{data_path}/output/visualizations/{scen}__map_ind_production.html')


    # PLOT AGGREGATED SCEN ------------------------------------------------------------------------------------------------------
    if len(list(set(T0_prediction_list))) ==1:
        T0_pred_agg = T0_prediction_list[0]


    # plot agg - line: Installed Capacity per Month ============================
    if visual_settings['plot_agg_line_installedCap_per_month']:
        checkpoint_to_logfile(f'plot_agg_line_installedCap_per_month', log_name)
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
            fig.add_trace(go.Scatter(x=pvinst_month_df['BeginOp_month'], y=pvinst_month_df['TotalPower'], name=f'built + predicted ({scen})',  mode='lines+markers', legendgroup = 'by Month', legendgrouptitle_text= 'by Month'))
            # fig.add_trace(go.Scatter(x=pvinst_month_built['BeginOp_month'], y=pvinst_month_built['TotalPower'], line = dict(color = 'deepskyblue'), name=f'built (month)',  mode='lines+markers', legendgroup = scen, legendgrouptitle_text=scen))
            # fig.add_trace(go.Scatter(x=capa_month_predicted['BeginOp_month'], y=capa_month_predicted['TotalPower'], line = dict(color = 'navy'), name=f'predicted (month)',  mode='lines+markers', legendgroup = scen, legendgrouptitle_text=scen))

            fig.add_trace(go.Scatter(x=pvinst_year_df['BeginOp_year'], y=pvinst_year_df['TotalPower'], name=f'built + predicted ({scen})',  mode='lines+markers', legendgroup = 'by Year', legendgrouptitle_text= 'by Year'))
            # fig.add_trace(go.Scatter(x=pvinst_year_built['BeginOp_year'], y=pvinst_year_built['TotalPower'], line = dict(color = 'lightgreen'), name=f'built (year)',  mode='lines+markers', legendgroup = scen, legendgrouptitle_text=scen))
            # fig.add_trace(go.Scatter(x=pvinst_year_predicted['BeginOp_year'], y=pvinst_year_predicted['TotalPower'], line = dict(color = 'forestgreen'), name=f'predicted (year)',  mode='lines+markers', legendgroup = scen, legendgrouptitle_text=scen))

        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Installed Capacity (kW)',
            legend_title='Scenarios',
            title = f'Agg. Installed Capacity per Month (weather year: {pvalloc_scen["weather_specs"]["weather_year"]})'
        )

        fig = add_T0_tick_to_plot(fig, T0_pred_agg, pvinst_year_df, 'TotalPower')
        fig = set_default_fig_zoom_year(fig, default_zoom_year, pvinst_year_df, 'BeginOp_year')
        if plot_show:
            fig.show()

        fig.write_html(f'{data_path}/output/visualizations/plot_agg_line_installedCap_per_month.html')


    # plot agg - line: Grid Premium per Hour of Year ============================
    if visual_settings['plot_agg_line_gridPremiumHOY_per_node']:
        checkpoint_to_logfile(f'plot_agg_line_gridPremiumHOY_per_node', log_name)
        fig = go.Figure()
        for i, scen in enumerate(scen_dir_export_list):
            # setup + import ----------
            scen_data_path = f'{data_path}/output/{scen}'
            pvalloc_scen = pvalloc_scen_list[i]

            gridprem_ts = pd.read_parquet(f'{scen_data_path}/gridprem_ts.parquet') 
            gridprem_ts['t_int'] = gridprem_ts['t'].str.extract(r't_(\d+)').astype(int)
            gridprem_ts.sort_values(by=['t_int', 'grid_node'], inplace=True)

            # plot ----------------
            for grid_node in gridprem_ts['grid_node'].unique():
                node_df = gridprem_ts[gridprem_ts['grid_node'] == grid_node]
                fig.add_trace(go.Scatter(
                    x=node_df['t_int'], 
                    y=node_df['prem_Rp_kWh'], 
                    mode='lines',
                    name=f'{scen} - {grid_node}',  # Include both scen and grid_node in the legend
                    showlegend=True
            ))

        fig.update_layout(
            xaxis_title='Hour of Year',
            yaxis_title='Grid Premium (CHF)',
            legend_title='Node ID',
            title = f'Agg Grid Premium per Hour of Year, by Scenario (CHF)'
        )
        fig = set_default_fig_zoom_hour(fig, default_zoom_hour)
        
        if plot_show:
            fig.show()
        fig.write_html(f'{data_path}/output/visualizations/plot_agg_line_gridPremiumHOY_per_node.html')


    # plot agg - line: Grid Structure ============================
    if visual_settings['plot_agg_line_gridpremium_structure']:
        checkpoint_to_logfile(f'plot_agg_line_gridpremium_structure', log_name)
        fig = go.Figure()
        for i, scen in enumerate(scen_dir_export_list):
            # setup + import ----------
            scen_data_path = f'{data_path}/output/{scen}'
            pvalloc_scen = pvalloc_scen_list[i]

            gridtiers = pvalloc_scen['gridprem_adjustment_specs']['tiers']
            gridtiers_colnames = pvalloc_scen['gridprem_adjustment_specs']['colnames']

            data = [(k, v[0], v[1]) for k, v in gridtiers.items()]
            gridtiers_df = pd.DataFrame(data, columns=gridtiers_colnames) 

            # plot ----------------
            if 'gridprem_plusRp_kWh'  in gridtiers_df.columns:
                fig.add_trace(go.Scatter(x=gridtiers_df['used_node_capa_rate'], y=gridtiers_df['gridprem_plusRp_kWh'], name=f'{scen}', mode='lines+markers'))
            else:
                fig.add_trace(go.Scatter(x=gridtiers_df['used_node_capa_rate'], y=gridtiers_df['gridprem_Rp_kWh'], name=f'{scen}', mode='lines+markers'))

        fig.update_layout(
            xaxis_title=r'Used Node Capacity Rate (% of individual node capacity)',
            yaxis_title='Grid Premium (Rp)',
            legend_title='Scenarios',
            title = f'Grid Premium Structure, by Scenario (Rp)'
        )
        if plot_show:
            fig.show()
        fig.write_html(f'{data_path}/output/visualizations/plot_agg_line_gridpremium_structure.html')


    # plot agg - line: PV Production / Feedin per Hour of Year ============================
    if visual_settings['plot_agg_line_productionHOY_per_node']:
        checkpoint_to_logfile(f'plot_agg_line_productionHOY_per_node', log_name)
        fig = go.Figure()
        i, scen = 0, scen_dir_export_list[0]
        for i, scen in enumerate(scen_dir_export_list):
            # setup + import ----------
            scen_data_path = f'{data_path}/output/{scen}'
            pvalloc_scen = pvalloc_scen_list[i]

            gridnode_df = pd.read_parquet(f'{scen_data_path}/gridnode_df.parquet') 
            gridnode_df['t_int'] = gridnode_df['t'].str.extract(r't_(\d+)').astype(int)
            gridnode_df.sort_values(by=['t_int'], inplace=True)

            # plot ----------------
            if False:
            # if 'pvsource' in gridnode_df.columns:
                pvsources = gridnode_df['pvsource'].unique()

                # gridnode_source_df = gridnode_df.groupby(['t', 't_int', 'pvsource']).agg({'pvprod_kW': 'sum'}).reset_index()
                gridnode_source_df = gridnode_df.groupby(['t', 't_int', 'pvsource']).agg({'pvprod_kW': 'sum', 'feedin_kW': 'sum', 'feedin_kW_taken': 'sum', 'feedin_kW_loss': 'sum'}).reset_index()
                gridnode_source_df.sort_values(by=['t_int'], inplace=True)
                for source in pvsources:
                    if source != '':
                        filter_df = gridnode_source_df.loc[gridnode_source_df['pvsource'] == source].copy()
                        fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['feedin_kW'], name = f'{node} - feedin (all), Source: {source}', mode = 'lines'))
                        fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['feedin_kW_taken'], name = f'{node} - feedin_taken, Source: {source}', mode = 'lines'))
                        fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['feedin_kW_loss'], name = f'{node} - feedin_loss, Source: {source}', mode = 'lines'))        
            
            # add total production
            gridnode_total_df = gridnode_df.groupby(['t', 't_int']).agg({'pvprod_kW': 'sum', 'feedin_kW': 'sum','feedin_kW_taken': 'sum','feedin_kW_loss': 'sum'}).reset_index()
            gridnode_total_df.sort_values(by=['t_int'], inplace=True)
            fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['pvprod_kW'], name=f'{scen}: Total production'))
            fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['feedin_kW'], name=f'{scen}: Total feedin'))
            fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['feedin_kW_taken'], name=f'{scen}: Total feedin_taken'))
            fig.add_trace(go.Scatter(x=gridnode_total_df['t'], y=gridnode_total_df['feedin_kW_loss'], name=f'{scen}: Total feedin_loss'))

                             
        fig.update_layout(
            xaxis_title='Hour of Year',
            yaxis_title='Production / Feedin (kW)',
            legend_title='Node ID',
            title = f'Agg. Production per Hour of Year, by Scenario (kW)'
        )
        fig = set_default_fig_zoom_hour(fig, default_zoom_hour)
        if plot_show:
            fig.show()

        fig.write_html(f'{data_path}/output/visualizations/plot_agg_line_productionHOY_per_node.html')


 



