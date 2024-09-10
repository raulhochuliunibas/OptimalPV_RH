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
    import glob
    import shutil
    import winsound
    import subprocess
    import pprint
    import json 
    import plotly.express as px
    import plotly.graph_objects as go

    from datetime import datetime
    from pprint import pformat


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
    plot_ind_line_productionHOY_per_node = True
    plot_ind_line_installedCap_per_month = True
    plot_ind_line_installedCap_per_BFS = True
    plot_ind_map_coveredArea = True

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



    # PLOT INDIVIDUAL SCEN --------------------------------------------------------------------


    # universal func for plot adjustments ===========================
    def add_scen_name_to_plot(fig, scen, pvalloc_scen):
        # add scenario name
        fig.add_annotation(
            text=f'Scen: {scen}, (start T0: {pvalloc_scen["T0_prediction"].split(" ")[0]}, {pvalloc_scen["months_prediction"]} prediction months)',
            xref="paper", yref="paper",
            x=0.5, y=1.05, showarrow=False,
            font=dict(size=12)
        )
        return fig
    
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
    # universal func for plot T0 tick ===========================
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



    
        
    # plot ind - line: Production HOY per Node ============================
    if plot_ind_line_productionHOY_per_node:
        i, scen = 0, scen_dir_export_list[0]
        for i, scen in enumerate(scen_dir_export_list):
            # setup + import ----------
            scen_data_path = f'{data_path}/output/{scen}'
            pvalloc_scen = pvalloc_scen_list[i]

            if not os.path.exists(f'{data_path}/output/visualizations/{scen}'):
                os.makedirs(f'{data_path}/output/visualizations/{scen}')
            elif os.path.exists(f'{data_path}/output/visualizations/{scen}'):
                shutil.rmtree(f'{data_path}/output/visualizations/{scen}')
                os.makedirs(f'{data_path}/output/visualizations/{scen}')

            gridnode_df = pd.read_parquet(f'{scen_data_path}/gridnode_df.parquet') 
            gridnode_df['t_int'] = gridnode_df['t'].str.extract('t_(\d+)').astype(int)
            gridnode_df.sort_values(by=['t_int'], inplace=True)

            # plot ----------------
            fig = px.line(gridnode_df, x='t', y='pvprod_kW', color = 'grid_node', title = f'prod kWh, weather year: {pvalloc_scen["weather_specs"]["weather_year"]})' )
    
            fig.update_layout(
                xaxis_title='Hour of Year',
                yaxis_title='Production (kWh)',
                legend_title='Node ID',
            )

            fig = add_scen_name_to_plot(fig, scen, pvalloc_scen_list[i])
            if plot_show:
                fig.show() 

            fig.write_html(f'{data_path}/output/visualizations/{scen}/plot_ind_line_productionHOY_per_node.html')


    # plot ind - line:
    #     plot ind - line: Installed Capacity per Month ===========================
    #     plot ind - line: Installed Capacity per BFS   ===========================

    
    if plot_ind_line_installedCap_per_month:
        for i, scen in enumerate(scen_dir_export_list):
            scen_data_path = f'{data_path}/output/{scen}'
            T0_prediction = T0_prediction_list[0]
            months_prediction = months_prediction_list[0]
            pvalloc_scen = pvalloc_scen_list[i]


            if not os.path.exists(f'{data_path}/output/visualizations/{scen}'):
                os.makedirs(f'{data_path}/output/visualizations/{scen}')
            elif os.path.exists(f'{data_path}/output/visualizations/{scen}'):
                shutil.rmtree(f'{data_path}/output/visualizations/{scen}')
                os.makedirs(f'{data_path}/output/visualizations/{scen}')

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
            pvinst_df['bf'] = pvinst_df['bfs'].astype(str)

            # REMOVE some old historic data, because not interesting on Plot
            pvinst_df = pvinst_df.loc[pvinst_df['BeginOp'] > pd.to_datetime('2015-01-01')]


            # plot ind - line: Installed Capacity per Month ===========================
            if True: 
                capa_inst_df = pvinst_df.copy()
                capa_inst_df['BeginOp_month'] = capa_inst_df['BeginOp'].dt.to_period('M')
                capa_month_df = capa_inst_df.groupby(['BeginOp_month', 'info_source'])['TotalPower'].sum().reset_index().copy()
                capa_month_df['BeginOp_month'] = capa_month_df['BeginOp_month'].dt.to_timestamp()
                capa_month_built = capa_month_df.loc[capa_month_df['info_source'] == 'pv_df'].copy()
                capa_month_predicted = capa_month_df.loc[capa_month_df['info_source'] == 'alloc_algorithm'].copy()

                capa_inst_df['BeginOp_year'] = capa_inst_df['BeginOp'].dt.to_period('Y')
                pvinst_year_df = capa_inst_df.groupby(['BeginOp_year', 'info_source'])['TotalPower'].sum().reset_index().copy()
                pvinst_year_df['BeginOp_year'] = pvinst_year_df['BeginOp_year'].dt.to_timestamp()
                pvinst_year_built = pvinst_year_df.loc[pvinst_year_df['info_source'] == 'pv_df'].copy()
                pvinst_year_predicted = pvinst_year_df.loc[pvinst_year_df['info_source'] == 'alloc_algorithm'].copy()

                # plot ----------------
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'], line = dict(color = 'navy'),name='built + predicted (month)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=capa_month_built['BeginOp_month'], y=capa_month_built['TotalPower'], line = dict(color = 'deepskyblue'), name='built (month)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=capa_month_predicted['BeginOp_month'], y=capa_month_predicted['TotalPower'], line = dict(color = 'cornflowerblue'), name='predicted (month)', mode='lines+markers'))

                fig1.add_trace(go.Scatter(x=pvinst_year_df['BeginOp_year'], y=pvinst_year_df['TotalPower'], line = dict(color = 'forestgreen'), name='built + predicted (year)', mode='lines+markers',))
                fig1.add_trace(go.Scatter(x=pvinst_year_built['BeginOp_year'], y=pvinst_year_built['TotalPower'], line = dict(color = 'lightgreen'), name='built (year)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=pvinst_year_predicted['BeginOp_year'], y=pvinst_year_predicted['TotalPower'], line = dict(color = 'limegreen'), name='predicted (year)', mode='lines+markers'))

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
                        y1=max(capa_month_df['TotalPower'].max(), pvinst_year_df['TotalPower'].max()),  # Dynamic height
                        line=dict(color="black", width=1, dash="dot"),
                    )
                )
                fig1.add_annotation(
                    x=  T0_prediction,
                    y=max(capa_month_df['TotalPower'].max(), pvinst_year_df['TotalPower'].max()),
                    text="T0 Prediction",
                    showarrow=False,
                    yshift=10
                )

                fig1 = add_scen_name_to_plot(fig, scen, pvalloc_scen_list[i])
                if plot_show:
                    fig1.show()
                fig1.write_html(f'{data_path}/output/visualizations/{scen}/plot_ind_line_installedCap_per_month.html')


            # plot ind - line: Installed Capacity per BFS ===========================
            if True: 
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

                fig2 = add_scen_name_to_plot(fig2, scen, pvalloc_scen_list[i])
                if plot_show:
                    fig2.show()
                fig2.write_html(f'{data_path}/output/visualizations/{scen}/plot_ind_line_installedCap_per_BFS.html')


    # plot ind - map:  Covered Area of Allocation Model ========================
    i = 0
    if plot_ind_map_coveredArea:
        scen = scen_dir_export_list[0]

        # scen_data_path = f'{data_path}/output/{scen}'

        # if not os.path.exists(f'{data_path}/output/visualizations/{scen}'):
        #     os.makedirs(f'{data_path}/output/visualizations/{scen}')
        # elif os.path.exists(f'{data_path}/output/visualizations/{scen}'):
        #     shutil.rmtree(f'{data_path}/output/visualizations/{scen}')
        #     os.makedirs(f'{data_path}/output/visualizations/{scen}')


        # # import data
        # topo = json.load(open(f'{scen_data_path}/topo_egid.json', 'r'))
        # egid_list, bfs_list, pv_inst_id_list = [], [], []

        # for k,v, in topo.items():
        #     egid_list.append(k)
        #     bfs_list.append(v['gwr_info']['bfs'])
        #     if v['pv_inst']['inst_TF'] == True:
        #         pv_inst_id_list.append(v['pv_inst']['xtf_id'])
        #     else:
        #         pv_inst_id_list.append('')
        # df = pd.DataFrame({'EGID': egid_list, 'bfs': bfs_list, 'pv_inst_id': pv_inst_id_list})

        # gm_shp = gpd.read_file(f'{data_path}/input\swissboundaries3d_2023-01_2056_5728.shp\swissBOUNDARIES3D_1_4_TLM_HOHEITSGRENZE.shp')
                            
        # gwr_gdf = gpd.read_file(f'{data_path}/output/{scen_dir_import_list[i]}/gwr_gdf.geojson')
        # gwr_gdf.crs
        # gdf = gwr_gdf.merge(df, on='EGID', how = 'left')
        # gdf.crs
        # gdf.head(10)
        # gdf.set_crs('EPSG:4326', allow_override=True, inplace=True)

        # gdf.set_crs('EPSG:2056', allow_override=True, inplace=True)
        # gdf.head(10)

        # gdf['lat'], gdf['lon'] = gdf['geometry'].x, gdf['geometry'].y
        # gdf['lat'], gdf['lon'] = gdf['geometry'].x, gdf['geometry'].y
        # gdf.crs
        # print(gdf[['lat', 'lon']].isna().sum())

        # # plot -----------

        # fig = px.scatter_mapbox(
        #     gdf, 
        #     lat='lat',
        #     lon='lon',
        #     hover_name='EGID',
        #     hover_data = ['bfs', 'pv_inst_id', 'GKLAS', 'GBAUJ'],
        #     title = 'Covered Area of Allocation Model',
        #     # mapbox_style='carto-positron',
        #     mapbox_style='open-street-map',
        # )

        # fig.show()

        # print(gdf_ch.head(10))
        # print(gdf.head(10))
        # print(f'{100*"*"}')



    # PLOT AGGREGATED SCEN --------------------------------------------------------------------
    if len(list(set(T0_prediction_list))) ==1:
        T0_pred_agg = T0_prediction_list[0]


    # plot agg - line: Installed Capacity per Month ============================
    fig = go.Figure()
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

        pvinst_df['BeginOp_month'] = pvinst_df['BeginOp'].dt.to_period('M')
        pvinst_month_df = pvinst_df.groupby(['BeginOp_month', 'info_source'])['TotalPower'].sum().reset_index().copy()
        pvinst_month_df['BeginOp_month'] = pvinst_month_df['BeginOp_month'].dt.to_timestamp()
        pvinst_month_built = pvinst_month_df.loc[pvinst_month_df['info_source'] == 'pv_df'].copy()
        capa_month_predicted = pvinst_month_df.loc[pvinst_month_df['info_source'] == 'alloc_algorithm'].copy()

        pvinst_df['BeginOp_year'] = pvinst_df['BeginOp'].dt.to_period('Y')
        pvinst_year_df = pvinst_df.groupby(['BeginOp_year', 'info_source'])['TotalPower'].sum().reset_index().copy()
        pvinst_year_df['BeginOp_year'] = pvinst_year_df['BeginOp_year'].dt.to_timestamp()
        pvinst_year_built = pvinst_year_df.loc[pvinst_year_df['info_source'] == 'pv_df'].copy()
        pvinst_year_predicted = pvinst_year_df.loc[pvinst_year_df['info_source'] == 'alloc_algorithm'].copy()


        # plot ----------------
        fig.add_trace(go.Scatter(x=pvinst_month_df['BeginOp_month'], y=pvinst_month_df['TotalPower'], line = dict(color = 'navy'),name=f'built + predicted (month)',  mode='lines+markers', legendgrouptitle_text=scen))
        fig.add_trace(go.Scatter(x=pvinst_month_built['BeginOp_month'], y=pvinst_month_built['TotalPower'], line = dict(color = 'deepskyblue'), name=f'built (month)',  mode='lines+markers', legendgrouptitle_text=scen))
        fig.add_trace(go.Scatter(x=capa_month_predicted['BeginOp_month'], y=capa_month_predicted['TotalPower'], line = dict(color = 'cornflowerblue'), name=f'predicted (month)',  mode='lines+markers', legendgrouptitle_text=scen))

        fig.add_trace(go.Scatter(x=pvinst_year_df['BeginOp_year'], y=pvinst_year_df['TotalPower'], line = dict(color = 'forestgreen'), name=f'built + predicted (year)',  mode='lines+markers', legendgrouptitle_text=scen))
        fig.add_trace(go.Scatter(x=pvinst_year_built['BeginOp_year'], y=pvinst_year_built['TotalPower'], line = dict(color = 'lightgreen'), name=f'built (year)',  mode='lines+markers', legendgrouptitle_text=scen))   
        fig.add_trace(go.Scatter(x=pvinst_year_predicted['BeginOp_year'], y=pvinst_year_predicted['TotalPower'], line = dict(color = 'limegreen'), name=f'predicted (year)',  mode='lines+markers', legendgrouptitle_text=scen))

    fig.update_layout(
        xaxis_title='Time',
        yaxis_title='Installed Capacity (kW)',
        legend_title='Scenarios',
        title = f'Installed Capacity per Month (weather year: {pvalloc_scen["weather_specs"]["weather_year"]})'
    )

    fig = add_T0_tick_to_plot(fig, T0_pred_agg)
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
        fig = px.line(gridnode_df, x='t', y='pvprod_kW', color = 'grid_node',name = f'{scen}, wy: {pvalloc_scen["weather_specs"]["weather_year"]})' )
        
        
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
        fig = px.line(gridprem_ts, x='t', y='gridprem', color = 'grid_node',name = f'{scen}, wy: {pvalloc_scen["weather_specs"]["weather_year"]})' )
    
    fig.update_layout(
        xaxis_title='Hour of Year',
        yaxis_title='Grid Premium (CHF)',
        legend_title='Node ID',
        title = f'Grid Premium per Hour of Year, by Scenario (CHF)'
    )
    if plot_show:
        fig.show()

    fig.write_html(f'{data_path}/output/visualizations/plot_agg_line_gridPremiumHOY_per_node.html')





    # topo = json.load(open("C:\Models\OptimalPV_RH_data\output\pvalloc_smallBL_SLCTN_npv_weighted/topo_egid.json", 'r'))
    topo = json.load(open("C:\Models\OptimalPV_RH_data\output\pvalloc_BL_SLCTN_random/topo_egid.json", 'r'))
    
    pvinst_source_list = []
    for k,v in topo.items():
        pvinst_source_list.append(v['pv_inst']['info_source'])

    pvinst_source_list.count('pv_df')
    pvinst_source_list.count('alloc_algorithm')

    key1 = list(topo.keys())[0]
    # topo[key1][]

    # topo_df = pd.read_parquet(f'{scen_data_path}/topo_df.parquet')
    # topo_df.head(10)
    # topo_df.dtypes




# plot ind - ???: Buliding Charachteristics ===========================







