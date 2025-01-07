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
    

    # own packages and functions
    import pv_allocation.default_settings as pvalloc_default_sett
    import visualisations.defaults_settings as visual_default_sett
    
    from auxiliary_functions import *
    
    import visualisations.plot_ind_var_summary_stats as plot_ind_var_summary_stats
    import visualisations.plot_ind_hist_pvcapaprod_sanitycheck as plot_ind_hist_pvcapaprod_sanitycheck
    import visualisations.plot_ind_charac_omitted_gwr as plot_ind_charac_omitted_gwr
    import visualisations.plot_ind_line_meteo_radiation as plot_ind_line_meteo_radiation
    import visualisations.plot_ind_line_installedCap as plot_ind_line_installedCap
    import visualisations.plot_ind_line_PVproduction as plot_ind_line_PVproduction
    import visualisations.plot_ind_line_productionHOY_per_node as plot_ind_line_productionHOY_per_node
    import visualisations.plot_ind_hist_NPV_freepartitions as plot_ind_hist_NPV_freepartitions
    
    import visualisations.plot_ind_map_base as plot_ind_map_base
    import visualisations.plot_ind_map_topo_egid as plot_ind_map_topo_egid
    import visualisations.plot_ind_map_node_connections as plot_ind_map_node_connections
    import visualisations.plot_ind_map_omitted_egids as plot_ind_map_omitted_egids

    # from pv_allocation.default_settings import *
    # from visualisations.defaults_settings import *
    # from visualisations.plot_auxiliary_functions import *

    # plot_show = visual_settings['plot_show']
    # default_zoom_year = visual_settings['default_zoom_year']
    # default_zoom_hour = visual_settings['default_zoom_hour']
    # mc_str = visual_settings["MC_subdir_for_plot"]


def visualization_MASTER(pvalloc_scenarios_func, visual_settings_func):
    # SETTINGS ------------------------------------------------------------------------------------------------------
    if True:
        if not isinstance(pvalloc_scenarios_func, dict):
            print(' USE LOCAL SETTINGS - DICT  ')
            pvalloc_scenarios = pvalloc_default_sett.get_default_pvalloc_settings()
        else:
            pvalloc_scenarios = pvalloc_scenarios_func

        if not isinstance(visual_settings_func, dict) or visual_settings_func == {}:
            visual_settings = visual_default_sett.get_default_visual_settings()
        else:
            visual_settings = visual_settings_func

        pvalloc_sett_run_on_server = pvalloc_scenarios.get(next(iter(pvalloc_scenarios))).get('script_run_on_server')

    # SETUP ------------------------------------------------------------------------------------------------------
    if True: 
        # general setup for paths etc.
        first_alloc_sett = pvalloc_scenarios[list(pvalloc_scenarios.keys())[0]]
        wd_path = first_alloc_sett['wd_path_laptop'] if not first_alloc_sett['script_run_on_server'] else first_alloc_sett['wd_path_server']
        data_path = f'{wd_path}_data'

        # create directory + log file
        visual_path = f'{data_path}/output/visualizations'
        if not os.path.exists(visual_path):
            os.makedirs(visual_path)

        log_name = f'{data_path}/output/visual_log.txt'
        total_runtime_start = datetime.now()


        # extract scenario information + import ------------------------

        # scen settings ----------------
        scen_dir_export_list, pvalloc_scen_list = [], []
        # scen_dir_import_list, T0_prediction_list, months_prediction_list = [], [], [], [] T0_prediction_list, months_lookback_list, months_prediction_list = [], [], [] pvalloc_scen_list = []
        for key, val in pvalloc_scenarios.items():
            pvalloc_settings_path = glob.glob(f'{data_path}/output/{key}/pvalloc_settings.json')
            
            if len(pvalloc_settings_path) == 1:
                try:
                    scen_sett = json.load(open(pvalloc_settings_path[0], 'r'))
                    pvalloc_scen_list.append(scen_sett)
                    scen_dir_export_list.append(scen_sett['name_dir_export'])
                except:
                    print(f'ERROR: could not load pvalloc_settings.json for {key}, take function input')
                    pvalloc_scen_list.append(val)
                    scen_dir_export_list.append(val['name_dir_export'])

            else:
                pvalloc_scen_list.append(val)
                scen_dir_export_list.append(val['name_dir_export'])
    

        # visual settings ----------------  
        # plot_show = visual_settings['plot_show']
        # default_zoom_year = visual_settings['default_zoom_year']
        # default_zoom_hour = visual_settings['default_zoom_hour']
        # mc_str = visual_settings['MC_subdir_for_plot']


        # create directory for plots by scen ----------------
        for key, val in pvalloc_scenarios.items():
            scen = val['name_dir_export']
            # scen = key
            scen_path = f'{data_path}/output/visualizations/{scen}'
            
            if os.path.exists(scen_path):
                n_same_names = len(glob.glob(f'{scen_path}*/'))
                old_dir_rename = f'{scen_path} ({n_same_names})'
                os.rename(scen_path, old_dir_rename)

            os.makedirs(scen_path)

        if visual_settings['remove_previous_plots']:
            all_html = glob.glob(f'{data_path}/output/visualizations/*.html')
            for f in all_html:
                os.remove(f)

        if visual_settings['remove_old_plot_scen_directories']:
            old_plot_scen_dirs = glob.glob(f'{data_path}/output/visualizations/*(*)')
            for dir in old_plot_scen_dirs:
                try:    
                    shutil.rmtree(dir)
                except:
                    print(f'Could not remove {dir}')

    chapter_to_logfile(f'start run_visualisations MASTER ', log_name, overwrite_file=True)




    # PLOT IND SCEN: pvalloc_initalization + sanitycheck ------------------------------------------------------------------------------------------------------


    # plot ind - var: summary statistics --------------------
    plot_ind_var_summary_stats.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)

    # plot ind - hist: sanity check capacity & production --------------------
    plot_ind_hist_pvcapaprod_sanitycheck.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)

    # plot ind - var: disc charac omitted gwr_egids --------------------
    plot_ind_charac_omitted_gwr.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)


    # plot ind - line: meteo radiation over time --------------------
    plot_ind_line_meteo_radiation.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)



    # PLOT IND SCEN: pvalloc_MC_algorithm ------------------------------------------------------------------------------------------------------

    
    # plot ind - line: Installed Capacity per Month & per BFS --------------------
    plot_ind_line_installedCap.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)

    # plot ind - hist: pv production deviation --------------------
    plot_ind_line_PVproduction.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)


    # plot ind - line: Production + Feedin HOY per Node --------------------
    plot_ind_line_productionHOY_per_node.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)   

    # plot ind - hist: NPV possible PV inst before / after --------------------
    plot_ind_hist_NPV_freepartitions.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name)
    

    # map ind - topo_egid --------------------
    plot_ind_map_topo_egid.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name, )

    # map ind - node_connections --------------------
    plot_ind_map_node_connections.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name,)

    # map ind - omitted gwr_egids --------------------
    plot_ind_map_omitted_egids.plot(pvalloc_scen_list, visual_settings, wd_path, data_path, log_name, )




    # PLOT AGGREGATED SCEN ------------------------------------------------------------------------------------------------------




    # ********************************************************************************************************************************************************
    # ********************************************************************************************************************************************************
    # ********************************************************************************************************************************************************
    # ********************************************************************************************************************************************************



    # PLOT AGGREGATED SCEN ------------------------------------------------------------------------------------------------------
    if False:
        if len(list(set(T0_prediction_list))) ==1:
            T0_pred_agg = T0_prediction_list[0]


    # plot agg - line: Installed Capacity per Month ============================
    if visual_settings['plot_agg_line_installedCap_per_month']:
        checkpoint_to_logfile(f'plot_agg_line_installedCap_per_month', log_name)
        fig = go.Figure()
        i_scen, scen = 0, scen_dir_export_list[0]
        # i_scen, scen = 1, scen_dir_export_list[1]
        for i_scen, scen in enumerate(scen_dir_export_list):
            scen_data_path = f'{data_path}/output/{scen}'
            T0_prediction = T0_prediction_list[0]
            months_prediction = months_prediction_list[0]
            pvalloc_scen = pvalloc_scen_list[i_scen]

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
        node_in_plot_selection = []

        checkpoint_to_logfile(f'plot_agg_line_gridPremiumHOY_per_node', log_name)
        fig = go.Figure()
        for i_scen, scen in enumerate(scen_dir_export_list):
            # setup + import ----------
            scen_data_path = f'{data_path}/output/{scen}'
            pvalloc_scen = pvalloc_scen_list[i_scen]

            gridprem_ts = pd.read_parquet(f'{scen_data_path}/gridprem_ts.parquet') 
            gridprem_ts['t_int'] = gridprem_ts['t'].str.extract(r't_(\d+)').astype(int)
            gridprem_ts.sort_values(by=['t_int', 'grid_node'], inplace=True)

            node_selection = visual_settings['node_selection_for_plots']

            # plot ----------------
            if isinstance(node_selection, list): 
                grid_node_loop_list = node_selection
            elif node_selection == None:
                grid_node_loop_list = gridprem_ts['grid_node'].unique()

            for grid_node in grid_node_loop_list:
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
        for i_scen, scen in enumerate(scen_dir_export_list):
            # setup + import ----------
            scen_data_path = f'{data_path}/output/{scen}'
            pvalloc_scen = pvalloc_scen_list[i_scen]

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
        i_scen, scen = 0, scen_dir_export_list[0]
        for i_scen, scen in enumerate(scen_dir_export_list):
            # setup + import ----------
            scen_data_path = f'{data_path}/output/{scen}'
            pvalloc_scen = pvalloc_scen_list[i_scen]

            gridnode_df = pd.read_parquet(f'{scen_data_path}/gridnode_df.parquet') 
            gridnode_df['t_int'] = gridnode_df['t'].str.extract(r't_(\d+)').astype(int)
            gridnode_df.sort_values(by=['t_int'], inplace=True)

            # plot ----------------
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


    # plot agg - line: PV Production / Feedin per Month ============================
    if visual_settings['plot_agg_line_production_per_month']:
    
        fig = go.Figure()
        i_scen, scen = 0, scen_dir_export_list[0]
        for i_scen, scen in enumerate(scen_dir_export_list):
            # setup + import ----------
            scen_data_path = f'{data_path}/output/{scen}'
            pvalloc_scen = pvalloc_scen_list[i_scen]
            T0_scen = pd.to_datetime(pvalloc_scen['T0_prediction'])
            months_prediction_scen = pvalloc_scen['months_prediction']

            plot_df = pd.DataFrame()
            months_prediction_range = pd.date_range(start=T0_scen + pd.DateOffset(days=1), periods=months_prediction_scen, freq='ME').to_period('M')
            m, month = 0, months_prediction_range[0]
            for i_m, m in enumerate(months_prediction_range):
                subgridnode_df = pd.read_parquet(f'{scen_data_path}/pred_gridprem_node_by_M/gridnode_df_{m}.parquet')
                subgridnode_df['scen'], subgridnode_df['month'] = scen, m
                subgridnode_total_df = subgridnode_df.groupby(['scen', 'month']).agg({'pvprod_kW': 'sum', 'feedin_kW': 'sum','feedin_kW_taken': 'sum','feedin_kW_loss': 'sum'}).reset_index()
                
                plot_df = pd.concat([plot_df, subgridnode_total_df])
            plot_df['month'] = plot_df['month'].dt.to_timestamp()

            # plot ----------------
            fig.add_trace(go.Scatter(x=plot_df['month'], y=plot_df['pvprod_kW'], name=f'{scen}: pv_production'))
            fig.add_trace(go.Scatter(x=plot_df['month'], y=plot_df['feedin_kW'], name=f'{scen}: feedin (prod-self_consum)'))
            fig.add_trace(go.Scatter(x=plot_df['month'], y=plot_df['feedin_kW_taken'], name=f'{scen}: feedin_taken (by grid)'))
            fig.add_trace(go.Scatter(x=plot_df['month'], y=plot_df['feedin_kW_loss'], name=f'{scen}: feedin_loss (excess grid nodes capa)'))

        fig.update_layout(
            xaxis_title='Perdicion Iterations',
            yaxis_title='Production / Feedin (kWh)',
            legend_title='Scenarios',
            title = f'Agg. Production per Month, by Iteration Step (diff Scenarios)'
        )

        if plot_show:
            fig.show()
        fig.write_html(f'{data_path}/output/visualizations/plot_agg_line_production_per_month.html')


    # plot agg - line: Charachteristics Newly Installed Buildings  per Month ============================
    if visual_settings['plot_agg_line_cont_charact_new_inst']:
        
        fig = go.Figure()
        i_scen, scen = 0, scen_dir_export_list[0]
        for i_scen, scen in enumerate(scen_dir_export_list):
        
            # setup + import ----------
            pvalloc_scen = pvalloc_scen_list[i_scen]
            T0_scen = pd.to_datetime(pvalloc_scen['T0_prediction'])
            months_prediction_scen = pvalloc_scen['months_prediction']
            scen_data_path = f'{data_path}/output/{scen}'
            
            colnames_cont_charact_installations = visual_settings['plot_agg_line_cont_charact_new_inst_specs']['colnames_cont_charact_installations']
            num_colors = (len(pvalloc_scen_list) * len(colnames_cont_charact_installations) ) + 5
            colors = pc.sample_colorscale('Turbo', [n/(num_colors-1) for n in range(num_colors)])
            
            predinst_all= pd.read_parquet(f'{scen_data_path}/pred_inst_df.parquet')


            # plot absolute values -------------------------------------
            preinst_absdf = copy.deepcopy(predinst_all)
            # months_prediction_range = pd.date_range(start=T0_scen + pd.DateOffset(days=1), periods=months_prediction_scen, freq='M').to_period('M')
            agg_dict ={}
            for colname in colnames_cont_charact_installations:
                agg_dict[f'{colname}'] = ['mean', 'std']

            # NOTE: remove if statement if 'iter_round' is present in later runs
            if not 'iter_round' in preinst_absdf.columns:
                agg_predinst_absdf = preinst_absdf.groupby('BeginOp').agg(agg_dict)
                agg_predinst_absdf['iter_round'] = range(1, len(agg_predinst_absdf)+1)   
            else:
                agg_predinst_absdf = preinst_absdf.groupby('iter_round').agg(agg_dict)
                agg_predinst_absdf['iter_round'] = agg_predinst_absdf.index

            agg_predinst_absdf.replace(np.nan, 0, inplace=True)   
            xaxis = agg_predinst_absdf['iter_round']


            # plot ----------------
            # fig = go.Figure()
            col = colnames_cont_charact_installations[0]    
            for i_col, col in enumerate(colnames_cont_charact_installations):
            # if True:
                scen_col = f'{scen}__{col}'                
                xaxis = agg_predinst_absdf['iter_round']
                y_mean, y_lower, y_upper = agg_predinst_absdf[col]['mean'], agg_predinst_absdf[col]['mean'] - agg_predinst_absdf[col]['std'], agg_predinst_absdf[col]['mean'] + agg_predinst_absdf[col]['std']
                line_color = colors[i_col % len(colors)]

                # mean
                fig.add_trace(go.Scatter(x=xaxis, y=y_mean, name=f'{scen_col}', legendgroup = f'{scen_col}', line=dict(color=line_color), mode='lines+markers', showlegend=True))
                # upper / lower bound
                fig.add_trace(go.Scatter(
                    x=xaxis.tolist() + xaxis.tolist()[::-1],  # Concatenate xaxis with its reverse
                    y=y_upper.tolist() + y_lower.tolist()[::-1],  # Concatenate y_upper with reversed y_lower
                    fill='toself',
                    fillcolor=line_color,  # Dynamic color with 50% transparency
                    opacity = 0.2,
                    line=dict(color='rgba(255,255,255,0)'),  # No boundary line
                    hoverinfo="skip",  # Don't show info on hover
                    showlegend=False,  # Do not show this trace in the legend
                    legendgroup=f'{scen_col}',  # Group with the mean line
                    visible=True  # Make this visible/toggleable with the mean line
                ))

        fig.update_layout(
            xaxis_title='Iteration Round',
            yaxis_title='Mean (+/- 1 std)',
            legend_title='Scenarios',
            title = f'Agg. Cont. Charact. of Newly Installed Buildings per Iteration Round'
        )
        
        if plot_show:
            fig.show()
        fig.write_html(f'{data_path}/output/visualizations/plot_agg_line_cont_charact_new_inst_abs_values.html')

        # add standardized version => where all values within the df are first standardized before plotted for more comparable/readable plot
        # pred_inst_df_stand = copy.deepcopy(pred_inst_df[['BeginOp']+colnames_cont_charact_installations])
        # pred_inst_df_stand = pred_inst_df_stand.groupby('BeginOp').transform(lambda x: (x - x.mean()) / x.std())


    # END ------------------------------------------------------------------------------------------------------




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
        for i_path, path in enumerate(topo_subdf_paths):
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
            color_continuous_scale="Turbo",
            range_color=(0, 100),
            mapbox_style="carto-positron",
            center={"lat": default_map_center[0], "lon": default_map_center[1]}, 
            zoom=default_map_zoom,
            opacity=0.5,
            hover_name="hover_text",
            title=f"Map of production per DF_UID ({scen})",
        )
        if plot_show:
            fig2.show()
        fig2.write_html(f'{data_path}/output/visualizations/{scen}__map_ind_production.html')

    # END  ================================================================
    chapter_to_logfile(f'END visualization_MASTER\n Runtime (hh:mm:ss):{datetime.now() - total_runtime_start}', log_name, overwrite_file=False)
    if not pvalloc_sett_run_on_server:
        winsound.Beep(1000, 30)
        winsound.Beep(1000, 30)
        winsound.Beep(1000, 30)
        winsound.Beep(1000, 30)
        winsound.Beep(1000, 30)
        winsound.Beep(1000, 1000)







# ********************************************************************************************************************
            
                

                
                            



                



    # plot for installation capacity + production => add to plots for an MC iteration
    if False:
        print('asdf')
        if False:

            # hist annual kWh production ------------
            topo = json.load(open(f'{scen_data_path}/topo_egid.json', 'r'))
            egid_list, inst_TF_list, info_source_list, TotalPower_list = [], [], [], []

            for k,v, in topo.items():
                egid_list.append(k)
                if v['pv_inst']['inst_TF'] == True:
                    inst_TF_list.append(v['pv_inst']['inst_TF'])
                    info_source_list.append(v['pv_inst']['info_source'])
                    TotalPower_list.append(v['pv_inst']['TotalPower'])
                else: 
                    inst_TF_list.append(False)
                    info_source_list.append('')
                    TotalPower_list.append(0)

            pvinst_df = pd.DataFrame({'EGID': egid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list, 'TotalPower': TotalPower_list})

            topo_subdf_paths = glob.glob(f'{scen_data_path}/topo_time_subdf/*.parquet')
            agg_subinst_df_list = []
            i_path, path = 0, topo_subdf_paths[0]
            for i_path, path in enumerate(topo_subdf_paths):
                subdf = pd.read_parquet(path)
                agg_subdf = subdf.groupby('EGID')['pvprod_kW'].sum().reset_index()
                agg_subinst_df_list.append(agg_subdf)

            agg_subinst_df = pd.concat(agg_subinst_df_list, axis=0)
            agg_subinst_df.rename(columns={'pvprod_kW': 'pvprod_kWh'}, inplace=True)
            agg_subinst_df = agg_subinst_df.merge(pvinst_df, on='EGID', how='left')

            fig = go.Figure()
            color_rest, color_pv_df, color_alloc_algo = visual_settings['plot_ind_map_topo_egid_specs']['point_color_rest'], visual_settings['plot_ind_map_topo_egid_specs']['point_color_pv_df'], visual_settings['plot_ind_map_topo_egid_specs']['point_color_alloc_algo']

            fig.add_trace(go.Histogram(x=agg_subinst_df['pvprod_kWh'], name='PV Production [kWh]', opacity = 0.2, marker_color = color_rest))
            fig.add_trace(go.Histogram(x=agg_subinst_df.loc[agg_subinst_df['info_source'] == 'pv_df', 'pvprod_kWh'], 
                                       name='PV Production pre-alloc installed', opacity=0.5, marker_color = color_pv_df))
            fig.add_trace(go.Histogram(x=agg_subinst_df.loc[agg_subinst_df['info_source'] == 'alloc_algorithm', 'pvprod_kWh'], 
                                       name='PV Production post-alloc installed', opacity=0.5, marker_color = color_alloc_algo))

            fig.update_layout(
                xaxis_title='PV Production kWh',
                yaxis_title='Frequency',
                title = f'PV Production Distribution (scen: {scen})',
                barmode = 'overlay')
            fig.update_traces(bingroup=1, opacity=0.5)

            fig = add_scen_name_to_plot(fig, scen, pvalloc_scen_list[i])
            if plot_show:
                fig.show()
            fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_hist_pvprod_kWh.html')

            
            # hist installation power kW ------------
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=pvinst_df.loc[pvinst_df['inst_TF']==True,'TotalPower'], name='Installed Capacity [kW]', opacity=0.2, marker_color = color_rest))
            fig.add_trace(go.Histogram(x=pvinst_df.loc[pvinst_df['info_source'] == 'pv_df', 'TotalPower'],
                                        name='Installed Capacity pre-alloc installed', opacity=0.5, marker_color = color_pv_df))
            fig.add_trace(go.Histogram(x=pvinst_df.loc[pvinst_df['info_source'] == 'alloc_algorithm', 'TotalPower'],
                                        name='Installed Capacity post-alloc installed', opacity=0.5, marker_color = color_alloc_algo))

            fig.update_layout(
                xaxis_title='Installed Capacity [kW]',
                yaxis_title='Frequency',
                title = f'Installed Capacity Distribution (scen: {scen})',
                barmode = 'overlay')
            fig.update_traces(bingroup=1, opacity=0.5)

            fig = add_scen_name_to_plot(fig, scen, pvalloc_scen_list[i])
            if plot_show:
                fig.show()
                fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_hist_installedCap_kW.html')
