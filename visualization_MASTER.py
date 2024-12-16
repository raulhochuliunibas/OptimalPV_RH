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
    

    # own packages and functions
    import pv_allocation.default_settings as pvalloc_default_sett
    import visualisations.defaults_settings as visual_default_sett

    from auxiliary_functions import *
    from pv_allocation.default_settings import *
    from visualisations.defaults_settings import *


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
        plot_show = visual_settings['plot_show']
        default_zoom_year = visual_settings['default_zoom_year']
        default_zoom_hour = visual_settings['default_zoom_hour']
        mc_str = visual_settings['MC_subdir_for_plot']


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
                shutil.rmtree(dir)

    chapter_to_logfile(f'start run_visualisations MASTER ', log_name, overwrite_file=True)


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




    # PLOT IND SCEN: pvalloc_initalization + sanitycheck ------------------------------------------------------------------------------------------------------


    # plot ind - var: summary statistics --------------------
    if visual_settings['plot_ind_var_summary_stats'][0]:
        checkpoint_to_logfile(f'plot_ind_var_summary_stats', log_name)
        i_scen, scen = 0, scen_dir_export_list[0]

        for i_scen, scen in enumerate(scen_dir_export_list):
            scen_data_path = f'{data_path}/output/{scen}'
            pvalloc_scen = pvalloc_scen_list[i_scen]

            # total kWh by demandtypes ------------------------
            demandtypes = pd.read_parquet(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/demandtypes.parquet')

            demandtypes_names = [col for col in demandtypes.columns if 't' not in col]
            totaldemand_kWh = [demandtypes[type].sum() for type in demandtypes_names]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=demandtypes_names, y=totaldemand_kWh, name='Total Demand [kWh]'))
            fig.update_layout(
                xaxis_title='Demand Type',
                yaxis_title='Total Demand [kWh], 1 year',
                title = f'Total Demand per Demand Type (scen: {scen})'
            )
            fig = add_scen_name_to_plot(fig, scen, pvalloc_scen)

            if plot_show and visual_settings['plot_ind_var_summary_stats'][1]:
                if visual_settings['plot_ind_var_summary_stats'][2]:
                    fig.show()
                elif not visual_settings['plot_ind_var_summary_stats'][2]:
                    fig.show() if i_scen == 0 else None
            if visual_settings['save_plot_by_scen_directory']:
                fig.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_ind_bar_totaldemand_by_type.html')
            else:
                fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_bar_totaldemand_by_type.html')
            print_to_logfile(f'\texport: plot_ind_bar_totaldemand_by_type.html (for: {scen})', log_name)
            

            # demand TS ------------------------
            demandtypes = pd.read_parquet(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/demandtypes.parquet')
            
            fig = px.line(demandtypes, x='t', y=demandtypes_names, title='Demand Time Series')
            fig.update_layout(
                xaxis_title='Time',
                yaxis_title='Demand [kWh]',
                title = f'Demand Time Series (scen: {scen})'
            )

            fig = add_scen_name_to_plot(fig, scen, pvalloc_scen)
            fig = set_default_fig_zoom_hour(fig, default_zoom_hour)

            if plot_show and visual_settings['plot_ind_var_summary_stats'][1]:
                if visual_settings['plot_ind_var_summary_stats'][2]:
                    fig.show()
                elif not visual_settings['plot_ind_var_summary_stats'][2]:
                    fig.show() if i_scen == 0 else None
            if visual_settings['save_plot_by_scen_directory']:
                fig.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_ind_line_demandTS.html')
            else:
                fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_line_demandTS.html')
            print_to_logfile(f'\texport: plot_ind_line_demandTS.html (for: {scen})', log_name)
            

    # plot ind - hist: sanity check capacity & production --------------------
    if visual_settings['plot_ind_hist_pvcapaprod_sanitycheck'][0]:
        checkpoint_to_logfile(f'plot_ind_hist_pvcapaprod_sanitycheck', log_name)
        i_scen, scen = 0, scen_dir_export_list[0]

        # fig_agg_abs, fig_agg_stand = make_subplots(specs=[[{"secondary_y": True}]]), make_subplots(specs=[[{"secondary_y": True}]])
        fig_agg_abs, fig_agg_stand = go.Figure(), go.Figure()

        for i_scen, scen in enumerate(scen_dir_export_list):
            pvalloc_scen = pvalloc_scen_list[i_scen]
            kWpeak_per_m2, share_roof_area_available = pvalloc_scen['tech_economic_specs']['kWpeak_per_m2'],pvalloc_scen['tech_economic_specs']['share_roof_area_available']
            inverter_efficiency = pvalloc_scen['tech_economic_specs']['inverter_efficiency']

            panel_efficiency_print = 'dynamic' if pvalloc_scen['panel_efficiency_specs']['variable_panel_efficiency_TF'] else 'static'
            color_pv_df, color_solkat, color_rest = visual_settings['plot_ind_map_topo_egid_specs']['point_color_pv_df'], visual_settings['plot_ind_map_topo_egid_specs']['point_color_solkat'],visual_settings['plot_ind_map_topo_egid_specs']['point_color_rest']
            xbins_hist_instcapa_abs, xbins_hist_instcapa_stand = visual_settings['plot_ind_hist_pvcapaprod_sanitycheck_specs']['xbins_hist_instcapa_abs'], visual_settings['plot_ind_hist_pvcapaprod_sanitycheck_specs']['xbins_hist_instcapa_stand']
            xbins_hist_totalprodkwh_abs, xbins_hist_totalprodkwh_stand = visual_settings['plot_ind_hist_pvcapaprod_sanitycheck_specs']['xbins_hist_totalprodkwh_abs'], visual_settings['plot_ind_hist_pvcapaprod_sanitycheck_specs']['xbins_hist_totalprodkwh_stand']

            sanity_scen_data_path = f'{data_path}/output/{scen}/sanity_check_byEGID'
            pv = pd.read_parquet(f'{data_path}/output/{scen}/pv.parquet')
            topo = json.load(open(f'{sanity_scen_data_path}/topo_egid.json', 'r'))
            egid_with_pvdf = [egid for egid in topo.keys() if topo[egid]['pv_inst']['info_source'] == 'pv_df']
            xtf_in_topo = [topo[egid]['pv_inst']['xtf_id'] for egid in egid_with_pvdf]
            topo_subdf_paths = glob.glob(f'{sanity_scen_data_path}/topo_subdf_*.parquet')
            topo.get(egid_with_pvdf[0])
            
            aggdf_combo_list = []
            path = topo_subdf_paths[0]
            for i_path, path in enumerate(topo_subdf_paths):
                subdf= pd.read_parquet(path)
                subdf = subdf.loc[subdf['EGID'].isin(egid_with_pvdf)]
                # compute pvprod by using TotalPower of pv_df, check if it overlaps with computation of STROMERTRAG
                subdf['pvprod_TotalPower_kW'] = subdf['radiation_rel_locmax'] * subdf['TotalPower'] *  inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] 


                agg_subdf = subdf.groupby(['EGID', 'df_uid', 'FLAECHE', 'STROMERTRAG']).agg({'pvprod_kW': 'sum', 
                                                                                             'pvprod_TotalPower_kW': 'sum'}).reset_index()
                aggsub_npry = np.array(agg_subdf)
                
                egid_list, flaeche_list, pvprod_list, pvprod_ByTotalPower_list, stromertrag_list = [], [], [], [], []                
                egid = list(topo.keys())[0]
                for egid in subdf['EGID'].unique():
                    mask_egid_subdf = np.isin(aggsub_npry[:,agg_subdf.columns.get_loc('EGID')], egid)
                    df_uids  = list(aggsub_npry[mask_egid_subdf, agg_subdf.columns.get_loc('df_uid')])

                    for r in range(1,len(df_uids)+1):
                        for combo in itertools.combinations(df_uids,r):
                            combo_key_str = '_'.join([str(c) for c in combo])
                            mask_dfuid_subdf = np.isin(aggsub_npry[:,agg_subdf.columns.get_loc('df_uid')], list(combo))

                            egid_list.append(egid)
                            flaeche_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('FLAECHE')].sum())
                            pvprod_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pvprod_kW')].sum())
                            pvprod_ByTotalPower_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pvprod_TotalPower_kW')].sum())
                            stromertrag_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('STROMERTRAG')].sum())
                            
                aggsubdf_combo = pd.DataFrame({'EGID': egid_list, 'FLAECHE': flaeche_list, 
                                               'pvprod_kW': pvprod_list, 'pvprod_ByTotalPower_kW': pvprod_ByTotalPower_list,
                                               'STROMERTRAG': stromertrag_list})
                
                aggdf_combo_list.append(aggsubdf_combo)
            
            aggdf_combo = pd.concat(aggdf_combo_list, axis=0)


            # installed Capapcity kW --------------------------------
            if True:            
                aggdf_combo['inst_capa_kW'] = aggdf_combo['FLAECHE'] * kWpeak_per_m2 * share_roof_area_available
                aggdf_combo['inst_capa_kW_stand'] = (aggdf_combo['inst_capa_kW'] - aggdf_combo['inst_capa_kW'].mean()) / aggdf_combo['inst_capa_kW'].std()
                pv['TotalPower_stand'] = (pv['TotalPower'] - pv['TotalPower'].mean()) / pv['TotalPower'].std()

                fig = make_subplots(specs=[[{"secondary_y": True}]])

                fig.add_trace(go.Histogram(x=aggdf_combo['inst_capa_kW'], 
                                        name='Modeled Potential Capacity (all partition combos, EGIDs in topo, in pv_df)', 
                                        opacity=0.5, marker_color = color_rest, 
                                        xbins=dict(size=xbins_hist_instcapa_abs)), 
                                        secondary_y=False)
                fig.add_trace(go.Histogram(x=pv.loc[pv['xtf_id'].isin(xtf_in_topo), 'TotalPower'], 
                                        name='Installed Capacity (pv_df in topo)', 
                                        opacity=0.5, marker_color = color_pv_df, 
                                        xbins=dict(size = xbins_hist_instcapa_abs)), 
                                        secondary_y=False)

                fig.add_trace(go.Histogram(x=aggdf_combo['inst_capa_kW_stand'], 
                                        name='Modeled Potential Capacity (all partition combos, EGIDs in topo, in pv_df), standardized', 
                                        opacity=0.5, marker_color = color_rest,
                                        xbins=dict(size= xbins_hist_instcapa_stand)),
                                        secondary_y=True)
                fig.add_trace(go.Histogram(x=pv['TotalPower_stand'],
                                            name='Installed Capacity (pv_df in topo), standardized',
                                            opacity=0.5, marker_color = color_pv_df,
                                            xbins=dict(size= xbins_hist_instcapa_stand)),
                                            secondary_y=True)

                fig.update_layout(
                    barmode='overlay', 
                    xaxis_title='Capacity [kW]',
                    yaxis_title='Frequency (Modelled Capacity, possible installations)',
                    title = f'SANITY CHECK: Modelled vs Installed Capacity (kWp_m2:{kWpeak_per_m2}, share roof: {share_roof_area_available})'
                ) 
                fig.update_yaxes(title_text="Frequency (standardized)", secondary_y=True)
                fig = add_scen_name_to_plot(fig, scen, pvalloc_scen)

                if plot_show and visual_settings['plot_ind_hist_pvcapaprod_sanitycheck'][1]:
                    if visual_settings['plot_ind_hist_pvcapaprod_sanitycheck'][2]:
                        fig.show()
                    elif not visual_settings['plot_ind_hist_pvcapaprod_sanitycheck'][2]:
                        fig.show() if i_scen == 0 else None
                if visual_settings['save_plot_by_scen_directory']:
                    fig.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_ind_hist_pvCapaProd_SanityCheck_instCapa_kW.html')
                else:
                    fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_hist_pvCapaProd_SanityCheck_instCapa_kW.html')    
                print_to_logfile(f'\texport: plot_ind_hist_SanityCheck_instCapa_kW.html (for: {scen})', log_name)

            # annual PV production kWh --------------------------------
            if True:
                # standardization for plot
                aggdf_combo['pvprod_kW_stand'] = (aggdf_combo['pvprod_kW'] - aggdf_combo['pvprod_kW'].mean()) / aggdf_combo['pvprod_kW'].std() 
                aggdf_combo['pvprod_ByTotalPower_kW_stand'] = (aggdf_combo['pvprod_ByTotalPower_kW'] - aggdf_combo['pvprod_ByTotalPower_kW'].mean()) / aggdf_combo['pvprod_ByTotalPower_kW'].std()
                aggdf_combo['STROMERTRAG_stand'] = (aggdf_combo['STROMERTRAG'] - aggdf_combo['STROMERTRAG'].mean()) / aggdf_combo['STROMERTRAG'].std()

                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Histogram(x=aggdf_combo['pvprod_kW'], 
                                        name='Modeled Potential Yearly Production (kWh)',
                                        opacity=0.5, marker_color = color_rest, 
                                        xbins = dict(size=xbins_hist_totalprodkwh_abs)), secondary_y=False)
                fig.add_trace(go.Histogram(x=aggdf_combo['STROMERTRAG'], 
                                        name='STROMERTRAG (solkat estimated production)',
                                        opacity=0.5, marker_color = color_solkat, 
                                        xbins = dict(size=xbins_hist_totalprodkwh_abs)), secondary_y=False)
                fig.add_trace(go.Histogram(x=aggdf_combo['pvprod_ByTotalPower_kW'],
                                            name='Yearly Prod. TotalPower (pvdf estimated production)', 
                                            opacity=0.5, marker_color = color_pv_df,
                                            xbins=dict(size=xbins_hist_totalprodkwh_abs)), secondary_y=False)

                fig.add_trace(go.Histogram(x=aggdf_combo['pvprod_kW_stand'], 
                                        name='Modeled Potential Yearly Production (kWh), standardized',
                                        opacity=0.5, marker_color = color_rest,
                                        xbins=dict(size=xbins_hist_totalprodkwh_stand)), secondary_y=True)
                fig.add_trace(go.Histogram(x=aggdf_combo['STROMERTRAG_stand'],
                                            name='STROMERTRAG (solkat estimated production), standardized',
                                            opacity=0.5, marker_color = color_solkat,
                                            xbins=dict(size=xbins_hist_totalprodkwh_stand)), secondary_y=True)
                fig.add_trace(go.Histogram(x=aggdf_combo['pvprod_ByTotalPower_kW_stand'],
                                            name='Yearly Prod. TotalPower (pvdf estimated production), standardized',
                                            opacity=0.5, marker_color = color_pv_df,
                                            xbins=dict(size=xbins_hist_totalprodkwh_stand)), secondary_y=True)
                fig.update_layout(
                    barmode = 'overlay', 
                    xaxis_title='Production [kWh]',
                    yaxis_title='Frequency, absolute',
                    title = f'SanityCheck: Modeled vs Estimated Yearly PRODUCTION (kWp_m2:{kWpeak_per_m2}, share roof available: {share_roof_area_available}, {panel_efficiency_print} panel efficiency, inverter efficiency: {inverter_efficiency})'
                )
                fig.update_yaxes(title_text="Frequency (standardized)", secondary_y=True)
                fig = add_scen_name_to_plot(fig, scen, pvalloc_scen)
                
                if plot_show and visual_settings['plot_ind_hist_pvcapaprod_sanitycheck'][1]:
                    if visual_settings['plot_ind_hist_pvcapaprod_sanitycheck'][2]:
                        fig.show()
                    elif not visual_settings['plot_ind_hist_pvcapaprod_sanitycheck'][2]:
                        fig.show() if i_scen == 0 else None
                if visual_settings['save_plot_by_scen_directory']:
                    fig.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_ind_hist_pvCapaProd_SanityCheck_annualPVprod_kWh.html')
                else:
                    fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_hist_pvCapaProd_SanityCheck_annualPVprod_kWh.html')
                print_to_logfile(f'\texport: plot_ind_hist_SanityCheck_annualPVprod_kWh.html (for: {scen})', log_name)


            # Agg: installed Capacity kW --------------------------------
            if True:
                fig_agg_abs.add_trace(go.Scatter(x=[0,], y=[0,], name=f'', opacity=0,))
                fig_agg_abs.add_trace(go.Scatter(x=[0,], y=[0,], name=f'{scen}', opacity=0,))
                fig_agg_abs.add_trace(go.Histogram(x=aggdf_combo['inst_capa_kW'], 
                                        name=' - Modeled Potential Capacity (all partition combos, EGIDs in topo, in pv_df)', 
                                        opacity=0.5, 
                                        xbins=dict(size=xbins_hist_instcapa_abs), 
                                        ))
                fig_agg_abs.add_trace(go.Histogram(x=pv.loc[pv['xtf_id'].isin(xtf_in_topo), 'TotalPower'],
                                            name=' - Installed Capacity (pv_df in topo)', 
                                            opacity=0.5, 
                                            xbins=dict(size = xbins_hist_instcapa_abs),
                                            ))
                
                fig_agg_stand.add_trace(go.Scatter(x=[0,], y=[0,], name=f'', opacity=0,))
                fig_agg_stand.add_trace(go.Scatter(x=[0,], y=[0,], name=f'{scen}', opacity=0,))                                                     
                fig_agg_stand.add_trace(go.Histogram(x=aggdf_combo['inst_capa_kW_stand'],
                                            name= ' - Modeled Potential Capacity (all partition combos, EGIDs in topo, in pv_df), standardized', 
                                            opacity=0.5, 
                                            xbins=dict(size= xbins_hist_instcapa_stand),
                                            ))
                fig_agg_stand.add_trace(go.Histogram(x=pv['TotalPower_stand'],
                                            name=' - Installed Capacity (pv_df in topo), standardized',
                                            opacity=0.5, 
                                            xbins=dict(size= xbins_hist_instcapa_stand),
                                            ))

            # Agg: annual PV production kWh --------------------------------
            if True:
                # kernel density traces
                x_range_pvproduction_abs = np.linspace(min(aggdf_combo['pvprod_kW'].min(), aggdf_combo['STROMERTRAG'].min(), aggdf_combo['pvprod_ByTotalPower_kW'].min()),
                                                    max(aggdf_combo['pvprod_kW'].max(), aggdf_combo['STROMERTRAG'].max(), aggdf_combo['pvprod_ByTotalPower_kW'].max()), 500)
                x_range_pvproduction_stand = np.linspace(min(aggdf_combo['pvprod_kW_stand'].min(), aggdf_combo['STROMERTRAG_stand'].min(), aggdf_combo['pvprod_ByTotalPower_kW_stand'].min()),
                                                    max(aggdf_combo['pvprod_kW_stand'].max(), aggdf_combo['STROMERTRAG_stand'].max(), aggdf_combo['pvprod_ByTotalPower_kW_stand'].max()), 500) 

                kde_pvprod_kW, kde_pvprod_kW_stand = stats.gaussian_kde(aggdf_combo['pvprod_kW']), stats.gaussian_kde(aggdf_combo['pvprod_kW_stand'])
                kde_STROMERTRAG, kde_STROMERTRAG_stand = stats.gaussian_kde(aggdf_combo['STROMERTRAG']), stats.gaussian_kde(aggdf_combo['STROMERTRAG_stand'])
                kde_pvprod_ByTotalPower_kW, kde_pvprod_ByTotalPower_kW_stand = stats.gaussian_kde(aggdf_combo['pvprod_ByTotalPower_kW']), stats.gaussian_kde(aggdf_combo['pvprod_ByTotalPower_kW_stand'])


                fig_agg_abs.add_trace(go.Histogram(x=aggdf_combo['pvprod_kW'],
                                        name=f' - Modeled Potential Yearly Production (kWh) {scen}',
                                        opacity=0.5, 
                                        xbins=dict(size=xbins_hist_totalprodkwh_abs)
                                        ))
                fig_agg_abs.add_trace(go.Histogram(x=aggdf_combo['STROMERTRAG'],
                                        name=f' - STROMERTRAG (solkat estimated production) {scen}',
                                        opacity=0.5, 
                                        xbins=dict(size=xbins_hist_totalprodkwh_abs),
                                        ))
                fig_agg_abs.add_trace(go.Histogram(x=aggdf_combo['pvprod_ByTotalPower_kW'],
                                        name=f' - Yearly Prod. TotalPower (pvdf estimated production) {scen}',
                                        opacity=0.5, 
                                        xbins=dict(size=xbins_hist_totalprodkwh_abs),
                                        ))     
                
                fig_agg_stand.add_trace(go.Histogram(x=aggdf_combo['pvprod_kW_stand'],
                                        name=f' - Modeled Potential Yearly Production (kWh), standardized {scen}',
                                        opacity=0.5, 
                                        xbins=dict(size=xbins_hist_totalprodkwh_stand),
                                        ))
                fig_agg_stand.add_trace(go.Histogram(x=aggdf_combo['STROMERTRAG_stand'],
                                        name=f' - STROMERTRAG (solkat estimated production), standardized {scen}',
                                        opacity=0.5, 
                                        xbins=dict(size=xbins_hist_totalprodkwh_stand),
                                        ))
                fig_agg_stand.add_trace(go.Histogram(x=aggdf_combo['pvprod_ByTotalPower_kW_stand'],
                                        name=f' - Yearly Prod. TotalPower (pvdf estimated production), standardized {scen}',
                                        opacity=0.5,
                                        xbins=dict(size=xbins_hist_totalprodkwh_stand),
                                        ))
                

        # Export Agg plots --------------------------------
        if True:
            fig_agg_abs.update_layout(
                barmode='overlay',
                xaxis_title='Capacity [kW]',
                yaxis_title='Frequency (Modelled Capacity, possible installations)',
                title = f'SANITY CHECK: Agg. Modelled vs Installed Cap. & Yearly Prod. ABSOLUTE (kWp_m2:{kWpeak_per_m2}, share roof available: {share_roof_area_available}, {panel_efficiency_print} panel eff, inverter eff: {inverter_efficiency})'
            )
            fig_agg_stand.update_layout(
                barmode='overlay',
                xaxis_title='Production [kWh]',
                yaxis_title='Frequency, absolute',
                title = f'SANITY CHECK: Agg. Modelled vs Installed Cap. & Yearly Prod. STANDARDIZED (kWp_m2:{kWpeak_per_m2}, share roof available: {share_roof_area_available}, {panel_efficiency_print} panel eff, inverter eff: {inverter_efficiency})'
            )


            if plot_show and visual_settings['plot_ind_hist_pvcapaprod_sanitycheck'][1]:
                fig_agg_abs.show()
                fig_agg_stand.show()
                # if visual_settings['save_plot_by_scen_directory']:
                #     fig_agg_abs.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_agg_hist_pvCapaProd_SanityCheck_instCapa_kW.html')
                #     fig_agg_stand.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_agg_hist_pvCapaProd_SanityCheck_annualPVprod_kWh.html')
                # else:
            fig_agg_abs.write_html(f'{data_path}/output/visualizations/{scen}__plot_agg_hist_pvCapaProd_SanityCheck_instCapa_kW.html')
            fig_agg_stand.write_html(f'{data_path}/output/visualizations/{scen}__plot_agg_hist_pvCapaProd_SanityCheck_annualPVprod_kWh.html')
            print_to_logfile(f'\texport: plot_agg_hist_SanityCheck_instCapa_kW.html (for: {scen})', log_name)

            

    # plot ind - var: disc charac omitted gwr_egids --------------------
    if visual_settings['plot_ind_charac_omitted_gwr'][0]:
        plot_ind_charac_omitted_gwr_specs = visual_settings['plot_ind_charac_omitted_gwr_specs']
        checkpoint_to_logfile(f'plot_ind_charac_omitted_gwr', log_name)

        for i_scen, scen in enumerate(scen_dir_export_list):
            scen_sett = pvalloc_scen_list[i_scen]

            # omitted egids from data prep -----            
            get_bfsnr_name_tuple_list()

            omitt_gwregid_gdf_geo = gpd.read_file(f'{data_path}/output/{scen_sett["name_dir_import"]}/omitt_gwregid_gdf.geojson')
            gwr_all_building_gdf = gpd.read_file(f'{data_path}/output/{scen_sett["name_dir_import"]}/gwr_all_building_gdf.geojson')
            omitt_gwregid_gdf_all = omitt_gwregid_gdf_geo.merge(gwr_all_building_gdf, on='EGID', how='left')
            omitt_gwregid_gdf_all.rename(columns={'GGDENR': 'BFS_NUMMER'}, inplace=True)
            omitt_gwregid_gdf_all['BFS_NUMMER'] = omitt_gwregid_gdf_all['BFS_NUMMER'].astype(int)
            omitt_gwregid_gdf = omitt_gwregid_gdf_all.loc[omitt_gwregid_gdf_all['BFS_NUMMER'].isin(scen_sett['bfs_numbers'])]


            # plot discrete characteristics -----
            disc_cols = plot_ind_charac_omitted_gwr_specs['disc_cols']
        
            fig = go.Figure()
            i, col = 0, disc_cols[1]
            for i, col in enumerate(disc_cols):
                unique_categories = omitt_gwregid_gdf[col].unique()
                col_df = omitt_gwregid_gdf[col].value_counts().to_frame().reset_index() 
                col_df ['count'] = col_df['count'] / col_df['count'].sum()
                col_df.sum(axis=0)
                                    
                # j, cat = 0, unique_categories[1]
                for j, cat in enumerate(unique_categories):
                    if col == 'BFS_NUMMER':
                        cat_label = f'{get_bfsnr_name_tuple_list([cat,])}'
                    elif col == 'GKLAS':
                        cat_label = f"{[x for x in plot_ind_charac_omitted_gwr_specs['gwr_code_name_tuples_GKLAS'] if x[0] == cat]}"                        
                    elif col == 'GSTAT':
                        cat_label = f"{[x for x in plot_ind_charac_omitted_gwr_specs['gwr_code_name_tuples_GSTAT'] if x[0] == cat]}"

                    count_value = col_df.loc[col_df[col] == cat, 'count'].values[0]
                    fig.add_trace(go.Bar(x=[col], y=[count_value], 
                        name=cat_label,
                        text=f'{count_value:.2f}',  # Add text to display the count
                        textposition='outside'    # Position the text outside the bar
                    ))
                fig.add_trace(go.Bar(x=[col], y=[0], name=col))  
                # fig.add_trace(go.Bar(x=[col], y=[0], name=''))  
            
            fig.update_layout(  
                barmode='stack',
                xaxis_title='Characteristics',
                yaxis_title='Frequency',
                title = f'Characteristics of omitted GWR EGIDs (scen: {scen})'
            )
                            
            if plot_show and visual_settings['plot_ind_charac_omitted_gwr'][1]:
                if visual_settings['plot_ind_charac_omitted_gwr'][2]:
                    fig.show()
                elif not visual_settings['plot_ind_charac_omitted_gwr'][2]:
                    fig.show() if i_scen == 0 else None
            if visual_settings['save_plot_by_scen_directory']:
                fig.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_ind_pie_disc_charac_omitted_gwr.html')
            else:
                fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_pie_disc_charac_omitted_gwr.html')
            print_to_logfile(f'\texport: plot_ind_pie_disc_charac_omitted_gwr.png (for: {scen})', log_name)



            # plot continuous characteristics -----
            cont_cols = plot_ind_charac_omitted_gwr_specs['cont_cols']
            ncols = 2
            nrows = int(np.ceil(len(cont_cols) / ncols))
            
            fig = make_subplots(rows = nrows, cols = ncols)

            i, col = 0, cont_cols[1]
            for i, col in enumerate(cont_cols):
                if col in omitt_gwregid_gdf.columns:
                    omitt_gwregid_gdf[col].value_counts()
                    col_df  = omitt_gwregid_gdf[col].replace('', np.nan).dropna().astype(float)
                    # if col in ['GBAUJ', 'GBAUM']:
                        # col_df.sort_values(inplace=True)
                    fig.add_trace(go.Histogram(x=col_df, name=col), row = int(i / ncols) + 1, col = i % ncols + 1)
                    fig.update_xaxes(title_text=col, row = int(i / ncols) + 1, col = i % ncols + 1)
                    fig.update_yaxes(title_text='Frequency', row = int(i / ncols) + 1, col = i % ncols + 1)
            fig.update_layout(
                title = f'Continuous Characteristics of omitted GWR EGIDs (scen: {scen})'
            )
            
            if plot_show and visual_settings['plot_ind_charac_omitted_gwr'][1]:
                if visual_settings['plot_ind_charac_omitted_gwr'][2]:
                    fig.show()
                elif not visual_settings['plot_ind_charac_omitted_gwr'][2]:
                    fig.show() if i_scen == 0 else None
            if visual_settings['save_plot_by_scen_directory']:
                fig.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_ind_hist_cont_charac_omitted_gwr.html')
            else:
                fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_hist_cont_charac_omitted_gwr.html')
            print_to_logfile(f'\texport: plot_ind_hist_cont_charac_omitted_gwr.png (for: {scen})', log_name)


    # plot ind - line: meteo radiation over time --------------------
    if visual_settings['plot_ind_line_meteo_radiation'][0]:
        checkpoint_to_logfile(f'plot_ind_line_meteo_radiation', log_name)

        i_scen, scen = 0, scen_dir_export_list[1]
        i_scen, scen = 0, scen_dir_export_list[2]
        for i_scen, scen in enumerate(scen_dir_export_list):
            scen_sett = pvalloc_scen_list[i_scen]
            scen_data_path = f'{data_path}/output/{scen}'
            meteo_col_dir_radiation = scen_sett['weather_specs']['meteo_col_dir_radiation']
            meteo_col_diff_radiation = scen_sett['weather_specs']['meteo_col_diff_radiation']
            meteo_col_temperature = scen_sett['weather_specs']['meteo_col_temperature']

            # import meteo data -----
            meteo = pd.read_parquet(f'{scen_data_path}/meteo_ts.parquet')


            # try to also get raw data to show how radidation is derived
            try: 
                meteo_raw = pd.read_parquet(f'{data_path}/output/{scen_sett["name_dir_import"]}/meteo.parquet')
                meteo_raw = meteo_raw.loc[meteo_raw['timestamp'].isin(meteo['timestamp'])]
                meteo_raw[meteo_col_temperature] = meteo_raw[meteo_col_temperature].astype(float)
            except:
                print('... no raw meteo data available')
                
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            try:  # necessary to accomodate older code versions where radiation is not strictly split into direct and diffuse
                fig.add_trace(go.Scatter(x=meteo['timestamp'], y=meteo[['rad_direct', 'rad_diffuse']].sum(axis = 1), name='Radiation [W/m^2]'))
            except:
                fig.add_trace(go.Scatter(x=meteo['timestamp'], y=meteo['radiation'], name='Radiation [W/m^2]'))
            fig.add_trace(go.Scatter(x=meteo['timestamp'], y=meteo['temperature'], name='Temperature [°C]'), secondary_y=True)
            
            radiation_cols = [meteo_col_dir_radiation, meteo_col_diff_radiation]
            try: 
                for col in radiation_cols:
                    fig.add_trace(go.Scatter(x=meteo_raw['timestamp'], y=meteo_raw[col], name=f'Rad. raw data: {col}'))

                fig.add_trace(go.Scatter(x=meteo_raw['timestamp'], y=meteo_raw[radiation_cols].sum(axis=1), name=f'Rad. raw data: sum of rad types'))
                fig.add_trace(go.Scatter(x=meteo_raw['timestamp'], y=meteo_raw[meteo_col_temperature], name=f'Temp. raw data: {temp_proxy}'))
            except:
                pass

            fig.update_layout(title_text = f'Meteo Data: Temperature and Radiation (if Direct & Diffuse. flat_diffuse_rad_factor: {scen_sett["weather_specs"]["flat_diffuse_rad_factor"]})')
            fig.update_xaxes(title_text='Time')
            fig.update_yaxes(title_text='Radiation [W/m^2]', secondary_y=False)
            fig.update_yaxes(title_text='Temperature [°C]', secondary_y=True)
            
            fig = add_scen_name_to_plot(fig, scen, pvalloc_scen_list[i_scen])
            # fig = set_default_fig_zoom_hour(fig, default_zoom_hour)

            if plot_show and visual_settings['plot_ind_line_meteo_radiation'][1]:
                if visual_settings['plot_ind_line_meteo_radiation'][2]:
                    fig.show()
                elif not visual_settings['plot_ind_line_meteo_radiation'][2]:
                    fig.show() if i_scen == 0 else None
            if visual_settings['save_plot_by_scen_directory']:
                fig.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_ind_line_meteo_radiation.html')
            else:
                fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_line_meteo_radiation.html')




    # PLOT IND SCEN: pvalloc_MC_algorithm ------------------------------------------------------------------------------------------------------

    
    # plot ind - line: Installed Capacity per Month & per BFS --------------------
    if visual_settings['plot_ind_line_installedCap'][0]:#  or visual_settings['plot_ind_line_installedCap_per_BFS']:
        i_scen, scen = 0, scen_dir_export_list[0]

        for i_scen, scen in enumerate(scen_dir_export_list):
            mc_data_path = glob.glob(f'{data_path}/output/{scen}/{mc_str}')[0] # take first path if multiple apply, so code can still run properly
            pvalloc_scen = pvalloc_scen_list[i_scen]

            topo = json.load(open(f'{mc_data_path}/topo_egid.json', 'r'))
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
            pvinst_df['BeginOp'] = pvinst_df['BeginOp'].apply(lambda x: x if len(x) == 10 else x + '-01') # add day to year-month string, to have a proper timestamp
            pvinst_df['BeginOp'] = pd.to_datetime(pvinst_df['BeginOp'], format='%Y-%m-%d')
            pvinst_df['bfs'] = pvinst_df['bfs'].astype(str)


            # plot ind - line: Installed Capacity per Month ===========================
            if visual_settings['plot_ind_line_installedCap'][0]:  #['plot_ind_line_installedCap_per_month']: 
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
                T0_prediction = pvalloc_scen['T0_prediction']
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

                fig1 = add_scen_name_to_plot(fig1, scen, pvalloc_scen_list[i_scen])
                fig1 = set_default_fig_zoom_year(fig1, default_zoom_year, capa_year_df, 'BeginOp_year')
                
                if plot_show and visual_settings['plot_ind_line_installedCap'][1]:
                    fig1.show()
                if visual_settings['save_plot_by_scen_directory']:
                    fig1.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_ind_line_installedCap_per_month.html')
                else:
                    fig1.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_line_installedCap_per_month.html')
                print_to_logfile(f'\texport: plot_ind_line_installedCap_per_month.html (for: {scen})', log_name)


            # plot ind - line: Installed Capacity per BFS ===========================
            if visual_settings['plot_ind_line_installedCap'][0]:  #plot_ind_line_installedCap_per_BFS']: 
                checkpoint_to_logfile(f'plot_ind_line_installedCap_per_BFS', log_name)
                capa_bfs_df = pvinst_df.copy()
                gm_gdf = gpd.read_file(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/gm_shp_gdf.geojson')                                         
                gm_gdf.rename(columns={'BFS_NUMMER': 'bfs'}, inplace=True)
                gm_gdf['bfs'] = gm_gdf['bfs'].astype(str)
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
                    fig2.add_trace(go.Scatter(x=subdf['BeginOp_month'], y=subdf['TotalPower'], name=f'{name} (by month)', legendgroup = 'By Month',  mode = 'lines'))

                for bfs in capa_bfs_year_df['bfs'].unique():
                    name = gm_gdf.loc[gm_gdf['bfs'] == bfs, 'NAME'].values[0]
                    subdf = capa_bfs_year_df.loc[capa_bfs_year_df['bfs'] == bfs].copy()
                    fig2.add_trace(go.Scatter(x=subdf['BeginOp_year'], y=subdf['TotalPower'], name=f'{name} (by year)', legendgroup = 'By Year', mode = 'lines'))

                fig2.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Installed Capacity (kW)',
                    legend_title='BFS',
                    title = f'Installed Capacity per Municipality (BFS) (weather year: {pvalloc_scen["weather_specs"]["weather_year"]})',
                    showlegend=True, 
                    legend=dict(
                        title='Legend',  # You can customize the legend title here
                        itemsizing='trace',  # Control the legend item sizing (can be 'trace' or 'constant')
                    )
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
                
                fig2 = add_scen_name_to_plot(fig2, scen, pvalloc_scen_list[i_scen])
                fig2 = set_default_fig_zoom_year(fig2, default_zoom_year, capa_bfs_year_df, 'BeginOp_year')
                # if plot_show:
                #     fig2.show()
                if visual_settings['save_plot_by_scen_directory']:
                    fig2.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_ind_line_installedCap_per_BFS.html')
                else:
                    fig2.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_line_installedCap_per_BFS.html')
                print_to_logfile(f'\texport: plot_ind_line_installedCap_per_BFS.html (for: {scen})', log_name)


    # plot ind - line: Production + Feedin HOY per Node --------------------
    if visual_settings['plot_ind_line_productionHOY_per_node'][0]:
        checkpoint_to_logfile(f'plot_ind_line_productionHOY_per_node', log_name)
        i_scen, scen = 0, scen_dir_export_list[0]
        for i_scen, scen in enumerate(scen_dir_export_list):

            # setup + import ----------
            mc_data_path = glob.glob(f'{data_path}/output/{scen}/{mc_str}')[0] # take first path if multiple apply, so code can still run properly
            pvalloc_scen = pvalloc_scen_list[i_scen]

            node_selection = visual_settings['node_selection_for_plots']

            gridnode_df = pd.read_parquet(f'{mc_data_path}/gridnode_df.parquet')
            gridnode_df['grid_node'].unique()
            gridnode_df['t_int'] = gridnode_df['t'].str.extract(r't_(\d+)').astype(int)
            gridnode_df.sort_values(by=['t_int'], inplace=True)

            # plot ----------------
            # unclear why if statement is necessary here? maybe older data versions featured col 'info_source'
            if 'info_source' in gridnode_df.columns:
                if isinstance(node_selection, list):
                    nodes = node_selection
                elif node_selection == None:
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
            
            else:
                if isinstance(node_selection, list):
                    nodes = node_selection
                elif node_selection == None:
                    nodes = gridnode_df['grid_node'].unique()

                fig = go.Figure()
                for node in nodes:
                    filter_df = copy.deepcopy(gridnode_df.loc[gridnode_df['grid_node'] == node])
                    fig.add_trace(go.Scatter(x=filter_df['t'], y=filter_df['feedin_kW'], name=f'{node} - feedin (all)'))
                    fig.add_trace(go.Scatter(x=filter_df['t'], y=filter_df['feedin_kW_taken'], name= f'{node} - feedin_taken'))
                    fig.add_trace(go.Scatter(x=filter_df['t'], y=filter_df['feedin_kW_loss'], name=f'{node} - feedin_loss'))

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
                title = f'Production per node (kW, weather year: {pvalloc_scen["weather_specs"]["weather_year"]}, self consum. rate: {pvalloc_scen["tech_economic_specs"]["self_consumption_ifapplicable"]})'
            )


            fig = add_scen_name_to_plot(fig, scen, pvalloc_scen_list[i_scen])
            fig = set_default_fig_zoom_hour(fig, default_zoom_hour)

            if plot_show and visual_settings['plot_ind_line_productionHOY_per_node'][1]:	
                fig.show() 
            if visual_settings['save_plot_by_scen_directory']:
                fig.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_ind_line_productionHOY_per_node.html')
            else:
                fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_line_productionHOY_per_node.html')


    # plot ind - hist: NPV possible PV inst before / after --------------------
    if visual_settings['plot_ind_hist_NPV_freepartitions'][0]:
        checkpoint_to_logfile(f'plot_ind_hist_NPV_freepartitions', log_name)
        i_scen, scen = 0, scen_dir_export_list[0]
        for i_scen, scen in enumerate(scen_dir_export_list):
            # setup + import ----------
            mc_data_path = glob.glob(f'{data_path}/output/{scen}/{mc_str}')[0] # take first path if multiple apply, so code can still run properly
            pvalloc_scen = pvalloc_scen_list[i_scen]

            npv_df_paths = glob.glob(f'{mc_data_path}/pred_npv_inst_by_M/npv_df_*.parquet')
            periods_list = [pd.to_datetime(path.split('npv_df_')[-1].split('.parquet')[0]) for path in npv_df_paths]
            before_period, after_period = min(periods_list), max(periods_list)

            npv_df_before = pd.read_parquet(f'{mc_data_path}/pred_npv_inst_by_M/npv_df_{before_period.to_period("M")}.parquet')
            npv_df_after  = pd.read_parquet(f'{mc_data_path}/pred_npv_inst_by_M/npv_df_{after_period.to_period("M")}.parquet')

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

            fig = add_scen_name_to_plot(fig, scen, pvalloc_scen_list[i_scen])
            
            if plot_show and visual_settings['plot_ind_hist_NPV_freepartitions'][1]:
                fig.show()
            if visual_settings['save_plot_by_scen_directory']:
                fig.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_ind_hist_NPV_freepartitions.html')
            else:
                fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_hist_NPV_freepartitions.html')


    # plot ind - hist: PV Capacity and Production per Year --------------------
    # if visual_settings['plot_ind_hist_pvcapaprod'][0]:
    if False:
        checkpoint_to_logfile(f'plot_ind_hist_pvcapaprod', log_name)
        i_scen, scen = 0, scen_dir_export_list[0]
        for i_scen, scen in enumerate(scen_dir_export_list):
            # setup + import ----------
            mc_data_path = glob.glob(f'{data_path}/output/{scen}/{mc_str}')[0]
            topo_subdf_paths = glob.glob(f'{data_path}/output/{scen}/topo_time_subdf/topo_subdf_*.parquet')
            pv = pd.read_parquet(f'{data_path}/output/{scen}/pv.parquet')

            kWpeak_per_m2, share_roof_area_available = pvalloc_scen['tech_economic_specs']['kWpeak_per_m2'],pvalloc_scen['tech_economic_specs']['share_roof_area_available']
            inverter_efficiency = pvalloc_scen['tech_economic_specs']['inverter_efficiency']
            panel_efficiency_print = 'dynamic' if pvalloc_scen['panel_efficiency_specs']['variable_panel_efficiency_TF'] else 'static'
            color_pv_df, color_alloc_algo, color_rest = visual_settings['plot_ind_map_topo_egid_specs']['point_color_pv_df'], visual_settings['plot_ind_map_topo_egid_specs']['point_color_alloc_algo'], visual_settings['plot_ind_map_topo_egid_specs']['point_color_rest']
            
            xbins_hist_instcapa_abs, xbins_hist_instcapa_stand = 0.5,0.1
            xbins_hist_totalprodkwh_abs, xbins_hist_totalprodkwh_stand = 500, 0.05

            topo = json.load(open(f'{mc_data_path}/topo_egid.json', 'r'))
            egid_in_topo = list(topo.keys())

            # installed Capapcity kW --------------------------------            
            egid_list, xtf_id_list, TotalPower_list, info_source_list = [], [], [], []
            for k, v in topo.items():
                if v['pv_inst']['inst_TF']:
                    egid_list.append(k)
                    TotalPower_list.append(v['pv_inst']['TotalPower'])
                    info_source_list.append(v['pv_inst']['info_source'])
                    xtf_id_list.append(v['pv_inst']['xtf_id'])
            pvinst_df = pd.DataFrame({'EGID': egid_list, 'info_source': info_source_list, 'xtf_id': xtf_id_list, 'TotalPower': TotalPower_list})

            # plot 
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=pvinst_df['TotalPower'], 
                                    name='Total Capacity (kW)', 
                                    opacity = 0.5, marker_color = color_rest,
                                    xbins=dict(size=xbins_hist_instcapa_abs)))
            fig.add_trace(go.Histogram(x=pvinst_df.loc[pvinst_df['info_source'] == 'pv_df', 'TotalPower'],
                                    name='Built Capacity (kW) - PV_df', 
                                    opacity = 0.5, marker_color = color_pv_df,
                                    xbins=dict(size=xbins_hist_instcapa_abs)))
            fig.add_trace(go.Histogram(x=pvinst_df.loc[pvinst_df['info_source'] == 'alloc_algorithm', 'TotalPower'],
                                    name='Modelled Capacity (kW) - Alloc. Algorithm', 
                                    opacity = 0.5, marker_color = color_alloc_algo,
                                    xbins=dict(size=xbins_hist_instcapa_abs)))      
            fig.update_layout(
                barmode='overlay',
                xaxis_title='Installed Capacity (kW)',
                yaxis_title='Frequency',
                title = f'Modelled vs Installed Capacity (MC_iter: {mc_str}, kWp_m2:{kWpeak_per_m2}, share roof available: {share_roof_area_available})'
            )
            fig = add_scen_name_to_plot(fig, scen, pvalloc_scen_list[i_scen])

            if plot_show and visual_settings['plot_ind_hist_pvcapaprod'][1]:
                fig.show()  
            fig.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_ind_hist_pvcapaprod.html')


            # Production kWh --------------------------------
            egid_with_inst = [k for k, v in topo.items() if v['pv_inst']['inst_TF']]

            aggdf_combo_list = []
            path = topo_subdf_paths[0]
            for i_path, path in enumerate(topo_subdf_paths):
                subdf = pd.read_parquet(path)
                subdf = subdf.loc[subdf['EGID'].isin(pvinst_df['EGID'])]

                agg_subdf = subdf.groupby(['EGID', 'df_uid', 'FLAECHE', 'STROMERTRAG',]).agg({'pvprod_kW': 'sum'}).reset_index()

                egid_list, flaeche_list, pvprod_list, stromertrag_list = [], [], [], []
                egid  = agg_subdf['EGID'].unique()[0]
                for egid in subdf['EGID'].unique():
                    egid_list.append(egid)

                    # BOOKMARK!!!
                    # > extract pvprod for all egids that have built in the MC iteration and sum them up
                    # > later plot histogram similar to sanity check of installed capacity. 
                    # problem !
                    # > how to treat the installation on existing buildings? 
                    # how did i treat these cases in the feedin to the grid part of the code?
                    # would have to be identical approach
                    
                    # if topo.get(egid).get('pv_inst').get('info_source') == 'pv_df':

                    # pvinst_df.loc[pvinst_df['EGID'] == egid, 'info_source'].values
                    # df_uids = pvinst_df.loc[pvinst_df['EGID'] == egid, 'xtf_id'].values


    # plot ind - map:  Model PV topology ========================
    default_map_zoom = visual_settings['default_map_zoom']
    default_map_center = visual_settings['default_map_center']

    # map ind - topo_egid --------------------
    if visual_settings['plot_ind_map_topo_egid'][0]:
        map_topo_egid_specs = visual_settings['plot_ind_map_topo_egid_specs']
        checkpoint_to_logfile(f'plot_ind_map_topo_egid', log_name)

        for i_scen, scen in enumerate(scen_dir_export_list):
            pvalloc_scen = pvalloc_scen_list[i_scen]
            
            # get pvinst_gdf ----------------
            if True: 
                mc_data_path = glob.glob(f'{data_path}/output/{scen}/{mc_str}')[0] # take first path if multiple apply, so code can still run properly
                
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


    # map ind - node_connections ============================
    if visual_settings['plot_ind_map_node_connections'][0]:
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







    # PLOT AGGREGATED SCEN ------------------------------------------------------------------------------------------------------




    # ********************************************************************************************************************************************************
    # ********************************************************************************************************************************************************
    # ********************************************************************************************************************************************************
    # ********************************************************************************************************************************************************






    # map ind - omitted gwr_egids ============================
    if visual_settings['plot_ind_map_omitted_gwr_egids']:
        map_topo_omit_specs = visual_settings['plot_ind_map_topo_omitt_specs']
        checkpoint_to_logfile(f'plot_ind_map_ommitted_gwr_egids', log_name)

        for i_scen, scen in enumerate(scen_dir_export_list):
            scen_sett = pvalloc_scenarios[f'{scen}']

            # omitted egids from data prep -----            
            omitt_gwregid_gdf_all = gpd.read_file(f'{data_path}/output/{scen_sett["name_dir_import"]}/omitt_gwregid_gdf.geojson')
            omitt_gwregid_gdf_all.rename(columns={'GGDENR': 'BFS_NUMMER'}, inplace=True)
            omitt_gwregid_gdf_all['BFS_NUMMER'] = omitt_gwregid_gdf_all['BFS_NUMMER'].astype(int)
            omitt_gwregid_gdf = omitt_gwregid_gdf_all.loc[omitt_gwregid_gdf_all['BFS_NUMMER'].isin(scen_sett['bfs_numbers'])]

            # topo omitt map ----------------
            fig_topoomitt = copy.deepcopy(fig_topoegid)
            
            omitt_gwregid_gdf = omitt_gwregid_gdf.set_crs('EPSG:2056', allow_override=True)
            omitt_gwregid_gdf = omitt_gwregid_gdf.to_crs('EPSG:4326')
            omitt_gwregid_gdf['geometry'] = omitt_gwregid_gdf['geometry'].apply(flatten_geometry)

            omitt_gwregid_gdf['hover_text'] = omitt_gwregid_gdf.apply(lambda row: f'EGID: {row["EGID"]}<br>BFS_NUMMER: {str(row["BFS_NUMMER"])}<br>GSTAT: {row["GSTAT"]}<br>GKAT: {row["GKAT"]}<br>GKLAS: {row["GKLAS"]}<br>GBAUJ: {row["GBAUJ"]}<br>GBAUM: {row["GBAUM"]}<br>GANZWHG: {row["GANZWHG"]}<br>GAREA: {row["GAREA"]}<br>WAZIM: {row["WAZIM"]} (no. rooms)<br>WAREA: {row["WAREA"]} (living area m2)', axis = 1)

            fig_topoomitt.add_trace(go.Scattermapbox(lat=omitt_gwregid_gdf.geometry.y,lon=omitt_gwregid_gdf.geometry.x, mode='markers',
                marker=dict(
                    size=map_topo_omit_specs['point_size'],
                    color=map_topo_omit_specs['point_color'],
                    opacity = map_topo_omit_specs['point_opacity']
                ),
                name = 'omitted EGIDs (not in solkat)',
                text=omitt_gwregid_gdf['hover_text'],
                hoverinfo='text'
                ))         

            fig_topoomitt.update_layout(
                        title=f"Map of PV Topology ({scen}) and *omitted* EGIDs",
                        mapbox=dict(
                            style="carto-positron",
                            center={"lat": default_map_center[0], "lon": default_map_center[1]},  # Center the map on the region
                            zoom=default_map_zoom
                        )
                    )                                           

            if plot_show:
                fig_topoomitt.show()
            fig_topoomitt.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_map_omitted_gwr_egids.html')




    # PLOT AGGREGATED SCEN ------------------------------------------------------------------------------------------------------
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
