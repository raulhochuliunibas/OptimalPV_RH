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
# PLOT INDIVIDUAL HISTOGRAM PV CAPACITY + PRODUCTION for SANITY CHECK
# ------------------------------------------------------------------------------------------------------

def plot(pvalloc_scen_list, 
         visual_settings, 
         wd_path, 
         data_path, 
         log_name):
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]
    plot_show = visual_settings['plot_show']    
    

    if visual_settings['plot_ind_hist_pvcapaprod_sanitycheck'][0]:

        checkpoint_to_logfile(f'plot_ind_hist_pvcapaprod_sanitycheck', log_name)

        # available color palettes
        trace_color_dict = {
            'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
            'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
            'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Plotly3': pc.sequential.Plotly3,
            'Viridis': pc.sequential.Viridis, 'Turbo': pc.sequential.Turbo, 'Blackbody': pc.sequential.Blackbody
        }        
        
        # visual settings
        plot_ind_hist_pvcapaprod_sanitycheck_specs= visual_settings['plot_ind_hist_pvcapaprod_sanitycheck_specs']
        xbins_hist_instcapa_abs, xbins_hist_instcapa_stand = plot_ind_hist_pvcapaprod_sanitycheck_specs['xbins_hist_instcapa_abs'], plot_ind_hist_pvcapaprod_sanitycheck_specs['xbins_hist_instcapa_stand']
        xbins_hist_totalprodkwh_abs, xbins_hist_totalprodkwh_stand = plot_ind_hist_pvcapaprod_sanitycheck_specs['xbins_hist_totalprodkwh_abs'], plot_ind_hist_pvcapaprod_sanitycheck_specs['xbins_hist_totalprodkwh_stand']
        
        trace_colval_max, trace_colincr, uniform_scencolor_and_KDE_TF = plot_ind_hist_pvcapaprod_sanitycheck_specs['trace_colval_max'], plot_ind_hist_pvcapaprod_sanitycheck_specs['trace_colincr'], plot_ind_hist_pvcapaprod_sanitycheck_specs['uniform_scencolor_and_KDE_TF']
        trace_color_palettes = plot_ind_hist_pvcapaprod_sanitycheck_specs['trace_color_palettes']
        trace_color_palettes_list= [trace_color_dict[color] for color in trace_color_palettes]

        color_pv_df, color_solkat, color_rest = visual_settings['plot_ind_map_topo_egid_specs']['point_color_pv_df'], visual_settings['plot_ind_map_topo_egid_specs']['point_color_solkat'],visual_settings['plot_ind_map_topo_egid_specs']['point_color_rest']
        
        # switch to rerun plot for uniform_scencolor_and_KDE_TF if turned on 
        if uniform_scencolor_and_KDE_TF:
            uniform_scencolor_and_KDE_TF_list = [True, False]
        elif not uniform_scencolor_and_KDE_TF:
            uniform_scencolor_and_KDE_TF_list = [False,]


        # plot --------------------
        for uniform_scencolor_and_KDE_TF in uniform_scencolor_and_KDE_TF_list:
            fig_agg_abs, fig_agg_stand = go.Figure(), go.Figure()

            i_scen, scen = 0, scen_dir_export_list[0]
            for i_scen, scen in enumerate(scen_dir_export_list):
                pvalloc_scen = pvalloc_scen_list[i_scen]

                kWpeak_per_m2, share_roof_area_available = pvalloc_scen['tech_economic_specs']['kWpeak_per_m2'],pvalloc_scen['tech_economic_specs']['share_roof_area_available']
                inverter_efficiency = pvalloc_scen['tech_economic_specs']['inverter_efficiency']
                panel_efficiency_print = 'dynamic' if pvalloc_scen['panel_efficiency_specs']['variable_panel_efficiency_TF'] else 'static'

                # data import
                sanity_scen_data_path = f'{data_path}/output/{scen}/sanity_check_byEGID'
                pv = pd.read_parquet(f'{data_path}/output/{scen}/pv.parquet')
                topo = json.load(open(f'{sanity_scen_data_path}/topo_egid.json', 'r'))
                gwr_gdf = gpd.read_file(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/gwr_gdf.geojson')

                egid_with_pvdf = [egid for egid in topo.keys() if topo[egid]['pv_inst']['info_source'] == 'pv_df']
                xtf_in_topo = [topo[egid]['pv_inst']['xtf_id'] for egid in egid_with_pvdf]
                topo_subdf_paths = glob.glob(f'{sanity_scen_data_path}/topo_subdf_*.parquet')
                topo.get(egid_with_pvdf[0])
                
                aggdf_combo_list = []
                path = topo_subdf_paths[0]
                for i_path, path in enumerate(topo_subdf_paths):
                    subdf= pd.read_parquet(path)
                    # subdf = subdf.loc[subdf['EGID'].isin(egid_with_pvdf)]
                    # compute pvprod by using TotalPower of pv_df, check if it overlaps with computation of STROMERTRAG
                    # subdf['pvprod_TotalPower_kW'] = subdf['radiation_rel_locmax'] * subdf['TotalPower'] *  inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] 
                    subdf.loc[:, 'pvprod_TotalPower_kW'] = subdf['radiation_rel_locmax'] * subdf['TotalPower'] * inverter_efficiency * share_roof_area_available * subdf['panel_efficiency']
                    

                    agg_subdf = subdf.groupby(['EGID', 'df_uid', 'FLAECHE', 'STROMERTRAG']).agg({'pvprod_kW': 'sum', 
                                                                                                'pvprod_TotalPower_kW': 'sum'}).reset_index()
                    aggsub_npry = np.array(agg_subdf)
                    
                    egid_list, dfuid_list, flaeche_list, pvprod_list, pvprod_ByTotalPower_list, stromertrag_list = [], [], [], [], [], []
                    egid = subdf['EGID'].unique()[0]

                    for egid in subdf['EGID'].unique():
                        mask_egid_subdf = np.isin(aggsub_npry[:,agg_subdf.columns.get_loc('EGID')], egid)
                        df_uids  = list(aggsub_npry[mask_egid_subdf, agg_subdf.columns.get_loc('df_uid')])

                        for r in range(1,len(df_uids)+1):
                            for combo in itertools.combinations(df_uids,r):
                                combo_key_str = '_'.join([str(c) for c in combo])
                                mask_dfuid_subdf = np.isin(aggsub_npry[:,agg_subdf.columns.get_loc('df_uid')], list(combo))

                                egid_list.append(egid)
                                dfuid_list.append(combo_key_str)
                                flaeche_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('FLAECHE')].sum())
                                pvprod_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pvprod_kW')].sum())
                                pvprod_ByTotalPower_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pvprod_TotalPower_kW')].sum())
                                stromertrag_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('STROMERTRAG')].sum())
                                
                    aggsubdf_combo = pd.DataFrame({'EGID': egid_list, 'df_uid': dfuid_list, 
                                                   'FLAECHE': flaeche_list, 'pvprod_kW': pvprod_list, 
                                                   'pvprod_ByTotalPower_kW': pvprod_ByTotalPower_list,
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


                # Functions for Agg traces plot  --------------------------------
                if True:
                    palette = trace_color_palettes_list[i_scen % len(trace_color_palettes_list)]
                    trace_colpal = pc.sample_colorscale(palette, [i/(trace_colval_max-1) for i in range(trace_colval_max)])
                    trace_colval = trace_colval_max

                    # get color from a specific palette
                    def col_from_palette(trace_colpal, trace_colval):
                        trace_colval = max(0, min(trace_colval_max, trace_colval))
                        return trace_colpal[trace_colval-1]
                    
                    def update_trace_color(trace_colval, trace_colincr):
                        return trace_colval - trace_colincr
                    

                    # kernel density traces
                    def add_kde_gaussian_trace(fig, data, name, color, uniform_col_AND_KDE = uniform_scencolor_and_KDE_TF):
                        kde = gaussian_kde(data)
                        x_range = np.linspace(min(data), max(data), 1000)
                        if uniform_col_AND_KDE:
                            fig.add_trace(go.Scatter(x=x_range, y=kde(x_range), mode='lines', line = dict(color = color), 
                                                        name=f'{name}', yaxis= 'y2'))  
                                    
                    # histogram traces
                    def add_histogram_trace(fig, data, name, color, xbins, uniform_col_AND_KDE = uniform_scencolor_and_KDE_TF):
                        if uniform_col_AND_KDE:
                            fig.add_trace(go.Histogram(x=data, name=name, opacity=0.5, xbins= dict(size=xbins), marker_color = color,))
                        else:
                            fig.add_trace(go.Histogram(x=data, name=name, opacity=0.5, xbins= dict(size=xbins),))

                    def add_scen_title_traces(fig, scen):
                        fig.add_trace(go.Scatter(x=[0,], y=[0,], name=f'', opacity=0,))
                        fig.add_trace(go.Scatter(x=[0,], y=[0,], name=f'{scen}', opacity=0,))         


                # Agg: Absolute Values --------------------------------   
                if True:
                    add_scen_title_traces(fig_agg_abs, scen)

                    # inst capacity kW
                    add_histogram_trace(fig_agg_abs, aggdf_combo['inst_capa_kW'],
                                        f' - Modeled Potential Capacity (kW)',   
                                        col_from_palette(trace_colpal, trace_colval), xbins_hist_instcapa_abs)
                    trace_colval = update_trace_color(trace_colval, trace_colincr)

                    add_histogram_trace(fig_agg_abs, pv.loc[pv['xtf_id'].isin(xtf_in_topo), 'TotalPower'],
                                        f' - Installed Capacity (pv_df in topo, kW)',
                                        col_from_palette(trace_colpal, trace_colval), xbins_hist_instcapa_abs)
                    trace_colval = update_trace_color(trace_colval, trace_colincr)

                    # annual PV production kWh
                    add_histogram_trace(fig_agg_abs, aggdf_combo['pvprod_kW'], 
                                        f' - Modeled Potential Yearly Production (kWh)', 
                                        col_from_palette(trace_colpal, trace_colval), xbins_hist_totalprodkwh_abs)
                    add_kde_gaussian_trace(fig_agg_abs, aggdf_combo['pvprod_kW'],f'  KDE Modeled Potential Yearly Production (kWh)',col_from_palette(trace_colpal,trace_colval), )
                    trace_colval = update_trace_color(trace_colval, trace_colincr)

                    add_histogram_trace(fig_agg_abs, aggdf_combo['STROMERTRAG'], 
                                        f' - STROMERTRAG (solkat estimated production)', 
                                        col_from_palette(trace_colpal, trace_colval), xbins_hist_totalprodkwh_abs)
                    add_kde_gaussian_trace(fig_agg_abs, aggdf_combo['STROMERTRAG'],f'  KDE STROMERTRAG (solkat estimated production)',col_from_palette(trace_colpal,trace_colval), )
                    trace_colval = update_trace_color(trace_colval, trace_colincr)

                    add_histogram_trace(fig_agg_abs, aggdf_combo['pvprod_ByTotalPower_kW'],
                                        f' - Yearly Prod. TotalPower (pvdf estimated production)', 
                                        col_from_palette(trace_colpal, trace_colval), xbins_hist_totalprodkwh_abs)
                    add_kde_gaussian_trace(fig_agg_abs, aggdf_combo['pvprod_ByTotalPower_kW'],f'  KDE Yearly Prod. TotalPower (pvdf estimated production)',col_from_palette(trace_colpal,trace_colval), )
                    trace_colval = update_trace_color(trace_colval, trace_colincr) 


                # Agg: Standardized Values --------------------------------
                if True:
                    trace_colval = trace_colval_max
                    add_scen_title_traces(fig_agg_stand, scen)  

                    # inst capacity kW
                    add_histogram_trace(fig_agg_stand, aggdf_combo['inst_capa_kW_stand'],
                                        f' - Modeled Potential Capacity (kW), standardized',   
                                        col_from_palette(trace_colpal, trace_colval), xbins_hist_instcapa_stand)
                    add_kde_gaussian_trace(fig_agg_stand, aggdf_combo['inst_capa_kW_stand'],f'  KDE Modeled Potential Capacity (kW)',col_from_palette(trace_colpal,trace_colval), )
                    trace_colval = update_trace_color(trace_colval, trace_colincr)

                    add_histogram_trace(fig_agg_stand, pv['TotalPower_stand'],
                                        f' - Installed Capacity (pv_df in topo), standardized',
                                        col_from_palette(trace_colpal, trace_colval), xbins_hist_instcapa_stand)
                    add_kde_gaussian_trace(fig_agg_stand, pv['TotalPower_stand'],f'  KDE Installed Capacity (pv_df in topo, kW)',col_from_palette(trace_colpal,trace_colval), )
                    trace_colval = update_trace_color(trace_colval, trace_colincr)

                    # annual PV production kWh
                    add_histogram_trace(fig_agg_stand, aggdf_combo['pvprod_kW_stand'],
                                        f' - Modeled Potential Yearly Production (kWh), standardized', 
                                        col_from_palette(trace_colpal, trace_colval), xbins_hist_totalprodkwh_stand)
                    add_kde_gaussian_trace(fig_agg_stand, aggdf_combo['pvprod_kW_stand'],f'  KDE Modeled Potential Yearly Production (kWh), standardized',col_from_palette(trace_colpal,trace_colval), )
                    trace_colval = update_trace_color(trace_colval, trace_colincr)

                    add_histogram_trace(fig_agg_stand, aggdf_combo['STROMERTRAG_stand'],
                                        f' - STROMERTRAG (solkat estimated production), standardized', 
                                        col_from_palette(trace_colpal, trace_colval), xbins_hist_totalprodkwh_stand)
                    add_kde_gaussian_trace(fig_agg_stand, aggdf_combo['STROMERTRAG_stand'],f'  KDE STROMERTRAG (solkat estimated production), standardized',col_from_palette(trace_colpal,trace_colval), )
                    trace_colval = update_trace_color(trace_colval, trace_colincr)

                    add_histogram_trace(fig_agg_stand, aggdf_combo['pvprod_ByTotalPower_kW_stand'],
                                        f' - Yearly Prod. TotalPower (pvdf estimated production), standardized', 
                                        col_from_palette(trace_colpal, trace_colval), xbins_hist_totalprodkwh_stand)
                    add_kde_gaussian_trace(fig_agg_stand, aggdf_combo['pvprod_ByTotalPower_kW_stand'],f'  KDE Yearly Prod. TotalPower (pvdf estimated production), standardized',col_from_palette(trace_colpal,trace_colval), )
                    trace_colval = update_trace_color(trace_colval, trace_colincr)


            # Export Agg plots --------------------------------
            if True:
                fig_agg_abs.update_layout(
                    barmode='overlay',
                    xaxis_title='Capacity [kW]',
                    yaxis_title='Frequency (Modelled Capacity, possible installations)',
                    title = f'SANITY CHECK: Agg. Modelled vs Installed Cap. & Yearly Prod. ABSOLUTE (kWp_m2:{kWpeak_per_m2}, share roof available: {share_roof_area_available}, {panel_efficiency_print} panel eff, inverter eff: {inverter_efficiency})',
                    yaxis2 = dict(overlaying='y', side='right', title='')
                )
                fig_agg_stand.update_layout(
                    barmode='overlay',
                    xaxis_title='Production [kWh]',
                    yaxis_title='Frequency, absolute',
                    title = f'SANITY CHECK: Agg. Modelled vs Installed Cap. & Yearly Prod. STANDARDIZED (kWp_m2:{kWpeak_per_m2}, share roof available: {share_roof_area_available}, {panel_efficiency_print} panel eff, inverter eff: {inverter_efficiency})',
                    yaxis2 = dict(overlaying='y', side='right', title='', position=0.95, ),
                )


                if plot_show and visual_settings['plot_ind_hist_pvcapaprod_sanitycheck'][1]:
                    fig_agg_abs.show()
                    fig_agg_stand.show()
                fig_agg_abs.write_html(f'{data_path}/output/visualizations/plot_agg_hist_pvCapaProd_SanityCheck_abs_values__{len(scen_dir_export_list)}scen_KDE{uniform_scencolor_and_KDE_TF}.html')   
                fig_agg_stand.write_html(f'{data_path}/output/visualizations/plot_agg_hist_pvCapaProd_SanityCheck_stand_values__{len(scen_dir_export_list)}scen_KDE{uniform_scencolor_and_KDE_TF}.html')
                print_to_logfile(f'\texport: plot_agg_hist_SanityCheck_instCapa_kW.html ({len(scen_dir_export_list)} scens, KDE: {uniform_scencolor_and_KDE_TF})', log_name)
    
            # Export shapes with 0 kWh annual production --------------------
            if plot_ind_hist_pvcapaprod_sanitycheck_specs['export_spatial_data_for_prod0']:
                # EGID_no_prod = aggdf_combo.loc[aggdf_combo['pvprod_kW'] == 0, 'EGID'].unique()
                aggdf_combo_noprod = aggdf_combo.loc[aggdf_combo['pvprod_kW'] == 0]

                # match GWR geom to gdf
                aggdf_noprod_gwrgeom_gdf = aggdf_combo_noprod.merge(gwr_gdf, on='EGID', how='left')
                aggdf_noprod_gwrgeom_gdf = gpd.GeoDataFrame(aggdf_noprod_gwrgeom_gdf, geometry='geometry')
                aggdf_noprod_gwrgeom_gdf.to_file(f'{data_path}/output/{scen}/topo_spatial_data/aggdf_noprod_gwrgeom_gdf.geojson', driver='GeoJSON')
                checkpoint_to_logfile(f'\texport: aggdf_noprod_gwrgeom_gdf.geojson (scen: {scen}) for sanity check', log_name)

                # try to match solkat geom to gdf
                solkat_gdf = gpd.read_file(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/solkat_gdf.geojson')
                aggdf_noprod_solkatgeom_gdf = copy.deepcopy(aggdf_combo_noprod)
                aggdf_noprod_solkatgeom_gdf['geometry'] = 'NA'
                
                # i, row = 0, aggdf_combo_noprod.iloc[0]
                # i, row = 2, aggdf_combo_noprod.iloc[2]
                for i, row in aggdf_combo_noprod.iterrows():
                    dfuid_row_list = row['df_uid'].split('_')
                    dfuid_row_solkatgeom = solkat_gdf.loc[solkat_gdf['DF_UID'].isin(dfuid_row_list)]
                    if dfuid_row_solkatgeom.shape[0] == 1:
                        aggdf_noprod_solkatgeom_gdf.loc[i, 'geometry'] = dfuid_row_solkatgeom.iloc[0]['geometry']
                    elif dfuid_row_solkatgeom.shape[0] > 1:
                        # aggdf_noprod_solkatgeom_gdf.loc[i, 'geometry'] = MultiPolygon([geom for geom in dfuid_row_solkatgeom['geometry']])
                        aggdf_noprod_gwrgeom_gdf.loc[i, 'geometry'] = dfuid_row_solkatgeom.unary_union
                    elif len(dfuid_row_solkatgeom) == 0:
                        aggdf_noprod_solkatgeom_gdf.loc[i, 'geometry'] = 'NA_dfuid_aggdf_combo_notin_solkat_gdf'

                aggdf_noprod_solkatgeom_gdf.loc[aggdf_noprod_solkatgeom_gdf['geometry'] == 'NA', 'geometry'] = None
                aggdf_noprod_solkatgeom_gdf = gpd.GeoDataFrame(aggdf_noprod_solkatgeom_gdf, geometry='geometry')
                aggdf_noprod_solkatgeom_gdf.to_file(f'{data_path}/output/{scen}/topo_spatial_data/aggdf_noprod_solkatgeom_gdf.geojson', driver='GeoJSON')
                checkpoint_to_logfile(f'\texport: aggdf_noprod_solkatgeom_gdf.geojson (scen: {scen}) for sanity check', log_name)







         
