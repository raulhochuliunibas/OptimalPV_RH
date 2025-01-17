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
# PLOT INDIVIDUAL LINE for INSTALLED CAPACITY
# ------------------------------------------------------------------------------------------------------

def plot(pvalloc_scen_list, 
         visual_settings, 
         wd_path, 
         data_path, 
         log_name):
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]
    plot_show = visual_settings['plot_show']
    default_zoom_year = visual_settings['default_zoom_year']


    if visual_settings['plot_ind_line_installedCap'][0]:#  or visual_settings['plot_ind_line_installedCap_per_BFS']:

        checkpoint_to_logfile(f'plot_ind_line_installedCap', log_name)
        # available color palettes
        trace_color_dict = {
            'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
            'Viridis': pc.sequential.Viridis, 'Plotly3': pc.sequential.Plotly3, 
            'Turbo': pc.sequential.Turbo, 'solar': pc.sequential.solar, 
            'RdBu': pc.diverging.RdBu, 'Rainbow': pc.sequential.Rainbow, 

            'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
            'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
            'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Blackbody': pc.sequential.Blackbody, 
            
            # 'Bluered': pc.sequential.Bluered, 
        }        


        i_scen, scen = 0, scen_dir_export_list[0]

        fig_agg_pmonth = go.Figure()
        for i_scen, scen in enumerate(scen_dir_export_list):
            mc_data_path = glob.glob(f'{data_path}/output/{scen}/{visual_settings['MC_subdir_for_plot']}')[0] # take first path if multiple apply, so code can still run properly
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

                capa_cumm_year_df =  pvinst_df.copy()
                capa_cumm_year_df['BeginOp_year'] = capa_cumm_year_df['BeginOp'].dt.to_period('Y')
                # capa_cumm_year_df.sort_values(by='BeginOp_year', inplace=True)
                capa_cumm_year_df = capa_cumm_year_df.groupby(['BeginOp_year',])['TotalPower'].sum().reset_index().copy()
                capa_cumm_year_df['Cumm_TotalPower'] = capa_cumm_year_df['TotalPower'].cumsum()
                capa_cumm_year_df['BeginOp_year'] = capa_cumm_year_df['BeginOp_year'].dt.to_timestamp()



                # plot ----------------
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'], line = dict(color = 'navy'),name='built + predicted (month)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=capa_month_built['BeginOp_month'], y=capa_month_built['TotalPower'], line = dict(color = 'deepskyblue'), name='built (month)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=capa_month_predicted['BeginOp_month'], y=capa_month_predicted['TotalPower'], line = dict(color = 'cornflowerblue'), name='predicted (month)', mode='lines+markers'))

                fig1.add_trace(go.Scatter(x=capa_year_df['BeginOp_year'], y=capa_year_df['TotalPower'], line = dict(color = 'forestgreen'), name='built + predicted (year)', mode='lines+markers',))
                fig1.add_trace(go.Scatter(x=capa_year_built['BeginOp_year'], y=capa_year_built['TotalPower'], line = dict(color = 'lightgreen'), name='built (year)', mode='lines+markers'))
                fig1.add_trace(go.Scatter(x=capa_year_predicted['BeginOp_year'], y=capa_year_predicted['TotalPower'], line = dict(color = 'limegreen'), name='predicted (year)', mode='lines+markers'))

                fig1.add_trace(go.Scatter(x=capa_cumm_year_df['BeginOp_year'], y=capa_cumm_year_df['Cumm_TotalPower'], line = dict(color ='purple'), name='cumulative built + pred (year)', mode='lines+markers'))

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
                    if visual_settings['plot_ind_line_installedCap'][2]:
                        fig1.show()
                    elif not visual_settings['plot_ind_line_installedCap'][2]:
                        fig1.show() if i_scen == 0 else None
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
           

            # plot add aggregated - line: Installed Capacity per Year ===========================
            if visual_settings['plot_ind_line_installedCap'][0]:  #['plot_ind_line_installedCap_per_month']: 
            
                color_allscen_list = [list(trace_color_dict.keys())[i_scen] for i_scen in range(len(scen_dir_export_list))]
                color_palette = trace_color_dict[list(trace_color_dict.keys())[i_scen]]

                # fig_agg_pmonth.add_trace(go.Scatter(x=[0,], y=[0,],  name=f'',opacity=1, ))
                fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'],  name=f'',opacity=0, ))
                fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'], line = dict(color = 'black'), name=f'{scen}',opacity=0, mode='lines+markers'))

                fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_df['BeginOp_month'], y=capa_month_df['TotalPower'],                opacity = 0.75, line = dict(color = color_palette[0]),                 name='-- built + predicted (month)', mode='lines+markers'))
                fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_built['BeginOp_month'], y=capa_month_built['TotalPower'],          opacity = 0.75, line = dict(color = color_palette[0+1]),               name='-- built (month)', mode='lines+markers'))
                fig_agg_pmonth.add_trace(go.Scatter(x=capa_month_predicted['BeginOp_month'], y=capa_month_predicted['TotalPower'],  opacity = 0.75, line = dict(color = color_palette[0+2]),               name='-- predicted (month)', mode='lines+markers'))
                fig_agg_pmonth.add_trace(go.Scatter(x=capa_year_df['BeginOp_year'], y=capa_year_df['TotalPower'],                   opacity = 0.75, line = dict(color = color_palette[0+3]),               name='-- built + predicted (year)', mode='lines+markers',))
                fig_agg_pmonth.add_trace(go.Scatter(x=capa_year_built['BeginOp_year'], y=capa_year_built['TotalPower'],             opacity = 0.75, line = dict(color = color_palette[0+4]),               name='-- built (year)', mode='lines+markers'))
                fig_agg_pmonth.add_trace(go.Scatter(x=capa_year_predicted['BeginOp_year'], y=capa_year_predicted['TotalPower'],     opacity = 0.75, line = dict(color = color_palette[0+5]),               name='-- predicted (year)', mode='lines+markers'))
                fig_agg_pmonth.add_trace(go.Scatter(x=capa_cumm_year_df['BeginOp_year'], y=capa_cumm_year_df['Cumm_TotalPower'],    opacity = 0.75, line = dict(color = color_palette[-1]),                name='-- cumulative built + pred (year)', mode='lines+markers'))



                # export plot add aggregated - line: Installed Capacity per Year 
                if i_scen == len(scen_dir_export_list)-1:
                    fig_agg_pmonth.update_layout(
                    xaxis_title='Time',
                    yaxis_title='Installed Capacity (kW)',
                    legend_title='Time steps',
                    title = f'Installed Capacity per Month/Year, {len(scen_dir_export_list)}scen (weather year: {pvalloc_scen["weather_specs"]["weather_year"]})'
                    )

                    # add T0 prediction
                    T0_prediction = pvalloc_scen['T0_prediction']
                    date = '2008-01-01 00:00:00'
                    fig_agg_pmonth.add_shape(
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
                    fig_agg_pmonth.add_annotation(
                        x=  T0_prediction,
                        y=max(capa_year_df['TotalPower'].max(), capa_year_df['TotalPower'].max()),
                        text="T0 Prediction",
                        showarrow=False,
                        yshift=10
                    )

                    fig_agg_pmonth = set_default_fig_zoom_year(fig_agg_pmonth, default_zoom_year, capa_year_df, 'BeginOp_year')

                    if plot_show and visual_settings['plot_ind_line_installedCap'][1]:
                        fig_agg_pmonth.show()

                    fig_agg_pmonth.write_html(f'{data_path}/output/visualizations/plot_agg_line_installedCap__{len(scen_dir_export_list)}scen.html')
                    print_to_logfile(f'\texport: plot_agg_line_installedCap__{len(scen_dir_export_list)}scen.html', log_name)
