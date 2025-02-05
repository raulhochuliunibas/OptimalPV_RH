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
# PLOT INDIVIDUAL LINE of PV PRODUCTION PER NODE
# ------------------------------------------------------------------------------------------------------

def plot(pvalloc_scen_list, 
         visual_settings, 
         wd_path, 
         data_path, 
         log_name):
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]
    plot_show = visual_settings['plot_show']


    if visual_settings['plot_ind_line_productionHOY_per_node'][0]:
        checkpoint_to_logfile(f'plot_ind_line_productionHOY_per_node', log_name)
        i_scen, scen = 0, scen_dir_export_list[0]
        for i_scen, scen in enumerate(scen_dir_export_list):

            # setup + import ----------
            mc_data_path = glob.glob(f'{data_path}/output/{scen}/{visual_settings["MC_subdir_for_plot"]}')[0] # take first path if multiple apply, so code can still run properly
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
            fig = set_default_fig_zoom_hour(fig, visual_settings['default_zoom_hour'])

            if plot_show and visual_settings['plot_ind_line_productionHOY_per_node'][1]:	
                if visual_settings['plot_ind_line_productionHOY_per_node'][2]:
                    fig.show()
                elif not visual_settings['plot_ind_line_productionHOY_per_node'][2]:
                    fig.show() if i_scen == 0 else None
            if visual_settings['save_plot_by_scen_directory']:
                fig.write_html(f'{data_path}/visualizations/{scen}/{scen}__plot_ind_line_productionHOY_per_node.html')
            else:
                fig.write_html(f'{data_path}/visualizations/{scen}__plot_ind_line_productionHOY_per_node.html')

