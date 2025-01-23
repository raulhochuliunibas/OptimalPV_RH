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
# PLOT INDIVIDUAL LINE GRID PREMIUM PER NODE
# ------------------------------------------------------------------------------------------------------

def plot(pvalloc_scen_list, 
         visual_settings, 
         wd_path, 
         data_path, 
         log_name):
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]
    plot_show = visual_settings['plot_show']


    if visual_settings['plot_ind_line_gridPremiumHOY_per_node'][0]:
        checkpoint_to_logfile(f'plot_ind_line_gridPremiumHOY_per_node', log_name)
        i_scen, scen = 0, scen_dir_export_list[0]

        for i_scen, scen in enumerate(scen_dir_export_list):

            # setup + import ----------
            mc_data_path = glob.glob(f'{data_path}/output/{scen}/{visual_settings["MC_subdir_for_plot"]}')[0] # take first path if multiple apply, so code can still run properly
            pvalloc_scen = pvalloc_scen_list[i_scen]

            node_selection = visual_settings['node_selection_for_plots']

            gridprem_ts = pd.read_parquet(f'{mc_data_path}/gridprem_ts.parquet')
            gridprem_ts['t_int'] = gridprem_ts['t'].str.extract(r't_(\d+)').astype(int)

            # plot ----------------
            fig = go.Figure()
            for node in  node_selection:
                gridprem_ts_node = gridprem_ts[gridprem_ts['grid_node'] == node]
                gridprem_ts_node.sort_values(by=['t_int'], inplace=True)

                fig.add_trace(go.Scatter(x=gridprem_ts_node['t_int'], y=gridprem_ts_node['prem_Rp_kWh'],
                                         mode='lines', name=f'grid_premium node: {node}'))
                
            # add average and std to gridprem_ts
            agg_gridprem = gridprem_ts.groupby('t_int').agg({
                't': 'first',
                'prem_Rp_kWh': ['mean', 'std']
            }).reset_index()                                           
            agg_gridprem.columns = ['t_int', 't', 'prem_Rp_kWh_mean', 'prem_Rp_kWh_std']

            fig.add_trace(go.Scatter
                (x=agg_gridprem['t_int'], 
                 y=agg_gridprem['prem_Rp_kWh_mean'],
                 mode='lines',
                 name='grid premium Rp/kWh (mean)',
                 line=dict(color='black', width=2)
                ))
            fig.add_trace(go.Scatter
                (x=agg_gridprem['t_int'], 
                 y=agg_gridprem['prem_Rp_kWh_std'],
                 mode='lines',
                 name='grid premium Rp/kWh (std)',
                 line=dict(color='darkgrey', width=1, 
                dash='dash')
                ))
            

            # layout ----------------
            fig.update_layout(
                title=f'Grid premium Rp/kWh per node (scen: {scen})',
                xaxis_title='Hour of year',
                yaxis_title='Grid premium Rp/kWh',
                showlegend=True,
            )
            fig = add_scen_name_to_plot(fig, scen, pvalloc_scen_list[i_scen])
            fig = set_default_fig_zoom_hour(fig, visual_settings['default_zoom_hour'])


            if plot_show and visual_settings['plot_ind_line_gridPremiumHOY_per_node'][1]:	
                if visual_settings['plot_ind_line_gridPremiumHOY_per_node'][2]:
                    fig.show()
                elif not visual_settings['plot_ind_line_gridPremiumHOY_per_node'][2]:
                    fig.show() if i_scen == 0 else None
            if visual_settings['save_plot_by_scen_directory']:
                fig.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_ind_line_gridPremiumHOY_per_node.html')
            else:
                fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_line_gridPremiumHOY_per_node.html')

