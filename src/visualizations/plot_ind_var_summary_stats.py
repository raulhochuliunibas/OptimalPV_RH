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
# PLOT INDIVIDUAL VARIABLE SUMMARY STATISTICS
# ------------------------------------------------------------------------------------------------------

def plot(pvalloc_scen_list, 
         visual_settings, 
         wd_path, 
         data_path, 
         log_name):
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]
    plot_show = visual_settings['plot_show']


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
                fig.write_html(f'{data_path}/visualizations/{scen}/{scen}__plot_ind_bar_totaldemand_by_type.html')
            else:
                fig.write_html(f'{data_path}/visualizations/{scen}__plot_ind_bar_totaldemand_by_type.html')
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
            fig = set_default_fig_zoom_hour(fig, visual_settings['default_zoom_hour'])

            if plot_show and visual_settings['plot_ind_var_summary_stats'][1]:
                if visual_settings['plot_ind_var_summary_stats'][2]:
                    fig.show()
                elif not visual_settings['plot_ind_var_summary_stats'][2]:
                    fig.show() if i_scen == 0 else None
            if visual_settings['save_plot_by_scen_directory']:
                fig.write_html(f'{data_path}/visualizations/{scen}/{scen}__plot_ind_line_demandTS.html')
            else:
                fig.write_html(f'{data_path}/visualizations/{scen}__plot_ind_line_demandTS.html')
            print_to_logfile(f'\texport: plot_ind_line_demandTS.html (for: {scen})', log_name)
            
