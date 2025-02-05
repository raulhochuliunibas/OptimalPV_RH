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
# PLOT INDIVIDUAL HISTOGRAMS of NPV for FREE PARTITIONS
# ------------------------------------------------------------------------------------------------------

def plot(pvalloc_scen_list, 
         visual_settings, 
         wd_path, 
         data_path, 
         log_name):
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]
    plot_show = visual_settings['plot_show']


    if visual_settings['plot_ind_hist_NPV_freepartitions'][0]:
        checkpoint_to_logfile(f'plot_ind_hist_NPV_freepartitions', log_name)
        fig_agg = go.Figure()

        i_scen, scen = 0, scen_dir_export_list[0]
        for i_scen, scen in enumerate(scen_dir_export_list):
            # setup + import ----------
            mc_data_path = glob.glob(f'{data_path}/output/{scen}/{visual_settings["MC_subdir_for_plot"]}')[0] # take first path if multiple apply, so code can still run properly
            pvalloc_scen = pvalloc_scen_list[i_scen]

            npv_df_paths = glob.glob(f'{mc_data_path}/pred_npv_inst_by_M/npv_df_*.parquet')
            periods_list = [pd.to_datetime(path.split('npv_df_')[-1].split('.parquet')[0]) for path in npv_df_paths]
            before_period, after_period = min(periods_list), max(periods_list)

            npv_df_before = pd.read_parquet(f'{mc_data_path}/pred_npv_inst_by_M/npv_df_{before_period.to_period("M")}.parquet')
            npv_df_after  = pd.read_parquet(f'{mc_data_path}/pred_npv_inst_by_M/npv_df_{after_period.to_period("M")}.parquet')

            # plot ----------------
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=npv_df_before['NPV_uid'], name='Before Allocation Algorithm', opacity=0.5))
            fig.add_trace(go.Histogram(x=npv_df_after['NPV_uid'], name='After Allocation Algorithm', opacity=0.5))

            fig.update_layout(
                xaxis_title=f'Net Present Value (NPV, interest rate: {pvalloc_scen["tech_economic_specs"]["interest_rate"]}, maturity: {pvalloc_scen["tech_economic_specs"]["invst_maturity"]} yr)',
                yaxis_title='Frequency',
                title = f'NPV Distribution of possible PV installations, first / last year (weather year: {pvalloc_scen["weather_specs"]["weather_year"]})',
                barmode = 'overlay')
            fig.update_traces(bingroup=1, opacity=0.5)

            fig = add_scen_name_to_plot(fig, scen, pvalloc_scen_list[i_scen])
            
            if plot_show and visual_settings['plot_ind_hist_NPV_freepartitions'][1]:
                if visual_settings['plot_ind_hist_NPV_freepartitions'][2]:
                    fig.show()
                elif not visual_settings['plot_ind_hist_NPV_freepartitions'][2]:
                    fig.show() if i_scen == 0 else None
            if visual_settings['save_plot_by_scen_directory']:
                fig.write_html(f'{data_path}/visualizations/{scen}/{scen}__plot_ind_hist_NPV_freepartitions.html')
            else:
                fig.write_html(f'{data_path}/visualizations/{scen}__plot_ind_hist_NPV_freepartitions.html')
           

            # aggregate plot ----------------
            fig_agg.add_trace(go.Scatter(x=[0,], y=[0,], name=f'', opacity=0,))
            fig_agg.add_trace(go.Scatter(x=[0,], y=[0,], name=f'{scen}', opacity=0,)) 

            fig_agg.add_trace(go.Histogram(x=npv_df_before['NPV_uid'], name=f'Before Allocation', opacity=0.7, xbins=dict(size=500)))
            fig_agg.add_trace(go.Histogram(x=npv_df_after['NPV_uid'],  name=f'After Allocation',  opacity=0.7, xbins=dict(size=500)))

        fig_agg.update_layout(
            xaxis_title=f'Net Present Value (NPV, interest rate: {pvalloc_scen["tech_economic_specs"]["interest_rate"]}, maturity: {pvalloc_scen["tech_economic_specs"]["invst_maturity"]} yr)',
            yaxis_title='Frequency',
            title = f'NPV Distribution of possible PV installations, first / last year ({len(scen_dir_export_list)} scen, weather year: {pvalloc_scen["weather_specs"]["weather_year"]})',
            barmode = 'overlay')
        # fig_agg.update_traces(bingroup=1, opacity=0.75)

        if plot_show and visual_settings['plot_ind_hist_NPV_freepartitions'][1]:
            fig_agg.show()
            fig_agg.write_html(f'{data_path}/visualizations/plot_agg_hist_NPV_freepartitions__{len(scen_dir_export_list)}scen.html')
            

