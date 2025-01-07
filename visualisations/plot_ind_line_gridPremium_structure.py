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
# PLOT INDIVIDUAL LINE w GRID PREMIUM STRUCTURE
# ------------------------------------------------------------------------------------------------------

def plot(pvalloc_scen_list, 
         visual_settings, 
         wd_path, 
         data_path, 
         log_name):
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]
    plot_show = visual_settings['plot_show']

    if visual_settings['plot_ind_line_gridPremium_structure'][0]:
        checkpoint_to_logfile(f'plot_ind_line_gridPremium_structure', log_name)

        i_scen, scen = 0, scen_dir_export_list[0]
        for i_scen, scen in enumerate(scen_dir_export_list):
            mc_data_path = glob.glob(f'{data_path}/output/{scen}/{visual_settings['MC_subdir_for_plot']}')[0] # take first path if multiple apply, so code can still run properly
            pvalloc_scen = pvalloc_scen_list[i_scen]
            
            # setup + import ----------
            gridprem_adjustment_specs = pvalloc_scen['gridprem_adjustment_specs']

            tiers_rel_treshold_list, gridprem_Rp_kWh_list = [], []
            tiers = gridprem_adjustment_specs['tiers']
            for k,v in tiers.items():
                tiers_rel_treshold_list.append(v[0])
                gridprem_Rp_kWh_list.append(v[1])

            gridprem_tiers_df = pd.DataFrame({'tiers_rel_treshold': tiers_rel_treshold_list, 'gridprem_Rp_kWh': gridprem_Rp_kWh_list})

            # plot ----------
            fig = go.Figure()

            fig.add_trace(go.Scatter
                          (x=gridprem_tiers_df['tiers_rel_treshold'], 
                           y=gridprem_tiers_df['gridprem_Rp_kWh'],
                           mode='lines+markers',
                           name='gridprem, marginal feedin premium (Rp)',
                           showlegend=True,
                           ))                
            fig.update_layout(
                title='Grid premium structure, feed-in premium for reaching relative grid node capacity (kVA)',
                xaxis_title='Relative grid node capacity threshold',
                yaxis_title='Feed-in premium (Rp/kWh)',
            )


            if plot_show and visual_settings['plot_ind_line_gridPremium_structure'][1]:	
                if visual_settings['plot_ind_line_gridPremium_structure'][2]:
                    fig.show()
                elif not visual_settings['plot_ind_line_gridPremium_structure'][2]:
                    fig.show() if i_scen == 0 else None
            if visual_settings['save_plot_by_scen_directory']:
                fig.write_html(f'{data_path}/output/visualizations/{scen}/{scen}__plot_ind_line_gridPremium_structure.html')
            else:
                fig.write_html(f'{data_path}/output/visualizations/{scen}__plot_ind_line_gridPremium_structure.html')
