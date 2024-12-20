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
# PLOT INDIVIDUAL LINE for METEO + RADIATION
# ------------------------------------------------------------------------------------------------------

def plot(pvalloc_scen_list, 
         visual_settings, 
         wd_path, 
         data_path, 
         log_name):
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]
    plot_show = visual_settings['plot_show']  
    
    if visual_settings['plot_ind_line_meteo_radiation'][0]:
        checkpoint_to_logfile(f'plot_ind_line_meteo_radiation', log_name)

        i_scen, scen = 0, scen_dir_export_list[1]
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
                fig.add_trace(go.Scatter(x=meteo_raw['timestamp'], y=meteo_raw[meteo_col_temperature], name=f'Temp. raw data: {meteo_col_temperature}'))
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
