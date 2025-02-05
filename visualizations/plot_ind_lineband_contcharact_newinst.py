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
# PLOT INDIVIDUAL LINE w confd BAND of CONTINUOUS CHARACTERISTICS for NEW INSTALLATIONS
# ------------------------------------------------------------------------------------------------------

def plot(pvalloc_scen_list, 
         visual_settings, 
         wd_path, 
         data_path, 
         log_name):
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]
    plot_show = visual_settings['plot_show']

    if visual_settings['plot_ind_lineband_contcharact_newinst'][0]:
        checkpoint_to_logfile(f'plot_ind_lineband_contcharact_newinst', log_name)

        # available color palettes
        trace_color_dict = {
            'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
            'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
            'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Plotly3': pc.sequential.Plotly3,
            'Viridis': pc.sequential.Viridis, 'Turbo': pc.sequential.Turbo, 'Blackbody': pc.sequential.Blackbody, 
            'Bluered': pc.sequential.Bluered, 'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
        }      

        i_scen, scen = 0, scen_dir_export_list[0]
        for i_scen, scen in enumerate(scen_dir_export_list):
            mc_data_path = glob.glob(f'{data_path}/output/{scen}/{visual_settings["MC_subdir_for_plot"]}')[0] # take first path if multiple apply, so code can still run properly
            pvalloc_scen = pvalloc_scen_list[i_scen]


            # setup + import ----------
            colnams_charac_AND_numerator = visual_settings['plot_ind_line_contcharact_newinst_specs']['colnames_cont_charact_installations_AND_numerator']
            trace_color_palette = visual_settings['plot_ind_line_contcharact_newinst_specs']['trace_color_palette']
            # col_colors = [val / len(colnams_charac_AND_numerator) for val in range(1,len(colnams_charac_AND_numerator)+1)]
            col_colors = list(range(1,len(colnams_charac_AND_numerator)+1))
            palette = trace_color_dict[trace_color_palette]
            
            predinst_all= pd.read_parquet( f'{mc_data_path}/pred_inst_df.parquet')
            predinst_absdf = copy.deepcopy(predinst_all)

            agg_dict ={}
            for col_tuple in colnams_charac_AND_numerator:
                agg_dict[f'{col_tuple[0]}'] = ['mean', 'std']
                predinst_absdf[f'{col_tuple[0]}'] = predinst_absdf[f'{col_tuple[0]}'] / col_tuple[1]

            agg_predinst_absdf = predinst_absdf.groupby('iter_round').agg(agg_dict)
            agg_predinst_absdf['iter_round'] = agg_predinst_absdf.index

            agg_predinst_absdf.replace(np.nan, 0, inplace=True) # replace NaNs with 0, needed if no deviation in std


            # plot ----------------
            fig = go.Figure()
            i_col = 3
            col, col_numerator = colnams_charac_AND_numerator[i_col][0], colnams_charac_AND_numerator[i_col][1]
            for i_col, col_tuple in enumerate(colnams_charac_AND_numerator):
                col = col_tuple[0]
                col_numerator = col_tuple[1]
                              
                xaxis   =           agg_predinst_absdf['iter_round']
                y_mean  =           agg_predinst_absdf[col]['mean'] 
                y_lower, y_upper =  agg_predinst_absdf[col]['mean'] - agg_predinst_absdf[col]['std'], agg_predinst_absdf[col]['mean'] + agg_predinst_absdf[col]['std']
                trace_color = palette[col_colors[i_col % len(col_colors)]]

                # mean trace
                fig.add_trace(go.Scatter(x=xaxis, y=y_mean,
                                        name=f'{col} mean (1/{col_numerator})',
                                        legendgroup=f'{col}',
                                        line=dict(color=trace_color),
                                        mode='lines+markers', showlegend=True))
                
                # upper / lower bound band
                fig.add_trace(go.Scatter(
                    x=xaxis.tolist() + xaxis.tolist()[::-1],  # Concatenate xaxis with its reverse
                    y=y_upper.tolist() + y_lower.tolist()[::-1],  # Concatenate y_upper with reversed y_lower
                    fill='toself',
                    fillcolor=trace_color,  # Dynamic color with 50% transparency
                    opacity=0.2,
                    line=dict(color='rgba(255,255,255,0)'),  # No boundary line
                    hoverinfo="skip",  # Don't show info on hover
                    showlegend=False,  # Do not show this trace in the legend
                    legendgroup=f'{col}',  # Group with the mean line
                    visible=True  # Make this visible/toggleable with the mean line
                ))

            fig.update_layout(
                xaxis_title='Iteration Round',
                yaxis_title='Mean (+/- 1 std)',
                legend_title='Scenarios',
                title = f'Agg. Cont. Charact. of Newly Installed Buildings per Iteration Round', 
                    uirevision='constant'  # Maintain the state of the plot when interacting

            )
        

            if plot_show and visual_settings['plot_ind_lineband_contcharact_newinst'][1]:	
                if visual_settings['plot_ind_lineband_contcharact_newinst'][2]:
                    fig.show()
                elif not visual_settings['plot_ind_lineband_contcharact_newinst'][2]:
                    fig.show() if i_scen == 0 else None
            if visual_settings['save_plot_by_scen_directory']:
                fig.write_html(f'{data_path}/visualizations/{scen}/{scen}__plot_ind_lineband_contcharact_newinst.html')
            else:
                fig.write_html(f'{data_path}/visualizations/{scen}__plot_ind_lineband_contcharact_newinst.html')
