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
from scipy.stats import norm, skewnorm


sys.path.append('..')
from auxiliary_functions import *
from .plot_auxiliary_functions import *


# ------------------------------------------------------------------------------------------------------
# PLOT INDIVIDUAL HISTOGRAM of RADIATION RANGE per Subfile and EGID
# ------------------------------------------------------------------------------------------------------

def plot(pvalloc_scen_list, 
         visual_settings, 
         wd_path, 
         data_path, 
         log_name):
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]
    plot_show = visual_settings['plot_show']    
    

    if visual_settings['plot_ind_hist_radiation_rng_sanitycheck'][0]:

        checkpoint_to_logfile(f'plot_ind_hist_radiation_rng_sanitycheck', log_name)

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

            
            # functions per subdf
            def distr_comp(df, colname):
                    srs = df[colname]
                    return(srs.mean(), srs.std(), srs.skew(), srs.kurt(), srs.min(), srs.max())

            def generate_distribution(mean, std, skew, kurt, min_val, max_val, num_points=1000):
                x = np.linspace(min_val, max_val, num_points)
                # Generate a normal distribution
                # y = norm.pdf(x, mean, std)
                # Adjust for skewness
                y = skewnorm.pdf(x, skew, mean, std)
                return x, y
            
            def generate_kde(df_srs, num_points=8000):
                kde = gaussian_kde(df_srs)
                x = np.linspace(min(df_srs), max(df_srs), num_points)
                y = kde(x)
                return kde
                # return x, y

            agg_by_method = "subdf"
            export_neg_rad_egid_counter = 0

            unit_id_list, debug_subdf_rad_dir_list, debug_subdf_rad_diff_list, debug_subdf_radiation_list, debug_subdf_radiation_rel_locmax_list = [], [], [], [], []
            kde_rad_dir_list, kde_rad_diff_list, kde_radiation_list, kde_radiation_rel_locmax_list = [], [], [], []
            kde_x_rad_dir_list, kde_x_rad_diff_list, kde_x_radiation_list, kde_x_radiation_rel_locmax_list = [], [], [], []
            kde_y_rad_dir_list, kde_y_rad_diff_list, kde_y_radiation_list, kde_y_radiation_rel_locmax_list = [], [], [], []

            i_path, path = 0, topo_subdf_paths[0]
            for i_path, path in enumerate(topo_subdf_paths):
                print(f'subdf {i_path+1}/{len(topo_subdf_paths)}')
                subdf= pd.read_parquet(path)
                
                if agg_by_method == "subdf":    # if agg debug subdf by subdf file "number"
                    unit_id_list.append(i_path)

                    debug_subdf_rad_dir_list.append(distr_comp(subdf, 'rad_direct'))
                    debug_subdf_rad_diff_list.append(distr_comp(subdf, 'rad_diffuse'))
                    debug_subdf_radiation_list.append(distr_comp(subdf, 'radiation'))
                    debug_subdf_radiation_rel_locmax_list.append(distr_comp(subdf, 'radiation_rel_locmax'))

                    # kde_rad_dir_list = generate_kde(subdf['rad_direct'])
                    # kde_rad_diff_list = generate_kde(subdf['rad_diffuse'])
                    # kde_radiation_list = generate_kde(subdf['radiation'])
                    # kde_radiation_rel_locmax_list = generate_kde(subdf['radiation_rel_locmax'])

                    while export_neg_rad_egid_counter < 5:
                         for egid in subdf['EGID'].unique():
                            egid_df = subdf[subdf['EGID'] == egid]
                            if egid_df['radiation'].min() < 0:
                                export_neg_rad_egid_counter += 1
                                egid_df.to_excel(f'{data_path}/output/subdf_egid{egid}_neg_rad.xlsx')
                                print(f'exported neg rad egid {export_neg_rad_egid_counter}')
                            if export_neg_rad_egid_counter == 5:
                                break
                         
                elif agg_by_method == "egid":
                        for egid in subdf['EGID'].unique():
                            egid_df = subdf[subdf['EGID'] == egid]

                            unit_id_list.append(egid)
                            debug_subdf_rad_dir_list.append(distr_comp(egid_df, 'rad_direct'))
                            debug_subdf_rad_diff_list.append(distr_comp(egid_df, 'rad_diffuse'))
                            debug_subdf_radiation_list.append(distr_comp(egid_df, 'radiation'))
                            debug_subdf_radiation_rel_locmax_list.append(distr_comp(egid_df, 'radiation_rel_locmax'))


            debug_rad_df = pd.DataFrame({'i_subdf_file': unit_id_list,
                                            'rad_direct_mean': [tupl_val[0] for tupl_val in debug_subdf_rad_dir_list],
                                            'rad_direct_std':  [tupl_val[1] for tupl_val in debug_subdf_rad_dir_list],
                                            'rad_direct_skew': [tupl_val[2] for tupl_val in debug_subdf_rad_dir_list],
                                            'rad_direct_kurt': [tupl_val[3] for tupl_val in debug_subdf_rad_dir_list],
                                            'rad_direct_min':  [tupl_val[4] for tupl_val in debug_subdf_rad_dir_list],
                                            'rad_direct_max':  [tupl_val[5] for tupl_val in debug_subdf_rad_dir_list],

                                            'rad_diff_mean':   [tupl_val[0] for tupl_val in debug_subdf_rad_diff_list],
                                            'rad_diff_std':    [tupl_val[1] for tupl_val in debug_subdf_rad_diff_list],
                                            'rad_diff_skew':   [tupl_val[2] for tupl_val in debug_subdf_rad_diff_list],
                                            'rad_diff_kurt':   [tupl_val[3] for tupl_val in debug_subdf_rad_diff_list],
                                            'rad_diff_min':    [tupl_val[4] for tupl_val in debug_subdf_rad_diff_list],
                                            'rad_diff_max':    [tupl_val[5] for tupl_val in debug_subdf_rad_diff_list],

                                            'radiation_mean':  [tupl_val[0] for tupl_val in debug_subdf_radiation_list],
                                            'radiation_std':   [tupl_val[1] for tupl_val in debug_subdf_radiation_list],
                                            'radiation_skew':  [tupl_val[2] for tupl_val in debug_subdf_radiation_list],  
                                            'radiation_kurt':  [tupl_val[3] for tupl_val in debug_subdf_radiation_list],
                                            'radiation_min':   [tupl_val[4] for tupl_val in debug_subdf_radiation_list],
                                            'radiation_max':   [tupl_val[5] for tupl_val in debug_subdf_radiation_list],

                                            'radiation_rel_locmax_mean': [tupl_val[0] for tupl_val in debug_subdf_radiation_rel_locmax_list],
                                            'radiation_rel_locmax_std':  [tupl_val[1] for tupl_val in debug_subdf_radiation_rel_locmax_list],
                                            'radiation_rel_locmax_skew': [tupl_val[2] for tupl_val in debug_subdf_radiation_rel_locmax_list],
                                            'radiation_rel_locmax_kurt': [tupl_val[3] for tupl_val in debug_subdf_radiation_rel_locmax_list],
                                            'radiation_rel_locmax_min':  [tupl_val[4] for tupl_val in debug_subdf_radiation_rel_locmax_list],
                                            'radiation_rel_locmax_max':  [tupl_val[5] for tupl_val in debug_subdf_radiation_rel_locmax_list],
                                            
                                            })


                # plot                
            fig = go.Figure()

            for colname in ['rad_direct', 'rad_diff', 'radiation', 'radiation_rel_locmax']:
                fig.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'', opacity=0))
                fig.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'{colname}', opacity=0))

                for index, row in debug_rad_df.iterrows():
                    x,y, = generate_distribution(row[f'{colname}_mean'], row[f'{colname}_std'], row[f'{colname}_skew'], row[f'{colname}_kurt'], row[f'{colname}_min'], row[f'{colname}_max'])
                    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'{colname}_dist_{index}'))

                # for index, row in debug_rad_df.iterrows():
                #     data = np.linspace(row[f'{colname}_min'], row[f'{colname}_max'], 2000)
                #     x_kde, y_kde = generate_kde(data)
                #     x_kde, y_kde = generate_kde(data)
                #     fig.add_trace(go.Scatter(x=x_kde, y=y_kde, mode='lines', name=f'{colname}_kde_{index}'))
                

            fig.update_layout(
                title='Distribution Functions and KDE',
                xaxis_title='Radiation [W/m2]',
                yaxis_title='Frequency',
                barmode='overlay'
            )
            fig.show()
            print('asdf')


