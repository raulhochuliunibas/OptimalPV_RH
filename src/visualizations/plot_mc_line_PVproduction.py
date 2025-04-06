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
# PLOT MC aggregated LINE for PV PRODUCTION 
# ------------------------------------------------------------------------------------------------------

def plot(pvalloc_scen_list, 
         visual_settings, 
         wd_path, 
         data_path, 
         log_name):
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]
    plot_show = visual_settings['plot_show']

    if visual_settings['plot_mc_line_PVproduction'][0]:
        checkpoint_to_logfile(f'plot_mc_line_PVproduction', log_name)

    i_scen, scen = 0, scen_dir_export_list[0]
    for i_scen, scen in enumerate(scen_dir_export_list):
        
        # setup ----------
        pvalloc_scen = pvalloc_scen_list[i_scen]
        mc_runs_paths = glob.glob(f'{data_path}/output/{scen}/*MC_*')

        mc_path = mc_runs_paths[0]
        for mc_path in mc_runs_paths:

            # import data ----------
            topo = json.load(open(f'{mc_path}/topo_egid.json', 'r'))
            topo_subdf_paths = glob.glob(f'{data_path}/output/{scen}/topo_time_subdf/topo_subdf_*.parquet')
            gridnode_df_paths = glob.glob(f'{mc_path}/pred_gridprem_node_by_M/gridnode_df_*.parquet')


            # get installations of topo over time
            egid_list, inst_TF_list, info_source_list, BeginOp_list, xtf_id_list, TotalPower_list, = [], [], [], [], [], []
            k = list(topo.keys())[0]
            for k, v in topo.items():
                egid_list.append(k)
                inst_TF_list.append(v['pv_inst']['inst_TF'])
                info_source_list.append(v['pv_inst']['info_source'])
                BeginOp_list.append(v['pv_inst']['BeginOp'])
                xtf_id_list.append(v['pv_inst']['xtf_id'])
                TotalPower_list.append(v['pv_inst']['TotalPower'])

            pvinst_df = pd.DataFrame({'EGID': egid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list, 
                                      'BeginOp': BeginOp_list, 'xtf_id': xtf_id_list, 'TotalPower': TotalPower_list,})
            pvinst_df = pvinst_df.loc[pvinst_df['inst_TF'] == True]
            # pvinst_df = pvinst_df.loc[pvinst_df['info_source'] == 'alloc_algorithm']
            
            pvinst_df['TotalPower'] = pd.to_numeric(pvinst_df['TotalPower'], errors='coerce')
            pvinst_df['BeginOp'] = pvinst_df['BeginOp'].apply(lambda x: x if len(x) == 10 else x + '-01') # add day to year-month string, to have a proper timestamp
            pvinst_df['BeginOp'] = pd.to_datetime(pvinst_df['BeginOp'], format='%Y-%m-%d')
            
            # attach annual production to each installation
            pvinst_df['pvprod_kW'] = 0
            aggdf_combo_list = []
            path = topo_subdf_paths[2]
                
            for ipath, path in enumerate(topo_subdf_paths):
                subdf = pd.read_parquet(path)
                subdf = subdf.loc[subdf['EGID'].isin(pvinst_df['EGID'])]

                agg_subdf = subdf.groupby(['EGID', 'df_uid', 'FLAECHE', 'STROMERTRAG']).agg({'pvprod_kW': 'sum',}).reset_index() 
                aggsub_npry = np.array(agg_subdf)


                # attach production to each installation                
                pvinst_egid_in_subdf = [egid for egid in pvinst_df['EGID'].unique() if egid in agg_subdf['EGID'].unique()]
                egid = pvinst_egid_in_subdf[0]
                for egid in pvinst_egid_in_subdf:
                    df_uid_combo = pvinst_df.loc[pvinst_df['EGID'] == egid]['xtf_id'].values[0].split('_')

                    if len(df_uid_combo) == 1:
                        pvinst_df.loc[pvinst_df['EGID'] == egid, 'pvprod_kW'] = agg_subdf.loc[agg_subdf['EGID'] == egid]['pvprod_kW'].values[0]
                    elif len(df_uid_combo) > 1:
                        pvinst_df.loc[pvinst_df['EGID'] == egid, 'pvprod_kW'] = agg_subdf.loc[agg_subdf['df_uid'].isin(df_uid_combo), 'pvprod_kW'].sum()
            

            # aggregate pvinst_df to monthly values
            prod_month_df = copy.deepcopy(pvinst_df)
            prod_month_df['BeginOp_month'] = prod_month_df['BeginOp'].dt.to_period('M')
            prod_month_df['BeginOp_month_str'] = prod_month_df['BeginOp_month'].astype(str)
            prod_month_df['TotalPower_month'] = prod_month_df.groupby(['BeginOp_month'])['TotalPower'].transform('sum')
            prod_month_df['pvprod_kW_month'] = prod_month_df.groupby(['BeginOp_month'])['pvprod_kW'].transform('sum')
            prod_month_df['BeginOp_month'] = prod_month_df['BeginOp_month'].dt.to_timestamp()
            prod_month_df.sort_values(by=['BeginOp_month'], inplace=True)


            month = prod_month_df['BeginOp_month'].unique()[0]
            for month in prod_month_df['BeginOp_month'].unique():
                month_str = prod_month_df.loc[prod_month_df['BeginOp_month'] == month, 'BeginOp_month_str'].values[0]
                grid_subdf = pd.read_parquet(f'{mc_path}/pred_gridprem_node_by_M/gridnode_df_{month_str}.parquet')
                
                prod_month_df.loc[prod_month_df['BeginOp_month'] == month, 'feedin_kW'] = grid_subdf['feedin_kW'].sum()
                prod_month_df.loc[prod_month_df['BeginOp_month'] == month, 'feedin_kW_taken'] = grid_subdf['feedin_kW_taken'].sum()
                prod_month_df.loc[prod_month_df['BeginOp_month'] == month, 'feedin_kW_loss'] = grid_subdf['feedin_kW_loss'].sum()


            # plot ----------------
            # fig = go.Figure()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Scatter(x=prod_month_df['BeginOp_month'], y=prod_month_df['pvprod_kW_month'], name='EGID Prod kWh (total pvprod_kW)', ))
            fig.add_trace(go.Scatter(x=prod_month_df['BeginOp_month'], y=prod_month_df['feedin_kW'], name='Grid feedin kWh (feedin_kwh)', ))
            fig.add_trace(go.Scatter(x=prod_month_df['BeginOp_month'], y=prod_month_df['feedin_kW_taken'], name='Grid feedin take kWh (feedin_taken kWh)', ))
            fig.add_trace(go.Scatter(x=prod_month_df['BeginOp_month'], y=prod_month_df['feedin_kW_loss'], name='Grid feedin loss kWh (feedin_loss kWh)', ))

            fig.add_trace(go.Scatter(x=prod_month_df['BeginOp_month'], y=prod_month_df['TotalPower_month'], name='Total installed capacity', line=dict(color='blue', width=2)), secondary_y=True)

            fig.update_layout(
                title=f'PV production per month',
                xaxis_title='Month',
                yaxis_title='Production [kW]',
                yaxis2_title='Installed capacity [kW]',
                legend_title='Legend',
            )
            fig.update_yaxes(title_text="Installed capacity [kW]", secondary_y=True)

            if plot_show and visual_settings['plot_ind_line_PVproduction'][1]:
                if visual_settings['plot_ind_line_PVproduction'][2]:
                    fig.show()
                elif not visual_settings['plot_ind_line_PVproduction'][2]:
                    fig.show() if i_scen == 0 else None
            if visual_settings['save_plot_by_scen_directory']:
                fig.write_html(f'{data_path}/visualizations/{scen}/{scen}__plot_ind_line_PVproduction.html')
            else:
                fig.write_html(f'{data_path}/visualizations/{scen}__plot_ind_line_PVproduction.html')
            print_to_logfile(f'\texport: plot_ind_line_PVproduction.html (for: {scen})', log_name)


    
        



