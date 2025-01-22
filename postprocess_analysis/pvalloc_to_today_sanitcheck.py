import os as os
import sys

import os as os
import pandas as pd
import geopandas as gpd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
import copy
import json
import glob
import matplotlib.pyplot as plt
import scipy.stats as stats

from datetime import datetime

sys.path.append('..')
from auxiliary_functions import *


# ------------------------------------------------------------------------------------------------------
# ANALYSIS: PVALLOCATION from the PAST until TODAY
# ------------------------------------------------------------------------------------------------------

def prediction_accuracy(pvalloc_scen_list, postprocess_analysis_settings, wd_path, data_path, log_name):
    scen_dir_export_list = [pvalloc_scen['name_dir_export'] for pvalloc_scen in pvalloc_scen_list]

    checkpoint_to_logfile('postprocessing analysis:  pvalloc_to_today_sanitycheck.prediction_accuracy.py', log_name)
    i_scen, scen = 0, scen_dir_export_list[0]

    for i_scen, scen in enumerate(scen_dir_export_list):
        # scen_data_path = f'{data_path}/output/{scen}'
        mc_data_path = glob.glob(f'{data_path}/output/{scen}/{postprocess_analysis_settings['MC_subdir_for_analysis']}')[0] # take first path if multiple apply, so code can still run properly
        pvalloc_scen = pvalloc_scen_list[i_scen]


        # load data ----------

        # from preprep
        pv_preprep_all  =    pd.read_parquet(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/pv.parquet')
        gwr_preprep_all =    pd.read_parquet(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/gwr.parquet')
        solkat_preprep_all = pd.read_parquet(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/solkat.parquet')
        Map_egid_pv =        pd.read_parquet(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/Map_egid_pv.parquet')

        # transformation + constrain comparison df to the same building park as in "historic model". Otherwise, including large number of houses (impossible to receive installation, distorting the accuracy)
        pv_preprep_all['BeginOp'] = pd.to_datetime(pv_preprep_all['BeginningOfOperation'], format='%Y-%m-%d')
        gwr_preprep_all = gwr_preprep_all.loc[
                            (gwr_preprep_all['GBAUJ'] >= pvalloc_scen.get('gwr_selection_specs').get('GBAUJ_minmax')[0]) & 
                            (gwr_preprep_all['GBAUJ'] <= pvalloc_scen.get('gwr_selection_specs').get('GBAUJ_minmax')[1])]

        # only historic selection
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
        
        pvinst_df['TotalPower'] = pd.to_numeric(pvinst_df['TotalPower'], errors='coerce')
        pvinst_df.loc[pvinst_df['inst_TF'] == True, 'BeginOp'] = pvinst_df.loc[pvinst_df['inst_TF'] == True, 'BeginOp'].apply(lambda x: x if len(x) == 10 else x + '-01') # add day to year-month string, to have a proper timestamp
        # pvinst_df['BeginOp'] = pvinst_df['BeginOp'].apply(lambda x: x if len(x) == 10 else x + '-01') # add day to year-month string, to have a proper timestamp
        pvinst_df['BeginOp'] = pd.to_datetime(pvinst_df['BeginOp'], format='%Y-%m-%d')
        pvinst_df['bfs'] = pvinst_df['bfs'].astype(str)        


        # built df containing all houses until today -----------
        pv_preprep_all = pv_preprep_all.merge(Map_egid_pv, on='xtf_id', how='left')
        
        df_comparison = gwr_preprep_all.merge(pv_preprep_all, on='EGID', how='left')
        # BOOKMARK! => this df has duplicates because some installations are built on the same house => assigned the same EGID, as pv_egid mapping is
        #              done through spatial join function. Workaround to remove some duplicates for now.
        df_comparison = df_comparison.loc[~df_comparison.duplicated(subset='EGID', keep = 'last')]

        # df_comparison = df_comparison.merge(solkat_preprep_all, on='EGID', how='left')
        
        # rename columns for better understanding and merge
        cols_to_add_suffix = ['']
        pvinst_df.rename(columns={'TotalPower': 'TotalPower_MOD', 
                                  'BeginOp': 'BeginOp_MOD'}, inplace=True)
        df_comparison.rename(columns={'TotalPower': 'TotalPower_RWD', 
                                      'BeginOp': 'BeginOp_RWD'}, inplace=True)
        df_comparison = df_comparison.merge(pvinst_df, on='EGID', how='left')

        # add columns for comparison
        df_comparison['TotalPower_DETLA'] = df_comparison['TotalPower_MOD'] - df_comparison['TotalPower_RWD']
        df_comparison['BeginOp_DELTAdays'] = (df_comparison['BeginOp_MOD'] - df_comparison['BeginOp_RWD']).dt.days

        df_comparison['d_TruePositive_Installation'] = np.where((df_comparison['TotalPower_MOD'] > 0) & 
                                                                (df_comparison['TotalPower_RWD'] > 0), True, False)
        df_comparison['d_FalsePositive_Installation'] = np.where((df_comparison['TotalPower_MOD'] > 0) & 
                                                                 (df_comparison['TotalPower_RWD'].isna()), True, False)
        df_comparison['d_TrueNegative_Installation'] = np.where((df_comparison['TotalPower_MOD'].isna()) &
                                                                (df_comparison['TotalPower_RWD'].isna()), True, False)
        df_comparison['d_FalseNegative_Installation'] = np.where((df_comparison['TotalPower_MOD'].isna()) &
                                                                 (df_comparison['TotalPower_RWD'] > 0), True, False)
        
        
        # plot ----------
        true_positive_count, false_positive_count, true_negative_count, false_negative_count = sum(df_comparison['d_TruePositive_Installation']), sum(df_comparison['d_FalsePositive_Installation']), sum(df_comparison['d_TrueNegative_Installation']), sum(df_comparison['d_FalseNegative_Installation'])
        true_positive_perct, false_positive_perct, true_negative_perct, false_negative_perct = true_positive_count / len(df_comparison)*100 , false_positive_count / len(df_comparison)*100, true_negative_count / len(df_comparison)*100, false_negative_count / len(df_comparison)*100
        heatmap_values = [
            [true_positive_count, false_positive_count],
            [false_negative_count, true_negative_count]
        ]
        heatmap_percent = [
            [true_positive_perct, false_positive_perct],
            [false_negative_perct, true_negative_perct]
        ]
        heatmap_labels = [
            [f"{true_positive_count}\n({true_positive_perct:.2f}%)", f"{false_positive_count}\n({false_positive_perct:.2f}%)"],
            [f"{false_negative_count}\n({false_negative_perct:.2f}%)", f"{true_negative_count}\n({true_negative_perct:.2f}%)"]
        ]
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_values,
            text=heatmap_labels,
            texttemplate="%{text}",  # Display the text in each cell
            hoverinfo='text',
            colorscale='RdBu',
            showscale=True, 
            opacity=0.8
        ))
        fig.update_traces(textfont=dict(size=14, color='white'), colorbar=dict(title='Counts'))

        fig.update_layout(
            title=f'Installation Prediction Accuracy (scen: {scen})',
            xaxis=dict(tickvals=[0, 1], ticktext=['Inst in Model', 'NO Modelled Inst']),
            yaxis=dict(tickvals=[0, 1], ticktext=['Inst in RWD', 'NO INST in RWD']),
            xaxis_title='Modelled (MOD)',
            yaxis_title='Real World Data (RWD)',
            height=750,  # Adjusting the height of the plot
            width=900,   # Adjusting the width of the plot to make it square
            margin=dict(l=50, r=50, t=50, b=50),  # Adjust the margins if needed
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            plot_bgcolor="white",  # Optional: set background color
        )

        # Show the plot
        fig.show()
        # steps
        # check if gwr no houses after T0 prediction
        # get houses until today for comparison
        # accuracy df => which features predict a good fit? marginal plots, which houses are correct 
        # make 2x2 accuracy matrix

