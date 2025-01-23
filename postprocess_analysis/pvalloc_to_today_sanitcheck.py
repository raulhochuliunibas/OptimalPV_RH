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


        # load data ------------------------------

        # from preprep
        pv_preprep_all  =    pd.read_parquet(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/pv.parquet')
        gwr_preprep_all =    pd.read_parquet(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/gwr.parquet')
        solkat_preprep_all = pd.read_parquet(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/solkat.parquet')
        Map_egid_pv =        pd.read_parquet(f'{data_path}/output/{pvalloc_scen["name_dir_import"]}/Map_egid_pv.parquet')

        # transformation + constrain comparison df to the same building park as in "historic model". Otherwise, including large number of houses (impossible to receive installation, distorting the accuracy)
        pv_preprep_all['BeginningOfOperation'] = pd.to_datetime(pv_preprep_all['BeginningOfOperation'], format='%Y-%m-%d')
        pv_preprep_all.rename(columns={'BeginningOfOperation': 'BeginOp'}, inplace=True)
        gwr_preprep_all = gwr_preprep_all.loc[
                            (gwr_preprep_all['GBAUJ'] >= pvalloc_scen.get('gwr_selection_specs').get('GBAUJ_minmax')[0]) & 
                            (gwr_preprep_all['GBAUJ'] <= pvalloc_scen.get('gwr_selection_specs').get('GBAUJ_minmax')[1])]

        # only historic selection of modelled houses
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
        # add day to year-month string, to have a proper timestamp
        pvinst_df.loc[pvinst_df['inst_TF'] == True, 'BeginOp'] = pvinst_df.loc[pvinst_df['inst_TF'] == True, 'BeginOp'].apply(lambda x: x if len(x) == 10 else x + '-01') 
        pvinst_df['BeginOp'] = pd.to_datetime(pvinst_df['BeginOp'], format='%Y-%m-%d')
        pvinst_df['bfs'] = pvinst_df['bfs'].astype(str)        
        pvinst_df.loc[pvinst_df['info_source'] == '', 'info_source'] = np.nan


        # built df containing all houses until today ------------------------------
        pv_preprep_all = pv_preprep_all.merge(Map_egid_pv, on='xtf_id', how='left')
        
        gwr_preprep_insample = gwr_preprep_all.loc[gwr_preprep_all['EGID'].isin(pvinst_df['EGID'])]
        df_comparison = gwr_preprep_insample.merge(pv_preprep_all, on='EGID', how='left')
        # BOOKMARK! => this df has duplicates because some installations are built on the same house => assigned the same EGID, as pv_egid mapping is
        #              done through spatial join function. Workaround to remove some duplicates for now.
        df_comparison = df_comparison.loc[~df_comparison.duplicated(subset='EGID', keep = 'last')]        

        
        # rename columns for better understanding and merge
        cols_to_add_suffix = ['']
        pvinst_df.rename(columns={'TotalPower': 'TotalPower_MOD', 
                                  'BeginOp': 'BeginOp_MOD', 
                                  'xtf_id': 'xtf_id_MOD',
                                  }, inplace=True)
        df_comparison.rename(columns={'TotalPower': 'TotalPower_RWD', 
                                      'BeginOp': 'BeginOp_RWD', 
                                      }, inplace=True)


        
        # 2x2 Matrix transformation ------------------------------
        # remove houses with EXISTING installations in the model, to really only look at the houses containing a predicted installation. 
        df_comp2x2 = copy.deepcopy(df_comparison)
        date_range_gwr_select = pd.date_range(start=f'{pvalloc_scen.get('gwr_selection_specs').get('GBAUJ_minmax')[0]}-01-01', 
                                       end  =f'{pvalloc_scen.get('gwr_selection_specs').get('GBAUJ_minmax')[1]}-12-31', freq='D')
        date_T0 = pd.to_datetime(f'{pvalloc_scen.get('gwr_selection_specs').get('GBAUJ_minmax')[1]+1}-01-01')

        # remove houses with existing installations that where already built before the inst allocation started
        # df_comp2x2 = df_comp2x2.loc[~df_comp2x2['BeginOp_RWD'].isin(date_range_gwr_select)]
        # df_comp2x2 = df_comp2x2.loc[(df_comp2x2['BeginOp_RWD'].isna()) |
        #                       (df_comp2x2['BeginOp_RWD'] < date_T0)]
        df_comp2x2 = df_comp2x2.loc[(df_comp2x2['BeginOp_RWD'].isna()) |
                              (df_comp2x2['BeginOp_RWD'] >= date_T0)]

        df_comp2x2 = df_comp2x2.merge(pvinst_df, on='EGID', how='left')
        

        # add columns for comparison
        df_comp2x2['TotalPower_DETLA'] = df_comp2x2['TotalPower_MOD'] - df_comp2x2['TotalPower_RWD']
        df_comp2x2['BeginOp_DELTAdays'] = (df_comp2x2['BeginOp_MOD'] - df_comp2x2['BeginOp_RWD']).dt.days

        df_comp2x2['d_TruePositive_Installation']  = np.where((df_comp2x2['TotalPower_MOD'].notna()) & 
                                                                 (df_comp2x2['TotalPower_RWD'].notna()), True, False)
        df_comp2x2['d_FalsePositive_Installation'] = np.where((df_comp2x2['TotalPower_MOD'].notna()) &
                                                                 (df_comp2x2['TotalPower_RWD'].isna()), True, False)
        df_comp2x2['d_TrueNegative_Installation']  = np.where((df_comp2x2['TotalPower_MOD'].isna()) &
                                                                 (df_comp2x2['TotalPower_RWD'].isna()), True, False)
        df_comp2x2['d_FalseNegative_Installation'] = np.where((df_comp2x2['TotalPower_MOD'].isna()) &
                                                                 (df_comp2x2['TotalPower_RWD'].notna()), True, False)


        # plot 2x2 Matrix ------------------------------
        if True:
            true_positive_count, false_positive_count, true_negative_count, false_negative_count = sum(df_comp2x2['d_TruePositive_Installation']), sum(df_comp2x2['d_FalsePositive_Installation']), sum(df_comp2x2['d_TrueNegative_Installation']), sum(df_comp2x2['d_FalseNegative_Installation'])
            true_positive_perct, false_positive_perct, true_negative_perct, false_negative_perct = true_positive_count / len(df_comp2x2)*100 , false_positive_count / len(df_comp2x2)*100, true_negative_count / len(df_comp2x2)*100, false_negative_count / len(df_comp2x2)*100
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
                title=f'Installation Prediction Accuracy, (only EGIDs w/o installation at T0, scen: {scen})',
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

            if postprocess_analysis_settings['prediction_accuracy_specs']['show_plot']:
                if postprocess_analysis_settings['prediction_accuracy_specs']['show_all_scen']:
                    fig.show()
                elif not postprocess_analysis_settings['prediction_accuracy_specs']['show_all_scen']:
                    fig.show() if i_scen == 0 else None

            os.makedirs(f'{data_path}/output/postprocess_analysis/{scen}', exist_ok=True)
            fig.write_html(f'{data_path}/output/postprocess_analysis/{scen}/plot_prediction_accuracy2x2.html')
        


        # plot line installed Capa ------------------------------
        capa_MOD_df = copy.deepcopy(pvinst_df.loc[pvinst_df['TotalPower_MOD'].notna()])
        capa_RWD_df = copy.deepcopy(df_comparison.loc[df_comparison['TotalPower_RWD'].notna()])
        
        def transf_to_lineCapa_df(df_original, by_MonthYear_str, ):
            df = copy.deepcopy(df_original)
            BeginOp_col = [col for col in df.columns if 'BeginOp' in col][0]
            TotalPower_col = [col for col in df.columns if 'TotalPower' in col][0]

            if by_MonthYear_str == 'year':
                df[BeginOp_col] = df[BeginOp_col].dt.to_period('Y')
                
            elif by_MonthYear_str == 'month':
                df[BeginOp_col] = df[BeginOp_col].dt.to_period('M')
                
            if 'info_source' in df.columns:
                df = df.groupby([BeginOp_col, 'info_source'])[TotalPower_col].sum().reset_index()
                df[f'{BeginOp_col}_{by_MonthYear_str}'] = df[BeginOp_col].dt.to_timestamp()

                df_built = df.loc[df['info_source'] == 'pv_df'].copy()
                df_built[f'{TotalPower_col}_{by_MonthYear_str}_cumsum'] = df_built[TotalPower_col].cumsum()
                
                df_pred = df.loc[df['info_source'] == 'alloc_algorithm'].copy()
                df_pred[f'{TotalPower_col}_{by_MonthYear_str}_cumsum'] = df_pred[TotalPower_col].cumsum()

                df[f'{TotalPower_col}_{by_MonthYear_str}_cumsum'] = df[TotalPower_col].cumsum()
                return df, df_built, df_pred
            
            elif 'info_source' not in df.columns:
                df = df.groupby(BeginOp_col)[TotalPower_col].sum().reset_index()
                df[f'{BeginOp_col}_{by_MonthYear_str}'] = df[BeginOp_col].dt.to_timestamp()
                df[f'{TotalPower_col}_{by_MonthYear_str}_cumsum'] = df[TotalPower_col].cumsum()
                return df
        
        capa_MOD_month_df, capa_MOD_month_built_df, capa_MOD_month_pred_df = transf_to_lineCapa_df(capa_MOD_df, 'month')
        capa_MOD_year_df, capa_MOD_year_built_df, capa_MOD_year_pred_df = transf_to_lineCapa_df(capa_MOD_df, 'year')
        capa_RWD_month_df = transf_to_lineCapa_df(capa_RWD_df, 'month')
        capa_RWD_year_df = transf_to_lineCapa_df(capa_RWD_df, 'year')

        def add_growth_rate_col(df, by_MonthYear_str):
            TotalPower_col = [col for col in df.columns if 'TotalPower' in col and not 'cumsum' in col][0]
            df[f'{TotalPower_col}_{by_MonthYear_str}_growthrate'] = df[TotalPower_col].pct_change()
            return df

        capa_RWD_month_df = add_growth_rate_col(capa_RWD_month_df, 'month')
        capa_RWD_year_df = add_growth_rate_col(capa_RWD_year_df, 'year')
        capa_MOD_month_df = add_growth_rate_col(capa_MOD_month_df, 'month')
        capa_MOD_year_df = add_growth_rate_col(capa_MOD_year_df, 'year')



        # plot line installed Capa ----------
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=capa_RWD_month_df['BeginOp_RWD_month'], y=np.zeros(capa_RWD_month_df.shape[0]), mode='lines+markers', name='-- RealWorldData ----------', opacity=0))
        fig.add_trace(go.Scatter(x=capa_RWD_month_df['BeginOp_RWD_month'], y=capa_RWD_month_df['TotalPower_RWD'], mode='lines+markers', name='RWD TotalPower (built, month)', opacity=0.7))
        fig.add_trace(go.Scatter(x=capa_RWD_year_df['BeginOp_RWD_year'], y=capa_RWD_year_df['TotalPower_RWD'], mode='lines+markers', name='RWD TotalPower (built, year)', opacity=0.7))
        fig.add_trace(go.Scatter(x=capa_RWD_month_df['BeginOp_RWD_month'], y=capa_RWD_month_df['TotalPower_RWD_month_cumsum'], mode='lines+markers', name='RWD TotalPower (built, cum.month)', opacity=0.7))
        
        fig.add_trace(go.Scatter(x=capa_MOD_month_df['BeginOp_MOD_month'], y=np.zeros(capa_MOD_month_df.shape[0]), mode='lines+markers', name='-- Modelled ----------', opacity=0))
        fig.add_trace(go.Scatter(x=capa_MOD_month_df['BeginOp_MOD_month'], y=capa_MOD_month_df['TotalPower_MOD'], mode='lines+markers', name='MOD TotalPower (built + predicted, month)', opacity=0.7))
        fig.add_trace(go.Scatter(x=capa_MOD_year_df['BeginOp_MOD_year'], y=capa_MOD_year_df['TotalPower_MOD'], mode='lines+markers', name='MOD TotalPower (built + predicted, year)', opacity=0.7))
        fig.add_trace(go.Scatter(x=capa_MOD_month_df['BeginOp_MOD_month'], y=capa_MOD_month_df['TotalPower_MOD_month_cumsum'], mode='lines+markers', name='MOD TotalPower (built + pred, cum.month)', opacity=0.7))     

        fig.add_trace(go.Scatter(x=capa_MOD_month_built_df['BeginOp_MOD_month'], y=np.zeros(capa_MOD_month_built_df.shape[0]), mode='lines+markers', name='-- Mod before T0 ----------', opacity=0))
        fig.add_trace(go.Scatter(x=capa_MOD_month_built_df['BeginOp_MOD_month'], y=capa_MOD_month_built_df['TotalPower_MOD'], mode='lines+markers', name='MOD TotalPower (built, month)', opacity=0.7))
        fig.add_trace(go.Scatter(x=capa_MOD_year_built_df['BeginOp_MOD_year'], y=capa_MOD_year_built_df['TotalPower_MOD'], mode='lines+markers', name='MOD TotalPower (built, year)', opacity=0.7))
        fig.add_trace(go.Scatter(x=capa_MOD_month_built_df['BeginOp_MOD_month'], y=capa_MOD_month_built_df['TotalPower_MOD_month_cumsum'], mode='lines+markers', name='MOD TotalPower (built, cum.month)', opacity=0.7))

        fig.add_trace(go.Scatter(x=capa_MOD_month_built_df['BeginOp_MOD_month'], y=np.zeros(capa_MOD_month_built_df.shape[0]), mode='lines+markers', name='-- growth rates ----------', opacity=0))
        fig.add_trace(go.Scatter(x=capa_RWD_month_df['BeginOp_RWD_month'], y=capa_RWD_month_df['TotalPower_RWD_month_growthrate'], mode='lines+markers', name='RWD TotalPower (built, growth month)', opacity=0.7))
        fig.add_trace(go.Scatter(x=capa_RWD_year_df['BeginOp_RWD_year'], y=capa_RWD_year_df['TotalPower_RWD_year_growthrate'], mode='lines+markers', name='RWD TotalPower (built, growth year)', opacity=0.7))
        fig.add_trace(go.Scatter(x=capa_MOD_month_df['BeginOp_MOD_month'], y=capa_MOD_month_df['TotalPower_MOD_month_growthrate'], mode='lines+markers', name='MOD TotalPower (built + pred, growth month)', opacity=0.7))
        fig.add_trace(go.Scatter(x=capa_MOD_year_df['BeginOp_MOD_year'], y=capa_MOD_year_df['TotalPower_MOD_year_growthrate'], mode='lines+markers', name='MOD TotalPower (built + pred, growth year)', opacity=0.7))

        fig.update_layout(
            xaxis_title='Time',
            yaxis_title='Installed Capacity [kWp]',
            title=f'Installed Capacity over Time (T0 to Today, scen: {scen}, T0: {pvalloc_scen.get('T0_prediction')})',
        )
        
        if postprocess_analysis_settings['prediction_accuracy_specs']['show_plot']:
            if postprocess_analysis_settings['prediction_accuracy_specs']['show_all_scen']:
                fig.show()
            elif not postprocess_analysis_settings['prediction_accuracy_specs']['show_all_scen']:
                fig.show() if i_scen == 0 else None

        os.makedirs(f'{data_path}/output/postprocess_analysis/{scen}', exist_ok=True)
        fig.write_html(f'{data_path}/output/postprocess_analysis/{scen}/plot_installedCapa_overTime.html')
        
        
        
        print('asdf')
