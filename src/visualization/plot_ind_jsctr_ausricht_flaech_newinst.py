# python integrated packages
import sys
import os as os
import glob

# external packages
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndJsctrAusrichtFlaechNewinst(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_jsctr_ausricht_flaech_newinst(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('Starting plot_ind_jsctr_ausricht_flaech_newinst', self.visual_sett.log_name)

        # create Plots and Color Dict ----------------------------------------
        # "settings" ----------
        n_cols_jsctr = 3

        n_rows_jsct= int(np.ceil(len(self.visual_sett.pvalloc_scen_list) / n_cols_jsctr))

        fig_jsctr = make_subplots(rows=n_rows_jsct, cols=n_cols_jsctr,
                                                subplot_titles=[f'{scen}' for scen in self.visual_sett.pvalloc_scen_list])

        # color dict ----------
        global_color_iter_rows = []
        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
            if not self._set_mc_data_path_or_skip(scen):
                continue
            list_pred_inst_df = glob.glob(f'{self.visual_sett.mc_data_path}/pred_npv_inst_by_M/pred_inst_df*.parquet')
            iter_list = [int(os.path.basename(f).split('pred_inst_df_')[1].split('.parquet')[0]) for f in list_pred_inst_df]
            max_iter = max(iter_list)
            global_color_iter_rows.append({'scen': scen, 'iter_list': iter_list, 'max_iter': max_iter})

        global_color_iter_df = pd.DataFrame(global_color_iter_rows)
        max_global_color_iter_df = global_color_iter_df.loc[global_color_iter_df['max_iter'].idxmax()]
        color_map_iter = {
            iter: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
            for i, iter in enumerate(range(0, max_global_color_iter_df['max_iter'] + 1))
        }

        fig = go.Figure()

        # Plots ----------------------------------------
        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)
            if not self._set_mc_data_path_or_skip(scen):
                continue
            list_pred_inst_df = glob.glob(f'{self.visual_sett.mc_data_path}/pred_npv_inst_by_M/pred_inst_df*.parquet')
            iter_list = [int(os.path.basename(f).split('pred_inst_df_')[1].split('.parquet')[0]) for f in list_pred_inst_df]
            max_iter = max(iter_list)

            last_pred_inst_df = pd.read_parquet(f'{self.visual_sett.mc_data_path}/pred_npv_inst_by_M/pred_inst_df_{max_iter}.parquet')

            for iter in last_pred_inst_df['iter_round'].unique():
                sub_predinst = last_pred_inst_df.loc[last_pred_inst_df['iter_round'] == float(iter)]

                for i_egid, egid in enumerate(sub_predinst['EGID'].unique()):
                    show_legend_TF = True if i_egid == 0 else False

                    egid_predinst = sub_predinst.loc[sub_predinst['EGID'] == egid]
                    # egid2 = '101228715'
                    # egid_predinst = sub_predinst.loc[sub_predinst['EGID'] == egid2]

                    hover_text1 = (
                        f'scen: {scen}<br>EGID: {egid}<br>iter: {iter}<br>*dfuids in EGID*'
                    )
                    if egid_predinst.shape[0] > 1:
                        hover_text2 = '<br>'.join([f"df_uid: {df_uid}, AUSRICHTUNG: {ausrichtung}, FLAECHE: {flaeche}" 
                                for df_uid, ausrichtung, flaeche in zip(
                                    egid_predinst['df_uid'], 
                                    egid_predinst['AUSRICHTUNG'], 
                                    egid_predinst['FLAECHE']
                                )])
                    else: 
                        hover_text2 = f"df_uid: {egid_predinst['df_uid'].values[0]}, AUSRICHTUNG: {egid_predinst['AUSRICHTUNG'].values[0]}, FLAECHE: {egid_predinst['FLAECHE'].values[0]}"

                    hover_text = hover_text1 + '<br>' + hover_text2 + '<extra></extra>'

                    fig.add_trace(go.Scatter(
                        x=egid_predinst['AUSRICHTUNG'],
                        y=egid_predinst['FLAECHE'],
                        mode='markers',
                        name=f'{scen}_iter{iter}',
                        legendgroup=f'{scen}_iter{iter}',  # Legend group by iter
                        showlegend=show_legend_TF,  # Show legend only for the first iteration
                        marker=dict(
                            color=color_map_iter[iter],
                            size=8,
                        ),
                        opacity=0.7,
                        hovertemplate= hover_text,
                      ))
                    if egid_predinst.shape[0] > 1: 
                        fig.add_trace(go.Scatter(
                            x=egid_predinst['AUSRICHTUNG'],
                            y=egid_predinst['FLAECHE'],
                            mode='lines+text',  # Connect the dots with lines
                            fill='toself',      # Fill the polygon
                            name=f'{scen}_iter{iter}',
                            legendgroup=f'{scen}_iter{iter}',  # Legend group by iter
                            show_legend=False,  # Do not show legend for the filled area
                            line=dict(
                                color= color_map_iter[iter], 
                                width = 1, 
                            ),
                            opacity = 0.4, 
                            fillcolor= color_map_iter[iter], 
                        ))

        if plot_show:
            fig.show()

        if plot_export:
            if os.path.exists(f'{self.visual_sett.visual_path}/plot_agg_jsctr_ausricht_flaech_newinst___{len(self.visual_sett.pvalloc_scen_list)}scen.html'):
                n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_path}/plot_agg_jsctr_ausricht_flaech_newinst___{len(self.visual_sett.pvalloc_scen_list)}scen*.html'))
                os.rename(f'{self.visual_sett.visual_path}/plot_agg_jsctr_ausricht_flaech_newinst___{len(self.visual_sett.pvalloc_scen_list)}scen.html',
                          f'{self.visual_sett.visual_path}/plot_agg_jsctr_ausricht_flaech_newinst___{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.html')
            fig.write_html(f'{self.visual_sett.visual_path}/plot_agg_jsctr_ausricht_flaech_newinst___{len(self.visual_sett.pvalloc_scen_list)}scen.html')
            print_to_logfile(f'\texport: plot_agg_jsctr_ausricht_flaech_newinst___{len(self.visual_sett.pvalloc_scen_list)}scen.html', self.visual_sett.log_name)
