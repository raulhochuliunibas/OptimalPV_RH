# python integrated packages
import sys
import os as os
import glob
import json

# external packages
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndHistContcharactNewinst(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_hist_contcharact_newinst(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('plot_ind_hist_contcharact_newinst', self.visual_sett.log_name)

        # available color palettes
        trace_color_dict = {
            'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
            'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
            'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Plotly3': pc.sequential.Plotly3,
            'Viridis': pc.sequential.Viridis, 'Turbo': pc.sequential.Turbo, 'Blackbody': pc.sequential.Blackbody, 
            'Bluered': pc.sequential.Bluered, 'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
            'Rainbow': pc.sequential.Rainbow, 'Burg': pc.sequential.Burg, 'Cividis': pc.sequential.Cividis,
            'Inferno': pc.sequential.Inferno, 'Magma': pc.sequential.Magma, 'Plasma': pc.sequential.Plasma,
        }     
        iter_color_map = {}
        export_plot_data_list = []


        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):

            if not self._set_mc_data_path_or_skip(scen):
                continue
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            # setup + import ----------
            colnames_binsize_tuples = self.visual_sett.plot_ind_XX_charact_newinst_specs['cols_cont_charact_inst_binsize']
            n_cols                  = self.visual_sett.plot_ind_XX_charact_newinst_specs['n_cols_cont_charact']
            n_rows                   = int(np.ceil(len(colnames_binsize_tuples) / n_cols))


            topo         = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
            pred_inst_df = pl.read_parquet( f'{self.visual_sett.mc_data_path}/pred_inst_df.parquet')
            npv_df       = pl.read_parquet( f'{self.visual_sett.mc_data_path}/npv_df.parquet')


            # add iter_round col if missing
            if 'iter_round' not in pred_inst_df.columns:
                pred_inst_df = pred_inst_df.with_columns([
                    pl.col('BeginOp').str.strptime(pl.Date, fmt="%Y-%m-%d", strict=False).dt.year().alias('BeginOp_year')
                ])
                year_to_index = {year: i+1 for i, year in enumerate(sorted(pred_inst_df['BeginOp_year'].unique()))}
                pred_inst_df = pred_inst_df.with_columns([
                    pl.col('BeginOp_year').map_dict(year_to_index).alias('iter_round')
                ])
                # predinst_all['BeginOp_date'] = pd.to_datetime(predinst_all['BeginOp'], errors='coerce')
                # year_to_index = {year: i+1 for i, year in enumerate(sorted(predinst_all['BeginOp_date'].dt.year.unique()))}
                # predinst_all['iter_round'] = predinst_all['BeginOp_date'].dt.year.map(year_to_index) 


            # transform / adjust cols
            GAREA_list =[]
            for k,v in topo.items():
                row = {
                    'EGID': k,
                    'GAREA': v['gwr_info']['garea'],    
                }
                GAREA_list.append(row)
            GAREA_df = pl.DataFrame(GAREA_list)
            pred_inst_df = pred_inst_df.join(GAREA_df, on='EGID', how='left')

            pred_inst_df = pred_inst_df.with_columns([
                pl.col('iter_round').cast(pl.Int64).alias('iter_round'),

                pl.col('AUSRICHTUNG').alias('AUSRICHTUNG_original'),
                pl.col('NEIGUNG').alias('NEIGUNG_original'),
                (pl.col('AUSRICHTUNG') * pl.col('share_pvprod_used')).alias('AUSRICHTUNG_share_used'),
                (pl.col('NEIGUNG') * pl.col('share_pvprod_used')).alias('NEIGUNG_share_used'),
            ])

            pred_inst_agg = pred_inst_df.group_by('EGID').agg([
                pl.col('df_uid').count().alias('n_dfuid'),
                pl.col('iter_round').first().alias('iter_round'),
                pl.col('FLAECHE').sum().alias('FLAECHE'),   
                pl.col('GAREA').sum().alias('GAREA'),   
                pl.col('AUSRICHTUNG_share_used').sum().alias('AUSRICHTUNG_share_used'),
                pl.col('NEIGUNG_share_used').sum().alias('NEIGUNG_share_used'),
                pl.col('AUSRICHTUNG').mean().alias('AUSRICHTUNG_mean'),

                pl.col('dfuidPower').sum().alias('dfuidPower'),
                pl.col('demand_kW').first().alias('demand_kW'),
                pl.col('pvprod_kW').first().alias('pvprod_kW'),
                pl.col('selfconsum_kW').first().alias('selfconsum_kW'),
                pl.col('NPV_uid').first().alias('NPV_uid'),
            ])

            # add iter_color_map
            unique_iter = list(pred_inst_df['iter_round'].unique())
            iter_not_in_map = [it for it in unique_iter if it not in iter_color_map.keys()]
            iter_color_map.update({
                it: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                for i, it in enumerate(iter_not_in_map)
                })


            # plots ----------
            fig = make_subplots(rows = n_rows, cols = n_cols,
                    subplot_titles = [f'{col_tuple[0]}' for col_tuple in colnames_binsize_tuples],)

            for i_col, col_tuple in enumerate(colnames_binsize_tuples):
                col           = col_tuple[0]
                n_bins        = col_tuple[1]
                subplot_row   = (i_col // n_cols) + 1
                subplot_col   = (i_col % n_cols) + 1

                if 'original' in col:
                    df_plot = pred_inst_df.to_pandas().copy()
                else: 
                    df_plot = pred_inst_agg.to_pandas().copy() 

                global_x_min = df_plot[col].min()
                global_x_max = df_plot[col].max()
                global_bin_size = (global_x_max - global_x_min) / n_bins

                for i_iter, iter in enumerate(sorted(df_plot['iter_round'].unique())):
                    show_legend = True if i_col == 0 else False
                    bin_size  = int((df_plot[col].max() - df_plot[col].min()) / n_bins)
                    df_iter = df_plot.loc[df_plot['iter_round'] == iter]

                    fig.add_trace(
                        go.Histogram(
                            x=df_iter[col],
                            xbins=dict(
                                start=global_x_min,
                                end=global_x_max,
                                size=global_bin_size,
                            ),
                            # nbinsx=n_bins,
                            # name=f'col {col} iter {iter}',
                            name=f'iter {iter}',
                            marker_color=iter_color_map[iter],
                            opacity=0.7, 
                            showlegend=show_legend,
                            legendgroup = f'iter{iter}',
                        ),
                        row=subplot_row,
                        col=subplot_col,
                    )


            # add npv data to tmp_df
            cols_to_join = ['pred_instPower', 'estim_pvinstcost_chf', 'NPV_uid_before_subsidy', 'subs_nodeHC_chf', 'pena_nodeHC_chf', 'NPV_uid', 'econ_inc_chf', 'econ_spend_chf']
            cols_in_npvdfforjoin = [col for col in cols_to_join if col in npv_df.columns]
            if len(cols_in_npvdfforjoin) > 0:
                df_plot_npv = pl.DataFrame(df_plot).join(
                    npv_df.select(['EGID',] + cols_in_npvdfforjoin),
                    on =['EGID',],
                    how='left',
                ).to_pandas()
            else:
                df_plot_npv = df_plot.to_pandas()

            # append export plot data
            pred_inst_cols_to_csv = ['EGID', 'iter_round','GAREA', 'FLAECHE', 'dfuidPower', 'demand_kW', 'pvprod_kW', 'selfconsum_kW']
            # cols_in_npvdfforjoin = [col for col in cols_in_npvdfforjoin]
            # tmp_df = df_plot_npv[pred_inst_cols_to_csv + cols_in_npvdfforjoin]
            tmp_df = df_plot_npv[pred_inst_cols_to_csv ]
            # tmp_df.loc[:,'scen'] = scen
            tmp_df = tmp_df.assign(scen=scen)


            export_plot_data_list.append(tmp_df)



            # update layout
            fig.update_layout(
                title_text=f'Charact. of EGID with new PV (numeric by iteration, scen: {scen})',                         
                barmode= 'overlay',
                template= 'plotly_white',
            )
            if plot_show:
                fig.show()
            if plot_export and not self.visual_sett.export_only_agg_comparison_plots:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}/plot_ind_hist_contcharact_newinst___{scen}.html')
                else:
                    fig.write_html(f'{self.visual_sett.visual_path}/plot_ind_hist_contcharact_newinst___{scen}.html')
                print_to_logfile(f'\texport: plot_ind_hist_contcharact_newinst (for: {scen})', self.visual_sett.log_name)


        # export plot data as csv
        if plot_export:
            export_plot_data_df = pd.concat(export_plot_data_list)
            os.makedirs(f'{self.visual_sett.visual_static_export_path}', exist_ok=True)
            export_df_name         = 'plot_agg_hist_contcharact_newinst___export_plot_data___'
            export_df_name_scenlen = f'{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen'
            if os.path.exists(f'{self.visual_sett.visual_static_export_path}/{export_df_name_scenlen}.csv'):
                n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen*.csv'))
                os.rename(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', 
                          f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.csv')
            export_plot_data_df.to_csv(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', index=False)
            print_to_logfile(f'\texport: {export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', self.visual_sett.log_name)
