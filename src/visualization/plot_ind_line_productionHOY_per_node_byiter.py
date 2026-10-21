# python integrated packages
import sys
import os as os
import glob

# external packages
import pandas as pd
import plotly.graph_objects as go
import polars as pl

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndLineProductionHOYPerNodeByiter(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_line_productionHOY_per_node_byiter(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('plot_ind_line_productionHOY_per_node_byiter', self.visual_sett.log_name)
        # print_to_logfile(f'{self.visual_sett.pvalloc_scen_list}', self.visual_sett.log_name)

        fig_scen_comp = go.Figure()
        export_plot_data_list = []

        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):

            # setup + import ----------
            if not self._set_mc_data_path_or_skip(scen):
                continue
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            iter_to_plot = [ 1, 2, 3, 4, 5, 6, 7, 8]
            gridnode_paths = []
            for i in iter_to_plot:
                gridnode_path = f'{self.visual_sett.mc_data_path}/pred_gridprem_node_by_M/gridnode_df_{i}.parquet'
                if os.path.exists(gridnode_path):
                    gridnode_paths.append(gridnode_path)

            gridnode_paths = gridnode_paths + [f'{self.visual_sett.mc_data_path}/gridnode_df.parquet']

            for gridnode_i in gridnode_paths:
                gridnode_df = pl.read_parquet(f'{gridnode_i}')
                if 'pred_gridprem_node_by_M' in gridnode_i:
                    legend_suffix = gridnode_i.split('pred_gridprem_node_by_M/gridnode_df_')[-1].split('.parquet')[0]
                else:
                    legend_suffix = 'end_iter'

                gridnode_df = gridnode_df.with_columns([
                    pl.col('t').str.strip_chars('t_').cast(pl.Int64).alias('t_int'),
                ])
                gridnode_df = gridnode_df.sort("t_int", descending=False)

                # aggregate to total production per HOY
                gridnode_df = gridnode_df.with_columns([
                    ((pl.col('kW_threshold') - pl.col('max_demand_feedin_atnode_kW')) / pl.col('kW_threshold')).alias('holding_capacity')
                ])

                gridnode_total_df = gridnode_df.group_by(['t', 't_int']).agg([
                    pl.col('feedin_atnode_kW').sum().alias('feedin_atnode_kW'),
                    pl.col('demand_atnode_kW').sum().alias('demand_atnode_kW'),
                    pl.col('max_demand_feedin_atnode_kW').sum().alias('max_demand_feedin_atnode_kW'),
                    pl.col('feedin_atnode_taken_kW').sum().alias('feedin_atnode_taken_kW'),
                    pl.col('feedin_atnode_loss_kW').sum().alias('feedin_atnode_loss_kW'),
                    pl.col('kW_threshold').sum().alias('kW_threshold'),
                    pl.col('holding_capacity').mean().alias('holding_capacity_all'),
                ])
                gridnode_total_df = gridnode_total_df.sort("t_int", descending=False)

                if self.visual_sett.cut_timeseries_to_zoom_hour: 
                    gridnode_total_df = gridnode_total_df.filter(
                        (pl.col('t_int') >= self.visual_sett.default_zoom_hour[0] - (24 * 7)) &
                        (pl.col('t_int') <= self.visual_sett.default_zoom_hour[1] + (24 * 7))
                    ).clone()


                fig_scen_comp.add_trace(go.Scatter(x=[None, ], y=[None, ], name=f'-- Total aggregation: {scen} {10*"-"}', opacity = 0, ))
                fig_scen_comp.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['feedin_atnode_loss_kW'],       name=f'Total feedin_atnode_loss_kW {legend_suffix}',             mode='lines+markers', line=dict(width=2), )) #  marker=dict(symbol='cross',), ))
                fig_scen_comp.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['feedin_atnode_taken_kW'],      name=f'Total feedin_atnode_taken_kW {legend_suffix}',            mode='lines+markers', line=dict(width=2), )) #  marker=dict(symbol='cross',), ))
                # fig_scen_comp.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['max_demand_feedin_atnode_kW'], name=f'Total max_demand_feedin_atnode_kW {legend_suffix}',       mode='lines+markers', line=dict(width=2), )) # marker=dict(symbol='cross',), ))                   
                # fig_scen_comp.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['demand_atnode_kW'],            name=f'Total demand_atnode_kW {legend_suffix}',                  mode='lines+markers', line=dict(width=2), )) #   marker=dict(symbol='cross',), ))
                fig_scen_comp.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['feedin_atnode_kW'],            name=f'Total feedin_atnode_kW {legend_suffix}',                  mode='lines+markers', line=dict(width=2), )) #  marker=dict(symbol='cross',), ))
                fig_scen_comp.add_trace(go.Scatter(x=[None, ], y=[None, ], name='', opacity = 0, ))


                # append export plot data
                gridnode_total_df_pd = gridnode_total_df.to_pandas()
                tmp_df = gridnode_total_df_pd.loc[:,
                    [
                        't_int',
                        'feedin_atnode_kW',
                        'demand_atnode_kW',
                        'max_demand_feedin_atnode_kW',
                        'feedin_atnode_taken_kW',
                        'feedin_atnode_loss_kW',
                ]].copy()
                tmp_df.loc[:,'iter'] = legend_suffix
                tmp_df.loc[:,'scen'] = scen
                export_plot_data_list.append(tmp_df)


        # export comparison plot
        fig_scen_comp.update_layout(
            xaxis_title='Hour of Year',
            yaxis_title='Production / Feedin (kW)',
            legend_title='Scen',
            title = 'Production per node, by model iterations', 
            template = 'plotly_white', 
            )
        fig_scen_comp = self.set_default_fig_zoom_hour(fig_scen_comp, self.visual_sett.default_zoom_hour)

        if self.visual_sett.add_day_night_HOY_bands:
            fig_scen_comp = self.add_day_night_bands_HOY_plot(fig_scen_comp, gridnode_df['t_int'])

            if plot_show:
                fig_scen_comp.show()
        if plot_export:
            if os.path.exists(f'{self.visual_sett.visual_path}/plot_agg_line_productionHOY_per_node_byiter___{len(self.visual_sett.pvalloc_scen_list)}scen.html'):
                n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_path}/plot_agg_line_productionHOY_per_node_byiter___{len(self.visual_sett.pvalloc_scen_list)}scen*.html'))
                os.rename(f'{self.visual_sett.visual_path}/plot_agg_line_productionHOY_per_node_byiter___{len(self.visual_sett.pvalloc_scen_list)}scen.html', 
                          f'{self.visual_sett.visual_path}/plot_agg_line_productionHOY_per_node_byiter___{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.html')    
            fig_scen_comp.write_html(f'{self.visual_sett.visual_path}/plot_agg_line_productionHOY_per_node_byiter___{len(self.visual_sett.pvalloc_scen_list)}scen.html')
            print_to_logfile(f'\texport: plot_agg_line_productionHOY_per_node_byiter___{len(self.visual_sett.pvalloc_scen_list)}scen.html', self.visual_sett.log_name)


            # export plot data as csv
            export_plot_data_df = pd.concat(export_plot_data_list)
            os.makedirs(f'{self.visual_sett.visual_static_export_path}', exist_ok=True)
            export_df_name         = 'plot_agg_line_productionHOY_per_node_byiter___export_plot_data___'
            export_df_name_scenlen = f'{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen'
            if os.path.exists(f'{self.visual_sett.visual_static_export_path}/{export_df_name_scenlen}.csv'):
                n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen*.csv'))
                os.rename(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', 
                          f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.csv')
            export_plot_data_df.to_csv(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', index=False)
            print_to_logfile(f'\texport: {export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', self.visual_sett.log_name)
