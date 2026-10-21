# python integrated packages
import sys
import os as os
import copy
import glob

# external packages
import pandas as pd
import plotly.graph_objects as go
import polars as pl

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndLineProductionHOYPerNode(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_line_productionHOY_per_node(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('plot_ind_line_productionHOY_per_node', self.visual_sett.log_name)
        # print_to_logfile(f'{self.visual_sett.pvalloc_scen_list}', self.visual_sett.log_name)

        fig_scen_comp = go.Figure()
        export_plot_data_list = []

        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):

            # setup + import ----------
            if not self._set_mc_data_path_or_skip(scen):
                continue
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            gridnode_df = pl.read_parquet(f'{self.visual_sett.mc_data_path}/gridnode_df.parquet')
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

            # plot ----------------
            if any([n in self.visual_sett.node_selection_for_plots for n in gridnode_df['grid_node'].unique()]):
                nodes = [n for n in gridnode_df['grid_node'].unique() if n in self.visual_sett.node_selection_for_plots]
            elif len(gridnode_df['grid_node'].unique()) > 5: 
                nodes = gridnode_df['grid_node'].unique()[:2]
            else:
                nodes = gridnode_df['grid_node'].unique()


            fig = go.Figure()
            gridnode_total_df = gridnode_total_df.to_pandas()
            gridnode_df = gridnode_df.to_pandas()
            # reduce t_hours to have more "handable - plots" in plotly
            if self.visual_sett.cut_timeseries_to_zoom_hour: 
                gridnode_total_df = gridnode_total_df[
                    (gridnode_total_df['t_int'] >= self.visual_sett.default_zoom_hour[0] - (24 * 7)) &
                    (gridnode_total_df['t_int'] <= self.visual_sett.default_zoom_hour[1] + (24 * 7))
                ].copy()
            if self.visual_sett.cut_timeseries_to_zoom_hour: 
                gridnode_df = gridnode_df[
                    (gridnode_df['t_int'] >= self.visual_sett.default_zoom_hour[0] - (24 * 7)) &
                    (gridnode_df['t_int'] <= self.visual_sett.default_zoom_hour[1] + (24 * 7))
                ].copy()


            for node in nodes:
                filter_df = copy.deepcopy(gridnode_df.loc[gridnode_df['grid_node'] == node])
                # fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['demand_kW'],                    name=f'{node} - demand_kW',                   opacity = 0.5, line = dict( dash='dot' , ),                                   )) # color='rgb(95,70,144)',    )))
                # fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['pvprod_kW'],                    name=f'{node} - pvprod_kW',                   opacity = 0.5, line = dict( dash='dot' , ),                                   )) # color='rgb(29,105,150)',   )))
                # fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['selfconsum_kW'],                name=f'{node} - selfconsum_kW',               opacity = 0.5, line = dict( dash='dot' , ),                                   )) # color='rgb(56,166,165)',   )))
                # fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['netfeedin_kW'],                 name=f'{node} - netfeedin_kW',                opacity = 0.5, line = dict( dash='dot' , ), stackgroup ='feedin'              )) # color='rgb(204,80,62)',    )))
                # fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['netdemand_kW'],                 name=f'{node} - netdemand_kW',                opacity = 0.5, line = dict( dash='dot' , ), stackgroup ='netdem+outtopodem'   )) # color='rgb(148,52,110)',   )))
                # fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['demand_proxy_out_kW'],          name=f'{node} - demand_proxy_out_kW',         opacity = 0.5, line = dict( dash='dot' , ), stackgroup ='netdem+outtopodem'   )) # color='rgb(111,64,112)',   )))
                # fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['feedin_atnode_kW'],             name=f'{node} - feedin_atnode',               opacity = 1.0, line = dict(              ), stackgroup ='feedin_atnode'       )) # color='rgb(102,102,102)',  )))
                # fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['demand_atnode_kW'],             name=f'{node} - demand_atnode',               opacity = 1.0, line = dict(              ), stackgroup ='demand_at_node'      )) # color='rgb(179,179,179)',  )))
                # fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['max_demand_feedin_atnode_kW'],  name=f'{node} - max_demand_feedin_atnode_kW', opacity = 1.0, line = dict(              ),                                   )) # color='rgb(230,171,2)',    )))
                # fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['kW_threshold'],                 name=f'{node} - kW_threshold',                opacity = 1.0, line = dict(              ),                                   )) # color='rgb(47,138,196)',   )))
                fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['feedin_atnode_taken_kW'],       name= f'{node} - feedin_all_taken',           opacity = 1.0, line = dict(              ), stackgroup ='feedin_taken'        )) # color='rgb(17,119,51)',    )))
                fig.add_trace(go.Scatter(x=filter_df['t_int'], y=filter_df['feedin_atnode_loss_kW'],        name=f'{node} - feedin_all_loss',             opacity = 1.0, line = dict(              ), stackgroup ='feedin_loss'         )) # color='rgb(204,80,62)',    )))   
                fig.add_trace(go.Scatter(x=[None,], y=[None,], name=f'-- grid_node: {node} {10*"-"}', opacity=0, mode='lines+markers', line=dict(color='black', width=1, dash='dash')))

            fig.add_trace(go.Scatter(x=[None, ], y=[None, ], name='', opacity = 0, ))
            fig.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['feedin_atnode_kW'],            name='Total feedin_atnode_kW',            mode='lines+markers', line=dict(color='black',           width=3),  marker=dict(symbol='cross',), ))
            fig.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['demand_atnode_kW'],            name='Total demand_atnode_kW',            mode='lines+markers', line=dict(color='grey',            width=3),   marker=dict(symbol='cross',), ))
            fig.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['max_demand_feedin_atnode_kW'], name='Total max_demand_feedin_atnode_kW', mode='lines+markers', line=dict(color='rgb(255,217,47)', width=3), marker=dict(symbol='cross',), ))                   
            fig.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['feedin_atnode_taken_kW'],      name='Total feedin_atnode_taken_kW',      mode='lines+markers', line=dict(color='green',           width=3),  marker=dict(symbol='cross',), ))
            fig.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['feedin_atnode_loss_kW'],       name='Total feedin_atnode_loss_kW',       mode='lines+markers', line=dict(color='red',             width=3),  marker=dict(symbol='cross',), ))
            fig.add_trace(go.Scatter(x=[None, ], y=[None, ], name=f'-- Total aggregation: {scen} {10*"-"}', opacity = 0, ))
            # fig.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['kW_threshold'],           name='Total kW_threshold',           mode='lines+markers', line=dict(color='blue',  width=3), marker= dict(symbol='cross')))
            # fig.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['holding_capacity_all'],  name='Total holding_capacity_all',  mode='lines+markers', line=dict(color='purple', width=3), marker=dict(symbol='cross')))

            fig_scen_comp.add_trace(go.Scatter(x=[None, ], y=[None, ], name=f'-- Total aggregation: {scen} {10*"-"}', opacity = 0, ))
            fig_scen_comp.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['feedin_atnode_loss_kW'],       name='Total feedin_atnode_loss_kW',       mode='lines+markers', line=dict(width=2), )) #  marker=dict(symbol='cross',), ))
            fig_scen_comp.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['feedin_atnode_taken_kW'],      name='Total feedin_atnode_taken_kW',      mode='lines+markers', line=dict(width=2), )) #  marker=dict(symbol='cross',), ))
            fig_scen_comp.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['max_demand_feedin_atnode_kW'], name='Total max_demand_feedin_atnode_kW', mode='lines+markers', line=dict(width=2), )) # marker=dict(symbol='cross',), ))                   
            fig_scen_comp.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['demand_atnode_kW'],            name='Total demand_atnode_kW',            mode='lines+markers', line=dict(width=2), )) #   marker=dict(symbol='cross',), ))
            fig_scen_comp.add_trace(go.Scatter(x=gridnode_total_df['t_int'], y=gridnode_total_df['feedin_atnode_kW'],            name='Total feedin_atnode_kW',            mode='lines+markers', line=dict(width=2), )) #  marker=dict(symbol='cross',), ))
            fig_scen_comp.add_trace(go.Scatter(x=[None, ], y=[None, ], name='', opacity = 0, ))

            # append export plot data
            tmp_df = gridnode_total_df.loc[:,
                [
                    't_int',
                    'feedin_atnode_kW',
                    'demand_atnode_kW',
                    'max_demand_feedin_atnode_kW',
                    'feedin_atnode_taken_kW',
                    'feedin_atnode_loss_kW',
            ]]
            tmp_df.loc[:,'scen'] = scen
            export_plot_data_list.append(tmp_df)

            # update + export
            fig.update_layout(
                xaxis_title='Hour of Year',
                yaxis_title='Production / Feedin (kW)',
                legend_title='Node ID',
                title = f'Production per node (kW, weather year: __, self consum. rate: __)', 
                template = 'plotly_white', 
            )

            fig = self.add_scen_name_to_plot(fig, scen, self.pvalloc_scen)
            fig = self.set_default_fig_zoom_hour(fig, self.visual_sett.default_zoom_hour)

            if self.visual_sett.add_day_night_HOY_bands:
                fig = self.add_day_night_bands_HOY_plot(fig, gridnode_df['t_int'])

            if plot_show:
                fig.show()
            if plot_export and not self.visual_sett.export_only_agg_comparison_plots:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}/plot_ind_line_productionHOY_per_node___{scen}.html')
                else:   
                    fig.write_html(f'{self.visual_sett.visual_path}/plot_ind_line_productionHOY_per_node___{scen}.html')
                print_to_logfile(f'\texport: plot_ind_line_productionHOY_per_node.html (for: {scen})', self.visual_sett.log_name)



        # export comparison plot
        fig_scen_comp.update_layout(
            xaxis_title='Hour of Year',
            yaxis_title='Production / Feedin (kW)',
            legend_title='Scen',
            title = f'Production per node (kW, weather year: __, self consum. rate: __)', 
            template = 'plotly_white', 
            )
        fig_scen_comp = self.set_default_fig_zoom_hour(fig_scen_comp, self.visual_sett.default_zoom_hour)

        if self.visual_sett.add_day_night_HOY_bands:
            fig_scen_comp = self.add_day_night_bands_HOY_plot(fig_scen_comp, gridnode_df['t_int'])

            if plot_show:
                fig_scen_comp.show()
        if plot_export:
            if os.path.exists(f'{self.visual_sett.visual_path}/plot_agg_line_productionHOY_per_node___{len(self.visual_sett.pvalloc_scen_list)}scen.html'):
                n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_path}/plot_agg_line_productionHOY_per_node___{len(self.visual_sett.pvalloc_scen_list)}scen*.html'))
                os.rename(f'{self.visual_sett.visual_path}/plot_agg_line_productionHOY_per_node___{len(self.visual_sett.pvalloc_scen_list)}scen.html', 
                          f'{self.visual_sett.visual_path}/plot_agg_line_productionHOY_per_node___{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.html')    
            fig_scen_comp.write_html(f'{self.visual_sett.visual_path}/plot_agg_line_productionHOY_per_node___{len(self.visual_sett.pvalloc_scen_list)}scen.html')
            print_to_logfile(f'\texport: plot_agg_line_productionHOY_per_node___{len(self.visual_sett.pvalloc_scen_list)}scen.html', self.visual_sett.log_name)


            # export plot data as csv
            export_plot_data_df = pd.concat(export_plot_data_list, ignore_index=True)
            os.makedirs(f'{self.visual_sett.visual_static_export_path}', exist_ok=True)
            export_df_name         = 'plot_agg_line_productionHOY_per_node___export_plot_data___'
            export_df_name_scenlen = f'{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen'
            if os.path.exists(f'{self.visual_sett.visual_static_export_path}/{export_df_name_scenlen}.csv'):
                n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen*.csv'))
                os.rename(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', 
                          f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.csv')
            export_plot_data_df.to_csv(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', index=False)
            print_to_logfile(f'\texport: {export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', self.visual_sett.log_name)
