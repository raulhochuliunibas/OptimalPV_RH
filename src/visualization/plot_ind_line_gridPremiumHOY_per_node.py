# python integrated packages
import sys
import os as os

# external packages
import pandas as pd
import plotly.graph_objects as go

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndLineGridPremiumHOYPerNode(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_line_gridPremiumHOY_per_node(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('plot_ind_line_gridPremiumHOY_per_node', self.visual_sett.log_name)

        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
            # setup + import ----------
            if not self._set_mc_data_path_or_skip(scen):
                continue
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            gridprem_ts = pd.read_parquet(f'{self.visual_sett.mc_data_path}/gridprem_ts.parquet')
            gridprem_ts['t_int'] = gridprem_ts['t'].str.extract(r't_(\d+)').astype(int)


            # plot ----------------
            fig = go.Figure()
            for node in self.visual_sett.node_selection_for_plots:
                gridprem_ts_node = gridprem_ts[gridprem_ts['grid_node'] == node]
                gridprem_ts_node.sort_values(by=['t_int'], inplace=True)

                fig.add_trace(go.Scatter(x=gridprem_ts_node['t_int'], y=gridprem_ts_node['prem_Rp_kWh'],
                                         mode='lines', name=f'grid_premium node: {node}'))

            agg_gridprem = gridprem_ts.groupby('t_int').agg({
                't': 'first',
                'prem_Rp_kWh': ['mean', 'std']
            }).reset_index()
            agg_gridprem.columns = ['t_int', 't', 'prem_Rp_kWh_mean', 'prem_Rp_kWh_std']

            fig.add_trace(go.Scatter
                (x=agg_gridprem['t_int'], 
                 y=agg_gridprem['prem_Rp_kWh_mean'],
                 mode='lines',
                 name='grid premium Rp/kWh (mean)',
                 line=dict(color='black', width=2)
                ))
            fig.add_trace(go.Scatter
                (x=agg_gridprem['t_int'],
                 y=agg_gridprem['prem_Rp_kWh_std'],
                 mode='lines',
                 name='grid premium Rp/kWh (std)',
                 line=dict(color='darkgrey', width=1,
                dash='dash')
                ))

            # layout ----------------
            fig.update_layout(
                title=f'Grid premium Rp/kWh per node (scen: {scen})',
                xaxis_title='Hour of year',
                yaxis_title='Grid premium Rp/kWh',
                showlegend=True,
            )
            fig = self.add_scen_name_to_plot(fig, scen, self.visual_sett.pvalloc_scen_list[i_scen])
            fig = self.set_default_fig_zoom_hour(fig, self.visual_sett.default_zoom_hour)


            if plot_show:
                fig.show()
            if plot_export:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_ind_line_gridPremiumHOY_per_node.html')
                else:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_line_gridPremiumHOY_per_node.html')
