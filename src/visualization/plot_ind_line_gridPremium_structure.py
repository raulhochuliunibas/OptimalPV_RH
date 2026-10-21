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


class PlotIndLineGridPremiumStructure(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_line_gridPremium_structure(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('plot_ind_line_gridPremium_structure', self.visual_sett.log_name)

        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
            if not self._set_mc_data_path_or_skip(scen):
                continue
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            # setup + import ----------
            tiers = self.pvalloc_scen.GRIDspec_tiers
            tiers_rel_treshold_list, gridprem_Rp_kWh_list = [], []
            for k,v in tiers.items():
                tiers_rel_treshold_list.append(v[0])
                gridprem_Rp_kWh_list.append(v[1])

            gridprem_tiers_df = pd.DataFrame({'tiers_rel_treshold': tiers_rel_treshold_list, 'gridprem_Rp_kWh': gridprem_Rp_kWh_list})

            # plot ----------
            fig = go.Figure()

            fig.add_trace(go.Scatter
                            (x=gridprem_tiers_df['tiers_rel_treshold'], 
                             y=gridprem_tiers_df['gridprem_Rp_kWh'],
                             mode='lines+markers',
                             name='gridprem, marginal feedin premium (Rp)',
                             showlegend=True,
                             ))
            fig.update_layout(
                title='Grid premium structure, feed-in premium for reaching relative grid node capacity (kVA)',
                xaxis_title='Relative grid node capacity threshold',
                yaxis_title='Feed-in premium (Rp/kWh)',
            )

            fig = self.add_scen_name_to_plot(fig, scen, self.visual_sett.pvalloc_scen_list[i_scen])

            if plot_show:
                fig.show()
            if plot_export:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_ind_line_gridPremium_structure.html')
                else:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_line_gridPremium_structure.html')
