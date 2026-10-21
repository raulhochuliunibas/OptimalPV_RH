# python integrated packages
import sys
import os as os
import copy
import json

# external packages
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndLinebandContcharactNewinst(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_lineband_contcharact_newinst(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('plot_ind_lineband_contcharact_newinst', self.visual_sett.log_name)

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

        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):

                if not self._set_mc_data_path_or_skip(scen):
                    continue
                self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

                # setup + import ----------
                colnams_charac_AND_numerator = self.visual_sett.plot_ind_lineband_contcharact_newinst_specs['colnames_cont_charact_installations_AND_numerator']
                trace_color_palette = self.visual_sett.plot_ind_lineband_contcharact_newinst_specs['trace_color_palette']
                # col_colors = [val / len(colnams_charac_AND_numerator) for val in range(1,len(colnams_charac_AND_numerator)+1)]
                col_colors = list(range(1,len(colnams_charac_AND_numerator)+1))
                palette = trace_color_dict[trace_color_palette]

                topo = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
                predinst_all = pd.read_parquet( f'{self.visual_sett.mc_data_path}/pred_inst_df.parquet')

                # add iter_round col if missing
                if 'iter_round' not in predinst_all.columns:
                    predinst_all['BeginOp_date'] = pd.to_datetime(predinst_all['BeginOp'], errors='coerce')
                    year_to_index = {year: i+1 for i, year in enumerate(sorted(predinst_all['BeginOp_date'].dt.year.unique()))}
                    predinst_all['iter_round'] = predinst_all['BeginOp_date'].dt.year.map(year_to_index)                       

                # agg df by EGID because multiple dfuid with larger than 1 partition installations
                predinst_all['n_dfuids'] = predinst_all['EGID']
                predinst_all = predinst_all.groupby(['EGID', ]).agg({
                    'EGID': 'first',
                    'n_dfuids': 'count',
                    'iter_round': 'first',
                    'STROMERTRAG': 'sum',
                    'FLAECHE': 'sum', 
                    'inst_TF': 'first',
                    'info_source': 'first',
                    'xtf_id': 'first',
                    'BeginOp': 'first',
                    'dfuidPower': 'sum',
                    'demand_kW': 'sum',
                    'poss_pvprod': 'sum',
                    'pvprod_kW': 'sum',
                    'selfconsum_kW': 'sum',
                    'netfeedin_kW': 'sum',
                    'netdemand_kW': 'sum',
                }).reset_index(drop=True)
                predinst_absdf = copy.deepcopy(predinst_all)

                agg_dict ={}
                for col_tuple in colnams_charac_AND_numerator:
                    agg_dict[f'{col_tuple[0]}'] = ['mean', 'std', 'sum']
                    predinst_absdf[f'{col_tuple[0]}'] = predinst_absdf[f'{col_tuple[0]}'] / col_tuple[1]
                agg_dict['EGID'] = ['size',]


                agg_predinst_absdf = predinst_absdf.groupby('iter_round').agg(agg_dict)
                agg_predinst_absdf['iter_round'] = agg_predinst_absdf.index
                agg_predinst_absdf.reset_index(drop=True, inplace=True)
                agg_predinst_absdf.replace(np.nan, 0, inplace=True) # replace NaNs with 0, needed if no deviation in std

                agg_predinst_absdf = agg_predinst_absdf.rename(columns={'EGID': 'nEGID_count'})

                # plot ----------------
                fig = go.Figure()
                i_col = 3
                col, col_numerator = colnams_charac_AND_numerator[i_col][0], colnams_charac_AND_numerator[i_col][1]
                for i_col, col_tuple in enumerate(colnams_charac_AND_numerator):
                    col             = col_tuple[0]
                    col_numerator   = col_tuple[1]

                    xaxis   =           agg_predinst_absdf['iter_round']
                    y_mean  =           agg_predinst_absdf[col]['mean']
                    y_lower, y_upper =  agg_predinst_absdf[col]['mean'] - agg_predinst_absdf[col]['std'], agg_predinst_absdf[col]['mean'] + agg_predinst_absdf[col]['std']
                    trace_color = palette[col_colors[i_col % len(col_colors)]]

                    # mean trace
                    fig.add_trace(go.Scatter(x=xaxis, y=y_mean,
                                            name=f'{col} mean (1/{col_numerator})',
                                            legendgroup=f'{col}',
                                            line=dict(color=trace_color),
                                            mode='lines+markers', showlegend=True))
                    # upper / lower bound band
                    fig.add_trace(go.Scatter(
                        x=xaxis.tolist() + xaxis.tolist()[::-1],  # Concatenate xaxis with its reverse
                        y=y_upper.tolist() + y_lower.tolist()[::-1],  # Concatenate y_upper with reversed y_lower
                        fill='toself',
                        fillcolor=trace_color,  # Dynamic color with 50% transparency
                        opacity=0.2,
                        line=dict(color='rgba(255,255,255,0)'),  # No boundary line
                        hoverinfo="skip",  # Don't show info on hover
                        showlegend=False,  # Do not show this trace in the legend
                        legendgroup=f'{col}',  # Group with the mean line
                        visible=True  # Make this visible/toggleable with the mean line
                    ))

                # add nEGID count trace
                total_EGID_in_topo  = len(topo)
                # agg_predinst_counts = (
                #         agg_predinst_absdf.groupby('iter_round')['FLAECHE']
                #         .count()
                #         .reset_index(name='nEGID_count')
                #     )   
                # agg_predinst_absdf.groupby('iter_round').count().reset_index(name='nEGID_count')
                agg_predinst_counts = agg_predinst_absdf[['iter_round', 'nEGID_count']].copy()
                agg_predinst_counts['nEGID_cumm'] = agg_predinst_counts['nEGID_count'].cumsum()
                agg_predinst_counts['nEGID_cumm_rel'] = agg_predinst_counts['nEGID_cumm'] / total_EGID_in_topo 
                fig.add_trace(go.Scatter(x=agg_predinst_counts['iter_round'],
                                        y=agg_predinst_counts['nEGID_cumm_rel'],
                                        name='nEGID count (rel total EGID in topo)',
                                        legendgroup='nEGID count',
                                        line=dict(color='black', width=2),
                                        mode='lines+markers', showlegend=True))


                fig.update_layout(
                    xaxis_title='Iteration Round',
                    yaxis_title='Mean (+/- 1 std)',
                    legend_title='Scenarios',
                    title = 'Agg. Cont. Charact. of Newly Installed Buildings per Iteration Round', 
                        uirevision='constant'  # Maintain the state of the plot when interacting
                )
                fig = self.add_scen_name_to_plot(fig, scen, self.visual_sett.pvalloc_scen_list[i_scen])

                if plot_show:
                    fig.show()
                if plot_export:
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_ind_lineband_contcharact_newinst.html')
                    else:
                        fig.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_lineband_contcharact_newinst.html')
