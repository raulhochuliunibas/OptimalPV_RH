# python integrated packages
import sys
import os as os
import json

# external packages
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndSummaryStatsByNode(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_summary_stats_by_node(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('plot_ind_summary_stats_by_node', self.visual_sett.log_name)

        # "settings" ----------
        n_cols_cont = 1
        node_summary_list = self.visual_sett.plot_ind_summary_stats_by_node_specs['node_summary_list']
        cont_col_plot     = self.visual_sett.plot_ind_summary_stats_by_node_specs['cont_col_plot']


        # color dict ----------
        if True: 
            global_color_rows = []
            for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
                self.get_pvalloc_sett_output(pvalloc_scen_name = scen)
                if not self._set_mc_data_path_or_skip(scen):
                    continue

                # import
                topo         = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))

                # compute topo_df
                for k,v in topo.items():

                    partitions = v.get('solkat_partitions', {})
                    gwr_info = v.get('gwr_info', {})
                    pv_inst = v.get('pv_inst', {})

                    for k_p, v_p in partitions.items():
                        row = {
                            'EGID': k,
                            'df_uid': k_p,
                            'bfs': gwr_info.get('bfs'),
                            'GKLAS': gwr_info.get('gklas'),
                            'GSTAT': gwr_info.get('gstat'),
                            'GAREA': gwr_info.get('garea'),
                            'GBAUJ': gwr_info.get('gbauj'),
                            'GWAERZH1': gwr_info.get('gwaerzh1'),
                            'are_typ': gwr_info.get('are_typ'),
                            'sfhmfh_typ': gwr_info.get('sfhmfh_typ'),
                            'demand_arch_typ': v.get('demand_arch_typ'),
                            'demand_elec_dem_pGAREA': v.get('demand_elec_dem_pGAREA'),
                            'grid_node': v.get('grid_node'),

                            'inst_TF': pv_inst.get('inst_TF'),
                            'info_source': pv_inst.get('info_source'),
                            'pvid': pv_inst.get('xtf_id'),
                            'pvtarif_Rp_kWh': v.get('pvtarif_Rp_kWh'),
                            'TotalPower': pv_inst.get('TotalPower'),
                            'df_uid_w_inst': v.get('df_uid_w_inst'),

                            'FLAECHE': v_p.get('FLAECHE'),
                            'AUSRICHTUNG': v_p.get('AUSRICHTUNG'),
                            'STROMERTRAG': v_p.get('STROMERTRAG'),
                            'NEIGUNG': v_p.get('NEIGUNG'),
                            'MSTRAHLUNG': v_p.get('MSTRAHLUNG'),
                            'elecpri_Rp_kWh': v.get('elecpri_Rp_kWh'),
                        }
                        global_color_rows.append(row)

            topo_df_catg_raw = pl.DataFrame(global_color_rows)
            topo_df_cont_raw = pl.DataFrame(global_color_rows)


            # make color mapping CONT ----------
            topo_df_cont = topo_df_cont_raw.group_by('EGID').agg([
                pl.col('GAREA').sum().alias('GAREA'),
                pl.col('FLAECHE').sum().alias('FLAECHE'),
                pl.col('AUSRICHTUNG').mean().alias('AUSRICHTUNG_mean'),
            ])


            color_map_cont = {
                col: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                for i, col in enumerate(cont_col_plot)
            }




        # Plots ----------------------------------------
        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            if not self._set_mc_data_path_or_skip(scen):
                continue

            # import
            topo         = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
            gridnode_df = pl.read_parquet(f'{self.visual_sett.mc_data_path}/gridnode_df.parquet')


            # compute topo_df
            rows = []

            for k,v in topo.items():

                partitions = v.get('solkat_partitions', {})
                gwr_info = v.get('gwr_info', {})
                pv_inst = v.get('pv_inst', {})

                for k_p, v_p in partitions.items():
                    row = {
                        'EGID': k,
                        'df_uid': k_p,
                        'bfs': gwr_info.get('bfs'),
                        'GKLAS': gwr_info.get('gklas'),
                        'GSTAT': gwr_info.get('gstat'),
                        'GAREA': gwr_info.get('garea'),
                        'GBAUJ': gwr_info.get('gbauj'),
                        'are_typ': gwr_info.get('are_typ'),
                        'sfhmfh_typ': gwr_info.get('sfhmfh_typ'),
                        'demand_arch_typ': v.get('demand_arch_typ'),
                        'demand_elec_dem_pGAREA': v.get('demand_elec_dem_pGAREA'),
                        'grid_node': v.get('grid_node'),

                        'inst_TF': pv_inst.get('inst_TF'),
                        'info_source': pv_inst.get('info_source'),
                        'pvid': pv_inst.get('xtf_id'),
                        'pvtarif_Rp_kWh': v.get('pvtarif_Rp_kWh'),
                        'TotalPower': pv_inst.get('TotalPower'),
                        'df_uid_w_inst': v.get('df_uid_w_inst'),

                        'FLAECHE': v_p.get('FLAECHE'),
                        'AUSRICHTUNG': v_p.get('AUSRICHTUNG'),
                        'STROMERTRAG': v_p.get('STROMERTRAG'),
                        'NEIGUNG': v_p.get('NEIGUNG'),
                        'MSTRAHLUNG': v_p.get('MSTRAHLUNG'),
                        'elecpri_Rp_kWh': v.get('elecpri_Rp_kWh'),
                    }
                    rows.append(row)


            topo_df_cont_raw = pl.DataFrame(rows)


            # transform / adjust cols
            topo_df_cont_raw = topo_df_cont_raw.with_columns([
                (pl.col('AUSRICHTUNG')).alias('AUSRICHTUNG_original'),
                (pl.col('AUSRICHTUNG') * pl.col('FLAECHE')).alias('AUSRICHTUNG_FLAECHE'),
            ])

            topo_df_cont_agg = topo_df_cont_raw.group_by('EGID').agg([
                pl.col('grid_node').first().alias('grid_node'),
                pl.col('info_source').first().alias('info_source'),
                pl.col('FLAECHE').sum().alias('FLAECHE'),
                pl.col('GAREA').sum().alias('GAREA'),
                pl.col('AUSRICHTUNG_FLAECHE').sum().alias('AUSRICHTUNG_FLAECHE'),
            ])

            # add pv_df inst / nEGID ratio
            gridnode_infosource_count = topo_df_cont_agg.group_by(['grid_node', 'info_source']).agg([
                pl.col('info_source').count().alias('n_pv_inst'),
            ]).filter(pl.col('info_source') == 'pv_df')
            gridnode_egid_count = topo_df_cont_agg.group_by('grid_node').agg([
                pl.col('EGID').count().alias('nEGID_gridnode')
            ])
            pvdf_inst_ratio = gridnode_infosource_count.join(gridnode_egid_count, on = 'grid_node', how = 'left')
            pvdf_inst_ratio = pvdf_inst_ratio.with_columns([
                (pl.col('n_pv_inst') / pl.col('nEGID_gridnode')).alias('pvdf_inst_ratio')
            ]).select(['grid_node', 'pvdf_inst_ratio'])
            # topo_df_cont_agg = topo_df_cont_agg.join(pvdf_inst_ratio, on = 'grid_node', how = 'left')

            # check_nodes = ['524', '905', '10', '26', '399']
            # pvdf_inst_ratio.filter(pl.col('grid_node').is_in(check_nodes))



            # plots ----------------------------------------
            # plot for 5 nodes with most loss
            if len(node_summary_list)==0: 
                gridnode_df_any_loss = gridnode_df.filter(pl.col('feedin_atnode_loss_kW') > 0)
                if gridnode_df_any_loss.shape[0] > 0:
                    gridnode_df_top_loss = gridnode_df_any_loss.group_by('grid_node').agg([
                        pl.col('feedin_atnode_loss_kW').sum().alias('feedin_atnode_loss_kW')
                    ]).sort('feedin_atnode_loss_kW', descending=True).head(5).select('grid_node')
                    compare_nodes_list = gridnode_df_top_loss['grid_node'].to_list()
                else: 
                    compare_nodes_list = pvdf_inst_ratio.sort('pvdf_inst_ratio', descending=True).head(1)['grid_node'].to_list()

            else: 
                compare_nodes_list = node_summary_list

            # node with most pv, but not single-EGID-node
            EGID_node_map = topo_df_cont_raw.group_by('EGID').agg([
                pl.col('grid_node').first().alias('grid_node'),
            ])
            node_EGID_count = EGID_node_map.group_by('grid_node').agg([
                pl.col('EGID').count().alias('n_EGID')
            ])
            pvdf_inst_ratio = pvdf_inst_ratio.join(node_EGID_count, on = 'grid_node', how = 'left')

            pvdf_inst_ratio.filter(pl.col('n_EGID') > 0).sort('pvdf_inst_ratio', descending=True).head(10)
            max_pvinst_2nEGID = max( pvdf_inst_ratio.filter(pl.col('n_EGID') > 1).select('pvdf_inst_ratio').to_series().to_list())

            max_pvinst_nodes_2nEGID = pvdf_inst_ratio.filter(pl.col('pvdf_inst_ratio') == max_pvinst_2nEGID).head(5).select('grid_node').to_series().to_list()
            max_pvinst_node         = pvdf_inst_ratio.filter(pl.col('n_EGID') > 1).sort('pvdf_inst_ratio', descending=True).head(1).select('grid_node').to_series().to_list()[0]

            # create Plots 
            n_rows_cont = int(np.ceil(len(compare_nodes_list) / n_cols_cont))
            fig = make_subplots(rows=n_rows_cont, cols=n_cols_cont,
                                                subplot_titles=[f'node {node}' for node in compare_nodes_list],)
            fig_bar = go.Figure()

            for i_node, node in enumerate(compare_nodes_list): 

                for i_col, col in enumerate(cont_col_plot):

                    if 'original' in col: 
                        df_plot = topo_df_cont_raw.filter(pl.col('grid_node')==node).to_pandas()
                    elif 'pvdf_inst_ratio' in col:
                        df_plot = pvdf_inst_ratio.filter(pl.col('grid_node')==node).to_pandas()
                    else:
                        df_plot = topo_df_cont_agg.filter(pl.col('grid_node')==node).to_pandas()

                    if 'pvdf_inst_ratio' in col: 
                        fig.add_trace(
                            go.Histogram(
                                x=[0,],
                                name=f"node {node:4} – {col:15}: {df_plot[col].values[0] * 100:.2f}%",
                                marker=dict(color=color_map_cont[col]),
                                legendgroup=col,
                                showlegend=True,
                                nbinsx=30,
                                opacity=0.6,
                                texttemplate="%{y}",
                            ),
                            row = (i_node // n_cols_cont) + 1, 
                            col = (i_node % n_cols_cont) + 1,
                        )

                        node_loss_df = gridnode_df.group_by('grid_node').agg([
                                        pl.col('feedin_atnode_loss_kW').sum().alias('feedin_atnode_loss_kW')
                                        ]).sort('feedin_atnode_loss_kW', descending=True).select(['grid_node', 'feedin_atnode_loss_kW'])
                        node_loss = node_loss_df.filter(pl.col('grid_node')==node).select('feedin_atnode_loss_kW').to_numpy()[0][0]
                        n_EGID_reg = pvdf_inst_ratio.filter(pl.col('grid_node')==node).select('n_EGID').to_numpy()[0][0]

                        fig_bar.add_trace(
                            go.Bar(
                                x=[f"node {node} - loss"],
                                y=[node_loss / 1000000],
                                name=f"node {node} – loss /1e6 (nEGID: {n_EGID_reg})",
                                # marker=dict(color=color_map_cont[col]),
                                # legendgroup=col,
                                showlegend=True,
                                text=[f"{node_loss / 1000000:.2f}"],
                        ))
                        fig_bar.add_trace(
                            go.Bar(
                                x=[f"node {node} - pvratio"],
                                y=[df_plot[col].values[0]],
                                name=f"node {node} – {col}",
                                # marker=dict(color=color_map_cont[col]),
                                # legendgroup=col,
                                showlegend=True,
                                text=[f"{df_plot[col].values[0] * 100:.2f}%"],
                        ))


                        # add node with max pv ratio as comparison to max loss nodes 
                        if i_node == len(compare_nodes_list) -1:
                            for max_node in max_pvinst_nodes_2nEGID: 
                                n_EGID = pvdf_inst_ratio.filter(pl.col('grid_node')==max_node).select('n_EGID').to_numpy()[0][0]
                                df_plot = pvdf_inst_ratio.filter(pl.col('grid_node')==max_node).to_pandas()
                                node_loss2 = node_loss_df.filter(pl.col('grid_node')==max_node).select('feedin_atnode_loss_kW').to_numpy()[0][0]
                                fig_bar.add_trace(
                                    go.Bar(
                                        x=[f"max pvratio node {max_node} - loss"],
                                        y=[node_loss2 / 1000000],
                                        name=f"node {max_node} – loss /1e6, min 2nEGID (nEGID: {n_EGID})",
                                        # marker=dict(color=color_map_cont[col]),
                                        # legendgroup=col,
                                        showlegend=True,
                                        text=[f"{node_loss2 / 1000000:.2f}"],
                                ))
                                fig_bar.add_trace(
                                    go.Bar(
                                        x=[f"max pvratio node {max_node} - pvratio"],
                                        y=[df_plot[col].values[0]],
                                        name=f"node {max_node} – {col}",
                                        # marker=dict(color=color_map_cont[col]),
                                        # legendgroup=col,
                                        showlegend=True,
                                        text=[f"{df_plot[col].values[0] * 100:.2f}%"],
                                ))

                    else:
                        fig.add_trace(
                            go.Histogram(
                                x=df_plot[col],
                                name=f"node {node} – {col}",
                                marker=dict(color=color_map_cont[col]),
                                legendgroup=col,
                                showlegend=True,
                                nbinsx=30,
                                opacity=0.6,
                                texttemplate="%{y}",
                                hovertemplate=(
                                    f"Scenario: {scen}<br>"
                                    f"Column: {col}<br>"
                                    "Value: %{x}<br>"
                                    "Count: %{y}<extra></extra>"
                                )
                            ),
                            row = (i_node // n_cols_cont) + 1, 
                            col = (i_node % n_cols_cont) + 1,
                        )


            # update + export ----------------------------------------
            fig.update_layout(
                title = f'Comparison EGID charact. by Node (scen: {scen})', 
                barmode='overlay', 
                template='plotly_white',
            )
            fig_bar.update_layout(
                title = f'PVdf Inst Ratio and Node Loss (scen: {scen})', 
                barmode='group', 
                template='plotly_white',
            )

            if plot_show:
                fig.show()
                fig_bar.show()
            if plot_export and not self.visual_sett.export_only_agg_comparison_plots:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig.write_html(f'{self.visual_sett.visual_path}/{scen}/plot_ind_summary_stats_by_node___{scen}.html')
                    fig_bar.write_html(f'{self.visual_sett.visual_path}/{scen}/plot_ind_summary_stats_by_node_pvshare_bar___{scen}.html')
                else:
                    fig.write_html(f'{self.visual_sett.visual_path}/plot_ind_summary_stats_by_node___{scen}.html')
                    fig_bar.write_html(f'{self.visual_sett.visual_path}/plot_ind_summary_stats_by_node_pvshare_bar___{scen}.html')
                print_to_logfile(f'\texport: plot_ind_summary_stats_by_node (for: {scen})', self.visual_sett.log_name)
