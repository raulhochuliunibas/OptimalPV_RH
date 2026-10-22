# python integrated packages
import sys
import os as os
import glob
import json

# external packages
import plotly.graph_objects as go
import polars as pl

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndLineProductionHOYPerEGID(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_line_productionHOY_per_EGID(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('plot_ind_line_productionHOY_per_EGID', self.visual_sett.log_name)

        grid_nodes_counts_minmax    = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['grid_nodes_counts_minmax']
        specific_gridnodes_egid_HOY = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['specific_gridnodes_egid_HOY']
        egid_col_to_plot            = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['egid_col_to_plot']
        grid_col_to_plot_tuples     = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['grid_col_to_plot_tuples']
        gridnode_pick_col_to_agg    = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['gridnode_pick_col_to_agg']
        egid_trace_opacity          = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['egid_trace_opacity']
        grid_trace_opacity          = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['grid_trace_opacity']

        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):

            # setup + import --------------------------
            if not self._set_mc_data_path_or_skip(scen):
                continue
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)

            topo = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
            topo_subdf_paths = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/topo_time_subdf/topo_subdf_*.parquet')
            outtopo_subdf_paths = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/outtopo_time_subdf/*.parquet')

            gridnode_df_import = pl.read_parquet(f'{self.visual_sett.mc_data_path}/gridnode_df.parquet')
            dsonodes_df = pl.read_parquet(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/dsonodes_df.parquet')  

            share_roof_area_available               = self.pvalloc_scen.TECspec_share_roof_area_available
            kWpeak_per_m2                           = self.pvalloc_scen.TECspec_kWpeak_per_m2
            TECspec_self_consumption_ifapplicable   = self.pvalloc_scen.TECspec_self_consumption_ifapplicable
            GRIDspec_perf_factor_1kVA_to_XkW        = self.pvalloc_scen.GRIDspec_perf_factor_1kVA_to_XkW
            print_checkpoint_statements             = False


            # select EGIDs by EGID - DF_UID combo --------------------------
            egid_list, dfuid_list, info_source_list, TotalPower_list, inst_TF_list, grid_node_list, share_pvprod_used_list, dfuidPower_list = [], [], [], [], [], [], [], []

            for k,v in topo.items():
                dfuid_tupls = [tpl[1] for tpl in v['pv_inst']['dfuid_w_inst_tuples'] if tpl[3] > 0.0]
                for k_s, v_s in v['solkat_partitions'].items():
                    if k_s in dfuid_tupls:
                        # for tpl in v['pv_inst']['dfuid_w_inst_tuples']:
                        for tpl in v['pv_inst']['dfuid_w_inst_tuples']:
                            if tpl[1] == k_s:       
                                egid_list.append(k)
                                info_source_list.append(v['pv_inst']['info_source'])
                                inst_TF_list.append(True if tpl[3] > 0.0 else False)   
                                dfuid_list.append(tpl[1])
                                share_pvprod_used_list.append(tpl[2])
                                dfuidPower_list.append(tpl[3])
                                grid_node_list.append(v['grid_node']) 
                                TotalPower_list.append(v['pv_inst']['TotalPower'])
                    else: 
                        egid_list.append(k)
                        dfuid_list.append(k_s)
                        share_pvprod_used_list.append(0.0)
                        dfuidPower_list.append(0.0)
                        info_source_list.append('')
                        inst_TF_list.append(False)
                        grid_node_list.append(v['grid_node'])  
                        TotalPower_list.append(0.0)          



            Map_pvinfo_topo_egid  = pl.DataFrame({'EGID': egid_list, 'df_uid': dfuid_list, 'info_source': info_source_list, 'TotalPower': TotalPower_list, 
                                                  'inst_TF': inst_TF_list, 'share_pvprod_used': share_pvprod_used_list, 'dfuidPower': dfuidPower_list, 
                                                  'grid_node': grid_node_list})
            gridnode_freq = Map_pvinfo_topo_egid.to_pandas()


            # select gridnodes / EGIDs to plot ----------
            if specific_gridnodes_egid_HOY != None: 
                gridnode_pick = specific_gridnodes_egid_HOY
            else: 
                gridnodes_counts =  gridnode_freq.groupby(['EGID', 'inst_TF' ]).agg({
                    # 'inst_TF': 'first',
                    'info_source': 'first',
                    'grid_node': 'first',})
                egid_gridnode_counts = gridnodes_counts['grid_node'].value_counts()
                egid_gridnode_counts.sort_values(ascending=False)
                gridnode_selec = egid_gridnode_counts[(egid_gridnode_counts >= grid_nodes_counts_minmax[0]) & (egid_gridnode_counts <= grid_nodes_counts_minmax[1]) ]

                if len(gridnode_selec) == 0:
                    egid_gridnode_counts.sort_values(ascending=True, inplace=True)
                    gridnode_pick = egid_gridnode_counts.index[0]  # take the smallest gridnode
                else:
                    gridnode_selec.sort_values(ascending=True, inplace=True)
                    gridnode_pick = gridnode_selec.index[0]


            Map_pvinfo_gridnode = gridnode_freq.loc[gridnode_freq['grid_node'] == gridnode_pick].copy()


            agg_subdf_updated_pvdf_list = []
            agg_subdf_df_list, agg_egids_list, agg_egids_all_list = [], [], []
            for i_path, path in enumerate(topo_subdf_paths):
                subdf = pl.read_parquet(path)   

                # extend subdf with static data (safe disk space) --------------------
                subdf_static = subdf['EGID'].unique().to_list()
                subdf_static_list = []
                for egid in subdf_static:
                    static_topo = topo[egid]

                    for k,v in static_topo['solkat_partitions'].items():
                        egid_dfuid_row = {
                            'EGID':               egid,
                            'df_uid':             k,
                            'bfs':                static_topo['gwr_info']['bfs'], 
                            'GKLAS':              static_topo['gwr_info']['gklas'],
                            'GAREA':              static_topo['gwr_info']['garea'],
                            'GBAUJ':              static_topo['gwr_info']['gbauj'],
                            'GSTAT':              static_topo['gwr_info']['gstat'],
                            'GWAERZH1':           static_topo['gwr_info']['gwaerzh1'],
                            'GENH1':              static_topo['gwr_info']['genh1'],
                            'sfhmfh_typ':         static_topo['gwr_info']['sfhmfh_typ'],
                            'demand_arch_typ':    static_topo['demand_arch_typ'],
                            'demand_elec_pGAREA': static_topo['demand_elec_pGAREA'],
                            'grid_node':          static_topo['grid_node'],
                            'pvtarif_Rp_kWh':     static_topo['pvtarif_Rp_kWh'],
                            'elecpri_Rp_kWh':     static_topo['elecpri_Rp_kWh'],
                            'inst_TF':            static_topo['pv_inst']['inst_TF'],
                            'info_source':        static_topo['pv_inst']['info_source'],
                            'pvid':               static_topo['pv_inst']['xtf_id'],
                            'TotalPower':         static_topo['pv_inst']['TotalPower'],
                            'FLAECHE':            v['FLAECHE'],
                            'AUSRICHTUNG':        v['AUSRICHTUNG'],
                            'STROMERTRAG':        v['STROMERTRAG'],
                            'NEIGUNG':            v['NEIGUNG'],
                            'MSTRAHLUNG':         v['MSTRAHLUNG'],
                            'GSTRAHLUNG':         v['GSTRAHLUNG'],
                        }
                        subdf_static_list.append(egid_dfuid_row)

                subdf_static_df = pl.DataFrame(subdf_static_list)
                subdf = subdf.join(subdf_static_df, on=['EGID', 'df_uid'], how='left')

                # Only plot EGIDs for 1 node: 
                subdf = subdf.filter(pl.col('grid_node') == gridnode_pick)  


                # taken 1:1 from algo_update_gridnode_AND_gridprem_POLARS() =============================================
                subdf_updated = subdf.clone()
                cols_to_update = [ 'info_source', 'inst_TF', 'share_pvprod_used', 'dfuidPower' ]                                      
                subdf_updated = subdf_updated.drop(cols_to_update)                      

                checkpoint_to_logfile('gridprem > subdf: start pandas.merge subdf w Map_infosource_egid', self.sett.log_name, 0, self.sett.show_debug_prints) if print_checkpoint_statements else None
                subdf_updated = subdf_updated.join(Map_pvinfo_topo_egid[['EGID', 'df_uid',] + cols_to_update], on=['EGID', 'df_uid'], how='left')         

                # remove the nulls from the merged columns
                subdf_updated = subdf_updated.with_columns([
                    pl.when(pl.col('inst_TF').is_null())
                        .then(False).otherwise(pl.col('inst_TF')).alias('inst_TF'),
                    pl.when(pl.col('info_source').is_null())
                        .then(pl.lit("")).otherwise(pl.col('info_source')).alias('info_source'),
                    pl.when(pl.col('share_pvprod_used').is_null())
                        .then(pl.lit(0.0)).otherwise(pl.col('share_pvprod_used')).alias('share_pvprod_used'),
                    pl.when(pl.col('dfuidPower').is_null())
                        .then(pl.lit(0.0)).otherwise(pl.col('dfuidPower')).alias('dfuidPower'),

                ])
                checkpoint_to_logfile('gridprem > subdf: end pandas.merge subdf w Map_infosource_egid', self.sett.log_name, 0, self.sett.show_debug_prints) if print_checkpoint_statements else None

                # pvprod_kW for non_inst EGIDs to 0 --------------------
                # Only consider production for houses that have built a pv installation and substract selfconsumption from the production
                # EDIT 1: Switched off, because we also need the demand of those houses per grid node (and non_inst houses can not feed in by design, so no probelm there)
                # subdf_updated = subdf_updated.filter(pl.col('inst_TF'))       
                # EDIT 2: Previous comment not fully true, pvprod_kW is calculated as the pvproduction potential for a df_uid partition, so those columns are also positive for
                # non-installed houses. SOLUTION: set all pvproduction to 0 for all df_uid that are not in Map_pvinfo_topo_egid (derived from most recent version of topo_egid)
                # EDIT 3: Because demand is per unit of EGID but the subdf data frame is by df_uid, demand_kW will be double counted when simply sumed up like other pvproduction relevant columns.
                # So I need to devide the demand on a house level by n_dfuid (with and / or without installations) so the actuall amount of feedin is calculated and not distorted by double counting. 
                # Two cases should solve all possible applications for this transformation
                # EDIT 4: Division as described in edit3 is working, problems persist with EGIDs with partial pv installations (only 1/2 df_uid installed). Then selfconsumption is only applied to 
                # 1/n-inst_share per EGID, which is too low. Another probably essential problem is, that demand is uniformly distributed over all df_uid, but pvprod is not. So even if a southfacing
                # partition could cover all demand, and another east/west facing partition could do feedin, the total self consumption would be underestimated ) 
                #       -> move selfconsumption part to a later stage of the compuation! agg production by [EGID,t] and only then apply selfconsumption. "demand-splitting" per partition no longer 
                #          needed, use demand.first() in aggregation

                # calculated pvprod_kW, given inst (also partial inst on partition possible)
                subdf_updated = subdf_updated.with_columns([
                    (pl.col('poss_pvprod_kW') * pl.col('share_pvprod_used')).alias('pvprod_kW'),
                ])

                # --------------------
                # sanity_check
                subdf_sanity_check = subdf_updated.filter(
                    (pl.col('t').is_in([
                    # 't_1', 't_2', 't_3', 't_4', 't_5', 't_6', 
                    # 't_7', 't_8', 't_9', 't_10', 't_11', 't_12', 
                    # 't_13', 't_14', 't_15', 't_16','t_17', 't_18', 
                    # 't_19', 't_20', 't_21', 't_22', 't_23', 't_24', 
                    't_14', 't_15', 
                                                    ])) &
                    ((pl.col('info_source') == 'pv_df') | 
                    (pl.col('info_source') == 'alloc_algorithm') )
                    # (pl.col('info_source') == 'alloc_algorithm')
                    ).select([
                        'EGID', 'df_uid', 't', 'inst_TF', 'TotalPower', 'info_source', 'dfuidPower', 'poss_pvprod_kW', 'pvprod_kW'
                        ]).head(50)
                subdf_sanity_check
                # sanity_check
                # --------------------

                # force pvprod == 0 for EGID-df_uid without inst
                Map_pvinst_topo_egid = Map_pvinfo_topo_egid.filter(pl.col('inst_TF') == True)
                subdf_no_inst = subdf_updated.join(
                    Map_pvinst_topo_egid[['EGID', 'df_uid']], 
                    on=['EGID', 'df_uid'], 
                    how='anti'
                )

                subdf_updated = subdf_updated.with_columns([
                    pl.when(
                        (pl.col('EGID').is_in(subdf_no_inst['EGID'].implode())) &
                        (pl.col('df_uid').is_in(subdf_no_inst['df_uid'].implode()))
                    ).then(pl.lit(0.0)).otherwise(pl.col('pvprod_kW')).alias('pvprod_kW'),
                ])
                subdf_updated = subdf_updated.with_columns([
                    pl.when(
                        (pl.col('EGID').is_in(subdf_no_inst['EGID'].implode())) &
                        (pl.col('df_uid').is_in(subdf_no_inst['df_uid'].implode()))
                    ).then(pl.lit(0.0)).otherwise(pl.col('radiation')).alias('radiation'),
                ])

                # sorting necessary so that .first() statement captures inst_TF and info_source for EGIDS with partial installations
                subdf_updated = subdf_updated.sort(['EGID','inst_TF', 'df_uid', 't_int'], descending=[False, True, False, False])

                # agg per EGID to apply selfconsumption 
                agg_egids = subdf_updated.group_by(['EGID', 't', 't_int']).agg([
                    pl.col('inst_TF').first().alias('inst_TF'),
                    pl.col('info_source').first().alias('info_source'),
                    pl.col('grid_node').first().alias('grid_node'),
                    pl.col('demand_kW').first().alias('demand_kW'),
                    pl.col('poss_pvprod_kW').sum().alias('poss_pvprod_kW'),
                    pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                    pl.col('radiation').sum().alias('radiation'),
                ])

                # calc selfconsumption
                agg_egids = agg_egids.sort(['EGID', 't_int'], descending = [False, False])

                selfconsum_expr = pl.min_horizontal([pl.col("pvprod_kW"), pl.col("demand_kW")]) * TECspec_self_consumption_ifapplicable

                agg_egids = agg_egids.with_columns([        
                    selfconsum_expr.alias("selfconsum_kW"),
                    (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
                    (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
                ])

                # (for visualization later) -----
                # only select egids for grid_node mentioned above
                agg_egids_all_list.append(agg_egids)

                agg_egids = agg_egids.filter(pl.col('EGID').is_in(Map_pvinfo_gridnode['EGID'].to_list()))
                agg_egids_list.append(agg_egids)
                # -----

                # agg per gridnode
                agg_subdf = agg_egids.group_by(['grid_node', 't', 't_int']).agg([
                pl.col('inst_TF').first().alias('inst_TF'),
                pl.col('demand_kW').sum().alias('demand_kW'),
                pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                pl.col('netdemand_kW').sum().alias('netdemand_kW'), 
                pl.col('radiation').sum().alias('radiation'),
                ])

                # agg subdf_updated_dfuid_pvdf for later export
                # (for pvallocation only)
                subdf_updated_dfuid_pvdf = subdf_updated.filter(pl.col('info_source') != '').clone()
                subdf_updated_dfuid_pvdf = subdf_updated_dfuid_pvdf.group_by(['EGID', 'df_uid',]).agg([
                    pl.col('inst_TF').first().alias('inst_TF'), 
                    pl.col('info_source').first().alias('info_source'), 
                    pl.col('pvid').first().alias('pvid'), 
                    pl.col('grid_node').first().alias('grid_node'),
                    pl.col('FLAECHE').first().alias('FLAECHE'), 
                    pl.col('MSTRAHLUNG').first().alias('MSTRAHLUNG'), 

                    pl.col('t').len().alias('n_t'),
                    pl.col('month').len().alias('n_month'), 

                    pl.col('demand_kW').sum().alias('sum_demand_kW'), 
                    pl.col('pvprod_kW').first().alias('sum_pvprod_kW'), 

                    pl.col('TotalPower').first().alias('TotalPower'), 
                    ])

                subdf_updated_dfuid_pvdf = subdf_updated_dfuid_pvdf.with_columns([
                    (kWpeak_per_m2 * share_roof_area_available * pl.col('FLAECHE')).alias('poss_prod_capa_kW_peak')
                ])
                # -----

                agg_subdf_df_list.append(agg_subdf)
                agg_subdf_updated_pvdf_list.append(subdf_updated_dfuid_pvdf)


            agg_subdf_df = pl.concat(agg_subdf_df_list)
            topo_gridnode_df = agg_subdf_df.group_by(['grid_node', 't', 't_int']).agg([
                pl.col('inst_TF').first().alias('inst_TF'),
                pl.col('demand_kW').sum().alias('demand_kW'),
                pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                pl.col('netdemand_kW').sum().alias('netdemand_kW'),
                pl.col('radiation').sum().alias('radiation'),
            ])

            # (for visualization later) -----
            # MAIN DF of this plot, all feedin TS by EGID
            topo_agg_egids_df = pl.concat(agg_egids_list)
            topo_agg_egids_df = topo_agg_egids_df.with_columns([
                pl.col('t').str.strip_chars('t_').cast(pl.Int64).alias('t_int'),
            ])
            topo_agg_egids_df = topo_agg_egids_df.sort("t_int", descending=False)
            # -----

            agg_subdf_updated_pvdf = pl.concat(agg_subdf_updated_pvdf_list)


            # import outtopo_time_subfs -----------------------------------------------------
            agg_subdf_df_list = []
            for i, path in enumerate(outtopo_subdf_paths):
                outsubdf = pl.read_parquet(path)  
                agg_outsubdf = outsubdf.group_by(['grid_node', 't']).agg([
                    pl.col('demand_proxy_out_kW').sum().alias('demand_proxy_out_kW'),
                ])
                del outsubdf
                agg_subdf_df_list.append(agg_outsubdf)

            agg_outsubdf_df = pl.concat(agg_subdf_df_list)
            outtopo_gridnode_df = agg_outsubdf_df.group_by(['grid_node', 't']).agg([
                pl.col('demand_proxy_out_kW').sum().alias('demand_proxy_out_kW'),
            ])


            # build gridnode_df -----------------------------------------------------
            checkpoint_to_logfile('gridprem: start merge to (out)topo_gridnode_df to gridnode_ts', self.sett.log_name, 0, self.sett.show_debug_prints) if print_checkpoint_statements else None   
            gridnode_df = topo_gridnode_df.join(outtopo_gridnode_df, on=['grid_node', 't'], how='left')
            gridnode_df = gridnode_df.with_columns([
                pl.col('t').str.strip_chars('t_').cast(pl.Int64).alias('t_int'),
            ])

            gridnode_df = gridnode_df.with_columns([
                pl.col('netfeedin_kW').alias('feedin_atnode_kW'), 
                (pl.col('netdemand_kW') + pl.col('demand_proxy_out_kW')).alias('demand_atnode_kW'),
            ])
            gridnode_df = gridnode_df.with_columns([
                pl.max_horizontal(['feedin_atnode_kW', 'demand_atnode_kW']).alias('max_demand_feedin_atnode_kW')
            ])

            # sanity check
            gridnode_df.group_by(['grid_node',]).agg([pl.len()])
            gridnode_df.group_by(['t',]).agg([pl.len()])

            # attach node thresholds 
            gridnode_df = gridnode_df.join(dsonodes_df[['grid_node', 'kVA_threshold']], on='grid_node', how='left')
            gridnode_df = gridnode_df.with_columns((pl.col("kVA_threshold") * GRIDspec_perf_factor_1kVA_to_XkW).alias("kW_threshold"))

            gridnode_df = gridnode_df.with_columns([
                pl.when(pl.col("max_demand_feedin_atnode_kW") < 0)
                .then(0.0)
                .otherwise(pl.col("max_demand_feedin_atnode_kW"))
                .alias("max_demand_feedin_atnode_kW"),
                ])
            gridnode_df = gridnode_df.with_columns([
                pl.when(pl.col("feedin_atnode_kW") > pl.col("kW_threshold"))
                .then(pl.col("kW_threshold"))
                .otherwise(pl.col("feedin_atnode_kW"))
                .alias("feedin_atnode_taken_kW"),
                ])
            gridnode_df = gridnode_df.with_columns([
                pl.when(pl.col("feedin_atnode_kW") > pl.col("kW_threshold"))
                .then(pl.col("feedin_atnode_kW") - pl.col("kW_threshold"))
                .otherwise(0.0)
                .alias("feedin_atnode_loss_kW")
            ])
            checkpoint_to_logfile('gridprem: end merge with gridnode_df + pl.when()', self.sett.log_name, 0, self.sett.show_debug_prints) if print_checkpoint_statements else None   


            # end 1:1 copy from pvallocation =============================================
            gridnode_pick_df = gridnode_df.clone()



            # plot --------------------------
            fig_egid_traces = go.Figure()
            fig_demand, fig_radiation, fig_pvprod, fig_selfconsum, fig_netdemand, fig_netfeedin, fig_feedin_atnode, fig_demand_atnode = go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()
            fig_dict = {
            'fig_demand_kW':     fig_demand     ,
            'fig_radiation':  fig_radiation  ,
            'fig_pvprod_kW':     fig_pvprod     ,
            'fig_selfconsum_kW': fig_selfconsum ,
            'fig_netdemand_kW':  fig_netdemand  ,
            'fig_netfeedin_kW':  fig_netfeedin  ,
            'fig_feedin_atnode_kW': fig_feedin_atnode,
            }
            gridnode_pick_df = gridnode_pick_df.to_pandas()
            topo_agg_egids_df = topo_agg_egids_df.to_pandas()

            # Stack traces EGIDs in picked gridnode ----------
            for col in egid_col_to_plot:
                fig_sub = fig_dict[f'fig_{col}']

                for egid in topo_agg_egids_df['EGID'].unique():
                    stack_subdf = topo_agg_egids_df.loc[topo_agg_egids_df['EGID'] == egid]
                    stack_subdf = stack_subdf.sort_values(by=['t_int'])

                    # reduce t_hours to have more "handable - plots" in plotly
                    if self.visual_sett.cut_timeseries_to_zoom_hour: 
                        stack_subdf = stack_subdf[
                            (stack_subdf['t_int'] >= self.visual_sett.default_zoom_hour[0] - (24 * 7)) &
                            (stack_subdf['t_int'] <= self.visual_sett.default_zoom_hour[1] + (24 * 7))
                        ].copy()                              

                    fig_sub.add_trace(go.Scatter(x=stack_subdf['t_int'], y=stack_subdf[col],
                                                mode='lines', 
                                                stackgroup= col, 
                                                name=f'EGID: {egid} - {col}', 
                                                hoverinfo='skip',
                                                opacity=egid_trace_opacity
                                                ))

                    fig_egid_traces.add_trace(go.Scatter(x=stack_subdf['t_int'], y=stack_subdf[col],
                                                mode='lines', 
                                                # stackgroup= col, 
                                                name=f'EGID: {egid} - {col}', 
                                                hoverinfo='skip',
                                                opacity=egid_trace_opacity
                                                ))

                fig_sub.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'topo_agg_egids_df from subdf {20*"-"}', opacity=0)) 
                fig_egid_traces.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'{20*"-"}', opacity=0)) 


            # gridnode_pick_df for comparison ----------
            for col in egid_col_to_plot:
                fig_sub = fig_dict[f'fig_{col}']

                for node in gridnode_pick_df['grid_node'].unique():
                    grid_col_agg_tuples = [t[0] for t in grid_col_to_plot_tuples]
                    node_subdf = gridnode_pick_df.loc[gridnode_pick_df['grid_node'] == node]
                    node_subdf = node_subdf.sort_values(by=['t_int'])   

                    # reduce t_hours to have more "handable - plots" in plotly
                    if self.visual_sett.cut_timeseries_to_zoom_hour: 
                        node_subdf = node_subdf[
                            (node_subdf['t_int'] >= self.visual_sett.default_zoom_hour[0] - (24 * 7)) &
                            (node_subdf['t_int'] <= self.visual_sett.default_zoom_hour[1] + (24 * 7))
                        ].copy()

                    # add aggregated traces 
                    if col in grid_col_agg_tuples:
                        agg_columns = [t[1] for t in grid_col_to_plot_tuples if t[0] == col]
                        for agg_col in agg_columns:
                            fig_sub.add_trace(go.Scatter(x=node_subdf['t_int'], y=node_subdf[agg_col],
                                                        name=f'grid_node: {node} - {agg_col} (agg)', 
                                                        mode='lines+markers', 
                                                        marker=dict(symbol='cross', ), 
                                                        hoverinfo='skip',
                                                        opacity=grid_trace_opacity
                                                        ))
                    # add col trace
                    fig_sub.add_trace(go.Scatter(x=node_subdf['t_int'], y=node_subdf[col],
                        mode='lines+markers', name=f'grid_node: {node} - {col}', 
                        marker=dict(symbol='cross', ), 
                        hoverinfo='skip',
                        opacity=grid_trace_opacity
                        ))
                fig_sub.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'gridnode_pick_df from subdf {20*"-"}', opacity=0))


            # # Sanity Check: picked node from gridnode_df +  ----------
            # # add values form gridnode_df to make in-plot sanity check
            for col in egid_col_to_plot:
                fig_sub = fig_dict[f'fig_{col}']

                for node in gridnode_pick_df['grid_node'].unique():
                    grid_col_agg_tuples = [t[0] for t in grid_col_to_plot_tuples]
                    gridnode_subdf = gridnode_df_import.filter(pl.col('grid_node') == node)
                    gridnode_subdf = gridnode_subdf.sort("t_int", descending=False)

                    # reduce t_hours to have more "handable - plots" in plotly
                    if self.visual_sett.cut_timeseries_to_zoom_hour: 
                        gridnode_subdf = gridnode_subdf.filter(
                            (pl.col('t_int') >= self.visual_sett.default_zoom_hour[0]-(24*7)) &
                            (pl.col('t_int') <= self.visual_sett.default_zoom_hour[1]+(24*7))
                        ).clone()  

                    gridnode_subdf = gridnode_subdf.to_pandas()

                    if col in grid_col_agg_tuples:
                        agg_columns = [t[1] for t in grid_col_to_plot_tuples if t[0] == col]
                        for agg_col in agg_columns:
                            fig_sub.add_trace(go.Scatter(x=gridnode_subdf['t_int'], y=gridnode_subdf[agg_col],
                                        name=f'grid_node: {node} - {agg_col} (gridnode_df)',
                                        mode='lines+markers', 
                                        marker=dict(symbol='cross', ), 
                                        hoverinfo='skip',
                                        opacity=grid_trace_opacity
                                        ))
                    fig_sub.add_trace(go.Scatter(x=gridnode_subdf['t_int'], y=gridnode_subdf[col],
                                            name=f'grid_node: {node} - {col} (gridnode_df)',
                                            mode='lines+markers', 
                                            hoverinfo='skip',
                                            opacity=grid_trace_opacity
                                            ))
                fig_sub.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'picked node from gridnode_df {20*"-"}', opacity=0))
            # ----------


            # plot fig_netfeedin_ALL ----------
            # gridnode_pick_df['netfeedin_all_kW+demand_proxy_out_kW'] = gridnode_pick_df['netfeedin_all_kW'] + gridnode_pick_df['demand_proxy_out_kW']

            for node in gridnode_pick_df['grid_node'].unique():
                node_subdf = gridnode_pick_df.loc[gridnode_pick_df['grid_node'] == node]
                node_subdf = node_subdf.sort_values(by=['t_int'])   

                # reduce t_hours to have more "handable - plots" in plotly
                if self.visual_sett.cut_timeseries_to_zoom_hour: 
                    node_subdf = node_subdf[
                        (node_subdf['t_int'] >= self.visual_sett.default_zoom_hour[0] - (24 * 7)) &
                        (node_subdf['t_int'] <= self.visual_sett.default_zoom_hour[1] + (24 * 7))
                    ].copy()

                fig_feedin_atnode.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'gridnode_pick_df from subdf {20*"-"}', opacity=0))
                for agg_col in gridnode_pick_col_to_agg:    
                    fig_feedin_atnode.add_trace(go.Scatter(x=node_subdf['t_int'], y=node_subdf[agg_col],           
                                                           name= f'grid_node: {node} - {agg_col} (agg subdf)',             
                                                           mode = 'lines+markers', 
                                                           hoverinfo = 'skip', 
                                                           opacity= grid_trace_opacity))

                gridnode_subdf = gridnode_df_import.filter(pl.col('grid_node') == node)
                gridnode_subdf = gridnode_subdf.sort("t_int", descending=False)
                gridnode_subdf = gridnode_subdf.to_pandas()

                # reduce t_hours to have more "handable - plots" in plotly
                if self.visual_sett.cut_timeseries_to_zoom_hour: 
                    gridnode_subdf = gridnode_subdf[
                        (gridnode_subdf['t_int'] >= self.visual_sett.default_zoom_hour[0] - (24 * 7)) &
                        (gridnode_subdf['t_int'] <= self.visual_sett.default_zoom_hour[1] + (24 * 7))
                    ].copy()


                fig_feedin_atnode.add_trace(go.Scatter(x=[None, ], y=[None, ], mode='lines', name=f'picked node from gridnode_df {20*"-"}', opacity=0))
                for agg_col in gridnode_pick_col_to_agg:    
                    fig_feedin_atnode.add_trace(go.Scatter(x=gridnode_subdf['t_int'], y=gridnode_subdf[agg_col],
                        name=f'grid_node: {node} - {agg_col} (gridnode_df)',
                        mode='lines+markers', 
                        hoverinfo='skip',
                        opacity=grid_trace_opacity
                        ))  


            # export plot --------------------------
            for k,v in fig_dict.items():
                fig_sub = v
                col_name = k.split('fig_')[1]

                fig_sub.update_layout(
                    title = f'Production per EGID / Grid Node (col: {col_name}, Hour of Year)', 
                    xaxis_title='Hour of Year',
                    yaxis_title='Production / Feedin (kW)',
                    legend_title='EGID / Node ID',
                    template='plotly_white',
                    showlegend=True,
                )
                fig_sub = self.add_scen_name_to_plot(fig_sub, scen, self.pvalloc_scen)
                fig_sub = self.set_default_fig_zoom_hour(fig_sub, self.visual_sett.default_zoom_hour)

                if False: # never apply day_night_bands to these plots because it takes much too long  - self.visual_sett.add_day_night_HOY_bands:
                    fig_sub = self.add_day_night_bands_HOY_plot(fig_sub, gridnode_subdf['t_int'])

                if plot_show:
                    fig_sub.show()
                if plot_export:
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig_sub.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_ind_line_productionHOY_per_EGID_inclEGID_{col_name}.html')
                    else:
                        fig_sub.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_line_productionHOY_per_EGID_inclEGID_{col_name}.html')


            # export fig_egid_traces --------------------------
            fig_egid_traces.update_layout(
                title = 'Sanity Check Dashboard all Time Series per EGID', 
                xaxis_title='Hour of Year',
                yaxis_title='Production / Feedin (kW)',
                template='plotly_white',
                showlegend=True,
                )
            fig_egid_traces = self.add_scen_name_to_plot(fig_egid_traces, scen, self.pvalloc_scen)
            fig_egid_traces = self.set_default_fig_zoom_hour(fig_egid_traces, self.visual_sett.default_zoom_hour)

            if plot_show:
                fig_egid_traces.show()
            if plot_export:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig_egid_traces.write_html(f'{self.visual_sett.visual_path}/{scen}/plot_ind_line_productionHOY_per_EGID___{scen}.html')
                else:
                    fig_sub.write_html(f'{self.visual_sett.visual_path}/plot_ind_line_productionHOY_per_EGID___{scen}.html')

                print_to_logfile(f'\texport: plot_ind_line_productionHOY_per_EGID.html (for: {scen})', self.visual_sett.log_name)
