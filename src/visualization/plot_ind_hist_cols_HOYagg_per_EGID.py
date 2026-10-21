# python integrated packages
import sys
import os as os
import glob
import json

# external packages
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import polars as pl
from scipy.stats import gaussian_kde

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndHistColsHOYaggPerEGID(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_hist_cols_HOYagg_per_EGID(self, plot_show: bool = True, plot_export: bool = True):

        checkpoint_to_logfile('plot_ind_hist_cols_HOYagg_per_EGID', self.visual_sett.log_name)
        trace_color_dict = {
            'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
            'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
            'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Plotly3': pc.sequential.Plotly3,
            'Viridis': pc.sequential.Viridis, 'Turbo': pc.sequential.Turbo, 'Blackbody': pc.sequential.Blackbody, 
            'Bluered': pc.sequential.Bluered, 'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
            'Rainbow': pc.sequential.Rainbow, 'Burg': pc.sequential.Burg, 'Cividis': pc.sequential.Cividis,
            'Inferno': pc.sequential.Inferno, 'Magma': pc.sequential.Magma, 'Plasma': pc.sequential.Plasma,
        }     
        fig_agg_color_palettes = ['Mint', 'Oranges', 'Greys',  'Burg',  'Blues', 'Greens', 'Reds',
                                  'Mint', 'Oranges', 'Greys',  'Burg',  'Blues', 'Greens', 'Reds',
                                    ]

        grid_nodes_counts_minmax    = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['grid_nodes_counts_minmax']
        specific_gridnodes_egid_HOY = self.visual_sett.plot_ind_line_productionHOY_per_EGID_specs['specific_gridnodes_egid_HOY']

        # hist_n_bins                 = self.visual_sett.plot_ind_hist_cols_HOYagg_per_EGID_specs ['hist_n_bins']
        hist_cols_to_plot           = self.visual_sett.plot_ind_hist_cols_HOYagg_per_EGID_specs ['hist_cols_to_plot']
        n_hist_bins_bycol           = self.visual_sett.plot_ind_hist_cols_HOYagg_per_EGID_specs ['n_hist_bins_bycol']
        hist_opacity                = self.visual_sett.plot_ind_hist_cols_HOYagg_per_EGID_specs ['hist_opacity']

        fig_hist_agg = go.Figure()

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

            # data calculation from plot_ind_line_productionHOY_per_EGID ----------------------------------------
            if True:
                # select EGIDs by EGID - DF_UID combo --------------------------
                egid_list, dfuid_list, info_source_list, TotalPower_list, pvid_list, inst_TF_list, grid_node_list, share_pvprod_used_list, dfuidPower_list, FLAECHE_list, STROMERTRAG_list, MSTRAHLUNG_list = [], [], [], [], [], [], [], [], [], [], [], []
                for k,v in topo.items():                
                    dfuid_tupls = [tpl[1] for tpl in v['pv_inst']['dfuid_w_inst_tuples'] if tpl[3] > 0.0]
                    for k_s, v_s in v['solkat_partitions'].items():
                        if k_s in dfuid_tupls:
                            for tpl in v['pv_inst']['dfuid_w_inst_tuples']:
                                if tpl[1] == k_s:       
                                    egid_list.append(k)
                                    info_source_list.append(v['pv_inst']['info_source'])
                                    pvid_list.append(v['pv_inst']['xtf_id'])
                                    inst_TF_list.append(True if tpl[3] > 0.0 else False)   
                                    dfuid_list.append(tpl[1])
                                    share_pvprod_used_list.append(tpl[2])
                                    dfuidPower_list.append(tpl[3])
                                    FLAECHE_list.append(v_s['FLAECHE'])
                                    STROMERTRAG_list.append(v_s['STROMERTRAG'])
                                    MSTRAHLUNG_list.append(v_s['MSTRAHLUNG'])
                                    grid_node_list.append(v['grid_node']) 
                                    TotalPower_list.append(v['pv_inst']['TotalPower'])
                        else: 
                            egid_list.append(k)
                            info_source_list.append('')
                            inst_TF_list.append(False)
                            pvid_list.append('')
                            dfuid_list.append(k_s)
                            share_pvprod_used_list.append(0.0)
                            dfuidPower_list.append(0.0)
                            FLAECHE_list.append(0.0)
                            STROMERTRAG_list.append(0.0)
                            MSTRAHLUNG_list.append(0.0)
                            grid_node_list.append(v['grid_node'])  
                            TotalPower_list.append(0.0)          


                Map_pvinfo_topo_egid  = pl.DataFrame({'EGID': egid_list, 'df_uid': dfuid_list, 'info_source': info_source_list, 'TotalPower': TotalPower_list, 'pvid': pvid_list,
                                                    'inst_TF': inst_TF_list, 'share_pvprod_used': share_pvprod_used_list, 'dfuidPower': dfuidPower_list, 
                                                    'grid_node': grid_node_list, 'FLAECHE': FLAECHE_list, 'STROMERTRAG': STROMERTRAG_list, 'MSTRAHLUNG': MSTRAHLUNG_list})
                gridnode_freq = Map_pvinfo_topo_egid.to_pandas()

                # select gridnodes / EGIDs to plot ----------
                if specific_gridnodes_egid_HOY != None: 
                    gridnode_pick = specific_gridnodes_egid_HOY
                else: 
                    gridnodes_counts =  gridnode_freq.groupby(['EGID', ]).agg({
                        'inst_TF': 'first',
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

                    # Only plot EGIDs for 1 node: 
                    # subdf f= subdf.filter(pl.col('grid_node') == gridnode_pick)  

                    # Match missing topo data to subdf 
                    subdf = subdf.join(Map_pvinfo_topo_egid.select(['EGID', 'df_uid', 'grid_node', 'pvid', 'MSTRAHLUNG', 'FLAECHE']), 
                                      on=['EGID', 'df_uid'], how='left')

                    # taken 1:1 from algo_update_gridnode_AND_gridprem_POLARS() =============================================
                    subdf_updated = subdf.clone()
                    cols_to_update = [ 'info_source', 'inst_TF', 'share_pvprod_used', 'dfuidPower' ]                                      

                    subdf_updated = subdf_updated.drop(cols_to_update, strict = False)  # drop cols to update if they already exist in subdf to avoid duplicate columns after merge

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
                    # sanitycheck
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
                            'EGID', 'df_uid', 't', 'inst_TF', 'info_source', 'dfuidPower', 'poss_pvprod_kW', 'pvprod_kW'
                            ]).head(50)
                    subdf_sanity_check
                    # sanitycheck
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

                        pl.col('dfuidPower').first().alias('dfuidPower'), 
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


                # MAIN DF of this plot, all feedin TS by EGID
                topo_agg_egids_all_df = pl.concat(agg_egids_all_list)
                topo_agg_egids_all_df = topo_agg_egids_all_df.sort(['EGID', 't_int'], descending = [False, False])


                gridnode_pick_df = gridnode_pick_df.to_pandas()
                topo_agg_egids_all_df = topo_agg_egids_all_df.to_pandas()
                topo_agg_egids_all_df = topo_agg_egids_all_df.sort_values(by=['EGID', 't_int'], ascending=[True, True]).copy()





            # plot hist ============================
            if True:
                fig_hist = go.Figure()

                # aggregate cols ALL EGID over year
                topo_agg_egids_hist = topo_agg_egids_all_df.groupby(['EGID',]).agg({
                    'inst_TF'       : 'first',
                    'info_source'   : 'first',
                    'demand_kW'     : 'sum' ,
                    'radiation'     : 'sum' ,
                    'pvprod_kW'     : 'sum' ,
                    'selfconsum_kW' : 'sum' ,
                    'netdemand_kW'  : 'sum' ,
                    'netfeedin_kW'  : 'sum' ,
                }).reset_index()
                topo_agg_egids_hist['selfconsum_pvprod_ratio'] = np.zeros(len(topo_agg_egids_hist), dtype=float)
                topo_agg_egids_hist.loc[topo_agg_egids_hist['inst_TF'] == True, 'selfconsum_pvprod_ratio'] = topo_agg_egids_hist['selfconsum_kW'] / topo_agg_egids_hist['pvprod_kW']

                # subset all w_inst EGIDs
                # topo_agg_egids_pvdf_hist = topo_agg_egids_hist.loc[topo_agg_egids_hist['info_source'] == 'pv_df'].copy()
                # topo_agg_egids_alloc_hist = topo_agg_egids_hist.loc[topo_agg_egids_hist['info_source'] == 'alloc_algorithm'].copy()
                topo_agg_egids_winst_hist = topo_agg_egids_hist.loc[topo_agg_egids_hist['inst_TF'] == True].copy()



                # set coloring         
                subset_trace_color_palette      = trace_color_dict[self.visual_sett.plot_ind_hist_cols_HOYagg_per_EGID_specs['subset_trace_color_palette']]
                agg_subset_color_palette        = trace_color_dict[fig_agg_color_palettes[i_scen]]

                all_egid_trace_color_palette    = trace_color_dict[self.visual_sett.plot_ind_hist_cols_HOYagg_per_EGID_specs['all_egid_trace_color_palette']]
                agg_allegid_color_palette       = trace_color_dict[fig_agg_color_palettes[-i_scen]]



                # add trace func 
                def add_hist_cols_trace(figfunc, hist_df, col_func, col_NAME, mean_line_dash, color_func, ):

                    # Clean data to avoid inf/nan issues
                    clean_data = hist_df[col_func].replace([np.inf, -np.inf], np.nan).dropna()

                    # adjust binsize
                    bin_edges = np.linspace(clean_data.min(), clean_data.max(), 100 + 1)

                    figfunc.add_trace(go.Histogram(
                        x=hist_df[col_func],
                        name=f'hist: {col_func} {col_NAME}',
                        histnorm='percent',
                        opacity=hist_opacity-0.2,
                        marker_color= color_func, 
                        xbins=dict(
                            start=bin_edges[0],
                            end=bin_edges[-1],
                            size=bin_edges[1] - bin_edges[0]  # Calculate the bin size from the edges
                            )
                        ))

                    return figfunc

                def add_kde_cols_trace(figfunc, hist_df, col_func, col_NAME, mean_line_dash, color_func):
                    # Clean data to avoid inf/nan issues
                    clean_data = hist_df[col_func].replace([np.inf, -np.inf], np.nan).dropna()

                    kde = gaussian_kde(clean_data)
                    x_range = np.linspace(clean_data.min(), clean_data.max(), 100)

                    # add mean trace
                    col_mean_func = clean_data.mean()
                    figfunc.add_trace(go.Scatter(
                        x=[col_mean_func, col_mean_func],  # Constant line at the mean value
                        y=[0, max(kde(x_range))],
                        mode='lines+text',
                        name=f'mean: {col_func} {col_NAME}',
                        opacity=1.0,
                        line=dict(color=color_func, width=2, dash = 'dash'), 
                        text=[None, f'{col_mean_func:.2f}'],  # Show label only at the top
                        textposition='top center',
                        textfont=dict(color=color_func),
                        yaxis='y2',
                    ))

                    # add kde trace
                    figfunc.add_trace(go.Scatter(
                        x=x_range,
                        y=kde(x_range),
                        mode='lines',
                        name=f'kde : {col_func} {col_NAME}',
                        opacity=hist_opacity,
                        line=dict(color=color_func, width=2),
                        yaxis='y2',
                    ))

                    return figfunc

                # hist traces - SUBSET EGID W INST ---------------
                nunique_egid_winst = topo_agg_egids_winst_hist['EGID'].nunique()
                fig_hist.add_trace(go.Histogram(
                    x=[None, ],
                    name=f'EGIDs with pvinst, {nunique_egid_winst} nunique {20*"-"}',
                    opacity=0,
                ))
                fig_hist_agg.add_trace(go.Histogram(
                    x=[None, ],
                    name=f'===== {scen} =====',
                    opacity=0,
                ))
                fig_hist_agg.add_trace(go.Histogram(
                    x=[None, ],
                    name=f'EGIDs with pvinst, {nunique_egid_winst} nunique {15*"-"}',
                    opacity=0,
                ))

                subset_color_pal_idx    = len(subset_trace_color_palette)
                agg_subset_color_pal_idx = len(agg_subset_color_palette)

                for i_col, col in enumerate(hist_cols_to_plot):
                    color      = subset_trace_color_palette[subset_color_pal_idx-(i_col+1) ]
                    agg_color  = agg_subset_color_palette[agg_subset_color_pal_idx-(i_col+1) ]

                    try:
                        fig_hist     = add_kde_cols_trace(fig_hist,     topo_agg_egids_winst_hist, col, 'subset egid w/inst', 'solid', color)
                        fig_hist_agg = add_kde_cols_trace(fig_hist_agg, topo_agg_egids_winst_hist, col, 'subset egid w/inst', 'dash',  agg_color)

                        if self.visual_sett.plot_ind_hist_cols_HOYagg_per_EGID_specs['show_hist_traces_TF']:
                            fig_hist     = add_hist_cols_trace(fig_hist,     topo_agg_egids_winst_hist, col, 'subset egid w/inst', 'solid', color) 
                            fig_hist_agg = add_hist_cols_trace(fig_hist_agg, topo_agg_egids_winst_hist, col, 'subset egid w/inst', 'dash',  agg_color)

                    except Exception as e:
                        print_to_logfile(f'Error in add_hist_cols_trace for col {col}: {e}', self.visual_sett.log_name)
                        continue


                # hist traces - ALL EGID in topo ---------------  
                nunique_egid = topo_agg_egids_hist['EGID'].nunique()
                fig_hist.add_trace(go.Histogram(
                    x=[None, ],
                    name=f'All EGIDs in topo , {nunique_egid} nunique {20*"-"}',
                    opacity=0,
                ))         
                fig_hist_agg.add_trace(go.Histogram(
                    x=[None, ],
                    name=f'All EGIDs in topo , {nunique_egid} nunique {15*"-"}',
                    opacity=0,
                ))

                all_color_pal_idx    = len(all_egid_trace_color_palette)
                agg_all_color_pal_idx = len(agg_allegid_color_palette)             

                for i_col, col in enumerate(hist_cols_to_plot):
                    color      = all_egid_trace_color_palette[all_color_pal_idx-(i_col+1) ]
                    agg_color  = agg_allegid_color_palette[agg_all_color_pal_idx-(i_col+1) ]

                    try:
                        fig_hist     = add_kde_cols_trace(fig_hist,     topo_agg_egids_hist, col, 'all egids in topo', 'solid', color)  
                        fig_hist_agg = add_kde_cols_trace(fig_hist_agg, topo_agg_egids_hist, col, 'all egids in topo', 'dash',  agg_color)

                        if self.visual_sett.plot_ind_hist_cols_HOYagg_per_EGID_specs['show_hist_traces_TF']:
                            fig_hist     = add_hist_cols_trace(fig_hist,     topo_agg_egids_hist, col, 'all egids in topo', 'solid', color) 
                            fig_hist_agg = add_hist_cols_trace(fig_hist_agg, topo_agg_egids_hist, col, 'all egids in topo', 'dash',  agg_color) 

                    except Exception as e:
                        print_to_logfile(f'Error in add_hist_cols_trace for col {col}: {e}', self.visual_sett.log_name)
                        continue


                fig_hist.update_layout(
                    title=f'Histogramm Annual Numbers per EGID ({scen})', 
                    xaxis_title='Total (sum kWh per year)',
                    yaxis_title='Frequency',
                    barmode='overlay',
                    template='plotly_white',
                    yaxis2=dict(
                        title='Density',
                        overlaying='y',
                        side='right',
                        showgrid=False,
                        zeroline=False,
                        ticks='',
                    ),
                )


                # export plot --------------------------
                if plot_show:
                    fig_hist.show()
                if plot_export:
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig_hist.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_ind_hist_cols_HOYagg_per_EGID.html')
                    else:
                        fig_hist.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_hist_cols_HOYagg_per_EGID.html')


                    # for debugging: export csv
                    if self.visual_sett.save_plot_by_scen_directory:
                        topo_agg_egids_hist.to_csv(f'{self.visual_sett.visual_path}/{scen}___topo_agg_egids_hist.csv', index=False)
                        topo_agg_egids_winst_hist.to_csv(f'{self.visual_sett.visual_path}/{scen}___topo_agg_egids_winst_hist.csv', index=False)
                    else:
                        topo_agg_egids_hist.to_csv(f'{self.visual_sett.visual_path}/{scen}___topo_agg_egids_hist.csv', index=False)
                        topo_agg_egids_winst_hist.to_csv(f'{self.visual_sett.visual_path}/{scen}___topo_agg_egids_winst_hist.csv', index=False)


            # plot joint scatter ============================
            if True:

                Map_pvinfo_check = Map_pvinfo_topo_egid.filter(pl.col('EGID').is_in(['2362095', '2362096'])).clone()
                Map_pvinfo_check = Map_pvinfo_check.group_by('EGID').agg([
                    # pl.col('share_FLAECHE_used').sum().alias('share_FLAECHE_used'),
                    pl.col('FLAECHE').sum().alias('FLAECHE'),
                    pl.col('STROMERTRAG').sum().alias('STROMERTRAG'),
                    pl.col('dfuidPower').sum().alias('dfuidPower'),

                    pl.col('TotalPower').max().alias('TotalPower'),
                ])
                Map_pvinfo_check

                Map_pvinfo_recall_egid = Map_pvinfo_topo_egid.clone()
                Map_pvinfo_recall_egid = Map_pvinfo_recall_egid.with_columns([
                    (pl.col('FLAECHE') * pl.col('share_pvprod_used')).alias('share_FLAECHE_used')
                ])

                Map_pvinfo_recall_egid = Map_pvinfo_recall_egid.group_by('EGID').agg([
                    pl.col('share_FLAECHE_used').sum().alias('share_FLAECHE_used'),
                    pl.col('FLAECHE').sum().alias('FLAECHE'),
                    pl.col('STROMERTRAG').sum().alias('STROMERTRAG'),
                    pl.col('dfuidPower').sum().alias('dfuidPower'),

                    pl.col('TotalPower').max().alias('TotalPower'),
                ]).to_pandas()


                # aggregate cols ALL EGID over year
                topo_agg_egids_jsctr = topo_agg_egids_all_df.groupby(['EGID',]).agg({
                    'inst_TF'       : 'first',
                    'info_source'   : 'first',
                    'demand_kW'     : 'sum' ,
                    'radiation'     : 'sum' ,
                    'pvprod_kW'     : 'sum' ,
                    'selfconsum_kW' : 'sum' ,
                    'netdemand_kW'  : 'sum' ,
                    'netfeedin_kW'  : 'sum' ,
                }).reset_index()

                topo_agg_egids_jsctr = topo_agg_egids_jsctr.merge(Map_pvinfo_recall_egid, on='EGID', how='left')

                # subset all w_inst EGIDs
                topo_agg_egids_jsctr = topo_agg_egids_jsctr.loc[topo_agg_egids_jsctr['inst_TF'] == True].copy()

                # Create subplots: 1 column, N rows (or change layout if preferred)
                col_tupl_list = self.visual_sett.plot_ind_hist_cols_HOYagg_per_EGID_specs['xy_tpl_cols_to_plot_joint_scatter']


                info_source_color_map = {
                    'pv_df': 'green', 
                    'alloc_algorithm': 'orange'
                }

                for i, (col_x, col_y) in enumerate(col_tupl_list):
                    # row = i // n_cols + 1
                    # col = i % n_cols + 1

                    fig_jsctr = go.Figure()


                    for info_source, color in info_source_color_map.items():
                        clean_data = topo_agg_egids_jsctr.loc[
                            topo_agg_egids_jsctr['info_source'] == info_source,
                            ['EGID', col_x, col_y]
                        ].replace([np.inf, -np.inf], np.nan).dropna()

                        fig_jsctr.add_trace(
                            go.Scatter(
                                x=clean_data[col_x],
                                y=clean_data[col_y],
                                mode='markers',
                                name=f'{info_source} - x: {col_x}, y: {col_y}',
                                showlegend=True,
                                opacity=0.7,
                                marker=dict(size=10,), # color=color),
                                customdata=clean_data[['EGID', col_x, col_y]].values,
                                hovertemplate=
                                    "EGID: %{customdata[0]}<br>" +
                                    f"{col_x}: " + "%{customdata[1]:.3f}<br>" +
                                    f"{col_y}: " + "%{customdata[2]:.3f}<br>" +
                                    "<extra></extra>"
                            )
                        )                

                    fig_jsctr.update_layout(
                        title='Joint Scatter EGIDs with inst by info_source',
                        xaxis_title=col_x,
                        yaxis_title=col_y,
                        template='plotly_white',
                        showlegend=True
                    )

                    fig_jsctr = self.add_scen_name_to_plot(fig_jsctr, scen, self.pvalloc_scen)

                    if plot_show:
                        fig_jsctr.show()
                    if plot_export:
                        if self.visual_sett.save_plot_by_scen_directory:
                            fig_jsctr.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_jsctr_EGIDS_{col_x}_{col_y}.html')
                        else:
                            fig_jsctr.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_jsctr_EGIDS_{col_x}_{col_y}.html')



        # export agg plot --------------------------
        fig_hist_agg.update_layout(
            title='Histogramm Annual Numbers per EGID', 
            xaxis_title='Total (sum kWh per year)',
            yaxis_title='Frequency',
            barmode='overlay',
            template='plotly_white',
            yaxis2=dict(
                title='Density',
                overlaying='y',
                side='right',
                showgrid=False,
                zeroline=False,
                ticks='',
            ),
        )

        if plot_show:
            fig_hist_agg.show()
        if plot_export:
            if os.path.exists(f'{self.visual_sett.visual_path}/plot_agg_hist_cols_HOYagg___{len(self.visual_sett.pvalloc_scen_list)}scen.html'):
                n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_path}/plot_agg_hist_cols_HOYagg___{len(self.visual_sett.pvalloc_scen_list)}scen*.html'))
                os.rename(f'{self.visual_sett.visual_path}/plot_agg_hist_cols_HOYagg___{len(self.visual_sett.pvalloc_scen_list)}scen.html', 
                          f'{self.visual_sett.visual_path}/plot_agg_hist_cols_HOYagg___{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.html')
            fig_hist_agg.write_html(f'{self.visual_sett.visual_path}/plot_agg_hist_cols_HOYagg___{len(self.visual_sett.pvalloc_scen_list)}scen.html')
            print_to_logfile(f'\texport: plot_agg_hist_cols_HOYagg___{len(self.visual_sett.pvalloc_scen_list)}scen.html', self.visual_sett.log_name)
