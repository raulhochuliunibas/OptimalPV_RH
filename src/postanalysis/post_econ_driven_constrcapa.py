import os
import json
import glob
import joblib
import sys

import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, brentq

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.auxiliary_functions import add_static_topo_data_to_subdf
from src.pvalloc.pvalloc_core import PVAllocCore
# from src.postanalysis.post_worstnode_worstweek import _PlotExportMixin


class EconDrivenConstrcapa():
    """
    Price-sensitivity analysis: how NPV distribution and required construction-capacity
    schedule change under a caller-supplied alternative elecpri/pvtarif, holding the
    scenario's NPV-rank cutoff policy fixed.
    """

    # ---------------- loading helpers ----------------
    def _load_pvalloc_sett(self, scen: str) -> dict:
        path = os.path.join(self.data_path, 'pvalloc', scen, 'pvalloc_sett.json')
        return json.load(open(path, 'r'))



    # ---------------- discontinued  ----------------
    def recompute_npv2(self, name_str):

        subdir_path = os.path.join(self.data_path, 'pvalloc', self.pvalloc_sett["name_dir_export"], )
        # import -----------------------------------------------------
        # gridprem_ts = pl.read_parquet(f'{subdir_path}/gridprem_ts.parquet')    
        topo = json.load(open(os.path.join(subdir_path, 'topo_egid.json'), 'r'))
            # self.data_path, 'pvalloc', self.pvalloc_sett["name_dir_export"], 'topo_egid.json'), 'r'))
        
        # remove all simulated installations
        for k,v in topo.items():
            if  v['pv_inst']['info_source'] == 'alloc_algorithm':
                topo[k]['pv_inst']['inst_TF']     = False
                topo[k]['pv_inst']['info_source'] = ''

        # import topo_time_subdfs -----------------------------------------------------
        topo_subdf_paths = glob.glob(f'{subdir_path}/topo_time_subdf/topo_subdf_*.parquet') 

        # additional imports ---- 
        rfr_model = joblib.load(os.path.join(self.data_path, 'calibration', self.pvalloc_sett["ALGOspec_calib_estim_dir_name"],
                            f'{self.pvalloc_sett["ALGOspec_calib_estim_mod_name_pkl"]}_model.pkl'))
        encoder   = joblib.load(os.path.join(self.data_path, 'calibration', self.pvalloc_sett["ALGOspec_calib_estim_dir_name"],
                            f'{self.pvalloc_sett["ALGOspec_calib_estim_mod_name_pkl"]}_encoder.pkl'))

        agg_npv_df_list = []
        for i, path in enumerate(topo_subdf_paths):
            if i < 3:
                print(f'recompute_npv subdf{i+1}/{len(topo_subdf_paths)+1} (scen: {self.pvalloc_sett["name_dir_export"]}, case: {name_str})')
        
            subdf = pl.read_parquet(path) # subdf_t0 = pd.read_parquet(path)
            
            subdf = subdf.with_columns(
                pl.lit(0).alias("prem_Rp_kWh")
            )   

            # # drop egids with pv installations
            # subdf = subdf_t0.filter(pl.col("EGID").is_in(no_pv_egid))   

            # # drop egids with closed grid nodes
            # if self.sett.GRIDspec_node_1hll_closed_TF:
            #     subdf = subdf.filter( ~pl.col("EGID").is_in(closed_nodes_egid))

            if subdf.shape[0] > 0:

                # extend subdf with static data (safe disk space) --------------------
                # subdf = add_static_topo_data_to_subdf(subdf, topo)


                # # merge gridprem_ts
                # checkpoint_to_logfile('npv > subdf: start merge subdf w gridprem_ts', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None
                # subdf = subdf.join(gridprem_ts[['t', 'grid_node', 'prem_Rp_kWh']], on=['t', 'grid_node'], how='left')  
                # checkpoint_to_logfile('npv > subdf: start merge subdf w gridprem_ts', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None

                # checkpoint_to_logfile('npv > subdf - all df_uid-combinations: start calc selfconsumption + netdemand', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None

                # egid = '2362103'
                agg_npv_list = []                    
                n_egid_econ_functions_counter = 0

                # if True: 
                for egid in list(subdf['EGID'].unique()):
                    egid_subdf = subdf.filter(pl.col('EGID') == egid).clone()

                    # arrange data to fit stat estimation model --------------------

                    # egid_dfuid_subagg = egid_subdf.group_by(['EGID', 'df_uid', ]).agg([
                    sub_egiddfuid = egid_subdf.group_by(['EGID', 'df_uid', ]).agg([
                        pl.col('bfs').first().alias('BFS_NUMMER'),
                        pl.col('GKLAS').first().alias('GKLAS'),
                        pl.col('GAREA').first().alias('GAREA'),
                        pl.col('GBAUJ').first().alias('GBAUJ'),
                        pl.col('GSTAT').first().alias('GSTAT'),
                        pl.col('GWAERZH1').first().alias('GWAERZH1'),
                        pl.col('GENH1').first().alias('GENH1'),
                        pl.col('sfhmfh_typ').first().alias('sfhmfh_typ'),
                        pl.col('demand_arch_typ').first().alias('demand_arch_typ'),
                        pl.col('demand_elec_pGAREA').first().alias('demand_elec_pGAREA'),
                        pl.col('grid_node').first().alias('grid_node'),
                        pl.col('inst_TF').first().alias('inst_TF'),
                        pl.col('info_source').first().alias('info_source'),
                        pl.col('pvid').first().alias('pvid'),
                        pl.col('pvtarif_Rp_kWh').first().alias('pvtarif_Rp_kWh'),
                        pl.col('TotalPower').first().alias('TotalPower'),
                        pl.col('FLAECHE').first().alias('FLAECHE'),
                        pl.col('AUSRICHTUNG').first().alias('AUSRICHTUNG'),
                        pl.col('STROMERTRAG').first().alias('STROMERTRAG'),
                        pl.col('NEIGUNG').first().alias('NEIGUNG'),
                        pl.col('MSTRAHLUNG').first().alias('MSTRAHLUNG'),
                        pl.col('GSTRAHLUNG').first().alias('GSTRAHLUNG'),
                        pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                        pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),
                        ])

                    # create direction classes
                    subagg_dir = sub_egiddfuid.with_columns([
                        pl.when((pl.col("AUSRICHTUNG") > 135) | (pl.col("AUSRICHTUNG") <= -135))
                        .then(pl.lit("north_max_flaeche"))
                        .when((pl.col("AUSRICHTUNG") > -135) & (pl.col("AUSRICHTUNG") <= -45))
                        .then(pl.lit("east_max_flaeche"))
                        .when((pl.col("AUSRICHTUNG") > -45) & (pl.col("AUSRICHTUNG") <= 45))
                        .then(pl.lit("south_max_flaeche"))
                        .when((pl.col("AUSRICHTUNG") > 45) & (pl.col("AUSRICHTUNG") <= 135))
                        .then(pl.lit("west_max_flaeche"))
                        .otherwise(pl.lit("Unkown"))
                        .alias("Direction")
                        ])
                    subagg_dir = subagg_dir.with_columns([
                        pl.col("Direction").fill_null(0).alias("Direction")
                        ])

                    topo_pivot = (
                        subagg_dir
                        .group_by(['EGID', 'Direction'])
                        .agg(
                            pl.col('FLAECHE').max().alias('max_flaeche'), 
                            )
                        .pivot(
                            values='max_flaeche',
                            index='EGID', 
                            on='Direction')
                            .sort('EGID')
                        )
                    topo_rest = (
                        sub_egiddfuid
                        .group_by(['EGID', ])
                        .agg(
                            pl.col('BFS_NUMMER').first().alias('BFS_NUMMER'),
                            pl.col('GAREA').first().alias('GAREA'),
                            pl.col('GBAUJ').first().alias('GBAUJ'),
                            pl.col('GKLAS').first().alias('GKLAS'),
                            pl.col('GSTAT').first().alias('GSTAT'),
                            pl.col('GWAERZH1').first().alias('GWAERZH1'),
                            pl.col('GENH1').first().alias('GENH1'),
                            pl.col('sfhmfh_typ').first().alias('sfhmfh_typ'),
                            pl.col('demand_arch_typ').first().alias('demand_arch_typ'),
                            pl.col('demand_elec_pGAREA').first().alias('demand_elec_pGAREA'),
                            pl.col('grid_node').first().alias('grid_node'),
                            pl.col('inst_TF').first().alias('inst_TF'),
                            pl.col('info_source').first().alias('info_source'),
                            pl.col('pvid').first().alias('pvid'),
                            pl.col('pvtarif_Rp_kWh').first().alias('pvtarif_Rp_kWh'),
                            pl.col('TotalPower').first().alias('TotalPower'),
                            pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                            pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),

                            pl.col('FLAECHE').first().alias('FLAECHE_total'),
                            )
                        )
                    subagg = topo_rest.join(topo_pivot, on=['EGID'], how='left')

                    # fill empty classes with 0
                    for direction in [
                        'north_max_flaeche',
                        'east_max_flaeche',
                        'south_max_flaeche',
                        'west_max_flaeche',
                        ]:
                        if direction not in subagg.columns:
                            subagg = subagg.with_columns([
                            pl.lit(0).alias(direction)
                            ])
                        else:
                            subagg = subagg.with_columns([
                                pl.col(direction).fill_null(0).alias(direction)
                                ])
                        

                    # apply estim model prediction --------------------
                    df = subagg.to_pandas()
                    df['GWAERZH1_str'] = np.where(df['GWAERZH1'].isin(['7410', '7411']), 'heatpump', 'no_heatpump')

                    cols_dtypes_tupls = {
                        # 'year': 'int64',
                        'BFS_NUMMER': 'category',
                        'GAREA': 'float64',
                        # 'GBAUJ': 'int64',   
                        'GKLAS': 'category',
                        # 'GSTAT': 'category',
                        'GWAERZH1': 'category',
                        'GENH1': 'category',
                        'GWAERZH1_str': 'category',
                        # 'InitialPower': 'float64',
                        'TotalPower': 'float64',
                        'elecpri_Rp_kWh': 'float64',
                        'pvtarif_Rp_kWh': 'float64',
                        'FLAECHE_total': 'float64',
                        'east_max_flaeche': 'float64',
                        'west_max_flaeche': 'float64',
                        'north_max_flaeche': 'float64',
                        'south_max_flaeche': 'float64',
                    }
                    df = df[[col for col in cols_dtypes_tupls.keys() if col in df.columns]]

                    df = df.dropna().copy()
                        
                    for col, dtype in cols_dtypes_tupls.items():
                        df[col] = df[col].astype(dtype)
                        

                    X = df.drop(columns=['TotalPower',])
                    cat_cols = X.select_dtypes(include=["object", "category"]).columns

                    encoded_array = encoder.transform(X[cat_cols].astype(str))
                    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cat_cols))
                        
                    # Final feature set
                    X_final = pd.concat(
                        [X.drop(columns=cat_cols).reset_index(drop=True), encoded_df.reset_index(drop=True)],
                        axis=1)
                    X_final = X_final[rfr_model.feature_names_in_]

                    pred_instPower = rfr_model.predict(X_final)[0]
                    df['pred_dfuidPower'] = pred_instPower


                    # distribute kWp to partition(s) -----------------
                    egid_list, dfuid_list, STROMERTRAG_list, FLAECHE_list, AUSRICHTUNG_list, NEIGUNG_list = [], [], [], [], [], []

                    for i, row in sub_egiddfuid.to_pandas().iterrows():
                        egid_list.append(row['EGID'])
                        dfuid_list.append(row['df_uid'])
                        STROMERTRAG_list.append(row['STROMERTRAG'])
                        FLAECHE_list.append(row['FLAECHE'])
                        AUSRICHTUNG_list.append(row['AUSRICHTUNG'])
                        NEIGUNG_list.append(row['NEIGUNG'])
                        
                    topo_egid_df = pd.DataFrame({
                        'EGID': egid_list,
                        'df_uid': dfuid_list,
                        'STROMERTRAG': STROMERTRAG_list,
                        'FLAECHE': FLAECHE_list,
                        'AUSRICHTUNG': AUSRICHTUNG_list, 
                        'NEIGUNG': NEIGUNG_list, 
                    })

                    # unsuitable variable naming ("pick(ed)") because it is copied from algo_Select_AND_adjust_topology_OPTIMIZED()
                    topo_pick_df = topo_egid_df.sort_values(by=['STROMERTRAG', ], ascending = [False,])
                    inst_power = pred_instPower
                    # inst_power = picked_flaeche * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available
                    # picked_flaeche = inst_power / (self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available)
                    picked_flaeche = inst_power / (self.pvalloc_sett['TECspec_kWpeak_per_m2'] * self.pvalloc_sett['TECspec_share_roof_area_available'] )
                    remaining_flaeche = picked_flaeche

                    for i in range(0, topo_pick_df.shape[0]):
                        dfuid_flaeche = topo_pick_df['FLAECHE'].iloc[i]
                        # dfuid_inst_power = dfuid_flaeche * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available
                        dfuid_inst_power = dfuid_flaeche * self.pvalloc_sett['TECspec_kWpeak_per_m2'] * self.pvalloc_sett['TECspec_share_roof_area_available']

                        total_ratio = remaining_flaeche / dfuid_flaeche
                        flaeche_ratio = 1       if total_ratio >= 1 else total_ratio
                        remaining_flaeche -= topo_pick_df['FLAECHE'].iloc[i]
                            
                        idx = topo_pick_df.index[i]
                        topo_pick_df.loc[idx, 'share_pvprod_used'] = flaeche_ratio                     if flaeche_ratio > 0.0 else 0.0
                        topo_pick_df.loc[idx, 'inst_TF']           = True                              if flaeche_ratio > 0.0 else False
                        topo_pick_df.loc[idx, 'TotalPower']        = inst_power  
                        topo_pick_df.loc[idx, 'dfuidPower']        = flaeche_ratio * dfuid_inst_power  if flaeche_ratio > 0.0 else 0.0

                    df_uid_w_inst = [dfuid for dfuid in topo_pick_df['df_uid'] if topo_pick_df.loc[topo_pick_df['df_uid'] == dfuid, 'inst_TF'].values[0] ]
                    df_uid_w_inst_str = '_'.join([str(dfuid) for dfuid in df_uid_w_inst])


                    # calculate selfconsumption + netdemand -----------------
                    topo_pick_pl = pl.from_pandas(topo_pick_df)
                    egid_subdf = egid_subdf.drop(['share_pvprod_used', 'inst_TF', 'TotalPower' ])
                    egid_subdf = egid_subdf.join(topo_pick_pl.select(['EGID', 'df_uid', 'share_pvprod_used', 'inst_TF', 'TotalPower' ]), on=['EGID', 'df_uid'], how='left')

                    egid_subdf = egid_subdf.with_columns([
                        (pl.col("poss_pvprod_kW") * pl.col("share_pvprod_used")).alias("pvprod_kW")
                    ])


                    egid_agg = egid_subdf.group_by(['EGID', 't', 't_int' ]).agg([
                        pl.lit(df_uid_w_inst_str).alias('df_uid_winst'), 
                        pl.col('df_uid').count().alias('n_dfuid'),
                        pl.col('grid_node').first().alias('grid_node'),
                        pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                        pl.col('pvtarif_Rp_kWh').first().alias('pvtarif_Rp_kWh'), 
                        pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),
                        pl.col('TotalPower').first().alias('TotalPower'),
                        pl.col('AUSRICHTUNG').mean().alias('AUSRICHTUNG_mean'),
                        pl.col('NEIGUNG').mean().alias('NEIGUNG_mean'),
                        # ((pl.col('NEIGUNG') * pl.col('FLAECHE')).sum() / pl.col('FLAECHE').sum()).alias('NEIGUNG_wmean'),

                        pl.col('FLAECHE').sum().alias('FLAECHE'),
                        pl.col('poss_pvprod_kW').sum().alias('poss_pvprod_kW'),
                        pl.col('demand_kW').first().alias('demand_kW'),
                        pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                    ])

                    # ----------------------------------
                    #sanity check
                    egid_subdf.filter(pl.col('t').is_in(['t_10', 't_11', 't_12', 't_13'])).select(['EGID', 'df_uid', 'share_pvprod_used', 'poss_pvprod_kW', 'inst_TF', 'pvprod_kW', 't'])
                    egid_agg.filter(pl.col('t').is_in(['t_10', 't_11'])).select(['EGID', 'poss_pvprod_kW', 'pvprod_kW', 't'])
                    # ----------------------------------


                    # calc selfconsumption
                    egid_agg = egid_agg.sort(['EGID', 't_int'], descending = [False, False])

                    # selfconsum_expr = pl.min_horizontal([pl.col("pvprod_kW"), pl.col("demand_kW")]) * self.sett.TECspec_self_consumption_ifapplicable
                    selfconsum_expr = pl.min_horizontal([pl.col("pvprod_kW"), pl.col("demand_kW")]) * self.pvalloc_sett['TECspec_self_consumption_ifapplicable']

                    egid_agg = egid_agg.with_columns([        
                        selfconsum_expr.alias("selfconsum_kW"),
                        (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
                        (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
                    ])

                    # Apply per-household peak shaving for later iterations so NPV reflects withheld feed-in.
                    # ( ...skipped ) 
                    egid_agg = egid_agg.with_columns([
                        pl.col('netfeedin_kW').alias('netfeedin_prepeakshave_kW'),
                        pl.lit(0.0).alias('peak_shaved_kW'),
                    ])

                    # calc econ spend/inc chf
                    egid_agg = egid_agg.with_columns([
                        ((pl.col("netfeedin_kW") * pl.col("pvtarif_Rp_kWh")) / 100 + (pl.col("selfconsum_kW") * pl.col("elecpri_Rp_kWh")) / 100).alias("econ_inc_chf")
                    ])
                        
                    # if not self.sett.ALGOspec_tweak_npv_excl_elec_demand:
                    if not self.pvalloc_sett['ALGOspec_tweak_npv_excl_elec_demand']:
                        egid_agg = egid_agg.with_columns([
                            ((pl.col("netfeedin_kW") * pl.col("prem_Rp_kWh")) / 100 +
                            (pl.col("demand_kW") * pl.col("elecpri_Rp_kWh")) / 100).alias("econ_spend_chf")
                        ])
                    else:
                        egid_agg = egid_agg.with_columns([
                            ((pl.col("netfeedin_kW") * pl.col("prem_Rp_kWh")) / 100).alias("econ_spend_chf")
                        ])


                    # NPV calculation -----------------
                    # -----
                    def _get_instcost_interpolate_function(self, scen: str):
                        path = os.path.join(self.data_path, 'pvalloc', scen, 'pvinstcost_coefficients.json')
                        with open(path, 'r') as f:
                            coefs = json.load(f)
                        params_pkW, params_total = coefs['params_pkW'], coefs['params_total']
                        def estim_instcost_chfpkW(x):
                            a, b, c, d = params_pkW
                            return a + d * ((x ** b) / (x ** c))
                        def estim_instcost_chftotal(x):
                            a, b, c = params_total
                            return a + b * (x ** c)
                        return estim_instcost_chfpkW, estim_instcost_chftotal
                    # -----

                    estim_instcost_chfpkW, estim_instcost_chftotal = _get_instcost_interpolate_function(self, self.pvalloc_sett["name_dir_export"])
                    estim_instcost_chftotal(pd.Series([10, 20, 30, 40, 50, 60, 70]))

                    annual_cashflow = (egid_agg["econ_inc_chf"].sum() - egid_agg["econ_spend_chf"].sum())
                    # installation_cost = estim_instcost_chftotal(pred_instPower) * (1-self.sett.TECspec_KEVflatsubs)
                    installation_cost = estim_instcost_chftotal(pred_instPower) * (1-self.pvalloc_sett['TECspec_KEVflatsubs'])
                               
                    # discount_factor = np.array([(1 + self.sett.TECspec_interest_rate)**-i for i in range(1, self.sett.TECspec_invst_maturity + 1)])
                    discount_factor = np.array([(1 + self.pvalloc_sett['TECspec_interest_rate'])**-i for i in range(1, self.pvalloc_sett['TECspec_invst_maturity'] + 1)])
                    disc_cashflow = annual_cashflow * np.sum(discount_factor)
                    npv = -installation_cost + disc_cashflow

                    egid_npv = egid_agg.group_by(['EGID', ]).agg([
                        pl.col('df_uid_winst').first().alias('df_uid_winst'),
                        pl.col('n_dfuid').first().alias('n_dfuid'),
                        pl.col('grid_node').first().alias('grid_node'),
                        pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                        pl.col('pvtarif_Rp_kWh').first().alias('pvtarif_Rp_kWh'), 
                        pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),
                        pl.col('TotalPower').first().alias('TotalPower'),
                        pl.col('AUSRICHTUNG_mean').first().alias('AUSRICHTUNG_mean'),
                        pl.col('NEIGUNG_mean').first().alias('NEIGUNG_mean'),
                        # pl.col('NEIGUNG_wmean').first().alias('NEIGUNG_wmean'),
                        pl.col('FLAECHE').first().alias('FLAECHE'),
                          
                        pl.col('poss_pvprod_kW').sum().alias('poss_pvprod_kW'),
                        pl.col('demand_kW').sum().alias('demand_kW'),
                        pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                        pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                        pl.col('netfeedin_prepeakshave_kW').sum().alias('netfeedin_prepeakshave_kW'),
                        pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                        pl.col('peak_shaved_kW').sum().alias('peak_shaved_kW'),
                        pl.col('netdemand_kW').sum().alias('netdemand_kW'),
                        pl.col('econ_inc_chf').sum().alias('econ_inc_chf'),
                        pl.col('econ_spend_chf').sum().alias('econ_spend_chf'),
                        ])
                    egid_npv = egid_npv.with_columns([
                        pl.lit(pred_instPower).alias("pred_instPower"),
                        pl.lit(installation_cost).alias("estim_pvinstcost_chf"),
                        pl.lit(disc_cashflow).alias("disc_cashflow"),
                        pl.lit(npv).alias("NPV_uid"),
                        ])
                        
                    agg_npv_df_list.append(egid_npv)

        
        # concat all egid_agg
        npv_df = pl.concat(agg_npv_df_list).clone()


        # add roof specific filter tag VERSION 2 -----------------
        topo_filter_egid = self.get_topo_filter_tag_mapping(topo)

        npv_df = npv_df.join(topo_filter_egid, on='EGID', how='left')

            
        # add subsidy where applicable -----------------
        # node_subsidy_monitor_iter = node_subsidy_monitor_dict[str(i_m)]
        # subsidy_node_name   = self.sett.GRIDspec_subsidy_name
        # subsidy_filtag_node_schemes = self.sett.GRIDspec_subsidy_filtag_node_schemes[subsidy_node_name]
        # npv_df = npv_df.with_columns([
        #     pl.lit(0.0).alias('subs_filter_tags_chf'),
        #     pl.lit(0.0).alias('subs_nodeHC_chf'),
        #     pl.lit(0.0).alias('pena_nodeHC_chf'),
        # ])



    # -- recompute NPV (reused for base AND adjusted) ----------------
    def recompute_npv(self,
                       pvalloc_sett:    dict,  
                       npv_df:          pl.DataFrame, 
                       elecpri:         float = None, 
                       pvtarif:         float = None, 
                       ) -> pl.DataFrame:
        """
        Recompute econ_inc/spend, disc_cashflow, NPV_uid_before_subsidy, NPV_uid under a given
        elecpri/pvtarif. subs_filter_tags_chf / subs_nodeHC_chf / pena_nodeHC_chf are read as-is
        from npv_df (price-independent, carried over unchanged). Settings are read from the
        scenario's own pvalloc_sett.json dict, not dataclass defaults. Original on-disk values
        for the 7 touched columns are preserved under '*_asexported' for traceability/verification.
        """
        interest_rate    = float(pvalloc_sett.get('TECspec_interest_rate', 0.015))
        invst_maturity   = int(pvalloc_sett.get('TECspec_invst_maturity', 25))
        excl_elec_demand = bool(pvalloc_sett.get('ALGOspec_tweak_npv_excl_elec_demand', True))
        discount_factor  = sum((1 + interest_rate) ** -i for i in range(1, invst_maturity + 1))
        
        elecpri_Rp_kWh = float(elecpri) if elecpri is not None else float(pvalloc_sett.get('TECspec_generic_elecpri_Rp_kWh', 0.0))
        pvtarif_Rp_kWh = float(pvtarif) if pvtarif is not None else float(pvalloc_sett.get('TECspec_generic_pvtarif_Rp_kWh', 0.0)) 

        rename_map = {
            'elecpri_Rp_kWh':         'elecpri_Rp_kWh_asexported',
            'pvtarif_Rp_kWh':         'pvtarif_Rp_kWh_asexported',
            'econ_inc_chf':           'econ_inc_chf_asexported',
            'econ_spend_chf':         'econ_spend_chf_asexported',
            'disc_cashflow':          'disc_cashflow_asexported',
            'NPV_uid_before_subsidy': 'NPV_uid_before_subsidy_asexported',
            'NPV_uid':                'NPV_uid_asexported',
        }
        npv_adj_df = npv_df.rename(rename_map)
        npv_adj_df = npv_adj_df.with_columns([
            pl.lit(float(elecpri_Rp_kWh)).alias('elecpri_Rp_kWh'),
            pl.lit(float(pvtarif_Rp_kWh)).alias('pvtarif_Rp_kWh'),
        ])
        npv_adj_df = npv_adj_df.with_columns([
            (pl.col('netfeedin_kW') * pl.col('pvtarif_Rp_kWh') / 100
             + pl.col('selfconsum_kW') * pl.col('elecpri_Rp_kWh') / 100).alias('econ_inc_chf'),
        ])
        if excl_elec_demand:
            npv_adj_df = npv_adj_df.with_columns([
                (pl.col('netfeedin_kW') * pl.col('prem_Rp_kWh') / 100).alias('econ_spend_chf'),
            ])
        else:
            npv_adj_df = npv_adj_df.with_columns([
                (pl.col('netfeedin_kW') * pl.col('prem_Rp_kWh') / 100
                 + pl.col('demand_kW') * pl.col('elecpri_Rp_kWh') / 100).alias('econ_spend_chf'),
            ])
        npv_adj_df = npv_adj_df.with_columns([(pl.col('econ_inc_chf') - pl.col('econ_spend_chf')).alias('_annual_cashflow')])
        npv_adj_df = npv_adj_df.with_columns([(pl.col('_annual_cashflow') * discount_factor).alias('disc_cashflow')])
        npv_adj_df = npv_adj_df.with_columns([
            (-pl.col('estim_pvinstcost_chf') + pl.col('disc_cashflow')).alias('NPV_uid_before_subsidy'),
        ])
        npv_adj_df = npv_adj_df.with_columns([
            (pl.col('NPV_uid_before_subsidy')).alias('NPV_uid'),
        ])
        return npv_adj_df.drop('_annual_cashflow')


    # -- base threshold derivation ----------------
    def _derive_threshold_npv(self, npv_df_base: pl.DataFrame, constrcapa_df: pl.DataFrame) -> pl.DataFrame:
        """Rank by NPV_uid desc, cumsum pred_instPower, find NPV at which the cumsum
        crosses each year's constr_capacity_kw budget. Returns constrcapa_df (date/year/month)
        plus constr_capacity_kw_input, threshold_npv, constr_capacity_kw_required."""
        ranked = (
            npv_df_base
            .sort('NPV_uid', descending=True)
            .with_columns(pl.col('pred_instPower').cum_sum().alias('cum_instPower_kw'))
            .select(['EGID', 'NPV_uid', 'pred_instPower', 'cum_instPower_kw' ])
        )
        total_capacity_kw = ranked.select(pl.col('cum_instPower_kw').max()).item()
        min_npv = ranked.select(pl.col('NPV_uid').min()).item()

        thresholds = []
        ranked_loop = ranked.clone()
        for row in constrcapa_df.iter_rows(named=True):
            budget = row['constr_capacity_kw']
            if budget >= total_capacity_kw:
                thr, achieved = min_npv, total_capacity_kw   # budget exceeds all available capacity
            else:
                ranked_loop = ranked_loop.with_columns(pl.col('pred_instPower').cum_sum().alias('cum_loop_instPower_kw'))
                ranked_in_thr = ranked_loop.filter( pl.col('cum_loop_instPower_kw') <= budget)
                thr      = ranked_in_thr.select(pl.col('NPV_uid').min()).item()
                achieved = ranked_in_thr.select(pl.col('cum_loop_instPower_kw').max()).item()
                
                thresholds.append({'year': row['year'], 'constr_capacity_kw_required': achieved, 'threshold_npv': thr })
                ranked_loop = ranked_loop.filter(~pl.col('EGID').is_in(ranked_in_thr['EGID'].to_list()))  # remove already counted EGIDs for next iteration           


        thr_df = pl.DataFrame(thresholds)
        thr_compare_df = constrcapa_df.rename({'constr_capacity_kw': 'constr_capacity_kw_input'}).join(thr_df, on='year', how='left')
        return thr_compare_df


    # -- extend capacity nulls until 2050 ----------------
    def _extend_constrcapa_until_2050(self, cap_df: pl.DataFrame, thr_compare_df: pl.DataFrame,
                                      target_year: int = 2050,
                                      last_years: list = [2051, 2052, 2053],
                                    #   n_disregard_tail: int = 1,
                                      last_years_capacity_kw: float = None,
                                      max_asymptote_kw: float = 250000.0,
                                      adj_tupl: tuple = None,
                                      show_export_plots: tuple = (False, True),
                                      plot_width_func: int = None,
                                      plot_height_func: int = None,
                                      ) -> pl.DataFrame:
        """Replace the exhausted tail of cap_df.constr_capacity_kw (nulls / 0.0 once the NPV pool
        is used up, plus the last n_disregard_tail valid values) with a logistic curve
            1/f(x) = 1/K + d * exp(-r*x),  x = year - first_year,   K = asymptote
        that is pinned at two points: the last kept value (anchor, so the extension continues from it
        without a jump) and thr_compare_df['constr_capacity_kw_input'] at target_year. Given r, K and d
        follow in closed form; r is fitted (relative residuals) to the kept values, with K <= max_asymptote_kw
        (None = no bound). Rows in last_years are set to last_years_capacity_kw (effectively unconstrained)."""
        n_disregard_tail = adj_tupl[3] if adj_tupl is not None and len(adj_tupl) > 3 else 1

        thr_compare_df = thr_compare_df.sort('year')
        original_years = thr_compare_df.get_column('year').to_numpy()
        original_vals  = thr_compare_df.get_column('constr_capacity_kw_input').to_numpy()

        cap_df = cap_df.sort('year')
        years  = cap_df.get_column('year').to_numpy()
        vals   = cap_df.get_column('constr_capacity_kw').to_numpy()       # nulls -> nan
        is_last = np.isin(years, last_years)

        target_rows = thr_compare_df.filter(pl.col('year') == target_year)
        if target_rows.height != 1:
            raise ValueError(f'thr_compare_df needs exactly one row for target_year {target_year}, found {target_rows.height}.')
        target_kw = float(target_rows.get_column('constr_capacity_kw_input').item())

        if last_years_capacity_kw is None:
            last_years_capacity_kw = target_kw * 1.05

        valid  = np.nan_to_num(vals, nan=0.0) > 0       # 0.0 counts as exhausted, like null
        not_valid = [not v for v in valid]
        n_good = len(vals) if valid.all() else int(np.argmin(valid))      # length of the leading valid run
        n_fit  = n_good - n_disregard_tail

        new_vals = vals.copy()
        if n_good > 0 and years[n_good - 1] >= target_year:
            print(f'extend_constrcapa: valid capacities already reach {target_year}, no fit needed.')
        else:
            if n_fit < 3:
                raise ValueError(f'extend_constrcapa: only {n_fit} valid values left to fit (need >= 3).')

            x, x_target = (years - years[0]).astype(float), float(target_year - years[0])
            v_fit, x_fit = vals[:n_fit], x[:n_fit]
            x_anchor, v_anchor = x_fit[-1], v_fit[-1]                     # last kept value: the curve must continue from here

            if v_anchor >= target_kw:
                raise ValueError(f'extend_constrcapa: last kept value {v_anchor:,.0f} kW ({int(years[n_fit - 1])}) is not below '
                                 f'the target {target_kw:,.0f} kW ({target_year}), cannot extend upwards.')
            if max_asymptote_kw is not None and max_asymptote_kw <= target_kw:
                raise ValueError(f'extend_constrcapa: max_asymptote_kw {max_asymptote_kw:,.0f} must exceed the target {target_kw:,.0f} kW.')

            def logistic_cd(r):                                           # 1/f = c + d*exp(-r*x); pins f(x_anchor) = v_anchor, f(x_target) = target_kw
                d = (1 / v_anchor - 1 / target_kw) / (np.exp(-r * x_anchor) - np.exp(-r * x_target))
                return 1 / target_kw - d * np.exp(-r * x_target), d       # c = 1/K

            def logistic(r, xx):
                c, d = logistic_cd(r)
                return 1 / (c + d * np.exp(-r * xx))

            r_lo = np.log(target_kw / v_anchor) / (x_target - x_anchor) * 1.0001   # c > 0 (finite asymptote) only for r above this
            if max_asymptote_kw is not None:
                r_lo = max(r_lo, brentq(lambda r: logistic_cd(r)[0] - 1 / max_asymptote_kw, r_lo, 5.0))   # K <= max_asymptote_kw
            r_start = min(max(1.5 * r_lo, 0.2), 4.9)

            sol = least_squares(
                lambda p: (logistic(p[0], x_fit) - v_fit) / v_fit,
                x0     = [r_start],
                bounds = ([r_lo], [5.0]),
            )
            if not sol.success:
                raise ValueError(f'extend_constrcapa: logistic fit did not converge ({sol.message}).')

            r, K = sol.x[0], 1 / logistic_cd(sol.x[0])[0]
            fitted = logistic(r, x)
            fill = (np.arange(len(vals)) >= n_fit) & ~is_last
            new_vals[fill] = fitted[fill]
            print(f'extend_constrcapa: logistic anchored at {int(years[n_fit - 1])} ({v_anchor:,.0f} kW) and {target_year} ({target_kw:,.0f} kW) '
                  f'(r={r:.3f}, asymptote K={K:,.0f} kW{" [at max_asymptote_kw]" if max_asymptote_kw is not None and K >= 0.999 * max_asymptote_kw else ""}, '
                  f'max rel. err on kept values={np.max(np.abs(sol.fun)):.1%}); replaced {int(fill.sum())} rows {int(years[n_fit])}-{int(years[fill].max())}.')

        new_vals[is_last] = last_years_capacity_kw

        # plot extension values
        plot_width  = self.plot_width  if plot_width_func  is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func


        plt.figure(figsize=(plot_width, plot_height))
        plt.plot(original_years, original_vals, 'o', label='npv base (ep2050+)', color='gray')
        # plt.plot(years, vals, 'o', label='npv recalc', color='red')
        plt.plot(years[valid], vals[valid], 'o', label='npv recalc', color='red')
        plt.plot(years[not_valid], new_vals[not_valid], '-', label='extended', color='blue')
        plt.axvline(target_year, color='black', linestyle=':', label=f'target year {target_year}')
        plt.axhline(target_kw, color='black', linestyle='--', label=f'target cap. {target_kw:,.0f} kW')
        plt.xlabel('Year'); plt.ylabel('Constr. Capacity (kW)'); plt.title(f'Ext. ConstCapa ({adj_tupl[0]}: elecp {adj_tupl[1]} Rp, pv {adj_tupl[2]} Rp)')
        plt.legend(); plt.tight_layout()
        if show_export_plots[1]:    # export BEFORE show: closing the plt.show() window destroys the figure, savefig would then write a new blank one
            export_file = f'constrcapa_extension_{adj_tupl[0]}.png'
            plt.savefig(os.path.join(self.dir_path_postanalysis, export_file), dpi=self.plot_dpi)
        if show_export_plots[0]:
            plt.show()
        plt.close()

        return cap_df.with_columns(pl.Series('constr_capacity_kw', new_vals, dtype=pl.Float64))


    # -- histogram (adapted from visualization_static.NPVhist) ----------------
    def _plot_npv_histogram(self, npv_df: pl.DataFrame, scen: str, run_tag: str,
                             export_name: str = None, title: str = None,
                             npv_hist_xrange: tuple = None,
                             x_label: str = 'NPV (CHF)', y_label: str = 'Count',
                             negative_rgb=(214, 39, 40), positive_rgb=(31, 119, 180),
                             plot_width_func=None, plot_height_func=None, 
                             adj_tupl: tuple = None, show_export_plots = (False, True)):
        values = npv_df.get_column('NPV_uid').drop_nulls().to_numpy()
        if values.size == 0:
            print(f'No NPV_uid values to plot for {scen} ({run_tag}).')
            return

        mean_val, median_val = float(np.mean(values)), float(np.median(values))

        def _rgb_to_mpl(rgb):
            arr = np.asarray(rgb, dtype=float)
            return tuple((arr / 255.0 if arr.max() > 1.0 else arr).tolist())

        negative_color, positive_color = _rgb_to_mpl(negative_rgb), _rgb_to_mpl(positive_rgb)
        plot_width  = self.plot_width  if plot_width_func  is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func

        data_min, data_max = float(values.min()), float(values.max())
        hist_range = (data_min, data_max)
        if npv_hist_xrange is not None:
            hist_min, hist_max = max(float(npv_hist_xrange[0]), data_min), min(float(npv_hist_xrange[1]), data_max)
            if hist_min < hist_max:
                hist_range = (hist_min, hist_max)

        bins = np.linspace(hist_range[0], hist_range[1], 41)
        counts, bin_edges = np.histogram(values, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bar_colors = [negative_color if c < 0 else positive_color for c in bin_centers]

        plt.figure(figsize=(plot_width, plot_height))
        ax = plt.gca()
        ax.bar(bin_edges[:-1], counts, width=np.diff(bin_edges), align='edge',
               color=bar_colors, edgecolor='white', linewidth=0.2, alpha=0.75)
        ax.axvline(0, color='black', linestyle=':', linewidth=1.0, label='Zero')
        ax.axvline(mean_val,   color='black', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:,.0f}')
        ax.axvline(median_val, color='black', linestyle='-',  linewidth=1.5, label=f'Median: {median_val:,.0f}')
        ax.set_xlim(hist_range[0], hist_range[1])
        plt.xlabel(x_label); plt.ylabel(y_label)
        if adj_tupl is not None:
            plt.title(title or f'NPV Dist. - {adj_tupl[0]} (elecp {adj_tupl[1]} Rp, pv {adj_tupl[2]} Rp)')
        else:
            plt.title(title or f'NPV Dist. - {run_tag}' )
        plt.legend(title=None)
        plt.tight_layout()

        export_file = export_name or f'npv_hist_{run_tag}.png'
        if show_export_plots[1]:
            plt.savefig(os.path.join(self.dir_path_postanalysis, export_file), dpi=self.plot_dpi)
        if show_export_plots[0]:
            plt.show()
        plt.close()


    # -- overlay histogram: base vs adjusted, in levels + standardized ----------------
    def plot_overlay_npv_histo(self, npv_df_base: pl.DataFrame, npv_df_adjusted: pl.DataFrame, scen: str, run_tag: str,
                               export_name: str = None, title: str = None, threshold_df: pl.DataFrame = None,
                               npv_hist_xrange: tuple = None, n_bins: int = 40,
                               x_label: str = 'NPV (CHF)', y_label: str = 'Count',
                               base_rgb=(31, 119, 180), adjusted_rgb=(255, 127, 14),
                               alpha_levels: float = 0.55, alpha_standardized: float = 0.35,
                               plot_width_func=None, plot_height_func=None,
                               adj_tupl: tuple = None, show_export_plots = (False, True)):
        """Overlay of the base and the adjusted NPV_uid distribution on identical bins.
        Top panel: histograms in levels (CHF). Bottom panel: both cases standardized separately,
        z = (NPV - mean) / std, which removes shift and scale between the cases so that only differences
        in shape remain (lighter fill). The z-scores live on their own x-axis, hence the second panel.
        The prices in the legend are read from the elecpri/pvtarif columns of the two dataframes.
        threshold_df (output of _derive_threshold_npv): its yearly 'threshold_npv' values (derived from the base
        ranking, applied to the adjusted NPVs in the orchestrator) are drawn as vertical lines in the top panel."""
        cases = []
        for df, name, rgb in [(npv_df_base,     'Base',                                             base_rgb),
                              (npv_df_adjusted, adj_tupl[0] if adj_tupl is not None else 'Adjusted', adjusted_rgb)]:
            values = df.get_column('NPV_uid').drop_nulls().to_numpy()
            if values.size == 0:
                print(f'No NPV_uid values to plot for {scen} ({name}).')
                return
            mean_val, median_val, std_val = float(np.mean(values)), float(np.median(values)), float(np.std(values))
            cases.append({
                'name':   name,
                'values': values,
                'z':      (values - mean_val) / std_val if std_val > 0 else np.zeros_like(values),
                'mean':   mean_val,
                'median': median_val,
                'color':  tuple(c / 255.0 for c in rgb),
                'label':  f"{name} (elecp {df.get_column('elecpri_Rp_kWh')[0]:g} Rp, pv {df.get_column('pvtarif_Rp_kWh')[0]:g} Rp)\n"
                          f"mean {mean_val:,.0f} | median {median_val:,.0f}",
            })

        # common bins for both cases (levels), and the same observations re-expressed as z-scores
        data_min, data_max = min(c['values'].min() for c in cases), max(c['values'].max() for c in cases)
        hist_min, hist_max = data_min, data_max
        if npv_hist_xrange is not None:
            xr_min, xr_max = max(float(npv_hist_xrange[0]), data_min), min(float(npv_hist_xrange[1]), data_max)
            if xr_min < xr_max:
                hist_min, hist_max = xr_min, xr_max
        bins = np.linspace(hist_min, hist_max, n_bins + 1)
        z_all = np.concatenate([c['z'][(c['values'] >= hist_min) & (c['values'] <= hist_max)] for c in cases])
        z_bins = np.linspace(z_all.min(), z_all.max(), n_bins + 1)

        plot_width  = self.plot_width  if plot_width_func  is None else plot_width_func
        plot_height = self.plot_height if plot_height_func is None else plot_height_func

        fig, (ax_lvl, ax_std) = plt.subplots(2, 1, figsize=(plot_width, 1.6 * plot_height))
        for c in cases:                     # translucent fill (alpha), solid outline -> overlap stays readable
            ax_lvl.hist(c['values'], bins=bins, histtype='stepfilled', facecolor=(*c['color'], alpha_levels),
                        edgecolor=c['color'], linewidth=1.0, label=c['label'])
            # ax_lvl.axvline(c['mean'],   color=c['color'], linestyle='--', linewidth=1.5)
            # ax_lvl.axvline(c['median'], color=c['color'], linestyle='-',  linewidth=1.5)
            ax_std.hist(c['z'], bins=z_bins, histtype='stepfilled', facecolor=(*c['color'], alpha_standardized),
                        edgecolor=c['color'], linewidth=1.0)
        ax_lvl.axvline(0, color='black', linestyle=':', linewidth=1.0, label='Zero')
        # ax_lvl.plot([], [], color='black', linestyle='--', label='Mean')       # legend proxies for the mean/median lines above
        # ax_lvl.plot([], [], color='black', linestyle='-',  label='Median')
        ax_std.axvline(0, color='black', linestyle=':', linewidth=1.0)         # z = 0 is the mean of each case

        # yearly NPV thresholds (base ranking): dashed lines, year written above the axes; only those inside the plotted range
        thr_by_year = []
        if threshold_df is not None:
            thr_by_year = [(int(row['year']), row['threshold_npv']) for row in threshold_df.sort('year').iter_rows(named=True)
                           if row['threshold_npv'] is not None and hist_min <= row['threshold_npv'] <= hist_max]
        for i, (year, thr) in enumerate(thr_by_year):
            ax_lvl.axvline(thr, color='0.35', linestyle='--', linewidth=0.8, label='NPV threshold per year (base)' if i == 0 else None)
            ax_lvl.text(thr, 1.0, str(year), transform=ax_lvl.get_xaxis_transform(), rotation=90, ha='center', va='bottom',
                        fontsize='xx-small', color='0.2')

        ax_lvl.set_xlim(hist_min, hist_max)
        ax_lvl.xaxis.set_major_locator(plt.MaxNLocator(nbins=6))               # CHF tick labels are long, avoid overlapping labels
        ax_lvl.set_xlabel(x_label); ax_lvl.set_ylabel(y_label)
        ax_lvl.set_title(title or f'NPV Dist. - Base vs {cases[1]["name"]}', pad=22 if thr_by_year else None)     # room for the year labels
        ax_lvl.legend(fontsize='x-small', loc='upper right')
        ax_std.set_xlim(z_bins[0], z_bins[-1])
        ax_std.set_xlabel('Standardized NPV (z = (NPV - mean) / std)'); ax_std.set_ylabel(y_label)
        fig.tight_layout()

        export_file = export_name or f'npv_hist_overlay_{run_tag}.png'
        if show_export_plots[1]:            # export BEFORE show: closing the plt.show() window destroys the figure
            fig.savefig(os.path.join(self.dir_path_postanalysis, export_file), dpi=self.plot_dpi)
        if show_export_plots[0]:
            plt.show()
        plt.close(fig)


    # -- orchestrator (public entry point) ----------------
    def econ_driven_constrcapa_analysis(self,
                                         scen: str = 'pvalloc_LRG3_max',
                                         adjustment_tupls: list[tuple[str, float, float]] = [
                                             ('test', 30.0, 6.0 ),],
                                         elecpri_Rp_kWh_base: float = None,
                                         pvtarif_Rp_kWh_base: float = None,
                                        #  scen_suffix: str = 'adjusted',
                                        #  elecpri_Rp_kWh_adjusted: float = None,
                                        #  pvtarif_Rp_kWh_adjusted: float = None,
                                         constrcapa_override=None,
                                         export_csv: bool = False,
                                         npv_hist_kwargs: dict = None,
                                         overlay_hist_kwargs: dict = None,
                                         show_export_plots: tuple= (False, True),
                                         ) -> dict:
        npv_hist_kwargs = npv_hist_kwargs or {}
        overlay_hist_kwargs = overlay_hist_kwargs or {}

        self.pvalloc_sett  = self._load_pvalloc_sett(scen)
        npv_df_raw    = pl.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'zMC1', 'pred_npv_inst_by_M', 'npv_df_1.parquet'))

        if constrcapa_override is not None:
            if isinstance(constrcapa_override, dict):
                df = pl.DataFrame({
                'year': list(constrcapa_override.keys()),
                    'constr_capacity_kw': list(constrcapa_override.values()),
                }).with_columns(pl.col('year').cast(pl.Int32), pl.col('constr_capacity_kw').cast(pl.Float64))
                df = df.with_columns(pl.lit(12).alias('month'))
                df = df.with_columns(pl.date(pl.col('year'), 12, 31).alias('date'))
            elif isinstance(constrcapa_override, str):
                df = pl.read_parquet(constrcapa_override)
        else: 
            df = pl.read_parquet(os.path.join(self.data_path, 'pvalloc', scen, 'constrcapa.parquet'))
            
        constrcapa_in = df

        elecpri_base = elecpri_Rp_kWh_base if elecpri_Rp_kWh_base is not None \
            else self.pvalloc_sett.get('TECspec_generic_elecpri_Rp_kWh')
        pvtarif_base = pvtarif_Rp_kWh_base if pvtarif_Rp_kWh_base is not None \
            else self.pvalloc_sett.get('TECspec_generic_pvtarif_Rp_kWh')
        if elecpri_base is None or pvtarif_base is None:
            raise ValueError(f'pvalloc_sett.json for {scen} is missing generic price fields; '
                              f'pass elecpri_Rp_kWh_base/pvtarif_Rp_kWh_base explicitly.')


        # ---- base ----
        npv_df_base  = self.recompute_npv(pvalloc_sett = self.pvalloc_sett, npv_df = npv_df_raw, 
                                          elecpri = elecpri_base, 
                                          pvtarif = pvtarif_base
                                          )
        runtag = f'{scen}_base e{elecpri_base}Rp pv{pvtarif_base}Rp'
        self._plot_npv_histogram(npv_df_base, scen=scen, run_tag=runtag, show_export_plots = show_export_plots, # npv_hist_xrange = (60000, 150000),
                                 **npv_hist_kwargs)
        
        # -- check debug-----
        check_df  = npv_df_base.filter(pl.col('NPV_uid') > 100000).sort('NPV_uid',descending = True).with_columns([
            # (pl.col('demand_kW') / 1000.0 ).alias('demand_MW'),
            # (pl.col('netfeedin_kW') / 1000.0 ).alias('netfeedin_MW'),
            # (pl.col('selfconsum_kW') / 1000.0 ).alias('selfconsum_MW'),
        ])
        check_df[['EGID','NPV_uid', 
                  'elecpri_Rp_kWh_asexported', 'elecpri_Rp_kWh', 
                  'TotalPower', 'FLAECHE', 
                #   'demand_kW', 'pvprod_kW', 'selfconsum_kW', 'netfeedin_kW', 'netdemand_kW',  
                  ]]
        # -----
        
        threshold_df = self._derive_threshold_npv(npv_df_base, constrcapa_in)

        npv_df_base.write_csv(    os.path.join(self.dir_path_postanalysis, f'npv_df_{scen}_base.csv'))
        npv_df_base.write_parquet(os.path.join(self.dir_path_postanalysis, f'npv_df_{scen}_base.parquet'))

        threshold_df.write_csv(    os.path.join(self.dir_path_postanalysis,   f'constrcapa_{scen}_base.csv'))
        threshold_df.write_parquet(os.path.join(self.dir_path_postanalysis,   f'constrcapa_{scen}_base.parquet'))
        constrcapa_in.write_csv(    os.path.join(self.dir_path_postanalysis,   f'constrcapa_{scen}_input.csv'))
        constrcapa_in.write_parquet(os.path.join(self.dir_path_postanalysis,   f'constrcapa_{scen}_input.parquet'))


        # ---- adjusted ----
        for adj_tupl in adjustment_tupls or []:
            
            run_tag_adj     = f'{scen}_{adj_tupl[0]}'
            npv_df_adjusted = self.recompute_npv(pvalloc_sett = self.pvalloc_sett, npv_df = npv_df_raw, 
                                                 elecpri = adj_tupl[1],
                                                 pvtarif = adj_tupl[2],
                                            )
            self._plot_npv_histogram(npv_df_adjusted, scen=scen, run_tag=run_tag_adj, adj_tupl = adj_tupl, show_export_plots = show_export_plots, **npv_hist_kwargs)
            self.plot_overlay_npv_histo(npv_df_base, npv_df_adjusted, scen=scen, run_tag=run_tag_adj, threshold_df = threshold_df, adj_tupl = adj_tupl, show_export_plots = show_export_plots, **overlay_hist_kwargs)

            results = []
            npv_df_loop = npv_df_adjusted.clone()
            for row in threshold_df.iter_rows(named=True):
                thr = row['threshold_npv']
                npv_in_cap = npv_df_loop.filter(pl.col('NPV_uid') >= thr)
                capa = npv_in_cap.select(pl.col('pred_instPower').sum()).item()

                results.append({
                    'date': f"{row['year']}-12-31",
                    'year': row['year'],
                    'month': 12,
                    'constr_capacity_kw': float(capa or 0.0)
                    })
                npv_df_loop = npv_df_loop.filter(~ pl.col('EGID').is_in(npv_in_cap.get_column('EGID'))).clone()
                
            cap_df = pl.DataFrame(results)

            cap_df = self._extend_constrcapa_until_2050(cap_df = cap_df, 
                                                        thr_compare_df = threshold_df, 
                                                        adj_tupl = adj_tupl, 
                                                        show_export_plots = show_export_plots)

            npv_df_adjusted.write_csv(    os.path.join(self.dir_path_postanalysis, f'npv_df_{run_tag_adj}.csv'))
            npv_df_adjusted.write_parquet(os.path.join(self.dir_path_postanalysis, f'npv_df_{run_tag_adj}.parquet'))
            
            cap_df.write_csv(    os.path.join(self.dir_path_postanalysis,   f'constrcapa_{run_tag_adj}.csv'))
            cap_df.write_parquet(os.path.join(self.dir_path_postanalysis,   f'constrcapa_{run_tag_adj}.parquet'))
            
            cap_df.write_csv(    os.path.join(self.dir_path_constrcapa_adj, f'constrcapa_{run_tag_adj}.csv'))
            cap_df.write_parquet(os.path.join(self.dir_path_constrcapa_adj, f'constrcapa_{run_tag_adj}.parquet'))


