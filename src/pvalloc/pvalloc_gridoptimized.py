# python integrated packages
import sys
import os as os 
import glob
import shutil
import json
import copy
import itertools
import time
import joblib

# external packages
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import polars as pl
from shapely import union_all
from  dataclasses import dataclass, field, asdict
from dataclasses import replace
from typing_extensions import List, Dict
from scipy.optimize import curve_fit
from scipy import optimize
from scipy.stats import pearson3
# from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp



# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, print_to_logfile, checkpoint_to_logfile, get_bfs_from_ktnr, add_static_topo_data_to_subdf
from src.pvalloc.pvallocation_inst_selection_optimization import run_pv_selection_optimization, prepare_feedin_matrix, optimize_pv_selection_greedy , optimize_pv_selection_pulp

# settings
pl.Config.set_tbl_rows(30)  # Adjust 50 to whatever number of rows you want
pd.set_option('display.max_rows', 30) 

from src.pvalloc.pvalloc_core import PVAllocCore


class PVAllocGridOpt(PVAllocCore):
    """
    Grid-optimized installation ordering + expansion selection stage of a PV
    allocation scenario. Split out of MAIN_pvallocation.py (formerly
    PVAllocScenario.run_gridoptimized_orderinst and run_gridoptimized_expansion).
    """

    def run_gridoptimized_orderinst(self):
        """
        Input: 
            > PVAllocScenario class containing a data class (_Settings) which specifies all scenraio settings (e.g. also path to preprep data directory where all data 
                for theexecuted computations is stored).
        Output:
            > within the scenario name defined in pvalloc_settings, the grid optimal PV expansion is replicated by finding the optimal installations per grid node. 
            This is done by checking which installations lead to the highest PV production witihn each grid node. 

        """

        # SETUP -----------------------------------------------------------------------------
        self.sett.log_name = os.path.join(self.sett.name_dir_export_path, 'pvalloc_OptOrderinst_log.txt')
        self.sett.timing_marks_csv_path = os.path.join(self.sett.name_dir_export_path, 'timing_marks.csv')

        # create log file
        chapter_to_logfile(f'start MAIN_pvalloc_GridOptim_Orderinst for : {self.sett.name_dir_export}', self.sett.log_name, overwrite_file=True)
        print_to_logfile('*model allocation specifications*:', self.sett.log_name)

        print_to_logfile(f'> n_bfs_municipalities: {len(self.sett.bfs_numbers)} \n> n_trange_prediction: {self.sett.months_prediction}', self.sett.log_name)

        print_to_logfile(f'> pvalloc_settings, OptimExpa_{self.sett.name_dir_export}', self.sett.log_name)  
        for k, v in vars(self).items():              
            print_to_logfile(f'{k}: {v}', self.sett.log_name)

        start_opt_expansion = datetime.datetime.now()                       
        self.mark_to_timing_csv('OptExpa', 'START_OptExpa', start_opt_expansion, np.nan, '-')


        # CREATE MC DIR + TRANSFER INITIAL DATA FILES ----------------------------------------------
        # (identical to mcalgo, because then plots for visualization can be kept the same!)

        self.sett.optim_path = os.path.join(self.sett.name_dir_export_path, 'zMC1_OptimExpa')
        os.makedirs(self.sett.optim_path, exist_ok=True)

        fresh_initial_paths = [f'{self.sett.name_dir_export_path}/{file}' for file in self.sett.MCspec_fresh_initial_files]
        for f in fresh_initial_paths:
            file_name = os.path.basename(f) 
            shutil.copy(f, os.path.join(self.sett.optim_path, file_name))

        topo_time_paths = glob.glob(os.path.join(self.sett.name_dir_export_path, 'topo_time_subdf', '*.parquet'))
        for f in topo_time_paths:
            file_name = os.path.basename(f) 
            shutil.copy(f, os.path.join(self.sett.optim_path, file_name))

        # create grid node monitoring for >1HOY of lost load
        node_1hll_closed_dict = {'k_descrip': 'k: iter_round of algorightm, v: list of nodes that have >1HOY of lost load'}
        with open(f'{self.sett.optim_path}/node_1hll_closed_dict.json', 'w') as f:
            json.dump(node_1hll_closed_dict, f)

        # create grid node monitoring for subsidy
        node_subsidy_monitor_dict = {'k_descrip': 'k: iter_round of algorightm, v: list of nodes that have subsidy applied'}
        with open(f'{self.sett.optim_path}/node_subsidy_monitor_dict.json', 'w') as f:
            json.dump(node_subsidy_monitor_dict, f)


        # copy gridprem_ts to keep code structre below as is from npv_update
        shutil.copy(os.path.join(self.sett.name_dir_export_path, 'sanity_check_byEGID', 'gridprem_ts.parquet'), 
                        os.path.join(self.sett.optim_path, 'gridprem_ts.parquet'))
        shutil.copy(os.path.join(self.sett.name_dir_export_path, 'sanity_check_byEGID', 'gridnode_df.parquet'), 
                        os.path.join(self.sett.optim_path, 'gridnode_df.parquet'))



        # OPTIMIZATION SETUP -----------------------------------------------------------------------------    

        with open(f'{self.sett.optim_path}/topo_egid.json', 'r', encoding='utf-8') as f:
            topo = json.load(f)
        topo_time_paths = glob.glob(os.path.join(self.sett.optim_path, 'topo_subdf*.parquet'))
        # gridprem_ts = pl.read_parquet(f'{self.sett.optim_path}/gridprem_ts.parquet')
        gridnode_df = pl.read_parquet(f'{self.sett.optim_path}/gridnode_df.parquet')
        elecpri_vese, elecpri_entsoe = self.load_elecpri_inputs()
        pvtarif_vese, pvtarif_entsoe = self.load_pvtarif_inputs()


        rfr_model = joblib.load(f'{self.sett.calib_model_coefs}/{self.sett.ALGOspec_calib_estim_mod_name_pkl}_model.pkl')
        encoder   = joblib.load(f'{self.sett.calib_model_coefs}/{self.sett.ALGOspec_calib_estim_mod_name_pkl}_encoder.pkl')

        with open(f'{self.sett.optim_path}/node_1hll_closed_dict.json', 'r', encoding='utf-8') as f:
            node_1hll_closed_dict = json.load(f)
        with open(f'{self.sett.optim_path}/node_subsidy_monitor_dict.json', 'r', encoding='utf-8') as f:
            node_subsidy_monitor_dict = json.load(f)
                

        # filter topo / subdf + calculate NPV ======================
        topo_df_list = []
        for k, v in topo.items():
            row = {
                'EGID': k,
                'grid_node': v.get('grid_node'),
                }
            topo_df_list.append(row)
        topo_df = pl.DataFrame(topo_df_list)
        


        def single_gridoptimized_orderinst(self, gridnode_subsample, topo_df, topo_time_paths, topo, gridnode_df, elecpri_vese, elecpri_entsoe, pvtarif_vese, pvtarif_entsoe, rfr_model, encoder):  
            """ 
            Process a single grid optimization
            """

            topo_node_df = topo_df.filter(pl.col('grid_node') == gridnode_subsample).clone()
            node_egids = topo_node_df.select('EGID').to_series().to_list()
            

            node_subdf_list = []
            for i, path in enumerate(topo_time_paths):
                if i < 5 and self.sett.OPTIMspecs_gridnode_subsample != 'all_nodes_pyparallel':
                    print('OptExpa', f'extract + prep subdf, {i} / {len(topo_time_paths)} for node: {gridnode_subsample}' ) 

                subdf = pl.read_parquet(path)   
                node_subdf = subdf.filter(pl.col('EGID').is_in(node_egids))
                
                if node_subdf.shape[0] > 0:

                    # extend subdf with static data (safe disk space) --------------------
                    node_subdf = add_static_topo_data_to_subdf(node_subdf, topo)
                    if 'elecpri_Rp_kWh' in subdf.columns:
                        node_subdf = node_subdf.drop(['elecpri_Rp_kWh', ])
              
                    # merge gridprem_ts
                    # node_subdf = node_subdf.join(gridprem_ts[['t', 'grid_node', 'prem_Rp_kWh']], on=['t', 'grid_node'], how='left')  

                    # add static electricity price for random forest --------------------
                    node_subdf = self.add_elecpri_to_subdf(node_subdf, elecpri_vese, elecpri_entsoe)

                    # add static pvtarif for random forest --------------------
                    node_subdf = self.add_pvtarif_to_subdf(node_subdf, pvtarif_vese, pvtarif_entsoe)


                    for egid in list(node_subdf['EGID'].unique()):
                        egid_subdf = node_subdf.filter(pl.col('EGID') == egid).clone()

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
                            # pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),
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
                                # pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),

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
                        picked_flaeche = inst_power / (self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available)
                        remaining_flaeche = picked_flaeche

                        for i in range(0, topo_pick_df.shape[0]):
                            dfuid_flaeche = topo_pick_df['FLAECHE'].iloc[i]
                            dfuid_inst_power = dfuid_flaeche * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available

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
                            pl.col('bfs').first().alias('bfs'),
                            pl.col('grid_node').first().alias('grid_node'),
                            # pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                            # pl.col('pvtarif_Rp_kWh').first().alias('pvtarif_Rp_kWh'), 
                            # pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),
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
                        # egid_subdf.filter(pl.col('t').is_in(['t_10', 't_11', 't_12', 't_13'])).select(['EGID', 'df_uid', 'share_pvprod_used', 'poss_pvprod_kW', 'inst_TF', 'pvprod_kW', 't'])
                        # egid_agg.filter(pl.col('t').is_in(['t_10', 't_11'])).select(['EGID', 'poss_pvprod_kW', 'pvprod_kW', 't'])
                        # ----------------------------------


                        # calc selfconsumption
                        egid_agg = egid_agg.sort(['EGID', 't_int'], descending = [False, False])

                        selfconsum_expr = pl.min_horizontal([pl.col("pvprod_kW"), pl.col("demand_kW")]) * self.sett.TECspec_self_consumption_ifapplicable

                        egid_agg = egid_agg.with_columns([        
                            selfconsum_expr.alias("selfconsum_kW"),
                            (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
                            (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
                        ])


                        # add ACTUAL electricity price + pvtarif for NPV (per hour if the setting is a parquet) --------------------
                        egid_agg = self.add_elecpri_to_subdf(egid_agg, elecpri_vese, elecpri_entsoe, hourly=True)
                        egid_agg = self.add_pvtarif_to_subdf(egid_agg, pvtarif_vese, pvtarif_entsoe, hourly=True)

                        # calc econ spend/inc chf -----------------
                        egid_agg = egid_agg.with_columns([
                            ((pl.col("netfeedin_kW") * pl.col("pvtarif_Rp_kWh")) / 100 + (pl.col("selfconsum_kW") * pl.col("elecpri_Rp_kWh")) / 100).alias("econ_inc_chf")
                        ])
                        
                        egid_agg = egid_agg.with_columns([
                            (pl.lit(0.0)).alias("econ_spend_chf")
                        ])


                        # NPV calculation -----------------
                        estim_instcost_chfpkW, estim_instcost_chftotal = self.initial_sml_get_instcost_interpolate_function()
                        estim_instcost_chftotal(pd.Series([10, 20, 30, 40, 50, 60, 70]))

                        annual_cashflow = (egid_agg["econ_inc_chf"].sum() - egid_agg["econ_spend_chf"].sum())
                        installation_cost = estim_instcost_chftotal(pred_instPower)
                                
                        discount_factor = np.array([(1 + self.sett.TECspec_interest_rate)**-i for i in range(1, self.sett.TECspec_invst_maturity + 1)])
                        disc_cashflow = annual_cashflow * np.sum(discount_factor)
                        npv = -installation_cost + disc_cashflow

                        # egid_npv = egid_agg.group_by(['EGID', ]).agg([
                        #     pl.col('df_uid_winst').first().alias('df_uid_winst'),
                        #     pl.col('n_dfuid').first().alias('n_dfuid'),
                        #     pl.col('grid_node').first().alias('grid_node'),
                        #     pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                        #     pl.col('pvtarif_Rp_kWh').first().alias('pvtarif_Rp_kWh'), 
                        #     pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),
                        #     pl.col('TotalPower').first().alias('TotalPower'),
                        #     pl.col('AUSRICHTUNG_mean').first().alias('AUSRICHTUNG_mean'),
                        #     pl.col('NEIGUNG_mean').first().alias('NEIGUNG_mean'),
                        #     pl.col('FLAECHE').first().alias('FLAECHE'),
                            
                        #     pl.col('poss_pvprod_kW').sum().alias('poss_pvprod_kW'),
                        #     pl.col('demand_kW').sum().alias('demand_kW'),
                        #     pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                        #     pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                        #     pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                        #     pl.col('netdemand_kW').sum().alias('netdemand_kW'),
                        #     pl.col('econ_inc_chf').sum().alias('econ_inc_chf'),
                        #     pl.col('econ_spend_chf').sum().alias('econ_spend_chf'),
                        #     ])
                        egid_agg = egid_agg.with_columns([
                            pl.lit(pred_instPower).alias("pred_instPower"),
                            pl.lit(installation_cost).alias("estim_pvinstcost_chf"),
                            pl.lit(disc_cashflow).alias("disc_cashflow"),
                            pl.lit(npv).alias("NPV_uid"),
                            ])
                        
                        node_subdf_list.append(egid_agg)

            topo_node_subdf = pl.concat(node_subdf_list)


            # add roof specific filter tag VERSION 2 -----------------
            topo_filter_egid = self.get_topo_filter_tag_mapping(topo)
            # npv_df = npv_df.join(topo_filter_egid, on='EGID', how='left')
            topo_node_subdf = topo_node_subdf.join(topo_filter_egid, on='EGID', how='left')
            topo_node_df = topo_node_df.join(topo_filter_egid, on='EGID', how='left')

            # export node_subdf
            # topo_node_subdf.write_parquet(f'{self.sett.optim_path}/topo_node_subdf.parquet')
            # topo_node_df.write_parquet(f'{self.sett.optim_path}/topo_node_df.parquet')


            # Grid node OPTIMIZATION ======================

            # def run_pv_selection_optimization()
            os.makedirs(f'{self.sett.optim_path}/grid_optorder', exist_ok=True)
            for method in self.sett.OPTIMspecs_solving_method:
                # method                  = self.sett.OPTIMspecs_solving_method
                node_limit_kW           = gridnode_df.filter(pl.col('grid_node') == gridnode_subsample).select('kW_threshold').to_series()[0]
                selection_strategy      = self.sett.OPTIMspecs_greedy_selection_strategy
                peak_limit_kW           = node_limit_kW + self.sett.OPTIMspecs_peak_ext_limit_kW
                cumulative_limit_kWh    = self.sett.OPTIMspecs_cumulative_limit_kWh # left out for now, because it just means the threshold exceeding limit, 

                # Prepare data
                egid_to_idx, feedin_matrix, prep_stats = prepare_feedin_matrix(topo_node_subdf)
                
                # Run optimization
                if method == 'greedy':
                    print('OptExpa', f'start "greedy" optimization node:{gridnode_subsample}') #, datetime.datetime.now(), np.nan, '-')
                    selection_order, results = optimize_pv_selection_greedy(
                        feedin_matrix, egid_to_idx, peak_limit_kW, selection_strategy = selection_strategy
                    )
                elif method == 'pulp':
                    print('OptExpa', f'start "pulp" optimization node:{gridnode_subsample}') #, datetime.datetime.now(), np.nan, '-')
                    selection_order, results = optimize_pv_selection_pulp(
                        feedin_matrix, egid_to_idx, peak_limit_kW, 
                    )
                else:
                    raise ValueError(f"Unknown method: {method}. Use 'greedy' or 'pulp'")
                
                # Add preparation stats to results
                results['preparation_stats'] = prep_stats
                
                # Create installation order mapping from selected_egids sequence
                selection_order_egids = [k for k, v in selection_order.items()]
                selection_order_inst  = [v for k, v in selection_order.items()]
                inst_order_df = pl.DataFrame({
                    'EGID':                  selection_order_egids,
                    'grid_optim_inst_order': selection_order_inst
                })

                egid_within_limit = results['selected_egids']
                # egid_outside_limit = results['remaining_sorted']
                inst_order_df = inst_order_df.with_columns([
                    pl.when(pl.col('EGID').is_in(egid_within_limit))
                    .then(pl.lit('within_limit'))
                    .otherwise(pl.lit('outside_limit'))
                    .alias('selection_category')
                    ])
                #     pl.lit('within_limit').when(pl.col('EGID').is_in(egid_within_limit)).otherwise('outside_limit').alias('selection_category')
                # ])

                
                # Filter original dataframe to selected EGIDs and sort by selection order
                node_subdf_order_selected = topo_node_subdf.filter(pl.col('EGID').is_in(selection_order_egids)).join(
                    inst_order_df, on='EGID', how='left'
                ).sort('grid_optim_inst_order')


                node_select_df = node_subdf_order_selected.group_by('EGID').agg([
                    pl.col('grid_optim_inst_order').first().alias('grid_optim_inst_order'),
                    pl.col('df_uid_winst').first().alias('df_uid_winst'),
                    pl.col('n_dfuid').first().alias('n_dfuid'),
                    pl.col('grid_node').first().alias('grid_node'),
                    pl.col('TotalPower').first().alias('TotalPower'),
                    pl.col('FLAECHE').first().alias('FLAECHE'),
                    pl.col('pred_instPower').first().alias('pred_instPower'),
                    pl.col('estim_pvinstcost_chf').first().alias('estim_pvinstcost_chf'),
                    pl.col('NPV_uid').first().alias('NPV_uid'),

                    pl.col('poss_pvprod_kW').sum().alias('poss_pvprod_kW'),
                    pl.col('demand_kW').sum().alias('demand_kW'),
                    pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                    pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                    pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                    pl.col('netdemand_kW').sum().alias('netdemand_kW'),
                ])


                # Map roof spec filter tags back to node_select_df 
                node_select_df = node_select_df.join(topo_filter_egid, on='EGID', how='left')

                # export
                grid_optorder_df = node_select_df
                grid_optorder_df.write_parquet(f'{self.sett.optim_path}/grid_optorder/grid_optorder_df_{method}_node{gridnode_subsample}.parquet')
                if self.sett.export_csvs:
                    grid_optorder_df.write_csv(f'{self.sett.optim_path}/grid_optorder/grid_optorder_df_{method}_node{gridnode_subsample}.csv')
                

                with open(f'{self.sett.optim_path}/optim_results.txt', 'w') as f:
                    f.write(str(results))


        # RUN OPTIMIZATION -----------------------------------------------------------------------------
        if self.sett.OPTIMspecs_gridnode_subsample == 'max_node':
            gridnode_subsample = topo_df['grid_node'].value_counts().sort(by='count', descending=True).head(1)['grid_node'][0]
            single_gridoptimized_orderinst(self, gridnode_subsample, topo_df, topo_time_paths, topo, gridnode_df, elecpri_vese, elecpri_entsoe, pvtarif_vese, pvtarif_entsoe, rfr_model, encoder)
            
        elif self.sett.OPTIMspecs_gridnode_subsample == 'all_nodes_loop':
            for gridnode_subsample in list(topo_df['grid_node'].unique()):
                single_gridoptimized_orderinst(self, gridnode_subsample, topo_df, topo_time_paths, topo, gridnode_df, elecpri_vese, elecpri_entsoe, pvtarif_vese, pvtarif_entsoe, rfr_model, encoder)
                
        else:
            if self.sett.OPTIMspecs_gridnode_subsample == 'all_nodes_pyparallel':
                gridnode_subsample_list = list(topo_df['grid_node'].unique())
            else:
                gridnode_subsample_list = self.sett.OPTIMspecs_gridnode_subsample
                
            # in-python-parallelization
            n_workers = min(max(1, mp.cpu_count() - 1), len(gridnode_subsample_list))
            print(f'Processing {len(gridnode_subsample_list)} nodes with {n_workers} threads')
            
            futures = {}
            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                for node in gridnode_subsample_list:
                    fut = ex.submit(
                        single_gridoptimized_orderinst,  # nested function in scope
                        self, node, topo_df, topo_time_paths, topo, gridnode_df, elecpri_vese, elecpri_entsoe, pvtarif_vese, pvtarif_entsoe, rfr_model, encoder
                    )
                    futures[fut] = node

                completed = 0
                for fut in as_completed(futures):
                    print('')
                    node = futures[fut]
                    try:
                        fut.result()
                        completed += 1
                        print(f'Completed node: {node} ({completed}/{len(gridnode_subsample_list)})')
                    except Exception as e:
                        print(f'Error processing node {node}: {e}')

        chapter_to_logfile(f'end MAIN_pvalloc_GridOptim_Orderinst for : {self.sett.name_dir_export}', self.sett.log_name, overwrite_file=True)

                


    def run_gridoptimized_expansion(self):

        # SETUP -----------------------------------------------------------------------------
        self.sett.log_name = os.path.join(self.sett.name_dir_export_path, 'pvalloc_OptExpansion_log.txt')
        self.sett.timing_marks_csv_path = os.path.join(self.sett.name_dir_export_path, 'timing_marks.csv')

        # create log file
        chapter_to_logfile(f'start MAIN_pvalloc_GridOptim_Expansion for : {self.sett.name_dir_export}', self.sett.log_name, overwrite_file=True)
        print_to_logfile('*model allocation specifications*:', self.sett.log_name)

        self.sett.optim_path = os.path.join(self.sett.name_dir_export_path, 'zMC1_OptimExpa')
        # safety cap of the installation while loop: number of EGIDs in the topology (each EGID can get only one installation, so the cap only stops a malfunction)
        with open(f'{self.sett.name_dir_export_path}/topo_egid.json', 'r', encoding='utf-8') as f:
            safety_counter_max = len(json.load(f))
        print_to_logfile(f'> safety cap of the installation while loop: {safety_counter_max} (number of EGIDs in topo_egid.json)', self.sett.log_name)


        # CONCATENATE NPV_DF -----------------------------------------------------------------------------    
        optorder_paths = glob.glob(f'{self.sett.optim_path}/grid_optorder/grid_optorder_df_{self.sett.OPTEXPApecs_solving_method}_node*.parquet')
        npv_df_list = []
        for path in optorder_paths:
            npv_df_part = pl.read_parquet(path)
            npv_df_list.append(npv_df_part)
        npv_df = pl.concat(npv_df_list)
        npv_df.write_parquet(f'{self.sett.optim_path}/npv_df.parquet')
        npv_df.write_csv(f'{self.sett.optim_path}/npv_df.csv') if self.sett.export_csvs else None


        # ALLOCATION ALGORITHM -----------------------------------------------------------------------------    
        start_expan_algo = datetime.datetime.now()
        trange_prediction_df = pd.read_parquet(f'{self.sett.optim_path}/trange_prediction.parquet')
        trange_prediction = [m.date() for m in trange_prediction_df['date']]
        constrcapa = pd.read_parquet(f'{self.sett.optim_path}/constrcapa.parquet')


        for i_m, m in enumerate(trange_prediction):
            i_m = i_m + 1    

            print_to_logfile(f'\n-- n_iter {i_m} -- month {m} -- iter OptExpan -- {self.sett.name_dir_export} --', self.sett.log_name)
            start_allocation_month = datetime.datetime.now()
            with open(f'{self.sett.optim_path}/topo_egid.json', 'r', encoding='utf-8') as f:
                topo = json.load(f)
            egid_without_pv = [k for k,v in topo.items() if not v['pv_inst']['inst_TF']]


            # GRIDPREM + NPV_DF UPDATE ==========
            # update gridnode_df one last time, after npv_df is empty to still capture full effect of expansion
            start_time_update_gridprem = datetime.datetime.now()
            print_to_logfile('- START update gridprem', self.sett.log_name)
            self.algo_update_gridnode_AND_gridprem_POLARS(self.sett.optim_path, i_m, m)
            end_time_update_gridprem = datetime.datetime.now()
            print_to_logfile(f'- END update gridprem: {self.timediff_to_str_hhmmss(start_time_update_gridprem, end_time_update_gridprem)} (hh:mm:ss.s)', self.sett.log_name)

            if len(egid_without_pv) > 0:

                # check if any EGID w/o PV in open node
                with open(f'{self.sett.optim_path}/node_1hll_closed_dict.json', 'r', encoding='utf-8') as f:
                    node_1hll_closed_dict = json.load(f)
                closed_nodes = node_1hll_closed_dict[str(i_m)]['all_nodes_abv_1hll']
                closed_nodes_egid = [k for k, v in topo.items() if v.get('grid_node')  in closed_nodes ]
                egid_wo_pv_open_node = [egid for egid in egid_without_pv if egid not in closed_nodes_egid]


                # init constr capa + safety_counter ==========
                constr_built_m = 0
                if m.year != (m.year-1):
                    constr_built_y = 0
                constr_capa_m = constrcapa.loc[constrcapa['date'] == str(m), 'constr_capacity_kw'].iloc[0]
                constr_capa_y = constrcapa.loc[constrcapa['year'].isin([m.year]), 'constr_capacity_kw'].sum()

                safety_counter = 0 if len(egid_wo_pv_open_node) > 0 else safety_counter_max


                # INST PICK ==========
                start_time_installation_whileloop = datetime.datetime.now()
                inst_counter = 0
                print_to_logfile('- START inst while loop', self.sett.log_name)

                while( (constr_built_m <= constr_capa_m) & (constr_built_y <= constr_capa_y) & (safety_counter <= safety_counter_max) ):
                    with open(f'{self.sett.optim_path}/topo_egid.json', 'r', encoding='utf-8') as f:
                        topo = json.load(f)
                    with open(f'{self.sett.optim_path}/node_1hll_closed_dict.json', 'r', encoding='utf-8') as f:
                        node_1hll_closed_dict = json.load(f)

                    npv_df = pl.read_parquet(f'{self.sett.optim_path}/npv_df.parquet')

                    #  remove all EGIDs with pv ----------------
                    egid_without_pv = [k for k,v in topo.items() if not v['pv_inst']['inst_TF']]
                    npv_df = npv_df.filter(pl.col('EGID').is_in(egid_without_pv))
                    
                    #  remove all closed nodes EGIDs if applicable ----------------
                    if self.sett.GRIDspec_node_1hll_closed_TF:
                        closed_nodes = node_1hll_closed_dict[str(i_m)]['all_nodes_abv_1hll']
                        closed_nodes_egid = [k for k, v in topo.items() if v.get('grid_node')  in closed_nodes ]

                        npv_df = npv_df.filter(~pl.col('EGID').is_in(closed_nodes_egid)).clone()
                        # npv_df = copy.deepcopy(npv_df.loc[~npv_df['EGID'].isin(closed_nodes_egid)])
                        
                    npv_df_empty_TF = npv_df.shape[0] == 0
                    if npv_df_empty_TF:
                        safety_counter = safety_counter_max + 1

                    if not npv_df_empty_TF: 
                        inst_counter += 1
                        if self.sett.ALGOspec_pvinst_size_calculation == 'estim_rfr':
                            inst_power = self.algo_select_AND_adjust_topology_RFR(self.sett.optim_path, i_m, m, safety_counter, require_positive_npv = False)   # grid optimized order ignores NPV
                        else:
                            break


                    # Loop Exit + adjust constr_built capacity ----------
                    constr_built_m, constr_built_y, safety_counter  = constr_built_m + inst_power, constr_built_y + inst_power, safety_counter + 1
                    overshoot_rate                                  = self.sett.CSTRspec_constr_capa_overshoot_fact
                    constr_m_TF, constr_y_TF, safety_TF             = constr_built_m > constr_capa_m*overshoot_rate, constr_built_y > constr_capa_y, safety_counter > safety_counter_max

                    # print statements ----------
                    if any([constr_m_TF, constr_y_TF, safety_TF]):
                        print_str = 'exit while loop -> '
                        if constr_m_TF:
                            print_str += f'\n\t* exceeded constr_limit month (constr_m_TF:{constr_m_TF}), {round(constr_built_m,1)} of {round(constr_capa_m,1)} kW capacity built; '                    
                        if constr_y_TF:
                            print_str += f'\n\t* exceeded constr_limit year (constr_y_TF:{constr_y_TF}), {round(constr_built_y,1)} of {round(constr_capa_y,1)} kW capacity built; '
                        if safety_TF:
                            if npv_df_empty_TF:
                                print_str += f'\n\t* exceeded safety counter (safety_TF:{safety_TF}), NO MORE EGID to install PV on; '
                            else:
                                print_str += f'\n\t* exceeded safety counter (safety_TF:{safety_TF}), {safety_counter} rounds for safety counter max of: {safety_counter_max}; '
                        checkpoint_to_logfile(print_str, self.sett.log_name, 0, True)


            end_time_installation_whileloop = datetime.datetime.now()
            checkpoint_to_logfile(f'{inst_counter} installations installed', self.sett.log_name, 0, True)
            print_to_logfile(f'- END inst while loop: {self.timediff_to_str_hhmmss(start_time_installation_whileloop, end_time_installation_whileloop)} (hh:mm:ss.s)', self.sett.log_name)
            self.mark_to_timing_csv('GridOptim Expansion', f'end inst_whileloop_{i_m}', end_time_installation_whileloop, self.timediff_to_str_hhmmss(start_time_installation_whileloop, end_time_installation_whileloop),  '-')  #if i_m < 7 else None
            checkpoint_to_logfile(f'end month allocation, runtime: {datetime.datetime.now() - start_allocation_month} (hh:mm:ss.s)', self.sett.log_name, 0, True)                    


        # END ---------------------------------------------------
        end_expan_algo = datetime.datetime.now()
        self.mark_to_timing_csv('MCalgo', 'END_MC_algo', end_expan_algo, self.timediff_to_str_hhmmss(start_expan_algo, end_expan_algo),  '-')

        if 'start_total_runtime' in dir(self.sett): 
            self.mark_to_timing_csv('TOTAL', 'END_total_runtime', end_expan_algo, self.timediff_to_str_hhmmss(self.sett.start_total_runtime, end_expan_algo),  '-')
            os.rename(self. sett.timing_marks_csv_path, f'{self.sett.timing_marks_csv_path.split(".cs")[0]}_{self.sett.name_dir_export}.csv')

        chapter_to_logfile(f'end MAIN_pvalloc_GridOptim_Expansion for : {self.sett.name_dir_export} \n Runtime (hh:mm:ss):{datetime.datetime.now() - start_expan_algo}', self.sett.log_name, overwrite_file=False)

