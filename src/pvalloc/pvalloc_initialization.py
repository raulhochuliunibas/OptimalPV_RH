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
from pathlib import Path

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


class PVAllocInitialization(PVAllocCore):
    """
    Initialization stage of a PV allocation scenario: builds the topology,
    calculates economics and runs sanity checks. Split out of MAIN_pvallocation.py
    (formerly PVAllocScenario.run_pvalloc_initialization and its exclusive helpers).
    """

    def run_pvalloc_initialization(self):
        """
        Input:
            > PVAllocScenario class containing a data class (_Settings) which specifies all scenraio settings (e.g. also path to preprep data directory where all data for the
              executed computations is stored).

        Output (no function return but export to dir):
            > directory renamed after scenario name (name_dir_export), containing all data files form the INITIALIZATION of the pv allocation run.

        Description: 
            > Depending on the settings, certain steps of the model initalization can be run. (Debug function to only run certain steps, based on interim 
            file exports to save time).
            > First the prepared data (geo and time series) from the preprep_[scenario] directory is imported (based on sencario selection criteria) 
            and a topology is created (dict with EGID as keys, containing all relevant information for each individual house).
            > Then the a the future construction capacity for each month is defined (based on the scenario settings and past construction volume (kWP 
            in the smaple area and time window)).
            > Next, the topology dictionary is transformed to a dataframe, to then be merged with the radiation time series. This step is necessary, as 
            I consider individual roof parts for each hour of the year). The total radiation potential per roof partition is calculated. This huge data
            frame is then partitioned into smaller subfiles to be "operatable" by my python IDE  economic components. This iterative subfile strucutre can
            be "switched off" (set n houses per subfile large enough) for larger computers or high performance computing clusters.
            > The next scetion of the MAIN file runs a number of sanity checks on the initalization of the pv allocation run. 
            - The first check runs the allocation algorithm (identical to later Monte Carlo iterations), to extract plots and visualiations, accessible already
                after only a few monthly iterations. 
            - Another check exports all the relevant data from the topo dict and the economic components for each house to an xlsx file for comparison. 
            - Another check runs a simple check for multiple installations per EGID (which should not happen in the current model).
            > The final step is to copy all relevant files to the output directory, which is then renamed after the scenario name.

        """
        # SETUP ---------------------------------------------------------------------------------------------
        self.sett.start_total_runtime = datetime.datetime.now()
        self.sett.log_name = os.path.join(self.sett.name_dir_export_path, 'pvalloc_Initial_log.txt')
        self.sett.summary_name = os.path.join(self.sett.name_dir_export_path, 'summary_data_selection_log.txt')
        self.sett.timing_marks_csv_path = os.path.join(self.sett.name_dir_export_path, 'timing_marks.csv')

        self.sett.bfs_numbers: List[str] = get_bfs_from_ktnr(self.sett.kt_numbers, self.sett.data_path, self.sett.log_name) if self.sett.kt_numbers != [] else [str(bfs) for bfs in self.sett.bfs_numbers]

        # create dir for export, rename old export dir not to overwrite
        if os.path.exists(self.sett.name_dir_export_path):
            export_name     = self.sett.name_dir_export
            export_to_path  = self.sett.name_dir_export_path.split(export_name)[0]
            n_same_names = len(glob.glob(f'{export_to_path}*{export_name}*'))
            while os.path.exists(f'{self.sett.data_path}/pvalloc/x_{self.sett.name_dir_export}_{n_same_names}_old_vers'):
                n_same_names +=1
            os.rename(self.sett.name_dir_export_path, f'{self.sett.data_path}/pvalloc/x_{self.sett.name_dir_export}_{n_same_names}_old_vers')
            
        os.makedirs(self.sett.name_dir_export_path, exist_ok=True)

        # export class instance settings to dir        
        self.export_pvalloc_scen_settings()


        # export slurm ID to find log files on HPC
        if hasattr(self.sett, 'slurm_full_id'):
            with open(f'{self.sett.name_dir_export_path}/0_HPC_job_{self.sett.slurm_full_id}.txt', 'w') as f:
                f.write(f'Combined: {self.sett.slurm_full_id}\n')
        if hasattr(self.sett, 'pvalloc_scen_index'):
            with open(f'{self.sett.name_dir_export_path}/0_HPC_job_{self.sett.slurm_full_id}.txt', 'a') as f:
                f.write(f'PVAlloc Scenario Index: {self.sett.pvalloc_scen_index}\n')


        # create log file
        chapter_to_logfile(f'start MAIN_pvalloc_INITIALIZATION for: {self.sett.name_dir_export}', self.sett.log_name, overwrite_file=True)
        subchapter_to_logfile('pvalloc_settings', self.sett.log_name)
        for k, v in vars(self).items():
            print_to_logfile(f'{k}: {v}', self.sett.log_name)

        # create summary file + Timing file
        chapter_to_logfile(f'OptimalPV Sample Summary of Building Topology, scen: {self.sett.name_dir_export}', self.sett.summary_name, overwrite_file=True)

        # create timing cwith open(src, 'rb') as fsrc:v
        start_initalization = datetime.datetime.now()
        self.mark_to_timing_csv
        self.mark_to_timing_csv('init', 'END_INIT', start_initalization, np.nan, '-')



        # CREATE TOPOLOGY ---------------------------------------------------------------------------------------------
        subchapter_to_logfile('initialization: CREATE SMALLER AID DFs', self.sett.log_name)
        start_create_topo = datetime.datetime.now()
        self.mark_to_timing_csv('init', 'start_create_topo', start_create_topo, np.nan, '-'),

        self.initial_sml_HOY_weatheryear_df()
        self.initial_sml_get_DSO_nodes_df_AND_ts()
        self.initial_sml_iterpolate_instcost_function()

        if self.sett.recreate_topology:
            subchapter_to_logfile('initialization: IMPORT PREPREP DATA & CREATE (building) TOPOLOGY', self.sett.log_name)
            topo, df_list, df_names = self.initial_lrg_import_preprep_AND_create_topology()
            self.approx_outtopo_df_griddemand()

            subchapter_to_logfile('initialization: IMPORT TS DATA', self.sett.log_name)
            ts_list, ts_names = self.initial_lrg_import_ts_data()

            subchapter_to_logfile('initialization: DEFINE T-RANGE PREDICTION', self.sett.log_name)
            trange_prediction, months_lookback = self.initial_define_trange_prediction()

            subchapter_to_logfile('initialization: DEFINE CONSTRUCTION CAPACITY', self.sett.log_name)
            if self.sett.CSTRspec_ext_constrcapa_path is None:
                self.initial_lrg_define_construction_capacity(topo = topo, 
                                                              trange_prediction = trange_prediction, months_lookback = months_lookback,
                                                              df_list=df_list, df_names=df_names, ts_list=ts_list, ts_names=ts_names)
            elif self.sett.CSTRspec_ext_constrcapa_path is not None:
                self.initial_copy_ext_construction_capacity_to_dir()

            end_create_topo = datetime.datetime.now()
            self.mark_to_timing_csv('init', 'end_create_topo', end_create_topo, self.timediff_to_str_hhmmss(start_create_topo, end_create_topo), '-')
        

        # CALC ECONOMICS + (OUT)TOPO_TIME SPECIFIC DFs ---------------------------------------------------------------------------------------------
        subchapter_to_logfile('prep: CALC ECONOMICS for TOPO_DF', self.sett.log_name)
        start_calc_economics = datetime.datetime.now()
        self.mark_to_timing_csv('init', 'start_calc_economics', start_calc_economics, np.nan, '-')
        
        # algo.calc_economics_in_topo_df(self, topo, df_list, df_names, ts_list, ts_names)
        self.algo_calc_production_in_topo_df_AND_topo_time_subdf(topo, df_list, df_names, ts_list, ts_names)
        shutil.copy(f'{self.sett.name_dir_export_path}/topo_egid.json', f'{self.sett.name_dir_export_path}/topo_egid_before_alloc.json')

        end_calc_economics = datetime.datetime.now()
        self.mark_to_timing_csv('init', 'end_calc_economics', end_calc_economics, self.timediff_to_str_hhmmss(start_calc_economics, end_calc_economics),  '-')


 
        # SANITY CHECK: CALC FEW ITERATION OF NPV AND FEEDIN for check ---------------------------------------------------------------
        subchapter_to_logfile('sanity_check: RUN FEW ITERATION for byCHECK', self.sett.log_name)
        start_sanity_check = datetime.datetime.now()
        self.mark_to_timing_csv('init', 'start_sanity_check', start_sanity_check, np.nan, '-')

        # make sanitycheck folder and move relevant initial files there (delete all old files, not distort results)
        if os.path.exists(self.sett.sanity_check_path):
            shutil.rmtree(self.sett.sanity_check_path)
        os.makedirs(self.sett.sanity_check_path)

        fresh_initial_files = [os.path.join(self.sett.name_dir_export_path, file) for file in self.sett.MCspec_fresh_initial_files] 
        for f in fresh_initial_files:
            file_name = os.path.basename(f) #f.split('\\')[-1]
            shutil.copy(f, os.path.join(self.sett.sanity_check_path, file_name))

        topo_time_paths = glob.glob(os.path.join(self.sett.name_dir_export_path, 'topo_time_subdf', '*.parquet'))
        for f in topo_time_paths:
            file_name = os.path.basename(f) #f.split('\\')[-1]
            shutil.copy(f, os.path.join(self.sett.sanity_check_path, file_name))

        # create grid node monitoring for >1HOY of lost load
        node_1hll_closed_dict = {'k_descrip': 'k: iter_round of algorightm, v: list of nodes that have >1HOY of lost load'}
        with open(f'{self.sett.sanity_check_path}/node_1hll_closed_dict.json', 'w') as f:
            json.dump(node_1hll_closed_dict, f)

        # create grid node monitoring for subsidy
        node_subsidy_monitor_dict = {'k_descrip': 'k: iter_round of algorightm, v: list of nodes that have subsidy applied'}
        with open(f'{self.sett.sanity_check_path}/node_subsidy_monitor_dict.json', 'w') as f:
            json.dump(node_subsidy_monitor_dict, f)


                
        # ALLOCATION RUN ====================
        start_sanity_check_allocation = datetime.datetime.now()
        dfuid_installed_list = []
        pred_inst_df = pd.DataFrame()
        trange_prediction_df = pd.read_parquet(f'{self.sett.name_dir_export_path}/trange_prediction.parquet')
        trange_prediction = [str(m.date()) for m in trange_prediction_df['date']]

        # Load models once before loop to avoid repeated memory allocation
        if self.sett.ALGOspec_pvinst_size_calculation == 'estim_rfr':
            rfr_model = joblib.load(f'{self.sett.calib_model_coefs}/{self.sett.ALGOspec_calib_estim_mod_name_pkl}_model.pkl')
            encoder   = joblib.load(f'{self.sett.calib_model_coefs}/{self.sett.ALGOspec_calib_estim_mod_name_pkl}_encoder.pkl')
        else:
            rfr_model, encoder = None, None

        for i_m, m in enumerate(trange_prediction[0:self.sett.CHECKspec_n_iterations_before_sanitycheck]):
            i_m = i_m + 1
            print_to_logfile(f'\n-- month {m} -- sanity check -- {self.sett.name_dir_export} --', self.sett.log_name)
            self.algo_update_gridnode_AND_gridprem_POLARS(self.sett.sanity_check_path, i_m, m)

            if self.sett.ALGOspec_pvinst_size_calculation == 'estim_rfr':
                self.algo_update_npv_df_RFR(self.sett.sanity_check_path, i_m, m, rfr_model, encoder)
                self.algo_select_AND_adjust_topology_RFR(self.sett.sanity_check_path, i_m, m)
            else:
                print_to_logfile('ERROR: specified "ALGOspec_pvinst_size_calculation" is not available', self.sett.log_name)
                break           
            # if self.sett.ALGOspec_pvinst_size_calculation == 'inst_full_partition':
            #     self.algo_update_npv_df_POLARS(self.sett.sanity_check_path, i_m, m)
            #     self.algo_select_AND_adjust_topology(self.sett.sanity_check_path, i_m, m)
            # elif self.sett.ALGOspec_pvinst_size_calculation == 'npv_optimized':
            #     self.algo_update_npv_df_OPTIMIZED(self.sett.sanity_check_path, i_m, m)
            #     self.algo_select_AND_adjust_topology_OPTIMIZED(self.sett.sanity_check_path, i_m, m)                
            # elif self.sett.ALGOspec_pvinst_size_calculation == 'estim_rf_segdist':
            #     self.algo_update_npv_df_RF_SEGMDIST(self.sett.sanity_check_path, i_m, m)
            #     self.algo_select_AND_adjust_topology_RFR(self.sett.sanity_check_path, i_m, m)


        end_sanity_check_allocation = datetime.datetime.now()
        self.mark_to_timing_csv('init', 'end_sanity_check_allocation', end_sanity_check_allocation, self.timediff_to_str_hhmmss(start_sanity_check_allocation, end_sanity_check_allocation), '-')
        
        start_sanity_check_summary_by_EGID = datetime.datetime.now()
        self.sanity_check_summary_byEGID(self.sett.sanity_check_path )

        end_sanity_check_summary_by_EGID = datetime.datetime.now()
        self.mark_to_timing_csv('init', 'end_sanity_check_summary_by_EGID', end_sanity_check_summary_by_EGID, self.timediff_to_str_hhmmss(start_sanity_check_summary_by_EGID, end_sanity_check_summary_by_EGID), '-')
        
        # EXPORT SPATIAL DATA ====================
        if self.sett.create_gdf_export_of_topology:
            start_sanity_check_export_spatial_data = datetime.datetime.now()

            subchapter_to_logfile('sanity_check: CREATE SPATIAL EXPORTS OF TOPOLOGY_DF', self.sett.log_name)
            self.sanity_create_gdf_export_of_topo()

            subchapter_to_logfile('sanity_check: MULTIPLE INSTALLATIONS PER EGID', self.sett.log_name)
            self.sanity_check_multiple_xtf_ids_per_EGID()

            end_sanity_check_export_spatial_data = datetime.datetime.now()
            self.mark_to_timing_csv('init', 'end_sanity_check_export_spatial_data', end_sanity_check_export_spatial_data, self.timediff_to_str_hhmmss(start_sanity_check_export_spatial_data, end_sanity_check_export_spatial_data), '-')
        

        # CLEANUP ====================
        self.sanity_check_cleanup_obsolete_data()


        end_sanity_check = datetime.datetime.now()
        self.mark_to_timing_csv('init', 'end_sanity_check', end_sanity_check, self.timediff_to_str_hhmmss(start_sanity_check, end_sanity_check), '-')


        # END ---------------------------------------------------
        end_initalization = datetime.datetime.now()
        self.mark_to_timing_csv('init', 'END_INIT', end_initalization, self.timediff_to_str_hhmmss(start_initalization, end_initalization), '-')

        chapter_to_logfile(f'end MAIN_pvalloc_INITIALIZATION for: {self.sett.name_dir_export}\n Runtime (hh:mm:ss):{datetime.datetime.now() - start_initalization}', self.sett.log_name)


    # ------------------------------------------------------------------------------------------------------
    # MONTE CARLO ALGORITHM of PV Allocatoin Scenario
    # ------------------------------------------------------------------------------------------------------


    # ------------------------------------------------------------------------------------------------------
    # INITIALIZATION helpers
    # ------------------------------------------------------------------------------------------------------
    def algo_calc_production_in_topo_df_AND_topo_time_subdf(self, 
                                       topo, 
                                       df_list, df_names, 
                                       ts_list, ts_names,
    ): 
                    
        # setup -----------------------------------------------------
        print_to_logfile('run function: algo_calc_production_in_topo_df_AND_topo_time_subdf', self.sett.log_name)


        # import -----------------------------------------------------
        angle_tilt_df = df_list[df_names.index('angle_tilt_df')]
        solkat_month = df_list[df_names.index('solkat_month')]
        demandtypes_ts = ts_list[ts_names.index('demandtypes_ts')]
        meteo_ts = ts_list[ts_names.index('meteo_ts')]

        angle_tilt_df = pl.DataFrame(angle_tilt_df.reset_index())  # If it was a multi-index
        solkat_month = pl.DataFrame(solkat_month.reset_index())  
        demandtypes_ts = pl.DataFrame(demandtypes_ts.reset_index())
        meteo_ts = pl.DataFrame(meteo_ts.reset_index())


        # TOPO to DF =============================================
        # solkat_combo_df_exists = os.path.exists(f'{pvalloc_settings["interim_path"]}/solkat_combo_df.parquet')
        # if pvalloc_settings['recalc_economics_topo_df']:

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
                    'GAREA': gwr_info.get('garea'),
                    'GBAUJ': gwr_info.get('gbaubj'),
                    'GSTAT': gwr_info.get('gstat'),
                    'GWAERZH1': gwr_info.get('gwaerzh1'),
                    'GENH1': gwr_info.get('genh1'),
                    'sfhmfh_typ': gwr_info.get('sfhmfh_typ'),
                    'demand_arch_typ': v.get('demand_arch_typ'),
                    'demand_elec_pGAREA': v.get('demand_elec_pGAREA'),
                    'grid_node': v.get('grid_node'),

                    'inst_TF': pv_inst.get('inst_TF'),
                    'info_source': pv_inst.get('info_source'),
                    'pvid': pv_inst.get('xtf_id'),
                    'pvtarif_Rp_kWh': v.get('pvtarif_Rp_kWh'),
                    'TotalPower': pv_inst.get('TotalPower'),
                    'dfuid_w_inst_tuples': pv_inst.get('dfuid_w_inst_tuples'),

                    'FLAECHE': v_p.get('FLAECHE'),
                    'AUSRICHTUNG': v_p.get('AUSRICHTUNG'),
                    'STROMERTRAG': v_p.get('STROMERTRAG'),
                    'NEIGUNG': v_p.get('NEIGUNG'),
                    'MSTRAHLUNG': v_p.get('MSTRAHLUNG'),
                    'GSTRAHLUNG': v_p.get('GSTRAHLUNG'),
                    'elecpri_Rp_kWh': v.get('elecpri_Rp_kWh'),
                }
                rows.append(row)

        topo_df = pl.DataFrame(rows)
            

        # make or clear dir for subdfs ----------------------------------------------
        subdf_path = f'{self.sett.name_dir_export_path}/topo_time_subdf'

        if not os.path.exists(subdf_path):
            os.makedirs(subdf_path)
        else:
            old_files = glob.glob(f'{subdf_path}/*')
            for f in old_files:
                os.remove(f)
            

        # round NEIGUNG + AUSRICHTUNG to 5 for easier computation
        topo_df = topo_df.with_columns([
            # Round and cast to int64 for NEIGUNG and AUSRICHTUNG
            ((pl.col("NEIGUNG") / 5).round(0) * 5).cast(pl.Int64).alias("NEIGUNG"),
            ((pl.col("AUSRICHTUNG") / 10).round(0) * 10).cast(pl.Int64).alias("AUSRICHTUNG")
        ])        
        angle_tilt_df = angle_tilt_df.rename({'angle': 'AUSRICHTUNG', 'tilt': 'NEIGUNG'})
        topo_df = topo_df.join(
            angle_tilt_df,
            on=["AUSRICHTUNG", "NEIGUNG"],
            how="left"
        )

        # # transform TotalPower
        # topo_df = topo_df.with_columns([
        #     pl.col("TotalPower")
        #     .cast(str)                         # Ensure it's treated as string first
        #     .str.replace_all("", "0")         # Replace empty string with "0"
        #     .cast(pl.Float64)                 # Convert to float
        # ])

        # ===========================================================                    
        # I  MERGE WEATHER DATA - CALC PRODUCTION PER PARTITION 
        # II ADJUST INST SETTINGS FOR EXISITN PV (INST_TF / SOURCE)  
        # =========================================================== 
        topo_subdf_partitioner = self.sett.ALGOspec_topo_subdf_partitioner
            
        share_roof_area_available = self.sett.TECspec_share_roof_area_available
        inverter_efficiency       = self.sett.TECspec_inverter_efficiency
        panel_efficiency          = self.sett.TECspec_panel_efficiency
        pvprod_calc_method        = self.sett.TECspec_pvprod_calc_method
        kWpeak_per_m2             = self.sett.TECspec_kWpeak_per_m2

        flat_direct_rad_factor  = self.sett.WEAspec_flat_direct_rad_factor
        flat_diffuse_rad_factor = self.sett.WEAspec_flat_diffuse_rad_factor

        subdf_pvdf_agg_before_list = [] 
        subdf_pvdf_agg_after_list = []    
        tranche_counter = 0
        egids = topo_df['EGID'].unique()
        stepsize = topo_subdf_partitioner if len(egids) > topo_subdf_partitioner else len(egids)
        checkpoint_to_logfile(' * * DEBUGGIGN * * *: START loop subdfs', self.sett.log_name, 0)
        for i in range(0, len(egids), stepsize):

            tranche_counter += 1
            # print_to_logfile(f'-- merges to topo_time_subdf {tranche_counter}/{len(range(0, len(egids), stepsize))} tranches ({i} to {i+stepsize-1} egids.iloc) ,  {7*"-"}  (stamp: {datetime.now()})', self.sett.log_name)
            subdf = topo_df.filter(pl.col("EGID").is_in(list(egids[i:i+stepsize]))).clone()


            # I  MERGE WEATHER DATA - CALC PRODUCTION PER PARTITION ===========================================================
            if True: 
                # merge production, grid prem + demand to partitions --------------------
                subdf = subdf.with_columns(pl.lit("Basel").alias("meteo_loc"))          
                meteo_ts = meteo_ts.with_columns(pl.lit("Basel").alias("meteo_loc"))    
                    
                # subdf = subdf.merge(meteo_ts[['rad_direct', 'rad_diffuse', 'temperature', 't', 'meteo_loc']], how='left', on='meteo_loc')
                subdf = subdf.join(meteo_ts, on="meteo_loc", how="left")  

                # add date specific columns
                subdf = subdf.with_columns([
                    pl.col("timestamp").dt.month().cast(pl.Int32).alias("month")
                ])
                        
                    

                # add radiation per h to subdf, "flat" OR "dfuid_ind" --------------------
                if self.sett.WEAspec_radiation_to_pvprod_method == 'flat':
                    subdf = subdf.with_columns(
                    (pl.col("rad_direct") * flat_direct_rad_factor + pl.col("rad_diffuse") * flat_diffuse_rad_factor)
                    .alias("radiation")
                    )
                    meteo_ts = meteo_ts.with_columns(
                        (pl.col("rad_direct") * flat_direct_rad_factor + pl.col("rad_diffuse") * flat_diffuse_rad_factor)
                        .alias("radiation")
                    )
                    mean_top_radiation = meteo_ts.sort("radiation", descending=True).select(
                        pl.col("radiation").head(10).mean()
                    ).item()  # `.item()` extracts scalar value from single-element DataFrame

                    subdf = subdf.with_columns(
                        (pl.col("radiation") / mean_top_radiation).alias("radiation_rel_locmax")
                    )
                        

                elif self.sett.WEAspec_radiation_to_pvprod_method == 'dfuid_ind':
                    if 'DF_UID' in solkat_month.columns:
                        solkat_month = solkat_month.rename({"DF_UID": "df_uid"})
                    if 'MONAT' in solkat_month.columns:
                        solkat_month = solkat_month.rename({"MONAT": "month"})

                    checkpoint_to_logfile(f'  start merge solkat_month to subdf {i} to {i+stepsize-1}', self.sett.log_name, 0) if i < 2 else None
                    subdf = subdf.join(
                        solkat_month.select(["df_uid", "month", "A_PARAM", "B_PARAM", "C_PARAM"]),
                        on=["df_uid", "month"],
                        how="left"
                    )   
                    checkpoint_to_logfile(f'  end merge solkat_month to subdf {i} to {i+stepsize-1}', self.sett.log_name, 0) if i < 2 else None
                    subdf = subdf.with_columns([
                        (
                            pl.col("A_PARAM") * pl.col("rad_direct") +
                            pl.col("B_PARAM") * pl.col("rad_diffuse") +
                            pl.col("C_PARAM")
                        ).alias("radiation")
                    ])
                    # some radiation values are negative, because of the linear transformation with abc parameters. 
                    # force all negative values to 0
                    subdf = subdf.with_columns([
                        pl.when((pl.col("rad_direct") == 0) & (pl.col("rad_diffuse") == 0))
                        .then(0.0)
                        .when(pl.col("radiation") < 0)
                        .then(0.0)
                        .otherwise(pl.col("radiation"))
                        .alias("radiation")
                    ])

                    meteo_ts = meteo_ts.with_columns([
                        (pl.col("rad_direct") * flat_direct_rad_factor + pl.col("rad_diffuse") * flat_diffuse_rad_factor)
                        .alias("radiation")
                    ])


                    # radiation_rel_locmax by "df_uid_specific" vs "all_HOY" ---------- 
                    if self.sett.WEAspec_rad_rel_loc_max_by == 'dfuid_specific':
                        subdf_dfuid_topradation = (
                            subdf
                            .group_by("df_uid")
                            .agg([
                                pl.col("radiation").top_k(10).mean().alias("mean_top_radiation")
                            ])
                        )
                        subdf = subdf.join(subdf_dfuid_topradation, on="df_uid", how="left")
                        subdf = subdf.with_columns([
                            (pl.col("radiation") / pl.col("mean_top_radiation")).alias("radiation_rel_locmax")
                        ])

                    elif self.sett.WEAspec_rad_rel_loc_max_by == 'all_HOY':
                        mean_nlargest_rad_all_HOY = (
                            meteo_ts
                            .select(pl.col("radiation").top_k(10).mean())
                            .item()
                        )
                        subdf = subdf.with_columns([
                            (pl.col("radiation") / mean_nlargest_rad_all_HOY).alias("radiation_rel_locmax")
                        ])
                    

                # add panel_efficiency by time --------------------
                if self.sett.PEFspec_variable_panel_efficiency_TF:
                    summer_months      = self.sett.PEFspec_summer_months
                    hotsummer_hours    = self.sett.PEFspec_hotsummer_hours
                    hot_hours_discount = self.sett.PEFspec_hot_hours_discount

                    HOY_weatheryear_df = pl.read_parquet(f'{self.sett.name_dir_export_path}/HOY_weatheryear_df.parquet')
                    hot_hours_in_year = HOY_weatheryear_df.filter(
                        pl.col("month").is_in(summer_months) & pl.col("hour").is_in(hotsummer_hours)
                    )
                    hot_t_set = set(hot_hours_in_year["t"].to_list())

                    subdf = subdf.with_columns([
                        pl.when(pl.col("t").is_in(hot_t_set))
                        .then(panel_efficiency * (1 - hot_hours_discount))
                        .otherwise(panel_efficiency)
                        .alias("panel_efficiency")
                    ])
                elif self.sett.PEFspec_variable_panel_efficiency_TF:
                    subdf = subdf.with_columns([
                        pl.lit(panel_efficiency).alias("panel_efficiency")
                    ])
                        

                # attach / calculate demand profiles --------------------
                demandtypes_unpivot = demandtypes_ts.unpivot(
                    on = ['SFH', 'MFH', ],
                    index=['t', 't_int'],  # col that stays unchanged
                    value_name='demand_profile',  # name of the column that will hold the values
                    variable_name='sfhmfh_typ'  # name of the column that will hold the original column names
                )
                subdf = subdf.join(demandtypes_unpivot, on=['t', 'sfhmfh_typ'], how="left")
                subdf = subdf.with_columns([
                    (pl.col("demand_elec_pGAREA") * pl.col("demand_profile") * pl.col("GAREA") * self.sett.ALGOspec_tweak_demand_profile ).alias("demand_kW")  # convert to kW
                ])



                # add heatpump to demand profiles --------------------
                if self.sett.TECspec_add_heatpump_demand_TF:
                    # classification heating system 
                    if 'GWAERZH2' in subdf.columns:
                        subdf = subdf.with_columns([
                            pl.when( (pl.col('GWAERZH1').is_in(self.sett.TECspec_heatpump_indicator)) | 
                                    (pl.col('GWAERZH2').is_in(self.sett.TECspec_heatpump_indicator)) )
                                .then(pl.lit('heatpump'))
                                .otherwise(pl.lit('no_heatpump'))
                                .alias('heating_system')
                        ])
                    else:
                        subdf = subdf.with_columns([
                            pl.when( pl.col('GWAERZH1').is_in(self.sett.TECspec_heatpump_indicator) )
                                .then(pl.lit('heatpump'))
                                .otherwise(pl.lit('no_heatpump'))
                                .alias('heating_system')
                        ])

                        
                    subdf = subdf.with_columns([
                        pl.col('demand_kW').alias('demand_sfhmfh_kW'), 
                        pl.col('demand_kW').alias('demand_appliances_kW'), 
                    ])
                        
                    month_list, factor_list = [], []
                    for tupl in self.sett.TECspec_heatpump_months_factor:
                        month_list.append(tupl[0])
                        factor_list.append(tupl[1])
                    heatpump_months_factor = pl.DataFrame({'month': month_list, 'heatpump_factor': factor_list})
                        
                    subdf = subdf.join(heatpump_months_factor, on='month', how='left')
                    subdf = subdf.with_columns([
                        (pl.col('demand_sfhmfh_kW') * pl.col('heatpump_factor')).alias('demand_heatpump_kW'),
                    ])
                    subdf = subdf.with_columns([
                        # (pl.col('demand_sfhmfh_kW') + pl.col('demand_heatpump_kW')).alias('demand_kW'),
                        pl.when(pl.col('heating_system') == 'heatpump')
                            .then(pl.col('demand_sfhmfh_kW') + pl.col('demand_heatpump_kW'))
                            .otherwise(pl.col('demand_appliances_kW'))
                            .alias('demand_kW')
                    ])
                    subdf = subdf.drop('heatpump_factor')



                # attach FLAECH_angletilt, might be usefull for later calculations
                subdf = subdf.with_columns([
                    (pl.col("FLAECHE") * pl.col("efficiency_factor")).alias("FLAECH_angletilt")
                ])


                # compute production ---------- 
                # pvprod method 1 (false, presented to frank 8.11.24. missing efficiency grade)
                if pvprod_calc_method == "method1":
                    subdf = subdf.with_columns([
                        (pl.col("radiation") * pl.col("FLAECHE") * pl.col("angletilt_factor") / 1000)
                        .alias("pvprod_kW")
                    ])
                    formla_for_log_print = "method1"
                    
                # pvprod method 2.1
                elif pvprod_calc_method == "method2.1":
                    subdf = subdf.with_columns([
                        (
                            (pl.col("radiation") / 1000) * pl.col("panel_efficiency") * inverter_efficiency * share_roof_area_available *  pl.col("FLAECHE") * pl.col("efficiency_factor")
                        ).alias("pvprod_kW")
                    ])
                    formla_for_log_print = "inPOLARS: subdf['pvprod_kW'] = subdf['radiation'] / 1000 * subdf['panel_efficiency'] * inverter_efficiency * share_roof_area_available *  subdf['FLAECHE'] * subdf['efficiency_factor']"
                    
                # pvprod method 2.2
                elif pvprod_calc_method == "method2.2":
                    subdf = subdf.with_columns([
                        (
                            (pl.col("radiation") / 1000) * pl.col("panel_efficiency") * inverter_efficiency * share_roof_area_available * pl.col("FLAECHE")
                        ).alias("pvprod_kW")
                    ])
                    formla_for_log_print = "inPOLARS: subdf['pvprod_kW'] = subdf['radiation'] / 1000 * subdf['panel_efficiency'] * inverter_efficiency * share_roof_area_available * subdf['FLAECHE']"

                # pvprod method 3.1
                elif pvprod_calc_method == "method3.1":
                    subdf = subdf.with_columns([
                        (
                            pl.col("radiation_rel_locmax") * pl.col("panel_efficiency") * kWpeak_per_m2 * inverter_efficiency * share_roof_area_available * pl.col("FLAECHE") * pl.col("efficiency_factor")
                        ).alias("pvprod_kW")
                    ])
                    formla_for_log_print = "inPOLARS: subdf['pvprod_kW'] = subdf['radiation_rel_locmax'] * subdf['panel_efficiency'] * kWpeak_per_m2 * inverter_efficiency * share_roof_area_available * subdf['FLAECHE'] * subdf['efficiency_factor']"

                # pvprod method 3.2
                elif pvprod_calc_method == "method3.2":
                    subdf = subdf.with_columns([
                        (
                            pl.col("radiation_rel_locmax") * pl.col("panel_efficiency") * kWpeak_per_m2 * inverter_efficiency * share_roof_area_available * pl.col("FLAECHE")
                        ).alias("pvprod_kW")
                    ])
                    formla_for_log_print = "inPOLARS: subdf['pvprod_kW'] = subdf['radiation_rel_locmax'] * subdf['panel_efficiency'] * kWpeak_per_m2 * inverter_efficiency * share_roof_area_available * subdf['FLAECHE']"
                
                # print computation formula for comparing methods
                print_to_logfile(f'* Computation formula for pv production per roof:\nmethod: {pvprod_calc_method} >> formula: {formla_for_log_print}', self.sett.log_name)


            # II ADJUST INST SETTINGS FOR EXISITN PV (INST_TF / SOURCE) =========================================================== 
            # create new columns for kWp installed per partition and % of partition used
            subdf = subdf.with_columns([
                (pl.lit(0.0).alias('dfuidPower')), 
                (pl.lit(0.0).alias('share_pvprod_used')),
                (pl.col('pvprod_kW').alias('poss_pvprod_kW')),
            ])


            if self.sett.ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF: 

                # loop through all egids with existing pv ----------
                egid_w_pvdf_inst = subdf.filter(pl.col('info_source') == 'pv_df')['EGID'].unique()
        
                for egid_inst in egid_w_pvdf_inst:

                    egid_capa_kW_pvdf = topo[egid_inst]['pv_inst']['TotalPower']
                    dfuid_w_inst_tuples_list = []

                    # create subdf_aggregated, find pvdf_capa to partition_calculated_capa faster 
                    def create_subdf_egid_pvdf(sdf, egid_inst): 
                        subdf_egid_pvdf = sdf.filter(pl.col('EGID') == egid_inst).clone()
                        subdf_egid_pvdf = subdf_egid_pvdf.group_by('df_uid').agg(
                            pl.col('EGID').first().alias('EGID'), 
                            pl.col('inst_TF').first().alias('inst_TF'), 
                            pl.col('info_source').first().alias('info_source'), 
                            pl.col('pvid').first().alias('pvid'), 
                            pl.col('grid_node').first().alias('grid_node'),
                            # pl.col('TotalPower').first().alias('TotalPower'), 
                            pl.col('FLAECHE').first().alias('FLAECHE'), 
                            pl.col('MSTRAHLUNG').first().alias('MSTRAHLUNG'), 
                            pl.col('STROMERTRAG').first().alias('STROMERTRAG'), 

                            pl.col('timestamp').len().alias('n_timestamp'), 
                            pl.col('t').len().alias('n_t'),
                            pl.col('month').len().alias('n_month'), 

                            pl.col('demand_kW').sum().alias('sum_demand_kW'), 
                            pl.col('pvprod_kW').sum().alias('sum_pvprod_kW'), 

                            pl.col('TotalPower').first().alias('TotalPower'), 
                            pl.col('dfuidPower').first().alias('dfuidPower'),
                            pl.col('share_pvprod_used').first().alias('share_pvprod_used'),
                        )
                        
                        subdf_egid_pvdf = subdf_egid_pvdf.with_columns([
                            (kWpeak_per_m2 * share_roof_area_available * pl.col('FLAECHE')).alias('poss_prod_capa_kW_peak')
                        ])
                            
                        subdf_egid_pvdf = subdf_egid_pvdf.sort('STROMERTRAG'  , descending=True)
                        # print(subdf_egid_pvdf[['df_uid', 'EGID', 'inst_TF', 'info_source', 'MSTRAHLUNG', 'sum_pvprod_kW', 'TotalPower', 'poss_prod_capa_kW_peak']])
                        return subdf_egid_pvdf
                        
                    subdf_egid_pvdf = create_subdf_egid_pvdf(subdf, egid_inst)

                    # loop through single PARTITIONS to adjust pvprod ----------
                    df_uid_inst = subdf_egid_pvdf['df_uid'][0]                    
                    for df_uid_inst in subdf_egid_pvdf['df_uid']:
                        subdf_dfuid_pvdf = subdf_egid_pvdf.filter(pl.col('df_uid') == df_uid_inst).clone()


                        # adjust pvprod, actual / partition kWpeak-ratio ----------
                        if subdf_dfuid_pvdf.shape[0] > 1:
                            print_to_logfile(f'  **WARNING**: df_uid {df_uid_inst} has more than one row in subdf_egid_pvdf', self.sett.log_name, 0)
                            
                            
                        # different options to allocate existing installations to the topo_egid 
                        kW_ratio_dfuidcalc_to_pvdf = egid_capa_kW_pvdf / subdf_dfuid_pvdf['poss_prod_capa_kW_peak'][0]
                            
                        if self.sett.ALGOspec_adjust_existing_pvdf_capa_topartition == 'capa_no_adj_pvprod_no_adj':
                            adjust_pvprod_kW = 1 # no adjustment of pvprod_kW
                            if (kW_ratio_dfuidcalc_to_pvdf <= 1.0) & (kW_ratio_dfuidcalc_to_pvdf > 0.0):
                                adjust_dfuidPower = egid_capa_kW_pvdf
                                inst_TF = True
                                share_pvprod_used = kW_ratio_dfuidcalc_to_pvdf

                            elif kW_ratio_dfuidcalc_to_pvdf > 1.0:
                                adjust_dfuidPower = subdf_dfuid_pvdf['poss_prod_capa_kW_peak'][0] 
                                inst_TF = True
                                share_pvprod_used = 1.0

                            elif kW_ratio_dfuidcalc_to_pvdf == 0.0:
                                adjust_dfuidPower = 0.0
                                inst_TF = False
                                share_pvprod_used = 0.0
                                # adjust_pvprod_kW = 0.0
                            

                        elif self.sett.ALGOspec_adjust_existing_pvdf_capa_topartition == 'capa_roundup_pvprod_no_adj':
                            adjust_pvprod_kW = 1
                            # DEBATABLE: to take the original TotalPower value from pv_df or the adjusted TotalPower value, based on the possible production potential of the roof partition. 
                            # for now it is kept at egid_capa_kW_pvdf, to keep consistency with the topo_egid dictionary
                            adjust_dfuidPower = subdf_dfuid_pvdf['poss_prod_capa_kW_peak'][0] if kW_ratio_dfuidcalc_to_pvdf > 0.0 else 0.0
                            inst_TF = True                                                    if kW_ratio_dfuidcalc_to_pvdf > 0.0 else False
                            share_pvprod_used = 1.0                                           if kW_ratio_dfuidcalc_to_pvdf > 0.0 else 0.0


                        elif self.sett.ALGOspec_adjust_existing_pvdf_capa_topartition == 'capa_roundup_pvprod_adjusted':
                            adjust_pvprod_kW = 1                                                if kW_ratio_dfuidcalc_to_pvdf > 0.0 else 0.0
                            adjust_dfuidPower = subdf_dfuid_pvdf['poss_prod_capa_kW_peak'][0]   if kW_ratio_dfuidcalc_to_pvdf > 0.0 else 0.0
                            inst_TF = True                                                      if kW_ratio_dfuidcalc_to_pvdf > 0.0 else False
                            share_pvprod_used = 1.0                                             if kW_ratio_dfuidcalc_to_pvdf > 0.0 else 0.0


                        elif self.sett.ALGOspec_adjust_existing_pvdf_capa_topartition == 'capa_no_adj_pvprod_adjusted':
                            adjust_pvprod_kW =  1                               if kW_ratio_dfuidcalc_to_pvdf > 1.0 else kW_ratio_dfuidcalc_to_pvdf
                            adjust_dfuidPower = egid_capa_kW_pvdf
                            inst_TF = True                                      if kW_ratio_dfuidcalc_to_pvdf > 0.0 else False
                            share_pvprod_used = kW_ratio_dfuidcalc_to_pvdf      if kW_ratio_dfuidcalc_to_pvdf > 0.0 else 0.0



                        # adjust pvprod_kW
                        subdf = subdf.with_columns([
                            pl.when((pl.col('EGID') == egid_inst) & (pl.col('df_uid') == df_uid_inst))
                            .then(pl.col('pvprod_kW') * adjust_pvprod_kW)
                            .otherwise(pl.col('pvprod_kW'))
                            .alias('pvprod_kW')
                        ])
                        #  adjust dfuidPower value
                        subdf = subdf.with_columns([
                            pl.when((pl.col('EGID') == egid_inst) & (pl.col('df_uid') == df_uid_inst))
                            .then(adjust_dfuidPower)
                            .otherwise(pl.col('dfuidPower'))
                            .alias('dfuidPower')
                        ])
                        # adjust inst_TF
                        subdf = subdf.with_columns([
                            pl.when((pl.col('EGID') == egid_inst) & (pl.col('df_uid') == df_uid_inst))
                            .then(inst_TF) 
                            .otherwise(pl.col('inst_TF'))
                            .alias('inst_TF')
                        ])
                        # adjust share_pvprod_used
                        subdf = subdf.with_columns([
                            pl.when((pl.col('EGID') == egid_inst) & (pl.col('df_uid') == df_uid_inst))
                            .then(share_pvprod_used)
                            .otherwise(pl.col('share_pvprod_used'))
                            .alias('share_pvprod_used')
                        ])

                        egid_capa_kW_pvdf = max(egid_capa_kW_pvdf - adjust_dfuidPower, 0.0)
                        dfuid_w_inst_tuple = ( 'tuple_names: df_uid_inst, share_pvprod_used, kWpeak', df_uid_inst, share_pvprod_used, adjust_dfuidPower )
                        dfuid_w_inst_tuples_list.append(dfuid_w_inst_tuple)


                    # adjust topo to know which partitions produce ----------
                    subdf_egid_pvdf_AFTER = create_subdf_egid_pvdf(subdf, egid_inst)

                    topo[egid_inst]['pv_inst']['dfuid_w_inst_tuples'] = dfuid_w_inst_tuples_list


                    #sanity check
                    subdf_pvdf_agg_before_list.append(subdf_egid_pvdf)
                    subdf_pvdf_agg_after_list.append(subdf_egid_pvdf_AFTER)
                    subdf_egid_pvdf[['df_uid', 'EGID', 'inst_TF', 'info_source', 'MSTRAHLUNG', 'sum_pvprod_kW', 'TotalPower', 'poss_prod_capa_kW_peak']]
                    subdf_egid_pvdf_AFTER[['df_uid', 'EGID', 'inst_TF', 'info_source', 'MSTRAHLUNG', 'sum_pvprod_kW', 'TotalPower', 'poss_prod_capa_kW_peak']]                          

            # drop unnecessary columns to keep topo_time_subdf a bit slim
            for col in self.sett.ALGOspec_drop_cols_topo_time_subdf_list:
                if col in subdf.columns:
                    subdf = subdf.drop(col)

            # export subdf ----------------------------------------------
            subdf.write_parquet(f'{subdf_path}/topo_subdf_{i}to{i+stepsize-1}.parquet')
            if self.sett.export_csvs:
                if 'dfuid_w_inst_tuples' in subdf.columns:
                    subdf = subdf.drop('dfuid_w_inst_tuples')  # remove this column, because it is not needed in the csv export
                subdf.write_csv(f'{subdf_path}/topo_subdf_{i}to{i+stepsize-1}.csv')

            if (i == 0) & self.sett.export_csvs:
                subdf.write_csv(f'{subdf_path}/topo_subdf_{i}to{i+stepsize-1}.csv' )
            checkpoint_to_logfile(f'end merge to topo_time_subdf (tranche {tranche_counter}/{len(range(0, len(egids), stepsize))}, size {stepsize})', self.sett.log_name, 0)
            checkpoint_to_logfile(' * * DEBUGGIGN * * *: END loop subdfs', self.sett.log_name, 0)


        # merge subdf_pvdf and export ----------------------------------------------
        if self.sett.ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF and len(subdf_pvdf_agg_before_list) > 0: 

            subdf_pvdf_agg_before = pl.concat(subdf_pvdf_agg_before_list)
            subdf_pvdf_agg_after = pl.concat(subdf_pvdf_agg_after_list)
            subdf_pvdf_agg_before.write_parquet(f'{subdf_path}/SANITYCHECK_topo_subdf_pvdf_agg_before.parquet')
            subdf_pvdf_agg_after.write_parquet(f'{subdf_path}/SANITYCHECK_topo_subdf_pvdf_agg_after.parquet')

            if subdf_pvdf_agg_after.shape[0] > 50000:
                subdf_pvdf_agg_before.head(50000).write_csv(f'{subdf_path}/SANITYCHECK_topo_subdf_pvdf_agg_before.csv')
                subdf_pvdf_agg_after.head(50000).write_csv(f'{subdf_path}/SANITYCHECK_topo_subdf_pvdf_agg_after.csv')
            else:
                subdf_pvdf_agg_before.write_csv(f'{subdf_path}/SANITYCHECK_topo_subdf_pvdf_agg_before.csv')
                subdf_pvdf_agg_after.write_csv(f'{subdf_path}/SANITYCHECK_topo_subdf_pvdf_agg_after.csv')

            shutil.copy(f'{self.sett.name_dir_export_path}/topo_egid.json', f'{self.sett.name_dir_export_path}/topo_egid_before_pvdf_existing_inst_adjustment.json')
            shutil.copy(f'{self.sett.name_dir_export_path}/topo_egid.txt', f'{self.sett.name_dir_export_path}/topo_egid_before_pvdf_existing_inst_adjustment.txt')


        # export topo ----------------------------------------------
        with open (f'{self.sett.name_dir_export_path}/topo_egid.json', 'w') as f:
            json.dump(topo, f, indent=4)
        with open(f'{self.sett.name_dir_export_path}/topo_egid.txt', 'w') as f:
            f.write(str(topo))




    def initial_sml_HOY_weatheryear_df(self, ):
        """
        Input:
            PVAlloc_Scenario + PVAllocScenario_Settings dataclass containing all scenarios settings + methods
        Output to pvalloc dir: 
            - dependent on the selected weather year, create a pd data frame for every hour of the year to be used later
        """
        print_to_logfile('run function: HOY_weatheryear_df', self.sett.log_name)

        # get every HOY of weather year ----------
        HOY_weatheryear_df = pd.DataFrame({'timestamp': pd.date_range(start=f'{self.sett.WEAspec_weather_year}-01-01 00:00:00',end=f'{self.sett.WEAspec_weather_year}-12-31 23:00:00', freq='h')})
        HOY_weatheryear_df['t'] = HOY_weatheryear_df.index.to_series().apply(lambda idx: f't_{idx + 1}')        
        HOY_weatheryear_df['month'] = HOY_weatheryear_df['timestamp'].dt.month
        HOY_weatheryear_df['day'] = HOY_weatheryear_df['timestamp'].dt.day
        HOY_weatheryear_df['hour'] = HOY_weatheryear_df['timestamp'].dt.hour

        # export df ----------
        HOY_weatheryear_df.to_parquet(f'{self.sett.name_dir_export_path}/HOY_weatheryear_df.parquet')

        
    def approx_outtopo_df_griddemand(self,):
        """
        Input:
            PVAllocScenario + PVAllocScenario_Settings dataclass containing all scenarios settings + methods
        Tasks:
            - import Map_egid_dsonode and gwr_gdf from the import path
            - create outtopo_df for all EGIDs not covered in the sample area (their demand over time)
            - assgin one / many load profile(s) to the EGIDs in sample to proxy demand over time.
            - sub partition the entire outtopo_df into subdfs (otherwise too large)
            - export the subdfs to the export path of the scenario
        Output to pvalloc dir:
            - subdf files of the outtopo_df, partitioned into nEGID-sized chunks            
        """

        # import settings + setup -------------------
        print_to_logfile('run function: approx_outtopo_df_griddemand', self.sett.log_name)

        # import --------------------
        # gwr_gdf = gpd.read_file(f'{self.sett.name_dir_import_path}/gwr_gdf.geojson')
        gwr_all_building_df = pd.read_parquet(f'{self.sett.name_dir_import_path}/gwr_all_building_df.parquet')
        Map_egid_dsonode = pl.read_parquet(f'{self.sett.name_dir_import_path}/Map_egid_dsonode.parquet')
        with open(f'{self.sett.name_dir_export_path}/topo_egid.json') as f:
            topo = json.load(f)

        # transformations in sample topo selection BUT in Polars
        gwr_all_building_df['EGID'] = gwr_all_building_df['EGID'].astype(str)
        gwr_all_building_df.loc[gwr_all_building_df['GBAUJ'] == '', 'GBAUJ'] = 0  # transform GBAUJ to apply filter and transform back
        gwr_all_building_df['GBAUJ'] = gwr_all_building_df['GBAUJ'].astype(int)
        # filtering for self.sett.GWR_specs
        gwr_all_building_df = gwr_all_building_df.loc[(gwr_all_building_df['GSTAT'].isin(self.sett.GWRspec_GSTAT)) &
                    # (gwr_all_building_df['GKLAS'].isin(self.sett.GWRspec_GKLAS)) &         # => EXCEPTION TO IN SAMPLE SELECTION!
                    (gwr_all_building_df['GGDENR'].isin(self.sett.bfs_numbers))              
                    # (gwr_all_building_df['GBAUJ'] >= self.sett.GWRspec_GBAUJ_minmax[0]) &  # => excluded because GBAUJ inconsistent 
                    # (gwr_all_building_df['GBAUJ'] <= self.sett.GWRspec_GBAUJ_minmax[1])    
                    ]
        gwr_all_building_df['GBAUJ'] = gwr_all_building_df['GBAUJ'].astype(str)
        gwr_all_building_df.loc[gwr_all_building_df['GBAUJ'] == '0', 'GBAUJ'] = ''
        # because not all buldings have dwelling information, need to remove dwelling columns and rows again (remove duplicates where 1 building had multiple dwellings)
        gwr_all_building_df.loc[gwr_all_building_df['GAREA'] == '', 'GAREA'] = 0
        gwr_all_building_df['GAREA'] = gwr_all_building_df['GAREA'].astype(float)

        gwr_all_building_df = pl.from_pandas(gwr_all_building_df)


        # clear old subdf files and create dir
        subdf_path = f'{self.sett.name_dir_export_path}/outtopo_time_subdf'

        if not os.path.exists(subdf_path):
            os.makedirs(subdf_path)
        else:
            old_files = glob.glob(f'{subdf_path}/*')
            for f in old_files:
                os.remove(f)

        # create outttopo_df -----------------------
        # create single demand profile for each EGID, not covered in sample  
        if False: #self.sett.GRIDspec_flat_profile_demand_type_col == 'flat':
            t_HOY_range_df = pl.DataFrame({
                "t": [f"t_{hoy}" for hoy in range(1, 8761)]
            }) 
            demand_outsample_ts = t_HOY_range_df.clone()
            demand_outsample_ts = demand_outsample_ts.with_columns([
                pl.col("t").str.strip_chars("t_").cast(pl.Int64).alias("t_int")
            ])                
            demand_outsample_ts = demand_outsample_ts.with_columns([
                # Calculate 'DayOfYear' using vectorized operations
                ((pl.col("t_int") - 1) // 24 + 1).alias("DayOfYear"),
                    
                # Calculate 'HourOfDay' using vectorized operations
                ((pl.col("t_int") - 1) % 24 + 1).alias("HourOfDay")
            ])
            demand_outsample_ts.group_by(["DayOfYear"]).agg([
                pl.len()
            ])
            demand_outsample_ts.group_by(["HourOfDay"]).agg([
                pl.len()
            ])


            # devide assumed yearly demand onto days to then distribute it to hours based weights for assumed high usage hours
            demand_outsample_ts = demand_outsample_ts.with_columns([
                pl.lit(0).alias("demand_proxy_out_kW")
            ])                
            ndays = demand_outsample_ts['DayOfYear'].n_unique()
            demand_pday = self.sett.GRIDspec_flat_profile_demand_total_EGID / ndays
            flat_profile_demand_dict = self.sett.GRIDspec_flat_profile_demand_dict
            hours_range = pl.DataFrame({'hour': [x for x in range(1,24+1)]})

            # count n of hours in window to allocate proper x-share of demand_pday to the single hour
            for i, (_, value) in enumerate(flat_profile_demand_dict.items()):
                t_start, t_end = value['t']
                if t_start <= t_end:
                    mask_hoy_in_window = list((hours_range['hour'] >= t_start) & (hours_range['hour'] <= t_end))
                else:
                    mask_hoy_in_window = list((hours_range['hour'] >= t_start) | (hours_range['hour'] <= t_end))

                filtered_hours = hours_range.filter(mask_hoy_in_window)
                hours_int_range = filtered_hours["hour"].to_list()
                nhours = sum(mask_hoy_in_window)
                demand_phour = demand_pday * value['demand_share'] / nhours
                    
                demand_outsample_ts = demand_outsample_ts.with_columns([
                    pl.when(pl.col("HourOfDay").is_in(hours_int_range))  # Condition: HourOfDay is in hours_int_range
                    .then(demand_phour)  # Set to demand_phour where the condition is True
                    .otherwise(pl.col("demand_proxy_out_kW"))  # Keep the original value where the condition is False
                    .alias("demand_proxy_out_kW")  # Assign to the same column
                ])

            notneeded_cols = [col for col in demand_outsample_ts.columns if col not in ['t', 'demand_proxy_out_kW']]
            demand_outsample_ts = demand_outsample_ts.drop(notneeded_cols)
  
            # calculate node usage through out of sample EGIDs
            egids_topo = list(topo.keys())
            outtopo_df = gwr_all_building_df.filter(~pl.col('EGID').is_in(egids_topo)).clone()

            outtopo_df = outtopo_df.join(Map_egid_dsonode, how='left', on='EGID') 

            # partition the outtopo_df into subdfs because otherwise not readable with enough memory
            topo_subdf_partitioner = self.sett.ALGOspec_topo_subdf_partitioner
                
            # add key for merge             
            outtopo_df = outtopo_df.with_columns([
                pl.lit(1).alias("key_for_merge_1")
            ])
            demand_outsample_ts = demand_outsample_ts.with_columns([
                pl.lit(1).alias("key_for_merge_1")
            ])

            # attach demand ts to outtopo_df by subdf
            egids_outsample = outtopo_df['EGID'].unique().to_list()
            stepsize = topo_subdf_partitioner if len(egids_outsample) > topo_subdf_partitioner else len(egids_outsample)
            tranche_counter = 0
            for i in range(0, len(egids_outsample), stepsize):
                tranche_counter += 1
                subdf = outtopo_df.filter(pl.col('EGID').is_in(egids_outsample[i:i + stepsize])).clone()

                # merge proxy demand to gwr EGIDs out of sample
                subdf_ts = subdf.join(demand_outsample_ts, how='left', on='key_for_merge_1')
                subdf_ts = subdf_ts.drop('key_for_merge_1')

                # export 
                subdf_ts.write_parquet(f'{subdf_path}/outtopo_subdf_{i}to{i+stepsize-1}.parquet')
                # if self.sett.export_csvs:
                if (i<2) & self.sett.export_csvs:
                    subdf_ts.write_csv(f'{subdf_path}/outtopo_subdf_{i}to{i+stepsize-1}.csv')
                checkpoint_to_logfile(f'export outsample subdf_ts {i}to{i+stepsize-1}', self.sett.log_name, 0, self.sett.show_debug_prints)


        # or call a demand profile from the demandtypes
        elif (self.sett.GRIDspec_flat_profile_demand_type_col == 'MFH_swstore') or (self.sett.GRIDspec_flat_profile_demand_type_col == 'outtopo_demand_zero') : 

            demandtypes_ts  = pl.read_parquet(f'{self.sett.name_dir_import_path}/demandtypes_ts.parquet')
            demandtypes_unpivot = demandtypes_ts.unpivot(
                on = ['SFH', 'MFH', ],
                index=['t', 't_int'],  # col that stays unchanged
                value_name='demand_profile',  # name of the column that will hold the values
                variable_name='sfhmfh_typ'  # name of the column that will hold the original column names
            )                

            # calculate node usage through out of sample EGIDs
            egids_topo = list(topo.keys())
            outtopo_df = gwr_all_building_df.filter(~pl.col('EGID').is_in(egids_topo)).clone()
            outtopo_df = outtopo_df.join(Map_egid_dsonode, how='left', on='EGID') 

            # partition the outtopo_df into subdfs because otherwise not readable with enough memory
            topo_subdf_partitioner = self.sett.ALGOspec_topo_subdf_partitioner
                
            # attach demand ts to outtopo_df by subdf
            egids_outsample = outtopo_df['EGID'].unique().to_list()
            stepsize = topo_subdf_partitioner if len(egids_outsample) > topo_subdf_partitioner else len(egids_outsample)
            tranche_counter = 0
            for i in range(0, len(egids_outsample), stepsize):
                tranche_counter += 1
                subdf = outtopo_df.filter(pl.col('EGID').is_in(egids_outsample[i:i + stepsize])).clone()

                # merge demand profile by SFHMFH and scale by GAREA
                if 'elec_dem_pGAREA' in subdf.columns:
                    subdf= subdf.rename({"elec_dem_pGAREA": "demand_elec_pGAREA"})
                subdf_ts = subdf.join(demandtypes_unpivot, on='sfhmfh_typ', how='left') 
                subdf_ts = subdf_ts.with_columns([
                    (pl.col("demand_elec_pGAREA") * pl.col("demand_profile") * pl.col("GAREA") ).alias("demand_proxy_out_kW")  # convert to kW
                ])

                if self.sett.GRIDspec_flat_profile_demand_type_col == 'outtopo_demand_zero':
                    subdf_ts = subdf_ts.with_columns([
                        pl.lit(0).alias("demand_proxy_out_kW")  # set demand to 0
                    ])

                # export 
                subdf_ts.write_parquet(f'{subdf_path}/outtopo_subdf_{i}to{i+stepsize-1}.parquet')
                # if self.sett.export_csvs:
                if (i<2) & self.sett.export_csvs:
                    subdf_ts.write_csv(f'{subdf_path}/outtopo_subdf_{i}to{i+stepsize-1}.csv')
                checkpoint_to_logfile(f'export outsample subdf_ts {i}to{i+stepsize-1}', self.sett.log_name, 0, self.sett.show_debug_prints)




    def initial_sml_get_DSO_nodes_df_AND_ts(self,):
        """
        Input: 
            PVAllocScenario + PVAllocScenario_Settings dataclass containing all scenarios settings + methods
        Tasks: 
            - import Map_egid_dsonode and gwr_gdf from the import path
            - merge DSO node and gwr dataframe 
            - add centroids as approximative node locations to data frame. 
        Output to pvalloc dir:
            - dsonodes_df  containing nodes incl threshold
            - dsonodes_gdf containing nodes incl threshold and approx location
        """    
        # import settings + setup -------------------
        print_to_logfile('run function: get_gridnodes_DSO', self.sett.log_name)

        # import ----------------------
        Map_egid_dsonode = pd.read_parquet(f'{self.sett.name_dir_import_path}/Map_egid_dsonode.parquet')
        gwr_gdf = gpd.read_file(f'{self.sett.name_dir_import_path}/gwr_gdf.geojson')
        gwr_all_building_gdf = gpd.read_file(f'{self.sett.name_dir_import_path}/gwr_all_building_gdf.geojson')


        # transformations ----------------------
        Map_egid_dsonode['EGID'], Map_egid_dsonode['grid_node'] = Map_egid_dsonode['EGID'].astype(str), Map_egid_dsonode['grid_node'].astype(str)
            
        gwr_all_building_gdf = gwr_all_building_gdf.merge(Map_egid_dsonode, how='left', on='EGID')
        gwr_all_building_gdf.set_crs('EPSG:2056', inplace=True)
        node_centroid_list = []
        node = Map_egid_dsonode['grid_node'].unique()[0]
        for node in gwr_all_building_gdf['grid_node'].unique():
            if pd.notna(node):
                sub_gwr_node = gwr_all_building_gdf.loc[gwr_all_building_gdf['grid_node'] == node]
                node_centroid = union_all(sub_gwr_node['geometry']).centroid
                kVA_treshold = sub_gwr_node['kVA_threshold'].unique()[0]

                node_centroid_list.append([node, kVA_treshold, node_centroid])

        dsonodes_gdf = gpd.GeoDataFrame(node_centroid_list, columns=['grid_node', 'kVA_threshold', 'geometry'], crs='EPSG:2056')
        dsonodes_df = dsonodes_gdf.loc[:,dsonodes_gdf.columns != 'geometry']

        dsonodes_in_gwr_df = gwr_gdf.merge(Map_egid_dsonode, how='left', on='EGID')

        # summary prints ----------------------
        print_to_logfile('DSO grid nodes information:', self.sett.summary_name)
        checkpoint_to_logfile(f'Total: {Map_egid_dsonode["grid_node"].nunique()} DSO grid nodes for {Map_egid_dsonode["EGID"].nunique()} unique EGIDs (Map_egid_dsonode.shape {Map_egid_dsonode.shape[0]}, node/egid ratio: {round(Map_egid_dsonode["grid_node"].nunique() / Map_egid_dsonode["EGID"].nunique(),4)*100}%', self.sett.summary_name)
        checkpoint_to_logfile(f'In sample: {dsonodes_in_gwr_df["grid_node"].nunique()} DSO grid nodes for {dsonodes_in_gwr_df["EGID"].nunique()} EGIDs in {len(self.sett.bfs_numbers)} BFSs , (node/egid ratio: {round(dsonodes_in_gwr_df["grid_node"].nunique()/dsonodes_in_gwr_df["EGID"].nunique(),4)*100}%)', self.sett.summary_name)
            

        # create dsonodes_ts -----------------------
        checkpoint_to_logfile('create dsonodes_ts to assign grid usage for outsample gwr (later)', self.sett.log_name, 0, self.sett.show_debug_prints)

        # create empty time series for all node demand, through proxy and in sample
        t_HOY_range_df = pl.DataFrame({
            "t": [f"t_{hoy}" for hoy in range(1, 8761)]
        })        
        dsonodes_df_copy = pl.from_pandas(dsonodes_df.copy())          
        dsonodes_df_copy = dsonodes_df_copy.with_columns([
            pl.lit(1).alias("key_for_merge_1")
        ])
        t_HOY_range_df = t_HOY_range_df.with_columns([
            pl.lit(1).alias("key_for_merge_1")
        ])
            
        dsonodes_ts = dsonodes_df_copy.join(t_HOY_range_df, how='left', on='key_for_merge_1')
        dsonodes_ts = dsonodes_ts.drop('key_for_merge_1')
        t_HOY_range_df = t_HOY_range_df.drop('key_for_merge_1')
        dsonodes_ts = dsonodes_ts.with_columns([
            pl.lit(0).alias("demand_outtopo"),
            pl.lit(0).alias("demand_topo"),
            pl.lit(0).alias("pvprod_kW"),
        ])


        # export ----------------------
        # These files live in the preprep folder, which is SHARED by all scenarios running in parallel (HPC array tasks). Every task writes
        # identical content, but a plain open(path, 'w') empties the file first, so a task reading it at that moment sees an empty / partial file.
        # atomic_export writes to a private temp file and swaps it in in one step, so readers always see a complete file.
        import_path = self.sett.name_dir_import_path
        self.atomic_export(f'{import_path}/dsonodes_df.parquet',  lambda p: dsonodes_df.to_parquet(p))
        self.atomic_export(f'{import_path}/dsonodes_df.csv',      lambda p: dsonodes_df.to_csv(p)) if self.sett.export_csvs else None
        self.atomic_export(f'{import_path}/dsonodes_ts.parquet',  lambda p: dsonodes_ts.write_parquet(p))
        self.atomic_export(f'{import_path}/dsonodes_gdf.geojson', lambda p: Path(p).write_text(dsonodes_gdf.to_json()))




    def initial_sml_iterpolate_instcost_function(self, ):
        """
        Input:
            PVAllocScenario + PVAllocScenario_Settings dataclass containing all scenarios settings + methods    
        Tasks:
            - use installation cost data from energie rechner schweiz documentation
            - create interpolation function for installation cost per kW and total installation cost, dependent on a certain kwp-installation range suitable to my modeling aim
            - export the coefficients of the function to a json file and a numpy array for later use in the model
        Output:
            - cost_coefficients.json,       -> cost coefficients from intrapolation
            - pvinstcost_coefficients.npy     -> cost coefficients from intrapolation
            - pvinstcost_table.png         -> plot of the installation cost data and the fitted function
            
        """
        # setup --------
        print_to_logfile('run function: estimate_iterpolate_instcost_function', self.sett.log_name)

        # data import ----- (copied from energie rechner schweiz doucmentation)
        installation_cost_dict = {
        "on_roof_installation_cost_pkW": {
            2:   4636,
            3:   3984,
            5:   3373,
            10:  2735,
            15:  2420,
            20:  2219,
            30:  1967,
            50:  1710,
            75:  1552,
            100: 1463,
            125: 1406,
            150: 1365
        },
        "on_roof_installation_cost_total": {
            2:   9272,
            3:   11952,
            5:   16863,
            10:  27353,
            15:  36304,
            20:  44370,
            30:  59009,
            50:  85478,
            75:  116420,
            100: 146349,
            125: 175748,
            150: 204816
        },}

        installation_cost_df = pd.DataFrame({
            'kw': list(installation_cost_dict['on_roof_installation_cost_pkW'].keys()),
            'chf_pkW': list(installation_cost_dict['on_roof_installation_cost_pkW'].values()),
            'chf_total': list(installation_cost_dict['on_roof_installation_cost_total'].values())
        })
        installation_cost_df.reset_index(inplace=True)


        # select kWp range --------
        kW_range_for_estim = self.sett.TECspec_kW_range_for_pvinst_cost_estim
        installation_cost_df['kw_in_estim_range'] = installation_cost_df['kw'].apply(
                                                            lambda x: True if (x >= kW_range_for_estim[0]) & 
                                                                            (x <= kW_range_for_estim[1]) else False)
            

        # define intrapolation functions for cost structure -----
            
        # chf_pkW
        def func_chf_pkW(x, a, b, c, d):
            return a +  d*((x ** b) /  (x ** c))
        params_pkW, covar = curve_fit(func_chf_pkW, 
                                    installation_cost_df.loc[installation_cost_df['kw_in_estim_range'], 'kw'],
                                    installation_cost_df.loc[installation_cost_df['kw_in_estim_range'], 'chf_pkW'])
        def estim_instcost_chfpkW(x):
            return func_chf_pkW(x, *params_pkW)
            
        checkpoint_to_logfile('created intrapolation function for chf_pkW using "cureve_fit" to receive curve parameters', self.sett.log_name)
        print_to_logfile(f'params_pkW: {params_pkW}', self.sett.log_name)
            

        # chf_total
        def func_chf_total(x, a, b, c):
            return a +  b*(x**c) 
        params_total, covar = curve_fit(func_chf_total,
                                        installation_cost_df.loc[installation_cost_df['kw_in_estim_range'], 'kw'],
                                        installation_cost_df.loc[installation_cost_df['kw_in_estim_range'], 'chf_total'])
        def estim_instcost_chftotal(x):
            return func_chf_total(x, *params_total)
            
        checkpoint_to_logfile('created intrapolation function for chf_total using "Polynomial.fit" to receive curve coefficients', self.sett.log_name)
        print_to_logfile(f'coefs_total: {params_total}', self.sett.log_name)

        pvinstcost_coefficients = {
            'params_pkW': list(params_pkW),
            'params_total': list(params_total)
        }

        # export ---------
        with open(f'{self.sett.name_dir_export_path}/pvinstcost_coefficients.json', 'w') as f:
            json.dump(pvinstcost_coefficients, f)

        np.save(f'{self.sett.name_dir_export_path}/pvinstcost_coefficients.npy', pvinstcost_coefficients)


        # plot installation cost df + intrapolation functions -------------------
        if True: 
            fig, axs = plt.subplots(1, 2, figsize=(12, 6)) 
            kw_range = np.linspace(installation_cost_df['kw'].min(), installation_cost_df['kw'].max(), 100)
            chf_pkW_fitted = estim_instcost_chfpkW(kw_range)
            chf_total_fitted = estim_instcost_chftotal(kw_range)

            # Scatter plots + interpolation -----
            # Interpolated line kWp Cost per kW
            axs[0].plot(kw_range, chf_pkW_fitted, label='Interpolated chf_pkW', color='red', alpha = 0.5)  
            axs[0].scatter(installation_cost_df.loc[installation_cost_df['kw_in_estim_range'], 'kw'],
                        installation_cost_df.loc[installation_cost_df['kw_in_estim_range'], 'chf_pkW'], 
                        label='point range used for function estimation', color='purple', s=160, alpha=0.5)
            axs[0].scatter(installation_cost_df['kw'], installation_cost_df['chf_pkW'], label='chf_pkW', color='blue', )

                            
            axs[0].set(xlabel='kW', ylabel='CHF', title='Cost per kW')
            axs[0].legend()
                

            # Interpolated line kWp Total Cost
            axs[1].plot(kw_range, chf_total_fitted, label='Interpolated chf_total', color='green', alpha = 0.5)
            axs[1].scatter(installation_cost_df.loc[installation_cost_df['kw_in_estim_range'], 'kw'],
                        installation_cost_df.loc[installation_cost_df['kw_in_estim_range'], 'chf_total'], 
                        label='point range used for function estimation', color='purple', s=160, alpha=0.5)
            axs[1].scatter(installation_cost_df['kw'], installation_cost_df['chf_total'], label='chf_total', color='orange')

            axs[1].set(xlabel='kW', ylabel='CHF', title='Total Cost')
            axs[1].legend()

            # Export the plots
            plt.tight_layout()
            plt.savefig(f'{self.sett.name_dir_export_path}/pvinstcost_table.png')


        # export cost df -------------------
        # installation_cost_df.to_parquet(f'{self.sett.name_dir_import_path}/pvinstcost_table.parquet')
        installation_cost_df.to_parquet(f'{self.sett.name_dir_export_path}/pvinstcost_table.parquet')
        installation_cost_df.to_csv(f'{self.sett.name_dir_export_path}/pvinstcost_table.csv')
        checkpoint_to_logfile('exported pvinstcost_table', self.sett.log_name, 0)

        return estim_instcost_chfpkW, estim_instcost_chftotal




    def initial_lrg_import_preprep_AND_create_topology(self, ):
        """
        Input: 
            - PVAllocScenario + PVAllocScenario_Settings dataclass containing all scenarios settings + methods
        Tasks: 
            - import all necessary data objects from preprep
            - transform different data sets and frames into useful formats
            - create topology data frame for all buildings in the selected municipalities including: 
                . gwr_infos
                . grid_node connection
                . pv_installation status
                . solkat assignmnet of roof partitions
                . pvtarif from DSO
                . proxy demand type
                . DSO operator 
                . electricity price
        Export: 
            - topo_egid.JSON, containing all relevant data for the selected municipalities
            - df_list containing all the indiviudal data sets to be possibly called later, 
            - df_names containing all the names of the data sets in df_list (to call up elelments of df_list by name)
            
        """
                    
        # import settings + setup -------------------
        print_to_logfile('run function: import_prepreped_data', self.sett.log_name)


        # IMPORT & TRANSFORM ============================================================================
        # Import all necessary data objects from prepreped folder and transform them for later calculations
        print_to_logfile('import & transform data', self.sett.log_name)
        if True:
                    
            # GWR -------
            gwr_gdf = gpd.read_file(f'{self.sett.name_dir_import_path}/gwr_gdf.geojson')
            gwr = pd.read_parquet(f'{self.sett.name_dir_import_path}/gwr.parquet')

            gwr['EGID'] = gwr['EGID'].astype(str)
            gwr.loc[gwr['GBAUJ'] == '', 'GBAUJ'] = 0  # transform GBAUJ to apply filter and transform back
            gwr['GBAUJ'] = gwr['GBAUJ'].astype(int)
            # filtering for self.sett.GWR_specs
            if self.sett.apply_GWRspec_GBAUJ_minmax_TF:
                gwr = gwr.loc[(gwr['GSTAT'].isin(self.sett.GWRspec_GSTAT)) &
                            (gwr['GKLAS'].isin(self.sett.GWRspec_GKLAS)) &
                            (gwr['GBAUJ'] >= self.sett.GWRspec_GBAUJ_minmax[0]) &  # exclude GBAUJ, because missing in a lot of instances -> droped observations predominantly in rural areas
                            (gwr['GBAUJ'] <= self.sett.GWRspec_GBAUJ_minmax[1])
                            ]
            elif not self.sett.apply_GWRspec_GBAUJ_minmax_TF:
                gwr = gwr.loc[(gwr['GSTAT'].isin(self.sett.GWRspec_GSTAT)) &
                            (gwr['GKLAS'].isin(self.sett.GWRspec_GKLAS)) 
                            # exclude GBAUJ, because missing in a lot of instances -> droped observations predominantly in rural areas
                            # (gwr['GBAUJ'] >= self.sett.GWRspec_GBAUJ_minmax[0]) &
                            # (gwr['GBAUJ'] <= self.sett.GWRspec_GBAUJ_minmax[1])
                            ]
            gwr['GBAUJ'] = gwr['GBAUJ'].astype(str)
            gwr.loc[gwr['GBAUJ'] == '0', 'GBAUJ'] = ''
            # because not all buldings have dwelling information, need to remove dwelling columns and rows again (remove duplicates where 1 building had multiple dwellings)
            gwr.loc[gwr['GAREA'] == '', 'GAREA'] = 0
            gwr['GAREA'] = gwr['GAREA'].astype(float)

            if self.sett.GWRspec_dwelling_cols == []:
                gwr = copy.deepcopy(gwr.loc[:, self.sett.GWRspec_building_cols + self.sett.GWRspec_swstore_demand_cols])
                gwr = gwr.drop_duplicates(subset=['EGID'])
            gwr = gwr.loc[gwr['GGDENR'].isin(self.sett.bfs_numbers)]
            gwr = copy.deepcopy(gwr)
                

            # SOLKAT -------
            solkat = pd.read_parquet(f'{self.sett.name_dir_import_path}/solkat.parquet')

            solkat['EGID'] = solkat['EGID'].fillna('').astype(str)
            solkat['DF_UID'] = solkat['DF_UID'].fillna('').astype(str)
            solkat['DF_NUMMER'] = solkat['DF_NUMMER'].fillna('').astype(str)
            solkat['SB_UUID'] = solkat['SB_UUID'].fillna('').astype(str)
            solkat['FLAECHE'] = solkat['FLAECHE'].fillna(0).astype(float)
            solkat['STROMERTRAG'] = solkat['STROMERTRAG'].fillna(0).astype(float)
            solkat['MSTRAHLUNG'] = solkat['MSTRAHLUNG'].fillna(0).astype(float)
            solkat['GSTRAHLUNG'] = solkat['GSTRAHLUNG'].fillna(0).astype(float)
            solkat['AUSRICHTUNG'] = solkat['AUSRICHTUNG'].astype(int)
            solkat['NEIGUNG'] = solkat['NEIGUNG'].astype(int)

            # remove building with maximal (outlier large) number of partitions => complicates the creation of partition combinations
            solkat['EGID'].value_counts()
            egid_counts = solkat['EGID'].value_counts()
            egids_below_max = list(egid_counts[egid_counts < self.sett.GWRspec_solkat_max_n_partitions].index)
            solkat = solkat.loc[solkat['EGID'].isin(egids_below_max)]

            # remove buildings with a certain roof surface because they are too large to be residential houses
            solkat_area_per_EGID_range = self.sett.GWRspec_solkat_area_per_EGID_range
            if solkat_area_per_EGID_range != []:
                solkat_agg_FLAECH = solkat.groupby('EGID')['FLAECHE'].sum()
                solkat = solkat.merge(solkat_agg_FLAECH, how='left', on='EGID', suffixes=('', '_sum'))
                solkat = solkat.rename(columns={'FLAECHE_sum': 'FLAECHE_total'})
                solkat = solkat.loc[(solkat['FLAECHE_total'] >= solkat_area_per_EGID_range[0]) & 
                                    (solkat['FLAECHE_total'] < solkat_area_per_EGID_range[1])]
                solkat.drop(columns='FLAECHE_total', inplace=True)

            solkat = solkat.loc[solkat['BFS_NUMMER'].isin(self.sett.bfs_numbers)]
            solkat = copy.deepcopy(solkat)


            # SOLKAT MONTH -------
            solkat_month = pd.read_parquet(f'{self.sett.name_dir_import_path}/solkat_month.parquet')
            solkat_month['DF_UID'] = solkat_month['DF_UID'].fillna('').astype(str)


            # PV -------
            pv = pd.read_parquet(f'{self.sett.name_dir_import_path}/pv.parquet').drop(columns='__index_level_0__', errors='ignore')   # stale index column of the preprep file, would clash with the index written on export
            pv['xtf_id'] = pv['xtf_id'].fillna(0).astype(int).replace(0, '').astype(str)    
            pv['TotalPower'] = pv['TotalPower'].fillna(0).astype(float)

            pv['BeginningOfOperation'] = pd.to_datetime(pv['BeginningOfOperation'], format='%Y-%m-%d', errors='coerce')
            gbauj_range = [pd.to_datetime(f'{self.sett.GWRspec_GBAUJ_minmax[0]}-01-01'), 
                        pd.to_datetime(f'{self.sett.GWRspec_GBAUJ_minmax[1]}-12-31')]
            pv = pv.loc[(pv['BeginningOfOperation'] >= gbauj_range[0]) & (pv['BeginningOfOperation'] <= gbauj_range[1])]
            pv['BeginningOfOperation'] = pv['BeginningOfOperation'].dt.strftime('%Y-%m-%d')

            pv = pv.loc[pv["BFS_NUMMER"].isin(self.sett.bfs_numbers)]
            pv = pv.copy()


            # PV TARIF -------
            pvtarif_year = self.sett.TECspec_pvtarif_year
            pvtarif_col =  self.sett.TECspec_pvtarif_col
                
            Map_gm_ewr = pd.read_parquet(f'{self.sett.name_dir_import_path}/Map_gm_ewr.parquet')
            pvtarif = pd.read_parquet(f'{self.sett.name_dir_import_path}/pvtarif.parquet')
            if '\ufeff' in pvtarif.columns:
                pvtarif.columns = pvtarif.columns.str.replace('﻿', '', regex=False)   # guard: API data may carry a BOM in 'nrElcom'
            pvtarif = pvtarif.merge(Map_gm_ewr, how='left', on='nrElcom')

            pvtarif['bfs'] = pvtarif['bfs'].astype(str)
            # pvtarif[pvtarif_col] = pvtarif[pvtarif_col].fillna(0).astype(float)
            pvtarif[pvtarif_col] = pvtarif[pvtarif_col].replace('', 0).astype(float)

            # transformation
            pvtarif = pvtarif.loc[(pvtarif['year'] == str(pvtarif_year)[2:4]) & 
                                (pvtarif['bfs'].isin((self.sett.bfs_numbers)))]

            # empty table: fine for a generic / hourly pvtarif (TECspec_pvtarif_year may lie outside the VESE data), for the None mode it must still fail here
            if pvtarif.shape[0] > 0 or self.sett.TECspec_pvtarif_Rp_kWh is None:
                empty_cols = [col for col in pvtarif.columns if pvtarif[col].isna().all()]
                pvtarif = pvtarif.drop(columns=empty_cols)

                select_cols = ['nrElcom', 'nomEw', 'year', 'bfs', 'idofs'] + pvtarif_col
                pvtarif = pvtarif[select_cols].copy()


            # ELECTRICITY PRICE -------
            elecpri = pd.read_parquet(f'{self.sett.name_dir_import_path}/elecpri.parquet')
            elecpri['bfs_number'] = elecpri['bfs_number'].astype(str)


            # Map solkat_egid > pv -------
            Map_egid_pv = pd.read_parquet(f'{self.sett.name_dir_import_path}/Map_egid_pv.parquet')
            Map_egid_pv = Map_egid_pv.dropna()
            Map_egid_pv['EGID'] = Map_egid_pv['EGID'].astype(int).astype(str)
            Map_egid_pv['xtf_id'] = Map_egid_pv['xtf_id'].fillna('').astype(int).astype(str)


            # Map egid > node -------
            Map_egid_dsonode = pd.read_parquet(f'{self.sett.name_dir_import_path}/Map_egid_dsonode.parquet')
            Map_egid_dsonode['EGID'] = Map_egid_dsonode['EGID'].astype(str)
            Map_egid_dsonode['grid_node'] = Map_egid_dsonode['grid_node'].astype(str)
            Map_egid_dsonode.index = Map_egid_dsonode['EGID']


            # dsonodes data -------
            dsonsodes_df  = pd.read_parquet(f'{self.sett.name_dir_import_path}/dsonodes_df.parquet')
            dsonodes_gdf = gpd.read_file(f'{self.sett.name_dir_import_path}/dsonodes_gdf.geojson')


            # angle_tilt_df -------
            angle_tilt_df = pd.read_parquet(f'{self.sett.name_dir_import_path}/angle_tilt_df.parquet')


            # PV Cost functions --------
            """
                # Define the interpolation functions using the imported coefficients
                def func_chf_pkW(x, a, b):
                    return a + b / x

                estim_instcost_chfpkW = lambda x: func_chf_pkW(x, *params_pkW)

                def func_chf_total_poly(x, coefs_total):
                    return sum(c * x**i for i, c in enumerate(coefs_total))

                estim_instcost_chftotal = lambda x: func_chf_total_poly(x, coefs_total)
            """


        # EGID SELECTION / EXCLUSION ============================================================================
        # check how many of gwr's EGIDs are in solkat and pv
        len(np.intersect1d(gwr['EGID'].unique(), solkat['EGID'].unique()))
        len(np.intersect1d(gwr['EGID'].unique(), Map_egid_pv['EGID'].unique()))


        # gwr/solkat mismatch ----------
        # throw out all EGIDs of GWR that are not in solkat
        # >  NOTE: this could be troublesome :/ check in QGIS if large share of buildings are missing.  
        gwr_before_solkat_selection = copy.deepcopy(gwr)
        gwr = copy.deepcopy(gwr.loc[gwr['EGID'].isin(solkat['EGID'].unique())])


        # gwr/Map_egid_dsonodes mismatch ----------
        # > Case 1 EGID in Map but not in GWR: => drop EGID; no problem, connection to a house that is not in sample; will happen automatically when creating topology on gwr EGIDs
        # > Case 2 EGID in GWR but not in Map: more problematic; EGID "close" to next node => Match to nearest node; EGID "far away" => drop EGID
        gwr_wo_node = gwr.loc[~gwr['EGID'].isin(Map_egid_dsonode['EGID'].unique()),]
        Map_egid_dsonode_appendings =[]
            
                
        for egid in gwr_wo_node['EGID']:
            egid_point = gwr_gdf.loc[gwr_gdf['EGID'] == egid, 'geometry'].iloc[0]
            dsonodes_gdf['distances'] = dsonodes_gdf['geometry'].distance(egid_point)
            min_idx = dsonodes_gdf['distances'].idxmin()
            min_dist = dsonodes_gdf['distances'].min()
                
            if min_dist < self.sett.TECspec_max_distance_m_for_EGID_node_matching:
                Map_egid_dsonode_appendings.append([egid, dsonodes_gdf.loc[min_idx, 'grid_node'], dsonodes_gdf.loc[min_idx, 'kVA_threshold']])
            
        if len(Map_egid_dsonode_appendings) > 0:
            Map_appendings_df = pd.DataFrame(Map_egid_dsonode_appendings, columns=['EGID', 'grid_node', 'kVA_threshold'])
            Map_appendings_df = Map_appendings_df.dropna(how='all')

            Map_egid_dsonode = pd.concat([Map_egid_dsonode, Map_appendings_df], axis=0)

        gwr_before_dsonode_selection = copy.deepcopy(gwr)
        gwr = copy.deepcopy(gwr.loc[gwr['EGID'].isin(Map_egid_dsonode['EGID'].unique())])
        check_egids = [egid for egid in self.sett.mini_sub_model_select_EGIDs if egid in gwr['EGID'].unique()]
        check_egids


        # mini model for exploratory work ----------
        if self.sett.mini_sub_model_TF:

            if self.sett.mini_sub_model_by_X == 'by_gridnode':
                gridnodes_in_gwr = Map_egid_dsonode.loc[Map_egid_dsonode['EGID'].isin(gwr['EGID'])]['grid_node'].unique()
                if any([node in gridnodes_in_gwr for node in self.sett.mini_sub_model_grid_nodes]):
                    mini_sub_model_nodes = self.sett.mini_sub_model_grid_nodes
                else:
                    mini_sub_model_nodes = gridnodes_in_gwr[0:self.sett.mini_sub_model_ngridnodes]
                        
                mini_submodel_EGIDs = Map_egid_dsonode.loc[Map_egid_dsonode['grid_node'].isin(mini_sub_model_nodes)]['EGID'].unique()
                gwr = copy.deepcopy(gwr.loc[gwr['EGID'].isin(mini_submodel_EGIDs)])

                if (self.sett.mini_sub_model_nEGIDs is not None) & (self.sett.mini_sub_model_nEGIDs < gwr['EGID'].nunique()): 
                    # gwr = copy.deepcopy(gwr.head(self.sett.mini_sub_model_nEGIDs))
                    gwr = copy.deepcopy(gwr.sample(n=self.sett.mini_sub_model_nEGIDs, random_state=self.sett.ALGOspec_rand_seed))
                elif (self.sett.mini_sub_model_nEGIDs is not None) & (self.sett.mini_sub_model_nEGIDs >= gwr['EGID'].nunique()):
                    gwr = copy.deepcopy(gwr)  


            elif self.sett.mini_sub_model_by_X == 'by_EGID':
                gwr_selected = []
                egid_in_gwr = [egid for egid in self.sett.mini_sub_model_select_EGIDs if egid in gwr['EGID'].unique()]
                gwr_select_EGID = copy.deepcopy(gwr.loc[gwr['EGID'].isin(egid_in_gwr)])
                rest_to_sample = self.sett.mini_sub_model_nEGIDs - len(egid_in_gwr)
                gwr_selected.append(gwr_select_EGID)

                if (rest_to_sample > 0) & (rest_to_sample < gwr['EGID'].nunique()):
                    gwr = copy.deepcopy(gwr.loc[~gwr['EGID'].isin(egid_in_gwr)])
                    gwr_rest = copy.deepcopy(gwr.sample(n=rest_to_sample, random_state=self.sett.ALGOspec_rand_seed))
                        
                elif (rest_to_sample > 0) & (rest_to_sample >= gwr['EGID'].nunique()):
                    gwr_rest = copy.deepcopy(gwr)

                gwr_selected.append(gwr_rest)
                gwr = pd.concat(gwr_selected, axis=0)
                    
                gwr = copy.deepcopy(gwr)
                

        # summary prints ----------
        print_to_logfile('\nEGID selection for TOPOLOGY:', self.sett.summary_name)
        checkpoint_to_logfile('Loop for topology creation over GWR EGIDs', self.sett.summary_name, 0, True)
        checkpoint_to_logfile('In Total: {gwr["EGID"].nunique()} gwrEGIDs ({round(gwr["EGID"].nunique() / gwr_before_solkat_selection["EGID"].nunique() * 100,1)}% of {gwr_before_solkat_selection["EGID"].nunique()} total gwrEGIDs) are used for topology creation', self.sett.summary_name, 0, True)
        checkpoint_to_logfile('  The rest drops out because gwrEGID not present in all data sources', self.sett.summary_name, 0, True)
            
        subtraction1 = gwr_before_solkat_selection["EGID"].nunique() - gwr_before_dsonode_selection["EGID"].nunique()
        checkpoint_to_logfile(f'  > {subtraction1} ({round(subtraction1 / gwr_before_solkat_selection["EGID"].nunique()*100,1)} % ) gwrEGIDs missing in solkat', self.sett.summary_name, 0, True)
            
        subtraction2 = gwr_before_dsonode_selection["EGID"].nunique() - gwr["EGID"].nunique()
        checkpoint_to_logfile(f'  > {subtraction2} ({round(subtraction2 / gwr_before_dsonode_selection["EGID"].nunique()*100,1)} % ) gwrEGIDs missing in dsonodes', self.sett.summary_name, 0, True)
        # if Map_appendings_df.shape[0] > 0:

        #     checkpoint_to_logfile(f'  > (REMARK: Even matched {Map_appendings_df.shape[0]} EGIDs matched artificially to gridnode, because EGID lies in close node range, max_distance_m_for_EGID_node_matching: {self.sett.TECspec_max_distance_m_for_EGID_node_matching} meters', self.sett.summary_name, 0, True)
        # elif Map_appendings_df.shape[0] == 0:
        #     checkpoint_to_logfile(f'  > (REMARK: No EGIDs matched to nearest gridnode, max_distance_m_for_EGID_node_matching: {self.sett.TECspec_max_distance_m_for_EGID_node_matching} meters', self.sett.summary_name, 0, True)



        # CREATE TOPOLOGY ============================================================================
        print_to_logfile('start creating topology - Taking EGIDs from GWR', self.sett.log_name)
        log_str1 = f'Of {gwr["EGID"].nunique()} gwrEGIDs, {len(np.intersect1d(gwr["EGID"].unique(), solkat["EGID"].unique()))} covered by solkatEGIDs ({round(len(np.intersect1d(gwr["EGID"].unique(), solkat["EGID"].unique()))/gwr["EGID"].nunique()*100,2)} % covered)'
        log_str2 = f'Solkat specs (WTIH assigned EGID): {solkat.loc[solkat["EGID"] !="", "SB_UUID"].nunique()} of {solkat.loc[:, "SB_UUID"].nunique()} ({round((solkat.loc[solkat["EGID"] !="", "SB_UUID"].nunique() / solkat.loc[:, "SB_UUID"].nunique())*100,2)} %); {solkat.loc[solkat["EGID"] !="", "DF_UID"].nunique()} of {solkat.loc[:, "DF_UID"].nunique()} ({round((solkat.loc[solkat["EGID"] !="", "DF_UID"].nunique() / solkat.loc[:, "DF_UID"].nunique())*100,2)} %)'
        checkpoint_to_logfile(log_str1, self.sett.log_name)
        checkpoint_to_logfile(log_str2, self.sett.log_name)


        # start loop ------------------------------------------------
        topo_egid = {}
        modulus_print = int(len(gwr['EGID'])//5)
        CHECK_egid_with_problems = []
        print_to_logfile('\n', self.sett.log_name)
        checkpoint_to_logfile('start attach to topo', self.sett.log_name, 0 , True)

        # transform to np.array for faster lookups
        pv_npry, gwr_npry, elecpri_npry = np.array(pv), np.array(gwr), np.array(elecpri) 



        for i, egid in enumerate(gwr['EGID']):
                    
            # add pv data --------
            pv_inst = {
                'inst_TF': False,
                'info_source': '',
                'xtf_id': '',
                'BeginOp': '',
                'InitialPower': 0.0,
                'TotalPower': 0.0,
                'dfuid_w_inst_tuples': [], 
            }
            egid_without_pv = []
            Map_xtf = Map_egid_pv.loc[Map_egid_pv['EGID'] == egid, 'xtf_id']

            if Map_xtf.empty:
                egid_without_pv.append(egid)

            elif not Map_xtf.empty:
                xtfid = Map_xtf.iloc[0]
                if xtfid not in pv['xtf_id'].values:
                    checkpoint_to_logfile(f'---- pv xtf_id {xtfid} in Mapping_egid_pv, but NOT in pv data', self.sett.log_name, 0, False)
                        
                if (Map_xtf.shape[0] == 1) and (xtfid in pv['xtf_id'].values):
                    mask_xtfid = np.isin(pv_npry[:, pv.columns.get_loc('xtf_id')], [xtfid,])

                    pv_inst['inst_TF'] = True
                    pv_inst['info_source'] = 'pv_df'
                    pv_inst['xtf_id'] = str(xtfid)
                    pv_inst['dfuid_w_inst_tuples'] = []
                        
                    pv_inst['BeginOp'] = pv_npry[mask_xtfid, pv.columns.get_loc('BeginningOfOperation')][0]
                    pv_inst['InitialPower'] = pv_npry[mask_xtfid, pv.columns.get_loc('InitialPower')][0]
                    pv_inst['TotalPower'] = pv_npry[mask_xtfid, pv.columns.get_loc('TotalPower')][0]
                    # pv_inst['TotalPower_'] = pv_npry[mask_xtfid, pv.columns.get_loc('TotalPower_pv_df')][0]
                    # pv_inst['BeginOp'] = pv.loc[pv['xtf_id'] == xtfid, 'BeginningOfOperation'].iloc[0]
                    # pv_inst['InitialPower'] = pv.loc[pv['xtf_id'] == xtfid, 'InitialPower'].iloc[0]
                    # pv_inst['TotalPower'] = pv.loc[pv['xtf_id'] == xtfid, 'TotalPower'].iloc[0]
                        
                elif Map_xtf.shape[0] > 1:
                    checkpoint_to_logfile(f'ERROR: multiple xtf_ids for EGID: {egid}', self.sett.log_name, 0, self.sett.show_debug_prints)
                    CHECK_egid_with_problems.append((egid, 'multiple xtf_ids'))


            # add solkat data --------
            if egid in solkat['EGID'].unique():
                solkat_sub = solkat.loc[solkat['EGID'] == egid]
                if solkat.duplicated(subset=['DF_UID', 'EGID']).any():
                    solkat_sub = solkat_sub.drop_duplicates(subset=['DF_UID', 'EGID'])
                solkat_partitions = solkat_sub.set_index('DF_UID')[['FLAECHE', 'STROMERTRAG', 'AUSRICHTUNG', 'NEIGUNG', 'MSTRAHLUNG', 'GSTRAHLUNG']].to_dict(orient='index')                   
                
            elif egid not in solkat['EGID'].unique():
                solkat_partitions = {}
                checkpoint_to_logfile(f'egid {egid} not in solkat', self.sett.log_name, 0, self.sett.show_debug_prints)


            # add pvtarif --------
            bfs_of_egid = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GGDENR')][0]
            if self.sett.TECspec_pvtarif_Rp_kWh is None:
                pvtarif_egid = sum([pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), col].sum() for col in pvtarif_col])

                pvtarif_sub = pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid)]
                if pvtarif_sub.shape[0] == 1:
                    ewr_info = {
                        'nrElcom': pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), 'nrElcom'].iloc[0],
                        'name': pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), 'nomEw'].iloc[0],
                        'energy1': pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), 'energy1'].sum()     if 'energy1' in pvtarif_col else 0.0,
                        'eco1': pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), 'eco1'].sum()           if 'eco1' in pvtarif_col else 0.0,
                    }
                elif pvtarif_sub.shape[0] > 1:
                    ewr_info = {
                        'nrElcom': pvtarif_sub['nrElcom'].unique().tolist(),
                        'name': pvtarif_sub['nomEw'].unique().tolist(),
                        'energy1': pvtarif_sub['energy1'].mean()    if 'energy1' in pvtarif_col else 0.0,
                        'eco1': pvtarif_sub['eco1'].mean()          if 'eco1' in pvtarif_col else 0.0,
                    }
                    
                    # checkpoint_to_logfile(f'multiple pvtarif data for EGID {egid}', self.sett.log_name, 0, self.sett.show_debug_prints)
                    CHECK_egid_with_problems.append((egid, 'multiple pvtarif data'))


            elif isinstance(self.sett.TECspec_pvtarif_Rp_kWh, (int, float)):
                pvtarif_egid = self.sett.TECspec_pvtarif_Rp_kWh
                ewr_info = {
                    'nrElcom': '-',
                    'name': 'generic pvtarif',
                    'energy1': '-',
                    'eco1': '-',
                }

            elif isinstance(self.sett.TECspec_pvtarif_Rp_kWh, str):
                pvtarif_egid = None     # hourly pvtarif is added to the subdf later (add_pvtarif_to_subdf), no static value per EGID
                ewr_info = {
                    'nrElcom': '-',
                    'name': 'hourly pvtarif',
                    'energy1': '-',
                    'eco1': '-',
                }



            # add elecpri --------
            elecpri_egid = {}
            elecpri_info = {}

            if self.sett.TECspec_elecpri_Rp_kWh is None:

                mask_bfs = np.isin(elecpri_npry[:, elecpri.columns.get_loc('bfs_number')], [bfs_of_egid,]) 
                mask_year = np.isin(elecpri_npry[:, elecpri.columns.get_loc('year')],    self.sett.TECspec_elecpri_year)
                mask_cat = np.isin(elecpri_npry[:, elecpri.columns.get_loc('category')], self.sett.TECspec_elecpri_category)

                if sum(mask_bfs & mask_year & mask_cat) < 1:
                    checkpoint_to_logfile(f'ERROR: no elecpri data for EGID {egid}', self.sett.log_name, 0, self.sett.show_debug_prints)
                    CHECK_egid_with_problems.append((egid, 'no elecpri data'))
                elif sum(mask_bfs & mask_year & mask_cat) > 1:
                    checkpoint_to_logfile(f'ERROR: multiple elecpri data for EGID {egid}', self.sett.log_name, 0, self.sett.show_debug_prints)
                    CHECK_egid_with_problems.append((egid, 'multiple elecpri data'))
                elif sum(mask_bfs & mask_year & mask_cat) == 1:
                    energy =   elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('energy')].sum()
                    grid =     elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('grid')].sum()
                    aidfee =   elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('aidfee')].sum()
                    taxes =    elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('taxes')].sum()
                    fixcosts = elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('fixcosts')].sum()

                    elecpri_egid = energy + grid + aidfee + taxes # + fixcosts
                    elecpri_info = {
                        'energy': energy,
                        'grid': grid,
                        'aidfee': aidfee,
                        'taxes': taxes,
                        'fixcosts': fixcosts,
                    }

            elif isinstance(self.sett.TECspec_elecpri_Rp_kWh, float):
                elecpri_egid = self.sett.TECspec_elecpri_Rp_kWh
                elecpri_info = {
                    'energy': '- (generic elecpri)',
                    'grid': '- (generic elecpri)',
                    'aidfee': '- (generic elecpri)',
                    'taxes': '- (generic elecpri)',
                    'fixcosts': '- (generic elecpri)',
                }
            elif isinstance(self.sett.TECspec_elecpri_Rp_kWh, str):
                elecpri_info = {
                    'energy': '- (hourly elecpri)',
                    'grid': '- (hourly elecpri)',
                    'aidfee': '- (hourly elecpri)',
                    'taxes': '- (hourly elecpri)',
                    'fixcosts': '- (hourly elecpri)',
                }


            # add GWR --------
            if all([col in gwr.columns for col in ['GWAERZH2', 'GENH2']]):
                gwaerzh1  = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GWAERZH1')][0]
                gwaerzh2  = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GWAERZH2')][0]
                genh2     = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GENH2')][0]
                if any([tech in self.sett.TECspec_heatpump_indicator for tech in [gwaerzh1, gwaerzh2, ]]):
                    heating_system = 'heatpump'
                else:
                    heating_system = 'no_heatpump'
            else:
                gwaerzh1  = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GWAERZH1')][0]
                gwaerzh2  = None
                genh2     = None
                if any([gwaerzh1 in self.sett.TECspec_heatpump_indicator]):
                    heating_system = 'heatpump'
                else:
                    heating_system = 'no_heatpump'


            gwr_info ={
                'bfs':            gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GGDENR')][0],
                'gklas':          gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GKLAS')][0],
                'garea':          gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GAREA')][0],
                'gstat':          gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GSTAT')][0],
                'gbauj':          gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GBAUJ')][0],
                'are_typ':        gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('ARE_typ')][0],
                'sfhmfh_typ':     gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('sfhmfh_typ')][0],
                'genh1':          gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GENH1')][0],   
                'genh2':          genh2, 
                'gwaerzh1':       gwaerzh1,
                'gwaerzh2':       gwaerzh2,
                'heating_system': heating_system,
            }
                

            # add demand type --------
            demand_arch_typ         = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('arch_typ')][0]
            demand_elec_pGAREA  = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('demand_elec_pGAREA')][0]


            # add grid node --------
            if isinstance(Map_egid_dsonode.loc[egid, 'grid_node'], str):
                grid_node = Map_egid_dsonode.loc[egid, 'grid_node']
            elif isinstance(Map_egid_dsonode.loc[egid, 'grid_node'], pd.Series):
                grid_node = Map_egid_dsonode.loc[egid, 'grid_node'].iloc[0]

                
            # add subsidy (placeholder) --------
            subsidy_egid = {
                'dfuid_subsidy_fix_chf': 0.0, 
                'node_subsidy_fix_chf': 0.0,
            }

                        

            # attach to topo --------
            # topo['EGID'][egid] = {
            topo_egid[egid] = {
                'gwr_info': gwr_info,
                'grid_node': grid_node,
                'pv_inst': pv_inst,
                'solkat_partitions': solkat_partitions, 
                'demand_arch_typ': demand_arch_typ,
                'demand_elec_pGAREA': demand_elec_pGAREA,
                'pvtarif_Rp_kWh': pvtarif_egid, 
                'EWR': ewr_info, 
                'elecpri_Rp_kWh': elecpri_egid,
                'elecpri_info': elecpri_info,
                'subsidy': subsidy_egid,
                }  

            # Checkpoint prints
            if i % modulus_print == 0:
                print_to_logfile(f'\t -- EGID {i} of {len(gwr["EGID"])} {15*"-"}', self.sett.log_name)

                
        # end loop ------------------------------------------------
        checkpoint_to_logfile('end attach to topo', self.sett.log_name, 0 , True)
        print_to_logfile('\nsanity check for installtions in topo_egid', self.sett.summary_name)
        checkpoint_to_logfile(f'number of EGIDs with multiple installations: {CHECK_egid_with_problems.count("multiple xtf_ids")}', self.sett.summary_name)


        # EXPORT TOPO + Mappings ============================================================================
            
        with open(f'{self.sett.name_dir_export_path}/topo_egid.json', 'w') as f:
            json.dump(topo_egid, f)
        with open(f'{self.sett.name_dir_export_path}/topo_egid.txt', 'w') as f:
            f.write(str(topo_egid))


        # Export CHECK_egid_with_problems to txt file for trouble shooting
        with open(f'{self.sett.name_dir_export_path}/CHECK_egid_with_problems.txt', 'w') as f:
            f.write(f'\n ** EGID with problems: {len(CHECK_egid_with_problems)} **\n\n')
            f.write(str(CHECK_egid_with_problems))

        CHECK_egid_with_problems_dict = {egid: problem for egid, problem in CHECK_egid_with_problems}
        with open(f'{self.sett.name_dir_export_path}/CHECK_egid_with_problems.json', 'w') as f:
            json.dump(CHECK_egid_with_problems_dict, f)


        # EXPORT ============================================================================
        # pvalloc_run folder gets crowded, > only keep the most important files
        df_names = ['Map_egid_pv', 'solkat_month', 'pv', 'pvtarif', 'elecpri', 'Map_egid_dsonode', 'angle_tilt_df',]      # 'dsonodes_df', 'dsonodes_gdf', 
        df_list =  [ Map_egid_pv,   solkat_month,   pv,   pvtarif,   elecpri,   Map_egid_dsonode,   angle_tilt_df,]      #  dsonodes_df,   dsonodes_gdf,  
        for i, m in enumerate(df_list): 
            if isinstance(m, pd.DataFrame):
                m.to_parquet(f'{self.sett.name_dir_export_path}/{df_names[i]}.parquet')
            elif isinstance(m, dict):
                with open(f'{self.sett.name_dir_export_path}/{df_names[i]}.json', 'w') as f:
                    json.dump(m, f)        
            elif isinstance(m, gpd.GeoDataFrame):
                m.to_file(f'{self.sett.name_dir_export_path}/{df_names[i]}.geojson', driver='GeoJSON')


        # RETURN OBJECTS ============================================================================
        return topo_egid, df_list, df_names
        # setup --------
        print_to_logfile('run function: get_estim_instcost_function', self.sett.log_name)

        with open(f'{self.sett.name_dir_export_path}/pvinstcost_coefficients.json', 'r') as file:
            pvinstcost_coefficients = json.load(file)
        params_pkW = pvinstcost_coefficients['params_pkW']
        # coefs_total = pvinstcost_coefficients['coefs_total']
        params_total = pvinstcost_coefficients['params_total']

        # PV Cost functions --------
        # Define the interpolation functions using the imported coefficients
        def func_chf_pkW(x, a, b, c, d):
            return a +  d*((x ** b) /  (x ** c))
        def estim_instcost_chfpkW(x):
            return func_chf_pkW(x, *params_pkW)

        def func_chf_total(x, a, b, c):
            return a +  b*(x**c) 
        def estim_instcost_chftotal(x):
            return func_chf_total(x, *params_total)
            
        return estim_instcost_chfpkW, estim_instcost_chftotal




    def initial_lrg_import_ts_data(self,):
        """
        Import: 
            - PVAllocScenario + PVAllocScenario_Settings dataclass containing all scenarios settings + methods
        Tasks: 
            - import all time series data needed for PV allocation algorithm
        Export to pvalloc dir: 
            - all the time series data objects that are needed for later calculations
        """

        # import settings + setup -------------------
        print_to_logfile('run function: import_ts_data', self.sett.log_name)


        # create time structure for TS
        T0 = pd.to_datetime(f'{self.sett.T0_prediction}')
        start_loockback = T0 - pd.DateOffset(months = self.sett.months_lookback) # + pd.DateOffset(hours=1)
        end_prediction = T0 + pd.DateOffset(months = self.sett.months_prediction) - pd.DateOffset(hours=1)
        date_range = pd.date_range(start=start_loockback, end=end_prediction, freq='h')
        checkpoint_to_logfile(f'import TS: lookback range   {start_loockback} to {T0-pd.DateOffset(hours=1)}', self.sett.log_name, 0)
        checkpoint_to_logfile(f'import TS: prediction range {T0} to {end_prediction}', self.sett.log_name, 0)

        Map_daterange = pd.DataFrame({'date_range': date_range, 'DoY': date_range.dayofyear, 'hour': date_range.hour})
        Map_daterange['HoY'] = (Map_daterange['DoY'] - 1) * 24 + (Map_daterange['hour']+1)
        Map_daterange['t'] = Map_daterange['HoY'].apply(lambda x: f't_{x}')


        # IMPORT ----------------------------------------------------------------------------
        demandtypes_ts = pd.read_parquet(f'{self.sett.name_dir_import_path}/demandtypes_ts.parquet')

        nas =   sum([demandtypes_ts[col].isna().sum() for col in demandtypes_ts.columns])
        nulls = sum([demandtypes_ts[col].isnull().sum() for col in demandtypes_ts.columns])
        checkpoint_to_logfile(f'sanity check demand_ts: {nas} NaNs or {nulls} Nulls for any column in df', self.sett.log_name)


        # meteo (radiation & temperature) --------
        meteo_col_dir_radiation =  self.sett.WEAspec_meteo_col_dir_radiation
        meteo_col_diff_radiation = self.sett.WEAspec_meteo_col_diff_radiation
        meteo_col_temperature =    self.sett.WEAspec_meteo_col_temperature
        weater_year =              self.sett.WEAspec_weather_year

        meteo = pd.read_parquet(f'{self.sett.name_dir_import_path}/meteo.parquet')
        meteo_cols = ['timestamp', meteo_col_dir_radiation, meteo_col_diff_radiation, meteo_col_temperature]
        meteo = meteo.loc[:,meteo_cols]

        # get radiation
        meteo['rad_direct'] = meteo[meteo_col_dir_radiation]
        meteo['rad_diffuse'] = meteo[meteo_col_diff_radiation]
        meteo.drop(columns=[meteo_col_dir_radiation, meteo_col_diff_radiation], inplace=True)

        # get temperature
        meteo['temperature'] = meteo[meteo_col_temperature]
        meteo.drop(columns=meteo_col_temperature, inplace=True)

        start_wy, end_wy = pd.to_datetime(f'{weater_year}-01-01 00:00:00'), pd.to_datetime(f'{weater_year}-12-31 23:00:00')
        meteo = meteo.loc[(meteo['timestamp'] >= start_wy) & (meteo['timestamp'] <= end_wy)]

        meteo['t']= meteo['timestamp'].apply(lambda x: f't_{(x.dayofyear -1) * 24 + x.hour +1}')
        meteo_ts = meteo.copy()



        # # grid premium --------
        # if not self.sett.test_faster_array_computation:
        #     # setup 
        #     if os.path.exists(f'{self.sett.name_dir_export_path}/gridprem_ts.parquet'):
        #         os.remove(f'{self.sett.name_dir_export_path}/gridprem_ts.parquet')    

        #     # import 
        #     dsonodes_df = pd.read_parquet(f'{self.sett.name_dir_import_path}/dsonodes_df.parquet')
        #     t_range = [f't_{t}' for t in range(1,8760 + 1)]

        #     gridprem_ts = pd.DataFrame(np.repeat(dsonodes_df.values, len(t_range), axis=0), columns=dsonodes_df.columns)  
        #     gridprem_ts['t'] = np.tile(t_range, len(dsonodes_df))
        #     gridprem_ts['prem_Rp_kWh'] = 0

        #     gridprem_ts = gridprem_ts[['t', 'grid_node', 'kVA_threshold', 'prem_Rp_kWh']]
        #     gridprem_ts.drop(columns='kVA_threshold', inplace=True)

        #     # export 
        #     gridprem_ts.to_parquet(f'{self.sett.name_dir_export_path}/gridprem_ts.parquet')

            

        # EXPORT ----------------------------------------------------------------------------
        ts_names = ['Map_daterange', 'demandtypes_ts', 'meteo_ts', ] #'gridprem_ts' ]
        ts_list =  [ Map_daterange,   demandtypes_ts,   meteo_ts,  ] # gridprem_ts]
        for i, ts in enumerate(ts_list):
            ts.to_parquet(f'{self.sett.name_dir_export_path}/{ts_names[i]}.parquet')


        # RETURN ----------------------------------------------------------------------------
        return ts_list, ts_names




    def initial_define_trange_prediction(self,):
        """
        Input: 
            - PVAllocScenario + PVAllocScenario_Settings dataclass containing all scenarios settings + methods
        Tasks:
            - define the time range for the prediction period based on the settings
        Export to pvalloc dir:

        """
        # import settings + setup -------------------
        print_to_logfile('run function: initial_define_trange_prediction.py', self.sett.log_name)

        # create monthly time structure
        T0 = pd.to_datetime(f'{self.sett.T0_year_prediction}-01-01 00:00:00')
        start_loockback = T0 - pd.DateOffset(months=self.sett.months_lookback) #+ pd.DateOffset(hours=1)
        end_prediction = T0 + pd.DateOffset(months=self.sett.months_prediction) - pd.DateOffset(hours=1)
        months_lookback = pd.date_range(start=start_loockback, end=T0, freq='ME').to_period('M')

        prediction_year_diff = end_prediction.year - T0.year
        if prediction_year_diff > 0:
            trange_prediction = pd.date_range(start=T0, end=end_prediction, freq='YE')
        elif prediction_year_diff == 0:
            trange_prediction = pd.date_range(start=T0, end=end_prediction + pd.DateOffset(capa_years=1), freq='YE')

        trange_prediction_df = pd.DataFrame({'n_iter': range(1,len(trange_prediction) + 1), 'date': trange_prediction, 
                                             'year': trange_prediction.year, 'month': trange_prediction.month, })

        trange_prediction_df.to_parquet(f'{self.sett.name_dir_export_path}/trange_prediction.parquet')
        trange_prediction_df.to_csv(f'{self.sett.name_dir_export_path}/trange_prediction.csv', index=False)

        return trange_prediction, months_lookback


    def initial_lrg_define_construction_capacity(self,
                                                topo,
                                                trange_prediction, months_lookback, 
                                                df_list, df_names, 
                                                ts_list, ts_names, 
                                                ): 
        """
        Input: 
            -   PVAllocScenario + PVAllocScenario_Settings dataclass containing all scenarios settings + methods
            - topo_egd, df_list and ts_list (incl df_names and ts_names to be callabale by name)
        Tasks:
            - import pv data set and calculate the constructed capacity in the defined lookback period and sample range
            - based on assumptions, calculate the capacity for every month and total year in the prediciton period. 
        Export to pvalloc dir:
            - constrcapa.parquet    ->
        """

        # IMPORT ----------------------------------------------------------------------------
        print_to_logfile('run function: initial_lrg_define_construction_capacity.py', self.sett.log_name)

        gwr_allch_summary = pl.read_parquet(f'{self.sett.name_dir_import_path}/gwr_all_ch_summary.parquet')
        gwr_all_building_df = pl.read_parquet(f'{self.sett.name_dir_import_path}/gwr_all_building_df.parquet')
        pv = df_list[df_names.index('pv')]
        Map_egid_pv = df_list[df_names.index('Map_egid_pv')]
        topo_keys = list(topo.keys())
        
        T0 = pd.to_datetime(f'{self.sett.T0_year_prediction}-01-01 00:00:00')
        start_loockback = T0 - pd.DateOffset(months=self.sett.months_lookback) #+ pd.DateOffset(hours=1)


        # # debugging -------------------
        # pv          = pl.read_parquet(r"C:\Models\OptimalPV_RH\data\pvalloc\pvalloc_29nbfs_LRG2_max\pv.parquet").to_pandas()
        # Map_egid_pv = pd.read_parquet(r"C:\Models\OptimalPV_RH\data\pvalloc\pvalloc_29nbfs_LRG2_max\Map_egid_pv.parquet")
        # topo = json.load(open(r"C:\Models\OptimalPV_RH\data\pvalloc\pvalloc_29nbfs_LRG2_max\topo_egid.json", 'r')   )
        # topo_keys = list(topo.keys())
        # self.sett.bfs_numbers =  [str(bfs) for bfs in [
        #     # RURAL 
        #     2612, 2889, 2883, 2621, 2622,
        #     2620, 2615, 2614, 2616, 2480,
        #     2617, 2611, 2788, 2619, 2783, 2477, 
        #     # SUBURBAN
        #     2613, 2782, 2618, 2786, 2785, 
        #     2772, 2761, 2743, 2476, 2768,
        #     # URBAN
        #     2773, 2769, 2770,
        #     ]]
        # # debugging -------------------

        # subset pv to EGIDs in TOPO, and LOOKBACK period of pvalloc settings
        pv_sub = copy.deepcopy(pv)
        del_cols = ['MainCategory', 'SubCategory', 'PlantCategory']
        pv_sub.drop(columns=del_cols, inplace=True)

        pv_sub = pv_sub.merge(Map_egid_pv, how='left', on='xtf_id')
        pv_sub = pv_sub.loc[pv_sub['EGID'].isin(topo_keys)]
        pv_plot = copy.deepcopy(pv_sub) # used for plotting later

        pv_sub['BeginningOfOperation'] = pd.to_datetime(pv_sub['BeginningOfOperation'])
        pv_sub['MonthPeriod'] = pv_sub['BeginningOfOperation'].dt.to_period('M')
        pv_sub_idx = pv_sub['MonthPeriod'].isin(months_lookback)
        if pv_sub_idx.sum() >= 1:
            pv_sub = pv_sub.loc[pv_sub_idx]
        elif pv_sub_idx.sum() < 1: 
            pv_sub = pv_sub.loc[pv_sub['MonthPeriod'] == max(pv_sub['MonthPeriod'])]


        # HISTORIC CAPACITY ASSIGNMENT ----------------------------------------------------------------------------
        if True: 
            capacity_growth = self.sett.CSTRspec_ann_capacity_growth
            month_constr_capa_tuples = self.sett.CSTRspec_month_constr_capa_tuples
            n_years_lookback = len(months_lookback) / 12
            sum_TP_kW_lookback = pv_sub['TotalPower'].sum() / n_years_lookback

            # per year
            # elif self.sett.CSTRspec_iter_time_unit == 'year':
            constrcapa_hist_year = pd.DataFrame({'date': trange_prediction, 'year': trange_prediction.year, 'month': trange_prediction.month})

            capa_years_prediction = trange_prediction.year.unique()
            for i,y in enumerate(capa_years_prediction):
                TP_y = sum_TP_kW_lookback * (1 + capacity_growth)**(i+1)
                constrcapa_hist_year.loc[(constrcapa_hist_year['year'] == y), 'constr_capacity_kw'] = TP_y 
             
            

        # EP2050+ CAPACITY ASSIGNMENT ----------------------------------------------------------------------------
        if True:
            # extract ep2050 settings data
            def build_eb2050_pvcapa_df(ep2050_zerobasis_dict):
                capa_year_values = ep2050_zerobasis_dict['pvcapa_total']
                share_year_values = ep2050_zerobasis_dict['share_instclass']
                    
                capa_years =  [int(year) for year in capa_year_values.keys()]
                share_years = [int(year) for year in share_year_values.keys()]
                    
                # Define the full year range from min to max year
                min_year_int = min(capa_years)
                max_year_int = max(capa_years)
                epzb_year_range = [year for year in range(min_year_int, max_year_int + 1)]

                epzb_capa_value_list = []
                class1to4_tuple_list = []
                for year in epzb_year_range: 
                    if year in capa_years:
                        capa_value = capa_year_values[str(year)]
                    else:
                        lower_year = max([y for y in capa_years if y < year])
                        upper_year = min([y for y in capa_years if y > year])
                        lower_value = capa_year_values[str(lower_year)]
                        upper_value = capa_year_values[str(upper_year)]

                        capa_value = lower_value + (upper_value - lower_value) * ((year - lower_year) / (upper_year - lower_year))
                    epzb_capa_value_list.append(capa_value)

                    if year in share_years:
                        class1to4_tuple = (
                            share_year_values[str(year)]['class1'],
                            share_year_values[str(year)]['class2'],
                            share_year_values[str(year)]['class3'],
                            share_year_values[str(year)]['class4'],
                        )
                    elif year <= max(share_years) and year >= min(share_years):
                        lower_year = max([y for y in share_years if y < year])
                        # upper_year1 = min([y for y in share_years if y > year else max(share_years)])
                        upper_year = min([y for y in share_years if y > year], default=max(share_years))
                        lower_tuple = (
                            share_year_values[str(lower_year)]['class1'],
                            share_year_values[str(lower_year)]['class2'],
                            share_year_values[str(lower_year)]['class3'],
                            share_year_values[str(lower_year)]['class4'],
                        )
                        upper_tuple = (
                            share_year_values[str(upper_year)]['class1'],
                            share_year_values[str(upper_year)]['class2'],
                            share_year_values[str(upper_year)]['class3'],
                            share_year_values[str(upper_year)]['class4'],
                        )
                        class1to4_tuple = tuple(
                            lower + (upper - lower) * ((year - lower_year) / (upper_year - lower_year))
                            for lower, upper in zip(lower_tuple, upper_tuple)
                        )
                    elif year > max(share_years):
                        class1to4_tuple = (
                            share_year_values[str(max(share_years))]['class1'],
                            share_year_values[str(max(share_years))]['class2'],
                            share_year_values[str(max(share_years))]['class3'],
                            share_year_values[str(max(share_years))]['class4'],
                        )
                    class1to4_tuple_list.append(class1to4_tuple)

                epzb_capa_df = pd.DataFrame({
                    'date': [pd.to_datetime(f'{year}-12-31') for year in epzb_year_range],
                    'year': epzb_year_range, 
                    'month': [int(12) for i in epzb_year_range],
                    'epzb_capa_GW': epzb_capa_value_list, 
                    'epzb_capa_kw': [value * 1e6 for value in epzb_capa_value_list],
                    'class1': [t[0] for t in class1to4_tuple_list],
                    'class2': [t[1] for t in class1to4_tuple_list],
                    'class3': [t[2] for t in class1to4_tuple_list],
                    'class4': [t[3] for t in class1to4_tuple_list],
                })

                return epzb_capa_df
                           
            epzb_capa_df = build_eb2050_pvcapa_df(self.sett.CSTRspec_ep2050_capa_dict['ep2050_zerobasis'])


            # adjust allCH capa to sample size
            GSTAT_adj_list      = self.sett.CSTRspec_ep2050_capa_dict['ep2050_zerobasis']['CHcapa_adjustment_filter']['GSTAT_list']
            GKLAS_adj_list      = self.sett.CSTRspec_ep2050_capa_dict['ep2050_zerobasis']['CHcapa_adjustment_filter']['GKLAS_list']
            classes_adj_list    = self.sett.CSTRspec_ep2050_capa_dict['ep2050_zerobasis']['CHcapa_adjustment_filter']['classes_adj_list']
            epzb_scaling_factor = self.sett.CSTRspec_ep2050_rescale_fact

            nEGIDs_all_CH = gwr_allch_summary \
                .filter( 
                    (pl.col('GSTAT').is_in(GSTAT_adj_list)) & 
                    (pl.col('GKLAS').is_in(GKLAS_adj_list)) ) \
                .select('nEGID').sum().item()

            # nEGIDs_SAMPLE = len(topo)
            nEGIDs_SAMPLE = gwr_allch_summary \
                .filter( 
                    (pl.col('GSTAT').is_in(GSTAT_adj_list)) & 
                    (pl.col('GKLAS').is_in(GKLAS_adj_list)) & 
                    (pl.col('BFS_NUMMER').is_in(self.sett.bfs_numbers)) ) \
                .select('nEGID').sum().item()
            # identical coutcome
            # gwr_in_scenbfs = gwr_all_building_df.filter(
            #     (pl.col('GSTAT').is_in(GSTAT_adj_list)) & 
            #     (pl.col('GKLAS').is_in(GKLAS_adj_list)) & 
            #     (pl.col('GGDENR').is_in(self.sett.bfs_numbers))
            #     )
            # gwr_in_scenbfs.shape[0]
                
            epzb_capa_df['ratio_sample_allCH'] = nEGIDs_SAMPLE / nEGIDs_all_CH
            epzb_capa_df['epzb_capa_sample_kw'] = epzb_capa_df['epzb_capa_kw'] * epzb_capa_df[classes_adj_list].sum(axis=1) * epzb_capa_df['ratio_sample_allCH'] 
            epzb_capa_df['constr_capacity_kw'] = epzb_capa_df['epzb_capa_sample_kw'] * epzb_scaling_factor
            constrcapa_epzb = epzb_capa_df[['date', 'year', 'month', 'constr_capacity_kw', 'epzb_capa_kw', 'class1', 'class2', 'class3', 'class4']].copy()
                

        # ADJUSTED HISTORIC CAPACITY ASSIGNMENT ----------------------------------------------------------------------------
        if True: 
            n_adj_months_lookback  = self.sett.CSTRspec_histadj_dict['adj_months_lookback']
            adjann_capacity_growth = self.sett.CSTRspec_histadj_dict['adjann_capacity_growth']
            adj_target_margin      = self.sett.CSTRspec_histadj_dict['adj_target_margin']
            adj_capa_asym_2050 = constrcapa_epzb.loc[epzb_capa_df['year'] == 2050, 'constr_capacity_kw'].values[0]


            adj_start_loockback = T0 - pd.DateOffset(months=n_adj_months_lookback) #+ pd.DateOffset(hours=1)
            adj_months_lookback = pd.date_range(start=adj_start_loockback, end=T0, freq='ME').to_period('M')

            # subset pv to EGIDs in TOPO, and LOOKBACK period of pvalloc settings
            pv_sub = copy.deepcopy(pv)
            del_cols = ['MainCategory', 'SubCategory', 'PlantCategory']
            pv_sub.drop(columns=del_cols, inplace=True)

            pv_sub = pv_sub.merge(Map_egid_pv, how='left', on='xtf_id')
            pv_sub = pv_sub.loc[pv_sub['EGID'].isin(topo_keys)]

            pv_sub['BeginningOfOperation'] = pd.to_datetime(pv_sub['BeginningOfOperation'])
            pv_sub['MonthPeriod'] = pv_sub['BeginningOfOperation'].dt.to_period('M')
            pv_sub_idx = pv_sub['MonthPeriod'].isin(adj_months_lookback)        # <= different than hist capa!
            if pv_sub_idx.sum() >= 1:
                pv_sub = pv_sub.loc[pv_sub_idx]
            elif pv_sub_idx.sum() < 1: 
                pv_sub = pv_sub.loc[pv_sub['MonthPeriod'] == max(pv_sub['MonthPeriod'])]

            n_years_lookback = len(adj_months_lookback) / 12
            sum_TP_kW_lookback = pv_sub['TotalPower'].sum() / n_years_lookback

            constrcapa_adjhist_year = pd.DataFrame({'date': trange_prediction, 'year': trange_prediction.year, 'month': trange_prediction.month})


            def capa_sigmoid_func(
                    n_year, 
                    growth_rate     = adjann_capacity_growth,
                    capa_target     = adj_capa_asym_2050,
                    target_margin   = adj_target_margin ,
                    capa_hist_start = sum_TP_kW_lookback, 
                    ):                 
                capa_target2 = capa_target * (1 + target_margin)
                capa_init = (capa_target2 - capa_hist_start) / capa_hist_start
                constrcapa_adj_val = capa_target2 / (1 + capa_init * np.exp(-growth_rate * n_year))

                return constrcapa_adj_val

            capa_years_prediction = trange_prediction.year.unique()
            for i,y in enumerate(capa_years_prediction):   
                # constrcapa_adj_lastyear = constrcapa_adjhist_year.loc[constrcapa_adjhist_year['year'] == y-1, 'constr_capacity_kw'].values[0] if i > 0 else sum_TP_kW_lookback
                # constrcapa_adj_now = constrcapa_adj_lastyear * ((adj_capa_asym_2050 / constrcapa_adj_lastyear) ** adjann_capacity_growth )
                # constrcapa_adjhist_year.loc[(constrcapa_adjhist_year['year'] == y), 'constr_capacity_kw'] = constrcapa_adj_now
                constrcapa_adj_val = capa_sigmoid_func(n_year = i + 1  )
                constrcapa_adjhist_year.loc[(constrcapa_adjhist_year['year'] == y), 'constr_capacity_kw'] = constrcapa_adj_val



        # PLOT  COMPARISON  ----------------------------------------------------------------------------
        # plot total power over time
        pv_plot['BeginningOfOperation'] = pd.to_datetime(pv_plot['BeginningOfOperation'])
        pv_plot.set_index('BeginningOfOperation', inplace=True)

        # Resample by week, month, and year and calculate the sum of TotalPower
        monthly_sum = pv_plot['TotalPower'].resample('ME').sum()
        yearly_sum = pv_plot['TotalPower'].resample('YE').sum()

        if True: 

            fig = go.Figure()
            export_csv_columns_list = []
            # add historic traces
            fig.add_trace(go.Scatter(x=[T0, ], y=[None, ], mode='lines', name='Hist. Built Capa. in Sample ---------------- ', opacity=0.0))
            # fig.add_trace(go.Scatter(x=monthly_sum.index, y=monthly_sum.values, mode='lines+markers', name='Monthly Built', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=yearly_sum.index, y=yearly_sum.values, mode='lines+markers', name='Yearly Built'))
                
            # add historically based constr capcacity 
            fig.add_trace(go.Scatter(x=[T0, ], y=[None, ], mode='lines', name='Hist. Based Future Capa. ---------------- ', opacity=0.0))
            fig.add_vrect(
                x0=start_loockback, 
                x1=T0,
                fillcolor="LightSalmon",
                opacity=0.3,
                layer="below",
                line_width=0,
                annotation_text="Lookback period",
                annotation_position="top left",
            )
            # fig.add_trace(go.Scatter(x=constrcapa_hist_month['date'], y=constrcapa_hist_month['constr_capacity_kw'], mode='lines+markers', name='Monthly Hist. based', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=constrcapa_hist_year['date'], y=constrcapa_hist_year['constr_capacity_kw'], mode='lines+markers', name='constrcapa - Yearly Hist. based (Sample)'))
            export_csv_columns_list.append(pd.DataFrame({'date': constrcapa_hist_year['date'], 
                                                         'constr_capacity_kw_HIST': constrcapa_hist_year['constr_capacity_kw']}))

            for r in [0.05, 0.075, 0.1, 0.125, 0.15, 0.16, 0.17,  0.2]:
                loop_hist_year = []
                for i,y in enumerate(capa_years_prediction):
                    val = sum_TP_kW_lookback * (1 + r)**(i+1)
                    loop_hist_year.append(val)
                fig.add_trace(go.Scatter(x=constrcapa_hist_year['date'], y=loop_hist_year, mode='lines+markers', name=f'Yearly Hist. based + {int(r*100)}% p.a.', line=dict(dash='dot')))
                export_csv_columns_list.append(pd.DataFrame({'date': constrcapa_hist_year['date'], 
                                                             f'constr_capacity_kw_Hist_r{r}': loop_hist_year}))
                            

            # add EP2050 based constr capacity
            classes_str = ', '.join(classes_adj_list) if len(classes_adj_list) > 1 else classes_adj_list[0]
            fig.add_trace(go.Scatter(x=[T0, ], y=[None, ], mode='lines', name='EP2050 Based Future Capa. ---------------- ', opacity=0.0))
            fig.add_trace(go.Scatter(x=[T0, ], y=[None, ], mode='lines', name=f'Adj_factor: {round(nEGIDs_SAMPLE / nEGIDs_all_CH,2) } - {nEGIDs_SAMPLE} nEGIDs in Sample / {nEGIDs_all_CH} nEGIDs allCH', opacity=0.0))
            fig.add_trace(go.Scatter(x=epzb_capa_df['date'], y=epzb_capa_df['epzb_capa_kw']/1000, mode='lines+markers', name='allCH capacity kW (1/1000)',))
            # fig.add_trace(go.Scatter(x=epzb_capa_df['date'], y=epzb_capa_df['epzb_capa_sample_kw'], mode='lines+markers', name='Yearly EP2050 based (Sample, all classes)',))

            rescale_epzb2050 = [0.1, 0.25, 0.5, 0.75, 1.0]
            for rescale in rescale_epzb2050:               
                fig.add_trace(go.Scatter(x=epzb_capa_df['date'], y=epzb_capa_df['epzb_capa_sample_kw'] * rescale, mode='lines+markers', name=f'constrcapa - EP2050 based, rescale: {rescale}'))
                export_csv_columns_list.append(pd.DataFrame({'date': epzb_capa_df['date'], 
                                                             f'constr_capacity_kw_EP2050_rescale{rescale}': epzb_capa_df['epzb_capa_sample_kw'] * rescale}))
                    
            fig.add_trace(go.Scatter(x=epzb_capa_df['date'], y=epzb_capa_df['epzb_capa_sample_kw'],           mode='lines+markers', name=f'constrcapa - size_class_adjusted ({classes_str})',))


            # add adjusted historic capacity
            fig.add_trace(go.Scatter(x=[T0, ], y=[None, ], mode='lines', name='ADJU Hist. to EPZB2050. ---------------- ', opacity=0.0))
            fig.add_vrect(
                x0=adj_start_loockback, 
                x1=T0,
                fillcolor="darkturquoise",
                opacity=0.3,
                layer="below",
                line_width=0,
                annotation_text="Adj lookback period",
                annotation_position="bottom left",
            )

            refactor_histadjust =  [
                # 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                # 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 
                # 1.0, 1.2, 1.4, 1.6, 1.8, 2.0
                # 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                # 5.0, 7.5, 10.0, 12.5, 15.0, 20.0
                0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.9, 
                ]
            for refactor in refactor_histadjust: 
                loop_adj_hist_year = []
                for i,y in enumerate(capa_years_prediction):
                    val = capa_sigmoid_func(n_year = i + 1 , growth_rate = refactor,capa_target = adj_capa_asym_2050, )
                    # val_lastyear = constrcapa_adjhist_year.loc[constrcapa_adjhist_year['year'] == y-1, 'constr_capacity_kw'].values[0] if i > 0 else sum_TP_kW_lookback
                    # val = val_lastyear * (adj_capa_asym_2050 / val_lastyear) ** r
                    loop_adj_hist_year.append(val)
                fig.add_trace(go.Scatter(x=constrcapa_hist_year['date'], y=loop_adj_hist_year, mode='lines+markers', name=f'Adj. Hist. based refact {refactor}', line=dict(dash='dot')))
                export_csv_columns_list.append(pd.DataFrame({'date': constrcapa_hist_year['date'], 
                                                             f'constr_capacity_kw_AdjHist_refact{refactor}': loop_adj_hist_year}))


            # update layout + export
            fig.update_layout(
                title=f'Construction Capacity in Sample Over Time ({len(self.sett.bfs_numbers)} nBFS, {len(topo.keys())} nEGIDs)',
                xaxis_title='Time',
                yaxis_title='Constructed Capacity (kW)',
                legend_title='Legend',
                template='plotly_white',
            )
            fig.write_html(f'{self.sett.name_dir_export_path}/construction_capacity_over_time_{self.sett.name_dir_export}.html')
            # fig.show()

            # crop dfs to match one date range for csv export
            # trange_csv = 
            min_date_range_hist, max_date_range_hist = min( constrcapa_adjhist_year['date'] ), max( constrcapa_adjhist_year['date'] )
            max_date_range_epzb = max( epzb_capa_df['date'] )
            start_date_csv = min_date_range_hist
            end_date_csv   = min( max_date_range_hist, max_date_range_epzb) 

            constracapa_comparison_df = None
            for df in export_csv_columns_list:
                df = df.copy()
                df['date'] = pd.to_datetime(df['date'])
                df = df.loc[(df['date'] >= start_date_csv) & (df['date'] <= end_date_csv)]

                if constracapa_comparison_df is None:
                    constracapa_comparison_df = df
                else:
                    constracapa_comparison_df = constracapa_comparison_df.merge(df, on='date', how='inner')

            constracapa_comparison_df = constracapa_comparison_df.sort_values('date').reset_index(drop=True)
            constracapa_comparison_df.to_csv(f'{self.sett.name_dir_export_path}/constracapa_comparison.csv', index=False)


        # EXPORT COMPARISON CSV  ----------------------------------------------------------------------------



        # SELECTION + PRINTs to LOGFILE ----------------------------------------------------------------------------
        if self.sett.CSTRspec_capacity_type == 'hist_constr_capa_year':
            constrcapa = constrcapa_hist_year.copy()
        elif self.sett.CSTRspec_capacity_type == 'ep2050_zerobasis':
            constrcapa = constrcapa_epzb.copy()
        elif self.sett.CSTRspec_capacity_type == 'adj_hist_to_ep2050':
            constrcapa = constrcapa_adjhist_year.copy()
                
        

        checkpoint_to_logfile(f'constr_capacity month lookback, between :                {months_lookback[0]} to {months_lookback[-1]}', self.sett.log_name, 0)
        checkpoint_to_logfile(f'constr_capacity KW built in period (sum_TP_kW_lookback): {round(sum_TP_kW_lookback,2)} kW', self.sett.log_name, 0)
        print_to_logfile('\n', self.sett.log_name)
        checkpoint_to_logfile(f'constr_capacity: month prediction {trange_prediction[0]} to {trange_prediction[-1]}', self.sett.log_name, 0)
        checkpoint_to_logfile(f'sum_TP_kw_lookback {round(sum_TP_kW_lookback,3)} kW to distribute across trange_prediction', self.sett.log_name, 0)
        print_to_logfile('\n', self.sett.log_name)
        checkpoint_to_logfile(f'sum_TP_kW_lookback (T0: {round(sum_TP_kW_lookback,2)} kW) increase by {capacity_growth*100}% per year', self.sett.log_name, 0)


        # EXPORT ----------------------------------------------------------------------------
        constrcapa.to_parquet(f'{self.sett.name_dir_export_path}/constrcapa.parquet')
        constrcapa.to_csv(f'{self.sett.name_dir_export_path}/constrcapa.csv', index=False)

        return constrcapa



    def initial_copy_ext_construction_capacity_to_dir(self, ):
        constrcapa_path = os.path.join(self.sett.data_path, self.sett.CSTRspec_ext_constrcapa_path)
        constrcapa = pd.read_parquet(constrcapa_path)

        constrcapa.to_parquet(f'{self.sett.name_dir_export_path}/constrcapa.parquet')
        constrcapa.to_csv(f'{self.sett.name_dir_export_path}/constrcapa.csv', index=False)



    # ------------------------------------------------------------------------------------------------------
    # SANITY CHECK helpers
    # ------------------------------------------------------------------------------------------------------
    def sanity_check_summary_byEGID(self, subdir_path: str): 
        """
        Input: 
            - PVAllocScenario + PVAllocScenario_Settings dataclass containing all scenarios settings + methods
            - subdir_path: path to the subfolder where the data is stored
        Tasks:
            - import data from the topo_egid dict and npv_df by extracting data for a list of EGIDs defined in the settings
            - extract e.g. buidling specs, demand type, roof partitions, pvprod capacity per roof etc. 
            - append all these extractions to rows and those in turn to a data frame which exported as an xlsx file.
        Export to subdir_path:
            - sanity_check_summary_byEGID.xlsx ->

        """
        print_to_logfile('run function: sanity_check_summary_byEGID', self.sett.log_name)


        # import -----------------------------------------------------
        with open(f'{subdir_path}/topo_egid.json', 'r', encoding='utf-8') as f:
            topo = json.load(f)
        npv_df = pd.read_parquet(f'{subdir_path}/npv_df.parquet')
        path_pred_inst = glob.glob(f'{subdir_path}/pred_npv_inst_by_M/pred_inst_df_*.parquet')
        pred_inst_df = pd.read_parquet(f'{subdir_path}/pred_inst_df.parquet')

        # add a EGID of model algorithm to the list
        if pred_inst_df.shape[0]< self.sett.CHECKspec_n_EGIDs_of_alloc_algorithm:
            n_EGIDs_of_alloc_algorithm = list(np.random.choice(pred_inst_df['EGID'], pred_inst_df.shape[0], replace=False))
        else:
            n_EGIDs_of_alloc_algorithm = list(np.random.choice(pred_inst_df['EGID'], self.sett.CHECKspec_n_EGIDs_of_alloc_algorithm, replace=False))
        pred_inst_df.loc[pred_inst_df['EGID'].isin(n_EGIDs_of_alloc_algorithm), ['EGID','info_source']]
            
        # remove any duplicates + add to pvalloc_settings
        self.sett.CHECKspec_egid_list = list(set(self.sett.CHECKspec_egid_list))
        # pvalloc_settings['sanitycheck_summary_byEGID_specs']['egid_list'] = list(set(self.sett.CHECKspec_egid_list + n_EGIDs_of_alloc_algorithm ))
            

        # information extraction -----------------------------------------------------
        colnames = ['key', 'descr', 'partition_id', 'col1', 'col2', 'val', 'unit']
        def get_new_row():
            return {col: None for col in colnames}
            
        summary_toExcel_list = []
        # egid = sanitycheck_summary_byEGID_specs['egid_list'][3]
        for n_egid, egid in enumerate(self.sett.CHECKspec_egid_list):
            if egid not in topo.keys():
                single_val_list = [row_egid_not_in_topo, ] = [get_new_row(), ]
                row_egid_not_in_topo['key'], row_egid_not_in_topo['descr'], row_egid_not_in_topo['val'] = 'EGID', 'EGID NOT in topo', egid
            elif egid in topo.keys():
                # single values ----------
                if True:
                    single_val_list = [
                        row_egid, row_bfs, row_gklas, row_node, row_demand_type, 
                        row_pvinst_info, row_pvinst_BeginOp, row_pvinst_TotalPower,
                        row_elecpri, row_pvtarif, 
                        row_interest_rate, row_capa_years_maturity, row_selfconsumption, row_pvprod_method, 
                        row_panel_efficiency, row_inverter_efficiency, row_kWpeak_per_m2, row_share_roof_area, 
                        empty_row ] = [get_new_row(), get_new_row(), get_new_row(), get_new_row(),
                                    get_new_row(), get_new_row(), get_new_row(), get_new_row(), 
                                    get_new_row(), get_new_row(), get_new_row(), get_new_row(), 
                                    get_new_row(), get_new_row(), get_new_row(), get_new_row(),
                                    get_new_row(), get_new_row(), get_new_row(),  ]
                        
                    # row_egid, row_bfs, row_gklas, row_node, row_demand_type = get_new_row(), get_new_row(), get_new_row(), get_new_row(), get_new_row()
                    row_egid['key'], row_egid['descr'], row_egid['val'] = 'EGID', 'house identifier ID', egid
                    row_bfs['key'], row_bfs['descr'], row_bfs['val'] = 'BFS', 'municipality identifier ID', topo.get(egid).get('gwr_info').get('bfs')
                    row_gklas['key'], row_gklas['descr'], row_gklas['val'] = 'GKLAS', 'building type classification', topo.get(egid).get('gwr_info').get('gklas')
                    row_node['key'], row_node['descr'], row_node['val'] = 'node', 'grid node identifier (artificial)', topo.get(egid).get('node')
                    row_demand_type['key'], row_demand_type['descr'], row_demand_type['val'] = 'demand_arch_typ', 'type of artifical demand profile (Netflex, maybe CKW later)', topo.get(egid).get('demand_arch_typ')

                    # row_pvinst_info, row_pvinst_BeginOp, row_pvinst_TotalPower = get_new_row(), get_new_row(), get_new_row() 
                    row_pvinst_info['key'], row_pvinst_info['descr'], row_pvinst_info['val'] = 'pv_inst > info_source', 'Origin behind pv inst on house (real data or model alloc)', topo.get(egid).get('pv_inst').get('info_source')
                    row_pvinst_BeginOp['key'], row_pvinst_BeginOp['descr'], row_pvinst_BeginOp['val'] = 'pv_inst > BeginOp', 'begin of operation', topo.get(egid).get('pv_inst').get('BeginOp')
                    row_pvinst_TotalPower['key'], row_pvinst_TotalPower['descr'], row_pvinst_TotalPower['val'], row_pvinst_TotalPower['unit'] = 'pv_inst > TotalPower', 'total power of PV installation', topo.get(egid).get('pv_inst').get('TotalPower'), 'kW'

                    # row_elecpri, row_pvtarif = get_new_row(), get_new_row() 
                    row_elecpri['key'], row_elecpri['descr'], row_elecpri['val'], row_elecpri['unit'], row_elecpri['col1'], row_elecpri['col2'] = 'elecpri', 'mean electricity price per BFS area', topo.get(egid).get('elecpri_Rp_kWh'), 'Rp/kWh', f"elecpri_info: {topo.get(egid).get('elecpri_info')}",f"year: {self.sett.TECspec_elecpri_year}"
                    row_pvtarif['key'], row_pvtarif['descr'], row_pvtarif['val'], row_pvtarif['unit'], row_pvtarif['col1'], row_pvtarif['col2'] = 'pvtarif', 'tariff for PV feedin to EWR',topo.get(egid).get('pvtarif_Rp_kWh'), 'Rp/kWh', f"EWRs: {topo.get(egid).get('EWR').get('name')}", f"year: {self.sett.TECspec_pvtarif_year}"
                    row_interest_rate['key'], row_interest_rate['descr'],row_interest_rate['val'] = 'interest_rate', 'generic interest rate used for dicsounting NPV calculation',              self.sett.TECspec_interest_rate
                    row_capa_years_maturity['key'], row_capa_years_maturity['descr'], row_capa_years_maturity['val'] = 'invst_maturity', 'number of capa_years that consider pv production for NPV calculation',    self.sett.TECspec_invst_maturity

                    # row_selfconsumption, row_interest_rate, row_capa_years_maturity, row_kWpeak_per_m2  = get_new_row(), get_new_row(), get_new_row(), get_new_row()
                    row_selfconsumption['key'], row_selfconsumption['descr'], row_selfconsumption['val'] = 'self_consumption_ifapplicable', 'amount of production that can be consumed by the house at any hour during the year',   self.sett.TECspec_self_consumption_ifapplicable
                    row_pvprod_method['key'], row_pvprod_method['descr'], row_pvprod_method['val'] = 'pvprod_calc_method', 'method used to calculate PV production',                                                                self.sett.TECspec_pvprod_calc_method
                    row_panel_efficiency['key'], row_panel_efficiency['descr'], row_panel_efficiency['val'] = 'panel_efficiency', 'transformation factor, how much solar energy can be transformed into electricity',               self.sett.TECspec_panel_efficiency
                    row_inverter_efficiency['key'], row_inverter_efficiency['descr'], row_inverter_efficiency['val'] = 'inverter_efficiency', 'transformation factor, how much DC can be transformed into AC',                      self.sett.TECspec_inverter_efficiency
                    row_kWpeak_per_m2['key'], row_kWpeak_per_m2['descr'], row_kWpeak_per_m2['val'] = 'kWpeak_per_m2', 'transformation factor, how much kWp can be put on a square meter',                                           self.sett.TECspec_kWpeak_per_m2
                    row_share_roof_area['key'], row_share_roof_area['descr'], row_share_roof_area['val'] = 'share_roof_area_available',  'share of roof area that can be effectively used for PV installation',                     self.sett.TECspec_share_roof_area_available

                # df_uid (roof partition) values ----------   
                no_pv_TF = not topo.get(egid).get('pv_inst').get('inst_TF') 
                if no_pv_TF:
                    npv_sub = npv_df.loc[npv_df['EGID'] == egid]
                    npv_val_list = [
                        row_demand_kW,
                        row_FLAECHE_m2_mean, row_FLAECHE_m2_sum, 
                        row_AUSRICHTUNG_mean, row_AUSRICHTUNG_sum,
                        row_NEIGUNG_mean, row_NEIGUNG_sum,
                        row_pvprod_kW_min, row_pvprod_kW_max, row_pvprod_kW_mean, row_pvprod_kW_std, 
                        row_stromertrag_kWh_min, row_stromertrag_kWh_max, row_stromertrag_kWh_mean, row_stromertrag_kWh_std,
                        row_netfeedin_kW_min, row_netfeedin_kW_max, row_netfeedin_kW_mean, row_netfeedin_kW_std,
                        row_econ_inc_chf_min, row_econ_inc_chf_max, row_econ_inc_chf_mean, row_econ_inc_chf_std,
                        row_estim_pvinstcost_chf_min, row_estim_pvinstcost_chf_max, row_estim_pvinstcost_chf_mean, row_estim_pvinstcost_chf_std,
                        row_npv_chf_min, row_npv_chf_max, row_npv_chf_mean, row_npv_chf_std,
                        ] = [get_new_row(), 
                            get_new_row(), get_new_row(), 
                            get_new_row(), get_new_row(), 
                            get_new_row(), get_new_row(), 
                            get_new_row(), get_new_row(), get_new_row(), get_new_row(),
                            get_new_row(), get_new_row(), get_new_row(), get_new_row(),
                            get_new_row(), get_new_row(), get_new_row(), get_new_row(),
                            get_new_row(), get_new_row(), get_new_row(), get_new_row(),
                            get_new_row(), get_new_row(), get_new_row(), get_new_row(),
                            get_new_row(), get_new_row(), get_new_row(), get_new_row(),]
                        
                    row_demand_kW['key'], row_demand_kW['descr'], row_demand_kW['val'], row_demand_kW['unit'] = 'demand_kW_min', 'total demand of house over 1 year', npv_sub['demand_kW'].mean(), 'kWh'

                    row_pvprod_kW_mean['key'], row_pvprod_kW_mean['descr'], row_pvprod_kW_mean['val'], row_pvprod_kW_mean['unit'] = 'pvprod_kW_mean', 'mean of possible production within all partition combinations',  npv_sub['pvprod_kW'].mean(), 'kWh'
                    row_pvprod_kW_std['key'], row_pvprod_kW_std['descr'], row_pvprod_kW_std['val'], row_pvprod_kW_std['unit'] = 'pvprod_kW_std', 'std of possible production within all partition combinations',  npv_sub['pvprod_kW'].std(), 'kWh'
                    row_pvprod_kW_min['key'], row_pvprod_kW_min['descr'], row_pvprod_kW_min['val'], row_pvprod_kW_min['unit'] = 'pvprod_kW_min', 'min of possible production within all partition combinations',  npv_sub['pvprod_kW'].min(), 'kWh'
                    row_pvprod_kW_max['key'], row_pvprod_kW_max['descr'], row_pvprod_kW_max['val'], row_pvprod_kW_max['unit'] = 'pvprod_kW_max', 'max of possible production within all partition combinations',  npv_sub['pvprod_kW'].max(), 'kWh'

                    row_FLAECHE_m2_mean['key'], row_FLAECHE_m2_mean['descr'], row_FLAECHE_m2_mean['val'], row_FLAECHE_m2_mean['unit'] = 'FLAECHE_m2_mean', 'mean of possible roof area within all partition combinations',  npv_sub['FLAECHE'].mean(), 'm2'  
                    row_FLAECHE_m2_sum['key'], row_FLAECHE_m2_sum['descr'], row_FLAECHE_m2_sum['val'], row_FLAECHE_m2_sum['unit'] = 'FLAECHE_m2_sum', 'sum of possible roof area within all partition combinations',  npv_sub['FLAECHE'].sum(), 'm2'
                    row_AUSRICHTUNG_mean['key'], row_AUSRICHTUNG_mean['descr'], row_AUSRICHTUNG_mean['val'], row_AUSRICHTUNG_mean['unit'] = 'AUSRICHTUNG_mean', 'mean of possible orientation within all partition combinations',  npv_sub['AUSRICHTUNG'].mean(), 'degree'
                    row_AUSRICHTUNG_sum['key'], row_AUSRICHTUNG_sum['descr'], row_AUSRICHTUNG_sum['val'], row_AUSRICHTUNG_sum['unit'] = 'AUSRICHTUNG_sum', 'sum of possible orientation within all partition combinations',  npv_sub['AUSRICHTUNG'].sum(), 'degree'
                    row_NEIGUNG_mean['key'], row_NEIGUNG_mean['descr'], row_NEIGUNG_mean['val'], row_NEIGUNG_mean['unit'] = 'NEIGUNG_mean', 'mean of possible tilt within all partition combinations',  npv_sub['NEIGUNG'].mean(), 'degree'
                    row_NEIGUNG_sum['key'], row_NEIGUNG_sum['descr'], row_NEIGUNG_sum['val'], row_NEIGUNG_sum['unit'] = 'NEIGUNG_sum', 'sum of possible tilt within all partition combinations',  npv_sub['NEIGUNG'].sum(), 'degree'

                    # row_stromertrag_kWh_mean['key'], row_stromertrag_kWh_mean['descr'], row_stromertrag_kWh_mean['val'], row_stromertrag_kWh_mean['unit'] = 'STROMERTRAG_mean', 'mean of possible STROMERTRAG (solkat data)',  npv_sub['stromertrag_kWh'].mean(), 'kWh/year'
                    # row_stromertrag_kWh_std['key'], row_stromertrag_kWh_std['descr'], row_stromertrag_kWh_std['val'], row_stromertrag_kWh_std['unit'] = 'STROMERTRAG_std', 'std of possible STROMERTRAG (solkat data)',  npv_sub['stromertrag_kWh'].std(), 'kWh/year'
                    # row_stromertrag_kWh_min['key'], row_stromertrag_kWh_min['descr'], row_stromertrag_kWh_min['val'], row_stromertrag_kWh_min['unit'] = 'STROMERTRAG_min', 'min of possible STROMERTRAG (solkat data)',  npv_sub['stromertrag_kWh'].min(), 'kWh/year'
                    # row_stromertrag_kWh_max['key'], row_stromertrag_kWh_max['descr'], row_stromertrag_kWh_max['val'], row_stromertrag_kWh_max['unit'] = 'STROMERTRAG_max', 'max of possible STROMERTRAG (solkat data)',  npv_sub['stromertrag_kWh'].max(), 'kWh/year'
                    #  STROMERTRAG IS NOT IN NPV_DFL

                    row_netfeedin_kW_mean['key'], row_netfeedin_kW_mean['descr'], row_netfeedin_kW_mean['val'], row_netfeedin_kW_mean['unit'] = 'netfeedin_kW_mean', 'mean of possible feedin within all partition combinations',  npv_sub['netfeedin_kW'].mean(), 'kWh'
                    row_netfeedin_kW_std['key'], row_netfeedin_kW_std['descr'], row_netfeedin_kW_std['val'], row_netfeedin_kW_std['unit'] = 'netfeedin_kW_std', 'std of possible feedin within all partition combinations',  npv_sub['netfeedin_kW'].std(), 'kWh'
                    row_netfeedin_kW_min['key'], row_netfeedin_kW_min['descr'], row_netfeedin_kW_min['val'], row_netfeedin_kW_min['unit'] = 'netfeedin_kW_min', 'min of possible feedin within all partition combinations',  npv_sub['netfeedin_kW'].min(), 'kWh'
                    row_netfeedin_kW_max['key'], row_netfeedin_kW_max['descr'], row_netfeedin_kW_max['val'], row_netfeedin_kW_max['unit'] = 'netfeedin_kW_max', 'max of possible feedin within all partition combinations',  npv_sub['netfeedin_kW'].max(), 'kWh'
                        
                    row_econ_inc_chf_mean['key'], row_econ_inc_chf_mean['descr'], row_econ_inc_chf_mean['val'], row_econ_inc_chf_mean['unit'] = 'econ_inc_chf_mean', 'mean of possible economic income within all partition combinations',  npv_sub['econ_inc_chf'].mean(), 'CHF'
                    row_econ_inc_chf_std['key'], row_econ_inc_chf_std['descr'], row_econ_inc_chf_std['val'], row_econ_inc_chf_std['unit'] = 'econ_inc_chf_std', 'std of possible economic income within all partition combinations',  npv_sub['econ_inc_chf'].std(), 'CHF'
                    row_econ_inc_chf_min['key'], row_econ_inc_chf_min['descr'], row_econ_inc_chf_min['val'], row_econ_inc_chf_min['unit'] = 'econ_inc_chf_min', 'min of possible economic income within all partition combinations',  npv_sub['econ_inc_chf'].min(), 'CHF'
                    row_econ_inc_chf_max['key'], row_econ_inc_chf_max['descr'], row_econ_inc_chf_max['val'], row_econ_inc_chf_max['unit'] = 'econ_inc_chf_max', 'max of possible economic income within all partition combinations',  npv_sub['econ_inc_chf'].max(), 'CHF'

                    row_estim_pvinstcost_chf_mean['key'], row_estim_pvinstcost_chf_mean['descr'], row_estim_pvinstcost_chf_mean['val'], row_estim_pvinstcost_chf_mean['unit'] = 'estim_pvinstcost_chf_mean', 'mean of possible installation costs within all partition combinations',  npv_sub['estim_pvinstcost_chf'].mean(), 'CHF'
                    row_estim_pvinstcost_chf_std['key'], row_estim_pvinstcost_chf_std['descr'], row_estim_pvinstcost_chf_std['val'], row_estim_pvinstcost_chf_std['unit'] = 'estim_pvinstcost_chf_std', 'std of possible installation costs within all partition combinations',  npv_sub['estim_pvinstcost_chf'].std(), 'CHF'
                    row_estim_pvinstcost_chf_min['key'], row_estim_pvinstcost_chf_min['descr'], row_estim_pvinstcost_chf_min['val'], row_estim_pvinstcost_chf_min['unit'] = 'estim_pvinstcost_chf_min', 'min of possible installation costs within all partition combinations',  npv_sub['estim_pvinstcost_chf'].min(), 'CHF'
                    row_estim_pvinstcost_chf_max['key'], row_estim_pvinstcost_chf_max['descr'], row_estim_pvinstcost_chf_max['val'], row_estim_pvinstcost_chf_max['unit'] = 'estim_pvinstcost_chf_max', 'max of possible installation costs within all partition combinations',  npv_sub['estim_pvinstcost_chf'].max(), 'CHF'

                    row_npv_chf_mean['key'], row_npv_chf_mean['descr'], row_npv_chf_mean['val'], row_npv_chf_mean['unit'] = 'npv_chf_mean', 'mean of possible NPV within all partition combinations',  npv_sub['NPV_uid'].mean(), 'CHF'
                    row_npv_chf_std['key'], row_npv_chf_std['descr'], row_npv_chf_std['val'], row_npv_chf_std['unit'] = 'npv_chf_std', 'std of possible NPV within all partition combinations',  npv_sub['NPV_uid'].std(), 'CHF'
                    row_npv_chf_min['key'], row_npv_chf_min['descr'], row_npv_chf_min['val'], row_npv_chf_min['unit'] = 'npv_chf_min', 'min of possible NPV within all partition combinations',  npv_sub['NPV_uid'].min(), 'CHF'
                    row_npv_chf_max['key'], row_npv_chf_max['descr'], row_npv_chf_max['val'], row_npv_chf_max['unit'] = 'npv_chf_max', 'max of possible NPV within all partition combinations',  npv_sub['NPV_uid'].max(), 'CHF'

                alloc_algo_pv_TF = topo.get(egid).get('pv_inst').get('inst_TF') and topo.get(egid).get('pv_inst').get('info_source') == 'alloc_algorithm'
                if alloc_algo_pv_TF:
                    pred_inst_sub = pred_inst_df.loc[pred_inst_df['EGID'] == egid]
                    npv_val_list = [
                        row_demand_kW,
                        row_df_uid,
                        row_pvprod_kW,
                        row_FLAECHE, 
                        row_FLAECH_angletilt,
                        row_AUSRICHTUNG,
                        row_NEIGUNG, 
                        row_STROMERTRAG_kWh,
                        row_netfeedin_kW, 
                        row_econ_inc_chf, 
                        row_estim_pvinstcost_chf, 
                        row_npv_chf
                        ] = [get_new_row(),get_new_row(),get_new_row(),get_new_row(),get_new_row(),get_new_row(),get_new_row(),get_new_row(),get_new_row(), get_new_row(), get_new_row(),get_new_row(), ] 
                        
                    row_demand_kW['key'], row_demand_kW['descr'], row_demand_kW['val'], row_demand_kW['unit'] = 'demand_kW_min', 'total demand of house over 1 year', pred_inst_sub['demand_kW'].values[0], 'kWh'
                    row_df_uid['key'], row_df_uid['descr'], row_df_uid['val'], row_df_uid['unit'] = 'df_uid', 'roof partition identifier ID', pred_inst_sub['df_uid_combo'].values[0], 'ID'

                    row_pvprod_kW['key'], row_pvprod_kW['descr'], row_pvprod_kW['val'], row_pvprod_kW['unit'] = 'pvprod_kW', 'total production of house over 1 year', pred_inst_sub['pvprod_kW'].values[0], 'kWh'
                    row_FLAECHE['key'], row_FLAECHE['descr'], row_FLAECHE['val'], row_FLAECHE['unit'] = 'FLAECHE_m2', 'total roof area of house', pred_inst_sub['FLAECHE'].values[0], 'm2'
                    row_FLAECH_angletilt['key'], row_FLAECH_angletilt['descr'], row_FLAECH_angletilt['val'], row_FLAECH_angletilt['unit'] = 'FLAECH_angletilt', 'total roof area of house with angle tilt', pred_inst_sub['FLAECH_angletilt'].values[0], 'm2'
                    row_AUSRICHTUNG['key'], row_AUSRICHTUNG['descr'], row_AUSRICHTUNG['val'], row_AUSRICHTUNG['unit'] = 'AUSRICHTUNG', 'total orientation of house', pred_inst_sub['AUSRICHTUNG'].values[0], 'degree'
                    row_NEIGUNG['key'], row_NEIGUNG['descr'], row_NEIGUNG['val'], row_NEIGUNG['unit'] = 'NEIGUNG', 'total tilt of house', pred_inst_sub['NEIGUNG'].values[0], 'degree'
                    row_STROMERTRAG_kWh['key'], row_STROMERTRAG_kWh['descr'], row_STROMERTRAG_kWh['val'], row_STROMERTRAG_kWh['unit'] = 'STROMERTRAG_kWh', 'total STROMERTRAG of house over 1 year', pred_inst_sub['STROMERTRAG'].values[0], 'kWh/year'
                    #  STROMERTRAG IS NOT IN NPV_DFL
                    row_netfeedin_kW['key'], row_netfeedin_kW['descr'], row_netfeedin_kW['val'], row_netfeedin_kW['unit'] = 'netfeedin_kW', 'total feedin of house over 1 year', pred_inst_sub['netfeedin_kW'].values[0], 'kWh'
                    row_econ_inc_chf['key'], row_econ_inc_chf['descr'], row_econ_inc_chf['val'], row_econ_inc_chf['unit'] = 'econ_inc_chf', 'economic income of house over 1 year', pred_inst_sub['econ_inc_chf'].values[0], 'CHF'
                    row_estim_pvinstcost_chf['key'], row_estim_pvinstcost_chf['descr'], row_estim_pvinstcost_chf['val'], row_estim_pvinstcost_chf['unit'] = 'estim_pvinstcost_chf', 'estimated installation costs of house over 1 year', pred_inst_sub['estim_pvinstcost_chf'].values[0], 'CHF'
                    row_npv_chf['key'], row_npv_chf['descr'], row_npv_chf['val'], row_npv_chf['unit'] = 'npv_chf', 'net present value of house over 1 year', pred_inst_sub['NPV_uid'].values[0], 'CHF'
                    
                
            # attache all rows to summary_df ----------
            summary_rows = []

            for row in single_val_list:
                summary_rows.append(row)
            if egid in topo.keys():
                if no_pv_TF or alloc_algo_pv_TF:
                    for row in npv_val_list:
                        summary_rows.append(row)

            egid_summary_df = pd.DataFrame(summary_rows)
            egid_summary_df.to_csv(f'{subdir_path}/summary_{egid}.csv')
            summary_toExcel_list.append(egid_summary_df)
        
        try:
            with pd.ExcelWriter(f'{subdir_path}/summary_all.xlsx') as writer:
                for i, df in enumerate(summary_toExcel_list):
                    df.to_excel(writer, sheet_name=f'EGID_{i}', index=False)
            checkpoint_to_logfile(f'exported summary for {len(self.sett.CHECKspec_egid_list)} EGIDs to excel', self.sett.log_name)

        except Exception as e:
            checkpoint_to_logfile(f'Failed to export summary_all.xlsx. Error: {e}', self.sett.log_name)




    def sanity_create_gdf_export_of_topo(self,): 
        """
        Input:
            - PVAllocScenario + PVAllocScenario_Settings dataclass containing all scenarios settings + methods
        Tasks:
            - transform topo_egid to a data frame
            - import all the geo data and filter by the observations in topo_egid (EGIDs, DF_UIDs)
            - export filtered gdfs to local dir
        Output to pvalloc dir: 
            - solkat_gdf_in_topo, gwr_gdf_in_topo, pv_gdf_in_topo, 
            - solkat_gdf_notin_topo, gwr_gdf_notin_topo, pv_gdf_notin_topo, 
            - topo_gdf, single_part_houses_w_tilt, dsonodes_withegids_gdf, gm_gdf,             """
        print_to_logfile('run function: create_gdf_export_of_topology', self.sett.log_name)


        # create topo_df -----------------------------------------------------
        with open(f'{self.sett.name_dir_export_path}/topo_egid.json', 'r', encoding='utf-8') as f:
            topo = json.load(f)
        egid_list, gklas_list, inst_tf_list, inst_info_list, inst_id_list, beginop_list, power_list = [], [], [], [], [], [], []
        topo_df_uid_list = []    
        for k,v in topo.items():
            egid_list.append(k)
            gklas_list.append(v.get('gwr_info').get('gklas'))
            inst_tf_list.append(v.get('pv_inst').get('inst_TF'))
            inst_info_list.append(v.get('pv_inst').get('inst_info'))
            inst_id_list.append(v.get('pv_inst').get('xtf_id'))
            beginop_list.append(v.get('pv_inst').get('BeginOp'))
            power_list.append(v.get('pv_inst').get('TotalPower'))

            for k_sub, v_sub in v.get('solkat_partitions').items():
                topo_df_uid_list.append(k_sub)


        topo_df = pd.DataFrame({'EGID': egid_list,'gklas': gklas_list,
                                'inst_tf': inst_tf_list,'inst_info': inst_info_list,'inst_id': inst_id_list,'beginop': beginop_list,'power': power_list,
        })
        # topo_df['power'] = topo_df['power'].replace('', 0).infer_objects(copy=False).astype(float)
        # topo_df['power'] = topo_df['power'].replace('', 0).astype(object)
        # topo_df['power'] = pd.to_numeric(topo_df['power'], errors='coerce').fillna(0)
        topo_df['power'] = pd.to_numeric(topo_df['power'].replace('', '0'), errors='coerce').fillna(0)


        # import geo data -----------------------------------------------------
        solkat_gdf = gpd.read_file(f'{self.sett.name_dir_import_path}/solkat_gdf.geojson')
        gwr_gdf = gpd.read_file(f'{self.sett.name_dir_import_path}/gwr_gdf.geojson')
        pv_gdf = gpd.read_file(f'{self.sett.name_dir_import_path}/pv_gdf.geojson')

        Map_egid_dsonode = pd.read_parquet(f'{self.sett.name_dir_import_path}/Map_egid_dsonode.parquet')
        gwr_bsblso_gdf = gpd.read_file(f'{self.sett.data_path}/input_split_data_geometry/gwr_bsblso_gdf.geojson')
        gm_shp_gdf = gpd.read_file(f'{self.sett.name_dir_import_path}/gm_shp_gdf.geojson')


        # transformations
        gm_shp_gdf['BFS_NUMMER'] = gm_shp_gdf['BFS_NUMMER'].astype(int)
        gm_gdf = gm_shp_gdf.loc[gm_shp_gdf['BFS_NUMMER'].isin(self.sett.bfs_numbers)]


        pv_gdf['xtf_id'] = pv_gdf['xtf_id'].astype(int).replace(np.nan, "").astype(str)
        solkat_gdf['DF_UID'] = solkat_gdf['DF_UID'].astype(int).replace(np.nan, "").astype(str)
        solkat_gdf.rename(columns={'DF_UID': 'df_uid'}, inplace=True)

        # DSO whole gridnet
        dsonodes_withegids_gdf = Map_egid_dsonode.merge(gwr_bsblso_gdf, on='EGID', how='left')
        dsonodes_withegids_gdf = gpd.GeoDataFrame(dsonodes_withegids_gdf, crs='EPSG:2056', geometry='geometry')


        # subset gwr + pv -----------------------------------------------------
        solkat_gdf_in_topo = copy.deepcopy(solkat_gdf.loc[solkat_gdf['df_uid'].isin(topo_df_uid_list)])
        gwr_gdf_in_topo = copy.deepcopy(gwr_gdf.loc[gwr_gdf['EGID'].isin(topo_df['EGID'].unique())])
        pv_gdf_in_topo = copy.deepcopy(pv_gdf.loc[pv_gdf['xtf_id'].isin(topo_df['inst_id'].unique())])
        for column in pv_gdf_in_topo.columns:
            if pd.api.types.is_datetime64_any_dtype(pv_gdf_in_topo[column]):
                pv_gdf_in_topo[column] = pv_gdf_in_topo[column].astype(str)  # Conv

        solkat_gdf_notin_topo = copy.deepcopy(solkat_gdf.loc[~solkat_gdf['df_uid'].isin(topo_df_uid_list)])
        gwr_gdf_notin_topo = copy.deepcopy(gwr_gdf.loc[~gwr_gdf['EGID'].isin(topo_df['EGID'].unique())])
        pv_gdf_notin_topo = copy.deepcopy(pv_gdf.loc[~pv_gdf['xtf_id'].isin(topo_df['inst_id'].unique())])
        for column in pv_gdf_notin_topo.columns:
            if pd.api.types.is_datetime64_any_dtype(pv_gdf_notin_topo[column]):
                pv_gdf_notin_topo[column] = pv_gdf_notin_topo[column].astype(str)  # Conv
            

        topo_gdf = topo_df.merge(gwr_gdf[['EGID', 'geometry']], on='EGID', how='left')
        topo_gdf = gpd.GeoDataFrame(topo_gdf, crs='EPSG:2056', geometry='geometry')


        solkat_in_grid = solkat_gdf.loc[solkat_gdf['EGID'].isin(Map_egid_dsonode['EGID'].unique())]
        solkat_in_grid = solkat_in_grid.loc[solkat_in_grid['BFS_NUMMER'].isin(self.sett.bfs_numbers)]
        single_partition_houses = copy.deepcopy(solkat_in_grid[solkat_in_grid['EGID'].map(solkat_in_grid['EGID'].value_counts()) == 1])
        single_part_houses_w_tilt = copy.deepcopy(single_partition_houses.loc[single_partition_houses['NEIGUNG'] > 0])
        print_to_logfile('\n\nSINGLE PARTITION HOUSES WITH TILT for debugging:', self.sett.log_name)
        checkpoint_to_logfile(f'First 10 EGID rows: {single_part_houses_w_tilt["EGID"][0:10]}', self.sett.log_name) 


        # EXPORT to geojson -----------------------------------------------------
        if not os.path.exists(f'{self.sett.name_dir_export_path}/topo_spatial_data'):
            os.makedirs(f'{self.sett.name_dir_export_path}/topo_spatial_data', exist_ok=True)

        shp_to_export=[
            (solkat_gdf_in_topo, 'solkat_gdf_in_topo.geojson'),
            (gwr_gdf_in_topo, 'gwr_gdf_in_topo.geojson'),
            (pv_gdf_in_topo, 'pv_gdf_in_topo.geojson'),
                
            (solkat_gdf_notin_topo, 'solkat_gdf_notin_topo.geojson'),
            (gwr_gdf_notin_topo, 'gwr_gdf_notin_topo.geojson'),
            (pv_gdf_notin_topo, 'pv_gdf_notin_topo.geojson'),
                        
            (topo_gdf, 'topo_gdf.geojson'),
            (single_part_houses_w_tilt, 'single_part_houses_w_tilt.geojson'),
                
            (dsonodes_withegids_gdf, 'dsonodes_withegids_gdf.geojson'),
            (gm_gdf, 'gm_gdf.geojson')
        ]

        for gdf, file_name in shp_to_export:
            path_file = f'{self.sett.name_dir_export_path}/topo_spatial_data/{file_name}'
            try:
                with open (path_file, 'w') as f:
                    f.write(gdf.to_json())

            except Exception as e:
                print(f"Failed to export {path_file}. Error: {e}")        


        # subset to > max n partitions -----------------------------------------------------
        max_partitions = self.sett.GWRspec_solkat_max_n_partitions
        topo_above_npart_gdf = copy.deepcopy(topo_gdf)
        counts = topo_above_npart_gdf['EGID'].value_counts()
        topo_above_npart_gdf['EGID_count'] = topo_above_npart_gdf['EGID'].map(counts)
        topo_above_npart_gdf = topo_above_npart_gdf[topo_above_npart_gdf['EGID_count'] > max_partitions]

        solkat_above_npart_gdf = copy.deepcopy(solkat_gdf_in_topo)
        solkat_gdf_in_topo[solkat_gdf_in_topo['df_uid'].isin(topo_df_uid_list)].copy()

        # export to shp -----------------------------------------------------
        gdf_to_export2 = [
                        (topo_above_npart_gdf, f'{self.sett.name_dir_export_path}/topo_spatial_data/topo_above_{max_partitions}_npart_gdf.geojson'),
                        (solkat_above_npart_gdf, f'{self.sett.name_dir_export_path}/topo_spatial_data/solkat_above_{max_partitions}_npart_gdf.geojson')]
        for gdf, path in gdf_to_export2:
            try:
                with open (path, 'w') as f:
                    f.write(gdf.to_json())
            except Exception as e:
                print(f"Failed to export {path}. Error: {e}")

        print_to_logfile('Exported topo spatial data to shp files (with possible expections, see prints statments).', self.sett.log_name)




    def sanity_check_multiple_xtf_ids_per_EGID(self, ):
        """ 
        Input: 
            - PVAllocScenario + PVAllocScenario_Settings dataclass containing all scenarios settings + methods
        Tasks:
            - Import the CHECK_egid_with_problems.json file created previously (during the creation of the topology)
            - filter through the rows and extract the issue occurences with "multiple xtf_ids"
            - filter building (gwr with EGIDs) and pv installation (pv with xtf_ids) gdfs for these occurences and export them as spatial data
        Output: 
            - gwr_gdf_multiple_xtf_id.geojson; pv_gdf_multiple_xtf_id.geojson

        """

        print_to_logfile('run function: check_multiple_xtf_ids_per_EGID', self.sett.log_name)

        # import -----------------------------------------------------
        gwr_gdf = gpd.read_file(f'{self.sett.name_dir_import_path}/gwr_gdf.geojson')
        pv_gdf = gpd.read_file(f'{self.sett.name_dir_import_path}/pv_gdf.geojson')
        Map_egid_pv = pd.read_parquet(f'{self.sett.name_dir_import_path}/Map_egid_pv.parquet')

        with open(f'{self.sett.name_dir_export_path}/CHECK_egid_with_problems.json', 'r', encoding='utf-8') as f:
            check_egid = json.load(f)
        egid_list, issue_list= [], []
        for k,v in check_egid.items():
            egid_list.append(k)
            issue_list.append(v)
        check_df = pd.DataFrame({'EGID': egid_list,'issue': issue_list})

        check_df = check_df.loc[check_df['issue'] == 'multiple xtf_ids']

        # Map egid to xtf_id 
        multip_xtf_list = []
        for i, row in Map_egid_pv.iterrows():
            if row['EGID'] in check_df['EGID'].unique():
                multip_xtf_list.append(row['xtf_id'])
            
        multip_xtf_list_unique = list(set(multip_xtf_list))
            

        gwr_gdf_multiple_xtf_id = gwr_gdf[gwr_gdf['EGID'].isin(check_df['EGID'].unique())].copy()
        pv_gdf_multiple_xtf_id = pv_gdf[pv_gdf['xtf_id'].isin(multip_xtf_list_unique)].copy()

        # export to shp -----------------------------------------------------
        if not os.path.exists(f'{self.sett.name_dir_export_path}/topo_spatial_data'):
            os.makedirs(f'{self.sett.name_dir_export_path}/topo_spatial_data')
            
        with open (f'{self.sett.name_dir_export_path}/topo_spatial_data/gwr_gdf_multiple_xtf_id.geojson', 'w') as f:
            f.write(gwr_gdf_multiple_xtf_id.to_json())

        pv_gdf_multiple_xtf_id['BeginningOfOperation'] = pv_gdf_multiple_xtf_id['BeginningOfOperation'].astype(str)
        with open (f'{self.sett.name_dir_export_path}/topo_spatial_data/pv_gdf_multiple_xtf_id.geojson', 'w') as f:
            f.write(pv_gdf_multiple_xtf_id.to_json())





    def sanity_check_cleanup_obsolete_data(self,):

        # remove topo_time_subdf in Sanity Check Allocation
        topo_time_paths_in_sanity_dir = glob.glob(f'{self.sett.sanity_check_path}/topo_subdf*.parquet')
        for path in topo_time_paths_in_sanity_dir:
            try: 
                os.remove(path)
            except Exception as e:
                print(f'Failed to remove obsolete topo subdf file in sanity dir: {path}. Error: {e}')


        # remove CSVs in  topo_time_subdf 
        topo_time_paths_csvs = glob.glob(f'{self.sett.name_dir_export_path}/topo_time_subdf/*topo_subdf*.csv')
        topo_time_paths_csvs = topo_time_paths_csvs[:-1]
        for path in topo_time_paths_csvs: 
            try: 
                os.remove(path)
            except Exception as e:
                print(f'Failed to remove obsolete topo subdf.CSV file in: {path}. Error: {e}')

