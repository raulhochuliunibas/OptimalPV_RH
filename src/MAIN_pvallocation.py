# python integrated packages
import sys
import os as os 
import glob
import shutil
import json
import copy
import itertools

# external packages
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import polars as pl
from  dataclasses import dataclass, field, asdict
from typing_extensions import List, Dict
from scipy.optimize import curve_fit

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, print_to_logfile, checkpoint_to_logfile, get_bfs_from_ktnr



@dataclass
class PVAllocScenario_Settings:
    # DEFAULT SETTINGS ---------------------------------------------------
    name_dir_export: str                        = 'pvalloc_BL_smallsample'   # name of the directory where the data is exported to (name to replace/ extend the name of the folder "preprep_data" in the end)
    name_dir_import: str                        = 'preprep_BL_22to23_extSolkatEGID'
    show_debug_prints: bool                     = True                    # F: certain print statements are omitted, T: includes print statements that help with debugging
    export_csvs: bool                           = False
    
    kt_numbers: List[int]                       = field(default_factory=list)  # list of cantons to be considered
    bfs_numbers: List[int]                      = field(default_factory=lambda: [
                                                    2767, 2771,                                               # BL mini with inst before 2006: Bottmingen, Oberwil
                                                    # 2767, 2771, 2765, 2764,                                 # BLsml with inst before 2008: Bottmingen, Oberwil, Binningen, Biel-Benken
                                                    # 2767, 2771, 2761, 2762, 2769, 2764, 2765, 2773,         # BLmed with inst with / before 2008: Bottmingen, Oberwil, Aesch, Allschwil, Münchenstein, Biel-Benken, Binningen, Reinach
                                                    # 2473, 2475, 2480,                                       # SOsml: Dornach, Hochwald, Seewen

                                                    # 2768, 2761, 2772, 2785,                                 # BLsml: Ettingen, Aesch, Pfeffingen, Duggingen; + Laufen for comparison with own PV installation
                                                    # 2763, 2773, 2775, 2764, 2471, 2481, 2476, 2786,2787,    # BLmed: Arlesheim, Reinach, Therwil, Biel-Benken, Bättwil, Witterswil, Hofstetten-Flüh, Grellingen
                                                    # 2618, 2621, 2883, 2622, 2616,                           # SOmed: Himmelried, Nunningen, Bretzwil, Zullwil, Fehre
                            ])
    T0_year_prediction: int                     = 2022                          # year for the prediction of the future construction capacity
    # T0_prediction: str                          = f'{T0_year_prediction}-01-01 00:00:00'         # start date for the prediction of the future construction capacity
    months_lookback: int                        = 12                           # number of months to look back for the prediction of the future construction capacity
    months_prediction: int                      = 12                         # number of months to predict the future construction capacity
    
    mcalgo_w_np_TF: bool                        = False
    recreate_topology: bool                     = True
    recalc_economics_topo_df: bool              = True
    sanitycheck_byEGID: bool                    = True
    create_gdf_export_of_topology: bool         = True
    test_faster_array_computation: bool         = False
    
    # PART I: settings for alloc_initialization --------------------
    GWRspec_solkat_max_n_partitions: int                = 10          # larger number of partitions make all combos of roof partitions practically impossible to calculate
    GWRspec_solkat_area_per_EGID_range: List[int]       = field(default_factory=lambda: [2, 600])  # for 100kWp inst, need 500m2 roof area => just above the threshold for residential subsidies KLEIV, below 2m2 too small to mount installations
    GWRspec_building_cols: List[str]                    = field(default_factory=lambda: [
                                                            'EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                                                            'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                                                            'GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1', 'GEBF', 'GAREA'
                                                        ])
    
    GWRspec_dwelling_cols: List[str]                    = field(default_factory=list)
    GWRspec_DEMAND_proxy: str                           = 'GAREA'
    GWRspec_GSTAT: List[str]                            = field(default_factory=lambda: ['1004'])
    GWRspec_GKLAS: List[str]                            = field(default_factory=lambda: ['1110', '1121'])
    GWRspec_GBAUJ_minmax: List[int]                     = field(default_factory=lambda: [1950, 2021])
    
    # weather_specs
    WEAspec_meteo_col_dir_radiation: str                = 'Basel Direct Shortwave Radiation'
    WEAspec_meteo_col_diff_radiation: str               = 'Basel Diffuse Shortwave Radiation'
    WEAspec_meteo_col_temperature: str                  = 'Basel Temperature [2 m elevation corrected]'
    WEAspec_weather_year: int                           = 2022
    WEAspec_radiation_to_pvprod_method: str             = 'dfuid_ind'
    WEAspec_rad_rel_loc_max_by: str                     = 'dfuid_specific'
    WEAspec_flat_direct_rad_factor: int                 = 1
    WEAspec_flat_diffuse_rad_factor: int                = 1

    # constr_capacity_specs
    CSTRspec_ann_capacity_growth: float                 = 0.05
    CSTRspec_constr_capa_overshoot_fact: int            = 1
    CSTRspec_month_constr_capa_tuples: List[tuple]      = field(default_factory=lambda: [
                                                            (1, 0.06), (2, 0.06), (3, 0.06), (4, 0.06),
                                                            (5, 0.08), (6, 0.08), (7, 0.08), (8, 0.08),
                                                            (9, 0.10), (10, 0.10), (11, 0.12), (12, 0.12)
                                                        ])
    
    # tech_economic_specs
    TECspec_self_consumption_ifapplicable: float            = 1
    TECspec_interest_rate: float                            = 0.01
    TECspec_pvtarif_year: int                               = 2022
    TECspec_pvtarif_col: List[str]                          = field(default_factory=lambda: ['energy1', 'eco1'])
    TECspec_pvprod_calc_method: str                         = 'method2.2'
    TECspec_panel_efficiency: float                         = 0.21
    TECspec_inverter_efficiency: int                        = 0.95
    TECspec_elecpri_year: int                               = 2022
    TECspec_elecpri_category: str                           = 'H4'
    TECspec_invst_maturity: int                             = 25
    TECspec_kWpeak_per_m2: float                            = 0.2
    TECspec_share_roof_area_available: float                = 1
    TECspec_max_distance_m_for_EGID_node_matching: float    = 0
    TECspec_kW_range_for_pvinst_cost_estim: List[int]       = field(default_factory=lambda: [0, 61])
    TECspec_estim_pvinst_cost_correctionfactor: float       = 1

    # panel_efficiency_specs
    PEFspec_variable_panel_efficiency_TF: bool              = True
    PEFspec_summer_months: List[int]                        = field(default_factory=lambda: [6, 7, 8, 9])
    PEFspec_hotsummer_hours: List[int]                      = field(default_factory=lambda: [11, 12, 13, 14, 15, 16, 17])
    PEFspec_hot_hours_discount: float                       = 0.1
    
    # sanitycheck_summary_byEGID_specs
    CHECKspec_egid_list: List[str]                          = field(default_factory=lambda: [
                                                                '391292', '390601', '2347595', '401781',  # single roof houses in Aesch, Ettingen
                                                                '391263', '245057295', '401753',  # houses with built pv in Aesch, Ettingen
                                                                '245054165', '245054166', '245054175', '245060521', # EGID selection of neighborhood within Aesch to analyse closer
                                                                '391253', '391255', '391257', '391258', '391262',
                                                                '391263', '391289', '391290', '391291', '391292',
                                                                '245057295', '245057294', '245011456', '391379', '391377'
                                                            ])
    CHECKspec_n_EGIDs_of_alloc_algorithm: int               = 20
    CHECKspec_n_iterations_before_sanitycheck: int          = 1

    # PART II: settings for MC algorithm --------------------
    MCspec_montecarlo_iterations: int                           = 1
    MCspec_fresh_initial_files: List[str]                       = field(default_factory=lambda: [
                                                                    'topo_egid.json', 'months_prediction.parquet', 'gridprem_ts.parquet', 
                                                                    'constrcapa.parquet', 'dsonodes_df.parquet'
                                                                ])
    MCspec_keep_files_month_iter_TF: bool                       = True
    MCspec_keep_files_month_iter_max: int                       = 9999999999
    MCspec_keep_files_month_iter_list: List[str]                = field(default_factory=lambda: [
                                                                    'topo_egid.json', 'npv_df.parquet', 'pred_inst_df.parquet', 'gridprem_ts.parquet'
                                                                ])

    # algorithm_specs
    ALGOspec_inst_selection_method: str                         = 'random'
    ALGOspec_rand_seed: bool                                    = None
    ALGOspec_while_inst_counter_max: int                        = 5000
    ALGOspec_topo_subdf_partitioner: int                        = 400
    ALGOspec_npv_update_groupby_cols_topo_aggdf: List[str]      = field(default_factory=lambda: [
                                                                    'EGID', 'df_uid', 'grid_node', 'bfs', 'gklas', 'demandtype', 'inst_TF', 'info_source',
                                                                    'pvid', 'pv_tarif_Rp_kWh', 'elecpri_Rp_kWh', 'FLAECHE', 'FLAECH_angletilt', 'AUSRICHTUNG', 
                                                                    'NEIGUNG', 'STROMERTRAG'
                                                                ])
    ALGOspec_npv_update_agg_cols_topo_aggdf: Dict[str, str]     = field(default_factory=lambda: {
                                                                    'pvprod_kW': 'sum', 'demand_kW': 'sum', 'selfconsum_kW': 'sum', 'netdemand_kW': 'sum',
                                                                    'netfeedin_kW': 'sum', 'econ_inc_chf': 'sum', 'econ_spend_chf': 'sum'
                                                                })
    ALGOspec_tweak_constr_capacity_fact: float                  = 1
    ALGOspec_tweak_npv_calc: float                              = 1
    ALGOspec_tweak_npv_excl_elec_demand: bool                   = True
    ALGOspec_tweak_gridnode_df_prod_demand_fact: float          = 1
    ALGOspec_constr_capa_overshoot_fact: float                  = 1

    # gridprem_adjustment_specs
    GRIDspec_tier_description: str                              = 'tier_level: (voltage_threshold, gridprem_Rp_kWh)'
    GRIDspec_power_factor: float                                = 1
    GRIDspec_perf_factor_1kVA_to_XkW: float                     = 0.8
    GRIDspec_colnames: List[str]                                = field(default_factory=lambda: ['tier_level', 'used_node_capa_rate', 'gridprem_Rp_kWh'])
    GRIDspec_tiers: Dict[int, List[float]]                      = field(default_factory=lambda: {
                                                                   1: [0.7, 1], 2: [0.8, 3], 3: [0.85, 5], 
                                                                   4: [0.9, 7], 5: [0.95, 15], 6: [1, 100]
                                                                })
    

    def __post_init__(self):
        # have post init for less error prone scen setting. define T0 year and reference remaining settings to that.
        self.T0_prediction: str                 = f'{self.T0_year_prediction}-01-01 00:00:00'  
        # GBAUJ_min, GBAUJ_max = 1920, self.T0_year_prediction-1
        # self.GWRspec_GBAUJ_minmax: List[int]    = field(default_factory=lambda: [GBAUJ_min, GBAUJ_max])
        self.GWRspec_GBAUJ_minmax[0] = 1920
        self.GWRspec_GBAUJ_minmax[1] = self.T0_year_prediction-1


class PVAllocScenario:
    def __init__(self, settings: PVAllocScenario_Settings): 
        self.sett = settings

        self.sett.wd_path = os.getcwd()
        self.sett.data_path                = os.path.join(self.sett.wd_path, 'data')
        self.sett.pvalloc_path             = os.path.join(self.sett.data_path, 'pvalloc', 'pvalloc_scen__temp_to_be_renamed')
        self.sett.name_dir_export_path     = os.path.join(self.sett.data_path, 'pvalloc', self.sett.name_dir_export)
        self.sett.name_dir_import_path     = os.path.join(self.sett.data_path, 'preprep', self.sett.name_dir_import)
        self.sett.sanity_check_path        = os.path.join(self.sett.name_dir_export_path, 'sanity_check_byEGID')




    # ------------------------------------------------------------------------------------------------------
    # INITIALIZATION of PV Allocatoin Scenario
    # ------------------------------------------------------------------------------------------------------
    def run_pvalloc_initalization(self):
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
        self.sett.log_name = os.path.join(self.sett.name_dir_export_path, 'pvalloc_Initial_log.txt')
        self.sett.summary_name = os.path.join(self.sett.name_dir_export_path, 'summary_data_selection_log.txt')
        
        self.sett.bfs_numbers: List[str] = get_bfs_from_ktnr(self.sett.kt_numbers, self.sett.data_path, self.sett.log_name) if self.sett.kt_numbers != [] else [str(bfs) for bfs in self.sett.bfs_numbers]
        self.sett.total_runtime_start = datetime.datetime.now()

        # create dir for export, rename old export dir not to overwrite
        if os.path.exists(self.sett.name_dir_export_path):
            n_same_names = len(glob.glob(f'{self.sett.name_dir_export_path}*'))
            os.rename(self.sett.name_dir_export_path, f'{self.sett.name_dir_export_path}_{n_same_names}_old_vers')
        os.makedirs(self.sett.name_dir_export_path, exist_ok=True)

        # export class instance settings to dir        
        self.export_pvalloc_scen_settings()


        # create log file
        chapter_to_logfile(f'start MAIN_pvalloc_INITIALIZATION for: {self.sett.name_dir_export}', self.sett.log_name, overwrite_file=True)
        subchapter_to_logfile('pvalloc_settings', self.sett.log_name)
        for k, v in vars(self).items():
            print_to_logfile(f'{k}: {v}', self.sett.log_name)

        # create summary file + Timing file
        chapter_to_logfile(f'OptimalPV Sample Summary of Building Topology, scen: {self.sett.name_dir_export}', self.sett.summary_name, overwrite_file=True)

        # create timing csv
        self.sett.timing_marks_csv_path = os.path.join(self.sett.name_dir_export_path, 'timing_marks.csv')
        self.mark_to_timing_csv


        # CREATE TOPOLOGY ---------------------------------------------------------------------------------------------
        subchapter_to_logfile('initialization: CREATE SMALLER AID DFs', self.sett.log_name)
        start_create_topo = datetime.datetime.now()
        self.mark_to_timing_csv('init', 'start_create_topo', start_create_topo, np.nan, '-'),

        self.initial_sml_HOY_weatheryear_df()
        self.initial_sml_get_gridnodes_DSO()
        self.initial_sml_iterpolate_instcost_function()

        if self.sett.recreate_topology:
            subchapter_to_logfile('initialization: IMPORT PREPREP DATA & CREATE (building) TOPOLOGY', self.sett.log_name)
            topo, df_list, df_names = self.initial_lrg_import_preprep_AND_create_topology()

            subchapter_to_logfile('initialization: IMPORT TS DATA', self.sett.log_name)
            ts_list, ts_names = self.initial_lrg_import_ts_data()

            subchapter_to_logfile('initialization: DEFINE CONSTRUCTION CAPACITY', self.sett.log_name)
            constrcapa, months_prediction, months_lookback = self.initial_lrg_define_construction_capacity(topo, df_list, df_names, ts_list, ts_names)

            end_create_topo = datetime.datetime.now()
            self.mark_to_timing_csv('init', 'end_create_topo', end_create_topo, end_create_topo - start_create_topo, '-') 
        

        # CALC ECONOMICS + TOPO_TIME SPECIFIC DFs ---------------------------------------------------------------------------------------------
        subchapter_to_logfile('prep: CALC ECONOMICS for TOPO_DF', self.sett.log_name)
        start_calc_economics = datetime.datetime.now()
        self.mark_to_timing_csv('init', 'start_calc_economics', start_calc_economics, np.nan, '-')
        
        # algo.calc_economics_in_topo_df(self, topo, df_list, df_names, ts_list, ts_names)
        self.algo_calc_economics_in_topo_df(topo, df_list, df_names, ts_list, ts_names)

        end_calc_economics = datetime.datetime.now()
        self.mark_to_timing_csv('init', 'end_calc_economics', end_calc_economics, end_calc_economics - start_calc_economics, '-')
        shutil.copy(f'{self.sett.name_dir_export_path}/topo_egid.json', f'{self.sett.name_dir_export_path}/topo_egid_before_alloc.json')



        # SANITY CHECK: CALC FEW ITERATION OF NPV AND FEEDIN for check ---------------------------------------------------------------
        subchapter_to_logfile('sanity_check: RUN FEW ITERATION for byCHECK', self.sett.log_name)
        start_sanity_check = datetime.datetime.now()
        self.mark_to_timing_csv('init', 'start_sanity_check', start_sanity_check, np.nan, '-')

        # make sanitycheck folder and move relevant initial files there (delete all old files, not distort results)
        os.makedirs(self.sett.sanity_check_path, exist_ok=False) 

        fresh_initial_files = [f'{self.sett.name_dir_export_path}/{file}' for file in ['topo_egid.json', 'gridprem_ts.parquet', 'dsonodes_df.parquet']]
        topo_time_paths = glob.glob(f'{self.sett.name_dir_export_path}/topo_time_subdf/*.parquet')
        for f in fresh_initial_files + topo_time_paths:
            shutil.copy(f, f'{self.sett.sanity_check_path}/')

        # ALLOCATION RUN ====================
        dfuid_installed_list = []
        pred_inst_df = pd.DataFrame()
        months_prediction_pq = pd.read_parquet(f'{self.sett.name_dir_export_path}/months_prediction.parquet')['date']
        months_prediction = [str(m) for m in months_prediction_pq]

        for i_m, m in enumerate(months_prediction[0:self.sett.CHECKspec_n_iterations_before_sanitycheck]):
            print_to_logfile(f'\n-- month {m} -- sanity check -- {self.sett.name_dir_export} --', self.sett.log_name)
            self.algo_update_gridprem(self.sett.sanity_check_path, i_m, m)
            self.algo_update_npv_df(self.sett.sanity_check_path, i_m, m)
            self.algo_select_AND_adjust_topology(self.sett.sanity_check_path, 
                                                 i_m, m)

        
        # sanity.sanity_check_summary_byEGID(self, self.sanity_check_path)
        self.sanity_check_summary_byEGID(self.sett.sanity_check_path )
        
        # EXPORT SPATIAL DATA ====================
        if self.sett.create_gdf_export_of_topology:
            subchapter_to_logfile('sanity_check: CREATE SPATIAL EXPORTS OF TOPOLOGY_DF', self.sett.log_name)
            # sanity.create_gdf_export_of_topology(self)  
            self.sanity_create_gdf_export_of_topo()

            subchapter_to_logfile('sanity_check: MULTIPLE INSTALLATIONS PER EGID', self.sett.log_name)
            # sanity.check_multiple_xtf_ids_per_EGID(self)
            self.sanity_check_multiple_xtf_ids_per_EGID()

        end_sanity_check = datetime.datetime.now()
        self.mark_to_timing_csv('init', 'end_sanity_check', end_sanity_check, end_sanity_check - start_sanity_check, '-')

        # END ---------------------------------------------------
        chapter_to_logfile(f'end start MAIN_pvalloc_INITIALIZATION\n Runtime (hh:mm:ss):{datetime.datetime.now() - self.sett.total_runtime_start}', self.sett.log_name)


    # ------------------------------------------------------------------------------------------------------
    # MONTE CARLO ALGORITHM of PV Allocatoin Scenario
    # ------------------------------------------------------------------------------------------------------
    def run_pvalloc_mcalgorithm(self,):
        """
        Input: 
            > PVAllocScenario class containing a data class (_Settings) which specifies all scenraio settings (e.g. also path to preprep data directory where all data for the
              executed computations is stored).
    
        Output:
            > within the scenario name defined in pvalloc_settings, the MAIN_pvalloc_MCalgorithm function 
            creates a new directory "MCx" folder directory containing each individual Monte Carlo iteration.

        Description:
            > The MAIN_pvalloc_MCalgorithm function calls the exact same functions as previously used in santiy check of
            pv allocation initializations' sanity check for direct comparison of debugging and testing. 
            > First the script updates the grid premium values for the current month, based on existing installtions and annual radiation. 
            > Then the script updates the NPV values for all houses not yet having a PV installation. 
            > Based on scenario settings, installations are selected from the NPV dataframe until the construction capacity for the given month 
            is reached (or the total capacity for the year; while loop exit criteria).
            
            > This process is repeated for as many Monte Carlo iterations as defined in the scenario settings.
            
        """
        # SETUP -----------------------------------------------------------------------------
        self.sett.log_name = os.path.join(self.sett.name_dir_export_path, 'pvalloc_MCalgo_log.txt')
        self.sett.timing_marks_csv_path = os.path.join(self.sett.name_dir_export_path, 'timing_marks.csv')
        total_runtime_start = datetime.datetime.now()

        # create log file
        chapter_to_logfile(f'start MAIN_pvalloc_MCalgorithm for : {self.sett.name_dir_export}', self.sett.log_name, overwrite_file=True)
        print_to_logfile('*model allocation specifications*:', self.sett.log_name)
        print_to_logfile(f'> n_bfs_municipalities: {len(self.sett.bfs_numbers)} \n> n_months_prediction: {self.sett.months_prediction} \n> n_montecarlo_iterations: {self.sett.MCspec_montecarlo_iterations}', self.sett.log_name)
        print_to_logfile(f'> pvalloc_settings, MCalloc_{self.sett.name_dir_export}', self.sett.log_name)
        for k, v in vars(self).items():
            print_to_logfile(f'{k}: {v}', self.sett.log_name)

        start_mc_algo = datetime.datetime.now()
        self.mark_to_timing_csv('MCalgo', 'START_MC_algo', start_mc_algo, np.nan, '-')



        # CREATE MC DIR + TRANSFER INITIAL DATA FILES ----------------------------------------------
        montecarlo_iterations = [*range(1, self.sett.MCspec_montecarlo_iterations+1, 1)]
        safety_counter_max = self.sett.ALGOspec_while_inst_counter_max
        
        # get all initial files to start a fresh MC iteration
        fresh_initial_paths = [f'{self.sett.name_dir_export_path}/{file}' for file in self.sett.MCspec_fresh_initial_files]
        topo_time_paths = glob.glob(f'{self.sett.name_dir_export_path}/topo_time_subdf/topo_subdf*.parquet')

        max_digits = len(str(max(montecarlo_iterations)))
        # mc_iter = montecarlo_iterations[0]
        # if True:    
        for mc_iter in montecarlo_iterations:
            start_mc_iter = datetime.datetime.now()
            subchapter_to_logfile(f'START MC{mc_iter:0{max_digits}} iteration', self.sett.log_name)
            self.mark_to_timing_csv('MCalgo', f'start_MC_iter_{mc_iter:0{max_digits}}', start_mc_iter, np.nan, '-')

            # copy all initial files to MC directory
            self.sett.mc_iter_path = os.path.join(self.sett.name_dir_export_path, f'zMC{mc_iter:0{max_digits}}')
            shutil.rmtree(self.sett.mc_iter_path) if os.path.exists(self.sett.mc_iter_path) else None
            os.makedirs(self.sett.mc_iter_path, exist_ok=False)
            for f in fresh_initial_paths + topo_time_paths:
                shutil.copy(f, self.sett.mc_iter_path)



            # ALLOCATION ALGORITHM -----------------------------------------------------------------------------    
            dfuid_installed_list = []
            pred_inst_df = pd.DataFrame()  
            months_prediction_df = pd.read_parquet(f'{self.sett.mc_iter_path}/months_prediction.parquet')
            months_prediction = months_prediction_df['date']
            constrcapa = pd.read_parquet(f'{self.sett.mc_iter_path}/constrcapa.parquet')

            for i_m, m in enumerate(months_prediction):
                print_to_logfile(f'\n-- month {m} -- iter MC{mc_iter:0{max_digits}} -- {self.sett.name_dir_export} --', self.sett.log_name)
                start_allocation_month = datetime.datetime.now()
                i_m = i_m + 1        

                # GRIDPREM + NPV_DF UPDATE ==========
                # if self.sett.mcalgo_w_np_TF:
                #     self.algo_update_gridprem_np(self.sett.mc_iter_path, i_m, m)
                # else:
                start_time_update_gridprem = datetime.datetime.now()
                print_to_logfile('- START update gridprem', self.sett.log_name)
                if not self.sett.test_faster_array_computation:
                    self.algo_update_gridprem(self.sett.mc_iter_path, i_m, m)
                elif self.sett.test_faster_array_computation:
                    self.algo_update_gridprem_POLARS(self.sett.mc_iter_path, i_m, m)
                end_time_update_gridprem = datetime.datetime.now()
                print_to_logfile(f'- END update gridprem: {end_time_update_gridprem - start_time_update_gridprem} (hh:mm:ss.s)', self.sett.log_name)
                self.mark_to_timing_csv('MCalgo', f'update_gridprem_{i_m:0{max_digits}}', end_time_update_gridprem - start_time_update_gridprem, end_time_update_gridprem, '-')  if i_m < 7 else None
                
                start_time_update_npv = datetime.datetime.now()
                print_to_logfile('- START update npv', self.sett.log_name)
                if not self.sett.test_faster_array_computation:
                    npv_df = self.algo_update_npv_df(self.sett.mc_iter_path, i_m, m)
                elif self.sett.test_faster_array_computation:
                    npv_df = self.algo_update_npv_df_POLARS(self.sett.mc_iter_path, i_m, m)
                end_time_update_npv = datetime.datetime.now()
                print_to_logfile(f'- END update npv: {end_time_update_npv - start_time_update_npv} (hh:mm:ss.s)', self.sett.log_name)
                self.mark_to_timing_csv('MCalgo', f'update_npv_{i_m:0{max_digits}}', end_time_update_npv - start_time_update_npv, end_time_update_npv, '-')  if i_m < 7 else None



                # init constr capa ==========
                constr_built_m = 0
                if m.year != (m-1).year:
                    constr_built_y = 0
                constr_capa_m = constrcapa.loc[constrcapa['date'] == m, 'constr_capacity_kw'].iloc[0]
                constr_capa_y = constrcapa.loc[constrcapa['year'].isin([m.year]), 'constr_capacity_kw'].sum()

                # INST PICK ==========
                start_time_installation_whileloop = datetime.datetime.now()
                print_to_logfile('- START inst while loop', self.sett.log_name)

                safety_counter = 0
                print_to_logfile('start inst pick while loop', self.sett.log_name)
                while( (constr_built_m <= constr_capa_m) & (constr_built_y <= constr_capa_y) & (safety_counter <= safety_counter_max) ):
                    
                    if npv_df.shape[0] == 0:
                        checkpoint_to_logfile(' npv_df is EMPTY, exit while loop', self.sett.log_name, 1, self.sett.self.sett.show_debug_prints)                    
                        safety_counter = safety_counter_max

                    if npv_df.shape[0] > 0: 
                        inst_power, npv_df = self.algo_select_AND_adjust_topology(self.sett.mc_iter_path,
                                                                                    i_m, m)

                    # Loop Exit + adjust constr_built capacity ----------
                    end_time_installation_whileloop = datetime.datetime.now()
                    print_to_logfile(f'- END inst while loop: {end_time_installation_whileloop - start_time_installation_whileloop} (hh:mm:ss.s)', self.sett.log_name)
                    self.mark_to_timing_csv('MCalgo', f'inst_whileloop_{i_m:0{max_digits}}', end_time_installation_whileloop - start_time_installation_whileloop, end_time_installation_whileloop, '-')  if i_m < 7 else None
                    
                    constr_built_m, constr_built_y, safety_counter = constr_built_m + inst_power, constr_built_y + inst_power, safety_counter + 1
                    overshoot_rate = self.sett.CSTRspec_constr_capa_overshoot_fact
                    constr_m_TF, constr_y_TF, safety_TF = constr_built_m > constr_capa_m*overshoot_rate, constr_built_y > constr_capa_y*overshoot_rate, safety_counter > safety_counter_max

                    if any([constr_m_TF, constr_y_TF, safety_TF]):
                        print_to_logfile('exit While Loop', self.sett.log_name)
                        if constr_m_TF:
                            checkpoint_to_logfile(f'exceeded constr_limit month (constr_m_TF:{constr_m_TF}), {round(constr_built_m,1)} of {round(constr_capa_m,1)} kW capacity built', self.sett.log_name, 1, self.sett.show_debug_prints)                    
                        if constr_y_TF:
                            checkpoint_to_logfile(f'exceeded constr_limit year (constr_y_TF:{constr_y_TF}), {round(constr_built_y,1)} of {round(constr_capa_y,1)} kW capacity built', self.sett.log_name, 1, self.sett.show_debug_prints)                    
                        if safety_TF:
                            checkpoint_to_logfile(f'exceeded safety counter (safety_TF:{safety_TF}), {safety_counter} rounds for safety counter max of: {safety_counter_max}', self.sett.log_name, 1, self.sett.show_debug_prints)                    

                        if constr_m_TF or constr_y_TF:    
                            checkpoint_to_logfile(f'{safety_counter} pv installations allocated', self.sett.log_name, 3, self.sett.show_debug_prints)                                        

                checkpoint_to_logfile(f'end month allocation, runtime: {datetime.datetime.now() - start_allocation_month} (hh:mm:ss.s)', self.sett.log_name, 1, self.sett.show_debug_prints)                    

            # CLEAN UP interim files of MC run ==========
            files_to_remove_paths =  glob.glob(f'{self.sett.mc_iter_path}/topo_subdf_*.parquet')
            for f in files_to_remove_paths:
                os.remove(f)

            mc_iter_time = datetime.datetime.now() - start_mc_iter
            subchapter_to_logfile(f'END MC{mc_iter:0{max_digits}}, runtime: {mc_iter_time} (hh:mm:ss.s)', self.sett.log_name)
            end_mc_iter = datetime.datetime.now()
            self.mark_to_timing_csv('MCalgo', f'end_MC_iter_{mc_iter:0{max_digits}}', end_mc_iter, end_mc_iter-start_mc_iter, '-')
            print_to_logfile('\n', self.sett.log_name)

        
        end_mc_algo = datetime.datetime.now()
        self.mark_to_timing_csv('MCalgo', 'END_MC_algo', start_mc_algo, end_mc_algo-start_mc_algo, '-')


        # END ---------------------------------------------------
        chapter_to_logfile(f'end MAIN_pvalloc_MCalgorithm\n Runtime (hh:mm:ss):{datetime.datetime.now() - total_runtime_start}', self.sett.log_name, overwrite_file=False)


    # ------------------------------------------------------------------------------------------------------
    # POSTPROCESSING of PV Allocatoin Scenario
    # ------------------------------------------------------------------------------------------------------
    def run_pvalloc_postprocess(self):
            """
            Input: 
                > PVAllocScenario class containing a data class (_Settings) which specifies all scenraio settings (e.g. also path to preprep data directory where all data 
                  for theexecuted computations is stored).

            """

            # SETUP -----------------------------------------------------------------------------
            self.log_name = os.path.join(self.sett.name_dir_export_path, 'pvalloc_postprocess_log.txt')
            self.sett.postprocess_path = os.path.join(self.sett.data_path, 'postprocess')
            os.makedirs(self.sett.postprocess_path)

            total_runtime_start = datetime.datetime.now()

            # create log file
            chapter_to_logfile(f'start MAIN_pvalloc_postprocess for : {self.sett.name_dir_export}', self.sett.log_name, overwrite_file=True)
            print_to_logfile('*model allocation specifications*:', self.sett.log_name)
            print_to_logfile(f'> n_bfs_municipalities: {len(self.sett.bfs_numbers)} \n> n_months_prediction: {self.sett.months_prediction} \n> n_montecarlo_iterations: {self.sett.MCspec_montecarlo_iterations}', self.sett.log_name)
            print_to_logfile(f'> pvalloc_settings, MCalloc_{self.sett.name_dir_export}', self.sett.log_name)
            for k, v in vars(self).items():
                print_to_logfile(f'{k}: {v}', self.sett.log_name)

    
            # POSTPROCESSING  ----------------------------------------------
            # mc_dir_paths = glob.glob(f'{self.sett.name_dir_export_path}/zMC*')

            # for i_mc_dir, mc_dir in enumerate(mc_dir_paths):

                # PREDICTION ACCURACY CALCULATION
                

  

    # ======================================================================================================
    # ALL METHODS CALLED ABOVE in the MAIN METHODS
    # ======================================================================================================
    
    # AUXILIARY ---------------------------------------------------------------------------
    def export_pvalloc_scen_settings(self):
            """
            Input:
                PVAllocScenario including the PVAllocScenario_Settings dataclass containing all scenarios settings. 
            (Output:)
                > Exports a JSON dict containing elements of the data class PVAllocScenario_Settings for pvallocation initialization.
                (all settings for the scenario).
            """
            sett_dict = asdict(self.sett)

            with open(f'{self.sett.name_dir_export_path}/pvalloc_sett.json', 'w') as f:
                json.dump(sett_dict, f, indent=4)


    def mark_to_timing_csv(self, step = None, substep = None,timestamp = None, runtime = None, descr = None):
        """
        Input:
            > PVAllocScenario class containing a data class (_Settings) which specifies all scenraio settings (e.g. also path to preprep data directory where all data for the
              executed computations is stored).
        Tasks: 
            > if not existing, create a timing csv file in the export directory of the scenario.
            > append the current date and time to the csv file from a given timestamp

        Output:
            > csv file in the export directory of the scenario containing the date and time of the last run of the script.           
        """
        # create timing file if not existing
        if not os.path.exists(self.sett.timing_marks_csv_path):
            df = pd.DataFrame({
                'step': [step,],
                'substep': [substep,],
                'timestamp': [timestamp,], 
                'runtime': [runtime,], 
                'descr': [descr,]
            })
            df.to_csv(self.sett.timing_marks_csv_path, index=False)
        
        elif os.path.exists(self.sett.timing_marks_csv_path):
            df = pd.read_csv(self.sett.timing_marks_csv_path)   

            new_row = {
                'step': step,
                'substep': substep,
                'timestamp': timestamp, 
                'runtime': runtime, 
                'descr': descr
            }
            
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(self.sett.timing_marks_csv_path, index=False)




    # INITIALIZATION ---------------------------------------------------------------------------
    if True:

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


        def initial_sml_get_gridnodes_DSO(self,):
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
                    node_centroid = sub_gwr_node.unary_union.centroid
                    kVA_treshold = sub_gwr_node['kVA_threshold'].unique()[0]

                    node_centroid_list.append([node, kVA_treshold, node_centroid])

            dsonodes_gdf = gpd.GeoDataFrame(node_centroid_list, columns=['grid_node', 'kVA_threshold', 'geometry'], crs='EPSG:2056')
            dsonodes_df = dsonodes_gdf.loc[:,dsonodes_gdf.columns != 'geometry']

            dsonodes_in_gwr_df = gwr_gdf.merge(Map_egid_dsonode, how='left', on='EGID')

            # summary prints ----------------------
            print_to_logfile('DSO grid nodes information:', self.sett.summary_name)
            checkpoint_to_logfile(f'Total: {Map_egid_dsonode["grid_node"].nunique()} DSO grid nodes for {Map_egid_dsonode["EGID"].nunique()} unique EGIDs (Map_egid_dsonode.shape {Map_egid_dsonode.shape[0]}, node/egid ratio: {round(Map_egid_dsonode["grid_node"].nunique() / Map_egid_dsonode["EGID"].nunique(),4)*100}%', self.sett.summary_name)
            checkpoint_to_logfile(f'In sample: {dsonodes_in_gwr_df["grid_node"].nunique()} DSO grid nodes for {dsonodes_in_gwr_df["EGID"].nunique()} EGIDs in {len(self.sett.bfs_numbers)} BFSs , (node/egid ratio: {round(dsonodes_in_gwr_df["grid_node"].nunique()/dsonodes_in_gwr_df["EGID"].nunique(),4)*100}%)', self.sett.summary_name)
            

            # export ----------------------
            dsonodes_df.to_parquet(f'{self.sett.name_dir_export_path}/dsonodes_df.parquet')
            dsonodes_df.to_csv(f'{self.sett.name_dir_export_path}/dsonodes_df.csv') if self.sett.export_csvs else None
            with open(f'{self.sett.name_dir_export_path}/dsonodes_gdf.geojson', 'w') as f:
                f.write(dsonodes_gdf.to_json())


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
            checkpoint_to_logfile('exported pvinstcost_table', self.sett.log_name, 5)

            return estim_instcost_chfpkW, estim_instcost_chftotal


        def initial_sml_get_instcost_interpolate_function(self, ): 
            """
            Input:
                PVAllocScenario + PVAllocScenario_Settings dataclass containing all scenarios settings + methods
            Tasks:
                - import the coefficients for the interpolated cost function from energie rechner schweiz
                - bring the coefficients back into their functional form. 
            Output:
                - estim_instcost_chfpkW, estim_instcost_chftotal: cost estimation functions to estimate the installation cost per kW and total installation cost
            """
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
                gwr = gwr.loc[(gwr['GSTAT'].isin(self.sett.GWRspec_GSTAT)) &
                            (gwr['GKLAS'].isin(self.sett.GWRspec_GKLAS)) &
                            (gwr['GBAUJ'] >= self.sett.GWRspec_GBAUJ_minmax[0]) &
                            (gwr['GBAUJ'] <= self.sett.GWRspec_GBAUJ_minmax[1])]
                gwr['GBAUJ'] = gwr['GBAUJ'].astype(str)
                gwr.loc[gwr['GBAUJ'] == '0', 'GBAUJ'] = ''
                # because not all buldings have dwelling information, need to remove dwelling columns and rows again (remove duplicates where 1 building had multiple dwellings)
                if self.sett.GWRspec_dwelling_cols == []:
                    gwr = copy.deepcopy(gwr.loc[:, self.sett.GWRspec_building_cols])
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
                pv = pd.read_parquet(f'{self.sett.name_dir_import_path}/pv.parquet')
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
                pvtarif = pvtarif.merge(Map_gm_ewr, how='left', on='nrElcom')

                pvtarif['bfs'] = pvtarif['bfs'].astype(str)
                # pvtarif[pvtarif_col] = pvtarif[pvtarif_col].fillna(0).astype(float)
                pvtarif[pvtarif_col] = pvtarif[pvtarif_col].replace('', 0).astype(float)

                # transformation
                pvtarif = pvtarif.loc[(pvtarif['year'] == str(pvtarif_year)[2:4]) & 
                                    (pvtarif['bfs'].isin((self.sett.bfs_numbers)))]

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


                # Map demandtypes > egid -------
                with open(f'{self.sett.name_dir_import_path}/Map_demandtype_EGID.json', 'r') as file:
                    Map_demandtypes_egid = json.load(file)


                # Map egid > demandtypes -------
                with open(f'{self.sett.name_dir_import_path}/Map_EGID_demandtypes.json', 'r') as file:
                    Map_egid_demandtypes = json.load(file)


                # Map egid > node -------
                Map_egid_dsonode = pd.read_parquet(f'{self.sett.name_dir_import_path}/Map_egid_dsonode.parquet')
                Map_egid_dsonode['EGID'] = Map_egid_dsonode['EGID'].astype(str)
                Map_egid_dsonode['grid_node'] = Map_egid_dsonode['grid_node'].astype(str)
                Map_egid_dsonode.index = Map_egid_dsonode['EGID']


                # dsonodes data -------
                dsonodes_df  = pd.read_parquet(f'{self.sett.name_dir_import_path}/dsonodes_df.parquet')
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
            
            Map_appendings_df = pd.DataFrame(Map_egid_dsonode_appendings, columns=['EGID', 'grid_node', 'kVA_threshold'])
            Map_egid_dsonode = pd.concat([Map_egid_dsonode, Map_appendings_df], axis=0)

            gwr_before_dsonode_selection = copy.deepcopy(gwr)
            gwr = copy.deepcopy(gwr.loc[gwr['EGID'].isin(Map_egid_dsonode['EGID'].unique())])
                

            # summary prints ----------
            print_to_logfile('\nEGID selection for TOPOLOGY:', self.sett.summary_name)
            checkpoint_to_logfile('Loop for topology creation over GWR EGIDs', self.sett.summary_name, 5, True)
            checkpoint_to_logfile('In Total: {gwr["EGID"].nunique()} gwrEGIDs ({round(gwr["EGID"].nunique() / gwr_before_solkat_selection["EGID"].nunique() * 100,1)}% of {gwr_before_solkat_selection["EGID"].nunique()} total gwrEGIDs) are used for topology creation', self.sett.summary_name, 3, True)
            checkpoint_to_logfile('  The rest drops out because gwrEGID not present in all data sources', self.sett.summary_name, 3, True)
            
            subtraction1 = gwr_before_solkat_selection["EGID"].nunique() - gwr_before_dsonode_selection["EGID"].nunique()
            checkpoint_to_logfile(f'  > {subtraction1} ({round(subtraction1 / gwr_before_solkat_selection["EGID"].nunique()*100,1)} % ) gwrEGIDs missing in solkat', self.sett.summary_name, 5, True)
            
            subtraction2 = gwr_before_dsonode_selection["EGID"].nunique() - gwr["EGID"].nunique()
            checkpoint_to_logfile(f'  > {subtraction2} ({round(subtraction2 / gwr_before_dsonode_selection["EGID"].nunique()*100,1)} % ) gwrEGIDs missing in dsonodes', self.sett.summary_name, 5, True)
            if Map_appendings_df.shape[0] > 0:

                checkpoint_to_logfile(f'  > (REMARK: Even matched {Map_appendings_df.shape[0]} EGIDs matched artificially to gridnode, because EGID lies in close node range, max_distance_m_for_EGID_node_matching: {self.sett.TECspec_max_distance_m_for_EGID_node_matching} meters', self.sett.summary_name, 3, True)
            elif Map_appendings_df.shape[0] == 0:
                checkpoint_to_logfile(f'  > (REMARK: No EGIDs matched to nearest gridnode, max_distance_m_for_EGID_node_matching: {self.sett.TECspec_max_distance_m_for_EGID_node_matching} meters', self.sett.summary_name, 3, True)



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
            checkpoint_to_logfile('start attach to topo', self.sett.log_name, 1 , True)

            # transform to np.array for faster lookups
            pv_npry, gwr_npry, elecpri_npry = np.array(pv), np.array(gwr), np.array(elecpri) 



            for i, egid in enumerate(gwr['EGID']):

                # add pv data --------
                pv_inst = {
                    'inst_TF': False,
                    'info_source': '',
                    'xtf_id': '',
                    'BeginOp': '',
                    'InitialPower': '',
                    'TotalPower': '',
                }
                egid_without_pv = []
                Map_xtf = Map_egid_pv.loc[Map_egid_pv['EGID'] == egid, 'xtf_id']

                if Map_xtf.empty:
                    egid_without_pv.append(egid)

                elif not Map_xtf.empty:
                    xtfid = Map_xtf.iloc[0]
                    if xtfid not in pv['xtf_id'].values:
                        checkpoint_to_logfile(f'---- pv xtf_id {xtfid} in Mapping_egid_pv, but NOT in pv data', self.sett.log_name, 3, False)
                        
                    if (Map_xtf.shape[0] == 1) and (xtfid in pv['xtf_id'].values):
                        mask_xtfid = np.isin(pv_npry[:, pv.columns.get_loc('xtf_id')], [xtfid,])

                        pv_inst['inst_TF'] = True
                        pv_inst['info_source'] = 'pv_df'
                        pv_inst['xtf_id'] = str(xtfid)
                        
                        pv_inst['BeginOp'] = pv_npry[mask_xtfid, pv.columns.get_loc('BeginningOfOperation')][0]
                        pv_inst['InitialPower'] = pv_npry[mask_xtfid, pv.columns.get_loc('InitialPower')][0]
                        pv_inst['TotalPower'] = pv_npry[mask_xtfid, pv.columns.get_loc('TotalPower')][0]
                    
                        # pv_inst['BeginOp'] = pv.loc[pv['xtf_id'] == xtfid, 'BeginningOfOperation'].iloc[0]
                        # pv_inst['InitialPower'] = pv.loc[pv['xtf_id'] == xtfid, 'InitialPower'].iloc[0]
                        # pv_inst['TotalPower'] = pv.loc[pv['xtf_id'] == xtfid, 'TotalPower'].iloc[0]
                        
                    elif Map_xtf.shape[0] > 1:
                        checkpoint_to_logfile(f'ERROR: multiple xtf_ids for EGID: {egid}', self.sett.log_name, 3, self.sett.show_debug_prints)
                        CHECK_egid_with_problems.append((egid, 'multiple xtf_ids'))


                # add solkat data --------
                if egid in solkat['EGID'].unique():
                    solkat_sub = solkat.loc[solkat['EGID'] == egid]
                    if solkat.duplicated(subset=['DF_UID', 'EGID']).any():
                        solkat_sub = solkat_sub.drop_duplicates(subset=['DF_UID', 'EGID'])
                    solkat_partitions = solkat_sub.set_index('DF_UID')[['FLAECHE', 'STROMERTRAG', 'AUSRICHTUNG', 'NEIGUNG']].to_dict(orient='index')                   
                
                elif egid not in solkat['EGID'].unique():
                    solkat_partitions = {}
                    checkpoint_to_logfile(f'egid {egid} not in solkat', self.sett.log_name, 3, self.sett.show_debug_prints)


                # add demand type --------
                if egid in Map_egid_demandtypes.keys():
                    demand_type = Map_egid_demandtypes[egid]
                elif egid not in Map_egid_demandtypes.keys():
                    print_to_logfile(f'\n ** ERROR ** EGID {egid} not in Map_egid_demandtypes, but must be because Map file is based on GWR', self.sett.log_name)
                    demand_type = 'NA'
                    CHECK_egid_with_problems.append((egid, 'not in Map_egid_demandtypes (both based on GWR)'))

                # add pvtarif --------
                bfs_of_egid = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GGDENR')][0]
                pvtarif_egid = sum([pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), col].sum() for col in pvtarif_col])

                pvtarif_sub = pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid)]
                if pvtarif_sub.empty:
                    checkpoint_to_logfile(f'ERROR: no pvtarif data for EGID {egid}', self.sett.log_name, 3, self.sett.show_debug_prints)
                    ewr_info = {}
                    CHECK_egid_with_problems.append((egid, 'no pvtarif data'))
                elif pvtarif_sub.shape[0] == 1:
                    ewr_info = {
                        'nrElcom': pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), 'nrElcom'].iloc[0],
                        'name': pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), 'nomEw'].iloc[0],
                        'energy1': pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), 'energy1'].sum(),
                        'eco1': pvtarif.loc[pvtarif['bfs'] == str(bfs_of_egid), 'eco1'].sum(),
                    }
                elif pvtarif_sub.shape[0] > 1:
                    ewr_info = {
                        'nrElcom': pvtarif_sub['nrElcom'].unique().tolist(),
                        'name': pvtarif_sub['nomEw'].unique().tolist(),
                        'energy1': pvtarif_sub['energy1'].mean(),
                        'eco1': pvtarif_sub['eco1'].mean(),
                    }
                
                    # checkpoint_to_logfile(f'multiple pvtarif data for EGID {egid}', self.sett.log_name, 3, self.sett.show_debug_prints)
                    CHECK_egid_with_problems.append((egid, 'multiple pvtarif data'))


                # add elecpri --------
                elecpri_egid = {}
                elecpri_info = {}

                mask_bfs = np.isin(elecpri_npry[:, elecpri.columns.get_loc('bfs_number')], [bfs_of_egid,]) 
                mask_year = np.isin(elecpri_npry[:, elecpri.columns.get_loc('year')],    self.sett.TECspec_elecpri_year)
                mask_cat = np.isin(elecpri_npry[:, elecpri.columns.get_loc('category')], self.sett.TECspec_elecpri_category)

                if sum(mask_bfs & mask_year & mask_cat) < 1:
                    checkpoint_to_logfile(f'ERROR: no elecpri data for EGID {egid}', self.sett.log_name, 3, self.sett.show_debug_prints)
                    CHECK_egid_with_problems.append((egid, 'no elecpri data'))
                elif sum(mask_bfs & mask_year & mask_cat) > 1:
                    checkpoint_to_logfile(f'ERROR: multiple elecpri data for EGID {egid}', self.sett.log_name, 3, self.sett.show_debug_prints)
                    CHECK_egid_with_problems.append((egid, 'multiple elecpri data'))
                elif sum(mask_bfs & mask_year & mask_cat) == 1:
                    energy =   elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('energy')].sum()
                    grid =     elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('grid')].sum()
                    aidfee =   elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('aidfee')].sum()
                    taxes =    elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('taxes')].sum()
                    fixcosts = elecpri_npry[(mask_bfs & mask_year & mask_cat), elecpri.columns.get_loc('fixcosts')].sum()

                    elecpri_egid = energy + grid + aidfee + taxes + fixcosts
                    elecpri_info = {
                        'energy': energy,
                        'grid': grid,
                        'aidfee': aidfee,
                        'taxes': taxes,
                        'fixcosts': fixcosts,
                    }


                    # add GWR --------
                    bfs_of_egid = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GGDENR')][0] 
                    glkas_of_egid = gwr_npry[np.isin(gwr_npry[:, gwr.columns.get_loc('EGID')], [egid,]), gwr.columns.get_loc('GKLAS')][0]
                    gwr_info ={
                        'bfs': bfs_of_egid,
                        'gklas': glkas_of_egid,
                    }

                    # add grid node --------
                    if isinstance(Map_egid_dsonode.loc[egid, 'grid_node'], str):
                        grid_node = Map_egid_dsonode.loc[egid, 'grid_node']
                    elif isinstance(Map_egid_dsonode.loc[egid, 'grid_node'], pd.Series):
                        grid_node = Map_egid_dsonode.loc[egid, 'grid_node'].iloc[0]
                        

                # attach to topo --------
                # topo['EGID'][egid] = {
                topo_egid[egid] = {
                    'gwr_info': gwr_info,
                    'grid_node': grid_node,
                    'pv_inst': pv_inst,
                    'solkat_partitions': solkat_partitions, 
                    'demand_type': demand_type,
                    'pvtarif_Rp_kWh': pvtarif_egid, 
                    'EWR': ewr_info, 
                    'elecpri_Rp_kWh': elecpri_egid,
                    'elecpri_info': elecpri_info,
                    }  

                # Checkpoint prints
                if i % modulus_print == 0:
                    print_to_logfile(f'\t -- EGID {i} of {len(gwr["EGID"])} {15*"-"}', self.sett.log_name)

                
            # end loop ------------------------------------------------
            checkpoint_to_logfile('end attach to topo', self.sett.log_name, 1 , True)
            print_to_logfile('\nsanity check for installtions in topo_egid', self.sett.summary_name)
            checkpoint_to_logfile(f'number of EGIDs with multiple installations: {CHECK_egid_with_problems.count("multiple xtf_ids")}', self.sett.summary_name)


            # EXPORT TOPO + Mappings ============================================================================

            with open(f'{self.sett.name_dir_export_path}/topo_egid.txt', 'w') as f:
                f.write(str(topo_egid))
            
            with open(f'{self.sett.name_dir_export_path}/topo_egid.json', 'w') as f:
                json.dump(topo_egid, f)

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
            checkpoint_to_logfile(f'import TS: lookback range   {start_loockback} to {T0-pd.DateOffset(hours=1)}', self.sett.log_name, 2)
            checkpoint_to_logfile(f'import TS: prediction range {T0} to {end_prediction}', self.sett.log_name, 2)

            Map_daterange = pd.DataFrame({'date_range': date_range, 'DoY': date_range.dayofyear, 'hour': date_range.hour})
            Map_daterange['HoY'] = (Map_daterange['DoY'] - 1) * 24 + (Map_daterange['hour']+1)
            Map_daterange['t'] = Map_daterange['HoY'].apply(lambda x: f't_{x}')


            # IMPORT ----------------------------------------------------------------------------

            # demand types --------
            demandtypes_tformat = pd.read_parquet(f'{self.sett.name_dir_import_path}/demandtypes.parquet')
            demandtypes_ts = demandtypes_tformat.copy()

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
            # setup 
            if os.path.exists(f'{self.sett.name_dir_export_path}/gridprem_ts.parquet'):
                os.remove(f'{self.sett.name_dir_export_path}/gridprem_ts.parquet')    

            # import 
            dsonodes_df = pd.read_parquet(f'{self.sett.name_dir_import_path}/dsonodes_df.parquet')
            t_range = [f't_{t}' for t in range(1,8760 + 1)]

            gridprem_ts = pd.DataFrame(np.repeat(dsonodes_df.values, len(t_range), axis=0), columns=dsonodes_df.columns)  
            gridprem_ts['t'] = np.tile(t_range, len(dsonodes_df))
            gridprem_ts['prem_Rp_kWh'] = 0

            gridprem_ts = gridprem_ts[['t', 'grid_node', 'kVA_threshold', 'prem_Rp_kWh']]
            gridprem_ts.drop(columns='kVA_threshold', inplace=True)

            # export 
            gridprem_ts.to_parquet(f'{self.sett.name_dir_export_path}/gridprem_ts.parquet')

            

            # EXPORT ----------------------------------------------------------------------------
            ts_names = ['Map_daterange', 'demandtypes_ts', 'meteo_ts', 'gridprem_ts' ]
            ts_list =  [ Map_daterange,   demandtypes_ts,   meteo_ts,   gridprem_ts]
            for i, ts in enumerate(ts_list):
                ts.to_parquet(f'{self.sett.name_dir_export_path}/{ts_names[i]}.parquet')


            # RETURN ----------------------------------------------------------------------------
            return ts_list, ts_names


        def initial_lrg_define_construction_capacity(self,
                                                    topo, 
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
                - months_prediction_df  ->
            """

            # import settings + setup -------------------
            print_to_logfile('run function: define_construction_capacity.py', self.sett.log_name)



            # create monthly time structure
            T0 = pd.to_datetime(f'{self.sett.T0_prediction}')
            start_loockback = T0 - pd.DateOffset(months=self.sett.months_lookback) #+ pd.DateOffset(hours=1)
            end_prediction = T0 + pd.DateOffset(months=self.sett.months_prediction) - pd.DateOffset(hours=1)
            months_lookback = pd.date_range(start=start_loockback, end=T0, freq='ME').to_period('M')
            months_prediction = pd.date_range(start=(T0 + pd.DateOffset(days=1)), end=end_prediction, freq='ME').to_period('M')


            # IMPORT ----------------------------------------------------------------------------
            pv = df_list[df_names.index('pv')]
            Map_egid_pv = df_list[df_names.index('Map_egid_pv')]

            topo_keys = list(topo.keys())

            # subset pv to EGIDs in TOPO, and LOOKBACK period of pvalloc settings
            pv_sub = copy.deepcopy(pv)
            del_cols = ['MainCategory', 'SubCategory', 'PlantCategory']
            pv_sub.drop(columns=del_cols, inplace=True)

            pv_sub = pv_sub.merge(Map_egid_pv, how='left', on='xtf_id')
            pv_sub = pv_sub.loc[pv_sub['EGID'].isin(topo_keys)]
            pv_plot = copy.deepcopy(pv_sub) # used for plotting later

            pv_sub['BeginningOfOperation'] = pd.to_datetime(pv_sub['BeginningOfOperation'])
            pv_sub['MonthPeriod'] = pv_sub['BeginningOfOperation'].dt.to_period('M')
            pv_sub = pv_sub.loc[pv_sub['MonthPeriod'].isin(months_lookback)]

            # plot total power over time
            if True: 
                pv_plot['BeginningOfOperation'] = pd.to_datetime(pv_plot['BeginningOfOperation'])
                pv_plot.set_index('BeginningOfOperation', inplace=True)

                # Resample by week, month, and year and calculate the sum of TotalPower
                weekly_sum = pv_plot['TotalPower'].resample('W').sum()
                monthly_sum = pv_plot['TotalPower'].resample('ME').sum()
                yearly_sum = pv_plot['TotalPower'].resample('YE').sum()

                # Create traces for each time period
                trace_weekly = go.Scatter(x=weekly_sum.index, y=weekly_sum.values, mode='lines', name='Weekly')
                trace_monthly = go.Scatter(x=monthly_sum.index, y=monthly_sum.values, mode='lines', name='Monthly')
                trace_yearly = go.Scatter(x=yearly_sum.index, y=yearly_sum.values, mode='lines', name='Yearly')

                layout = go.Layout(
                    title='Built PV Capacity within Sample of GWR EGIDs',
                    xaxis=dict(title='Time',
                            range = ['2010-01-01', '2024-5-31']),
                    yaxis=dict(title='Total Power'),
                    legend=dict(x=0, y=1),
                        shapes=[
                            # Shaded region for months_lookback
                            dict(
                                type="rect",
                                xref="x",
                                yref="paper",
                                x0=months_lookback[0].start_time,
                                x1=months_lookback[-1].end_time,
                                y0=0,
                                y1=1,
                                fillcolor="LightSalmon",
                                opacity=0.3,
                                layer="below",
                                line_width=0,
                    )
                ]
                )
                fig = go.Figure(data=[trace_weekly, trace_monthly, trace_yearly], layout=layout)
                # fig.show()
                fig.write_html(f'{self.sett.name_dir_export_path}/pv_total_power_over_time.html')


            # CAPACITY ASSIGNMENT ----------------------------------------------------------------------------
            capacity_growth = self.sett.CSTRspec_ann_capacity_growth
            month_constr_capa_tuples = self.sett.CSTRspec_month_constr_capa_tuples

            sum_TP_kW_lookback = pv_sub['TotalPower'].sum()

            constrcapa = pd.DataFrame({'date': months_prediction, 'year': months_prediction.year, 'month': months_prediction.month})
            years_prediction = months_prediction.year.unique()
            i, y = 0, years_prediction[0]
            for i,y in enumerate(years_prediction):

                TP_y = sum_TP_kW_lookback * (1 + capacity_growth)**(i+1)
                for m, TP_m in month_constr_capa_tuples:
                    constrcapa.loc[(constrcapa['year'] == y) & 
                                (constrcapa['month'] == m), 'constr_capacity_kw'] = TP_y * TP_m
                
            months_prediction_df = pd.DataFrame({'date': months_prediction, 'year': months_prediction.year, 'month': months_prediction.month})

            # PRINTs to LOGFILE ----------------------------------------------------------------------------
            checkpoint_to_logfile(f'constr_capacity month lookback, between :                {months_lookback[0]} to {months_lookback[-1]}', self.sett.log_name, 2)
            checkpoint_to_logfile(f'constr_capacity KW built in period (sum_TP_kW_lookback): {round(sum_TP_kW_lookback,2)} kW', self.sett.log_name, 2)
            print_to_logfile('\n', self.sett.log_name)
            checkpoint_to_logfile(f'constr_capacity: month prediction {months_prediction[0]} to {months_prediction[-1]}', self.sett.log_name, 2)
            checkpoint_to_logfile(f'sum_TP_kw_lookback {round(sum_TP_kW_lookback,3)} kW to distribute across months_prediction', self.sett.log_name, 2)
            print_to_logfile('\n', self.sett.log_name)
            checkpoint_to_logfile(f'sum_TP_kW_lookback (T0: {round(sum_TP_kW_lookback,2)} kW) increase by {capacity_growth*100}% per year', self.sett.log_name, 2)


            # EXPORT ----------------------------------------------------------------------------
            constrcapa.to_parquet(f'{self.sett.name_dir_export_path}/constrcapa.parquet')
            constrcapa.to_csv(f'{self.sett.name_dir_export_path}/constrcapa.csv', index=False)

            months_prediction_df.to_parquet(f'{self.sett.name_dir_export_path}/months_prediction.parquet')
            months_prediction_df.to_csv(f'{self.sett.name_dir_export_path}/months_prediction.csv', index=False)

            return constrcapa, months_prediction, months_lookback



    # SANITY CHECK ---------------------------------------------------------------------------
    if True:
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
            topo = json.load(open(f'{subdir_path}/topo_egid.json', 'r'))
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
                            row_interest_rate, row_years_maturity, row_selfconsumption, row_pvprod_method, 
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
                        row_demand_type['key'], row_demand_type['descr'], row_demand_type['val'] = 'demand_type', 'type of artifical demand profile (Netflex, maybe CKW later)', topo.get(egid).get('demand_type')

                        # row_pvinst_info, row_pvinst_BeginOp, row_pvinst_TotalPower = get_new_row(), get_new_row(), get_new_row() 
                        row_pvinst_info['key'], row_pvinst_info['descr'], row_pvinst_info['val'] = 'pv_inst > info_source', 'Origin behind pv inst on house (real data or model alloc)', topo.get(egid).get('pv_inst').get('info_source')
                        row_pvinst_BeginOp['key'], row_pvinst_BeginOp['descr'], row_pvinst_BeginOp['val'] = 'pv_inst > BeginOp', 'begin of operation', topo.get(egid).get('pv_inst').get('BeginOp')
                        row_pvinst_TotalPower['key'], row_pvinst_TotalPower['descr'], row_pvinst_TotalPower['val'], row_pvinst_TotalPower['unit'] = 'pv_inst > TotalPower', 'total power of PV installation', topo.get(egid).get('pv_inst').get('TotalPower'), 'kW'

                        # row_elecpri, row_pvtarif = get_new_row(), get_new_row() 
                        row_elecpri['key'], row_elecpri['descr'], row_elecpri['val'], row_elecpri['unit'], row_elecpri['col1'], row_elecpri['col2'] = 'elecpri', 'mean electricity price per BFS area', topo.get(egid).get('elecpri_Rp_kWh'), 'Rp/kWh', f"elecpri_info: {topo.get(egid).get('elecpri_info')}",f"year: {self.sett.TECspec_elecpri_year}"
                        row_pvtarif['key'], row_pvtarif['descr'], row_pvtarif['val'], row_pvtarif['unit'], row_pvtarif['col1'], row_pvtarif['col2'] = 'pvtarif', 'tariff for PV feedin to EWR',topo.get(egid).get('pvtarif_Rp_kWh'), 'Rp/kWh', f"EWRs: {topo.get(egid).get('EWR').get('name')}", f"year: {self.sett.TECspec_pvtarif_year}"
                        row_interest_rate['key'], row_interest_rate['descr'],row_interest_rate['val'] = 'interest_rate', 'generic interest rate used for dicsounting NPV calculation',              self.sett.TECspec_interest_rate
                        row_years_maturity['key'], row_years_maturity['descr'], row_years_maturity['val'] = 'invst_maturity', 'number of years that consider pv production for NPV calculation',    self.sett.TECspec_invst_maturity

                        # row_selfconsumption, row_interest_rate, row_years_maturity, row_kWpeak_per_m2  = get_new_row(), get_new_row(), get_new_row(), get_new_row()
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
            topo = json.load(open(f'{self.sett.name_dir_export_path}/topo_egid.json', 'r'))
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

            check_egid = json.load(open(f'{self.sett.name_dir_export_path}/CHECK_egid_with_problems.json', 'r'))
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


 
    # MC ALGORITHM ---------------------------------------------------------------------------
    if True: 
        
        def algo_calc_economics_in_topo_df(self, 
                                           topo, 
                                           df_list, df_names, 
                                           ts_list, ts_names,
        ): 
                    
            # setup -----------------------------------------------------
            print_to_logfile('run function: calc_economics_in_topo_df', self.sett.log_name)


            # import -----------------------------------------------------
            angle_tilt_df = df_list[df_names.index('angle_tilt_df')]
            solkat_month = df_list[df_names.index('solkat_month')]
            demandtypes_ts = ts_list[ts_names.index('demandtypes_ts')]
            meteo_ts = ts_list[ts_names.index('meteo_ts')]


            # TOPO to DF =============================================
            # solkat_combo_df_exists = os.path.exists(f'{pvalloc_settings["interim_path"]}/solkat_combo_df.parquet')
            # if pvalloc_settings['recalc_economics_topo_df']:
            topo = topo

            egid_list, df_uid_list, bfs_list, gklas_list, demandtype_list, grid_node_list  = [], [], [], [], [], []
            inst_list, info_source_list, pvdf_totalpower_list, pvid_list, pv_tarif_Rp_kWh_list = [], [], [], [], []
            flaeche_list, stromertrag_list, ausrichtung_list, neigung_list, elecpri_list = [], [], [], [], []

            keys = list(topo.keys())

            for k,v in topo.items():
                # if k in no_pv_egid:
                # ADJUSTMENT: this needs to be removed, because I also need to calculate the pvproduction_kW per house 
                # later when quantifying the grid feedin per grid node
                partitions = v.get('solkat_partitions')

                for k_p, v_p in partitions.items():
                    egid_list.append(k)
                    df_uid_list.append(k_p)
                    bfs_list.append(v.get('gwr_info').get('bfs'))
                    gklas_list.append(v.get('gwr_info').get('gklas'))
                    demandtype_list.append(v.get('demand_type'))
                    grid_node_list.append(v.get('grid_node'))

                    inst_list.append(v.get('pv_inst').get('inst_TF'))
                    info_source_list.append(v.get('pv_inst').get('info_source'))
                    pvid_list.append(v.get('pv_inst').get('xtf_id'))
                    pv_tarif_Rp_kWh_list.append(v.get('pvtarif_Rp_kWh'))
                    pvdf_totalpower_list.append(v.get('pv_inst').get('TotalPower'))

                    flaeche_list.append(v_p.get('FLAECHE'))
                    ausrichtung_list.append(v_p.get('AUSRICHTUNG'))
                    stromertrag_list.append(v_p.get('STROMERTRAG'))
                    neigung_list.append(v_p.get('NEIGUNG'))
                    elecpri_list.append(v.get('elecpri_Rp_kWh'))
                        
                
            topo_df = pd.DataFrame({'EGID': egid_list, 'df_uid': df_uid_list, 'bfs': bfs_list,
                                    'gklas': gklas_list, 'demandtype': demandtype_list, 'grid_node': grid_node_list,

                                    'inst_TF': inst_list, 'info_source': info_source_list, 'pvid': pvid_list,
                                    'pv_tarif_Rp_kWh': pv_tarif_Rp_kWh_list, 'TotalPower': pvdf_totalpower_list,

                                    'FLAECHE': flaeche_list, 'AUSRICHTUNG': ausrichtung_list, 
                                    'STROMERTRAG': stromertrag_list, 'NEIGUNG': neigung_list, 
                                    'elecpri_Rp_kWh': elecpri_list})
            

            # make or clear dir for subdfs ----------------------------------------------
            subdf_path = f'{self.sett.name_dir_export_path}/topo_time_subdf'

            if not os.path.exists(subdf_path):
                os.makedirs(subdf_path)
            else:
                old_files = glob.glob(f'{subdf_path}/*')
                for f in old_files:
                    os.remove(f)
            

            # round NEIGUNG + AUSRICHTUNG to 5 for easier computation
            topo_df['NEIGUNG'] = topo_df['NEIGUNG'].apply(lambda x: round(x / 5) * 5)
            topo_df['AUSRICHTUNG'] = topo_df['AUSRICHTUNG'].apply(lambda x: round(x / 10) * 10)
            
            def lookup_angle_tilt_efficiency(row, angle_tilt_df):
                try:
                    return angle_tilt_df.loc[(row['AUSRICHTUNG'], row['NEIGUNG']), 'efficiency_factor']
                except KeyError:
                    return 0
            topo_df['angletilt_factor'] = topo_df.apply(lambda r: lookup_angle_tilt_efficiency(r, angle_tilt_df), axis=1)

            # transform TotalPower
            topo_df['TotalPower'] = topo_df['TotalPower'].replace('', '0').astype(float)

                    
            # MERGE + GET ECONOMIC VALUES FOR NPV CALCULATION =============================================
            topo_subdf_partitioner = self.sett.ALGOspec_topo_subdf_partitioner
            
            share_roof_area_available = self.sett.TECspec_share_roof_area_available
            inverter_efficiency       = self.sett.TECspec_inverter_efficiency
            panel_efficiency          = self.sett.TECspec_panel_efficiency
            pvprod_calc_method        = self.sett.TECspec_pvprod_calc_method
            kWpeak_per_m2             = self.sett.TECspec_kWpeak_per_m2

            flat_direct_rad_factor  = self.sett.WEAspec_flat_direct_rad_factor
            flat_diffuse_rad_factor = self.sett.WEAspec_flat_diffuse_rad_factor


            egids = topo_df['EGID'].unique()

            stepsize = topo_subdf_partitioner if len(egids) > topo_subdf_partitioner else len(egids)
            tranche_counter = 0
            checkpoint_to_logfile(' * * DEBUGGIGN * * *: START loop subdfs', self.sett.log_name, 1)
            for i in range(0, len(egids), stepsize):

                tranche_counter += 1
                # print_to_logfile(f'-- merges to topo_time_subdf {tranche_counter}/{len(range(0, len(egids), stepsize))} tranches ({i} to {i+stepsize-1} egids.iloc) ,  {7*"-"}  (stamp: {datetime.now()})', self.sett.log_name)
                subdf = copy.deepcopy(topo_df[topo_df['EGID'].isin(egids[i:i+stepsize])])


                # merge production, grid prem + demand to partitions ----------
                subdf['meteo_loc'] = 'Basel'
                meteo_ts['meteo_loc'] ='Basel' 
                
                # subdf = subdf.merge(meteo_ts[['rad_direct', 'rad_diffuse', 'temperature', 't', 'meteo_loc']], how='left', on='meteo_loc')
                subdf = subdf.merge(meteo_ts, how='left', on='meteo_loc')
                

                # add radiation per h to subdf, "flat" OR "dfuid_ind" ----------
                if self.sett.WEAspec_radiation_to_pvprod_method == 'flat':
                    subdf['radiation'] = subdf['rad_direct'] * flat_direct_rad_factor + subdf['rad_diffuse'] * flat_diffuse_rad_factor
                    meteo_ts['radiation'] = meteo_ts['rad_direct'] * flat_direct_rad_factor + meteo_ts['rad_diffuse'] * flat_diffuse_rad_factor
                    mean_top_radiation = meteo_ts['radiation'].nlargest(10).mean()

                    subdf['radiation_rel_locmax'] = subdf['radiation'] / mean_top_radiation


                elif self.sett.WEAspec_radiation_to_pvprod_method == 'dfuid_ind':
                    solkat_month.rename(columns={'DF_UID': 'df_uid', 'MONAT': 'month'}, inplace=True)
                    solkat_month['month'] = solkat_month['month'].astype(int)
                    subdf['month'] = subdf['timestamp'].dt.month.astype(int)
                    
                
                    checkpoint_to_logfile(f'  start merge solkat_month to subdf {i} to {i+stepsize-1}', self.sett.log_name, 1) if i < 2 else None
                    subdf = subdf.merge(solkat_month[['df_uid', 'month', 'A_PARAM', 'B_PARAM', 'C_PARAM']], how='left', on=['df_uid', 'month'])
                    checkpoint_to_logfile(f'  end merge solkat_month to subdf {i} to {i+stepsize-1}', self.sett.log_name, 1) if i < 2 else None
                    subdf['radiation'] = subdf['A_PARAM'] * subdf['rad_direct'] + subdf['B_PARAM'] * subdf['rad_diffuse'] + subdf['C_PARAM']
                    # some radiation values are negative, because of the linear transformation with abc parameters. 
                    # force all negative values to 0
                    subdf.loc[subdf['radiation'] < 0, 'radiation'] = 0
                    subdf.loc[(subdf['rad_direct'] == 0) & (subdf['rad_diffuse'] == 0), 'radiation'] = 0
                    # subdf['radiation'] = np.where(
                    #                         (subdf['rad_direct'] != 0) | (subdf['rad_diffuse'] != 0),
                    #                         subdf['A_PARAM'] * subdf['rad_direct'] + subdf['B_PARAM'] * subdf['rad_diffuse'] + subdf['C_PARAM'],
                    #                         0)

                    meteo_ts['radiation'] = meteo_ts['rad_direct'] * flat_direct_rad_factor + meteo_ts['rad_diffuse'] * flat_diffuse_rad_factor
                    # meteo_ts['radiation_abc_param_1dfuid'] = meteo_ts['rad_direct'] * subdf['A_PARAM'].mean() + meteo_ts['rad_diffuse'] * subdf['B_PARAM'].mean() + subdf['C_PARAM'].mean()


                    # radiation_rel_locmax by "df_uid_specific" vs "all_HOY" ---------- 
                    if self.sett.WEAspec_rad_rel_loc_max_by == 'dfuid_specific':
                        subdf_dfuid_topradation = subdf.groupby('df_uid')['radiation'].apply(lambda x: x.nlargest(10).mean()).reset_index()
                        subdf_dfuid_topradation.rename(columns={'radiation': 'mean_top_radiation'}, inplace=True)
                        subdf = subdf.merge(subdf_dfuid_topradation, how='left', on='df_uid')

                        subdf['radiation_rel_locmax'] = subdf['radiation'] / subdf['mean_top_radiation']

                    elif self.sett.WEAspec_rad_rel_loc_max_by == 'all_HOY':
                        mean_nlargest_rad_all_HOY = meteo_ts['radiation'].nlargest(10).mean()
                        subdf['radiation_rel_locmax'] = subdf['radiation'] / mean_nlargest_rad_all_HOY


                # add panel_efficiency by time ----------
                if self.sett.PEFspec_variable_panel_efficiency_TF:
                    summer_months      = self.sett.PEFspec_summer_months
                    hotsummer_hours    = self.sett.PEFspec_hotsummer_hours
                    hot_hours_discount = self.sett.PEFspec_hot_hours_discount

                    HOY_weatheryear_df = pd.read_parquet(f'{self.sett.name_dir_export_path}/HOY_weatheryear_df.parquet')
                    hot_hours_in_year = HOY_weatheryear_df.loc[(HOY_weatheryear_df['month'].isin(summer_months)) & (HOY_weatheryear_df['hour'].isin(hotsummer_hours))]
                    subdf['panel_efficiency'] = np.where(
                        subdf['t'].isin(hot_hours_in_year['t']),
                        panel_efficiency * (1-hot_hours_discount),
                        panel_efficiency)
                    
                elif not self.sett.PEFspec_variable_panel_efficiency_TF:
                    subdf['panel_efficiency'] = panel_efficiency
                    

                # attach demand profiles ----------
                demandtypes_names = [c for c in demandtypes_ts.columns if 'DEMANDprox' in c]
                demandtypes_melt = demandtypes_ts.melt(id_vars='t', value_vars=demandtypes_names, var_name= 'demandtype', value_name= 'demand')
                subdf = subdf.merge(demandtypes_melt, how='left', on=['t', 'demandtype'])
                subdf.rename(columns={'demand': 'demand_kW'}, inplace=True)
                # checkpoint_to_logfile(f'  end merge demandtypes for subdf {i} to {i+stepsize-1}', self.sett.log_name, 1)


                # attach FLAECH_angletilt, might be usefull for later calculations
                subdf = subdf.assign(FLAECH_angletilt = subdf['FLAECHE'] * subdf['angletilt_factor'])


                # compute production ---------- 
                # pvprod method 1 (false, presented to frank 8.11.24. missing efficiency grade)
                if pvprod_calc_method == 'method1':    
                    subdf = subdf.assign(pvprod_kW = (subdf['radiation'] * subdf['FLAECHE'] * subdf['angletilt_factor']) / 1000).drop(columns=['meteo_loc', 'radiation'])

                # pvprod method 2.1
                elif pvprod_calc_method == 'method2.1':   
                    subdf['pvprod_kW'] = (subdf['radiation'] / 1000 ) *                     inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] * subdf['angletilt_factor']
                    formla_for_log_print = "subdf['pvprod_kW'] = subdf['radiation'] / 1000 * inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] * subdf['angletilt_factor']"

                # pvprod method 2.2
                elif pvprod_calc_method == 'method2.2':   
                    subdf['pvprod_kW'] = (subdf['radiation'] / 1000 ) *                     inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] 
                    formla_for_log_print = "subdf['pvprod_kW'] = subdf['radiation'] / 1000 * inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE']"

                # pvprod method 3.1
                elif pvprod_calc_method == 'method3.1':
                    subdf['pvprod_kW'] =  subdf['radiation_rel_locmax'] * kWpeak_per_m2 *   inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] * subdf['angletilt_factor']
                    formla_for_log_print = "subdf['pvprod_kW'] = subdf['radiation_rel_locmax'] * kWpeak_per_m2 * inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] * subdf['angletilt_factor']"

                # pvprod method 3.2
                elif pvprod_calc_method == 'method3.2':
                    subdf['pvprod_kW'] =  subdf['radiation_rel_locmax'] * kWpeak_per_m2 *   inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] 
                    formla_for_log_print = "subdf['pvprod_kW'] = subdf['radiation_rel_locmax'] * kWpeak_per_m2 * inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE']"


                # pvprod method 3
                    # > 19.11.2024: no longer needed. from previous runs where I wanted to compare different pvprod_computations methods
                elif False:   
                    subdf['pvprod_kW'] = inverter_efficiency * share_roof_area_available * (subdf['radiation'] / 1000 ) * subdf['FLAECHE'] * subdf['angletilt_factor']
                    subdf.drop(columns=['meteo_loc', 'radiation'], inplace=True)
                    print_to_logfile("* calculation formula for pv production per roof:\n   > subdf['pvprod_kW'] = inverter_efficiency * share_roof_area_available * (subdf['radiation'] / 1000 ) * subdf['FLAECHE'] * subdf['angletilt_factor']\n", self.sett.log_name)
                    
                # pvprod method 4
                    # > 19.11.2024: because I dont have the same weather year as the calculations for the STROMERTRAG in solkat, it is not really feasible to back-engineer any type of shade deduction 
                    #   coefficient that might bring any additional information. 
                elif False:  
                    subdf['pvprod_kW_noshade'] =   (subdf['radiation'] / 1000 ) * subdf['FLAECHE'] # * subdf['angletilt_factor']
                    # check if no_shade production calculation is larger than STROMERTRAG (should be, and then later corrected...)
                    sum(subdf.loc[subdf['df_uid'] == subdf['df_uid'].unique()[0], 'pvprod_kW_noshade']), subdf.loc[subdf['df_uid'] == subdf['df_uid'].unique()[0], 'STROMERTRAG'].iloc[0]
                    
                    dfuid_subdf = subdf['df_uid'].unique()
                    dfuid = dfuid_subdf[0]
                    for dfuid in dfuid_subdf:
                        dfuid_TF = subdf['df_uid'] == dfuid
                        pvprod_kWhYear_noshade = subdf.loc[dfuid_TF, 'pvprod_kW_noshade'].sum()
                        stromertrag_dfuid = subdf.loc[dfuid_TF, 'STROMERTRAG'].iloc[0]
                        shading_factor = stromertrag_dfuid / pvprod_kWhYear_noshade
                        
                        if shading_factor > 1:
                            checkpoint_to_logfile(f' *ERROR* shading factor > 1 for df_uid: {dfuid}, EGID: {subdf.loc[dfuid_TF, "EGID"].unique()} ', self.sett.log_name, 1)
                        subdf.loc[dfuid_TF, 'pvprod_kW'] = subdf.loc[dfuid_TF, 'pvprod_kW_noshade'] * shading_factor
                    subdf.drop(columns=['meteo_loc', 'radiation', 'pvprod_kW_noshade'], inplace=True)
                    print_to_logfile("* calculation formula for pv production per roof:\n   > subdf['pvprod_kW'] = <retrofitted_shading_factor> * inverter_efficiency  * (subdf['radiation'] / 1000 ) * subdf['FLAECHE'] * subdf['angletilt_factor'] \n", self.sett.log_name)
                    

                # export subdf ----------------------------------------------
                subdf.to_parquet(f'{subdf_path}/topo_subdf_{i}to{i+stepsize-1}.parquet')
                if self.sett.export_csvs:
                    subdf.to_csv(f'{subdf_path}/topo_subdf_{i}to{i+stepsize-1}.csv', index=False)
                if (i == 0) & self.sett.export_csvs:
                    subdf.to_csv(f'{subdf_path}/topo_subdf_{i}to{i+stepsize-1}.csv', index=False)
                checkpoint_to_logfile(f'end merge to topo_time_subdf (tranche {tranche_counter}/{len(range(0, len(egids), stepsize))}, size {stepsize})', self.sett.log_name, 1)
                checkpoint_to_logfile(' * * DEBUGGIGN * * *: END loop subdfs', self.sett.log_name, 1)


            # print computation formula for comparing methods
            print_to_logfile(f'* Computation formula for pv production per roof:\n{formla_for_log_print}', self.sett.log_name)


        def algo_update_gridprem_POLARS(self, subdir_path: str, i_m: int, m): 
    
            # setup -----------------------------------------------------
            print_to_logfile('run function: update_gridprem', self.sett.log_name)
            gridtiers_power_factor  = self.sett.GRIDspec_power_factor

            # import  -----------------------------------------------------
            topo = json.load(open(f'{subdir_path}/topo_egid.json', 'r'))
            dsonodes_df = pl.read_parquet(f'{subdir_path}/dsonodes_df.parquet')    # dsonodes_df = pd.read_parquet(f'{subdir_path}/dsonodes_df.parquet')
            gridprem_ts = pl.read_parquet(f'{subdir_path}/gridprem_ts.parquet')    # gridprem_ts = pd.read_parquet(f'{subdir_path}/gridprem_ts.parquet')

            data = [(k, v[0], v[1]) for k, v in self.sett.GRIDspec_tiers.items()]
            gridtiers_df = pd.DataFrame(data, columns=self.sett.GRIDspec_colnames)

            checkpoint_to_logfile('gridprem: start loop Map_infosource_egid', self.sett.log_name, 1, self.sett.show_debug_prints)
            egid_list, info_source_list, inst_TF_list = [], [], []
            for k,v in topo.items():
                egid_list.append(k)
                if v.get('pv_inst', {}).get('inst_TF'):
                    info_source_list.append(v.get('pv_inst').get('info_source'))
                    inst_TF_list.append(v.get('pv_inst').get('inst_TF'))
                else: 
                    info_source_list.append('')
                    inst_TF_list.append(False)
            Map_infosource_egid = pd.DataFrame({'EGID': egid_list, 'info_source': info_source_list, 'inst_TF': inst_TF_list}, index=egid_list)

            checkpoint_to_logfile('gridprem: end loop Map_infosource_egid', self.sett.log_name, 1, self.sett.show_debug_prints)


            # import topo_time_subdfs -----------------------------------------------------
            # topo_subdf_paths = glob.glob(f'{self.sett.pvalloc_path}/topo_time_subdf/*.parquet')
            checkpoint_to_logfile('gridprem: start read subdf', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None

            topo_subdf_paths = glob.glob(f'{subdir_path}/topo_subdf_*.parquet')
            agg_subinst_df_list = []
            # no_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == False]
            # wi_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == True]

            i, path = 0, topo_subdf_paths[0]
            for i, path in enumerate(topo_subdf_paths):
                checkpoint_to_logfile('gridprem > subdf: start read subdf', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None
                subdf = pl.read_parquet(path)           # subdf = pd.read_parquet(path)
                Map_infosource_egid = pl.from_pandas(Map_infosource_egid) if isinstance(Map_infosource_egid, pd.DataFrame) else Map_infosource_egid
                
                checkpoint_to_logfile('gridprem > subdf: end read subdf', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None

                subdf_updated = subdf.clone()                                       # subdf_updated = copy.deepcopy(subdf)
                subdf_updated = subdf_updated.drop(['info_source', 'inst_TF'])      # subdf_updated.drop(columns=['info_source', 'inst_TF'], inplace=True)

                checkpoint_to_logfile('gridprem > subdf: start pandas.merge subdf w Map_infosource_egid', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None
                subdf_updated = subdf_updated.join(Map_infosource_egid[['EGID', 'info_source', 'inst_TF']], on='EGID', how='left')          # subdf_updated = subdf_updated.join(Map_infosource_egid.select(['EGID', 'info_source', 'inst_TF']), on='EGID', how='left')        # subdf_updated = subdf_updated.merge(Map_infosource_egid[['EGID', 'info_source', 'inst_TF']], how='left', on='EGID')
                checkpoint_to_logfile('gridprem > subdf: end pandas.merge subdf w Map_infosource_egid', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None

                # Only consider production for houses that have built a pv installation and substract selfconsumption from the production
                subinst = subdf_updated.filter(pl.col('inst_TF') == True)       # subinst = copy.deepcopy(subdf_updated.loc[subdf_updated['inst_TF']==True])
                checkpoint_to_logfile('gridprem > subdf: start calc + update feedin_kw', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None   
                # pvprod_kW, demand_kW = subinst['pvprod_kW'].to_numpy(), subinst['demand_kW'].to_numpy()
                # selfconsum_kW = np.minimum(pvprod_kW, demand_kW) * self.sett.TECspec_self_consumption_ifapplicable
                # netdemand_kW = demand_kW - selfconsum_kW
                # netfeedin_kW = pvprod_kW - selfconsum_kW
                # subinst = subinst.with_column(pl.Series('feedin_kW', netfeedin_kW))   # subinst['feedin_kW'] = netfeedin_kW
                selfconsum_expr = pl.min_horizontal([pl.col("pvprod_kW"), pl.col("demand_kW")]) * self.sett.TECspec_self_consumption_ifapplicable

                subinst = subinst.with_columns([
                selfconsum_expr.alias("selfconsum_kW"),
                (pl.col("pvprod_kW") - selfconsum_expr).alias("feedin_kW"),
                (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
                ])
                checkpoint_to_logfile('gridprem > subdf: end calc + update feedin_kw', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None   
                # NOTE: attempt for a more elaborate way to handle already installed installations
                if False:
                    pv = pd.read_parquet(f'{subdir_path}/pv.parquet')
                    pv['pvsource'] = 'pv_df'
                    pv['pvid'] = pv['xtf_id']

                    # if 'pv_df' in subinst['pvsource'].unique():
                    # TotalPower = pv.loc[pv['xtf_id'].isin(subinst.loc[subinst['EGID'] == egid, 'pvid']), 'TotalPower'].sum()

                    subinst = subinst.sort_values(by = 'STROMERTRAG', ascending=False)
                    subinst['pvprod_kW'] = 0
                    
                    # t_steps = subinst['t'].unique()
                    for t in subinst['t'].unique():
                        timestep_df = subinst.loc[subinst['t'] == t]
                        total_stromertrag = timestep_df['STROMERTRAG'].sum()

                        for idx, row in timestep_df.iterrows():
                            share = row['STROMERTRAG'] / total_stromertrag
                            # subinst.loc[idx, 'pvprod_kW'] = share * TotalPower
                            print(share)

                subinst = pl.from_pandas(subinst) if isinstance(subinst, pd.DataFrame) else subinst
                # agg_subinst = subinst.groupby(['grid_node', 't']).agg({'feedin_kW': 'sum', 'pvprod_kW':'sum'}).reset_index()
                agg_subinst = subinst.group_by(["grid_node", "t"]).agg([
                    pl.col('feedin_kW').sum().alias('feedin_kW'),
                    pl.col('pvprod_kW').sum().alias('pvprod_kW')
                ])
                
                del subinst
                agg_subinst_df_list.append(agg_subinst)
            

            # build gridnode_df -----------------------------------------------------
            checkpoint_to_logfile('gridprem: start merge with gridnode_df + np.where', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None   

            # gridnode_df = pd.concat(agg_subinst_df_list)
            gridnode_df = pl.concat(agg_subinst_df_list)
            # # groupby df again because grid nodes will be spreach accross multiple tranches
            # gridnode_df = gridnode_df.groupby(['grid_node', 't']).agg({'feedin_kW': 'sum', 'pvprod_kW':'sum'}).reset_index() 
            gridnode_df = gridnode_df.group_by(['grid_node', 't']).agg([
                pl.col('feedin_kW').sum().alias('feedin_kW'),
                pl.col('pvprod_kW').sum().alias('pvprod_kW')
            ])


            # attach node thresholds 
            # gridnode_df = gridnode_df.merge(dsonodes_df[['grid_node', 'kVA_threshold']], how='left', on='grid_node')
            # gridnode_df['kW_threshold'] = gridnode_df['kVA_threshold'] * self.sett.GRIDspec_perf_factor_1kVA_to_XkW
            
            # gridnode_df['feedin_kW_taken'] = np.where(gridnode_df['feedin_kW'] > gridnode_df['kW_threshold'], gridnode_df['kW_threshold'], gridnode_df['feedin_kW'])
            # gridnode_df['feedin_kW_loss'] =  np.where(gridnode_df['feedin_kW'] > gridnode_df['kW_threshold'], gridnode_df['feedin_kW'] - gridnode_df['kW_threshold'], 0)
            gridnode_df = gridnode_df.join(dsonodes_df[['grid_node', 'kVA_threshold']], on='grid_node', how='left')
            gridnode_df = gridnode_df.with_columns((pl.col("kVA_threshold") * self.sett.GRIDspec_perf_factor_1kVA_to_XkW).alias("kW_threshold"))
            gridnode_df = gridnode_df.with_columns([
                pl.when(pl.col("feedin_kW") > pl.col("kW_threshold"))
                .then(pl.col("kW_threshold"))
                .otherwise(pl.col("feedin_kW"))
                .alias("feedin_kW_taken"),

                pl.when(pl.col("feedin_kW") > pl.col("kW_threshold"))
                .then(pl.col("feedin_kW") - pl.col("kW_threshold"))
                .otherwise(0)
                .alias("feedin_kW_loss")
            ])

            checkpoint_to_logfile('gridprem: end merge with gridnode_df + np.where', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None   


            # update gridprem_ts -----------------------------------------------------
            checkpoint_to_logfile('gridprem: start update gridprem_ts', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None   
            gridnode_df = gridnode_df.sort("feedin_kW_taken", descending=True)            # gridnode_df_for_prem = gridnode_df.groupby(['grid_node','kW_threshold', 't']).agg({'feedin_kW_taken': 'sum'}).reset_index().copy()
            # gridprem_ts = gridprem_ts.merge(gridnode_df_for_prem[['grid_node', 't', 'kW_threshold', 'feedin_kW_taken']], how='left', on=['grid_node', 't'])
            # gridprem_ts['feedin_kW_taken'] = gridprem_ts['feedin_kW_taken'].replace(np.nan, 0)
            gridnode_df_for_prem = gridnode_df.group_by(['grid_node', 'kW_threshold', 't']).agg(
                pl.col('feedin_kW_taken').sum().alias('feedin_kW_taken')
            )            
            gridprem_ts = gridprem_ts.join(
                gridnode_df_for_prem.select(['grid_node', 't', 'kW_threshold', 'feedin_kW_taken']),
                on=['grid_node', 't'],
                how='left'
            )
            gridprem_ts = gridprem_ts.with_columns(
                pl.col('feedin_kW_taken').fill_null(0)
            )

            # gridtiers_df['kW_threshold'] = gridtiers_df['kVA_threshold'] / gridtiers_power_factor
            # conditions, choices = [], []
            # for i in range(len(gridtiers_df)):
            #     i_adj = len(gridtiers_df) - i -1 # order needs to be reversed, because otherwise first condition is always met and disregards the higher tiers
            #     conditions.append((gridprem_ts['feedin_kW_taken'] / gridprem_ts['kW_threshold'])  > gridtiers_df.loc[i_adj, 'used_node_capa_rate'])
            #     choices.append(gridtiers_df.loc[i_adj, 'gridprem_Rp_kWh'])
            # gridprem_ts['prem_Rp_kWh'] = np.select(conditions, choices, default=gridprem_ts['prem_Rp_kWh'])
            # gridprem_ts.drop(columns=['feedin_kW_taken', 'kW_threshold'], inplace=True)
            expr = pl.col('prem_Rp_kWh')
            for i in reversed(range(len(gridtiers_df))):
                capa_rate = gridtiers_df.loc[i, 'used_node_capa_rate']
                price = gridtiers_df.loc[i, 'gridprem_Rp_kWh']
                
                expr = pl.when(pl.col('feedin_kW_taken') / pl.col('kW_threshold') > capa_rate)\
                        .then(price)\
                        .otherwise(expr)
            gridprem_ts = gridprem_ts.with_columns(expr.alias('prem_Rp_kWh'))
            gridprem_ts = gridprem_ts.drop(['feedin_kW_taken', 'kW_threshold'])

            checkpoint_to_logfile('gridprem: end update gridprem_ts', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None   


            # EXPORT -----------------------------------------------------
            gridnode_df.write_parquet(f'{subdir_path}/gridnode_df.parquet')      # gridnode_df.to_parquet(f'{subdir_path}/gridnode_df.parquet')
            gridprem_ts.write_parquet(f'{subdir_path}/gridprem_ts.parquet')      # gridprem_ts.to_parquet(f'{subdir_path}/gridprem_ts.parquet')
            if self.sett.export_csvs:                                            # if self.sett.export_csvs:
                gridnode_df.write_csv(f'{subdir_path}/gridnode_df.csv')               # gridnode_df.to_csv(f'{subdir_path}/gridnode_df.csv', index=False)
                gridprem_ts.write_csv(f'{subdir_path}/gridprem_ts.csv')               # gridprem_ts.to_csv(f'{subdir_path}/gridprem_ts.csv', index=False)


            # export by Month -----------------------------------------------------
            if self.sett.MCspec_keep_files_month_iter_TF:
                if i_m < self.sett.MCspec_keep_files_month_iter_max:
                    # gridprem_node_by_M_path = f'{self.sett.pvalloc_path}/pred_gridprem_node_by_M'
                    gridprem_node_by_M_path = f'{subdir_path}/pred_gridprem_node_by_M'
                    if not os.path.exists(gridprem_node_by_M_path):
                        os.makedirs(gridprem_node_by_M_path)

                        gridnode_df.write_parquet(f'{gridprem_node_by_M_path}/gridnode_df_{m}.parquet')     # gridnode_df.to_parquet(f'{gridprem_node_by_M_path}/gridnode_df_{m}.parquet')
                        gridprem_ts.write_parquet(f'{gridprem_node_by_M_path}/gridprem_ts_{m}.parquet')     # gridprem_ts.to_parquet(f'{gridprem_node_by_M_path}/gridprem_ts_{m}.parquet')

                    if self.sett.export_csvs:
                        gridnode_df.write_csv(f'{gridprem_node_by_M_path}/gridnode_df_{m}.csv')    # gridnode_df.to_csv(f'{gridprem_node_by_M_path}/gridnode_df_{m}.csv', index=False)
                        gridprem_ts.write_csv(f'{gridprem_node_by_M_path}/gridprem_ts_{m}.csv')    # gridprem_ts.to_csv(f'{gridprem_node_by_M_path}/gridprem_ts_{m}.csv', index=False)
                    if i_m < 5:
                        gridnode_df.write_csv(f'{gridprem_node_by_M_path}/gridnode_df_{m}.csv')   # gridnode_df.to_csv(f'{gridprem_node_by_M_path}/gridnode_df_{m}.csv', index=False)
                        gridprem_ts.write_csv(f'{gridprem_node_by_M_path}/gridprem_ts_{m}.csv')   # gridprem_ts.to_csv(f'{gridprem_node_by_M_path}/gridprem_ts_{m}.csv', index=False)

            checkpoint_to_logfile('exported gridprem_ts and gridnode_df', self.sett.log_name, 1) if i_m < 3 else None
            

        def algo_update_gridprem(self, subdir_path: str, i_m: int, m): 
    
            # setup -----------------------------------------------------
            print_to_logfile('run function: update_gridprem', self.sett.log_name)
            gridtiers_power_factor  = self.sett.GRIDspec_power_factor

            # import  -----------------------------------------------------
            topo = json.load(open(f'{subdir_path}/topo_egid.json', 'r'))
            dsonodes_df = pd.read_parquet(f'{subdir_path}/dsonodes_df.parquet')
            gridprem_ts = pd.read_parquet(f'{subdir_path}/gridprem_ts.parquet')

            data = [(k, v[0], v[1]) for k, v in self.sett.GRIDspec_tiers.items()]
            gridtiers_df = pd.DataFrame(data, columns=self.sett.GRIDspec_colnames)

            checkpoint_to_logfile('**DEBUGGIG** > START LOOP through topo_egid', self.sett.log_name, 1, self.sett.show_debug_prints)
            egid_list, info_source_list, inst_TF_list = [], [], []
            for k,v in topo.items():
                egid_list.append(k)
                if v.get('pv_inst', {}).get('inst_TF'):
                    info_source_list.append(v.get('pv_inst').get('info_source'))
                    inst_TF_list.append(v.get('pv_inst').get('inst_TF'))
                else: 
                    info_source_list.append('')
                    inst_TF_list.append(False)
            Map_infosource_egid = pd.DataFrame({'EGID': egid_list, 'info_source': info_source_list, 'inst_TF': inst_TF_list}, index=egid_list)

            checkpoint_to_logfile('**DEBUGGIG** > end loop through topo_egid', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None


            # import topo_time_subdfs -----------------------------------------------------
            # topo_subdf_paths = glob.glob(f'{self.sett.pvalloc_path}/topo_time_subdf/*.parquet')
            checkpoint_to_logfile('**DEBUGGIG** > start loop through subdfs', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None

            topo_subdf_paths = glob.glob(f'{subdir_path}/topo_subdf_*.parquet')
            agg_subinst_df_list = []
            # no_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == False]
            # wi_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == True]

            i, path = 0, topo_subdf_paths[0]
            for i, path in enumerate(topo_subdf_paths):
                checkpoint_to_logfile('**DEBUGGIG** \t> start read subdfs', self.sett.log_name, 2) if i < 2 else None
                subdf = pd.read_parquet(path)
                checkpoint_to_logfile('**DEBUGGIG** \t> end read subdfs', self.sett.log_name, 2) if i < 2 else None

                subdf_updated = copy.deepcopy(subdf)
                subdf_updated.drop(columns=['info_source', 'inst_TF'], inplace=True)

                checkpoint_to_logfile('**DEBUGGIG** \t> start Map_infosource_egid', self.sett.log_name, 1, self.sett.show_debug_prints) if i < 2 else None
                subdf_updated = subdf_updated.merge(Map_infosource_egid[['EGID', 'info_source', 'inst_TF']], how='left', on='EGID')
                checkpoint_to_logfile('**DEBUGGIG** \t> end Map_infosource_egid', self.sett.log_name, 1, self.sett.show_debug_prints) if i < 2 else None
                # updated_instTF_srs, update_infosource_srs = subdf_updated['inst_TF'].fillna(subdf['inst_TF']), subdf_updated['info_source'].fillna(subdf['info_source'])
                # subdf['inst_TF'], subdf['info_source'] = updated_instTF_srs.infer_objects(copy=False), update_infosource_srs.infer_objects(copy=False)

                # Only consider production for houses that have built a pv installation and substract selfconsumption from the production
                subinst = copy.deepcopy(subdf_updated.loc[subdf_updated['inst_TF']==True])
                checkpoint_to_logfile('**DEBUGGIG** \t> pvprod_kw_to_numpy', self.sett.log_name, 2) if i < 2 else None
                pvprod_kW, demand_kW = subinst['pvprod_kW'].to_numpy(), subinst['demand_kW'].to_numpy()
                selfconsum_kW = np.minimum(pvprod_kW, demand_kW) * self.sett.TECspec_self_consumption_ifapplicable
                netdemand_kW = demand_kW - selfconsum_kW
                netfeedin_kW = pvprod_kW - selfconsum_kW

                subinst['feedin_kW'] = netfeedin_kW
                
                checkpoint_to_logfile('**DEBUGGIG** > end pvprod_kw_to_numpy', self.sett.log_name, 2, self.sett.show_debug_prints) if i < 2 else None
                # NOTE: attempt for a more elaborate way to handle already installed installations
                if False:
                    pv = pd.read_parquet(f'{subdir_path}/pv.parquet')
                    pv['pvsource'] = 'pv_df'
                    pv['pvid'] = pv['xtf_id']

                    # if 'pv_df' in subinst['pvsource'].unique():
                    # TotalPower = pv.loc[pv['xtf_id'].isin(subinst.loc[subinst['EGID'] == egid, 'pvid']), 'TotalPower'].sum()

                    subinst = subinst.sort_values(by = 'STROMERTRAG', ascending=False)
                    subinst['pvprod_kW'] = 0
                    
                    # t_steps = subinst['t'].unique()
                    for t in subinst['t'].unique():
                        timestep_df = subinst.loc[subinst['t'] == t]
                        total_stromertrag = timestep_df['STROMERTRAG'].sum()

                        for idx, row in timestep_df.iterrows():
                            share = row['STROMERTRAG'] / total_stromertrag
                            # subinst.loc[idx, 'pvprod_kW'] = share * TotalPower
                            print(share)

                agg_subinst = subinst.groupby(['grid_node', 't']).agg({'feedin_kW': 'sum', 'pvprod_kW':'sum'}).reset_index()
                del subinst
                agg_subinst_df_list.append(agg_subinst)
            
            checkpoint_to_logfile('**DEBUGGIG** > end loop through subdfs', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None


            # build gridnode_df -----------------------------------------------------
            gridnode_df = pd.concat(agg_subinst_df_list)
            # groupby df again because grid nodes will be spreach accross multiple tranches
            gridnode_df = gridnode_df.groupby(['grid_node', 't']).agg({'feedin_kW': 'sum', 'pvprod_kW':'sum'}).reset_index() 
            

            # attach node thresholds 
            gridnode_df = gridnode_df.merge(dsonodes_df[['grid_node', 'kVA_threshold']], how='left', on='grid_node')
            gridnode_df['kW_threshold'] = gridnode_df['kVA_threshold'] * self.sett.GRIDspec_perf_factor_1kVA_to_XkW

            gridnode_df['feedin_kW_taken'] = np.where(gridnode_df['feedin_kW'] > gridnode_df['kW_threshold'], gridnode_df['kW_threshold'], gridnode_df['feedin_kW'])
            gridnode_df['feedin_kW_loss'] =  np.where(gridnode_df['feedin_kW'] > gridnode_df['kW_threshold'], gridnode_df['feedin_kW'] - gridnode_df['kW_threshold'], 0)

            checkpoint_to_logfile('**DEBUGGIG** > end merge + npwhere subdfs', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None


            # update gridprem_ts -----------------------------------------------------
            gridnode_df.sort_values(by=['feedin_kW_taken'], ascending=False)
            gridnode_df_for_prem = gridnode_df.groupby(['grid_node','kW_threshold', 't']).agg({'feedin_kW_taken': 'sum'}).reset_index().copy()
            gridprem_ts = gridprem_ts.merge(gridnode_df_for_prem[['grid_node', 't', 'kW_threshold', 'feedin_kW_taken']], how='left', on=['grid_node', 't'])
            gridprem_ts['feedin_kW_taken'] = gridprem_ts['feedin_kW_taken'].replace(np.nan, 0)
            gridprem_ts.sort_values(by=['feedin_kW_taken'], ascending=False)

            # gridtiers_df['kW_threshold'] = gridtiers_df['kVA_threshold'] / gridtiers_power_factor
            conditions, choices = [], []
            for i in range(len(gridtiers_df)):
                i_adj = len(gridtiers_df) - i -1 # order needs to be reversed, because otherwise first condition is always met and disregards the higher tiers
                conditions.append((gridprem_ts['feedin_kW_taken'] / gridprem_ts['kW_threshold'])  > gridtiers_df.loc[i_adj, 'used_node_capa_rate'])
                choices.append(gridtiers_df.loc[i_adj, 'gridprem_Rp_kWh'])
            gridprem_ts['prem_Rp_kWh'] = np.select(conditions, choices, default=gridprem_ts['prem_Rp_kWh'])
            gridprem_ts.drop(columns=['feedin_kW_taken', 'kW_threshold'], inplace=True)

            checkpoint_to_logfile('**DEBUGGIG** > end update gridprem_ts', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None


            # EXPORT -----------------------------------------------------
            gridnode_df.to_parquet(f'{subdir_path}/gridnode_df.parquet')
            gridprem_ts.to_parquet(f'{subdir_path}/gridprem_ts.parquet')
            if self.sett.export_csvs:
                gridnode_df.to_csv(f'{subdir_path}/gridnode_df.csv', index=False)
                gridprem_ts.to_csv(f'{subdir_path}/gridprem_ts.csv', index=False)


            # export by Month -----------------------------------------------------
            if self.sett.MCspec_keep_files_month_iter_TF:
                if i_m < self.sett.MCspec_keep_files_month_iter_max:
                    # gridprem_node_by_M_path = f'{self.sett.pvalloc_path}/pred_gridprem_node_by_M'
                    gridprem_node_by_M_path = f'{subdir_path}/pred_gridprem_node_by_M'
                    if not os.path.exists(gridprem_node_by_M_path):
                        os.makedirs(gridprem_node_by_M_path)

                    gridnode_df.to_parquet(f'{gridprem_node_by_M_path}/gridnode_df_{m}.parquet')
                    gridprem_ts.to_parquet(f'{gridprem_node_by_M_path}/gridprem_ts_{m}.parquet')

                    if self.sett.export_csvs:
                        gridnode_df.to_csv(f'{gridprem_node_by_M_path}/gridnode_df_{m}.csv', index=False)
                        gridprem_ts.to_csv(f'{gridprem_node_by_M_path}/gridprem_ts_{m}.csv', index=False)
                    if i_m < 5:
                        gridnode_df.to_csv(f'{gridprem_node_by_M_path}/gridnode_df_{m}.csv', index=False)
                        gridprem_ts.to_csv(f'{gridprem_node_by_M_path}/gridprem_ts_{m}.csv', index=False)

            checkpoint_to_logfile('exported gridprem_ts and gridnode_df', self.sett.log_name, 1) if i_m < 3 else None


        def algo_update_npv_df_POLARS(self, subdir_path: str, i_m: int, m):

            # setup -----------------------------------------------------
            print_to_logfile('run function: update_npv_df', self.sett.log_name)         

            # import -----------------------------------------------------
            gridprem_ts = pl.read_parquet(f'{subdir_path}/gridprem_ts.parquet')    
            topo = json.load(open(f'{subdir_path}/topo_egid.json', 'r'))


            # import topo_time_subdfs -----------------------------------------------------
            topo_subdf_paths = glob.glob(f'{subdir_path}/topo_subdf_*.parquet') 
            no_pv_egid = [k for k, v in topo.items() if not v.get('pv_inst', {}).get('inst_TF') ]
            agg_npv_df_list = []

            j = 0
            i, path = j, topo_subdf_paths[j]
            for i, path in enumerate(topo_subdf_paths):
                print_topo_subdf_TF = len(topo_subdf_paths) > 5 and i <5  # i% (len(topo_subdf_paths) //3 ) == 0:
                if print_topo_subdf_TF:
                    print_to_logfile(f'updated npv (tranche {i+1}/{len(topo_subdf_paths)})', self.sett.log_name)
                subdf_t0 = pl.read_parquet(path) # subdf_t0 = pd.read_parquet(path)

                # drop egids with pv installations
                subdf = subdf_t0.filter(pl.col("EGID").is_in(no_pv_egid))   

                if subdf.shape[0] > 0:

                    # merge gridprem_ts
                    checkpoint_to_logfile('npv > subdf: start merge subdf w gridprem_ts', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None
                    subdf = subdf.join(gridprem_ts[['t', 'grid_node', 'prem_Rp_kWh']], on=['t', 'grid_node'], how='left')  
                    checkpoint_to_logfile('npv > subdf: start merge subdf w gridprem_ts', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None


                    # compute selfconsumption + netdemand ----------------------------------------------
                    checkpoint_to_logfile('npv > subdf: start calc selfconsumption + netdemand', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None
                    prod_demand_fact = self.sett.ALGOspec_tweak_gridnode_df_prod_demand_fact
                    selfcons_fact = self.sett.TECspec_self_consumption_ifapplicable
                    exclude_elec_demand = self.sett.ALGOspec_tweak_npv_excl_elec_demand

                    subdf = subdf.with_columns([
                        (pl.col("demand_kW") * prod_demand_fact).alias("demand_kW"),
                        (pl.min_horizontal([pl.col("pvprod_kW"), pl.col("demand_kW") * prod_demand_fact]) * selfcons_fact).alias("selfconsum_kW"),
                    ])
                    subdf = subdf.with_columns([
                        (pl.col("demand_kW") - pl.col("selfconsum_kW")).alias("netdemand_kW"),
                        (pl.col("pvprod_kW") - pl.col("selfconsum_kW")).alias("netfeedin_kW"),
                    ])
                    subdf = subdf.with_columns([
                        ((pl.col("netfeedin_kW") * pl.col("pv_tarif_Rp_kWh")) / 100 + (pl.col("selfconsum_kW") * pl.col("elecpri_Rp_kWh")) / 100).alias("econ_inc_chf")
                    ])
                    
                    if not exclude_elec_demand:
                        subdf = subdf.with_columns([
                            ((pl.col("netfeedin_kW") * pl.col("prem_Rp_kWh")) / 100 +
                            (pl.col("netdemand_kW") * pl.col("elecpri_Rp_kWh")) / 100).alias("econ_spend_chf")
                        ])
                    else:
                        subdf = subdf.with_columns([
                            ((pl.col("netfeedin_kW") * pl.col("prem_Rp_kWh")) / 100).alias("econ_spend_chf")
                        ])

                    checkpoint_to_logfile('npv > subdf: end calc selfconsumption + netdemand', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None


                    checkpoint_to_logfile('npv > subdf: start groupgy agg_subdf', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None
                    
                    group_cols = self.sett.ALGOspec_npv_update_groupby_cols_topo_aggdf
                    agg_map = self.sett.ALGOspec_npv_update_agg_cols_topo_aggdf
                    agg_exprs = [
                        getattr(pl.col(col), agg_func)().alias(f"{col}")
                        for col, agg_func in agg_map.items()
                    ]
                    agg_subdf = subdf.group_by(group_cols).agg(agg_exprs)


                    checkpoint_to_logfile('npv > subdf: end groupgy agg_subdf', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None

                    # create combinations ----------------------------------------------
                    checkpoint_to_logfile('npv > subdf: end groupgy agg_subdf', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None

                    agg_sub_grouped = agg_subdf.group_by(['EGID'])

                    combo_rows = []

                    for egid , egid_subdf in agg_sub_grouped:
                        df_uids = egid_subdf['df_uid'].unique().to_list()

                        for r in range(1, len(df_uids)+1):
                            for combo in itertools.combinations(df_uids,r):
                                combo_str = '_'.join([str(c) for c in combo])

                                combo_df = egid_subdf.filter(pl.col('df_uid').is_in(combo))
                                row = {
                                    "EGID": egid,
                                    "df_uid_combo": combo_str,
                                    "bfs": combo_df[0, "bfs"],
                                    "gklas": combo_df[0, "gklas"],
                                    "demandtype": combo_df[0, "demandtype"],
                                    "grid_node": combo_df[0, "grid_node"],
                                    "inst_TF": combo_df[0, "inst_TF"],
                                    "info_source": combo_df[0, "info_source"],
                                    "pvid": combo_df[0, "pvid"],
                                    "pv_tarif_Rp_kWh": combo_df[0, "pv_tarif_Rp_kWh"],
                                    "elecpri_Rp_kWh": combo_df[0, "elecpri_Rp_kWh"],
                                    "demand_kW": combo_df[0, "demand_kW"],
                                    "AUSRICHTUNG": combo_df[0, "AUSRICHTUNG"],
                                    "NEIGUNG": combo_df[0, "NEIGUNG"],
                                    "FLAECHE": combo_df["FLAECHE"].sum(),
                                    "STROMERTRAG": combo_df["STROMERTRAG"].sum(),
                                    "FLAECH_angletilt": combo_df["FLAECH_angletilt"].sum(),
                                    "pvprod_kW": combo_df["pvprod_kW"].sum(),
                                    "selfconsum_kW": combo_df["selfconsum_kW"].sum(),
                                    "netdemand_kW": combo_df["netdemand_kW"].sum(),
                                    "netfeedin_kW": combo_df["netfeedin_kW"].sum(),
                                    "econ_inc_chf": combo_df["econ_inc_chf"].sum(),
                                    "econ_spend_chf": combo_df["econ_spend_chf"].sum()
                                }
                                combo_rows.append(row)
                    aggsubdf_combo = pl.DataFrame(combo_rows)


                    checkpoint_to_logfile('npv > subdf: end groupgy agg_subdf', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None

                # if (i <3) and (i_m <3): 
                #     checkpoint_to_logfile(f'\t created df_uid combos for {agg_subdf["EGID"].nunique()} EGIDs', self.sett.log_name, 1, self.sett.show_debug_prints)

                

                # NPV calculation -----------------------------------------------------
                estim_instcost_chfpkW, estim_instcost_chftotal = self.initial_sml_get_instcost_interpolate_function()
                estim_instcost_chftotal(pd.Series([10, 20, 30, 40, 50, 60, 70]))



                # correct cost estimation by a factor based on insights from pvprod_correction.py
                aggsubdf_combo = aggsubdf_combo.with_columns([
                    (pl.col("FLAECHE") * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available).alias("roof_area_for_cost"),
                ])

                # aggsubdf_combo = aggsubdf_combo.to_pandas()
                # kwp_peak_array = aggsubdf_combo['FLAECHE'] * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available / self.sett.TECspec_estim_pvinst_cost_correctionfactor
                # aggsubdf_combo['estim_pvinstcost_chf'] = estim_instcost_chftotal(kwp_peak_array) 
                # aggsubdf_combo = aggsubdf_combo.from_pandas()
                estim_instcost_chftotal_srs = estim_instcost_chftotal(aggsubdf_combo['roof_area_for_cost'] )
                aggsubdf_combo = aggsubdf_combo.with_columns(
                    pl.Series("estim_pvinstcost_chf", estim_instcost_chftotal_srs)
                )


                checkpoint_to_logfile('npv > subdf: start calc npv', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None

                cashflow_srs =  aggsubdf_combo['econ_inc_chf'] - aggsubdf_combo['econ_spend_chf']
                cashflow_disc_list = []
                for j in range(1, self.sett.TECspec_invst_maturity+1):
                    cashflow_disc_list.append(cashflow_srs / (1+self.sett.TECspec_interest_rate)**j)
                cashflow_disc_srs = sum(cashflow_disc_list)
                
                npv_srs = (-aggsubdf_combo['estim_pvinstcost_chf']) + cashflow_disc_srs

                aggsubdf_combo = aggsubdf_combo.with_columns(
                    pl.Series("npv_chf", npv_srs)
                )

                checkpoint_to_logfile('npv > subdf: end calc npv', self.sett.log_name, 1, self.sett.show_debug_prints) if i_m < 3 else None

                agg_npv_df_list.append(aggsubdf_combo)

            agg_npv_df = pl.concat(agg_npv_df_list)
            npv_df = agg_npv_df.clone()


            # export npv_df -----------------------------------------------------
            npv_df.write_parquet(f'{subdir_path}/npv_df.parquet')
            # if self.sett.export_csvs:
            #     npv_df.write_csv(f'{subdir_path}/npv_df.csv', index=False)
                

            # export by Month -----------------------------------------------------
            if self.sett.MCspec_keep_files_month_iter_TF:
                if i_m < self.sett.MCspec_keep_files_month_iter_max:
                    pred_npv_inst_by_M_path = f'{subdir_path}/pred_npv_inst_by_M'
                    if not os.path.exists(pred_npv_inst_by_M_path):
                        os.makedirs(pred_npv_inst_by_M_path)

                    npv_df.write_parquet(f'{pred_npv_inst_by_M_path}/npv_df_{m}.parquet')
                    # if self.sett.export_csvs:
                    #     npv_df.write_csv(f'{pred_npv_inst_by_M_path}/npv_df_{m}.csv', separator=',' )                    
                    # if i_m < 5:
                    #     npv_df.write_csv(f'{pred_npv_inst_by_M_path}/npv_df_{m}.csv', index=False)

            checkpoint_to_logfile('exported npv_df', self.sett.log_name, 1)
                
            return npv_df




        def algo_update_npv_df(self, subdir_path: str, i_m: int, m):

            # setup -----------------------------------------------------
            print_to_logfile('run function: update_npv_df', self.sett.log_name)         

            # import -----------------------------------------------------
            gridprem_ts = pd.read_parquet(f'{subdir_path}/gridprem_ts.parquet')
            topo = json.load(open(f'{subdir_path}/topo_egid.json', 'r'))


            # import topo_time_subdfs -----------------------------------------------------
            topo_subdf_paths = glob.glob(f'{subdir_path}/topo_subdf_*.parquet') 
            no_pv_egid = [k for k, v in topo.items() if not v.get('pv_inst', {}).get('inst_TF') ]
            agg_npv_df_list = []

            j = 0
            i, path = j, topo_subdf_paths[j]
            for i, path in enumerate(topo_subdf_paths):
                print_topo_subdf_TF = len(topo_subdf_paths) > 5 and i <5  # i% (len(topo_subdf_paths) //3 ) == 0:
                if print_topo_subdf_TF:
                    print_to_logfile(f'updated npv (tranche {i+1}/{len(topo_subdf_paths)})', self.sett.log_name)
                subdf_t0 = pd.read_parquet(path)

                # drop egids with pv installations
                subdf = copy.deepcopy(subdf_t0[subdf_t0['EGID'].isin(no_pv_egid)])

                if not subdf.empty:

                    # merge gridprem_ts
                    subdf = subdf.merge(gridprem_ts[['t', 'prem_Rp_kWh', 'grid_node']], how='left', on=['t', 'grid_node']) 

                    # compute selfconsumption + netdemand ----------------------------------------------
                    subdf_array = subdf[['pvprod_kW', 'demand_kW', 'pv_tarif_Rp_kWh', 'elecpri_Rp_kWh', 'prem_Rp_kWh']].to_numpy()
                    pvprod_kW, demand_kW, pv_tarif_Rp_kWh, elecpri_Rp_kWh, prem_Rp_kWh = subdf_array[:,0], subdf_array[:,1], subdf_array[:,2], subdf_array[:,3], subdf_array[:,4]

                    demand_kW = demand_kW * self.sett.ALGOspec_tweak_gridnode_df_prod_demand_fact
                    selfconsum_kW = np.minimum(pvprod_kW, demand_kW) * self.sett.TECspec_self_consumption_ifapplicable
                    netdemand_kW = demand_kW - selfconsum_kW
                    netfeedin_kW = pvprod_kW - selfconsum_kW

                    econ_inc_chf = ((netfeedin_kW * pv_tarif_Rp_kWh) /100) + ((selfconsum_kW * elecpri_Rp_kWh) /100)
                    if not self.sett.ALGOspec_tweak_npv_excl_elec_demand:
                        econ_spend_chf = ((netfeedin_kW * prem_Rp_kWh) / 100)  + ((netdemand_kW * elecpri_Rp_kWh) /100)
                    else:
                        econ_spend_chf = ((netfeedin_kW * prem_Rp_kWh) / 100)

                    subdf['demand_kW'], subdf['pvprod_kW'], subdf['selfconsum_kW'], subdf['netdemand_kW'], subdf['netfeedin_kW'], subdf['econ_inc_chf'], subdf['econ_spend_chf'] = demand_kW, pvprod_kW, selfconsum_kW, netdemand_kW, netfeedin_kW, econ_inc_chf, econ_spend_chf
                    

                    if (i <3) and (i_m <3): 
                        checkpoint_to_logfile('\t end compute econ factors', self.sett.log_name, 1, self.sett.show_debug_prints) #for subdf EGID {path.split("topo_subdf_")[1].split(".parquet")[0]}', self.sett.log_name, 1, self.sett.show_debug_prints)

                    agg_subdf = subdf.groupby(
                                        self.sett.ALGOspec_npv_update_groupby_cols_topo_aggdf).agg(
                                        self.sett.ALGOspec_npv_update_agg_cols_topo_aggdf).reset_index()
                        
                    
                    if (i <3) and (i_m <3): 
                        checkpoint_to_logfile('\t groupby subdf to agg_subdf', self.sett.log_name, 1, self.sett.show_debug_prints)


                    # create combinations ----------------------------------------------
                    aggsub_npry = np.array(agg_subdf)

                    egid_list, combo_df_uid_list, df_uid_list, bfs_list, gklas_list, demandtype_list, grid_node_list = [], [], [], [], [], [], []
                    inst_list, info_source_list, pvid_list, pv_tarif_Rp_kWh_list = [], [], [], []
                    flaeche_list, stromertrag_list, ausrichtung_list, neigung_list, elecpri_Rp_kWh_list = [], [], [], [], []
                
                    flaech_angletilt_list = []
                    demand_list, pvprod_list, selfconsum_list, netdemand_list, netfeedin_list = [], [], [], [], []
                    econ_inc_chf_list, econ_spend_chf_list = [], []

                    egid = agg_subdf['EGID'].unique()[0]
                    for i, egid in enumerate(agg_subdf['EGID'].unique()):

                        mask_egid_subdf = np.isin(aggsub_npry[:,agg_subdf.columns.get_loc('EGID')], egid)
                        df_uids  = list(aggsub_npry[mask_egid_subdf, agg_subdf.columns.get_loc('df_uid')])

                        for r in range(1,len(df_uids)+1):
                            for combo in itertools.combinations(df_uids, r):
                                combo_key_str = '_'.join([str(c) for c in combo])
                                mask_dfuid_only = np.isin(aggsub_npry[:,agg_subdf.columns.get_loc('df_uid')], list(combo))
                                mask_dfuid_subdf = mask_egid_subdf & mask_dfuid_only

                                egid_list.append(egid)
                                combo_df_uid_list.append(combo_key_str)
                                # df_uid_list.append(list(combo))
                                bfs_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('bfs')][0])
                                gklas_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('gklas')][0])
                                demandtype_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('demandtype')][0])
                                grid_node_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('grid_node')][0])

                                inst_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('inst_TF')][0])
                                info_source_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('info_source')][0])
                                pvid_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pvid')][0])
                                pv_tarif_Rp_kWh_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pv_tarif_Rp_kWh')][0]) 
                                elecpri_Rp_kWh_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('elecpri_Rp_kWh')][0])
                                demand_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('demand_kW')][0])

                                ausrichtung_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('AUSRICHTUNG')][0])
                                neigung_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('NEIGUNG')][0])

                                flaeche_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('FLAECHE')].sum())
                                stromertrag_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('STROMERTRAG')].sum())                    
                                flaech_angletilt_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('FLAECH_angletilt')].sum())
                                pvprod_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pvprod_kW')].sum())
                                selfconsum_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('selfconsum_kW')].sum())
                                netdemand_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('netdemand_kW')].sum())
                                netfeedin_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('netfeedin_kW')].sum())
                                econ_inc_chf_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('econ_inc_chf')].sum())
                                econ_spend_chf_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('econ_spend_chf')].sum())



                    aggsubdf_combo = pd.DataFrame({'EGID': egid_list, 'df_uid_combo': combo_df_uid_list, 'bfs': bfs_list,
                                                'gklas': gklas_list, 'demandtype': demandtype_list, 'grid_node': grid_node_list,

                                                'inst_TF': inst_list, 'info_source': info_source_list, 'pvid': pvid_list,
                                                'pv_tarif_Rp_kWh': pv_tarif_Rp_kWh_list, 'elecpri_Rp_kWh': elecpri_Rp_kWh_list,
                                                'demand_kW': demand_list,

                                                'AUSRICHTUNG': ausrichtung_list, 'NEIGUNG': neigung_list,
                                                
                                                'FLAECHE': flaeche_list, 'STROMERTRAG': stromertrag_list,
                                                'FLAECH_angletilt': flaech_angletilt_list,
                                                'pvprod_kW': pvprod_list,
                                                'selfconsum_kW': selfconsum_list, 'netdemand_kW': netdemand_list, 'netfeedin_kW': netfeedin_list,
                                                'econ_inc_chf': econ_inc_chf_list, 'econ_spend_chf': econ_spend_chf_list})
                            
                if (i <3) and (i_m <3): 
                    checkpoint_to_logfile(f'\t created df_uid combos for {agg_subdf["EGID"].nunique()} EGIDs', self.sett.log_name, 1, self.sett.show_debug_prints)

                

                # NPV calculation -----------------------------------------------------
                estim_instcost_chfpkW, estim_instcost_chftotal = self.initial_sml_get_instcost_interpolate_function()
                estim_instcost_chftotal(pd.Series([10, 20, 30, 40, 50, 60, 70]))

                # # estim_instcost_chfpkW, estim_instcost_chftotal = initial.estimate_iterpolate_instcost_function(pvalloc_settings)

                # if not os.path.exists(f'{preprep_name_dir_path }/pvinstcost_coefficients.json') == True:
                #     estim_instcost_chfpkW, estim_instcost_chftotal = initial.estimate_iterpolate_instcost_function(pvalloc_settings)
                #     estim_instcost_chftotal(pd.Series([10, 20, 30, 40, 50, 60, 70]))

                # elif os.path.exists(f'{preprep_name_dir_path }/pvinstcost_coefficients.json') == True:    
                #     estim_instcost_chfpkW, estim_instcost_chftotal = initial.get_estim_instcost_function(pvalloc_settings)
                #     estim_instcost_chftotal(pd.Series([10, 20, 30, 40, 50, 60, 70]))

                # correct cost estimation by a factor based on insights from pvprod_correction.py
                # aggsubdf_combo['estim_pvinstcost_chf'] = estim_instcost_chftotal(aggsubdf_combo['FLAECHE'] * 
                #                                                                  self.sett.TECspec_kWpeak_per_m2 * 
                #                                                                  self.sett.TECspec_share_roof_area_available) / self.sett.TECspec_estim_pvinst_cost_correctionfactor
                kwp_peak_array = aggsubdf_combo['FLAECHE'] * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available / self.sett.TECspec_estim_pvinst_cost_correctionfactor
                aggsubdf_combo['estim_pvinstcost_chf'] = estim_instcost_chftotal(kwp_peak_array) 
                
                

                def compute_npv(row):
                    pv_cashflow = (row['econ_inc_chf'] - row['econ_spend_chf']) / (1+self.sett.TECspec_interest_rate)**np.arange(1, self.sett.TECspec_invst_maturity+1)
                    npv = (-row['estim_pvinstcost_chf']) + np.sum(pv_cashflow)
                    return npv
                aggsubdf_combo['NPV_uid'] = aggsubdf_combo.apply(compute_npv, axis=1)

                if (i <3) and (i_m <3): 
                    checkpoint_to_logfile('\t computed NPV for agg_subdf', self.sett.log_name, 1, self.sett.show_debug_prints)

                agg_npv_df_list.append(aggsubdf_combo)

            agg_npv_df = pd.concat(agg_npv_df_list)
            npv_df = copy.deepcopy(agg_npv_df)


            # export npv_df -----------------------------------------------------
            npv_df.to_parquet(f'{subdir_path}/npv_df.parquet')
            if self.sett.export_csvs:
                npv_df.to_csv(f'{subdir_path}/npv_df.csv', index=False)


            # export by Month -----------------------------------------------------
            if self.sett.MCspec_keep_files_month_iter_TF:
                if i_m < self.sett.MCspec_keep_files_month_iter_max:
                    pred_npv_inst_by_M_path = f'{subdir_path}/pred_npv_inst_by_M'
                    if not os.path.exists(pred_npv_inst_by_M_path):
                        os.makedirs(pred_npv_inst_by_M_path)

                    npv_df.to_parquet(f'{pred_npv_inst_by_M_path}/npv_df_{m}.parquet')
                    if self.sett.export_csvs:
                        npv_df.to_csv(f'{pred_npv_inst_by_M_path}/npv_df_{m}.csv', index=False)
                    if i_m < 5:
                        npv_df.to_csv(f'{pred_npv_inst_by_M_path}/npv_df_{m}.csv', index=False)


            checkpoint_to_logfile('exported npv_df', self.sett.log_name, 1)
                
            return npv_df


        def algo_select_AND_adjust_topology(self, subdir_path: str, i_m: int, m):
            
    
            print_to_logfile('run function: select_AND_adjust_topology', self.sett.log_name)

            # import ----------
            topo = json.load(open(f'{subdir_path}/topo_egid.json', 'r'))
            npv_df = pd.read_parquet(f'{subdir_path}/npv_df.parquet') 
            pred_inst_df = pd.read_parquet(f'{subdir_path}/pred_inst_df.parquet') if os.path.exists(f'{subdir_path}/pred_inst_df.parquet') else pd.DataFrame()


            # drop installed partitions from npv_df 
            #   -> otherwise multiple selection possible
            #   -> easier to drop inst before each selection than to create a list / df and carry it through the entire code)
            npv_df_start_inst_selection = copy.deepcopy(npv_df)
            egid_wo_inst = [egid for egid in topo if topo.get(egid, {}).get('pv_inst', {}).get('inst_TF') == False]
            npv_df = copy.deepcopy(npv_df.loc[npv_df['EGID'].isin(egid_wo_inst)])


            # SELECTION BY METHOD ---------------
            # set random seed
            if self.sett.ALGOspec_rand_seed is not None:
                np.random.seed(self.sett.ALGOspec_rand_seed)

            # have a list of egids to install on for sanity check. If all build, start building on the rest of EGIDs
            install_EGIDs_summary_sanitycheck = self.sett.CHECKspec_egid_list

            if isinstance(install_EGIDs_summary_sanitycheck, list):
                # remove duplicates from install_EGIDs_summary_sanitycheck
                unique_EGID = []
                for e in install_EGIDs_summary_sanitycheck:
                        if e not in unique_EGID:
                            unique_EGID.append(e)
                install_EGIDs_summary_sanitycheck = unique_EGID
                # get remaining EGIDs of summary_sanitycheck_list that are not yet installed
                # > not even necessary if installed EGIDs get dropped from npv_df?
                remaining_egids = [
                    egid for egid in install_EGIDs_summary_sanitycheck 
                    if not topo.get(egid, {}).get('pv_inst', {}).get('inst_TF', False) == False ]
                
                if any([True if egid in npv_df['EGID'] else False for egid in remaining_egids]):
                    npv_df = npv_df.loc[npv_df['EGID'].isin(remaining_egids)].copy()
                else:
                    npv_df = npv_df.copy()
                    

            if self.sett.ALGOspec_inst_selection_method == 'random':
                npv_pick = npv_df.sample(n=1).copy()
            
            elif self.sett.ALGOspec_inst_selection_method == 'max_npv':
                npv_pick = npv_df[npv_df['NPV_uid'] == max(npv_df['NPV_uid'])].copy()

            elif self.sett.ALGOspec_inst_selection_method == 'prob_weighted_npv':
                rand_num = np.random.uniform(0, 1)
                
                npv_df['NPV_stand'] = npv_df['NPV_uid'] / max(npv_df['NPV_uid'])
                npv_df['diff_NPV_rand'] = abs(npv_df['NPV_stand'] - rand_num)
                npv_pick = npv_df[npv_df['diff_NPV_rand'] == min(npv_df['diff_NPV_rand'])].copy()
                
                # if multiple rows at min to rand num 
                if npv_pick.shape[0] > 1:
                    rand_row = np.random.randint(0, npv_pick.shape[0])
                    npv_pick = npv_pick.iloc[rand_row]


            # remove cols for uniform format between selection methods
            for col in ['NPV_stand', 'diff_NPV_rand']:
                if col in npv_df.columns:
                    npv_df.drop(columns=['NPV_stand', 'diff_NPV_rand'], inplace=True)
            # ---------------


            if isinstance(npv_pick, pd.DataFrame):
                picked_egid = npv_pick['EGID'].values[0]
                picked_uid = npv_pick['df_uid_combo'].values[0]
                picked_flaech = npv_pick['FLAECHE'].values[0]
                for col in ['NPV_stand', 'diff_NPV_rand']:
                    if col in npv_pick.columns:
                        npv_pick.drop(columns=['NPV_stand', 'diff_NPV_rand'], inplace=True)

            elif isinstance(npv_pick, pd.Series):
                picked_egid = npv_pick['EGID']
                picked_uid = npv_pick['df_uid_combo']
                picked_flaech = npv_pick['FLAECHE']
                for col in ['NPV_stand', 'diff_NPV_rand']:
                    if col in npv_pick.index:
                        npv_pick.drop(index=['NPV_stand', 'diff_NPV_rand'], inplace=True)
                        
            inst_power = picked_flaech * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available
            npv_pick['inst_TF'], npv_pick['info_source'], npv_pick['xtf_id'], npv_pick['BeginOp'], npv_pick['TotalPower'], npv_pick['iter_round'] = [True, 'alloc_algorithm', picked_uid, str(m), inst_power, i_m]
            

            # Adjust export lists / df
            if '_' in picked_uid:
                picked_combo_uid = list(picked_uid.split('_'))
            else:
                picked_combo_uid = [picked_uid]

            if isinstance(npv_pick, pd.DataFrame):
                pred_inst_df = pd.concat([pred_inst_df, npv_pick])
            elif isinstance(npv_pick, pd.Series):
                pred_inst_df = pd.concat([pred_inst_df, npv_pick.to_frame().T])
            

            # Adjust topo
            topo[picked_egid]['pv_inst'] = {'inst_TF': True, 'info_source': 'alloc_algorithm', 'xtf_id': picked_uid, 'BeginOp': f'{m}', 'TotalPower': inst_power}


            # export main dfs ------------------------------------------
            # do not overwrite the original npv_df, this way can reimport it every month and filter for sanitycheck
            pred_inst_df.to_parquet(f'{subdir_path}/pred_inst_df.parquet')
            pred_inst_df.to_csv(f'{subdir_path}/pred_inst_df.csv') if self.sett.export_csvs else None
            with open (f'{subdir_path}/topo_egid.json', 'w') as f:
                json.dump(topo, f)


            # export by Month ------------------------------------------
            pred_inst_df.to_parquet(f'{subdir_path}/pred_npv_inst_by_M/pred_inst_df_{m}.parquet')
            pred_inst_df.to_csv(f'{subdir_path}/pred_npv_inst_by_M/pred_inst_df_{m}.csv') if self.sett.export_csvs else None
            with open(f'{subdir_path}/pred_npv_inst_by_M/topo_{m}.json', 'w') as f:
                json.dump(topo, f)
                        
            return  inst_power, npv_df  # , picked_uid, picked_combo_uid, pred_inst_df, dfuid_installed_list, topo



                    

# ======================================================================================================
# RUN SCENARIOS
# ======================================================================================================
if __name__ == '__main__':

    pvalloc_scen_list = [
        PVAllocScenario_Settings(
                name_dir_export    = 'pvalloc_BFS2761_2m_f2021_1mc_meth2.2_rnd_DEBUG',
                name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
                show_debug_prints  = True,
                export_csvs        = True,
                T0_year_prediction = 2021,
                months_prediction  = 2,
                ALGOspec_inst_selection_method = 'prob_weighted_npv',
                TECspec_pvprod_calc_method = 'method2.2',
                MCspec_montecarlo_iterations = 2,
        ), 

    
    ]


    for pvalloc_scen in pvalloc_scen_list:
        pvalloc_class = PVAllocScenario(pvalloc_scen)
        
        # pvalloc_class.export_pvalloc_scen_settings()

        pvalloc_class.run_pvalloc_initalization()
        pvalloc_class.run_pvalloc_mcalgorithm()
        # pvalloc_self.sett.run_pvalloc_postprocess()









