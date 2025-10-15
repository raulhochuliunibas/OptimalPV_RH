import os as os
import numpy as np
import polars as pl
import pandas as pd
import geopandas as gpd
import sqlite3
import copy
import itertools
import time 
import datetime
import seaborn as sns
import glob
import shutil
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV  
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split

from shapely.geometry import Point
from shapely.ops import unary_union
from dataclasses import dataclass, field
from typing_extensions import List, Dict, Tuple


@dataclass
class Calibration_Settings:
    # DEFAULT SETTINGS --------------------------------------------------- 
    name_dir_export: str                    = 'calib_mini_debug'
    name_preprep_subsen: str                = 'preprep_class_default_sett'
    name_calib_subscen: str                 = 'calib_class_default_sett'
    scicore_concat_data_path: str           = None


    kt_numbers: List[int]                    = field(default_factory=lambda: [
        # 1, 2, 3, 4, 5, 6, 7, 89, 10, 11, 12, 13, 
        # 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 
        ]
        )                               # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain individual municipalities
    bfs_numbers: List[int]                   = field(default_factory=lambda: [
        # 2883,
        # 2546,  # Grenchen

        # subest in n_rows solkat subset 
        1201, 1205, 96, 1033, 4237, 4239, 
        # 96, 1033, 4237, 4239, 1201, 1205, 1218, 1207
        ])                                      
    
    rerun_import_and_preprp_data_TF: bool           = True 
    export_gwr_ALL_building_gdf_TF: bool            = False  

    run_concatenate_preprep_data_TF: bool           = True
    run_approach1_fit_optim_costfunction_TF: bool   = True  
    run_approach2_regression_instsize_TF: bool      = True
    run_appr2_random_forest_reg_TF: bool            = True


    pvinst_pvtrif_elecpri_range_minmax:Tuple[int,int]  = (2018, 2023)                             # min and max year of PV installation to be considered 
    pvinst_capacity_minmax:Tuple[float,float]= (0.5, 50)                             # min and max capacity (in kWp) of PV installation to be considered

    reg2_random_forest_reg_settings: Dict = field(default_factory=lambda: {
        'run_ML_rfr_TF': True,
        'comment_settings_RandomForestRegressor()': 
        """
            n_estimators:       Defines the number of decision trees in the Random Forest.
            random_state=0:     Ensures the randomness in model training is controlled for reproducibility.
            oob_score=True:     Enables out-of-bag scoring which evaluates the model's performance using data 
                                not seen by individual trees during training
            max_depth:          The maximum depth of the tree. If None, then nodes are expanded until all 
                                leaves are pure or until all leaves contain less than min_samples_split 
                                samples
        """,
        # 'n_estimators':         100 ,    # default: 100   # | 1,       
        # 'min_samples_split':    5   ,    # default: 2     # | 1000,    
        # 'max_depth':            20  ,    # default: None  # | 3,       
        'random_state':         None,    # default: None  # | None,    
        'n_jobs':               -1,      # default: None  # | -1,  
        'cross_validation':     2, 
        'n_estimators':         [1, ]  ,    # default: 100   # | 1,       
        'min_samples_split':    [5, ]    ,    # default: 2     # | 1000,    
        'max_depth':            [3, ]   ,    # default: None  # | 3,       

        'reg2_rfrname_dfsuffix_tupls': [
            ('_rfr1', ''),
        ]
    })




    # settings from DATA_AGGREGATION ---------------------------------------------------
    # new ones
    n_rows_import: int                       = None
    SOLKAT_PV_buffer_size: int               = 20                          # buffer size in meters for the GWR selection   
    
    # existing ones
    if True: 
        GWR_building_cols: List[str]    = field(default_factory=lambda: ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE',
                                                'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ',
                                                'GANZWHG','GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1',
                                                'GEBF', 'GAREA'])
        GWR_dwelling_cols: List[str]    = field(default_factory=lambda: ['EGID', 'EWID', 'WAZIM', 'WAREA', ])
        GWR_GSTAT: List[str]            = field(default_factory=lambda: [
                                                '1001', # GSTAT - 1001: in planing
                                                '1002', # GSTAT - 1002: construction right granted 
                                                '1003', # GSTAT - 1003: in construction
                                                '1004', # GSTAT - 1004: fully constructed, existing buildings
                                                ])                                 
        GWR_GKLAS: List[str]            = field(default_factory=lambda: [
                                                '1110', # GKLAS - 1110: only 1 living space per building
                                                '1121', # GKLAS - 1121: Double-, row houses with each appartment (living unit) having it's own roof;
                                                '1122', # GKLAS - 1122: Buildings with three or more appartments
                                                '1276', # GKLAS - 1276: structure for animal keeping (most likely still one owner)
                                                '1278', # GKLAS - 1278: structure for agricultural use (not anmial or plant keeping use, e.g. barns, machinery storage, silos),
                                                ])
        GWR_GBAUJ_minmax: List[int]     = field(default_factory=lambda: [1920, 2022])                       # GBAUJ_minmax: range of years of construction
        GWR_AREtypology : Dict          = field(default_factory=lambda:  {
                                                'Urban': [2, 4, ],
                                                'Suburban': [3, 5, 4, 6 ], 
                                                'Rural': [7, 8,],                        
                                                # 1 - big centers   # https://map.geo.admin.ch/#/map?lang=en&center=2611872.51,1270543.42&z=3.703&topic=ech&layers=ch.swisstopo.zeitreihen@year=1864,f;ch.bfs.gebaeude_wohnungs_register,f;ch.bav.haltestellen-oev,f;ch.swisstopo.swisstlm3d-wanderwege,f;ch.vbs.schiessanzeigen,f;ch.astra.wanderland-sperrungen_umleitungen,f;ch.are.gemeindetypen;ch.swisstopo.swissboundaries3d-kanton-flaeche.fill&bgLayer=ch.swisstopo.pixelkarte-farbe            # '1' - big centers => URBAN
                                                # 2 - secondary centers of big centers  => URBAN 
                                                # 3 - crown big centers => SEMI-URBAN
                                                # 4 - medium centers => 
                                                # 5 - crown medium centers =>
                                                # 6 - small centers => 
                                                # 7 - peri-urban rural communes => RURAL
                                                # 8 - agricultural communes => RURAL
                                                # 9 - tourist communes => RURAL
        })
        GWR_SFHMFHtypology: Dict       = field(default_factory=lambda: {
                                                'SFH': ['1110', ], 
                                                'MFH': ['1121', '1122', '1276', '1278', ],
        })
        GWR_SFHMFH_outsample_proxy:str = 'MFH'
        SOLKAT_GWR_EGID_buffer_size: int = 10                          # buffer size in meters for the GWR selection
        
        SOLKAT_extend_dfuid_for_missing_EGIDs_to_be_unique: bool = False
        SOLKAT_cols_adjust_for_missEGIDs_to_solkat: List[str] = field(default_factory=lambda: ['FLAECHE', 'STROMERTRAG', 'GSTRAHLUNG', 'MSTRAHLUNG' ])
        DEMAND_input_data_source: str = 'SwissStore'#    # "NETFLEX"  OR "SwissStore"
        
        # * not used settings *
        # GWR_DEMAND_proxy: str           = 'GAREA'
        # GWR_GWAERZH: List[str]          = field(default_factory=lambda: ['7410', '7411',])                       # GWAERZH - 7410: heat pumpt for 1 building, 7411: heat pump for multiple buildings
        # SOLKAT_col_partition_union: str = 'SB_UUID'                   # column name used for the union of partitions
        # SOLKAT_test_loop_optim_buff_size_TF: bool = False
        # SOLKAT_test_loop_optim_buff_arang: List[float] = field(default_factory=lambda: [0, 10, 0.1])


    # settings from PVALLOCATION --------------------------------------------------- 
    # new ones
    topo_df_excl_gwr_cols: List[str] = field(default_factory=lambda: [
        'GKSCE', 'GKODE', 'GKODN', 
        ])
    topo_df_excl_solkat_cols: List[str] = field(default_factory=lambda: [
        'SB_UUID', 'SB_OBJEKTART', 'SB_DATUM_ERSTELLUNG', 'SB_DATUM_AENDERUNG', 'KLASSE', 
        'WAERMEERTRAG', 'DUSCHGAENGE', 'DG_HEIZUNG', 'DG_WAERMEBEDARF', 'BEDARF_WARMWASSER', 
        'BEDARF_HEIZUNG', 'FLAECHE_KOLLEKTOREN', 'VOLUMEN_SPEICHER',
    ])
    topo_df_excl_pv_cols: List[str] = field(default_factory=lambda: [
        'Address', 'PostCode', 'Municipality', 'Canton', 'MainCategory',
        'SubCategory', 'PlantCategory',
    ])


    # existing ones
    if True:
        # # PART I: settings for alloc_initialization --------------------
        GWRspec_solkat_max_n_partitions: int                = 10          # larger number of partitions make all combos of roof partitions practically impossible to calculate
        GWRspec_solkat_area_per_EGID_range: List[int]       = field(default_factory=lambda: [2, 600])  # for 100kWp inst, need 500m2 roof area => just above the threshold for residential subsidies KLEIV, below 2m2 too small to mount installations
        GWRspec_building_cols: List[str]                    = field(default_factory=lambda: [
                                                                'EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                                                                'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                                                                'GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1', 'GEBF', 'GAREA'
                                                            ])
        
        GWRspec_dwelling_cols: List[str]                    = field(default_factory=list)
        GWRspec_swstore_demand_cols: List[str]              = field(default_factory=lambda: ['ARE_typ', 'sfhmfh_typ', 'arch_typ', 'demand_elec_pGAREA'])
        GWRspec_DEMAND_proxy: str                           = 'GAREA'
        # gwr topo_egid selection
        GWRspec_GSTAT: List[str]                            = field(default_factory=lambda: [
                                                                    '1001', # GSTAT - 1001: in planing
                                                                    '1002', # GSTAT - 1002: construction right granted 
                                                                    '1003', # GSTAT - 1003: in construction
                                                                    '1004', # GSTAT - 1004: fully constructed, existing buildings
                                                                    ])    
        GWRspec_GKLAS: List[str]                            = field(default_factory=lambda: [
                                                                    '1110', # GKLAS - 1110: only 1 living space per building
                                                                    '1121', # GKLAS - 1121: Double-, row houses with each appartment (living unit) having it's own roof;
                                                                    '1122', # GKLAS - 1122: Buildings with three or more appartments
                                                                    # '1276', # GKLAS - 1276: structure for animal keeping (most likely still one owner)
                                                                    # '1278', # GKLAS - 1278: structure for agricultural use (not anmial or plant keeping use, e.g. barns, machinery storage, silos),
                                                                    ])
        GWRspec_GBAUJ_minmax: List[int]                     = field(default_factory=lambda: [1920, 2022])           
        

        # weather_specs
        WEAspec_meteo_col_dir_radiation: str                = 'Basel Direct Shortwave Radiation'
        WEAspec_meteo_col_diff_radiation: str               = 'Basel Diffuse Shortwave Radiation'
        WEAspec_meteo_col_temperature: str                  = 'Basel Temperature [2 m elevation corrected]'
        WEAspec_weather_year: int                           = 2022
        WEAspec_radiation_to_pvprod_method: str             = 'dfuid_ind'
        # WEAspec_rad_rel_loc_max_by: str                     = 'dfuid_specific'
        # WEAspec_flat_direct_rad_factor: int                 = 1
        # WEAspec_flat_diffuse_rad_factor: int                = 1

        # # constr_capacity_specs
        # CSTRspec_iter_time_unit: str                        = 'year'   # month, year
        # CSTRspec_ann_capacity_growth: float                 = 0.15
        # CSTRspec_constr_capa_overshoot_fact: int            = 1
        # CSTRspec_month_constr_capa_tuples: List[tuple]      = field(default_factory=lambda: [
        #                                                         (1,  0.04), 
        #                                                         (2,  0.04), 
        #                                                         (3,  0.04), 
        #                                                         (4,  0.06),
        #                                                         (5,  0.06), 
        #                                                         (6,  0.06), 
        #                                                         (7,  0.1), 
        #                                                         (8,  0.1),
        #                                                         (9,  0.1), 
        #                                                         (10, 0.1), 
        #                                                         (11, 0.14), 
        #                                                         (12, 0.16)
        #                                                     ])
        
        # tech_economic_specs
        TECspec_self_consumption_ifapplicable: float            = 1.0
        TECspec_interest_rate: float                            = 0.01
        # TECspec_pvtarif_year: int                               = 2022
        TECspec_pvtarif_col: List[str]                          = field(default_factory=lambda: ['energy1', ])  # 'energy1', 'eco1'
        TECspec_generic_pvtarif_Rp_kWh: float                   = None 
        TECspec_pvprod_calc_method: str                         = 'method2.2'
        TECspec_panel_efficiency: float                         = 0.21
        TECspec_inverter_efficiency: int                        = 0.8
        # TECspec_elecpri_year: int                               = 2022
        TECspec_elecpri_category: str                           = 'H4'
        TECspec_invst_maturity: int                             = 25
        TECspec_kWpeak_per_m2: float                            = 0.2
        TECspec_share_roof_area_available: float                = 1
        TECspec_max_distance_m_for_EGID_node_matching: float    = 0
        TECspec_kW_range_for_pvinst_cost_estim: List[int]       = field(default_factory=lambda: [0, 61])
        TECspec_estim_pvinst_cost_correctionfactor: float       = 1
        TECspec_opt_max_flaeche_factor: float                   = 1.5
        TECspec_add_heatpump_demand_TF: bool                    = True   
        TECspec_heatpump_months_factor: List[tuple]             = field(default_factory=lambda: [
                                                                (10, 1.0 ),
                                                                (11, 1.0 ), 
                                                                (12, 1.0 ), 
                                                                (1 , 1.0 ), 
                                                                (2 , 1.0 ), 
                                                                (3 , 1.0 ), 
                                                                (4 , 1.0 ), 
                                                                (5 , 1.0 ), 
                                                                (6 , 1.0 ), 
                                                                (7 , 1.0 ), 
                                                                (8 , 1.0 ), 
                                                                (9 , 1.0 )  
                                                                ])
        # # panel_efficiency_specs
        # PEFspec_variable_panel_efficiency_TF: bool              = True
        # PEFspec_summer_months: List[int]                        = field(default_factory=lambda: [6, 7, 8, 9])
        # PEFspec_hotsummer_hours: List[int]                      = field(default_factory=lambda: [11, 12, 13, 14, 15, 16, 17])
        # PEFspec_hot_hours_discount: float                       = 0.1
        
        # # sanitycheck_summary_byEGID_specs
        # CHECKspec_egid_list: List[str]                          = field(default_factory=lambda: [])
        #                                                         #     '391292', '390601', '2347595', '401781',  # single roof houses in Aesch, Ettingen
        #                                                         #     '391263', '245057295', '401753',  # houses with built pv in Aesch, Ettingen
        #                                                         #     '245054165', '245054166', '245054175', '245060521', # EGID selection of neighborhood within Aesch to analyse closer
        #                                                         #     '391253', '391255', '391257', '391258', '391262',
        #                                                         #     '391263', '391289', '391290', '391291', '391292',
        #                                                         #     '245057295', '245057294', '245011456', '391379', '391377'
        #                                                         # ])
        # CHECKspec_n_EGIDs_of_alloc_algorithm: int               = 20
        # CHECKspec_n_iterations_before_sanitycheck: int          = 1

        # # PART II: settings for MC algorithm --------------------
        # MCspec_montecarlo_iterations_fordev_sequentially: int        = 1
        # MCspec_fresh_initial_files: List[str]                       = field(default_factory=lambda: [
        #                                                                 'topo_egid.json', 'trange_prediction.parquet',# 'gridprem_ts.parquet', 
        #                                                                 'constrcapa.parquet', # 'dsonodes_df.parquet'
        #                                                             ])
        # MCspec_keep_files_month_iter_TF: bool                       = True
        # MCspec_keep_files_month_iter_max: int                       = 9999999999
        # MCspec_keep_files_month_iter_list: List[str]                = field(default_factory=lambda: [
        #                                                                 'topo_egid.json', 'npv_df.parquet', 'pred_inst_df.parquet', 'gridprem_ts.parquet'
        #                                                             ])

        # algorithm_specs
        ALGOspec_inst_selection_method: str                         = 'random'          # 'random', max_npv', 'prob_weighted_npv'
        ALGOspec_rand_seed: bool                                    = None
        ALGOspec_while_inst_counter_max: int                        = 5000
        ALGOspec_topo_subdf_partitioner: int                        = 250  # 9999999
        ALGOspec_npv_update_groupby_cols_topo_aggdf: List[str]      = field(default_factory=lambda: [
                                                                        'EGID', 'df_uid', 'grid_node', 'bfs', 'GKLAS', 'GAREA', 'sfhmfh_typ', 'demand_arch_typ', 'inst_TF', 'info_source',
                                                                        'pvid', 'pvtarif_Rp_kWh', 'elecpri_Rp_kWh', 'FLAECHE', 'FLAECH_angletilt', 'AUSRICHTUNG', 
                                                                        'NEIGUNG', 'STROMERTRAG'
                                                                    ])
        ALGOspec_npv_update_agg_cols_topo_aggdf: Dict[str, str]     = field(default_factory=lambda: {
                                                                        'pvprod_kW': 'sum', 'demand_kW': 'sum', 'selfconsum_kW': 'sum', 'netdemand_kW': 'sum',
                                                                        'netfeedin_kW': 'sum', 'econ_inc_chf': 'sum', 'econ_spend_chf': 'sum'
                                                                    })
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF: bool   = False
        ALGOspec_adjust_existing_pvdf_capa_topartition: str         = 'capa_no_adj_pvprod_no_adj'     
                                                                        # 'capa_no_adj_pvprod_no_adj',              : assigns df_uid_winst to topo in (df_uid, dfuidPower) - tuples, to be easily accessed later, pvprod_kW is not altered at all.
                                                                        # 'capa_roundup_pvprod_no_adj'              : assigns df_uid_w_inst to topo, based on pv_df TotalPower value (rounded up), pvprod_kW remains "untouched" and is still equivalent to production potential per roof partition
                                                                        # 'capa_roundup_pvprod_adjusted' - ATTENTION: will activate an if statement which will adjust pvprod_kW in topo_time_subdfs, so no longer pure production potential per roof partition
                                                                        # 'capa_no_adj_pvprod_adjusted' - ATTENTION: will activate an if statement which will adjust pvprod_kW in topo_time_subdfs, so no longer pure production potential per roof partition
        ALGOspec_pvinst_option_to_EGID: str                         = 'max_dfuid_EGIDcombos'    # 'EGIDitercombos_maxdfuid' / 'EGIDoptimal__partial_dfuid'

        ALGOspec_constr_capa_overshoot_fact: float                  = 1
        ALGOspec_subselec_filter_criteria: str                      = None  # 'southfacing_1spec' / 'eastwestfacing_3spec' / 'southwestfacing_2spec'
        ALGOspec_drop_cols_topo_time_subdf_list: List[str]          = field(default_factory=lambda: [
                                                                        'index', 'timestamp', 'rad_direct', 'rad_diffuse', 'temperature', 
                                                                        'A_PARAM', 'B_PARAM', 'C_PARAM', 'mean_top_radiation', 
                                                                        'radiation_rel_locmax'])
        
        ALGOspec_reinstall_inst_EGID_pvdf_for_check_TF :bool        = False  # True: will reinstall the dfuid_winst to EGIDs that already have a inst in reality in pv_df to check accuracy of allocation kWp estimates
        ALGOspec_pvinst_size_calculation: str                       = 'npv_optimized'   # 'inst_full_partition' / 'npv_optimized'
        
        ALGOspec_tweak_constr_capacity_fact: float                  = 1
        ALGOspec_tweak_npv_calc: float                              = 1
        ALGOspec_tweak_npv_excl_elec_demand: bool                   = True
        ALGOspec_tweak_gridnode_df_prod_demand_fact: float          = 1
        ALGOspec_tweak_demand_profile: float                        = 1.8

        # # dsonodes_ts_specs
        # GRIDspec_flat_profile_demand_dict: Dict                     = field(default_factory=lambda: {
        #                                                                 '_window1':{'t': [6,21],  'demand_share': 0.9}, 
        #                                                                 '_window2':{'t': [22, 5], 'demand_share': 0.1},
        #                                                                 })
        # GRIDspec_flat_profile_demand_total_EGID: float              = 4500
        # GRIDspec_flat_profile_demand_type_col: set                  = 'MFH_swstore'  # 'flat' / 'MFH_swstore' / 'outtopo_demand_zero'

        # # gridprem_adjustment_specs
        # GRIDspec_tier_description: str                              = 'tier_level: (voltage_threshold, gridprem_Rp_kWh)'
        # GRIDspec_power_factor: float                                = 1
        # GRIDspec_perf_factor_1kVA_to_XkW: float                     = 0.8
        # GRIDspec_colnames: List[str]                                = field(default_factory=lambda: ['tier_level', 'used_node_capa_rate', 'gridprem_Rp_kWh'])
        # GRIDspec_tiers: Dict[int, List[float]]                      = field(default_factory=lambda: {
        #                                                             1: [0.7, 1], 2: [0.8, 3], 3: [0.85, 5], 
        #                                                             4: [0.9, 7], 5: [0.95, 15], 6: [1, 100]
        #                                                             })
        

class Calibration:
    def __init__(self, settings: Calibration_Settings):
        self.sett = settings
        
        # SETUP --------------------
        self.sett.wd_path = os.getcwd()
        self.sett.data_path = os.path.join(self.sett.wd_path, 'data')
        self.sett.split_data_path = os.path.join(self.sett.data_path, 'input_split_data_geometry')
        # self.sett.preprep_import_path = os.path.join(self.sett.data_path, 'preprep', self.sett.name_dir_preprep_import)
        self.sett.calib_path = os.path.join(self.sett.data_path, 'calibration')
        self.sett.calib_scen_path = os.path.join(self.sett.calib_path, self.sett.name_dir_export)
        self.sett.calib_scen_preprep_path = os.path.join(self.sett.calib_scen_path, 'preprep_data')
        self.sett.subscen_time_log_path = f'{self.sett.calib_scen_preprep_path}/{self.sett.name_preprep_subsen}_preprep_time_log.txt'

        
        os.makedirs(self.sett.calib_path, exist_ok=True)
        os.makedirs(self.sett.calib_scen_path, exist_ok=True)

        print(f'{30*"="}\n  START CALIBRATION: {self.sett.name_dir_export}\n  > name_preprep_subsen: {self.sett.name_preprep_subsen}\n  > name_calib_subscen: {self.sett.name_calib_subscen}\n{30*"="}\n')


        # bfs numbers selection
        gm_shp_gdf = gpd.read_file(f'{self.sett.data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
        gm_shp_gdf = gm_shp_gdf.drop(columns = ['UUID', 'DATUM_AEND', 'DATUM_ERST'])
        gm_shp_gdf = gm_shp_gdf.loc[gm_shp_gdf['KANTONSNUM'].notna()]
        gm_shp_gdf['KANTONSNUM'] = gm_shp_gdf['KANTONSNUM'].astype(int)
        gm_shp_gdf['BFS_NUMMER'] = gm_shp_gdf['BFS_NUMMER'].astype(str)

        gm_shp_gdf['BFS_NUMMER_int'] = gm_shp_gdf['BFS_NUMMER'].astype(int)
        gm_shp_gdf = gm_shp_gdf.loc[gm_shp_gdf['BFS_NUMMER_int'] < 9000].copy()
        gm_shp_gdf.crs = 'EPSG:2056'

        if False: 
            bfs_list_all = gm_shp_gdf['BFS_NUMMER'].unique().tolist()
            with open(f'{os.getcwd()}/calib_ary_bfs_launch_{len(bfs_list_all)}nbfs.cmd', 'a') as f:
                for bfs in bfs_list_all:
                    f.write(f'python src/calibration_array_by_bfs.py {bfs}\n')


        if self.sett.kt_numbers != []:
            gm_select_kt = gm_shp_gdf.loc[gm_shp_gdf['KANTONSNUM'].isin(self.sett.kt_numbers),['KANTONSNUM', 'BFS_NUMMER', 'NAME',]]
            self.sett.bfs_numbers = gm_select_kt['BFS_NUMMER'].unique().tolist()
        else:
            self.sett.bfs_numbers = [str(bfs) for bfs in self.sett.bfs_numbers]


    def write_to_logfile(self, str_text, log_time, log_file_path = None):
        if log_file_path is None:
            log_file_path = self.sett.subscen_time_log_path
        with open(self.sett.subscen_time_log_path, 'a') as f:
            str_to_print = f'{str_text:70}, time: {time.ctime()}, to complete: {round((time.time()-log_time),2)} sec'
            f.write(str_to_print)
            print(str_to_print)
            log_time = time.time()
        return log_time


    def import_and_preprep_data(self,):
        os.makedirs(self.sett.calib_scen_preprep_path, exist_ok=True)
        name_preprep_subsen = self.sett.name_preprep_subsen
        
        log_time = time.time()
        log_time = self.write_to_logfile('\n * import_and_preprep_data()', log_time)


        start_time = time.time()
        with open(self.sett.subscen_time_log_path, 'w') as f:
            f.write(f'calibration - prerep_data: {self.sett.name_dir_export}>{self.sett.name_preprep_subsen} \n')
            f.write(f'start time: {time.ctime()} \n\n')
            
        # bfs numbers selection
        gm_shp_gdf = gpd.read_file(f'{self.sett.data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
        gm_shp_gdf = gm_shp_gdf.drop(columns = ['DATUM_AEND', 'DATUM_ERST'])
        gm_shp_gdf = gm_shp_gdf.loc[gm_shp_gdf['KANTONSNUM'].notna()]
        gm_shp_gdf['KANTONSNUM'] = gm_shp_gdf['KANTONSNUM'].astype(int)
        gm_shp_gdf['BFS_NUMMER'] = gm_shp_gdf['BFS_NUMMER'].astype(str)
        gm_shp_gdf.crs = 'EPSG:2056'

        gm_shp_gdf = gm_shp_gdf.loc[gm_shp_gdf['BFS_NUMMER'].isin(self.sett.bfs_numbers)].copy()

        # import aid functions + data files ====================
        def attach_bfs_to_spatial_data(gdf, gm_shp_df, keep_cols = ['BFS_NUMMER', 'geometry' ]):
            """
            Function to attach BFS numbers to spatial data sources
            """
            gdf = copy.deepcopy(gdf)
            gdf.set_crs(gm_shp_df.crs, allow_override=True, inplace=True)
            gdf = gpd.sjoin(gdf, gm_shp_df, how="left", predicate="within")
            dele_cols = ['index_right'] + [col for col in gm_shp_df.columns if col not in keep_cols]
            gdf.drop(columns = dele_cols, inplace = True)
            if 'BFS_NUMMER' in gdf.columns:
                # transform BFS_NUMMER to str, np.nan to ''
                gdf['BFS_NUMMER'] = gdf['BFS_NUMMER'].apply(lambda x: '' if pd.isna(x) else str(int(x)))

            return gdf

        def set_crs_to_gm_shp(gdf_CRS, gdf_a, gdf_b = None):
            gdf_a.set_crs(gdf_CRS.crs, allow_override=True, inplace=True)
            if gdf_b is not None:
                gdf_b.set_crs(gdf_CRS.crs, allow_override=True, inplace=True)
            
            if gdf_b is None: 
                return gdf_a
            if gdf_b is not None:
                return gdf_a, gdf_b



        # import local data ====================
        if True:
            # pv ----------
            elec_prod_gdf = gpd.read_file(
                f'{self.sett.data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg', 
                layer ='ElectricityProductionPlant', 
                )
            pv_all_gdf = copy.deepcopy(elec_prod_gdf.loc[elec_prod_gdf['SubCategory'] == 'subcat_2'])
            pv_all_gdf['xtf_id'] = pv_all_gdf['xtf_id'].astype(str)

            pv_all_gdf = attach_bfs_to_spatial_data(pv_all_gdf, gm_shp_gdf)
            pv_select_gdf = pv_all_gdf.loc[pv_all_gdf['BFS_NUMMER'].isin(self.sett.bfs_numbers)]

            # pv_select_gdf.set_crs("EPSG:2056", allow_override=True, inplace=True)
            pv_select_gdf.crs = 'EPSG:2056'
            pv_gdf = pv_select_gdf.copy()
            pv_pq  = pv_select_gdf.loc[:,pv_select_gdf.columns !='geometry'].copy()
            pv_geo = pv_select_gdf.loc[:,['xtf_id', 'BFS_NUMMER', 'geometry']].copy()
    
            log_time = self.write_to_logfile('local data imported - pv, finished', log_time)



            # solkat ----------
            solkat_gpkg_path = os.path.join(self.sett.data_path, 'input', 'solarenergie-eignung-daecher_2056.gpkg', 'SOLKAT_DACH.gpkg')
            if self.sett.n_rows_import is not None:
                solkat_all_gdf = gpd.read_file(solkat_gpkg_path, layer ='SOLKAT_CH_DACH', rows = self.sett.n_rows_import)
            else:
                solkat_all_gdf = gpd.read_file(solkat_gpkg_path, layer ='SOLKAT_CH_DACH')

            # solkat_all_gdf = solkat_all_gdf.drop(columns = ['DATUM_ERSTELLUNG', 'DATUM_AENDERUNG', 'SB_DATUM_ERSTELLUNG', 'SB_DATUM_AENDERUNG',])
            for col in ['DATUM_ERSTELLUNG', 'DATUM_AENDERUNG', 'SB_DATUM_ERSTELLUNG', 'SB_DATUM_AENDERUNG',]:
                solkat_all_gdf[col] = solkat_all_gdf[col].astype(str)
            
            # minor transformations to str (with removing nan values)
            solkat_all_gdf['DF_UID'] = solkat_all_gdf['DF_UID'].astype(str)
            solkat_all_gdf['GWR_EGID'] = solkat_all_gdf['GWR_EGID'].fillna(0).astype(int).astype(str)
            solkat_all_gdf.loc[solkat_all_gdf['GWR_EGID'] == '0', 'GWR_EGID'] = 'NAN'
            solkat_all_gdf.rename(columns={'GWR_EGID': 'EGID'}, inplace=True)
            solkat_all_gdf = solkat_all_gdf.dropna(subset=['DF_UID'])
            # solkat_all_gdf['EGID_count'] = solkat_all_gdf.groupby('EGID')['EGID'].transform('count')

            solkat_all_gdf = attach_bfs_to_spatial_data(solkat_all_gdf, gm_shp_gdf)
            solkat_select_gdf = solkat_all_gdf.loc[solkat_all_gdf['BFS_NUMMER'].isin(self.sett.bfs_numbers)]

            # select solkat with pv installations
            # solkat_select_gdf.set_crs("EPSG:2056", allow_override=True, inplace=True)
            solkat_select_gdf.crs = 'EPSG:2056'
            pv_buffer_gdf = pv_gdf.drop(columns=['BFS_NUMMER']).copy()
            # pv_buffer_gdf['geometry'] = pv_buffer_gdf.geometry.buffer(self.sett.SOLKAT_PV_buffer_size)
            # solkat_pvinst = gpd.sjoin(solkat_select_gdf, pv_buffer_gdf, how='inner', predicate='intersects')
            
            solkat_export = solkat_select_gdf.copy()
            solkat_gdf  = solkat_export.copy()
            solkat_pq   = solkat_export.loc[:,solkat_export.columns !='geometry'].copy()
            solkat_geo  = solkat_export.loc[:,['DF_UID', 'BFS_NUMMER', 'geometry']].copy()
            log_time = self.write_to_logfile('local data imported - solkat, finished', log_time)



            # solkat month ----------pv_all_gdf
            # try to import only rows for selected solkats
            df_uid_sql_list = solkat_gdf['DF_UID'].unique().tolist()
            df_uid_sql_str = ','.join([f"'{dfuid}'" for dfuid in df_uid_sql_list])
            df_uid_sql_query = f"SELECT * FROM SOLKAT_CH_DACH_MONAT WHERE DF_UID IN ({df_uid_sql_str})"

            solkat_month_gpkg_path = os.path.join(self.sett.data_path, 'input', 'solarenergie-eignung-daecher_2056_monthlydata.gpkg', 'SOLKAT_DACH_MONAT.gpkg')
            solkat_month_pq = gpd.read_file(solkat_month_gpkg_path, sql=df_uid_sql_query)
            
            solkat_month_pq['SB_UUID'] = solkat_month_pq['SB_UUID'].astype(str)
            solkat_month_pq['DF_UID'] = solkat_month_pq['DF_UID'].astype(str)
            solkat_month_pq = solkat_month_pq.merge(solkat_pq[['DF_UID', 'BFS_NUMMER']], how = 'left', on = 'DF_UID')
            # solkat_month = solkat_month_pq[solkat_month_pq['BFS_NUMMER'].isin(self.sett.bfs_numbers)]
            log_time = self.write_to_logfile('local data imported - solkat month, finished', log_time)



            # gwr -------------------
                
            # querys ------

            # get DWELLING data
            # select cols
            query_columns = self.sett.GWR_dwelling_cols
            query_columns_str = ', '.join(query_columns)
            query_bfs_numbers = ', '.join([str(i) for i in self.sett.bfs_numbers])

            conn = sqlite3.connect(f'{self.sett.data_path}/input/GebWohnRegister.CH/data.sqlite')
            cur = conn.cursor()
            cur.execute(f'SELECT {query_columns_str} FROM dwelling')
            sqlrows = cur.fetchall()
            conn.close()

            gwr_dwelling_df = pd.DataFrame(sqlrows, columns=query_columns)
            gwr_dwelling_df[['WAZIM', 'WAREA']] = gwr_dwelling_df[['WAZIM', 'WAREA']].replace('', 0).astype(float)

            log_time = self.write_to_logfile('local data imported - gwr dwelling, finished', log_time)

            # get ALL BUILDING data
            # select cols
            query_columns = self.sett.GWR_building_cols
            query_columns_str = ', '.join(query_columns)
            query_bfs_numbers = ', '.join([str(i) for i in self.sett.bfs_numbers])

            conn = sqlite3.connect(f'{self.sett.data_path}/input/GebWohnRegister.CH/data.sqlite')
            cur = conn.cursor()
            cur.execute(f'SELECT {query_columns_str} FROM building WHERE GGDENR IN ({query_bfs_numbers})')
            sqlrows = cur.fetchall()
            conn.close()

            gwr_all_building_df = pd.DataFrame(sqlrows, columns=query_columns)
            log_time = self.write_to_logfile('local data imported - gwr all building, finished', log_time)

            # merger ------
            if False: 
                gwr_mrg = gwr_all_building_df.merge(gwr_dwelling_df, on='EGID', how='left')

                bldg_agg_cols = copy.deepcopy(self.sett.GWR_building_cols)
                bldg_agg_cols.remove('EGID')
                bldg_agg_meth = {col: 'first' for col in bldg_agg_cols}

                gwr_mrg['nEWID'] = gwr_mrg['EWID']
                def concat_strings(x):
                    return '_'.join(x.dropna().astype(str))
                dwel_agg_meth = {'EWID':concat_strings,'nEWID': 'count', 'WAZIM': 'sum', 'WAREA': 'sum'}

                agg_meth = {**bldg_agg_meth, **dwel_agg_meth}
                gwr_mrg_after_agg =           gwr_mrg.groupby('EGID').agg(agg_meth).reset_index()
                gwr_mrg_all_building_in_bfs = gwr_mrg.groupby('EGID').agg(agg_meth).reset_index()
                gwr_mrg = copy.deepcopy(gwr_mrg_after_agg)
            else: 
                gwr_mrg = gwr_all_building_df.copy()
                


            # filter for spces ------
            gwr_mrg0 = gwr_mrg.copy()  
            gwr_mrg0['GBAUJ'] = gwr_mrg0['GBAUJ'].replace('', 0).astype(int)

            # gwr_mrg0['GBAUJ'] = gwr_mrg0['GBAUJ'].replace('', 0).astype(int)
            # gwr_mrg1 = gwr_mrg0[(gwr_mrg0['GSTAT'].isin(self.sett.GWR_GSTAT))]
            # gwr_mrg2 = gwr_mrg1[(gwr_mrg1['GKLAS'].isin(self.sett.GWR_GKLAS))]
            # gwr_mrg3 = gwr_mrg2[(gwr_mrg2['GBAUJ'] >= self.sett.GWR_GBAUJ_minmax[0]) & (gwr_mrg2['GBAUJ'] <= self.sett.GWR_GBAUJ_minmax[1])]
            # gwr_pq = copy.deepcopy(gwr_mrg3)
            gwr_pq = gwr_mrg0[
                (gwr_mrg0['GSTAT'].isin(self.sett.GWR_GSTAT)) &
                (gwr_mrg0['GKLAS'].isin(self.sett.GWR_GKLAS)) 
                # (gwr_mrg0['GBAUJ'] >= self.sett.GWR_GBAUJ_minmax[0]) &
                # (gwr_mrg0['GBAUJ'] <= self.sett.GWR_GBAUJ_minmax[1])
            ].copy()  # shallow copy for the final result

            def gwr_to_gdf(df):
                df = copy.deepcopy(df)                
                df = df.loc[(df['GKODE'] != '') & (df['GKODN'] != '')]
                df[['GKODE', 'GKODN']] = df[['GKODE', 'GKODN']].astype(float)
                df['geometry'] = df.apply(lambda row: Point(row['GKODE'], row['GKODN']), axis=1)
                gdf = gpd.GeoDataFrame(df, geometry='geometry')
                gdf.crs = 'EPSG:2056'
                return gdf
            
            gwr_gdf = gwr_to_gdf(gwr_pq)
            gwr_all_building_gdf = gwr_to_gdf(gwr_all_building_df) if self.sett.export_gwr_ALL_building_gdf_TF else None

            log_time = self.write_to_logfile('local data imported - gwr merge + filter', log_time)
        
        log_time = self.write_to_logfile('local data imported - gwr, finished', log_time)


        # continue only if houses in BFS =-=-=-=
        if ((gwr_gdf.shape[0] > 0) & (solkat_gdf.shape[0] > 0)):


            # transformations + spatial mappings ====================
            # add omitted EGIDs to SOLKAT ------------------------------
            if True:
                if solkat_select_gdf.shape[0] > 0:
                    # the solkat df has missing EGIDs, for example row houses where the entire roof is attributed to one EGID. Attempt to 
                    # 1 - add roof (perfectly overlapping roofpartitions) to solkat for all the EGIDs within the unions shape
                    # 2- reduce the FLAECHE for all theses partitions by dividing it through the number of EGIDs in the union shape
                    cols_adjust_for_missEGIDs_to_solkat = self.sett.SOLKAT_cols_adjust_for_missEGIDs_to_solkat

                    # solkat_v2 = copy.deepcopy(solkat_all_pq[solkat_all_pq['BFS_NUMMER'].isin(self.sett.bfs_numbers)])
                    # solkat_v2_wgeo = solkat_v2.merge(solkat_all_geo[['DF_UID', 'geometry']], how = 'left', on = 'DF_UID')
                    # solkat_v2_gdf = gpd.GeoDataFrame(solkat_v2_wgeo, geometry='geometry')
                    solkat_v2_gdf = solkat_select_gdf.copy()
                    solkat_v2_gdf = solkat_v2_gdf[solkat_v2_gdf['EGID'] != 'NAN']
                        
                    # create mapping of solkatEGIDs and missing gwrEGIDs 
                    # union all shapes with the same EGID 
                    solkat_union_v2EGID = solkat_v2_gdf.groupby('EGID').agg({
                        'geometry': lambda x: unary_union(x),  # Combine all geometries into one MultiPolygon
                        'DF_UID': lambda x: '_'.join(map(str, x))  # Concatenate DF_UID values as a single string
                        }).reset_index()
                    solkat_union_v2EGID = gpd.GeoDataFrame(solkat_union_v2EGID, geometry='geometry')

                    solkat_union_v2EGID = solkat_union_v2EGID.rename(columns = {'EGID': 'EGID_old_solkat'})  # rename EGID colum because gwr_EGIDs are now matched to union_shapes
                    solkat_union_v2EGID.set_crs(gwr_gdf.crs, allow_override=True, inplace=True)
                    join_gwr_solkat_union = gpd.sjoin(solkat_union_v2EGID, gwr_gdf, how='left')
                    join_gwr_solkat_union.rename(columns = {'EGID': 'EGID_gwradded'}, inplace = True)

                    # check EGID mapping case by case, add missing gwrEGIDs to solkat -------------------
                    EGID_old_solkat_list = join_gwr_solkat_union['EGID_old_solkat'].unique()
                    new_solkat_append_list = []
                    add_solkat_counter, add_solkat_partition = 1, 4
                    print_counter_max, i_print = 5, 0

                    # debug_idx = 274
                    # n_egid, egid = debug_idx, EGID_old_solkat_list[debug_idx]
                    for n_egid, egid in enumerate(EGID_old_solkat_list):
                        egid_join_union = join_gwr_solkat_union.loc[join_gwr_solkat_union['EGID_old_solkat'] == egid,]
                        egid_join_union = egid_join_union.reset_index(drop = True)

                        # Shapes of building that will not be included given GWR filter settings
                        if any(egid_join_union['EGID_gwradded'].isna()):  
                            solkat_subdf = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                            solkat_subdf['DF_UID_solkat'] = solkat_subdf['DF_UID']

                        elif all(egid_join_union['EGID_gwradded'] != np.nan): 

                            # cases
                            case1_TF = (egid_join_union.shape[0] == 1) & (egid_join_union['EGID_gwradded'].values[0] == egid)
                            case2_TF = (egid_join_union.shape[0] == 1) & (egid_join_union['EGID_gwradded'].values[0] != egid)
                            case3_TF = (egid_join_union.shape[0] > 1) & any(egid_join_union['EGID_gwradded'].isna())
                            case4_TF = (egid_join_union.shape[0] > 1) & (egid in egid_join_union['EGID_gwradded'].to_list())
                            case5_TF = (egid_join_union.shape[0] > 1) & (egid not in egid_join_union['EGID_gwradded'].to_list())

                            # "Best" case (unless step above applies): Shapes of building that only has 1 GWR EGID
                            if case1_TF:        # (egid_join_union.shape[0] == 1) & (egid_join_union['EGID_gwradded'].values[0] == egid): 
                                solkat_subdf = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                                solkat_subdf['DF_UID_solkat'] = solkat_subdf['DF_UID']
                                
                            # Not best case but for consistency better to keep individual solkatEGIs matches (otherwise missmatch of newer buildings with old shape partitions possible)
                            elif case2_TF:      # (egid_join_union.shape[0] == 1) & (egid_join_union['EGID_gwradded'].values[0] != egid): 
                                solkat_subdf = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                                solkat_subdf['DF_UID_solkat'] = solkat_subdf['DF_UID']

                            elif case3_TF:      # (egid_join_union.shape[0] > 1) & any(egid_join_union['EGID_gwradded'].isna()):
                                print(f'**MAJOR ERROR**: EGID {egid}, np.nan in egid_join_union[EGID_gwradded] column')

                            # Intended case: Shapes of building that has multiple GWR EGIDs within the shape boundaries
                            elif case4_TF:      # (egid_join_union.shape[0] > 1) & (egid in egid_join_union['EGID_gwradded'].to_list()):
                                
                                solkat_subdf_addedEGID_list = []
                                n, egid_to_add = 0, egid_join_union['EGID_gwradded'].unique()[0]
                                for n, egid_to_add in enumerate(egid_join_union['EGID_gwradded'].unique()):
                                    
                                    # add all partitions given the "old EGID" & change EGID to the acutal identifier (if not egid_to_add in EGID_old_solkat_list:)
                                    solkat_addedEGID = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                                    solkat_addedEGID['DF_UID_solkat'] = solkat_addedEGID['DF_UID']
                                    solkat_addedEGID['EGID'] = egid_to_add
                                    
                                    #extend the DF_UID with some numbers to have truely unique DF_UIDs
                                    if self.sett.SOLKAT_extend_dfuid_for_missing_EGIDs_to_be_unique:
                                        str_suffix = str(n+1).zfill(5)
                                        if isinstance(solkat_addedEGID['DF_UID'].iloc[0], str):
                                            solkat_addedEGID['DF_UID'] = solkat_addedEGID['DF_UID'].apply(lambda x: f'{x}{str_suffix}')
                                        elif isinstance(solkat_addedEGID['DF_UID'].iloc[0], int):   
                                            solkat_addedEGID['DF_UID'] = solkat_addedEGID['DF_UID'].apply(lambda x: int(f'{x}{str_suffix}'))

                                    # divide certain columns by the number of EGIDs in the union shape (e.g. FLAECHE)
                                    for col in cols_adjust_for_missEGIDs_to_solkat:
                                        solkat_addedEGID[col] =  solkat_addedEGID[col] / egid_join_union.shape[0]
                                    
                                    # shrink topology to see which partitions are affected by EGID extensions
                                    # solkat_addedEGID['geometry'] =solkat_addedEGID['geometry'].buffer(-0.5, resolution=16)

                                    solkat_subdf_addedEGID_list.append(solkat_addedEGID)
                                
                                # concat all EGIDs within the same shape that were previously missing
                                solkat_subdf = pd.concat(solkat_subdf_addedEGID_list, ignore_index=True)
                                
                            # Error case: Shapes of building that has multiple gwrEGIDs but does not overlap with the assigned / identical solkatEGID. 
                            # Not proper solution, but best for now: add matching gwrEGID to solkatEGID selection, despite the acutall gwrEGID being placed in another shape. 
                            elif case5_TF:      # (egid_join_union.shape[0] > 1) & (egid not in egid_join_union['EGID_gwradded'].to_list()):

                                # attach a copy of one solkatEGID partition and set the EGID to the gwrEGID
                                gwrEGID_row = copy.deepcopy(egid_join_union.iloc[0])
                                # solkat_addedEGID['DF_UID_solkat'] = solkat_addedEGID['DF_UID']
                                gwrEGID_row['EGID_gwradded'] = egid
                                egid_join_union = pd.concat([egid_join_union, gwrEGID_row.to_frame().T], ignore_index=True)

                                # next follow all steps as in "Intended Case" above (solkat_shape with solkatEGID and gwrEGIDs)
                                solkat_subdf_addedEGID_list = []
                                n, egid_to_add = 0, egid_join_union['EGID_gwradded'].unique()[0]
                                
                                for n, egid_to_add in enumerate(egid_join_union['EGID_gwradded'].unique()):

                                    # add all partitions given the "old EGID" & change EGID to the acutal identifier (if not egid_to_add in EGID_old_solkat_list:)
                                    solkat_addedEGID = copy.deepcopy(solkat_v2_gdf.loc[solkat_v2_gdf['EGID'] == egid,])
                                    solkat_addedEGID['EGID'] = egid_to_add
                                    
                                    #extend the DF_UID with some numbers to have truely unique DF_UIDs
                                    if self.sett.SOLKAT_extend_dfuid_for_missing_EGIDs_to_be_unique:
                                        str_suffix = str(n+1).zfill(3)
                                        if isinstance(solkat_addedEGID['DF_UID'].iloc[0], str):
                                            solkat_addedEGID['DF_UID'] = solkat_addedEGID['DF_UID'].apply(lambda x: f'{x}{str_suffix}')
                                        elif isinstance(solkat_addedEGID['DF_UID'].iloc[0], int):   
                                            solkat_addedEGID['DF_UID'] = solkat_addedEGID['DF_UID'].apply(lambda x: int(f'{x}{str_suffix}'))

                                    # divide certain columns by the number of EGIDs in the union shape (e.g. FLAECHE)
                                    for col in cols_adjust_for_missEGIDs_to_solkat:
                                        solkat_addedEGID[col] =  solkat_addedEGID[col] / egid_join_union.shape[0]
                                    
                                    # shrink topology to see which partitions are affected by EGID extensions
                                    # solkat_addedEGID['geometry'] =solkat_addedEGID['geometry'].buffer(-0.5, resolution=16)

                                    solkat_subdf_addedEGID_list.append(solkat_addedEGID)
                                
                                # concat all EGIDs within the same shape that were previously missing
                                solkat_subdf = pd.concat(solkat_subdf_addedEGID_list, ignore_index=True)

                                if i_print < print_counter_max:
                                    print(f'ERROR: EGID {egid}: multiple gwrEGIDs, outside solkatEGID / without solkatEGID amongst them')
                                    i_print += 1
                                elif i_print == print_counter_max:
                                    print(f'ERROR: EGID {egid}: {print_counter_max}+ ... more cases of multiple gwrEGIDs, outside solkatEGID / without solkatEGID amongst them')
                                    i_print += 1

                        if n_egid == int(len(EGID_old_solkat_list)/add_solkat_partition):
                            print(f'Match gwrEGID to solkat: {add_solkat_counter}/{add_solkat_partition} partition')
                
                        # merge all solkat partitions to new solkat df
                        new_solkat_append_list.append(solkat_subdf) 

                    new_solkat_gdf = gpd.GeoDataFrame(pd.concat(new_solkat_append_list, ignore_index=True), geometry='geometry')
                    new_solkat = new_solkat_gdf.drop(columns = ['geometry'])

                    solkat_pq, solkat_gdf = copy.deepcopy(new_solkat), copy.deepcopy(new_solkat_gdf)


                # other local data ------------------------------
                gemeinde_type_gdf = gpd.read_file(f'{self.sett.data_path}/input/gemeindetypen_2056.gpkg/gemeindetypen_2056.gpkg', layer=None)



                # MAP: egid > pv ---------------------------------------------------------------------------------
                # gwr_buff_gdf = copy.deepcopy(gwr_all_building_gdf)
                gwr_buff_gdf = copy.deepcopy(gwr_gdf)
                gwr_buff_gdf.set_crs("EPSG:32632", allow_override=True, inplace=True)
                gwr_buff_gdf['geometry'] = gwr_buff_gdf['geometry'].buffer(self.sett.SOLKAT_GWR_EGID_buffer_size)
                gwr_buff_gdf, pv_gdf = set_crs_to_gm_shp(gm_shp_gdf, gwr_buff_gdf, pv_gdf)

                gwregid_pvid_join = gpd.sjoin(pv_gdf,gwr_buff_gdf, how="left", predicate="within")
                gwregid_pvid_join.drop(columns = ['index_right'] + [col for col in gwr_buff_gdf.columns if col not in ['EGID', 'geometry']], inplace = True)

                # keep only unique xtf_ids 
                gwregid_pvid_unique = copy.deepcopy(gwregid_pvid_join.loc[~gwregid_pvid_join.duplicated(subset='xtf_id', keep=False)])
                xtf_duplicates =      copy.deepcopy(gwregid_pvid_join.loc[ gwregid_pvid_join.duplicated(subset='xtf_id', keep=False)])
            
                if xtf_duplicates.shape[0] > 0:
                    # match duplicates with nearest neighbour
                    xtf_nearestmatch_list = []
                    # xtf_id = xtf_duplicates['xtf_id'].unique()[0]
                    for xtf_id in xtf_duplicates['xtf_id'].unique():
                        gwr_sub = copy.deepcopy(gwr_buff_gdf.loc[gwr_buff_gdf['EGID'].isin(xtf_duplicates.loc[xtf_duplicates['xtf_id'] == xtf_id, 'EGID'])])
                        pv_sub = copy.deepcopy(pv_gdf.loc[pv_gdf['xtf_id'] == xtf_id])
                        
                        assert pv_sub.crs == gwr_sub.crs
                        gwr_sub['distance_to_pv'] = gwr_sub['geometry'].centroid.distance(pv_sub['geometry'].values[0])
                        pv_sub['EGID'] = gwr_sub.loc[gwr_sub['distance_to_pv'].idxmin()]['EGID']

                        xtf_nearestmatch_list.append(pv_sub)
                    
                    xtf_nearestmatches_df = pd.concat(xtf_nearestmatch_list, ignore_index=True)
                    gwregid_pvid = pd.concat([gwregid_pvid_unique, xtf_nearestmatches_df], ignore_index=True).drop_duplicates()

                else: 
                    gwregid_pvid = gwregid_pvid_unique


                # Map_egid_pv = gwregid_pvid.loc[gwregid_pvid['EGID'].notna(), ['EGID', 'xtf_id']].copy()
                Map_egid_pv = gwregid_pvid[['EGID', 'xtf_id', 'BFS_NUMMER']].copy()
                Map_egid_pv = Map_egid_pv.dropna(subset=['EGID'])


                # MAP: BFSGM > EWR ------------------------------
                Map_gm_ewr = pd.read_parquet(f'{self.sett.data_path}/input_api/Map_gm_ewr.parquet')
            
                log_time = self.write_to_logfile('\nadd omitted EGIDs to SOLKAT, finished', log_time) 


            # import ts data and match households ====================
            if True: 
                # demand data ------------------------------
                if self.sett.DEMAND_input_data_source == "SwissStore" :
                    swstore_arch_typ_factors  = pd.read_excel(f'{self.sett.data_path}/input/SwissStore_DemandData/12.swisstore_table12_unige.xlsx', sheet_name='Feuil1')
                    swstore_arch_typ_master   = pd.read_csv(f'{self.sett.data_path}/input/SwissStore_DemandData/Master_table_archetype.csv', sep=';')
                    swstore_sfhmfh_ts         = pd.read_excel(f'{self.sett.data_path}/input/SwissStore_DemandData/Electricity_demand_SFH_MFH.xlsx', sheet_name='dmnd_prof_sfh_mfh_avg')
                    
                    # gwr                       = pd.read_parquet(f'{self.sett.preprep_path}/gwr.parquet')
                    # gwr_all_building_gdf      = gpd.read_file(f'{self.sett.preprep_path}/gwr_all_building_gdf.geojson')
                    # gemeinde_type_gdf         = gpd.read_file(f'{self.sett.preprep_path}/gemeinde_type_gdf.geojson')

                    # classify EGIDs into SFH / MFH, Rural / Urban -------------

                    gwr_join_gdf = gwr_gdf.copy()

                    # get ARE type classification
                    gwr_join_gdf['ARE_typ'] = ''
                    gwr_join_gdf = gpd.sjoin(gwr_join_gdf, gemeinde_type_gdf[['NAME', 'TYP', 'BFS_NO', 'geometry']],
                                            how='left', predicate='intersects')
                    # gemeinde_type_gdf['BFS_NO'] = gemeinde_type_gdf['BFS_NO'].astype(str)
                    # gwr_join_gdf = gwr_join_gdf.merge(gemeinde_type_gdf[['NAME', 'TYP', 'BFS_NO']], left_on='GGDENR', right_on='BFS_NO', how='left')

                    gwr_join_gdf.rename(columns={'NAME': 'ARE_NAME', 'TYP': 'ARE_TYP', }, inplace=True)
                    for k,v in self.sett.GWR_AREtypology.items():
                        gwr_join_gdf.loc[gwr_join_gdf['ARE_TYP'].isin(v), 'ARE_typ'] = k

                    # get SFH / MFH classification from GWR data
                    gwr_join_gdf['sfhmfh_typ'] = ''
                    for k,v in self.sett.GWR_SFHMFHtypology.items():
                        gwr_join_gdf.loc[gwr_join_gdf['GKLAS'].isin(v), 'sfhmfh_typ'] = k
                    gwr_join_gdf.loc[gwr_join_gdf['sfhmfh_typ'] == '', 'sfhmfh_typ'] = self.sett.GWR_SFHMFH_outsample_proxy

                    # build swstore_type to attach swstore factors
                    gwr_join_gdf['arch_typ'] = gwr_join_gdf['sfhmfh_typ'].str.cat(gwr_join_gdf['ARE_typ'], sep='-')
                    gwr_join_gdf = gwr_join_gdf.merge(swstore_arch_typ_factors[['arch_typ', 'elec_dem_ind_cecb', ]])
                    gwr_join_gdf.rename(columns={'elec_dem_ind_cecb': 'demand_elec_pGAREA'}, inplace=True)

                    # attach information to gwr and export
                    # gwr_pq = gwr_pq.merge(gwr_join_gdf[['EGID', 'ARE_typ', 'sfhmfh_typ', 'arch_typ', 'demand_elec_pGAREA']], on='EGID', how='left')
                    # gwr_all_building_df = gwr_join_gdf.drop(columns=['geometry', ]).copy()
                    gwr_gdf = gwr_join_gdf.copy()
                    gwr_pq  = gwr_join_gdf.drop(columns=['geometry', ]).copy()

                    # transform demand profiles to TS 
                    swstore_sfhmfh_ts = swstore_sfhmfh_ts.dropna(subset=['MFH', 'SFH'], how='all')
                    swstore_sfhmfh_ts['t'] = [f't_{i+1}' for i in range(len(swstore_sfhmfh_ts))]
                    swstore_sfhmfh_ts['t_int'] = [i+1 for i in range(len(swstore_sfhmfh_ts))]
                    demandtypes_ts = copy.deepcopy(swstore_sfhmfh_ts)


                # meteo data ------------------------------
                if True: 
                    meteo = pd.read_csv(f'{self.sett.data_path}/input/Meteoblue_BSBL/Meteodaten_Basel_2018_2024_reduziert_bereinigt.csv')

                    # transformations
                    meteo['timestamp'] = pd.to_datetime(meteo['timestamp'], format = '%d.%m.%Y %H:%M:%S')

                    # select relevant time frame
                    # start_stamp = pd.to_datetime(f'01.01.{self.sett.year_range[0]}', format = '%d.%m.%Y')
                    # end_stamp = pd.to_datetime(f'31.12.{self.sett.year_range[1]}', format = '%d.%m.%Y')
                    # meteo = meteo[(meteo['timestamp'] >= start_stamp) & (meteo['timestamp'] <= end_stamp)]
                

                # angle tilt reduction ------------------------------
                if True:
                    index_angle = [-180, -170, -160, -150, -140, -130, -120, -110, -100, -90, -80, -70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180]
                    index_tilt = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]
                    tuples_iter = list(itertools.product(index_angle, index_tilt))

                    tuples = [(-180, 0), (-180, 5), (-180, 10), (-180, 15), (-180, 20), (-180, 25), (-180, 30), (-180, 35), (-180, 40), (-180, 45), (-180, 50), (-180, 55), (-180, 60), (-180, 65), (-180, 70), (-180, 75), (-180, 80), (-180, 85), (-180, 90), 
                            (-170, 0), (-170, 5), (-170, 10), (-170, 15), (-170, 20), (-170, 25), (-170, 30), (-170, 35), (-170, 40), (-170, 45), (-170, 50), (-170, 55), (-170, 60), (-170, 65), (-170, 70), (-170, 75), (-170, 80), (-170, 85), (-170, 90), 
                            (-160, 0), (-160, 5), (-160, 10), (-160, 15), (-160, 20), (-160, 25), (-160, 30), (-160, 35), (-160, 40), (-160, 45), (-160, 50), (-160, 55), (-160, 60), (-160, 65), (-160, 70), (-160, 75), (-160, 80), (-160, 85), (-160, 90), 
                            (-150, 0), (-150, 5), (-150, 10), (-150, 15), (-150, 20), (-150, 25), (-150, 30), (-150, 35), (-150, 40), (-150, 45), (-150, 50), (-150, 55), (-150, 60), (-150, 65), (-150, 70), (-150, 75), (-150, 80), (-150, 85), (-150, 90), 
                            (-140, 0), (-140, 5), (-140, 10), (-140, 15), (-140, 20), (-140, 25), (-140, 30), (-140, 35), (-140, 40), (-140, 45), (-140, 50), (-140, 55), (-140, 60), (-140, 65), (-140, 70), (-140, 75), (-140, 80), (-140, 85), (-140, 90),
                            (-130, 0), (-130, 5), (-130, 10), (-130, 15), (-130, 20), (-130, 25), (-130, 30), (-130, 35), (-130, 40), (-130, 45), (-130, 50), (-130, 55), (-130, 60), (-130, 65), (-130, 70), (-130, 75), (-130, 80), (-130, 85), (-130, 90),
                            (-120, 0), (-120, 5), (-120, 10), (-120, 15), (-120, 20), (-120, 25), (-120, 30), (-120, 35), (-120, 40), (-120, 45), (-120, 50), (-120, 55), (-120, 60), (-120, 65), (-120, 70), (-120, 75), (-120, 80), (-120, 85), (-120, 90),
                            (-110, 0), (-110, 5), (-110, 10), (-110, 15), (-110, 20), (-110, 25), (-110, 30), (-110, 35), (-110, 40), (-110, 45), (-110, 50), (-110, 55), (-110, 60), (-110, 65), (-110, 70), (-110, 75), (-110, 80), (-110, 85), (-110, 90),
                            (-100, 0), (-100, 5), (-100, 10), (-100, 15), (-100, 20), (-100, 25), (-100, 30), (-100, 35), (-100, 40), (-100, 45), (-100, 50), (-100, 55), (-100, 60), (-100, 65), (-100, 70), (-100, 75), (-100, 80), (-100, 85), (-100, 90),
                            (-90, 0), (-90, 5), (-90, 10), (-90, 15), (-90, 20), (-90, 25), (-90, 30), (-90, 35), (-90, 40), (-90, 45), (-90, 50), (-90, 55), (-90, 60), (-90, 65), (-90, 70), (-90, 75), (-90, 80), (-90, 85), (-90, 90),
                            (-80, 0), (-80, 5), (-80, 10), (-80, 15), (-80, 20), (-80, 25), (-80, 30), (-80, 35), (-80, 40), (-80, 45), (-80, 50), (-80, 55), (-80, 60), (-80, 65), (-80, 70), (-80, 75), (-80, 80), (-80, 85), (-80, 90),
                            (-70, 0), (-70, 5), (-70, 10), (-70, 15), (-70, 20), (-70, 25), (-70, 30), (-70, 35), (-70, 40), (-70, 45), (-70, 50), (-70, 55), (-70, 60), (-70, 65), (-70, 70), (-70, 75), (-70, 80), (-70, 85), (-70, 90),
                            (-60, 0), (-60, 5), (-60, 10), (-60, 15), (-60, 20), (-60, 25), (-60, 30), (-60, 35), (-60, 40), (-60, 45), (-60, 50), (-60, 55), (-60, 60), (-60, 65), (-60, 70), (-60, 75), (-60, 80), (-60, 85), (-60, 90),
                            (-50, 0), (-50, 5), (-50, 10), (-50, 15), (-50, 20), (-50, 25), (-50, 30), (-50, 35), (-50, 40), (-50, 45), (-50, 50), (-50, 55), (-50, 60), (-50, 65), (-50, 70), (-50, 75), (-50, 80), (-50, 85), (-50, 90),
                            (-40, 0), (-40, 5), (-40, 10), (-40, 15), (-40, 20), (-40, 25), (-40, 30), (-40, 35), (-40, 40), (-40, 45), (-40, 50), (-40, 55), (-40, 60), (-40, 65), (-40, 70), (-40, 75), (-40, 80), (-40, 85), (-40, 90),
                            (-30, 0), (-30, 5), (-30, 10), (-30, 15), (-30, 20), (-30, 25), (-30, 30), (-30, 35), (-30, 40), (-30, 45), (-30, 50), (-30, 55), (-30, 60), (-30, 65), (-30, 70), (-30, 75), (-30, 80), (-30, 85), (-30, 90),
                            (-20, 0), (-20, 5), (-20, 10), (-20, 15), (-20, 20), (-20, 25), (-20, 30), (-20, 35), (-20, 40), (-20, 45), (-20, 50), (-20, 55), (-20, 60), (-20, 65), (-20, 70), (-20, 75), (-20, 80), (-20, 85), (-20, 90),
                            (-10, 0), (-10, 5), (-10, 10), (-10, 15), (-10, 20), (-10, 25), (-10, 30), (-10, 35), (-10, 40), (-10, 45), (-10, 50), (-10, 55), (-10, 60), (-10, 65), (-10, 70), (-10, 75), (-10, 80), (-10, 85), (-10, 90),
                            (0, 0), (0, 5), (0, 10), (0, 15), (0, 20), (0, 25), (0, 30), (0, 35), (0, 40), (0, 45), (0, 50), (0, 55), (0, 60), (0, 65), (0, 70), (0, 75), (0, 80), (0, 85), (0, 90),
                            (10, 0), (10, 5), (10, 10), (10, 15), (10, 20), (10, 25), (10, 30), (10, 35), (10, 40), (10, 45), (10, 50), (10, 55), (10, 60), (10, 65), (10, 70), (10, 75), (10, 80), (10, 85), (10, 90),
                            (20, 0), (20, 5), (20, 10), (20, 15), (20, 20), (20, 25), (20, 30), (20, 35), (20, 40), (20, 45), (20, 50), (20, 55), (20, 60), (20, 65), (20, 70), (20, 75), (20, 80), (20, 85), (20, 90),
                            (30, 0), (30, 5), (30, 10), (30, 15), (30, 20), (30, 25), (30, 30), (30, 35), (30, 40), (30, 45), (30, 50), (30, 55), (30, 60), (30, 65), (30, 70), (30, 75), (30, 80), (30, 85), (30, 90),
                            (40, 0), (40, 5), (40, 10), (40, 15), (40, 20), (40, 25), (40, 30), (40, 35), (40, 40), (40, 45), (40, 50), (40, 55), (40, 60), (40, 65), (40, 70), (40, 75), (40, 80), (40, 85), (40, 90),
                            (50, 0), (50, 5), (50, 10), (50, 15), (50, 20), (50, 25), (50, 30), (50, 35), (50, 40), (50, 45), (50, 50), (50, 55), (50, 60), (50, 65), (50, 70), (50, 75), (50, 80), (50, 85), (50, 90),
                            (60, 0), (60, 5), (60, 10), (60, 15), (60, 20), (60, 25), (60, 30), (60, 35), (60, 40), (60, 45), (60, 50), (60, 55), (60, 60), (60, 65), (60, 70), (60, 75), (60, 80), (60, 85), (60, 90),
                            (70, 0), (70, 5), (70, 10), (70, 15), (70, 20), (70, 25), (70, 30), (70, 35), (70, 40), (70, 45), (70, 50), (70, 55), (70, 60), (70, 65), (70, 70), (70, 75), (70, 80), (70, 85), (70, 90),
                            (80, 0), (80, 5), (80, 10), (80, 15), (80, 20), (80, 25), (80, 30), (80, 35), (80, 40), (80, 45), (80, 50), (80, 55), (80, 60), (80, 65), (80, 70), (80, 75), (80, 80), (80, 85), (80, 90),
                            (90, 0), (90, 5), (90, 10), (90, 15), (90, 20), (90, 25), (90, 30), (90, 35), (90, 40), (90, 45), (90, 50), (90, 55), (90, 60), (90, 65), (90, 70), (90, 75), (90, 80), (90, 85), (90, 90),
                            (100, 0), (100, 5), (100, 10), (100, 15), (100, 20), (100, 25), (100, 30), (100, 35), (100, 40), (100, 45), (100, 50), (100, 55), (100, 60), (100, 65), (100, 70), (100, 75), (100, 80), (100, 85), (100, 90),
                            (110, 0), (110, 5), (110, 10), (110, 15), (110, 20), (110, 25), (110, 30), (110, 35), (110, 40), (110, 45), (110, 50), (110, 55), (110, 60), (110, 65), (110, 70), (110, 75), (110, 80), (110, 85), (110, 90),
                            (120, 0), (120, 5), (120, 10), (120, 15), (120, 20), (120, 25), (120, 30), (120, 35), (120, 40), (120, 45), (120, 50), (120, 55), (120, 60), (120, 65), (120, 70), (120, 75), (120, 80), (120, 85), (120, 90),
                            (130, 0), (130, 5), (130, 10), (130, 15), (130, 20), (130, 25), (130, 30), (130, 35), (130, 40), (130, 45), (130, 50), (130, 55), (130, 60), (130, 65), (130, 70), (130, 75), (130, 80), (130, 85), (130, 90),
                            (140, 0), (140, 5), (140, 10), (140, 15), (140, 20), (140, 25), (140, 30), (140, 35), (140, 40), (140, 45), (140, 50), (140, 55), (140, 60), (140, 65), (140, 70), (140, 75), (140, 80), (140, 85), (140, 90),
                            (150, 0), (150, 5), (150, 10), (150, 15), (150, 20), (150, 25), (150, 30), (150, 35), (150, 40), (150, 45), (150, 50), (150, 55), (150, 60), (150, 65), (150, 70), (150, 75), (150, 80), (150, 85), (150, 90),
                            (160, 0), (160, 5), (160, 10), (160, 15), (160, 20), (160, 25), (160, 30), (160, 35), (160, 40), (160, 45), (160, 50), (160, 55), (160, 60), (160, 65), (160, 70), (160, 75), (160, 80), (160, 85), (160, 90),
                            (170, 0), (170, 5), (170, 10), (170, 15), (170, 20), (170, 25), (170, 30), (170, 35), (170, 40), (170, 45), (170, 50), (170, 55), (170, 60), (170, 65), (170, 70), (170, 75), (170, 80), (170, 85), (170, 90),
                            (180, 0), (180, 5), (180, 10), (180, 15), (180, 20), (180, 25), (180, 30), (180, 35), (180, 40), (180, 45), (180, 50), (180, 55), (180, 60), (180, 65), (180, 70), (180, 75), (180, 80), (180, 85), (180, 90)
                            ]
                    index = pd.MultiIndex.from_tuples(tuples, names=['angle', 'tilt'])

                    values = [89.0, 85.5, 81.5, 77.3, 72.7, 68.3, 64.0, 59.8, 55.6, 51.5, 47.6, 44.1, 40.7, 37.9, 35.8, 34.1, 32.7, 31.4, 30.2, 
                            89.0, 85.5, 81.6, 77.4, 72.9, 68.5, 64.2, 60.0, 55.9, 51.9, 48.1, 44.5, 41.2, 38.5, 36.4, 34.8, 33.3, 31.9, 30.7, 
                            89.0, 85.7, 81.9, 77.8, 73.5, 69.2, 65.0, 60.9, 56.9, 53.0, 49.4, 46.0, 42.9, 40.6, 38.6, 36.8, 35.2, 33.7, 32.2, 
                            89.0, 85.9, 82.4, 78.6, 74.6, 70.5, 66.4, 62.5, 58.7, 55.0, 51.6, 48.6, 46.1, 43.8, 41.7, 39.8, 38.0, 36.3, 34.6, 
                            89.0, 86.3, 83.1, 79.6, 75.9, 72.2, 68.4, 64.8, 61.3, 58.1, 55.1, 52.4, 49.9, 47.6, 45.4, 43.3, 41.3, 39.4, 37.5, 
                            89.0, 86.7, 84.0, 80.8, 77.7, 74.3, 71.1, 67.8, 64.8, 61.9, 59.1, 56.5, 54.1, 51.8, 49.4, 47.2, 45.0, 42.8, 40.7, 
                            89.0, 87.1, 84.9, 82.4, 79.6, 76.8, 74.0, 71.3, 68.6, 66.0, 63.4, 61.0, 58.6, 56.2, 53.8, 51.4, 49.0, 46.6, 44.2, 
                            89.0, 87.7, 85.9, 84.0, 81.8, 79.5, 77.2, 74.9, 72.5, 70.2, 67.9, 65.5, 63.1, 60.7, 58.8, 55.7, 53.1, 50.6, 48.0, 
                            89.0, 88.3, 87.1, 85.6, 84.0, 82.2, 80.4, 78.5, 76.5, 74.4, 72.2, 69.9, 67.6, 65.2, 62.7, 60.1, 57.3, 54.5, 51.8, 
                            89.0, 88.8, 88.2, 87.3, 86.2, 84.9, 83.6, 82.0, 80.3, 78.4, 76.4, 74.3, 71.9, 69.5, 66.8, 64.1, 61.3, 58.3, 55.2, 
                            89.0, 89.4, 89.3, 89.0, 88.4, 87.6, 86.6, 85.4, 84.0, 82.3, 80.4, 78.3, 75.9, 73.4, 70.9, 67.9, 64.8, 61.8, 58.5, 
                            89.0, 89.9, 90.5, 90.6, 90.5, 90.1, 89.5, 88.6, 87.3, 85.8, 84.0, 82.0, 79.7, 77.1, 74.3, 71.4, 68.2, 64.7, 61.3, 
                            89.0, 90.5, 91.4, 92.1, 92.4, 92.4, 92.1, 91.4, 90.4, 89.0, 87.4, 85.2, 83.0, 80.5, 77.5, 74.3, 71.0, 67.4, 63.7, 
                            89.0, 90.9, 92.4, 93.5, 94.2, 94.5, 94.4, 93.9, 93.0, 91.7, 90.2, 88.3, 85.8, 83.1, 80.2, 76.9, 73.3, 69.5, 65.6, 
                            89.0, 91.4, 93.2, 94.6, 95.6, 96.2, 96.4, 96.1, 95.4, 94.2, 92.5, 90.6, 88.3, 85.5, 82.3, 78.9, 75.2, 71.2, 66.9, 
                            89.0, 91.7, 93.9, 95.5, 96.8, 97.7, 98.0, 97.7, 97.1, 96.1, 94.5, 92.5, 90.0, 87.1, 84.0, 80.4, 76.4, 72.2, 67.8, 
                            89.0, 91.9, 94.3, 96.3, 97.7, 98.6, 99.1, 99.0, 98.5, 97.4, 95.8, 93.8, 91.4, 88.4, 85.0, 81.3, 77.2, 72.8, 68.1, 
                            89.0, 92.1, 94.6, 96.7, 98.2, 99.2, 99.8, 99.8, 99.3, 98.3, 96.7, 94.6, 92.0, 89.0, 85.5, 81.8, 77.5, 73.0, 68.2, 
                            89.0, 92.1, 94.7, 96.8, 98.4, 99.5, 100,  100 , 99.5, 98.3, 96.8, 94.8, 92.3, 89.3, 85.8, 81.9, 77.6, 73.1, 68.1,
                            89.0, 92.1, 94.6, 96.7, 98.2, 99.2, 99.8, 99.8, 99.3, 98.3, 96.7, 94.6, 92.0, 89.0, 85.5, 81.8, 77.5, 73.0, 68.2, 
                            89.0, 91.9, 94.3, 96.3, 97.7, 98.6, 99.1, 99.0, 98.5, 97.4, 95.8, 93.8, 91.4, 88.4, 85.0, 81.3, 77.2, 72.8, 68.1, 
                            89.0, 91.7, 93.9, 95.5, 96.8, 97.7, 98.0, 97.7, 97.1, 96.1, 94.5, 92.5, 90.0, 87.1, 84.0, 80.4, 76.4, 72.2, 67.8, 
                            89.0, 91.4, 93.2, 94.6, 95.6, 96.2, 96.4, 96.1, 95.4, 94.2, 92.5, 90.6, 88.3, 85.5, 82.3, 78.9, 75.2, 71.2, 66.9, 
                            89.0, 90.9, 92.4, 93.5, 94.2, 94.5, 94.4, 93.9, 93.0, 91.7, 90.2, 88.3, 85.8, 83.1, 80.2, 76.9, 73.3, 69.5, 65.6, 
                            89.0, 90.5, 91.4, 92.1, 92.4, 92.4, 92.1, 91.4, 90.4, 89.0, 87.4, 85.2, 83.0, 80.5, 77.5, 74.3, 71.0, 67.4, 63.7, 
                            89.0, 89.9, 90.5, 90.6, 90.5, 90.1, 89.5, 88.6, 87.3, 85.8, 84.0, 82.0, 79.7, 77.1, 74.3, 71.4, 68.2, 64.7, 61.3, 
                            89.0, 89.4, 89.3, 89.0, 88.4, 87.6, 86.6, 85.4, 84.0, 82.3, 80.4, 78.3, 75.9, 73.4, 70.9, 67.9, 64.8, 61.8, 58.5, 
                            89.0, 88.8, 88.2, 87.3, 86.2, 84.9, 83.6, 82.0, 80.3, 78.4, 76.4, 74.3, 71.9, 69.5, 66.8, 64.1, 61.3, 58.3, 55.2, 
                            89.0, 88.3, 87.1, 85.6, 84.0, 82.2, 80.4, 78.5, 76.5, 74.4, 72.2, 69.9, 67.6, 65.2, 62.7, 60.1, 57.3, 54.5, 51.8, 
                            89.0, 87.7, 85.9, 84.0, 81.8, 79.5, 77.2, 74.9, 72.5, 70.2, 67.9, 65.5, 63.1, 60.7, 58.8, 55.7, 53.1, 50.6, 48.0, 
                            89.0, 87.1, 84.9, 82.4, 79.6, 76.8, 74.0, 71.3, 68.6, 66.0, 63.4, 61.0, 58.6, 56.2, 53.8, 51.4, 49.0, 46.6, 44.2, 
                            89.0, 86.7, 84.0, 80.8, 77.7, 74.3, 71.1, 67.8, 64.8, 61.9, 59.1, 56.5, 54.1, 51.8, 49.4, 47.2, 45.0, 42.8, 40.7, 
                            89.0, 86.3, 83.1, 79.6, 75.9, 72.2, 68.4, 64.8, 61.3, 58.1, 55.1, 52.4, 49.9, 47.6, 45.4, 43.3, 41.3, 39.4, 37.5, 
                            89.0, 85.9, 82.4, 78.6, 74.6, 70.5, 66.4, 62.5, 58.7, 55.0, 51.6, 48.6, 46.1, 43.8, 41.7, 39.8, 38.0, 36.3, 34.6, 
                            89.0, 85.7, 81.9, 77.8, 73.5, 69.2, 65.0, 60.9, 56.9, 53.0, 49.4, 46.0, 42.9, 40.6, 38.6, 36.8, 35.2, 33.7, 32.2, 
                            89.0, 85.5, 81.6, 77.4, 72.9, 68.5, 64.2, 60.0, 55.9, 51.9, 48.1, 44.5, 41.2, 38.5, 36.4, 34.8, 33.3, 31.9, 30.7, 
                            89.0, 85.5, 81.5, 77.3, 72.7, 68.3, 64.0, 59.8, 55.6, 51.5, 47.6, 44.1, 40.7, 37.9, 35.8, 34.1, 32.7, 31.4, 30.2
                            ] 
                    
                    angle_tilt_df = pd.DataFrame(data = values, index = index, columns = ['efficiency_factor'])
                    angle_tilt_df['efficiency_factor'] = angle_tilt_df['efficiency_factor'] / 100
                    angle_tilt_df = angle_tilt_df.reset_index()


                # HOY weateryear --------------------------------
                if True:
                    HOY_weatheryear_df = pd.DataFrame({'timestamp': pd.date_range(start=f'{self.sett.WEAspec_weather_year}-01-01 00:00:00',end=f'{self.sett.WEAspec_weather_year}-12-31 23:00:00', freq='h')})
                    HOY_weatheryear_df['t'] = HOY_weatheryear_df.index.to_series().apply(lambda idx: f't_{idx + 1}')        
                    HOY_weatheryear_df['month'] = HOY_weatheryear_df['timestamp'].dt.month
                    HOY_weatheryear_df['day'] = HOY_weatheryear_df['timestamp'].dt.day
                    HOY_weatheryear_df['hour'] = HOY_weatheryear_df['timestamp'].dt.hour


                # elecpri + pvtarif --------------------------------
                elecpri_all = pd.read_parquet(f'{self.sett.data_path}/input/ElCom_consum_price_api_data/elecpri.parquet')
                elecpri = elecpri_all.loc[elecpri_all['bfs_number'].isin(self.sett.bfs_numbers)]

                pvtarif_all = pd.read_parquet(f'{self.sett.data_path}/input_api/pvtarif.parquet')
                # year_range_2int = [str(year % 100).zfill(2) for year in range(self.sett.year_range[0], self.sett.year_range[1]+1)]
                # pvtarif = copy.deepcopy(pvtarif_all.loc[pvtarif_all['year'].isin(year_range_2int), :])
                pvtarif = copy.deepcopy(pvtarif_all)
            log_time = self.write_to_logfile('import ts data and match households, finished', log_time)


            # export and store to class ====================
            export_pq_list = [
                ('pv_pq', pv_pq),
                ('solkat_pq', solkat_pq),
                ('solkat_month_pq', solkat_month_pq),
                ('gwr_pq', gwr_pq),
                ('Map_egid_pv', Map_egid_pv),
                ('Map_gm_ewr', Map_gm_ewr),
                ('demandtypes_ts', demandtypes_ts),
                ('swstore_arch_typ_factors', swstore_arch_typ_factors),
                ('swstore_arch_typ_master', swstore_arch_typ_master),
                ('meteo', meteo),
                ('angle_tilt_df', angle_tilt_df),
                ('HOY_weatheryear_df', HOY_weatheryear_df),
                ('elecpri', elecpri),
                ('pvtarif', pvtarif),
            ]
            for name, df in export_pq_list:
                df.to_parquet(f'{self.sett.calib_scen_preprep_path}/{name_preprep_subsen}_{name}.parquet', index=False)
                df.to_csv(f'{self.sett.calib_scen_preprep_path}/{name_preprep_subsen}_{name}.csv', index=False)

            export_gdf_list = [
                ('gm_shp_gdf', gm_shp_gdf),
                ('pv_gdf', pv_gdf),
                ('solkat_gdf', solkat_gdf),
                ('gwr_gdf', gwr_gdf),
                ('gemeinde_type_gdf', gemeinde_type_gdf),               
            ]
            if self.sett.export_gwr_ALL_building_gdf_TF: 
                export_gdf_list =  export_gdf_list + [('gwr_all_building_gdf', gwr_all_building_gdf),]

            for name, gdf in export_gdf_list:
                with open(f'{self.sett.calib_scen_preprep_path}/{name_preprep_subsen}_{name}.geojson', 'w') as f:
                    f.write(gdf.to_json())

            end_time = time.time()
            with open(self.sett.subscen_time_log_path, 'a') as f:
                f.write(f'\n\nend time: {time.ctime()}\n')
                f.write(f'run time prep: {format(str(datetime.timedelta(seconds=end_time - start_time)))} hh:mm:ss\n')
                
        print('end import_and_preprep_data()')


    def concatenate_prerep_data(self,):
        log_time = time.time()
        self.sett.concat_time_log_path = f'{self.sett.calib_scen_path}/{self.sett.name_dir_export}_concat_time_log.txt'
        log_time = self.write_to_logfile('\n * concatenate_prerep_data()', log_time= log_time, log_file_path= self.sett.concat_time_log_path )

        # remove all old concatenated files
        rm_old_files = glob.glob(os.path.join(f'{self.sett.calib_scen_preprep_path}', f'*{self.sett.name_dir_export}*'))
        for path in rm_old_files:
            os.remove(path)
            
        # list of subfiles
        preprep_subscen_paths_raw = glob.glob(os.path.join(f'{self.sett.calib_scen_preprep_path}','*log.txt'))
        if 'scicore' in preprep_subscen_paths_raw[0]:
            preprep_subscen_names = [path.split(f'{self.sett.calib_scen_preprep_path}/')[-1].split('_preprep_time_log.txt')[0] for path in preprep_subscen_paths_raw]
        else:
            preprep_subscen_names = [path.split(f'{self.sett.calib_scen_preprep_path}\\')[-1].split('_preprep_time_log.txt')[0] for path in preprep_subscen_paths_raw]

        name_list, n_file_list = [],[]
        files_to_select_list = [
            'angle_tilt_df.parquet', 
            'demandtypes_ts.parquet', 
            'elecpri.parquet', 
            'gwr_pq.parquet', 
            'HOY_weatheryear_df.parquet', 
            'Map_egid_pv.parquet', 
            'Map_gm_ewr.parquet', 
            'meteo.parquet', 
            'pvtarif.parquet', 
            'pv_pq.parquet', 
            'solkat_month_pq.parquet', 
            'solkat_pq.parquet', 
            'swstore_arch_typ_factors.parquet', 
            'swstore_arch_typ_master.parquet', 
            
            'gemeinde_type_gdf.geojson', 
            'gm_shp_gdf.geojson', 
            'gwr_gdf.geojson', 
            'pv_gdf.geojson', 
            'solkat_gdf.geojson', 
        ]
        for i_name, subscen_name in enumerate(preprep_subscen_names):
            file_list = []
            for file_pattern in files_to_select_list:
                if len(glob.glob(os.path.join(f'{self.sett.calib_scen_preprep_path}',f'{subscen_name}_{file_pattern}'))) == 0:
                    print(f'-> missing file: {subscen_name}_{file_pattern}')
                else: 
                    file_list.append(glob.glob(os.path.join(f'{self.sett.calib_scen_preprep_path}',f'{subscen_name}_{file_pattern}')))
            
            name_list.append(subscen_name)
            n_file_list.append(len(file_list))
            print(f'{subscen_name:15} n_files:{len(file_list)}') if i_name < 50 else None

        name_nfile_df = pd.DataFrame({'subscen_name': name_list, 'n_files': n_file_list})
        name_for_file_pattern_pick = name_nfile_df.loc[name_nfile_df['n_files'] == name_nfile_df['n_files'].max(), 'subscen_name'].values[0]
        print(f'-> picked: <{name_for_file_pattern_pick}>')

        file_types_paths_raw = glob.glob(os.path.join(f'{self.sett.calib_scen_preprep_path}',f'{name_for_file_pattern_pick}*'))
        file_types_names = [path.split(f'{name_for_file_pattern_pick}_')[-1] for path in file_types_paths_raw if path.split(f'{name_for_file_pattern_pick}')[-1] not in ['_preprep_time_log.txt', '_log.txt',]]
        file_types_names = [ft for ft in file_types_names if ('.parquet' in ft) | ('.geojson' in ft) ]
        
        # skip files not to be concatenated
        single_file_types =  [
            'Map_gm_ewr.parquet',

            'angle_tilt_df.parquet',
            'demandtypes_ts.parquet',
            'HOY_weatheryear_df.parquet', 
            'pvtarif.parquet',
            'gemeinde_type_gdf.geojson',
            'meteo.parquet',
            'swstore_arch_typ_factors.parquet', 
            'swstore_arch_typ_master.parquet', 
                         ]

        # file_type = file_types_names[0]
        for i_ft, file_type in enumerate(file_types_names):
            print(file_type)
            df_agg_list = []
            file_type_paths = glob.glob(os.path.join(f'{self.sett.calib_scen_preprep_path}',f'*{file_type}'))

            if file_type in single_file_types: 
                path = file_type_paths[0]
                shutil.copy(path, f'{self.sett.calib_scen_preprep_path}/0_{self.sett.name_dir_export}_{file_type}')

            else:
                if '.parquet' in file_type:
                    for path in file_type_paths:
                        df = pl.read_parquet(path) 

                        if file_type == 'solkat_month_pq.parquet':
                            if 'objectid' in df.columns:
                                df = df.drop('objectid')
                        
                        if file_type == 'solkat_pq.parquet':
                            float_cols = [
                                'MSTRAHLUNG', 'GSTRAHLUNG', 'STROMERTRAG', 
                                'STROMERTRAG_SOMMERHALBJAHR', 'STROMERTRAG_WINTERHALBJAHR', ]
                            for col in float_cols:
                                df = df.with_columns([
                                    pl.col(col).cast(pl.Float64)
                                ])

                        if file_type == 'gwr_pq.parquet':
                            drop_cols = [
                                'index_right',
                                'BFS_NO',
                            ]
                            int_cols = [
                                'GBAUJ',
                                'ARE_TYP',
                            ]
                            float_cols = [ 
                                'GKODE','GKODN', 
                                'GAREA',
                                'demand_elec_pGAREA', 
                            ]
                            df = df.drop(drop_cols)
                            str_cols = [col for col in df.columns if (col not in int_cols) & (col not in float_cols)]
                            for col in str_cols:
                                df = df.with_columns([
                                    pl.col(col).cast(pl.Utf8)
                                ])
                            for col in float_cols:
                                df = df.with_columns([
                                    df[col]
                                    .cast(pl.Utf8)
                                    .str.strip_chars()
                                    .fill_null('0.0')
                                    .replace('', '0.0')
                                    .cast(pl.Float64)
                                    .alias(col)
                                ])
                            for col in int_cols:
                                df = df.with_columns([
                                    pl.when(pl.col(col).cast(pl.Utf8).str.strip_chars().is_in(['', None]))
                                    .then(pl.lit(0))
                                    .otherwise(pl.col(col).cast(pl.Int64))
                                    .alias(col)
                                ])
                            
                            # print(f'\n{path}')
                            # for col in df.columns:
                            #     print(f'{col:15}, {df[col].dtype}')
 
                        if (df.null_count().to_numpy().sum()== 0) & (
                           (df.shape[0] >= 1) ):
                            df_agg_list.append(df)

                    df_agg = pl.concat(df_agg_list)
                    df_agg.write_parquet(os.path.join(f'{self.sett.calib_scen_preprep_path}', f'0_{self.sett.name_dir_export}_{file_type}'))
                    # file_type_to_csv = file_type.split('.parquet')[0] + '.csv'
                    # df_agg.write_csv(os.path.join(f'{self.sett.calib_scen_preprep_path}', f'0_{self.sett.name_dir_export}_{file_type_to_csv}'))

                elif '.geojson' in file_type:
                    for path in file_type_paths:
                        df = gpd.read_file(path)
                        date_types = ['datetime64[ms]', 'datetime64[ns]', 'datetime64[ms, UTC]', ]
                        any_date_in_df = [True for dt in date_types if dt in df.dtypes.values]
                        if any(any_date_in_df):
                            date_cols = [col for col in df.columns if df[col].dtype in date_types]
                            df[date_cols] = df[date_cols].astype(str)
                        df_agg_list.append(df)
                    df_agg = gpd.GeoDataFrame(pd.concat(df_agg_list, ignore_index=True), crs=df.crs)
                    # with open(f'{self.sett.calib_scen_preprep_path}/0_{self.sett.name_dir_export}_{file_type}', 'w') as f:
                    with open(os.path.join(f'{self.sett.calib_scen_preprep_path}', f'0_{self.sett.name_dir_export}_{file_type}'), 'w') as f:
                        f.write(df_agg.to_json())
                    
            log_time = self.write_to_logfile(f'exported: ./{self.sett.name_dir_export}_{file_type}', log_time, self.sett.concat_time_log_path )
        print('\n')


    def approach1_fit_optim_cost_function(self,):
        print('asdf')


    def approach2_regression_instsize(self,):
        start_time = time.time()
        with open(f'{self.sett.calib_scen_path}/approach2_time_log.txt', 'a') as f:
            f.write(f'start time: {start_time}\n')
        
        # create topo_df =====================================
        if True:
            # import dfs and merge --------------------------------

            if self.sett.scicore_concat_data_path == None: 
                gwr                = pd.read_parquet(f'{self.sett.calib_scen_preprep_path}/0_{self.sett.name_dir_export}_gwr_pq.parquet')
                solkat             = pd.read_parquet(f'{self.sett.calib_scen_preprep_path}/0_{self.sett.name_dir_export}_solkat_pq.parquet')
                solkat_month       = pd.read_parquet(f'{self.sett.calib_scen_preprep_path}/0_{self.sett.name_dir_export}_solkat_month_pq.parquet')
                pv                 = pd.read_parquet(f'{self.sett.calib_scen_preprep_path}/0_{self.sett.name_dir_export}_pv_pq.parquet')
                meteo              = pd.read_parquet(f'{self.sett.calib_scen_preprep_path}/0_{self.sett.name_dir_export}_meteo.parquet')
                Map_egid_pv        = pl.read_parquet(f'{self.sett.calib_scen_preprep_path}/0_{self.sett.name_dir_export}_Map_egid_pv.parquet')
                Map_gm_ewr         = pl.read_parquet(f'{self.sett.calib_scen_preprep_path}/0_{self.sett.name_dir_export}_Map_gm_ewr.parquet')
                pvtarif            = pl.read_parquet(f'{self.sett.calib_scen_preprep_path}/0_{self.sett.name_dir_export}_pvtarif.parquet')
                elecpri            = pl.read_parquet(f'{self.sett.calib_scen_preprep_path}/0_{self.sett.name_dir_export}_elecpri.parquet')
                HOY_weatheryear_df = pl.read_parquet(f'{self.sett.calib_scen_preprep_path}/0_{self.sett.name_dir_export}_HOY_weatheryear_df.parquet')
                demandtypes_ts     = pl.read_parquet(f'{self.sett.calib_scen_preprep_path}/0_{self.sett.name_dir_export}_demandtypes_ts.parquet')
                angle_tilt_df      = pl.read_parquet(f'{self.sett.calib_scen_preprep_path}/0_{self.sett.name_dir_export}_angle_tilt_df.parquet')

            else: 
                scicore_path = os.path.join(self.sett.scicore_concat_data_path, self.sett.name_dir_export, 'preprep_data')
                gwr                = pd.read_parquet(os.path.join(scicore_path, f'0_{self.sett.name_dir_export}_gwr_pq.parquet'))
                solkat             = pd.read_parquet(os.path.join(scicore_path, f'0_{self.sett.name_dir_export}_solkat_pq.parquet'))
                solkat_month       = pd.read_parquet(os.path.join(scicore_path, f'0_{self.sett.name_dir_export}_solkat_month_pq.parquet'))
                pv                 = pd.read_parquet(os.path.join(scicore_path, f'0_{self.sett.name_dir_export}_pv_pq.parquet'))
                meteo              = pd.read_parquet(os.path.join(scicore_path, f'0_{self.sett.name_dir_export}_meteo.parquet'))
                Map_egid_pv        = pl.read_parquet(os.path.join(scicore_path, f'0_{self.sett.name_dir_export}_Map_egid_pv.parquet'))
                Map_gm_ewr         = pl.read_parquet(os.path.join(scicore_path, f'0_{self.sett.name_dir_export}_Map_gm_ewr.parquet'))
                pvtarif            = pl.read_parquet(os.path.join(scicore_path, f'0_{self.sett.name_dir_export}_pvtarif.parquet'))
                elecpri            = pl.read_parquet(os.path.join(scicore_path, f'0_{self.sett.name_dir_export}_elecpri.parquet'))
                HOY_weatheryear_df = pl.read_parquet(os.path.join(scicore_path, f'0_{self.sett.name_dir_export}_HOY_weatheryear_df.parquet'))
                demandtypes_ts     = pl.read_parquet(os.path.join(scicore_path, f'0_{self.sett.name_dir_export}_demandtypes_ts.parquet'))
                angle_tilt_df      = pl.read_parquet(os.path.join(scicore_path, f'0_{self.sett.name_dir_export}_angle_tilt_df.parquet'))




            # GWR -------
            # gwr = pd.read_parquet(f'{self.sett.calib_scen_preprep_path}/gwr_pq.parquet')

            gwr['EGID'] = gwr['EGID'].astype(str)
            gwr.loc[gwr['GBAUJ'] == '', 'GBAUJ'] = 0  # transform GBAUJ to apply filter and transform back
            gwr['GBAUJ'] = gwr['GBAUJ'].astype(int)
            gwr.loc['GBAUJ'] = gwr['GBAUJ'].astype(str)
            # filtering for self.sett.GWR_specs
            gwr = gwr.loc[(gwr['GSTAT'].isin(self.sett.GWRspec_GSTAT)) &
                        (gwr['GKLAS'].isin(self.sett.GWRspec_GKLAS)) 
                        # (gwr['GBAUJ'] >= self.sett.GWRspec_GBAUJ_minmax[0]) &
                        # (gwr['GBAUJ'] <= self.sett.GWRspec_GBAUJ_minmax[1])
                        ].copy()
            # because not all buldings have dwelling information, need to remove dwelling columns and rows again (remove duplicates where 1 building had multiple dwellings)
            gwr.loc[gwr['GAREA'] == '', 'GAREA'] = 0
            gwr['GAREA'] = gwr['GAREA'].astype(float)

            if self.sett.GWRspec_dwelling_cols == []:
                gwr = copy.deepcopy(gwr.loc[:, self.sett.GWRspec_building_cols + self.sett.GWRspec_swstore_demand_cols])
                gwr = gwr.drop_duplicates(subset=['EGID'])
            gwr.rename(columns={'GGDENR': 'BFS_NUMMER'}, inplace=True)  # rename for merge with other dfs
            gwr = gwr.loc[gwr['BFS_NUMMER'].isin(self.sett.bfs_numbers)]
            gwr = copy.deepcopy(gwr)


            # SOLKAT -------
            # solkat = pd.read_parquet(f'{self.sett.calib_scen_preprep_path}/solkat_pq.parquet')

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
            # solkat_month = pd.read_parquet(f'{self.sett.calib_scen_preprep_path}/solkat_month_pq.parquet')

            solkat_month['DF_UID'] = solkat_month['DF_UID'].fillna('').astype(str)


            # PV -------
            # pv = pd.read_parquet(f'{self.sett.calib_scen_preprep_path}/pv_pq.parquet')
            pv['xtf_id'] = pv['xtf_id'].fillna('NA').astype(str)
            pv['TotalPower'] = pv['TotalPower'].fillna(0).astype(float)
            
            # date filter by GWR year range + extra PVINST year range
            pv['BeginningOfOperation'] = pd.to_datetime(pv['BeginningOfOperation'], format='%Y-%m-%d', errors='coerce')
            gbauj_range = [pd.to_datetime(f'{self.sett.GWRspec_GBAUJ_minmax[0]}-01-01'), 
                           pd.to_datetime(f'{self.sett.GWRspec_GBAUJ_minmax[1]}-12-31')]
            pv = pv.loc[(pv['BeginningOfOperation'] >= gbauj_range[0]) & (pv['BeginningOfOperation'] <= gbauj_range[1])]
            
            pvinst_date_range = [pd.to_datetime(f'{self.sett.pvinst_pvtrif_elecpri_range_minmax[0]}-01-01'),
                            pd.to_datetime(f'{self.sett.pvinst_pvtrif_elecpri_range_minmax[1]}-12-31')]
            pv = pv.loc[(pv['BeginningOfOperation'] >= pvinst_date_range[0]) & (pv['BeginningOfOperation'] <= pvinst_date_range[1])]
            pv['BeginningOfOperation'] = pv['BeginningOfOperation'].dt.strftime('%Y-%m-%d')

            # filter by PVINST peak capacity
            pv = pv.loc[(pv['TotalPower'] >= self.sett.pvinst_capacity_minmax[0]) & (pv['TotalPower'] <= self.sett.pvinst_capacity_minmax[1])]

            pv = pv.loc[pv["BFS_NUMMER"].isin(self.sett.bfs_numbers)]
            pv = pv.copy()
          
            
            # METEO -------
            # meteo = pd.read_parquet(f'{self.sett.calib_scen_preprep_path}/meteo.parquet')
            
            meteo_col_dir_radiation =  self.sett.WEAspec_meteo_col_dir_radiation
            meteo_col_diff_radiation = self.sett.WEAspec_meteo_col_diff_radiation
            meteo_col_temperature =    self.sett.WEAspec_meteo_col_temperature
            weater_year =              self.sett.WEAspec_weather_year

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
            meteo_ts = pl.from_pandas(meteo.copy())

            
            # REST -------
            year_rng = [str(yr) for yr in range(self.sett.pvinst_pvtrif_elecpri_range_minmax[0], self.sett.pvinst_pvtrif_elecpri_range_minmax[1]+1)]
            pvtarif_year_range_list = [str(yr_str) for yr_str in year_rng]    
            elecpri_year_range_list = [int(yr_str) for yr_str in year_rng]


            # Map_egid_pv = pl.read_parquet(f'{self.sett.calib_scen_preprep_path}/Map_egid_pv.parquet')
            # Map_egid_pv = Map_egid_pv.with_columns([pl.col('xtf_id').fill_null(0)])
            # Map_egid_pv = Map_egid_pv.with_columns([pl.col('xtf_id').cast(pl.Int64).cast(pl.Utf8)])
            Map_egid_pv = Map_egid_pv.with_columns([pl.col('xtf_id').cast(pl.Utf8)])
          
            # Map_gm_ewr = pl.read_parquet(f'{self.sett.calib_scen_preprep_path}/Map_gm_ewr.parquet')
            Map_gm_ewr = Map_gm_ewr.with_columns([pl.col('bfs').fill_null(0)])
            Map_gm_ewr = Map_gm_ewr.with_columns([pl.col('bfs').cast(pl.Int64).cast(pl.Utf8)])

            # pvtarif = pl.read_parquet(f'{self.sett.calib_scen_preprep_path}/pvtarif.parquet')
            # transform 2 digit to 4 digit year
            pvtarif = pvtarif.with_columns([
                    pl.col('year').cast(pl.Int32).alias('year_int'),
                    pl.lit('19').alias('year_prefix'),  # default 1900s                
                 ])
            pvtarif = pvtarif.with_columns([
                pl.when((pl.col('year_int') < 50) | (pl.col('year_int') >= 0)).then(pl.lit('20')).otherwise(pl.lit('19')).alias('year_prefix')
            ])
            pvtarif = pvtarif.with_columns([
                (pl.col('year_prefix') + pl.col('year')).alias('year')
                ])
            pvtarif = pvtarif.filter(pl.col('year').is_in(pvtarif_year_range_list))

            pvtarif = pvtarif.join(Map_gm_ewr, left_on='nrElcom', right_on='nrElcom', how='inner', suffix='_map_gmewr')
            pvtarif = pvtarif.with_columns([pl.col(self.sett.TECspec_pvtarif_col).replace('', 0).cast(pl.Float64)])

            empty_cols = [col for col in pvtarif.columns if pvtarif[col].is_null().all()]
            pvtarif = pvtarif.select([col for col in pvtarif.columns if col not in empty_cols])
            select_cols = ['nrElcom', 'nomEw', 'year', 'bfs', 'idofs'] + self.sett.TECspec_pvtarif_col
            pvtarif = pvtarif.select(select_cols).clone()
            pvtarif = pvtarif.rename({self.sett.TECspec_pvtarif_col[0]: 'pvtarif_Rp_kWh'})
            pvtarif = pvtarif.select(['nrElcom', 'nomEw', 'year', 'bfs', 'pvtarif_Rp_kWh']).clone()

            # elecpri = pl.read_parquet(f'{self.sett.calib_scen_preprep_path}/elecpri.parquet')
            elecpri = elecpri.filter( ( pl.col('category')== self.sett.TECspec_elecpri_category) &
                                      ( pl.col('year').is_in(elecpri_year_range_list)) )  
                                    #   ( pl.col('year') == self.sett.TECspec_elecpri_year) )
            elecpri = elecpri.with_columns([
                (pl.col('energy') + pl.col('grid') + pl.col('aidfee') + pl.col('taxes') ).alias('elecpri_Rp_kWh'), 
                pl.col('year').cast(pl.Utf8),
            ])
            elecpri = elecpri.select(['bfs_number', 'year', 'elecpri_Rp_kWh']).clone()


            # HOY_weatheryear_df  = pl.read_parquet(f'{self.sett.calib_scen_preprep_path}/HOY_weatheryear_df.parquet')
            # demandtypes_ts      = pl.read_parquet(f'{self.sett.calib_scen_preprep_path}/demandtypes_ts.parquet')
            # angle_tilt_df       = pl.read_parquet(f'{self.sett.calib_scen_preprep_path}/angle_tilt_df.parquet')


                       
            # build topo_df --------------------------------
            gwr_pl      = pl.from_pandas(gwr.drop(columns=self.sett.topo_df_excl_gwr_cols))
            solkat_pl   = pl.from_pandas(solkat.drop(columns=self.sett.topo_df_excl_solkat_cols))
            pv_pl       = pl.from_pandas(pv.drop(columns=self.sett.topo_df_excl_pv_cols))
            Map_egid_pv = Map_egid_pv.filter(pl.col('xtf_id').is_in(pv_pl['xtf_id'].implode()))


            topo_df_join0 = gwr_pl.filter(
                pl.col('EGID').is_in(Map_egid_pv.select('EGID').to_series().implode())
            )
            topo_df_join1 = topo_df_join0.join(solkat_pl, left_on='EGID', right_on='EGID', how='left', suffix='_solkat')
            topo_df_join2 = topo_df_join1.join(Map_egid_pv, left_on='EGID', right_on='EGID', how='left', suffix='_map_egidpv')
            topo_df_join3 = topo_df_join2.join(pv_pl, left_on='xtf_id', right_on='xtf_id', how='left', suffix='_pv')
            topo_df_join4 = topo_df_join3.join(elecpri, left_on='BFS_NUMMER', right_on='bfs_number', how='left', suffix='_elecpri')
            topo_df_join5 = topo_df_join4.join(pvtarif, left_on=['BFS_NUMMER', 'year'],  right_on=['bfs', 'year'], how='left', suffix='_pvtarif')

            topo_df = topo_df_join5.clone()
            del topo_df_join0, topo_df_join1, topo_df_join2, topo_df_join3, topo_df_join4, topo_df_join5


        # add direction classification =====================================
        if True:
            topo_df_complete = topo_df.filter(pl.col('DF_UID').is_not_null())
            topo_dir = topo_df_complete.with_columns([
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
            topo_dir = topo_dir.with_columns([
                pl.col("Direction").fill_null(0).alias("Direction")
                ])

            topo_pivot = (
                topo_dir
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
                topo_dir
                .group_by(['EGID', 'year'])
                .agg(
                    pl.col('BFS_NUMMER').first().alias('BFS_NUMMER'),
                    pl.col('xtf_id').first().alias('xtf_id'),
                    pl.col('DF_UID').count().alias('n_DF_UID'), 
                    pl.col('GAREA').first().alias('GAREA'),
                    pl.col('GBAUJ').first().alias('GBAUJ'),
                    pl.col('GKLAS').first().alias('GKLAS'),
                    pl.col('GSTAT').first().alias('GSTAT'),
                    pl.col('GWAERZH1').first().alias('GWAERZH1'),
                    pl.col('GENH1').first().alias('GENH1'),

                    pl.col('InitialPower').first().alias('InitialPower'),
                    pl.col('TotalPower').first().alias('TotalPower'), 
                    pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                    pl.col('pvtarif_Rp_kWh').first().alias('pvtarif_Rp_kWh'),

                    pl.col('FLAECHE').sum().alias('FLAECHE_total'),
                    )
                )
            topo_agg = topo_rest.join(topo_pivot, on='EGID', how='left', )

            for direction in [
                'north_max_flaeche',
                'east_max_flaeche',
                'south_max_flaeche',
                'west_max_flaeche',
                ]:
                if direction not in topo_agg.columns:
                    topo_agg = topo_agg.with_columns([
                    pl.lit(0).alias(direction)
                    ])
                else:
                    topo_agg = topo_agg.with_columns([
                        pl.col(direction).fill_null(0).alias(direction)
                        ])


        # export =====================================
        df_approach2 = topo_agg.clone()
        df_approach2.write_parquet( f'{self.sett.calib_scen_path}/{self.sett.name_calib_subscen}_df_approach2.parquet')
        df_approach2.write_csv(     f'{self.sett.calib_scen_path}/{self.sett.name_calib_subscen}_df_approach2.csv')

        # export gdf
        pv_gdf = gpd.read_file(f'{self.sett.calib_scen_preprep_path}/0_{self.sett.name_dir_export}_pv_gdf.geojson')
        df_approach2_pd = df_approach2.to_pandas()
        df_approach2_gdf = df_approach2_pd.merge(pv_gdf[['xtf_id', 'geometry']], how='left', on='xtf_id')
        df_approach2_gdf = gpd.GeoDataFrame(df_approach2_gdf, geometry='geometry', crs=pv_gdf.crs)
        df_approach2_gdf.to_file(f'{self.sett.calib_scen_path}/{self.sett.name_calib_subscen}_df_approach2_gdf.geojson', driver='GeoJSON')


        # regression =====================================
        if False:

            # exploratory data analysis --------------------------------
            cols_reg = ['TotalPower', 'n_DF_UID', 'elecpri_Rp_kWh', 'pvtarif_Rp_kWh', 'north_max_flaeche', 'east_max_flaeche', 'south_max_flaeche', 'west_max_flaeche']
            df_sctr_plot = topo_agg.select(['EGID'] + cols_reg).to_pandas()

            # joint_sctr = sns.pairplot(df_sctr_plot, kind = 'scatter', plot_kws={'alpha':0.5})
            # joint_sctr.savefig(f'{self.sett.calib_scen_path}/regression_scatterplot_matrix.png', dpi=300)

            # sctr1 = sns.jointplot(x='TotalPower', y='south_max_flaeche', data=df_sctr_plot, kind='scatter', alpha=0.5)
            # sctr1.savefig(f'{self.sett.calib_scen_path}/sctr1_TotalPower_SouthMaxFlaeche.png', dpi=300)



            # regressions --------------------------------
            df_pd = topo_agg.to_pandas()

            # missing values
            for col in cols_reg:
                n_miss = df_pd[col].isna().sum()
                print(f'col {col} has {n_miss} missing values')
            
            cols_reg = ['TotalPower', 'n_DF_UID', 'elecpri_Rp_kWh', 'pvtarif_Rp_kWh', 'north_max_flaeche', 'east_max_flaeche', 'south_max_flaeche', 'west_max_flaeche']
            df_pd = df_pd.dropna(subset=cols_reg)
            
            # train test split
            numeric_features = [
                'elecpri_Rp_kWh', 'pvtarif_Rp_kWh',
                'north_max_flaeche', 'east_max_flaeche', 'south_max_flaeche', 'west_max_flaeche'
            ]
            categorical_features = ['year', 'BFS_NUMMER']

            x = df_pd[numeric_features + categorical_features]
            y = df_pd['TotalPower']


            x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.33, random_state=42)
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(drop='first'), categorical_features),  # drop='first' avoids dummy trap
                    ('num', 'passthrough', numeric_features)
                ]
            )

            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])
            model.fit(x_train, y_train)

            df_test = df_pd.sample(frac=0.33, random_state=1).copy()
            df_train = df_pd.loc[~df_pd['EGID'].isin(df_test['EGID'])].copy()
            

    def random_forest_regression(self,):
        rfr_settings = self.sett.reg2_random_forest_reg_settings

        # data prep ====================

        def split_train_test(df_input, subset_type='train', split_ratio=0.7, seed=42):
            train_df, test_df = train_test_split(df_input, train_size=split_ratio, random_state=seed)
            if subset_type == 'train':
                return train_df
            elif subset_type == 'test':
                return test_df
            else:
                raise ValueError("subset_type must be either 'train' or 'test'")


        # data import + transform 
        df = pd.read_csv(f'{self.sett.calib_scen_path}'f'/{self.sett.name_calib_subscen}_df_approach2.csv')

        categorical_cols = ['GBAUJ', 'GKLAS', 'GSTAT', 'GWAERZH1', 'GENH1']
        df[categorical_cols] = df[categorical_cols].astype('str')
        df['GWAERZH1_str'] = np.where(df['GWAERZH1'].isin(['7410', '7411']), 'heatpump', 'no_heatpump')


        # filtering + splitting 
        df_train = split_train_test(df, 'train')
        df_test = split_train_test(df, 'test')  
        df_train.to_csv(f'{self.sett.calib_scen_path}/df_train.csv', index=False)
        df_test.to_csv(f'{self.sett.calib_scen_path}/df_test.csv', index=False)

        # Residential buildings 3+ units
        df__res3plus = df.loc[df['GKLAS'].isin(['1110', '1121', '1122'])]
        if df__res3plus.shape[0] > 0:
            df_train_res3plus = split_train_test(df__res3plus, 'train')
            df_test_res3plus = split_train_test(df__res3plus, 'test')
            df_train_res3plus.to_csv(f'{self.sett.calib_scen_path}/df_train_res3plus.csv', index=False)
            df_test_res3plus.to_csv(f'{self.sett.calib_scen_path}/df_test_res3plus.csv', index=False)

        # Residential buildings 1-2 units
        df__res1to2 = df.loc[df['GKLAS'].isin(['1110', '1121'])]
        if df__res1to2.shape[0] > 0:
            df_train_res1to2 = split_train_test(df__res1to2, 'train')
            df_test_res1to2 = split_train_test(df__res1to2, 'test')
            df_train_res1to2.to_csv(f'{self.sett.calib_scen_path}/df_train_res1to2.csv', index=False)
            df_test_res1to2.to_csv(f'{self.sett.calib_scen_path}/df_test_res1to2.csv', index=False)

        # Subset: Residential 1-2 units + TotalPower < 20
        df__kwpmax20 = df.loc[(df['GKLAS'].isin(['1110', '1121'])) & (df['TotalPower'] < 20)]
        if df__kwpmax20.shape[0] > 0:
            df_train_kwpmax20 = split_train_test(df__kwpmax20, 'train')
            df_test_kwpmax20 = split_train_test(df__kwpmax20, 'test')
            df_train_kwpmax20.to_csv(f'{self.sett.calib_scen_path}/df_train_kwpmax20.csv', index=False)
            df_test_kwpmax20.to_csv(f'{self.sett.calib_scen_path}/df_test_kwpmax20.csv', index=False)
            

        # random forest regression ====================

        for i, (rfr_mod_name, df_suffix) in enumerate(rfr_settings['reg2_rfrname_dfsuffix_tupls']):

            df_train_rfr = pd.read_csv(f'{self.sett.calib_scen_path}/df_train{df_suffix}.csv')
            df_test_rfr  = pd.read_csv(f'{self.sett.calib_scen_path}/df_test{df_suffix}.csv')

            # transformations
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
            df_train_rfr = df_train_rfr[[col for col in cols_dtypes_tupls.keys() if col in df_train_rfr.columns]].copy()
            df_test_rfr  = df_test_rfr[[col for col in cols_dtypes_tupls.keys() if col in df_test_rfr.columns]].copy()

            df_train_rfr = df_train_rfr.dropna().copy()
            df_test_rfr  = df_test_rfr.dropna().copy()

            for col, dtype in cols_dtypes_tupls.items():
                df_train_rfr[col] = df_train_rfr[col].astype(dtype)
                df_test_rfr[col]  = df_test_rfr[col].astype(dtype)

            X = df_train_rfr.drop(columns=['TotalPower', ])
            y = df_train_rfr['TotalPower']

            # encode categorical variables
            cat_cols = X.select_dtypes(include=["object", "category"]).columns
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            encoded_Arry = encoder.fit_transform(X[cat_cols].astype(str))
            encoded_df = pd.DataFrame(encoded_Arry, columns=encoder.get_feature_names_out(cat_cols))
            X = pd.concat([X.drop(columns=cat_cols).reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)


            # rf model
            if rfr_settings['run_ML_rfr_TF']:
                # rfr_model = RandomForestRegressor(
                #     n_estimators   = rfr_settings['n_estimators'],
                #     max_depth      = rfr_settings['max_depth'],
                #     random_state   = rfr_settings['random_state'],
                #     n_jobs         = rfr_settings['n_jobs'],
                # )
                # # cross validation
                # kf = KFold(n_splits=10, shuffle=True, random_state=42)
                # cv_scores = cross_val_score(rfr_model, X, y, cv=kf, scoring='neg_mean_absolute_error')
                # rfr_model.fit(X, y)

                rfr_model = RandomForestRegressor(random_state = rfr_settings['random_state'])
                param_grid = {
                    'n_estimators':      rfr_settings['n_estimators'],
                    'min_samples_split': rfr_settings['min_samples_split'],
                    'max_depth':         rfr_settings['max_depth'],
                }
                    
                grid_search = GridSearchCV(
                    rfr_model,
                    param_grid,
                    cv=rfr_settings['cross_validation'],
                    scoring='neg_mean_absolute_error',
                    n_jobs=rfr_settings['n_jobs'],
                    return_train_score=True,
                )
                grid_search.fit(X, y)
                rfr_model = grid_search.best_estimator_


                # save model + encoder
                joblib.dump(rfr_model, f'{self.sett.calib_scen_path}/{rfr_mod_name}_model.pkl')
                joblib.dump(encoder, f'{self.sett.calib_scen_path}/{rfr_mod_name}_encoder.pkl')

                os.makedirs(f'{self.sett.calib_path}/PVALLOC_calibration_model_coefs', exist_ok=True)
                joblib.dump(rfr_model, f'{self.sett.calib_path}/PVALLOC_calibration_model_coefs/{rfr_mod_name}_model.pkl')
                joblib.dump(encoder, f'{self.sett.calib_path}/PVALLOC_calibration_model_coefs/{rfr_mod_name}_encoder.pkl')
                


            # prediction
            if rfr_settings['run_ML_rfr_TF']:
                X_test = df_test_rfr.drop(columns=['TotalPower', ])
                encoded_test_array = encoder.transform(X_test[cat_cols].astype(str))
                encoded_test_df = pd.DataFrame(encoded_test_array, columns=encoder.get_feature_names_out(cat_cols))

                X_test_final = pd.concat([X_test.drop(columns=cat_cols).reset_index(drop=True), encoded_test_df.reset_index(drop=True)], axis=1)
                X_test_final = X_test_final[X.columns]

                test_preds = rfr_model.predict(X_test_final)

                df_test_rfr[f'pred_{rfr_mod_name}'] = test_preds

            else: 
                df_test_rfr[f'pred_{rfr_mod_name}'] = np.zeros(df_test_rfr.shape[0])
        
            # df_test_rfr.to_csv(f'{self.sett.calib_scen_path}/df_test_rfr_{rfr_mod_name}{df_suffix}.csv', index=False)
            df_test_rfr.to_csv(f'{self.sett.calib_scen_path}/df_test_rfr_{rfr_mod_name}.csv', index=False)
            del df_train_rfr, df_test_rfr
            





if __name__ == '__main__':

    # preprep calibration --------------------------------
    preprep_list = [
    #     # Calibration_Settings(), 

        Calibration_Settings(
            name_dir_export='calib_mini_debug',
            name_preprep_subsen='bfs1201',
            bfs_numbers=[1201,],
            n_rows_import= 4000,
            rerun_import_and_preprp_data_TF = True,
            export_gwr_ALL_building_gdf_TF = True
        ), 
        # Calibration_Settings(
        #     name_dir_export='calib_mini_debug',
        #     name_preprep_subsen='bfs1033',
        #     bfs_numbers=[1033,],
        #     n_rows_import= 4000,
        #     rerun_import_and_preprp_data_TF = True,
        #     export_gwr_ALL_building_gdf_TF = True
        # ), 
        # Calibration_Settings(
        #     name_dir_export='calib_mini_debug',
        #     name_preprep_subsen='bfs1205',
        #     bfs_numbers=[1205,],
        #     # n_rows_import= 2000,
        #     rerun_import_and_preprp_data_TF = True,
        #     export_gwr_ALL_building_gdf_TF = True
        # ), 
        # Calibration_Settings(
        #     name_dir_export='calib_mini_debug',
        #     name_preprep_subsen='bfs3788',
        #     bfs_numbers=[3788,],
        #     # n_rows_import= 2000,
        #     rerun_import_and_preprp_data_TF = True,
        #     export_gwr_ALL_building_gdf_TF = True
        # ),
        # Calibration_Settings(
        #     name_dir_export='calib_mini_debug',
        #     name_preprep_subsen='bfs3764',
        #     bfs_numbers=[3764,],
        #     # n_rows_import= 2000,
        #     rerun_import_and_preprp_data_TF = True,
        #     export_gwr_ALL_building_gdf_TF = True
        # ),
        # Calibration_Settings(
        #     name_dir_export='calib_mini_debug',
        #     name_preprep_subsen='bfs3762',
        #     bfs_numbers=[3762,],
        #     # n_rows_import= 2000,
        #     rerun_import_and_preprp_data_TF = True,
        #     export_gwr_ALL_building_gdf_TF = True
        # ),
        # Calibration_Settings(
        #     name_dir_export='calib_mini_debug',
        #     name_preprep_subsen='bfs1631',
        #     bfs_numbers=[1631,],
        #     # n_rows_import= 2000,
        #     rerun_import_and_preprp_data_TF = True,
        #     export_gwr_ALL_building_gdf_TF = True
        # ),
        # Calibration_Settings(
        #     name_dir_export='calib_mini_debug',
        #     name_preprep_subsen='bfs3746',
        #     bfs_numbers=[3746,],
        #     # n_rows_import= 2000,
        #     rerun_import_and_preprp_data_TF = True,
        #     export_gwr_ALL_building_gdf_TF = True
        # ),
        # Calibration_Settings(
        #     name_dir_export='calib_mini_debug',
        #     name_preprep_subsen='bfs3543',
        #     bfs_numbers=[3543,],
        #     # n_rows_import= 2000,
        #     rerun_import_and_preprp_data_TF = True,
        #     export_gwr_ALL_building_gdf_TF = True
        # ),
        # Calibration_Settings(
        #     name_dir_export='calib_mini_debug',
        #     name_preprep_subsen='bfs6037',
        #     bfs_numbers=[6037,],
        #     # n_rows_import= 2000,
        #     rerun_import_and_preprp_data_TF = True,
        #     export_gwr_ALL_building_gdf_TF = True
        # ),
        # Calibration_Settings(
        #     name_dir_export='calib_mini_debug',
        #     name_preprep_subsen='bfs3851',
        #     bfs_numbers=[3851,],
        #     # n_rows_import= 2000,
        #     rerun_import_and_preprp_data_TF = True,
        #     export_gwr_ALL_building_gdf_TF = True
        # ),
        # Calibration_Settings(
        #     name_dir_export='calib_mini_debug',
        #     name_preprep_subsen='bfs3792',
        #     bfs_numbers=[3792,],
        #     # n_rows_import= 2000,
        #     rerun_import_and_preprp_data_TF = True,
        #     export_gwr_ALL_building_gdf_TF = True
        # ),
        # Calibration_Settings(
        #     name_dir_export='calib_mini_debug',
        #     name_preprep_subsen='bfs6252',
        #     bfs_numbers=[6252,],
        #     # n_rows_import= 2000,
        #     rerun_import_and_preprp_data_TF = True,
        #     export_gwr_ALL_building_gdf_TF = True
        # ),
        # Calibration_Settings(
        #     name_dir_export='calib_mini_debug',
        #     name_preprep_subsen='bfs6300',
        #     bfs_numbers=[6300,],
        #     # n_rows_import= 2000,
        #     rerun_import_and_preprp_data_TF = True,
        #     export_gwr_ALL_building_gdf_TF = True
        # ),
    
    ]   

    for i_prep, prep_sett in enumerate(preprep_list):
        print('')
        preprep_class = Calibration(prep_sett)
        # preprep_class.import_and_preprep_data() if preprep_class.sett.rerun_import_and_preprp_data_TF else None

    # preprep_class = Calibration(preprep_list[0])
    calib_class = Calibration(preprep_list[0])
    calib_class.concatenate_prerep_data()           if calib_class.sett.run_concatenate_preprep_data_TF else None

    calib_class.approach2_regression_instsize()     if calib_class.sett.run_approach2_regression_instsize_TF else None
    calib_class.random_forest_regression()          if calib_class.sett.run_appr2_random_forest_reg_TF else None




    # run calibration approaches --------------------------------
    calibration_list = [
        Calibration_Settings(
            name_dir_export='calib_mini_debug',
            name_preprep_subsen='bfs_mini_try',
            name_calib_subscen='reg2_mini_bfs',
            pvinst_pvtrif_elecpri_range_minmax = (2017, 2023),
            rerun_import_and_preprp_data_TF = False,
            export_gwr_ALL_building_gdf_TF = False
        ), 
    ]
    for calib_settings in calibration_list:
        calib_class = Calibration(calib_settings)


