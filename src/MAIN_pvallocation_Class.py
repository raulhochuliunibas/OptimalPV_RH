# python integrated packages
import sys
import os as os 

# external packages
import pandas as pd
import datetime as datetime
import polars as pl
from  dataclasses import dataclass, field, asdict
from dataclasses import replace
from typing_extensions import List, Dict, Union
from scipy import optimize
from scipy.stats import pearson3
# from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp


# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, print_to_logfile, checkpoint_to_logfile, get_bfs_from_ktnr, add_static_topo_data_to_subdf
from src.pvalloc.pvallocation_inst_selection_optimization import run_pv_selection_optimization, prepare_feedin_matrix, optimize_pv_selection_greedy , optimize_pv_selection_pulp


# settings
pl.Config.set_tbl_rows(30)  # Adjust 50 to whatever number of rows you want
pd.set_option('display.max_rows', 30) 

from src.pvalloc.pvalloc_initialization import PVAllocInitialization
from src.pvalloc.pvalloc_mcalgorithm import PVAllocMCAlgo
from src.pvalloc.pvalloc_gridoptimized import PVAllocGridOpt


# settings
pl.Config.set_tbl_rows(30)  # Adjust 50 to whatever number of rows you want
pd.set_option('display.max_rows', 30) 


@dataclass
class PVAllocScenario_Settings:
    # DEFAULT SETTINGS ---------------------------------------------------
    name_dir_export: str                        = 'pvalloc_BL_smallsample'   # name of the directory where the data is exported to (name to replace/ extend the name of the folder "preprep_data" in the end)
    name_dir_import: str                        = 'preprep_BLSO_15to24_extSolkatEGID__Oct26'     # 'preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI'     #'preprep_BLSO_22to23_extSolkatEGID_aggrfarms'
    run_pvalloc_initalization_TF: bool          = True
    run_pvalloc_mcalgorithm_TF: bool            = True
    run_gridoptimized_orderinst_TF: bool        = False
    run_gridoptimized_expansion_TF: bool        = False

    # region general settings
    show_debug_prints: bool                     = False                    # F: certain print statements are omitted, T: includes print statements that help with debugging
    export_csvs: bool                           = False
    
    kt_numbers: List[int]                       = field(default_factory=list)  # list of cantons to be considered
    bfs_numbers: List[int]                      = field(default_factory=lambda: [
                                                    2641, 2615, 
                                                    # # RURAL - Beinwil, Lauwil, Bretzwil, Nunningen, Zullwil, Meltingen, Erschwil, Büsserach, Fehren, Seewen
                                                    # 2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616, 2480,
                                                    # # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
                                                    # 2613, 27  2, 2618, 2786, 2785, 2772, 2761, 2743, 
                                                    # # URBAN: Reinach, Münchenstein, Muttenz
                                                    # 2773, 2769, 2770,
                                                    ])
    mini_sub_model_TF: bool                     = False
    mini_sub_model_grid_nodes: List[str]        = field(default_factory=lambda: [ 
                                                                    #ew nodes
                                                                    '524', 
                                                                    '743',
                                                                    '724', 
                                                                    # regular nodes problematic
                                                                    '81', '867', '79',
                                                                                 ]) 
    mini_sub_model_ngridnodes: int              = 6
    mini_sub_model_nEGIDs: int                  = 30
    mini_sub_model_by_X: str                    = 'by_EGID'       # 'by_gridnode' / 'by_EGID' 
    mini_sub_model_select_EGIDs: List[str]      = field(default_factory=lambda: [
                                                    # pv_df EGIDs in BFS 2889 - Lauwil
                                                    # '3032150', '2362100', '245052560', '245048760', '434178', '245057989', '245044986', 
                                                    # pv_df EGIDs in 9 BFS RUR selection
                                                    # '101428161', '11513725', '190001512', '190004146', '190024109', '190033245', 
                                                    # '190048248', '190083872', '190109228', '190116571', '190144906', '190178022', 
                                                    # '190183577', '190185552', '190251628', '190251772', '190251828', '190296588', 
                                                    # '190491308', '190694269', '190709629', '190814490', '190912633', '190960689', 
                                                    # '2125434', '2362100', '245044986', '245048760', '245052560', '245053405', 
                                                    # '245057989', '245060059', '3030694', '3030905', '3031761', '3032150', '3033714', 
                                                    # '3075084', '386736', '432600', '432638', '432671', '432683', '432701', '432729', '434178',
                                                    ])
    
    recreate_topology: bool                     = True
    recalc_economics_topo_df: bool              = True
    sanitycheck_byEGID: bool                    = True
    create_gdf_export_of_topology: bool         = True
    # endregion

    
    # PART I: settings for alloc_initialization --------------------
    # region part I: initialization settings
    # gwr_specs
    GWRspec_solkat_max_n_partitions: int                = 10          # larger number of partitions make all combos of roof partitions practically impossible to calculate
    GWRspec_solkat_area_per_EGID_range: List[int]       = field(default_factory=lambda: [2, 600])  # for 100kWp inst, need 500m2 roof area => just above the threshold for residential subsidies KLEIV, below 2m2 too small to mount installations
    GWRspec_building_cols: List[str]                    = field(default_factory=lambda: [
                                                            'EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                                                            'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                                                            'GEBF', 'GAREA', 
                                                            'GWAERZH1', 'GENH1',   # 'GWAERSCEH1', 'GWAERDATH1',
                                                            'GWAERZH2', 'GENH2',   #'GWAERSCEH2', 'GWAERDATH2'
                                                        ])
    
    GWRspec_dwelling_cols: List[str]                    = field(default_factory=list)
    GWRspec_swstore_demand_cols: List[str]              = field(default_factory=lambda: ['ARE_typ', 'sfhmfh_typ', 'arch_typ', 'demand_elec_pGAREA'])
    GWRspec_DEMAND_proxy: str                           = 'GAREA'
    # gwr selection
    GWRspec_GSTAT: List[str]                            = field(default_factory=lambda: [
                                                                # '1001', # GSTAT - 1001: in planing
                                                                # '1002', # GSTAT - 1002: construction right granted 
                                                                # '1003', # GSTAT - 1003: in construction
                                                                '1004', # GSTAT - 1004: fully constructed, existing buildings
                                                                ])    
    GWRspec_GKLAS: List[str]                            = field(default_factory=lambda: [
                                                                '1110', # GKLAS - 1110: only 1 living space per building
                                                                '1121', # GKLAS - 1121: Double-, row houses with each appartment (living unit) having it's own roof;
                                                                '1122', # GKLAS - 1122: Buildings with three or more appartments
                                                                # '1276', # GKLAS - 1276: structure for animal keeping (most likely still one owner)
                                                                # '1278', # GKLAS - 1278: structure for agricultural use (not anmial or plant keeping use, e.g. barns, machinery storage, silos),
                                                                ])
    GWRspec_GBAUJ_minmax: List[int]                     = field(default_factory=lambda: [1950, 2024])     # GBAUJ_max value is replaced later with T0_year_prediction
    apply_GWRspec_GBAUJ_minmax_TF: bool                 = False
    

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
    T0_year_prediction: int                             = 2024                          # year for the prediction of the future construction capacity
    # T0_prediction: str                                = f'{T0_year_prediction}-01-01 00:00:00'         # start date for the prediction of the future construction capacity
    months_prediction: int                              = 360                         # number of months to predict the future construction capacity

    months_lookback: int                                = 24                           # number of months to look back for the prediction of the future construction capacity
    CSTRspec_capacity_type: str                         ='adj_hist_to_ep2050' # adj_hist_to_ep2050 / ep2050_zerobasis / hist_constr_capa_year >> discarded: hist_constr_capa_month
    # CSTRspec_iter_time_unit: str                        = 'year'   # month (not really feasible), year
    CSTRspec_ann_capacity_growth: float                 = 0.05
    CSTRspec_constr_capa_overshoot_fact: int            = 1
    
    CSTRspec_ep2050_rescale_fact: int                   = 1.0,
    CSTRspec_ep2050_capa_dict: Dict[str, float]         = field(default_factory=lambda: {
        'ep2050_zerobasis':{
            'pvcapa_total': {
                '2020' : 2.5,  # <= this is for year 2019 in source, but for ease of use changed to 2020
                '2025' : 4.8,
                '2030' : 9.8,
                '2035' : 16.2,
                '2040' : 24.1,
                '2045' : 31.0,
                '2050' : 37.5,
                '2055' : 40.4,
                '2060' : 41.9
                }, 
            'share_instclass': {
                '2020':{
                    'class1': 0.389, 
                    'class2': 0.1569, 
                    'class3': 0.3819, 
                    'class4': 0.0712
                    }, 
                '2035':{
                    'class1': 0.42, 
                    'class2': 0.1639,
                    'class3': 0.3786, 
                    'class4': 0.0354,
                    }, 
                '2050':{
                    'class1': 0.4617, 
                    'class2': 0.182,
                    'class3': 0.3349,
                    'class4': 0.0213,
                    }, 
            },
            'CHcapa_adjustment_filter': {
                'classes_adj_list': [
                    'class1', 
                    # 'class2'
                    ],
                'GSTAT_list': [
                    '1003', '1004',
                    ],
                'GKLAS_list': [
                    '1110',    # : 'Gebäude mit einer Wohnung', 
                    '1121',    # : 'Gebäude mit zwei Wohnungen', 
                    '1122',    # : 'Gebäude drei oder mehr Wohnungen', 
                    '1130',    # : 'Wohngebäude für Gemeinschaften', 
                    '1211',    # : 'Hotels und ähnliche Gebäude', 
                    '1212',    # : 'Kurzfristige Beherbergungen, Mobilheime', 
                    '1220',    # : 'Büro- und Verwaltungsgebäude', 
                    '1230',    # : 'Gross- und Einzelhandelsgebäude', 
                    '1231',    # : 'Restaurants, Bars (ohne Wohnnutzung)', 
                    '1241',    # : 'Bahnhöfe, Kommunikation, Verkehr', 
                    '1242',    # : 'Garagengebäude, überdachte Parkplätze', 
                    '1251',    # : 'Industriegebäude und Fabriken', 
                    '1252',    # : 'Behälter, Silos, Lagergebäude', 
                    '1261',    # : 'Kultur- und Freizeitzwecke',
                    '1262',    # : 'Museen und Bibliotheken', 
                    '1263',    # : 'Schulen, Hochschulen, Forschung', 
                    '1264',    # : 'Krankenhäuser, Pflegeeinrichtungen',
                    '1265',    # : 'Gebäude für Hallensport',
                    # '1271',    # : 'Landwirtschaftliche Betriebsgebäude (ersetzt)',
                    '1272',    # : 'Kirchen und Kultgebäude',
                    # '1273',    # : 'Denkmäler, unter Denkmalschutz',
                    # '1274',    # : 'Sonstige Hochbauten ungenannt',
                    # '1275',    # : 'Andere Gebäude kollektive Unterkunft',
                    # '1276',    # : 'Gebäude für Tierhaltung',
                    # '1277',    # : 'Gebäude für Pflanzenbau',
                    # '1278',    # : 'Andere landwirtschaftliche Gebäude'
                ]
            },
      },


    })
    CSTRspec_ep2050_share_inst_classes: List[str]       = field(default_factory=lambda: [
                                                                                    'class1', 
                                                                                    #  'class2',
                                                                                        ])  # 'class1', 'class2', 'class3', 'class4'
    CSTRspec_histadj_dict: Dict[str, float]             = field(default_factory=lambda: {
        'adj_months_lookback': 24, 
        'adjann_capacity_growth': 0.25,
        'rescale_epzb2050_target': 1.0,
        'adj_target_margin': 0.05,
    })
    CSTRspec_month_constr_capa_tuples: List[tuple]      = field(default_factory=lambda: [
                                                            (1,  0.04), 
                                                            (2,  0.04), 
                                                            (3,  0.04), 
                                                            (4,  0.06),
                                                            (5,  0.06), 
                                                            (6,  0.06), 
                                                            (7,  0.1), 
                                                            (8,  0.1),
                                                            (9,  0.1), 
                                                            (10, 0.1), 
                                                            (11, 0.14), 
                                                            (12, 0.16)
                                                        ])    
    CSTRspec_ext_constrcapa_path: str                   = None  
    
    # tech_economic_specs
    TECspec_elecpri_year: int                               = None
    TECspec_pvtarif_year: int                               = None
    TECspec_elecpri_Rp_kWh: Union[None, float, str]         = 31.98   # None: elecpri of T0 year; float: generic elecpri; str: path to hourly elecpri data (parquet) for T0 year
    # TECspec_generic_elecpri_Rp_kWh: float                   = 31.98     # Elecpri Elcom in 2026   # None: elecpri of previous year is applied (T0_prediction -1 )
    TECspec_pvtarif_Rp_kWh: Union[None, float, str]         = 9.03     # None: pvtarif of T0 year; float: generic pvtarif; str: path to hourly pvtarif data (parquet) for T0 year
    TECspec_pvtarif_Rp_floor: float                         = 6.0      # lower bound (Rp/kWh) clipped onto the hourly pvtarif data; only used if TECspec_pvtarif_Rp_kWh is a str
    # TECspec_generic_pvtarif_Rp_kWh: float                   = 6.0     # energy 6.94 Rp, herkunfsn. 2 Rp PVtarif of DSO in mid-late 2026  # None: pvtarif of previous year is applied (T0_prediction -1 )
    TECspec_KEVflatsubs: float                              = 0.3
    TECspec_interest_rate: float                            = 0.015
    TECspec_self_consumption_ifapplicable: float            = 1.0
    TECspec_pvtarif_col: List[str]                          = field(default_factory=lambda: ['energy1', ])  # 'energy1', 'eco1'
    TECspec_pvprod_calc_method: str                         = 'method2.2'
    TECspec_panel_efficiency: float                         = 0.21
    TECspec_inverter_efficiency: int                        = 0.8
    TECspec_elecpri_category: str                           = 'H4'
    TECspec_invst_maturity: int                             = 25
    TECspec_kWpeak_per_m2: float                            = 0.2
    TECspec_share_roof_area_available: float                = 0.7
    TECspec_max_distance_m_for_EGID_node_matching: float    = 0
    TECspec_kW_range_for_pvinst_cost_estim: List[int]       = field(default_factory=lambda: [0, 61])
    TECspec_estim_pvinst_cost_correctionfactor: float       = 1
    TECspec_opt_max_flaeche_factor: float                   = 1.5
    TECspec_add_heatpump_demand_TF: bool                    = True   
    TECspec_heatpump_indicator: List[str]                   = field(default_factory=lambda: ['7410',    # '7410' heatpump for 1 building, 
                                                                                            #  '7411'     # '7411' heatpump for multiple buildings, 
                                                                                             ] ) 
    TECspec_heatpump_months_factor: List[tuple]             = field(default_factory=lambda: [
                                                            (9 ,  2.0),                                                            
                                                            (10,  4.0),
                                                            (11,  8.0), 
                                                            (12, 10.0), 
                                                            (1 , 10.0), 
                                                            (2 , 10.0), 
                                                            (3 ,  8.0), 
                                                            (4 ,  4.0), 
                                                            (5 ,  2.0),     
                                                            (6 ,     1.0), 
                                                            (7 ,     1.0), 
                                                            (8 ,     1.0), 
                                                            ])
    # panel_efficiency_specs
    PEFspec_variable_panel_efficiency_TF: bool              = True
    PEFspec_summer_months: List[int]                        = field(default_factory=lambda: [6, 7, 8, 9])
    PEFspec_hotsummer_hours: List[int]                      = field(default_factory=lambda: [11, 12, 13, 14, 15, 16, 17])
    PEFspec_hot_hours_discount: float                       = 0.1
    
    # sanitycheck_summary_byEGID_specs
    CHECKspec_egid_list: List[str]                          = field(default_factory=lambda: [])
                                                            #     '391292', '390601', '2347595', '401781',  # single roof houses in Aesch, Ettingen
                                                            #     '391263', '245057295', '401753',  # houses with built pv in Aesch, Ettingen
                                                            #     '245054165', '245054166', '245054175', '245060521', # EGID selection of neighborhood within Aesch to analyse closer
                                                            #     '391253', '391255', '391257', '391258', '391262',
                                                            #     '391263', '391289', '391290', '391291', '391292',
                                                            #     '245057295', '245057294', '245011456', '391379', '391377'
                                                            # ])
    CHECKspec_n_EGIDs_of_alloc_algorithm: int               = 20
    CHECKspec_n_iterations_before_sanitycheck: int          = 2
    # endregion

    # PART II: settings for MC algorithm --------------------
    # region part II: MC algorithm settings
    MCspec_montecarlo_iterations_fordev_sequentially: int        = 1
    MCspec_fresh_initial_files: List[str]                       = field(default_factory=lambda: [
                                                                    'topo_egid.json', 'trange_prediction.parquet',# 'gridprem_ts.parquet', 
                                                                    'constrcapa.parquet', # 'dsonodes_df.parquet'
                                                                ])
    MCspec_keep_files_month_iter_TF: bool                       = True
    MCspec_keep_files_month_iter_max: int                       = 9999999999
    MCspec_keep_files_month_iter_list: List[str]                = field(default_factory=lambda: [
                                                                    'topo_egid.json', 'npv_df.parquet', 'pred_inst_df.parquet', 'gridprem_ts.parquet'
                                                                ])

    # algorithm_specs
    ALGOspec_inst_selection_method: str                         = 'prob_weighted_npv'          # 'random', max_npv', 'prob_weighted_npv'
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
    ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF: bool   = True
    ALGOspec_adjust_existing_pvdf_capa_topartition: str         = 'capa_no_adj_pvprod_no_adj'     
                                                                    # 'capa_no_adj_pvprod_no_adj',              : assigns df_uid_winst to topo in (df_uid, dfuidPower) - tuples, to be easily accessed later, pvprod_kW is not altered at all.
                                                                    # 'capa_roundup_pvprod_no_adj'              : assigns df_uid_w_inst to topo, based on pv_df TotalPower value (rounded up), pvprod_kW remains "untouched" and is still equivalent to production potential per roof partition
                                                                    # 'capa_roundup_pvprod_adjusted' - ATTENTION: will activate an if statement which will adjust pvprod_kW in topo_time_subdfs, so no longer pure production potential per roof partition
                                                                    # 'capa_no_adj_pvprod_adjusted' - ATTENTION: will activate an if statement which will adjust pvprod_kW in topo_time_subdfs, so no longer pure production potential per roof partition
    ALGOspec_pvinst_option_to_EGID: str                         = 'max_dfuid_EGIDcombosc'    # 'EGIDitercombos_maxdfuid' / 'EGIDoptimal__partial_dfuid'

    ALGOspec_constr_capa_overshoot_fact: float                  = 1
    ALGOspec_subselec_filter_criteria: str                      = None  # 'southfacing_1spec' / 'eastwestfacing_3spec' / 'southwestfacing_2spec'
                                                                        # edit: new a tuple of order filtering, basically install inst on EGIDs with this filter_tag == True first
                                                                        # df_tag_south_nr  df_tag_south_1r  eastwest_2r  eastwest_nr
    ALGOspec_subselec_filter_method: str                        = 'pooled'  # 'ordered' / 'pooled',   # 'ordered': filter criteria are applied in order of ALGOspec_subselec_filter_criteria list, 'pooled': all filter criteria are applied at once and inst are selected from the pool of EGIDs that fulfill any of the filter criteria
    # drop cols in subdf (safe disk space)                                                                    
    ALGOspec_drop_cols_topo_time_subdf_list: List[str]          = field(default_factory=lambda: [
                                                                       'index', 'timestamp', 'rad_direct', 'rad_diffuse', 'temperature', 
                                                                       'A_PARAM', 'B_PARAM', 'C_PARAM', 'mean_top_radiation', 
                                                                       'radiation_rel_locmax', 
                                                                       'GKLAS', 'GAREA', 'GBAUJ', 'GSTAT', 'GWAERZH1', 'GENH1', 'GWAERZH2', 'GENH2', 
                                                                       'sfhmfh_typ', 'arch_typ', 'demand_arch_typ', 'demand_elec_pGAREA',
                                                                       'pvid', 'inst_TF', 'info_source', 'TotalPower', 'efficiency_factor', 
                                                                       'grid_node',
                                                                       'pvtarif_Rp_kWh', 'elecpri_Rp_kWh',
                                                                       'FLAECHE', 'FLAECH_angletilt', 'AUSRICHTUNG', 'NEIGUNG', 'STROMERTRAG', 'MSTRAHLUNG', 'GSTRAHLUNG',
                                                                       'meteo_loc', ])
    
    ALGOspec_reinstall_inst_EGID_pvdf_for_check_TF :bool        = False  # True: will reinstall the dfuid_winst to EGIDs that already have a inst in reality in pv_df to check accuracy of allocation kWp estimates
    ALGOspec_tweak_constr_capacity_fact: float                  = 1
    ALGOspec_tweak_npv_calc: float                              = 1
    ALGOspec_tweak_gridnode_df_prod_demand_fact: float          = 1
    ALGOspec_tweak_demand_profile: float                        = 1.8
    ALGOspec_pvinst_size_calculation: str                       = 'estim_rfr'   # 'estim_rf_segdist' / 'estim_rfr' / 'inst_full_partition' / 'npv_optimized'
    ALGOspec_calib_estim_dir_name: str                          = 'PVALLOC_calibration_model_coefs'
    ALGOspec_calib_estim_mod_name_pkl: str                      = 'allCHbfs_rfr2b'
    ALGOspec_sleep_bfr_MCiter_TF: bool                          = False

    # dsonodes_ts_specs
    GRIDspec_flat_profile_demand_dict: Dict                     = field(default_factory=lambda: {
                                                                    '_window1':{'t': [6,21],  'demand_share': 0.9}, 
                                                                    '_window2':{'t': [22, 5], 'demand_share': 0.1},
                                                                    })
    GRIDspec_flat_profile_demand_total_EGID: float              = 4500
    GRIDspec_flat_profile_demand_type_col: set                  = 'MFH_swstore'  # 'flat' / 'MFH_swstore' / 'outtopo_demand_zero'

    # gridprem_adjustment_specs
    GRIDspec_apply_prem_tiers_TF: bool                          = False 
    GRIDspec_tier_description: str                              = 'tier_level: (voltage_threshold, gridprem_Rp_kWh)'
    GRIDspec_power_factor: float                                = 1
    GRIDspec_perf_factor_1kVA_to_XkW: float                     = 0.8
    GRIDspec_colnames: List[str]                                = field(default_factory=lambda: ['tier_level', 'used_node_capa_rate', 'gridprem_Rp_kWh'])
    GRIDspec_tiers: Dict[int, List[float]]                      = field(default_factory=lambda: {
                                                                   1: [0.7,   0], 
                                                                   2: [0.8,   0], 
                                                                   3: [0.85,  0], 
                                                                   4: [0.9,   0], 
                                                                   5: [0.95,  0], 
                                                                   6: [1,     0]
                                                                })
    GRIDspec_node_1hll_closed_TF: bool                         = False       # F: installations can still be built in grid nodes that have > 1 HOY Lost Load, T: no installations in circuits which have just 1 hour of lost load in the grid_updating stage. 
    GRIDspec_subsidy_name: str                                 = 'default_subsidy'   # 'subsidy_name' / None
    
    # peak-shaving settings
    GRID_peak_shaving_enabled_tupl: tuple                         = (False, 9999, 1 )         # (Enable/disable peak-shaving feature, node HC ratio threshold for peak-shaving activation (e.g. 0.9  = 90%), production reduction factor during peak-shaving (e.g. 0.7 = 70% of production))
    # GRID_peak_shaving_enabled_TF: bool                          = False                # Enable/disable peak-shaving feature
    # GRID_peak_shaving_node_criticality_threshold: float          = 0.9                 # Node HC ratio threshold (e.g., 0.9 = 90%)
    # GRID_peak_shaving_production_factor: float                   = 0.7                 # Production reduction factor (e.g., 0.7 = 70% of production)
    
    GRIDspec_subsidy_filtag_node_schemes: Dict[str, float]     = field(default_factory=lambda: {

        # new naming convention ---------------------------------------

        'default_subsidy': {
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 0.0),
                ('filter_tag__eastwest_70pr', 0.0),
                ],
            'subs_nodeHC_chf_tuples':(0.5,   0.0),
            'pena_nodeHC_chf_tuples':(0.95,  0.0),
        },
        'As2p0':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 2000.0),
                ('filter_tag__eastwest_70pr', 2000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.5,   0.0), 
            'pena_nodeHC_chf_tuples':(0.95,  0.0), 
        },
        'As4p0':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 4000.0),
                ('filter_tag__eastwest_70pr', 4000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.5,   0.0), 
            'pena_nodeHC_chf_tuples':(0.95,  0.0), 
        },
        'As6p0':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 6000.0),
                ('filter_tag__eastwest_70pr', 6000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.5,   0.0), 
            'pena_nodeHC_chf_tuples':(0.95,  0.0), 
        },
        'As8p0':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 8000.0),
                ('filter_tag__eastwest_70pr', 8000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.5,   0.0), 
            'pena_nodeHC_chf_tuples':(0.95,  0.0), 
        },

        'Bs0p4':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 0.0),
                ('filter_tag__eastwest_70pr', 0.0),
            ],
            'subs_nodeHC_chf_tuples':(0.5,   0.0),
            'pena_nodeHC_chf_tuples':(0.90,  4000.0), 
        },
        'Bs0p6':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 0.0),
                ('filter_tag__eastwest_70pr', 0.0),
            ],
            'subs_nodeHC_chf_tuples':(0.7,   0.0),
            'pena_nodeHC_chf_tuples':(0.90,  8000.0), 
        },
        'Bs0p8':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 0.0),
                ('filter_tag__eastwest_70pr', 0.0),
            ],
            'subs_nodeHC_chf_tuples':(0.7,   0.0),
            'pena_nodeHC_chf_tuples':(0.90,  8000.0), 
        },


        'Cs2p4':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 2000.0),
                ('filter_tag__eastwest_70pr', 2000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.7,   0.0),
            'pena_nodeHC_chf_tuples':(0.90,  4000.0), 
        },
        'Cs2p6':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 2000.0),
                ('filter_tag__eastwest_70pr', 2000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.7,   0.0),
            'pena_nodeHC_chf_tuples':(0.90,  6000.0), 
        },
        'Cs2p8':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 2000.0),
                ('filter_tag__eastwest_70pr', 2000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.7,   0.0),
            'pena_nodeHC_chf_tuples':(0.90,  8000.0), 
        },


        'Cs4p4':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 4000.0),
                ('filter_tag__eastwest_70pr', 4000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.7,   0.0),
            'pena_nodeHC_chf_tuples':(0.90,  4000.0), 
        },
        'Cs4p6':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 4000.0),
                ('filter_tag__eastwest_70pr', 4000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.7,   0.0),
            'pena_nodeHC_chf_tuples':(0.90,  6000.0), 
        },
        'Cs4p8':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 4000.0),
                ('filter_tag__eastwest_70pr', 4000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.7,   0.0),
            'pena_nodeHC_chf_tuples':(0.90,  8000.0), 
        },

        'Cs6p4':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 6000.0),
                ('filter_tag__eastwest_70pr', 6000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.7,   0.0),
            'pena_nodeHC_chf_tuples':(0.90,  4000.0), 
        },
        'Cs6p6':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 6000.0),
                ('filter_tag__eastwest_70pr', 6000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.7,   0.0),
            'pena_nodeHC_chf_tuples':(0.90,  6000.0), 
        },
        'Cs6p8':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 6000.0),
                ('filter_tag__eastwest_70pr', 6000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.7,   0.0),
            'pena_nodeHC_chf_tuples':(0.90,  8000.0), 
        },       
        'Cs8p4':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 8000.0),
                ('filter_tag__eastwest_70pr', 8000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.7,   0.0),
            'pena_nodeHC_chf_tuples':(0.90,  4000.0), 
        },
        'Cs8p6':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 8000.0),
                ('filter_tag__eastwest_70pr', 8000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.7,   0.0),
            'pena_nodeHC_chf_tuples':(0.90,  6000.0), 
        },
        'Cs8p8':{
            'subs_filter_tags_chf_tuples': [
                ('filter_tag__eastwest_80pr', 8000.0),
                ('filter_tag__eastwest_70pr', 8000.0),
            ],
            'subs_nodeHC_chf_tuples':(0.7,   0.0),
            'pena_nodeHC_chf_tuples':(0.90,  8000.0), 
        },
    })
    # endregion


    # PART III: settings for Grid Optimal PV Expansion --------------------
    # region part III: grid optimal pv expansion settings
    OPTIMspecs_gridnode_subsample: str                = 'all_nodes_pyparallel', # 'max_node' / 'all_nodes_pyparallel / 'all_nodes_loop'/ 'custom-> grid_node: str'
    # OPTIMspecs_solving_method: str                    = 'greedy'       # 'greedy' / 'pulp'
    OPTIMspecs_solving_method: List[str]              = field(default_factory=lambda: [
                                                    # 'greedy', 
                                                    'pulp',
                                                    ])
    OPTIMspecs_greedy_selection_strategy: str         = 'peak_efficient'       # # - 'peak_efficient': Select houses with lowest peak contribution|  - 'energy_max': Select houses with highest energy contribution|  - 'random': Random selection (for baseline comparison)| # endregion
    OPTIMspecs_peak_ext_limit_kW: float               = 0
    OPTIMspecs_cumulative_limit_kWh: float            = 0
    
    OPTEXPApecs_apply_gridoptim_order_TF: bool         = True  # similar to ALGOspec_subselec_filter_criteria, npv_df is sorted, such that min(grid_optim_inst_order) rows are always picked first. 
    OPTEXPApecs_solving_method: str                    = 'pulp'       # 'greedy' / 'pulp'
    

        

    

    def __post_init__(self):
        # have post init for less error prone scen setting. define T0 year and reference remaining settings to that.
        self.T0_prediction: str                 = f'{self.T0_year_prediction}-01-01 00:00:00'  
        # GBAUJ_min, GBAUJ_max = 1920, self.T0_year_prediction-1
        # self.GWRspec_GBAUJ_minmax: List[int]    = field(default_factory=lambda: [GBAUJ_min, GBAUJ_max])
        # self.GWRspec_GBAUJ_minmax[0] = 1920
        self.GWRspec_GBAUJ_minmax[1] = self.T0_year_prediction - 1
        self.TECspec_pvtarif_year    = self.T0_year_prediction - 1 if self.TECspec_pvtarif_year is None else self.TECspec_pvtarif_year

        self.TECspec_elecpri_year    = self.T0_year_prediction - 1 if self.TECspec_elecpri_year is None else self.TECspec_elecpri_year




class PVAllocScenario:
    """
    Entry point for a PV allocation scenario. Composes the three pipeline
    stages (PVAllocInitialization, PVAllocMCAlgo, PVAllocGridOpt) which all
    share the same PVAllocScenario_Settings instance, and dispatches to them
    the same way PVAllocScenario.RUN_pvalloc_scenario did in MAIN_pvallocation.py.
    """

    def __init__(self, settings: PVAllocScenario_Settings):
        self.sett = settings
        self.init_stage = PVAllocInitialization(settings)
        self.mc_stage = PVAllocMCAlgo(settings)
        self.gridopt_stage = PVAllocGridOpt(settings)


    # ------------------------------------------------------------------------------------------------------
    # EXECUTE PVALLOC SCENARIO
    # ------------------------------------------------------------------------------------------------------
    def RUN_pvalloc_scenario(self,):
        if self.sett.run_pvalloc_initalization_TF:
            self.init_stage.run_pvalloc_initialization()
        if self.sett.run_pvalloc_mcalgorithm_TF:
            self.mc_stage.run_pvalloc_mcalgorithm()
        if self.sett.run_gridoptimized_orderinst_TF:
            self.gridopt_stage.run_gridoptimized_orderinst()
        if self.sett.run_gridoptimized_expansion_TF:
            self.gridopt_stage.run_gridoptimized_expansion()




def make_scenario(default_scen, name_dir_export=None, name_dir_suffix=None, bfs_numbers=None, **overrides):
    if name_dir_export is not None and name_dir_suffix is not None:
        raise ValueError("Pass either name_dir_export or name_dir_suffix, not both.")
    kwargs = {}
    if name_dir_export is not None:
        kwargs['name_dir_export'] = name_dir_export
    elif name_dir_suffix is not None:
        kwargs['name_dir_export'] = f'{default_scen.name_dir_export}{name_dir_suffix}'
    if bfs_numbers is not None:
        kwargs['bfs_numbers'] = bfs_numbers
    if overrides:
        kwargs.update(overrides)
    return replace(default_scen, **kwargs)



if __name__ == "__main__":

    mini_dev_base_scen = PVAllocScenario_Settings(
        name_dir_export = 'pvalloc_minidev',
        name_dir_import = 'preprep_BLSO_15to24_extSolkatEGID__Oct26',
        run_pvalloc_initalization_TF      = True,     
        run_pvalloc_mcalgorithm_TF        = True, 
        # run_gridoptimized_orderinst_TF    = True,     
        # run_gridoptimized_expansion_TF    = True,     


        bfs_numbers                       = [
                                            2641, 2615,
                                            2761,
                                            ],         
        mini_sub_model_TF                 = True,
        mini_sub_model_by_X               = 'by_gridnode',
        mini_sub_model_ngridnodes         = 5,
        mini_sub_model_grid_nodes         = [
                                             '3', 
                                             '16', 
                                             # '28', 
                                             ],
        # mini_sub_model_select_EGIDs        = [
        #     '390449', '3029927', 
        # ],
        mini_sub_model_nEGIDs             = 500,
        create_gdf_export_of_topology     = False,
        export_csvs                       = False,
        months_prediction                 = 120,
        ALGOspec_topo_subdf_partitioner   = 250, 
        ALGOspec_rand_seed                = 42,
        # TECspec_elecpri_Rp_kWh            = 'elecpri_entsoe.parquet'
    )

    mini_dev_scen_list = [
        make_scenario(
            mini_dev_base_scen, 
            name_dir_suffix = '_max',
            CSTRspec_ext_constrcapa_path = os.path.join("postanalysis", "constracpa_adjustments", "constrcapa_pvalloc_LRG3_max_adj_13.parquet")
        ), 

    ]


    pvalloc_scen_list = mini_dev_scen_list  

    for pvalloc_scen in pvalloc_scen_list:
        pvalloc_class = PVAllocScenario(pvalloc_scen)
        
        pvalloc_class.RUN_pvalloc_scenario()


    print(f'... end of <{os.path.basename(__file__)}> ...')
    