import sys
import os as os
import pandas as pd
import glob
import datetime as datetime
import shutil
import pickle
from dataclasses import dataclass, field
from typing_extensions import List, Dict

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from auxiliary.auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, print_to_logfile, checkpoint_to_logfile, get_bfs_from_ktnr

import pv_allocation.initialization_small_functions as initial_sml
import pv_allocation.initialization_large_functions as  initial_lrg
import pv_allocation.alloc_algorithm as algo
# import pv_allocation.alloc_algorithm_SPARK as SPARKalgo
import pv_allocation.alloc_sanitychecks as sanity
import pv_allocation.inst_selection as select


# ---------------------
# *** PV ALLOCATION ***
# ---------------------


from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class PVAllocScenario_Defaults:
    # DEFAULT SETTINGS ---------------------------------------------------
    name_dir_export: str                        = 'pvalloc_BL_smallsample'   # name of the directory where the data is exported to (name to replace/ extend the name of the folder "preprep_data" in the end)
    name_dir_import: str                        = 'preprep_BL_22to23_extSolkatEGID'
    show_debug_prints: bool                     = False                    # F: certain print statements are omitted, T: includes print statements that help with debugging
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
    
    T0_prediction: str                          = '2022-01-01 00:00:00'         # start date for the prediction of the future construction capacity
    months_lookback: int                        = 12                           # number of months to look back for the prediction of the future construction capacity
    months_prediction: int                      = 12                         # number of months to predict the future construction capacity
    
    recreate_topology: bool                     = True
    recalc_economics_topo_df: bool              = True
    sanitycheck_byEGID: bool                    = True
    create_gdf_export_of_topology: bool         = True
    
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
    GWRspec_GBAUJ_minmax: List[int]                     = field(default_factory=lambda: [1950, 2022])
    
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


class PVAllocScenario:
    # DEFAULT SETTINGS ---------------------------------------------------
    def __init__(self, 
                 name_dir_export:str   =  1, #'pvalloc_BL_smallsample',             # name of the directory where the data is exported to (name to replace/ extend the name of the folder "preprep_data" in the end)
                 name_dir_import: str   = 'preprep_BL_22to23_extSolkatEGID',
                 show_debug_prints: bool = False,                       # F: certain print statements are omitted, T: includes print statements that help with debugging
                 export_csvs: bool      = False,

                 kt_numbers: List[int] = [],                               # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain individual municipalities
                 bfs_numbers: List[int] = [
                     ],                                      # list of municipalites to select for allocation (only used if kt_numbers == 0)

                 T0_prediction: str = '2022-01-01 00:00:00',            # start date for the prediction of the future construction capacity
                 months_lookback: int    = 12,                              # number of months to look back for the prediction of the future construction capacity
                 months_prediction: int  = 12,                            # number of months to predict the future construction capacity

                 recreate_topology: bool              = True, 
                 recalc_economics_topo_df: bool       = True, 
                 sanitycheck_byEGID: bool             = True, 
                 create_gdf_export_of_topology: bool  = True, 
                 
                 # PART I: settings for alloc_initialization --------------------
                 # gwr_selection_specs
                 GWRspec_solkat_max_n_partitions: int    = 10,               # larger number of partitions make all combos of roof partitions practically impossible to calculate
                 GWRspec_solkat_area_per_EGID_range: List[int] = [2,600],          # for 100kWp inst, need 500m2 roof area => just above the threshold for residential subsidies KLEIV, below 2m2 too small to mount installations
                 GWRspec_building_cols: List[str]              =  ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                                                        'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                                                        'GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1', 'GEBF', 'GAREA'],
                 GWRspec_dwelling_cols: List[str]              = [],             # ['EGID', 'WAZIM', 'WAREA', ],
                 GWRspec_DEMAND_proxy: str               = 'GAREA',          # because WAZIM and WAREA are not available for all buildings (because not all building EGIDs have entries with WEIDs)
                 GWRspec_GSTAT: List[str]                      = ['1004',],        # GSTAT - 1004: only existing, fully constructed buildings 
                 GWRspec_GKLAS: List[str]                      = ['1110','1121'], #,'1276',],      # GKLAS - 1110: only 1 living space per building 
                 GWRspec_GBAUJ_minmax: List[int]               = [1950, 2022],     # GBAUJ_minmax: range of years of construction
                        # 'GWAERZH': ['7410', '7411',],       # GWAERZH - 7410: heat pumpt for 1 building, 7411: heat pump for multiple buildings
                        # 'GENH': ['7580', '7581', '7582'],   # GENHZU - 7580 to 7582: any type of Fernwärme/district heating        
                        # GANZWHG - total number of apartments in building
                        # GAZZI - total number of rooms in building

                # weather_specs
                WEAspec_meteo_col_dir_radiation: str     = 'Basel Direct Shortwave Radiation', 
                WEAspec_meteo_col_diff_radiation: str    = 'Basel Diffuse Shortwave Radiation',
                WEAspec_meteo_col_temperature: str       = 'Basel Temperature [2 m elevation corrected]', 
                WEAspec_weather_year: int                 = 2022,
                WEAspec_radiation_to_pvprod_method: str  = 'dfuid_ind',          #'flat', 'dfuid_ind'
                WEAspec_rad_rel_loc_max_by: str          = 'dfuid_specific',     # 'all_HOY', 'dfuid_specific'
                WEAspec_flat_direct_rad_factor: int      = 1,
                WEAspec_flat_diffuse_rad_factor: int     = 1,

                # constr_capacity_specs
                CSTRspec_ann_capacity_growth: int        = 0.05,         # annual growth of installed capacity# each year, X% more PV capacity can be built, 100% in year T0
                CSTRspec_constr_capa_overshoot_fact: int = 1, 
                CSTRspec_month_constr_capa_tuples: List   = [(1,  0.06), (2,  0.06), (3,  0.06), (4,  0.06), 
                                                       (5,  0.08), (6,  0.08), (7,  0.08), (8,  0.08), 
                                                       (9, 0.10),  (10, 0.10), (11, 0.12), (12, 0.12),
                                                       ],

                # tech_economic_specs
                TECspec_self_consumption_ifapplicable: float       = 1,
                TECspec_interest_rate: float                       = 0.01,
                TECspec_pvtarif_year: int                        = 2022, 
                TECspec_pvtarif_col: List[str]                         = ['energy1', 'eco1'],
                TECspec_pvprod_calc_method: str               = 'method2.2',
                TECspec_panel_efficiency: float                    = 0.21,         # XY% Wirkungsgrad PV Modul
                TECspec_inverter_efficiency: int                 = 0.95,         # XY% Wirkungsgrad Wechselrichter
                TECspec_elecpri_year: int                        = 2022,
                TECspec_elecpri_category: str                    = 'H4', 
                TECspec_invst_maturity: int                  = 25,
                TECspec_kWpeak_per_m2: float                       = 0.2,          # A 1m2 area can fit 0.2 kWp of PV Panels, 10kWp per 50m2; ASSUMPTION HECTOR: 300 Wpeak / 1.6 m2
                TECspec_share_roof_area_available: float           = 1,            # x% of the roof area is effectively available for PV installation  ASSUMPTION HECTOR: 70%¨
                TECspec_max_distance_m_for_EGID_node_matching: float = 0,          # max distance in meters for matching GWR EGIDs that have no node assignment to the next grid node
                TECspec_kW_range_for_pvinst_cost_estim: List[int]      =[0 , 61],      # max range 2 kW to 150
                TECspec_estim_pvinst_cost_correctionfactor: float = 1,

                # panel_efficiency_specs
                PEFspec_variable_panel_efficiency_TF: bool = True,
                PEFspec_summer_months: List[int]                = [6,7,8,9],
                PEFspec_hotsummer_hours: List[int]              = [11, 12, 13, 14, 15, 16, 17,],
                PEFspec_hot_hours_discount: float           = 0.1,

                # sanitycheck_summary_byEGID_specs
                CHECKspec_egid_list: List[str] = [                                             # ['3031017','1367570', '3030600',], # '1367570', '245017418'      # known houses in the sample in Laufen
                        '391292', '390601', '2347595', '401781'        # single roof houses in Aesch, Ettingen, 
                        '391263', '245057295', '401753',               # houses with built pv in Aesch, Ettingen,
                        
                        '245054165','245054166','245054175','245060521', # EGID selection of neighborhood within Aesch to analyse closer
                        '391253','391255','391257','391258','391262',
                        '391263','391289','391290','391291','391292',
                        '245057295', '245057294', '245011456', '391379', '391377'
                           ],
                CHECKspec_n_EGIDs_of_alloc_algorithm: int        = 20,
                CHECKspec_n_iterations_before_sanitycheck: int   = 1,

    
                # PART II: settings for MC algorithm --------------------
                # MC_loop_specs
                MCspec_montecarlo_iterations: int      = 1,
                MCspec_fresh_initial_files: List[str]        = ['topo_egid.json', 'months_prediction.parquet', 
                                                                'gridprem_ts.parquet', 'constrcapa.parquet', 
                                                                'dsonodes_df.parquet'],  #'gridnode_df.parquet',
                MCspec_keep_files_month_iter_TF: bool   = True,
                MCspec_keep_files_month_iter_max: int   = 9999999999,
                MCspec_keep_files_month_iter_list: List[str] = ['topo_egid.json', 'npv_df.parquet', 'pred_inst_df.parquet', 'gridprem_ts.parquet',], 

                # algorithm_specs
                ALGOspec_inst_selection_method: str             = 'random',   # random, prob_weighted_npv, max_npv 
                ALGOspec_rand_seed: bool                          = None,      # random seed set to int or None
                ALGOspec_while_inst_counter_max: int             = 5000,
                ALGOspec_topo_subdf_partitioner: int              = 400,
                ALGOspec_npv_update_groupby_cols_topo_aggdf: List[str]  =  ['EGID', 'df_uid', 'grid_node', 'bfs', 
                                                                'gklas', 'demandtype','inst_TF', 'info_source', 
                                                                'pvid', 'pv_tarif_Rp_kWh', 'elecpri_Rp_kWh', 
                                                                'FLAECHE', 'FLAECH_angletilt', 'AUSRICHTUNG', 
                                                                'NEIGUNG','STROMERTRAG'], 
                ALGOspec_npv_update_agg_cols_topo_aggdf: Dict     = {'pvprod_kW': 'sum', 'demand_kW': 'sum', 
                                                               'selfconsum_kW': 'sum', 'netdemand_kW': 'sum', 
                                                               'netfeedin_kW': 'sum', 'econ_inc_chf': 'sum', 
                                                               'econ_spend_chf': 'sum'}, 
                ALGOspec_tweak_constr_capacity_fact: float         = 1,
                ALGOspec_tweak_npv_calc: float                     = 1,
                ALGOspec_tweak_npv_excl_elec_demand: bool         = True,
                ALGOspec_tweak_gridnode_df_prod_demand_fact: float = 1,
                ALGOspec_constr_capa_overshoot_fact: float         =1, # not in that dir but should be a single tweak factor dict. 
                
                # gridprem_adjustment_specs
                GRIDspec_tier_description: str = 'tier_level: (voltage_threshold, gridprem_Rp_kWh)',
                GRIDspec_power_factor: float = 1, 
                GRIDspec_perf_factor_1kVA_to_XkW: float = 0.8,
                GRIDspec_colnames: List[str] = ['tier_level', 'used_node_capa_rate', 'gridprem_Rp_kWh'],
                GRIDspec_tiers: Dict = { 1: [0.7, 1], 2: [0.8,  3],  3: [0.85, 5], 
                                   4: [0.9, 7], 5: [0.95, 15], 6: [1, 100], 
                                   },


                # PART III: post processing of MC algorithm --------------------
                # ...
                ):

        # INITIALIZATION --------------------
        self.name_dir_export: str = name_dir_export
        self.name_dir_import: str = name_dir_import
        self.show_debug_prints: bool = show_debug_prints
        self.export_csvs: bool = export_csvs

        self.kt_numbers: List[int] = kt_numbers
        self.bfs_numbers: List[int] = bfs_numbers
        self.T0_prediction: str = T0_prediction
        self.months_lookback: int = months_lookback
        self.months_prediction: int = months_prediction

        self.recreate_topology: bool = recreate_topology
        self.recalc_economics_topo_df: bool = recalc_economics_topo_df
        self.sanitycheck_byEGID: bool = sanitycheck_byEGID
        self.create_gdf_export_of_topology: bool = create_gdf_export_of_topology

        self.GWRspec_solkat_max_n_partitions: int = GWRspec_solkat_max_n_partitions
        self.GWRspec_solkat_area_per_EGID_range: List[int] = GWRspec_solkat_area_per_EGID_range
        self.GWRspec_building_cols: List[str] = GWRspec_building_cols
        self.GWRspec_dwelling_cols: List[str] = GWRspec_dwelling_cols 
        self.GWRspec_DEMAND_proxy: str = GWRspec_DEMAND_proxy
        self.GWRspec_GSTAT: List[str] = GWRspec_GSTAT
        self.GWRspec_GKLAS: List[str] = GWRspec_GKLAS
        self.GWRspec_GBAUJ_minmax: List[int] = GWRspec_GBAUJ_minmax
        
        self.WEAspec_meteo_col_dir_radiation: str = WEAspec_meteo_col_dir_radiation
        self.WEAspec_meteo_col_diff_radiation: str = WEAspec_meteo_col_diff_radiation
        self.WEAspec_meteo_col_temperature: str = WEAspec_meteo_col_temperature
        self.WEAspec_weather_year: int = WEAspec_weather_year
        self.WEAspec_radiation_to_pvprod_method: str = WEAspec_radiation_to_pvprod_method
        self.WEAspec_rad_rel_loc_max_by: str = WEAspec_rad_rel_loc_max_by
        self.WEAspec_flat_direct_rad_factor: float = WEAspec_flat_direct_rad_factor
        self.WEAspec_flat_diffuse_rad_factor: float = WEAspec_flat_diffuse_rad_factor
        
        self.CSTRspec_ann_capacity_growth: float = CSTRspec_ann_capacity_growth
        self.CSTRspec_constr_capa_overshoot_fact: float = CSTRspec_constr_capa_overshoot_fact
        self.CSTRspec_month_constr_capa_tuples: List[tuple] = CSTRspec_month_constr_capa_tuples
        
        self.TECspec_self_consumption_ifapplicable: int = TECspec_self_consumption_ifapplicable
        self.TECspec_interest_rate: float = TECspec_interest_rate
        self.TECspec_pvtarif_year: int = TECspec_pvtarif_year
        self.TECspec_pvtarif_col: List[str] = TECspec_pvtarif_col
        self.TECspec_pvprod_calc_method: str = TECspec_pvprod_calc_method
        self.TECspec_panel_efficiency: float = TECspec_panel_efficiency
        self.TECspec_inverter_efficiency: float = TECspec_inverter_efficiency
        self.TECspec_elecpri_year: int = TECspec_elecpri_year
        self.TECspec_elecpri_category: str = TECspec_elecpri_category
        self.TECspec_invst_maturity: int = TECspec_invst_maturity
        self.TECspec_kWpeak_per_m2: float = TECspec_kWpeak_per_m2
        self.TECspec_share_roof_area_available: float = TECspec_share_roof_area_available
        self.TECspec_max_distance_m_for_EGID_node_matching: int = TECspec_max_distance_m_for_EGID_node_matching
        self.TECspec_kW_range_for_pvinst_cost_estim: List[float] = TECspec_kW_range_for_pvinst_cost_estim
        self.TECspec_estim_pvinst_cost_correctionfactor: float = TECspec_estim_pvinst_cost_correctionfactor
        
        self.PEFspec_variable_panel_efficiency_TF: bool = PEFspec_variable_panel_efficiency_TF
        self.PEFspec_summer_months: List[int] = PEFspec_summer_months
        self.PEFspec_hotsummer_hours: List[int] = PEFspec_hotsummer_hours
        self.PEFspec_hot_hours_discount: float = PEFspec_hot_hours_discount
        
        self.CHECKspec_egid_list: List[str] = CHECKspec_egid_list
        self.CHECKspec_n_EGIDs_of_alloc_algorithm: int = CHECKspec_n_EGIDs_of_alloc_algorithm
        self.CHECKspec_n_iterations_before_sanitycheck: int = CHECKspec_n_iterations_before_sanitycheck
        
        self.MCspec_montecarlo_iterations: int = MCspec_montecarlo_iterations
        self.MCspec_fresh_initial_files: List[str] = MCspec_fresh_initial_files
        self.MCspec_keep_files_month_iter_TF: bool = MCspec_keep_files_month_iter_TF
        self.MCspec_keep_files_month_iter_max: int = MCspec_keep_files_month_iter_max
        self.MCspec_keep_files_month_iter_list: List[str] = MCspec_keep_files_month_iter_list
        
        self.ALGOspec_inst_selection_method: str = ALGOspec_inst_selection_method
        self.ALGOspec_rand_seed: int = ALGOspec_rand_seed
        self.ALGOspec_while_inst_counter_max: int = ALGOspec_while_inst_counter_max
        self.ALGOspec_topo_subdf_partitioner: int = ALGOspec_topo_subdf_partitioner
        self.ALGOspec_npv_update_groupby_cols_topo_aggdf: List[str] = ALGOspec_npv_update_groupby_cols_topo_aggdf
        self.ALGOspec_npv_update_agg_cols_topo_aggdf: Dict[str, str] = ALGOspec_npv_update_agg_cols_topo_aggdf
        self.ALGOspec_tweak_constr_capacity_fact: float = ALGOspec_tweak_constr_capacity_fact
        self.ALGOspec_tweak_npv_calc: float = ALGOspec_tweak_npv_calc
        self.ALGOspec_tweak_npv_excl_elec_demand: bool = ALGOspec_tweak_npv_excl_elec_demand
        self.ALGOspec_tweak_gridnode_df_prod_demand_fact: float = ALGOspec_tweak_gridnode_df_prod_demand_fact
        self.ALGOspec_constr_capa_overshoot_fact: float = ALGOspec_constr_capa_overshoot_fact

        self.GRIDspec_tier_description: str = GRIDspec_tier_description
        self.GRIDspec_power_factor: float = GRIDspec_power_factor
        self.GRIDspec_perf_factor_1kVA_to_XkW: float = GRIDspec_perf_factor_1kVA_to_XkW
        self.GRIDspec_colnames: List[str] = GRIDspec_colnames
        self.GRIDspec_tiers: Dict[int, List[float]] = GRIDspec_tiers

        # SETUP --------------------
        self.wd_path = os.getcwd()
        self.data_path = os.path.join(self.wd_path, 'data')
        self.pvalloc_path = os.path.join(self.data_path, 'pvalloc', 'pvalloc_scen__temp_to_be_renamed')
        self.name_dir_export_path = os.path.join(self.data_path, 'pvalloc', self.name_dir_export)
        self.name_dir_import_path = os.path.join(self.data_path, 'preprep', self.name_dir_import)

    def export_class_attr_to_pickle(self):
        # export class instance to pickle
        with open(f'{self.name_dir_export_path}/{self.name_dir_export}_attributes.pkl', 'wb') as f: 
            pickle.dump(self, f)


    def run_pvalloc_initalization(self):
        """
        Input:
            (preprep data directory defined in the pv allocation scenario settings)
            dict: pvalloc_settings_func
                > settings for pv allocation scenarios, for initalization and Monte Carlo iterations

        Output (no function return but export to dir):
            > directory renamed after scenario name (pvalloc_scenario), containing all data files form the INITIALIZATION of the pv allocation run.

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
        self.sanity_check_path = f'{self.name_dir_export_path}/sanity_check_byEGID'

        # self.dir_move_to = os.path.join(self.data_path, 'pvalloc', self.name_dir_export)

        self.log_name = os.path.join(self.name_dir_export_path, 'pvalloc_log.txt')
        self.summary_name = os.path.join(self.name_dir_export_path, 'summary_data_selection_log.txt')
        
        self.bfs_numbers: List[str] = get_bfs_from_ktnr(self.kt_numbers, self.data_path, self.log_name) if self.kt_numbers != [] else [str(bfs) for bfs in self.bfs_numbers]
        self.total_runtime_start = datetime.datetime.now()

        # create dir for export, rename old export dir not to overwrite
        if os.path.exists(self.name_dir_export_path):
            n_same_names = len(glob.glob(f'{self.name_dir_export_path}*'))
            os.rename(self.name_dir_export_path, f'{self.name_dir_export_path}_{n_same_names}_old_vers')
        os.makedirs(self.name_dir_export_path, exist_ok=True)

        # export class instance to pickle
        self.export_class_attr_to_pickle()


        # create log file
        chapter_to_logfile(f'start MAIN_pvalloc_INITIALIZATION for: {self.name_dir_export}', self.log_name, overwrite_file=True)
        subchapter_to_logfile('pvalloc_settings', self.log_name)
        for k, v in vars(self).items():
            print_to_logfile(f'{k}: {v}', self.log_name)

        # create summary file
        chapter_to_logfile('OptimalPV - Sample Summary of Building Topology', self.summary_name, overwrite_file=True)
        # chapter_to_logfile('pv_allocation', self.summary_name)



        # CREATE TOPOLOGY ---------------------------------------------------------------------------------------------
        subchapter_to_logfile('initialization: CREATE SMALLER AID DFs', self.log_name)
        initial_sml.HOY_weatheryear_df(self)
        initial_sml.get_gridnodes_DSO(self)
        initial_sml.estimate_iterpolate_instcost_function(self)

        if self.recreate_topology:
            subchapter_to_logfile('initialization: IMPORT PREPREP DATA & CREATE (building) TOPOLOGY', self.log_name)
            topo, df_list, df_names = initial_lrg.import_prepre_AND_create_topology(self)

            subchapter_to_logfile('initialization: IMPORT TS DATA', self.log_name)
            ts_list, ts_names = initial_lrg.import_ts_data(self)

            subchapter_to_logfile('initialization: DEFINE CONSTRUCTION CAPACITY', self.log_name)
            constrcapa, months_prediction, months_lookback = initial_lrg.define_construction_capacity(self, topo, df_list, df_names, ts_list, ts_names)

        

        # CALC ECONOMICS + TOPO_TIME SPECIFIC DFs ---------------------------------------------------------------------------------------------
        subchapter_to_logfile('prep: CALC ECONOMICS for TOPO_DF', self.log_name)
        # SPARKalgo.calc_economics_in_topo_df(self, topo, df_list, df_names, ts_list, ts_names)
        algo.calc_economics_in_topo_df(self, topo, df_list, df_names, ts_list, ts_names)
        shutil.copy(f'{self.name_dir_export_path}/topo_egid.json', f'{self.name_dir_export_path}/topo_egid_before_alloc.json')



        # SANITY CHECK: CALC FEW ITERATION OF NPV AND FEEDIN for check ---------------------------------------------------------------
        subchapter_to_logfile('sanity_check: RUN FEW ITERATION for byCHECK', self.log_name)
        # make sanitycheck folder and move relevant initial files there (delete all old files, not distort results)
        os.makedirs(self.sanity_check_path, exist_ok=False) 

        fresh_initial_files = [f'{self.name_dir_export_path}/{file}' for file in ['topo_egid.json', 'gridprem_ts.parquet', 'dsonodes_df.parquet']]
        topo_time_paths = glob.glob(f'{self.name_dir_export_path}/topo_time_subdf/*.parquet')
        for f in fresh_initial_files + topo_time_paths:
            shutil.copy(f, f'{self.sanity_check_path}/')

        # ALLOCATION RUN ====================
        dfuid_installed_list = []
        pred_inst_df = pd.DataFrame()
        months_prediction_pq = pd.read_parquet(f'{self.name_dir_export_path}/months_prediction.parquet')['date']
        months_prediction = [str(m) for m in months_prediction_pq]
        # i_m, m = 1, months_prediction[0:2]
        for i_m, m in enumerate(months_prediction[0:self.CHECKspec_n_iterations_before_sanitycheck]):
            print_to_logfile(f'\n-- month {m} -- sanity check -- {self.name_dir_export} --', self.log_name)
            algo.update_gridprem(self, self.sanity_check_path, m, i_m)
            algo.update_npv_df(self, self.sanity_check_path, m, i_m)
            select.select_AND_adjust_topology(self, self.sanity_check_path,
                                            dfuid_installed_list,pred_inst_df,
                                            m, i_m)
        
        sanity.sanity_check_summary_byEGID(self, self.sanity_check_path)
        
        # EXPORT SPATIAL DATA ====================
        if self.create_gdf_export_of_topology:
            subchapter_to_logfile('sanity_check: CREATE SPATIAL EXPORTS OF TOPOLOGY_DF', self.log_name)
            sanity.create_gdf_export_of_topology(self)  

            subchapter_to_logfile('sanity_check: MULTIPLE INSTALLATIONS PER EGID', self.log_name)
            sanity.check_multiple_xtf_ids_per_EGID(self)


        # END ---------------------------------------------------
        chapter_to_logfile(f'end start MAIN_pvalloc_INITIALIZATION\n Runtime (hh:mm:ss):{datetime.datetime.now() - self.total_runtime_start}', self.log_name)


    def run_pvalloc_mcalgorithm(self):
        """
        Input: 
            (preprep data directory defined in the pv allocation scenario settings)
            (pvalloc data directory defined in the pv allocation scenario settings)
            dict: pvalloc_settings_func
                > settings for pv allocation scenarios, for initalization and Monte Carlo iterations
       
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
        self.log_name = os.path.join(self.name_dir_export_path, 'pvalloc_MCalgo_log.txt')
        # self.summary_name = os.path.join(self.name_dir_export_path, 'summary_data_selection_log_MC.txt')
        total_runtime_start = datetime.datetime.now()

        # create log file
        chapter_to_logfile(f'start MAIN_pvalloc_MCalgorithm for : {self.name_dir_export}', self.log_name, overwrite_file=True)
        print_to_logfile('*model allocation specifications*:', self.log_name)
        print_to_logfile(f'> n_bfs_municipalities: {len(self.bfs_numbers)} \n> n_months_prediction: {self.months_prediction} \n> n_montecarlo_iterations: {self.MCspec_montecarlo_iterations}', self.log_name)
        print_to_logfile(f'> pvalloc_settings, MCalloc_{self.name_dir_export}', self.log_name)
        for k, v in vars(self).items():
            print_to_logfile(f'{k}: {v}', self.log_name)



        # CREATE MC DIR + TRANSFER INITIAL DATA FILES ----------------------------------------------
        montecarlo_iterations = [*range(1, self.MCspec_montecarlo_iterations+1, 1)]
        safety_counter_max = self.ALGOspec_while_inst_counter_max
        
        # get all initial files to start a fresh MC iteration
        fresh_initial_paths = [f'{self.name_dir_export_path}/{file}' for file in self.MCspec_fresh_initial_files]
        topo_time_paths = glob.glob(f'{self.name_dir_export_path}/topo_time_subdf/topo_subdf*.parquet')

        max_digits = len(str(max(montecarlo_iterations)))
        # mc_iter = montecarlo_iterations[0]
        # if True:    
        for mc_iter in montecarlo_iterations:
            mc_iter_start = datetime.datetime.now()
            subchapter_to_logfile(f'START MC{mc_iter:0{max_digits}} iteration', self.log_name)

            # copy all initial files to MC directory
            self.mc_iter_path = os.path.join(self.name_dir_export_path, f'zMC{mc_iter:0{max_digits}}')
            shutil.rmtree(self.mc_iter_path) if os.path.exists(self.mc_iter_path) else None
            os.makedirs(self.mc_iter_path, exist_ok=False)
            for f in fresh_initial_paths + topo_time_paths:
                shutil.copy(f, self.mc_iter_path)



            # ALLOCATION ALGORITHM -----------------------------------------------------------------------------    
            dfuid_installed_list = []
            pred_inst_df = pd.DataFrame()  
            months_prediction_df = pd.read_parquet(f'{self.mc_iter_path}/months_prediction.parquet')
            months_prediction = months_prediction_df['date']
            constrcapa = pd.read_parquet(f'{self.mc_iter_path}/constrcapa.parquet')

            for i_m, m in enumerate(months_prediction):
                print_to_logfile(f'\n-- month {m} -- iter MC{mc_iter:0{max_digits}} -- {self.name_dir_export} --', self.log_name)
                start_allocation_month = datetime.datetime.now()
                i_m = i_m + 1        

                # GRIDPREM + NPV_DF UPDATE ==========
                algo.update_gridprem(self, self.mc_iter_path, m, i_m)
                npv_df = algo.update_npv_df(self, self.mc_iter_path, m, i_m)

                # init constr capa ==========
                constr_built_m = 0
                if m.year != (m-1).year:
                    constr_built_y = 0
                constr_capa_m = constrcapa.loc[constrcapa['date'] == m, 'constr_capacity_kw'].iloc[0]
                constr_capa_y = constrcapa.loc[constrcapa['year'].isin([m.year]), 'constr_capacity_kw'].sum()

                # INST PICK ==========
                safety_counter = 0
                print_to_logfile('start inst pick while loop', self.log_name)
                while( (constr_built_m <= constr_capa_m) & (constr_built_y <= constr_capa_y) & (safety_counter <= safety_counter_max) ):
                    
                    if npv_df.shape[0] == 0:
                        checkpoint_to_logfile(' npv_df is EMPTY, exit while loop', self.log_name, 1, self.self.show_debug_prints)                    
                        safety_counter = safety_counter_max

                    if npv_df.shape[0] > 0: 
                        # checkpoint_to_logfile(f' npv_df with 0 < rows, select inst and adjust topology', log_name, 1, self.show_debug_prints)                    
                        inst_power, npv_df = select.select_AND_adjust_topology(self, 
                                                        self.mc_iter_path,
                                                        dfuid_installed_list, 
                                                        pred_inst_df,
                                                        m, i_m)  

                    # Loop Exit + adjust constr_built capacity ----------
                    constr_built_m, constr_built_y, safety_counter = constr_built_m + inst_power, constr_built_y + inst_power, safety_counter + 1
                    overshoot_rate = self.CSTRspec_constr_capa_overshoot_fact
                    constr_m_TF, constr_y_TF, safety_TF = constr_built_m > constr_capa_m*overshoot_rate, constr_built_y > constr_capa_y*overshoot_rate, safety_counter > safety_counter_max

                    if any([constr_m_TF, constr_y_TF, safety_TF]):
                        print_to_logfile('exit While Loop', self.log_name)
                        if constr_m_TF:
                            checkpoint_to_logfile(f'exceeded constr_limit month (constr_m_TF:{constr_m_TF}), {round(constr_built_m,1)} of {round(constr_capa_m,1)} kW capacity built', self.log_name, 1, self.show_debug_prints)                    
                        if constr_y_TF:
                            checkpoint_to_logfile(f'exceeded constr_limit year (constr_y_TF:{constr_y_TF}), {round(constr_built_y,1)} of {round(constr_capa_y,1)} kW capacity built', self.log_name, 1, self.show_debug_prints)                    
                        if safety_TF:
                            checkpoint_to_logfile(f'exceeded safety counter (safety_TF:{safety_TF}), {safety_counter} rounds for safety counter max of: {safety_counter_max}', self.log_name, 1, self.show_debug_prints)                    

                        if constr_m_TF or constr_y_TF:    
                            checkpoint_to_logfile(f'{safety_counter} pv installations allocated', self.log_name, 3, self.show_debug_prints)                                        

                checkpoint_to_logfile(f'end month allocation, runtime: {datetime.datetime.now() - start_allocation_month} (hh:mm:ss.s)', self.log_name, 1, self.show_debug_prints)                    

            # CLEAN UP interim files of MC run ==========
            files_to_remove_paths =  glob.glob(f'{self.mc_iter_path}/topo_subdf_*.parquet')
            for f in files_to_remove_paths:
                os.remove(f)

            mc_iter_time = datetime.datetime.now() - mc_iter_start
            subchapter_to_logfile(f'END MC{mc_iter:0{max_digits}}, runtime: {mc_iter_time} (hh:mm:ss.s)', self.log_name)
            print_to_logfile('\n', self.log_name)

        
        # END ---------------------------------------------------
        chapter_to_logfile(f'end MAIN_pvalloc_MCalgorithm\n Runtime (hh:mm:ss):{datetime.datetime.now() - total_runtime_start}', self.log_name, overwrite_file=False)
        os.rename(self.log_name, f'{self.name_dir_export_path}/pvalloc_MCalgo_log_{self.name_dir_export}.txt')


    def run_pvalloc_postprocess(self):

        # SETUP -----------------------------------------------------------------------------
        self.log_name = os.path.join(self.name_dir_export_path, 'pvalloc_postprocess_log.txt')
        self.postprocess_path = os.path.join(self.data_path, 'postprocess')
        os.makedirs(self.postprocess_path)

        total_runtime_start = datetime.datetime.now()

        # create log file
        chapter_to_logfile(f'start MAIN_pvalloc_postprocess for : {self.name_dir_export}', self.log_name, overwrite_file=True)
        print_to_logfile('*model allocation specifications*:', self.log_name)
        print_to_logfile(f'> n_bfs_municipalities: {len(self.bfs_numbers)} \n> n_months_prediction: {self.months_prediction} \n> n_montecarlo_iterations: {self.MCspec_montecarlo_iterations}', self.log_name)
        print_to_logfile(f'> pvalloc_settings, MCalloc_{self.name_dir_export}', self.log_name)
        for k, v in vars(self).items():
            print_to_logfile(f'{k}: {v}', self.log_name)

 
        # POSTPROCESSING  ----------------------------------------------
        mc_dir_paths = glob.glob(f'{self.name_dir_export_path}/zMC*')

        for i_mc_dir, mc_dir in enumerate(mc_dir_paths):

            # PREDICTION ACCURACY CALCULATION
             


            
            print('asdf')


# ==================================================================================================================
pvalloc_scen_list = [
    # PVAllocScenario(
    #     name_dir_export    = 'pvalloc_BFS2761_2m_f2021_1mc_meth2.2_rnd_DEBUG',
    #     name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
    #     show_debug_prints  = True,
    #     export_csvs        = True,
    #     T0_prediction      = '2021-01-01 00:00:00',            # start date for the prediction of the future construction capacity
    #     months_prediction  = 2,
    #     GWRspec_GBAUJ_minmax = [1920, 2020],
    #     ALGOspec_inst_selection_method = 'random',
    #     TECspec_pvprod_calc_method = 'method2.2',
    #     MCspec_montecarlo_iterations = 2,
    #     ),

    # pvalloc_BLsml_13y_f2010_1mc_meth2.2_npv        
    PVAllocScenario(
        name_dir_export    = 'pvalloc_BLsml_13y_f2010_1mc_meth2.2_npv',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_prediction      = '2010-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 156,
        GWRspec_GBAUJ_minmax = [1920, 2009],
        ALGOspec_inst_selection_method = 'prob_weighted_npv',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 1,
        ),
    # pvalloc_BLsml_13y_f2010_1mc_meth2.2_rnd
    PVAllocScenario(
        name_dir_export    = 'pvalloc_BLsml_13y_f2010_1mc_meth2.2_rnd',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_prediction      = '2010-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 156,
        GWRspec_GBAUJ_minmax = [1920, 2009],
        ALGOspec_inst_selection_method = 'random',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 1,
        ),
    # pvalloc_BLsml_13y_f2010_1mc_meth2.2_max
    PVAllocScenario(
        name_dir_export    = 'pvalloc_BLsml_13y_f2010_1mc_meth2.2_max',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_prediction      = '2010-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 156,
        GWRspec_GBAUJ_minmax = [1920, 2009],
        ALGOspec_inst_selection_method = 'max_npv',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 1,
        ),
   
   
    # pvalloc_BLsml_13y_f2010_5mc_meth2.2_npv        
    PVAllocScenario(
        name_dir_export    = 'pvalloc_BLsml_13y_f2010_5mc_meth2.2_npv',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_prediction      = '2010-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 156,
        GWRspec_GBAUJ_minmax = [1920, 2009],
        ALGOspec_inst_selection_method = 'prob_weighted_npv',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 5,
        ),
    # pvalloc_BLsml_13y_f2010_5mc_meth2.2_rnd
    PVAllocScenario(
        name_dir_export    = 'pvalloc_BLsml_13y_f2010_5mc_meth2.2_rnd',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_prediction      = '2010-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 156,
        GWRspec_GBAUJ_minmax = [1920, 2009],
        ALGOspec_inst_selection_method = 'random',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 5,
        ),
    # pvalloc_BLsml_13y_f2010_5mc_meth2.2_max
    PVAllocScenario(
        name_dir_export    = 'pvalloc_BLsml_13y_f2010_5mc_meth2.2_max',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [2767, 2771, 2765, 2764,  ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_prediction      = '2010-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 156,
        GWRspec_GBAUJ_minmax = [1920, 2009],
        ALGOspec_inst_selection_method = 'max_npv',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 5,
        ),
    

    # pvalloc_BLSOmed_10y_f2013_5mc_meth2.2_npv        
    PVAllocScenario(
        name_dir_export    = 'pvalloc_BLSOmed_10y_f2013_5mc_meth2.2_npv',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [        
            2768, 2761, 2772, 2785, 2787,
            2473, 2475, 2480,        
            ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
        T0_prediction      = '2013-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 120,
        GWRspec_GBAUJ_minmax = [1920, 2012],
        ALGOspec_inst_selection_method = 'prob_weighted_npv',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 5,
        ),
    # pvalloc_BLSOmed_10y_f2013_5mc_meth2.2_rnd
    PVAllocScenario(
        name_dir_export    = 'pvalloc_BLSOmed_10y_f2013_5mc_meth2.2_rnd',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [        
            2768, 2761, 2772, 2785, 2787,
            2473, 2475, 2480,        
            ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)        
        T0_prediction      = '2013-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 120,
        GWRspec_GBAUJ_minmax = [1920, 2012],
        ALGOspec_inst_selection_method = 'random',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 5,
        ),
    # pvalloc_BLSOmed_10y_f2013_5mc_meth2.2_max
    PVAllocScenario(
        name_dir_export    = 'pvalloc_BLSOmed_10y_f2013_5mc_meth2.2_max',
        name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
        show_debug_prints  = True,
        bfs_numbers        = [        
            2768, 2761, 2772, 2785, 2787,
            2473, 2475, 2480,        
            ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)       
        T0_prediction      = '2013-01-01 00:00:00',            # start date for the prediction of the future construction capacity
        months_prediction  = 120,
        GWRspec_GBAUJ_minmax = [1920, 2012],
        ALGOspec_inst_selection_method = 'max_npv',
        TECspec_pvprod_calc_method = 'method2.2',
        MCspec_montecarlo_iterations = 5,
        ),


]

if __name__ == '__main__':

    for pvalloc_scen in pvalloc_scen_list:
        pvalloc_scen.export_class_attr_to_pickle()
        # pvalloc_scen.run_pvalloc_initalization()
        # pvalloc_scen.run_pvalloc_mcalgorithm()
        # pvalloc_scen.run_pvalloc_postprocess()

print('done')






    # # pvalloc_BLsml_20y_f2003_1mc_meth2.2_npv
    # PVAllocScenario(
    #     name_dir_export    = 'pvalloc_BLsml_20y_f2003_1mc_meth2.2_npv',
    #     name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
    #     bfs_numbers        = [2768, 2761, 2772, 2785, ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
    #     T0_prediction      = '2003-01-01 00:00:00',            # start date for the prediction of the future construction capacity
    #     months_prediction  = 240,
    #     GWRspec_GBAUJ_minmax = [1920, 2012],
    #     ALGOspec_inst_selection_method = 'prob_weighted_npv',
    #     TECspec_pvprod_calc_method = 'method2.2',
    #     MCspec_montecarlo_iterations = 1,
    #     ),
    # # pvalloc_BLsml_20y_f2003_1mc_meth2.2_rnd        
    # PVAllocScenario(
    #     name_dir_export    = 'pvalloc_BLsml_20y_f2003_1mc_meth2.2_rnd',
    #     name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
    #     bfs_numbers        = [2768, 2761, 2772, 2785, ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
    #     T0_prediction      = '2003-01-01 00:00:00',            # start date for the prediction of the future construction capacity
    #     months_prediction  = 240,
    #     GWRspec_GBAUJ_minmax = [1920, 2012],
    #     ALGOspec_inst_selection_method = 'random',
    #     TECspec_pvprod_calc_method = 'method2.2',
    #     MCspec_montecarlo_iterations = 1,
    #     ),
    # # pvalloc_BLsml_20y_f2003_1mc_meth2.2_max
    # PVAllocScenario(
    #     name_dir_export    = 'pvalloc_BLsml_20y_f2003_1mc_meth2.2_max',
    #     name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
    #     bfs_numbers        = [2768, 2761, 2772, 2785, ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
    #     T0_prediction      = '2003-01-01 00:00:00',            # start date for the prediction of the future construction capacity
    #     months_prediction  = 240,
    #     GWRspec_GBAUJ_minmax = [1920, 2012],
    #     ALGOspec_inst_selection_method = 'max_npv',
    #     TECspec_pvprod_calc_method = 'method2.2',
    #     MCspec_montecarlo_iterations = 1,
    #     ),


    # # pvalloc_BLsml_40y_f1983_1mc_meth2.2_npv
    # PVAllocScenario(
    #     name_dir_export    = 'pvalloc_BLsml_40y_f1983_1mc_meth2.2_npv',
    #     name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
    #     bfs_numbers        = [2768, 2761, 2772, 2785, ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
    #     T0_prediction      = '1983-01-01 00:00:00',            # start date for the prediction of the future construction capacity
    #     months_prediction  = 480,
    #     GWRspec_GBAUJ_minmax = [1920, 2012],
    #     ALGOspec_inst_selection_method = 'prob_weighted_npv',
    #     TECspec_pvprod_calc_method = 'method2.2',
    #     MCspec_montecarlo_iterations = 1,
    #     ),
    # # pvalloc_BLsml_40y_f1983_1mc_meth2.2_rnd
    # PVAllocScenario(
    #     name_dir_export    = 'pvalloc_BLsml_40y_f1983_1mc_meth2.2_rnd',
    #     name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
    #     bfs_numbers        = [2768, 2761, 2772, 2785, ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
    #     T0_prediction      = '1983-01-01 00:00:00',            # start date for the prediction of the future construction capacity
    #     months_prediction  = 480,
    #     GWRspec_GBAUJ_minmax = [1920, 2012],
    #     ALGOspec_inst_selection_method = 'max',
    #     TECspec_pvprod_calc_method = 'method2.2',
    #     MCspec_montecarlo_iterations = 1,
    #     ),
    # # pvalloc_BLsml_40y_f1983_1mc_meth2.2_max
    # PVAllocScenario(
    #     name_dir_export    = 'pvalloc_BLsml_40y_f1983_1mc_meth2.2_max',
    #     name_dir_import    = 'preprep_BL_22to23_extSolkatEGID',
    #     bfs_numbers        = [2768, 2761, 2772, 2785, ],        # list of municipalites to select for allocation (only used if kt_numbers == 0)
    #     T0_prediction      = '1983-01-01 00:00:00',            # start date for the prediction of the future construction capacity
    #     months_prediction  = 480,
    #     GWRspec_GBAUJ_minmax = [1920, 2012],
    #     ALGOspec_inst_selection_method = 'max_npv',
    #     TECspec_pvprod_calc_method = 'method2.2',
    #     MCspec_montecarlo_iterations = 1,
    #     ),





class PVAllocScenario_OLD:
    # DEFAULT SETTINGS ---------------------------------------------------
    def __init__(self, 
                 name_dir_export:str   =  1, #'pvalloc_BL_smallsample',             # name of the directory where the data is exported to (name to replace/ extend the name of the folder "preprep_data" in the end)
                 name_dir_import: str   = 'preprep_BL_22to23_extSolkatEGID',
                 show_debug_prints: bool = False,                       # F: certain print statements are omitted, T: includes print statements that help with debugging
                 export_csvs: bool      = False,

                 kt_numbers: List[int] = [],                               # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain individual municipalities
                 bfs_numbers: List[int] = [
                     ],                                      # list of municipalites to select for allocation (only used if kt_numbers == 0)

                 T0_prediction: str = '2022-01-01 00:00:00',            # start date for the prediction of the future construction capacity
                 months_lookback: int    = 12,                              # number of months to look back for the prediction of the future construction capacity
                 months_prediction: int  = 12,                            # number of months to predict the future construction capacity

                 recreate_topology: bool              = True, 
                 recalc_economics_topo_df: bool       = True, 
                 sanitycheck_byEGID: bool             = True, 
                 create_gdf_export_of_topology: bool  = True, 
                 
                 # PART I: settings for alloc_initialization --------------------
                 # gwr_selection_specs
                 GWRspec_solkat_max_n_partitions: int    = 10,               # larger number of partitions make all combos of roof partitions practically impossible to calculate
                 GWRspec_solkat_area_per_EGID_range: List[int] = [2,600],          # for 100kWp inst, need 500m2 roof area => just above the threshold for residential subsidies KLEIV, below 2m2 too small to mount installations
                 GWRspec_building_cols: List[str]              =  ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                                                        'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                                                        'GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1', 'GEBF', 'GAREA'],
                 GWRspec_dwelling_cols: List[str]              = [],             # ['EGID', 'WAZIM', 'WAREA', ],
                 GWRspec_DEMAND_proxy: str               = 'GAREA',          # because WAZIM and WAREA are not available for all buildings (because not all building EGIDs have entries with WEIDs)
                 GWRspec_GSTAT: List[str]                      = ['1004',],        # GSTAT - 1004: only existing, fully constructed buildings 
                 GWRspec_GKLAS: List[str]                      = ['1110','1121'], #,'1276',],      # GKLAS - 1110: only 1 living space per building 
                 GWRspec_GBAUJ_minmax: List[int]               = [1950, 2022],     # GBAUJ_minmax: range of years of construction
                        # 'GWAERZH': ['7410', '7411',],       # GWAERZH - 7410: heat pumpt for 1 building, 7411: heat pump for multiple buildings
                        # 'GENH': ['7580', '7581', '7582'],   # GENHZU - 7580 to 7582: any type of Fernwärme/district heating        
                        # GANZWHG - total number of apartments in building
                        # GAZZI - total number of rooms in building

                # weather_specs
                WEAspec_meteo_col_dir_radiation: str     = 'Basel Direct Shortwave Radiation', 
                WEAspec_meteo_col_diff_radiation: str    = 'Basel Diffuse Shortwave Radiation',
                WEAspec_meteo_col_temperature: str       = 'Basel Temperature [2 m elevation corrected]', 
                WEAspec_weather_year: int                 = 2022,
                WEAspec_radiation_to_pvprod_method: str  = 'dfuid_ind',          #'flat', 'dfuid_ind'
                WEAspec_rad_rel_loc_max_by: str          = 'dfuid_specific',     # 'all_HOY', 'dfuid_specific'
                WEAspec_flat_direct_rad_factor: int      = 1,
                WEAspec_flat_diffuse_rad_factor: int     = 1,

                # constr_capacity_specs
                CSTRspec_ann_capacity_growth: int        = 0.05,         # annual growth of installed capacity# each year, X% more PV capacity can be built, 100% in year T0
                CSTRspec_constr_capa_overshoot_fact: int = 1, 
                CSTRspec_month_constr_capa_tuples: List   = [(1,  0.06), (2,  0.06), (3,  0.06), (4,  0.06), 
                                                       (5,  0.08), (6,  0.08), (7,  0.08), (8,  0.08), 
                                                       (9, 0.10),  (10, 0.10), (11, 0.12), (12, 0.12),
                                                       ],

                # tech_economic_specs
                TECspec_self_consumption_ifapplicable: float       = 1,
                TECspec_interest_rate: float                       = 0.01,
                TECspec_pvtarif_year: int                        = 2022, 
                TECspec_pvtarif_col: List[str]                         = ['energy1', 'eco1'],
                TECspec_pvprod_calc_method: str               = 'method2.2',
                TECspec_panel_efficiency: float                    = 0.21,         # XY% Wirkungsgrad PV Modul
                TECspec_inverter_efficiency: int                 = 0.95,         # XY% Wirkungsgrad Wechselrichter
                TECspec_elecpri_year: int                        = 2022,
                TECspec_elecpri_category: str                    = 'H4', 
                TECspec_invst_maturity: int                  = 25,
                TECspec_kWpeak_per_m2: float                       = 0.2,          # A 1m2 area can fit 0.2 kWp of PV Panels, 10kWp per 50m2; ASSUMPTION HECTOR: 300 Wpeak / 1.6 m2
                TECspec_share_roof_area_available: float           = 1,            # x% of the roof area is effectively available for PV installation  ASSUMPTION HECTOR: 70%¨
                TECspec_max_distance_m_for_EGID_node_matching: float = 0,          # max distance in meters for matching GWR EGIDs that have no node assignment to the next grid node
                TECspec_kW_range_for_pvinst_cost_estim: List[int]      =[0 , 61],      # max range 2 kW to 150
                TECspec_estim_pvinst_cost_correctionfactor: float = 1,

                # panel_efficiency_specs
                PEFspec_variable_panel_efficiency_TF: bool = True,
                PEFspec_summer_months: List[int]                = [6,7,8,9],
                PEFspec_hotsummer_hours: List[int]              = [11, 12, 13, 14, 15, 16, 17,],
                PEFspec_hot_hours_discount: float           = 0.1,

                # sanitycheck_summary_byEGID_specs
                CHECKspec_egid_list: List[str] = [                                             # ['3031017','1367570', '3030600',], # '1367570', '245017418'      # known houses in the sample in Laufen
                        '391292', '390601', '2347595', '401781'        # single roof houses in Aesch, Ettingen, 
                        '391263', '245057295', '401753',               # houses with built pv in Aesch, Ettingen,
                        
                        '245054165','245054166','245054175','245060521', # EGID selection of neighborhood within Aesch to analyse closer
                        '391253','391255','391257','391258','391262',
                        '391263','391289','391290','391291','391292',
                        '245057295', '245057294', '245011456', '391379', '391377'
                           ],
                CHECKspec_n_EGIDs_of_alloc_algorithm: int        = 20,
                CHECKspec_n_iterations_before_sanitycheck: int   = 1,

    
                # PART II: settings for MC algorithm --------------------
                # MC_loop_specs
                MCspec_montecarlo_iterations: int      = 1,
                MCspec_fresh_initial_files: List[str]        = ['topo_egid.json', 'months_prediction.parquet', 
                                                                'gridprem_ts.parquet', 'constrcapa.parquet', 
                                                                'dsonodes_df.parquet'],  #'gridnode_df.parquet',
                MCspec_keep_files_month_iter_TF: bool   = True,
                MCspec_keep_files_month_iter_max: int   = 9999999999,
                MCspec_keep_files_month_iter_list: List[str] = ['topo_egid.json', 'npv_df.parquet', 'pred_inst_df.parquet', 'gridprem_ts.parquet',], 

                # algorithm_specs
                ALGOspec_inst_selection_method: str             = 'random',   # random, prob_weighted_npv, max_npv 
                ALGOspec_rand_seed: bool                          = None,      # random seed set to int or None
                ALGOspec_while_inst_counter_max: int             = 5000,
                ALGOspec_topo_subdf_partitioner: int              = 400,
                ALGOspec_npv_update_groupby_cols_topo_aggdf: List[str]  =  ['EGID', 'df_uid', 'grid_node', 'bfs', 
                                                                'gklas', 'demandtype','inst_TF', 'info_source', 
                                                                'pvid', 'pv_tarif_Rp_kWh', 'elecpri_Rp_kWh', 
                                                                'FLAECHE', 'FLAECH_angletilt', 'AUSRICHTUNG', 
                                                                'NEIGUNG','STROMERTRAG'], 
                ALGOspec_npv_update_agg_cols_topo_aggdf: Dict     = {'pvprod_kW': 'sum', 'demand_kW': 'sum', 
                                                               'selfconsum_kW': 'sum', 'netdemand_kW': 'sum', 
                                                               'netfeedin_kW': 'sum', 'econ_inc_chf': 'sum', 
                                                               'econ_spend_chf': 'sum'}, 
                ALGOspec_tweak_constr_capacity_fact: float         = 1,
                ALGOspec_tweak_npv_calc: float                     = 1,
                ALGOspec_tweak_npv_excl_elec_demand: bool         = True,
                ALGOspec_tweak_gridnode_df_prod_demand_fact: float = 1,
                ALGOspec_constr_capa_overshoot_fact: float         =1, # not in that dir but should be a single tweak factor dict. 
                
                # gridprem_adjustment_specs
                GRIDspec_tier_description: str = 'tier_level: (voltage_threshold, gridprem_Rp_kWh)',
                GRIDspec_power_factor: float = 1, 
                GRIDspec_perf_factor_1kVA_to_XkW: float = 0.8,
                GRIDspec_colnames: List[str] = ['tier_level', 'used_node_capa_rate', 'gridprem_Rp_kWh'],
                GRIDspec_tiers: Dict = { 1: [0.7, 1], 2: [0.8,  3],  3: [0.85, 5], 
                                   4: [0.9, 7], 5: [0.95, 15], 6: [1, 100], 
                                   },


                # PART III: post processing of MC algorithm --------------------
                # ...
                ):

        # INITIALIZATION --------------------
        self.name_dir_export: str = name_dir_export
        self.name_dir_import: str = name_dir_import
        self.show_debug_prints: bool = show_debug_prints
        self.export_csvs: bool = export_csvs

        self.kt_numbers: List[int] = kt_numbers
        self.bfs_numbers: List[int] = bfs_numbers
        self.T0_prediction: str = T0_prediction
        self.months_lookback: int = months_lookback
        self.months_prediction: int = months_prediction

        self.recreate_topology: bool = recreate_topology
        self.recalc_economics_topo_df: bool = recalc_economics_topo_df
        self.sanitycheck_byEGID: bool = sanitycheck_byEGID
        self.create_gdf_export_of_topology: bool = create_gdf_export_of_topology

        self.GWRspec_solkat_max_n_partitions: int = GWRspec_solkat_max_n_partitions
        self.GWRspec_solkat_area_per_EGID_range: List[int] = GWRspec_solkat_area_per_EGID_range
        self.GWRspec_building_cols: List[str] = GWRspec_building_cols
        self.GWRspec_dwelling_cols: List[str] = GWRspec_dwelling_cols 
        self.GWRspec_DEMAND_proxy: str = GWRspec_DEMAND_proxy
        self.GWRspec_GSTAT: List[str] = GWRspec_GSTAT
        self.GWRspec_GKLAS: List[str] = GWRspec_GKLAS
        self.GWRspec_GBAUJ_minmax: List[int] = GWRspec_GBAUJ_minmax
        
        self.WEAspec_meteo_col_dir_radiation: str = WEAspec_meteo_col_dir_radiation
        self.WEAspec_meteo_col_diff_radiation: str = WEAspec_meteo_col_diff_radiation
        self.WEAspec_meteo_col_temperature: str = WEAspec_meteo_col_temperature
        self.WEAspec_weather_year: int = WEAspec_weather_year
        self.WEAspec_radiation_to_pvprod_method: str = WEAspec_radiation_to_pvprod_method
        self.WEAspec_rad_rel_loc_max_by: str = WEAspec_rad_rel_loc_max_by
        self.WEAspec_flat_direct_rad_factor: float = WEAspec_flat_direct_rad_factor
        self.WEAspec_flat_diffuse_rad_factor: float = WEAspec_flat_diffuse_rad_factor
        
        self.CSTRspec_ann_capacity_growth: float = CSTRspec_ann_capacity_growth
        self.CSTRspec_constr_capa_overshoot_fact: float = CSTRspec_constr_capa_overshoot_fact
        self.CSTRspec_month_constr_capa_tuples: List[tuple] = CSTRspec_month_constr_capa_tuples
        
        self.TECspec_self_consumption_ifapplicable: int = TECspec_self_consumption_ifapplicable
        self.TECspec_interest_rate: float = TECspec_interest_rate
        self.TECspec_pvtarif_year: int = TECspec_pvtarif_year
        self.TECspec_pvtarif_col: List[str] = TECspec_pvtarif_col
        self.TECspec_pvprod_calc_method: str = TECspec_pvprod_calc_method
        self.TECspec_panel_efficiency: float = TECspec_panel_efficiency
        self.TECspec_inverter_efficiency: float = TECspec_inverter_efficiency
        self.TECspec_elecpri_year: int = TECspec_elecpri_year
        self.TECspec_elecpri_category: str = TECspec_elecpri_category
        self.TECspec_invst_maturity: int = TECspec_invst_maturity
        self.TECspec_kWpeak_per_m2: float = TECspec_kWpeak_per_m2
        self.TECspec_share_roof_area_available: float = TECspec_share_roof_area_available
        self.TECspec_max_distance_m_for_EGID_node_matching: int = TECspec_max_distance_m_for_EGID_node_matching
        self.TECspec_kW_range_for_pvinst_cost_estim: List[float] = TECspec_kW_range_for_pvinst_cost_estim
        self.TECspec_estim_pvinst_cost_correctionfactor: float = TECspec_estim_pvinst_cost_correctionfactor
        
        self.PEFspec_variable_panel_efficiency_TF: bool = PEFspec_variable_panel_efficiency_TF
        self.PEFspec_summer_months: List[int] = PEFspec_summer_months
        self.PEFspec_hotsummer_hours: List[int] = PEFspec_hotsummer_hours
        self.PEFspec_hot_hours_discount: float = PEFspec_hot_hours_discount
        
        self.CHECKspec_egid_list: List[str] = CHECKspec_egid_list
        self.CHECKspec_n_EGIDs_of_alloc_algorithm: int = CHECKspec_n_EGIDs_of_alloc_algorithm
        self.CHECKspec_n_iterations_before_sanitycheck: int = CHECKspec_n_iterations_before_sanitycheck
        
        self.MCspec_montecarlo_iterations: int = MCspec_montecarlo_iterations
        self.MCspec_fresh_initial_files: List[str] = MCspec_fresh_initial_files
        self.MCspec_keep_files_month_iter_TF: bool = MCspec_keep_files_month_iter_TF
        self.MCspec_keep_files_month_iter_max: int = MCspec_keep_files_month_iter_max
        self.MCspec_keep_files_month_iter_list: List[str] = MCspec_keep_files_month_iter_list
        
        self.ALGOspec_inst_selection_method: str = ALGOspec_inst_selection_method
        self.ALGOspec_rand_seed: int = ALGOspec_rand_seed
        self.ALGOspec_while_inst_counter_max: int = ALGOspec_while_inst_counter_max
        self.ALGOspec_topo_subdf_partitioner: int = ALGOspec_topo_subdf_partitioner
        self.ALGOspec_npv_update_groupby_cols_topo_aggdf: List[str] = ALGOspec_npv_update_groupby_cols_topo_aggdf
        self.ALGOspec_npv_update_agg_cols_topo_aggdf: Dict[str, str] = ALGOspec_npv_update_agg_cols_topo_aggdf
        self.ALGOspec_tweak_constr_capacity_fact: float = ALGOspec_tweak_constr_capacity_fact
        self.ALGOspec_tweak_npv_calc: float = ALGOspec_tweak_npv_calc
        self.ALGOspec_tweak_npv_excl_elec_demand: bool = ALGOspec_tweak_npv_excl_elec_demand
        self.ALGOspec_tweak_gridnode_df_prod_demand_fact: float = ALGOspec_tweak_gridnode_df_prod_demand_fact
        self.ALGOspec_constr_capa_overshoot_fact: float = ALGOspec_constr_capa_overshoot_fact

        self.GRIDspec_tier_description: str = GRIDspec_tier_description
        self.GRIDspec_power_factor: float = GRIDspec_power_factor
        self.GRIDspec_perf_factor_1kVA_to_XkW: float = GRIDspec_perf_factor_1kVA_to_XkW
        self.GRIDspec_colnames: List[str] = GRIDspec_colnames
        self.GRIDspec_tiers: Dict[int, List[float]] = GRIDspec_tiers

        # SETUP --------------------
        self.wd_path = os.getcwd()
        self.data_path = os.path.join(self.wd_path, 'data')
        self.pvalloc_path = os.path.join(self.data_path, 'pvalloc', 'pvalloc_scen__temp_to_be_renamed')
        self.name_dir_export_path = os.path.join(self.data_path, 'pvalloc', self.name_dir_export)
        self.name_dir_import_path = os.path.join(self.data_path, 'preprep', self.name_dir_import)

    def export_class_attr_to_pickle(self):
        # export class instance to pickle
        with open(f'{self.name_dir_export_path}/{self.name_dir_export}_attributes.pkl', 'wb') as f: 
            pickle.dump(self, f)

