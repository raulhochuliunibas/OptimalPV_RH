import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.MAIN_pvallocation_Class import PVAllocScenario_Settings, PVAllocScenario, make_scenario


# ==============================
# Default Sub-Scenarios 
# ==============================
if True: 
    # make_scenario() is imported from src.MAIN_pvallocation_Class (single shared definition)
    


    # region: Default Scenario Init ------------------------------------------------


    pvalloc_Xnbfs_DEFAULT = PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_DEFAULT',
        name_dir_import                 = 'preprep_BLSO_15to24_extSolkatEGID__Oct2_26', 
        bfs_numbers                     = [
                                    # RURAL 
                                    2612, 2889, 2883, 2621, 2622,
                                    2620, 2615, 2614, 2616, 2480,
                                    2617, 2611, 2788, 2619, 2783, 2477, 
                                    # SUBURBAN
                                    2613, 2782, 2618, 2786, 2785, 
                                    2772, 2761, 2743, 2476, 2768,
                                    # URBAN
                                    2773, 2769, 2770,
        ],

        run_pvalloc_initalization_TF    = True,
        run_pvalloc_mcalgorithm_TF      = True,
        run_gridoptimized_orderinst_TF  = False,
        run_gridoptimized_expansion_TF  = False,

        create_gdf_export_of_topology   = True,
        export_csvs                     = False,
        T0_year_prediction              = 2024,
        months_prediction               = 360,
        ALGOspec_inst_selection_method  = 'max_npv',
        ALGOspec_topo_subdf_partitioner = 250,

        TECspec_elecpri_Rp_kWh          = 31.98,
        TECspec_pvtarif_Rp_kWh          = 9.03,
    ) 


    DEV_default_scen = make_scenario(pvalloc_Xnbfs_DEFAULT,
                                        name_dir_export = 'pvalloc_DEV', 
                                        bfs_numbers = [
                                            2614, 2615, # RUR
                                            2761, 2785, # SUB
                                            2621,

                                            # critical nodes - max npv
                                            2762, 2771, 
                                            # critical nodes - ew 
                                            2768, 2769,
                                        ], 
                                        months_prediction         = 360,
                                        mini_sub_model_TF         = True,
                                        mini_sub_model_by_X       = 'by_gridnode',
                                        mini_sub_model_ngridnodes = 50,
                                        mini_sub_model_grid_nodes = [
                                                                        '514', 
                                                                        '511', 
                                                                        '412', 
                                                                        '411', 
                                                                        '415', 
                                                                        ],
                                        # mini_sub_model_nEGIDs     = 5000,
    )

    # DEV_default_scen = make_scenario(pvalloc_Xnbfs_DEFAULT,
    #                                     name_dir_export = 'pvalloc_DEV', 
    #                                     months_prediction         = 360,
    #                                     mini_sub_model_TF         = True,
    #                                     ALGOspec_rand_seed        = 42,

    #                                     # mini_sub_model_by_X       = 'by_gridnode',
    #                                     # mini_sub_model_ngridnodes = 200,
    #                                     # mini_sub_model_grid_nodes = [
    #                                     #                                 '514', 
    #                                     #                                 '511', 
    #                                     #                                 '412', 
    #                                     #                                 '411', 
    #                                     #                                 '415', 
    #                                     #                                 ],
    #                                     # mini_sub_model_nEGIDs     = 500,

    #                                     mini_sub_model_by_X       = 'by_EGID',
    #                                     mini_sub_model_nEGIDs     = 5000,

    # )
    # endregion


# ==============================
# Lists Sub-Scenarios 
# ==============================

# Helper Functions
def keep_scen_pattern_filter(scen_list           = None, 
                             scen_pattern_filter = None,
                             ): 
    scen_list = [] if scen_list is None else scen_list
    if scen_pattern_filter is None:
        return scen_list

    remain_scen = [scen for scen in scen_list 
                   if any(pattern in scen.name_dir_export for pattern in scen_pattern_filter)]

    return remain_scen   

        
# Sub-Scenario Functions
def create_MAIN_scen_list(scen_list          = None, 
                                 default_scenario   = pvalloc_Xnbfs_DEFAULT, 
                                 scen_pattern_filter = None):
    scen_list = [] if scen_list is None else scen_list      # no mutable default: a shared [] would accumulate across calls
    f_scen_list = [

        make_scenario(default_scenario,
                      name_dir_suffix = '_max_fixep2023'
                      ),
        make_scenario(default_scenario,
                      name_dir_suffix = '_max_hoyep2023',
                      TECspec_elecpri_Rp_kWh   = 'elecpri_entsoe.parquet',
                      ),
    ]

    scen_list.extend(f_scen_list)
    scen_list = keep_scen_pattern_filter(scen_list, scen_pattern_filter)
    
    return scen_list


def add_MAIN_subsidy_scen_list(scen_list          = None,
                               default_scenario   = pvalloc_Xnbfs_DEFAULT, 
                               scen_pattern_filter = None):
    scen_list = [] if scen_list is None else scen_list      # no mutable default: a shared [] would accumulate across calls
    f_scen_list = [

        # subsidies
        make_scenario(default_scenario,
                      name_dir_suffix = '_max_kev',
                      ),
        make_scenario(default_scenario,
                      name_dir_suffix = '_max_nokev',
                      TECspec_KEVflatsubs = 0.0,
                      ),

        make_scenario(default_scenario,
                      name_dir_suffix = '_max_As8p0',
                      GRIDspec_subsidy_name = 'As8p0',
                      ),
        make_scenario(default_scenario, 
                      name_dir_suffix = '_max_Bs0p6',
                      GRIDspec_subsidy_name = 'Bs0p6',
                      GRIDspec_reupdate_npv_df_TF = True, 
                      ),
        make_scenario(default_scenario,
                      name_dir_suffix = '_max_Bs0p8',
                      GRIDspec_subsidy_name = 'Bs0p8',
                      GRIDspec_reupdate_npv_df_TF = True, 
                      ),
        make_scenario(default_scenario,
                      name_dir_suffix = '_max_Cs6p6',
                      GRIDspec_subsidy_name = 'Cs6p6',
                      GRIDspec_reupdate_npv_df_TF = True, 
                      ),
        make_scenario(default_scenario,
                      name_dir_suffix = '_max_Cs6p8',
                      GRIDspec_subsidy_name = 'Cs6p8',
                      GRIDspec_reupdate_npv_df_TF = True, 
                      ),
        make_scenario(default_scenario,
                      name_dir_suffix = '_max_Cs8p6',
                      GRIDspec_subsidy_name = 'Cs8p6',
                      GRIDspec_reupdate_npv_df_TF = True, 
                      ),
        make_scenario(default_scenario,
                      name_dir_suffix = '_max_Cs8p8',
                      GRIDspec_subsidy_name = 'Cs8p8',
                      GRIDspec_reupdate_npv_df_TF = True, 
                      )
    ]

    scen_list.extend(f_scen_list)
    scen_list = keep_scen_pattern_filter(scen_list, scen_pattern_filter)
    
    return scen_list


def add_ADDITIONAL_scen_list(scen_list          = None,
                             default_scenario   = pvalloc_Xnbfs_DEFAULT, 
                             scen_pattern_filter = None):
    scen_list = [] if scen_list is None else scen_list      # no mutable default: a shared [] would accumulate across calls
    f_scen_list = [

        make_scenario(default_scenario,
                      name_dir_suffix = '_fixpv2023_rnd', 
                      ALGOspec_inst_selection_method = 'random',
                      ),
        make_scenario(default_scenario,
                      name_dir_suffix = '_fixpv2025_max', 
                      TECspec_elecpri_Rp_kWh          = 29.0,
                      TECspec_pvtarif_Rp_kWh          = 6.0,
                      ),
        make_scenario(default_scenario,
                      name_dir_suffix = '_hoypv2025_max',
                      TECspec_pvtarif_Rp_kWh   = 'pvtarif_entsoe.parquet',
                      TECspec_pvtarif_year     = 2025,
                      ),

        make_scenario(default_scenario,
                      name_dir_suffix = '_fixpv2023_gridoptim',
                      run_pvalloc_initalization_TF     = True,
                      run_pvalloc_mcalgorithm_TF       = False,
                      run_gridoptimized_orderinst_TF   = True,
                      run_gridoptimized_expansion_TF   = True,
                      OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel',
                      ),
        make_scenario(default_scenario,
                      name_dir_suffix = '_fixpv2023_max_histconstr_e40pv6', 
                      CSTRspec_ext_constrcapa_path = os.path.join("postanalysis", "constracpa_adjustments", "constrcapa_pvalloc_LRG3_max_adj_13.parquet"),
                    ),

        # DSO response
        make_scenario(default_scenario,
                      name_dir_suffix = '_fixpv2023_max_1hll',
                      GRIDspec_node_1hll_closed_TF = True,
                      ),
        make_scenario(default_scenario,
                      name_dir_suffix = '_fixpv2023_max_pksh07',
                      GRID_peak_shaving_enabled_tupl  = (True, 0.7, 0.7),
                      ),
    ]

    scen_list.extend(f_scen_list)
    scen_list = keep_scen_pattern_filter(scen_list, scen_pattern_filter)
    
    return scen_list




# less used scen lists
if False: 

    RUR_default_scen = make_scenario(pvalloc_Xnbfs_DEFAULT, 
                                     name_dir_export = 'pvalloc_16nbfs_RUR', 
                                     bfs_numbers = [
                                        # RURAL
                                        2612, 2889, 2883, 2621, 2622,
                                        2620, 2615, 2614, 2616, 2480,
                                        2617, 2611, 2788, 2619, 2783, 2477, 
                      ],
    )

    RUR_scen_list = [
    # pvalloc_scen_list = [
        make_scenario(RUR_default_scen, 
                      name_dir_suffix = '_max'
        ),
        make_scenario(RUR_default_scen, 
                      name_dir_suffix = '_max_sA1',
                      GRIDspec_subsidy_name = 'A1',
        ),      
    ]



    SUB_default_scen = make_scenario(pvalloc_Xnbfs_DEFAULT, 
                                     name_dir_export = 'pvalloc_10nbfs_SUB',
                                     bfs_numbers = [
                                        # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
                                        2613, 2782, 2618, 2786, 2785, 
                                        2772, 2761, 2743, 2476, 2768,
                                     ]
    )
                                     
    SUB_scen_list = [
    # pvalloc_scen_list = [
        make_scenario(SUB_default_scen,
                        name_dir_suffix = '_max'
            ),
        make_scenario(SUB_default_scen,
                        name_dir_suffix = '_max_sA1',
                        GRIDspec_subsidy_name = 'A1',
        ),
    ]



# DEV --------------------------------------------
DEV_scen_list = []
DEV_May26 = make_scenario(default_scen = DEV_default_scen, 
                          name_dir_suffix = '_May26', 
                          name_dir_import = 'preprep_BLSO_15to24_extSolkatEGID__May26',
)
DEV_scen_list = add_MAIN_subsidy_scen_list(scen_list = DEV_scen_list, default_scenario = DEV_May26)

DEV_Oct2_26 = make_scenario(default_scen = DEV_default_scen, 
                            name_dir_suffix = '_Oct2_26',
                            name_dir_import = 'preprep_BLSO_15to24_extSolkatEGID__Oct2_26', 
)
DEV_scen_list = add_MAIN_subsidy_scen_list(scen_list = DEV_scen_list, default_scenario = DEV_Oct2_26)

DEV_additional_scen = [
    # DEV_May26_gridoptim + rnd
    make_scenario(default_scen = DEV_May26, 
                  name_dir_suffix = '_gridoptim',
                  run_pvalloc_initalization_TF     = True,
                  run_pvalloc_mcalgorithm_TF       = False,
                  run_gridoptimized_orderinst_TF   = True,
                  run_gridoptimized_expansion_TF   = True,
                  OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel',
                  OPTEXPApecs_apply_gridoptim_order_TF     = True,
    ), 
    make_scenario(default_scen = DEV_May26, 
                  name_dir_suffix = '_rnd',
                  ALGOspec_inst_selection_method = 'random',
    ), 

    # DEV_Oct2_26_gridoptim
    make_scenario(default_scen = DEV_Oct2_26,
                  name_dir_suffix = '_gridoptim',
                  run_pvalloc_initalization_TF     = True,
                  run_pvalloc_mcalgorithm_TF       = False,
                  run_gridoptimized_orderinst_TF   = True,
                  run_gridoptimized_expansion_TF   = True,
                  OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel',
                  OPTEXPApecs_apply_gridoptim_order_TF     = True,
    ),    
    make_scenario(default_scen = DEV_Oct2_26, 
                  name_dir_suffix = '_rnd',
                  ALGOspec_inst_selection_method = 'random',
    ), 

    #  DEV_May26 cummax, sumFLACH, cummax_sumFLACH
    make_scenario(default_scen = DEV_default_scen,
                  name_dir_suffix = '_May26_cummax_Cs8p8', 
                  name_dir_import = 'preprep_BLSO_15to24_extSolkatEGID__May26',
                  GRIDspec_subsidy_name = 'Cs8p8',
                  GRIDspec_reupdate_npv_df_TF = True, 
                  ALGOspec_inst_selection_cumulative_TF = True, 
                  #   ALGOspec_rfr_FLAECHE_total_sum_TF = True, 
    ), 
    make_scenario(default_scen = DEV_default_scen,
                  name_dir_suffix = '_May26_max_sumFLACH_Cs8p8', 
                  name_dir_import = 'preprep_BLSO_15to24_extSolkatEGID__May26',
                  GRIDspec_subsidy_name = 'Cs8p8',
                  GRIDspec_reupdate_npv_df_TF = True, 
                  #   ALGOspec_inst_selection_cumulative_TF = True, 
                  ALGOspec_rfr_FLAECHE_total_sum_TF = True, 
    ), 
    make_scenario(default_scen = DEV_default_scen,
                  name_dir_suffix = '_May26_cummax_sumFLACH_Cs8p8', 
                  name_dir_import = 'preprep_BLSO_15to24_extSolkatEGID__May26',
                  GRIDspec_subsidy_name = 'Cs8p8',
                  GRIDspec_reupdate_npv_df_TF = True, 
                  ALGOspec_inst_selection_cumulative_TF = True, 
                  ALGOspec_rfr_FLAECHE_total_sum_TF = True, 
    )
]

DEV_scen_list = DEV_scen_list + DEV_additional_scen

DEV_suffix_filters = [
    # '_May26_rnd',
    # '_May26_max_kev', 
    # '_May26_max_As8p0', 
    '_May26_max_Bs0p8',
    # '_May26_max_Cs8p8',
    # '_May26_gridoptim', 

    # '_Oct2_26_rnd',
    # '_Oct2_26_max_kev', 
    # '_Oct2_26_max_As8p0',
    # '_Oct2_26_max_Bs0p8',
    # '_Oct2_26_max_Cs8p8',
    # '_Oct2_26_gridoptim',

    # '_May26_cummax_Cs8p8', 
    # '_May26_max_sumFLACH_Cs8p8',
    # '_May26_cummax_sumFLACH_Cs8p8',

]                    
DEV_scen_list = keep_scen_pattern_filter(scen_list = DEV_scen_list, scen_pattern_filter = DEV_suffix_filters)


for scen in DEV_scen_list:
    print(scen.name_dir_export)




# LRG --------------------------------------------
LRG_default_scen = make_scenario(pvalloc_Xnbfs_DEFAULT,
                                    name_dir_export = 'pvalloc_LRG',   # 'pvalloc_26nbfs_LRG',
                                    bfs_numbers = [
                                    # RURAL 
                                    2612, 2889, 2883, 2621, 2622,
                                    2620, 2615, 2614, 2616, 2480,
                                    2617, 2611, 2788, 2619, 2783, 2477, 
                                    # SUBURBAN
                                    2613, 2782, 2618, 2786, 2785, 
                                    2772, 2761, 2743, 2476, 2768,
                                    # URBAN
                                    2773, 2769, 2770,
                                    ],
)


LRG_scen_list = []
# fixed_elecpri_2023 = make_scenario(LRG_default_scen, name_dir_suffix = '_max_fixep2023')
# dynam_elecpri_2023 = make_scenario(LRG_default_scen, name_dir_suffix = '_max_hoyep2023',
#                                                           TECspec_elecpri_year     = 2023,
#                                                           TECspec_elecpri_Rp_kWh   = 'elecpri_entsoe.parquet',
# )
# LRG_scen_list = add_MAIN_subsidy_scen_list(scen_list = LRG_scen_list, default_scenario = fixed_elecpri_2023)
# LRG_scen_list = add_MAIN_subsidy_scen_list(scen_list = LRG_scen_list, default_scenario = dynam_elecpri_2023)
# LRG_scen_list = add_ADDITIONAL_scen_list(scen_list = LRG_scen_list, default_scenario = fixed_elecpri_2023)

fixed_pvt2023 = make_scenario(LRG_default_scen, name_dir_suffix = '_fixpv2023')
LRG_scen_list = add_MAIN_subsidy_scen_list(scen_list = LRG_scen_list, default_scenario = fixed_pvt2023)


dynam_pvt2023 = make_scenario(LRG_default_scen, name_dir_suffix = '_hoypv2023',
                                                TECspec_pvtarif_year     = 2023,
                                                TECspec_pvtarif_Rp_kWh   = 'pvtarif_entsoe.parquet',)               
LRG_scen_list = add_MAIN_subsidy_scen_list(scen_list = LRG_scen_list, default_scenario = dynam_pvt2023)


add_fixed_pvt2023 = make_scenario(LRG_default_scen, name_dir_suffix = '')
LRG_scen_list = add_ADDITIONAL_scen_list(  scen_list = LRG_scen_list, default_scenario = add_fixed_pvt2023)

LRG_filters = [
    'pvalloc_LRG5_fixpv2023_max_kev', 
    # 'pvalloc_LRG5_fixpv2023_max_nokev', 
    # 'pvalloc_LRG5_fixpv2023_max_As8p0', 
    # 'pvalloc_LRG5_fixpv2023_max_Bs0p6', 
    'pvalloc_LRG5_fixpv2023_max_Bs0p8', 
    # 'pvalloc_LRG5_fixpv2023_max_Cs6p6', 
    # 'pvalloc_LRG5_fixpv2023_max_Cs6p8', 
    # 'pvalloc_LRG5_fixpv2023_max_Cs8p6', 
    'pvalloc_LRG5_fixpv2023_max_Cs8p8', 
    'pvalloc_LRG5_hoypv2023_max_kev', 
    # 'pvalloc_LRG5_hoypv2023_max_nokev', 
    # 'pvalloc_LRG5_hoypv2023_max_As8p0', 
    # 'pvalloc_LRG5_hoypv2023_max_Bs0p6', 
    'pvalloc_LRG5_hoypv2023_max_Bs0p8', 
    # 'pvalloc_LRG5_hoypv2023_max_Cs6p6', 
    # 'pvalloc_LRG5_hoypv2023_max_Cs6p8', 
    # 'pvalloc_LRG5_hoypv2023_max_Cs8p6', 
    'pvalloc_LRG5_hoypv2023_max_Cs8p8', 
    # 'pvalloc_LRG5_fixpv2023_rnd', 
    # 'pvalloc_LRG5_fixpv2025_max', 
    # 'pvalloc_LRG5_hoypv2025_max', 
    'pvalloc_LRG5_fixpv2023_gridoptim', 
    # 'pvalloc_LRG5_fixpv2023_max_histconstr_e40pv6', 
    # 'pvalloc_LRG5_fixpv2023_max_1hll', 
    # 'pvalloc_LRG5_fixpv2023_max_pksh07', 
]   

LRG_scen_list = keep_scen_pattern_filter(scen_list = LRG_scen_list, scen_pattern_filter = LRG_filters)
# for scen in LRG_scen_list:
#     print(scen.name_dir_export)





# LRG 5 ----------------------------------------------------------------


# LRG 5 + 6 Adjustments 
LRG_filter_patterns = [
    '_fixpv2023_max_kev', 
    '_fixpv2023_max_Bs0p8', 
    '_fixpv2023_max_Cs8p8', 
    '_hoypv2023_max_kev', 
    '_hoypv2023_max_Bs0p8', 
    '_hoypv2023_max_Cs8p8', 
    '_fixpv2023_gridoptim',  
]   

LRG_5_scen_list = []
LRG_5_default_scen = make_scenario(pvalloc_Xnbfs_DEFAULT,name_dir_export = 'pvalloc_LRG5', )

LRG_5_fixed_pvt2023 = make_scenario(LRG_5_default_scen, name_dir_suffix = '_fixpv2023')
LRG_5_scen_list = add_MAIN_subsidy_scen_list(scen_list = LRG_5_scen_list, default_scenario = LRG_5_fixed_pvt2023)

LRG_5_dynam_pvt2023 = make_scenario(LRG_5_default_scen, name_dir_suffix = '_hoypv2023',
                                                TECspec_pvtarif_year     = 2023,
                                                TECspec_pvtarif_Rp_kWh   = 'pvtarif_entsoe.parquet',)               
LRG_5_scen_list = add_MAIN_subsidy_scen_list(scen_list = LRG_5_scen_list, default_scenario = LRG_5_dynam_pvt2023)

LRG_5_add_fixed_pvt2023 = make_scenario(LRG_5_default_scen, name_dir_suffix = '')
LRG_5_scen_list = add_ADDITIONAL_scen_list(  scen_list = LRG_5_scen_list, default_scenario = LRG_5_add_fixed_pvt2023)


LRG_5_filter = [f'pvalloc_LRG5{pattern}' for pattern in LRG_filter_patterns]
LRG_5_scen_list = keep_scen_pattern_filter(scen_list = LRG_5_scen_list, scen_pattern_filter = LRG_5_filter)


# LRG 6 ----------------------------------------------------------------
LRG_6_scen_list = []
LRG_6_default_scen = make_scenario(pvalloc_Xnbfs_DEFAULT,name_dir_export = 'pvalloc_LRG6',
                                   ALGOspec_inst_selection_cumulative_TF = True
                                     )
LRG_6_fixed_pvt2023 = make_scenario(LRG_6_default_scen, name_dir_suffix = '_fixpv2023')
LRG_6_scen_list = add_MAIN_subsidy_scen_list(scen_list = LRG_6_scen_list, default_scenario = LRG_6_fixed_pvt2023)

LRG_6_dynam_pvt2023 = make_scenario(LRG_6_default_scen, name_dir_suffix = '_hoypv2023',
                                                TECspec_pvtarif_year     = 2023,
                                                TECspec_pvtarif_Rp_kWh   = 'pvtarif_entsoe.parquet',)               
LRG_6_scen_list = add_MAIN_subsidy_scen_list(scen_list = LRG_6_scen_list, default_scenario = LRG_6_dynam_pvt2023)


LRG_6_add_fixed_pvt2023 = make_scenario(LRG_6_default_scen, name_dir_suffix = '')
LRG_6_scen_list = add_ADDITIONAL_scen_list(  scen_list = LRG_6_scen_list, default_scenario = LRG_6_add_fixed_pvt2023)


LRG_6_filter = [f'pvalloc_LRG6{pattern}' for pattern in LRG_filter_patterns]
LRG_6_scen_list = keep_scen_pattern_filter(scen_list = LRG_6_scen_list, scen_pattern_filter = LRG_6_filter)


# LRG 7 ----------------------------------------------------------------
LRG_7_scen_list = []
LRG_7_default_scen = make_scenario(pvalloc_Xnbfs_DEFAULT,name_dir_export = 'pvalloc_LRG7',
                                   ALGOspec_inst_selection_cumulative_TF = True, 
                                   ALGOspec_rfr_FLAECHE_total_sum_TF     = True
                                     )
LRG_7_fixed_pvt2023 = make_scenario(LRG_7_default_scen, name_dir_suffix = '_fixpv2023')
LRG_7_scen_list = add_MAIN_subsidy_scen_list(scen_list = LRG_7_scen_list, default_scenario = LRG_7_fixed_pvt2023)

LRG_7_dynam_pvt2023 = make_scenario(LRG_7_default_scen, name_dir_suffix = '_hoypv2023',
                                                TECspec_pvtarif_year     = 2023,
                                                TECspec_pvtarif_Rp_kWh   = 'pvtarif_entsoe.parquet',)               
LRG_7_scen_list = add_MAIN_subsidy_scen_list(scen_list = LRG_7_scen_list, default_scenario = LRG_7_dynam_pvt2023)


LRG_7_add_fixed_pvt2023 = make_scenario(LRG_7_default_scen, name_dir_suffix = '')
LRG_7_scen_list = add_ADDITIONAL_scen_list(  scen_list = LRG_7_scen_list, default_scenario = LRG_7_add_fixed_pvt2023)


LRG_7_filter = [f'pvalloc_LRG6{pattern}' for pattern in LRG_filter_patterns]
LRG_7_scen_list = keep_scen_pattern_filter(scen_list = LRG_7_scen_list, scen_pattern_filter = LRG_7_filter)





asdf = LRG_5_scen_list + LRG_6_scen_list






XLRG_default_scen = make_scenario(pvalloc_Xnbfs_DEFAULT,
                                    name_dir_export = 'pvalloc_46nbfs_XLRG',
                                    bfs_numbers = [
                                        # RURAL 
                                        2612, 2889, 2883, 2621, 2622,
                                        2620, 2615, 2614, 2616, 2480,
                                        2617, 2611, 2788, 2619, 2783, 2477, 
                                        # SUBURBAN
                                        2613, 2782, 2618, 2786, 2785, 
                                        2772, 2761, 2743, 2476, 2768,
                                        2471, 2481, 2775, 2764, 2771, 
                                        2763, 2473, 2475, 2474, 2472, 
                                        2478, 2830, 2766, 2767, 2774, 
                                        # URBAN
                                        2773, 2769, 2770,
                                        2762, 2765,                                         
                                    ],
)


XALL_default_scen = make_scenario(pvalloc_Xnbfs_DEFAULT,
                                    name_dir_export = 'pvalloc_XALL_79nbfs',
                                    bfs_numbers = [
                                        # RURAL 
                                        2612, 2889, 2883, 2621, 2622,
                                        2620, 2615, 2614, 2616, 2480,
                                        2617, 2611, 2788, 2619, 2783, 2477, 
                                        2790, 2426, 2428, 2893, 
                                        2885, 2852, 2491, 2502, 2499,
                                        2585,
                                        # SUBURBAN
                                        2613, 2782, 2618, 2786, 2785, 
                                        2772, 2761, 2743, 2476, 2768,
                                        2471, 2481, 2775, 2764, 2771, 
                                        2763, 2473, 2475, 2474, 2472, 
                                        2478, 2830, 2766, 2767, 2774, 
                                        2422, 2792, 2787, 2789, 2479, 
                                        2834, 2833, 
                                        2579, 2582, 2586, 2500, 2501, 
                                        2493, 2497, 2495, 2584, 2573, 
                                        2572, 2576, 2583,
                                        # URBAN
                                        2773, 2769, 2770,
                                        2762, 2765, 
                                        2829, 2831, 
                                        2581,
        ],
)



# ==============================
# EXPORT Sub-Scenarios 
# ==============================
def get_subscen_list(sub_scen_str = 'test'):
    if sub_scen_str == 'RUR':
        return []
    elif sub_scen_str == 'SUB':
        return []
    elif sub_scen_str == 'LRG':
        return LRG_scen_list
    elif sub_scen_str == 'DEV':
        return DEV_scen_list
    else:
        return []
    

if __name__ == '__main__':

    # call scen in array and run ------------------------------------------

    for i in range(0,len(LRG_scen_list)):
        print(f"python HPC_scenario_runs/pvalloc_LRG.py    {i}")


    pvalloc_scen_list = get_subscen_list(sub_scen_str = 'DEV', )
    
    # # slurm_job_id = os.environ.get('SLURM_JOB_ID_ENV', 'unknown')
    # slurm_job_id = os.environ.get('SLURM_ARRAY_JOB_ID_ENV', 'unknown')
    # slurm_array_id = os.environ.get('SLURM_ARRAY_TASK_ID_ENV', 'unknown')
    # slurm_full_id = f"{slurm_job_id}_{slurm_array_id}"

    # if len(sys.argv) > 1:
    #     pvalloc_scen_index = int(sys.argv[1])
    #     if pvalloc_scen_index < len(pvalloc_scen_list):
    #         pvalloc_scen = pvalloc_scen_list[pvalloc_scen_index]
            
    #         scen_class = PVAllocScenario(pvalloc_scen)

    #         scen_class.sett.slurm_full_id        = slurm_full_id
    #         scen_class.sett.pvalloc_scen_index   = pvalloc_scen_index
    
    #         scen_class.RUN_pvalloc_scenario()

    if 'scicore' in os.getcwd() or 'SCICORE' in os.getcwd():

        for pvalloc_scen_index in range(0, len(pvalloc_scen_list)):
            pvalloc_scen = pvalloc_scen_list[pvalloc_scen_index]
            
            scen_class = PVAllocScenario(pvalloc_scen)
            scen_class.RUN_pvalloc_scenario()

    
    if 'c:\\Models' in os.getcwd() or 'C:\\Models' in os.getcwd():

        for pvalloc_scen_index in range(0, len(pvalloc_scen_list)):
            pvalloc_scen = pvalloc_scen_list[pvalloc_scen_index]
            
            scen_class = PVAllocScenario(pvalloc_scen)
            scen_class.RUN_pvalloc_scenario()

    print(f"end of file: {os.path.basename(__file__)} | n_scenarios: {len(pvalloc_scen_list)}")





# old scen lists (commented out in Oct 26) ===========================================================================================


if False:
    import json
    import pandas as pd
    topo = json.load(open(r"C:\Models\OptimalPV_RH\data\pvalloc\pvalloc_26nbfs_LRG_max\topo_egid.json", "r"))

    rows = []
    for k, v in topo.items():
        d = {
            "EGID": k,
            "grid_node": v['grid_node'],
        }
        rows.append(d)

    topo_df = pd.DataFrame(rows)

# LRG_constrcapa_scen_list = [

#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max', 
#     ),
#     # make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_epzb1', 
#     #               bfs_numbers                      = LRG_bfs_list, 
#     #               CSTRspec_capacity_type           = 'ep2050_zerobasis',
#     #               CSTRspec_ep2050_rescale_fact     = 1.0,
#     # ),
#     # make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_epzb025', 
#     #               bfs_numbers                      = LRG_bfs_list, 
#     #               CSTRspec_capacity_type           = 'ep2050_zerobasis',
#     #               CSTRspec_ep2050_rescale_fact     = 0.25,
#     # ),
#     # make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_hist01', 
#     #               bfs_numbers                      = LRG_bfs_list, 
#     #               CSTRspec_capacity_type           = 'hist_constr_capa_year',
#     #               CSTRspec_ann_capacity_growth     = 0.1,
#     # ),
#     # make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_elecpri60rp', 
#     #               bfs_numbers                      = LRG_bfs_list, 
#     #               TECspec_generic_elecpri_Rp_kWh   = 60.0,
#     # ),
#     # make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_elecpri15rp', 
#     #               bfs_numbers                      = LRG_bfs_list, 
#     #               TECspec_generic_elecpri_Rp_kWh   = 15.0,
#     # ),
    
#     # make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_rnd', 
#     #               bfs_numbers                      = LRG_bfs_list, 
#     #               ALGOspec_inst_selection_method   = 'random',
#     # ),
#     # make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_1hll', 
#     #               bfs_numbers                      = LRG_bfs_list, 
#     #               GRIDspec_node_1hll_closed_TF     = True, 
#     # ),
#     # make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_gridoptim', 
#     #               bfs_numbers                      = LRG_bfs_list, 
#     #               run_pvalloc_initalization_TF     = True,
#     #               run_pvalloc_mcalgorithm_TF       = False,
#     #               run_gridoptimized_orderinst_TF   = True,
#     #               run_gridoptimized_expansion_TF   = True,
#     #               OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
#     #               OPTEXPApecs_apply_gridoptim_order_TF     = True,
#     # ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_pksh',
#                     bfs_numbers                      = LRG_bfs_list, 
#                     GRID_peak_shaving_enabled_tupl  = (True, 0.7, 0.7),
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_nokev',
#                     bfs_numbers                      = LRG_bfs_list, 
#                     TECspec_KEVflatsubs              = 0.0,    
#     ),
# ]


# LRG_subsidy_scen_list = [ 

#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_sAs4p0',
#                                     bfs_numbers                      = LRG_bfs_list,
#                                     GRIDspec_subsidy_name             = 'As4p0',
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_sAs6p0',
#                                     bfs_numbers                      = LRG_bfs_list,
#                                     GRIDspec_subsidy_name             = 'As6p0',
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_sAs8p0',
#                                     bfs_numbers                      = LRG_bfs_list,
#                                     GRIDspec_subsidy_name             = 'As8p0',
#     ),

#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_sBs0p4',
#                                     bfs_numbers                      = LRG_bfs_list,
#                                     GRIDspec_apply_prem_tiers_TF      = True,
#                                     GRIDspec_subsidy_name             = 'Bs0p4',
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_sBs0p6',
#                                     bfs_numbers                      = LRG_bfs_list,
#                                     GRIDspec_apply_prem_tiers_TF      = True,
#                                     GRIDspec_subsidy_name             = 'Bs0p6',
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_sBs0p8',
#                                     bfs_numbers                      = LRG_bfs_list,
#                                     GRIDspec_apply_prem_tiers_TF      = True,
#                                     GRIDspec_subsidy_name             = 'Bs0p8',
#     ),

#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_sCs4p4',
#                                     bfs_numbers                      = LRG_bfs_list,
#                                     GRIDspec_apply_prem_tiers_TF      = True,
#                                     GRIDspec_subsidy_name             = 'Cs4p4',
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_sCs4p6',
#                                     bfs_numbers                      = LRG_bfs_list,
#                                     GRIDspec_apply_prem_tiers_TF      = True,
#                                     GRIDspec_subsidy_name             = 'Cs4p6',
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_sCs4p8',
#                                     bfs_numbers                      = LRG_bfs_list,
#                                     GRIDspec_apply_prem_tiers_TF      = True,
#                                     GRIDspec_subsidy_name             = 'Cs4p8',
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_sCs6p4',
#                                     bfs_numbers                      = LRG_bfs_list,
#                                     GRIDspec_apply_prem_tiers_TF      = True,
#                                     GRIDspec_subsidy_name             = 'Cs6p4',
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_sCs6p6',
#                                     bfs_numbers                      = LRG_bfs_list,
#                                     GRIDspec_apply_prem_tiers_TF      = True,
#                                     GRIDspec_subsidy_name             = 'Cs6p6',
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_sCs6p8',
#                                     bfs_numbers                      = LRG_bfs_list,
#                                     GRIDspec_apply_prem_tiers_TF      = True,
#                                     GRIDspec_subsidy_name             = 'Cs6p8',
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_sCs8p4',
#                                     bfs_numbers                      = LRG_bfs_list,
#                                     GRIDspec_apply_prem_tiers_TF      = True,
#                                     GRIDspec_subsidy_name             = 'Cs8p4',
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_sCs8p6',
#                                     bfs_numbers                      = LRG_bfs_list,
#                                     GRIDspec_apply_prem_tiers_TF      = True,
#                                     GRIDspec_subsidy_name             = 'Cs8p6',
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_sCs8p8',
#                                     bfs_numbers                      = LRG_bfs_list,
#                                     GRIDspec_apply_prem_tiers_TF      = True,
#                                     GRIDspec_subsidy_name             = 'Cs8p8',
#     ),
    
# ]


# LRG_scen_APPEND_list = [
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_epzb1', 
#                     CSTRspec_ep2050_rescale_fact     = 1.0,
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_histcnstrcapgr0_3', 
#                     CSTRspec_capacity_type           = 'hist_constr_capa_year',
#                     CSTRspec_ann_capacity_growth     = 0.3,
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_histcnstrcapgr0_2', 
#                     CSTRspec_capacity_type           = 'hist_constr_capa_year',
#                     CSTRspec_ann_capacity_growth     = 0.2,
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_histcnstrcapgr0_1', 
#                     CSTRspec_capacity_type           = 'hist_constr_capa_year',
#                     CSTRspec_ann_capacity_growth     = 0.1,
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_pksh991',
#                     GRID_peak_shaving_enabled_tupl  = (True, 99, 1), 
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_pksh',
#                     GRID_peak_shaving_enabled_tupl  = (True, 0.9, 0.7),
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_pksh_sCs2p8',
#                     GRIDspec_apply_prem_tiers_TF      = True,
#                     GRIDspec_subsidy_name             = 'Cs2p8',
#                     GRID_peak_shaving_enabled_tupl  = (True, 0.9, 0.7),
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_pksh_sCs4p8',
#                     GRIDspec_apply_prem_tiers_TF      = True,
#                     GRIDspec_subsidy_name             = 'Cs4p8',
#                     GRID_peak_shaving_enabled_tupl  = (True, 0.9, 0.7),
#     ),
#     make_scenario(pvalloc_Xnbfs_DEFAULT, f'{LRG_bfs_name}_max_pksh_sCs6p8',
#                     GRIDspec_apply_prem_tiers_TF      = True,
#                     GRIDspec_subsidy_name             = 'Cs6p8',
#                     GRID_peak_shaving_enabled_tupl  = (True, 0.9, 0.7),
#     ),




    
# ]



# old scen lists ===========================================================================================

#     DEV_newpreprep_SUB__scen_list = [
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV3_{SUB_bfs_name_MFHhp}_max_epzb0_75', 
#         #               bfs_numbers                       = SUB_bfs_list,
#         #               TECspec_heatpump_indicator        = ['7410', '7411'],
#         #               CSTRspec_ep2050_rescale_fact      = 0.75,
#         # ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV3_{SUB_bfs_name_MFHhp}_max_epzb0_2', 
#         #               bfs_numbers                       = SUB_bfs_list,
#         #               TECspec_heatpump_indicator        = ['7410', '7411'],
#         #               CSTRspec_ep2050_rescale_fact      = 0.2,
#         # ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV3_{SUB_bfs_name_MFHhp}_max_epzb0_1', 
#         #               bfs_numbers                       = SUB_bfs_list,
#         #               TECspec_heatpump_indicator        = ['7410', '7411'],
#         #               CSTRspec_ep2050_rescale_fact      = 0.1,
#         # ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV3_{SUB_bfs_name_MFHhp}_max', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', '7411'],
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV3_{SUB_bfs_name_MFHhp}_max_epzb1', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', '7411'],
#                       CSTRspec_ep2050_rescale_fact      = 1.0,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV3_{SUB_bfs_name_MFHhp}_max_epzb0_5', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', '7411'],
#                       CSTRspec_ep2050_rescale_fact      = 0.5,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV3_{SUB_bfs_name_MFHhp}_max_hist_constrgr0_05', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', '7411'],
#                       CSTRspec_capacity_type            = 'hist_constr_capa_year',
#                       CSTRspec_ann_capacity_growth      = 0.05,                         
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV3_{SUB_bfs_name_MFHhp}_max_hist_constrgr0_01', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', '7411'],
#                       CSTRspec_capacity_type            = 'hist_constr_capa_year',
#                       CSTRspec_ann_capacity_growth      = 0.01,                         
#         ),
        

#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV3_{SUB_bfs_name_SFHhp}_max', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', ],
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV3_{SUB_bfs_name_SFHhp}_max_epzb1', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', ],
#                       CSTRspec_ep2050_rescale_fact      = 1.0,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV3_{SUB_bfs_name_SFHhp}_max_epzb0_5', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', ],
#                       CSTRspec_ep2050_rescale_fact      = 0.5,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV3_{SUB_bfs_name_SFHhp}_max_hist_constrgr0_05', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', ],
#                       CSTRspec_capacity_type            = 'hist_constr_capa_year',
#                       CSTRspec_ann_capacity_growth      = 0.05,                         
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV3_{SUB_bfs_name_SFHhp}_max_hist_constrgr0_01', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', ],
#                       CSTRspec_capacity_type            = 'hist_constr_capa_year',
#                       CSTRspec_ann_capacity_growth      = 0.01,                         
#         ),


#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV3_{SUB_bfs_name_MFHhp}_max_beforeFeb26', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', '7411'],
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV3_{SUB_bfs_name_MFHhp}_max_epzb1_beforeFeb26', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', '7411'],
#                       CSTRspec_ep2050_rescale_fact      = 1.0,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV3_{SUB_bfs_name_MFHhp}_max_epzb0_5_beforeFeb26', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', '7411'],
#                       CSTRspec_ep2050_rescale_fact      = 0.5,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV3_{SUB_bfs_name_MFHhp}_max_hist_constrgr0_05_beforeFeb26', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', '7411'],
#                       CSTRspec_capacity_type            = 'hist_constr_capa_year',
#                       CSTRspec_ann_capacity_growth      = 0.05,                         
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV3_{SUB_bfs_name_MFHhp}_max_hist_constrgr0_01_beforeFeb26', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', '7411'],
#                       CSTRspec_capacity_type            = 'hist_constr_capa_year',
#                       CSTRspec_ann_capacity_growth      = 0.01,                         
#         ),
        

#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV3_{SUB_bfs_name_SFHhp}_max_beforeFeb26', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', ],
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV3_{SUB_bfs_name_SFHhp}_max_epzb1_beforeFeb26', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', ],
#                       CSTRspec_ep2050_rescale_fact      = 1.0,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV3_{SUB_bfs_name_SFHhp}_max_epzb0_5_beforeFeb26', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', ],
#                       CSTRspec_ep2050_rescale_fact      = 0.5,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV3_{SUB_bfs_name_SFHhp}_max_hist_constrgr0_05_beforeFeb26', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', ],
#                       CSTRspec_capacity_type            = 'hist_constr_capa_year',
#                       CSTRspec_ann_capacity_growth      = 0.05,                         
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV3_{SUB_bfs_name_SFHhp}_max_hist_constrgr0_01_beforeFeb26', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       TECspec_heatpump_indicator        = ['7410', ],
#                       CSTRspec_capacity_type            = 'hist_constr_capa_year',
#                       CSTRspec_ann_capacity_growth      = 0.01,                         
#         ),



#     ]

#     DEV_newpreprep__scen_list = [

#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{RUR_bfs_name}_max', 
#                       bfs_numbers                       = RUR_bfs_list,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{RUR_bfs_name}_max_1hll', 
#                       bfs_numbers                       = RUR_bfs_list,
#                       GRIDspec_node_1hll_closed_TF      = True,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{RUR_bfs_name}_gridoptim_max',
#                       bfs_numbers                       = RUR_bfs_list,
#                       run_pvalloc_initalization_TF      = True,
#                       run_pvalloc_mcalgorithm_TF        = False,
#                       run_gridoptimized_orderinst_TF    = True,
#                       run_gridoptimized_expansion_TF    = True,
#                       OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
#                       OPTEXPApecs_apply_gridoptim_order_TF     = True,
#                 ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{RUR_bfs_name}_max_sCs4p6',
#                         bfs_numbers                       = RUR_bfs_list,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p6',
#         ),


#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{SUB_bfs_name}_max', 
#                       bfs_numbers                       = SUB_bfs_list,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{SUB_bfs_name}_max_1hll', 
#                       bfs_numbers                       = SUB_bfs_list,
#                       GRIDspec_node_1hll_closed_TF      = True,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{SUB_bfs_name}_gridoptim_max',
#                       bfs_numbers                       = SUB_bfs_list,
#                       run_pvalloc_initalization_TF      = True,
#                       run_pvalloc_mcalgorithm_TF        = False,
#                       run_gridoptimized_orderinst_TF    = True,
#                       run_gridoptimized_expansion_TF    = True,
#                       OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
#                       OPTEXPApecs_apply_gridoptim_order_TF     = True,
#                 ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{SUB_bfs_name}_max_sCs4p6',
#                         bfs_numbers                       = SUB_bfs_list,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p6',
#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_max', 
#                       bfs_numbers                       = LRG_bfs_list,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_max_epzb1', 
#                       bfs_numbers                       = LRG_bfs_list,
#                       CSTRspec_ep2050_rescale_fact      = 1.0,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_max_epzb0_75', 
#                       bfs_numbers                       = LRG_bfs_list,
#                       CSTRspec_ep2050_rescale_fact      = 0.75,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_max_epzb0_5', 
#                       bfs_numbers                       = LRG_bfs_list,
#                       CSTRspec_ep2050_rescale_fact      = 0.5,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_max_epzb0_2', 
#                       bfs_numbers                       = LRG_bfs_list,
#                       CSTRspec_ep2050_rescale_fact      = 0.2,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_max_epzb0_1', 
#                       bfs_numbers                       = LRG_bfs_list,
#                       CSTRspec_ep2050_rescale_fact      = 0.1,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_max_hist_constrgr0_05', 
#                       bfs_numbers                       = LRG_bfs_list,
#                       CSTRspec_capacity_type            = 'hist_constr_capa_year',
#                       CSTRspec_ann_capacity_growth      = 0.05,                         
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_max_hist_constrgr0_01', 
#                       bfs_numbers                       = LRG_bfs_list,
#                       CSTRspec_capacity_type            = 'hist_constr_capa_year',
#                       CSTRspec_ann_capacity_growth      = 0.01,                         
#         ),


#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_max_1hll', 
#                       bfs_numbers                       = LRG_bfs_list,
#                       GRIDspec_node_1hll_closed_TF      = True,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_gridoptim_max',
#                       bfs_numbers                       = LRG_bfs_list,
#                       run_pvalloc_initalization_TF      = True,
#                       run_pvalloc_mcalgorithm_TF        = False,
#                       run_gridoptimized_orderinst_TF    = True,
#                       run_gridoptimized_expansion_TF    = True,
#                       OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
#                       OPTEXPApecs_apply_gridoptim_order_TF     = True,
#                 ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_max_sCs4p6',
#                         bfs_numbers                       = LRG_bfs_list,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p6',
#         ),


#         # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{XLRG_bfs_name}_max', 
#         #               bfs_numbers                       = XLRG_bfs_list,
#         # ),
#         # # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{XLRG_bfs_name}_max_1hll', 
#         # #               bfs_numbers                       = XLRG_bfs_list,
#         # #               GRIDspec_node_1hll_closed_TF      = True,
#         # # ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{XLRG_bfs_name}_gridoptim_max',
#         #               bfs_numbers                       = XLRG_bfs_list,
#         #               run_pvalloc_initalization_TF      = True,
#         #               run_pvalloc_mcalgorithm_TF        = False,
#         #               run_gridoptimized_orderinst_TF    = True,
#         #               run_gridoptimized_expansion_TF    = True,
#         #               OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
#         #               OPTEXPApecs_apply_gridoptim_order_TF     = True,
#         # ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{XLRG_bfs_name}_max_sCs4p6',
#         #                 bfs_numbers                       = XLRG_bfs_list,
#         #                 GRIDspec_apply_prem_tiers_TF      = True,
#         #                 GRIDspec_subsidy_name             = 'Cs4p6',
#         # ),

#         # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{X2XL_bfs_name}_max', 
#         #               bfs_numbers                       = X2XL_bfs_list,
#         # ),
#         # # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{X2XL_bfs_name}_max_1hll', 
#         # #               bfs_numbers                       = X2XL_bfs_list,
#         # #               GRIDspec_node_1hll_closed_TF      = True,
#         # # ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{X2XL_bfs_name}_gridoptim_max',
#         #               bfs_numbers                       = X2XL_bfs_list,
#         #               run_pvalloc_initalization_TF      = True,
#         #               run_pvalloc_mcalgorithm_TF        = False,
#         #               run_gridoptimized_orderinst_TF    = True,
#         #               run_gridoptimized_expansion_TF    = True,
#         #               OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
#         #               OPTEXPApecs_apply_gridoptim_order_TF     = True,
#         #         ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{X2XL_bfs_name}_max_sCs4p6',
#         #                 bfs_numbers                       = X2XL_bfs_list,
#         #                 GRIDspec_apply_prem_tiers_TF      = True,
#         #                 GRIDspec_subsidy_name             = 'Cs4p6',
#         # ),


#     ]

#     DEV_OLDpreprep__scen_list = [
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{RUR_bfs_name}_max_prepFeb26', 
#                       bfs_numbers                       = RUR_bfs_list,
#         ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{RUR_bfs_name}_max_1hll', 
#         #               bfs_numbers                       = RUR_bfs_list,
#         #               GRIDspec_node_1hll_closed_TF      = True,
#         # ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{RUR_bfs_name}_gridoptim_max_prepFeb26',
#                       bfs_numbers                       = RUR_bfs_list,
#                       run_pvalloc_initalization_TF      = True,
#                       run_pvalloc_mcalgorithm_TF        = False,
#                       run_gridoptimized_orderinst_TF    = True,
#                       run_gridoptimized_expansion_TF    = True,
#                       OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
#                       OPTEXPApecs_apply_gridoptim_order_TF     = True,
#                 ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{RUR_bfs_name}_max_sCs4p6_prepFeb26',
#                         bfs_numbers                       = RUR_bfs_list,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p6',
#         ),


#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{SUB_bfs_name}_max_prepFeb26', 
#                       bfs_numbers                       = SUB_bfs_list,
#         ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{SUB_bfs_name}_max_1hll', 
#         #               bfs_numbers                       = SUB_bfs_list,
#         #               GRIDspec_node_1hll_closed_TF      = True,
#         # ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{SUB_bfs_name}_gridoptim_max_prepFeb26',
#                       bfs_numbers                       = SUB_bfs_list,
#                       run_pvalloc_initalization_TF      = True,
#                       run_pvalloc_mcalgorithm_TF        = False,
#                       run_gridoptimized_orderinst_TF    = True,
#                       run_gridoptimized_expansion_TF    = True,
#                       OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
#                       OPTEXPApecs_apply_gridoptim_order_TF     = True,
#                 ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{SUB_bfs_name}_max_sCs4p6_prepFeb26',
#                         bfs_numbers                       = SUB_bfs_list,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p6',
#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{LRG_bfs_name}_max_prepFeb26', 
#                       bfs_numbers                       = LRG_bfs_list,
#         ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{LRG_bfs_name}_max_1hll', 
#         #               bfs_numbers                       = LRG_bfs_list,
#         #               GRIDspec_node_1hll_closed_TF      = True,
#         # ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{LRG_bfs_name}_gridoptim_max_prepFeb26',
#                       bfs_numbers                       = LRG_bfs_list,
#                       run_pvalloc_initalization_TF      = True,
#                       run_pvalloc_mcalgorithm_TF        = False,
#                       run_gridoptimized_orderinst_TF    = True,
#                       run_gridoptimized_expansion_TF    = True,
#                       OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
#                       OPTEXPApecs_apply_gridoptim_order_TF     = True,
#                 ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{LRG_bfs_name}_max_sCs4p6_prepFeb26',
#                         bfs_numbers                       = LRG_bfs_list,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p6',
#         ),

#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{XLRG_bfs_name}_max_prepFeb26', 
#         #               bfs_numbers                       = XLRG_bfs_list,
#         # ),
#         # # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{XLRG_bfs_name}_max_1hll', 
#         # #               bfs_numbers                       = XLRG_bfs_list,
#         # #               GRIDspec_node_1hll_closed_TF      = True,
#         # # ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{XLRG_bfs_name}_gridoptim_max_prepFeb26',
#         #               bfs_numbers                       = XLRG_bfs_list,
#         #               run_pvalloc_initalization_TF      = True,
#         #               run_pvalloc_mcalgorithm_TF        = False,
#         #               run_gridoptimized_orderinst_TF    = True,
#         #               run_gridoptimized_expansion_TF    = True,
#         #               OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel_prepFeb26', 
#         #               OPTEXPApecs_apply_gridoptim_order_TF     = True,
#         # ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{XLRG_bfs_name}_max_sCs4p6_prepFeb26',
#         #                 bfs_numbers                       = XLRG_bfs_list,
#         #                 GRIDspec_apply_prem_tiers_TF      = True,
#         #                 GRIDspec_subsidy_name             = 'Cs4p6',
#         # ),

#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{X2XL_bfs_name}_max_prepFeb26', 
#         #               bfs_numbers                       = X2XL_bfs_list,
#         # ),
#         # # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{X2XL_bfs_name}_max_1hll', 
#         # #               bfs_numbers                       = X2XL_bfs_list,
#         # #               GRIDspec_node_1hll_closed_TF      = True,
#         # # ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{X2XL_bfs_name}_gridoptim_max_prepFeb26',
#         #               bfs_numbers                       = X2XL_bfs_list,
#         #               run_pvalloc_initalization_TF      = True,
#         #               run_pvalloc_mcalgorithm_TF        = False,
#         #               run_gridoptimized_orderinst_TF    = True,
#         #               run_gridoptimized_expansion_TF    = True,
#         #               OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
#         #               OPTEXPApecs_apply_gridoptim_order_TF     = True,
#         #         ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{X2XL_bfs_name}_max_sCs4p6_prepFeb26',
#         #                 bfs_numbers                       = X2XL_bfs_list,
#         #                 GRIDspec_apply_prem_tiers_TF      = True,
#         #                 GRIDspec_subsidy_name             = 'Cs4p6',
#         # ),

#     ]
    


#     LRG_OLDpreprep__scen_list = [

#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max', 
#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_rnd',
#             ALGOspec_inst_selection_method    = 'random',
#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_gridoptim_max', 
#             # run_pvalloc_initalization_TF    = True,
#             run_pvalloc_initalization_TF    = True,
#             run_pvalloc_mcalgorithm_TF      = False,
#             run_gridoptimized_orderinst_TF  = True,
#             run_gridoptimized_expansion_TF  = True,
#             OPTIMspecs_gridnode_subsample           = 'all_nodes_pyparallel', 
#             OPTEXPApecs_apply_gridoptim_order_TF     = True,
#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_epzb0_75', 
#                       CSTRspec_ep2050_rescale_fact     = 0.75,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_epzb0_50', 
#                       CSTRspec_ep2050_rescale_fact     = 0.50,
#         ),


#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_sAs2p0',
#                         GRIDspec_subsidy_name             = 'As2p0',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_sAs4p0',
#                         GRIDspec_subsidy_name             = 'As4p0',    
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_sAs6p0',
#                         GRIDspec_subsidy_name             = 'As6p0',
#         ),


#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_sBs0p4',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Bs0p4',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_sBs0p6',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Bs0p6',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_sBs0p8',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Bs0p8',
#         ),


#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_sCs2p4',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs2p4',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_sCs2p6',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs2p6',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_sCs2p8',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs2p8',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_sCs4p4',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p4',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_sCs4p6',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p6',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_sCs4p8',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p8',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_sCs6p4',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs6p4',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_sCs6p6',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs6p6',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_max_sCs6p8',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs6p8',
#         ),
#    ]
    
#     LRG_OLDpreprep_1hll__scen_list = [

#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'{LRG_bfs_name}_rnd',
#             ALGOspec_inst_selection_method    = 'random',
#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll', 
#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_gridoptim_max_1hll', 
#             # run_pvalloc_initalization_TF    = True,
#             run_pvalloc_initalization_TF    = True,
#             run_pvalloc_mcalgorithm_TF      = False,
#             run_gridoptimized_orderinst_TF  = True,
#             run_gridoptimized_expansion_TF  = True,
#             OPTIMspecs_gridnode_subsample           = 'all_nodes_pyparallel', 
#             OPTEXPApecs_apply_gridoptim_order_TF     = True,
#         ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_rnd',
#         #                 ALGOspec_inst_selection_method    = 'random',
#         # ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_epzb0_75', 
#                       CSTRspec_ep2050_rescale_fact     = 0.75,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_epzb0_50', 
#                       CSTRspec_ep2050_rescale_fact     = 0.50,
#         ),


#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_sAs2p0',
#         #                 GRIDspec_subsidy_name             = 'As2p0',
#         # ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_sAs4p0',
#                         GRIDspec_subsidy_name             = 'As4p0',    
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_sAs6p0',
#                         GRIDspec_subsidy_name             = 'As6p0',
#         ),


#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_sBs0p4',
#         #                 GRIDspec_apply_prem_tiers_TF      = True,
#         #                 GRIDspec_subsidy_name             = 'Bs0p4',
#         # ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_sBs0p6',
#         #                 GRIDspec_apply_prem_tiers_TF      = True,
#         #                 GRIDspec_subsidy_name             = 'Bs0p6',
#         # ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_sBs0p8',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Bs0p8',
#         ),


#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_sCs2p4',
#         #                 GRIDspec_apply_prem_tiers_TF      = True,
#         #                 GRIDspec_subsidy_name             = 'Cs2p4',
#         # ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_sCs2p6',
#         #                 GRIDspec_apply_prem_tiers_TF      = True,
#         #                 GRIDspec_subsidy_name             = 'Cs2p6',
#         # ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_sCs2p8',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs2p8',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_sCs4p4',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p4',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_sCs4p6',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p6',
#         ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_sCs4p8',
#         #                 GRIDspec_apply_prem_tiers_TF      = True,
#         #                 GRIDspec_subsidy_name             = 'Cs4p8',
#         # ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_sCs6p4',
#         #                 GRIDspec_apply_prem_tiers_TF      = True,
#         #                 GRIDspec_subsidy_name             = 'Cs6p4',
#         # ),
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_sCs6p6',
#         #                 GRIDspec_apply_prem_tiers_TF      = True,
#         #                 GRIDspec_subsidy_name             = 'Cs6p6',
#         # ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP_1hll, f'{LRG_bfs_name}_max_1hll_sCs6p8',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs6p8',
#         ),


# ]
    
         
    
#     LRG_final_WorkingPaper_JanFeb26_scen_list = [
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max', 
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll',
#                         GRIDspec_node_1hll_closed_TF      = True,
#         ),  


#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sAs2p0',
#                 GRIDspec_subsidy_name             = 'As2p0',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs2p0',
#                 GRIDspec_node_1hll_closed_TF      = True,
#                 GRIDspec_subsidy_name             = 'As2p0',
#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sAs4p0',
#                 GRIDspec_subsidy_name             = 'As4p0',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs4p0',
#                 GRIDspec_node_1hll_closed_TF      = True,
#                 GRIDspec_subsidy_name             = 'As4p0',
#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sAs6p0',
#                 GRIDspec_subsidy_name             = 'As6p0',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs6p0',
#                 GRIDspec_node_1hll_closed_TF      = True,
#                 GRIDspec_subsidy_name             = 'As6p0',
#         ),


#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sBs0p4',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Bs0p4',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sBs0p4',
#                         GRIDspec_node_1hll_closed_TF      = True,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Bs0p4',
#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sBs0p6',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Bs0p6',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sBs0p6',
#                         GRIDspec_node_1hll_closed_TF      = True,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Bs0p6',
#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sBs0p8',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Bs0p8',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sBs0p8',
#                         GRIDspec_node_1hll_closed_TF      = True,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Bs0p8',
#         ),


#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sCs2p4',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs2p4',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs2p4',
#                         GRIDspec_node_1hll_closed_TF      = True,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs2p4',
#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sCs4p6',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p6',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs4p6',
#                         GRIDspec_node_1hll_closed_TF      = True,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p6',
#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sCs6p8',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs6p8',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs6p8',
#                         GRIDspec_node_1hll_closed_TF      = True,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs6p8',
#         ),



#         # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_rnd',
#         #                 ALGOspec_inst_selection_method    = 'random',
#         # ),  
#         # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_rnd_1hll',
#         #                 GRIDspec_node_1hll_closed_TF      = True,
#         #                 ALGOspec_inst_selection_method    = 'random',
#         # ),  
#     ]


#     XLRG_scen_list = [

#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max', 
#                         bfs_numbers                       = RUR_bfs_list,

#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_1hll',
#                         bfs_numbers                       = RUR_bfs_list,
#                         GRIDspec_node_1hll_closed_TF      = True, 
#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_gridoptim_max', 
#                         bfs_numbers                       = RUR_bfs_list,
#                         # run_pvalloc_initalization_TF    = True,
#                         run_pvalloc_initalization_TF    = True,
#                         run_pvalloc_mcalgorithm_TF      = False,
#                         run_gridoptimized_orderinst_TF  = True,
#                         run_gridoptimized_expansion_TF  = True,
#                         OPTIMspecs_gridnode_subsample           = 'all_nodes_pyparallel', 
#                         OPTEXPApecs_apply_gridoptim_order_TF     = True,
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_rnd',
#                         bfs_numbers                       = RUR_bfs_list,
#                         ALGOspec_inst_selection_method    = 'random',
#         ),


#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_epzb0_25', 
#                         bfs_numbers                       = RUR_bfs_list,
#                         CSTRspec_ep2050_rescale_fact     = 0.250,
#         ),

#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_histcnstrcapgr0_02', 
#                         bfs_numbers                       = RUR_bfs_list,
#                         CSTRspec_capacity_type           = 'hist_constr_capa_year',
#                         CSTRspec_ann_capacity_growth     = 0.02,
#         ),

        

#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sAs6p0',
#                         bfs_numbers                       = RUR_bfs_list,
#                         GRIDspec_subsidy_name             = 'As6p0',
#         ),



#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sBs0p8',
#                         bfs_numbers                       = RUR_bfs_list,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Bs0p8',
#         ),


#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sCs2p4',
#                         bfs_numbers                       = RUR_bfs_list,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs2p4',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sCs2p6',
#                         bfs_numbers                       = RUR_bfs_list,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs2p6',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sCs2p8',
#                         bfs_numbers                       = RUR_bfs_list,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs2p8',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sCs4p4',
#                         bfs_numbers                       = RUR_bfs_list,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p4',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sCs4p6',
#                         bfs_numbers                       = RUR_bfs_list,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p6',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sCs4p8',
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs4p8',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sCs6p4',
#                         bfs_numbers                       = RUR_bfs_list,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs6p4',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sCs6p6',
#                         bfs_numbers                       = RUR_bfs_list,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs6p6',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sCs6p8',
#                         bfs_numbers                       = RUR_bfs_list,
#                         GRIDspec_apply_prem_tiers_TF      = True,
#                         GRIDspec_subsidy_name             = 'Cs6p8',
#         ),
#     ]


#     XLRG_final_scen_list = [
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max', 
#                 bfs_numbers                       = XLRG_bfs_list,

#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_1hll',
#                 bfs_numbers                       = XLRG_bfs_list,
#                 GRIDspec_node_1hll_closed_TF      = True,
#         ),  


#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sAs4p0',
#                 bfs_numbers                       = XLRG_bfs_list,
#                 GRIDspec_subsidy_name             = 'As4p0',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_1hll_sAs4p0',
#                 bfs_numbers                       = XLRG_bfs_list,
#                 GRIDspec_node_1hll_closed_TF      = True,
#                 GRIDspec_subsidy_name             = 'As4p0',
#         ),


#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sCs4p6',
#                 bfs_numbers                       = XLRG_bfs_list,
#                 GRIDspec_apply_prem_tiers_TF      = True,
#                 GRIDspec_subsidy_name             = 'Cs4p6',
#         ),
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_1hll_sCs4p6',
#                 bfs_numbers                       = XLRG_bfs_list,
#                 GRIDspec_node_1hll_closed_TF      = True,
#                 GRIDspec_apply_prem_tiers_TF      = True,
#                 GRIDspec_subsidy_name             = 'Cs4p6',
#         ),


#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_rnd',
#                 bfs_numbers                       = XLRG_bfs_list,
#                 ALGOspec_inst_selection_method    = 'random',
#         ),  
#         make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_rnd_1hll',
#                 bfs_numbers                       = XLRG_bfs_list,
#                 GRIDspec_node_1hll_closed_TF      = True,
#                 ALGOspec_inst_selection_method    = 'random',
#         ),  
#     ]


