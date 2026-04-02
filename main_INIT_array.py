import os
import sys
from dataclasses import replace
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


# ==============================
# Default Sub-Scenarios 
# ==============================
if True: 
    def make_scenario(default_scen, name_dir_export, bfs_numbers=None, **overrides):
        kwargs = {'name_dir_export': name_dir_export}
        if bfs_numbers is not None:
            kwargs['bfs_numbers'] = bfs_numbers
        if overrides:
            kwargs.update(overrides)
        return replace(default_scen, **kwargs)


    pvalloc_2nbfs_test_DEFAULT = PVAllocScenario_Settings(name_dir_export ='pvalloc_2nbfs_test_DEFAULT',
            bfs_numbers                                          = [
                                                                    # critical nodes - max npv
                                                                    2762, 2771, 
                                                                    # critical nodes - ew 
                                                                    2768, 2769,
                                                        # 2641, 2615,
                                                                    ],         
            # mini_sub_model_TF                                    = False,
            # mini_sub_model_ngridnodes                            = 20, 
            # mini_sub_model_nEGIDs                                = 100,
            create_gdf_export_of_topology                        = True,
            export_csvs                                          = False,
            T0_year_prediction                                   = 2022,
            months_lookback                                      = 12,
            months_prediction                                    = 240,
            TECspec_add_heatpump_demand_TF                       = True,   
            ALGOspec_topo_subdf_partitioner                      = 250, 
            ALGOspec_inst_selection_method                       = 'max_npv',     # 'random', max_npv', 'prob_weighted_npv'
            # CSTRspec_ann_capacity_growth                         = 0.1,
            CSTRspec_capacity_type                               = 'ep2050_zerobasis',
            ALGOspec_subselec_filter_method                      = 'pooled',

    )
    test_scen_name = 'pvalloc_2nbfs_2y_testMC'


    pvalloc_Xnbfs_ARE_20y_DEFAULT = PVAllocScenario_Settings(
        name_dir_export = 'pvalloc_29nbfs_20y_DEFAULT',
        name_dir_import = 'preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI_Apr26', 
        bfs_numbers                                          = [
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

        create_gdf_export_of_topology                        = True,
        export_csvs                                          = False,
        T0_year_prediction                                   = 2024,
        months_lookback                                      = 12,
        months_prediction                                    = 240,
        TECspec_add_heatpump_demand_TF                       = True,
        ALGOspec_topo_subdf_partitioner                      = 250,
        ALGOspec_inst_selection_method                       = 'max_npv',     # 'random', max_npv', 'prob_weighted_npv'
        CSTRspec_ann_capacity_growth                         = 0.1,
        ALGOspec_subselec_filter_method                      = 'pooled',
        CSTRspec_capacity_type                               = 'ep2050_zerobasis',

    ) 
    pvalloc_Xnbfs_ARE_20y_OLDPREPREP = PVAllocScenario_Settings(
        name_dir_export ='pvalloc_29nbfs_20y_OLDPREPREP',
        name_dir_import = 'preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI_Apr26',

        GWRspec_building_cols             = ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                                             'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                                             'GEBF', 'GAREA', 
                                             'GWAERZH1', 'GENH1',],
        bfs_numbers                    = [
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

        create_gdf_export_of_topology                        = True,
        export_csvs                                          = False,
        T0_year_prediction                                   = 2024,
        months_lookback                                      = 12,
        months_prediction                                    = 240,
        TECspec_add_heatpump_demand_TF                       = True,
        ALGOspec_topo_subdf_partitioner                      = 250,
        ALGOspec_inst_selection_method                       = 'max_npv',     # 'random', max_npv', 'prob_weighted_npv'
        CSTRspec_ann_capacity_growth                         = 0.1,
        ALGOspec_subselec_filter_method                      = 'pooled',
        CSTRspec_capacity_type                               = 'ep2050_zerobasis',

    ) 
    RUR_bfs_name = 'pvalloc_16nbfs_RUR'
    RUR_bfs_list =[
        # RURAL
        2612, 2889, 2883, 2621, 2622,
        2620, 2615, 2614, 2616, 2480,
        2617, 2611, 2788, 2619, 2783, 2477, 
    ]
    SUB_bfs_name = 'pvalloc_10nbfs_SUB'
    SUB_bfs_list = [
        # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
        2613, 2782, 2618, 2786, 2785, 
        2772, 2761, 2743, 2476, 2768,
    ]
    LRG_bfs_name = 'pvalloc_29nbfs_LRG'
    # v5 -> T0_prediction: 2024
    LRG_bfs_list = [
        # RURAL 
        2612, 2889, 2883, 2621, 2622,
        2620, 2615, 2614, 2616, 2480,
        2617, 2611, 2788, 2619, 2783, 2477, 
        # SUBURBAN
        2613, 2782, 2618, 2786, 2785, 
        2772, 2761, 2743, 2476, 2768,
        # URBAN
        2773, 2769, 2770,
        ]
    XLRG_bfs_name = 'pvalloc_46nbfs_XLRG'
    XLRG_bfs_list = [
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
        ]
    
    X2XL_bfs_name = 'pvalloc_59nbfs_XXL'
    X2XL_bfs_list = [
        # RURAL 
        2612, 2889, 2883, 2621, 2622,
        2620, 2615, 2614, 2616, 2480,
        2617, 2611, 2788, 2619, 2783, 2477, 
        2790, 2426, 2428, 2893, 
        # SUBURBAN
        2613, 2782, 2618, 2786, 2785, 
        2772, 2761, 2743, 2476, 2768,
        2471, 2481, 2775, 2764, 2771, 
        2763, 2473, 2475, 2474, 2472, 
        2478, 2830, 2766, 2767, 2774, 
        2422, 2792, 2787, 2789, 2479, 
        2834, 2833, 
        # URBAN
        2773, 2769, 2770,
        2762, 2765, 
        2829, 2831, 
    ]

    ALLDSO_bfs_name = 'pvalloc_79nbfs_ALLDSO'
    ALLDSO_bfs_list = [
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
    ]


# ==============================
# Lists Sub-Scenarios 
# ==============================
if True: 
    test_scen_list = [
        make_scenario(pvalloc_2nbfs_test_DEFAULT, f'{test_scen_name}_max', 
        ),
    ]
    
    DEV_newpreprep__scen_list = [

        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{RUR_bfs_name}_max', 
                      bfs_numbers                       = RUR_bfs_list,
        ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{RUR_bfs_name}_max_1hll', 
        #               bfs_numbers                       = RUR_bfs_list,
        #               GRIDspec_node_1hll_closed_TF      = True,
        # ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{RUR_bfs_name}_gridoptim_max',
                      bfs_numbers                       = RUR_bfs_list,
                      run_pvalloc_initalization_TF      = True,
                      run_pvalloc_mcalgorithm_TF        = False,
                      run_gridoptimized_orderinst_TF    = True,
                      run_gridoptimized_expansion_TF    = True,
                      OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
                      OPTEXPApecs_apply_gridoptim_order_TF     = True,
                ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{RUR_bfs_name}_max_sCs4p6',
                        bfs_numbers                       = RUR_bfs_list,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p6',
        ),


        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{SUB_bfs_name}_max', 
                      bfs_numbers                       = SUB_bfs_list,
        ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{SUB_bfs_name}_max_1hll', 
        #               bfs_numbers                       = SUB_bfs_list,
        #               GRIDspec_node_1hll_closed_TF      = True,
        # ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{SUB_bfs_name}_gridoptim_max',
                      bfs_numbers                       = SUB_bfs_list,
                      run_pvalloc_initalization_TF      = True,
                      run_pvalloc_mcalgorithm_TF        = False,
                      run_gridoptimized_orderinst_TF    = True,
                      run_gridoptimized_expansion_TF    = True,
                      OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
                      OPTEXPApecs_apply_gridoptim_order_TF     = True,
                ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{SUB_bfs_name}_max_sCs4p6',
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p6',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_max', 
                      bfs_numbers                       = LRG_bfs_list,
        ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_max_1hll', 
        #               bfs_numbers                       = LRG_bfs_list,
        #               GRIDspec_node_1hll_closed_TF      = True,
        # ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_gridoptim_max',
                      bfs_numbers                       = LRG_bfs_list,
                      run_pvalloc_initalization_TF      = True,
                      run_pvalloc_mcalgorithm_TF        = False,
                      run_gridoptimized_orderinst_TF    = True,
                      run_gridoptimized_expansion_TF    = True,
                      OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
                      OPTEXPApecs_apply_gridoptim_order_TF     = True,
                ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_max_sCs4p6',
                        bfs_numbers                       = LRG_bfs_list,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p6',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{XLRG_bfs_name}_max', 
                      bfs_numbers                       = XLRG_bfs_list,
        ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{XLRG_bfs_name}_max_1hll', 
        #               bfs_numbers                       = XLRG_bfs_list,
        #               GRIDspec_node_1hll_closed_TF      = True,
        # ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{XLRG_bfs_name}_gridoptim_max',
                      bfs_numbers                       = XLRG_bfs_list,
                      run_pvalloc_initalization_TF      = True,
                      run_pvalloc_mcalgorithm_TF        = False,
                      run_gridoptimized_orderinst_TF    = True,
                      run_gridoptimized_expansion_TF    = True,
                      OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
                      OPTEXPApecs_apply_gridoptim_order_TF     = True,
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{XLRG_bfs_name}_max_sCs4p6',
                        bfs_numbers                       = XLRG_bfs_list,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p6',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{X2XL_bfs_name}_max', 
                      bfs_numbers                       = X2XL_bfs_list,
        ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{X2XL_bfs_name}_max_1hll', 
        #               bfs_numbers                       = X2XL_bfs_list,
        #               GRIDspec_node_1hll_closed_TF      = True,
        # ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{X2XL_bfs_name}_gridoptim_max',
                      bfs_numbers                       = X2XL_bfs_list,
                      run_pvalloc_initalization_TF      = True,
                      run_pvalloc_mcalgorithm_TF        = False,
                      run_gridoptimized_orderinst_TF    = True,
                      run_gridoptimized_expansion_TF    = True,
                      OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
                      OPTEXPApecs_apply_gridoptim_order_TF     = True,
                ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{X2XL_bfs_name}_max_sCs4p6',
                        bfs_numbers                       = X2XL_bfs_list,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p6',
        ),


        


        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{RUR_bfs_name}_max__preprep_before_Feb26', 
        #             bfs_numbers                       = RUR_bfs_list,
        #             #   name_dir_import                   = 'preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26'
        #             name_dir_import                   = 'preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI__before_Feb26',
        #             GWRspec_building_cols             = ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
        #                                                  'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
        #                                                  'GEBF', 'GAREA', 
        #                                                  'GWAERZH1', 'GENH1',],
        # ),

        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'DEV2_{LRG_bfs_name}_max__preprep_before_Feb26', 
        #             bfs_numbers                       = LRG_bfs_list,
        #             #   name_dir_import                   = 'preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI-COPYpreprep_used_untilFeb26'
        #             name_dir_import                   = 'preprep_BLSO_15to24_extSolkatEGID_aggrfarms_reimportAPI__before_Feb26',
        #             GWRspec_building_cols             = ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
        #                                                  'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
        #                                                  'GEBF', 'GAREA', 
        #                                                  'GWAERZH1', 'GENH1',],
        # ),

    ]

    DEV_OLDpreprep__scen_list = [
        make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{RUR_bfs_name}_max_OLDpreprep', 
                      bfs_numbers                       = RUR_bfs_list,
        ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{RUR_bfs_name}_max_1hll', 
        #               bfs_numbers                       = RUR_bfs_list,
        #               GRIDspec_node_1hll_closed_TF      = True,
        # ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{RUR_bfs_name}_gridoptim_max_OLDpreprep',
                      bfs_numbers                       = RUR_bfs_list,
                      run_pvalloc_initalization_TF      = True,
                      run_pvalloc_mcalgorithm_TF        = False,
                      run_gridoptimized_orderinst_TF    = True,
                      run_gridoptimized_expansion_TF    = True,
                      OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
                      OPTEXPApecs_apply_gridoptim_order_TF     = True,
                ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{RUR_bfs_name}_max_sCs4p6_OLDpreprep',
                        bfs_numbers                       = RUR_bfs_list,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p6',
        ),


        make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{SUB_bfs_name}_max_OLDpreprep', 
                      bfs_numbers                       = SUB_bfs_list,
        ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{SUB_bfs_name}_max_1hll', 
        #               bfs_numbers                       = SUB_bfs_list,
        #               GRIDspec_node_1hll_closed_TF      = True,
        # ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{SUB_bfs_name}_gridoptim_max_OLDpreprep',
                      bfs_numbers                       = SUB_bfs_list,
                      run_pvalloc_initalization_TF      = True,
                      run_pvalloc_mcalgorithm_TF        = False,
                      run_gridoptimized_orderinst_TF    = True,
                      run_gridoptimized_expansion_TF    = True,
                      OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
                      OPTEXPApecs_apply_gridoptim_order_TF     = True,
                ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{SUB_bfs_name}_max_sCs4p6_OLDpreprep',
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p6',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{LRG_bfs_name}_max_OLDpreprep', 
                      bfs_numbers                       = LRG_bfs_list,
        ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{LRG_bfs_name}_max_1hll', 
        #               bfs_numbers                       = LRG_bfs_list,
        #               GRIDspec_node_1hll_closed_TF      = True,
        # ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{LRG_bfs_name}_gridoptim_max_OLDpreprep',
                      bfs_numbers                       = LRG_bfs_list,
                      run_pvalloc_initalization_TF      = True,
                      run_pvalloc_mcalgorithm_TF        = False,
                      run_gridoptimized_orderinst_TF    = True,
                      run_gridoptimized_expansion_TF    = True,
                      OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
                      OPTEXPApecs_apply_gridoptim_order_TF     = True,
                ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{LRG_bfs_name}_max_sCs4p6_OLDpreprep',
                        bfs_numbers                       = LRG_bfs_list,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p6',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{XLRG_bfs_name}_max_OLDpreprep', 
                      bfs_numbers                       = XLRG_bfs_list,
        ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{XLRG_bfs_name}_max_1hll', 
        #               bfs_numbers                       = XLRG_bfs_list,
        #               GRIDspec_node_1hll_closed_TF      = True,
        # ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{XLRG_bfs_name}_gridoptim_max_OLDpreprep',
                      bfs_numbers                       = XLRG_bfs_list,
                      run_pvalloc_initalization_TF      = True,
                      run_pvalloc_mcalgorithm_TF        = False,
                      run_gridoptimized_orderinst_TF    = True,
                      run_gridoptimized_expansion_TF    = True,
                      OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel_OLDpreprep', 
                      OPTEXPApecs_apply_gridoptim_order_TF     = True,
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{XLRG_bfs_name}_max_sCs4p6_OLDpreprep',
                        bfs_numbers                       = XLRG_bfs_list,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p6',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{X2XL_bfs_name}_max_OLDpreprep', 
                      bfs_numbers                       = X2XL_bfs_list,
        ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{X2XL_bfs_name}_max_1hll', 
        #               bfs_numbers                       = X2XL_bfs_list,
        #               GRIDspec_node_1hll_closed_TF      = True,
        # ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{X2XL_bfs_name}_gridoptim_max_OLDpreprep',
                      bfs_numbers                       = X2XL_bfs_list,
                      run_pvalloc_initalization_TF      = True,
                      run_pvalloc_mcalgorithm_TF        = False,
                      run_gridoptimized_orderinst_TF    = True,
                      run_gridoptimized_expansion_TF    = True,
                      OPTIMspecs_gridnode_subsample            = 'all_nodes_pyparallel', 
                      OPTEXPApecs_apply_gridoptim_order_TF     = True,
                ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_OLDPREPREP, f'DEV_{X2XL_bfs_name}_max_sCs4p6_OLDpreprep',
                        bfs_numbers                       = X2XL_bfs_list,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p6',
        ),

    ]
    # DEV_scen_list = DEV_newpreprep__scen_list + DEV_OLDpreprep__scen_list
    DEV_scen_list = DEV_newpreprep__scen_list


    RUR_scen_list = [
    # pvalloc_scen_list = [
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{RUR_bfs_name}', 
                      bfs_numbers                       = RUR_bfs_list,
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{RUR_bfs_name}_sA1', 
                      bfs_numbers                       = RUR_bfs_list,
                      GRIDspec_subsidy_name             = 'A1',
        ),
       
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{RUR_bfs_name}_1hll', 
                        bfs_numbers                       = RUR_bfs_list,
                        GRIDspec_node_1hll_closed_TF      = True,
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{RUR_bfs_name}_1hll_sA1',
                        bfs_numbers                       = RUR_bfs_list,
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'A1',
        ),  


      
    ]

    SUB_scen_list = [
    # pvalloc_scen_list = [

        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{SUB_bfs_name}', 
                       bfs_numbers                       = SUB_bfs_list,
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{SUB_bfs_name}_sA1', 
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_subsidy_name             = 'A1',
        ),


        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{SUB_bfs_name}_1hll', 
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_node_1hll_closed_TF      = True,
        ),
    ]


    LRG_scen_list_asdf = [

        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max', 
        ),

        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_gridoptim_max', 
            # run_pvalloc_initalization_TF    = True,
            run_pvalloc_initalization_TF    = False,
            run_pvalloc_mcalgorithm_TF      = False,
            run_gridoptimized_orderinst_TF  = True,
            run_gridoptimized_expansion_TF  = True,
            OPTIMspecs_gridnode_subsample           = 'all_nodes_pyparallel', 
            OPTEXPApecs_apply_gridoptim_order_TF     = True,
        ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_rnd',
        #                 ALGOspec_inst_selection_method    = 'random',
        # ),

        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_epzb0_75', 
                      CSTRspec_ep2050_rescale_fact     = 0.75,
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_epzb0_50', 
                      CSTRspec_ep2050_rescale_fact     = 0.50,
        ),


        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sAs2p0',
                        GRIDspec_subsidy_name             = 'As2p0',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sAs4p0',
                        GRIDspec_subsidy_name             = 'As4p0',    
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sAs6p0',
                        GRIDspec_subsidy_name             = 'As6p0',
        ),


        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sBs0p4',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Bs0p4',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sBs0p6',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Bs0p6',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sBs0p8',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Bs0p8',
        ),


        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sCs2p4',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs2p4',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sCs2p6',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs2p6',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sCs2p8',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs2p8',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sCs4p4',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p4',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sCs4p6',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p6',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sCs4p8',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p8',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sCs6p4',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs6p4',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sCs6p6',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs6p6',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sCs6p8',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs6p8',
        ),
   ]
    
    LRG_scen_list = [
        
     make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_gridoptim_max', 
            # run_pvalloc_initalization_TF    = True,
            run_pvalloc_initalization_TF    = False,
            run_pvalloc_mcalgorithm_TF      = False,
            run_gridoptimized_orderinst_TF  = True,
            run_gridoptimized_expansion_TF  = True,
            OPTIMspecs_gridnode_subsample           = 'all_nodes_pyparallel', 
            OPTEXPApecs_apply_gridoptim_order_TF     = True,
        ),
    ]
    
    # LRG__max_1hll__scen_list = [

        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        # ),  
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_gridoptim_max_1hll',
        #     # run_pvalloc_initalization_TF    = True,                      
        #     run_pvalloc_initalization_TF    = False,                      
        #     run_pvalloc_mcalgorithm_TF      = False,
        #     run_gridoptimized_orderinst_TF  = True,
        #     run_gridoptimized_expansion_TF  = True,
        #     OPTIMspecs_gridnode_subsample           = 'all_nodes_pyparallel', 
        #     OPTEXPApecs_apply_gridoptim_order_TF     = True,

        #     GRIDspec_node_1hll_closed_TF      = True,
        # ),  
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_rnd_1hll',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 ALGOspec_inst_selection_method    = 'random',
        # ),  


        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs2p0',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_subsidy_name             = 'As2p0',
        # ),  
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs4p0',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_subsidy_name             = 'As4p0',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs6p0',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_subsidy_name             = 'As6p0',
        # ),

        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sBs0p4',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Bs0p4',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sBs0p6',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Bs0p6',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sBs0p8',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Bs0p8',
        # ),


        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs2p4',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Cs2p4',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs2p6',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Cs2p6',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs2p8',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Cs2p8',
        # ),

        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs4p4',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Cs4p4',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs4p6',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Cs4p6',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs4p8',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Cs4p8',
        # ),

        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs6p4',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Cs6p4',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs6p6',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Cs6p6',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs6p8',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Cs6p8',
        # ),
    
         
    
    LRG_final_WorkingPaper_JanFeb26_scen_list = [
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max', 
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll',
                        GRIDspec_node_1hll_closed_TF      = True,
        ),  


        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sAs2p0',
                GRIDspec_subsidy_name             = 'As2p0',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs2p0',
                GRIDspec_node_1hll_closed_TF      = True,
                GRIDspec_subsidy_name             = 'As2p0',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sAs4p0',
                GRIDspec_subsidy_name             = 'As4p0',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs4p0',
                GRIDspec_node_1hll_closed_TF      = True,
                GRIDspec_subsidy_name             = 'As4p0',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sAs6p0',
                GRIDspec_subsidy_name             = 'As6p0',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs6p0',
                GRIDspec_node_1hll_closed_TF      = True,
                GRIDspec_subsidy_name             = 'As6p0',
        ),


        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sBs0p4',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Bs0p4',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sBs0p4',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Bs0p4',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sBs0p6',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Bs0p6',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sBs0p6',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Bs0p6',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sBs0p8',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Bs0p8',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sBs0p8',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Bs0p8',
        ),


        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sCs2p4',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs2p4',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs2p4',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs2p4',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sCs4p6',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p6',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs4p6',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p6',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_sCs6p8',
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs6p8',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs6p8',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs6p8',
        ),



        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_rnd',
        #                 ALGOspec_inst_selection_method    = 'random',
        # ),  
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{LRG_bfs_name}_rnd_1hll',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 ALGOspec_inst_selection_method    = 'random',
        # ),  
    ]


    XLRG_scen_list = [
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max', 
        #         bfs_numbers                       = XLRG_bfs_list,
        # ),

        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_1hll',
        #         bfs_numbers                       = XLRG_bfs_list,
        #         GRIDspec_node_1hll_closed_TF      = True,
        # ),  

        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_gridoptim_max', 
            bfs_numbers                       = XLRG_bfs_list,
            # run_pvalloc_initalization_TF    = True,
            run_pvalloc_initalization_TF    = False,
            run_pvalloc_mcalgorithm_TF      = False,
            run_gridoptimized_orderinst_TF  = True,
            run_gridoptimized_expansion_TF  = True,
            OPTIMspecs_gridnode_subsample           = 'all_nodes_pyparallel', 
            OPTEXPApecs_apply_gridoptim_order_TF     = True,
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_gridoptim_max_1hll',
            bfs_numbers                       = XLRG_bfs_list,
            # run_pvalloc_initalization_TF    = True,
            run_pvalloc_initalization_TF    = False,
            run_pvalloc_mcalgorithm_TF      = False,
            run_gridoptimized_orderinst_TF  = True,
            run_gridoptimized_expansion_TF  = True,
            OPTIMspecs_gridnode_subsample           = 'all_nodes_pyparallel', 
            OPTEXPApecs_apply_gridoptim_order_TF     = True,

            GRIDspec_node_1hll_closed_TF      = True,
        ),  

        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_rnd',
        #                 bfs_numbers                       = XLRG_bfs_list,
        #                 ALGOspec_inst_selection_method    = 'random',
        # ),  
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_rnd_1hll',
        #                 bfs_numbers                       = XLRG_bfs_list,
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 ALGOspec_inst_selection_method    = 'random',
        # ),  

        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sAs4p0',
        #                 bfs_numbers                       = XLRG_bfs_list,
        #                 GRIDspec_subsidy_name             = 'As4p0',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_1hll_sAs4p0',
        #                 bfs_numbers                       = XLRG_bfs_list,
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_subsidy_name             = 'As4p0',
        # ),

        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sBs0p6',
        #                 bfs_numbers                       = XLRG_bfs_list,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Bs0p6',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_1hll_sBs0p6',
        #                 bfs_numbers                       = XLRG_bfs_list,
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Bs0p6',
        # ),

        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sCs4p6',
        #                 bfs_numbers                       = XLRG_bfs_list,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Cs4p6',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_1hll_sCs4p6',
        #                 bfs_numbers                       = XLRG_bfs_list,
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Cs4p6',
        # ),

    ]

    XLRG_final_scen_list = [
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max', 
                bfs_numbers                       = XLRG_bfs_list,

        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_1hll',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_node_1hll_closed_TF      = True,
        ),  


        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sAs4p0',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_subsidy_name             = 'As4p0',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_1hll_sAs4p0',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_node_1hll_closed_TF      = True,
                GRIDspec_subsidy_name             = 'As4p0',
        ),


        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_sCs4p6',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_apply_prem_tiers_TF      = True,
                GRIDspec_subsidy_name             = 'Cs4p6',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_max_1hll_sCs4p6',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_node_1hll_closed_TF      = True,
                GRIDspec_apply_prem_tiers_TF      = True,
                GRIDspec_subsidy_name             = 'Cs4p6',
        ),


        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_rnd',
                bfs_numbers                       = XLRG_bfs_list,
                ALGOspec_inst_selection_method    = 'random',
        ),  
        make_scenario(pvalloc_Xnbfs_ARE_20y_DEFAULT, f'{XLRG_bfs_name}_rnd_1hll',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_node_1hll_closed_TF      = True,
                ALGOspec_inst_selection_method    = 'random',
        ),  
    ]


# ==============================
# EXPORT Sub-Scenarios 
# ==============================
def get_subscen_list(sub_scen_str = 'test'):
    if sub_scen_str == 'test':
        return DEV_scen_list
    elif sub_scen_str == 'RUR':
        return RUR_scen_list
    elif sub_scen_str == 'SUB':
        return SUB_scen_list
    elif sub_scen_str == 'RUR_and_SUB':
        return RUR_scen_list + SUB_scen_list
    elif sub_scen_str == 'LRG':
        return LRG_scen_list
    elif sub_scen_str == 'XLRG':
        return XLRG_scen_list
    elif sub_scen_str == 'XLRG_final':
        return XLRG_final_scen_list
    elif sub_scen_str == 'DEV':
        return DEV_scen_list
    else:
        return []
    

if __name__ == '__main__':

    # call scen in array and run ------------------------------------------

    pvalloc_scen_list = test_scen_list   

    # for pvalloc_scen_index in range(0,10):
    #     print(f'idx < len(list)-1 ->i: {pvalloc_scen_index} | {pvalloc_scen_index < len(pvalloc_scen_list)-1}')

    # slurm_job_id = os.environ.get('SLURM_JOB_ID_ENV', 'unknown')
    slurm_job_id = os.environ.get('SLURM_ARRAY_JOB_ID_ENV', 'unknown')
    slurm_array_id = os.environ.get('SLURM_ARRAY_TASK_ID_ENV', 'unknown')
    slurm_full_id = f"{slurm_job_id}_{slurm_array_id}"

    if len(sys.argv) > 1:
        pvalloc_scen_index = int(sys.argv[1])
        if pvalloc_scen_index < len(pvalloc_scen_list):
            pvalloc_scen = pvalloc_scen_list[pvalloc_scen_index]
            
            scen_class = PVAllocScenario(pvalloc_scen)

            scen_class.sett.slurm_full_id        = slurm_full_id
            scen_class.sett.pvalloc_scen_index   = pvalloc_scen_index
    
            scen_class.RUN_pvalloc_scenario()
    
    elif 'c:\\Models' in os.getcwd():

        for pvalloc_scen_index in range(0, len(pvalloc_scen_list)):
            pvalloc_scen = pvalloc_scen_list[pvalloc_scen_index]
            
            scen_class = PVAllocScenario(pvalloc_scen)
            scen_class.RUN_pvalloc_scenario()

        print('done')







