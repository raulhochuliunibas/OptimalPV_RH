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
            TECspec_heatpump_months_factor                       = [
                                                                    (10, 7.0),
                                                                    (11, 7.0), 
                                                                    (12, 7.0), 
                                                                    (1 , 7.0), 
                                                                    (2 , 7.0), 
                                                                    (3 , 7.0), 
                                                                    (4 , 7.0), 
                                                                    (5 , 7.0),     
                                                                    (6 , 1.0), 
                                                                    (7 , 1.0), 
                                                                    (8 , 1.0), 
                                                                    (9 , 1.0),
                                                                    ], 
            ALGOspec_topo_subdf_partitioner                      = 250, 
            ALGOspec_inst_selection_method                       = 'max_npv',     # 'random', max_npv', 'prob_weighted_npv'
            # CSTRspec_ann_capacity_growth                         = 0.1,
            CSTRspec_capacity_type                               = 'ep2050_zerobasis',
            ALGOspec_subselec_filter_method                      = 'pooled',

    )


    pvalloc_Xnbfs_ARE_30y_DEFAULT = PVAllocScenario_Settings(
        name_dir_export ='pvalloc_29nbfs_30y_DEFAULT',
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
        create_gdf_export_of_topology                        = True,
        export_csvs                                          = False,
        T0_year_prediction                                   = 2024,
        months_lookback                                      = 12,
        months_prediction                                    = 360,
        TECspec_add_heatpump_demand_TF                       = True,
        TECspec_heatpump_months_factor                       = [
                                                                (10, 7.0),
                                                                (11, 7.0), 
                                                                (12, 7.0), 
                                                                (1 , 7.0), 
                                                                (2 , 7.0), 
                                                                (3 , 7.0), 
                                                                (4 , 7.0), 
                                                                (5 , 7.0),     
                                                                (6 , 1.0), 
                                                                (7 , 1.0), 
                                                                (8 , 1.0), 
                                                                (9 , 1.0),
                                                                ],
        ALGOspec_topo_subdf_partitioner                      = 250,
        ALGOspec_inst_selection_method                       = 'max_npv',     # 'random', max_npv', 'prob_weighted_npv'
        CSTRspec_ann_capacity_growth                         = 0.1,
        ALGOspec_subselec_filter_method                      = 'pooled',
        CSTRspec_capacity_type                               = 'ep2050_zerobasis',

    ) 

    test_scen_name = 'pvalloc_2nbfs_2y_testMC'

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
    LRG_bfs_name = 'pvalloc_29nbfs_30y5'
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
        ],

    XLRG_bfs_name = 'pvalloc_47nbfs_30y4'
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


# ==============================
# Lists Sub-Scenarios 
# ==============================
if True: 
    test_scen_list = [
    # pvalloc_scen_list = [

        # make_scenario(pvalloc_2nbfs_test_DEFAULT, f'{test_scen_name}',
        # ),
        make_scenario(pvalloc_2nbfs_test_DEFAULT, f'{test_scen_name}_probnpv',
                        ALGOspec_inst_selection_method    = 'prob_weighted_npv',
        ),

        # make_scenario(pvalloc_2nbfs_test_DEFAULT, f'{test_scen_name}_sA1',
        #               GRIDspec_subsidy_name = 'A1',
        # ),
        # make_scenario(pvalloc_2nbfs_test_DEFAULT, f'{test_scen_name}_sA3',
        #               GRIDspec_subsidy_name = 'A3',
        # ),
        # make_scenario(pvalloc_2nbfs_test_DEFAULT, f'{test_scen_name}_sB1',
        #               GRIDspec_subsidy_name = 'B1',
        # ),
        # make_scenario(pvalloc_2nbfs_test_DEFAULT, f'{test_scen_name}_sB2',
        #               GRIDspec_subsidy_name = 'B2',
        # ),
        # make_scenario(pvalloc_2nbfs_test_DEFAULT, f'{test_scen_name}_sC1',
        #               GRIDspec_subsidy_name = 'C1',
        # ),
        # make_scenario(pvalloc_2nbfs_test_DEFAULT, f'{test_scen_name}_sC2',
        #               GRIDspec_subsidy_name = 'C2',
        # ),

        
    ]
        
    RUR_scen_list = [
    # pvalloc_scen_list = [
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}', 
                      bfs_numbers                       = RUR_bfs_list,
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_sA1', 
                      bfs_numbers                       = RUR_bfs_list,
                      GRIDspec_subsidy_name             = 'A1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_sA3',
                        bfs_numbers                       = RUR_bfs_list,
                        GRIDspec_subsidy_name             = 'A3',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_sB1',
                        bfs_numbers                       = RUR_bfs_list,
                        GRIDspec_subsidy_name             = 'B1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_sB2',
                        bfs_numbers                       = RUR_bfs_list,
                        GRIDspec_subsidy_name             = 'B2',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_sC1',
                        bfs_numbers                       = RUR_bfs_list,
                        GRIDspec_subsidy_name             = 'C1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_sC2',
                        bfs_numbers                       = RUR_bfs_list,
                        GRIDspec_subsidy_name             = 'C2',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_1hll', 
                        bfs_numbers                       = RUR_bfs_list,
                        GRIDspec_node_1hll_closed_TF      = True,
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_1hll_sA1',
                        bfs_numbers                       = RUR_bfs_list,
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'A1',
        ),  
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_1hll_sA3',
                        bfs_numbers                       = RUR_bfs_list,
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'A3',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_1hll_sB1',
                        bfs_numbers                       = RUR_bfs_list,   
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'B1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_1hll_sB2',
                        bfs_numbers                       = RUR_bfs_list,
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'B2',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_1hll_sC1',
                        bfs_numbers                       = RUR_bfs_list,
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'C1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_1hll_sC2',
                        bfs_numbers                       = RUR_bfs_list,
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'C2',
        ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_ew1pool', 
        #                 bfs_numbers                       = RUR_bfs_list,
        #                 ALGOspec_subselec_filter_criteria =  ('filter_tag__eastwest_80pr', 'filter_tag__eastwest_70pr',),
        #     ),
        # # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_ew2pool', 
        # #               bfs_numbers                       = RUR_bfs_list,
        # #               ALGOspec_subselec_filter_criteria =  ('filter_tag__eastwest_80pr', 'filter_tag__eastwest_70pr',
        # #                                                     'filter_tag__eastORwest_50pr', 'filter_tag__eastORwest_40pr'),
        # #     ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_1hll', 
        #                 bfs_numbers                       = RUR_bfs_list,
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #     ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_1hll_ew1pool', 
        #                 bfs_numbers                       = RUR_bfs_list,
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 ALGOspec_subselec_filter_criteria = ('filter_tag__eastwest_80pr', 'filter_tag__eastwest_70pr', )
        #     ),
        # # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{RUR_bfs_name}_1hll_ew2pool',
        # #                 bfs_numbers                       = RUR_bfs_list,
        # #                 GRIDspec_node_1hll_closed_TF      = True,
        # #                 ALGOspec_subselec_filter_criteria =  ('filter_tag__eastwest_80pr', 'filter_tag__eastwest_70pr',
        # #                                                       'filter_tag__eastORwest_50pr', 'filter_tag__eastORwest_40pr'),
        # #         ),
    ]

    SUB_scen_list = [
    # pvalloc_scen_list = [

        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}', 
                       bfs_numbers                       = SUB_bfs_list,
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_sA1', 
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_subsidy_name             = 'A1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_sA3',
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_subsidy_name             = 'A3',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_sB1',
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_subsidy_name             = 'B1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_sB2',
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_subsidy_name             = 'B2',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_sC1',
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_subsidy_name             = 'C1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_sC2',
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_subsidy_name             = 'C2',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_1hll', 
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_node_1hll_closed_TF      = True,
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_1hll_sA1',
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'A1',
        ),  
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_1hll_sA3',
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'A3',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_1hll_sB1',
                        bfs_numbers                       = SUB_bfs_list,   
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'B1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_1hll_sB2',
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'B2',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_1hll_sC1',
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'C1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_1hll_sC2',
                        bfs_numbers                       = SUB_bfs_list,
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'C2',
        ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_ew1pool',
        #                 bfs_numbers                       = SUB_bfs_list,
        #                 ALGOspec_subselec_filter_criteria =  ('filter_tag__eastwest_80pr', 'filter_tag__eastwest_70pr',),
        #     ),
        # # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_ew2pool',
        # #                 CSTRspec_capacity_type            ='ep2050_zerobasis',
        # #                 ALGOspec_subselec_filter_criteria =  ('filter_tag__eastwest_80pr', 'filter_tag__eastwest_70pr',
        # #                                                       'filter_tag__eastORwest_50pr', 'filter_tag__eastORwest_40pr'),
        # #     ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_1hll',
        #                 bfs_numbers                       = SUB_bfs_list,
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #     ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_1hll_ew1pool',
        #                 bfs_numbers                       = SUB_bfs_list,
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 ALGOspec_subselec_filter_criteria = ('filter_tag__eastwest_80pr', 'filter_tag__eastwest_70pr', )
        #     ),
        # # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_1hll_ew2pool',
        # #                 bfs_numbers                       = SUB_bfs_list,
        # #                 GRIDspec_node_1hll_closed_TF      = True,
        # #                 ALGOspec_subselec_filter_criteria =  ('filter_tag__eastwest_80pr', 'filter_tag__eastwest_70pr',
        # #                                                       'filter_tag__eastORwest_50pr', 'filter_tag__eastORwest_40pr'),
        # #     ),
    ]

    LRG_scen_list = [
    # pvalloc_scen_list = [

        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max', 
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll',
                        GRIDspec_node_1hll_closed_TF      = True,
        ),  
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_rnd',
                        ALGOspec_inst_selection_method    = 'random',
        ),  
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_rnd_1hll',
                        GRIDspec_node_1hll_closed_TF      = True,
                        ALGOspec_inst_selection_method    = 'random',
        ),  
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_probnpv_1hll',
                        GRIDspec_node_1hll_closed_TF      = True,
                        ALGOspec_inst_selection_method    = 'prob_weighted_npv',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs2p0',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'As2p0',
        ),  
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs4p0',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'As4p0',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs6p0',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_subsidy_name             = 'As6p0',
        ),


        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sBs0p4',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Bs0p4',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sBs0p6',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Bs0p6',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sBs0p8',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Bs0p8',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs2p4',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs2p4',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs2p6',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs2p6',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs2p8',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs2p8',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs4p4',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p4',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs4p6',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p6',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs4p8',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs4p8',
        ),

        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs6p4',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs6p4',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs6p6',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs6p6',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs6p8',
                        GRIDspec_node_1hll_closed_TF      = True,
                        GRIDspec_apply_prem_tiers_TF      = True,
                        GRIDspec_subsidy_name             = 'Cs6p8',
        ),
    


            ]
    
    LRG_final_max_scen_list = [
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max', 
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        # ),  

        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_sAs6p0',
        #         GRIDspec_subsidy_name             = 'As6p0',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs6p0',
        #         GRIDspec_node_1hll_closed_TF      = True,
        #         GRIDspec_subsidy_name             = 'As6p0',
        # ),

        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_sAs4p0',
                GRIDspec_subsidy_name             = 'As4p0',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sAs4p0',
                GRIDspec_node_1hll_closed_TF      = True,
                GRIDspec_subsidy_name             = 'As4p0',
        ),

        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_sCs2p4',
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Cs2p4',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs2p4',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Cs2p4',
        # ),

        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_sCs4p6',
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Cs4p6',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_max_1hll_sCs4p6',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 GRIDspec_subsidy_name             = 'Cs4p6',
        # ),

    ]
    
    LRG_final_rnd_scen_list = [
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_rnd',
        #                 ALGOspec_inst_selection_method    = 'random',
        # ),  
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_rnd_1hll',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 ALGOspec_inst_selection_method    = 'random',
        # ),  
    ]

    LRG_final_probnpv_scen_list = [
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_probnpv', 
        #                 ALGOspec_inst_selection_method    = 'prob_weighted_npv',

        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_probnpv_1hll',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 ALGOspec_inst_selection_method    = 'prob_weighted_npv',
        # ),

        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_probnpv_sAs6p0',
        #                 ALGOspec_inst_selection_method    = 'prob_weighted_npv',
        #                 GRIDspec_subsidy_name             = 'As6p0',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_probnpv_1hll_sAs6p0',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 ALGOspec_inst_selection_method    = 'prob_weighted_npv',
        #                 GRIDspec_subsidy_name             = 'As6p0',
        # ),

        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_probnpv_sAs4p0',
                        ALGOspec_inst_selection_method    = 'prob_weighted_npv',
                        GRIDspec_subsidy_name             = 'As4p0',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_probnpv_1hll_sAs4p0',
                        GRIDspec_node_1hll_closed_TF      = True,
                        ALGOspec_inst_selection_method    = 'prob_weighted_npv',
                        GRIDspec_subsidy_name             = 'As4p0',
        ),

        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_probnpv_sCs2p4',
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 ALGOspec_inst_selection_method    = 'prob_weighted_npv',
        #                 GRIDspec_subsidy_name             = 'Cs2p4',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_probnpv_1hll_sCs2p4',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 ALGOspec_inst_selection_method    = 'prob_weighted_npv',
        #                 GRIDspec_subsidy_name             = 'Cs2p4',
        # ),

        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_probnpv_sCs4p6',
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 ALGOspec_inst_selection_method    = 'prob_weighted_npv',
        #                 GRIDspec_subsidy_name             = 'Cs4p6',
        # ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{LRG_bfs_name}_probnpv_1hll_sCs4p6',
        #                 GRIDspec_node_1hll_closed_TF      = True,
        #                 GRIDspec_apply_prem_tiers_TF      = True,
        #                 ALGOspec_inst_selection_method    = 'prob_weighted_npv',
        #                 GRIDspec_subsidy_name             = 'Cs4p6',
        # ),

    ]

    


    XLRG_scen_list = [
    # pvalloc_scen_list = [
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}',
                bfs_numbers                       = XLRG_bfs_list,
        ), 
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}_sA1',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_subsidy_name             = 'A1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}_sA2',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_subsidy_name             = 'A2',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}_sA3',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_subsidy_name             = 'A3',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}_sB1',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_subsidy_name             = 'B1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}_sB2',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_subsidy_name             = 'B2',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}_sC1',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_subsidy_name             = 'C1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}_sC2',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_subsidy_name             = 'C2',
        ),
        
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}_1hll',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_node_1hll_closed_TF      = True,
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}_1hll_sA1',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_node_1hll_closed_TF      = True,
                GRIDspec_subsidy_name             = 'A1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}_1hll_sA2',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_node_1hll_closed_TF      = True,
                GRIDspec_subsidy_name             = 'A2',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}_1hll_sA3',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_node_1hll_closed_TF      = True,
                GRIDspec_subsidy_name             = 'A3',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}_1hll_sB1',
                bfs_numbers                       = XLRG_bfs_list,  
                GRIDspec_node_1hll_closed_TF      = True,
                GRIDspec_subsidy_name             = 'B1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}_1hll_sB2',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_node_1hll_closed_TF      = True,
                GRIDspec_subsidy_name             = 'B2',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}_1hll_sC1',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_node_1hll_closed_TF      = True,
                GRIDspec_subsidy_name             = 'C1',
        ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{XLRG_bfs_name}_1hll_sC2',
                bfs_numbers                       = XLRG_bfs_list,
                GRIDspec_node_1hll_closed_TF      = True,
                GRIDspec_subsidy_name             = 'C2',
        ),
    ]




# ==============================
# EXPORT Sub-Scenarios 
# ==============================
def get_subscen_list(sub_scen_str = 'test'):
    if sub_scen_str == 'test':
        return test_scen_list
    elif sub_scen_str == 'RUR':
        return RUR_scen_list
    elif sub_scen_str == 'SUB':
        return SUB_scen_list
    elif sub_scen_str == 'RUR_and_SUB':
        return RUR_scen_list + SUB_scen_list
    elif sub_scen_str == 'LRG':
        return LRG_scen_list
    elif sub_scen_str == 'LRG_final':
        return LRG_final_max_scen_list + LRG_final_rnd_scen_list + LRG_final_probnpv_scen_list
    elif sub_scen_str == 'XLRG':
        return XLRG_scen_list
    else:
        return []
    

if __name__ == '__main__':

    # call scen in array and run ------------------------------------------
    # SUB_and_RUR_list = SUB_scen_list + RUR_scen_list
    # test_scen_list
    # RUR_scen_list
    # SUB_scen_list
    # LRG_scen_list

    # XLRG_scen_list
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
    
            scen_class.run_pvalloc_initalization()
            scen_class.run_pvalloc_mcalgorithm()
    
    elif 'c:\\Models' in os.getcwd():

        for pvalloc_scen_index in range(0, len(pvalloc_scen_list)):
            pvalloc_scen = pvalloc_scen_list[pvalloc_scen_index]
            
            scen_class = PVAllocScenario(pvalloc_scen)
            scen_class.run_pvalloc_initalization()
            scen_class.run_pvalloc_mcalgorithm()
    
        print('done')







