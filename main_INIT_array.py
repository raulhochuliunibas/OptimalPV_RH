import os
import sys
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


if __name__ == '__main__':

    # pvalloc list ------------------------------------------
    pvalloc_scen_list = [

        # PVAllocScenario_Settings(name_dir_export ='pvalloc_2nbfs_2y_testMC',
        #     bfs_numbers                                          = [
        #                                                 2641, 2615,
        #                                                 # # RURAL - Beinwil, Lauwil, Bretzwil, Nunningen, Zullwil, Meltingen, Erschwil, Büsserach, Fehren, Seewen
        #                                                 # 2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616, 2480,
        #                                                 # # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
        #                                                 # 2613, 2782, 2618, 2786, 2785, 2772, 2761, 2743, 
        #                                                 # # URBAN: Reinach, Münchenstein, Muttenz
        #                                                 # 2773, 2769, 2770,
        #                                                             ],         
        #     # mini_sub_model_TF                                    = False,
        #     # mini_sub_model_ngridnodes                            = 20, 
        #     # mini_sub_model_nEGIDs                                = 100,
        #     create_gdf_export_of_topology                        = True,
        #     export_csvs                                          = False,
        #     T0_year_prediction                                   = 2022,
        #     months_lookback                                      = 12,
        #     months_prediction                                    = 240,
        #     TECspec_add_heatpump_demand_TF                       = True,   
        #     TECspec_heatpump_months_factor                       = [
        #                                                             (10, 7.0),
        #                                                             (11, 7.0), 
        #                                                             (12, 7.0), 
        #                                                             (1 , 7.0), 
        #                                                             (2 , 7.0), 
        #                                                             (3 , 7.0), 
        #                                                             (4 , 7.0), 
        #                                                             (5 , 7.0),     
        #                                                             (6 , 1.0), 
        #                                                             (7 , 1.0), 
        #                                                             (8 , 1.0), 
        #                                                             (9 , 1.0),
        #                                                             ], 
        #     ALGOspec_topo_subdf_partitioner                      = 250, 
        #     ALGOspec_inst_selection_method                       = 'max_npv',     # 'random', max_npv', 'prob_weighted_npv'
        #     ALGOspec_rand_seed                                   = 123,
        #     # GRIDspec_apply_prem_tiers_TF                         = True,
        #     # GRIDspec_node_1hll_closed_TF                         = True,
        # ),
        # PVAllocScenario_Settings(name_dir_export ='pvalloc_2nbfs_2y_testMC_1hll',
        #     bfs_numbers                                          = [
        #                                                 2641, 2615,
        #                                                 # # RURAL - Beinwil, Lauwil, Bretzwil, Nunningen, Zullwil, Meltingen, Erschwil, Büsserach, Fehren, Seewen
        #                                                 # 2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616, 2480,
        #                                                 # # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
        #                                                 # 2613, 2782, 2618, 2786, 2785, 2772, 2761, 2743, 
        #                                                 # # URBAN: Reinach, Münchenstein, Muttenz
        #                                                 # 2773, 2769, 2770,
        #                                                             ],         
        #     # mini_sub_model_TF                                    = False,
        #     # mini_sub_model_ngridnodes                            = 20, 
        #     # mini_sub_model_nEGIDs                                = 100,
        #     create_gdf_export_of_topology                        = True,
        #     export_csvs                                          = False,
        #     T0_year_prediction                                   = 2022,
        #     months_lookback                                      = 12,
        #     months_prediction                                    = 240,
        #     TECspec_add_heatpump_demand_TF                       = True,   
        #     TECspec_heatpump_months_factor                       = [
        #                                                             (10, 7.0),
        #                                                             (11, 7.0), 
        #                                                             (12, 7.0), 
        #                                                             (1 , 7.0), 
        #                                                             (2 , 7.0), 
        #                                                             (3 , 7.0), 
        #                                                             (4 , 7.0), 
        #                                                             (5 , 7.0),     
        #                                                             (6 , 1.0), 
        #                                                             (7 , 1.0), 
        #                                                             (8 , 1.0), 
        #                                                             (9 , 1.0),
        #                                                             ], 
        #     ALGOspec_topo_subdf_partitioner                      = 250, 
        #     ALGOspec_inst_selection_method                       = 'max_npv',     # 'random', max_npv', 'prob_weighted_npv'
        #     ALGOspec_rand_seed                                   = 123,
        #     GRIDspec_apply_prem_tiers_TF                         = True,
        #     GRIDspec_node_1hll_closed_TF                         = True,
        # ),

        PVAllocScenario_Settings(name_dir_export ='pvalloc_16nbfs_RUR_max_30y_histgr0-05',
            bfs_numbers                                          = [
                                                        # RURAL
                                                        2612, 2889, 2883, 2621, 2622,
                                                        2620, 2615, 2614, 2616, 2480,
                                                        2617, 2611, 2788, 2619, 2783, 2477, 

                                                                    ],          
            create_gdf_export_of_topology                        = True,
            export_csvs                                          = False,
            T0_year_prediction                                   = 2022,
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
            ALGOspec_rand_seed                                   = 123,
            # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
            CSTRspec_capacity_type                               = 'hist_constr_capa_year', 
            CSTRspec_ann_capacity_growth                         = 0.05,  

        ),
        
        PVAllocScenario_Settings(name_dir_export ='pvalloc_10nbfs_SUB_max_30y_histgr0-05',
            bfs_numbers                                          = [
                                                        # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
                                                        2613, 2782, 2618, 2786, 2785, 
                                                        2772, 2761, 2743, 2476, 2768,
                                                                    ],          
            create_gdf_export_of_topology                        = True,
            export_csvs                                          = False,
            T0_year_prediction                                   = 2022,
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
            ALGOspec_rand_seed                                   = 123,
            # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
            CSTRspec_capacity_type                               = 'hist_constr_capa_year', 
            CSTRspec_ann_capacity_growth                         = 0.05,  
        ),


        PVAllocScenario_Settings(name_dir_export ='pvalloc_16nbfs_RUR_max_30y_eb2050',
            bfs_numbers                                          = [
                                                        # RURAL
                                                        2612, 2889, 2883, 2621, 2622,
                                                        2620, 2615, 2614, 2616, 2480,
                                                        2617, 2611, 2788, 2619, 2783, 2477, 

                                                                    ],          
            create_gdf_export_of_topology                        = True,
            export_csvs                                          = False,
            T0_year_prediction                                   = 2022,
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
            ALGOspec_rand_seed                                   = 123,
            # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
        ),

        PVAllocScenario_Settings(name_dir_export ='pvalloc_10nbfs_SUB_max_30y_eb2050',
            bfs_numbers                                          = [
                                                        # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
                                                        2613, 2782, 2618, 2786, 2785, 
                                                        2772, 2761, 2743, 2476, 2768,
                                                                    ],          
            create_gdf_export_of_topology                        = True,
            export_csvs                                          = False,
            T0_year_prediction                                   = 2022,
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
            ALGOspec_rand_seed                                   = 123,
            # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
        ),


        PVAllocScenario_Settings(name_dir_export ='pvalloc_16nbfs_RUR_max_30y_eb2050_1hll',
            bfs_numbers                                          = [
                                                        # RURAL
                                                        2612, 2889, 2883, 2621, 2622,
                                                        2620, 2615, 2614, 2616, 2480,
                                                        2617, 2611, 2788, 2619, 2783, 2477, 

                                                                    ],          
            create_gdf_export_of_topology                        = True,
            export_csvs                                          = False,
            T0_year_prediction                                   = 2022,
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
            ALGOspec_rand_seed                                   = 123,
            GRIDspec_apply_prem_tiers_TF                         = True,
            GRIDspec_node_1hll_closed_TF                         = True,
        ),

        PVAllocScenario_Settings(name_dir_export ='pvalloc_10nbfs_SUB_max_30y_eb2050_1hll',
            bfs_numbers                                          = [
                                                        # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
                                                        2613, 2782, 2618, 2786, 2785, 
                                                        2772, 2761, 2743, 2476, 2768,
                                                                    ],          
            create_gdf_export_of_topology                        = True,
            export_csvs                                          = False,
            T0_year_prediction                                   = 2022,
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
            ALGOspec_rand_seed                                   = 123,
            GRIDspec_apply_prem_tiers_TF                         = True,
            GRIDspec_node_1hll_closed_TF                         = True,
        ),




    ]

    LARGE_scen_list = [

        PVAllocScenario_Settings(name_dir_export = 'pvalloc_29nbfs_max_30y_histgr0-05',
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
            create_gdf_export_of_topology                        = True,
            export_csvs                                          = False,
            T0_year_prediction                                   = 2022,
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
            ALGOspec_rand_seed                                   = 123,
            # GRIDspec_apply_prem_tiers_TF                         = False,
            # GRIDspec_node_1hll_closed_TF                         = False,
            CSTRspec_capacity_type                               = 'hist_constr_capa_year', 
            CSTRspec_ann_capacity_growth                         = 0.05,  

        ),

        PVAllocScenario_Settings(name_dir_export = 'pvalloc_29nbfs_max_30y_eb2050',
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
            create_gdf_export_of_topology                        = True,
            export_csvs                                          = False,
            T0_year_prediction                                   = 2022,
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
            ALGOspec_rand_seed                                   = 123,
            # GRIDspec_apply_prem_tiers_TF                         = False,
            # GRIDspec_node_1hll_closed_TF                         = False,
        ),

        PVAllocScenario_Settings(name_dir_export = 'pvalloc_29nbfs_max_30y_eb2050_1hll',
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
        create_gdf_export_of_topology                        = True,
        export_csvs                                          = False,
        T0_year_prediction                                   = 2022,
        months_lookback                                      = 12,
        months_prediction                                    = 480,
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
        ALGOspec_rand_seed                                   = 123,
        GRIDspec_apply_prem_tiers_TF                         = True,
        GRIDspec_node_1hll_closed_TF                         = True,
    ),

    ]

    ARE_type_scens = [

        PVAllocScenario_Settings(name_dir_export ='pvalloc_10nbfs_RUR_max_30y',
            bfs_numbers                                          = [
                                                        # # RURAL - Beinwil, Lauwil, Bretzwil, Nunningen, Zullwil, Meltingen, Erschwil, Büsserach, Fehren, Seewen
                                                        2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616, 2480,
                                                        # # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
                                                        # 2613, 2782, 2618, 2786, 2785, 2772, 2761, 2743, 
                                                        # # URBAN: Reinach, Münchenstein, Muttenz
                                                        # 2773, 2769, 2770,
                                                                    ],          
            create_gdf_export_of_topology                        = True,
            export_csvs                                          = False,
            T0_year_prediction                                   = 2022,
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
            ALGOspec_rand_seed                                   = 123,
            # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
        ),

        PVAllocScenario_Settings(name_dir_export ='pvalloc_10nbfs_RUR_rnd_30y',
            bfs_numbers                                          = [
                                                        # # RURAL - Beinwil, Lauwil, Bretzwil, Nunningen, Zullwil, Meltingen, Erschwil, Büsserach, Fehren, Seewen
                                                        2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616, 2480,
                                                        # # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
                                                        # 2613, 2782, 2618, 2786, 2785, 2772, 2761, 2743, 
                                                        # # URBAN: Reinach, Münchenstein, Muttenz
                                                        # 2773, 2769, 2770,
                                                                    ],          
            create_gdf_export_of_topology                        = True,
            export_csvs                                          = False,
            T0_year_prediction                                   = 2022,
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
            ALGOspec_inst_selection_method                       = 'random',     # 'random', max_npv', 'prob_weighted_npv'
            ALGOspec_rand_seed                                   = 123,
            # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
        ),
        

        PVAllocScenario_Settings(name_dir_export ='pvalloc_8nbfs_SUB_max_30y',
            bfs_numbers                                          = [
                                                        # # RURAL - Beinwil, Lauwil, Bretzwil, Nunningen, Zullwil, Meltingen, Erschwil, Büsserach, Fehren, Seewen
                                                        # 2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616, 2480,
                                                        # # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
                                                        2613, 2782, 2618, 2786, 2785, 2772, 2761, 2743, 
                                                        # # URBAN: Reinach, Münchenstein, Muttenz
                                                        # 2773, 2769, 2770,
                                                                    ],          
            create_gdf_export_of_topology                        = True,
            export_csvs                                          = False,
            T0_year_prediction                                   = 2022,
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
            ALGOspec_rand_seed                                   = 123,
            # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
        ),

        PVAllocScenario_Settings(name_dir_export ='pvalloc_8nbfs_SUB_rnd_30y',
            bfs_numbers                                          = [
                                                        # # RURAL - Beinwil, Lauwil, Bretzwil, Nunningen, Zullwil, Meltingen, Erschwil, Büsserach, Fehren, Seewen
                                                        # 2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616, 2480,
                                                        # # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
                                                        2613, 2782, 2618, 2786, 2785, 2772, 2761, 2743, 
                                                        # # URBAN: Reinach, Münchenstein, Muttenz
                                                        # 2773, 2769, 2770,
                                                                    ],          
            create_gdf_export_of_topology                        = True,
            export_csvs                                          = False,
            T0_year_prediction                                   = 2022,
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
            ALGOspec_inst_selection_method                       = 'random',     # 'random', max_npv', 'prob_weighted_npv'
            ALGOspec_rand_seed                                   = 123,
            # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
        ),

  
        PVAllocScenario_Settings(name_dir_export ='pvalloc_3nbfs_URB_max_30y',
            bfs_numbers                                          = [
                                                        # # RURAL - Beinwil, Lauwil, Bretzwil, Nunningen, Zullwil, Meltingen, Erschwil, Büsserach, Fehren, Seewen
                                                        # 2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616, 2480,
                                                        # # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
                                                        # 2613, 2782, 2618, 2786, 2785, 2772, 2761, 2743, 
                                                        # # URBAN: Reinach, Münchenstein, Muttenz
                                                        2773, 2769, 2770,
                                                                    ],          
            create_gdf_export_of_topology                        = True,
            export_csvs                                          = False,
            T0_year_prediction                                   = 2022,
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
            ALGOspec_rand_seed                                   = 123,
            # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
        ),
    
        PVAllocScenario_Settings(name_dir_export ='pvalloc_3nbfs_URB_rnd_30y',
            bfs_numbers                                          = [
                                                        # # RURAL - Beinwil, Lauwil, Bretzwil, Nunningen, Zullwil, Meltingen, Erschwil, Büsserach, Fehren, Seewen
                                                        # 2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616, 2480,
                                                        # # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
                                                        # 2613, 2782, 2618, 2786, 2785, 2772, 2761, 2743, 
                                                        # # URBAN: Reinach, Münchenstein, Muttenz
                                                        2773, 2769, 2770,
                                                                    ],          
            create_gdf_export_of_topology                        = True,
            export_csvs                                          = False,
            T0_year_prediction                                   = 2022,
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
            ALGOspec_inst_selection_method                       = 'random',     # 'random', max_npv', 'prob_weighted_npv'
            ALGOspec_rand_seed                                   = 123,
            # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
        ),
      
    ]


    slurm_job_id = os.environ.get('SLURM_JOB_ID_ENV', 'unknown')

    if len(sys.argv) > 1:
        pvalloc_scen_index = int(sys.argv[1])
        pvalloc_scen = pvalloc_scen_list[pvalloc_scen_index]

    else: 
        pvalloc_scen = PVAllocScenario_Settings()
    
    scen_class = PVAllocScenario(pvalloc_scen)
    scen_class.run_pvalloc_initalization()
    scen_class.run_pvalloc_mcalgorithm()
    
print('done')

