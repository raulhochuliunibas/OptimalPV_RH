import os
import sys
from dataclasses import replace
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


if __name__ == '__main__':

    def make_scenario(default_scen, name_dir_export, bfs_numbers=None, **overrides):
        kwargs = {'name_dir_export': name_dir_export}
        if bfs_numbers is not None:
            kwargs['bfs_numbers'] = bfs_numbers
        if overrides:
            kwargs.update(overrides)
        return replace(default_scen, **kwargs)

    
    pvalloc_2nbfs_test_DEFAULT = PVAllocScenario_Settings(name_dir_export ='pvalloc_2nbfs_test_DEFAULT',
            bfs_numbers                                          = [
                                                        2641, 2615,
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
            CSTRspec_ann_capacity_growth                         = 0.1,
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
        CSTRspec_ann_capacity_growth                         = 0.1,
    ) 
    RUR_bfs_list =[
        # RURAL
        2612, 2889, 2883, 2621, 2622,
        2620, 2615, 2614, 2616, 2480,
        2617, 2611, 2788, 2619, 2783, 2477, 
    ]
    SUB_bfs_list = [
        # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
        2613, 2782, 2618, 2786, 2785, 
        2772, 2761, 2743, 2476, 2768,
    ]

        


    # pvalloc list ------------------------------------------
    # pvalloc_scen_list = []
    asdf = [

        make_scenario(pvalloc_2nbfs_test_DEFAULT, 'pvalloc_2nbfs_2y_testMC'),

        make_scenario(pvalloc_2nbfs_test_DEFAULT, 'pvalloc_2nbfs_2y_testMC_eb2050',
                      CSTRspec_capacity_type        ='ep2050_zerobasis', # hist_constr_capa_year / hist_constr_capa_month / ep2050_zerobasis
        ),
        make_scenario(pvalloc_2nbfs_test_DEFAULT, 'pvalloc_2nbfs_2y_testMC_eb2050_1hll',
                      CSTRspec_capacity_type        ='ep2050_zerobasis', # hist_constr_capa_year / hist_constr_capa_month / ep2050_zerobasis
                      GRIDspec_node_1hll_closed_TF  = True,
        ),
        make_scenario(pvalloc_2nbfs_test_DEFAULT, 'pvalloc_2nbfs_2y_testMC_eb2050_1hll_ewfirst',
                      CSTRspec_capacity_type        ='ep2050_zerobasis', # hist_constr_capa_year / hist_constr_capa_month / ep2050_zerobasis
                      GRIDspec_node_1hll_closed_TF  = True,
                      ALGOspec_subselec_filter_criteria =  ('eastwest_2r', 'eastwest_nr' )
        ),


    ]
    # RUR_SUB_scen_list = [
    pvalloc_scen_list = [
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, 'pvalloc_16nbfs_RUR_30y_histgr',
        #               bfs_numbers                       = RUR_bfs_list,
        #               CSTRspec_capacity_type            ='hist_constr_capa_year',
        #     ), 
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, 'pvalloc_16nbfs_RUR_30y_eb2050', 
        #               bfs_numbers                       = RUR_bfs_list,
        #               CSTRspec_capacity_type            ='ep2050_zerobasis',
        #     ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, 'pvalloc_16nbfs_RUR_30y_eb2050_ewfirst', 
                      bfs_numbers                       = RUR_bfs_list,
                      CSTRspec_capacity_type            ='ep2050_zerobasis',
                        ALGOspec_subselec_filter_criteria =  ('eastwest_2r', 'eastwest_nr'),
            ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, 'pvalloc_16nbfs_RUR_30y_eb2050_1hll', 
        #               bfs_numbers                       = RUR_bfs_list,
        #               CSTRspec_capacity_type            ='ep2050_zerobasis',
        #               GRIDspec_node_1hll_closed_TF      = True,
        #     ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, 'pvalloc_16nbfs_RUR_30y_1hll_ewfirst', 
        #               bfs_numbers                       = RUR_bfs_list,
        #               CSTRspec_capacity_type            ='ep2050_zerobasis',
        #               GRIDspec_node_1hll_closed_TF      = True,
        #               ALGOspec_subselec_filter_criteria =  ('eastwest_2r', 'eastwest_nr' )
        #     ),

        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, 'pvalloc_10nbfs_SUB_30y_histgr',
        #               bfs_numbers                       = SUB_bfs_list,
        #               CSTRspec_capacity_type            ='hist_constr_capa_year',
        #     ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, 'pvalloc_10nbfs_SUB_30y_eb2050', 
        #               bfs_numbers                       = SUB_bfs_list,
        #               CSTRspec_capacity_type            ='ep2050_zerobasis',
        #     ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, 'pvalloc_10nbfs_SUB_30y_eb2050_ewfirst', 
                      bfs_numbers                       = SUB_bfs_list,
                      CSTRspec_capacity_type            ='ep2050_zerobasis',
                        ALGOspec_subselec_filter_criteria =  ('eastwest_2r', 'eastwest_nr'),
            ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, 'pvalloc_10nbfs_SUB_30y_eb2050_1hll', 
        #               bfs_numbers                       = SUB_bfs_list,
        #               CSTRspec_capacity_type            ='ep2050_zerobasis',
        #               GRIDspec_node_1hll_closed_TF      = True,
        #     ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, 'pvalloc_10nbfs_SUB_30y_1hll_ewfirst', 
        #               bfs_numbers                       = SUB_bfs_list,
        #               CSTRspec_capacity_type            ='ep2050_zerobasis',
        #               GRIDspec_node_1hll_closed_TF      = True,
        #               ALGOspec_subselec_filter_criteria =  ('eastwest_2r', 'eastwest_nr' )
        #     ),

        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, 'pvalloc_29nbfs_30y_histgr0-05'),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, 'pvalloc_29nbfs_30y_eb2050',
        #               CSTRspec_capacity_type            ='ep2050_zerobasis',
        #     ),
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, 'pvalloc_29nbfs_30y_eb2050_ewfirst',
                      CSTRspec_capacity_type            ='ep2050_zerobasis',
                      ALGOspec_subselec_filter_criteria =  ('eastwest_2r', 'eastwest_nr'),
            ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, 'pvalloc_29nbfs_30y_eb2050_1hll',
        #               CSTRspec_capacity_type            ='ep2050_zerobasis',
        #               GRIDspec_node_1hll_closed_TF      = True,
        #     ),
        # make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, 'pvalloc_29nbfs_30y_eb2050_1hll_ewfirst',
        #               CSTRspec_capacity_type            ='ep2050_zerobasis',
        #               GRIDspec_node_1hll_closed_TF      = True,
        #               ALGOspec_subselec_filter_criteria =  ('eastwest_2r', 'eastwest_nr' )
        #     ),
        
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







