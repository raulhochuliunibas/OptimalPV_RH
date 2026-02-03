import os
import sys
from dataclasses import replace
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.auxiliary_functions import make_scenario

if __name__ == "__main__":

    # pvalloc_scen_list = get_subscen_list('LRG')
    # pvalloc_scen_list = get_subscen_list('LRG_final')

    slurm_job_id = os.environ.get('SLURM_ARRAY_JOB_ID_ENV', 'unknown')
    slurm_array_id = os.environ.get('SLURM_ARRAY_TASK_ID_ENV', 'unknown')
    slurm_full_id = f"{slurm_job_id}_{slurm_array_id}"


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

            # OPTIMspecs_gridnode_subsample                        = grid_node_str,
        ) 
    SUB_bfs_name = 'pvalloc_10nbfs_SUB'
    SUB_bfs_list = [
        # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
        2613, 2782, 2618, 2786, 2785, 
        2772, 2761, 2743, 2476, 2768,
    ]


    pvalloc_scen_list =[
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_gridoptim_max', 
                                 bfs_numbers                       = SUB_bfs_list,
                                 run_pvalloc_initalization_TF    = True,
                                 run_pvalloc_mcalgorithm_TF      = False,
                                 run_gridoptimized_orderinst_TF  = True,
                                 run_gridoptimized_expansion_TF  = True,
                                 OPTIMspecs_gridnode_subsample           = 'all_nodes_pyparallel', 
                                 OPTEXPApecs_apply_gridoptim_order_TF     = True,
                                 ), 
        make_scenario(pvalloc_Xnbfs_ARE_30y_DEFAULT, f'{SUB_bfs_name}_gridoptim_loop_max', 
                                 bfs_numbers                       = SUB_bfs_list,
                                 run_pvalloc_initalization_TF    = True,
                                 run_pvalloc_mcalgorithm_TF      = False,
                                 run_gridoptimized_orderinst_TF  = True,
                                 run_gridoptimized_expansion_TF  = True,
                                 OPTIMspecs_gridnode_subsample           = 'all_nodes_loop', 
                                 OPTEXPApecs_apply_gridoptim_order_TF     = True,
                                 ), 
    ]
                                 


    for pvalloc_scen in pvalloc_scen_list:
            
        scen_class = PVAllocScenario(pvalloc_scen)
        scen_class.sett.slurm_full_id        = slurm_full_id

        if scen_class.sett.run_pvalloc_initalization_TF:
            scen_class.run_pvalloc_initialization()

        if scen_class.sett.run_gridoptimized_orderinst_TF:
            scen_class.run_gridoptimized_orderinst()

        if scen_class.sett.run_gridoptimized_expansion_TF:
            scen_class.run_gridoptimized_expansion()

    print('done')
        