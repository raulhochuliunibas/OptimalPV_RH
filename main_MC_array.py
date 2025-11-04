import os
import sys
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


if __name__ == '__main__':
    
    slurm_job_id = os.environ.get('SLURM_JOB_ID_ENV', 'unknown')

    if len(sys.argv) > 1:
        MC_iter = int(sys.argv[1])
        pvalloc_scen_list = [

            PVAllocScenario_Settings(name_dir_export ='pvalloc_2nbfs_2y_testMC',
            MCspec_montecarlo_iterations_fordev_sequentially     = 2,
            export_csvs                                          = True,
            
            bfs_numbers                                          = [2612, ],          
            create_gdf_export_of_topology                        = True,
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
            ALGOspec_rand_seed                                   = 123,
            # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
            ),

        ]

    else:
        pvalloc_scen_list = [
            PVAllocScenario_Settings(),
        ]

    # pv alloctaion ---------------------
    for pvalloc_scen in pvalloc_scen_list:
        scen_class = PVAllocScenario(pvalloc_scen)
        # scen_class.run_pvalloc_initalization()
        scen_class.run_pvalloc_mcalgorithm()

print('done')

