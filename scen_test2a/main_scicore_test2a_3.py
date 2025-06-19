import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization

pvalloc_scen_list = [

    # SCENARIOS 2A: test2 scens with SINGLE HOUSING BUILDINGS

    PVAllocScenario_Settings(
        name_dir_export                 = 'pvalloc_URB_test2a_southfacing_rnd',
        bfs_numbers                     = [2773, 2769, 2770,                    # URBAN: Reinach, Münchenstein, Muttenz
                                        ], 
        T0_year_prediction              = 2021,
        months_prediction               = 360,
        CSTRspec_iter_time_unit         = 'year',
        overwrite_scen_init             = False,
        CHECKspec_n_iterations_before_sanitycheck   = 2,
        ALGOspec_inst_selection_method              = 'random', 
        ALGOspec_rand_seed                          = 123,
        TECspec_pvprod_calc_method                  = 'method2.2',
        MCspec_montecarlo_iterations_fordev_sequentially                = 1,
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True,

        ALGOspec_subselec_filter_criteria = 'southfacing_1spec', 
        ),


]


visualization_list = [

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*', 
                ], 
            pvalloc_include_pattern_list = [
                '*test2*', 
            ],
            save_plot_by_scen_directory        = True, 
            remove_old_plot_scen_directories   = False,  
            remove_old_plots_in_visualization  = False,  
            remove_old_csvs_in_visualization   = False, 
            ),    
    ]       


if __name__ == '__main__':

    # pv alloctaion ---------------------
    for pvalloc_scen in pvalloc_scen_list:
        pvalloc_class = PVAllocScenario(pvalloc_scen)
        
        if (pvalloc_class.sett.overwrite_scen_init) or (not os.path.exists(pvalloc_class.sett.name_dir_export_path)): 
            pvalloc_class.run_pvalloc_initalization()

        pvalloc_class.run_pvalloc_mcalgorithm()
        # pvalloc_self.sett.run_pvalloc_postprocess()



print('\n\n -------- done -------')
