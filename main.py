
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


pvalloc_scen_list = [
    PVAllocScenario_Settings(
        name_dir_export='pvalloc_mini_BYMONTH_rnd',
        mini_sub_model_TF= True,
        test_faster_array_computation= True,
        create_gdf_export_of_topology = False,
        T0_year_prediction                                   = 2021,
        months_prediction                                    = 30,
        CSTRspec_iter_time_unit                              = 'month',
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF  = True, 
        ALGOspec_topo_subdf_partitioner                      = 250, 
        ALGOspec_inst_selection_method                       = 'random', 
        # ALGOspec_inst_selection_method                     = 'prob_weighted_npv',
        ALGOspec_rand_seed                                   = 123,
    ), 
    PVAllocScenario_Settings(
        name_dir_export='pvalloc_mini_BYYEAR_rnd',
        mini_sub_model_TF= True,
        test_faster_array_computation= True,
        create_gdf_export_of_topology = False,
        T0_year_prediction                                   = 2021,
        months_prediction                                    = 30,
        CSTRspec_iter_time_unit                              = 'year',
        ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF  = True, 
        ALGOspec_topo_subdf_partitioner                      = 250, 
        ALGOspec_inst_selection_method                       = 'random', 
        # ALGOspec_inst_selection_method                     = 'prob_weighted_npv',
        ALGOspec_rand_seed                                   = 123,
    ),    

]

visualization_list = [

        Visual_Settings(
            pvalloc_exclude_pattern_list = [
                '*.txt','*old_vers*', 
                'pvalloc_BLsml_10y_f2013_1mc_meth2.2_npv'
                ], 
            save_plot_by_scen_directory        = False, 
            remove_old_plot_scen_directories   = False,  
            remove_old_plots_in_visualization = False,  
            ),    
    ]    


if __name__ == '__main__':

    # pv alloctaion ---------------------
    for pvalloc_scen in pvalloc_scen_list:
        scen_class = PVAllocScenario(pvalloc_scen)

        scen_class.run_pvalloc_initalization()
        scen_class.run_pvalloc_mcalgorithm()
        # pvalloc_self.sett.run_pvalloc_postprocess()

    # visualization ---------------------
    for visual_scen in visualization_list:
        visual_class = Visualization(visual_scen)

        try: 
            visual_class.plot_ALL_init_sanitycheck()

        except Exception as e:
            print(f'ERROR: could not plot ALL init_sanitycheck or mcalgorithm: {e}')
            pass

        try:
            visual_class.plot_ALL_mcalgorithm()
        except Exception as e:
            print(f'ERROR: could not plot ALL mcalgorithm: {e}')
            pass

print('done')

