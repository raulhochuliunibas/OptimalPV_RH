
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


# pv alloctaion ------------------------------------------
pvalloc_scen_list = [

    PVAllocScenario_Settings(name_dir_export ='pvalloc_2nbfs_2y_testMC',
        bfs_numbers                                          = [
                                                    2641, 2615,
                                                    # # RURAL - Beinwil, Lauwil, Bretzwil, Nunningen, Zullwil, Meltingen, Erschwil, Büsserach, Fehren, Seewen
                                                    # 2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616, 2480,
                                                    # # SUBURBAN - Breitenbach, Brislach, Himmelried, Grellingen, Duggingen, Pfeffingen, Aesch, Dornach
                                                    # 2613, 2782, 2618, 2786, 2785, 2772, 2761, 2743, 
                                                    # # URBAN: Reinach, Münchenstein, Muttenz
                                                    # 2773, 2769, 2770,
                                                                ],         
        mini_sub_model_TF                                    = False,
        mini_sub_model_ngridnodes                            = 20, 
        mini_sub_model_nEGIDs                                = 100,
        create_gdf_export_of_topology                        = True,
        export_csvs                                          = True,
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

    PVAllocScenario_Settings(name_dir_export = 'pvalloc_Xnbfs_LRG_max_40y',
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
        # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
    ),

    PVAllocScenario_Settings(name_dir_export = 'pvalloc_Xnbfs_LRG_rnd_40y',
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
        ALGOspec_inst_selection_method                       = 'random',     # 'random', max_npv', 'prob_weighted_npv'
        ALGOspec_rand_seed                                   = 123,
        # ALGOspec_subselec_filter_criteria = 'southwestfacing_2spec', 
    ),



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
        export_csvs                                          = True,
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
        export_csvs                                          = True,
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
        export_csvs                                          = True,
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
        export_csvs                                          = True,
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
        export_csvs                                          = True,
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
        export_csvs                                          = True,
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


    
    ]


# visualization ------------------------------------------
visualization_list = [
    Visual_Settings(
        pvalloc_exclude_pattern_list = [
            '*.txt','*.xlsx','*.csv','*.parquet',
            '*old_vers*',
            ], 
        pvalloc_include_pattern_list = [
            # 'pvalloc_1bfs_RUR_r0-02_max',  
            # 'pvalloc_9bfs_RUR_r0-02_max_15y',
            # 'pvalloc_9bfs_RUR_r0-02_max_TEST,
        ],
        plot_show                          = False,
        # save_plot_by_scen_directory        = False,
            
        # remove_old_plot_scen_directories   = True,  
        # remove_old_plots_in_visualization  = True,  
        # remove_old_csvs_in_visualization   = True, 
    
        # cut_timeseries_to_zoom_hour        = True,
        # add_day_night_HOY_bands            = True,

        
        # # # -- def plot_ALL_init_sanitycheck(self, ): --- [run plot,  show plot,  show all scen] ---------
        # plot_ind_var_summary_stats_TF                   = [True,      False,       False],
        # plot_ind_line_meteo_radiation_TF                = [True,      True,       False],

        # # # # -- def plot_ALL_mcalgorithm(self,): --------- [run plot,  show plot,  show all scen] ---------
        # plot_ind_hist_pvcapaprod_TF                     = [True,      True,       False],
        # plot_ind_lineband_contcharact_newinst_TF        = [True,      True,       False],
    
        plot_ind_mapline_prodHOY_EGIDrfcombo_TF         = [True,      True,       False],                   
        plot_ind_line_productionHOY_per_EGID_TF         = [True,      True,       False],                
        plot_ind_line_productionHOY_per_node_TF         = [True,      True,       False],                   
        plot_ind_line_PVproduction_TF                   = [True,      True,       False],      #TO BE ADJUSTED!!             
        # plot_ind_hist_cols_HOYagg_per_EGID_TF           = [True,      True,       False],                   
        # plot_ind_map_topo_egid_TF                       = [True,      True,       False],
        plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False],

            plot_ind_mapline_prodHOY_EGIDrfcombo_specs = {
                    'specific_exclude_EGIDs': [
                            '190178022', # strange large installation              
                            '190296588',      
                            ],
                    'specific_selected_EGIDs': [
                        # bfs: 2883 - Bretzwil
                        # '3030694',      # pv_df
                        # '2362100',    # pv_df  2 part
                        # '245052560',    # pv_df 2 part
                        # '3032150',    # pv_df  3 part
                        # '434234',
    
                       # 9bfs: all of RUR bfs
                    #    '101428161', '11513725', '190001512',
                        '101428161', '11513725', '190001512', '190004146', '190024109', '190033245', 
                        '190048248', '190083872', '190109228', '190116571', '190144906', '190178022', 
                        '190183577', '190185552', '190251628', '190251772', '190251828', '190296588', 
                        '190491308', '190694269', '190709629', '190814490', '190912633', '190960689', 
                        '2125434', '2362100', '245044986', '245048760', '245052560', '245053405', 
                        '245057989', '245060059', '3030694', '3030905', '3031761', '3032150', '3033714', 
                        '3075084', '386736', '432600', '432638', '432671', '432683', '432701', '432729', '434178',

                                                ], 
                    'run_selected_plot_parts': [
                        # 'map', 
                        # 'all_combo_TS',
                        # 'actual_inst_TS',
                        # 'outsample_TS',
                        'recalc_opt_inst',
                        # 'summary'
                    ], 
                    'show_selected_plots': [
                                                # 'map',  
                                                # 'timeseries', 
                                                # 'summary', 
                                                # 'econ_func', 
                                                'joint_scatter'
                                                ],
                    'traces_in_timeseries_plot': [
                                                'full_partition_combo', 
                                                'actual_pv_inst', 
                                                'recalc_pv_inst',
                                                ], 
                    'rndm_sample_seed': 123,
                    'n_rndm_egid_winst_pvdf': 0, 
                    'n_rndm_egid_winst_alloc': 0,
                    'n_rndm_egid_woinst': 0,
                    'n_rndm_egid_outsample': 0, 
                    'n_partition_pegid_minmax': (1,2), 
                    'roofpartition_color': '#fff2ae',
                    'actual_ts_trace_marker': 'cross',
                    'summary_cols_only_in_legend': ['EGID', 'inst_TF', 'info_source', 'grid_node',
                                                    'GKLAS', 'GAREA', 'are_typ', 'sfhmfh_typ', 
                                                    'TotalPower', 'demand_elec_pGAREA', 
                                                    'pvtarif_Rp_kWh', 'elecpri_Rp_kWh', ],
                    'recalc_opt_max_flaeche_factor': 1.5, #  1.5,  
                    'recalc_interest_rate_list': [
                        0.020, 
                        # 0.030,
                        # 0.040, 
                        # 0.050,
                        # 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049,
                        # 0.051, 0.052, 0.053, 0.054, 0.055, 0.056, 0.057, 0.058, 0.059,
                        ],
                    # 'tryout_generic_pvtariff_elecpri_Rp_kWh': (5.0, 25.0),
                    'tryout_generic_pvtariff_elecpri_Rp_kWh': (0.6, 25.0),
                    'generic_pvinstcost_coeff_total': {
                                                        'use_generic_instcost_coeff_total_TF': False, 
                                                          'coeff_total_dict_list': [
                                                            {'a': 1193.8, 'b': 4860.3, 'c': 0.7291},  #  estimates pvinst_coefficients 
                                                            {'a': 6000, 'b': 5000, 'c': 0.65},  #  higher fix cost, more curvutre, lower cost for large inst
                                                            {'a': 4000, 'b': 3300, 'c': 0.83},  #  higher fix cost, then much closes to estim function, higher cost for large inst
                                                            {'a': 7000, 'b': 1500, 'c': 0.85},  #  much higher fix costs, much less change in curvature, almost linear, much lower cost for large inst
                                                            ],                                                         
                  }, 
                    'noisy_demand_TF': False, 

                    'noisy_demand_factor': 0.5
                    },
        
    )
    ]


if __name__ == '__main__':

    # pv alloctaion ---------------------
    for pvalloc_scen in pvalloc_scen_list:
        scen_class = PVAllocScenario(pvalloc_scen)
        scen_class.run_pvalloc_initalization()
        scen_class.run_pvalloc_mcalgorithm()

    # visualization ---------------------
    for visual_scen in visualization_list:
        visual_class = Visualization(visual_scen)
        # visual_class.plot_ALL_init_sanitycheck()
        # visual_class.plot_ALL_mcalgorithm()

print('done')

