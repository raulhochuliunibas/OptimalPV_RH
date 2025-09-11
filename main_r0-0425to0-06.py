
from src.MAIN_pvallocation import PVAllocScenario_Settings, PVAllocScenario
from src.MAIN_visualization import Visual_Settings, Visualization


# pv alloctaion ------------------------------------------
pvalloc_scen_list = [

PVAllocScenario_Settings(name_dir_export ='pvalloc_9bfs_RUR_r0-042_max_TEST',
    bfs_numbers = [
        2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616,
    ],
    mini_sub_model_TF = False,
    mini_sub_model_nEGIDs = 50,
    mini_sub_model_select_EGIDs = [
        '101428161', '11513725', '190001512', '190004146', '190024109', '190033245', 
        '190048248', '190083872', '190109228', '190116571', '190144906', '190178022', 
        '190183577', '190185552', '190251628', '190251772', '190251828', '190296588', 
        '190491308', '190694269', '190709629', '190814490', '190912633', '190960689', 
        '2125434', '2362100', '245044986', '245048760', '245052560', '245053405', 
        '245057989', '245060059', '3030694', '3030905', '3031761', '3032150', '3033714', 
        '3075084', '386736', '432600', '432638', '432671', '432683', '432701', '432729', '434178',
    ],
    T0_year_prediction = 2021,
    months_prediction = 12,
    TECspec_self_consumption_ifapplicable = 1.0,
    TECspec_interest_rate = 0.042,
    CSTRspec_ann_capacity_growth = 0.2,
    ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True, 
    ALGOspec_topo_subdf_partitioner = 250, 
    ALGOspec_inst_selection_method = 'max_npv', 
    ALGOspec_rand_seed = 123,
),

PVAllocScenario_Settings(name_dir_export ='pvalloc_9bfs_RUR_r0-044_max_TEST',
    bfs_numbers = [2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616],
    mini_sub_model_TF = False,
    mini_sub_model_nEGIDs = 50,
    T0_year_prediction = 2021,
    months_prediction = 12,
    TECspec_self_consumption_ifapplicable = 1.0,
    TECspec_interest_rate = 0.044,
    CSTRspec_ann_capacity_growth = 0.2,
    ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True, 
    ALGOspec_topo_subdf_partitioner = 250, 
    ALGOspec_inst_selection_method = 'max_npv', 
    ALGOspec_rand_seed = 123,
),

PVAllocScenario_Settings(name_dir_export ='pvalloc_9bfs_RUR_r0-046_max_TEST',
    bfs_numbers = [2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616],
    mini_sub_model_TF = False,
    mini_sub_model_nEGIDs = 50,
    T0_year_prediction = 2021,
    months_prediction = 12,
    TECspec_self_consumption_ifapplicable = 1.0,
    TECspec_interest_rate = 0.046,
    CSTRspec_ann_capacity_growth = 0.2,
    ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True, 
    ALGOspec_topo_subdf_partitioner = 250, 
    ALGOspec_inst_selection_method = 'max_npv', 
    ALGOspec_rand_seed = 123,
),

PVAllocScenario_Settings(name_dir_export ='pvalloc_9bfs_RUR_r0-048_max_TEST',
    bfs_numbers = [2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616],
    mini_sub_model_TF = False,
    mini_sub_model_nEGIDs = 50,
    T0_year_prediction = 2021,
    months_prediction = 12,
    TECspec_self_consumption_ifapplicable = 1.0,
    TECspec_interest_rate = 0.048,
    CSTRspec_ann_capacity_growth = 0.2,
    ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True, 
    ALGOspec_topo_subdf_partitioner = 250, 
    ALGOspec_inst_selection_method = 'max_npv', 
    ALGOspec_rand_seed = 123,
),

PVAllocScenario_Settings(name_dir_export ='pvalloc_9bfs_RUR_r0-050_max_TEST',
    bfs_numbers = [2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616],
    mini_sub_model_TF = False,
    mini_sub_model_nEGIDs = 50,
    T0_year_prediction = 2021,
    months_prediction = 12,
    TECspec_self_consumption_ifapplicable = 1.0,
    TECspec_interest_rate = 0.050,
    CSTRspec_ann_capacity_growth = 0.2,
    ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True, 
    ALGOspec_topo_subdf_partitioner = 250, 
    ALGOspec_inst_selection_method = 'max_npv', 
    ALGOspec_rand_seed = 123,
),

PVAllocScenario_Settings(name_dir_export ='pvalloc_9bfs_RUR_r0-052_max_TEST',
    bfs_numbers = [2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616],
    mini_sub_model_TF = False,
    mini_sub_model_nEGIDs = 50,
    T0_year_prediction = 2021,
    months_prediction = 12,
    TECspec_self_consumption_ifapplicable = 1.0,
    TECspec_interest_rate = 0.052,
    CSTRspec_ann_capacity_growth = 0.2,
    ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True, 
    ALGOspec_topo_subdf_partitioner = 250, 
    ALGOspec_inst_selection_method = 'max_npv', 
    ALGOspec_rand_seed = 123,
),

PVAllocScenario_Settings(name_dir_export ='pvalloc_9bfs_RUR_r0-054_max_TEST',
    bfs_numbers = [2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616],
    mini_sub_model_TF = False,
    mini_sub_model_nEGIDs = 50,
    T0_year_prediction = 2021,
    months_prediction = 12,
    TECspec_self_consumption_ifapplicable = 1.0,
    TECspec_interest_rate = 0.054,
    CSTRspec_ann_capacity_growth = 0.2,
    ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True, 
    ALGOspec_topo_subdf_partitioner = 250, 
    ALGOspec_inst_selection_method = 'max_npv', 
    ALGOspec_rand_seed = 123,
),

PVAllocScenario_Settings(name_dir_export ='pvalloc_9bfs_RUR_r0-056_max_TEST',
    bfs_numbers = [2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616],
    mini_sub_model_TF = False,
    mini_sub_model_nEGIDs = 50,
    T0_year_prediction = 2021,
    months_prediction = 12,
    TECspec_self_consumption_ifapplicable = 1.0,
    TECspec_interest_rate = 0.056,
    CSTRspec_ann_capacity_growth = 0.2,
    ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True, 
    ALGOspec_topo_subdf_partitioner = 250, 
    ALGOspec_inst_selection_method = 'max_npv', 
    ALGOspec_rand_seed = 123,
),
PVAllocScenario_Settings(name_dir_export ='pvalloc_9bfs_RUR_r0-058_max_TEST',
    bfs_numbers = [2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616],
    mini_sub_model_TF = False,
    mini_sub_model_nEGIDs = 50,
    T0_year_prediction = 2021,
    months_prediction = 12,
    TECspec_self_consumption_ifapplicable = 1.0,
    TECspec_interest_rate = 0.058,
    CSTRspec_ann_capacity_growth = 0.2,
    ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True, 
    ALGOspec_topo_subdf_partitioner = 250, 
    ALGOspec_inst_selection_method = 'max_npv', 
    ALGOspec_rand_seed = 123,
),
PVAllocScenario_Settings(name_dir_export ='pvalloc_9bfs_RUR_r0-060_max_TEST',
    bfs_numbers = [2612, 2889, 2883, 2621, 2622, 2620, 2615, 2614, 2616],
    mini_sub_model_TF = False,
    mini_sub_model_nEGIDs = 50,
    T0_year_prediction = 2021,
    months_prediction = 12,
    TECspec_self_consumption_ifapplicable = 1.0,
    TECspec_interest_rate = 0.006,
    CSTRspec_ann_capacity_growth = 0.2,
    ALGOspec_adjust_existing_pvdf_pvprod_bypartition_TF = True, 
    ALGOspec_topo_subdf_partitioner = 250, 
    ALGOspec_inst_selection_method = 'max_npv', 
    ALGOspec_rand_seed = 123,
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
            # 'pvalloc_mini_aggr_RUR_default_rnd_selfconsum1',
            # 'pvalloc_mini_aggr_RUR_southf_rnd_selfconsum1', 
            # 'pvalloc_mini_aggr_RUR*', 
            'pvalloc_9bfs_RUR_*TEST*',  
        ],
        plot_show                          = True,
        save_plot_by_scen_directory        = False, 
        remove_old_plot_scen_directories   = True,  
        remove_old_plots_in_visualization  = True,  
        remove_old_csvs_in_visualization   = False, 
        
        cut_timeseries_to_zoom_hour        = True,
        add_day_night_HOY_bands            = False,

        default_zoom_hour                  = [212*24, (212*24)+(24*14)], 

        # # # -- def plot_ALL_init_sanitycheck(self, ): --- [run plot,  show plot,  show all scen] ---------
        # plot_ind_var_summary_stats_TF                   = [True,      True,       False],
        # plot_ind_charac_omitted_gwr_TF                  = [True,      True,       False],
        # plot_ind_line_meteo_radiation_TF                = [True,      True,       False],

        plot_ind_mapline_prodHOY_EGIDrfcombo_TF         = [True,      True,       False],                   
        # plot_ind_line_productionHOY_per_EGID_TF         = [True,      True,       False],                   
        # plot_ind_line_productionHOY_per_node_TF         = [True,      True,       False],                   
        # plot_ind_line_PVproduction_TF                   = [True,      True,       False],                   
        # plot_ind_hist_cols_HOYagg_per_EGID_TF           = [True,      True,       False],                   
        plot_ind_map_topo_egid_TF                       = [True,      True,       True],
        # plot_ind_map_topo_egid_incl_gridarea_TF         = [True,      True,       False],

        plot_ind_mapline_prodHOY_EGIDrfcombo_specs = {
                'specific_selected_EGIDs': [
                    # bfs: 2883 - Bretzwil
                    '3030694',      # pv_df
                    '2362100',    # pv_df  2 part
                    # '245052560',    # pv_df 2 part
                    # '3032150',    # pv_df  3 part
                    # '11513725',

                    # 9bfs: all of RUR bfs
                    '101428161', '11513725', '190001512', '190004146', '190024109', '190033245', 
                    '190048248', '190083872', '190109228', '190116571', '190144906', '190178022', 
                    '190183577', '190185552', '190251628', '190251772', '190251828', '190296588', 
                    '190491308', '190694269', '190709629', '190814490', '190912633', '190960689', 
                    '2125434', '2362100', '245044986', '245048760', '245052560', '245053405', 
                    '245057989', '245060059', '3030694', '3030905', '3031761', '3032150', '3033714', 
                    '3075084', '386736', '432600', '432638', '432671', '432683', '432701', '432729', '434178',

                                            ], 
                'show_selected_plots': [
                                            'map',  
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
                'n_rndm_egid_winst_pvdf': 50, 
                'n_rndm_egid_winst_alloc': 0,
                'n_rndm_egid_woinst': 0,
                'n_rndm_egid_outsample': 0, 
                'n_partition_pegid_minmax': (1,2), 
                'roofpartition_color': '#fff2ae',
                'actual_ts_trace_marker': 'cross',
                'tweak_denominator_list': [
                                            1.0, 
                                        ], 
                'summary_cols_only_in_legend': ['EGID', 'inst_TF', 'info_source', 'grid_node',
                                                'GKLAS', 'GAREA', 'are_typ', 'sfhmfh_typ', 
                                                'TotalPower', 'demand_elec_pGAREA', 
                                                'pvtarif_Rp_kWh', 'elecpri_Rp_kWh', ],
                'max_flaeche_pvinst_area_m2': 185,  
                # 'tryout_generic_pvtariff_elecpri_Rp_kWh': (5.0, 25.0),
                'tryout_generic_pvtariff_elecpri_Rp_kWh': (None, None),
                'generic_pvinstcost_coeff_total': {
                                                    'use_generic_instcost_coeff_total_TF': False, 
                                                    'a': 1193, 
                                                    'b': 9340, 
                                                    'c': 0.45,
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
        visual_class.plot_ALL_init_sanitycheck()
        visual_class.plot_ALL_mcalgorithm()

print('done')

