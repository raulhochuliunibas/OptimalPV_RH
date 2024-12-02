visual_default_settings = {
        'plot_show': True,
        'remove_previous_plots': False,
        'remove_old_plot_scen_directories': False,
        'MC_subdir_for_plot': '*MC*1',

        'default_zoom_year': [2012, 2030],
        'default_zoom_hour': [2400, 2400+(24*7)],
        'default_map_zoom': 11, 
        'default_map_center': [47.48, 7.57],
        'node_selection_for_plots': ['node1', 'node3', 'node5'], # or None for all nodes

        # PLOT CHUCK -------------------------> [run plot,  show plot]
        # for pvalloc_inital + sanitycheck
        'plot_ind_var_summary_stats':           [True,      True],
        'plot_ind_hist_pvcapaprod_sanitycheck': [True,      True],
        'plot_ind_charac_omitted_gwr':          [True,      True],
            'plot_ind_charac_omitted_gwr_specs':{
                'disc_cols': ['BFS_NUMMER','GSTAT','GKAT','GKLAS'], 
                'disc_ncols': 2, 
                'disc_figsize': [15, 10],
                'cont_cols': ['GBAUJ','GBAUM','GAREA','GEBF','WAZIM','WAREA'],
                'cont_ncols': 3,
                'cont_figsize': [15, 10],
                'cont_bins': 20,
            },
        'plot_ind_line_meteo_radiation':        [True,      True],

        # for pvalloc_MC_algorithm 
        'plot_ind_line_installedCap':            [True,      True],
        'plot_ind_line_productionHOY_per_node':  [True,      True],
        'plot_ind_hist_NPV_freepartitions':      [True,      True],
        'plot_ind_hist_pvcapaprod':              [True,    True],


        'plot_ind_map_topo_egid':                [True,    False],
            'plot_ind_map_topo_egid_specs': {
                'uniform_municip_color': '#fff2ae',
                'shape_opacity': 0.2,
                'point_opacity': 0.7,
                'point_opacity_sanity_check': 0.4,
                'point_size_pv': 6,
                'point_size_rest': 4.5,
                'point_size_sanity_check': 20,
                'point_color_pv_df': '#54f533',      # green
                'point_color_solkat': '#f06a1d',     # turquoise
                'point_color_alloc_algo': '#ffa600', # yellow 
                'point_color_rest': '#383838',       # dark grey
                'point_color_sanity_check': '#0041c2', # blue
            }, 
        'plot_ind_map_node_connections':         [True,    False],
            'plot_ind_map_node_connections_specs': {
                'uniform_municip_color': '#fff2ae',
                'shape_opacity': 0.2,   
                'point_opacity_all': 0.5,
                'point_size_all': 4,
                'point_opacity_bynode': 0.7,
                'point_size_bynode': 6,
                'point_color_all': '#383838',       # dark grey
                'point_color_palette': 'Turbo',
                'point_size_dsonode_loc': 15,
                'point_opacity_dsonode_loc': 1
            },


        
        'plot_ind_map_topo_omitt':              True,
            'plot_ind_map_topo_omitt_specs':{
                'point_opacity': 0.6, 
                'point_size': 7.5, 
                'point_color': '#f54242',           # red
            },


        'plot_agg_line_installedCap_per_month':  True,
        'plot_agg_line_productionHOY_per_node':  True,
        'plot_agg_line_gridPremiumHOY_per_node': True,
        'plot_agg_line_gridpremium_structure':   True,
        'plot_agg_line_production_per_month':    True,


        'plot_agg_line_cont_charact_new_inst':   True,
            'plot_agg_line_cont_charact_new_inst_specs': {
                'colnames_cont_charact_installations': ['pv_tarif_Rp_kWh', 'elecpri_Rp_kWh','selfconsum_kW','FLAECHE', 'netdemand_kW', 'estim_pvinstcost_chf', 'TotalPower']},

        # single plots (just show once, not for all scenarios)
        'map_ind_production': False, # NOT WORKING YET

        
    }

def extend_visual_sett_with_defaults(sett_dict, defaults=visual_default_settings):
    default_dict = defaults.copy()

    for sett_name, sett_val in sett_dict.items():
        if not isinstance(sett_val, dict):
            default_dict[sett_name] = sett_val

        elif isinstance(sett_val, dict) and sett_name in default_dict.keys():
            for k_sett, v_sett in sett_val.items():
                default_dict[sett_name][k_sett] = v_sett  

    return default_dict


def get_default_visual_settings():
    return visual_default_settings

