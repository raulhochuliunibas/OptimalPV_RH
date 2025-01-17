visual_default_settings = {
        'plot_show': True,
        'remove_previous_plots': False,
        'remove_old_plot_scen_directories': False,
        'MC_subdir_for_plot': '*MC*1',
        'mc_plots_individual_traces': True,

        'default_zoom_year': [2012, 2030],
        'default_zoom_hour': [2400, 2400+(24*7)],
        'default_map_zoom': 11, 
        'default_map_center': [47.48, 7.57],
        'node_selection_for_plots': ['1', '3', '5'],


        # PLOT CHUCK -------------------------> [run plot,  show plot,  show all scen]
      
        # for pvalloc_inital + sanitycheck
        'plot_ind_var_summary_stats':               [False,     True,       False],
        'plot_ind_hist_pvcapaprod_sanitycheck':     [False,     True,       False],
            'plot_ind_hist_pvcapaprod_sanitycheck_specs': {
                'xbins_hist_instcapa_abs': 0.5,
                'xbins_hist_instcapa_stand': 0.1,
                'xbins_hist_totalprodkwh_abs': 500, 
                'xbins_hist_totalprodkwh_stand': 0.05,
                'trace_color_palettes': ['Turbo', 'Viridis', 'Aggrnyl', 'Agsunset'],    #  ['Blues', 'Greens', 'Reds', 'Oranges', 'Purples', 'Greys', 'Mint', 'solar', 'Teal', 'Magenta', 'Plotly3', 'Viridis', 'Turbo', 'Blackbody']
                'trace_colval_max': 60,                            # max value for color scale; the higher the max value and the lower the increments, the more colors will be picked within the same color range of the palette
                'trace_colincr': 10,                                # increment for color scale
                'uniform_scencolor_and_KDE_TF': True,
                'export_spatial_data_for_prod0': True, 
                },
        'plot_ind_hist_radiation_rng_sanitycheck':  [False,     True,       False],
        'plot_ind_hist_pvprod_deviation':           [False,     True,       False],
        'plot_ind_charac_omitted_gwr':              [False,     True,       False],
            'plot_ind_charac_omitted_gwr_specs':{
                'disc_cols': ['BFS_NUMMER','GSTAT','GKAT','GKLAS'], 
                'disc_ncols': 2, 
                'disc_figsize': [15, 10],
                'cont_cols': ['GBAUJ','GBAUM','GAREA','GEBF','WAZIM','WAREA'],
                'cont_ncols': 3,
                'cont_figsize': [15, 10],
                'cont_bins': 20,
                'gwr_code_name_tuples_GKLAS': [
                    ('1110', 'Bldg. w one flat (incl double, row houses, w indiv roofs)'),
                    ('1121', 'Bldg. w two flat (incl double, row houses, w 2 flats'),
                    ('1276', 'Bldg. for animal shelter'), ],
                'gwr_code_name_tuples_GSTAT': [
                    ('1004', 'Existing bldg.'),]
            },
        'plot_ind_line_meteo_radiation':            [False,     True,       False],

        # for pvalloc_MC_algorithm 
        'plot_ind_line_installedCap':               [False,    True,       False],
        'plot_ind_line_PVproduction':               [False,    True,       False],
        'plot_ind_line_productionHOY_per_node':     [False,    True,       False],
        'plot_ind_line_gridPremiumHOY_per_node':    [False,    True,       False],
        'plot_ind_line_gridPremium_structure':      [False,    True,       False],
        'plot_ind_hist_NPV_freepartitions':         [False,    True,       False],
        'plot_ind_hist_pvcapaprod':                 [False,    True,       False],
        'plot_ind_map_topo_egid':                   [False,    False,      False],
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
        'plot_ind_map_node_connections':            [False,    False,      False],
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
        'plot_ind_map_omitted_egids':               [False,    True,       False],
            'plot_ind_map_omitted_egids_specs': {
                'point_opacity': 0.7,
                'point_size_select_but_omitted': 10,
                'point_size_rest_not_selected': 1, # 4.5,
                'point_color_select_but_omitted': '#ed4242', # red
                'point_color_rest_not_selected': '#ff78ef',  # pink
                'export_gdfs_to_shp': True, 
            }, 
        'plot_ind_lineband_contcharact_newinst':    [False,    True,       False],
            'plot_ind_line_contcharact_newinst_specs': {
                'trace_color_palette': 'Turbo',
                'upper_lower_bound_interval': [0.05, 0.95],
                'colnames_cont_charact_installations_AND_numerator': 
                [('pv_tarif_Rp_kWh',        1), 
                 ('elecpri_Rp_kWh',         1),
                 ('selfconsum_kW',          1),
                 ('FLAECHE',                1), 
                 ('netdemand_kW',           1000), 
                 ('estim_pvinstcost_chf',   1000),
                 ('TotalPower',             1),
                ], 
                },

        # for aggregated MC_algorithms
        'plot_mc_line_PVproduction':                [False,    True,       False],

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

