visual_default_settings = {
        'plot_show': True,

        'default_zoom_year': [2012, 2030],
        'default_zoom_hour': [2400, 2400+(24*7)],
        'default_map_zoom': 11, 
    
        'plot_ind_line_productionHOY_per_node': True,
        'plot_ind_line_installedCap_per_month': True,
        'plot_ind_line_installedCap_per_BFS': False,
        'map_ind_topo_egid': False,

        'plot_agg_line_installedCap_per_month': True,
        'plot_agg_line_productionHOY_per_node': True,
        'plot_agg_line_gridPremiumHOY_per_node': True,

        # single plots (just show once, not for all scenarios)
        'map_ind_production': False, # NOT WORKING YET

        'map_topo_specs': {
            'uniform_municip_color': '#fff2ae',
            'shape_opacity': 0.2,
            'point_opacity': 0.7,
            'point_size_pv': 6,
            'point_size_rest': 4.5,
            'point_color_pv_df': '#54f533',      # green
            'point_color_alloc_algo': '#ffa600', # yellow 
            'point_color_rest': '#383838',       # dark grey

        },         
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

