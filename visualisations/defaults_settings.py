visual_default_settings = {
        'plot_show': True,

        'default_zoom_year': [2012, 2030],
        'default_map_zoom': 11, 
        
        'plot_ind_line_productionHOY_per_node': False,
        'plot_ind_line_installedCap_per_month': False,
        'plot_ind_line_installedCap_per_BFS': False,
        'map_ind_topo_egid': True,

    }

def extend_visual_sett_with_defaults(sett_dict, defaults=visual_default_settings):
    default_dict = defaults.copy()
    sett_dict_return = {}
    # scen_dicts = {}

    for sett_name, sett_val in sett_dict.items():
        if not isinstance(sett_val, dict):
            default_dict[sett_name] = sett_val

        elif isinstance(sett_val, dict) and sett_name in default_dict.keys():
            for k_sett, v_sett in sett_val.items():
                default_dict[sett_name][k_sett] = v_sett

    sett_dict_return[sett_name] = default_dict  

    return sett_dict_return 
def get_default_visual_settings():
    return visual_default_settings

