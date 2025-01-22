postprocess_default_settings = {
    'asdf': 1, 
}

def extend_postprocess_scen_with_defaults(sett_dict, defaults=postprocess_default_settings):
    default_dict = defaults.copy()

    for sett_name, sett_val in sett_dict.items():
        if not isinstance(sett_val, dict):
            default_dict[sett_name] = sett_val

        elif isinstance(sett_val, dict) and sett_name in default_dict.keys():
            for k_sett, v_sett in sett_val.items():
                default_dict[sett_name][k_sett] = v_sett  

    return default_dict


def get_default_postprocess_analysis_settings():
    return postprocess_default_settings