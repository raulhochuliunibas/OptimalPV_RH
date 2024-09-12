import os

# import data_aggregation_MASTER
import pv_allocation_MASTER
import visualization_MASTER

from data_aggregation.default_settings import extend_dataag_scen_with_defaults
from pv_allocation.default_settings import extend_pvalloc_scen_with_defaults
from visualisations.defaults_settings import extend_visual_sett_with_defaults


# SETTINGS DEFINITION ==================================================================================================================
os.chdir('C:/Models/OptimalPV_RH')


# data_aggregation 
datagg_scenarios = {
    'preprep_BSBLSO_18to22':{
        'show_debug_prints': True,
        }, 
}
datagg_scenarios = extend_dataag_scen_with_defaults(datagg_scenarios)


# pv_allocation 
pvalloc_scenarios={
    # BL small sample, 1 y ~ca. 3h 1 scenario
    'pvalloc_smallBL_1y_SLCTN_npv_weighted': {
            'algorithm_specs': {
                'inst_selection_method': 'prob_weighted_npv',
                'rand_seed': None,
                'tweak_constr_capacity_fact': 10,},
            'months_prediction': 12*1,

    },
    'pvalloc_smallBL_3y_SLCTN_random': {
            'algorithm_specs': {
                'inst_selection_method': 'random',
                'rand_seed': None,
                'tweak_constr_capacity_fact': 10,},
            'months_prediction': 12*1,

    },

}

parkplatz = {
    'pvalloc_smallBL_5y_npv_weighted': {
            'name_dir_import': 'preprep_BSBLSO_18to22_20240826_22h',
            'months_prediction': 12*5,
            'algorithm_specs': {
                'inst_selection_method': 'prob_weighted_npv',
                'topo_subdf_partitioner' : 500,},
    },
    'pvalloc_smallBL_5y_random': {
            'name_dir_import': 'preprep_BSBLSO_18to22_20240826_22h',    
            'months_prediction': 12*5,
            'algorithm_specs': {
                'inst_selection_method': 'random',
                'topo_subdf_partitioner' : 500,},
    },

}

pvalloc_scenarios = extend_pvalloc_scen_with_defaults(pvalloc_scenarios)
print(pvalloc_scenarios)


# vsualiastion 
visual_settings = {
        'check_defaults 2' : 'deez nuts',
        'plot_show': True,

        'default_zoom_year': [2012, 2030],
        'default_map_zoom': 11, 
        

        'plot_ind_line_productionHOY_per_node': False,
        'plot_ind_line_installedCap_per_month': False,
        'plot_ind_line_installedCap_per_BFS': False,
        'map_ind_topo_egid': True,
        'map_topo_specs': {
            'uniform_municip_color ': '#fff2ae',
        }
        
    }
visual_settings = extend_visual_sett_with_defaults(visual_settings)




# EXECUTION ==================================================================================================================


# DATA AGGREGATION RUNs  ------------------------------------------------------------------------
for k_sett, scen_sett in datagg_scenarios.items():
    dataagg_settings = scen_sett
    # data_aggregation_MASTER.data_aggregation_MASTER(dataagg_settings)


# ALLOCATION RUNs  ------------------------------------------------------------------------
for k_sett, scen_sett in pvalloc_scenarios.items():
    pvalloc_settings = scen_sett
    # pv_allocation_MASTER.pv_allocation_MASTER(pvalloc_settings)
    

# VISUALISATION RUNs  ------------------------------------------------------------------------
visualization_MASTER.visualization_MASTER(pvalloc_scenarios, visual_settings)



# END ==========================================================================
print(f'{54*"="}\n{5*" "}{10*"*"}{5*" "}END of RUNFILE{5*" "}{10*"*"}{5*" "}\n{54*"="}\n')




