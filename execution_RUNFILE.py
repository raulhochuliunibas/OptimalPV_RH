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
    'pvalloc_smallBL_6y_SLCTN_npv_weighted': {
            'algorithm_specs': {
                'inst_selection_method': 'prob_weighted_npv',
                'rand_seed': None,
                'tweak_constr_capacity_fact': 5,},
            'months_prediction': 12*6,

    },
    'pvalloc_smallBL_6y_SLCTN_random': {
            'algorithm_specs': {
                'inst_selection_method': 'random',
                'rand_seed': None,
                'tweak_constr_capacity_fact': 5,},
            'months_prediction': 12*6,

    },

}

parkplatz = {
    'pvalloc_smallBL_5y_SLCTN_npv_weighted': {
            'algorithm_specs': {
                'inst_selection_method': 'prob_weighted_npv',
                'rand_seed': None,
                'tweak_constr_capacity_fact': 20,},
            'months_prediction': 12*5,

    },
    'pvalloc_smallBL_5y_SLCTN_random': {
            'algorithm_specs': {
                'inst_selection_method': 'random',
                'rand_seed': None,
                'tweak_constr_capacity_fact': 20,},
            'months_prediction': 12*5,

    },

        'pvalloc_smallBL_3m_npv_weighted': {
            'algorithm_specs': {
                'inst_selection_method': 'prob_weighted_npv',
                'rand_seed': None,
                'tweak_constr_capacity_fact': 20,},
            'months_prediction': 3,

    },
    'pvalloc_smallBL_3m_random': {
            'algorithm_specs': {
                'inst_selection_method': 'random',
                'rand_seed': None,
                'tweak_constr_capacity_fact': 20,},
            'months_prediction': 3,

    },

}

pvalloc_scenarios = extend_pvalloc_scen_with_defaults(pvalloc_scenarios)
print(pvalloc_scenarios)


# vsualiastion 
visual_settings = {
        'plot_show': True,

        'plot_ind_line_productionHOY_per_node': True,
        'plot_ind_line_installedCap_per_month': True,
        'plot_ind_line_installedCap_per_BFS': False,
        'map_ind_topo_egid': True,

        'plot_agg_line_installedCap_per_month': True,
        'plot_agg_line_productionHOY_per_node': True,
        'plot_agg_line_gridPremiumHOY_per_node': True,

        # single plots (just show once, not for all scenarios)
        'map_ind_production': False, # NOT WORKING YET

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
print(f'\n\n{54*"="}\n{5*" "}{10*"*"}{5*" "}END of RUNFILE{5*" "}{10*"*"}{5*" "}\n{54*"="}\n')




