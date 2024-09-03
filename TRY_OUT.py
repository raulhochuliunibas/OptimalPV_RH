import os
import winsound

visual_settings = {
    'wd_path_laptop': 'C:/Models/OptimalPV_RH',

    'general_specs': {
    }, 
    'asdf': {
        'plot_show': True, 
        'plot_save': True,
        'import_path': 'C:\Models\OptimalPV_RH_data\output\pvalloc_BSBLSO_wrkn_prgrss_20240826_9h_BREAK',
        'preprep_path': 'C:/Models/OptimalPV_RH_data/output/preprep_BSBLSO_15to23_20240821_02h',
        'title': 'TOPO interim development',

    }, 
    'fdsa': {
        'plot_show': True, 
        'plot_save': True,
        'import_path': 'preprep_BSBLSO_18to22_20240826_22h',
        'preprep_path': '',
        'title': 'TOPO interim development',

    },
}

wd_path = f'C:/Models/OptimalPV_RH'


# RUN TEST: -----------------------------------------------------------
with open(f'{wd_path}/visualizations_MASTER.py', 'r') as f:
    script_code = f.read()

exec(script_code)


print('end)')
winsound.Beep(2000, 100)
winsound.Beep(2000, 100)

