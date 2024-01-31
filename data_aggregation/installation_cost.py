import sys
import os
import pandas as pd
import numpy as np

sys.path.append('..')
from functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile

# ------------------------------------------------------------------------------------------------------
# INVESTMENT COSTS
# ------------------------------------------------------------------------------------------------------

# COST DATA FRAME --------------------------------------------
def attach_pv_cost(
        script_run_on_server_def = None,
        recreate_parquet_files_def = None,
        smaller_import_def = None,
        log_file_name_def = None,
        wd_path_def = None,
        data_path_def = None,
        show_debug_prints_def = None,
        ):        
    """
    Function to create assumed cost df
    Source: https://www.energieschweiz.ch/tools/solarrechner/
    """ 


    # setup -------------------
    wd_path = "D:/RaulHochuli_inuse/Models/OptimalPV_RH" if script_run_on_server_def else "C:/Models/OptimalPV_RH"   # path for private computer
    data_path = f'{wd_path}_data'

    # create directory + log file
    if not os.path.exists(f'{data_path}/output/preprep_data'):
        os.makedirs(f'{data_path}/output/preprep_data')
    checkpoint_to_logfile('run function: create_cost_df_and_func.py', log_file_name_def=log_file_name_def, n_tabs_def = 5) 
    print_to_logfile(f' > assuming cost of energieschweiz.ch/tools/solarrechner/; January 2024', log_file_name_def= log_file_name_def)


    # create cost df -------------------
    installation_cost_dict = {
        "on_roof_installation_cost_pkW": {
            2: 4636,
            3: 3984,
            5: 3373,
            10: 2735,
            15: 2420,
            20: 2219,
            30: 1967,
            50: 1710,
            75: 1552,
            100: 1463,
            125: 1406,
            150: 1365
        },}
    
    installation_cost_list = [(k, v) for k, v in installation_cost_dict["on_roof_installation_cost_pkW"].items()]
    installation_cost_df = pd.DataFrame(installation_cost_list, columns=['kw', 'chf_pkW'])

    def Map_kw_pvcost(kw_def):
        """
        Cost Assumption set in def pv_cost_df_and_func():!
        Function to return the cost (CHF) of a pv system (input kW)  
        """
        return np.interp(kw_def, installation_cost_df['kw'], installation_cost_df['chf_pkW'])


    # export -------------------
    installation_cost_df.to_parquet(f'{data_path}/output/preprep_data/pvinstcost.parquet')


    # attach cost per roof partition -------------------
    solkat = pd.read_parquet(f'{data_path}/output/preprep_data/solkat_by_gm.parquet')
    solkat['GSTRAH_kw'] = solkat['GSTRAHLUNG'] / (365*24)
    solkat['cost_tot_kw_chf'] = solkat['GSTRAH_kw'].apply(Map_kw_pvcost)


    # export -------------------
    solkat.to_parquet(f'{data_path}/output/preprep_data/solkat_w_cost_by_gm.parquet')


    return Map_kw_pvcost


    
    


