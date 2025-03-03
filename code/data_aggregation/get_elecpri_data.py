import os as os
import sys
import pandas as pd

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from auxiliary.auxiliary_functions import checkpoint_to_logfile, print_to_logfile, get_bfs_from_ktnr



# ------------------------------------------------------------------------------------------------------
# GET ELECTRICITY PRICES FROM FORMER API IMPORT 
# ------------------------------------------------------------------------------------------------------
def get_elecpri_data_earlier_api_import(scen,):
    """
    Get electricity prices from former api import
    """

    # SETUP --------------------------------------
    print_to_logfile(f'run function: get_elecpri_data_earlier_api_import', scen.log_name)

    # IMPORT + SUBSET DATA --------------------------------------
    # elecpri_all = pd.read_parquet(f'{scen.data_path}/input_api/elecpri.parquet')
    elecpri_all = pd.read_parquet(f'{scen.data_path}/input/ElCom_consum_price_api_data/elecpri.parquet')
    elecpri = elecpri_all.loc[elecpri_all['bfs_number'].isin(scen.bfs_numbers)]

    # EXPORT --------------------------------------
    checkpoint_to_logfile('export elecpri of local data from former api import', scen.log_name)
    elecpri.to_parquet(f'{scen.preprep_path}/elecpri.parquet')
    

