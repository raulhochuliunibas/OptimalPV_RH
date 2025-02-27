import sys
import pandas as pd

sys.path.append('..')
from auxiliary_functions import checkpoint_to_logfile, print_to_logfile


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
    elecpri_all = pd.read_parquet(f'{scen.data_path}/input_api/elecpri.parquet')
    elecpri = elecpri_all.loc[elecpri_all['bfs_number'].isin(scen.bfs_numbers)]

    # EXPORT --------------------------------------
    checkpoint_to_logfile('export elecpri of local data from former api import', scen.log_name)
    elecpri.to_parquet(f'{scen.data_path}/input_split_data_geometry/elecpri.parquet')
    

