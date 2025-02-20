import sys
import pandas as pd

sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile


# ------------------------------------------------------------------------------------------------------
# GET ELECTRICITY PRICES FROM FORMER API IMPORT 
# ------------------------------------------------------------------------------------------------------
def get_elecpri_data_earlier_api_import(dataagg_settings_def):

    # import settings + setup -------------------
    data_path_def = dataagg_settings_def['data_path']
    preprep_path_def = dataagg_settings_def['preprep_path']
    log_name = dataagg_settings_def['log_file_name']
    print_to_logfile(f'run function: get_elecpri_data_earlier_api_import', log_name)

    # import + subset data -------------------
    elecpri_all = pd.read_parquet(f'{data_path_def}/input/ElCom_consum_price_api_data/elecpri.parquet')
    elecpri = elecpri_all.loc[elecpri_all['bfs_number'].isin(dataagg_settings_def['bfs_numbers'])]
    
    # export -------------------
    checkpoint_to_logfile(f'export elecpri of local data from former api import', log_name)
    elecpri.to_parquet(f'{preprep_path_def}/elecpri.parquet')
    elecpri.to_csv(f'{preprep_path_def}/elecpri.csv', index=False)
    

