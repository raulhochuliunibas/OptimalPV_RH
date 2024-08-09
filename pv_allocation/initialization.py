import sys
import os as os
import numpy as np
import pandas as pd
import json

from pyarrow.parquet import ParquetFile
import pyarrow as pa

# own functions 
sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile


# ------------------------------------------------------------------------------------------------------
# IMPORT PREPREPED DATA & CREATE TOPOLOGY
# ------------------------------------------------------------------------------------------------------

def import_prepre_AND_create_topology(
        pvalloc_settings, ):
    """ 
    Reimport all the data that is intermediatley stored in the prepreped folder. Transform it such that it can be used throughout the
    PV allocation algorithm. Return all the data objects back to the main code file. 
    - Input: PV allocation settings
    - Output: all the data objects that are needed for later calculations
    """

    # import settings + setup -------------------
    script_run_on_server_def = pvalloc_settings['script_run_on_server']
    name_dir_import_def = pvalloc_settings['name_dir_import']
    fast_debug_def = pvalloc_settings['fast_debug_run']
    show_debug_prints_def = pvalloc_settings['show_debug_prints']
    wd_path_def = pvalloc_settings['wd_path']
    data_path_def = pvalloc_settings['data_path']
    log_file_name_def = pvalloc_settings['log_file_name']
    print_to_logfile('run function: import_prepreped_data', log_file_name_def)


    # IMPORT & TRANSFORM ============================================================================
    # Import all necessary data objects from prepreped folder and transform them for later calculations
    print_to_logfile('import & transform data', log_file_name_def)

    # transformation functions --------
    def import_large_or_small_pq_file(path, batchsize, fast_run):
        # import a parquet file, either in full or only small batch for faster developping
        if not fast_run: 
            df = pd.read_parquet(path)
        elif fast_run:
            pf = ParquetFile(path)
            first_n_rows = next(pf.iter_batches(batch_size = batchsize))
            df = first_n_rows.to_pandas()
        return df
        

    # GWR 
    gwr = import_large_or_small_pq_file(f'{data_path_def}/output/{name_dir_import_def}/gwr.parquet', 100, fast_debug_def)

    # SOLKAT 
    solkat = import_large_or_small_pq_file(f'{data_path_def}/output/{name_dir_import_def}/solkat.parquet', 100, fast_debug_def)

    # PV 
    pv = import_large_or_small_pq_file(f'{data_path_def}/output/{name_dir_import_def}/pv.parquet', 100, fast_debug_def)

    # Map solkat_egid > pv 
    Map_solkategid_pv = import_large_or_small_pq_file(f'{data_path_def}/output/{name_dir_import_def}/Map_solkategid_pv.parquet', 100, fast_debug_def)
    
    Map_solkategid_pv = Map_solkategid_pv.dropna()
    Map_solkategid_pv['EGID'] = Map_solkategid_pv['EGID'].astype(int).astype(str)
    Map_solkategid_pv['xtf_id'] = Map_solkategid_pv['xtf_id'].fillna(0).astype(int).astype(str)


    # CREATE TOPOLOGY ============================================================================
    print_to_logfile('create topology', log_file_name_def)
    topo = {'EGID': {}}

    len(gwr['EGID'])
    gwr = gwr.head(10)

    # add EGID specific data
    def populate_topo_byEGID(egid):
        installation = 1 if (Map_solkategid_pv.loc[Map_solkategid_pv['EGID'] == egid, 'xtf_id'] != 0).any() else 0
        xtf_id_series = Map_solkategid_pv.loc[Map_solkategid_pv['EGID'] == egid, 'xtf_id']
        xtf_id = xtf_id_series.iloc[0] if not xtf_id_series.empty else None

        topo['EGID'][egid] = {
            'pv': {
                'installation': installation,
                'xtf_id': xtf_id,
            }
        }
    gwr['EGID'].apply(populate_topo_byEGID)


    # export topo to txt file
    with open(f'{data_path_def}/output/pvalloc_run/topo.txt', 'w') as f:
        f.write(str(topo))
    
    with open(f'{data_path_def}/output/pvalloc_run/topo.json', 'w') as f:
        json.dump(topo, f)


    print('asdf')    




