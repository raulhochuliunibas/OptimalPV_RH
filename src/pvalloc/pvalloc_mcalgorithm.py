# python integrated packages
import sys
import os as os 
import glob
import shutil
import json
import copy
import itertools
import time
import joblib

# external packages
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime as datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import polars as pl
from shapely import union_all
from  dataclasses import dataclass, field, asdict
from dataclasses import replace
from typing_extensions import List, Dict
from scipy.optimize import curve_fit
from scipy import optimize
from scipy.stats import pearson3
# from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp



# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, print_to_logfile, checkpoint_to_logfile, get_bfs_from_ktnr, add_static_topo_data_to_subdf
from src.pvalloc.pvallocation_inst_selection_optimization import run_pv_selection_optimization, prepare_feedin_matrix, optimize_pv_selection_greedy , optimize_pv_selection_pulp


# settings
pl.Config.set_tbl_rows(30)  # Adjust 50 to whatever number of rows you want
pd.set_option('display.max_rows', 30) 

from src.pvalloc.pvalloc_core import PVAllocCore


class PVAllocMCAlgo(PVAllocCore):
    """
    Monte Carlo installation algorithm stage of a PV allocation scenario.
    Split out of MAIN_pvallocation.py (formerly PVAllocScenario.run_pvalloc_mcalgorithm).
    """

    def run_pvalloc_mcalgorithm(self,):
        """
        Input: 
            > PVAllocScenario class containing a data class (_Settings) which specifies all scenraio settings (e.g. also path to preprep data directory where all data for the
              executed computations is stored).
    
        Output:
            > within the scenario name defined in pvalloc_settings, the MAIN_pvalloc_MCalgorithm function 
            creates a new directory "MCx" folder directory containing each individual Monte Carlo iteration.

        Description:
            > The MAIN_pvalloc_MCalgorithm function calls the exact same functions as previously used in santiy check of
            pv allocation initializations' sanity check for direct comparison of debugging and testing. 
            > First the script updates the grid premium values for the current month, based on existing installtions and annual radiation. 
            > Then the script updates the NPV values for all houses not yet having a PV installation. 
            > Based on scenario settings, installations are selected from the NPV dataframe until the construction capacity for the given month 
            is reached (or the total capacity for the year; while loop exit criteria).
            
            > This process is repeated for as many Monte Carlo iterations as defined in the scenario settings.
            
        """
        # SETUP -----------------------------------------------------------------------------
        self.sett.log_name = os.path.join(self.sett.name_dir_export_path, 'pvalloc_MCalgo_log.txt')
        self.sett.timing_marks_csv_path = os.path.join(self.sett.name_dir_export_path, 'timing_marks.csv')

        # create log file
        chapter_to_logfile(f'start MAIN_pvalloc_MCalgorithm for : {self.sett.name_dir_export}', self.sett.log_name, overwrite_file=True)
        print_to_logfile('*model allocation specifications*:', self.sett.log_name)
        print_to_logfile(f'> n_bfs_municipalities: {len(self.sett.bfs_numbers)} \n> n_trange_prediction: {self.sett.months_prediction} \n> n_montecarlo_iterations: {self.sett.MCspec_montecarlo_iterations_fordev_sequentially}', self.sett.log_name)
        print_to_logfile(f'> pvalloc_settings, MCalloc_{self.sett.name_dir_export}', self.sett.log_name)
        for k, v in vars(self).items():
            print_to_logfile(f'{k}: {v}', self.sett.log_name)

        start_mc_algo = datetime.datetime.now()
        self.mark_to_timing_csv('MCalgo', 'START_MC_algo', start_mc_algo, np.nan, '-')



        # CREATE MC DIR + TRANSFER INITIAL DATA FILES ----------------------------------------------
        montecarlo_iterations_fordev_seq = [*range(1, self.sett.MCspec_montecarlo_iterations_fordev_sequentially+1, 1)]
        safety_counter_max = self.sett.ALGOspec_while_inst_counter_max
        
        # get all initial files to start a fresh MC iteration
        fresh_initial_paths = [f'{self.sett.name_dir_export_path}/{file}' for file in self.sett.MCspec_fresh_initial_files]
        topo_time_paths = glob.glob(f'{self.sett.name_dir_export_path}/topo_time_subdf/topo_subdf*.parquet')

        max_digits = len(str(max(montecarlo_iterations_fordev_seq)))
        for _ in montecarlo_iterations_fordev_seq:

            # set a random sleep so array tasks in scicore don't overwrite MC directories
            if self.sett.ALGOspec_sleep_bfr_MCiter_TF:
                sleep_range = list(range(10, 1201, 5))
                sleep_time = np.random.choice(sleep_range)
                time.sleep(sleep_time)

            # create additional next MC dir and copy init files
            n_mc_dir = len(glob.glob(f'{self.sett.name_dir_export_path}/zMC*'))
            mc_iter = n_mc_dir + 1 

            start_mc_iter = datetime.datetime.now()
            subchapter_to_logfile(f'START MC{mc_iter:0{max_digits}} iteration', self.sett.log_name)
            self.mark_to_timing_csv('MCalgo', f'start_MC_iter_{mc_iter:0{max_digits}}', start_mc_iter, np.nan, '-')

            # copy all initial files to MC directory
            self.sett.mc_iter_path = os.path.join(self.sett.name_dir_export_path, f'zMC{mc_iter:0{max_digits}}')
            if os.path.exists(self.sett.mc_iter_path):
                shutil.rmtree(self.sett.mc_iter_path)
            os.makedirs(self.sett.mc_iter_path, exist_ok=False)

            fresh_initial_files = [os.path.join(self.sett.name_dir_export_path, file) for file in self.sett.MCspec_fresh_initial_files] 
            for f in fresh_initial_files:
                file_name = os.path.basename(f) 
                shutil.copy(f, os.path.join(self.sett.mc_iter_path, file_name))

            topo_time_paths = glob.glob(os.path.join(self.sett.name_dir_export_path, 'topo_time_subdf', '*.parquet'))
            for f in topo_time_paths:
                file_name = os.path.basename(f) 
                shutil.copy(f, os.path.join(self.sett.mc_iter_path, file_name))
            
            # shutil.rmtree(self.sett.mc_iter_path) if os.path.exists(self.sett.mc_iter_path) else None
            # os.makedirs(self.sett.mc_iter_path, exist_ok=False)
            # for f in fresh_initial_paths + topo_time_paths:
                    # shutil.copy(f, self.sett.mc_iter_path)


            # create grid node monitoring for >1HOY of lost load
            node_1hll_closed_dict = {'k_descrip': 'k: iter_round of algorightm, v: list of nodes that have >1HOY of lost load'}
            with open(f'{self.sett.mc_iter_path}/node_1hll_closed_dict.json', 'w') as f:
                json.dump(node_1hll_closed_dict, f)

            # create grid node monitoring for subsidy
            node_subsidy_monitor_dict = {'k_descrip': 'k: iter_round of algorightm, v: list of nodes that have subsidy applied'}
            with open(f'{self.sett.mc_iter_path}/node_subsidy_monitor_dict.json', 'w') as f:
                json.dump(node_subsidy_monitor_dict, f)


            # ALLOCATION ALGORITHM -----------------------------------------------------------------------------    
            dfuid_installed_list = []
            pred_inst_df = pd.DataFrame()  
            trange_prediction_df = pd.read_parquet(f'{self.sett.mc_iter_path}/trange_prediction.parquet')
            trange_prediction = [m.date() for m in trange_prediction_df['date']]
            constrcapa = pd.read_parquet(f'{self.sett.mc_iter_path}/constrcapa.parquet')

            # Load models once before loop to avoid repeated memory allocation
            if self.sett.ALGOspec_pvinst_size_calculation == 'estim_rfr':
                rfr_model = joblib.load(f'{self.sett.calib_model_coefs}/{self.sett.ALGOspec_calib_estim_mod_name_pkl}_model.pkl')
                encoder   = joblib.load(f'{self.sett.calib_model_coefs}/{self.sett.ALGOspec_calib_estim_mod_name_pkl}_encoder.pkl')
            else:
                rfr_model, encoder = None, None

            for i_m, m in enumerate(trange_prediction):
                i_m = i_m + 1    

                print_to_logfile(f'\n-- n_iter {i_m} -- month {m} -- iter MC{mc_iter:0{max_digits}} -- {self.sett.name_dir_export} --', self.sett.log_name)
                start_allocation_month = datetime.datetime.now()
                with open(f'{self.sett.mc_iter_path}/topo_egid.json', 'r', encoding='utf-8') as f:
                    topo = json.load(f)
                egid_without_pv = [k for k,v in topo.items() if not v['pv_inst']['inst_TF']]


                # GRIDPREM UPDATE ==========
                start_time_update_gridprem = datetime.datetime.now()
                print_to_logfile('- START update gridprem', self.sett.log_name)
                self.algo_update_gridnode_AND_gridprem_POLARS(self.sett.mc_iter_path, i_m, m)
                end_time_update_gridprem = datetime.datetime.now()
                
                print_to_logfile(f'- END update gridprem: {self.timediff_to_str_hhmmss(start_time_update_gridprem, end_time_update_gridprem)} (hh:mm:ss.s)', self.sett.log_name)
                self.mark_to_timing_csv('MCalgo', f'end update_gridprem_{i_m:0{max_digits}}', end_time_update_gridprem, self.timediff_to_str_hhmmss(start_time_update_gridprem, end_time_update_gridprem), '-')  #if i_m < 7 else None

                if len(egid_without_pv) > 0:

                    # check if any EGID w/o PV in open node
                    with open(f'{self.sett.mc_iter_path}/node_1hll_closed_dict.json', 'r', encoding='utf-8') as f:
                        node_1hll_closed_dict = json.load(f)
                    closed_nodes = node_1hll_closed_dict[str(i_m)]['all_nodes_abv_1hll']
                    closed_nodes_egid = [k for k, v in topo.items() if v.get('grid_node')  in closed_nodes ]
                    egid_wo_pv_open_node = [egid for egid in egid_without_pv if egid not in closed_nodes_egid]

                    
                    # NPV_DF UPDATE ==========
                    if len(egid_wo_pv_open_node) > 0:
                                                                                                                            
                        # (only updated each iteration if feedin premium adjusts)
                        if (i_m == 1) or (self.sett.GRIDspec_reupdate_npv_df_TF): 
                            start_time_update_npv = datetime.datetime.now()
                            print_to_logfile('- START update npv', self.sett.log_name)
                            if self.sett.ALGOspec_pvinst_size_calculation == 'estim_rfr':
                                self.algo_update_npv_df_RFR(self.sett.mc_iter_path, i_m, m, rfr_model, encoder)
                            else:
                                break

                            end_time_update_npv = datetime.datetime.now()
                            print_to_logfile(f'- END update npv: {self.timediff_to_str_hhmmss(start_time_update_npv, end_time_update_npv)} (hh:mm:ss.s)', self.sett.log_name)

                            self.mark_to_timing_csv('MCalgo', f'end update_npv_{i_m:0{max_digits}}', end_time_update_npv, self.timediff_to_str_hhmmss(start_time_update_npv, end_time_update_npv), '-')  #if i_m < 7 else None


                # init constr capa + safety_counter ==========
                constr_built_m = 0
                if m.year != (m.year-1):
                    constr_built_y = 0
                constr_capa_m = constrcapa.loc[constrcapa['date'] == str(m), 'constr_capacity_kw'].iloc[0]
                constr_capa_y = constrcapa.loc[constrcapa['year'].isin([m.year]), 'constr_capacity_kw'].sum()

                safety_counter = 0 if len(egid_wo_pv_open_node) > 0 else safety_counter_max


                # INST PICK ==========
                start_time_installation_whileloop = datetime.datetime.now()
                inst_counter = 0
                print_to_logfile('- START inst while loop', self.sett.log_name)

                while( (constr_built_m <= constr_capa_m) & (constr_built_y <= constr_capa_y) & (safety_counter <= safety_counter_max) ):
                    with open(f'{self.sett.mc_iter_path}/topo_egid.json', 'r', encoding='utf-8') as f:
                        topo = json.load(f)
                    with open(f'{self.sett.mc_iter_path}/node_1hll_closed_dict.json', 'r', encoding='utf-8') as f:
                        node_1hll_closed_dict = json.load(f)

                    npv_df = pl.read_parquet(f'{self.sett.mc_iter_path}/npv_df.parquet')

                    #  remove all EGIDs with pv ----------------
                    egid_without_pv = [k for k,v in topo.items() if not v['pv_inst']['inst_TF']]
                    npv_df = npv_df.filter(pl.col('EGID').is_in(egid_without_pv))
                    
                    #  remove all closed nodes EGIDs if applicable ----------------
                    if self.sett.GRIDspec_node_1hll_closed_TF:
                        closed_nodes = node_1hll_closed_dict[str(i_m)]['all_nodes_abv_1hll']
                        closed_nodes_egid = [k for k, v in topo.items() if v.get('grid_node')  in closed_nodes ]

                        npv_df = npv_df.filter(~pl.col('EGID').is_in(closed_nodes_egid)).clone()
                        # npv_df = copy.deepcopy(npv_df.loc[~npv_df['EGID'].isin(closed_nodes_egid)])

                    #  remove all EGIDs with negative NPV ----------------
                    npv_df = npv_df.filter(pl.col('NPV_uid') > 0).clone()

                        
                    npv_df_empty_TF = npv_df.shape[0] == 0
                    if npv_df_empty_TF:
                        safety_counter = safety_counter_max + 1

                    if not npv_df_empty_TF: 
                        inst_counter += 1
                        if self.sett.ALGOspec_pvinst_size_calculation == 'estim_rfr':
                            inst_power = self.algo_select_AND_adjust_topology_RFR(self.sett.mc_iter_path, i_m, m, safety_counter)
                        else:
                            break


                    # Loop Exit + adjust constr_built capacity ----------
                    constr_built_m, constr_built_y, safety_counter  = constr_built_m + inst_power, constr_built_y + inst_power, safety_counter + 1
                    overshoot_rate                                  = self.sett.CSTRspec_constr_capa_overshoot_fact
                    constr_m_TF, constr_y_TF, safety_TF             = constr_built_m > constr_capa_m*overshoot_rate, constr_built_y > constr_capa_y, safety_counter > safety_counter_max

                    # print statements ----------
                    if any([constr_m_TF, constr_y_TF, safety_TF]):
                        print_str = 'exit while loop -> '
                        if constr_m_TF:
                            print_str += f'\n* exceeded constr_limit month (constr_m_TF:{constr_m_TF}), {round(constr_built_m,1)} of {round(constr_capa_m,1)} kW capacity built; '                    
                        if constr_y_TF:
                            print_str += f'\n* exceeded constr_limit year (constr_y_TF:{constr_y_TF}), {round(constr_built_y,1)} of {round(constr_capa_y,1)} kW capacity built; '
                        if safety_TF:
                            if npv_df_empty_TF:
                                print_str += f'\n* exceeded safety counter (safety_TF:{safety_TF}), NO MORE EGID to install PV on; '
                            else:
                                print_str += f'\n* exceeded safety counter (safety_TF:{safety_TF}), {safety_counter} rounds for safety counter max of: {safety_counter_max}; '
                        checkpoint_to_logfile(print_str, self.sett.log_name, 0, True)

                               
                end_time_installation_whileloop = datetime.datetime.now()
                checkpoint_to_logfile(f'{inst_counter} installations installed', self.sett.log_name, 0, True)
                print_to_logfile(f'- END inst while loop: {self.timediff_to_str_hhmmss(start_time_installation_whileloop, end_time_installation_whileloop)} (hh:mm:ss.s)', self.sett.log_name)
                self.mark_to_timing_csv('MCalgo', f'end inst_whileloop_{i_m:0{max_digits}}', end_time_installation_whileloop, self.timediff_to_str_hhmmss(start_time_installation_whileloop, end_time_installation_whileloop),  '-')  #if i_m < 7 else None
                                            
                checkpoint_to_logfile(f'end month allocation, runtime: {datetime.datetime.now() - start_allocation_month} (hh:mm:ss.s)', self.sett.log_name, 0, True)                    
                                                                           

            # CLEAN UP interim files of MC run ==========
            # topo_time_subdf
            files_to_remove_paths =  glob.glob(f'{self.sett.mc_iter_path}/topo_subdf_*.parquet')
            for f in files_to_remove_paths:
                os.remove(f)
            
            # all CSVs in pred_gridprem_node_by_M
            gridnode_gridprem_by_M_paths = glob.glob(f'{self.sett.mc_iter_path}/pred_gridprem_node_by_M/*.csv')
            for path in files_to_remove_paths:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        print(f'Failed to remove obsolete CSV file in: {path}. Error: {e}')
            

            mc_iter_time = datetime.datetime.now() - start_mc_iter
            subchapter_to_logfile(f'END MC{mc_iter:0{max_digits}}, runtime: {mc_iter_time} (hh:mm:ss.s)', self.sett.log_name)
            end_mc_iter = datetime.datetime.now()
            self.mark_to_timing_csv('MCalgo', f'end_MC_iter_{mc_iter:0{max_digits}}', end_mc_iter, self.timediff_to_str_hhmmss(start_mc_iter, end_mc_iter),  '-')
            print_to_logfile('\n', self.sett.log_name)

        
        # END ---------------------------------------------------
        end_mc_algo = datetime.datetime.now()
        self.mark_to_timing_csv('MCalgo', 'END_MC_algo', end_mc_algo, self.timediff_to_str_hhmmss(start_mc_algo, end_mc_algo),  '-')

        if 'start_total_runtime' in dir(self.sett): 
            self.mark_to_timing_csv('TOTAL', 'END_total_runtime', end_mc_algo, self.timediff_to_str_hhmmss(self.sett.start_total_runtime, end_mc_algo),  '-')
            os.rename(self. sett.timing_marks_csv_path, f'{self.sett.timing_marks_csv_path.split(".cs")[0]}_{self.sett.name_dir_export}.csv')

        chapter_to_logfile(f'end MAIN_pvalloc_MCalgorithm\n Runtime (hh:mm:ss):{datetime.datetime.now() - start_mc_algo}', self.sett.log_name, overwrite_file=False)

