# python integrated packages
import sys
import os as os 
import glob
import shutil
import json
import copy
import itertools
import time
import uuid
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

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.MAIN_pvallocation_Class import PVAllocScenario_Settings


class PVAllocCore:
    """
    Shared base class for the PV allocation pipeline stages.
    Holds the scenario settings plus the algorithm and logging/timing utility
    methods that are called from two or more of PVAllocInitialization,
    PVAllocMCAlgo and PVAllocGridOpt. Split out of MAIN_pvallocation.py.
    """

    def __init__(self, settings: "PVAllocScenario_Settings"):
        self.sett = settings

        self.sett.wd_path = os.getcwd()
        self.sett.data_path                = os.path.join(self.sett.wd_path, 'data')
        self.sett.pvalloc_path             = os.path.join(self.sett.data_path, 'pvalloc', 'pvalloc_scen__temp_to_be_renamed')
        self.sett.name_dir_export_path     = os.path.join(self.sett.data_path, 'pvalloc', self.sett.name_dir_export)
        self.sett.name_dir_import_path     = os.path.join(self.sett.data_path, 'preprep', self.sett.name_dir_import)
        self.sett.sanity_check_path        = os.path.join(self.sett.name_dir_export_path, 'sanity_check_byEGID')
        self.sett.calib_model_coefs        = os.path.join(self.sett.data_path, 'calibration', self.sett.ALGOspec_calib_estim_dir_name)



    # ------------------------------------------------------------------------------------------------------
    # AUXILIARY (used by every stage)
    # ------------------------------------------------------------------------------------------------------
    def export_pvalloc_scen_settings(self):
            """
            Input:
                PVAllocScenario including the PVAllocScenario_Settings dataclass containing all scenarios settings. 
            (Output:)
                > Exports a JSON dict containing elements of the data class PVAllocScenario_Settings for pvallocation initialization.
                (all settings for the scenario).
            """
            sett_dict = asdict(self.sett)

            with open(f'{self.sett.name_dir_export_path}/pvalloc_sett.json', 'w') as f:
                json.dump(sett_dict, f, indent=4)




    def atomic_export(self, final_path: str, write_fn) -> None:
        """
        Input:
            > final_path: path of a file that parallel scenario runs (e.g. HPC array tasks) SHARE, e.g. in the preprep folder.
            > write_fn:   function taking one path argument and writing the complete file to it, e.g. lambda p: df.to_parquet(p).

        Output:
            > final_path contains the complete new file. Nobody ever reads a half-written (or empty) file, because
              write_fn writes to a private temp file in the SAME folder (same file system) which is then swapped in
              with os.replace (an atomic rename: readers see either the old or the new complete file, never something in between).
              If writing fails (or on Ctrl-C) the temp file is removed; after a hard kill (SIGTERM, SLURM timeout) a stray *.tmp_<id> file may remain. final_path is never left truncated.
            > (Windows only) os.replace fails with PermissionError while another process (a reader, a virus scanner) holds one of the two
              files open, so it is retried briefly. On Linux (HPC) this never triggers: renaming over a file that is open elsewhere is allowed.
        """
        tmp_path = f'{final_path}.tmp_{uuid.uuid4().hex}'     # random name -> no two tasks (also on different nodes) ever share it
        try:
            write_fn(tmp_path)
            for i_try in range(5):
                try:
                    os.replace(tmp_path, final_path)
                    break
                except PermissionError:
                    if i_try == 4:
                        raise
                    time.sleep(0.2)
        except BaseException:                                  # also Ctrl-C (KeyboardInterrupt) and sys.exit (SystemExit), not just normal errors
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise




    def mark_to_timing_csv(self, step = None, substep = None,timestamp = None, runtime = None, descr = None):
        """
        Input:
            > PVAllocScenario class containing a data class (_Settings) which specifies all scenraio settings (e.g. also path to preprep data directory where all data for the
              executed computations is stored).
        Tasks: 
            > if not existing, create a timing csv file in the export directory of the scenario.
            > append the current date and time to the csv file from a given timestamp

        Output:
            > csv file in the export directory of the scenario containing the date and time of the last run of the script.           
        """
        # create timing file if not existing
        if not os.path.exists(self.sett.timing_marks_csv_path):
            df = pd.DataFrame({
                'step': [step,],
                'substep': [substep,],
                'timestamp': [timestamp,], 
                'runtime': [runtime,], 
                'descr': [descr,]
            })
            df.to_csv(self.sett.timing_marks_csv_path, index=False)
        
        elif os.path.exists(self.sett.timing_marks_csv_path):
            df = pd.read_csv(self.sett.timing_marks_csv_path)   

            new_row = {
                'step': step,
                'substep': substep,
                'timestamp': timestamp, 
                'runtime': runtime, 
                'descr': descr
            }
            
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(self.sett.timing_marks_csv_path, index=False)




    def timediff_to_str_hhmmss(self, start_time, end_time):
        """
        Input:
            > start_time: datetime object of the start time of a process
            > end_time: datetime object of the end time of a process

        Output:
            > formatted string of the time difference between start and end time in hh:mm:ss.s format.
        """
        timediff = end_time - start_time
        total_seconds = int(timediff.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds // 60) % 60
        seconds = total_seconds % 60
        milliseconds = timediff.microseconds // 1000
        formatted_time = f"{hours:2}:{minutes:02}:{seconds:02}.{milliseconds:03} (hh:mm:ss.s)"

        return formatted_time




    # ------------------------------------------------------------------------------------------------------
    # SHARED ALGORITHM METHODS (called from Initialization, MCAlgo and/or GridOpt)
    # ------------------------------------------------------------------------------------------------------
    def get_topo_filter_tag_mapping(self, topo_func):
        # get topo filter df
        topo_filter_list = []
        for k,v in topo_func.items():
            for k_dfuid, v_dfuid in v['solkat_partitions'].items():
                row = {
                    'EGID': k,
                    'df_uid': k_dfuid, 
                    'FLAECHE': v_dfuid['FLAECHE'],
                    'AUSRICHTUNG': v_dfuid['AUSRICHTUNG'],
                    'NEIGUNG': v_dfuid['NEIGUNG'],
                }
                topo_filter_list.append(row)
        
        topo_filter = pl.DataFrame(topo_filter_list)
        
        # get flaeche ratios
        flaeche_by_egid = topo_filter.group_by('EGID').agg([
            pl.col('FLAECHE').sum().alias('total_flaeche_by_egid')
        ])  
        topo_filter = topo_filter.join(flaeche_by_egid, on='EGID', how='left')

        topo_filter = topo_filter.with_columns([
            (pl.col('FLAECHE') / pl.col('total_flaeche_by_egid')).alias('FLAECHE_ratio')
        ])

        # get dfuid specific filters -----
        topo_filter = topo_filter.with_columns([
            # EAST WEST
            pl.when(
                (pl.col('FLAECHE_ratio') >= 0.5 ) & (pl.col('AUSRICHTUNG') > -135) & (pl.col('AUSRICHTUNG') < -45)
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias('filt_dfuid_east_50pr'),
            pl.when(
                (pl.col('FLAECHE_ratio') >= 0.5 ) & (pl.col('AUSRICHTUNG') > 45) & (pl.col('AUSRICHTUNG') < 135)
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias('filt_dfuid_west_50pr'),

            pl.when(
                (pl.col('FLAECHE_ratio') >= 0.4 ) & (pl.col('AUSRICHTUNG') > -135) & (pl.col('AUSRICHTUNG') < -45)
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias('filt_dfuid_east_40pr'),
            pl.when(
                (pl.col('FLAECHE_ratio') >= 0.4 ) & (pl.col('AUSRICHTUNG') > 45) & (pl.col('AUSRICHTUNG') < 135)
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias('filt_dfuid_west_40pr'),
            
            pl.when(
                (pl.col('FLAECHE_ratio') >= 0.35 ) & (pl.col('AUSRICHTUNG') > -135) & (pl.col('AUSRICHTUNG') < -45)
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias('filt_dfuid_east_35pr'),
            pl.when(
                (pl.col('FLAECHE_ratio') >= 0.35 ) & (pl.col('AUSRICHTUNG') > 45) & (pl.col('AUSRICHTUNG') < 135)
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias('filt_dfuid_west_35pr'),

            # SOUTH
            pl.when(
                (pl.col('FLAECHE_ratio') >= 0.5) & (pl.col('AUSRICHTUNG') > -45) & (pl.col('AUSRICHTUNG') < 45)
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias('filt_dfuid_south_50pr'),

            pl.when(
                (pl.col('FLAECHE_ratio') >= 0.4) & (pl.col('AUSRICHTUNG') > -45) & (pl.col('AUSRICHTUNG') < 45)
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias('filt_dfuid_south_40pr'),

        ])

        #  group_by egid + derive filter_tags
        topo_filter_egid = topo_filter.group_by('EGID').agg([
            pl.col('filt_dfuid_east_50pr').any().alias('filt_dfuid_east_50pr'),
            pl.col('filt_dfuid_east_40pr').any().alias('filt_dfuid_east_40pr'),
            pl.col('filt_dfuid_east_35pr').any().alias('filt_dfuid_east_35pr'),

            pl.col('filt_dfuid_west_50pr').any().alias('filt_dfuid_west_50pr'),
            pl.col('filt_dfuid_west_40pr').any().alias('filt_dfuid_west_40pr'), 
            pl.col('filt_dfuid_west_35pr').any().alias('filt_dfuid_west_35pr'),

            pl.col('filt_dfuid_south_50pr').any().alias('filt_dfuid_south_50pr'),
            pl.col('filt_dfuid_south_40pr').any().alias('filt_dfuid_south_40pr'),
        ])

        # get egid specific filters -----
        topo_filter_egid = topo_filter_egid.with_columns([
            pl.when(
                (pl.col('filt_dfuid_east_40pr') == True) & (pl.col('filt_dfuid_west_40pr') == True)
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias('filter_tag__eastwest_80pr'),              
            pl.when(
                (pl.col('filt_dfuid_east_35pr') == True) & (pl.col('filt_dfuid_west_35pr') == True)
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias('filter_tag__eastwest_70pr'),
            pl.when(
                (pl.col('filt_dfuid_east_50pr') == True) | (pl.col('filt_dfuid_west_50pr') == True)
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias('filter_tag__eastORwest_50pr'),
            pl.when(
                (pl.col('filt_dfuid_east_40pr') == True) | (pl.col('filt_dfuid_west_40pr') == True)
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias('filter_tag__eastORwest_40pr'),

            pl.when(
                (pl.col('filt_dfuid_south_50pr') == True)
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias('filter_tag__south_50pr'),
            pl.when(
                (pl.col('filt_dfuid_south_40pr') == True)
            ).then(pl.lit(True)).otherwise(pl.lit(False)).alias('filter_tag__south_40pr'),
        ])

        return topo_filter_egid


    def initial_sml_get_instcost_interpolate_function(self, ): 
        """
        Input:
            PVAllocScenario + PVAllocScenario_Settings dataclass containing all scenarios settings + methods
        Tasks:
            - import the coefficients for the interpolated cost function from energie rechner schweiz
            - bring the coefficients back into their functional form. 
        Output:
            - estim_instcost_chfpkW, estim_instcost_chftotal: cost estimation functions to estimate the installation cost per kW and total installation cost
        """
        # setup --------
        # print_to_logfile('run function: get_estim_instcost_function', self.sett.log_name) if i_m < 5 else None
        checkpoint_to_logfile('npv > subdf: get_estim_instcost_function', self.sett.log_name, 0, self.sett.show_debug_prints)

        with open(f'{self.sett.name_dir_export_path}/pvinstcost_coefficients.json', 'r') as file:
            pvinstcost_coefficients = json.load(file)
        params_pkW = pvinstcost_coefficients['params_pkW']
        # coefs_total = pvinstcost_coefficients['coefs_total']
        params_total = pvinstcost_coefficients['params_total']

        # PV Cost functions --------
        # Define the interpolation functions using the imported coefficients
        def func_chf_pkW(x, a, b, c, d):
            return a +  d*((x ** b) /  (x ** c))
        def estim_instcost_chfpkW(x):
            return func_chf_pkW(x, *params_pkW)

        def func_chf_total(x, a, b, c):
            return a +  b*(x**c) 
        def estim_instcost_chftotal(x):
            return func_chf_total(x, *params_total)
            
        return estim_instcost_chfpkW, estim_instcost_chftotal            




    def algo_update_gridnode_AND_gridprem_POLARS(self, subdir_path: str, i_m: int, m): 
    
        # setup -----------------------------------------------------
        checkpoint_to_logfile('run function: update_gridprem', self.sett.log_name, 0, True)
        gridtiers_power_factor                = self.sett.GRIDspec_power_factor
        share_roof_area_available             = self.sett.TECspec_share_roof_area_available
        kWpeak_per_m2                         = self.sett.TECspec_kWpeak_per_m2
        TECspec_self_consumption_ifapplicable = self.sett.TECspec_self_consumption_ifapplicable
        GRIDspec_perf_factor_1kVA_to_XkW      = self.sett.GRIDspec_perf_factor_1kVA_to_XkW
        print_checkpoint_statements           = True if i_m < 3 else False


        # import  -----------------------------------------------------
        with open(f'{subdir_path}/topo_egid.json', 'r', encoding='utf-8') as f:
            topo = json.load(f)
        topo_subdf_paths = glob.glob(f'{subdir_path}/topo_subdf_*.parquet')
        outtopo_subdf_paths = glob.glob(f'{self.sett.name_dir_export_path}/outtopo_time_subdf/*.parquet')
        dsonodes_df = pl.read_parquet(f'{self.sett.name_dir_import_path}/dsonodes_df.parquet')
        with open(f'{subdir_path}/node_1hll_closed_dict.json', 'r', encoding='utf-8') as f:
            node_1hll_closed_dict = json.load(f)
        with open(f'{subdir_path}/node_subsidy_monitor_dict.json', 'r', encoding='utf-8') as f:
            node_subsidy_monitor_dict = json.load(f)

        data = [(k, v[0], v[1]) for k, v in self.sett.GRIDspec_tiers.items()]
        gridtiers_df = pd.DataFrame(data, columns=self.sett.GRIDspec_colnames)
        if not self.sett.GRIDspec_reupdate_npv_df_TF: 
            gridtiers_df['gridprem_Rp_kWh'] = 0.0   



        # create Map_infosource_egid ----------------------------------------------
        checkpoint_to_logfile('gridprem: start loop Map_infosource_egid', self.sett.log_name, 0, self.sett.show_debug_prints)
        k,v = list(topo.items())[0]
        egid_list, dfuid_list, info_source_list, TotalPower_list, inst_TF_list, grid_node_list, share_pvprod_used_list, dfuidPower_list = [], [], [], [], [], [], [], []
        for k,v in topo.items():
            dfuid_tupls = [tpl[1] for tpl in v['pv_inst']['dfuid_w_inst_tuples'] if tpl[3] > 0.0]
            for k_s, v_s in v['solkat_partitions'].items():
                if k_s in dfuid_tupls:
                    # for tpl in v['pv_inst']['dfuid_w_inst_tuples']:
                    for tpl in v['pv_inst']['dfuid_w_inst_tuples']:
                        if tpl[1] == k_s:       
                            egid_list.append(k)
                            info_source_list.append(v['pv_inst']['info_source'])
                            inst_TF_list.append(True if tpl[3] > 0.0 else False)   
                            dfuid_list.append(tpl[1])
                            share_pvprod_used_list.append(tpl[2])
                            dfuidPower_list.append(tpl[3])
                            grid_node_list.append(v['grid_node']) 
                            TotalPower_list.append(v['pv_inst']['TotalPower'])
                else: 
                    egid_list.append(k)
                    dfuid_list.append(k_s)
                    share_pvprod_used_list.append(0.0)
                    dfuidPower_list.append(0.0)
                    info_source_list.append('')
                    inst_TF_list.append(False)
                    grid_node_list.append(v['grid_node'])  
                    TotalPower_list.append(0.0)          

        Map_pvinfo_topo_egid = pl.DataFrame({'EGID': egid_list, 'df_uid': dfuid_list, 'info_source': info_source_list, 'TotalPower': TotalPower_list,
                                             'inst_TF': inst_TF_list, 'share_pvprod_used': share_pvprod_used_list, 'dfuidPower': dfuidPower_list, 
                                             'grid_node': grid_node_list
                                             })
        Map_pvinfo_gridnode = Map_pvinfo_topo_egid.to_pandas()

        checkpoint_to_logfile('gridprem: end loop Map_infosource_egid', self.sett.log_name, 0, self.sett.show_debug_prints)
            

        # import topo_time_subdfs -----------------------------------------------------
        checkpoint_to_logfile('gridprem: start read subdf', self.sett.log_name, 0, self.sett.show_debug_prints) if print_checkpoint_statements else None

        agg_subdf_updated_pvdf_list = []
        agg_subdf_df_list, agg_egids_list, agg_egids_all_list = [], [], []

        for i, path in enumerate(topo_subdf_paths):
            checkpoint_to_logfile('gridprem > subdf: start read subdf', self.sett.log_name, 0, self.sett.show_debug_prints) if print_checkpoint_statements else None
            subdf = pl.read_parquet(path)      

            # extend subdf with static data (safe disk space) --------------------
            subdf = add_static_topo_data_to_subdf(subdf, topo)
    

            checkpoint_to_logfile('gridprem > subdf: end read subdf', self.sett.log_name, 0, self.sett.show_debug_prints) if print_checkpoint_statements else None

            # start for 1:1 copy for visualization
            subdf_updated = subdf.clone()
            cols_to_update = [ 'info_source', 'inst_TF', 'share_pvprod_used', 'dfuidPower' ]                                      
            subdf_updated = subdf_updated.drop(cols_to_update)                      

            checkpoint_to_logfile('gridprem > subdf: start pandas.merge subdf w Map_infosource_egid', self.sett.log_name, 0, self.sett.show_debug_prints) if print_checkpoint_statements else None
            subdf_updated = subdf_updated.join(Map_pvinfo_topo_egid[['EGID', 'df_uid',] + cols_to_update], on=['EGID', 'df_uid'], how='left')         
                
            # remove the nulls from the merged columns
            subdf_updated = subdf_updated.with_columns([
                pl.when(pl.col('inst_TF').is_null())
                    .then(False).otherwise(pl.col('inst_TF')).alias('inst_TF'),
                pl.when(pl.col('info_source').is_null())
                    .then(pl.lit("")).otherwise(pl.col('info_source')).alias('info_source'),
                pl.when(pl.col('share_pvprod_used').is_null())
                    .then(pl.lit(0.0)).otherwise(pl.col('share_pvprod_used')).alias('share_pvprod_used'),
                pl.when(pl.col('dfuidPower').is_null())
                    .then(pl.lit(0.0)).otherwise(pl.col('dfuidPower')).alias('dfuidPower'),

            ])
            checkpoint_to_logfile('gridprem > subdf: end pandas.merge subdf w Map_infosource_egid', self.sett.log_name, 0, self.sett.show_debug_prints) if print_checkpoint_statements else None

            # pvprod_kW for non_inst EGIDs to 0 --------------------
            # Only consider production for houses that have built a pv installation and substract selfconsumption from the production
            # EDIT 1: Switched off, because we also need the demand of those houses per grid node (and non_inst houses can not feed in by design, so no probelm there)
            # subdf_updated = subdf_updated.filter(pl.col('inst_TF'))       
            # EDIT 2: Previous comment not fully true, pvprod_kW is calculated as the pvproduction potential for a df_uid partition, so those columns are also positive for
            # non-installed houses. SOLUTION: set all pvproduction to 0 for all df_uid that are not in Map_pvinfo_topo_egid (derived from most recent version of topo_egid)
            # EDIT 3: Because demand is per unit of EGID but the subdf data frame is by df_uid, demand_kW will be double counted when simply sumed up like other pvproduction relevant columns.
            # So I need to devide the demand on a house level by n_dfuid (with and / or without installations) so the actuall amount of feedin is calculated and not distorted by double counting. 
            # Two cases should solve all possible applications for this transformation
            # EDIT 4: Division as described in edit3 is working, problems persist with EGIDs with partial pv installations (only 1/2 df_uid installed). Then selfconsumption is only applied to 
            # 1/n-inst_share per EGID, which is too low. Another probably essential problem is, that demand is uniformly distributed over all df_uid, but pvprod is not. So even if a southfacing
            # partition could cover all demand, and another east/west facing partition could do feedin, the total self consumption would be underestimated ) 
            #       -> move selfconsumption part to a later stage of the compuation! agg production by [EGID,t] and only then apply selfconsumption. "demand-splitting" per partition no longer 
            #          needed, use demand.first() in aggregation

            # calculated pvprod_kW, given inst (also partial inst on partition possible)
            subdf_updated = subdf_updated.with_columns([
                (pl.col('poss_pvprod_kW') * pl.col('share_pvprod_used')).alias('pvprod_kW'),
            ])


            # force pvprod == 0 for EGID-df_uid without inst
            Map_pvinst_topo_egid = Map_pvinfo_topo_egid.filter(pl.col('inst_TF') == True)
            subdf_no_inst = subdf_updated.join(
                Map_pvinst_topo_egid[['EGID', 'df_uid']], 
                on=['EGID', 'df_uid'], 
                how='anti'
            )

            subdf_no_inst_EGID = subdf_no_inst['EGID'].to_list()
            subdf_no_inst_dfuid = subdf_no_inst['df_uid'].to_list()
            subdf_updated = subdf_updated.with_columns([
                pl.when(
                    (pl.col('EGID').is_in(subdf_no_inst_EGID) &
                    (pl.col('df_uid').is_in(subdf_no_inst_dfuid)))
                ).then(pl.lit(0.0)).otherwise(pl.col('pvprod_kW')).alias('pvprod_kW'),
            ])
            subdf_updated = subdf_updated.with_columns([
                pl.when(
                    (pl.col('EGID').is_in(subdf_no_inst_EGID) &
                    (pl.col('df_uid').is_in(subdf_no_inst_dfuid)))
                ).then(pl.lit(0.0)).otherwise(pl.col('radiation')).alias('radiation'),
            ])

            # sorting necessary so that .first() statement captures inst_TF and info_source for EGIDS with partial installations
            subdf_updated = subdf_updated.sort(['EGID','inst_TF', 'df_uid', 't_int'], descending=[False, True, False, False])

            # agg per EGID to apply selfconsumption 
            agg_egids = subdf_updated.group_by(['EGID', 't', 't_int']).agg([
                pl.col('inst_TF').first().alias('inst_TF'),
                pl.col('info_source').first().alias('info_source'),
                pl.col('grid_node').first().alias('grid_node'),
                pl.col('demand_kW').first().alias('demand_kW'),

                pl.col('poss_pvprod_kW').sum().alias('poss_pvprod_kW'),
                pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                pl.col('radiation').sum().alias('radiation'),
            ])

            # calc selfconsumption
            agg_egids = agg_egids.sort(['EGID', 't_int'], descending = [False, False])

            selfconsum_expr = pl.min_horizontal([pl.col("pvprod_kW"), pl.col("demand_kW")]) * TECspec_self_consumption_ifapplicable

            agg_egids = agg_egids.with_columns([        
                selfconsum_expr.alias("selfconsum_kW"),
                (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
                (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
            ])

            # Apply per-house (per-EGID) peak-shaving when in later iterations (i_m > 1).
            # This shifts the withholding to the individual-house level so node aggregation
            # later sums already reflect shaved production. When not applied, we still
            # provide the pre-peak-shave column for consistency.
            GRID_peak_shave_TF          = self.sett.GRID_peak_shaving_enabled_tupl[0]
            GRID_peak_shave_crit_level  = self.sett.GRID_peak_shaving_enabled_tupl[1]
            GRID_peak_shave_reduct_fact = self.sett.GRID_peak_shaving_enabled_tupl[2]

            if i_m > 1 and GRID_peak_shave_TF:

                # join gridnode_df to agg_egids, to find critical hours
                gridnode_df = pl.read_parquet(f'{subdir_path}/gridnode_df.parquet')
                gridnode_df.columns
 
                agg_egids = agg_egids.join(gridnode_df[['grid_node', 't', 'max_demand_feedin_atnode_kW', 'kW_threshold']], on=['grid_node', 't'], how='left')                  
                agg_egids = agg_egids.with_columns([
                    (pl.col('max_demand_feedin_atnode_kW') / pl.col('kW_threshold')).alias('HC_ratio'), 
                ])

                # apply peak shaving in relevant hours
                agg_egids = (
                    agg_egids
                    .with_columns(
                        pl.col('netfeedin_kW').alias('netfeedin_prepeakshave_kW')
                    )
                    .with_columns([
                        pl.when(
                            (pl.col('HC_ratio').is_not_null()) &
                            (pl.col('HC_ratio') > GRID_peak_shave_crit_level)
                        )
                        .then(
                            pl.col('netfeedin_prepeakshave_kW') *
                            pl.lit(GRID_peak_shave_reduct_fact)
                        )
                        .otherwise(pl.col('netfeedin_prepeakshave_kW'))
                        .alias('netfeedin_kW'),

                        pl.when(
                            (pl.col('HC_ratio').is_not_null()) &
                            (pl.col('HC_ratio') > GRID_peak_shave_crit_level)
                        )
                        .then(
                            pl.col('netfeedin_prepeakshave_kW') *
                            (pl.lit(1.0) - pl.lit(GRID_peak_shave_reduct_fact))
                        )
                        .otherwise(0.0)
                        .alias('peak_shaved_kW')
                    ])
                )


                # cleanup helper cols
                for col in ['max_demand_feedin_atnode_kW', 'kW_threshold', 'HC_ratio']:
                    if col in agg_egids.columns:
                        agg_egids = agg_egids.drop(col)

            else:
                # keep prepeak column for consistent downstream processing
                agg_egids = agg_egids.with_columns([
                    pl.col('netfeedin_kW').alias('netfeedin_prepeakshave_kW'),
                    pl.lit(0.0).alias('peak_shaved_kW')
                ])

                    

            # (for visualization later) -----
            # only select egids for grid_node mentioned above
            agg_egids_all_list.append(agg_egids)

            agg_egids = agg_egids.filter(pl.col('EGID').is_in(Map_pvinfo_gridnode['EGID'].to_list()))
            agg_egids_list.append(agg_egids)
            # -----

            # agg per gridnode
            agg_subdf = agg_egids.group_by(['grid_node', 't', 't_int']).agg([
            pl.col('inst_TF').first().alias('inst_TF'),
            pl.col('demand_kW').sum().alias('demand_kW'),
            pl.col('pvprod_kW').sum().alias('pvprod_kW'),
            pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
            pl.col('netfeedin_prepeakshave_kW').sum().alias('netfeedin_prepeakshave_kW'),
            pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
            pl.col('peak_shaved_kW').sum().alias('peak_shaved_kW'),
            pl.col('netdemand_kW').sum().alias('netdemand_kW'), 
            pl.col('radiation').sum().alias('radiation'),
            ])

            # agg subdf_updated_dfuid_pvdf for later export
            # (for pvallocation only)
            subdf_updated_dfuid_pvdf = subdf_updated.filter(pl.col('info_source') != '').clone()
            subdf_updated_dfuid_pvdf = subdf_updated_dfuid_pvdf.group_by(['EGID', 'df_uid',]).agg([
                pl.col('inst_TF').first().alias('inst_TF'), 
                pl.col('info_source').first().alias('info_source'), 
                pl.col('pvid').first().alias('pvid'), 
                pl.col('grid_node').first().alias('grid_node'),
                pl.col('FLAECHE').first().alias('FLAECHE'), 
                pl.col('MSTRAHLUNG').first().alias('MSTRAHLUNG'), 

                pl.col('t').len().alias('n_t'),
                pl.col('month').len().alias('n_month'), 

                pl.col('demand_kW').sum().alias('sum_demand_kW'), 
                pl.col('pvprod_kW').first().alias('sum_pvprod_kW'), 

                pl.col('TotalPower').first().alias('TotalPower'), 
                ])
                    
            subdf_updated_dfuid_pvdf = subdf_updated_dfuid_pvdf.with_columns([
                (kWpeak_per_m2 * share_roof_area_available * pl.col('FLAECHE')).alias('poss_prod_capa_kW_peak')
            ])
            # -----
                
            agg_subdf_df_list.append(agg_subdf)
            agg_subdf_updated_pvdf_list.append(subdf_updated_dfuid_pvdf)


        agg_subdf_df = pl.concat(agg_subdf_df_list)
        topo_gridnode_df = agg_subdf_df.group_by(['grid_node', 't', 't_int']).agg([
            pl.col('inst_TF').first().alias('inst_TF'),
            pl.col('demand_kW').sum().alias('demand_kW'),
            pl.col('pvprod_kW').sum().alias('pvprod_kW'),
            pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
            pl.col('netfeedin_prepeakshave_kW').sum().alias('netfeedin_prepeakshave_kW'),
            pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
            pl.col('peak_shaved_kW').sum().alias('peak_shaved_kW'),
            pl.col('netdemand_kW').sum().alias('netdemand_kW'),
            pl.col('radiation').sum().alias('radiation'),
        ])
            
        # (for visualization later) -----
        # MAIN DF of this plot, all feedin TS by EGID
        topo_agg_egids_df = pl.concat(agg_egids_list)
        topo_agg_egids_df = topo_agg_egids_df.with_columns([
            pl.col('t').str.strip_chars('t_').cast(pl.Int64).alias('t_int'),
        ])
        topo_agg_egids_df = topo_agg_egids_df.sort("t_int", descending=False)
        # -----

        agg_subdf_updated_pvdf = pl.concat(agg_subdf_updated_pvdf_list)


        # import outtopo_time_subfs -----------------------------------------------------
        agg_subdf_df_list = []
        for i, path in enumerate(outtopo_subdf_paths):
            outsubdf = pl.read_parquet(path)  
            agg_outsubdf = outsubdf.group_by(['grid_node', 't']).agg([
                pl.col('demand_proxy_out_kW').sum().alias('demand_proxy_out_kW'),
            ])
            del outsubdf
            agg_subdf_df_list.append(agg_outsubdf)
                
        agg_outsubdf_df = pl.concat(agg_subdf_df_list)
        outtopo_gridnode_df = agg_outsubdf_df.group_by(['grid_node', 't']).agg([
            pl.col('demand_proxy_out_kW').sum().alias('demand_proxy_out_kW'),
        ])


        # build gridnode_df -----------------------------------------------------
        checkpoint_to_logfile('gridprem: start merge to (out)topo_gridnode_df to gridnode_ts', self.sett.log_name, 0, self.sett.show_debug_prints) if print_checkpoint_statements else None   
        gridnode_df = topo_gridnode_df.join(outtopo_gridnode_df, on=['grid_node', 't'], how='left')
        gridnode_df = gridnode_df.with_columns([
            pl.col('t').str.strip_chars('t_').cast(pl.Int64).alias('t_int'),
        ])
            
        gridnode_df = gridnode_df.with_columns([
            pl.col('netfeedin_kW').alias('feedin_atnode_kW'), 
            (pl.col('netdemand_kW') + pl.col('demand_proxy_out_kW')).alias('demand_atnode_kW'),
        ])
        gridnode_df = gridnode_df.with_columns([
            pl.max_horizontal(['feedin_atnode_kW', 'demand_atnode_kW']).alias('max_demand_feedin_atnode_kW')
        ])

        # sanity check
        gridnode_df.group_by(['grid_node',]).agg([pl.len()])
        gridnode_df.group_by(['t',]).agg([pl.len()])

        # attach node thresholds 
        gridnode_df = gridnode_df.join(dsonodes_df[['grid_node', 'kVA_threshold']], on='grid_node', how='left')
        gridnode_df = gridnode_df.with_columns((pl.col("kVA_threshold") * GRIDspec_perf_factor_1kVA_to_XkW).alias("kW_threshold"))
            
        gridnode_df = gridnode_df.with_columns([
            pl.when(pl.col("max_demand_feedin_atnode_kW") < 0)
            .then(0.0)
            .otherwise(pl.col("max_demand_feedin_atnode_kW"))
            .alias("max_demand_feedin_atnode_kW"),
            ])
        gridnode_df = gridnode_df.with_columns([
            pl.when(pl.col("feedin_atnode_kW") > pl.col("kW_threshold"))
            .then(pl.col("kW_threshold"))
            .otherwise(pl.col("feedin_atnode_kW"))
            .alias("feedin_atnode_taken_kW"),
            ])
        gridnode_df = gridnode_df.with_columns([
            pl.when(pl.col("feedin_atnode_kW") > pl.col("kW_threshold"))
            .then(pl.col("feedin_atnode_kW") - pl.col("kW_threshold"))
            .otherwise(0.0)
            .alias("feedin_atnode_loss_kW")
        ])
        checkpoint_to_logfile('gridprem: end merge with gridnode_df + pl.when()', self.sett.log_name, 0, self.sett.show_debug_prints) if print_checkpoint_statements else None   


        # update node 1hll monitor  -----------------------------------------------------
        # # --- only for dev ---
        # gridnode_df = gridnode_df.with_columns(
        #     pl.when((pl.col('grid_node') == '397') & (pl.col('t') == 't_10'))
        #     .then(1)  # Assign value 1
        #     .otherwise(pl.col('feedin_atnode_loss_kW'))  # Keep existing value if condition is not met
        #     .alias('feedin_atnode_loss_kW')  # Update column in DataFrame
        # )
        # # --- --- ---          
        gridnode_df = gridnode_df.sort(['grid_node', 't_int'], descending=[False, False])
        gridnode_df_hours_agg = gridnode_df.group_by(['grid_node', ]).agg([
            pl.col('feedin_atnode_taken_kW').sum().alias('feedin_atnode_taken_kW'),
            pl.col('feedin_atnode_loss_kW').sum().alias('feedin_atnode_loss_kW'),
        ])
        gridnode_df_abv_1hll = gridnode_df_hours_agg.filter(pl.col('feedin_atnode_loss_kW') > 0).select('grid_node').to_series().to_list()

        if str(i_m-1) in node_1hll_closed_dict.keys(): 
            prev_nodes_closed = node_1hll_closed_dict[str(i_m-1)]['all_nodes_abv_1hll']
            new_nodes_closed = [node for node in gridnode_df_abv_1hll if node not in prev_nodes_closed]
            add_dict = {'all_nodes_abv_1hll': gridnode_df_abv_1hll, 
                        'new_nodes_abv_1hll': new_nodes_closed
                        }
        else: 
            add_dict = {'all_nodes_abv_1hll': gridnode_df_abv_1hll, 
                        'new_nodes_abv_1hll': gridnode_df_abv_1hll}
                
            
        node_1hll_closed_dict[str(i_m)] = add_dict
        with open(f'{subdir_path}/node_1hll_closed_dict.json', 'w') as f:
            json.dump(node_1hll_closed_dict, f, indent=4)


        # update node subsidy monitor  -----------------------------------------------------
        subsidy_node_name   = self.sett.GRIDspec_subsidy_name
        subsidy_node_scheme = self.sett.GRIDspec_subsidy_filtag_node_schemes[subsidy_node_name]
        subs_node_tpl = subsidy_node_scheme['subs_nodeHC_chf_tuples']
        pena_node_tpl = subsidy_node_scheme['pena_nodeHC_chf_tuples']


        gridnode_df = gridnode_df.sort(['grid_node', 't_int'], descending=[False, False])
        gridnode_max_HC = gridnode_df.group_by(['grid_node', ]).agg([
            pl.col('max_demand_feedin_atnode_kW').max().alias('max_demand_feedin_atnode_kW'),
            pl.col('kW_threshold').first().alias('kW_threshold'),
        ])
        gridnode_max_HC = gridnode_max_HC.with_columns([
            (pl.col('max_demand_feedin_atnode_kW') / pl.col('kW_threshold')).alias('HC_ratio'),
        ])

        subs_gridnodes_list = gridnode_max_HC.filter(pl.col('HC_ratio') <= subs_node_tpl[0]).select('grid_node').to_series().to_list()
        pena_gridnodes_list = gridnode_max_HC.filter(pl.col('HC_ratio') >  pena_node_tpl[0]).select('grid_node').to_series().to_list()

        node_subsidy_monitor_dict[str(i_m)] = {
            'subs_gridnodes': list(set(subs_gridnodes_list)),
            'pena_gridnodes': list(set(pena_gridnodes_list)),
        }
        with open(f'{subdir_path}/node_subsidy_monitor_dict.json', 'w') as f:
            json.dump(node_subsidy_monitor_dict, f, indent=4)


        # update gridprem_ts -----------------------------------------------------
        checkpoint_to_logfile('gridprem: start update gridprem_ts', self.sett.log_name, 0, self.sett.show_debug_prints) if print_checkpoint_statements else None   
        gridnode_df = gridnode_df.sort("feedin_atnode_taken_kW", descending=True)            
        gridnode_df_for_prem = gridnode_df.group_by(['grid_node', 'kW_threshold', 't']).agg(
            pl.col('feedin_atnode_taken_kW').sum().alias('feedin_atnode_taken_kW'), 
            pl.col('feedin_atnode_loss_kW').sum().alias('feedin_atnode_loss_kW'),
            pl.col('kW_threshold').first().alias('kW_threshold_first'),
        ).clone()
        gridnode_df_for_prem = (
            gridnode_df_for_prem
            .drop('kW_threshold')
            .rename({'kW_threshold_first': 'kW_threshold'})
        )            

        gridnode_df_for_prem = gridnode_df_for_prem.with_columns(
            pl.lit(0).alias("prem_Rp_kWh")
        )   
            
        for i in reversed(range(len(gridtiers_df))):
            capa_tier_rate_lo = gridtiers_df.loc[i, 'used_node_capa_rate']
            capa_tier_rate_up = gridtiers_df.loc[i+1, 'used_node_capa_rate'] if i < len(gridtiers_df)-1 else 1
            prem = gridtiers_df.loc[i, 'gridprem_Rp_kWh']

            condition = (
                ((pl.col("feedin_atnode_taken_kW") / pl.col("kW_threshold")) > capa_tier_rate_lo) &
                ((pl.col("feedin_atnode_taken_kW") / pl.col("kW_threshold")) <= capa_tier_rate_up)
            )

            gridnode_df_for_prem = gridnode_df_for_prem.with_columns(
                pl.when(condition)
                .then(pl.lit(prem))
                .otherwise(pl.col("prem_Rp_kWh"))
                .alias("prem_Rp_kWh")
            )

        checkpoint_to_logfile('gridprem: end update gridprem_ts', self.sett.log_name, 0, self.sett.show_debug_prints) if print_checkpoint_statements else None   
            
        # sanity check
        gridnode_df_for_prem.filter(pl.col('prem_Rp_kWh') > 0)

        gridprem_ts = gridnode_df_for_prem.drop(['feedin_atnode_taken_kW', 'feedin_atnode_loss_kW', 'kW_threshold']).clone()   
    

        # EXPORT -----------------------------------------------------
        gridnode_df = gridnode_df.sort("t_int", descending=False)            
        gridnode_df.write_parquet(f'{subdir_path}/gridnode_df.parquet')    
        gridprem_ts.write_parquet(f'{subdir_path}/gridprem_ts.parquet')    
        if self.sett.export_csvs and (i_m < 3):                                          
            gridnode_df.write_csv(f'{subdir_path}/gridnode_df.csv')        
            gridprem_ts.write_csv(f'{subdir_path}/gridprem_ts.csv')        

        if self.sett.export_csvs and (i_m < 3):
            agg_subdf_updated_pvdf.write_parquet(f'{subdir_path}/SANITYCHECK_topo_agg_subdf_updated_pvdf_iter{i_m}.parquet')
            agg_subdf_updated_pvdf.write_csv(f'{subdir_path}/SANITYCHECK_topo_agg_subdf_updated_pvdf_iter{i_m}.csv')       



        # export by Month -----------------------------------------------------
        if self.sett.MCspec_keep_files_month_iter_TF:
            if i_m < self.sett.MCspec_keep_files_month_iter_max:
                # gridprem_node_by_M_path = f'{self.sett.pvalloc_path}/pred_gridprem_node_by_M'
                gridprem_node_by_M_path = f'{subdir_path}/pred_gridprem_node_by_M'
                if not os.path.exists(gridprem_node_by_M_path):
                    os.makedirs(gridprem_node_by_M_path)

                gridnode_df.write_parquet(f'{gridprem_node_by_M_path}/gridnode_df_{i_m}.parquet')   
                gridprem_ts.write_parquet(f'{gridprem_node_by_M_path}/gridprem_ts_{i_m}.parquet')   

                if self.sett.export_csvs:
                    gridnode_df.write_csv(f'{gridprem_node_by_M_path}/gridnode_df_{i_m}.csv')           
                    gridprem_ts.write_csv(f'{gridprem_node_by_M_path}/gridprem_ts_{i_m}.csv')           
        
        checkpoint_to_logfile('exported gridprem_ts and gridnode_df', self.sett.log_name, self.sett.show_debug_prints) if print_checkpoint_statements else None
            



    def add_elecpri_to_subdf(self, subdf_func, elecpri_vese, elecpri_entsoe, hourly=False): 
        if self.sett.TECspec_elecpri_Rp_kWh is None: 
            # elecpri_raw = pl.read_parquet(os.path.join(self.sett.name_dir_import_path, 'elecpri.parquet'))
            elecpri_select = elecpri_vese.filter(
                (pl.col('bfs_number').is_in([str(bfs) for bfs in self.sett.bfs_numbers]) ) ,
                (pl.col('year') == self.sett.TECspec_elecpri_year)  , 
                (pl.col('category') == self.sett.TECspec_elecpri_category)
            ).rename({
                'bfs_number': 'bfs',})
            # multiple prices per bfs -> mean
            if pd.Series(elecpri_select['bfs']).duplicated().any():
                print_to_logfile(f'elecpri: multiple prices per bfs, mean taken', self.sett.log_name)
            elecpri_select = elecpri_select.group_by('bfs').agg([
                pl.col('energy').mean().alias('energy'),
                pl.col('grid').mean().alias('grid'),
                pl.col('aidfee').mean().alias('aidfee'),
                pl.col('taxes').mean().alias('taxes'),
                pl.col('fixcosts').mean().alias('fixcosts'),
            ])

            price_component_expr = pl.sum_horizontal([
                pl.col('energy'),
                pl.col('grid'),
                pl.col('aidfee'),
                pl.col('taxes'),
                pl.col('fixcosts'),
            ])
            elecpri_select = elecpri_select.with_columns([
                price_component_expr.alias('elecpri_Rp_kWh'), 
            ]).select(['bfs', 'elecpri_Rp_kWh'])
            subdf_func = subdf_func.join(elecpri_select, on='bfs', how='left')
                    
        elif isinstance(self.sett.TECspec_elecpri_Rp_kWh, float) or isinstance(self.sett.TECspec_elecpri_Rp_kWh, int):
            elecpri = self.sett.TECspec_elecpri_Rp_kWh
            subdf_func = subdf_func.with_columns(
                pl.lit(elecpri).alias('elecpri_Rp_kWh')
            )

        elif isinstance(self.sett.TECspec_elecpri_Rp_kWh, str) and hourly:
            # price per hour of the elecpri year, joined on hour of year (t_int)  > used for the NPV
            elecpri_year = elecpri_entsoe.filter(pl.col('year') == self.sett.TECspec_elecpri_year)
            if elecpri_year.shape[0] == 0:
                raise ValueError(f'elecpri: year {self.sett.TECspec_elecpri_year} not found in the hourly elecpri data')
            if 'elecpri_Rp_kWh' in subdf_func.columns:
                subdf_func = subdf_func.drop('elecpri_Rp_kWh')
            subdf_func = subdf_func.join(
                elecpri_year.select([pl.col('t_int').cast(subdf_func.schema['t_int']), 'elecpri_Rp_kWh']),
                on='t_int', how='left', maintain_order='left')
            if subdf_func['elecpri_Rp_kWh'].null_count() > 0:
                raise ValueError(f"elecpri: {subdf_func['elecpri_Rp_kWh'].null_count()} rows without an hourly price after the join on t_int")

        elif isinstance(self.sett.TECspec_elecpri_Rp_kWh, str) and not hourly:
            # yearly mean as one scalar  > used as feature for the random forest
            elecpri_year = elecpri_entsoe.filter(pl.col('year') == self.sett.TECspec_elecpri_year)
            if elecpri_year.shape[0] == 0:
                raise ValueError(f'elecpri: year {self.sett.TECspec_elecpri_year} not found in the hourly elecpri data')
            elecpri = elecpri_year['elecpri_Rp_kWh'].mean()                    
            subdf_func = subdf_func.with_columns(
                pl.lit(elecpri).alias('elecpri_Rp_kWh')
            )

        return subdf_func




    def load_elecpri_inputs(self):
        """
        Load the price tables that add_elecpri_to_subdf needs for the current TECspec_elecpri_Rp_kWh setting.
        The table of a mode that is not active stays None (old preprep folders without elecpri_entsoe.parquet keep working for None / float).
            - None  : elecpri_vese   = ElCom elecpri table (elecpri.parquet), filtered by bfs, year and category in add_elecpri_to_subdf
            - float : nothing to load
            - str   : elecpri_entsoe = hourly elecpri data from the preprep folder, file name = TECspec_elecpri_Rp_kWh
        """
        elecpri_vese, elecpri_entsoe = None, None

        if self.sett.TECspec_elecpri_Rp_kWh is None:
            elecpri_vese = pl.read_parquet(os.path.join(self.sett.name_dir_import_path, 'elecpri.parquet'))

        elif isinstance(self.sett.TECspec_elecpri_Rp_kWh, str):
            elecpri_entsoe = pl.read_parquet(os.path.join(self.sett.name_dir_import_path, self.sett.TECspec_elecpri_Rp_kWh))

        return elecpri_vese, elecpri_entsoe



    def load_pvtarif_inputs(self):
        """
        Load the price tables that add_pvtarif_to_subdf needs for the current TECspec_pvtarif_Rp_kWh setting.
        The table of a mode that is not active stays None (old preprep folders without pvtarif_entsoe.parquet keep working for None / float).
            - None  : pvtarif_vese   = [bfs, pvtarif_Rp_kWh], official VESE tariff of TECspec_pvtarif_year by municipality
            - float : nothing to load
            - str   : pvtarif_entsoe = hourly pvtarif data from the preprep folder, clipped to TECspec_pvtarif_Rp_floor
        """
        pvtarif_vese, pvtarif_entsoe = None, None

        if self.sett.TECspec_pvtarif_Rp_kWh is None:
            pvtarif_col = self.sett.TECspec_pvtarif_col
            pvtarif_raw = pl.read_parquet(os.path.join(self.sett.name_dir_import_path, 'pvtarif.parquet'))
            Map_gm_ewr  = pl.read_parquet(os.path.join(self.sett.name_dir_import_path, 'Map_gm_ewr.parquet'))

            pvtarif_select = (
                pvtarif_raw
                .select([pl.col('nrElcom').cast(pl.Utf8), pl.col('year').cast(pl.Utf8)] + [pl.col(col) for col in pvtarif_col])
                .join(
                    Map_gm_ewr.select([pl.col('nrElcom').cast(pl.Utf8), pl.col('bfs').fill_null(0).cast(pl.Int64).cast(pl.Utf8)]),
                    on='nrElcom', how='inner')
                .filter(
                    (pl.col('year') == str(self.sett.TECspec_pvtarif_year)[2:4]) &              # VESE years are 2-digit strings
                    (pl.col('bfs').is_in([str(bfs) for bfs in self.sett.bfs_numbers])))
                .with_columns([pl.col(col).cast(pl.Utf8).replace('', '0').cast(pl.Float64, strict=False) for col in pvtarif_col])    # as in initialization: '' = 0, null = no tariff reported (skipped in the mean)
            )
            # multiple tariffs per bfs (several EWRs in one municipality) -> mean, then sum of the tariff columns
            if pvtarif_select.shape[0] > pvtarif_select['bfs'].n_unique():
                print_to_logfile(f'pvtarif: multiple tariffs per bfs, mean taken', self.sett.log_name)
            pvtarif_vese = (
                pvtarif_select
                .group_by('bfs')
                .agg([pl.col(col).mean() for col in pvtarif_col])
                .with_columns(pl.sum_horizontal(pvtarif_col).alias('pvtarif_Rp_kWh'))
                .select(['bfs', 'pvtarif_Rp_kWh'])
            )

        elif isinstance(self.sett.TECspec_pvtarif_Rp_kWh, str):
            pvtarif_entsoe = pl.read_parquet(os.path.join(self.sett.name_dir_import_path, self.sett.TECspec_pvtarif_Rp_kWh))
            n_below_floor = pvtarif_entsoe.filter(
                (pl.col('year') == self.sett.TECspec_pvtarif_year) & (pl.col('pvtarif_Rp_kWh') < self.sett.TECspec_pvtarif_Rp_floor)
            ).shape[0]
            print_to_logfile(f'pvtarif: {n_below_floor} hours of {self.sett.TECspec_pvtarif_year} raised to the floor of {self.sett.TECspec_pvtarif_Rp_floor} Rp/kWh', self.sett.log_name)
            # clip keeps nulls as nulls, so missing prices are still caught by the null check in add_pvtarif_to_subdf
            pvtarif_entsoe = pvtarif_entsoe.with_columns(pl.col('pvtarif_Rp_kWh').clip(lower_bound=self.sett.TECspec_pvtarif_Rp_floor))

        return pvtarif_vese, pvtarif_entsoe



    def add_pvtarif_to_subdf(self, subdf_func, pvtarif_vese, pvtarif_entsoe, hourly=False):
        """
        Add pvtarif_Rp_kWh to subdf_func, same logic as add_elecpri_to_subdf (inputs from load_pvtarif_inputs):
            - None  : official VESE tariff of TECspec_pvtarif_year by municipality (pvtarif_vese, join on bfs)
            - float : generic tariff
            - str   : hourly data (pvtarif_entsoe, already clipped to TECspec_pvtarif_Rp_floor)
                        hourly=True : price per hour (join on t_int)  > used for the NPV
                        hourly=False: yearly mean as one scalar       > used as feature for the random forest
        There is deliberately no final else: for an unexpected setting no column is created and the next pl.col('pvtarif_Rp_kWh') fails loudly.
        """
        if 'pvtarif_Rp_kWh' in subdf_func.columns:      # a stale column of an earlier step must not mask an unmatched setting
            subdf_func = subdf_func.drop('pvtarif_Rp_kWh')

        if self.sett.TECspec_pvtarif_Rp_kWh is None:
            subdf_func = subdf_func.join(pvtarif_vese, on='bfs', how='left', maintain_order='left')
            if subdf_func['pvtarif_Rp_kWh'].null_count() > 0:
                bfs_without_tarif = subdf_func.filter(pl.col('pvtarif_Rp_kWh').is_null())['bfs'].unique().to_list()
                raise ValueError(f'pvtarif: no VESE tariff for year {self.sett.TECspec_pvtarif_year} in bfs {bfs_without_tarif}')

        elif isinstance(self.sett.TECspec_pvtarif_Rp_kWh, (int, float)):
            subdf_func = subdf_func.with_columns(
                pl.lit(float(self.sett.TECspec_pvtarif_Rp_kWh)).alias('pvtarif_Rp_kWh')
            )

        elif isinstance(self.sett.TECspec_pvtarif_Rp_kWh, str) and hourly:
            pvtarif_year = pvtarif_entsoe.filter(pl.col('year') == self.sett.TECspec_pvtarif_year)
            if pvtarif_year.shape[0] == 0:
                raise ValueError(f'pvtarif: year {self.sett.TECspec_pvtarif_year} not found in the hourly pvtarif data')
            subdf_func = subdf_func.join(
                pvtarif_year.select([pl.col('t_int').cast(subdf_func.schema['t_int']), 'pvtarif_Rp_kWh']),
                on='t_int', how='left', maintain_order='left')
            if subdf_func['pvtarif_Rp_kWh'].null_count() > 0:
                raise ValueError(f"pvtarif: {subdf_func['pvtarif_Rp_kWh'].null_count()} rows without an hourly price after the join on t_int")

        elif isinstance(self.sett.TECspec_pvtarif_Rp_kWh, str) and not hourly:
            pvtarif_year = pvtarif_entsoe.filter(pl.col('year') == self.sett.TECspec_pvtarif_year)
            if pvtarif_year.shape[0] == 0:
                raise ValueError(f'pvtarif: year {self.sett.TECspec_pvtarif_year} not found in the hourly pvtarif data')
            pvtarif = pvtarif_year['pvtarif_Rp_kWh'].mean()
            subdf_func = subdf_func.with_columns(
                pl.lit(pvtarif).alias('pvtarif_Rp_kWh')
            )

        return subdf_func




    def algo_update_npv_df_RFR(self, subdir_path: str, i_m: int, m, rfr_model, encoder):
        """
            This function estimates the installation size of all houses in sample, based on a previously run statistical model calibration. 
            This stat model coefficients are imported and used to determine the most realistic installation size chose for the house
            Model used: 
                - Random Forest Regression 
                  (directly estiamting the installation size based on building characteristics)
        """

        # setup -----------------------------------------------------
        checkpoint_to_logfile(f'run function: algo_update_npv_df_RFR, GRIDspec_node_1hll_closed_TF:{self.sett.GRIDspec_node_1hll_closed_TF}', self.sett.log_name, 0, True)         



        # import -----------------------------------------------------
        # gridprem_ts = pl.read_parquet(f'{subdir_path}/gridprem_ts.parquet')    
        with open(f'{subdir_path}/topo_egid.json', 'r', encoding='utf-8') as f:
            topo = json.load(f)

        with open(f'{subdir_path}/node_1hll_closed_dict.json', 'r', encoding='utf-8') as f:
            node_1hll_closed_dict = json.load(f)
        with open(f'{subdir_path}/node_subsidy_monitor_dict.json', 'r', encoding='utf-8') as f:
            node_subsidy_monitor_dict = json.load(f)

        elecpri_vese, elecpri_entsoe = self.load_elecpri_inputs()
        pvtarif_vese, pvtarif_entsoe = self.load_pvtarif_inputs()

        GRID_peak_shave_TF          = self.sett.GRID_peak_shaving_enabled_tupl[0]
        GRID_peak_shave_crit_level  = self.sett.GRID_peak_shaving_enabled_tupl[1]
        GRID_peak_shave_reduct_fact = self.sett.GRID_peak_shaving_enabled_tupl[2]

        gridnode_df_peakshave = None
        if i_m > 1 and GRID_peak_shave_TF:
            gridnode_df_peakshave = pl.read_parquet(f'{subdir_path}/gridnode_df.parquet').select([
                'grid_node',
                't',
                'max_demand_feedin_atnode_kW',
                'kW_threshold',
            ])
                    

        # import topo_time_subdfs -----------------------------------------------------
        topo_subdf_paths = glob.glob(f'{subdir_path}/topo_subdf_*.parquet') 
        no_pv_egid = [k for k, v in topo.items() if not v.get('pv_inst', {}).get('inst_TF') ]

        closed_nodes = node_1hll_closed_dict[str(i_m)]['all_nodes_abv_1hll']
        closed_nodes_egid = [k for k, v in topo.items() if v.get('grid_node')  in closed_nodes ]
                        
            
        agg_npv_df_list = []
        j = 0
        i, path = j, topo_subdf_paths[j]
        for i, path in enumerate(topo_subdf_paths):
            print_topo_subdf_TF = len(topo_subdf_paths) > 5 and i <5  # i% (len(topo_subdf_paths) //3 ) == 0:
            if print_topo_subdf_TF:
                print_to_logfile(f'updated npv (tranche {i+1}/{len(topo_subdf_paths)})', self.sett.log_name)
            subdf_t0 = pl.read_parquet(path) # subdf_t0 = pd.read_parquet(path)

            # drop egids with pv installations
            subdf = subdf_t0.filter(pl.col("EGID").is_in(no_pv_egid))   

            # drop egids with closed grid nodes
            if self.sett.GRIDspec_node_1hll_closed_TF:
                subdf = subdf.filter( ~pl.col("EGID").is_in(closed_nodes_egid))

            if subdf.shape[0] > 0:

                # extend subdf with static data (safe disk space) --------------------
                subdf = add_static_topo_data_to_subdf(subdf, topo)
                if 'elecpri_Rp_kWh' in subdf.columns:
                    subdf = subdf.drop(['elecpri_Rp_kWh', ])

                # # merge gridprem_ts
                # checkpoint_to_logfile('npv > subdf: start merge subdf w gridprem_ts', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None
                # subdf = subdf.join(gridprem_ts[['t', 'grid_node', 'prem_Rp_kWh']], on=['t', 'grid_node'], how='left')  
                # checkpoint_to_logfile('npv > subdf: start merge subdf w gridprem_ts', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None
                

                # add static electricity price for random forest --------------------
                subdf = self.add_elecpri_to_subdf(subdf, elecpri_vese, elecpri_entsoe)

                # add static pvtarif for random forest --------------------
                subdf = self.add_pvtarif_to_subdf(subdf, pvtarif_vese, pvtarif_entsoe)

                checkpoint_to_logfile('npv > subdf - all df_uid-combinations: start calc selfconsumption + netdemand', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None

                # egid = '2362103'
                agg_npv_list = []                    
                n_egid_econ_functions_counter = 0

                # if True: 
                for egid in list(subdf['EGID'].unique()):
                    egid_subdf = subdf.filter(pl.col('EGID') == egid).clone()

                    # arrange data to fit stat estimation model --------------------

                    # egid_dfuid_subagg = egid_subdf.group_by(['EGID', 'df_uid', ]).agg([
                    sub_egiddfuid = egid_subdf.group_by(['EGID', 'df_uid', ]).agg([
                        pl.col('bfs').first().alias('BFS_NUMMER'),
                        pl.col('GKLAS').first().alias('GKLAS'),
                        pl.col('GAREA').first().alias('GAREA'),
                        pl.col('GBAUJ').first().alias('GBAUJ'),
                        pl.col('GSTAT').first().alias('GSTAT'),
                        pl.col('GWAERZH1').first().alias('GWAERZH1'),
                        pl.col('GENH1').first().alias('GENH1'),
                        pl.col('sfhmfh_typ').first().alias('sfhmfh_typ'),
                        pl.col('demand_arch_typ').first().alias('demand_arch_typ'),
                        pl.col('demand_elec_pGAREA').first().alias('demand_elec_pGAREA'),
                        pl.col('grid_node').first().alias('grid_node'),
                        pl.col('inst_TF').first().alias('inst_TF'),
                        pl.col('info_source').first().alias('info_source'),
                        pl.col('pvid').first().alias('pvid'),
                        pl.col('pvtarif_Rp_kWh').first().alias('pvtarif_Rp_kWh'),
                        pl.col('TotalPower').first().alias('TotalPower'),
                        pl.col('FLAECHE').first().alias('FLAECHE'),
                        pl.col('AUSRICHTUNG').first().alias('AUSRICHTUNG'),
                        pl.col('STROMERTRAG').first().alias('STROMERTRAG'),
                        pl.col('NEIGUNG').first().alias('NEIGUNG'),
                        pl.col('MSTRAHLUNG').first().alias('MSTRAHLUNG'),
                        pl.col('GSTRAHLUNG').first().alias('GSTRAHLUNG'),
                        pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                        # pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),
                        ])

                    # create direction classes
                    subagg_dir = sub_egiddfuid.with_columns([
                        pl.when((pl.col("AUSRICHTUNG") > 135) | (pl.col("AUSRICHTUNG") <= -135))
                        .then(pl.lit("north_max_flaeche"))
                        .when((pl.col("AUSRICHTUNG") > -135) & (pl.col("AUSRICHTUNG") <= -45))
                        .then(pl.lit("east_max_flaeche"))
                        .when((pl.col("AUSRICHTUNG") > -45) & (pl.col("AUSRICHTUNG") <= 45))
                        .then(pl.lit("south_max_flaeche"))
                        .when((pl.col("AUSRICHTUNG") > 45) & (pl.col("AUSRICHTUNG") <= 135))
                        .then(pl.lit("west_max_flaeche"))
                        .otherwise(pl.lit("Unkown"))
                        .alias("Direction")
                        ])
                    subagg_dir = subagg_dir.with_columns([
                        pl.col("Direction").fill_null(0).alias("Direction")
                        ])

                    topo_pivot = (
                        subagg_dir
                        .group_by(['EGID', 'Direction'])
                        .agg(
                            pl.col('FLAECHE').max().alias('max_flaeche'), 
                            )
                        .pivot(
                            values='max_flaeche',
                            index='EGID', 
                            on='Direction')
                            .sort('EGID')
                        )
                    topo_rest = (
                        sub_egiddfuid
                        .group_by(['EGID', ])
                        .agg(
                            pl.col('BFS_NUMMER').first().alias('BFS_NUMMER'),
                            pl.col('GAREA').first().alias('GAREA'),
                            pl.col('GBAUJ').first().alias('GBAUJ'),
                            pl.col('GKLAS').first().alias('GKLAS'),
                            pl.col('GSTAT').first().alias('GSTAT'),
                            pl.col('GWAERZH1').first().alias('GWAERZH1'),
                            pl.col('GENH1').first().alias('GENH1'),
                            pl.col('sfhmfh_typ').first().alias('sfhmfh_typ'),
                            pl.col('demand_arch_typ').first().alias('demand_arch_typ'),
                            pl.col('demand_elec_pGAREA').first().alias('demand_elec_pGAREA'),
                            pl.col('grid_node').first().alias('grid_node'),
                            pl.col('inst_TF').first().alias('inst_TF'),
                            pl.col('info_source').first().alias('info_source'),
                            pl.col('pvid').first().alias('pvid'),
                            pl.col('pvtarif_Rp_kWh').first().alias('pvtarif_Rp_kWh'),
                            pl.col('TotalPower').first().alias('TotalPower'),
                            pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                            # pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),

                            pl.col('FLAECHE').first().alias('FLAECHE_total'),
                            )
                        )
                    subagg = topo_rest.join(topo_pivot, on=['EGID'], how='left')

                    # fill empty classes with 0
                    for direction in [
                        'north_max_flaeche',
                        'east_max_flaeche',
                        'south_max_flaeche',
                        'west_max_flaeche',
                        ]:
                        if direction not in subagg.columns:
                            subagg = subagg.with_columns([
                            pl.lit(0).alias(direction)
                            ])
                        else:
                            subagg = subagg.with_columns([
                                pl.col(direction).fill_null(0).alias(direction)
                                ])
                        

                    # apply estim model prediction --------------------
                    df = subagg.to_pandas()
                    df['GWAERZH1_str'] = np.where(df['GWAERZH1'].isin(['7410', '7411']), 'heatpump', 'no_heatpump')

                    cols_dtypes_tupls = {
                        # 'year': 'int64',
                        'BFS_NUMMER': 'category',
                        'GAREA': 'float64',
                        # 'GBAUJ': 'int64',   
                        'GKLAS': 'category',
                        # 'GSTAT': 'category',
                        'GWAERZH1': 'category',
                        'GENH1': 'category',
                        'GWAERZH1_str': 'category',
                        # 'InitialPower': 'float64',
                        'TotalPower': 'float64',
                        'elecpri_Rp_kWh': 'float64',
                        'pvtarif_Rp_kWh': 'float64',
                        'FLAECHE_total': 'float64',
                        'east_max_flaeche': 'float64',
                        'west_max_flaeche': 'float64',
                        'north_max_flaeche': 'float64',
                        'south_max_flaeche': 'float64',
                    }
                    df = df[[col for col in cols_dtypes_tupls.keys() if col in df.columns]]

                    df = df.dropna().copy()
                        
                    for col, dtype in cols_dtypes_tupls.items():
                        df[col] = df[col].astype(dtype)
                        

                    X = df.drop(columns=['TotalPower',])
                    cat_cols = X.select_dtypes(include=["object", "category"]).columns

                    encoded_array = encoder.transform(X[cat_cols].astype(str))
                    encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cat_cols))
                        
                    # Final feature set
                    X_final = pd.concat(
                        [X.drop(columns=cat_cols).reset_index(drop=True), encoded_df.reset_index(drop=True)],
                        axis=1)
                    X_final = X_final[rfr_model.feature_names_in_]

                    pred_instPower = rfr_model.predict(X_final)[0]
                    df['pred_dfuidPower'] = pred_instPower


                    # distribute kWp to partition(s) -----------------
                    egid_list, dfuid_list, STROMERTRAG_list, FLAECHE_list, AUSRICHTUNG_list, NEIGUNG_list = [], [], [], [], [], []

                    for i, row in sub_egiddfuid.to_pandas().iterrows():
                        egid_list.append(row['EGID'])
                        dfuid_list.append(row['df_uid'])
                        STROMERTRAG_list.append(row['STROMERTRAG'])
                        FLAECHE_list.append(row['FLAECHE'])
                        AUSRICHTUNG_list.append(row['AUSRICHTUNG'])
                        NEIGUNG_list.append(row['NEIGUNG'])
                        
                    topo_egid_df = pd.DataFrame({
                        'EGID': egid_list,
                        'df_uid': dfuid_list,
                        'STROMERTRAG': STROMERTRAG_list,
                        'FLAECHE': FLAECHE_list,
                        'AUSRICHTUNG': AUSRICHTUNG_list, 
                        'NEIGUNG': NEIGUNG_list, 
                    })

                    # unsuitable variable naming ("pick(ed)") because it is copied from algo_Select_AND_adjust_topology_OPTIMIZED()
                    topo_pick_df = topo_egid_df.sort_values(by=['STROMERTRAG', ], ascending = [False,])
                    inst_power = pred_instPower
                    # inst_power = picked_flaeche * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available
                    picked_flaeche = inst_power / (self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available)
                    remaining_flaeche = picked_flaeche

                    for i in range(0, topo_pick_df.shape[0]):
                        dfuid_flaeche = topo_pick_df['FLAECHE'].iloc[i]
                        dfuid_inst_power = dfuid_flaeche * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available

                        total_ratio = remaining_flaeche / dfuid_flaeche
                        flaeche_ratio = 1       if total_ratio >= 1 else total_ratio
                        remaining_flaeche -= topo_pick_df['FLAECHE'].iloc[i]
                            
                        idx = topo_pick_df.index[i]
                        topo_pick_df.loc[idx, 'share_pvprod_used'] = flaeche_ratio                     if flaeche_ratio > 0.0 else 0.0
                        topo_pick_df.loc[idx, 'inst_TF']           = True                              if flaeche_ratio > 0.0 else False
                        topo_pick_df.loc[idx, 'TotalPower']        = inst_power  
                        topo_pick_df.loc[idx, 'dfuidPower']        = flaeche_ratio * dfuid_inst_power  if flaeche_ratio > 0.0 else 0.0

                    df_uid_w_inst = [dfuid for dfuid in topo_pick_df['df_uid'] if topo_pick_df.loc[topo_pick_df['df_uid'] == dfuid, 'inst_TF'].values[0] ]
                    df_uid_w_inst_str = '_'.join([str(dfuid) for dfuid in df_uid_w_inst])


                    # calculate selfconsumption + netdemand -----------------
                    topo_pick_pl = pl.from_pandas(topo_pick_df)
                    egid_subdf = egid_subdf.drop(['share_pvprod_used', 'inst_TF', 'TotalPower' ])
                    egid_subdf = egid_subdf.join(topo_pick_pl.select(['EGID', 'df_uid', 'share_pvprod_used', 'inst_TF', 'TotalPower' ]), on=['EGID', 'df_uid'], how='left')

                    egid_subdf = egid_subdf.with_columns([
                        (pl.col("poss_pvprod_kW") * pl.col("share_pvprod_used")).alias("pvprod_kW")
                    ])


                    egid_agg = egid_subdf.group_by(['EGID', 't', 't_int' ]).agg([
                        pl.lit(df_uid_w_inst_str).alias('df_uid_winst'), 
                        pl.col('df_uid').count().alias('n_dfuid'),
                        pl.col('bfs').first().alias('bfs'),
                        pl.col('grid_node').first().alias('grid_node'),
                        # pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                        # pl.col('pvtarif_Rp_kWh').first().alias('pvtarif_Rp_kWh'), 
                        # pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),
                        pl.col('TotalPower').first().alias('TotalPower'),
                        pl.col('AUSRICHTUNG').mean().alias('AUSRICHTUNG_mean'),
                        pl.col('NEIGUNG').mean().alias('NEIGUNG_mean'),
                        # ((pl.col('NEIGUNG') * pl.col('FLAECHE')).sum() / pl.col('FLAECHE').sum()).alias('NEIGUNG_wmean'),

                        pl.col('FLAECHE').sum().alias('FLAECHE'),
                        pl.col('poss_pvprod_kW').sum().alias('poss_pvprod_kW'),
                        pl.col('demand_kW').first().alias('demand_kW'),
                        pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                    ])

                    # ----------------------------------
                    #sanity check
                    egid_subdf.filter(pl.col('t').is_in(['t_10', 't_11', 't_12', 't_13'])).select(['EGID', 'df_uid', 'share_pvprod_used', 'poss_pvprod_kW', 'inst_TF', 'pvprod_kW', 't'])
                    egid_agg.filter(pl.col('t').is_in(['t_10', 't_11'])).select(['EGID', 'poss_pvprod_kW', 'pvprod_kW', 't'])
                    # ----------------------------------


                    # calc selfconsumption --------------------
                    egid_agg = egid_agg.sort(['EGID', 't_int'], descending = [False, False])

                    selfconsum_expr = pl.min_horizontal([pl.col("pvprod_kW"), pl.col("demand_kW")]) * self.sett.TECspec_self_consumption_ifapplicable

                    egid_agg = egid_agg.with_columns([        
                        selfconsum_expr.alias("selfconsum_kW"),
                        (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
                        (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
                    ])

                    # Apply per-household peak shaving for later iterations so NPV reflects withheld feed-in.
                    if i_m > 1 and GRID_peak_shave_TF and gridnode_df_peakshave is not None:
                        egid_agg = egid_agg.join(gridnode_df_peakshave, on=['grid_node', 't'], how='left')
                        egid_agg = egid_agg.with_columns([
                            (pl.col('max_demand_feedin_atnode_kW') / pl.col('kW_threshold')).alias('HC_ratio'), 
                        ])
                        # apply peak shaving in relevant hours
                        egid_agg = (
                            egid_agg
                            .with_columns(
                                pl.col('netfeedin_kW').alias('netfeedin_prepeakshave_kW')
                            )
                            .with_columns([
                                pl.when(
                                    (pl.col('HC_ratio').is_not_null()) &
                                    (pl.col('HC_ratio') > GRID_peak_shave_crit_level)
                                )
                                .then(
                                    pl.col('netfeedin_prepeakshave_kW') *
                                    pl.lit(GRID_peak_shave_reduct_fact)
                                )
                                .otherwise(pl.col('netfeedin_prepeakshave_kW'))
                                .alias('netfeedin_kW'),

                                pl.when(
                                    (pl.col('HC_ratio').is_not_null()) &
                                    (pl.col('HC_ratio') > GRID_peak_shave_crit_level)
                                )
                                .then(
                                    pl.col('netfeedin_prepeakshave_kW') *
                                    (pl.lit(1.0) - pl.lit(GRID_peak_shave_reduct_fact))
                                )
                                .otherwise(0.0)
                                .alias('peak_shaved_kW')
                            ])
                        )
                        # egid_agg = egid_agg.with_columns([
                        #     (pl.col('max_demand_feedin_atnode_kW') / pl.col('kW_threshold')).alias('HC_ratio'),
                        #     pl.col('netfeedin_kW').alias('netfeedin_prepeakshave_kW'),
                        # ])
                        # egid_agg = egid_agg.with_columns([
                        #     pl.when(
                        #         (pl.col('HC_ratio').is_not_null()) & (pl.col('HC_ratio') > GRID_peak_shave_crit_level)
                        #     )
                        #     .then(pl.col('netfeedin_prepeakshave_kW') * pl.lit(GRID_peak_shave_reduct_fact))
                        #     .otherwise(pl.col('netfeedin_prepeakshave_kW'))
                        #     .alias('netfeedin_kW'),
                        #     pl.when(
                        #         (pl.col('HC_ratio').is_not_null()) & (pl.col('HC_ratio') > GRID_peak_shave_crit_level)
                        #     )
                        #     .then(pl.col('netfeedin_prepeakshave_kW') * (pl.lit(1.0) - pl.lit(GRID_peak_shave_reduct_fact)))
                        #     .otherwise(0.0)
                        #     .alias('peak_shaved_kW'),
                        # ])

                        for col in ['max_demand_feedin_atnode_kW', 'kW_threshold', 'HC_ratio']:
                            if col in egid_agg.columns:
                                egid_agg = egid_agg.drop(col)
                    else:
                        egid_agg = egid_agg.with_columns([
                            pl.col('netfeedin_kW').alias('netfeedin_prepeakshave_kW'),
                            pl.lit(0.0).alias('peak_shaved_kW'),
                        ])


                    # add ACTUAL electricity price + pvtarif for NPV (per hour if the setting is a parquet) --------------------
                    egid_agg = self.add_elecpri_to_subdf(egid_agg, elecpri_vese, elecpri_entsoe, hourly=True)
                    egid_agg = self.add_pvtarif_to_subdf(egid_agg, pvtarif_vese, pvtarif_entsoe, hourly=True)


                    # calc econ spend/inc chf --------------------
                    egid_agg = egid_agg.with_columns([
                        ((pl.col("netfeedin_kW") * pl.col("pvtarif_Rp_kWh")) / 100 + (pl.col("selfconsum_kW") * pl.col("elecpri_Rp_kWh")) / 100).alias("econ_inc_chf")
                    ])
                        
                    egid_agg = egid_agg.with_columns([
                        (pl.lit(0.0)).alias("econ_spend_chf")
                    ])


                    # NPV calculation -----------------
                    estim_instcost_chfpkW, estim_instcost_chftotal = self.initial_sml_get_instcost_interpolate_function()
                    estim_instcost_chftotal(pd.Series([10, 20, 30, 40, 50, 60, 70]))

                    annual_cashflow = (egid_agg["econ_inc_chf"].sum() - egid_agg["econ_spend_chf"].sum())
                    installation_cost = estim_instcost_chftotal(pred_instPower) * (1-self.sett.TECspec_KEVflatsubs)
                               
                    discount_factor = np.array([(1 + self.sett.TECspec_interest_rate)**-i for i in range(1, self.sett.TECspec_invst_maturity + 1)])
                    disc_cashflow = annual_cashflow * np.sum(discount_factor)
                    npv = -installation_cost + disc_cashflow

                    egid_npv = egid_agg.group_by(['EGID', ]).agg([
                        pl.col('df_uid_winst').first().alias('df_uid_winst'),
                        pl.col('n_dfuid').first().alias('n_dfuid'),
                        pl.col('grid_node').first().alias('grid_node'),
                        # pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                        pl.col('pvtarif_Rp_kWh').mean().alias('pvtarif_Rp_kWh'),            # mean: with hourly pvtarif first() would only report hour 1 
                        # pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),
                        pl.col('TotalPower').first().alias('TotalPower'),
                        pl.col('AUSRICHTUNG_mean').first().alias('AUSRICHTUNG_mean'),
                        pl.col('NEIGUNG_mean').first().alias('NEIGUNG_mean'),
                        # pl.col('NEIGUNG_wmean').first().alias('NEIGUNG_wmean'),
                        pl.col('FLAECHE').first().alias('FLAECHE'),
                          
                        pl.col('poss_pvprod_kW').sum().alias('poss_pvprod_kW'),
                        pl.col('demand_kW').sum().alias('demand_kW'),
                        pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                        pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                        pl.col('netfeedin_prepeakshave_kW').sum().alias('netfeedin_prepeakshave_kW'),
                        pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                        pl.col('peak_shaved_kW').sum().alias('peak_shaved_kW'),
                        pl.col('netdemand_kW').sum().alias('netdemand_kW'),
                        pl.col('econ_inc_chf').sum().alias('econ_inc_chf'),
                        pl.col('econ_spend_chf').sum().alias('econ_spend_chf'),
                        ])
                    egid_npv = egid_npv.with_columns([
                        pl.lit(pred_instPower).alias("pred_instPower"),
                        pl.lit(installation_cost).alias("estim_pvinstcost_chf"),
                        pl.lit(disc_cashflow).alias("disc_cashflow"),
                        pl.lit(npv).alias("NPV_uid"),
                        ])
                        
                    agg_npv_df_list.append(egid_npv)

        
        # concat all egid_agg
        npv_df = pl.concat(agg_npv_df_list).clone()


        # add roof specific filter tag VERSION 2 -----------------
        topo_filter_egid = self.get_topo_filter_tag_mapping(topo)

        npv_df = npv_df.join(topo_filter_egid, on='EGID', how='left')

            
        # add subsidy where applicable -----------------
        node_subsidy_monitor_iter = node_subsidy_monitor_dict[str(i_m)]
        subsidy_node_name   = self.sett.GRIDspec_subsidy_name
        subsidy_filtag_node_schemes = self.sett.GRIDspec_subsidy_filtag_node_schemes[subsidy_node_name]
        npv_df = npv_df.with_columns([
            pl.lit(0.0).alias('subs_filter_tags_chf'),
            pl.lit(0.0).alias('subs_nodeHC_chf'),
            pl.lit(0.0).alias('pena_nodeHC_chf'),
        ])

        # filter_tag subsidy ----
        subs_filter_tags_chf_tuples = subsidy_filtag_node_schemes['subs_filter_tags_chf_tuples']  # ['filter_tags_subs_chf_tuples']
        for filt_tag, subs_chf in reversed(subs_filter_tags_chf_tuples):
            npv_df = npv_df.with_columns([
                pl.when(pl.col(filt_tag) == True
                        ).then(pl.lit(subs_chf)
                ).otherwise(pl.col('subs_filter_tags_chf')
                ).alias('subs_filter_tags_chf'),
            ])

        # node subsidy / penalty ----
        subs_node_tpl = subsidy_filtag_node_schemes['subs_nodeHC_chf_tuples']
        pena_node_tpl = subsidy_filtag_node_schemes['pena_nodeHC_chf_tuples']

        npv_df = npv_df.with_columns([
            pl.when(pl.col('grid_node').is_in(node_subsidy_monitor_iter['subs_gridnodes'])
                    ).then(pl.lit(subs_node_tpl[1])
            ).otherwise(pl.col('subs_nodeHC_chf')
            ).alias('subs_nodeHC_chf'),

            pl.when(pl.col('grid_node').is_in(node_subsidy_monitor_iter['pena_gridnodes'])
                    ).then(pl.lit(pena_node_tpl[1])
            ).otherwise(pl.col('pena_nodeHC_chf')
            ).alias('pena_nodeHC_chf'),
        ])

            
        # apply subsidy / penality ----
        npv_df = npv_df.with_columns([
            pl.col('NPV_uid').alias('NPV_uid_before_subsidy'),
            (pl.col('NPV_uid') + pl.col('subs_filter_tags_chf') + pl.col('subs_nodeHC_chf') - pl.col('pena_nodeHC_chf')
            ).alias('NPV_uid'),
        ])  


        # export npv_df -----------------------------------------------------
        npv_df.write_parquet(f'{subdir_path}/npv_df.parquet')
        if (self.sett.export_csvs) & ( i_m < 3):
            npv_df.write_csv(f'{subdir_path}/npv_df.csv')


        # export by Month -----------------------------------------------------
        if self.sett.MCspec_keep_files_month_iter_TF:
            if i_m < self.sett.MCspec_keep_files_month_iter_max:
                pred_npv_inst_by_M_path = f'{subdir_path}/pred_npv_inst_by_M'
                if not os.path.exists(pred_npv_inst_by_M_path):
                    os.makedirs(pred_npv_inst_by_M_path)

                npv_df.write_parquet(f'{pred_npv_inst_by_M_path}/npv_df_{i_m}.parquet')

                if self.sett.export_csvs:
                    npv_df.write_csv(f'{pred_npv_inst_by_M_path}/npv_df_{i_m}.csv')               
                    
        checkpoint_to_logfile('exported npv_df', self.sett.log_name, 0)





    def algo_select_AND_adjust_topology_RFR(self, subdir_path: str, i_m: int, m, while_safety_counter: int = 0, require_positive_npv: bool = True):

        checkpoint_to_logfile('run function: algo_select_AND_adjust_topology_RFR', self.sett.log_name, 0, True) if while_safety_counter < 5 else None

        # import ----------------
        with open(f'{subdir_path}/topo_egid.json', 'r', encoding='utf-8') as f:
            topo = json.load(f)
        npv_df = pd.read_parquet(f'{subdir_path}/npv_df.parquet')
        pred_inst_df = pd.read_parquet(f'{subdir_path}/pred_inst_df.parquet') if os.path.exists(f'{subdir_path}/pred_inst_df.parquet') else pd.DataFrame()

        with open(f'{subdir_path}/node_1hll_closed_dict.json', 'r', encoding='utf-8') as f:
            node_1hll_closed_dict = json.load(f)


        #  remove all EGIDs with pv ----------------
        egid_without_pv = [k for k,v in topo.items() if not v['pv_inst']['inst_TF']]
        npv_df = copy.deepcopy(npv_df.loc[npv_df['EGID'].isin(egid_without_pv)])         


        #  remove all closed nodes EGIDs if applicable ----------------
        if self.sett.GRIDspec_node_1hll_closed_TF:
            closed_nodes = node_1hll_closed_dict[str(i_m)]['all_nodes_abv_1hll']
            closed_nodes_egid = [k for k, v in topo.items() if v.get('grid_node')  in closed_nodes ]

            npv_df = copy.deepcopy(npv_df.loc[~npv_df['EGID'].isin(closed_nodes_egid)])


        #  remove all EGIDs with NPV <= 0 (MC algorithm + sanity check; the grid optimized expansion passes require_positive_npv=False) ----------------
        if require_positive_npv:
            npv_df = copy.deepcopy(npv_df.loc[npv_df['NPV_uid'] > 0])

        if npv_df.shape[0] == 0:
            checkpoint_to_logfile('no free EGID to select (with NPV > 0), no installation built', self.sett.log_name, 0, True)
            return 0.0


        #  SUBSELECTION FILTER specific scenarios ----------------
        if self.sett.ALGOspec_subselec_filter_criteria is not None:
            # export geojson for sanity check  ---
            if (i_m == 1) and ('sanity' not in subdir_path):
                gwr_all_building_gdf = gpd.read_file(f'{self.sett.name_dir_import_path}/gwr_all_building_gdf.geojson')
                gwr_filter_tag_facing_gdf = gwr_all_building_gdf.merge(npv_df, on = 'EGID', how = 'inner')
                filter_tag_list = [col for col in npv_df.columns if 'filter_tag__' in col ]
                for filter_tag in filter_tag_list:
                    gwr_filter_tag = gwr_filter_tag_facing_gdf.loc[gwr_filter_tag_facing_gdf[filter_tag] == True]
                    with open(f'{self.sett.name_dir_export_path}/gwr_{filter_tag}_gdf.geojson', 'w') as f:
                        f.write(gwr_filter_tag.to_json())
            # ------
            if self.sett.ALGOspec_subselec_filter_method == 'ordered':
                subselec_npv_df_empty = True
                for filter_tag in self.sett.ALGOspec_subselec_filter_criteria: 
                    subselec_npv_df = npv_df.loc[npv_df[filter_tag] == True]
                    if subselec_npv_df_empty and subselec_npv_df.shape[0] > 0:
                        npv_df = copy.deepcopy(subselec_npv_df)
                        subselec_npv_df_empty = False

            elif self.sett.ALGOspec_subselec_filter_method == 'pooled':
                mask = np.zeros(npv_df.shape[0], dtype=bool)
                for filter_tag in self.sett.ALGOspec_subselec_filter_criteria: 
                    mask = mask | (npv_df[filter_tag] == True)
                subselec_npv_df = npv_df.loc[mask]
                if subselec_npv_df.shape[0] > 0:
                    npv_df = copy.deepcopy(subselec_npv_df)

            
        #  SUBSELECTION GRIDOPTIM installation order scenario ----------------
        if self.sett.OPTEXPApecs_apply_gridoptim_order_TF and 'OptimExpa' in subdir_path:
            min_grid_optim_inst_order = npv_df['grid_optim_inst_order'].min()
            npv_df = copy.deepcopy(npv_df.loc[npv_df['grid_optim_inst_order'] == min_grid_optim_inst_order])

                    

        # SELECTION BY METHOD ---------------
        # set random seed
        if self.sett.ALGOspec_rand_seed is not None:
            np.random.seed(self.sett.ALGOspec_rand_seed)

        # have a list of egids to install on for sanity check. If all build, start building on the rest of EGIDs
        install_EGIDs_summary_sanitycheck = self.sett.CHECKspec_egid_list


        # installation selelction ---------------
        if self.sett.ALGOspec_inst_selection_method == 'random':
            npv_pick = npv_df.sample(n=1).copy()
            
        elif self.sett.ALGOspec_inst_selection_method == 'max_npv':
            npv_pick = npv_df[npv_df['NPV_uid'] == max(npv_df['NPV_uid'])].copy()

        elif self.sett.ALGOspec_inst_selection_method == 'prob_weighted_npv':
            rand_num = np.random.uniform(0, 1)
                
            npv_df['NPV_stand'] = npv_df['NPV_uid'] / max(npv_df['NPV_uid'])
            npv_df['diff_NPV_rand'] = abs(npv_df['NPV_stand'] - rand_num)
            npv_pick = npv_df[npv_df['diff_NPV_rand'] == min(npv_df['diff_NPV_rand'])].copy()
                
            # if multiple rows at min to rand num 
            if npv_pick.shape[0] > 1:
                rand_row = np.random.randint(0, npv_pick.shape[0])
                npv_pick = npv_pick.iloc[rand_row]

        # ---------------------------------------------            
            

        # extract selected inst info -----------------
        if isinstance(npv_pick, pd.DataFrame):
            picked_egid              = npv_pick['EGID'].values[0]
            picked_power             = npv_pick['pred_instPower'].values[0]
            df_uid_winst             = npv_pick['df_uid_winst'].values[0]
            picked_flaeche           = npv_pick['FLAECHE'].values[0]
            picked_npv               = npv_pick['NPV_uid'].values[0]
            picked_demand_kW         = npv_pick['demand_kW'].values[0]
            picked_poss_pvprod       = npv_pick['poss_pvprod_kW'].values[0]
            picked_pvprod_kW         = npv_pick['pvprod_kW'].values[0]
            picked_selfconsum_kW     = npv_pick['selfconsum_kW'].values[0]
            picked_netfeedin_kW      = npv_pick['netfeedin_kW'].values[0]
            picked_netdemand_kW      = npv_pick['netdemand_kW'].values[0]

            picked_grid_node         = topo[picked_egid]['grid_node']

            topo_pick_df = npv_pick
                
        elif isinstance(npv_pick, pd.Series):
            picked_egid  = npv_pick['EGID']
            picked_power = npv_pick['pred_instPower']
            df_uid_winst = npv_pick['df_uid_winst']
            picked_flaeche = npv_pick['FLAECHE']
            picked_npv               = npv_pick['NPV_uid']
            picked_demand_kW         = npv_pick['demand_kW']
            picked_poss_pvprod       = npv_pick['poss_pvprod_kW']
            picked_pvprod_kW         = npv_pick['pvprod_kW']
            picked_selfconsum_kW     = npv_pick['selfconsum_kW']
            picked_netfeedin_kW      = npv_pick['netfeedin_kW']
            picked_netdemand_kW      = npv_pick['netdemand_kW']

            picked_grid_node         = topo[picked_egid]['grid_node']

            topo_pick_df = pd.DataFrame(npv_pick).T


        # distribute kWp to partition(s) -----------------
        egid_list, dfuid_list, STROMERTRAG_list, FLAECHE_list, AUSRICHTUNG_list, NEIGUNG_list = [], [], [], [], [], []
        topo_egid = {picked_egid: topo[picked_egid].copy()}
        for k,v in topo_egid.items():
            for sub_k, sub_v in v['solkat_partitions'].items():
                egid_list.append(k)
                dfuid_list.append(sub_k)
                STROMERTRAG_list.append(sub_v['STROMERTRAG'])
                FLAECHE_list.append(sub_v['FLAECHE'])
                AUSRICHTUNG_list.append(sub_v['AUSRICHTUNG'])
                NEIGUNG_list.append(sub_v['NEIGUNG'])
            
        topo_egid_df = pd.DataFrame({
            'EGID': egid_list,
            'df_uid': dfuid_list,
            'STROMERTRAG': STROMERTRAG_list,
            'FLAECHE': FLAECHE_list,
            'AUSRICHTUNG': AUSRICHTUNG_list, 
            'NEIGUNG': NEIGUNG_list, 
            })

        topo_pick_df = topo_egid_df.sort_values(by=['STROMERTRAG', ], ascending = [False,])
        inst_power = picked_power
        # inst_power = picked_flaeche * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available
        picked_flaeche = inst_power / (self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available)
        remaining_flaeche = picked_flaeche

        # add empty cols to fill in later
        cols_to_add = ['inst_TF', 'info_source', 'xtf_id', 'BeginOp', 'dfuidPower', 
                       'share_pvprod_used', 'demand_kW', 'poss_pvprod', 'pvprod_kW', 
                       'selfconsum_kW', 'netfeedin_kW', 'netdemand_kW', 
                       ]
        for col in cols_to_add: 
            if col not in topo_pick_df.columns:
                if col in ['inst_TF']:                              # boolean
                    topo_pick_df[col] = False
                elif col in ['info_source', 'xtf_id', 'BeginOp']:   # string
                    topo_pick_df[col] = ''
                else:                                               # numeric                    
                    topo_pick_df[col] = np.nan

        for i in range(0, topo_pick_df.shape[0]):
            dfuid_flaeche = topo_pick_df['FLAECHE'].iloc[i]
            dfuid_inst_power = dfuid_flaeche * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available

            total_ratio = remaining_flaeche / dfuid_flaeche
            flaeche_ratio = 1       if total_ratio >= 1 else total_ratio
            remaining_flaeche -= topo_pick_df['FLAECHE'].iloc[i]

            idx = topo_pick_df.index[i]

            topo_pick_df.loc[idx, 'inst_TF'] =             True                                   if flaeche_ratio > 0.0 else False
            topo_pick_df.loc[idx, 'share_pvprod_used'] =   flaeche_ratio                          if flaeche_ratio > 0.0 else 0.0
            topo_pick_df.loc[idx, 'info_source'] =         'alloc_algorithm'                      if flaeche_ratio > 0.0 else ''
            topo_pick_df.loc[idx, 'BeginOp'] =             str(m)                                 if flaeche_ratio > 0.0 else ''
            topo_pick_df.loc[idx, 'iter_round'] =          i_m                                    if flaeche_ratio > 0.0 else -1
            topo_pick_df.loc[idx, 'NPV_uid'] =             picked_npv                             if flaeche_ratio > 0.0 else 0.0
            topo_pick_df.loc[idx, 'grid_node'] =           picked_grid_node                       if flaeche_ratio > 0.0 else ''
            topo_pick_df.loc[idx, 'xtf_id'] =              df_uid_winst                           if flaeche_ratio > 0.0 else ''
            topo_pick_df.loc[idx, 'dfuidPower'] =          dfuid_inst_power                       if flaeche_ratio > 0.0 else 0.0
            topo_pick_df.loc[idx, 'demand_kW'] =           picked_demand_kW                       if flaeche_ratio > 0.0 else 0.0
            topo_pick_df.loc[idx, 'poss_pvprod'] =         picked_poss_pvprod                     if flaeche_ratio > 0.0 else 0.0
            topo_pick_df.loc[idx, 'pvprod_kW'] =           picked_pvprod_kW                       if flaeche_ratio > 0.0 else 0.0
            topo_pick_df.loc[idx, 'selfconsum_kW'] =       picked_selfconsum_kW                   if flaeche_ratio > 0.0 else 0.0
            topo_pick_df.loc[idx, 'netfeedin_kW'] =        picked_netfeedin_kW                    if flaeche_ratio > 0.0 else 0.0
            topo_pick_df.loc[idx, 'netdemand_kW'] =        picked_netdemand_kW                    if flaeche_ratio > 0.0 else 0.0

            if 'OptimExpa' in subdir_path:
                topo_pick_df.loc[idx, 'grid_optim_inst_order'] = npv_pick.loc[npv_pick['EGID'] == picked_egid, 'grid_optim_inst_order'].values[0] 


            
        topo_pick_df = topo_pick_df.loc[topo_pick_df['inst_TF'] == True].copy()
        pred_inst_df = pd.concat([pred_inst_df, topo_pick_df], ignore_index=True)


        # Adjust topo + npv_df -----------------
        dfuid_w_inst_tuples = []
        for _, row in topo_pick_df.iterrows():
            tpl = ('tuple_names: df_uid_inst, share_pvprod_used, kWpeak', 
                                    row['df_uid'], row['share_pvprod_used'], row['dfuidPower'] )
            dfuid_w_inst_tuples.append(tpl)

        topo[picked_egid]['pv_inst'] = {'inst_TF': True, 
                                        'info_source': 'alloc_algorithm', 
                                        'xtf_id': df_uid_winst, 
                                        'BeginOp': f'{m}', 
                                        'TotalPower': inst_power, 
                                        'dfuid_w_inst_tuples': dfuid_w_inst_tuples
                                        }


        # export main dfs ------------------------------------------
        pred_inst_df.to_parquet(f'{subdir_path}/pred_inst_df.parquet')
        pred_inst_df.to_csv(f'{subdir_path}/pred_inst_df.csv') if self.sett.export_csvs else None
        with open (f'{subdir_path}/topo_egid.json', 'w') as f:
            json.dump(topo, f)


        # export by Month ------------------------------------------
        os.makedirs(f'{subdir_path}/pred_npv_inst_by_M', exist_ok=True)
        pred_inst_df.to_parquet(f'{subdir_path}/pred_npv_inst_by_M/pred_inst_df_{i_m}.parquet')
        pred_inst_df.to_csv(f'{subdir_path}/pred_npv_inst_by_M/pred_inst_df_{i_m}.csv') if self.sett.export_csvs else None
        with open(f'{subdir_path}/pred_npv_inst_by_M/topo_{i_m}.json', 'w') as f:
            json.dump(topo, f)

        return  inst_power    #, npv_df  # , picked_uid, picked_combo_uid, pred_inst_df, dfuid_installed_list, topo

