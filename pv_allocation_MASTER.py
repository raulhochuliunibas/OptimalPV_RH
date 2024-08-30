# -----------------------------------------------------------------------------
# pv_allocation_MASTER.py 
# -----------------------------------------------------------------------------
# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024
# > description: 


# SETTIGNS --------------------------------------------------------------------
pvalloc_settings = {
        'name_dir_export': 'pvalloc_BSBLSO_wrkn_prgrss',              # name of the directory where all proccessed data is stored at the end of the code file 
        'name_dir_import': 'preprep_BSBLSO_18to22_20240826_22h', # name of the directory where preprepared data is stored and accessed by the code
        'script_run_on_server': False,                           # F: run on private computer, T: run on server
        'fast_debug_run': False,                                 # T: run the code with a small subset of data, F: run the code with the full dataset
        'show_debug_prints': True,                              # F: certain print statements are omitted, T: includes print statements that help with debugging
        'n_egid_in_topo': 200, 
        'wd_path_laptop': 'C:/Models/OptimalPV_RH',              # path to the working directory on Raul's laptop
        'wd_path_server': 'D:/RaulHochuli_inuse/OptimalPV_RH',   # path to the working directory on the server

        'kt_numbers': [11,12,13,], #[11,12,13],                           # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain indiviual municipalities
        'bfs_numbers': [2549, 2574, 2612, 2541, 2445, 2424, 2463, 2524, 2502, 2492], # list of bfs numbers to be considered
        
        # 'topology_year_range':[2019, 2022],
        # 'prediction_year_range':[2023, 2025],
        'T0_prediction': '2023-01-01 00:00:00', 
        'months_lookback': 12*1,
        'months_prediction': 12*2,
        'recreate_topology':            True, 
        'recalc_economics_topo_df':     True,
        'create_map_of_topology':       True,
        'recalc_npv_all_combinations':  True,

        'test_faster_if_subdf_deleted': False,

        'algorithm_specs': {
            'rand_seed': 42, 
            'safety_counter_max': 5000,
            'capacity_tweak_fact': 1, 
            'topo_subdf_partitioner': 1000,
        },
        'gridprem_adjustment_specs': {
            'voltage_assumption': '',
            'tier_description': 'tier_level: (voltage_threshold, gridprem_plusRp_kWh)',
            'colnames': ['tier_level', 'vltg_threshold', 'gridprem_plusRp_kWh'],
            'tiers': { 
                1: [0.5, 1], 
                2: [0.7, 3],
                4: [0.9, 7],
                5: [0.95, 13], 
                6: [1, 20],
                },},
        'tech_economic_specs': {
            'interest_rate': 0.01,
            'pvtarif_year': 2022, 
            'pvtarif_col': ['energy1', 'eco1'],
            'elecpri_year': 2022,
            'elecpri_category': 'H8', 
            'invst_maturity': 25,
            'self_consumption_ifapplicable': 1,
            'conversion_m2tokW': 0.1,  # A 1m2 area can fit 0.1 kWp of PV Panels
            },
        'weather_specs': {
            'meteoblue_col_radiation_proxy': 'Basel Direct Shortwave Radiation',
            'weather_year': 2022,
        },
        'constr_capacity_specs': {
            'ann_capacity_growth': 0.1,         # annual growth of installed capacity# each year, X% more PV capacity can be built, 100% in year T0
            'summer_months': [4,5,6,7,8,9,],
            'winter_months': [10,11,12,1,2,3,],
            'share_to_summer': 0.6, 
            'share_to_winter': 0.4,
        },
        'gwr_selection_specs': {
            'solkat_max_n_partitions': 20,
            'building_cols': ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                        'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                        'GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1', 'GEBF', 'GAREA'],
            'dwelling_cols': None, # ['EGID', 'WAZIM', 'WAREA', ],
            'DEMAND_proxy': 'GAREA',
            'GSTAT': ['1004',],                 # GSTAT - 1004: only existing, fully constructed buildings
            'GKLAS': ['1110','1121','1276',],                 # GKLAS - 1110: only 1 living space per building
            'GBAUJ_minmax': [1950, 2023],       # GBAUJ_minmax: range of years of construction
            # 'GWAERZH': ['7410', '7411',],       # GWAERZH - 7410: heat pumpt for 1 building, 7411: heat pump for multiple buildings
            # 'GENH': ['7580', '7581', '7582'],   # GENHZU - 7580 to 7582: any type of Fernwärme/district heating        
                                                # GANZWHG - total number of apartments in building
                                                # GAZZI - total number of rooms in building
            },
        'assumed_parameters': {
        },

        'topo_type': 1,              # 1: all data, all egid  2: all data, only egid in solkat,  3: only partitions + Mappings, all egid, 4: only partitions + Mappings, only egid in solkat
        'rate_operation_cost': 0.01,                # assumed rate of operation cost (of investment cost)
        'NPV_include_wealth_tax': False,            # F: exclude wealth tax from NPV calculation, T: include wealth tax in NPV calculation
        'solkat_house_type_class': [0,],            # list of house type classes to be considered
         }



# PACKAGES --------------------------------------------------------------------
import sys
sys.path.append(pvalloc_settings['wd_path_laptop']) if pvalloc_settings['script_run_on_server'] else sys.path.append(pvalloc_settings['wd_path_server'])

# external packages
import os as os
import pandas as pd
import geopandas as gpd
import numpy as np
import dask.dataframe as dd
from datetime import datetime
from pprint import pformat

import glob
import shutil
import winsound
import subprocess
import pprint

# own packages and functions
import auxiliary_functions
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile, get_bfs_from_ktnr, format_MASTER_settings
import pv_allocation.initialization as  initial
import pv_allocation.alloc_algorithm as algo
import pv_allocation.topo_visualization as visual

from pv_allocation.initialization import *
from pv_allocation.alloc_algorithm import *
from pv_allocation.topo_visualization import *



# SETUP ================================================================
# set working directory
wd_path = pvalloc_settings['wd_path_laptop'] if not pvalloc_settings['script_run_on_server'] else pvalloc_settings['wd_path_server']
data_path = f'{wd_path}_data'

# create directory + log file
pvalloc_path = f'{data_path}/output/pvalloc_run'
if not os.path.exists(pvalloc_path):
    os.makedirs(pvalloc_path)
log_name = f'{data_path}/output/pvalloc_log.txt'
total_runtime_start = datetime.now()


# extend settings dict with relevant informations for later functions
if not not pvalloc_settings['kt_numbers']:
    pvalloc_settings['bfs_numbers'] = auxiliary_functions.get_bfs_from_ktnr(pvalloc_settings['kt_numbers'], data_path, log_name)
    print_to_logfile(f' > no. of kt  numbers in selection: {len(pvalloc_settings["kt_numbers"])}', log_name)
    print_to_logfile(f' > no. of bfs numbers in selection: {len(pvalloc_settings["bfs_numbers"])}', log_name) 

elif (not pvalloc_settings['kt_numbers']) and (not not pvalloc_settings['bfs_numbers']):
    pvalloc_settings['bfs_numbers'] = [str(bfs) for bfs in pvalloc_settings['bfs_numbers']]

pvalloc_settings['log_file_name'] = log_name
pvalloc_settings['wd_path'] = wd_path
pvalloc_settings['data_path'] = data_path
pvalloc_settings['pvalloc_path'] = pvalloc_path
interim_path = get_interim_path(pvalloc_settings)
pvalloc_settings['interim_path'] = interim_path
show_debug_prints = pvalloc_settings['show_debug_prints']
conv_m2toKWP = pvalloc_settings['tech_economic_specs']['conversion_m2tokW']


# pvalloc_settings['disc_rate'] = (pvalloc_settings['assumed_parameters']['interest_rate'] + pvalloc_settings['assumed_parameters']['inflation_rate'] )
# pvalloc_settings['disc_denom'] = np.sum((1+pvalloc_settings['disc_rate'])** np.arange(1, pvalloc_settings['assumed_parameters']['invest_maturity']+1))

chapter_to_logfile(f'start pv_allocation_MASTER for: {pvalloc_settings["name_dir_export"]}', log_name, overwrite_file=True)
formated_pvalloc_settings = format_MASTER_settings(pvalloc_settings)
print_to_logfile(f'pvalloc_settings: \n{pformat(formated_pvalloc_settings)}', log_name)



angle_tilt_df = initial.get_angle_tilt_table(pvalloc_settings)
Map_egid_nodes = initial.get_fake_gridnodes(pvalloc_settings)

# INITIALIZATION ================================================================
if pvalloc_settings['recreate_topology']:
    subchapter_to_logfile('initialization: IMPORT PREPREP DATA & CREATE (building) TOPOLOGY', log_name)
    topo, df_list, df_names = initial.import_prepre_AND_create_topology(pvalloc_settings)

elif not pvalloc_settings['recreate_topology']:
    subchapter_to_logfile('initialization: IMPORT EXISITNG TOPOLOGY', log_name) 
    topo, df_list, df_names = initial.import_exisitng_topology(pvalloc_settings, 
                    df_search_names = ['Map_solkatdfuid_egid', 'Map_egid_pv', 'Map_demandtypes_egid', 'Map_egid_demandtypes', 'pvtarif', 'elecpri'])

subchapter_to_logfile('initialization: IMPORT TS DATA', log_name)
ts_list, ts_names = initial.import_ts_data(pvalloc_settings)

subchapter_to_logfile('initialization: DEFINE CONSTRUCTION CAPACITY', log_name)
constrcapa, months_prediction, months_lookback = define_construction_capacity(pvalloc_settings, topo, ts_list)

# df_list.append(angle_tilt_df)
# df_names.append('angle_tilt_df')





# ALLOCATION ALGORITHM ================================================================

# CALC ECONOMICS for TOPO_DF ----------------------------------------------------------------
if pvalloc_settings['recalc_economics_topo_df']:
    subchapter_to_logfile('allocation algorithm: CALC ECONOMICS for TOPO_DF', log_name)
    topo_df, topo_agg_df = algo.calc_economics_in_topo_df(pvalloc_settings, topo, df_list, df_names, ts_list, ts_names)

    """
    [Map_solkatdfuid_egid, Map_egid_pv, Map_demandtypes_egid, Map_egid_demandtypes, pvtarif, elecpri] = df_list
    [Map_daterange, demandtypes_ts, meteo_ts, gridprem_ts] = ts_list
    print(f'Reatach elements of df_list: {df_names}')
    print(f'Reatach elements of ts_list: {ts_names}')

    # NOTE: clean up when no longer necessary
    if 'high_wiHP' in demandtypes_ts.columns:
        demandtypes_ts.rename(columns={'high_wiHP': 'high_DEMANDprox_wiHP', 'low_wiHP': 'low_DEMANDprox_wiHP', 'high_noHP': 'high_DEMANDprox_noHP', 'low_noHP': 'low_DEMANDprox_noHP'}, inplace=True)


    # TOPO to DF ----------------------------------------------------------------
    # solkat_combo_df_exists = os.path.exists(f'{pvalloc_settings["interim_path"]}/solkat_combo_df.parquet')
    # if pvalloc_settings['recalc_economics_topo_df']:
    no_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == False]
    with_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == True]

    egid_list, bfs_list, gklas_list, pvinst_list, pvsource_list, pvid_list, demandtype_list = [], [], [], [], [], [], []
    pv_tarif_Rp_kWh_list, df_uid_list, flaeche_list, stromertrag_list, estim_instcost_list, elecpri_list = [], [], [], [], [], []

    keys = list(topo.keys())

    for k,v in topo.items():
        partitions = v.get('solkat_partitions')

        for k_p, v_p in partitions.items():
            egid_list.append(k)
            bfs_list.append(v.get('gwr_info').get('bfs'))
            gklas_list.append(v.get('gwr_info').get('gklas'))
            pvinst_list.append(v.get('pv_inst').get('inst_TF'))
            pvsource_list.append(v.get('pv_inst').get('info_source'))
            pvid_list.append(v.get('pv_inst').get('xtf_id'))
            demandtype_list.append(v.get('demand_type'))
            pv_tarif_Rp_kWh_list.append(v.get('pvtarif_Rp_kWh'))
            elecpri_list.append(v.get('elecpri_Rp_kWh'))
            
            df_uid_list.append(k_p)
            flaeche_list.append(v_p.get('FLAECHE'))
            stromertrag_list.append(v_p.get('STROMERTRAG'))

        
    topo_df = pd.DataFrame({'EGID': egid_list, 'bfs': bfs_list, 'gklas': gklas_list, 'pvinst_TF': pvinst_list, 
                            'pvsource': pvsource_list, 'pvid': pvid_list, 'demandtype': demandtype_list, 
                            'pv_tarif_Rp_kWh': pv_tarif_Rp_kWh_list, 'df_uid': df_uid_list, 'FLAECHE': flaeche_list,
                            'STROMERTRAG': stromertrag_list, 'elecpri_Rp_kWh': elecpri_list})

    groupby_cols = ['EGID', 'df_uid', 'FLAECHE', 'bfs', 'gklas', 'pvinst_TF', 'pvsource', 'pvid', 'pv_tarif_Rp_kWh', 'elecpri_Rp_kWh']
    agg_cols = ['econ_inc_chf', 'econ_spend_chf']




    # CREATE MAP OF TOPO_DF ----------------------------------------------------------------
    if pvalloc_settings['create_map_of_topology']:
        print_to_logfile('visualization: CREATE MAP OF TOPOLOGY_DF', log_name)
        create_map_of_topology(pvalloc_settings, topo_df)    



    # GET ECONOMIC VALUES FOR NPV CALCULATION ----------------------------------------------
    topo_subdf_partitioner = pvalloc_settings['algorithm_specs']['topo_subdf_partitioner']
    selfconsum_rate = pvalloc_settings['tech_economic_specs']['self_consumption_ifapplicable']

    subdf_list = []
    agg_subdf_list = []
    egids = topo_df['EGID'].unique()
    dfuids = topo_df['df_uid'].unique()
    subdf_path = f'{data_path}/output/pvalloc_run/topo_subdf'

    if not os.path.exists(subdf_path):
        os.makedirs(subdf_path)

    stepsize = topo_subdf_partitioner if len(dfuids) > topo_subdf_partitioner else len(dfuids)
    tranche_counter = 0
    for i in range(0, len(dfuids), stepsize):
        # print(f'  > {i} to {i+stepsize-1}'
        tranche_counter += 1
        print_to_logfile(f'topo_subdf {tranche_counter}/{len(range(0, len(dfuids), stepsize))} ({i} to {i+stepsize-1} df_uid.iloc) , calculate gains/spending per roof partition \t (stamp: {datetime.now()})', log_name)
        # print_to_logfile(f'Calculate gains/spending per roof partition, df_uid.iloc ({i} to {i+stepsize-1})', log_name)
        subdf = topo_df[topo_df['df_uid'].isin(dfuids[i:i+stepsize])].copy()


        # merge production, grid prem + demand to partitions ----------
        subdf['meteo_loc'] = 'Basel'
        meteo_ts['meteo_loc'] = 'Basel' 
        subdf = subdf.merge(meteo_ts[['t', 'radiation', 'meteo_loc']], how='left', on='meteo_loc')
        subdf = subdf.assign(pvprod = subdf['radiation'] * subdf['FLAECHE'] / 1000).drop(columns=['meteo_loc', 'radiation', 'STROMERTRAG'])
        checkpoint_to_logfile(f'  end merge meteo for subdf {i} to {i+stepsize-1}', log_name, 1)

        subdf['grid_node_loc'] = 'BS001'
        gridprem_ts['grid_node_loc'] = 'BS001'
        subdf = subdf.merge(gridprem_ts[['t', 'prem_Rp_kWh', 'grid_node_loc']], how='left', on=['t', 'grid_node_loc']).drop(columns='grid_node_loc')
        checkpoint_to_logfile(f'  end merge gridprem for subdf {i} to {i+stepsize-1}', log_name, 1)

        demandtypes_names = [c for c in demandtypes_ts.columns if 'DEMANDprox' in c]
        demandtypes_melt = demandtypes_ts.melt(id_vars='t', value_vars=demandtypes_names, var_name= 'demandtype', value_name= 'demand')
        subdf = subdf.merge(demandtypes_melt, how='left', on=['t', 'demandtype'])
        checkpoint_to_logfile(f'  end merge demandtypes for subdf {i} to {i+stepsize-1}', log_name, 1)


        # compute production + selfconsumption + netdemand ----------
        subdf['selfconsum'] = np.minimum(subdf['pvprod'], subdf['demand']) * selfconsum_rate
        subdf['netdemand'] = subdf['demand'] - subdf['selfconsum']
        subdf['netfeedin'] = subdf['pvprod'] - subdf['selfconsum']

        subdf['econ_inc_chf'] = ((subdf['netfeedin'] * subdf['pv_tarif_Rp_kWh']) /100) + ((subdf['selfconsum'] * subdf['elecpri_Rp_kWh']) /100) 
        subdf['econ_spend_chf'] = ((subdf['netfeedin'] * subdf['prem_Rp_kWh'])) + ((subdf['netdemand'] * subdf['elecpri_Rp_kWh']) /100 )
        checkpoint_to_logfile(f'  end compute econ factors for subdf {i} to {i+stepsize-1}', log_name, 1)

        # subdf_list.append(subdf)
        subdf.to_parquet(f'{subdf_path}/subdf_{i}to{i+stepsize-1}.parquet')

        if not pvalloc_settings['test_faster_if_subdf_deleted']:            
            # aggregate gains/spending per EGID for NPV calculation ----------
            agg_subdf = subdf.groupby(groupby_cols).agg({agg_cols[0]: 'sum', agg_cols[1]: 'sum'}).reset_index()
            agg_subdf_list.append(agg_subdf)
            agg_subdf.to_parquet(f'{subdf_path}/agg_subdf_{i}to{i+stepsize-1}.parquet')
            # del subdf

        if pvalloc_settings['test_faster_if_subdf_deleted']:
            del subdf

    if pvalloc_settings['test_faster_if_subdf_deleted']:
        checkpoint_to_logfile(f'  start agg topo_subdf to topo_agg_df', log_name, 1)
        subdfs_path = glob.glob(f'{subdf_path}/*.parquet')
        agg_subdf_list = []

        for f in subdfs_path:
            subdf = pd.read_parquet(f)
            if 'pv_inst_TF' in subdf.columns:
                subdf.rename(columns={'pv_inst_TF': 'pvinst_TF'}, inplace=True)
            agg_subdf = subdf.groupby(groupby_cols).agg({col: 'sum' for col in agg_cols}).reset_index()
            agg_subdf_list.append(agg_subdf)
            topo_agg_df = pd.concat(agg_subdf_list)
            # topo_agg_df.to_parquet(f'{data_path}/output/pvalloc_run/topo_agg_df.parquet')

    topo_agg_df = pd.concat(agg_subdf_list)
    topo_agg_df.to_parquet(f'{data_path}/output/pvalloc_run/topo_agg_df.parquet')
    checkpoint_to_logfile(f'  end agg topo_subdf to topo_agg_df', log_name, 1)
    """

if not pvalloc_settings['recalc_economics_topo_df']:
    topo_agg_df_in_pvallocrun = os.path.exists(f'{data_path}/output/pvalloc_run/topo_agg_df.parquet')
    topo_agg_df_in_interim = os.path.exists(f'{interim_path}/topo_agg_df.parquet')
    topo_agg_df_path = f'{interim_path}/topo_agg_df.parquet' if topo_agg_df_in_interim else f'{data_path}/output/pvalloc_run/topo_agg_df.parquet'

    subchapter_to_logfile('allocation algorithm: GET TOPO_DF from pvalloc or interim_path', log_name)
    topo_agg_df = pd.read_parquet(topo_agg_df_path)
    # NOTE: Remove if statement if no longer necessary
    if os.path.exists(f'{topo_agg_df_path}/topo_df.parquet'):
        topo_df = pd.read_parquet(f'{topo_agg_df_path}/topo_df.parquet')


# CREATE MAP OF TOPO_DF ----------------------------------------------------------------
if pvalloc_settings['create_map_of_topology']:
    subchapter_to_logfile('visualization: CREATE MAP OF TOPOLOGY_DF', log_name)
    visual.create_map_of_topology(pvalloc_settings, topo_df)


# NPV CALCULATION for ALL COMBINATIONS ----------------------------------------------
if pvalloc_settings['recalc_npv_all_combinations']:
    subchapter_to_logfile('allocation algorithm: NPV CALCULATION for ALL COMBINATIONS', log_name)
    npv_df = algo.calc_npv_partition_combinations(pvalloc_settings, topo_agg_df)  

    """
    interest_rate = pvalloc_settings['tech_economic_specs']['interest_rate']
    invst_maturity = pvalloc_settings['tech_economic_specs']['invst_maturity']

    # NOTE: needs to be adressed that some buildings have over 50, 100 or even 150 partitions
    df_before = topo_agg_df.copy()
    counts = topo_agg_df['EGID'].value_counts()
    topo_agg_df['EGID_count'] = topo_agg_df['EGID'].map(counts)
    topo_agg_df = topo_agg_df.loc[topo_agg_df['EGID_count'] < pvalloc_settings['gwr_selection_specs']['solkat_max_n_partitions']]
    print_to_logfile(f'ATTENTION: needed to drop {df_before["EGID"].nunique() - topo_agg_df["EGID"].nunique()}/{df_before["EGID"].nunique()} EGID, {df_before.shape[0] - topo_agg_df.shape[0]} partitions, because n_partitions > {pvalloc_settings["gwr_selection_specs"]["solkat_max_n_partitions"]}', log_name)


    # create all partition combos of topo_df ----------
    print_to_logfile(f'\nCreate all possible combos of partitions for {topo_agg_df["EGID"].nunique()} EGIDs', log_name)
    estim_instcost_chfpkW, estim_instcost_chftotal = get_estim_instcost_function(pvalloc_settings)
    agg_npry = np.array(topo_agg_df)

    egid_list, combo_df_uid_list, df_uid_list, flaeche_list, bfs_list, pv_tarif_Rp_kWh_list, econ_inc_chf_list, econ_spend_chf_list = [], [], [], [], [], [], [], [] 

    egid = topo_agg_df['EGID'].unique()[0]
    combos_counter = topo_agg_df['EGID'].nunique() // 10
    for i, egid in enumerate(topo_agg_df['EGID'].unique()):
        sub_egid = topo_agg_df.loc[topo_agg_df['EGID'] == egid].copy()
        
        mask_egid_topo_agg_df = np.isin(agg_npry[:,topo_agg_df.columns.get_loc('EGID')], egid)
        df_uids = list(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('df_uid')])

        for r in range(1,len(df_uids)+1): 
            for combo in itertools.combinations(df_uids, r):
                combo_key_str = '_'.join([str(c) for c in combo])

                egid_list.append(egid)
                combo_df_uid_list.append(combo_key_str)
                flaeche_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('FLAECHE')].sum())
                bfs_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('bfs')][0])
                pv_tarif_Rp_kWh_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('pv_tarif_Rp_kWh')][0])
                econ_inc_chf_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('econ_inc_chf')].sum())
                econ_spend_chf_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('econ_spend_chf')].sum())
        
        if i % combos_counter == 0:
            checkpoint_to_logfile(f'  combo creation complete for {i} of {topo_agg_df["EGID"].nunique()} EGIDs', log_name, 1)

    npv_df = pd.DataFrame({'EGID': egid_list, 'df_uid_combo': combo_df_uid_list, 'FLAECHE': flaeche_list, 
                            'bfs': bfs_list, 'pv_tarif_Rp_kWh': pv_tarif_Rp_kWh_list, 'econ_inc_chf': econ_inc_chf_list, 
                            'econ_spend_chf': econ_spend_chf_list})

    npv_df.to_parquet(f'{data_path}/output/pvalloc_run/npv_df.parquet')

    npv_df['estim_pvinstcost_chf'] = estim_instcost_chfpkW(npv_df['FLAECHE'] * conv_m2toKWP)
    npv_df.to_parquet(f'{data_path}/output/pvalloc_run/npv_df.parquet')

    def compute_npv(row):
        pv_cashflow = (row['econ_inc_chf'] - row['econ_spend_chf']) / (1+interest_rate)**np.arange(1, invst_maturity+1)
        npv = (-row['estim_pvinstcost_chf']) + np.sum(pv_cashflow)
        return npv
    npv_df['NPV_uid'] = npv_df.apply(compute_npv, axis=1)
    npv_df.to_parquet(f'{data_path}/output/pvalloc_run/npv_df.parquet')
    npv_df.to_csv(f'{wd_path}/npv_df.csv', index=False)
    """

elif not pvalloc_settings['recalc_npv_all_combinations']:
    subchapter_to_logfile('allocation algorithm: IMPORT NPV_DF', log_name)

    npv_df_in_pvallocrun = os.path.exists(f'{data_path}/output/pvalloc_run/npv_df.parquet')
    npv_df_in_interim = os.path.exists(f'{interim_path}/npv_df.parquet')
    npv_path = f'{interim_path}/npv_df.parquet' if npv_df_in_interim else f'{data_path}/output/pvalloc_run/npv_df.parquet'

    npv_df = pd.read_parquet(f'{data_path}/output/pvalloc_run/npv_df.parquet')



# ALLOCATION LOOP ----------------------------------------------
subchapter_to_logfile('allocation algorithm: START LOOP FOR PRED MONTH', log_name)
months_lookback = pvalloc_settings['months_lookback']
rand_seed = pvalloc_settings['algorithm_specs']['rand_seed']
safety_counter_max = pvalloc_settings['algorithm_specs']['safety_counter_max']
gridprem_ts = ts_list[ts_names.index('gridprem_ts')]

dfuid_installed_list = []
pred_inst_df = pd.DataFrame()

if 'pvinst_TF' not in npv_df.columns:
    npv_nopv_df = npv_df.copy()
    npv_nopv_df['pvinst_TF'] = False
elif 'pvinst_TF' in npv_df.columns:
    npv_nopv_df = npv_df.loc[npv_df['pvinst_TF'] == False].copy()

for i, m in enumerate(months_prediction):
    print_to_logfile(f'-- Allocation for month: {m} {15*"-"}', log_name)
    start_allocation_month = datetime.now()

    # initialize constr capacity ----------
    constr_built_m = 0
    if m.year != (m-1).year:
        constr_built_y = 0
    constr_capa_m = constrcapa.loc[constrcapa['date'] == m, 'constr_capacity_kw'].iloc[0]
    constr_capa_y = constrcapa.loc[constrcapa['year'].isin([m.year]), 'constr_capacity_kw'].sum()


    # INSTALLATION PICK ==========
    safety_counter = 0
    while (constr_built_m < constr_capa_m) & (constr_built_y < constr_capa_y) & (safety_counter < safety_counter_max ):
        # draw a random number between 0 and 1
        if rand_seed is not None:
            np.random.seed(rand_seed)
        rand_num = np.random.uniform(0, 1)
        npv_nopv_df['NPV_stand'] = npv_nopv_df['NPV_uid'] / max(npv_nopv_df['NPV_uid'])

        # find the NPV_stand that is closest to the random number
        npv_nopv_df['diff_NPV_rand'] = abs(npv_nopv_df['NPV_stand'] - rand_num)
        npv_nopv_min_pick = npv_nopv_df.loc[npv_nopv_df['diff_NPV_rand']  == npv_nopv_df['diff_NPV_rand'].min()]

        if npv_nopv_min_pick.shape[0] > 1:
            rand_row = np.random.randint(0, npv_nopv_min_pick.shape[0])
            npv_nopv_min_pick = npv_nopv_min_pick.iloc[[rand_row]]

        inst_power = npv_nopv_min_pick['FLAECHE'].values[0]  * conv_m2toKWP * pvalloc_settings['algorithm_specs']['capacity_tweak_fact']
        npv_nopv_min_pick['inst_power'] = inst_power


        # Adjust DFs ----------
        # NOTE: col name of df_uid will be different!
        picked_egid = npv_nopv_min_pick['EGID'].values[0]
        npv_nopv_df = npv_nopv_df.loc[npv_nopv_df['EGID'] != picked_egid].copy()

        picked_uid =  npv_nopv_min_pick['df_uid_combo']
        dfuid_installed_list.append(picked_uid)
        picked_combo_uid = list(picked_uid.values[0].split('_'))
        pred_inst_df = pd.concat([pred_inst_df, npv_nopv_min_pick], ignore_index=True)

        # Adjust TOPO ----------
        topo[picked_egid]['pv_inst'] = {'inst_TF': True, 'info_source': 'alloc_algorithm', 'BeginOp': f'{m}', 'TotalPower': inst_power}

        # Adjust constr_built capacity----------
        constr_built_m += inst_power
        constr_built_y += inst_power
        safety_counter += 1

        # State Loop Exit ----------
        constr_m_TF  = constr_built_m >= constr_capa_m
        constr_y_TF  = constr_built_y >= constr_capa_y
        safety_TF = safety_counter >= safety_counter_max
        if any([constr_m_TF, constr_y_TF, safety_TF]):
            checkpoint_to_logfile(f'Exit While Loop, constraint met : constr_m_TF: {constr_m_TF}, constr_y_TF: {constr_y_TF}, safety_TF: {safety_TF}', log_name, 1, True)
            checkpoint_to_logfile(f'end month allocation, runtime: {datetime.now() - start_allocation_month} (hh:mm:ss.s)', log_name, 1, show_debug_prints)
    
            # if len(dfuid_installed_list) % 100 == 0:
            #     checkpoint_to_logfile(f' picked {len(dfuid_installed_list) }EGIDs, inst_power: {round(inst_power,2)}, constr_capa m: {round(constr_built_m,1)}/{round(constr_capa_m,1)}; y: {round(constr_built_y,1)}/{round(constr_capa_y,1)}', log_name, 2)

            # UPDATE GRID PREM TS ------------------------------------------------------------
            algo.update_gridprem(pvalloc_settings, topo, pred_inst_df,
                                 ts_list, ts_names)



npv_nopv_df.to_parquet(f'{data_path}/output/pvalloc_run/npv_nopv_df.parquet')
pred_inst_df.to_parquet(f'{data_path}/output/pvalloc_run/pred_inst_df.parquet')

######################################
# BOOKMARK
######################################





######################################
# OLD VERSION
######################################
if False: 
    # export ----------
    topo_df.to_parquet(f'{data_path}/output/pvalloc_run/solkat_combo_df.parquet')
    topo_df.to_csv(f'{data_path}/output/pvalloc_run/solkat_combo_df.csv', index=False)
    checkpoint_to_logfile(f'Transformed topo dict ({len(topo.keys())}) to solkat_combo_df ({topo_df["df_uid"].nunique()}) for NPV calculations, ', log_name, 1)
    if True: 
        print('asdf')

    elif (not pvalloc_settings['rebuild_solkat_combo_from_topo']) and (solkat_combo_df_exists):
        topo_df = pd.read_parquet(f'{data_path}/output/pvalloc_run/topo_df.parquet')
        checkpoint_to_logfile(f'Imported solkat_combo_df from parquet', log_name, 1)

    else: 
        checkpoint_to_logfile(f'solkat_combo_df not built and also not found in interim path: {pvalloc_settings["interim_path"]}', log_name, 1)



    # START LOOP FOR PRED MONTH ---------------------------------------------------
    subchapter_to_logfile('allocation algorithm: START LOOP FOR PRED MONTH', log_name)


    for i, m in enumerate(months_prediction):
    # if True:
        # m = months_prediction[0]

        # define lookback period ---------- 
        start_allocation_month = datetime.now() 

        constr_capa_m = constrcapa.loc[constrcapa['date'] == m, 'constr_capacity_kw'].iloc[0]
        constr_capa_y = constrcapa.loc[constrcapa['year'].isin([m.year]), 'constr_capacity_kw'].sum()

        constr_built_m = 0
        if m.year != (m-1).year:
            constr_built_y = 0
        
        checkpoint_to_logfile(f'ALLOCATION MONTH: {m}', log_name, 2)
        print(f'  > prediction for: {m} - loockback start: {m-12}, end: {m-1}')

        # set lookback periods ----------
        start_lookback = pd.to_datetime(f'{m - months_lookback}-1 00:00:00')
        last_day_of_month = pd.to_datetime(f'{m-1}-01').to_period('M').to_timestamp('M') + pd.offsets.MonthEnd(0)
        end_lookback = last_day_of_month.replace(hour=23, minute=00, second=00)
        lookback = pd.DataFrame(pd.date_range(start_lookback, end_lookback, freq='H'), columns=['timestamp'])
        lookback['t'] = lookback['timestamp'].apply(lambda x: f't_{(x.dayofyear -1) * 24 + x.hour +1}')

        # filter TS data to hour of the year in lookback period
        meteo_lb = meteo_ts.loc[meteo_ts['t'].isin(lookback['t'])]
        meteo_lb.set_index('t', inplace=True)
        demandtypes_lb = demandtypes_ts.loc[demandtypes_ts['t'].isin(lookback['t'])].copy()
        demandtypes_lb.set_index('t', inplace=True)
        gridprem_lb = gridprem_ts.loc[gridprem_ts['t'].isin(lookback['t'])].copy()
        gridprem_lb.set_index('t', inplace=True)


        # NPV CALCULATION ==========    
        print_to_logfile(f'    NPV calculation for {len(solkat_combo_df["df_uid_combo"].unique())} UID combos', log_name)
        uid_counter = len(solkat_combo_df['df_uid_combo'].unique()) // 5

        # NOTE: build a better path for combo npv
        # if not os.path.exists(f'C:\Models\OptimalPV_RH_data\output\pvalloc_BSBLSO_wrkn_prgrss_20240823_13h\solkat_combo_df_npv.parquet'):
        combo_npv = solkat_combo_df.copy()
        uid = combo_npv['df_uid_combo'][0]

        for i, uid in enumerate(combo_npv['df_uid_combo'].unique()):
            # extract TS for production and demand
            flaeche_uid =          combo_npv.loc[combo_npv['df_uid_combo'] == uid, 'FLAECHE'].iloc[0]
            pvtarif_uid =          combo_npv.loc[combo_npv['df_uid_combo'] == uid, 'pv_tarif_Rp_kWh'].iloc[0]
            estim_pvinstcost_uid = combo_npv.loc[combo_npv['df_uid_combo'] == uid, 'estim_pvinstcost_chf'].iloc[0]
            elecpri_uid =          combo_npv.loc[combo_npv['df_uid_combo'] == uid, 'elecpri_Rp_kWh'].iloc[0]

            pvprod_uid_ts = (meteo_lb['radiation']/1000) * flaeche_uid
            demand_uid_ts = demandtypes_lb.loc[:, combo_npv.loc[combo_npv['df_uid_combo'] == uid, 'demandtype'].iloc[0]]

            type(pvprod_uid_ts)
            type(demand_uid_ts)

            pvprod_uid_npry = pvprod_uid_ts.values
            demand_uid_npry = demand_uid_ts.values

            selfconsum_uid_npry = np.minimum(pvprod_uid_npry, demand_uid_npry) * selfconsum_rate
            # netdemand_uid_npry = demand_uid_npry - selfconsum_uid_npry
            netfeedin_uid_npry = pvprod_uid_npry - selfconsum_uid_npry

            CF_y =   sum(netfeedin_uid_npry) *  pvtarif_uid  + sum(selfconsum_uid_npry) * elecpri_uid  
            
            NVP_uid = (-estim_pvinstcost_uid) + np.sum( CF_y / (1 + interest_rate)**np.arange(1, invst_maturity+1)) 
            combo_npv.loc[combo_npv['df_uid_combo'] == uid, 'NPV'] = NVP_uid

            ammort_uid = np.ceil(estim_pvinstcost_uid / ((sum(netfeedin_uid_npry)*  pvtarif_uid  + sum(selfconsum_uid_npry) * elecpri_uid)))
            combo_npv.loc[combo_npv['df_uid_combo'] == uid, 'ammort_years'] = ammort_uid

            if i % uid_counter == 0:
                checkpoint_to_logfile(f'  NPV calc complete for {i} of {len(combo_npv["df_uid_combo"].unique())} UID combos', log_name, 2)

        combo_npv.to_parquet(f'{data_path}/output/pvalloc_run/combo_npv.parquet')
        combo_npv.to_csv(f'{wd_path}/combo_npv.csv', index=False)

        combo_npv_before = combo_npv.copy()
        combo_npv_before.to_parquet(f'{data_path}/output/pvalloc_run/combo_npv_before.parquet')

        # elif os.path.exists(f'C:\Models\OptimalPV_RH_data\output\pvalloc_BSBLSO_wrkn_prgrss_20240823_13h\combo_npv.parquet'):
        #     combo_npv = pd.read_parquet(f'{data_path}/output/pvalloc_run/combo_npv.parquet')

        # # NOTE: remove this part when combo_npv is built
        # elif os.path.exists(f'C:\Models\OptimalPV_RH_data\output\pvalloc_BSBLSO_wrkn_prgrss_20240823_13h\solkat_combo_df_npv.parquet'):
        #     combo_npv = pd.read_parquet(f'C:\Models\OptimalPV_RH_data\output\pvalloc_BSBLSO_wrkn_prgrss_20240823_13h\solkat_combo_df_npv.parquet')




        # INSTALLATION PICK ==========
        rand_seed = pvalloc_settings['algorithm_specs']['rand_seed']
        safety_counter_max = pvalloc_settings['algorithm_specs']['safety_counter_max']
        dfuid_installed_list = []


        safety_counter = 0
        while (constr_built_m < constr_capa_m) & (constr_built_y < constr_capa_y) & (safety_counter < safety_counter_max):
            # draw a random number between 0 and 1
            if rand_seed is not None:
                np.random.seed(rand_seed)
            rand_num = np.random.uniform(0, 1)
            combo_npv['NPV_stand'] = combo_npv['NPV'] / max(combo_npv['NPV'])


            # find the NPV_stand that is closest to the random number
            combo_npv['diff_NPV_rand'] = abs(combo_npv['NPV_stand'] - rand_num)
            combo_npv_sub = combo_npv.loc[combo_npv['diff_NPV_rand']  == combo_npv['diff_NPV_rand'].min()]

            if combo_npv_sub.shape[0] > 1: 
                rand_row = np.random.randint(0, combo_npv_sub.shape[0])
                combo_npv_sub = combo_npv_sub.iloc[rand_row] 

            picked_uid =  combo_npv_sub['df_uid_combo'].iloc[0]    
            picked_egid = combo_npv_sub['EGID'].iloc[0]
            
            dfuid_installed_list.append(picked_uid)

            # Adjust combo_npv ----------
            picked_combo = combo_npv.loc[combo_npv['df_uid_combo'] == picked_uid, ]
            combo_npv = combo_npv.loc[combo_npv['EGID'] != picked_egid].copy()

            # Adjust topo ----------
            # look up picked_egid in topo keys 
            inst_power = picked_combo['FLAECHE'].iloc[0]  * pvalloc_settings['assumed_parameters']['conversion_m2_to_kw'] * pvalloc_settings['algorithm_specs']['capacity_tweak_fact']
            topo[picked_egid]['pv_inst'] = {'inst_TF': True, 'info_source': 'alloc_algorithm', 'BeginOp': f'{m}', 'TotalPower': inst_power}

            # Adjust constr_built capacity----------
            constr_built_m += inst_power
            constr_built_y += inst_power
            safety_counter += 1
            
            if len(dfuid_installed_list) % 100 == 0:
                checkpoint_to_logfile(f' picked {len(dfuid_installed_list) }EGIDs, inst_power: {inst_power}, constr_capa m: {round(constr_built_m,1)}/{round(constr_capa_m,1)}; y: {round(constr_built_y,1)}/{round(constr_capa_y,1)}', log_name, 2)


        # ADJUST GRID PREMIUM ==========


        # end ----------
        end_allocation_month = datetime.now()
        checkpoint_to_logfile(f'END ALLOCATION MONTH: {m} \n Runtime (hh:mm:ss):{end_allocation_month - start_allocation_month}', log_name, 2)


    # EXPORT TOPO & interim files ----------------------------------------------------------------  
    with open(f'{data_path}/output/pvalloc_run/topo_after.json', 'w') as f:
        json.dump(topo, f)

    combo_installed_df = combo_npv_before.loc[combo_npv_before['df_uid_combo'].isin(dfuid_installed_list), ]
    combo_installed_df.to_parquet(f'{data_path}/output/pvalloc_run/combo_installed_df.parquet')
    combo_installed_df.to_csv(f'{data_path}/output/pvalloc_run/combo_installed_df.csv', index=False)
    print('')




# COPY & RENAME AGGREGATED DATA FOLDER ---------------------------------------------------------------
# > not to overwrite completed preprep folder while debugging 

if pvalloc_settings['name_dir_export'] is None:    
    today = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dirs_alloc_data_DATE = f'{data_path}/output/pvalloc_data_{today.split("-")[0]}{today.split("-")[1]}{today.split("-")[2]}_{today.split("-")[3]}h'
    if os.path.exists(dirs_alloc_data_DATE):
        shutil.rmtree(dirs_alloc_data_DATE)
    if not os.path.exists(dirs_alloc_data_DATE):
        os.makedirs(dirs_alloc_data_DATE)
    file_to_move = glob.glob(f'{data_path}/output/pvalloc_data/*')
    for f in file_to_move:
        if os.path.isfile(f):
            shutil.copy(f, dirs_alloc_data_DATE)
        elif os.path.isdir(f):
            shutil.copytree(f, os.path.join(dirs_alloc_data_DATE, os.path.basename(f)))
    shutil.copy(glob.glob(f'{data_path}/output/pvalloc*_log.txt')[0], dirs_alloc_data_DATE)

elif pvalloc_settings['name_dir_export'] is not None:
    today = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dirs_alloc_data_DATE = f'{data_path}/output/{pvalloc_settings["name_dir_export"]}_{today.split("-")[0]}{today.split("-")[1]}{today.split("-")[2]}_{today.split("-")[3]}h'
    if os.path.exists(dirs_alloc_data_DATE):
        shutil.rmtree(dirs_alloc_data_DATE)
    if not os.path.exists(dirs_alloc_data_DATE):
        os.makedirs(dirs_alloc_data_DATE)
    file_to_move = glob.glob(f'{data_path}/output/pvalloc_run/*')
    for f in file_to_move:
        if os.path.isfile(f):
            shutil.copy(f, dirs_alloc_data_DATE)
        elif os.path.isdir(f):
            shutil.copytree(f, os.path.join(dirs_alloc_data_DATE, os.path.basename(f)))
    shutil.copy(glob.glob(f'{data_path}/output/pvalloc_log.txt')[0], f'{dirs_alloc_data_DATE}/pvalloc_log_{pvalloc_settings["name_dir_export"]}.txt')
    

# -----------------------------------------------------------------------------
# END 
chapter_to_logfile(f'END pv_allocation_MASTER\n Runtime (hh:mm:ss):{datetime.now() - total_runtime_start}', log_name, overwrite_file=False)


if not pvalloc_settings['script_run_on_server']:
    winsound.Beep(1000, 300)
    winsound.Beep(1000, 300)
    winsound.Beep(1000, 1000)
# -----------------------------------------------------------------------------







# ===========================================================================================
# ===========================================================================================
# ===========================================================================================



"""


bfs_list = pvalloc_settings['bfs_numbers']
bfs_list_str = [str(bfs) for bfs in bfs_list]

# mappings
Map_egroof_sbroof = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/Map_egroof_sbroof.parquet')
Map_egroof_pv = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/Map_egroof_pv.parquet')
Map_gm_ewr = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/Map_gm_ewr.parquet')

# gwr
checkpoint_to_logfile(f'start import with DD > GWR', log_name, n_tabs_def=2, show_debug_prints_def=pvalloc_settings['show_debug_prints'])
gwr_dd = dd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/gwr.parquet')
gwr = gwr_dd[gwr_dd['GGDENR'].isin(bfs_list_str)].compute()

# solkat
checkpoint_to_logfile(f'start import with DD > SOLKAT', log_name, n_tabs_def=2, show_debug_prints_def=pvalloc_settings['show_debug_prints'])
solkat_dd = dd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/solkat_by_gm.parquet')
solkat = solkat_dd[solkat_dd['BFS_NUMMER'].isin(bfs_list)].compute()
solkat_cumm_dd = dd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/solkatcost_cumm.parquet')
solkat_cumm = solkat_cumm_dd[solkat_cumm_dd['GWR_EGID'].isin(gwr['EGID'].unique())].compute()
solkat = solkat_cumm.copy()

# pv
pv_dd = dd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/pv_by_gm.parquet')
pv = pv_dd[pv_dd['BFS_NUMMER'].isin(bfs_list)].compute()
checkpoint_to_logfile(f'end import with DD', log_name, n_tabs_def=2, show_debug_prints_def=pvalloc_settings['show_debug_prints'])

# pvtarif
pvtarif = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/pvtarif.parquet')

# electricity prices
elecpri = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/elecpri.parquet')

# also import parquet files for comparison
checkpoint_to_logfile(f'start import ALL in PQ', log_name, n_tabs_def=2, show_debug_prints_def=pvalloc_settings['show_debug_prints'])
solkat_pq = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/solkat_by_gm.parquet')
gwr_pq = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/gwr.parquet')
pv_pq = pd.read_parquet(f'{data_path}/output/{pvalloc_settings["name_dir_import"]}/pv_by_gm.parquet')
checkpoint_to_logfile(f'end import ALL in PQ', log_name, n_tabs_def=2, show_debug_prints_def=pvalloc_settings['show_debug_prints'])



# DATA TRANSFORMATIONS -------------------------------------------------------
# convert all ID columns to string
def convert_srs_to_str(df, col_name):
    df[col_name] = df[col_name].fillna(-1).astype(int).astype(str)          # Fill NaN values with -1, convert to integers, and then to stringsMap_egroof_sbroof['EGID'] = Map_egroof_sbroof['EGID'].fillna(-1).astype(int).astype(str)    # Fill NaN values with -1, convert to integers, and then to strings
    df[col_name] = df[col_name].replace('-1', np.nan)                       # Replace '-1' with 'nan'
    return df

Map_egroof_sbroof = convert_srs_to_str(Map_egroof_sbroof, 'EGID')
Map_egroof_pv = convert_srs_to_str(Map_egroof_pv, 'EGID')
Map_egroof_pv = convert_srs_to_str(Map_egroof_pv, 'xtf_id')
Map_gm_ewr = convert_srs_to_str(Map_gm_ewr, 'bfs')

solkat = convert_srs_to_str(solkat, 'GWR_EGID')
pv = convert_srs_to_str(pv, 'xtf_id')
pv = convert_srs_to_str(pv, 'BFS_NUMMER')

# converte other data types
gwr['GBAUJ'] = pd.to_numeric(gwr['GBAUJ'], errors='coerce')
gwr['GBAUJ'] = gwr['GBAUJ'].fillna(0)  # replace NaN with 0
gwr['GBAUJ'] = gwr['GBAUJ'].astype(int)

# transform to date
pv_capa = pv.copy()
pv_capa['BeginningOfOperation'] = pd.to_datetime(pv_capa['BeginningOfOperation'])


# ----------------------------------------------------------------------------------------------------------------------------------




# plan:
#-- stil before pv_allocation
#   > create a py file that creates all assumptions, cost etc. 

#-- prepare all data computations

# --- cost computation
#   > compute cost per roof partition - CHECK
#   >> "downwoard" computation -> compute NPV for best partition, second best and best partition, etc.
#   >> include ratio of self consumption

#   > (compute elec demand by heating squrare)
#   > 

#-- subset selection


#-- initiate topology
#   > create dict with gwr id and year
#   >> subsetable by bfs number and building type

#   > define thresholds for installations per year

#   > assign all partitions to dict
#   > assign production of installed pv to dict
#   > (assign grid connection to dict)

#-- calculate NPV
#   > select free gwrs 
#   > calculate NPV by partition (possible with switch to only consider KLASSE 3+)
#   > select best NPV
# ----------------------------------------------------------------------------------------------------------------------------------




# INITIATE TOPOLOGY ----------------------------------------------------------------


# create topo with EGID for certain GWR filter criteria
gwr_sub = gwr[
    # (gwr['GBAUJ'] >= pvalloc_settings['topology_year_range'][0]) &
    (gwr['GBAUJ'] <= pvalloc_settings['topology_year_range'][1]) &
    (gwr['GKLAS'].isin(pvalloc_settings['gwr_house_type_class']))&
    (gwr['GSTAT'] == '1004')].copy() # only consider buildings that are standing

pvtopo_df = gwr_sub[['EGID', 'GBAUJ', 'GKLAS']].copy()
pvtopo = pvtopo_df

# pvtopo_df.set_index('EGID', inplace=True)
# pvtopo = pvtopo_df.to_dict('index')
# for key, value in pvtopo.items():
#     value['EGID'] = key


# attach pv xtf_id to topo
# pvtopo = {k: {**v, 'pv': np.nan} for k, v in pvtopo.items()} # create new key with value np.nan
# Map_egroof_pv_dict = Map_egroof_pv.set_index('EGID')['xtf_id'].to_dict() # Convert the Map_egroof_pv DataFrame to a dictionary
Map_pv_BegOfOp = pv[['xtf_id','BeginningOfOperation']].copy()
Map_pv_BegOfOp.rename(columns={'BeginningOfOperation': 'BegOfOp'}, inplace=True)


pvtopo = pvtopo.merge(Map_egroof_pv[['EGID', 'xtf_id']], on='EGID', how='left', suffixes=('_pvtopo', '_Map_egroof_pv'))
pvtopo[['pv',]] = np.nan
pvtopo.loc[pd.notna(pvtopo['xtf_id']), 'pv'] = 1
pvtopo = pvtopo.merge(Map_pv_BegOfOp, on='xtf_id', how='left', suffixes=('_pvtopo', '_Map_pv_BegOfOp'))
pvtopo['BegOfOp'] = pvtopo['BegOfOp'].replace({'<NA>': np.nan})


# COMPUTE NPV ----------------------------------------------------------------

# ASSUMPTION / PARAMETERs
interest_rate = 0.01
inflation_rate = 0.018  
maturity = 25
disc_rate = (interest_rate+ inflation_rate + interest_rate * inflation_rate) # NOTE: Source?
disc_denominator = np.sum((1+disc_rate)** np.arange(1, maturity+1))
capa_years = [2019, 2023]
ann_inst_capa = pv_capa.loc[pv_capa['BeginningOfOperation'].dt.year.isin(capa_years), 'InitialPower'].sum() / len(capa_years)




year = pvalloc_settings['topology_year_range'][1]
year2d = str(year % 100).zfill(2)
solkat_t = solkat.copy()

pvtarif_t = pvtarif.loc[pvtarif['year'] == year2d, :].copy()
pvtarif_bygm_t = Map_gm_ewr.copy().merge(pvtarif_t, on='nrElcom', how='left', suffixes=('_pvtarif_bygm_t', '_pvtarif'))
# pvtarif_bygm_t['energy1'].replace({np.nan: 0}, inplace=True)    # distorts the mean with a value of 0
pvtarif_bygm_t['energy1'] = pvtarif_bygm_t['energy1'].astype(float) 
pvtarif_bygm_t = pvtarif_bygm_t.dropna(subset=['energy1'])

# group over bfs and average power1           
pvtarif_bygm_t = pvtarif_bygm_t.groupby('bfs').agg({'energy1': 'mean'}).reset_index()
pvtarif_bygm_t.rename(columns={'bfs': 'BFS_NUMMER', 'energy1': 'avg_pvrate_RpkWh'}, inplace=True)

# attach BFS_NUMMER to solkat
attach_BFS = gwr[['EGID', 'GGDENR']].copy()
attach_BFS.rename(columns={'EGID': 'GWR_EGID', 'GGDENR': 'BFS_NUMMER'}, inplace=True)
solkat_t = solkat_t.merge(attach_BFS, on='GWR_EGID', how='left', suffixes=('_solkat_t', '_attach_BFS'))

# attach avg_pvrate_RpkWh to solkat
solkat_t = solkat_t.merge(pvtarif_bygm_t, on='BFS_NUMMER', how='left', suffixes=('_solkat_t', '_pvtarif_bygm_t'))

# NPV CALCULATION -----------
solkat_t['pv_gain_1y'] = solkat_t['STROME_cumm'] * solkat_t['avg_pvrate_RpkWh']
solkat_t['NPV'] = np.nan
solkat_t['NPV'] = (maturity * solkat_t['pv_gain_1y'] /disc_denominator )-solkat_t['partition_pv_cost_chf']

# solkat_t.head(400).to_csv(f'{data_path}/output/solkat_t400.csv')
# solkat_t.head(400).to_excel(f'{data_path}/output/solkat_t400.xlsx')
# BOOKMARK: 

import matplotlib.pyplot as plt

# Plot the PDF of 'NPV'
solkat_t['NPV'].hist(density=True, bins=100)
plt.xlabel('NPV')
plt.ylabel('PDF')
plt.title('PDF of NPV')
plt.show()
# Plot the CDF of 'NPV'
solkat_t['NPV'].hist(cumulative=True, density=1, bins=100)
plt.xlabel('NPV')
plt.ylabel('CDF')
plt.title('CDF of NPV')
plt.show()

egid_noinst_t = pvtopo.loc[pvtopo['pv'] != 1, :]


"""


print("End of script")
# -----------------------------------------------------------------------------
# END
# -----------------------------------------------------------------------------

# =============================================================================
# ARCHIVE
# =============================================================================


"""
# adjust "bfs_numbers" to all bfs of a canton are selected

if (isinstance(pvalloc_settings['kt_numbers'], list)) and (not not pvalloc_settings['kt_numbers']): # check if canton selection is a list and not empty
    pvalloc_settings['bfs_numbers'] = get_bfs_from_ktnr(pvalloc_settings['kt_numbers'])
#     print_to_logfile(f' > kt_numbers: {pvalloc_settings["kt_numbers"]}; use the municipality bfs numbers from the following canton numbers', log_name)
#     gm_shp_sub = gm_shp[gm_shp['KANTONSNUM'].isin(pvalloc_settings['kt_numbers'])]
#     pvalloc_settings['bfs_numbers'] = gm_shp_sub['BFS_NUMMER'].unique().tolist()
elif (isinstance(pvalloc_settings['bfs_numbers'], list)) and (not not pvalloc_settings['bfs_numbers']): # check if bfs selection is a list and not empty
    print_to_logfile(f' > bfs_numbers: {pvalloc_settings["bfs_numbers"]}; use the following municipality bfs numbers, not cantonal selection specifies', log_name)
else:
    print_to_logfile(f' > ERROR: no canton or bfs selection applicables; NOT used any municipality selection', log_name)

# print all selected municipalities as a check
checkpoint_to_logfile('\n > selected municipalities for pv allocation', log_name, n_tabs_def=1, show_debug_prints_def=pvalloc_settings['show_debug_prints'])
select_bfs = gm_shp_sub[['NAME', 'BFS_NUMMER', 'KANTONSNUM']].copy()
for i, r in select_bfs.iterrows():
    row_data = ', '.join([f'{col}: {r[col]}' for col in select_bfs.columns])
    checkpoint_to_logfile(f'name: {r["NAME"]} \t BFS: {r["BFS_NUMMER"]} \t KT: {r["KANTONSNUM"]}', log_name, n_tabs_def=2, show_debug_prints_def=pvalloc_settings['show_debug_prints'])
print_to_logfile(f'\n', log_name)
"""

