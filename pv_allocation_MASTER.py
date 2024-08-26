# -----------------------------------------------------------------------------
# pv_allocation_MASTER.py 
# -----------------------------------------------------------------------------
# Preamble: 
# > author: Raul Hochuli (raul.hochuli@unibas.ch), University of Basel, spring 2024
# > description: 


# SETTIGNS --------------------------------------------------------------------
pvalloc_settings = {
        'name_dir_export': 'pvalloc_BSBLSO_wrkn_prgrss',              # name of the directory where all proccessed data is stored at the end of the code file 
        'name_dir_import': 'preprep_BSBLSO_15to23_20240821_02h', # name of the directory where preprepared data is stored and accessed by the code
        'script_run_on_server': False,                           # F: run on private computer, T: run on server
        'fast_debug_run': True,                                 # T: run the code with a small subset of data, F: run the code with the full dataset
        'show_debug_prints': True,                              # F: certain print statements are omitted, T: includes print statements that help with debugging
        'n_egid_in_topo':1000, 
        'wd_path_laptop': 'C:/Models/OptimalPV_RH',              # path to the working directory on Raul's laptop
        'wd_path_server': 'D:/RaulHochuli_inuse/OptimalPV_RH',   # path to the working directory on the server

        'kt_numbers': [11,12,13],                           # list of cantons to be considered, 0 used for NON canton-selection, selecting only certain individual municipalities
        'bfs_numbers': [], # 2761, 2772],             # list of municipalites to select for allocation (only used if kt_numbers == 0)
        # 'topology_year_range':[2019, 2022],
        # 'prediction_year_range':[2023, 2025],
        'T0_prediction': '2023-01-01 00:00:00', 
        'months_lookback': 12*1,
        'months_prediction': 12*3,
        'recreate_topology': True, 
        'rebuild_solkat_combo_from_topo': True,
        'MonteCarlo_iter': 1000, 

        'algorithm_specs': {
            'rand_seed': 42, 
            'safety_counter_max': 100,
            'capacity_tweak_fact': 0.01
        },

        'tech_economic_specs': {
            'interest_rate': 0.01,
            'pvtarif_year': 2022, 
            'pvtarif_col': ['energy1', 'eco1'],
            'elecpri_year': 2022,
            'elecpri_category': 'H8', 
            'invst_maturity': 25,
            'self_consumption_ifapplicable': 1,
            },
        'weather_specs': {},
        'meteoblue_col_radiation_proxy': 'Basel Direct Shortwave Radiation',
        'weather_year': 2022,

        'constr_capacity_specs': {
            'ann_capacity_growth': 0.1,         # annual growth of installed capacity# each year, X% more PV capacity can be built, 100% in year T0
            'summer_months': [4,5,6,7,8,9,],
            'winter_months': [10,11,12,1,2,3,],
            'share_to_summer': 0.6, 
            'share_to_winter': 0.4,
        },
        'gwr_selection_specs': {
            'building_cols': ['EGID', 'GDEKT', 'GGDENR', 'GKODE', 'GKODN', 'GKSCE', 
                        'GSTAT', 'GKAT', 'GKLAS', 'GBAUJ', 'GBAUM', 'GBAUP', 'GABBJ', 'GANZWHG', 
                        'GWAERZH1', 'GENH1', 'GWAERSCEH1', 'GWAERDATH1', 'GEBF', 'GAREA'],
            'dwelling_cols':['EGID', 'WAZIM', 'WAREA', ],
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
            'conversion_m2_to_kw': 0.1,  # A 1m2 area can fit 0.1 kWp of PV Panels
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
from pv_allocation.initialization import *



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
pvalloc_settings['interim_path'] = get_interim_path(pvalloc_settings)
# pvalloc_settings['disc_rate'] = (pvalloc_settings['assumed_parameters']['interest_rate'] + pvalloc_settings['assumed_parameters']['inflation_rate'] )
# pvalloc_settings['disc_denom'] = np.sum((1+pvalloc_settings['disc_rate'])** np.arange(1, pvalloc_settings['assumed_parameters']['invest_maturity']+1))

chapter_to_logfile(f'start pv_allocation_MASTER for: {pvalloc_settings["name_dir_export"]}', log_name, overwrite_file=True)
formated_pvalloc_settings = format_MASTER_settings(pvalloc_settings)
print_to_logfile(f'pvalloc_settings: \n{pformat(formated_pvalloc_settings)}', log_name)



# INITIALIZATION ================================================================
if pvalloc_settings['recreate_topology']:
    subchapter_to_logfile('initialization: IMPORT PREPREP DATA & CREATE (building) TOPOLOGY', log_name)
    topo, df_list, df_names = import_prepre_AND_create_topology(pvalloc_settings)

elif not pvalloc_settings['recreate_topology']:
    subchapter_to_logfile('initialization: IMPORT EXISITNG TOPOLOGY', log_name) 
    topo, df_list, df_names = import_exisitng_topology(pvalloc_settings, 
                    df_search_names = ['Map_solkatdfuid_egid', 'Map_egid_pv', 'Map_demandtypes_egid', 'Map_egid_demandtypes', 'pvtarif', 'elecpri'])

subchapter_to_logfile('initialization: IMPORT TS DATA', log_name)
ts_list, ts_names = import_ts_data(pvalloc_settings)
# elif not pvalloc_settings['reimport_ts_data']:
#     subchapter_to_logfile('initialization: IMPORT EXISITNG TS DATA', log_name)
#     ts_list = import_exisitng_ts_data(pvalloc_settings)

subchapter_to_logfile('initialization: DEFINE CONSTRUCTION CAPACITY', log_name)
constrcapa, months_prediction, months_lookback = define_construction_capacity(pvalloc_settings, topo, ts_list)




# ALLOCATION ALGORITHM ================================================================

# INITIATE ----------------------------------------------------------------
subchapter_to_logfile('allocation algorithm: INITIATE', log_name)
months_lookback = pvalloc_settings['months_lookback']
selfconsum_rate = pvalloc_settings['tech_economic_specs']['self_consumption_ifapplicable']
interest_rate = pvalloc_settings['tech_economic_specs']['interest_rate']
invst_maturity = pvalloc_settings['tech_economic_specs']['invst_maturity']

[Map_solkatdfuid_egid, Map_egid_pv, Map_demandtypes_egid, Map_egid_demandtypes, pvtarif, elecpri] = df_list
[Map_daterange, demandtypes_ts, meteo_ts, gridprem_ts] = ts_list
print(f'\nReatach elements of df_list: {df_names}')
print(f'Reatach elements of ts_list: {ts_names}')
print(f'\n')

# NOTE: clean up when no longer necessary
if 'high_wiHP' in demandtypes_ts.columns:
    demandtypes_ts.rename(columns={'high_wiHP': 'high_DEMANDprox_wiHP', 'low_wiHP': 'low_DEMANDprox_wiHP', 'high_noHP': 'high_DEMANDprox_noHP', 'low_noHP': 'low_DEMANDprox_noHP'}, inplace=True)


# all partition combos to df ----------
print_to_logfile(f'Create all partition combos for NPV calculations from topo', log_name)
solkat_combo_df_exists = os.path.exists(f'{pvalloc_settings["interim_path"]}/solkat_combo_df.parquet')
if pvalloc_settings['rebuild_solkat_combo_from_topo']:
    no_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == False]
    with_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == True]

    egid_list = []
    bfs_list = []
    demandtype_list = []
    pv_tarif_Rp_kWh_list = []
    df_uid_combo_list = []
    flaeche_list = []   
    stromertrag_list = []
    estim_instcost_list = []
    elecpri_list = []
    keys = list(topo.keys())

    for k,v in topo.items():
        demandtype = v.get('demand_type')
        pv_tarif_Rp_kWh = v.get('pvtarif_Rp_kWh')
        elecpri_Rp_kWh = v.get('elecpri_Rp_kWh')
        bfs_uid = v.get('gwr_info').get('bfs')

        solkat_combos = v.get('solkat_combos')
        for k_combo, v_combo in solkat_combos.items():
            egid_list.append(k)
            bfs_list.append(bfs_uid)
            demandtype_list.append(demandtype)
            pv_tarif_Rp_kWh_list.append(pv_tarif_Rp_kWh)
            elecpri_list.append(elecpri_Rp_kWh)

            df_uid_combo_list.append(k_combo)
            flaeche_list.append(v_combo.get('FLAECHE'))
            stromertrag_list.append(v_combo.get('STROMERTRAG'))
            estim_instcost_list.append(v_combo.get('estim_pvinstcost_chf'))
        df_uid_combos = [k_combo for k_combo, v_combo in solkat_combos.items()]

    solkat_combo_df = pd.DataFrame({'EGID': egid_list, 'bfs': bfs_list, 'demandtype': demandtype_list, 'pv_tarif_Rp_kWh': pv_tarif_Rp_kWh_list,
                                    'df_uid_combo': df_uid_combo_list, 'FLAECHE': flaeche_list, 'STROMERTRAG': stromertrag_list, 
                                    'estim_pvinstcost_chf': estim_instcost_list, 'elecpri_Rp_kWh': elecpri_list})


    # export ----------
    solkat_combo_df.to_parquet(f'{data_path}/output/pvalloc_run/solkat_combo_df.parquet')
    solkat_combo_df.to_csv(f'{data_path}/output/pvalloc_run/solkat_combo_df.csv', index=False)
    checkpoint_to_logfile(f'Transformed topo dict ({len(topo.keys())}) to solkat_combo_df ({solkat_combo_df["df_uid_combo"].nunique()}) for NPV calculations, ', log_name, 1)


elif (not pvalloc_settings['rebuild_solkat_combo_from_topo']) and (solkat_combo_df_exists):
    solkat_combo_df = pd.read_parquet(f'{data_path}/output/pvalloc_run/solkat_combo_df.parquet')
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

        
        combo_installed_df = combo_npv_before.loc[combo_npv_before['df_uid_combo'].isin(dfuid_installed_list), ]
        combo_installed_df.to_parquet(f'{data_path}/output/pvalloc_run/combo_installed_df.parquet')
        combo_installed_df.to_csv(f'{data_path}/output/pvalloc_run/combo_installed_df.csv', index=False)


    # ADJUST GRID PREMIUM ==========
    # TO BE CODED SOON! \_(^_^)_/ 


    # end ----------
    end_allocation_month = datetime.now()
    checkpoint_to_logfile(f'END ALLOCATION MONTH: {m} \n Runtime (hh:mm:ss):{end_allocation_month - start_allocation_month}', log_name, 2)


# EXPORT TOPO & interim files ----------------------------------------------------------------  
with open(f'{data_path}/output/pvalloc_run/topo_after.json', 'w') as f:
    json.dump(topo, f)




# COPY & RENAME AGGREGATED DATA FOLDER ---------------------------------------------------------------
# > not to overwrite completed preprep folder while debugging 

if pvalloc_settings['name_dir_export'] is None:    
    today = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dirs_alloc_data_DATE = f'{data_path}/output/pvalloc_data_{today.split("-")[0]}{today.split("-")[1]}{today.split("-")[2]}_{today.split("-")[3]}h'
    if not os.path.exists(dirs_alloc_data_DATE):
        os.makedirs(dirs_alloc_data_DATE)
    file_to_move = glob.glob(f'{data_path}/output/pvalloc_data/*')
    for f in file_to_move:
        shutil.copy(f, dirs_alloc_data_DATE)
    shutil.copy(glob.glob(f'{data_path}/output/pvalloc*_log.txt')[0], dirs_alloc_data_DATE)

elif pvalloc_settings['name_dir_export'] is not None:
    today = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    dirs_alloc_data_DATE = f'{data_path}/output/{pvalloc_settings["name_dir_export"]}_{today.split("-")[0]}{today.split("-")[1]}{today.split("-")[2]}_{today.split("-")[3]}h'
    if not os.path.exists(dirs_alloc_data_DATE):
        os.makedirs(dirs_alloc_data_DATE)
    file_to_move = glob.glob(f'{data_path}/output/pvalloc_run/*')
    for f in file_to_move:
        shutil.copy(f, dirs_alloc_data_DATE)
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

