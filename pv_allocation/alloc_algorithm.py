import sys
import os as os
import numpy as np
import pandas as pd
import json
import itertools
import math
import glob
import plotly.graph_objs as go
import plotly.offline as pyo
import geopandas as gpd
from datetime import datetime

from pyarrow.parquet import ParquetFile
import pyarrow as pa

# own functions 
sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile

import pv_allocation.initialization as initial

# ------------------------------------------------------------------------------------------------------
# CALCULATE ECONOMIC INDICATORS OF TOPOLOGY
# ------------------------------------------------------------------------------------------------------
def calc_economics_in_topo_df(
        pvalloc_settings, 
        topo_func, 
        df_list_func, df_names_func, 
        ts_list_func, ts_names_func,):
    
    # setup -----------------------------------------------------
    name_dir_import_def = pvalloc_settings['name_dir_import']
    data_path_def = pvalloc_settings['data_path']
    show_debug_prints_def = pvalloc_settings['show_debug_prints']
    log_file_name_def = pvalloc_settings['log_file_name']
    print_to_logfile(f'run function: calc_economics_in_topo_df', log_file_name_def)

    topo = topo_func
    data_path = data_path_def
    log_name = log_file_name_def
    df_list = df_list_func
    df_names = df_names_func
    ts_list = ts_list_func
    ts_names = ts_names_func



    # import -----------------------------------------------------
    [Map_solkatdfuid_egid, Map_egid_pv, Map_demandtypes_egid, Map_egid_demandtypes, pvtarif, elecpri, angle_tilt_df] = df_list
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

    egid_list, bfs_list, gklas_list, demandtype_list, grid_node_list = [], [], [], [], []
    pvinst_list, pvsource_list, pvid_list, pv_tarif_Rp_kWh_list = [], [], [], []
    df_uid_list, flaeche_list, stromertrag_list, ausrichtung_list, neigung_list, elecpri_list = [], [], [], [], [], []

    keys = list(topo.keys())

    for k,v in topo.items():
        partitions = v.get('solkat_partitions')

        for k_p, v_p in partitions.items():
            egid_list.append(k)
            grid_node_list.append(v.get('grid_node'))
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
            ausrichtung_list.append(v_p.get('AUSRICHTUNG'))
            stromertrag_list.append(v_p.get('STROMERTRAG'))
            neigung_list.append(v_p.get('NEIGUNG'))           


        
    topo_df = pd.DataFrame({'EGID': egid_list, 'bfs': bfs_list, 'grid_node' : grid_node_list, 'gklas': gklas_list,
                            'pvinst_TF': pvinst_list, 'pvsource': pvsource_list, 'pvid': pvid_list, 
                            'demandtype': demandtype_list, 'pv_tarif_Rp_kWh': pv_tarif_Rp_kWh_list, 
                            'df_uid': df_uid_list, 'FLAECHE': flaeche_list, 'AUSRICHTUNG': ausrichtung_list, 
                            'NEIGUNG': neigung_list, 'STROMERTRAG': stromertrag_list, 'elecpri_Rp_kWh': elecpri_list})
    
    groupby_cols = ['EGID', 'df_uid', 'grid_node', 'bfs', 'gklas', 
                    'pvinst_TF', 'pvsource', 'pvid', 'pv_tarif_Rp_kWh', 'elecpri_Rp_kWh', 
                    'FLAECHE', 'AUSRICHTUNG', 'STROMERTRAG', 'NEIGUNG']
    agg_cols = ['econ_inc_chf', 'econ_spend_chf']

    topo_df.to_parquet(f'{data_path}/output/pvalloc_run/topo_df.parquet')
    

    # GET ECONOMIC VALUES FOR NPV CALCULATION ----------------------------------------------
    topo_subdf_partitioner = pvalloc_settings['algorithm_specs']['topo_subdf_partitioner']
    selfconsum_rate = pvalloc_settings['tech_economic_specs']['self_consumption_ifapplicable']

    # round NEIGUNG + AUSRICHTUNG to 5 for easier computation
    topo_df['NEIGUNG'] = topo_df['NEIGUNG'].apply(lambda x: round(x / 5) * 5)
    topo_df['AUSRICHTUNG'] = topo_df['AUSRICHTUNG'].apply(lambda x: round(x / 5) * 5)
    
    def lookup_efficiency(row, angle_tilt_df):
        try:
            return angle_tilt_df.loc[(row['AUSRICHTUNG'], row['NEIGUNG']), 'efficiency_factor']
        except KeyError:
            return 0
    topo_df['efficiency_factor'] = topo_df.apply(lambda r: lookup_efficiency(r, angle_tilt_df), axis=1)

    agg_subdf_list = []
    dfuids = topo_df['df_uid'].unique()
    subdf_path = f'{data_path}/output/pvalloc_run/topo_subdf'

    if not os.path.exists(subdf_path):
        os.makedirs(subdf_path)

    stepsize = topo_subdf_partitioner if len(dfuids) > topo_subdf_partitioner else len(dfuids)
    tranche_counter = 0
    for i in range(0, len(dfuids), stepsize):
        # print(f'  > {i} to {i+stepsize-1}'
        tranche_counter += 1
        print_to_logfile(f'-- topo_subdf {tranche_counter}/{len(range(0, len(dfuids), stepsize))} ({i} to {i+stepsize-1} df_uid.iloc) , calc gains/spending per partition {7*"-"}  (stamp: {datetime.now()})', log_name)
        # print_to_logfile(f'Calculate gains/spending per roof partition, df_uid.iloc ({i} to {i+stepsize-1})', log_name)
        subdf = topo_df[topo_df['df_uid'].isin(dfuids[i:i+stepsize])].copy()


        # merge production, grid prem + demand to partitions ----------
        subdf['meteo_loc'] = 'Basel'
        meteo_ts['meteo_loc'] = 'Basel' 
        subdf = subdf.merge(meteo_ts[['t', 'radiation', 'meteo_loc']], how='left', on='meteo_loc')
        subdf = subdf.assign(pvprod = (subdf['radiation'] * subdf['FLAECHE'] * subdf['efficiency_factor']) / 1000).drop(columns=['meteo_loc', 'radiation'])
        # checkpoint_to_logfile(f'  end merge meteo for subdf {i} to {i+stepsize-1}', log_name, 1)

        subdf['grid_node_loc'] = 'BS001'
        gridprem_ts['grid_node_loc'] = 'BS001'
        subdf = subdf.merge(gridprem_ts[['t', 'prem_Rp_kWh', 'grid_node']], how='left', on=['t', 'grid_node']) #.drop(columns='grid_node')
        # checkpoint_to_logfile(f'  end merge gridprem for subdf {i} to {i+stepsize-1}', log_name, 1)

        demandtypes_names = [c for c in demandtypes_ts.columns if 'DEMANDprox' in c]
        demandtypes_melt = demandtypes_ts.melt(id_vars='t', value_vars=demandtypes_names, var_name= 'demandtype', value_name= 'demand')
        subdf = subdf.merge(demandtypes_melt, how='left', on=['t', 'demandtype'])
        # checkpoint_to_logfile(f'  end merge demandtypes for subdf {i} to {i+stepsize-1}', log_name, 1)


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

    return topo_df, topo_agg_df



# ------------------------------------------------------------------------------------------------------
# CALCULATE NPV OF ALL PARTITION COMBINATIONS
# ------------------------------------------------------------------------------------------------------
def calc_npv_partition_combinations(
        pvalloc_settings,
        topo_agg_df_func,):
    
    # setup -----------------------------------------------------
    name_dir_import_def = pvalloc_settings['name_dir_import']
    data_path_def = pvalloc_settings['data_path']
    wd_path_def = pvalloc_settings['wd_path']
    show_debug_prints_def = pvalloc_settings['show_debug_prints']
    log_file_name_def = pvalloc_settings['log_file_name']
    print_to_logfile(f'run function: calc_economics_in_topo_df', log_file_name_def)

    topo_agg_df = topo_agg_df_func
    data_path = data_path_def
    wd_path = wd_path_def
    log_name = log_file_name_def

    interest_rate = pvalloc_settings['tech_economic_specs']['interest_rate']
    invst_maturity = pvalloc_settings['tech_economic_specs']['invst_maturity']
    conv_m2toKWP = pvalloc_settings['tech_economic_specs']['conversion_m2tokW']

    # NOTE: needs to be adressed that some buildings have over 50, 100 or even 150 partitions
    df_before = topo_agg_df.copy()
    counts = topo_agg_df['EGID'].value_counts()
    topo_agg_df['EGID_count'] = topo_agg_df['EGID'].map(counts)
    topo_agg_df = topo_agg_df.loc[topo_agg_df['EGID_count'] < pvalloc_settings['gwr_selection_specs']['solkat_max_n_partitions']]
    print_to_logfile(f'ATTENTION: needed to drop {df_before["EGID"].nunique() - topo_agg_df["EGID"].nunique()}/{df_before["EGID"].nunique()} EGID, {df_before.shape[0] - topo_agg_df.shape[0]} partitions, because n_partitions > {pvalloc_settings["gwr_selection_specs"]["solkat_max_n_partitions"]}', log_name)


    # create all partition combos of topo_df ----------
    print_to_logfile(f'\nCreate all possible combos of partitions for {topo_agg_df["EGID"].nunique()} EGIDs', log_name)
    estim_instcost_chfpkW, estim_instcost_chftotal = initial.get_estim_instcost_function(pvalloc_settings)
    agg_npry = np.array(topo_agg_df)

    egid_list, combo_df_uid_list, df_uid_list, flaeche_list, bfs_list, pv_tarif_Rp_kWh_list, econ_inc_chf_list, econ_spend_chf_list = [], [], [], [], [], [], [], [] 

    egid = topo_agg_df['EGID'].unique()[0]
    combos_counter = topo_agg_df['EGID'].nunique() // 10
    checkpoint_to_logfile(f'start combo creation of partitions', log_name, 1, True)
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
            # checkpoint_to_logfile(f'  combo creation complete for {i} of {topo_agg_df["EGID"].nunique()} EGIDs', log_name, 1)
            checkpoint_to_logfile(f'-- combo complete for {i} of {topo_agg_df["EGID"].nunique()} EGIDs {7*"-"}  (stamp: {datetime.now()})', log_name)

    checkpoint_to_logfile(f'end combo creation of partitions', log_name, 1, True)

    npv_df = pd.DataFrame({'EGID': egid_list, 'df_uid_combo': combo_df_uid_list, 'FLAECHE': flaeche_list, 
                            'bfs': bfs_list, 'pv_tarif_Rp_kWh': pv_tarif_Rp_kWh_list, 'econ_inc_chf': econ_inc_chf_list, 
                            'econ_spend_chf': econ_spend_chf_list})
    npv_df.to_parquet(f'{data_path}/output/pvalloc_run/npv_df.parquet')

    npv_df['estim_pvinstcost_chf'] = estim_instcost_chfpkW(npv_df['FLAECHE'] * conv_m2toKWP)
    npv_df.to_parquet(f'{data_path}/output/pvalloc_run/npv_df.parquet')
    checkpoint_to_logfile(f'estimated installation costs added to npv_df', log_name, 1, show_debug_prints_def)

    def compute_npv(row):
        pv_cashflow = (row['econ_inc_chf'] - row['econ_spend_chf']) / (1+interest_rate)**np.arange(1, invst_maturity+1)
        npv = (-row['estim_pvinstcost_chf']) + np.sum(pv_cashflow)
        return npv
    npv_df['NPV_uid'] = npv_df.apply(compute_npv, axis=1)
    npv_df.to_parquet(f'{data_path}/output/pvalloc_run/npv_df.parquet')
    npv_df.to_csv(f'{wd_path}/npv_df.csv', index=False)
    checkpoint_to_logfile(f'NPV calculation complete', log_name, 1, show_debug_prints_def)

    return npv_df



# ------------------------------------------------------------------------------------------------------
# UPDATE GRID PREMIUM TS
# ------------------------------------------------------------------------------------------------------
def update_gridprem(
        pvalloc_settings,
        topo_func, 
        pred_inst_df_func,
        ts_list_func, 
        ts_names_func,):
    
    # setup -----------------------------------------------------
    name_dir_import_def = pvalloc_settings['name_dir_import']
    data_path_def = pvalloc_settings['data_path']
    wd_path_def = pvalloc_settings['wd_path']
    show_debug_prints_def = pvalloc_settings['show_debug_prints']
    log_file_name_def = pvalloc_settings['log_file_name']
    gridtiers = pvalloc_settings['gridprem_adjustment_specs']['tiers']
    gridtiers_colnames = pvalloc_settings['gridprem_adjustment_specs']['colnames']
    print_to_logfile(f'run function: update_gridprem', log_file_name_def)
    
    topo = topo_func
    pred_inst_df = pred_inst_df_func
    ts_list = ts_list_func
    ts_names = ts_names_func

    # get inst df -----------------------------------------------------
    egid_with_inst = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == True]

    egid_list, bfs_list, gklas_list, demandtype_list, grid_node_list = [], [], [], [], []
    pvinst_list, pvsource_list, pvid_list, pv_tarif_Rp_kWh_list = [], [], [], []
    df_uid_list, flaeche_list, stromertrag_list, ausrichtung_list, neigung_list, elecpri_list = [], [], [], [], [], []

    for egid in egid_with_inst:
        topo_egid = topo[egid]
        partitions = topo_egid.get('solkat_partitions')

        for k_p, v_p in partitions.items():
            egid_list.append(egid)
            bfs_list.append(topo_egid.get('gwr_info').get('bfs'))
            gklas_list.append(topo_egid.get('gwr_info').get('gklas'))
            # demandtype_list.append(topo_egid.get('demand_type'))
            grid_node_list.append(topo_egid.get('grid_node'))
            pvinst_list.append(topo_egid.get('pv_inst').get('inst_TF'))
            pvsource_list.append(topo_egid.get('pv_inst').get('info_source'))
            pvid_list.append(topo_egid.get('pv_inst').get('xtf_id'))
            df_uid_list.append(k_p)
            flaeche_list.append(v_p.get('FLAECHE'))
            stromertrag_list.append(v_p.get('STROMERTRAG'))
            ausrichtung_list.append(v_p.get('AUSRICHTUNG'))
            neigung_list.append(v_p.get('NEIGUNG'))
            # elecpri_list.append(topo_egid.get('elecpri_Rp_kWh'))
            
    inst_df = pd.DataFrame({'EGID': egid_list, 'bfs': bfs_list, 'gklas': gklas_list, 'grid_node' : grid_node_list,
                            'pvinst_TF': pvinst_list, 'pvsource': pvsource_list, 'pvid': pvid_list, 
                            'df_uid': df_uid_list, 'FLAECHE': flaeche_list, 'STROMERTRAG': stromertrag_list, 
                            'AUSRICHTUNG': ausrichtung_list, 'NEIGUNG': neigung_list})

    # get gridtiers_df
    data = [(k, v[0], v[1]) for k, v in gridtiers.items()]
    gridtiers_df = pd.DataFrame(data, columns=gridtiers_colnames)
    subdf = pd.read_parquet(f'{data_path_def}/output/pvalloc_run/topo_subdf/subdf_0to1499.parquet')

    # find meteo_ts in ts_names
    meteo_ts = ts_list[ts_names.index('meteo_ts')]

    # find gridprem_ts in ts_names
    gridprem_ts = ts_list[ts_names.index('gridprem_ts')]

    #####################
    # BOOKMARK
    #####################
    inst_df['pvsource'].value_counts()

    






    print('asdf')