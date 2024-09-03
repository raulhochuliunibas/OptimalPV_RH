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
        groubpby_cols_func, agg_cols_func,
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
    groupby_cols = groubpby_cols_func
    agg_cols = agg_cols_func


    # import -----------------------------------------------------
    angle_tilt_df = df_list[df_names.index('angle_tilt_df')]
    demandtypes_ts = ts_list[ts_names.index('demandtypes_ts')]
    meteo_ts = ts_list[ts_names.index('meteo_ts')]
    gridprem_ts = ts_list[ts_names.index('gridprem_ts')]

    # NOTE: clean up when no longer necessary
    if 'high_wiHP' in demandtypes_ts.columns:
        demandtypes_ts.rename(columns={'high_wiHP': 'high_DEMANDprox_wiHP', 'low_wiHP': 'low_DEMANDprox_wiHP', 'high_noHP': 'high_DEMANDprox_noHP', 'low_noHP': 'low_DEMANDprox_noHP'}, inplace=True)


    # TOPO to DF ----------------------------------------------------------------
    # solkat_combo_df_exists = os.path.exists(f'{pvalloc_settings["interim_path"]}/solkat_combo_df.parquet')
    # if pvalloc_settings['recalc_economics_topo_df']:
    no_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == False]
    with_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == True]

    egid_list, df_uid_list, bfs_list, gklas_list, demandtype_list, grid_node_list  = [], [], [], [], [], []
    pvinst_list, pvsource_list, pvid_list, pv_tarif_Rp_kWh_list = [], [], [], []
    flaeche_list, stromertrag_list, ausrichtung_list, neigung_list, elecpri_list = [], [], [], [], []

    keys = list(topo.keys())

    for k,v in topo.items():
        partitions = v.get('solkat_partitions')

        for k_p, v_p in partitions.items():
            egid_list.append(k)
            df_uid_list.append(k_p)
            bfs_list.append(v.get('gwr_info').get('bfs'))
            gklas_list.append(v.get('gwr_info').get('gklas'))
            demandtype_list.append(v.get('demand_type'))
            grid_node_list.append(v.get('grid_node'))

            pvinst_list.append(v.get('pv_inst').get('inst_TF'))
            pvsource_list.append(v.get('pv_inst').get('info_source'))
            pvid_list.append(v.get('pv_inst').get('xtf_id'))
            pv_tarif_Rp_kWh_list.append(v.get('pvtarif_Rp_kWh'))

            flaeche_list.append(v_p.get('FLAECHE'))
            ausrichtung_list.append(v_p.get('AUSRICHTUNG'))
            stromertrag_list.append(v_p.get('STROMERTRAG'))
            neigung_list.append(v_p.get('NEIGUNG'))
            elecpri_list.append(v.get('elecpri_Rp_kWh'))
            

        
    topo_df = pd.DataFrame({'EGID': egid_list, 'df_uid': df_uid_list, 'bfs': bfs_list,
                            'gklas': gklas_list, 'demandtype': demandtype_list, 'grid_node': grid_node_list,

                            'pvinst_TF': pvinst_list, 'pvsource': pvsource_list, 'pvid': pvid_list,
                            'pv_tarif_Rp_kWh': pv_tarif_Rp_kWh_list,

                            'FLAECHE': flaeche_list, 'AUSRICHTUNG': ausrichtung_list, 
                            'STROMERTRAG': stromertrag_list, 'NEIGUNG': neigung_list, 
                            'elecpri_Rp_kWh': elecpri_list})
    
    # groupby_cols = ['EGID', 'df_uid', 'grid_node', 'bfs', 'gklas', 'demandtype',
    #                 'pvinst_TF', 'pvsource', 'pvid', 'pv_tarif_Rp_kWh', 'elecpri_Rp_kWh', 
    #                 'FLAECHE', 'AUSRICHTUNG', 'STROMERTRAG', 'NEIGUNG', 'angletilt_factor']
    # agg_cols = ['econ_inc_chf', 'econ_spend_chf']

    topo_df.to_parquet(f'{data_path}/output/pvalloc_run/topo_df.parquet')
    

    # GET ECONOMIC VALUES FOR NPV CALCULATION ----------------------------------------------
    topo_subdf_partitioner = pvalloc_settings['algorithm_specs']['topo_subdf_partitioner']
    selfconsum_rate = pvalloc_settings['tech_economic_specs']['self_consumption_ifapplicable']

    # round NEIGUNG + AUSRICHTUNG to 5 for easier computation
    topo_df['NEIGUNG'] = topo_df['NEIGUNG'].apply(lambda x: round(x / 5) * 5)
    topo_df['AUSRICHTUNG'] = topo_df['AUSRICHTUNG'].apply(lambda x: round(x / 10) * 10)
    
    def lookup_angle_tilt_efficiency(row, angle_tilt_df):
        try:
            return angle_tilt_df.loc[(row['AUSRICHTUNG'], row['NEIGUNG']), 'efficiency_factor']
        except KeyError:
            return 0
    topo_df['angletilt_factor'] = topo_df.apply(lambda r: lookup_angle_tilt_efficiency(r, angle_tilt_df), axis=1)

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
        subdf = subdf.assign(pvprod_kW = (subdf['radiation'] * subdf['FLAECHE'] * subdf['angletilt_factor']) / 1000).drop(columns=['meteo_loc', 'radiation'])
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
        subdf['selfconsum'] = np.minimum(subdf['pvprod_kW'], subdf['demand']) * selfconsum_rate
        subdf['netdemand'] = subdf['demand'] - subdf['selfconsum']
        subdf['netfeedin'] = subdf['pvprod_kW'] - subdf['selfconsum']

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
    topo_agg_df['FLAECH_angletilt'] = topo_agg_df['FLAECHE'] * topo_agg_df['angletilt_factor']

    counts = topo_agg_df['EGID'].value_counts()
    topo_agg_df['EGID_count'] = topo_agg_df['EGID'].map(counts)
    topo_agg_df = topo_agg_df.loc[topo_agg_df['EGID_count'] < pvalloc_settings['gwr_selection_specs']['solkat_max_n_partitions']]
    print_to_logfile(f'ATTENTION: needed to drop {df_before["EGID"].nunique() - topo_agg_df["EGID"].nunique()}/{df_before["EGID"].nunique()} EGID, {df_before.shape[0] - topo_agg_df.shape[0]} partitions, because n_partitions > {pvalloc_settings["gwr_selection_specs"]["solkat_max_n_partitions"]}', log_name)


    # create all partition combos of topo_df ----------
    print_to_logfile(f'\nCreate all possible combos of partitions for {topo_agg_df["EGID"].nunique()} EGIDs', log_name)
    estim_instcost_chfpkW, estim_instcost_chftotal = initial.get_estim_instcost_function(pvalloc_settings)
    agg_npry = np.array(topo_agg_df)

    egid_list, combo_df_uid_list, df_uid_list, bfs_list, gklas_list, demandtype_list, grid_node_list = [], [], [], [], [], [], []
    pvinst_list, pvsource_list, pvid_list, pv_tarif_Rp_kWh_list = [], [], [], []
    flaeche_list, stromertrag_list, ausrichtung_list, neigung_list, elecpri_Rp_kWh_list = [], [], [], [], []
    flaech_angletilt_list, econ_inc_chf_list, econ_spend_chf_list = [], [], []

    egid = topo_agg_df['EGID'].unique()[0]
    combos_counter = topo_agg_df['EGID'].nunique() // 5
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
                # df_uid_list.append(list(combo))
                bfs_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('bfs')][0])
                gklas_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('gklas')][0])
                demandtype_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('demandtype')][0])
                grid_node_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('grid_node')][0])

                pvinst_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('pvinst_TF')][0])
                pvsource_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('pvsource')][0])
                pvid_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('pvid')][0])
                pv_tarif_Rp_kWh_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('pv_tarif_Rp_kWh')][0]) 
                elecpri_Rp_kWh_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('elecpri_Rp_kWh')][0])

                flaeche_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('FLAECHE')].sum())
                # stromertrag_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('STROMERTRAG')].sum())
                # ausrichtung_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('AUSRICHTUNG')][0])
                # neigung_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('NEIGUNG')][0])

                flaech_angletilt_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('FLAECH_angletilt')].sum())
                econ_inc_chf_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('econ_inc_chf')].sum())
                econ_spend_chf_list.append(agg_npry[mask_egid_topo_agg_df, topo_agg_df.columns.get_loc('econ_spend_chf')].sum())
        
        if i % combos_counter == 0:
            # checkpoint_to_logfile(f'  combo creation complete for {i} of {topo_agg_df["EGID"].nunique()} EGIDs', log_name, 1)
            checkpoint_to_logfile(f'-- combo complete for {i} of {topo_agg_df["EGID"].nunique()} EGIDs {7*"-"}  (stamp: {datetime.now()})', log_name)

    checkpoint_to_logfile(f'end combo creation of partitions', log_name, 1, True)

    npv_df = pd.DataFrame({'EGID': egid_list, 'df_uid_combo': combo_df_uid_list, 'bfs': bfs_list,
                           'gklas': gklas_list, 'demandtype': demandtype_list, 'grid_node': grid_node_list,

                            'pvinst_TF': pvinst_list, 'pvsource': pvsource_list, 'pvid': pvid_list,
                            'pv_tarif_Rp_kWh': pv_tarif_Rp_kWh_list, 'elecpri_Rp_kWh': elecpri_Rp_kWh_list,

                            'FLAECHE': flaeche_list, 
                            # 'STROMERTRAG': stromertrag_list,
                            # 'AUSRICHTUNG': ausrichtung_list, 'NEIGUNG': neigung_list,
                        
                            'FLAECH_angletilt': flaech_angletilt_list,
                            'econ_inc_chf': econ_inc_chf_list, 'econ_spend_chf': econ_spend_chf_list})
    
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
        df_list_func, df_names_func,
        ts_list_func, ts_names_func,
        month_func,):
    
    # setup -----------------------------------------------------
    name_dir_import_def = pvalloc_settings['name_dir_import']
    data_path_def = pvalloc_settings['data_path']
    wd_path_def = pvalloc_settings['wd_path']
    show_debug_prints_def = pvalloc_settings['show_debug_prints']
    log_file_name_def = pvalloc_settings['log_file_name']

    gridtiers = pvalloc_settings['gridprem_adjustment_specs']['tiers']
    gridtiers_colnames = pvalloc_settings['gridprem_adjustment_specs']['colnames']
    conv_m2toKWP = pvalloc_settings['tech_economic_specs']['conversion_m2tokW']
    topo_subdf_partitioner = pvalloc_settings['algorithm_specs']['topo_subdf_partitioner']
    checkpoint_to_logfile(f'    run function: update_gridprem', log_file_name_def, 1, show_debug_prints_def)
    
    topo = topo_func
    pred_inst_df = pred_inst_df_func
    df_list, df_names = df_list_func, df_names_func
    ts_list, ts_names = ts_list_func, ts_names_func
    m = month_func



    # import -----------------------------------------------------
    # get inst df ------
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

    # get gridtiers_df ------
    data = [(k, v[0], v[1]) for k, v in gridtiers.items()]
    gridtiers_df = pd.DataFrame(data, columns=gridtiers_colnames)

    # get data from df_list + ts_list ------
    meteo_ts = ts_list[ts_names.index('meteo_ts')]
    gridprem_ts = ts_list[ts_names.index('gridprem_ts')]
    angle_tilt_df = df_list[df_names.index('angle_tilt_df')]
    pv = df_list[df_names.index('pv')]
    Map_egid_nodes = df_list[df_names.index('Map_egid_nodes')]


    # calculate feedin HOY ----------------------------------------
    def lookup_angle_tilt_efficiency(row, angle_tilt_df):
        try:
            return angle_tilt_df.loc[(row['AUSRICHTUNG'], row['NEIGUNG']), 'efficiency_factor']
        except KeyError:
            return 0
    inst_df['NEIGUNG'] = inst_df['NEIGUNG'].apply(lambda x: round(x / 5) * 5)
    inst_df['AUSRICHTUNG'] = inst_df['AUSRICHTUNG'].apply(lambda x: round(x / 10) * 10)
    inst_df['angletilt_factor'] = inst_df.apply(lambda r: lookup_angle_tilt_efficiency(r, angle_tilt_df), axis=1)
    
    pv['pvsource'] = 'pv_df'
    pv['pvid'] = pv['xtf_id']

    agg_gridnode_df_list = []
    agg_gridnode_df_path = f'{data_path_def}output/pvalloc_run/agg_gridnode_df'
    if not os.path.exists(agg_gridnode_df_path):
        os.makedirs(agg_gridnode_df_path)
    
    # ---
    # dfuids = inst_df['df_uid'].unique()
    egids = inst_df['EGID'].unique()

    # ---
    stepsize = topo_subdf_partitioner if len(egids) > topo_subdf_partitioner else len(egids)
    tranche_counter = 0
    for i in range(1, len(egids), stepsize):
        tranche_counter += 1
        # print_to_logfile(f'-- gridnode_subdf {tranche_counter}/{len(range(0, len(egids), stepsize))} {7*"-"}  (stamp: {datetime.now()})', log_file_name_def)
        subinst = inst_df[inst_df['EGID'].isin(egids[i:i+stepsize])].copy()
    # ---
    # for i,egid in enumerate(egids):
    #     subinst = inst_df[inst_df['EGID'] == egid].copy()
        
        subinst['meteo_loc'] = 'Basel'
        meteo_ts['meteo_loc'] = 'Basel'
        subinst = subinst.merge(meteo_ts[['t', 'radiation', 'meteo_loc']], how='left', on='meteo_loc')

        # production per partition
        subinst = subinst.assign(pvprod_kW = (subinst['radiation'] * subinst['FLAECHE'] * subinst['angletilt_factor']) / 1000)
        subinst.drop(columns=['meteo_loc',], inplace=True)

        # special case: inst from pv_df
        if False:
        # if 'pv_df' in subinst['pvsource'].unique():
            TotalPower = pv.loc[pv['xtf_id'].isin(subinst.loc[subinst['EGID'] == egid, 'pvid']), 'TotalPower'].sum()

            subinst = subinst.sort_values(by = 'STROMERTRAG', ascending=False)
            subinst['pvprod_kW'] = 0
            
            # t_steps = subinst['t'].unique()
            for t in subinst['t'].unique():
                timestep_df = subinst.loc[subinst['t'] == t]
                total_stromertrag = timestep_df['STROMERTRAG'].sum()

                for idx, row in timestep_df.iterrows():
                    share = row['STROMERTRAG'] / total_stromertrag
                    subinst.loc[idx, 'pvprod_kW'] = share * TotalPower

        agg_subinst = subinst.groupby(['grid_node', 't']).agg({'pvprod_kW': 'sum'}).reset_index()
        agg_gridnode_df_list.append(subinst)
        agg_subinst.to_parquet(f'{agg_gridnode_df_path}/agg_gridnode_df_{i}to{i+stepsize-1}.parquet')

        checkpoint_to_logfile(f'    Tranche {i} of {len(range(0, len(egids), stepsize))} update gridprem, computed PV prod per node', log_file_name_def, 1, show_debug_prints_def)

    gridnode_df = pd.concat(agg_gridnode_df_list)
    gridnode_df = gridnode_df.groupby(['grid_node', 't']).agg({'pvprod_kW': 'sum'}).reset_index()
    gridnode_df.to_parquet(f'{data_path_def}output/pvalloc_run/gridnode_df.parquet')


    # update gridprem_ts ----------------------------------------
    gridnode_df.sort_values(by=['pvprod_kW'], ascending=False, inplace=True)
    gridprem_ts = gridprem_ts.merge(gridnode_df[['grid_node', 't', 'pvprod_kW']], how='left', on=['grid_node', 't'])

    conditions = [(gridprem_ts['pvprod_kW'] > gridtiers_df.loc[i, 'vltg_threshold']) for i in range(len(gridtiers_df))]
    choices = [gridtiers_df.loc[i, 'gridprem_plusRp_kWh'] for i in range(len(gridtiers_df))]
    gridprem_ts['prem_Rp_kWh'] = np.select(conditions, choices, default=0)

    gridprem_ts.drop(columns='pvprod_kW', inplace=True)



    # export gridprem_ts ----------------------------------------
    gridprem_ts.to_parquet(f'{data_path_def}output/pvalloc_run/gridprem_ts.parquet')

    gridprem_by_M_path = f'{data_path_def}output/pvalloc_run/pred_gridprem_by_M'
    if not os.path.exists(gridprem_by_M_path):
        os.makedirs(gridprem_by_M_path)
    gridprem_ts.to_parquet(f'{data_path_def}output/pvalloc_run/pred_gridprem_by_M/gridprem_ts_{m}.parquet')
    gridprem_ts.to_csv(f'{data_path_def}output/pvalloc_run/pred_gridprem_by_M/gridprem_ts_{m}.csv', index=False)

    return gridprem_ts


# ------------------------------------------------------------------------------------------------------
# UPDATE NPV_DF with NEW GRID PREMIUM TS
# ------------------------------------------------------------------------------------------------------
def update_npv_df(pvalloc_settings,
                  topo, 
                  pred_inst_df,
                  groupby_cols_func, agg_cols_func,
                  df_list, df_names,
                  ts_list, ts_names,
                  m, 
                  gridprem_ts_func,
                  npv_nopv_df_func,):
    
    # setup -----------------------------------------------------
    name_dir_import_def = pvalloc_settings['name_dir_import']
    data_path_def = pvalloc_settings['data_path']
    wd_path_def = pvalloc_settings['wd_path']
    show_debug_prints_def = pvalloc_settings['show_debug_prints']
    log_file_name_def = pvalloc_settings['log_file_name']

    topo_subdf_partitioner = pvalloc_settings['algorithm_specs']['topo_subdf_partitioner']
    selfconsum_rate = pvalloc_settings['tech_economic_specs']['self_consumption_ifapplicable']
    interest_rate = pvalloc_settings['tech_economic_specs']['interest_rate']
    invst_maturity = pvalloc_settings['tech_economic_specs']['invst_maturity']

    groupby_cols = groupby_cols_func
    agg_cols = agg_cols_func
    gridprem_ts = gridprem_ts_func
    npv_nopv_df = npv_nopv_df_func
    checkpoint_to_logfile(f'    run: update_npv_df', log_file_name_def, 1, show_debug_prints_def)


    # import -----------------------------------------------------
    meteo_ts = ts_list[ts_names.index('meteo_ts')]
    demandtypes_ts = ts_list[ts_names.index('demandtypes_ts')]
    Map_egid_demandtypes = df_list[df_names.index('Map_egid_demandtypes')]

    # recalc NPV -----------------------------------------------------
    agg_subdf_list = []
    dfuids = npv_nopv_df['df_uid_combo'].unique()
    subdf_path = f'{data_path_def}/output/pvalloc_run/npv_nopv_subdf'

    if not os.path.exists(subdf_path):
        os.makedirs(subdf_path)

    stepsize = topo_subdf_partitioner if len(dfuids) > topo_subdf_partitioner else len(dfuids)
    tranche_counter = 0
    for i in range(0, len(dfuids), stepsize):
        tranche_counter += 1
 
        # merge production, grid prem + demand to partitions ----------
        subnopv = npv_nopv_df[npv_nopv_df['df_uid_combo'].isin(dfuids[i:i+stepsize])].copy()

        # merge with meteo
        subnopv['meteo_loc'] = 'Basel'
        subnopv = subnopv.merge(meteo_ts[['t', 'radiation', 'meteo_loc']], how='left', on='meteo_loc')
        subnopv = subnopv.assign(pvprod_kW = (subnopv['radiation'] * subnopv['FLAECH_angletilt']) / 1000).drop(columns=['meteo_loc', 'radiation'])

        # merge with demand
        subnopv['grid_node_loc'] = 'BS001'
        gridprem_ts['grid_node_loc'] = 'BS001'
        subnopv = subnopv.merge(gridprem_ts[['t', 'prem_Rp_kWh', 'grid_node_loc']], how='left', on=['t', 'grid_node_loc'])

        # merge with demand 
        demandtypes_names = [c for c in demandtypes_ts.columns if 'DEMANDprox' in c]
        demandtypes_melt = demandtypes_ts.melt(id_vars='t', value_vars=demandtypes_names, var_name= 'demandtype', value_name= 'demand')
        subnopv = subnopv.merge(demandtypes_melt, how='left', on=['t', 'demandtype'])

        checkpoint_to_logfile(f'        Merged meteo, gridprem and demand to subnopv {i} to {i+stepsize-1}', log_file_name_def, 1, show_debug_prints_def)

        # compute production + selfconsumption + netdemand
        subnopv['selfconsum'] = np.minimum(subnopv['pvprod_kW'], subnopv['demand']) * selfconsum_rate
        subnopv['netdemand'] = subnopv['demand'] - subnopv['selfconsum']
        subnopv['netfeedin'] = subnopv['pvprod_kW'] - subnopv['selfconsum']

        subnopv['econ_inc_chf'] = ((subnopv['netfeedin'] * subnopv['pv_tarif_Rp_kWh']) /100) + ((subnopv['selfconsum'] * subnopv['elecpri_Rp_kWh']) /100)
        subnopv['econ_spend_chf'] = ((subnopv['netfeedin'] * subnopv['prem_Rp_kWh'])) + ((subnopv['netdemand'] * subnopv['elecpri_Rp_kWh']) /100)

        subnopv.to_parquet(f'{subdf_path}/subnopv_{i}to{i+stepsize-1}.parquet')

        agg_nopv = subnopv.groupby(groupby_cols).agg({agg_cols[0]: 'sum', agg_cols[1]: 'sum'}).reset_index()
        agg_subdf_list.append(agg_nopv)
        agg_nopv.to_parquet(f'{subdf_path}/agg_nopv_{i}to{i+stepsize-1}.parquet')

        checkpoint_to_logfile(f'        NPV calc for subnopv {i} to {i+stepsize-1}', log_file_name_def, 1, show_debug_prints_def)
        # checkpoint_to_logfile(f'    Tranche {i} of {len(range(0, len(dfuids), stepsize))} updating NPV per df_uids_combo', log_file_name_def, 1, show_debug_prints_def)
        del subnopv

    agg_nopv_df = pd.concat(agg_subdf_list)
    del agg_subdf_list

    # NPV calculation
    estim_instcost_chfpkW, estim_instcost_chftotal = initial.get_estim_instcost_function(pvalloc_settings)
    agg_nopv_df['estim_pvinstcost_chf'] = estim_instcost_chfpkW(agg_nopv_df['FLAECHE'] * pvalloc_settings['tech_economic_specs']['conversion_m2tokW'])
    def compute_npv(row):
        pv_cashflow = (row['econ_inc_chf'] - row['econ_spend_chf']) / (1+interest_rate)**np.arange(1, invst_maturity+1)
        npv = (-row['estim_pvinstcost_chf']) + np.sum(pv_cashflow)
        return npv

    agg_nopv_df['NPV_uid'] = agg_nopv_df.apply(compute_npv, axis=1)
    agg_nopv_df.to_parquet(f'{data_path_def}/output/pvalloc_run/agg_nopv_df.parquet')
    agg_nopv_df.to_csv(f'{wd_path_def}/agg_nopv_df.csv', index=False)
    
    return agg_nopv_df


