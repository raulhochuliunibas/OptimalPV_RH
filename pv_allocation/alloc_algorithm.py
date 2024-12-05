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
import copy
from datetime import datetime

from pyarrow.parquet import ParquetFile
import pyarrow as pa

# own functions 
sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile

import pv_allocation.initialization_large_functions as initial

# ------------------------------------------------------------------------------------------------------
# CALCULATE ECONOMIC INDICATORS OF TOPOLOGY
# ------------------------------------------------------------------------------------------------------
def calc_economics_in_topo_df(
        pvalloc_settings, 
        topo, 
        df_list, df_names, 
        ts_list, ts_names,):
    
    # setup -----------------------------------------------------
    name_dir_import = pvalloc_settings['name_dir_import']
    wd_path = pvalloc_settings['wd_path']
    data_path = f'{wd_path}_data'
    show_debug_prints = pvalloc_settings['show_debug_prints']
    log_file_name = pvalloc_settings['log_file_name']
    print_to_logfile(f'run function: calc_economics_in_topo_df', log_file_name)



    # import -----------------------------------------------------
    angle_tilt_df = df_list[df_names.index('angle_tilt_df')]
    solkat_month = df_list[df_names.index('solkat_month')]
    demandtypes_ts = ts_list[ts_names.index('demandtypes_ts')]
    meteo_ts = ts_list[ts_names.index('meteo_ts')]




    # TOPO to DF =============================================
    # solkat_combo_df_exists = os.path.exists(f'{pvalloc_settings["interim_path"]}/solkat_combo_df.parquet')
    # if pvalloc_settings['recalc_economics_topo_df']:
    topo = topo
    no_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == False]
    with_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == True]

    egid_list, df_uid_list, bfs_list, gklas_list, demandtype_list, grid_node_list  = [], [], [], [], [], []
    inst_list, info_source_list, pvdf_totalpower_list, pvid_list, pv_tarif_Rp_kWh_list = [], [], [], [], []
    flaeche_list, stromertrag_list, ausrichtung_list, neigung_list, elecpri_list = [], [], [], [], []

    keys = list(topo.keys())

    for k,v in topo.items():
        # if k in no_pv_egid:
        # ADJUSTMENT: this needs to be removed, because I also need to calculate the pvproduction_kW per house 
        # later when quantifying the grid feedin per grid node
        partitions = v.get('solkat_partitions')

        for k_p, v_p in partitions.items():
            egid_list.append(k)
            df_uid_list.append(k_p)
            bfs_list.append(v.get('gwr_info').get('bfs'))
            gklas_list.append(v.get('gwr_info').get('gklas'))
            demandtype_list.append(v.get('demand_type'))
            grid_node_list.append(v.get('grid_node'))

            inst_list.append(v.get('pv_inst').get('inst_TF'))
            info_source_list.append(v.get('pv_inst').get('info_source'))
            pvid_list.append(v.get('pv_inst').get('xtf_id'))
            pv_tarif_Rp_kWh_list.append(v.get('pvtarif_Rp_kWh'))
            pvdf_totalpower_list.append(v.get('pv_inst').get('TotalPower'))

            flaeche_list.append(v_p.get('FLAECHE'))
            ausrichtung_list.append(v_p.get('AUSRICHTUNG'))
            stromertrag_list.append(v_p.get('STROMERTRAG'))
            neigung_list.append(v_p.get('NEIGUNG'))
            elecpri_list.append(v.get('elecpri_Rp_kWh'))
                
        
    topo_df = pd.DataFrame({'EGID': egid_list, 'df_uid': df_uid_list, 'bfs': bfs_list,
                            'gklas': gklas_list, 'demandtype': demandtype_list, 'grid_node': grid_node_list,

                            'inst_TF': inst_list, 'info_source': info_source_list, 'pvid': pvid_list,
                            'pv_tarif_Rp_kWh': pv_tarif_Rp_kWh_list, 'TotalPower': pvdf_totalpower_list,

                            'FLAECHE': flaeche_list, 'AUSRICHTUNG': ausrichtung_list, 
                            'STROMERTRAG': stromertrag_list, 'NEIGUNG': neigung_list, 
                            'elecpri_Rp_kWh': elecpri_list})
    

    # make or clear dir for subdfs ----------------------------------------------
    subdf_path = f'{data_path}/output/pvalloc_run/topo_time_subdf'

    if not os.path.exists(subdf_path):
        os.makedirs(subdf_path)
    else:
        old_files = glob.glob(f'{subdf_path}/*')
        for f in old_files:
            os.remove(f)
    

    # round NEIGUNG + AUSRICHTUNG to 5 for easier computation
    topo_df['NEIGUNG'] = topo_df['NEIGUNG'].apply(lambda x: round(x / 5) * 5)
    topo_df['AUSRICHTUNG'] = topo_df['AUSRICHTUNG'].apply(lambda x: round(x / 10) * 10)
    
    def lookup_angle_tilt_efficiency(row, angle_tilt_df):
        try:
            return angle_tilt_df.loc[(row['AUSRICHTUNG'], row['NEIGUNG']), 'efficiency_factor']
        except KeyError:
            return 0
    topo_df['angletilt_factor'] = topo_df.apply(lambda r: lookup_angle_tilt_efficiency(r, angle_tilt_df), axis=1)

    # transform TotalPower
    topo_df['TotalPower'] = topo_df['TotalPower'].replace('', '0').astype(float)


    # MERGE + GET ECONOMIC VALUES FOR NPV CALCULATION =============================================
    topo_subdf_partitioner = pvalloc_settings['algorithm_specs']['topo_subdf_partitioner']
    
    share_roof_area_available = pvalloc_settings['tech_economic_specs']['share_roof_area_available']
    inverter_efficiency = pvalloc_settings['tech_economic_specs']['inverter_efficiency']
    panel_efficiency = pvalloc_settings['tech_economic_specs']['panel_efficiency']
    pvprod_calc_method = pvalloc_settings['tech_economic_specs']['pvprod_calc_method']
    kWpeak_per_m2 = pvalloc_settings['tech_economic_specs']['kWpeak_per_m2']

    flat_direct_rad_factor = pvalloc_settings['weather_specs']['flat_direct_rad_factor']
    flat_diffuse_rad_factor = pvalloc_settings['weather_specs']['flat_diffuse_rad_factor']


    egids = topo_df['EGID'].unique()

    stepsize = topo_subdf_partitioner if len(egids) > topo_subdf_partitioner else len(egids)
    tranche_counter = 0
    for i in range(0, len(egids), stepsize):

        tranche_counter += 1
        # print_to_logfile(f'-- merges to topo_time_subdf {tranche_counter}/{len(range(0, len(egids), stepsize))} tranches ({i} to {i+stepsize-1} egids.iloc) ,  {7*"-"}  (stamp: {datetime.now()})', log_file_name)
        subdf = copy.deepcopy(topo_df[topo_df['EGID'].isin(egids[i:i+stepsize])])


        # merge production, grid prem + demand to partitions ----------
        subdf['meteo_loc'] = 'Basel'
        meteo_ts['meteo_loc'] ='Basel' 
        
        # subdf = subdf.merge(meteo_ts[['rad_direct', 'rad_diffuse', 'temperature', 't', 'meteo_loc']], how='left', on='meteo_loc')
        subdf = subdf.merge(meteo_ts, how='left', on='meteo_loc')
        

        # add radiation per h to subdf ----------
        if pvalloc_settings['weather_specs']['radiation_to_pvprod_method'] == 'flat':
            subdf['radiation'] = subdf['rad_direct'] * flat_direct_rad_factor + subdf['rad_diffuse'] * flat_diffuse_rad_factor
            meteo_ts['radiation'] = meteo_ts['rad_direct'] * flat_direct_rad_factor + meteo_ts['rad_diffuse'] * flat_diffuse_rad_factor
            mean_top_radiation = meteo_ts['radiation'].nlargest(20).mean()

            subdf['radiation_rel_locmax'] = subdf['radiation'] / mean_top_radiation


        elif pvalloc_settings['weather_specs']['radiation_to_pvprod_method'] == 'dfuid_ind':
            solkat_month.rename(columns={'DF_UID': 'df_uid', 'MONAT': 'month'}, inplace=True)
            solkat_month['month'] = solkat_month['month'].astype(int)
            subdf['month'] = subdf['timestamp'].dt.month.astype(int)
            
        
            checkpoint_to_logfile(f'  start merge solkat_month to subdf {i} to {i+stepsize-1}', log_file_name, 1) if i < 2 else None
            subdf = subdf.merge(solkat_month[['df_uid', 'month', 'A_PARAM', 'B_PARAM', 'C_PARAM']], how='left', on=['df_uid', 'month'])
            checkpoint_to_logfile(f'  end merge solkat_month to subdf {i} to {i+stepsize-1}', log_file_name, 1) if i < 2 else None
            subdf['radiation'] = subdf['A_PARAM'] * subdf['rad_direct'] + subdf['B_PARAM'] * subdf['rad_diffuse'] + subdf['C_PARAM']

            meteo_ts['radiation'] = meteo_ts['rad_direct'] * flat_direct_rad_factor + meteo_ts['rad_diffuse'] * flat_diffuse_rad_factor
            meteo_ts['radiation_abc_param_1dfuid'] = meteo_ts['rad_direct'] * subdf['A_PARAM'].mean() + meteo_ts['rad_diffuse'] * subdf['B_PARAM'].mean() + subdf['C_PARAM'].mean()

            if pvalloc_settings['weather_specs']['rad_rel_loc_max_by'] == 'dfuid_specific':
                subdf_dfuid_topradation = subdf.groupby('df_uid')['radiation'].apply(lambda x: x.nlargest(20).mean()).reset_index()
                subdf_dfuid_topradation.rename(columns={'radiation': 'mean_top_radiation'}, inplace=True)
                subdf = subdf.merge(subdf_dfuid_topradation, how='left', on='df_uid')

                subdf['radiation_rel_locmax'] = subdf['radiation'] / subdf['mean_top_radiation']

            elif pvalloc_settings['weather_specs']['rad_rel_loc_max_by'] == 'all_HOY':
                mean_nlargest_rad_all_HOY = meteo_ts['radiation'].nlargest(20).mean()
                subdf['radiation_rel_locmax'] = subdf['radiation'] / mean_nlargest_rad_all_HOY

        # add panel_efficiency by time ----------
        if pvalloc_settings['panel_efficiency_specs']['variable_panel_efficiency_TF']:
            summer_months = pvalloc_settings['panel_efficiency_specs']['summer_months']
            hotsummer_hours = pvalloc_settings['panel_efficiency_specs']['hotsummer_hours']
            hot_hours_discount = pvalloc_settings['panel_efficiency_specs']['hot_hours_discount']

            HOY_weatheryear_df = pd.read_parquet(f'{data_path}/output/pvalloc_run/HOY_weatheryear_df.parquet')
            hot_hours_in_year = HOY_weatheryear_df.loc[(HOY_weatheryear_df['month'].isin(summer_months)) & (HOY_weatheryear_df['hour'].isin(hotsummer_hours))]
            subdf['panel_efficiency'] = np.where(
                subdf['t'].isin(hot_hours_in_year['t']),
                panel_efficiency * (1-hot_hours_discount),
                panel_efficiency)
            
        elif not pvalloc_settings['panel_efficiency_specs']['variable_panel_efficiency_TF']:
            subdf['panel_efficiency'] = panel_efficiency
            

        # attach demand profiles ----------
        demandtypes_names = [c for c in demandtypes_ts.columns if 'DEMANDprox' in c]
        demandtypes_melt = demandtypes_ts.melt(id_vars='t', value_vars=demandtypes_names, var_name= 'demandtype', value_name= 'demand')
        subdf = subdf.merge(demandtypes_melt, how='left', on=['t', 'demandtype'])
        subdf.rename(columns={'demand': 'demand_kW'}, inplace=True)
        # checkpoint_to_logfile(f'  end merge demandtypes for subdf {i} to {i+stepsize-1}', log_file_name, 1)


        # attach FLAECH_angletilt, might be usefull for later calculations
        subdf = subdf.assign(FLAECH_angletilt = subdf['FLAECHE'] * subdf['angletilt_factor'])

        # compute production ---------- 
        # pvprod method 1 (false, presented to frank 8.11.24. missing efficiency grade)
        if pvprod_calc_method == 'method1':    
            subdf = subdf.assign(pvprod_kW = (subdf['radiation'] * subdf['FLAECHE'] * subdf['angletilt_factor']) / 1000).drop(columns=['meteo_loc', 'radiation'])

        # pvprod method 2.1
        elif pvprod_calc_method == 'method2.1':   
            subdf['pvprod_kW'] = (subdf['radiation'] / 1000 ) *                     inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] * subdf['angletilt_factor']
            # subdf.drop(columns=['meteo_loc', 'radiation', 'radiation_rel_locmax'], inplace=True)
            formla_for_log_print = f"subdf['pvprod_kW'] = inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * (subdf['radiation'] / 1000 ) * subdf['FLAECHE'] * subdf['angletilt_factor']"

        # pvprod method 2.2
        elif pvprod_calc_method == 'method2.2':   
            subdf['pvprod_kW'] = (subdf['radiation'] / 1000 ) *                     inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] 
            # subdf.drop(columns=['meteo_loc', 'radiation', 'radiation_rel_locmax'], inplace=True)
            formla_for_log_print = f"subdf['pvprod_kW'] = inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * (subdf['radiation'] / 1000 ) * subdf['FLAECHE'] * subdf['angletilt_factor']"

        # pvprod method 3.1
        elif pvprod_calc_method == 'method3.1':
            subdf['pvprod_kW'] =  subdf['radiation_rel_locmax'] * kWpeak_per_m2 *   inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] * subdf['angletilt_factor']
            # subdf.drop(columns=['meteo_loc', 'radiation', 'radiation_rel_locmax'], inplace=True)
            formla_for_log_print = f"subdf['pvprod_kW'] = inverter_efficiency * share_roof_area_available * kWpeak_per_m2 * subdf['radiation_rel_locmax'] * subdf['FLAECHE'] * subdf['angletilt_factor']"

        # pvprod method 3.2
        elif pvprod_calc_method == 'method3.2':
            subdf['pvprod_kW'] =  subdf['radiation_rel_locmax'] * kWpeak_per_m2 *   inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] 
            # subdf.drop(columns=['meteo_loc', 'radiation', 'radiation_rel_locmax'], inplace=True)
            formla_for_log_print = f"subdf['pvprod_kW'] = inverter_efficiency * share_roof_area_available * kWpeak_per_m2 * subdf['radiation_rel_locmax'] * subdf['FLAECHE'] "


        # pvprod method 3
            # > 19.11.2024: no longer needed. from previous runs where I wanted to compare different pvprod_computations methods
        elif False:   
            subdf['pvprod_kW'] = inverter_efficiency * share_roof_area_available * (subdf['radiation'] / 1000 ) * subdf['FLAECHE'] * subdf['angletilt_factor']
            subdf.drop(columns=['meteo_loc', 'radiation'], inplace=True)
            print_to_logfile("* calculation formula for pv production per roof:\n   > subdf['pvprod_kW'] = inverter_efficiency * share_roof_area_available * (subdf['radiation'] / 1000 ) * subdf['FLAECHE'] * subdf['angletilt_factor']\n", log_file_name)
            
        # pvprod method 4
            # > 19.11.2024: because I dont have the same weather year as the calculations for the STROMERTRAG in solkat, it is not really feasible to back-engineer any type of shade deduction 
            #   coefficient that might bring any additional information. 
        elif False:  
            subdf['pvprod_kW_noshade'] =   (subdf['radiation'] / 1000 ) * subdf['FLAECHE'] # * subdf['angletilt_factor']
            # check if no_shade production calculation is larger than STROMERTRAG (should be, and then later corrected...)
            sum(subdf.loc[subdf['df_uid'] == subdf['df_uid'].unique()[0], 'pvprod_kW_noshade']), subdf.loc[subdf['df_uid'] == subdf['df_uid'].unique()[0], 'STROMERTRAG'].iloc[0]
            
            dfuid_subdf = subdf['df_uid'].unique()
            dfuid = dfuid_subdf[0]
            for dfuid in dfuid_subdf:
                dfuid_TF = subdf['df_uid'] == dfuid
                pvprod_kWhYear_noshade = subdf.loc[dfuid_TF, 'pvprod_kW_noshade'].sum()
                stromertrag_dfuid = subdf.loc[dfuid_TF, 'STROMERTRAG'].iloc[0]
                shading_factor = stromertrag_dfuid / pvprod_kWhYear_noshade
                
                if shading_factor > 1:
                    checkpoint_to_logfile(f' *ERROR* shading factor > 1 for df_uid: {dfuid}, EGID: {subdf.loc[dfuid_TF, "EGID"].unique()} ', log_file_name, 1)
                subdf.loc[dfuid_TF, 'pvprod_kW'] = subdf.loc[dfuid_TF, 'pvprod_kW_noshade'] * shading_factor
            subdf.drop(columns=['meteo_loc', 'radiation', 'pvprod_kW_noshade'], inplace=True)
            print_to_logfile("* calculation formula for pv production per roof:\n   > subdf['pvprod_kW'] = <retrofitted_shading_factor> * inverter_efficiency  * (subdf['radiation'] / 1000 ) * subdf['FLAECHE'] * subdf['angletilt_factor'] \n", log_file_name)
            

        # export subdf ----------------------------------------------
        subdf.to_parquet(f'{subdf_path}/topo_subdf_{i}to{i+stepsize-1}.parquet')
        if pvalloc_settings['export_csvs']:
            subdf.to_csv(f'{subdf_path}/topo_subdf_{i}to{i+stepsize-1}.csv', index=False)
        if i == 0:
            subdf.to_csv(f'{subdf_path}/topo_subdf_{i}to{i+stepsize-1}.csv', index=False)
        checkpoint_to_logfile(f'end merge to topo_time_subdf (tranche {tranche_counter}/{len(range(0, len(egids), stepsize))}, size {stepsize})', log_file_name, 1)


    # print computation formula for comparing methods
    print_to_logfile(f'* Computation formula for pv production per roof:\n{formla_for_log_print}', log_file_name)



# ------------------------------------------------------------------------------------------------------
# INITIATE GRID PREMIUM TS
# ------------------------------------------------------------------------------------------------------
def initiate_gridprem(
        pvalloc_settings,):
    wd_path = pvalloc_settings['wd_path']
    data_path = f'{wd_path}_data'
    name_dir_import = pvalloc_settings['name_dir_import']
    print_to_logfile(f'run function: initiate_gridprem', pvalloc_settings['log_file_name'])

    # setup -----------------------------------------------------
    if os.path.exists(f'{data_path}/output/pvalloc_run/gridprem_ts.parquet'):
        os.remove(f'{data_path}/output/pvalloc_run/gridprem_ts.parquet')    

    # import -----------------------------------------------------
    dsonodes_df = pd.read_parquet(f'{data_path}/output/{name_dir_import}/dsonodes_df.parquet')
    t_range = [f't_{t}' for t in range(1,8760 + 1)]

    dsonodes_df.drop(columns=['EGID'], inplace=True)
    gridprem_ts = pd.DataFrame(np.repeat(dsonodes_df.values, len(t_range), axis=0), columns=dsonodes_df.columns)  
    gridprem_ts['t'] = np.tile(t_range, len(dsonodes_df))
    gridprem_ts['prem_Rp_kWh'] = 0

    gridprem_ts = gridprem_ts[['t', 'grid_node', 'kVA_threshold', 'prem_Rp_kWh']]
    gridprem_ts.drop(columns='kVA_threshold', inplace=True)

    # export -----------------------------------------------------
    gridprem_ts.to_parquet(f'{data_path}/output/pvalloc_run/gridprem_ts.parquet')




# ------------------------------------------------------------------------------------------------------
# UPDATE GRID PREMIUM TS
# ------------------------------------------------------------------------------------------------------
def update_gridprem(
        pvalloc_settings,
        subdir_path,
        month_func, i_month_func):
    
    # setup -----------------------------------------------------
    name_dir_import = pvalloc_settings['name_dir_import']
    subdir_path_def = subdir_path
    wd_path = pvalloc_settings['wd_path']
    data_path = f'{wd_path}_data'
    show_debug_prints = pvalloc_settings['show_debug_prints']
    log_file_name = pvalloc_settings['log_file_name']

    gridtiers = pvalloc_settings['gridprem_adjustment_specs']['tiers']
    gridtiers_colnames = pvalloc_settings['gridprem_adjustment_specs']['colnames']
    gridtiers_power_factor = pvalloc_settings['gridprem_adjustment_specs']['power_factor']
    perf_factor_1kVA_to_XkW = pvalloc_settings['gridprem_adjustment_specs']['perf_factor_1kVA_to_XkW']
    print_to_logfile(f'run function: update_gridprem', log_file_name)

    m = month_func
    i_m = i_month_func

    # import  -----------------------------------------------------
    # topo = json.load(open(f'{data_path}/output/pvalloc_run/topo_egid.json', 'r'))
    # dsonodes_df = pd.read_parquet(f'{data_path}/output/{name_dir_import}/dsonodes_df.parquet')
    # gridprem_ts = pd.read_parquet(f'{data_path}/output/pvalloc_run/gridprem_ts.parquet')
    # pv = df_list[df_names.index('pv')]
    topo = json.load(open(f'{subdir_path_def}/topo_egid.json', 'r'))
    dsonodes_df = pd.read_parquet(f'{subdir_path_def}/dsonodes_df.parquet')
    gridprem_ts = pd.read_parquet(f'{subdir_path_def}/gridprem_ts.parquet')

    data = [(k, v[0], v[1]) for k, v in gridtiers.items()]
    gridtiers_df = pd.DataFrame(data, columns=gridtiers_colnames)

    egid_list, info_source_list, inst_TF_list = [], [], []
    for k,v in topo.items():
        egid_list.append(k)
        if v.get('pv_inst', {}).get('inst_TF') == True:
            info_source_list.append(v.get('pv_inst').get('info_source'))
            inst_TF_list.append(v.get('pv_inst').get('inst_TF'))
        else: 
            info_source_list.append('')
            inst_TF_list.append(False)
    Map_infosource_egid = pd.DataFrame({'EGID': egid_list, 'info_source': info_source_list, 'inst_TF': inst_TF_list}, index=egid_list)


    # import topo_time_subdfs -----------------------------------------------------
    # topo_subdf_paths = glob.glob(f'{data_path}/output/pvalloc_run/topo_time_subdf/*.parquet')
    topo_subdf_paths = glob.glob(f'{subdir_path_def}/topo_subdf_*.parquet')
    agg_subinst_df_list = []
    no_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == False]
    wi_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == True]

    i, path = 0, topo_subdf_paths[0]
    for i, path in enumerate(topo_subdf_paths):
        subdf = pd.read_parquet(path)

        subdf_updated = copy.deepcopy(subdf)
        subdf_updated.drop(columns=['info_source', 'inst_TF'], inplace=True)
        subdf_updated = subdf_updated.merge(Map_infosource_egid[['EGID', 'info_source', 'inst_TF']], how='left', on='EGID')
        # updated_instTF_srs, update_infosource_srs = subdf_updated['inst_TF'].fillna(subdf['inst_TF']), subdf_updated['info_source'].fillna(subdf['info_source'])
        # subdf['inst_TF'], subdf['info_source'] = updated_instTF_srs.infer_objects(copy=False), update_infosource_srs.infer_objects(copy=False)

        # Only consider production for houses that have built a pv installation and substract selfconsumption from the production
        subinst = copy.deepcopy(subdf_updated.loc[subdf_updated['inst_TF']==True])
        pvprod_kW, demand_kW = subinst['pvprod_kW'].to_numpy(), subinst['demand_kW'].to_numpy()
        selfconsum_kW = np.minimum(pvprod_kW, demand_kW) * pvalloc_settings['tech_economic_specs']['self_consumption_ifapplicable']
        netdemand_kW = demand_kW - selfconsum_kW
        netfeedin_kW = pvprod_kW - selfconsum_kW

        subinst['feedin_kW'] = netfeedin_kW

        # NOTE: attempt for a more elaborate way to handle already installed installations
        if False:
            pv = pd.read_parquet(f'{subdir_path_def}/pv.parquet')
            pv['pvsource'] = 'pv_df'
            pv['pvid'] = pv['xtf_id']

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

        agg_subinst = subinst.groupby(['grid_node', 't']).agg({'feedin_kW': 'sum', 'pvprod_kW':'sum'}).reset_index()
        del subinst
        agg_subinst_df_list.append(agg_subinst)
    

    # build gridnode_df -----------------------------------------------------
    gridnode_df = pd.concat(agg_subinst_df_list)
    # groupby df again because grid nodes will be spreach accross multiple tranches
    gridnode_df = gridnode_df.groupby(['grid_node', 't']).agg({'feedin_kW': 'sum', 'pvprod_kW':'sum'}).reset_index() 

    # attach node thresholds 
    gridnode_df = gridnode_df.merge(dsonodes_df[['grid_node', 'kVA_threshold']], how='left', on='grid_node')
    gridnode_df['kW_threshold'] = gridnode_df['kVA_threshold'] * perf_factor_1kVA_to_XkW

    gridnode_df['feedin_kW_taken'] = np.where(gridnode_df['feedin_kW'] > gridnode_df['kW_threshold'], gridnode_df['kW_threshold'], gridnode_df['feedin_kW'])
    gridnode_df['feedin_kW_loss'] =  np.where(gridnode_df['feedin_kW'] > gridnode_df['kW_threshold'], gridnode_df['feedin_kW'] - gridnode_df['kW_threshold'], 0)

    
    # update gridprem_ts -----------------------------------------------------
    gridnode_df.sort_values(by=['feedin_kW_taken'], ascending=False)
    gridnode_df_for_prem = gridnode_df.groupby(['grid_node','kW_threshold', 't']).agg({'feedin_kW_taken': 'sum'}).reset_index().copy()
    gridprem_ts = gridprem_ts.merge(gridnode_df_for_prem[['grid_node', 't', 'kW_threshold', 'feedin_kW_taken']], how='left', on=['grid_node', 't'])
    gridprem_ts['feedin_kW_taken'] = gridprem_ts['feedin_kW_taken'].replace(np.nan, 0)
    gridprem_ts.sort_values(by=['feedin_kW_taken'], ascending=False)

    # gridtiers_df['kW_threshold'] = gridtiers_df['kVA_threshold'] / gridtiers_power_factor
    conditions, choices = [], []
    for i in range(len(gridtiers_df)):
        i_adj = len(gridtiers_df) - i -1 # order needs to be reversed, because otherwise first condition is always met and disregards the higher tiers
        conditions.append((gridprem_ts['feedin_kW_taken'] / gridprem_ts['kW_threshold'])  > gridtiers_df.loc[i_adj, 'used_node_capa_rate'])
        choices.append(gridtiers_df.loc[i_adj, 'gridprem_Rp_kWh'])
    gridprem_ts['prem_Rp_kWh'] = np.select(conditions, choices, default=gridprem_ts['prem_Rp_kWh'])
    gridprem_ts.drop(columns=['feedin_kW_taken', 'kW_threshold'], inplace=True)


    # EXPORT -----------------------------------------------------
    gridnode_df.to_parquet(f'{subdir_path_def}/gridnode_df.parquet')
    gridprem_ts.to_parquet(f'{subdir_path_def}/gridprem_ts.parquet')
    if pvalloc_settings['export_csvs']:
        gridnode_df.to_csv(f'{subdir_path_def}/gridnode_df.csv', index=False)
        gridprem_ts.to_csv(f'{subdir_path_def}/gridprem_ts.csv', index=False)


    # export by Month -----------------------------------------------------
    if pvalloc_settings['MC_loop_specs']['keep_files_month_iter_TF']:
        if i_m < pvalloc_settings['MC_loop_specs']['keep_files_month_iter_max']:
            # gridprem_node_by_M_path = f'{data_path}/output/pvalloc_run/pred_gridprem_node_by_M'
            gridprem_node_by_M_path = f'{subdir_path_def}/pred_gridprem_node_by_M'
            if not os.path.exists(gridprem_node_by_M_path):
                os.makedirs(gridprem_node_by_M_path)
            # elif os.path.exists(gridprem_node_by_M_path):
            #     old_files = glob.glob(f'{gridprem_node_by_M_path}/*')
            #     for f in old_files:
            #         os.remove(f)

            gridnode_df.to_parquet(f'{gridprem_node_by_M_path}/gridnode_df_{m}.parquet')
            gridprem_ts.to_parquet(f'{gridprem_node_by_M_path}/gridprem_ts_{m}.parquet')

            if pvalloc_settings['export_csvs']:
                gridnode_df.to_csv(f'{gridprem_node_by_M_path}/gridnode_df_{m}.csv', index=False)
                gridprem_ts.to_csv(f'{gridprem_node_by_M_path}/gridprem_ts_{m}.csv', index=False)
            if i_m < 5:
                gridnode_df.to_csv(f'{gridprem_node_by_M_path}/gridnode_df_{m}.csv', index=False)
                gridprem_ts.to_csv(f'{gridprem_node_by_M_path}/gridprem_ts_{m}.csv', index=False)

    checkpoint_to_logfile(f'exported gridprem_ts and gridnode_df', log_file_name, 1)



# ------------------------------------------------------------------------------------------------------
# UPDATE NPV_DF with NEW GRID PREMIUM TS
# ------------------------------------------------------------------------------------------------------
def update_npv_df(pvalloc_settings,
                  subdir_path,
                #   groupby_cols_func, agg_cols_func, 
                  month_func, i_month_func
                ):
    
    # setup -----------------------------------------------------
    name_dir_import = pvalloc_settings['name_dir_import']
    wd_path = pvalloc_settings['wd_path']
    data_path = f'{wd_path}_data'
    show_debug_prints = pvalloc_settings['show_debug_prints']
    log_file_name = pvalloc_settings['log_file_name']

    topo_subdf_partitioner = pvalloc_settings['algorithm_specs']['topo_subdf_partitioner']
    selfconsum_rate = pvalloc_settings['tech_economic_specs']['self_consumption_ifapplicable']
    interest_rate = pvalloc_settings['tech_economic_specs']['interest_rate']
    invst_maturity = pvalloc_settings['tech_economic_specs']['invst_maturity']
    kWpeak_per_m2 = pvalloc_settings['tech_economic_specs']['kWpeak_per_m2']
    share_roof_area_available = pvalloc_settings['tech_economic_specs']['share_roof_area_available']
    tweak_npv_excl_elec_demand = pvalloc_settings['algorithm_specs']['tweak_npv_excl_elec_demand']

    groupby_cols = pvalloc_settings['algorithm_specs']['npv_update_grouby_cols_topo_aggdf']
    agg_cols = pvalloc_settings['algorithm_specs']['npv_update_agg_cols_topo_aggdf']

    estim_instcost_chfpkW, estim_instcost_chftotal = initial.get_estim_instcost_function(pvalloc_settings)

    subdir_path_def = subdir_path
    m = month_func
    i_m = i_month_func
    print_to_logfile(f'run function: update_npv_df', log_file_name)


    # import -----------------------------------------------------
    # gridprem_ts = pd.read_parquet(f'{data_path}/output/pvalloc_run/gridprem_ts.parquet')
    # topo = json.load(open(f'{data_path}/output/pvalloc_run/topo_egid.json', 'r'))
    gridprem_ts = pd.read_parquet(f'{subdir_path_def}/gridprem_ts.parquet')
    topo = json.load(open(f'{subdir_path_def}/topo_egid.json', 'r'))


    # import topo_time_subdfs -----------------------------------------------------
    topo_subdf_paths = glob.glob(f'{subdir_path_def}/topo_subdf_*.parquet') 
    no_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == False]
    agg_npv_df_list = []

    j =0
    i, path = j, topo_subdf_paths[j]
    for i, path in enumerate(topo_subdf_paths):
        if len(topo_subdf_paths) > 5 and i <5 : # i% (len(topo_subdf_paths) //3 ) == 0:
            # print_to_logfile(f'  {2*"-"} update npv (tranche {i}/{len(topo_subdf_paths)}) {6*"-"}', log_file_name)
            checkpoint_to_logfile(f'updated npv (tranche {i+1}/{len(topo_subdf_paths)})', log_file_name, 1, show_debug_prints)
        subdf_t0 = pd.read_parquet(path)

        # drop egids with pv installations
        subdf = copy.deepcopy(subdf_t0[subdf_t0['EGID'].isin(no_pv_egid)])

        if not subdf.empty:

            # merge gridprem_ts
            subdf = subdf.merge(gridprem_ts[['t', 'prem_Rp_kWh', 'grid_node']], how='left', on=['t', 'grid_node']) 

            # compute selfconsumption + netdemand ----------------------------------------------
            subdf_array = subdf[['pvprod_kW', 'demand_kW', 'pv_tarif_Rp_kWh', 'elecpri_Rp_kWh', 'prem_Rp_kWh']].to_numpy()
            pvprod_kW, demand_kW, pv_tarif_Rp_kWh, elecpri_Rp_kWh, prem_Rp_kWh = subdf_array[:,0], subdf_array[:,1], subdf_array[:,2], subdf_array[:,3], subdf_array[:,4]

            demand_kW = demand_kW * pvalloc_settings['algorithm_specs']['tweak_gridnode_df_prod_demand_fact']
            selfconsum_kW = np.minimum(pvprod_kW, demand_kW) * selfconsum_rate
            netdemand_kW = demand_kW - selfconsum_kW
            netfeedin_kW = pvprod_kW - selfconsum_kW

            econ_inc_chf = ((netfeedin_kW * pv_tarif_Rp_kWh) /100) + ((selfconsum_kW * elecpri_Rp_kWh) /100)
            if not tweak_npv_excl_elec_demand:
                econ_spend_chf = ((netfeedin_kW * prem_Rp_kWh) / 100)  + ((netdemand_kW * elecpri_Rp_kWh) /100)
            else:
                econ_spend_chf = ((netfeedin_kW * prem_Rp_kWh) / 100)

            subdf['demand_kW'], subdf['pvprod_kW'], subdf['selfconsum_kW'], subdf['netdemand_kW'], subdf['netfeedin_kW'], subdf['econ_inc_chf'], subdf['econ_spend_chf'] = demand_kW, pvprod_kW, selfconsum_kW, netdemand_kW, netfeedin_kW, econ_inc_chf, econ_spend_chf
            

            if (i <3) and (i_m <3): 
                checkpoint_to_logfile(f'\t end compute econ factors', log_file_name, 1, show_debug_prints) #for subdf EGID {path.split("topo_subdf_")[1].split(".parquet")[0]}', log_file_name, 1, show_debug_prints)

            agg_subdf = subdf.groupby(groupby_cols).agg(agg_cols).reset_index()
            
            if (i <3) and (i_m <3): 
                checkpoint_to_logfile(f'\t groupby subdf to agg_subdf', log_file_name, 1, show_debug_prints)


            # create combinations ----------------------------------------------
            aggsub_npry = np.array(agg_subdf)

            egid_list, combo_df_uid_list, df_uid_list, bfs_list, gklas_list, demandtype_list, grid_node_list = [], [], [], [], [], [], []
            inst_list, info_source_list, pvid_list, pv_tarif_Rp_kWh_list = [], [], [], []
            flaeche_list, stromertrag_list, ausrichtung_list, neigung_list, elecpri_Rp_kWh_list = [], [], [], [], []
        
            flaech_angletilt_list = []
            demand_list, pvprod_list, selfconsum_list, netdemand_list, netfeedin_list = [], [], [], [], []
            econ_inc_chf_list, econ_spend_chf_list = [], []

            egid = agg_subdf['EGID'].unique()[0]
            for i, egid in enumerate(agg_subdf['EGID'].unique()):

                mask_egid_subdf = np.isin(aggsub_npry[:,agg_subdf.columns.get_loc('EGID')], egid)
                df_uids  = list(aggsub_npry[mask_egid_subdf, agg_subdf.columns.get_loc('df_uid')])

                for r in range(1,len(df_uids)+1):
                    for combo in itertools.combinations(df_uids, r):
                        combo_key_str = '_'.join([str(c) for c in combo])
                        mask_dfuid_subdf = np.isin(aggsub_npry[:,agg_subdf.columns.get_loc('df_uid')], list(combo))

                        egid_list.append(egid)
                        combo_df_uid_list.append(combo_key_str)
                        # df_uid_list.append(list(combo))
                        bfs_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('bfs')][0])
                        gklas_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('gklas')][0])
                        demandtype_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('demandtype')][0])
                        grid_node_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('grid_node')][0])

                        inst_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('inst_TF')][0])
                        info_source_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('info_source')][0])
                        pvid_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pvid')][0])
                        pv_tarif_Rp_kWh_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pv_tarif_Rp_kWh')][0]) 
                        elecpri_Rp_kWh_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('elecpri_Rp_kWh')][0])
                        demand_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('demand_kW')][0])

                        ausrichtung_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('AUSRICHTUNG')][0])
                        neigung_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('NEIGUNG')][0])

                        flaeche_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('FLAECHE')].sum())
                        stromertrag_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('STROMERTRAG')].sum())                    
                        flaech_angletilt_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('FLAECH_angletilt')].sum())
                        pvprod_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pvprod_kW')].sum())
                        selfconsum_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('selfconsum_kW')].sum())
                        netdemand_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('netdemand_kW')].sum())
                        netfeedin_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('netfeedin_kW')].sum())
                        econ_inc_chf_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('econ_inc_chf')].sum())
                        econ_spend_chf_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('econ_spend_chf')].sum())



            aggsubdf_combo = pd.DataFrame({'EGID': egid_list, 'df_uid_combo': combo_df_uid_list, 'bfs': bfs_list,
                                        'gklas': gklas_list, 'demandtype': demandtype_list, 'grid_node': grid_node_list,

                                        'inst_TF': inst_list, 'info_source': info_source_list, 'pvid': pvid_list,
                                        'pv_tarif_Rp_kWh': pv_tarif_Rp_kWh_list, 'elecpri_Rp_kWh': elecpri_Rp_kWh_list,
                                        'demand_kW': demand_list,

                                        'AUSRICHTUNG': ausrichtung_list, 'NEIGUNG': neigung_list,
                                        
                                        'FLAECHE': flaeche_list, 'STROMERTRAG': stromertrag_list,
                                        'FLAECH_angletilt': flaech_angletilt_list,
                                        'pvprod_kW': pvprod_list,
                                        'selfconsum_kW': selfconsum_list, 'netdemand_kW': netdemand_list, 'netfeedin_kW': netfeedin_list,
                                        'econ_inc_chf': econ_inc_chf_list, 'econ_spend_chf': econ_spend_chf_list})
                     
        if (i <3) and (i_m <3): 
            checkpoint_to_logfile(f'\t created df_uid combos for {agg_subdf["EGID"].nunique()} EGIDs', log_file_name, 1, show_debug_prints)

        

        # NPV calculation -----------------------------------------------------
        aggsubdf_combo['estim_pvinstcost_chf'] = estim_instcost_chftotal(aggsubdf_combo['FLAECHE'] * kWpeak_per_m2 * share_roof_area_available)

        def compute_npv(row):
            pv_cashflow = (row['econ_inc_chf'] - row['econ_spend_chf']) / (1+interest_rate)**np.arange(1, invst_maturity+1)
            npv = (-row['estim_pvinstcost_chf']) + np.sum(pv_cashflow)
            return npv
        aggsubdf_combo['NPV_uid'] = aggsubdf_combo.apply(compute_npv, axis=1)

        if (i <3) and (i_m <3): 
            checkpoint_to_logfile(f'\t computed NPV for agg_subdf', log_file_name, 1, show_debug_prints)

        agg_npv_df_list.append(aggsubdf_combo)

    agg_npv_df = pd.concat(agg_npv_df_list)
    npv_df = copy.deepcopy(agg_npv_df)


    # export npv_df -----------------------------------------------------
    npv_df.to_parquet(f'{subdir_path_def}/npv_df.parquet')
    if pvalloc_settings['export_csvs']:
        npv_df.to_csv(f'{subdir_path_def}/npv_df.csv', index=False)


    # export by Month -----------------------------------------------------
    if pvalloc_settings['MC_loop_specs']['keep_files_month_iter_TF']:
        if i_m < pvalloc_settings['MC_loop_specs']['keep_files_month_iter_max']:
            pred_npv_inst_by_M_path = f'{subdir_path_def}/pred_npv_inst_by_M'
            if not os.path.exists(pred_npv_inst_by_M_path):
                os.makedirs(pred_npv_inst_by_M_path)
            # elif os.path.exists(pred_npv_inst_by_M_path):
            #     old_files = glob.glob(f'{pred_npv_inst_by_M_path}/*')
            #     for f in old_files:
            #         os.remove(f)

            npv_df.to_parquet(f'{pred_npv_inst_by_M_path}/npv_df_{m}.parquet')
            if pvalloc_settings['export_csvs']:
                npv_df.to_csv(f'{pred_npv_inst_by_M_path}/npv_df_{m}.csv', index=False)
            if i_m < 5:
                npv_df.to_csv(f'{pred_npv_inst_by_M_path}/npv_df_{m}.csv', index=False)


    checkpoint_to_logfile(f'exported npv_df', log_file_name, 1)
        
    return npv_df

