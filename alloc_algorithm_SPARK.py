import sys
import os as os
import numpy as np
import pandas as pd
import json
import itertools
import glob
import copy
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# own functions 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from auxiliary.auxiliary_functions import  checkpoint_to_logfile, print_to_logfile

import pv_allocation.initialization_small_functions as initial_sml


# ------------------------------------------------------------------------------------------------------
# CALCULATE ECONOMIC INDICATORS OF TOPOLOGY
# ------------------------------------------------------------------------------------------------------
def calc_economics_in_topo_df(
        scen, 
        topo, 
        df_list, df_names, 
        ts_list, ts_names,):
    
    # setup -----------------------------------------------------
    print_to_logfile('run function: calc_economics_in_topo_df', scen.log_name)


    # import -----------------------------------------------------
    angle_tilt_df = df_list[df_names.index('angle_tilt_df')]
    solkat_month = df_list[df_names.index('solkat_month')]
    demandtypes_ts = ts_list[ts_names.index('demandtypes_ts')]
    meteo_ts = ts_list[ts_names.index('meteo_ts')]


    # TOPO to DF =============================================
    # solkat_combo_df_exists = os.path.exists(f'{pvalloc_settings["interim_path"]}/solkat_combo_df.parquet')
    # if pvalloc_settings['recalc_economics_topo_df']:
    topo = topo

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
    subdf_path = f'{scen.name_dir_export_path}/topo_time_subdf'

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
    
    topo_subdf_partitioner = scen.ALGOspec_topo_subdf_partitioner
    
    share_roof_area_available = scen.TECspec_share_roof_area_available
    inverter_efficiency       = scen.TECspec_inverter_efficiency
    panel_efficiency          = scen.TECspec_panel_efficiency
    pvprod_calc_method        = scen.TECspec_pvprod_calc_method
    kWpeak_per_m2             = scen.TECspec_kWpeak_per_m2

    flat_direct_rad_factor  = scen.WEAspec_flat_direct_rad_factor
    flat_diffuse_rad_factor = scen.WEAspec_flat_diffuse_rad_factor


    # initiate spark session for computations --------------------
    spark = SparkSession.builder.appName("TopoDataFrame").getOrCreate()
    topo_sdf = spark.createDataFrame(topo_df)
    solkat_month_sdf = spark.createDataFrame(solkat_month)
    meteo_ts_sdf = spark.createDataFrame(meteo_ts)

    # merge production, grid prem + demand to partitions 
    topo_sdf = topo_sdf.withColumn('meteo_loc', F.lit('Basel'))
    meteo_ts_sdf = meteo_ts_sdf.withColumn('meteo_loc', F.lit('Basel'))
    
    topo_sdf = topo_sdf.join(meteo_ts_sdf, on='meteo_loc', how='left')  

    checkpoint_to_logfile(' * * DEBUGGIGN * * *: START SparkSession', scen.log_name, 1)

    # Adding radiation for 'flat' method
    if scen.WEAspec_radiation_to_pvprod_method == 'flat':
        # Apply flat radiation factor calculations
        topo_sdf = topo_sdf.withColumn(
            'radiation', 
            (topo_sdf['rad_direct'] * flat_direct_rad_factor + topo_sdf['rad_diffuse'] * flat_diffuse_rad_factor)
        )
        meteo_ts_sdf = meteo_ts_sdf.withColumn(
            'radiation', 
            (meteo_ts_sdf['rad_direct'] * flat_direct_rad_factor + meteo_ts_sdf['rad_diffuse'] * flat_diffuse_rad_factor)
        )
        
        # Calculate mean of the top 10 largest radiation values
        mean_top_radiation = meteo_ts_sdf.orderBy(F.desc('radiation')).limit(10).agg(F.mean('radiation')).collect()[0][0]
        # Add relative location max radiation column to topo_sdf
        topo_sdf = topo_sdf.withColumn('radiation_rel_locmax', topo_sdf['radiation'] / mean_top_radiation)

    # Adding radiation for 'dfuid_ind' method
    elif scen.WEAspec_radiation_to_pvprod_method == 'dfuid_ind':
        # Prepare solkat_month for joining by renaming columns
        solkat_month_sdf = solkat_month_sdf.withColumnRenamed('DF_UID', 'df_uid').withColumnRenamed('MONAT', 'month')
        solkat_month_sdf = solkat_month_sdf.withColumn('month', solkat_month_sdf['month'].cast('int'))
        # Add 'month' column to topo_sdf
        topo_sdf = topo_sdf.withColumn('month', topo_sdf['timestamp'].month)
        # You can then proceed to perform any additional joins or operations needed between 'topo_sdf' and 'solkat_month_sdf'
        topo_sdf = topo_sdf.join(solkat_month_sdf, on='month', how='left')



        # Radiation relative location max by "df_uid_specific" vs "all_HOY" ----------
        if scen.WEAspec_rad_rel_loc_max_by == 'dfuid_specific':
            # Group by 'df_uid' and calculate the mean of the top 10 largest radiation values for each df_uid
            subdf_dfuid_topradation = topo_sdf.groupBy('df_uid') \
                .agg(F.mean(F.when(F.row_number().over(Window.partitionBy('df_uid').orderBy(F.desc('radiation'))) <= 10, 'radiation'))
                    .otherwise(None)).alias('mean_top_radiation')
            
            # Merge the result with the original DataFrame (topo_sdf) to add the 'mean_top_radiation' column
            topo_sdf = topo_sdf.join(subdf_dfuid_topradation, on='df_uid', how='left')
            # Add the 'radiation_rel_locmax' column by dividing 'radiation' by 'mean_top_radiation'
            topo_sdf = topo_sdf.withColumn('radiation_rel_locmax', topo_sdf['radiation'] / topo_sdf['mean_top_radiation'])

        elif scen.WEAspec_rad_rel_loc_max_by == 'all_HOY':
            # Calculate the mean of the top 10 largest radiation values from meteo_ts_sdf
            mean_nlargest_rad_all_HOY = meteo_ts_sdf.orderBy(F.desc('radiation')).limit(10).agg(F.mean('radiation')).collect()[0][0]
            # Add the 'radiation_rel_locmax' column to topo_sdf
            topo_sdf = topo_sdf.withColumn('radiation_rel_locmax', topo_sdf['radiation'] / mean_nlargest_rad_all_HOY)

    checkpoint_to_logfile('end radiation calculations', scen.log_name, 1)
    # Add panel_efficiency by time ----------
    if scen.PEFspec_variable_panel_efficiency_TF:
        summer_months = scen.PEFspec_summer_months
        hotsummer_hours = scen.PEFspec_hotsummer_hours
        hot_hours_discount = scen.PEFspec_hot_hours_discount
        # Read the parquet file into a PySpark DataFrame
        HOY_weatheryear_df = spark.read.parquet(f'{scen.name_dir_export_path}/HOY_weatheryear_df.parquet')
                    # Filter for hot hours in summer months
        hot_hours_in_year_sdf = HOY_weatheryear_df.filter(
            (HOY_weatheryear_df['month'].isin(summer_months)) &
            (HOY_weatheryear_df['hour'].isin(hotsummer_hours))
        )
        
        # Join with the original topo_sdf on the 't' column to get the matching hot hours
        topo_sdf = topo_sdf.join(hot_hours_in_year_sdf.select('t'), on='t', how='left')

        # Apply the hot hours discount for panel efficiency
        topo_sdf = topo_sdf.withColumn(
            'panel_efficiency',
            F.when(topo_sdf['t'].isNotNull(), panel_efficiency * (1 - hot_hours_discount))
            .otherwise(panel_efficiency)
        )

    elif not scen.PEFspec_variable_panel_efficiency_TF:
        # If not variable efficiency, just assign the base panel efficiency
        topo_sdf = topo_sdf.withColumn('panel_efficiency', F.lit(panel_efficiency))


    # Attach demand profiles ----------
    demandtypes_names = [c for c in demandtypes_ts.columns if 'DEMANDprox' in c]
    # Melt the demandtypes DataFrame
    demandtypes_melt_sdf = spark.createDataFrame(demandtypes_ts.melt(id_vars='t', value_vars=demandtypes_names, var_name='demandtype', value_name='demand'))
    # Join demandtypes_melt_sdf with topo_sdf based on 't' and 'demandtype'
    topo_sdf = topo_sdf.join(demandtypes_melt_sdf, on=['t', 'demandtype'], how='left')
    # Rename 'demand' to 'demand_kW'
    topo_sdf = topo_sdf.withColumnRenamed('demand', 'demand_kW')

    # Attach FLAECH_angletilt, might be useful for later calculations
    topo_sdf = topo_sdf.withColumn('FLAECH_angletilt', topo_sdf['FLAECHE'] * topo_sdf['angletilt_factor'])

    checkpoint_to_logfile('end attach demand profiles', scen.log_name, 1)

    # Compute production ----------
    if pvprod_calc_method == 'method1':
        # Method 1: Calculate pvprod_kW using the formula and drop unnecessary columns
        topo_sdf = topo_sdf.withColumn(
            'pvprod_kW', 
            (topo_sdf['radiation'] * topo_sdf['FLAECHE'] * topo_sdf['angletilt_factor']) / 1000
        ).drop('meteo_loc', 'radiation')

    elif pvprod_calc_method == 'method2.1':
        # Method 2.1: Calculate pvprod_kW with inverter efficiency, roof area share, and panel efficiency
        topo_sdf = topo_sdf.withColumn(
            'pvprod_kW',
            (topo_sdf['radiation'] / 1000) * inverter_efficiency * share_roof_area_available * topo_sdf['panel_efficiency'] * topo_sdf['FLAECHE'] * topo_sdf['angletilt_factor']
        )
        formla_for_log_print = "topo_sdf['pvprod_kW'] = topo_sdf['radiation'] / 1000 * inverter_efficiency * share_roof_area_available * topo_sdf['panel_efficiency'] * topo_sdf['FLAECHE'] * topo_sdf['angletilt_factor']"

    elif pvprod_calc_method == 'method2.2':
        # Method 2.2: Similar to 2.1, but without the angletilt_factor
        topo_sdf = topo_sdf.withColumn(
            'pvprod_kW',
            (topo_sdf['radiation'] / 1000) * inverter_efficiency * share_roof_area_available * topo_sdf['panel_efficiency'] * topo_sdf['FLAECHE']
        )
        formla_for_log_print = "topo_sdf['pvprod_kW'] = topo_sdf['radiation'] / 1000 * inverter_efficiency * share_roof_area_available * topo_sdf['panel_efficiency'] * topo_sdf['FLAECHE']"

    elif pvprod_calc_method == 'method3.1':
        # Method 3.1: Use radiation relative location max and kWpeak_per_m2 to calculate pvprod_kW
        topo_sdf = topo_sdf.withColumn(
            'pvprod_kW',
            topo_sdf['radiation_rel_locmax'] * kWpeak_per_m2 * inverter_efficiency * share_roof_area_available * topo_sdf['panel_efficiency'] * topo_sdf['FLAECHE'] * topo_sdf['angletilt_factor']
        )
        formla_for_log_print = "topo_sdf['pvprod_kW'] = topo_sdf['radiation_rel_locmax'] * kWpeak_per_m2 * inverter_efficiency * share_roof_area_available * topo_sdf['panel_efficiency'] * topo_sdf['FLAECHE'] * topo_sdf['angletilt_factor']"

    elif pvprod_calc_method == 'method3.2':
        # Method 3.2: Similar to 3.1, but without the angletilt_factor
        topo_sdf = topo_sdf.withColumn(
            'pvprod_kW',
            topo_sdf['radiation_rel_locmax'] * kWpeak_per_m2 * inverter_efficiency * share_roof_area_available * topo_sdf['panel_efficiency'] * topo_sdf['FLAECHE']
        )
        formla_for_log_print = "topo_sdf['pvprod_kW'] = topo_sdf['radiation_rel_locmax'] * kWpeak_per_m2 * inverter_efficiency * share_roof_area_available * topo_sdf['panel_efficiency'] * topo_sdf['FLAECHE']"

            
    # export topo_sdf ----------------------------------------------
    topo_sdf.write.parquet(f'{subdf_path}/topo_subdf.parquet')
    checkpoint_to_logfile('end merge to topo_time_subdf', scen.log_name, 1)
    checkpoint_to_logfile(' * * DEBUGGIGN * * *: END SparkSession', scen.log_name, 1)




# ------------------------------------------------------------------------------------------------------
# INITIATE GRID PREMIUM TS
# ------------------------------------------------------------------------------------------------------
def initiate_gridprem(
        scen,):

    # setup -----------------------------------------------------
    print_to_logfile('run function: initiate_gridprem', scen.log_name)

    if os.path.exists(f'{scen.name_dir_export_path}/gridprem_ts.parquet'):
        os.remove(f'{scen.name_dir_export_path}/gridprem_ts.parquet')    

    # import -----------------------------------------------------
    dsonodes_df = pd.read_parquet(f'{scen.name_dir_export_path}/dsonodes_df.parquet')
    t_range = [f't_{t}' for t in range(1,8760 + 1)]

    dsonodes_df.drop(columns=['EGID'], inplace=True)
    gridprem_ts = pd.DataFrame(np.repeat(dsonodes_df.values, len(t_range), axis=0), columns=dsonodes_df.columns)  
    gridprem_ts['t'] = np.tile(t_range, len(dsonodes_df))
    gridprem_ts['prem_Rp_kWh'] = 0

    gridprem_ts = gridprem_ts[['t', 'grid_node', 'kVA_threshold', 'prem_Rp_kWh']]
    gridprem_ts.drop(columns='kVA_threshold', inplace=True)

    # export -----------------------------------------------------
    gridprem_ts.to_parquet(f'{scen.name_dir_export_path}/gridprem_ts.parquet')




# ------------------------------------------------------------------------------------------------------
# UPDATE GRID PREMIUM TS
# ------------------------------------------------------------------------------------------------------
def update_gridprem(
        scen,
        subdir_path,
        month_func, i_month_func):
    
    # setup -----------------------------------------------------
    print_to_logfile('run function: update_gridprem', scen.log_name)
    gridtiers_power_factor  = scen.GRIDspec_power_factor
    m = month_func
    i_m = i_month_func

    # import  -----------------------------------------------------
    topo = json.load(open(f'{subdir_path}/topo_egid.json', 'r'))
    dsonodes_df = pd.read_parquet(f'{subdir_path}/dsonodes_df.parquet')
    gridprem_ts = pd.read_parquet(f'{subdir_path}/gridprem_ts.parquet')

    data = [(k, v[0], v[1]) for k, v in scen.GRIDspec_tiers.items()]
    gridtiers_df = pd.DataFrame(data, columns=scen.GRIDspec_colnames)

    checkpoint_to_logfile('**DEBUGGIG** > START LOOP through topo_egid', scen.log_name, 1, scen.show_debug_prints)
    egid_list, info_source_list, inst_TF_list = [], [], []
    for k,v in topo.items():
        egid_list.append(k)
        if v.get('pv_inst', {}).get('inst_TF'):
            info_source_list.append(v.get('pv_inst').get('info_source'))
            inst_TF_list.append(v.get('pv_inst').get('inst_TF'))
        else: 
            info_source_list.append('')
            inst_TF_list.append(False)
    Map_infosource_egid = pd.DataFrame({'EGID': egid_list, 'info_source': info_source_list, 'inst_TF': inst_TF_list}, index=egid_list)

    checkpoint_to_logfile('**DEBUGGIG** > end loop through topo_egid', scen.log_name, 1, scen.show_debug_prints) if i_m < 3 else None


    # import topo_time_subdfs -----------------------------------------------------
    # topo_subdf_paths = glob.glob(f'{scen.pvalloc_path}/topo_time_subdf/*.parquet')
    checkpoint_to_logfile('**DEBUGGIG** > start loop through subdfs', scen.log_name, 1, scen.show_debug_prints) if i_m < 3 else None

    topo_subdf_paths = glob.glob(f'{subdir_path}/topo_subdf_*.parquet')
    agg_subinst_df_list = []
    # no_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == False]
    # wi_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == True]

    i, path = 0, topo_subdf_paths[0]
    for i, path in enumerate(topo_subdf_paths):
        checkpoint_to_logfile('**DEBUGGIG** \t> start read subdfs', scen.log_name, 2) if i < 2 else None
        subdf = pd.read_parquet(path)
        checkpoint_to_logfile('**DEBUGGIG** \t> end read subdfs', scen.log_name, 2) if i < 2 else None

        subdf_updated = copy.deepcopy(subdf)
        subdf_updated.drop(columns=['info_source', 'inst_TF'], inplace=True)

        checkpoint_to_logfile('**DEBUGGIG** \t> start Map_infosource_egid', scen.log_name, 1, scen.show_debug_prints) if i < 2 else None
        subdf_updated = subdf_updated.merge(Map_infosource_egid[['EGID', 'info_source', 'inst_TF']], how='left', on='EGID')
        checkpoint_to_logfile('**DEBUGGIG** \t> end Map_infosource_egid', scen.log_name, 1, scen.show_debug_prints) if i < 2 else None
        # updated_instTF_srs, update_infosource_srs = subdf_updated['inst_TF'].fillna(subdf['inst_TF']), subdf_updated['info_source'].fillna(subdf['info_source'])
        # subdf['inst_TF'], subdf['info_source'] = updated_instTF_srs.infer_objects(copy=False), update_infosource_srs.infer_objects(copy=False)

        # Only consider production for houses that have built a pv installation and substract selfconsumption from the production
        subinst = copy.deepcopy(subdf_updated.loc[subdf_updated['inst_TF']==True])
        checkpoint_to_logfile('**DEBUGGIG** \t> pvprod_kw_to_numpy', scen.log_name, 2) if i < 2 else None
        pvprod_kW, demand_kW = subinst['pvprod_kW'].to_numpy(), subinst['demand_kW'].to_numpy()
        selfconsum_kW = np.minimum(pvprod_kW, demand_kW) * scen.TECspec_self_consumption_ifapplicable
        netdemand_kW = demand_kW - selfconsum_kW
        netfeedin_kW = pvprod_kW - selfconsum_kW

        subinst['feedin_kW'] = netfeedin_kW
        
        checkpoint_to_logfile('**DEBUGGIG** > end pvprod_kw_to_numpy', scen.log_name, 2, scen.show_debug_prints) if i < 2 else None
        # NOTE: attempt for a more elaborate way to handle already installed installations
        if False:
            pv = pd.read_parquet(f'{subdir_path}/pv.parquet')
            pv['pvsource'] = 'pv_df'
            pv['pvid'] = pv['xtf_id']

            # if 'pv_df' in subinst['pvsource'].unique():
            # TotalPower = pv.loc[pv['xtf_id'].isin(subinst.loc[subinst['EGID'] == egid, 'pvid']), 'TotalPower'].sum()

            subinst = subinst.sort_values(by = 'STROMERTRAG', ascending=False)
            subinst['pvprod_kW'] = 0
            
            # t_steps = subinst['t'].unique()
            for t in subinst['t'].unique():
                timestep_df = subinst.loc[subinst['t'] == t]
                total_stromertrag = timestep_df['STROMERTRAG'].sum()

                for idx, row in timestep_df.iterrows():
                    share = row['STROMERTRAG'] / total_stromertrag
                    # subinst.loc[idx, 'pvprod_kW'] = share * TotalPower
                    print(share)

        agg_subinst = subinst.groupby(['grid_node', 't']).agg({'feedin_kW': 'sum', 'pvprod_kW':'sum'}).reset_index()
        del subinst
        agg_subinst_df_list.append(agg_subinst)
    
    checkpoint_to_logfile('**DEBUGGIG** > end loop through subdfs', scen.log_name, 1, scen.show_debug_prints) if i_m < 3 else None


    # build gridnode_df -----------------------------------------------------
    gridnode_df = pd.concat(agg_subinst_df_list)
    # groupby df again because grid nodes will be spreach accross multiple tranches
    gridnode_df = gridnode_df.groupby(['grid_node', 't']).agg({'feedin_kW': 'sum', 'pvprod_kW':'sum'}).reset_index() 

    # attach node thresholds 
    gridnode_df = gridnode_df.merge(dsonodes_df[['grid_node', 'kVA_threshold']], how='left', on='grid_node')
    gridnode_df['kW_threshold'] = gridnode_df['kVA_threshold'] * scen.GRIDspec_perf_factor_1kVA_to_XkW

    gridnode_df['feedin_kW_taken'] = np.where(gridnode_df['feedin_kW'] > gridnode_df['kW_threshold'], gridnode_df['kW_threshold'], gridnode_df['feedin_kW'])
    gridnode_df['feedin_kW_loss'] =  np.where(gridnode_df['feedin_kW'] > gridnode_df['kW_threshold'], gridnode_df['feedin_kW'] - gridnode_df['kW_threshold'], 0)

    checkpoint_to_logfile('**DEBUGGIG** > end merge + npwhere subdfs', scen.log_name, 1, scen.show_debug_prints) if i_m < 3 else None


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

    checkpoint_to_logfile('**DEBUGGIG** > end update gridprem_ts', scen.log_name, 1, scen.show_debug_prints) if i_m < 3 else None


    # EXPORT -----------------------------------------------------
    gridnode_df.to_parquet(f'{subdir_path}/gridnode_df.parquet')
    gridprem_ts.to_parquet(f'{subdir_path}/gridprem_ts.parquet')
    if scen.export_csvs:
        gridnode_df.to_csv(f'{subdir_path}/gridnode_df.csv', index=False)
        gridprem_ts.to_csv(f'{subdir_path}/gridprem_ts.csv', index=False)


    # export by Month -----------------------------------------------------
    if scen.MCspec_keep_files_month_iter_TF:
        if i_m < scen.MCspec_keep_files_month_iter_max:
            # gridprem_node_by_M_path = f'{scen.pvalloc_path}/pred_gridprem_node_by_M'
            gridprem_node_by_M_path = f'{subdir_path}/pred_gridprem_node_by_M'
            if not os.path.exists(gridprem_node_by_M_path):
                os.makedirs(gridprem_node_by_M_path)

            gridnode_df.to_parquet(f'{gridprem_node_by_M_path}/gridnode_df_{m}.parquet')
            gridprem_ts.to_parquet(f'{gridprem_node_by_M_path}/gridprem_ts_{m}.parquet')

            if scen.export_csvs:
                gridnode_df.to_csv(f'{gridprem_node_by_M_path}/gridnode_df_{m}.csv', index=False)
                gridprem_ts.to_csv(f'{gridprem_node_by_M_path}/gridprem_ts_{m}.csv', index=False)
            if i_m < 5:
                gridnode_df.to_csv(f'{gridprem_node_by_M_path}/gridnode_df_{m}.csv', index=False)
                gridprem_ts.to_csv(f'{gridprem_node_by_M_path}/gridprem_ts_{m}.csv', index=False)

    checkpoint_to_logfile('exported gridprem_ts and gridnode_df', scen.log_name, 1) if i_m < 3 else None



# ------------------------------------------------------------------------------------------------------
# UPDATE NPV_DF with NEW GRID PREMIUM TS
# ------------------------------------------------------------------------------------------------------
def update_npv_df(scen,
                  subdir_path,
                  month_func, i_month_func
                ):
    
    # setup -----------------------------------------------------
    print_to_logfile('run function: update_npv_df', scen.log_name)
    m = month_func
    i_m = i_month_func
    


    # import -----------------------------------------------------
    gridprem_ts = pd.read_parquet(f'{subdir_path}/gridprem_ts.parquet')
    topo = json.load(open(f'{subdir_path}/topo_egid.json', 'r'))


    # import topo_time_subdfs -----------------------------------------------------
    topo_subdf_paths = glob.glob(f'{subdir_path}/topo_subdf_*.parquet') 
    no_pv_egid = [k for k, v in topo.items() if not v.get('pv_inst', {}).get('inst_TF') ]
    agg_npv_df_list = []

    j = 0
    i, path = j, topo_subdf_paths[j]
    for i, path in enumerate(topo_subdf_paths):
        print_topo_subdf_TF = len(topo_subdf_paths) > 5 and i <5  # i% (len(topo_subdf_paths) //3 ) == 0:
        if print_topo_subdf_TF:
            print_to_logfile(f'updated npv (tranche {i+1}/{len(topo_subdf_paths)})', scen.log_name)
        subdf_t0 = pd.read_parquet(path)

        # drop egids with pv installations
        subdf = copy.deepcopy(subdf_t0[subdf_t0['EGID'].isin(no_pv_egid)])

        if not subdf.empty:

            # merge gridprem_ts
            subdf = subdf.merge(gridprem_ts[['t', 'prem_Rp_kWh', 'grid_node']], how='left', on=['t', 'grid_node']) 

            # compute selfconsumption + netdemand ----------------------------------------------
            subdf_array = subdf[['pvprod_kW', 'demand_kW', 'pv_tarif_Rp_kWh', 'elecpri_Rp_kWh', 'prem_Rp_kWh']].to_numpy()
            pvprod_kW, demand_kW, pv_tarif_Rp_kWh, elecpri_Rp_kWh, prem_Rp_kWh = subdf_array[:,0], subdf_array[:,1], subdf_array[:,2], subdf_array[:,3], subdf_array[:,4]

            demand_kW = demand_kW * scen.ALGOspec_tweak_gridnode_df_prod_demand_fact
            selfconsum_kW = np.minimum(pvprod_kW, demand_kW) * scen.TECspec_self_consumption_ifapplicable
            netdemand_kW = demand_kW - selfconsum_kW
            netfeedin_kW = pvprod_kW - selfconsum_kW

            econ_inc_chf = ((netfeedin_kW * pv_tarif_Rp_kWh) /100) + ((selfconsum_kW * elecpri_Rp_kWh) /100)
            if not scen.ALGOspec_tweak_npv_excl_elec_demand:
                econ_spend_chf = ((netfeedin_kW * prem_Rp_kWh) / 100)  + ((netdemand_kW * elecpri_Rp_kWh) /100)
            else:
                econ_spend_chf = ((netfeedin_kW * prem_Rp_kWh) / 100)

            subdf['demand_kW'], subdf['pvprod_kW'], subdf['selfconsum_kW'], subdf['netdemand_kW'], subdf['netfeedin_kW'], subdf['econ_inc_chf'], subdf['econ_spend_chf'] = demand_kW, pvprod_kW, selfconsum_kW, netdemand_kW, netfeedin_kW, econ_inc_chf, econ_spend_chf
            

            if (i <3) and (i_m <3): 
                checkpoint_to_logfile('\t end compute econ factors', scen.log_name, 1, scen.show_debug_prints) #for subdf EGID {path.split("topo_subdf_")[1].split(".parquet")[0]}', scen.log_name, 1, scen.show_debug_prints)

            agg_subdf = subdf.groupby(
                                scen.ALGOspec_npv_update_groupby_cols_topo_aggdf).agg(
                                scen.ALGOspec_npv_update_agg_cols_topo_aggdf).reset_index()
                
            
            if (i <3) and (i_m <3): 
                checkpoint_to_logfile('\t groupby subdf to agg_subdf', scen.log_name, 1, scen.show_debug_prints)


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
                        mask_dfuid_only = np.isin(aggsub_npry[:,agg_subdf.columns.get_loc('df_uid')], list(combo))
                        mask_dfuid_subdf = mask_egid_subdf & mask_dfuid_only

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
            checkpoint_to_logfile(f'\t created df_uid combos for {agg_subdf["EGID"].nunique()} EGIDs', scen.log_name, 1, scen.show_debug_prints)

        

        # NPV calculation -----------------------------------------------------
        if print_topo_subdf_TF:
            estim_instcost_chfpkW, estim_instcost_chftotal = initial_sml.get_estim_instcost_function(scen)
            estim_instcost_chftotal(pd.Series([10, 20, 30, 40, 50, 60, 70]))

        # # estim_instcost_chfpkW, estim_instcost_chftotal = initial.estimate_iterpolate_instcost_function(pvalloc_settings)

        # if not os.path.exists(f'{preprep_name_dir_path }/pvinstcost_coefficients.json') == True:
        #     estim_instcost_chfpkW, estim_instcost_chftotal = initial.estimate_iterpolate_instcost_function(pvalloc_settings)
        #     estim_instcost_chftotal(pd.Series([10, 20, 30, 40, 50, 60, 70]))

        # elif os.path.exists(f'{preprep_name_dir_path }/pvinstcost_coefficients.json') == True:    
        #     estim_instcost_chfpkW, estim_instcost_chftotal = initial.get_estim_instcost_function(pvalloc_settings)
        #     estim_instcost_chftotal(pd.Series([10, 20, 30, 40, 50, 60, 70]))

        # correct cost estimation by a factor based on insights from pvprod_correction.py
        aggsubdf_combo['estim_pvinstcost_chf'] = estim_instcost_chftotal(aggsubdf_combo['FLAECHE'] * 
                                                                         scen.TECspec_kWpeak_per_m2 * 
                                                                         scen.TECspec_share_roof_area_available) / scen.TECspec_estim_pvinst_cost_correctionfactor


        def compute_npv(row):
            pv_cashflow = (row['econ_inc_chf'] - row['econ_spend_chf']) / (1+scen.TECspec_interest_rate)**np.arange(1, scen.TECspec_invst_maturity+1)
            npv = (-row['estim_pvinstcost_chf']) + np.sum(pv_cashflow)
            return npv
        aggsubdf_combo['NPV_uid'] = aggsubdf_combo.apply(compute_npv, axis=1)

        if (i <3) and (i_m <3): 
            checkpoint_to_logfile('\t computed NPV for agg_subdf', scen.log_name, 1, scen.show_debug_prints)

        agg_npv_df_list.append(aggsubdf_combo)

    agg_npv_df = pd.concat(agg_npv_df_list)
    npv_df = copy.deepcopy(agg_npv_df)


    # export npv_df -----------------------------------------------------
    npv_df.to_parquet(f'{subdir_path}/npv_df.parquet')
    if scen.export_csvs:
        npv_df.to_csv(f'{subdir_path}/npv_df.csv', index=False)


    # export by Month -----------------------------------------------------
    if scen.MCspec_keep_files_month_iter_TF:
        if i_m < scen.MCspec_keep_files_month_iter_max:
            pred_npv_inst_by_M_path = f'{subdir_path}/pred_npv_inst_by_M'
            if not os.path.exists(pred_npv_inst_by_M_path):
                os.makedirs(pred_npv_inst_by_M_path)

            npv_df.to_parquet(f'{pred_npv_inst_by_M_path}/npv_df_{m}.parquet')
            if scen.export_csvs:
                npv_df.to_csv(f'{pred_npv_inst_by_M_path}/npv_df_{m}.csv', index=False)
            if i_m < 5:
                npv_df.to_csv(f'{pred_npv_inst_by_M_path}/npv_df_{m}.csv', index=False)


    checkpoint_to_logfile('exported npv_df', scen.log_name, 1)
        
    return npv_df

