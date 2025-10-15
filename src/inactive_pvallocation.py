############################################################
#           PARKPLATZ - INACTIVE CODE CHUNKs 
############################################################
# this file contains code chunks that are currently inactive, but should not be deleted yet. To not overburden the IDE in the MAIN files, these chunks are moved to this file.


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# algo_calc_economics_in_topo_df()
def algo_calc_economics_in_topo_df(self, 
                                    topo, 
                                    df_list, df_names, 
                                    ts_list, ts_names,
): 
            
    # setup -----------------------------------------------------
    print_to_logfile('run function: calc_economics_in_topo_df', self.sett.log_name)


    # import -----------------------------------------------------
    angle_tilt_df = df_list[df_names.index('angle_tilt_df')]
    solkat_month = df_list[df_names.index('solkat_month')]
    demandtypes_ts = ts_list[ts_names.index('demandtypes_ts')]
    meteo_ts = ts_list[ts_names.index('meteo_ts')]


    # TOPO to DF =============================================
    # solkat_combo_df_exists = os.path.exists(f'{pvalloc_settings["interim_path"]}/solkat_combo_df.parquet')
    # if pvalloc_settings['recalc_economics_topo_df']:
    topo = topo

    egid_list, df_uid_list, bfs_list, gklas_list, demand_arch_typ_list, grid_node_list  = [], [], [], [], [], []
    inst_list, info_source_list, pvdf_totalpower_list, pvid_list, pvtarif_Rp_kWh_list = [], [], [], [], []
    flaeche_list, stromertrag_list, ausrichtung_list, neigung_list, elecpri_list = [], [], [], [], []

    k,v = list(topo.items())[0]

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
            demand_arch_typ_list.append(v.get('demand_arch_typ'))
            grid_node_list.append(v.get('grid_node'))

            inst_list.append(v.get('pv_inst').get('inst_TF'))
            info_source_list.append(v.get('pv_inst').get('info_source'))
            pvid_list.append(v.get('pv_inst').get('xtf_id'))
            pvtarif_Rp_kWh_list.append(v.get('pvtarif_Rp_kWh'))
            pvdf_totalpower_list.append(v.get('pv_inst').get('TotalPower'))

            flaeche_list.append(v_p.get('FLAECHE'))
            ausrichtung_list.append(v_p.get('AUSRICHTUNG'))
            stromertrag_list.append(v_p.get('STROMERTRAG'))
            neigung_list.append(v_p.get('NEIGUNG'))
            elecpri_list.append(v.get('elecpri_Rp_kWh'))
                
        
    topo_df = pd.DataFrame({'EGID': egid_list, 'df_uid': df_uid_list, 'bfs': bfs_list,
                            'gklas': gklas_list, 'demand_arch_typ': demand_arch_typ_list, 'grid_node': grid_node_list,

                            'inst_TF': inst_list, 'info_source': info_source_list, 'pvid': pvid_list,
                            'pvtarif_Rp_kWh': pvtarif_Rp_kWh_list, 'TotalPower': pvdf_totalpower_list,

                            'FLAECHE': flaeche_list, 'AUSRICHTUNG': ausrichtung_list, 
                            'STROMERTRAG': stromertrag_list, 'NEIGUNG': neigung_list, 
                            'elecpri_Rp_kWh': elecpri_list})
    

    # make or clear dir for subdfs ----------------------------------------------
    subdf_path = f'{self.sett.name_dir_export_path}/topo_time_subdf'

    if not os.path.exists(subdf_path):
        os.makedirs(subdf_path)
    else:
        old_files = glob.glob(f'{subdf_path}/*')
        for f in old_files:
            os.remove(f)
    

    # angle/tilt transformation ----------------------------------------------
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

    # MERGE WEATHER DATA - CALC PRODUCTION PER PARTITION  +  ADJUST INST SETTINGS FOR EXISITN PV (INST_TF / SOURCE)   =========================================================== 
    topo_subdf_partitioner = self.sett.ALGOspec_topo_subdf_partitioner
    
    share_roof_area_available = self.sett.TECspec_share_roof_area_available
    inverter_efficiency       = self.sett.TECspec_inverter_efficiency
    panel_efficiency          = self.sett.TECspec_panel_efficiency
    pvprod_calc_method        = self.sett.TECspec_pvprod_calc_method
    kWpeak_per_m2             = self.sett.TECspec_kWpeak_per_m2

    flat_direct_rad_factor  = self.sett.WEAspec_flat_direct_rad_factor
    flat_diffuse_rad_factor = self.sett.WEAspec_flat_diffuse_rad_factor


    egids = topo_df['EGID'].unique()

    stepsize = topo_subdf_partitioner if len(egids) > topo_subdf_partitioner else len(egids)
    tranche_counter = 0
    checkpoint_to_logfile(' * * DEBUGGIGN * * *: START loop subdfs', self.sett.log_name, 0)
    for i in range(0, len(egids), stepsize):

        tranche_counter += 1
        # print_to_logfile(f'-- merges to topo_time_subdf {tranche_counter}/{len(range(0, len(egids), stepsize))} tranches ({i} to {i+stepsize-1} egids.iloc) ,  {7*"-"}  (stamp: {datetime.now()})', self.sett.log_name)
        subdf = copy.deepcopy(topo_df[topo_df['EGID'].isin(egids[i:i+stepsize])])


        # I  MERGE WEATHER DATA - CALC PRODUCTION PER PARTITION 

        # merge production, grid prem + demand to partitions ----------
        subdf['meteo_loc'] = 'Basel'
        meteo_ts['meteo_loc'] ='Basel' 
        
        # subdf = subdf.merge(meteo_ts[['rad_direct', 'rad_diffuse', 'temperature', 't', 'meteo_loc']], how='left', on='meteo_loc')
        subdf = subdf.merge(meteo_ts, how='left', on='meteo_loc')
        

        # add radiation per h to subdf, "flat" OR "dfuid_ind" ----------
        if self.sett.WEAspec_radiation_to_pvprod_method == 'flat':
            subdf['radiation'] = subdf['rad_direct'] * flat_direct_rad_factor + subdf['rad_diffuse'] * flat_diffuse_rad_factor
            meteo_ts['radiation'] = meteo_ts['rad_direct'] * flat_direct_rad_factor + meteo_ts['rad_diffuse'] * flat_diffuse_rad_factor
            mean_top_radiation = meteo_ts['radiation'].nlargest(10).mean()

            subdf['radiation_rel_locmax'] = subdf['radiation'] / mean_top_radiation


        elif self.sett.WEAspec_radiation_to_pvprod_method == 'dfuid_ind':
            solkat_month.rename(columns={'DF_UID': 'df_uid', 'MONAT': 'month'}, inplace=True)
            solkat_month['month'] = solkat_month['month'].astype(int)
            subdf['month'] = subdf['timestamp'].dt.month.astype(int)
            
        
            checkpoint_to_logfile(f'  start merge solkat_month to subdf {i} to {i+stepsize-1}', self.sett.log_name, 0) if i < 2 else None
            subdf = subdf.merge(solkat_month[['df_uid', 'month', 'A_PARAM', 'B_PARAM', 'C_PARAM']], how='left', on=['df_uid', 'month'])
            checkpoint_to_logfile(f'  end merge solkat_month to subdf {i} to {i+stepsize-1}', self.sett.log_name, 0) if i < 2 else None
            subdf['radiation'] = subdf['A_PARAM'] * subdf['rad_direct'] + subdf['B_PARAM'] * subdf['rad_diffuse'] + subdf['C_PARAM']
            # some radiation values are negative, because of the linear transformation with abc parameters. 
            # force all negative values to 0
            subdf.loc[subdf['radiation'] < 0, 'radiation'] = 0
            subdf.loc[(subdf['rad_direct'] == 0) & (subdf['rad_diffuse'] == 0), 'radiation'] = 0
            # subdf['radiation'] = np.where(
            #                         (subdf['rad_direct'] != 0) | (subdf['rad_diffuse'] != 0),
            #                         subdf['A_PARAM'] * subdf['rad_direct'] + subdf['B_PARAM'] * subdf['rad_diffuse'] + subdf['C_PARAM'],
            #                         0)

            meteo_ts['radiation'] = meteo_ts['rad_direct'] * flat_direct_rad_factor + meteo_ts['rad_diffuse'] * flat_diffuse_rad_factor
            # meteo_ts['radiation_abc_param_1dfuid'] = meteo_ts['rad_direct'] * subdf['A_PARAM'].mean() + meteo_ts['rad_diffuse'] * subdf['B_PARAM'].mean() + subdf['C_PARAM'].mean()


            # radiation_rel_locmax by "df_uid_specific" vs "all_HOY" ---------- 
            if self.sett.WEAspec_rad_rel_loc_max_by == 'dfuid_specific':
                subdf_dfuid_topradation = subdf.groupby('df_uid')['radiation'].apply(lambda x: x.nlargest(10).mean()).reset_index()
                subdf_dfuid_topradation.rename(columns={'radiation': 'mean_top_radiation'}, inplace=True)
                subdf = subdf.merge(subdf_dfuid_topradation, how='left', on='df_uid')

                subdf['radiation_rel_locmax'] = subdf['radiation'] / subdf['mean_top_radiation']

            elif self.sett.WEAspec_rad_rel_loc_max_by == 'all_HOY':
                mean_nlargest_rad_all_HOY = meteo_ts['radiation'].nlargest(10).mean()
                subdf['radiation_rel_locmax'] = subdf['radiation'] / mean_nlargest_rad_all_HOY


        # add panel_efficiency by time ----------
        if self.sett.PEFspec_variable_panel_efficiency_TF:
            summer_months      = self.sett.PEFspec_summer_months
            hotsummer_hours    = self.sett.PEFspec_hotsummer_hours
            hot_hours_discount = self.sett.PEFspec_hot_hours_discount

            HOY_weatheryear_df = pd.read_parquet(f'{self.sett.name_dir_export_path}/HOY_weatheryear_df.parquet')
            hot_hours_in_year = HOY_weatheryear_df.loc[(HOY_weatheryear_df['month'].isin(summer_months)) & (HOY_weatheryear_df['hour'].isin(hotsummer_hours))]
            subdf['panel_efficiency'] = np.where(
                subdf['t'].isin(hot_hours_in_year['t']),
                panel_efficiency * (1-hot_hours_discount),
                panel_efficiency)
            
        elif not self.sett.PEFspec_variable_panel_efficiency_TF:
            subdf['panel_efficiency'] = panel_efficiency
            

        # attach demand profiles ----------
        demandtypes_names = [c for c in demandtypes_ts.columns if 'DEMANDprox' in c]
        demandtypes_melt = demandtypes_ts.melt(id_vars='t', value_vars=demandtypes_names, var_name= 'demandtype', value_name= 'demand')
        subdf = subdf.merge(demandtypes_melt, how='left', on=['t', 'demandtype'])
        subdf.rename(columns={'demand': 'demand_kW'}, inplace=True)
        # checkpoint_to_logfile(f'  end merge demandtypes for subdf {i} to {i+stepsize-1}', self.sett.log_name, 0)


        # attach FLAECH_angletilt, might be usefull for later calculations
        subdf = subdf.assign(FLAECH_angletilt = subdf['FLAECHE'] * subdf['angletilt_factor'])


        # compute production -------------------------------------------------------------------------------- 
        # pvprod method 1 (false, presented to frank 8.11.24. missing efficiency grade)
        if pvprod_calc_method == 'method1':    
            subdf = subdf.assign(pvprod_kW = (subdf['radiation'] * subdf['FLAECHE'] * subdf['angletilt_factor']) / 1000).drop(columns=['meteo_loc', 'radiation'])

        # pvprod method 2.1
        elif pvprod_calc_method == 'method2.1':   
            subdf['pvprod_kW'] = (subdf['radiation'] / 1000 ) *                     inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] * subdf['angletilt_factor']
            formla_for_log_print = "subdf['pvprod_kW'] = subdf['radiation'] / 1000 * inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] * subdf['angletilt_factor']"

        # pvprod method 2.2
        elif pvprod_calc_method == 'method2.2':   
            subdf['pvprod_kW'] = (subdf['radiation'] / 1000 ) *                     inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] 
            formla_for_log_print = "subdf['pvprod_kW'] = subdf['radiation'] / 1000 * inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE']"

        # pvprod method 3.1
        elif pvprod_calc_method == 'method3.1':
            subdf['pvprod_kW'] =  subdf['radiation_rel_locmax'] * kWpeak_per_m2 *   inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] * subdf['angletilt_factor']
            formla_for_log_print = "subdf['pvprod_kW'] = subdf['radiation_rel_locmax'] * kWpeak_per_m2 * inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] * subdf['angletilt_factor']"

        # pvprod method 3.2
        elif pvprod_calc_method == 'method3.2':
            subdf['pvprod_kW'] =  subdf['radiation_rel_locmax'] * kWpeak_per_m2 *   inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE'] 
            formla_for_log_print = "subdf['pvprod_kW'] = subdf['radiation_rel_locmax'] * kWpeak_per_m2 * inverter_efficiency * share_roof_area_available * subdf['panel_efficiency'] * subdf['FLAECHE']"


        # pvprod method 3
            # > 19.11.2024: no longer needed. from previous runs where I wanted to compare different pvprod_computations methods
        elif False:   
            subdf['pvprod_kW'] = inverter_efficiency * share_roof_area_available * (subdf['radiation'] / 1000 ) * subdf['FLAECHE'] * subdf['angletilt_factor']
            subdf.drop(columns=['meteo_loc', 'radiation'], inplace=True)
            print_to_logfile("* calculation formula for pv production per roof:\n   > subdf['pvprod_kW'] = inverter_efficiency * share_roof_area_available * (subdf['radiation'] / 1000 ) * subdf['FLAECHE'] * subdf['angletilt_factor']\n", self.sett.log_name)
            
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
                    checkpoint_to_logfile(f' *ERROR* shading factor > 1 for df_uid: {dfuid}, EGID: {subdf.loc[dfuid_TF, "EGID"].unique()} ', self.sett.log_name, 0)
                subdf.loc[dfuid_TF, 'pvprod_kW'] = subdf.loc[dfuid_TF, 'pvprod_kW_noshade'] * shading_factor
            subdf.drop(columns=['meteo_loc', 'radiation', 'pvprod_kW_noshade'], inplace=True)
            print_to_logfile("* calculation formula for pv production per roof:\n   > subdf['pvprod_kW'] = <retrofitted_shading_factor> * inverter_efficiency  * (subdf['radiation'] / 1000 ) * subdf['FLAECHE'] * subdf['angletilt_factor'] \n", self.sett.log_name)
            


        # export subdf ----------------------------------------------
        subdf.to_parquet(f'{subdf_path}/topo_subdf_{i}to{i+stepsize-1}.parquet')
        topo_df.head(500).to_parquet(f'{subdf_path}/topo_df_500.parquet')

        if (i < 5) & self.sett.export_csvs:
            subdf.to_csv(f'{subdf_path}/topo_subdf_{i}to{i+stepsize-1}.csv', index=False)
            topo_df.head(500).to_csv(f'{subdf_path}/topo_df_500.csv', index=False) if i < 5 else None

        checkpoint_to_logfile(f'end merge to topo_time_subdf (tranche {tranche_counter}/{len(range(0, len(egids), stepsize))}, size {stepsize})', self.sett.log_name, 0)
        checkpoint_to_logfile(' * * DEBUGGIGN * * *: END loop subdfs', self.sett.log_name, 0)


    # print computation formula for comparing methods
    print_to_logfile(f'* Computation formula for pv production per roof:\n{formla_for_log_print}', self.sett.log_name)


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# algo_update_gridprem()
def algo_update_gridprem(self, subdir_path: str, i_m: int, m): 

    # setup -----------------------------------------------------
    print_to_logfile('run function: update_gridprem', self.sett.log_name)
    gridtiers_power_factor  = self.sett.GRIDspec_power_factor

    # import  -----------------------------------------------------
    topo = json.load(open(f'{subdir_path}/topo_egid.json', 'r'))
    dsonodes_df = pd.read_parquet(f'{self.sett.name_dir_import_path}/dsonodes_df.parquet')
    # gridprem_ts = pd.read_parquet(f'{subdir_path}/gridprem_ts.parquet')

    data = [(k, v[0], v[1]) for k, v in self.sett.GRIDspec_tiers.items()]
    gridtiers_df = pd.DataFrame(data, columns=self.sett.GRIDspec_colnames)

    checkpoint_to_logfile('**DEBUGGIG** > START LOOP through topo_egid', self.sett.log_name, 0, self.sett.show_debug_prints)
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

    checkpoint_to_logfile('**DEBUGGIG** > end loop through topo_egid', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None


    # import topo_time_subdfs -----------------------------------------------------
    # topo_subdf_paths = glob.glob(f'{self.sett.pvalloc_path}/topo_time_subdf/*.parquet')
    checkpoint_to_logfile('**DEBUGGIG** > start loop through subdfs', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None

    topo_subdf_paths = glob.glob(f'{subdir_path}/topo_subdf_*.parquet')
    agg_subinst_df_list = []
    # no_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == False]
    # wi_pv_egid = [k for k, v in topo.items() if v.get('pv_inst', {}).get('inst_TF') == True]

    i, path = 0, topo_subdf_paths[0]
    for i, path in enumerate(topo_subdf_paths):
        checkpoint_to_logfile('**DEBUGGIG** \t> start read subdfs', self.sett.log_name, 0) if i < 2 else None
        subdf = pd.read_parquet(path)
        checkpoint_to_logfile('**DEBUGGIG** \t> end read subdfs', self.sett.log_name, 0) if i < 2 else None

        subdf_updated = copy.deepcopy(subdf)
        subdf_updated.drop(columns=['info_source', 'inst_TF'], inplace=True)

        checkpoint_to_logfile('**DEBUGGIG** \t> start Map_infosource_egid', self.sett.log_name, 0, self.sett.show_debug_prints) if i < 2 else None
        subdf_updated = subdf_updated.merge(Map_infosource_egid[['EGID', 'info_source', 'inst_TF']], how='left', on='EGID')
        checkpoint_to_logfile('**DEBUGGIG** \t> end Map_infosource_egid', self.sett.log_name, 0, self.sett.show_debug_prints) if i < 2 else None
        # updated_instTF_srs, update_infosource_srs = subdf_updated['inst_TF'].fillna(subdf['inst_TF']), subdf_updated['info_source'].fillna(subdf['info_source'])
        # subdf['inst_TF'], subdf['info_source'] = updated_instTF_srs.infer_objects(copy=False), update_infosource_srs.infer_objects(copy=False)

        # Only consider production for houses that have built a pv installation and substract selfconsumption from the production
        subinst = copy.deepcopy(subdf_updated.loc[subdf_updated['inst_TF']==True])
        checkpoint_to_logfile('**DEBUGGIG** \t> pvprod_kw_to_numpy', self.sett.log_name, 0) if i < 2 else None
        pvprod_kW, demand_kW = subinst['pvprod_kW'].to_numpy(), subinst['demand_kW'].to_numpy()
        selfconsum_kW = np.minimum(pvprod_kW, demand_kW) * self.sett.TECspec_self_consumption_ifapplicable
        netdemand_kW = demand_kW - selfconsum_kW
        netfeedin_kW = pvprod_kW - selfconsum_kW

        subinst['netfeedin_kW'] = netfeedin_kW
        
        checkpoint_to_logfile('**DEBUGGIG** > end pvprod_kw_to_numpy', self.sett.log_name, 0, self.sett.show_debug_prints) if i < 2 else None
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

        agg_subinst = subinst.groupby(['grid_node', 't']).agg({'netfeedin_kW': 'sum', 'pvprod_kW':'sum'}).reset_index()
        del subinst
        agg_subinst_df_list.append(agg_subinst)
    
    checkpoint_to_logfile('**DEBUGGIG** > end loop through subdfs', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None


    # build gridnode_df -----------------------------------------------------
    gridnode_df = pd.concat(agg_subinst_df_list)
    # groupby df again because grid nodes will be spreach accross multiple tranches
    gridnode_df = gridnode_df.groupby(['grid_node', 't']).agg({'netfeedin_kW': 'sum', 'pvprod_kW':'sum'}).reset_index() 
    

    # attach node thresholds 
    gridnode_df = gridnode_df.merge(dsonodes_df[['grid_node', 'kVA_threshold']], how='left', on='grid_node')
    gridnode_df['kW_threshold'] = gridnode_df['kVA_threshold'] * self.sett.GRIDspec_perf_factor_1kVA_to_XkW

    gridnode_df['feedin_kW_taken'] = np.where(gridnode_df['netfeedin_kW'] > gridnode_df['kW_threshold'], gridnode_df['kW_threshold'], gridnode_df['netfeedin_kW'])
    gridnode_df['feedin_kW_loss'] =  np.where(gridnode_df['netfeedin_kW'] > gridnode_df['kW_threshold'], gridnode_df['netfeedin_kW'] - gridnode_df['kW_threshold'], 0)

    checkpoint_to_logfile('**DEBUGGIG** > end merge + npwhere subdfs', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None


    # update gridprem_ts -----------------------------------------------------
    gridnode_df.sort_values(by=['feedin_kW_taken'], ascending=False)
    gridnode_df_for_prem = gridnode_df.groupby(['grid_node','kW_threshold', 't']).agg({'feedin_kW_taken': 'sum'}).reset_index().copy()
    # gridprem_ts = gridprem_ts.merge(gridnode_df_for_prem[['grid_node', 't', 'kW_threshold', 'feedin_kW_taken']], how='left', on=['grid_node', 't'])
    gridprem_ts = gridnode_df_for_prem.copy()
    gridprem_ts['prem_Rp_kWh'] = 0.0
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

    checkpoint_to_logfile('**DEBUGGIG** > end update gridprem_ts', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None


    # EXPORT -----------------------------------------------------
    gridnode_df.to_parquet(f'{subdir_path}/gridnode_df.parquet')
    gridprem_ts.to_parquet(f'{subdir_path}/gridprem_ts.parquet')
    if self.sett.export_csvs:
        gridnode_df.to_csv(f'{subdir_path}/gridnode_df.csv', index=False)
        gridprem_ts.to_csv(f'{subdir_path}/gridprem_ts.csv', index=False)


    # export by Month -----------------------------------------------------
    if self.sett.MCspec_keep_files_month_iter_TF:
        if i_m < self.sett.MCspec_keep_files_month_iter_max:
            # gridprem_node_by_M_path = f'{self.sett.pvalloc_path}/pred_gridprem_node_by_M'
            gridprem_node_by_M_path = f'{subdir_path}/pred_gridprem_node_by_M'
            if not os.path.exists(gridprem_node_by_M_path):
                os.makedirs(gridprem_node_by_M_path)

            gridnode_df.to_parquet(f'{gridprem_node_by_M_path}/gridnode_df_{i_m}.parquet')
            gridprem_ts.to_parquet(f'{gridprem_node_by_M_path}/gridprem_ts_{i_m}.parquet')

            if self.sett.export_csvs:
                gridnode_df.to_csv(f'{gridprem_node_by_M_path}/gridnode_df_{i_m}.csv', index=False)
                gridprem_ts.to_csv(f'{gridprem_node_by_M_path}/gridprem_ts_{i_m}.csv', index=False)
            if i_m < 5:
                gridnode_df.to_csv(f'{gridprem_node_by_M_path}/gridnode_df_{i_m}.csv', index=False)
                gridprem_ts.to_csv(f'{gridprem_node_by_M_path}/gridprem_ts_{i_m}.csv', index=False)

    checkpoint_to_logfile('exported gridprem_ts and gridnode_df', self.sett.log_name, 0) if i_m < 3 else None


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# algo_update_npv_df()
def algo_update_npv_df(self, subdir_path: str, i_m: int, m):

    # setup -----------------------------------------------------
    print_to_logfile('run function: update_npv_df', self.sett.log_name)         

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
            print_to_logfile(f'updated npv (tranche {i+1}/{len(topo_subdf_paths)})', self.sett.log_name)
        subdf_t0 = pd.read_parquet(path)

        # drop egids with pv installations
        subdf = copy.deepcopy(subdf_t0[subdf_t0['EGID'].isin(no_pv_egid)])

        if not subdf.empty:

            # merge gridprem_ts
            subdf = subdf.merge(gridprem_ts[['t', 'prem_Rp_kWh', 'grid_node']], how='left', on=['t', 'grid_node']) 

            # compute selfconsumption + netdemand ----------------------------------------------
            subdf_array = subdf[['pvprod_kW', 'demand_kW', 'pvtarif_Rp_kWh', 'elecpri_Rp_kWh', 'prem_Rp_kWh']].to_numpy()
            pvprod_kW, demand_kW, pvtarif_Rp_kWh, elecpri_Rp_kWh, prem_Rp_kWh = subdf_array[:,0], subdf_array[:,1], subdf_array[:,2], subdf_array[:,3], subdf_array[:,4]

            demand_kW = demand_kW * self.sett.ALGOspec_tweak_gridnode_df_prod_demand_fact
            selfconsum_kW = np.minimum(pvprod_kW, demand_kW) * self.sett.TECspec_self_consumption_ifapplicable
            netdemand_kW = demand_kW - selfconsum_kW
            netfeedin_kW = pvprod_kW - selfconsum_kW

            econ_inc_chf = ((netfeedin_kW * pvtarif_Rp_kWh) /100) + ((selfconsum_kW * elecpri_Rp_kWh) /100)
            if not self.sett.ALGOspec_tweak_npv_excl_elec_demand:
                econ_spend_chf = ((netfeedin_kW * prem_Rp_kWh) / 100)  + ((netdemand_kW * elecpri_Rp_kWh) /100)
            else:
                econ_spend_chf = ((netfeedin_kW * prem_Rp_kWh) / 100)

            subdf['demand_kW'], subdf['pvprod_kW'], subdf['selfconsum_kW'], subdf['netdemand_kW'], subdf['netfeedin_kW'], subdf['econ_inc_chf'], subdf['econ_spend_chf'] = demand_kW, pvprod_kW, selfconsum_kW, netdemand_kW, netfeedin_kW, econ_inc_chf, econ_spend_chf
            

            if (i <3) and (i_m <3): 
                checkpoint_to_logfile('\t end compute econ factors', self.sett.log_name, 0, self.sett.show_debug_prints) #for subdf EGID {path.split("topo_subdf_")[1].split(".parquet")[0]}', self.sett.log_name, 0, self.sett.show_debug_prints)

            agg_subdf = subdf.groupby(
                                self.sett.ALGOspec_npv_update_groupby_cols_topo_aggdf).agg(
                                self.sett.ALGOspec_npv_update_agg_cols_topo_aggdf).reset_index()
                
            
            if (i <3) and (i_m <3): 
                checkpoint_to_logfile('\t groupby subdf to agg_subdf', self.sett.log_name, 0, self.sett.show_debug_prints)


            # create combinations ----------------------------------------------
            aggsub_npry = np.array(agg_subdf)

            egid_list, combo_df_uid_list, n_df_uid_list, bfs_list, gklas_list, demandtype_list, grid_node_list = [], [], [], [], [], [], []
            inst_list, info_source_list, pvid_list, pvtarif_Rp_kWh_list = [], [], [], []
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
                        bfs_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('bfs')][0])
                        gklas_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('gklas')][0])
                        demandtype_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('demandtype')][0])
                        grid_node_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('grid_node')][0])

                        inst_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('inst_TF')][0])
                        info_source_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('info_source')][0])
                        pvid_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pvid')][0])
                        pvtarif_Rp_kWh_list.append(aggsub_npry[mask_dfuid_subdf, agg_subdf.columns.get_loc('pvtarif_Rp_kWh')][0]) 
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



            aggsubdf_combo = pd.DataFrame({'EGID': egid_list, 'df_uid_combo': combo_df_uid_list, 
                                            'bfs': bfs_list, 'gklas': gklas_list, 
                                            'demandtype': demandtype_list, 'grid_node': grid_node_list,

                                        'inst_TF': inst_list, 'info_source': info_source_list, 'pvid': pvid_list,
                                        'pvtarif_Rp_kWh': pvtarif_Rp_kWh_list, 'elecpri_Rp_kWh': elecpri_Rp_kWh_list,
                                        'demand_kW': demand_list,

                                        'AUSRICHTUNG': ausrichtung_list, 'NEIGUNG': neigung_list,
                                        
                                        'FLAECHE': flaeche_list, 'STROMERTRAG': stromertrag_list,
                                        'FLAECH_angletilt': flaech_angletilt_list,
                                        'pvprod_kW': pvprod_list,
                                        'selfconsum_kW': selfconsum_list, 'netdemand_kW': netdemand_list, 'netfeedin_kW': netfeedin_list,
                                        'econ_inc_chf': econ_inc_chf_list, 'econ_spend_chf': econ_spend_chf_list})
                    
        if (i <3) and (i_m <3): 
            checkpoint_to_logfile(f'\t created df_uid combos for {agg_subdf["EGID"].nunique()} EGIDs', self.sett.log_name, 0, self.sett.show_debug_prints)

        

        # NPV calculation -----------------------------------------------------
        estim_instcost_chfpkW, estim_instcost_chftotal = self.initial_sml_get_instcost_interpolate_function(i_m)
        estim_instcost_chftotal(pd.Series([10, 20, 30, 40, 50, 60, 70]))

        # # estim_instcost_chfpkW, estim_instcost_chftotal = initial.estimate_iterpolate_instcost_function(pvalloc_settings)

        # if not os.path.exists(f'{preprep_name_dir_path }/pvinstcost_coefficients.json') == True:
        #     estim_instcost_chfpkW, estim_instcost_chftotal = initial.estimate_iterpolate_instcost_function(pvalloc_settings)
        #     estim_instcost_chftotal(pd.Series([10, 20, 30, 40, 50, 60, 70]))

        # elif os.path.exists(f'{preprep_name_dir_path }/pvinstcost_coefficients.json') == True:    
        #     estim_instcost_chfpkW, estim_instcost_chftotal = initial.get_estim_instcost_function(pvalloc_settings)
        #     estim_instcost_chftotal(pd.Series([10, 20, 30, 40, 50, 60, 70]))

        # correct cost estimation by a factor based on insights from pvprod_correction.py
        # aggsubdf_combo['estim_pvinstcost_chf'] = estim_instcost_chftotal(aggsubdf_combo['FLAECHE'] * 
        #                                                                  self.sett.TECspec_kWpeak_per_m2 * 
        #                                                                  self.sett.TECspec_share_roof_area_available) / self.sett.TECspec_estim_pvinst_cost_correctionfactor
        kwp_peak_array = aggsubdf_combo['FLAECHE'] * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available / self.sett.TECspec_estim_pvinst_cost_correctionfactor
        aggsubdf_combo['estim_pvinstcost_chf'] = estim_instcost_chftotal(kwp_peak_array) 
        
        

        def compute_npv(row):
            pv_cashflow = (row['econ_inc_chf'] - row['econ_spend_chf']) / (1+self.sett.TECspec_interest_rate)**np.arange(1, self.sett.TECspec_invst_maturity+1)
            npv = (-row['estim_pvinstcost_chf']) + np.sum(pv_cashflow)
            return npv
        aggsubdf_combo['NPV_uid'] = aggsubdf_combo.apply(compute_npv, axis=1)

        if (i <3) and (i_m <3): 
            checkpoint_to_logfile('\t computed NPV for agg_subdf', self.sett.log_name, 0, self.sett.show_debug_prints)

        agg_npv_df_list.append(aggsubdf_combo)

    agg_npv_df = pd.concat(agg_npv_df_list)
    npv_df = copy.deepcopy(agg_npv_df)


    # export npv_df -----------------------------------------------------
    npv_df.to_parquet(f'{subdir_path}/npv_df.parquet')
    if self.sett.export_csvs:
        npv_df.to_csv(f'{subdir_path}/npv_df.csv', index=False)


    # export by Month -----------------------------------------------------
    if self.sett.MCspec_keep_files_month_iter_TF:
        if i_m < self.sett.MCspec_keep_files_month_iter_max:
            pred_npv_inst_by_M_path = f'{subdir_path}/pred_npv_inst_by_M'
            if not os.path.exists(pred_npv_inst_by_M_path):
                os.makedirs(pred_npv_inst_by_M_path)

            npv_df.to_parquet(f'{pred_npv_inst_by_M_path}/npv_df_{i_m}.parquet')
            if self.sett.export_csvs:
                npv_df.to_csv(f'{pred_npv_inst_by_M_path}/npv_df_{i_m}.csv', index=False)
            if i_m < 5:
                npv_df.to_csv(f'{pred_npv_inst_by_M_path}/npv_df_{i_m}.csv', index=False)


    checkpoint_to_logfile('exported npv_df', self.sett.log_name, 0)
        

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

