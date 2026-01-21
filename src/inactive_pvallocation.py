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

# algo_update_npv_df_POLARS()
def algo_update_npv_df_POLARS(self, subdir_path: str, i_m: int, m):
    # setup -----------------------------------------------------
    print_to_logfile('run function: update_npv_df_POLARS', self.sett.log_name)         

    # import -----------------------------------------------------
    gridprem_ts = pl.read_parquet(f'{subdir_path}/gridprem_ts.parquet')    
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
        subdf_t0 = pl.read_parquet(path) # subdf_t0 = pd.read_parquet(path)

        # drop egids with pv installations
        subdf = subdf_t0.filter(pl.col("EGID").is_in(no_pv_egid))   

        if subdf.shape[0] > 0:

            # merge gridprem_ts
            checkpoint_to_logfile('npv > subdf: start merge subdf w gridprem_ts', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None
            subdf = subdf.join(gridprem_ts[['t', 'grid_node', 'prem_Rp_kWh']], on=['t', 'grid_node'], how='left')  
            checkpoint_to_logfile('npv > subdf: start merge subdf w gridprem_ts', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None


            # compute selfconsumption + netdemand ----------------------------------------------
            checkpoint_to_logfile('npv > subdf - all df_uid-combinations: start calc selfconsumption + netdemand', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None
            
            combo_rows = []
            
            for egid in list(subdf['EGID'].unique()):
                egid_subdf = subdf.filter(pl.col('EGID') == egid).clone()
                df_uids = list(egid_subdf['df_uid'].unique())

                for r in range(1, len(df_uids)+1):
                    for combo in itertools.combinations(df_uids,r):
                        combo_list = list(combo)
                        combo_str = '_'.join([str(c) for c in combo])

                        combo_subdf = egid_subdf.filter(pl.col('df_uid').is_in(combo_list)).clone()

                        # sorting necessary so that .first() statement captures inst_TF and info_source for EGIDS with partial installations
                        combo_subdf = combo_subdf.sort(['EGID','inst_TF', 'df_uid', 't_int'], descending=[False, True, False, False])
                        
                        # agg per EGID to apply selfconsumption, different to gridnode_update because more information needed in export csv/parquet
                        combo_agg_egid = combo_subdf.group_by(['EGID', 't', 't_int']).agg([
                            pl.col('inst_TF').first().alias('inst_TF'),
                            pl.col('info_source').first().alias('info_source'),
                            pl.col('grid_node').first().alias('grid_node'),
                            pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                            pl.col('pvtarif_Rp_kWh').first().alias('pvtarif_Rp_kWh'), 
                            pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),

                            pl.col('demand_kW').first().alias('demand_kW'),
                            pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                        ])

                        combo_agg_egid = combo_agg_egid.with_columns([
                            pl.lit(combo_str).alias('df_uid_combo')
                        ])

                        combo_agg_dfuid = combo_subdf.group_by(['EGID', 'df_uid']).agg([
                            pl.col('AUSRICHTUNG').first().alias('AUSRICHTUNG'),
                            pl.col('NEIGUNG').first().alias('NEIGUNG'), 
                            pl.col('FLAECHE').first().alias('FLAECHE'), 
                            pl.col('STROMERTRAG').first().alias('STROMERTRAG'), 
                            pl.col('GSTRAHLUNG').first().alias('GSTRAHLUNG'), 
                            pl.col('MSTRAHLUNG').first().alias('MSTRAHLUNG'), 
                        ])

                        # calc selfconsumption
                        combo_agg_egid = combo_agg_egid.sort(['EGID', 't_int'], descending = [False, False])

                        selfconsum_expr = pl.min_horizontal([pl.col("pvprod_kW"), pl.col("demand_kW")]) * self.sett.TECspec_self_consumption_ifapplicable

                        combo_agg_egid = combo_agg_egid.with_columns([        
                            selfconsum_expr.alias("selfconsum_kW"),
                            (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
                            (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
                        ])

                        # calc econ spend/inc chf
                        combo_agg_egid = combo_agg_egid.with_columns([
                            ((pl.col("netfeedin_kW") * pl.col("pvtarif_Rp_kWh")) / 100 + (pl.col("selfconsum_kW") * pl.col("elecpri_Rp_kWh")) / 100).alias("econ_inc_chf")
                        ])
                        
                        if not self.sett.ALGOspec_tweak_npv_excl_elec_demand:
                            combo_agg_egid = combo_agg_egid.with_columns([
                                ((pl.col("netfeedin_kW") * pl.col("prem_Rp_kWh")) / 100 +
                                (pl.col("demand_kW") * pl.col("elecpri_Rp_kWh")) / 100).alias("econ_spend_chf")
                            ])
                        else:
                            combo_agg_egid = combo_agg_egid.with_columns([
                                ((pl.col("netfeedin_kW") * pl.col("prem_Rp_kWh")) / 100).alias("econ_spend_chf")
                            ])

                        checkpoint_to_logfile('npv > subdf - all df_uid-combinations: end calc selfconsumption + netdemand', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None

                        row = {
                            'EGID':              combo_agg_egid['EGID'][0], 
                            'df_uid_combo':      combo_agg_egid['df_uid_combo'][0], 
                            'n_df_uid':          len(combo),
                            'inst_TF':           combo_agg_egid['inst_TF'][0],           
                            'info_source':       combo_agg_egid['info_source'][0],
                            'grid_node':         combo_agg_egid['grid_node'][0], 
                            'elecpri_Rp_kWh':    combo_agg_egid['elecpri_Rp_kWh'][0], 
                            'pvtarif_Rp_kWh':    combo_agg_egid['pvtarif_Rp_kWh'][0], 
                            'prem_Rp_kWh':       combo_agg_egid['prem_Rp_kWh'][0],                                     
                            'AUSRICHTUNG':       combo_agg_dfuid['AUSRICHTUNG'].mean(), 
                            'NEIGUNG':           combo_agg_dfuid['NEIGUNG'].mean(), 
                            'FLAECHE':           combo_agg_dfuid['FLAECHE'].sum(), 
                            'STROMERTRAG':       combo_agg_dfuid['STROMERTRAG'].sum(),
                            'GSTRAHLUNG':        combo_agg_dfuid['GSTRAHLUNG'].sum(),  
                            'MSTRAHLUNG':        combo_agg_dfuid['MSTRAHLUNG'].sum(), 
                            'demand_kW':         combo_agg_egid['demand_kW'].sum(), 
                            'pvprod_kW':         combo_agg_egid['pvprod_kW'].sum(), 
                            'selfconsum_kW':     combo_agg_egid['selfconsum_kW'].sum(), 
                            'netfeedin_kW':      combo_agg_egid['netfeedin_kW'].sum(), 
                            'netdemand_kW':      combo_agg_egid['netdemand_kW'].sum(), 
                            'econ_inc_chf':      combo_agg_egid['econ_inc_chf'].sum(), 
                            'econ_spend_chf':    combo_agg_egid['econ_spend_chf'].sum(), 

                        }

                        combo_rows.append(row)
                    aggsubdf_combo = pl.DataFrame(combo_rows)

        
        
        # NPV calculation -----------------------------------------------------
        estim_instcost_chfpkW, estim_instcost_chftotal = self.initial_sml_get_instcost_interpolate_function(i_m)
        estim_instcost_chftotal(pd.Series([10, 20, 30, 40, 50, 60, 70]))



        # correct cost estimation by a factor based on insights from pvprod_correction.py
        aggsubdf_combo = aggsubdf_combo.with_columns([
            (pl.col("FLAECHE") * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available).alias("roof_area_for_cost_kWpeak"),
        ])

        estim_instcost_chftotal_srs = estim_instcost_chftotal(aggsubdf_combo['roof_area_for_cost_kWpeak'] )
        aggsubdf_combo = aggsubdf_combo.with_columns(
            pl.Series("estim_pvinstcost_chf", estim_instcost_chftotal_srs)
        )


        checkpoint_to_logfile('npv > subdf: start calc npv', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None

        cashflow_srs =  aggsubdf_combo['econ_inc_chf'] - aggsubdf_combo['econ_spend_chf']
        cashflow_disc_list = []
        for j in range(1, self.sett.TECspec_invst_maturity+1):
            cashflow_disc_list.append(cashflow_srs / (1+self.sett.TECspec_interest_rate)**j)
        cashflow_disc_srs = sum(cashflow_disc_list)
        
        npv_srs = (-aggsubdf_combo['estim_pvinstcost_chf']) + cashflow_disc_srs

        aggsubdf_combo = aggsubdf_combo.with_columns(
            pl.Series("NPV_uid", npv_srs)
        )

        checkpoint_to_logfile('npv > subdf: end calc npv', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None

        agg_npv_df_list.append(aggsubdf_combo)

    agg_npv_df = pl.concat(agg_npv_df_list)
    npv_df = agg_npv_df.clone()

    # export npv_df -----------------------------------------------------
    npv_df.write_parquet(f'{subdir_path}/npv_df.parquet')
    # if self.sett.export_csvs:
    #     npv_df.write_csv(f'{subdir_path}/npv_df.csv', index=False)
        

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
        
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# algo_select_AND_adjust_topology()
def algo_select_AND_adjust_topology(self, subdir_path: str, i_m: int, m, while_safety_counter: int = 0):


    # print_to_logfile('run function: select_AND_adjust_topology', self.sett.log_name) if while_safety_counter < 5 else None

    # import ----------
    topo = json.load(open(f'{subdir_path}/topo_egid.json', 'r'))
    npv_df = pd.read_parquet(f'{subdir_path}/npv_df.parquet') 
    pred_inst_df = pd.read_parquet(f'{subdir_path}/pred_inst_df.parquet') if os.path.exists(f'{subdir_path}/pred_inst_df.parquet') else pd.DataFrame()


    # drop installed partitions from npv_df 
    #   -> otherwise multiple selection possible
    #   -> easier to drop inst before each selection than to create a list / df and carry it through the entire code)
    egid_wo_inst = [egid for egid in topo if  not topo.get(egid, {}).get('pv_inst', {}).get('inst_TF')]
    npv_df = copy.deepcopy(npv_df.loc[npv_df['EGID'].isin(egid_wo_inst)])


    #  SUBSELECTION FILTER specific scenarios ----------------
    
    if self.sett.ALGOspec_subselec_filter_criteria == 'southfacing_1spec':
        npv_subdf_angle_dfuid = copy.deepcopy(npv_df)
        npv_subdf_angle_dfuid = npv_subdf_angle_dfuid.loc[
                                    (npv_subdf_angle_dfuid['n_df_uid'] == 1 ) & 
                                    (npv_subdf_angle_dfuid['AUSRICHTUNG'] > -45) & 
                                    (npv_subdf_angle_dfuid['AUSRICHTUNG'] <  45)]
        
        if npv_subdf_angle_dfuid.shape[0] > 0:
            npv_df = copy.deepcopy(npv_subdf_angle_dfuid)

    elif self.sett.ALGOspec_subselec_filter_criteria == 'eastwestfacing_3spec':
        npv_subdf_angle_dfuid = copy.deepcopy(npv_df)
        
        selected_rows = []
        for egid, group in npv_subdf_angle_dfuid.groupby('EGID'):
            eastwest_spec = group[
                (group['n_df_uid'] == 2) &
                (group['AUSRICHTUNG'] > -30) &
                (group['AUSRICHTUNG'] < 30)
            ]
            east_spec = group[
                (group['n_df_uid'] == 1) &
                (group['AUSRICHTUNG'] > -135) &
                (group['AUSRICHTUNG'] < -45)
            ]
            west_spec = group[
                (group['n_df_uid'] == 1) &
                (group['AUSRICHTUNG'] > 45) &
                (group['AUSRICHTUNG'] < 135)
            ]
            
            if not eastwest_spec.empty:
                selected_rows.append(eastwest_spec)
            elif not west_spec.empty:
                selected_rows.append(west_spec)
            elif not east_spec.empty:
                selected_rows.append(east_spec)

        if len(selected_rows) > 0:
            npv_subdf_selected = pd.concat(selected_rows, ignore_index = True)
            # sanity check
            cols_to_show = ['EGID', 'df_uid_combo', 'n_df_uid', 'inst_TF', 'AUSRICHTUNG', 'NEIGUNG', 'FLAECHE']
            npv_subdf_angle_dfuid.loc[npv_subdf_angle_dfuid['EGID'].isin(['400507', '400614']), cols_to_show]
            npv_subdf_selected.loc[npv_subdf_selected['EGID'].isin(['400507', '400614']), cols_to_show]

            npv_df = copy.deepcopy(npv_subdf_selected)
            
    elif self.sett.ALGOspec_subselec_filter_criteria == 'southwestfacing_2spec':
        npv_subdf_angle_dfuid = copy.deepcopy(npv_df)
        
        selected_rows = []
        for egid, group in npv_subdf_angle_dfuid.groupby('EGID'):
            eastsouth_single_spec = group[
                (group['n_df_uid'] == 1) &
                (group['AUSRICHTUNG'] > -45) &
                (group['AUSRICHTUNG'] < 135)
            ]
            eastsouth_group_spec = group[
                (group['n_df_uid'] > 1) &
                (group['AUSRICHTUNG'] > 0) &    
                (group['AUSRICHTUNG'] < 90)
            ]
            
            if not eastsouth_group_spec.empty:
                selected_rows.append(eastsouth_group_spec)
            elif not eastsouth_single_spec.empty:
                selected_rows.append(eastsouth_single_spec)

        if len(selected_rows) > 0:
            npv_subdf_selected = pd.concat(selected_rows, ignore_index = True)
            # sanity check
            cols_to_show = ['EGID', 'df_uid_combo', 'n_df_uid', 'inst_TF', 'AUSRICHTUNG', 'NEIGUNG', 'FLAECHE']
            npv_subdf_angle_dfuid.loc[npv_subdf_angle_dfuid['EGID'].isin(['400507', '400614']), cols_to_show]
            npv_subdf_selected.loc[npv_subdf_selected['EGID'].isin(['400507', '400614']), cols_to_show]

            npv_df = copy.deepcopy(npv_subdf_selected)
            


    # SELECTION BY METHOD ---------------
    # set random seed
    if self.sett.ALGOspec_rand_seed is not None:
        np.random.seed(self.sett.ALGOspec_rand_seed)

    # have a list of egids to install on for sanity check. If all build, start building on the rest of EGIDs
    install_EGIDs_summary_sanitycheck = self.sett.CHECKspec_egid_list
    # if isinstance(install_EGIDs_summary_sanitycheck, list):
    if False:

        # remove duplicates from install_EGIDs_summary_sanitycheck
        unique_EGID = []
        for e in install_EGIDs_summary_sanitycheck:
                if e not in unique_EGID:
                    unique_EGID.append(e)
        install_EGIDs_summary_sanitycheck = unique_EGID
        # get remaining EGIDs of summary_sanitycheck_list that are not yet installed
        # > not even necessary if installed EGIDs get dropped from npv_df?
        remaining_egids = [
            egid for egid in install_EGIDs_summary_sanitycheck 
            if not topo.get(egid, {}).get('pv_inst', {}).get('inst_TF', False) == False ]
        
        if any([True if egid in npv_df['EGID'] else False for egid in remaining_egids]):
            npv_df = npv_df.loc[npv_df['EGID'].isin(remaining_egids)].copy()
        else:
            npv_df = npv_df.copy()
            

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
        picked_egid = npv_pick['EGID'].values[0]
        picked_uid = npv_pick['df_uid_combo'].values[0]
        picked_flaech = npv_pick['FLAECHE'].values[0]
        df_uid_w_inst = picked_uid.split('_')
        for col in ['NPV_stand', 'diff_NPV_rand']:
            if col in npv_pick.columns:
                npv_pick.drop(columns=['NPV_stand', 'diff_NPV_rand'], inplace=True)

    elif isinstance(npv_pick, pd.Series):
        picked_egid = npv_pick['EGID']
        picked_uid = npv_pick['df_uid_combo']
        picked_flaech = npv_pick['FLAECHE']
        df_uid_w_inst = picked_uid.split('_')
        for col in ['NPV_stand', 'diff_NPV_rand']:
            if col in npv_pick.index:
                npv_pick.drop(index=['NPV_stand', 'diff_NPV_rand'], inplace=True)
                
    inst_power = picked_flaech * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available
    npv_pick['inst_TF']          = True
    npv_pick['info_source']      = 'alloc_algorithm'
    npv_pick['xtf_id']           = picked_uid
    npv_pick['BeginOp']          = str(m)
    npv_pick['TotalPower']       = inst_power
    npv_pick['iter_round']       = i_m
    # npv_pick['df_uid_w_inst']  = df_uid_w_inst
    

    # Adjust export lists / df -----------------
    if '_' in picked_uid:
        picked_combo_uid = list(picked_uid.split('_'))
    else:
        picked_combo_uid = [picked_uid]

    if isinstance(npv_pick, pd.DataFrame):
        pred_inst_df = pd.concat([pred_inst_df, npv_pick])
    elif isinstance(npv_pick, pd.Series):
        pred_inst_df = pd.concat([pred_inst_df, npv_pick.to_frame().T])
    

    # Adjust topo + npv_df -----------------
    topo[picked_egid]['pv_inst'] = {'inst_TF': True, 
                                    'info_source': 'alloc_algorithm', 
                                    'xtf_id': picked_uid, 
                                    'BeginOp': f'{m}', 
                                    'TotalPower': inst_power, 
                                    'df_uid_w_inst': df_uid_w_inst}

    # again drop installed EGID (just to be sure, even though installed egids are excluded at the beginning)
    sum(npv_df['EGID'] != picked_egid)
    npv_df = copy.deepcopy(npv_df.loc[npv_df['EGID'] != picked_egid])


    # export main dfs ------------------------------------------
    # do not overwrite the original npv_df, this way can reimport it every month and filter for sanitycheck
    npv_df.to_parquet(f'{subdir_path}/npv_df.parquet')
    pred_inst_df.to_parquet(f'{subdir_path}/pred_inst_df.parquet')
    pred_inst_df.to_csv(f'{subdir_path}/pred_inst_df.csv') if self.sett.export_csvs else None
    with open (f'{subdir_path}/topo_egid.json', 'w') as f:
        json.dump(topo, f)


    # export by Month ------------------------------------------
    pred_inst_df.to_parquet(f'{subdir_path}/pred_npv_inst_by_M/pred_inst_df_{i_m}.parquet')
    pred_inst_df.to_csv(f'{subdir_path}/pred_npv_inst_by_M/pred_inst_df_{i_m}.csv') if self.sett.export_csvs else None
    with open(f'{subdir_path}/pred_npv_inst_by_M/topo_{i_m}.json', 'w') as f:
        json.dump(topo, f)
                
    return  inst_power    #, npv_df  # , picked_uid, picked_combo_uid, pred_inst_df, dfuid_installed_list, topo


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# algo_update_npv_df_OPTIMIZED()
def algo_update_npv_df_OPTIMIZED(self, subdir_path: str, i_m: int, m):
    
    # setup -----------------------------------------------------
    print_to_logfile('run function: update_npv_df_POLARS', self.sett.log_name)         

    # import -----------------------------------------------------
    gridprem_ts = pl.read_parquet(f'{subdir_path}/gridprem_ts.parquet')    
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
        subdf_t0 = pl.read_parquet(path) # subdf_t0 = pd.read_parquet(path)

        # drop egids with pv installations
        subdf = subdf_t0.filter(pl.col("EGID").is_in(no_pv_egid))   

        if subdf.shape[0] > 0:

            # merge gridprem_ts
            checkpoint_to_logfile('npv > subdf: start merge subdf w gridprem_ts', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None
            subdf = subdf.join(gridprem_ts[['t', 'grid_node', 'prem_Rp_kWh']], on=['t', 'grid_node'], how='left')  
            checkpoint_to_logfile('npv > subdf: start merge subdf w gridprem_ts', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None

            checkpoint_to_logfile('npv > subdf - all df_uid-combinations: start calc selfconsumption + netdemand', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None

            egid = '2362103'

            agg_npv_list = []                    
            n_egid_econ_functions_counter = 0
            for egid in list(subdf['EGID'].unique()):
                egid_subdf = subdf.filter(pl.col('EGID') == egid).clone()


                # compute npv of optimized installtion size ----------------------------------------------
                max_stromertrag = egid_subdf['STROMERTRAG'].max()
                max_dfuid_df = egid_subdf.filter(pl.col('STROMERTRAG') == max_stromertrag).sort(['t_int'], descending=[False,])
                max_dfuid_df.select(['EGID', 'df_uid', 't_int', 'STROMERTRAG', ])

                # find optimal installation size
                estim_instcost_chfpkW, estim_instcost_chftotal = self.initial_sml_get_instcost_interpolate_function(i_m)

                def calculate_npv(flaeche, max_dfuid_df, estim_instcost_chftotal, tweak_denominator=1.0
                                    ):
                    """
                    Calculate NPV for a given FLAECHE value
                    
                    Returns:
                    -------
                    float
                        Net Present Value (NPV) of the installation
                    """
                    # Copy the dataframe to avoid modifying the original
                    df = max_dfuid_df.clone()

                    if self.sett.TECspec_pvprod_calc_method == 'method2.2':
                        # Calculate production with the given FLAECHE
                        df = df.with_columns([
                            ((pl.col("radiation") / 1000) * 
                            pl.col("panel_efficiency") * 
                            self.sett.TECspec_inverter_efficiency * 
                            self.sett.TECspec_share_roof_area_available * 
                            flaeche).alias("pvprod_kW")
                        ])

                        # calc selfconsumption
                        selfconsum_expr = pl.min_horizontal([ pl.col("pvprod_kW"), pl.col("demand_kW") ]) * self.sett.TECspec_self_consumption_ifapplicable

                        df = df.with_columns([  
                            selfconsum_expr.alias("selfconsum_kW"),
                            (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
                            (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
                            ])
                        

                        df = df.with_columns([
                            (pl.col("pvtarif_Rp_kWh") / tweak_denominator).alias("pvtarif_Rp_kWh"),
                        ])
                        # calc economic income and spending
                        if not self.sett.ALGOspec_tweak_npv_excl_elec_demand:

                            df = df.with_columns([
                                ((pl.col("netfeedin_kW") * pl.col("pvtarif_Rp_kWh")) / 100 + 
                                (pl.col("selfconsum_kW") * pl.col("elecpri_Rp_kWh")) / 100).alias("econ_inc_chf"),

                                ((pl.col("netfeedin_kW") * pl.col("prem_Rp_kWh")) / 100 + 
                                (pl.col('demand_kW') * pl.col("elecpri_Rp_kWh")) / 100).alias("econ_spend_chf")
                                ])
                            
                        else:
                            df = df.with_columns([
                                ((pl.col("netfeedin_kW") * pl.col("pvtarif_Rp_kWh")) / 100 + 
                                (pl.col("selfconsum_kW") * pl.col("elecpri_Rp_kWh")) / 100).alias("econ_inc_chf"),
                                ((pl.col("netfeedin_kW") * pl.col("prem_Rp_kWh")) / 100).alias("econ_spend_chf")
                                ])

                        annual_cashflow = (df["econ_inc_chf"].sum() - df["econ_spend_chf"].sum())

                        # calc inst cost 
                        kWp = flaeche * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available
                        installation_cost = estim_instcost_chftotal(kWp)

                        # calc NPV
                        discount_factor = np.array([(1 + self.sett.TECspec_interest_rate)**-i for i in range(1, self.sett.TECspec_invst_maturity + 1)])
                        disc_cashflow = annual_cashflow * np.sum(discount_factor)
                        npv = -installation_cost + disc_cashflow
                        
                        pvprod_kW_sum = df['pvprod_kW'].sum()
                        demand_kW_sum = df['demand_kW'].sum()
                        selfconsum_kW_sum= df['selfconsum_kW'].sum()
                        rest = (installation_cost, disc_cashflow, pvprod_kW_sum, demand_kW_sum, selfconsum_kW_sum)
                        
                        # return npv, installation_cost, disc_cashflow, pvprod_kW_sum, demand_kW_sum, selfconsum_kW_sum
                        return npv, rest

                def optimize_pv_size(max_dfuid_df, estim_instcost_chftotal, max_flaeche_factor=None):
                    """
                    Find the optimal PV installation size (FLAECHE) that maximizes NPV
                    
                    """
                    def obj_func(flaeche):
                        npv, rest = calculate_npv(flaeche, max_dfuid_df, estim_instcost_chftotal)
                        return -npv  

                    
                    # Set bounds - minimum FLAECHE is 0, maximum is either specified or from the data
                    if max_flaeche_factor is not None:
                        max_flaeche = max(max_dfuid_df['FLAECHE']) * max_flaeche_factor
                    else:
                        max_flaeche = max(max_dfuid_df['FLAECHE'])

                        
                    
                    # Run the optimization
                    result = optimize.minimize_scalar(
                        obj_func,
                        bounds=(0, max_flaeche),
                        method='bounded'
                    )
                    
                    # optimal values
                    optimal_flaeche = result.x
                    optimal_npv = -result.fun
                                                
                    return optimal_flaeche, optimal_npv
                                                
                opt_flaeche, opt_npv = optimize_pv_size(max_dfuid_df, estim_instcost_chftotal, self.sett.TECspec_opt_max_flaeche_factor)
                opt_kWpeak = opt_flaeche * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available


                # plot economic functions
                if (i_m < 2) & (n_egid_econ_functions_counter < 3):
                    fig_econ_comp =  go.Figure()
                    # for tweak_denominator in [0.5, 1.0, 1.5, 2.0, 2.5, ]:
                    tweak_denominator = 1.0
                    # fig_econ_comp =  go.Figure()
                    flaeche_range = np.linspace(0, int(max_dfuid_df['FLAECHE'].max()) , 200)
                    kWpeak_range = self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available * flaeche_range
                    # cost_kWp = estim_instcost_chftotal(kWpeak_range)
                    npv_list, cost_list, cashflow_list, pvprod_kW_list, demand_kW_list, selfconsum_kW_list = [], [], [], [], [], []
                    for flaeche in flaeche_range:
                        npv, rest = calculate_npv(flaeche, max_dfuid_df, estim_instcost_chftotal, tweak_denominator)
                        npv_list.append(npv)

                        cost_list.append(         rest[0]) 
                        cashflow_list.append(     rest[1]) 
                        pvprod_kW_list.append(    rest[2]) 
                        demand_kW_list.append(    rest[3]) 
                        selfconsum_kW_list.append(rest[4]) 
                            
                    npv = np.array(npv_list)
                    cost_kWp = np.array(cost_list)
                    cashflow = np.array(cashflow_list)
                    pvprod_kW_sum = np.array(pvprod_kW_list)
                    demand_kW_sum = np.array(demand_kW_list)
                    selfconsum_kW_sum = np.array(selfconsum_kW_list)

                    pvtarif = max_dfuid_df['pvtarif_Rp_kWh'][0] / tweak_denominator
                    elecpri = max_dfuid_df['elecpri_Rp_kWh'][0]

                    fig_econ_comp.add_trace(go.Scatter( x=kWpeak_range,   y=cost_kWp,           mode='lines',  name=f'Installation Cost (CHF)   - pvtarif:{pvtarif}, elecpri:{elecpri} (Rp/kWh)', )) # line=dict(color='blue')))
                    fig_econ_comp.add_trace(go.Scatter( x=kWpeak_range,   y=cashflow,           mode='lines',  name=f'Cash Flow (CHF)           - pvtarif:{pvtarif}, elecpri:{elecpri} (Rp/kWh)',         )) # line=dict(color='magenta')))
                    fig_econ_comp.add_trace(go.Scatter( x=kWpeak_range,   y=npv,                mode='lines',  name=f'Net Present Value (CHF)   - pvtarif:{pvtarif}, elecpri:{elecpri} (Rp/kWh)', )) # line=dict(color='green')))
                    fig_econ_comp.add_trace(go.Scatter( x=kWpeak_range,   y=pvprod_kW_sum,      mode='lines',  name=f'PV Production (kWh)       - pvtarif:{pvtarif}, elecpri:{elecpri} (Rp/kWh)',     )) # line=dict(color='orange')))
                    fig_econ_comp.add_trace(go.Scatter( x=kWpeak_range,   y=demand_kW_sum,      mode='lines',  name=f'Demand (kWh)              - pvtarif:{pvtarif}, elecpri:{elecpri} (Rp/kWh)',            )) # line=dict(color='red')))
                    fig_econ_comp.add_trace(go.Scatter( x=kWpeak_range,   y=selfconsum_kW_sum,  mode='lines',  name=f'Self-consumption (kWh)    - pvtarif:{pvtarif}, elecpri:{elecpri} (Rp/kWh)',  )) # line=dict(color='purple')))
                    fig_econ_comp.add_trace(go.Scatter( x=[None,],        y=[None,],            mode='lines',  name='',  opacity = 0    ))
                    fig_econ_comp.update_layout(
                        title=f'Economic Comparison for EGID: {max_dfuid_df["EGID"][0]} (Tweak Denominator: {tweak_denominator})',
                        xaxis_title='System Size (kWp)',
                        yaxis_title='Value (CHF/kWh)',
                        legend=dict(x=0.99, y=0.99),
                        template='plotly_white'
                    )
                    fig_econ_comp.write_html(f'{subdir_path}/npv_kWp_optim_factors{egid}.html', auto_open=False)
                    n_egid_econ_functions_counter += 1
                    # fig_econ_comp.show()


                # calculate df for optim inst per egid ----------------------------------------------
                # optimal production
                flaeche = opt_flaeche
                npv, rest = calculate_npv(opt_flaeche, max_dfuid_df, estim_instcost_chftotal, tweak_denominator=1.0)
                installation_cost, disc_cashflow, pvprod_kW_sum, demand_kW_sum, selfconsum_kW_sum = rest[0], rest[1], rest[2], rest[3], rest[4]
                
                max_dfuid_df = max_dfuid_df.with_columns([
                    pl.lit(opt_flaeche).alias("opt_FLAECHE"),

                    pl.lit(opt_npv).alias("NPV_uid"),
                    pl.lit(opt_kWpeak).alias("dfuidPower"),
                    pl.lit(installation_cost).alias("estim_pvinstcost_chf"),
                    pl.lit(disc_cashflow).alias("disc_cashflow"),
                    ])
                
                # if self.sett.TECspec_pvprod_calc_method == 'method2.2':
                max_dfuid_df = max_dfuid_df.with_columns([
                    ((pl.col("radiation") / 1000) * 
                    pl.col("panel_efficiency") * 
                    self.sett.TECspec_inverter_efficiency * 
                    self.sett.TECspec_share_roof_area_available * 
                    pl.col("opt_FLAECHE")).alias("pvprod_kW")
                ])

                selfconsum_expr = pl.min_horizontal([ pl.col("pvprod_kW"), pl.col("demand_kW") ]) * self.sett.TECspec_self_consumption_ifapplicable
                
                max_dfuid_df = max_dfuid_df.with_columns([  
                    selfconsum_expr.alias("selfconsum_kW"),
                    (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
                    (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
                    ])                        
                
                
                egid_npv_optim = max_dfuid_df.group_by(['EGID', ]).agg([
                    pl.col('df_uid').first().alias('df_uid'),
                    pl.col('GKLAS').first().alias('GKLAS'),
                    pl.col('GAREA').first().alias('GAREA'),
                    pl.col('sfhmfh_typ').first().alias('sfhmfh_typ'),
                    pl.col('demand_arch_typ').first().alias('demand_arch_typ'),
                    pl.col('demand_elec_pGAREA').first().alias('demand_elec_pGAREA'),
                    pl.col('grid_node').first().alias('grid_node'),
                    pl.col('inst_TF').first().alias('inst_TF'),
                    pl.col('info_source').first().alias('info_source'),
                    pl.col('pvid').first().alias('pvid'),
                    pl.col('pvtarif_Rp_kWh').first().alias('pvtarif_Rp_kWh'),
                    pl.col('TotalPower').first().alias('TotalPower'),
                    # pl.col('dfuid_w_inst_tuples').first().alias('dfuid_w_inst_tuples'),
                    pl.col('FLAECHE').first().alias('FLAECHE'),
                    pl.col('AUSRICHTUNG').first().alias('AUSRICHTUNG'),
                    pl.col('STROMERTRAG').first().alias('STROMERTRAG'),
                    pl.col('NEIGUNG').first().alias('NEIGUNG'),
                    pl.col('MSTRAHLUNG').first().alias('MSTRAHLUNG'),
                    pl.col('GSTRAHLUNG').first().alias('GSTRAHLUNG'),
                    pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                    pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),

                    pl.col('opt_FLAECHE').first().alias('opt_FLAECHE'),
                    pl.col('NPV_uid').first().alias('NPV_uid'),
                    pl.col('estim_pvinstcost_chf').first().alias('estim_pvinstcost_chf'),
                    pl.col('disc_cashflow').first().alias('disc_cashflow'),
                    pl.col('dfuidPower').first().alias('dfuidPower'),
                    pl.col('share_pvprod_used').first().alias('share_pvprod_used'),

                    pl.col('demand_kW').sum().alias('demand_kW'),
                    pl.col('poss_pvprod_kW').sum().alias('poss_pvprod'),
                    pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                    pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                    pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                    pl.col('netdemand_kW').sum().alias('netdemand_kW'),
                ])
                
                agg_npv_list.append(egid_npv_optim)

            agg_npv_df = pl.concat(agg_npv_list)
            npv_df = agg_npv_df.clone()

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


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# algo_select_AND_adjust_topology_OPTIMIZED()
def algo_select_AND_adjust_topology_OPTIMIZED(self, subdir_path: str, i_m: int, m, while_safety_counter: int = 0):

    # print_to_logfile('run function: select_AND_adjust_topology', self.sett.log_name) if while_safety_counter < 5 else None

    # import ----------
    topo = json.load(open(f'{subdir_path}/topo_egid.json', 'r'))
    npv_df = pd.read_parquet(f'{subdir_path}/npv_df.parquet') 
    pred_inst_df = pd.read_parquet(f'{subdir_path}/pred_inst_df.parquet') if os.path.exists(f'{subdir_path}/pred_inst_df.parquet') else pd.DataFrame()


    #  SUBSELECTION FILTER specific scenarios ----------------
    
    if self.sett.ALGOspec_subselec_filter_criteria == 'southfacing_1spec':
        npv_subdf_angle_dfuid = copy.deepcopy(npv_df)
        npv_subdf_angle_dfuid = npv_subdf_angle_dfuid.loc[
                                    (npv_subdf_angle_dfuid['n_df_uid'] == 1 ) & 
                                    (npv_subdf_angle_dfuid['AUSRICHTUNG'] > -45) & 
                                    (npv_subdf_angle_dfuid['AUSRICHTUNG'] <  45)]
        
        if npv_subdf_angle_dfuid.shape[0] > 0:
            npv_df = copy.deepcopy(npv_subdf_angle_dfuid)

    elif self.sett.ALGOspec_subselec_filter_criteria == 'eastwestfacing_3spec':
        npv_subdf_angle_dfuid = copy.deepcopy(npv_df)
        
        selected_rows = []
        for egid, group in npv_subdf_angle_dfuid.groupby('EGID'):
            eastwest_spec = group[
                (group['n_df_uid'] == 2) &
                (group['AUSRICHTUNG'] > -30) &
                (group['AUSRICHTUNG'] < 30)
            ]
            east_spec = group[
                (group['n_df_uid'] == 1) &
                (group['AUSRICHTUNG'] > -135) &
                (group['AUSRICHTUNG'] < -45)
            ]
            west_spec = group[
                (group['n_df_uid'] == 1) &
                (group['AUSRICHTUNG'] > 45) &
                (group['AUSRICHTUNG'] < 135)
            ]
            
            if not eastwest_spec.empty:
                selected_rows.append(eastwest_spec)
            elif not west_spec.empty:
                selected_rows.append(west_spec)
            elif not east_spec.empty:
                selected_rows.append(east_spec)

        if len(selected_rows) > 0:
            npv_subdf_selected = pd.concat(selected_rows, ignore_index = True)
            # sanity check
            cols_to_show = ['EGID', 'df_uid_combo', 'n_df_uid', 'inst_TF', 'AUSRICHTUNG', 'NEIGUNG', 'FLAECHE']
            npv_subdf_angle_dfuid.loc[npv_subdf_angle_dfuid['EGID'].isin(['400507', '400614']), cols_to_show]
            npv_subdf_selected.loc[npv_subdf_selected['EGID'].isin(['400507', '400614']), cols_to_show]

            npv_df = copy.deepcopy(npv_subdf_selected)
            
    elif self.sett.ALGOspec_subselec_filter_criteria == 'southwestfacing_2spec':
        npv_subdf_angle_dfuid = copy.deepcopy(npv_df)
        
        selected_rows = []
        for egid, group in npv_subdf_angle_dfuid.groupby('EGID'):
            eastsouth_single_spec = group[
                (group['n_df_uid'] == 1) &
                (group['AUSRICHTUNG'] > -45) &
                (group['AUSRICHTUNG'] < 135)
            ]
            eastsouth_group_spec = group[
                (group['n_df_uid'] > 1) &
                (group['AUSRICHTUNG'] > 0) &    
                (group['AUSRICHTUNG'] < 90)
            ]
            
            if not eastsouth_group_spec.empty:
                selected_rows.append(eastsouth_group_spec)
            elif not eastsouth_single_spec.empty:
                selected_rows.append(eastsouth_single_spec)

        if len(selected_rows) > 0:
            npv_subdf_selected = pd.concat(selected_rows, ignore_index = True)
            # sanity check
            cols_to_show = ['EGID', 'df_uid_combo', 'n_df_uid', 'inst_TF', 'AUSRICHTUNG', 'NEIGUNG', 'FLAECHE']
            npv_subdf_angle_dfuid.loc[npv_subdf_angle_dfuid['EGID'].isin(['400507', '400614']), cols_to_show]
            npv_subdf_selected.loc[npv_subdf_selected['EGID'].isin(['400507', '400614']), cols_to_show]

            npv_df = copy.deepcopy(npv_subdf_selected)
            


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
        picked_dfuid             = npv_pick['df_uid'].values[0]
        picked_flaeche           = npv_pick['opt_FLAECHE'].values[0]
        # picked_dfuidPower        = npv_pick['dfuidPower'].values[0]
        # picked_share_pvprod_used = npv_pick['share_pvprod_used'].values[0]
        picked_demand_kW         = npv_pick['demand_kW'].values[0]
        picked_poss_pvprod       = npv_pick['poss_pvprod'].values[0]
        picked_pvprod_kW         = npv_pick['pvprod_kW'].values[0]
        picked_selfconsum_kW     = npv_pick['selfconsum_kW'].values[0]
        picked_netfeedin_kW      = npv_pick['netfeedin_kW'].values[0]
        picked_netdemand_kW      = npv_pick['netdemand_kW'].values[0]


    elif isinstance(npv_pick, pd.Series):
        picked_egid = npv_pick['EGID']



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
    inst_power = picked_flaeche * self.sett.TECspec_kWpeak_per_m2 * self.sett.TECspec_share_roof_area_available
    remaining_flaeche = picked_flaeche


    cols_to_add = ['inst_TF', 'info_source', 'xtf_id', 'BeginOp', 'dfuidPower', 
                    'share_pvprod_used', 'demand_kW', 'poss_pvprod', 'pvprod_kW', 
                    'selfconsum_kW', 'netfeedin_kW', 'netdemand_kW', 
                    ]
    for col in cols_to_add:  # add empty cols to fill in later
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
        topo_pick_df.loc[idx, 'info_source'] = '       alloc_algorithm'                       if flaeche_ratio > 0.0 else ''
        topo_pick_df.loc[idx, 'BeginOp'] =             str(m)                                 if flaeche_ratio > 0.0 else ''
        topo_pick_df.loc[idx, 'iter_round'] =          i_m                                    if flaeche_ratio > 0.0 else ''
        topo_pick_df.loc[idx, 'xtf_id'] =              picked_dfuid                           if flaeche_ratio > 0.0 else ''
        topo_pick_df.loc[idx, 'demand_kW'] =           picked_demand_kW                       if flaeche_ratio > 0.0 else 0.0
        topo_pick_df.loc[idx, 'dfuidPower'] =          flaeche_ratio * dfuid_inst_power       if flaeche_ratio > 0.0 else 0.0
        topo_pick_df.loc[idx, 'poss_pvprod'] =         flaeche_ratio * picked_poss_pvprod     if flaeche_ratio > 0.0 else 0.0
        topo_pick_df.loc[idx, 'pvprod_kW'] =           flaeche_ratio * picked_pvprod_kW       if flaeche_ratio > 0.0 else 0.0
        topo_pick_df.loc[idx, 'selfconsum_kW'] =       flaeche_ratio * picked_selfconsum_kW   if flaeche_ratio > 0.0 else 0.0
        topo_pick_df.loc[idx, 'netfeedin_kW'] =        flaeche_ratio * picked_netfeedin_kW    if flaeche_ratio > 0.0 else 0.0
        topo_pick_df.loc[idx, 'netdemand_kW'] =        flaeche_ratio * picked_netdemand_kW    if flaeche_ratio > 0.0 else 0.0
    
        
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
                                    'xtf_id': picked_dfuid, 
                                    'BeginOp': f'{m}', 
                                    'TotalPower': inst_power, 
                                    'dfuid_w_inst_tuples': dfuid_w_inst_tuples
                                    }

    # drop installed EGID (just to be sure, even though installed egids are excluded at the beginning)
    npv_df = copy.deepcopy(npv_df.loc[npv_df['EGID'] != picked_egid])



    # export main dfs ------------------------------------------
    # do not overwrite the original npv_df, this way can reimport it every month and filter for sanitycheck
    npv_df.to_parquet(f'{subdir_path}/npv_df.parquet')
    pred_inst_df.to_parquet(f'{subdir_path}/pred_inst_df.parquet')
    pred_inst_df.to_csv(f'{subdir_path}/pred_inst_df.csv') if self.sett.export_csvs else None
    with open (f'{subdir_path}/topo_egid.json', 'w') as f:
        json.dump(topo, f)


    # export by Month ------------------------------------------
    pred_inst_df.to_parquet(f'{subdir_path}/pred_npv_inst_by_M/pred_inst_df_{i_m}.parquet')
    pred_inst_df.to_csv(f'{subdir_path}/pred_npv_inst_by_M/pred_inst_df_{i_m}.csv') if self.sett.export_csvs else None
    with open(f'{subdir_path}/pred_npv_inst_by_M/topo_{i_m}.json', 'w') as f:
        json.dump(topo, f)
                
    return  inst_power    #, npv_df  # , picked_uid, picked_combo_uid, pred_inst_df, dfuid_installed_list, topo


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------

# algo_update_npv_df_RF_SEGMDIST()
def algo_update_npv_df_RF_SEGMDIST(self, subdir_path: str, i_m: int, m):
    """
        This function estimates the installation size of all houses in sample, based on a previously run statistical model calibration. 
        This stat model coefficients are imported and used to determine the most realistic installation size chose for the house
        Model used: 
            - Random Forest Classifer + skew.norm segment distribution of TotalPower (kWp)
                (RF to determine kWp segment. Then historic skew.norm distribution fitted to kWp segment of actual installtions. 
                Draw n random samples from the distribution to estimate PV installation size)

    """

    # setup -----------------------------------------------------
    # print_to_logfile('run function: algo_update_npv_df_STATESTIM', self.sett.log_name)         

    # import -----------------------------------------------------
    gridprem_ts = pl.read_parquet(f'{subdir_path}/gridprem_ts.parquet')    
    topo = json.load(open(f'{subdir_path}/topo_egid.json', 'r'))

    rfr_model    = joblib.load(f'{self.sett.calib_model_coefs}/{self.sett.ALGOspec_calib_estim_mod_name_pkl}_model.pkl')
    encoder      = joblib.load(f'{self.sett.calib_model_coefs}/{self.sett.ALGOspec_calib_estim_mod_name_pkl}_encoder.pkl')
    if os.path.exists(f'{self.sett.calib_model_coefs}/{self.sett.ALGOspec_calib_estim_mod_name_pkl}_kWp_segments.json'):
        try:
            kWp_segments = json.load(open(f'{self.sett.calib_model_coefs}/{self.sett.ALGOspec_calib_estim_mod_name_pkl}_kWp_segments.json', 'r'))
        except:
            print_to_logfile('Error loading kWp_segments json file', self.sett.log_name)
    elif not os.path.exists(f'{self.sett.calib_model_coefs}/{self.sett.ALGOspec_calib_estim_mod_name_pkl}_kWp_segments.json'):
        try:
            kWp_segments = json.load(open(f'{self.sett.calib_model_coefs}/rfr_segment_distribution_{self.sett.ALGOspec_calib_estim_mod_name_pkl}.json', 'r'))
        except:
            print_to_logfile('Error loading kWp_segments json file', self.sett.log_name)
        


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
        subdf_t0 = pl.read_parquet(path) # subdf_t0 = pd.read_parquet(path)

        # drop egids with pv installations
        subdf = subdf_t0.filter(pl.col("EGID").is_in(no_pv_egid))   

        if subdf.shape[0] > 0:

            # merge gridprem_ts
            checkpoint_to_logfile('npv > subdf: start merge subdf w gridprem_ts', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None
            subdf = subdf.join(gridprem_ts[['t', 'grid_node', 'prem_Rp_kWh']], on=['t', 'grid_node'], how='left')  
            checkpoint_to_logfile('npv > subdf: start merge subdf w gridprem_ts', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None

            checkpoint_to_logfile('npv > subdf - all df_uid-combinations: start calc selfconsumption + netdemand', self.sett.log_name, 0, self.sett.show_debug_prints) if i_m < 3 else None

            egid = '2362103'

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
                    pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),
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
                        pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),

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
                
                x_cols = [tupl[0] for tupl in cols_dtypes_tupls.items() if tupl[0] not in ['TotalPower', ]]


                # RF segment estimation ----------------
                X = df.drop(columns=['TotalPower',])
                cat_cols = X.select_dtypes(include=["object", "category"]).columns
                encoded_array = encoder.transform(X[cat_cols].astype(str))
                encoded_df = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(cat_cols))
                
                X_final = pd.concat(
                    [X.drop(columns=cat_cols).reset_index(drop=True), encoded_df.reset_index(drop=True)],
                    axis=1)
                X_final = X_final[rfr_model.feature_names_in_]

                pred_kwp_segm = rfr_model.predict(X_final)[0]
                df['pred_instPower_segm'] = pred_kwp_segm


                # inst kWp pick of distirbution ----------------
                df['pred_dfuidPower'] = np.nan
                for segment_str, segment_dict in kWp_segments.items():
                    mask = df['pred_instPower_segm'] == segment_str
                    n_rows = mask.sum()

                    if n_rows == 0:
                        continue

                    nEGID     = segment_dict['nEGID_in_segment']
                    mean      = segment_dict['TotalPower_mean_seg']
                    stdev     = segment_dict['TotalPower_std_seg']
                    skewness  = segment_dict['TotalPower_skew_seg']
                    kurto     = segment_dict['TotalPower_kurt_seg']

                    if stdev == 0:
                        df.loc[mask, 'pred_dfuidPower'] = mean
                        continue

                    pred_instPower = pearson3.rvs(skew=skewness, loc=mean, scale=stdev, size=n_rows)
                    df.loc[mask, 'pred_dfuidPower'] = pred_instPower


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
                        pl.col('grid_node').first().alias('grid_node'),
                        pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                        pl.col('pvtarif_Rp_kWh').first().alias('pvtarif_Rp_kWh'), 
                        pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),
                        pl.col('TotalPower').first().alias('TotalPower'),
                        pl.col('AUSRICHTUNG').first().alias('AUSRICHTUNG'),
                        pl.col('NEIGUNG').first().alias('NEIGUNG'),

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


                    # calc selfconsumption
                    egid_agg = egid_agg.sort(['EGID', 't_int'], descending = [False, False])

                    selfconsum_expr = pl.min_horizontal([pl.col("pvprod_kW"), pl.col("demand_kW")]) * self.sett.TECspec_self_consumption_ifapplicable

                    egid_agg = egid_agg.with_columns([        
                        selfconsum_expr.alias("selfconsum_kW"),
                        (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
                        (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
                    ])

                    # calc econ spend/inc chf
                    egid_agg = egid_agg.with_columns([
                        ((pl.col("netfeedin_kW") * pl.col("pvtarif_Rp_kWh")) / 100 + (pl.col("selfconsum_kW") * pl.col("elecpri_Rp_kWh")) / 100).alias("econ_inc_chf")
                    ])
                    
                    if not self.sett.ALGOspec_tweak_npv_excl_elec_demand:
                        egid_agg = egid_agg.with_columns([
                            ((pl.col("netfeedin_kW") * pl.col("prem_Rp_kWh")) / 100 +
                            (pl.col("demand_kW") * pl.col("elecpri_Rp_kWh")) / 100).alias("econ_spend_chf")
                        ])
                    else:
                        egid_agg = egid_agg.with_columns([
                            ((pl.col("netfeedin_kW") * pl.col("prem_Rp_kWh")) / 100).alias("econ_spend_chf")
                        ])


                    # NPV calculation -----------------
                    estim_instcost_chfpkW, estim_instcost_chftotal = self.initial_sml_get_instcost_interpolate_function(i_m)
                    estim_instcost_chftotal(pd.Series([10, 20, 30, 40, 50, 60, 70]))

                    annual_cashflow = (egid_agg["econ_inc_chf"].sum() - egid_agg["econ_spend_chf"].sum())
                    installation_cost = estim_instcost_chftotal(pred_instPower)
                        
                    discount_factor = np.array([(1 + self.sett.TECspec_interest_rate)**-i for i in range(1, self.sett.TECspec_invst_maturity + 1)])
                    disc_cashflow = annual_cashflow * np.sum(discount_factor)
                    npv = -installation_cost + disc_cashflow

                    egid_npv = egid_agg.group_by(['EGID', ]).agg([
                        pl.col('df_uid_winst').first().alias('df_uid_winst'),
                        pl.col('n_dfuid').first().alias('n_dfuid'),
                        pl.col('grid_node').first().alias('grid_node'),
                        pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                        pl.col('pvtarif_Rp_kWh').first().alias('pvtarif_Rp_kWh'), 
                        pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),
                        pl.col('TotalPower').first().alias('TotalPower'),
                        pl.col('AUSRICHTUNG').first().alias('AUSRICHTUNG'),
                        pl.col('NEIGUNG').first().alias('NEIGUNG'),
                        pl.col('FLAECHE').first().alias('FLAECHE'),
                    
                        pl.col('poss_pvprod_kW').sum().alias('poss_pvprod_kW'),
                        pl.col('demand_kW').sum().alias('demand_kW'),
                        pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                        pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                        pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
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
        agg_npv_df = pl.concat(agg_npv_df_list)
        npv_df = agg_npv_df.clone()

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


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------



# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------


