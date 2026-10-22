# python integrated packages
import sys
import os as os
import copy
import glob
import itertools
import json
import random

# external packages
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import polars as pl
from scipy import optimize

# own modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.auxiliary_functions import checkpoint_to_logfile, print_to_logfile
from src.visualization.visual_core import VisualizationCore


class PlotIndMaplineProdHOYEGIDrfcombo(VisualizationCore):
    """Plot of the pvalloc_MC_algorithm stage."""

    def plot_ind_mapline_prodHOY_EGIDrfcombo(self, plot_show: bool = True, plot_export: bool = True):

        trace_color_dict = {
            'Blues': pc.sequential.Blues, 'Greens': pc.sequential.Greens, 'Reds': pc.sequential.Reds, 'Oranges': pc.sequential.Oranges,
            'Purples': pc.sequential.Purples, 'Greys': pc.sequential.Greys, 'Mint': pc.sequential.Mint, 'solar': pc.sequential.solar,
            'Teal': pc.sequential.Teal, 'Magenta': pc.sequential.Magenta, 'Plotly3': pc.sequential.Plotly3,
            'Viridis': pc.sequential.Viridis, 'Turbo': pc.sequential.Turbo, 'Blackbody': pc.sequential.Blackbody, 
            'Bluered': pc.sequential.Bluered, 'Aggrnyl': pc.sequential.Aggrnyl, 'Agsunset': pc.sequential.Agsunset,
            'Rainbow': pc.sequential.Rainbow, 'Burg': pc.sequential.Burg, 'Cividis': pc.sequential.Cividis,
            'Inferno': pc.sequential.Inferno, 'Magma': pc.sequential.Magma, 'Plasma': pc.sequential.Plasma,
        }     
        fig_agg_color_palettes = [
            # 'Plotly3', 'Viridis', 'Cividis', 'Inferno', 'Turbo', 'Aggrnyl', 'Agsunset'  
            'Blues', 'Greens', 'Reds', 'Oranges', 'Purples', 'Mint', 'Aggrnyl', 'Agsunset'  
            ]
        plot_mapline_specs = self.visual_sett.plot_ind_mapline_prodHOY_EGIDrfcombo_specs

        checkpoint_to_logfile('plot_ind_mapline_prodHOY_EGIDrfcombo', self.visual_sett.log_name)

        fig_econfunc_comp_scen = go.Figure()
        fig_instpwr_jsctr = go.Figure()

        for i_scen, scen in enumerate(self.visual_sett.pvalloc_scen_list):
            self.get_pvalloc_sett_output(pvalloc_scen_name = scen)
            if not self._set_mc_data_path_or_skip(scen):
                continue

            TECspec_self_consumption_ifapplicable = self.pvalloc_scen.TECspec_self_consumption_ifapplicable

            # import
            topo = json.load(open(f'{self.visual_sett.mc_data_path}/topo_egid.json', 'r'))
            gridprem_ts = pl.read_parquet(f'{self.visual_sett.mc_data_path}/gridprem_ts.parquet')    
            topo_subdf_paths    = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/topo_time_subdf/topo_subdf_*.parquet')
            outtopo_subdf_paths = glob.glob(f'{self.visual_sett.data_path}/pvalloc/{scen}/outtopo_time_subdf/outtopo_subdf_*.parquet') 



            # solkat_gdf = gpd.read_file(f'{self.visual_sett.data_path}/preprep/{self.pvalloc_scen.name_dir_import}/solkat_gdf.geojson')
            solkat_gdf = gpd.read_file(f'{self.visual_sett.data_path}/pvalloc/{self.pvalloc_scen.name_dir_export}/topo_spatial_data/solkat_gdf_in_topo.geojson')


            # select EGIDs for analysis ---------------
            if True:    
                egid_list, dfuid_list, info_source_list, TotalPower_list, inst_TF_list, grid_node_list, share_pvprod_used_list, dfuidPower_list = [], [], [], [], [], [], [], []
                for k,v in topo.items():
                    if v['pv_inst']['inst_TF']:
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
                # asdf = Map_pvinfo_topo_egid.filter(pl.col('EGID') == egid)

                # select EGIDs for visualization
                selected_egid = []
                if True: 
                    selected_in_sample = [egid for egid in plot_mapline_specs['specific_selected_EGIDs'] if egid in topo.keys()]
                    selected_in_sample_after_exclude = [egid for egid in selected_in_sample if egid not in plot_mapline_specs['specific_exclude_EGIDs']]
                    selected_egid = selected_egid + selected_in_sample_after_exclude

                    # add random EGIDs
                    npartion_minmax = plot_mapline_specs['n_partition_pegid_minmax']
                    egid_counts = Map_pvinfo_topo_egid.group_by("EGID").agg(
                        pl.col("df_uid").n_unique().alias("df_uid_count")
                    )
                    Map_pvinfo_topo_egid = Map_pvinfo_topo_egid.join(egid_counts, on="EGID", how="left")
                    def sample_egids_of_df(df, n, seed = None):
                        unique_egids = df['EGID'].unique().to_list()
                        if seed is not None:
                            random.seed(seed)
                        if n < len(unique_egids):
                            sample_egids = random.sample(unique_egids, n)
                        else:
                            sample_egids = unique_egids

                        return df.filter(pl.col('EGID').is_in(sample_egids))

                        # if seed is None: 
                        #     # sample_egids = unique_egids.sample(n=n, with_replacement=False).clone()
                        # else: 
                        #     # sample_egids = unique_egids.sample(n=n, with_replacement=False, seed=seed).clone()

                    egid_winst_pvdf_df = Map_pvinfo_topo_egid.filter(
                        (pl.col('inst_TF') == True) & 
                        (pl.col('info_source') == 'pv_df') & 
                        (pl.col('df_uid_count') >= npartion_minmax[0]) &
                        (pl.col('df_uid_count') <= npartion_minmax[1])
                    )
                    # make selection for pv_df case dynamic to avoid unnecessary code break
                    max_n_dfuid = 10
                    dfuid_iter= 0
                    while  (len(egid_winst_pvdf_df['EGID'].unique().to_list()) < plot_mapline_specs['n_rndm_egid_winst_pvdf']) & (dfuid_iter < max_n_dfuid)  :
                        dfuid_iter += 1
                        egid_winst_pvdf_df = Map_pvinfo_topo_egid.filter(
                            (pl.col('inst_TF') == True) & 
                            (pl.col('info_source') == 'pv_df') &
                            (pl.col('df_uid_count') >= 1) &
                            (pl.col('df_uid_count') <= dfuid_iter)
                        )
                    egid_winst_pvdf_df = sample_egids_of_df(egid_winst_pvdf_df, plot_mapline_specs['n_rndm_egid_winst_pvdf'], plot_mapline_specs['rndm_sample_seed'])
                    egid_winst_pvdf = egid_winst_pvdf_df['EGID'].to_list()


                    egid_winst_alloc_df = Map_pvinfo_topo_egid.filter(
                        (pl.col('inst_TF') == True) & 
                        (pl.col('info_source') == 'alloc_algorithm') &
                        (pl.col('df_uid_count') >= npartion_minmax[0]) &
                        (pl.col('df_uid_count') <= npartion_minmax[1])
                    )
                    # make selection for alloc_algo case dynamic to avoid unnecessary code break
                    max_n_dfuid = 10
                    dfuid_iter= 0
                    while  (len(egid_winst_alloc_df['EGID'].unique().to_list()) < plot_mapline_specs['n_rndm_egid_winst_alloc']) & (dfuid_iter < max_n_dfuid)  :
                        dfuid_iter += 1
                        egid_winst_alloc_df = Map_pvinfo_topo_egid.filter(
                            (pl.col('inst_TF') == True) & 
                            (pl.col('info_source') == 'alloc_algorithm') &
                            (pl.col('df_uid_count') >= 1) &
                            (pl.col('df_uid_count') <= dfuid_iter)
                        )
                    egid_winst_alloc_df = sample_egids_of_df(egid_winst_alloc_df, plot_mapline_specs['n_rndm_egid_winst_alloc'], plot_mapline_specs['rndm_sample_seed'])
                    egid_winst_alloc = egid_winst_alloc_df['EGID'].to_list()


                    egid_woinst_df = Map_pvinfo_topo_egid.filter(
                        (pl.col('inst_TF') == False) & 
                        (pl.col('df_uid_count') >= npartion_minmax[0]) &
                        (pl.col('df_uid_count') <= npartion_minmax[1])
                    )
                    if plot_mapline_specs['rndm_sample_seed'] == None:
                        egid_woinst_df = egid_woinst_df.sample(n=plot_mapline_specs['n_rndm_egid_woinst'], with_replacement=False,).clone()
                    else:
                        egid_woinst_df = egid_woinst_df.sample(n=plot_mapline_specs['n_rndm_egid_woinst'], with_replacement=False, seed = plot_mapline_specs['rndm_sample_seed']).clone()
                    egid_woinst = egid_woinst_df['EGID'].to_list()

                    outtopo_egid_list = []
                    for i_path, path in enumerate(outtopo_subdf_paths):
                        outtopo_subdf = pl.read_parquet(path)
                        unique_egids_in_subdf = outtopo_subdf.select(pl.col("EGID").unique()).to_series().to_list()
                        outtopo_egid_list = outtopo_egid_list + unique_egids_in_subdf
                    # egid_outsample = outtopo_egid_list.sample(n=plot_mapline_specs['n_rndm_egid_outsample'], with_replacement=False,).copy()
                    egid_outsample = pl.Series(outtopo_egid_list).sample(n=min(len(outtopo_egid_list), plot_mapline_specs['n_rndm_egid_outsample']), with_replacement=False).to_list()


                    selected_egid = list(set(selected_egid + egid_winst_pvdf + egid_winst_alloc + egid_woinst + egid_outsample))


            # build topo_partitions_df --------------
            egid_list, dfuid_list, flaeche_list, stromertrag_list, ausrichtung_list, neigung_list, mstrahlung_list = [], [], [], [], [], [], []
            garea_list, are_typ_list, sfhmfh_typ_list, info_source_list = [], [], [], []
            for k,v in topo.items():
                if k in selected_egid:
                    for dfuid, partition in v['solkat_partitions'].items():
                        egid_list.append(k)
                        dfuid_list.append(dfuid)
                        flaeche_list.append(partition['FLAECHE'])
                        stromertrag_list.append(partition['STROMERTRAG'])
                        ausrichtung_list.append(partition['AUSRICHTUNG'])
                        neigung_list.append(partition['NEIGUNG'])
                        mstrahlung_list.append(partition['MSTRAHLUNG'])

                        garea_list.append(v['gwr_info']['garea'])
                        are_typ_list.append(v['gwr_info']['are_typ'])
                        sfhmfh_typ_list.append(v['gwr_info']['sfhmfh_typ'])

                        info_source_list.append(v['pv_inst']['info_source'])

            topo_partitions_df = pd.DataFrame({
                'EGID': egid_list, 
                'df_uid': dfuid_list,
                'FLAECHE': flaeche_list,
                'STROMERTRAG': stromertrag_list,
                'AUSRICHTUNG': ausrichtung_list,
                'NEIGUNG': neigung_list,
                'MSTRAHLUNG': mstrahlung_list, 
                'GAREA': garea_list, 
                'are_typ': are_typ_list, 
                'sfhmfh_typ': sfhmfh_typ_list,
                'info_source': info_source_list,
            })

            topo_partitions_gdf = topo_partitions_df.copy()
            topo_partitions_gdf = topo_partitions_gdf.merge(solkat_gdf[['geometry', 'df_uid']], on='df_uid', how='left')
            topo_partitions_gdf = gpd.GeoDataFrame(topo_partitions_gdf, crs='EPSG:2056', geometry='geometry')

            topo_partitions_gdf['hover_text'] = topo_partitions_gdf.apply(lambda row: f"EGID: {row['EGID']}<br>df_uid: {row['df_uid']}<br>FLAECHE: {row['FLAECHE']}<br>STROMERTRAG: {row['STROMERTRAG']}<br>AUSRICHTUNG: {row['AUSRICHTUNG']}<br>NEIGUNG: {row['NEIGUNG']}<br>MSTRAHLUNG: {row['MSTRAHLUNG']}", axis=1)
            topo_partitions_gdf = topo_partitions_gdf.to_crs('EPSG:4326')
            topo_partitions_gdf['geometry'] = topo_partitions_gdf['geometry'].apply(self.flatten_geometry)
            topo_partitions_gdf['roofpartition_color'] = topo_partitions_gdf['EGID'].apply(lambda x: plot_mapline_specs['roofpartition_color'] if x in selected_egid else 'rgba(0,0,0,0)')


            # loop interest rates  ---------------
            # for i_recalc_interest_rate, recalc_interest_rate in enumerate(plot_mapline_specs['recalc_interest_rate_list']):
            #     print(f'i_recalc_interest_rate: {i_recalc_interest_rate}, recalc_interest_rate: {recalc_interest_rate}')

            #     generic_pvinstcost_coeff_total = plot_mapline_specs['generic_pvinstcost_coeff_total']
            #     if not generic_pvinstcost_coeff_total['use_generic_instcost_coeff_total_TF']:
            #         generic_pvinstcost_coeff_total['coeff_total_dict_list'] = [{'a': 1, 'b': 1, 'c': 1}]

            #     for i_cost_coeff, cost_coeff_dict in enumerate(generic_pvinstcost_coeff_total['coeff_total_dict_list']): 
            #         egid_jsctr_list, TotalPower_jsctr_list, RecalcPowe_jsctr_list, sum_FLAECHE_list, max_optFLAECHE_list, interest_rate_list, cost_coeff_list = [], [], [], [], [], [], []


            #         egid = '190296588'
            #         fig_econfunc_comp = go.Figure()
            #         for i_egid, egid in enumerate(selected_egid):  # ['EGID'].unique():


            for i_egid, egid in enumerate(selected_egid):  # ['EGID'].unique():

                for i_recalc_interest_rate, recalc_interest_rate in enumerate(plot_mapline_specs['recalc_interest_rate_list']):
                    print(f'i_recalc_interest_rate: {i_recalc_interest_rate}, recalc_interest_rate: {recalc_interest_rate}')

                    generic_pvinstcost_coeff_total = plot_mapline_specs['generic_pvinstcost_coeff_total']
                    if not generic_pvinstcost_coeff_total['use_generic_instcost_coeff_total_TF']:
                        generic_pvinstcost_coeff_total['coeff_total_dict_list'] = [{'a': 1, 'b': 1, 'c': 1}]

                    fig_econfunc_comp = go.Figure()      
                    for i_cost_coeff, cost_coeff_dict in enumerate(generic_pvinstcost_coeff_total['coeff_total_dict_list']): 
                        egid_jsctr_list, TotalPower_jsctr_list, RecalcPowe_jsctr_list, sum_FLAECHE_list, max_optFLAECHE_list, interest_rate_list, cost_coeff_list = [], [], [], [], [], [], []


                        if (True) & (i_scen < 1) & (i_egid < 2 ):
                            checkpoint_print_TF = True
                        else: 
                            checkpoint_print_TF = False

                        # run non-recalculation part only once ==============================
                        if (i_recalc_interest_rate == 0) & (i_cost_coeff == 0): 


                            # traces partition map ==============================                        
                            fig_rfpart_map = go.Figure()
                            if 'map' in plot_mapline_specs['run_selected_plot_parts']:

                                if egid in list(topo_partitions_gdf['EGID'].unique()):

                                    sub_partitions_gdf = topo_partitions_gdf.loc[topo_partitions_gdf['EGID'] == egid].copy()

                                    center_lat = sub_partitions_gdf.geometry.union_all().centroid.y    # center_lat = sub_partitions_gdf.geometry.unary_union.centroid.y
                                    center_lon = sub_partitions_gdf.geometry.union_all().centroid.x    # center_lon = sub_partitions_gdf.geometry.unary_union.centroid.x

                                    # Add each polygon as a separate trace
                                    for idx, row in sub_partitions_gdf.iterrows():

                                        # Extract coordinates
                                        if row.geometry.geom_type == 'MultiPolygon':
                                            # Handle multiple polygons
                                            for poly in row.geometry.geoms:
                                                x, y = poly.exterior.xy
                                                fig_rfpart_map.add_trace(go.Scattermapbox(
                                                    fill="toself",
                                                    fillcolor=plot_mapline_specs['roofpartition_color'],
                                                    mode="lines",
                                                    line=dict(width=1, color="black"),
                                                    lat=y.tolist(),
                                                    lon=x.tolist(),
                                                    name=f"EGID: {row['EGID']} - df_uid: {row['df_uid']}",
                                                    text=row['hover_text'],
                                                    hoverinfo="text", 

                                                ))
                                        else:
                                            # Handle single polygon
                                            x, y = row.geometry.exterior.xy
                                            fig_rfpart_map.add_trace(go.Scattermapbox(
                                                fill="toself",
                                                fillcolor=plot_mapline_specs['roofpartition_color'],
                                                mode="lines",
                                                line=dict(width=1, color="black"),
                                                lat=y.tolist(),
                                                lon=x.tolist(),
                                                name=f"EGID: {row['EGID']} - df_uid: {row['df_uid']}",
                                                text=row['hover_text'],
                                                hoverinfo="text"
                                            ))
                            checkpoint_to_logfile(f'--> i_scen: {i_scen}, i_egid: {i_egid};/{len(selected_egid)} end fig_rfpart_map', self.visual_sett.log_name) if checkpoint_print_TF else None



                            # traces partition time series ==============================
                            fig_rfpart_ts = go.Figure()
                            rfpart_ts_list = []
                            summary_agg_egids_list = []

                            # all possible combinations ------------------------------
                            if 'all_combo_TS' in plot_mapline_specs['run_selected_plot_parts']:

                                # create list of df_uid combinations
                                df_uid_list = topo_partitions_df.loc[topo_partitions_df['EGID'] == egid, 'df_uid'].to_list()
                                combo_list = []
                                for r in range(1,len(df_uid_list)+1):
                                    for combo in itertools.combinations(df_uid_list,r):
                                        combo_list.append(combo)

                                idx = 0
                                i_path, path = idx, topo_subdf_paths[idx]
                                # extract value for df_uid combinations from topo_time_subdf
                                for df_uid_combo in combo_list:

                                    for i_path, path in enumerate(topo_subdf_paths):
                                        subdf = pl.read_parquet(path)

                                        if egid in subdf['EGID'].to_list():

                                            # extend subdf with static data (safe disk space) --------------------
                                            subdf_static = subdf['EGID'].unique().to_list()
                                            subdf_static_list = []
                                            for egid in subdf_static:
                                                static_topo = topo[egid]

                                                for k,v in static_topo['solkat_partitions'].items():
                                                    egid_dfuid_row = {
                                                        'EGID':               egid,
                                                        'df_uid':             k,
                                                        'bfs':                static_topo['gwr_info']['bfs'], 
                                                        'GKLAS':              static_topo['gwr_info']['gklas'],
                                                        'GAREA':              static_topo['gwr_info']['garea'],
                                                        'GBAUJ':              static_topo['gwr_info']['gbauj'],
                                                        'GSTAT':              static_topo['gwr_info']['gstat'],
                                                        'GWAERZH1':           static_topo['gwr_info']['gwaerzh1'],
                                                        'GENH1':              static_topo['gwr_info']['genh1'],
                                                        'sfhmfh_typ':         static_topo['gwr_info']['sfhmfh_typ'],
                                                        'demand_arch_typ':    static_topo['demand_arch_typ'],
                                                        'demand_elec_pGAREA': static_topo['demand_elec_pGAREA'],
                                                        'grid_node':          static_topo['grid_node'],
                                                        'pvtarif_Rp_kWh':     static_topo['pvtarif_Rp_kWh'],
                                                        'elecpri_Rp_kWh':     static_topo['elecpri_Rp_kWh'],
                                                        'inst_TF':            static_topo['pv_inst']['inst_TF'],
                                                        'info_source':        static_topo['pv_inst']['info_source'],
                                                        'pvid':               static_topo['pv_inst']['xtf_id'],
                                                        'TotalPower':         static_topo['pv_inst']['TotalPower'],
                                                        'FLAECHE':            v['FLAECHE'],
                                                        'AUSRICHTUNG':        v['AUSRICHTUNG'],
                                                        'STROMERTRAG':        v['STROMERTRAG'],
                                                        'NEIGUNG':            v['NEIGUNG'],
                                                        'MSTRAHLUNG':         v['MSTRAHLUNG'],
                                                        'GSTRAHLUNG':         v['GSTRAHLUNG'],
                                                    }
                                                    subdf_static_list.append(egid_dfuid_row)

                                            subdf_static_df = pl.DataFrame(subdf_static_list)
                                            subdf = subdf.join(subdf_static_df, on=['EGID', 'df_uid'], how='left')


                                            # topo_time_subdf aggregation
                                            subdf_combo = subdf.filter(
                                                (pl.col('EGID') == egid) & 
                                                (pl.col('df_uid').is_in(df_uid_combo))
                                                )
                                            # set inst_TF to True to select all partitions in combo
                                            subdf_combo = subdf_combo.with_columns([
                                                # pl.col('inst_TF')
                                                pl.lit(True).alias('inst_TF'), 
                                                pl.lit(1.0).alias('share_pvprod_used'),
                                            ])        
                                            subdf_combo = subdf_combo.with_columns([
                                                (pl.col('poss_pvprod_kW') * pl.col('share_pvprod_used')).alias('pvprod_kW'),
                                                ])

                                            # sorting necessary so that .first() statement captures inst_TF and info_source for EGIDS with partial installations
                                            subdf_combo = subdf_combo.sort(['EGID','inst_TF', 'df_uid', 't_int'], descending=[False, True, False, False])

                                            # agg per EGID to apply selfconsumption 
                                            agg_egids = subdf_combo.group_by(['EGID', 't', 't_int']).agg([
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

                                            agg_egids = agg_egids.with_columns([
                                                pl.when(pl.col('netfeedin_kW') > 0)
                                                .then(pl.col('selfconsum_kW') / pl.col('pvprod_kW'))
                                                .otherwise(0)  # or any other value that makes sense
                                                .alias('selfconsum_pvprod_ratio')
                                            ])


                                            # topo_partition_df aggregation
                                            topo_part_egid = topo_partitions_df.loc[(topo_partitions_df['EGID'] == egid) & 
                                                                                    (topo_partitions_df['df_uid'].isin(df_uid_combo))]
                                            topo_agg_egid = topo_part_egid.groupby('EGID').agg({
                                                'FLAECHE': 'sum',
                                                'STROMERTRAG': 'sum',
                                                'NEIGUNG': 'mean', 
                                                'AUSRICHTUNG': 'mean',
                                                'GAREA': 'first',
                                                'are_typ': 'first',
                                                'sfhmfh_typ': 'first',
                                                'info_source': 'first',
                                            })

                                            # add fig traces
                                            df_uid_combo_str = '-'.join(df_uid_combo) if len(df_uid_combo) >1 else df_uid_combo[0]
                                            legend_str1 = f"---- Possible PV p_combo - EGID: {egid}, df_uid_combo: {df_uid_combo_str} ------------"
                                            legend_str2 = f"     FLAECHE: {round(topo_agg_egid['FLAECHE'].values[0],3)}, STROMERTRAG: {round(topo_agg_egid['STROMERTRAG'].values[0],3)}, AUSRICHTUNG: {round(topo_agg_egid['AUSRICHTUNG'].values[0],3)}" 
                                            legend_str3 = f"     NEIGUNG: {topo_agg_egid['NEIGUNG'].values[0]}, GAREA: {topo_agg_egid['GAREA'].values[0]}, are_typ: {topo_agg_egid['are_typ'].values[0]}, sfhmfh_typ: {topo_agg_egid['sfhmfh_typ'].values[0]}"
                                            legend_str4 = f"     info_source: {topo_agg_egid['info_source'].values[0]}, max_kWp: {round(topo_agg_egid['FLAECHE'].values[0] * self.pvalloc_scen.TECspec_kWpeak_per_m2,3) }"

                                            legend_list = [legend_str1, legend_str2, legend_str3, legend_str4, ]

                                            if 'full_partition_combo' in  plot_mapline_specs['traces_in_timeseries_plot']:
                                                for legend_str in legend_list:
                                                    fig_rfpart_ts.add_trace(go.Scatter(x= [None, ], y = [None,] , mode = 'lines', 
                                                                                name = legend_str, 
                                                                                line = dict(color=plot_mapline_specs['roofpartition_color'], width=1.5),
                                                                                opacity = 0,
                                                                            ))

                                                for col in ['demand_kW', 'radiation', 'pvprod_kW', 'selfconsum_kW', 'netfeedin_kW', 'netdemand_kW', 'selfconsum_pvprod_ratio']:
                                                    if col == 'selfconsum_pvprod_ratio':
                                                        agg_value = agg_egids[col].mean()
                                                        agg_method = 'mean'
                                                    else:
                                                        agg_value = sum(agg_egids[col])
                                                        agg_method = 'sum'

                                                    if agg_value < 2: 
                                                        agg_round_value = round(agg_value,5)
                                                    else:
                                                        agg_round_value = round(agg_value,2)

                                                    # reduce t_hours to have more "handable - plots" in plotly
                                                    if self.visual_sett.cut_timeseries_to_zoom_hour: 
                                                        agg_egids = agg_egids.filter(
                                                            (pl.col('t_int') >= self.visual_sett.default_zoom_hour[0]-(24*7)) &
                                                            (pl.col('t_int') <= self.visual_sett.default_zoom_hour[1]+(24*7))
                                                        ).clone()

                                                    t_int_array_daynight_bands = copy.deepcopy(agg_egids['t_int'])

                                                    fig_rfpart_ts.add_trace(go.Scatter(
                                                        x=agg_egids['t_int'], 
                                                        y=agg_egids[col],
                                                        mode='lines',
                                                        name=f"{col:<25} - {df_uid_combo_str:<20}, {agg_method}: {agg_round_value}",
                                                        line=dict(width=1.5),
                                                        # hovertemplate=f"EGID: {egid}<br>df_uid_combo: {df_uid_combo_str}<br>FLAECHE: {topo_agg_egid['FLAECHE'].values[0]:3}, STROMERTRAG: {topo_agg_egid['STROMERTRAG'].values[0]:3}, AUSRICHTUNG: {topo_agg_egid['AUSRICHTUNG'].values[0]:3}<br> {col}: %{{y:.2f}}<extra></extra>"
                                                    ))


                            checkpoint_to_logfile(f'--> i_scen: {i_scen}, i_egid: {i_egid};/{len(selected_egid)} end calc all poss combinations', self.visual_sett.log_name) if checkpoint_print_TF else None


                            # actual installation time series ------------------------------
                            if 'actual_inst_TS' in plot_mapline_specs['run_selected_plot_parts']:
                                egid_list, dfuid_list, info_source_list, TotalPower_list, inst_TF_list, grid_node_list, share_pvprod_used_list, dfuidPower_list = [], [], [], [], [], [], [], []
                                for k,v in topo.items():
                                    dfuid_tupls = [tpl[1] for tpl in v['pv_inst']['dfuid_w_inst_tuples'] if tpl[3] > 0.0]
                                    for k_s, v_s in v['solkat_partitions'].items():
                                        if k_s in dfuid_tupls:
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
                                # gridnode_freq         = pd.DataFrame({'EGID': egid_list, 'df_uid': dfuid_list, 'inst_TF': inst_TF_list, 'info_source': info_source_list,  'grid_node': grid_node_list})


                                agg_subdf_df_list, agg_egids_list = [], []
                                for i_path, path in enumerate(topo_subdf_paths):

                                    subdf = pl.read_parquet(path)

                                    if egid in subdf['EGID'].to_list():

                                        # extend subdf with static data (safe disk space) --------------------
                                        subdf_static = subdf['EGID'].unique().to_list()
                                        subdf_static_list = []
                                        for egid in subdf_static:
                                            static_topo = topo[egid]

                                            for k,v in static_topo['solkat_partitions'].items():
                                                egid_dfuid_row = {
                                                    'EGID':               egid,
                                                    'df_uid':             k,
                                                    'bfs':                static_topo['gwr_info']['bfs'], 
                                                    'GKLAS':              static_topo['gwr_info']['gklas'],
                                                    'GAREA':              static_topo['gwr_info']['garea'],
                                                    'GBAUJ':              static_topo['gwr_info']['gbauj'],
                                                    'GSTAT':              static_topo['gwr_info']['gstat'],
                                                    'GWAERZH1':           static_topo['gwr_info']['gwaerzh1'],
                                                    'GENH1':              static_topo['gwr_info']['genh1'],
                                                    'sfhmfh_typ':         static_topo['gwr_info']['sfhmfh_typ'],
                                                    'demand_arch_typ':    static_topo['demand_arch_typ'],
                                                    'demand_elec_pGAREA': static_topo['demand_elec_pGAREA'],
                                                    'grid_node':          static_topo['grid_node'],
                                                    'pvtarif_Rp_kWh':     static_topo['pvtarif_Rp_kWh'],
                                                    'elecpri_Rp_kWh':     static_topo['elecpri_Rp_kWh'],
                                                    'inst_TF':            static_topo['pv_inst']['inst_TF'],
                                                    'info_source':        static_topo['pv_inst']['info_source'],
                                                    'pvid':               static_topo['pv_inst']['xtf_id'],
                                                    'TotalPower':         static_topo['pv_inst']['TotalPower'],
                                                    'FLAECHE':            v['FLAECHE'],
                                                    'AUSRICHTUNG':        v['AUSRICHTUNG'],
                                                    'STROMERTRAG':        v['STROMERTRAG'],
                                                    'NEIGUNG':            v['NEIGUNG'],
                                                    'MSTRAHLUNG':         v['MSTRAHLUNG'],
                                                    'GSTRAHLUNG':         v['GSTRAHLUNG'],
                                                }
                                                subdf_static_list.append(egid_dfuid_row)

                                        subdf_static_df = pl.DataFrame(subdf_static_list)
                                        subdf = subdf.join(subdf_static_df, on=['EGID', 'df_uid'], how='left')

                                        subdf = subdf.join(gridprem_ts[['t', 'grid_node', 'prem_Rp_kWh']], on=['t', 'grid_node'], how='left')  

                                        subdf_updated = subdf.filter(pl.col('EGID') == egid).clone()


                                        cols_to_update = [ 'info_source', 'inst_TF', 'share_pvprod_used', 'dfuidPower' ]                                      
                                        subdf_updated = subdf_updated.drop(cols_to_update)                      

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
                                            # so that pvprod_kW of only installed partitions is counted! will again be off if egid is a house of pvdf with more than 1 partition. 
                                            # -- edit: commented out because is already apply so below?
                                            # pl.when(pl.col('inst_TF') == False)
                                            #     .then(pl.lit(0.0)).otherwise(pl.col('poss_pvprod_kW')).alias('poss_pvprod_kW'),
                                            # pl.when(pl.col('inst_TF') == False)
                                            #     .then(pl.lit(0.0)).otherwise((pl.col('share_pvprod_used') * pl.col('radiation'))).alias('share_radiation_used'),
                                            # pl.when(pl.col('inst_TF') == False)
                                            #     .then(pl.lit(0.0)).otherwise(pl.col('radiation')).alias('radiation'), 
                                        ])

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
                                        subdf_updated = subdf_updated.with_columns([
                                            pl.when(
                                                (pl.col('EGID').is_in(subdf_no_inst['EGID'].implode())) &
                                                (pl.col('df_uid').is_in(subdf_no_inst['df_uid'].implode()))
                                            ).then(pl.lit(0.0)).otherwise(pl.col('pvprod_kW')).alias('pvprod_kW'),
                                        ])
                                        subdf_updated = subdf_updated.with_columns([
                                            pl.when(
                                                (pl.col('EGID').is_in(subdf_no_inst['EGID'].implode())) &
                                                (pl.col('df_uid').is_in(subdf_no_inst['df_uid'].implode()))
                                            ).then(pl.lit(0.0)).otherwise(pl.col('radiation')).alias('radiation'),
                                        ])


                                        # sorting necessary so that .first() statement captures inst_TF and info_source for EGIDS with partial installations
                                        subdf_updated = subdf_updated.sort(['EGID','inst_TF', 'df_uid', 't_int'], descending=[False, True, False, False])

                                        # agg per EGID to apply selfconsumption 
                                        agg_egids = subdf_updated.group_by(['EGID', 't', 't_int']).agg([
                                            pl.col('inst_TF').first().alias('inst_TF'),
                                            pl.col('info_source').first().alias('info_source'),
                                            pl.col('TotalPower').first().alias('TotalPower'),
                                            pl.col('grid_node').first().alias('grid_node'),
                                            pl.col('demand_kW').first().alias('demand_kW'),
                                            pl.col('panel_efficiency').first().alias('panel_efficiency'),
                                            pl.col('pvtarif_Rp_kWh').first().alias('pvtarif_Rp_kWh'),
                                            pl.col('elecpri_Rp_kWh').first().alias('elecpri_Rp_kWh'),
                                            pl.col('prem_Rp_kWh').first().alias('prem_Rp_kWh'),

                                            pl.col('demand_appliances_kW').first().alias('demand_appliances_kW') if self.pvalloc_scen.TECspec_add_heatpump_demand_TF else pl.lit(0.0).alias('demand_appliances_kW'),
                                            pl.col('demand_heatpump_kW').first().alias('demand_heatpump_kW') if self.pvalloc_scen.TECspec_add_heatpump_demand_TF else pl.lit(0.0).alias('demand_heatpump_kW'),

                                            # pl.col('share_radiation_used').sum().alias('share_radiation_used'),
                                            pl.col('poss_pvprod_kW').sum().alias('poss_pvprod_kW'),
                                            pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                                            pl.col('radiation').sum().alias('radiation'),

                                        ])

                                        # calc selfconsumption
                                        agg_egids = agg_egids.sort(['EGID', 't_int'], descending = [False, False])

                                        selfconsum_expr = pl.min_horizontal([pl.col("pvprod_kW"), pl.col("demand_kW")]) * self.pvalloc_scen.TECspec_self_consumption_ifapplicable

                                        agg_egids = agg_egids.with_columns([        
                                            selfconsum_expr.alias("selfconsum_kW"),
                                            (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
                                            (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
                                        ])

                                        agg_egids = agg_egids.with_columns([
                                            pl.when(pl.col('netfeedin_kW') > 0)
                                            .then(pl.col('selfconsum_kW') / pl.col('pvprod_kW'))
                                            .otherwise(0)  # or any other value that makes sense
                                            .alias('selfconsum_pvprod_ratio')
                                        ])



                                        agg_egids_list.append(agg_egids)
                                        summary_agg_egids_list.append(agg_egids)

                                        topo_agg_egids_df = pl.concat(agg_egids_list)


                                        # add fig traces
                                        name_str = f"---- Actual PV: EGID: {egid}, TotalPower: {round(topo_agg_egids_df['TotalPower'][0],3)}, info_source: {topo_agg_egids_df['info_source'][0]} {20*'-'}, "

                                        if 'actual_pv_inst' in  plot_mapline_specs['traces_in_timeseries_plot']:

                                            fig_rfpart_ts.add_trace(go.Scatter(x= [None, ], y = [None,] , mode = 'lines', 
                                                                        name = name_str, 
                                                                        line = dict(color=plot_mapline_specs['roofpartition_color'], width=1.5),
                                                                        # hovertemplate = f"EGID: {egid}<br>df_uid_combo: {df_uid_combo_str}<br>FLAECHE_sum: {topo_agg_egid['FLAECHE'].values[0]}<br>STROMERTRAG_sum: {topo_agg_egid['STROMERTRAG'].values[0]}<br>AUSRICHTUNG_mean: {topo_agg_egid['AUSRICHTUNG'].values[0]}<extra></extra>",
                                                                        opacity = 0,
                                                                    ))

                                        if self.pvalloc_scen.TECspec_add_heatpump_demand_TF:
                                            cols_for_traces = ['demand_kW', 'radiation', 'pvprod_kW', 'selfconsum_kW', 'netfeedin_kW', 'netdemand_kW', 'selfconsum_pvprod_ratio', 
                                                                'demand_appliances_kW', 'demand_heatpump_kW',                                                               
                                                                ]
                                        else: 
                                            cols_for_traces = ['demand_kW', 'radiation', 'pvprod_kW', 'selfconsum_kW', 'netfeedin_kW', 'netdemand_kW', 'selfconsum_pvprod_ratio']

                                        for col in cols_for_traces:
                                            if col == 'selfconsum_pvprod_ratio':
                                                agg_value = agg_egids[col].mean()
                                                agg_method = 'mean'
                                            else:
                                                agg_value = sum(agg_egids[col])
                                                agg_method = 'sum'

                                            if agg_value < 2: 
                                                agg_round_value = round(agg_value,5)
                                            else:
                                                agg_round_value = round(agg_value,2)

                                            # reduce t_hours to have more "handable - plots" in plotly
                                            if self.visual_sett.cut_timeseries_to_zoom_hour: 
                                                agg_egids_plot = agg_egids.filter(
                                                    (pl.col('t_int') >= self.visual_sett.default_zoom_hour[0]-(24*7)) &
                                                    (pl.col('t_int') <= self.visual_sett.default_zoom_hour[1]+(24*7))
                                                ).clone()
                                            else:
                                                agg_egids_plot = agg_egids.clone()

                                            fig_rfpart_ts.add_trace(go.Scatter(
                                                x=agg_egids_plot['t_int'], 
                                                y=agg_egids_plot[col],
                                                mode='lines+markers', 
                                                marker = dict(
                                                    symbol = plot_mapline_specs['actual_ts_trace_marker'], 
                                                    # size = 5,
                                                ),
                                                name=f"{col:<25} - {agg_method}: {agg_round_value}",
                                                line=dict(width=2.5),
                                                opacity = 0.6,
                                            ))


                                info_source_pd = Map_pvinfo_topo_egid.to_pandas()
                                info_source_str = info_source_pd.loc[info_source_pd["EGID"] == egid, "info_source"].iloc[0]
                            checkpoint_to_logfile(f'--> i_scen: {i_scen}, i_egid: {i_egid};/{len(selected_egid)} end calc actual inst', self.visual_sett.log_name) if checkpoint_print_TF else None


                            # outsample time series ------------------------------
                            if 'outsample_TS' in plot_mapline_specs['run_selected_plot_parts']:
                                if egid in outtopo_egid_list:

                                    for i_path, path in enumerate(outtopo_subdf_paths):
                                        outtopo_subdf = pl.read_parquet(path)

                                        if egid in outtopo_subdf['EGID'].to_list():
                                            # filter for egid
                                            outtopo_egid = outtopo_subdf.filter(pl.col('EGID') == egid).clone()

                                            # reduce t_hours to have more "handable - plots" in plotly
                                            if self.visual_sett.cut_timeseries_to_zoom_hour: 
                                                outtopo_egid = outtopo_egid.filter(
                                                    (pl.col('t_int') >= self.visual_sett.default_zoom_hour[0]-(24*7)) &
                                                    (pl.col('t_int') <= self.visual_sett.default_zoom_hour[1]+(24*7))
                                                ).clone()

                                            if 'actual_pv_inst' in plot_mapline_specs['traces_in_timeseries_plot']:

                                                fig_rfpart_ts.add_trace(go.Scatter(
                                                    x = outtopo_egid['t_int'],
                                                    y = outtopo_egid['demand_proxy_out_kW'],
                                                    mode = 'lines',
                                                    name = 'demand_proxy_out_kW - outsample EGIDs',
                                                    line = dict(width = 1.5)
                                                    ))
                            checkpoint_to_logfile(f'--> i_scen: {i_scen}, i_egid: {i_egid};/{len(selected_egid)} end outsample TS', self.visual_sett.log_name) if checkpoint_print_TF else None



                        # traces recalc kWp pvdf ==============================
                        fig_econfunc =  go.Figure()
                        if 'recalc_opt_inst' in plot_mapline_specs['run_selected_plot_parts']:    
                            # all necessary functions from pvalloc
                            if True:

                                def initial_sml_get_instcost_interpolate_function(
                                            use_generic_coeff_pkW_TF= False, 
                                            use_generic_coeff_total_TF = False, 
                                            generic_coeff_pkW = [1, 1, 1, 1],
                                            generic_coeff_total = [1, 1, 1, ]
                                    ): 
                                    # setup --------
                                    # print_to_logfile('run function: get_estim_instcost_function', self.pvalloc_scen.log_name) if i_m < 5 else None

                                    with open(f'{self.visual_sett.data_path}/pvalloc/{scen}/pvinstcost_coefficients.json', 'r') as file:
                                        pvinstcost_coefficients = json.load(file)

                                    if not use_generic_coeff_pkW_TF:
                                        params_pkW = pvinstcost_coefficients['params_pkW']
                                    else: 
                                        params_pkW = generic_coeff_pkW

                                    if not use_generic_coeff_total_TF:                                        
                                        # coefs_total = pvinstcost_coefficients['coefs_total']
                                        params_total = pvinstcost_coefficients['params_total']
                                    else: 
                                        params_total = generic_coeff_total

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

                                    return estim_instcost_chfpkW, estim_instcost_chftotal, params_pkW, params_total        


                                def calculate_npv(flaeche, max_dfuid_df, estim_instcost_chftotal,  npv_calc_interest_rate = None, 
                                                #   tweak_denominator=1.0,
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

                                    if self.pvalloc_scen.TECspec_pvprod_calc_method == 'method2.2':
                                        # Calculate production with the given FLAECHE
                                        df = df.with_columns([
                                            ((pl.col("radiation") / 1000) * 
                                            pl.col("panel_efficiency") * 
                                            self.pvalloc_scen.TECspec_inverter_efficiency * 
                                            self.pvalloc_scen.TECspec_share_roof_area_available * 
                                            flaeche).alias("pvprod_kW")
                                        ])

                                        # calc selfconsumption
                                        selfconsum_expr = pl.min_horizontal([ pl.col("pvprod_kW"), pl.col("demand_kW") ]) * self.pvalloc_scen.TECspec_self_consumption_ifapplicable

                                        df = df.with_columns([  
                                            selfconsum_expr.alias("selfconsum_kW"),
                                            (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
                                            (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
                                            ])


                                        # df = df.with_columns([
                                        #     (pl.col("pvtarif_Rp_kWh") / tweak_denominator).alias("pvtarif_Rp_kWh"),
                                        # ])
                                        # calc economic income and spending
                                        if not getattr(self.pvalloc_scen, 'ALGOspec_tweak_npv_excl_elec_demand', True):

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
                                        kWp = flaeche * self.pvalloc_scen.TECspec_kWpeak_per_m2 * self.pvalloc_scen.TECspec_share_roof_area_available
                                        installation_cost = estim_instcost_chftotal(kWp)

                                        # calc NPV
                                        if npv_calc_interest_rate is None:
                                            discount_factor = np.array([(1 + self.pvalloc_scen.TECspec_interest_rate)**-i for i in range(1, self.pvalloc_scen.TECspec_invst_maturity + 1)])
                                        else:
                                            discount_factor = np.array([(1 + npv_calc_interest_rate)**-i for i in range(1, self.pvalloc_scen.TECspec_invst_maturity + 1)])
                                        disc_cashflow = annual_cashflow * np.sum(discount_factor)
                                        npv = -installation_cost + disc_cashflow

                                        # calc breakeven
                                        cum_disc_cashflow = -installation_cost
                                        breakeven_year = None
                                        for yr in range(1, self.pvalloc_scen.TECspec_invst_maturity * 2 + 1):
                                            disc_cashflow_year = annual_cashflow / (1 + self.pvalloc_scen.TECspec_interest_rate) ** yr
                                            cum_disc_cashflow += disc_cashflow_year

                                            if cum_disc_cashflow > 0 and breakeven_year is None:
                                                breakeven_year = yr


                                        pvprod_kW_sum = df['pvprod_kW'].sum()
                                        demand_kW_sum = df['demand_kW'].sum()
                                        selfconsum_kW_sum= df['selfconsum_kW'].sum()
                                        rest = (installation_cost, disc_cashflow, pvprod_kW_sum, demand_kW_sum, selfconsum_kW_sum, breakeven_year)

                                        # return npv, installation_cost, disc_cashflow, pvprod_kW_sum, demand_kW_sum, selfconsum_kW_sum
                                        return npv, rest


                                def optimize_pv_size(max_dfuid_df, estim_instcost_chftotal, max_flaeche_factor=None, optim_calc_interest_rate = None):
                                    """
                                    Find the optimal PV installation size (FLAECHE) that maximizes NPV

                                    """
                                    def obj_func(flaeche):
                                        npv, rest = calculate_npv(flaeche, max_dfuid_df, estim_instcost_chftotal, optim_calc_interest_rate)
                                        return -npv  


                                    # Set bounds - minimum FLAECHE is 0, maximum is either specified or from the data
                                    if max_flaeche_factor is None:
                                        max_flaeche = max(max_dfuid_df['FLAECHE']) * self.pvalloc_scen.TECspec_opt_max_flaeche_factor
                                    else:
                                        max_flaeche = max(max_dfuid_df['FLAECHE']) * max_flaeche_factor



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



                            # optimized installation size                                   
                            if True:
                                pvdf_topo_egid = Map_pvinfo_topo_egid.filter(pl.col('info_source') == 'pv_df').clone()
                                # if egid in pvdf_topo_egid['EGID'].to_list():

                                for i_path, path in enumerate(topo_subdf_paths):
                                    subdf = pl.read_parquet(path)

                                    if egid in subdf['EGID'].to_list():

                                        # extend subdf with static data (safe disk space) --------------------
                                        subdf_static = subdf['EGID'].unique().to_list()
                                        subdf_static_list = []
                                        for egid in subdf_static:
                                            static_topo = topo[egid]

                                            for k,v in static_topo['solkat_partitions'].items():
                                                egid_dfuid_row = {
                                                    'EGID':               egid,
                                                    'df_uid':             k,
                                                    'bfs':                static_topo['gwr_info']['bfs'], 
                                                    'GKLAS':              static_topo['gwr_info']['gklas'],
                                                    'GAREA':              static_topo['gwr_info']['garea'],
                                                    'GBAUJ':              static_topo['gwr_info']['gbauj'],
                                                    'GSTAT':              static_topo['gwr_info']['gstat'],
                                                    'GWAERZH1':           static_topo['gwr_info']['gwaerzh1'],
                                                    'GENH1':              static_topo['gwr_info']['genh1'],
                                                    'sfhmfh_typ':         static_topo['gwr_info']['sfhmfh_typ'],
                                                    'demand_arch_typ':    static_topo['demand_arch_typ'],
                                                    'demand_elec_pGAREA': static_topo['demand_elec_pGAREA'],
                                                    'grid_node':          static_topo['grid_node'],
                                                    'pvtarif_Rp_kWh':     static_topo['pvtarif_Rp_kWh'],
                                                    'elecpri_Rp_kWh':     static_topo['elecpri_Rp_kWh'],
                                                    'inst_TF':            static_topo['pv_inst']['inst_TF'],
                                                    'info_source':        static_topo['pv_inst']['info_source'],
                                                    'pvid':               static_topo['pv_inst']['xtf_id'],
                                                    'TotalPower':         static_topo['pv_inst']['TotalPower'],
                                                    'FLAECHE':            v['FLAECHE'],
                                                    'AUSRICHTUNG':        v['AUSRICHTUNG'],
                                                    'STROMERTRAG':        v['STROMERTRAG'],
                                                    'NEIGUNG':            v['NEIGUNG'],
                                                    'MSTRAHLUNG':         v['MSTRAHLUNG'],
                                                    'GSTRAHLUNG':         v['GSTRAHLUNG'],
                                                }
                                                subdf_static_list.append(egid_dfuid_row)

                                        subdf_static_df = pl.DataFrame(subdf_static_list)
                                        subdf = subdf.join(subdf_static_df, on=['EGID', 'df_uid'], how='left')


                                        # feature: adjust pvtarif and elecpri to explore effects
                                        if plot_mapline_specs['tryout_generic_pvtariff_elecpri_Rp_kWh'][0] is not None: 
                                            subdf = subdf.with_columns([
                                                pl.lit(plot_mapline_specs['tryout_generic_pvtariff_elecpri_Rp_kWh'][0]).alias('pvtarif_Rp_kWh'),
                                            ])
                                        if plot_mapline_specs['tryout_generic_pvtariff_elecpri_Rp_kWh'][1] is not None: 
                                            subdf = subdf.with_columns([
                                                pl.lit(plot_mapline_specs['tryout_generic_pvtariff_elecpri_Rp_kWh'][1]).alias('elecpri_Rp_kWh'),
                                            ])

                                        # recalc inst
                                        subdf = subdf.join(gridprem_ts[['t', 'grid_node', 'prem_Rp_kWh']], on=['t', 'grid_node'], how='left')  
                                        egid_subdf = subdf.filter(pl.col('EGID') == egid).clone()


                                        # # feature: make demand more "spiky" to explore effect
                                        if plot_mapline_specs['noisy_demand_TF']:

                                            egid_subdf = egid_subdf.with_columns([
                                                pl.col('demand_kW').alias('original_sfhmfh_demand_kW')
                                            ])

                                            egid_subdf = egid_subdf.filter(pl.col('EGID') == egid).clone()
                                            egid_subdf = egid_subdf.with_columns(
                                                (pl.col("demand_kW") - pl.col("demand_kW").shift(-1)).alias("demand_change")
                                            )
                                            egid_subdf = egid_subdf.with_columns([
                                                pl.when(pl.col('demand_change').is_null())
                                                .then(pl.col('demand_change').shift(1))
                                                .otherwise(pl.col('demand_change')).alias('demand_change')
                                            ])


                                            # add more noise to demand_kW based on change
                                            noise_factor = plot_mapline_specs['noisy_demand_factor']
                                            def add_noise_demand_change(change_val):
                                                noise = np.random.normal(loc=0, scale=abs(change_val)* noise_factor)
                                                return noise

                                            egid_subdf = egid_subdf.with_columns(
                                                pl.col('original_sfhmfh_demand_kW').map_elements(
                                                    add_noise_demand_change, return_dtype=pl.Float64).alias('noise_by_demand_change')
                                            )                                        

                                            # Add the noise to the original demand
                                            egid_subdf = egid_subdf.with_columns(
                                                (pl.col('original_sfhmfh_demand_kW') + pl.col('noise_by_demand_change')).alias('demand_kW')
                                            )

                                            # make correction to annual aggregate
                                            total_original_demand = egid_subdf.select(pl.col("original_sfhmfh_demand_kW").sum()).to_numpy().item()
                                            total_noisy_demand = egid_subdf.select(pl.col("demand_kW").sum()).to_numpy().item()
                                            correction_factor = total_original_demand / total_noisy_demand
                                            egid_subdf = egid_subdf.with_columns(
                                                (pl.col("demand_kW") * correction_factor).alias("demand_kW")
                                            )
                                            egid_subdf.select(['EGID', 'df_uid', 'original_sfhmfh_demand_kW', 'demand_change', 'noise_by_demand_change', 'demand_kW', ])



                                        # compute npv of optimized installtion size ----------------------------------------------
                                        max_stromertrag = egid_subdf['STROMERTRAG'].max()
                                        max_dfuid_df = egid_subdf.filter(pl.col('STROMERTRAG') == max_stromertrag).sort(['t_int'], descending=[False,])
                                        max_dfuid_df.select(['EGID', 't_int', 'STROMERTRAG', 'TotalPower', 'demand_kW', ])
                                        max_dfuid_df['demand_kW'].sum()


                                        # find optimal installation size
                                        estim_instcost_chfpkW, estim_instcost_chftotal, params_pkw, params_total = initial_sml_get_instcost_interpolate_function(
                                            use_generic_coeff_pkW_TF = False, 
                                            use_generic_coeff_total_TF = generic_pvinstcost_coeff_total['use_generic_instcost_coeff_total_TF'],
                                            generic_coeff_pkW = [1, 1, 1, 1],
                                            generic_coeff_total = [ 
                                                cost_coeff_dict['a'], 
                                                cost_coeff_dict['b'],
                                                cost_coeff_dict['c'],
                                                ])


                                        opt_flaeche, opt_npv = optimize_pv_size(max_dfuid_df, estim_instcost_chftotal, plot_mapline_specs['recalc_opt_max_flaeche_factor'], optim_calc_interest_rate = recalc_interest_rate)
                                        opt_kWpeak = opt_flaeche * self.pvalloc_scen.TECspec_kWpeak_per_m2 * self.pvalloc_scen.TECspec_share_roof_area_available


                                        # calculate df for optim inst per egid ----------------------------------------------
                                        # optimal production
                                        flaeche = opt_flaeche
                                        npv, rest = calculate_npv(opt_flaeche, max_dfuid_df, estim_instcost_chftotal, npv_calc_interest_rate = recalc_interest_rate, )
                                        installation_cost, disc_cashflow, pvprod_kW_sum, demand_kW_sum, selfconsum_kW_sum, breakeven = rest[0], rest[1], rest[2], rest[3], rest[4], rest[5]

                                        max_dfuid_df = max_dfuid_df.with_columns([
                                            pl.lit(opt_flaeche).alias("opt_FLAECHE"),

                                            pl.lit(opt_npv).alias("NPV_uid"),
                                            pl.lit(opt_kWpeak).alias("TotalPower"),
                                            pl.lit(installation_cost).alias("estim_pvinstcost_chf"),
                                            pl.lit(disc_cashflow).alias("disc_cashflow"),
                                            pl.lit(breakeven).alias('breakeven'),
                                            ])

                                        # if self.pvalloc_scen.TECspec_pvprod_calc_method == 'method2.2':
                                        max_dfuid_df = max_dfuid_df.with_columns([
                                            ((pl.col("radiation") / 1000) * 
                                            pl.col("panel_efficiency") * 
                                            self.pvalloc_scen.TECspec_inverter_efficiency * 
                                            self.pvalloc_scen.TECspec_share_roof_area_available * 
                                            pl.col("opt_FLAECHE")).alias("pvprod_kW")
                                        ])

                                        selfconsum_expr = pl.min_horizontal([ pl.col("pvprod_kW"), pl.col("demand_kW") ]) * self.pvalloc_scen.TECspec_self_consumption_ifapplicable

                                        max_dfuid_df = max_dfuid_df.with_columns([  
                                            selfconsum_expr.alias("selfconsum_kW"),
                                            (pl.col("pvprod_kW") - selfconsum_expr).alias("netfeedin_kW"),
                                            (pl.col("demand_kW") - selfconsum_expr).alias("netdemand_kW")
                                            ])                        

                                        max_dfuid_df = max_dfuid_df.with_columns([
                                            pl.when(pl.col('netfeedin_kW') > 0)
                                            .then(pl.col('selfconsum_kW') / pl.col('pvprod_kW'))
                                            .otherwise(0)  # or any other value that makes sense
                                            .alias('selfconsum_pvprod_ratio')
                                        ])


                                        # add trace to timeseries plot ----------------------------------------------
                                        # name_str = f"---- RECALCULATED PV: EGID: {egid}, TotalPower: {round(max_dfuid_df['TotalPower'][0], 3)}, info_source: {max_dfuid_df['info_source'][0]} {20*'-'}, "
                                        name_str = f"---- RECALCULATED PV:"

                                        if 'recalc_pv_inst' in plot_mapline_specs['traces_in_timeseries_plot']:
                                            fig_rfpart_ts.add_trace(go.Scatter(x= [None, ], y = [None,] , mode = 'lines', 
                                                                        name = name_str, 
                                                                        line = dict(color=plot_mapline_specs['roofpartition_color'], width=1.5),
                                                                        opacity = 0,
                                                                    ))

                                            if plot_mapline_specs['noisy_demand_TF']:
                                                cols_for_traces = [
                                                    'demand_kW', 'radiation', 'pvprod_kW', 'selfconsum_kW', 'netfeedin_kW', 'netdemand_kW', 'selfconsum_pvprod_ratio', 
                                                    'noise_by_demand_change', 'original_sfhmfh_demand_kW', 'demand_change',
                                                    ]
                                            else: 
                                                cols_for_traces = [
                                                    'demand_kW', 'radiation', 'pvprod_kW', 'selfconsum_kW', 'netfeedin_kW', 'netdemand_kW', 'selfconsum_pvprod_ratio', 
                                                ]

                                            for col in cols_for_traces:
                                                if col == 'selfconsum_pvprod_ratio':
                                                    agg_value = max_dfuid_df[col].mean()
                                                    agg_method = 'mean'
                                                else:
                                                    agg_value = sum(max_dfuid_df[col])
                                                    agg_method = 'sum'

                                                if agg_value < 2: 
                                                    agg_round_value = round(agg_value,5)
                                                else:
                                                    agg_round_value = round(agg_value,2)

                                                # reduce t_hours to have more "handable - plots" in plotly
                                                if self.visual_sett.cut_timeseries_to_zoom_hour: 
                                                    max_dfuid_df_plot = max_dfuid_df.filter(
                                                        (pl.col('t_int') >= self.visual_sett.default_zoom_hour[0]-(24*7)) &
                                                        (pl.col('t_int') <= self.visual_sett.default_zoom_hour[1]+(24*7))
                                                    ).clone()
                                                else:
                                                    max_dfuid_df_plot = max_dfuid_df.clone()

                                                fig_rfpart_ts.add_trace(go.Scatter(
                                                    x=max_dfuid_df_plot['t_int'], 
                                                    y=max_dfuid_df_plot[col],
                                                    mode='lines+markers', 
                                                    marker = dict(
                                                        symbol = 'diamond',
                                                        # size = 5,
                                                    ),
                                                    name=f"{col:<25} - {agg_method}: {agg_round_value}",
                                                    line=dict(
                                                        width=2.5,
                                                        # dash = 'solid'
                                                        ),
                                                    opacity = 0.6,
                                                ))
                                                row = {
                                                    'EGID': [egid for i in range(len(max_dfuid_df_plot))],
                                                    'col': [col for i in range(len(max_dfuid_df_plot))],
                                                    'value': max_dfuid_df_plot[col].to_list(),
                                                    't_int': max_dfuid_df_plot['t_int'].to_list(),
                                                }
                                                t_int_array_daynight_bands = copy.deepcopy(max_dfuid_df_plot['t_int'])
                                                rfpart_ts_list.append(pd.DataFrame(row))

                                        # add traces for jscatter
                                        if topo[egid]['pv_inst']['info_source'] == 'pv_df':
                                            egid_jsctr_list.append(egid)
                                            TotalPower_jsctr_list.append(topo[egid]['pv_inst']['TotalPower'])
                                            RecalcPowe_jsctr_list.append(opt_kWpeak)
                                            sum_FLAECHE_list.append(sum([ v['FLAECHE'] for k,v in topo[egid]['solkat_partitions'].items()]))
                                            max_optFLAECHE_list.append(max_dfuid_df['FLAECHE'].first() * 1.5)
                                            interest_rate_list.append(recalc_interest_rate)
                                            cost_coeff_list.append(f'a:{cost_coeff_dict["a"]}, b:{cost_coeff_dict["b"]}, c:{cost_coeff_dict["c"]}')



                                        # agg df for later histogram 
                                        max_dfuid_hist = max_dfuid_df.group_by(['EGID']).agg([
                                            pl.col('inst_TF').first().alias('inst_TF'),
                                            pl.col('info_source').first().alias('info_source'),
                                            pl.col('grid_node').first().alias('grid_node'),
                                            pl.col('breakeven').first().alias('breakeven'), 

                                            pl.col('demand_kW').sum().alias('demand_kW'),
                                            pl.col('poss_pvprod_kW').sum().alias('poss_pvprod_kW'),
                                            pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                                            pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                                            pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                                            pl.col('netdemand_kW').sum().alias('netdemand_kW'),
                                            pl.col('selfconsum_pvprod_ratio').mean().alias('selfconsum_pvprod_ratio')
                                        ])


                            # plot economic functions 
                            if True:
                                # fig_econfunc =  go.Figure()
                                flaeche_range = np.linspace(0, int(max_dfuid_df['FLAECHE'].max())*2 , 200)
                                kWpeak_range = self.pvalloc_scen.TECspec_kWpeak_per_m2 * self.pvalloc_scen.TECspec_share_roof_area_available * flaeche_range
                                # cost_kWp = estim_instcost_chftotal(kWpeak_range)
                                npv_list, cost_list, cashflow_list, pvprod_kW_list, demand_kW_list, selfconsum_kW_list, breakeven_list = [], [], [], [], [], [], []
                                for flaeche in flaeche_range:
                                    npv, rest = calculate_npv(flaeche, max_dfuid_df, estim_instcost_chftotal, npv_calc_interest_rate = recalc_interest_rate, )    
                                    npv_list.append(npv)

                                    cost_list.append(         rest[0]) 
                                    cashflow_list.append(     rest[1]) 
                                    pvprod_kW_list.append(    rest[2]) 
                                    demand_kW_list.append(    rest[3]) 
                                    selfconsum_kW_list.append(rest[4]) 
                                    breakeven_list.append(    rest[5])

                                npv = np.array(npv_list)
                                cost_kWp = np.array(cost_list)
                                cashflow = np.array(cashflow_list)
                                pvprod_kW_sum = np.array(pvprod_kW_list)
                                demand_kW_sum = np.array(demand_kW_list)
                                selfconsum_kW_sum = np.array(selfconsum_kW_list)
                                breakeven = np.array(breakeven_list)

                                pvtarif = round(max_dfuid_df['pvtarif_Rp_kWh'][0], 4)
                                elecpri = round(max_dfuid_df['elecpri_Rp_kWh'][0], 4)
                                params_total_a = round(params_total[0], 1) 
                                params_total_b = round(params_total[1], 1) 
                                params_total_c = round(params_total[2], 5)

                                fig_econfunc.add_trace(go.Scatter( x=kWpeak_range,   y=cost_kWp,           mode='lines',  name=f'Installation Cost (CHF)   - pvtarif:{pvtarif}, elecpri:{elecpri} (Rp/kWh)', )) # line=dict(color='blue')))
                                fig_econfunc.add_trace(go.Scatter( x=kWpeak_range,   y=cashflow,           mode='lines',  name=f'Cash Flow (CHF)           - pvtarif:{pvtarif}, elecpri:{elecpri} (Rp/kWh)',         )) # line=dict(color='magenta')))
                                fig_econfunc.add_trace(go.Scatter( x=kWpeak_range,   y=npv,                mode='lines',  name=f'Net Present Value (CHF)   - pvtarif:{pvtarif}, elecpri:{elecpri} (Rp/kWh)', )) # line=dict(color='green')))
                                fig_econfunc.add_trace(go.Scatter( x=kWpeak_range,   y=pvprod_kW_sum,      mode='lines',  name=f'PV Production (kWh)       - pvtarif:{pvtarif}, elecpri:{elecpri} (Rp/kWh)',     )) # line=dict(color='orange')))
                                fig_econfunc.add_trace(go.Scatter( x=kWpeak_range,   y=demand_kW_sum,      mode='lines',  name=f'Demand (kWh)              - pvtarif:{pvtarif}, elecpri:{elecpri} (Rp/kWh)',            )) # line=dict(color='red')))
                                fig_econfunc.add_trace(go.Scatter( x=kWpeak_range,   y=selfconsum_kW_sum,  mode='lines',  name=f'Self-consumption (kWh)    - pvtarif:{pvtarif}, elecpri:{elecpri} (Rp/kWh)',  )) # line=dict(color='purple')))
                                fig_econfunc.add_trace(go.Scatter( x=kWpeak_range,   y=breakeven,          mode='lines',  name=f'Breakeven Year (years)    - pvtarif:{pvtarif}, elecpri:{elecpri} (Rp/kWh)', )) # line=dict(color='black')))   
                                fig_econfunc.add_trace(go.Scatter( x=[None,],        y=[None,],            mode='lines',  name='',  opacity = 0    ))

                                # add aggreagted traces
                                palette = random.choice(fig_agg_color_palettes)
                                egid_color = trace_color_dict[palette]                                    
                                for i_tupl, tupl in enumerate([
                                                (cost_kWp,          'Inst.Cost (CHF) ' ), 
                                                (cashflow,          'CF (CHF) ' ), 
                                                (npv,               'NPV (CHF)' ), 
                                                # (pvprod_kW_sum,     'PV PRod (kWh) ' ), 
                                                # (demand_kW_sum,     'Demand (kWh) ' ), 
                                                # (selfconsum_kW_sum, 'Self-cons. (kWh) ' ), 
                                ]):
                                    fig_econfunc_comp.add_trace(     go.Scatter( x=kWpeak_range, y=tupl[0], mode='lines',  name=       f'EGID: {egid}, {tupl[1]} - pvtarif:{pvtarif}, elecpri:{elecpri} (Rp/kWh), params_total a:{params_total_a} b:{params_total_b}, c:{params_total_c} ', line = dict(color = egid_color[-i_tupl -1 ]), ))
                                    fig_econfunc_comp_scen.add_trace(go.Scatter( x=kWpeak_range, y=tupl[0], mode='lines',  name=f'{scen} EGID: {egid}, {tupl[1]} - pvtarif:{pvtarif}, elecpri:{elecpri} (Rp/kWh), params_total a:{params_total_a} b:{params_total_b}, c:{params_total_c} ', line = dict(color = egid_color[-i_tupl -1 ]), ))

                                fig_econfunc_comp.add_trace(     go.Scatter( x=[None,],        y=[None,],            mode='lines',  name='',  opacity = 0    ))
                                fig_econfunc_comp_scen.add_trace(go.Scatter( x=[None,],        y=[None,],            mode='lines',  name='',  opacity = 0    ))

                        checkpoint_to_logfile(f'--> i_scen: {i_scen}, i_egid: {i_egid};/{len(selected_egid)} end recalc inst', self.visual_sett.log_name) if checkpoint_print_TF else None

                    # fig jscatter
                    if 'recalc_opt_inst' in plot_mapline_specs['run_selected_plot_parts']:    
                        random_color = '#%06x' % random.randint(0, 0xFFFFFF)
                        random_position = random.choice(['top center', 'bottom center', 'middle left', 'middle right',
                            'top left', 'top right', 'bottom left', 'bottom right'])

                        jsctr_scen_df = pd.DataFrame({
                            'EGID': egid_jsctr_list, 'TotalPower': TotalPower_jsctr_list, 'RecalcPower': RecalcPowe_jsctr_list, 
                            'sum_FLAECHE': sum_FLAECHE_list, 'max_opt_FLAECHE': max_optFLAECHE_list, 'interest_rate': interest_rate_list,
                            'cost_coeff': cost_coeff_list})

                        fig_instpwr_jsctr.add_trace(go.Scatter(
                            x=jsctr_scen_df['TotalPower'], 
                            y=jsctr_scen_df['RecalcPower'],
                            mode='markers+text',
                            text=jsctr_scen_df['EGID'],
                            textposition=random_position,
                            customdata=jsctr_scen_df[['EGID', 'TotalPower', 'RecalcPower', 'sum_FLAECHE', 'max_opt_FLAECHE', 'interest_rate', 'cost_coeff']].values,
                            hovertemplate=(
                                "EGID: %{customdata[0]}<br>"
                                "TotalPower: %{customdata[1]:.2f} kW<br>"
                                "RecalcPower: %{customdata[2]:.2f} kW<br>"
                                "sum_FLAECHE: %{customdata[3]:.2f} m²<br>"
                                "max_opt_FLAECHE: %{customdata[4]:.2f} m²<br>"
                                "interest_rate: %{customdata[5]:.2f}<br>"
                                "cost_coeff: %{customdata[6]}<br>"
                                "<extra></extra>"
                            ),
                            marker=dict(
                                size=10 + 4*i_recalc_interest_rate,
                                opacity=0.5,
                                color = random_color,
                            ),
                            textfont = dict(
                                color=random_color, 
                            ), 
                            name=f'{scen} - interest_rate: {recalc_interest_rate}, cost_coeffs a:{params_total_a} b:{params_total_b}, c:{params_total_c} ',
                            showlegend=True
                        ))                   



                # traces summary ==============================
                fig_rfpart_summary = go.Figure()
                if 'summary' in plot_mapline_specs['run_selected_plot_parts']: 
                    summary_agg_egids = pl.concat(summary_agg_egids_list)

                    flaeche_pvdf = topo[egid]['pv_inst']['TotalPower'] / (self.pvalloc_scen.TECspec_kWpeak_per_m2 * self.pvalloc_scen.TECspec_share_roof_area_available) 
                    # summary_agg_egids = summary_agg_egids.rename({'share_radiation_used': 'radiation'})
                    npv, rest = calculate_npv(flaeche_pvdf, summary_agg_egids, estim_instcost_chftotal, npv_calc_interest_rate = recalc_interest_rate)
                    breakeven = rest[5]


                    # aggregate to yearly values
                    summary_agg_egids = summary_agg_egids.group_by(['EGID']).agg([
                        pl.col('inst_TF').first().alias('inst_TF'),
                        pl.col('info_source').first().alias('info_source'),
                        pl.col('grid_node').first().alias('grid_node'),                                            
                        pl.lit(0.0).alias('breakeven'), 

                        pl.col('demand_kW').sum().alias('demand_kW'),
                        pl.col('poss_pvprod_kW').sum().alias('poss_pvprod_kW'),
                        pl.col('pvprod_kW').sum().alias('pvprod_kW'),
                        pl.col('selfconsum_kW').sum().alias('selfconsum_kW'),
                        pl.col('netfeedin_kW').sum().alias('netfeedin_kW'),
                        pl.col('netdemand_kW').sum().alias('netdemand_kW'),
                        pl.col('selfconsum_pvprod_ratio').mean().alias('selfconsum_pvprod_ratio'), 

                        # pl.col('noisyadj_demand_kW').sum().alias('noisyadj_demand_kW'),
                        # pl.col('original_sfhmfh_demand_kW').sum().alias('original_sfhmfh_demand_kW'),

                    ])

                    # extend with useful information
                    summary_agg_egids = summary_agg_egids.with_columns([
                        pl.lit(topo[egid]['gwr_info']['gklas']).alias('GKLAS'),
                        pl.lit(topo[egid]['gwr_info']['garea']).alias('GAREA'),
                        pl.lit(topo[egid]['gwr_info']['are_typ']).alias('are_typ'),
                        pl.lit(topo[egid]['gwr_info']['sfhmfh_typ']).alias('sfhmfh_typ'),
                        pl.lit(topo[egid]['pv_inst']['TotalPower']).alias('TotalPower'),
                        pl.lit(topo[egid]['demand_elec_pGAREA']).alias('demand_elec_pGAREA'),
                        pl.lit(topo[egid]['pvtarif_Rp_kWh']).alias('pvtarif_Rp_kWh'),
                        pl.lit(topo[egid]['elecpri_Rp_kWh']).alias('elecpri_Rp_kWh'),
                        pl.lit(breakeven).alias('breakeven'),
                    ])
                    max_dfuid_df



                    # add Bar traces
                    fig_rfpart_summary = go.Figure()
                    for col in summary_agg_egids.columns:
                        if col in self.visual_sett.plot_ind_mapline_prodHOY_EGIDrfcombo_specs['summary_cols_only_in_legend']:
                            fig_rfpart_summary.add_trace(go.Bar(
                                x=[col],
                                y=[None], 
                                name=f'{col:15} - {summary_agg_egids[col].to_list()[0]}',
                                marker=dict(color='rgba(0,0,0,0)'), 
                                showlegend=True  ,
                                xaxis='x2',  # Assign to a secondary x-axis (we can hide this x-axis later)
                            ))
                        else:

                            summary_agg_egids = summary_agg_egids.with_columns(
                                pl.when(pl.col("breakeven").is_null())
                                .then(9999)
                                .otherwise(pl.col("breakeven"))
                                .alias("breakeven")
                            )
                            y_val = round(summary_agg_egids[col][0],4)

                            fig_rfpart_summary.add_trace(go.Bar(
                                x = [col],
                                y = [y_val],
                                name = col,
                                showlegend = True,
                                text= y_val,
                                textposition='outside', 
                            ))


                            # add  values
                            if (col in ['demand_kW', 'poss_pvprod_kW', 'pvprod_kW', 'selfconsum_kW', 'netfeedin_kW', 'netdemand_kW', 'selfconsum_pvprod_ratio', 'breakeven' ] ):
                                # y_val = round(max_dfuid_hist[col][0],4)
                                y_val = round(summary_agg_egids[col][0],4)
                                fig_rfpart_summary.add_trace(go.Bar(
                                    x = [f'RECALC-{col}',],
                                    y = [y_val],
                                    name = f'RECALC-{col}',
                                    showlegend = True,
                                    text= y_val,
                                    textposition
                                    ='outside', 
                                ))

                    # add STROMERTRAG
                    for dfuid, partition in topo[egid]['solkat_partitions'].items():
                        y_val = round(partition['STROMERTRAG'],4)
                        fig_rfpart_summary.add_trace(go.Bar(
                            x = [f'dfuid-{dfuid}',],
                            y = [y_val],
                            name = f'STROMERTRAG {dfuid} (kwh/y)',
                            showlegend = True,
                            text= y_val,
                            textposition='outside', 
                        ))
                checkpoint_to_logfile(f'--> i_scen: {i_scen}, i_egid: {i_egid};/{len(selected_egid)} end summary', self.visual_sett.log_name) if checkpoint_print_TF else None



                # layout & export plots ==============================
                if ('map' in plot_mapline_specs['run_selected_plot_parts']) & ('map' in plot_mapline_specs['run_selected_plot_parts']) :


                    # partition map
                    fig_rfpart_map = self.add_scen_name_to_plot(fig_rfpart_map, scen, self.pvalloc_scen)
                    info_source_pd = Map_pvinfo_topo_egid.to_pandas()

                    for i, row in info_source_pd.loc[info_source_pd["EGID"] == egid].iterrows():
                        if row['inst_TF']==True:
                            info_source_str = row['info_source']

                    fig_rfpart_map.update_layout(
                        title=f'Roof Partitions Map for EGID: {egid} (info_source: {info_source_str})',
                        mapbox=dict(
                            style="carto-positron",
                            zoom=18,
                            center={"lat": center_lat, "lon": center_lon},
                        ),
                        margin={"r":0, "t":50, "l":0, "b":0},
                    )
                    if plot_show:
                        fig_rfpart_map.show() 
                    if plot_export:
                        if self.visual_sett.save_plot_by_scen_directory:
                            fig_rfpart_map.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}__plot_ind_mapline_prodHOY_EGID{egid}_maprfcombo.html')
                        else:
                            fig_rfpart_map.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_mapline_prodHOY_EGID{egid}_maprfcombo.html')    


                # timesieres for combo of poss_prod, installation + optimized calculation 
                fig_rfpart_ts = self.set_default_fig_zoom_hour(fig_rfpart_ts, self.visual_sett.default_zoom_hour)
                fig_rfpart_ts = self.add_scen_name_to_plot(fig_rfpart_ts, scen, self.pvalloc_scen)

                if self.visual_sett.add_day_night_HOY_bands:
                    fig_rfpart_ts = self.add_day_night_bands_HOY_plot(fig_rfpart_ts, t_int_array_daynight_bands)

                fig_rfpart_ts.update_layout(
                    xaxis_title = 't - Hours', 
                    yaxis_title = 'kW', 
                    title = f"Roof Partitions Time Series for EGID: {egid}, n_partitions: {len(topo[egid]['solkat_partitions'])} info_source: {topo[egid]['pv_inst']['info_source']}",
                    template = 'plotly_white',
                )
                if 'timeseries' in plot_mapline_specs['run_selected_plot_parts']:   
                    if plot_show:
                        fig_rfpart_ts.show()
                if plot_export:
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig_rfpart_ts.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_ind_mapline_prodHOY_EGID{egid}_timeseries.html')
                    else:
                        fig_rfpart_ts.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_mapline_prodHOY_EGID{egid}_timeseries.html')

                    # export csv for rfpart_ts
                    export_plot_data_df = pd.concat(rfpart_ts_list)
                    os.makedirs(f'{self.visual_sett.visual_static_export_path}', exist_ok=True)
                    export_df_name         = 'export_rfpart_ts_EGID'
                    export_df_name_scenlen = f'{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen'
                    if os.path.exists(f'{self.visual_sett.visual_static_export_path}/{export_df_name_scenlen}.csv'):
                        n_agg_plots = len(glob.glob(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen*.csv'))
                        os.rename(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', 
                                f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen_{n_agg_plots}nplot.csv')
                    export_plot_data_df.to_csv(f'{self.visual_sett.visual_static_export_path}/{export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', index=False)
                    print_to_logfile(f'\texport: {export_df_name}{len(self.visual_sett.pvalloc_scen_list)}scen.csv', self.visual_sett.log_name)

                # economic function optim installation
                fig_econfunc.update_layout(
                    title=f'Economic Comparison for EGID: {egid} ',
                    xaxis_title='System Size (kWp)',
                    yaxis_title='Value (CHF/kWh)',
                    legend=dict(x=0.99, y=0.99),
                    template='plotly_white'
                )
                # if 'recalc_pvdf' in plot_mapline_specs['run_selected_plot_parts']:
                #     if plot_show:
                #         fig_econfunc.show()
                if plot_export:
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig_econfunc.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_ind_mapline_prodHOY_EGID{egid}_econfunc.html')
                    else:
                        fig_econfunc.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_mapline_prodHOY_EGID{egid}_econfunc.html')


                # barplot annual summary aggregates
                fig_rfpart_summary = self.add_scen_name_to_plot(fig_rfpart_summary, scen, self.pvalloc_scen)
                fig_rfpart_summary.update_layout(
                    title = f'Roof Partitions Summary for EGID: {egid}',
                    xaxis_title = 'Columns',
                    yaxis_title = 'kWh / (other)',
                    template = 'plotly_white',
                    barmode = 'group',
                    xaxis2=dict(
                        showticklabels=False,  # Hide ticks and labels for the secondary x-axis
                        overlaying='x',  # Overlay on the main x-axis
                    ),
                )
                if 'summary' in plot_mapline_specs['run_selected_plot_parts']:
                    if plot_show:
                        fig_rfpart_summary.show()
                if plot_export:
                    if self.visual_sett.save_plot_by_scen_directory:
                        fig_rfpart_summary.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_ind_mapline_prodHOY_EGID{egid}_summary.html')
                    else:
                        fig_rfpart_summary.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_mapline_prodHOY_EGID{egid}_summary.html')


            # agg economic function optim installation
            fig_econfunc_comp.update_layout(
                title=f'Economic Comparison Aggregated for EGIDs (scen {scen})', 
                xaxis_title='System Size (kWp)',
                yaxis_title='Value (CHF/kWh)',
                legend=dict(x=0.99, y=0.99),
                template='plotly_white'
            )   
            if 'econ_func' in plot_mapline_specs['run_selected_plot_parts']:
                if plot_show:
                    fig_econfunc_comp.show()
            if plot_export:
                if self.visual_sett.save_plot_by_scen_directory:
                    fig_econfunc_comp.write_html(f'{self.visual_sett.visual_path}/{scen}/{scen}___plot_ind_mapline_prodHOY_EGID{egid}_econfunc_comp.html')
                else:
                    fig_econfunc_comp.write_html(f'{self.visual_sett.visual_path}/{scen}___plot_ind_mapline_prodHOY_EGID{egid}_econfunc_comp.html')


        # agg economic function optim installation for all scenarios
        fig_econfunc_comp_scen.update_layout(
                title='Economic Comparison Aggregated for EGIDs', 
                xaxis_title='System Size (kWp)',
                yaxis_title='Value (CHF/kWh)',
                legend=dict(x=0.99, y=0.99),
                template='plotly_white'
            )   
        if 'econ_func' in plot_mapline_specs['run_selected_plot_parts']:
            if plot_show:
                    fig_econfunc_comp_scen.show()
        if plot_export:
            if os.path.exists(f'{self.visual_sett.visual_path}/plot_ind_mapline_prodHOY_EGID_fig_econfunc_comp_scen___{len(self.visual_sett.pvalloc_scen_list)}scen.html'):
                n_econ_comp_plots = len(glob.glob(f'{self.visual_sett.visual_path}/plot_ind_mapline_prodHOY_EGID_fig_econfunc_comp_scen___{len(self.visual_sett.pvalloc_scen_list)}scen*.html'))
                os.rename(f'{self.visual_sett.visual_path}/plot_ind_mapline_prodHOY_EGID_fig_econfunc_comp_scen___{len(self.visual_sett.pvalloc_scen_list)}scen.html',
                          f'{self.visual_sett.visual_path}/plot_ind_mapline_prodHOY_EGID_fig_econfunc_comp_scen___{len(self.visual_sett.pvalloc_scen_list)}scen___{n_econ_comp_plots}nplot.html')
            fig_econfunc_comp_scen.write_html(f'{self.visual_sett.visual_path}/plot_ind_mapline_prodHOY_EGID_fig_econfunc_comp_scen___{len(self.visual_sett.pvalloc_scen_list)}scen.html')  


        # agg joint scatter plot Power installation
        if 'recalc_opt_inst' in plot_mapline_specs['run_selected_plot_parts']:
            min_totalpower, max_totalpower = min(jsctr_scen_df['TotalPower'])-1, max(jsctr_scen_df['TotalPower'])+1
            fig_instpwr_jsctr.add_trace(go.Scatter(
                x=[min_totalpower, max_totalpower],
                y=[min_totalpower, max_totalpower],
                mode='lines',
                name = 'x=y diagonal',
                line=dict(color='Red', dash='dash'),
            ))               
            fig_instpwr_jsctr.update_layout(
                title=f'Joint Scatter Plot of Installed Power vs Recalculated Power for EGIDs of pv_df ({len(self.visual_sett.pvalloc_scen_list)} n_scen)',
                xaxis_title='Total Power (kWp, real inst)', 
                yaxis_title='Recalc Power (kWp, modelled inst)',
                legend=dict(x=0.99, y=0.99),
                template='plotly_white',
            )
            if 'joint_scatter' in plot_mapline_specs['run_selected_plot_parts']:
                if plot_show:
                        fig_instpwr_jsctr.show()
            if plot_export:
                if os.path.exists(f'{self.visual_sett.visual_path}/plot_ind_mapline_fig_instpwr_jsctr___{len(self.visual_sett.pvalloc_scen_list)}scen.html'):
                    n_econ_comp_plots = len(glob.glob(f'{self.visual_sett.visual_path}/plot_ind_mapline_fig_instpwr_jsctr___{len(self.visual_sett.pvalloc_scen_list)}scen*.html'))

                    os.rename(f'{self.visual_sett.visual_path}/plot_ind_mapline_fig_instpwr_jsctr___{len(self.visual_sett.pvalloc_scen_list)}scen.html', 
                            f'{self.visual_sett.visual_path}/plot_ind_mapline_fig_instpwr_jsctr___{len(self.visual_sett.pvalloc_scen_list)}scen___{n_econ_comp_plots}nplot.html', )
                fig_instpwr_jsctr.write_html(f'{self.visual_sett.visual_path}/plot_ind_mapline_fig_instpwr_jsctr___{len(self.visual_sett.pvalloc_scen_list)}scen.html')
