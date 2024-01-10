import os
import numpy as np
import pandas as pd
import geopandas as gpd
import datetime
import glob

from datetime import datetime

# ------------------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------------------
def Lupien_aggregation(
        script_run_on_server_def = 0,
        check_vs_raw_input = False, 
        union_vs_hull_shape = 'union'):
    """
    script_run_on_server_def: 0 = private computer, 1 = server
    """
        
    # script_run_on_server_def = 0
    # check_vs_raw_input = False
    # union_vs_hull_shape = 'union'

    # ------------------------------------------------------------------------------

    if script_run_on_server_def == 0:
        wd_path = "C:/Models/OptimalPV_RH"   # path for private computer
        data_path = f'{wd_path}_data'
    elif script_run_on_server_def == 1:
        wd_path = "D:/RaulHochuli_inuse/OptimalPV_RH"
        data_path = f'{wd_path}_data'

    # wd_path = 'C:/Models/OptimalPV_RH'
    # data_path = f'{wd_path}_data'
    os.chdir(wd_path)
    os.listdir(data_path)

    # import raw input data
    # check_vs_raw_input = False
    # agg_version = 'agg_solkat_pv_gm_gwr_heat_buff10_KT'


    # create a export txt file for summary outputs
    print(f'\n\n ***** AGGREGATION roofkat pv munic ***** \t time: {datetime.now()}')
    if not os.path.exists(f'{data_path}/Lupien_aggregation'):
        os.makedirs(f'{data_path}/Lupien_aggregation')

    export_txt_name = f'{data_path}/Lupien_aggregation/aggregation_roofkat_pv_munic_log.txt'
    with open(export_txt_name, 'w') as export_txt:
        export_txt.write(f'\n')
        export_txt.write(f'\n *************************** \n     SANITY CHECK OUTPUT \n *************************** \n')
        export_txt.write(f'\n* start script: time: {datetime.now()} \n  settings: ')
        export_txt.write(f'\n -- script_run_on_server_def: {script_run_on_server_def}')
        export_txt.write(f'\n -- check_vs_raw_input: {check_vs_raw_input}')
        export_txt.write(f'\n -- union_vs_hull_shape: {union_vs_hull_shape}')

    # ------------------------------------------------------------------------------
    # DATA IMPORT
    # ------------------------------------------------------------------------------
        
    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')
    Map_roof_pv = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/Map_roof_pv_{union_vs_hull_shape}.parquet')
    Map_roof_gm = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/Map_roof_gm_{union_vs_hull_shape}.parquet')

    if check_vs_raw_input:
        print(f'\n\n* import raw input data: time: {datetime.now()}')
        roof_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
        print(f'imported roof_kat, time: {datetime.now()}')
        elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
        pv = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()
        print(f'imported pv, time: {datetime.now()}')

        with open(export_txt_name, 'a') as export_txt:
            export_txt.write(f'\n\n* use parquet RAW INPUT DATA for scanity check: time: {datetime.now()}')

    elif not check_vs_raw_input:
        print(f'\n\n* import parquet files: time: {datetime.now()}')
        roof_kat = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/roof_kat_by_gm.parquet')
        roof_kat.drop(columns=list(gm_shp.columns) + ['index_right',], inplace=True)

        pv = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/pv_by_gm.parquet')
        pv.drop(columns=list(gm_shp.columns) + ['index_right',], inplace=True)

        with open(export_txt_name, 'a') as export_txt:
            export_txt.write(f'\n\n* use parquet INTERCEPTS for scanity check: time: {datetime.now()}')
        
    # ------------------------------------------------------------------------------
    # AGGREGATE
    # ------------------------------------------------------------------------------

    # map to roofs -----------------------------------------------------------------

    # pv to roofs_agg
    if "SB_UUID" not in Map_roof_pv.columns:
        Map_roof_pv = Map_roof_pv.reset_index()
    # Map_roof_pv['SB_UUID'] = Map_roof_pv['SB_UUID'].astype('str')
    # Map_roof_pv['xtf_id'] = Map_roof_pv['xtf_id'].where(Map_roof_pv['xtf_id'].isna(), Map_roof_pv['xtf_id'].astype(str))
    # Map_roof_pv['xtf_id'] = Map_roof_pv['xtf_id'].astype('str')
    # Map_roof_pv.loc[Map_roof_pv['xtf_id'] == 'nan', 'xtf_id'] = np.nan
    Map_roof_pv['xtf_id'] = Map_roof_pv['xtf_id'].astype('Int64')
    # roof_kat['SB_UUID'] = roof_kat['SB_UUID'].astype('str')
    # pv['xtf_id'] = pv['xtf_id'].astype('Int64')


    df_agg_pq = roof_kat.merge(Map_roof_pv[['SB_UUID', 'xtf_id']], on='SB_UUID', how='left')
    df_agg_pq = df_agg_pq.merge(pv[['xtf_id', 'TotalPower', 'InitialPower',  'BeginningOfOperation']], on='xtf_id', how='left')

    # gm to roofs_agg
    if "SB_UUID" not in Map_roof_gm.columns:
        Map_roof_gm = Map_roof_gm.reset_index()
    # Map_roof_gm['SB_UUID'] = Map_roof_gm['SB_UUID'].astype('str')
    Map_roof_gm['BFS_NUMMER'] = Map_roof_gm['BFS_NUMMER'].astype('Int64')

    df_agg_pq = df_agg_pq.merge(Map_roof_gm[['SB_UUID', 'BFS_NUMMER']], on='SB_UUID', how='left')

    # gm to roof_kat
    roof_kat = roof_kat.merge(Map_roof_gm[['SB_UUID', 'BFS_NUMMER']], on='SB_UUID', how='left')


    # transform date to year -------------------------------------------------------
    df_agg_pq['BeginningOfOperation'] = pd.to_datetime(df_agg_pq['BeginningOfOperation'])
    df_agg_pq['year'] = df_agg_pq['BeginningOfOperation'].dt.year
    df_agg_pq['year'] = df_agg_pq['year'].astype('Int64')


    # transform data to be grouped later -------------------------------------------

    # EXPORT: Select Houses with PV installation ----------------------------------------------------
    print(f'* start aggregation <<agg_solkat_pv_gm_BY_INSTALLATION>>, time: {datetime.now()}')

    df_agg_pq_non_nan = df_agg_pq[df_agg_pq['xtf_id'].notna()]
    len(df_agg_pq_non_nan['xtf_id'].unique()) 
    len(pv['xtf_id'].unique())
    missing_xtf_id = pv[~pv['xtf_id'].isin(df_agg_pq_non_nan['xtf_id'].unique())]['xtf_id'].unique()

    # Sanity check
    with open(export_txt_name, 'a') as export_txt:
        export_txt.write(f'\n\n{10*"-"}\n sanity checks pv;  time: {datetime.now()} \n{10*"-"}')
        export_txt.write(f'\n length xtf_id in pv: {len(pv["xtf_id"].unique())} compared to total length in pv: {pv.shape[0]} => xtf_id is a unique identifier') 
        export_txt.write(f'\n length unique "xtf_id": {len(pv["xtf_id"].unique())} in pv ("raw import") | {len(df_agg_pq["xtf_id"].unique())} df_agg_pq ("aggregated") | {len(df_agg_pq_non_nan["xtf_id"].unique())} df_agg_pq_non_nan ("aggregated, by house shapes")')
        export_txt.write(f'\n number of missing installations, not covered by house shape: {len(missing_xtf_id)} | {len(missing_xtf_id)/len(pv["xtf_id"].unique())*100:.2f}% of total installations')
        export_txt.write(f'\n omitted xtf_ids: {missing_xtf_id}')
        export_txt.write(f'\n - ')
        # export_txt.write(f'\n "TotalPower": {pv["TotalPower"].sum()} in pv ("raw import") | {df_agg_pq["TotalPower"].sum()} df_agg_pq ("aggregated")')
        # export_txt.write(f'\n "InitialPower": {pv["InitialPower"].sum()} in pv ("raw import") | {df_agg_pq["InitialPower"].sum()} df_agg_pq ("aggregated")')


    # EXPORT:
    df_agg_pq_non_nan.to_csv(f'{data_path}/Lupien_aggregation/agg_solkat_pv_gm_BY_INSTALLATION.csv', index=False)
    df_agg_pq_non_nan.to_parquet(f'{data_path}/Lupien_aggregation/agg_solkat_pv_gm_BY_INSTALLATION.parquet')
    print(f'* export << agg_solkat_pv_gm_BY_INSTALLATION >> to parquet and csv, all roof shapes that contain a pv installation, intersected with gm and roof_kat, ')
    with open(export_txt_name, 'a') as export_txt:
        export_txt.write(f'\n\n *export << agg_solkat_pv_gm_BY_INSTALLATION >> to parquet and csv, time: {datetime.now()}')


    # EXPORT: aggregate data by municipality  ------------------------------------------------
    print(f'* start aggregation <<agg_solkat_pv_gm_gwr_heat_BY_gm>>, time: {datetime.now()}')

    # Sanity check
    with open(export_txt_name, 'a') as export_txt:
        export_txt.write(f'\n\n{10*"-"}\n sanity checks df_agg_pg by municipalities ONLY;  time: {datetime.now()} \n{10*"-"}')
        export_txt.write(f'\n length unique "SB_UUID": {len(roof_kat["SB_UUID"].unique())} in roof_kat ("raw import") | {len(df_agg_pq["SB_UUID"].unique())} df_agg_pq_non_nan ("aggregated, by house shapes")')

        print(f'** sanity check: STROMERTRAG')
        export_txt.write(f'\n - ')
        export_txt.write(f'\n total "STROMERTRAG":          {roof_kat["STROMERTRAG"].sum()} in roof_kat ("raw import") | {df_agg_pq["STROMERTRAG"].sum()} df_agg_pq_non_nan ("aggregated, by house shapes")')
        export_txt.write(f'\n total "STROMERTRAG_class2up": {roof_kat.loc[roof_kat["KLASSE"] >= 2, "STROMERTRAG"].sum()} in roof_kat ("raw import") | {df_agg_pq.loc[df_agg_pq["KLASSE"] >= 2, "STROMERTRAG"].sum()} df_agg_pq_non_nan ("aggregated, by house shapes")')
        export_txt.write(f'\n total "STROMERTRAG_class3up": {roof_kat.loc[roof_kat["KLASSE"] >= 3, "STROMERTRAG"].sum()} in roof_kat ("raw import") | {df_agg_pq.loc[df_agg_pq["KLASSE"] >= 3, "STROMERTRAG"].sum()} df_agg_pq_non_nan ("aggregated, by house shapes")')
        export_txt.write(f'\n total "STROMERTRAG_class4up": {roof_kat.loc[roof_kat["KLASSE"] >= 4, "STROMERTRAG"].sum()} in roof_kat ("raw import") | {df_agg_pq.loc[df_agg_pq["KLASSE"] >= 4, "STROMERTRAG"].sum()} df_agg_pq_non_nan ("aggregated, by house shapes")')
        export_txt.write(f'\n total "STROMERTRAG_class5up": {roof_kat.loc[roof_kat["KLASSE"] >= 5, "STROMERTRAG"].sum()} in roof_kat ("raw import") | {df_agg_pq.loc[df_agg_pq["KLASSE"] >= 5, "STROMERTRAG"].sum()} df_agg_pq_non_nan ("aggregated, by house shapes")') 

        print(f'** sanity check: STROMERTRAG - DIFFERENCES')
        export_txt.write(f'\n\n - DIFFERENCES')
        export_txt.write(f'\n total "STROMERTRAG":          {df_agg_pq["STROMERTRAG"].sum() - roof_kat["STROMERTRAG"].sum()}, df_agg_pq[STROMERTRAG] -roof_kat[STROMERTRAG]')
        export_txt.write(f'\n total "STROMERTRAG_class2up": {df_agg_pq.loc[df_agg_pq["KLASSE"] >= 2, "STROMERTRAG"].sum() - roof_kat.loc[roof_kat["KLASSE"] >= 2, "STROMERTRAG"].sum()}, df_agg_pq[STROMERTRAG_class2up] - roof_kat[STROMERTRAG_class2up]')
        export_txt.write(f'\n total "STROMERTRAG_class3up": {df_agg_pq.loc[df_agg_pq["KLASSE"] >= 3, "STROMERTRAG"].sum() - roof_kat.loc[roof_kat["KLASSE"] >= 3, "STROMERTRAG"].sum()}, df_agg_pq[STROMERTRAG_class3up] - roof_kat[STROMERTRAG_class3up]')
        export_txt.write(f'\n total "STROMERTRAG_class4up": {df_agg_pq.loc[df_agg_pq["KLASSE"] >= 4, "STROMERTRAG"].sum() - roof_kat.loc[roof_kat["KLASSE"] >= 4, "STROMERTRAG"].sum()}, df_agg_pq[STROMERTRAG_class4up] - roof_kat[STROMERTRAG_class4up]')
        export_txt.write(f'\n total "STROMERTRAG_class5up": {df_agg_pq.loc[df_agg_pq["KLASSE"] >= 5, "STROMERTRAG"].sum() - roof_kat.loc[roof_kat["KLASSE"] >= 5, "STROMERTRAG"].sum()}, df_agg_pq[STROMERTRAG_class5up] - roof_kat[STROMERTRAG_class5up]')

        print(f'** sanity check: STROMERTRAG - DIFFERENCES IN PERCENT')
        export_txt.write(f'\n\n - DIFFERENCES IN PERCENT')
        export_txt.write(f'\n total "STROMERTRAG":          {(df_agg_pq["STROMERTRAG"].sum() - roof_kat["STROMERTRAG"].sum()) / df_agg_pq["STROMERTRAG"].sum() * 100:.2f}%, df_agg_pq[STROMERTRAG] -roof_kat[STROMERTRAG]')
        export_txt.write(f'\n total "STROMERTRAG_class2up": {(df_agg_pq.loc[df_agg_pq["KLASSE"] >= 2, "STROMERTRAG"].sum() - roof_kat.loc[roof_kat["KLASSE"] >= 2, "STROMERTRAG"].sum()) / df_agg_pq.loc[df_agg_pq["KLASSE"] >= 2, "STROMERTRAG"].sum() * 100:.2f}%, df_agg_pq[STROMERTRAG_class2up] - roof_kat[STROMERTRAG_class2up]')
        export_txt.write(f'\n total "STROMERTRAG_class3up": {(df_agg_pq.loc[df_agg_pq["KLASSE"] >= 3, "STROMERTRAG"].sum() - roof_kat.loc[roof_kat["KLASSE"] >= 3, "STROMERTRAG"].sum()) / df_agg_pq.loc[df_agg_pq["KLASSE"] >= 3, "STROMERTRAG"].sum() * 100:.2f}%, df_agg_pq[STROMERTRAG_class3up] - roof_kat[STROMERTRAG_class3up]')
        export_txt.write(f'\n total "STROMERTRAG_class4up": {(df_agg_pq.loc[df_agg_pq["KLASSE"] >= 4, "STROMERTRAG"].sum() - roof_kat.loc[roof_kat["KLASSE"] >= 4, "STROMERTRAG"].sum()) / df_agg_pq.loc[df_agg_pq["KLASSE"] >= 4, "STROMERTRAG"].sum() * 100:.2f}%, df_agg_pq[STROMERTRAG_class4up] - roof_kat[STROMERTRAG_class4up]')
        export_txt.write(f'\n total "STROMERTRAG_class5up": {(df_agg_pq.loc[df_agg_pq["KLASSE"] >= 5, "STROMERTRAG"].sum() - roof_kat.loc[roof_kat["KLASSE"] >= 5, "STROMERTRAG"].sum()) / df_agg_pq.loc[df_agg_pq["KLASSE"] >= 5, "STROMERTRAG"].sum() * 100:.2f}%, df_agg_pq[STROMERTRAG_class5up] - roof_kat[STROMERTRAG_class5up]')

        print(f'** sanity check: STROMERTRAG - DIFFERENCES IN PERCENT, BY KLASSE')
        export_txt.write(f'\n\n - DIFFERENCES IN PERCENT, BY KLASSE')
        export_txt.write(f'\n total "STROMERTRAG_class1exact": {(df_agg_pq.loc[df_agg_pq["KLASSE"] == 1, "STROMERTRAG"].sum() - roof_kat.loc[roof_kat["KLASSE"] == 1, "STROMERTRAG"].sum()) / df_agg_pq.loc[df_agg_pq["KLASSE"] == 1, "STROMERTRAG"].sum() * 100:.2f}%, df_agg_pq[STROMERTRAG_class1exact] - roof_kat[STROMERTRAG_class1exact]')
        export_txt.write(f'\n total "STROMERTRAG_class2exact": {(df_agg_pq.loc[df_agg_pq["KLASSE"] == 2, "STROMERTRAG"].sum() - roof_kat.loc[roof_kat["KLASSE"] == 2, "STROMERTRAG"].sum()) / df_agg_pq.loc[df_agg_pq["KLASSE"] == 2, "STROMERTRAG"].sum() * 100:.2f}%, df_agg_pq[STROMERTRAG_class2exact] - roof_kat[STROMERTRAG_class2exact]')
        export_txt.write(f'\n total "STROMERTRAG_class3exact": {(df_agg_pq.loc[df_agg_pq["KLASSE"] == 3, "STROMERTRAG"].sum() - roof_kat.loc[roof_kat["KLASSE"] == 3, "STROMERTRAG"].sum()) / df_agg_pq.loc[df_agg_pq["KLASSE"] == 3, "STROMERTRAG"].sum() * 100:.2f}%, df_agg_pq[STROMERTRAG_class3exact] - roof_kat[STROMERTRAG_class3exact]')
        export_txt.write(f'\n total "STROMERTRAG_class4exact": {(df_agg_pq.loc[df_agg_pq["KLASSE"] == 4, "STROMERTRAG"].sum() - roof_kat.loc[roof_kat["KLASSE"] == 4, "STROMERTRAG"].sum()) / df_agg_pq.loc[df_agg_pq["KLASSE"] == 4, "STROMERTRAG"].sum() * 100:.2f}%, df_agg_pq[STROMERTRAG_class4exact] - roof_kat[STROMERTRAG_class4exact]')
        export_txt.write(f'\n total "STROMERTRAG_class5exact": {(df_agg_pq.loc[df_agg_pq["KLASSE"] == 5, "STROMERTRAG"].sum() - roof_kat.loc[roof_kat["KLASSE"] == 5, "STROMERTRAG"].sum()) / df_agg_pq.loc[df_agg_pq["KLASSE"] == 5, "STROMERTRAG"].sum() * 100:.2f}%, df_agg_pq[STROMERTRAG_class5exact] - roof_kat[STROMERTRAG_class5exact]')

        print(f'** sanity check: STROMERTRAG - DIFFERENCES IN PERCENT, BY BFS Municipalities')
        if check_vs_raw_input == False: 
            bfs_list = list(df_agg_pq['BFS_NUMMER'].dropna().unique())
            percent_diff_by_bfs = []
            for bfs_i in bfs_list:
                percent_diff_by_bfs.append((df_agg_pq.loc[df_agg_pq['BFS_NUMMER'] == bfs_i, 'STROMERTRAG'].sum() - roof_kat.loc[roof_kat['BFS_NUMMER'] == bfs_i, 'STROMERTRAG'].sum()) / df_agg_pq.loc[df_agg_pq['BFS_NUMMER'] == bfs_i, 'STROMERTRAG'].sum() * 100)
            
            export_txt.write(f'\n\n - DIFFERENCES IN PERCENT, BY BFS Municipalities')
            export_txt.write(f'\n average difference in STROMERTRAG by bfs: {sum(percent_diff_by_bfs)/len(percent_diff_by_bfs):.2f}%')
            export_txt.write(f'\n standard deviation in STROMERTRAG by bfs: {np.std(percent_diff_by_bfs):.2f}%')
            export_txt.write(f'\n min difference in STROMERTRAG by bfs: {min(percent_diff_by_bfs):.2f}%')
            export_txt.write(f'\n max difference in STROMERTRAG by bfs: {max(percent_diff_by_bfs):.2f}%')
        
        print(f'** sanity check: STROMERTRAG - DIFFERENCES IN PERCENT, ROOF WITH INSTALLATION VS WITHOUT')
        export_txt.write(f'\n\n - PERCENT OF PV PRODUCTION TO POTENTIAL') 
        export_txt.write(f'\n % of STROMERTRAG, pv installed roofs / total, df_agg_pq: {df_agg_pq.loc[df_agg_pq["xtf_id"].notna(), "STROMERTRAG"].sum() / df_agg_pq["STROMERTRAG"].sum() * 100:.2f}%')
        export_txt.write(f'\n % of STROMERTRAG, pv installed roofs / total, class2up : {df_agg_pq.loc[(df_agg_pq["KLASSE"] >= 2) & (df_agg_pq["xtf_id"].notna()), "STROMERTRAG"].sum() / df_agg_pq.loc[df_agg_pq["KLASSE"] >= 2, "STROMERTRAG"].sum() * 100:.2f}%')
        export_txt.write(f'\n % of STROMERTRAG, pv installed roofs / total, class3up : {df_agg_pq.loc[(df_agg_pq["KLASSE"] >= 3) & (df_agg_pq["xtf_id"].notna()), "STROMERTRAG"].sum() / df_agg_pq.loc[df_agg_pq["KLASSE"] >= 3, "STROMERTRAG"].sum() * 100:.2f}%')
        export_txt.write(f'\n % of STROMERTRAG, pv installed roofs / total, class4up : {df_agg_pq.loc[(df_agg_pq["KLASSE"] >= 4) & (df_agg_pq["xtf_id"].notna()), "STROMERTRAG"].sum() / df_agg_pq.loc[df_agg_pq["KLASSE"] >= 4, "STROMERTRAG"].sum() * 100:.2f}%') 
        export_txt.write(f'\n % of STROMERTRAG, pv installed roofs / total, class5up : {df_agg_pq.loc[(df_agg_pq["KLASSE"] >= 5) & (df_agg_pq["xtf_id"].notna()), "STROMERTRAG"].sum() / df_agg_pq.loc[df_agg_pq["KLASSE"] >= 5, "STROMERTRAG"].sum() * 100:.2f}%')

        export_txt.write(f'\n - ')

    """
    # Group by 'BFS_NUMMER' and 'year', and calculate aggregates

    # df_agg_BY_gm_year = df_agg_pq.groupby(['BFS_NUMMER', 'year']).agg(
    #     SB_UUID=('SB_UUID', 'nunique'),
    #     # stromertrag_pot_kwh=('STROMERTRAG', 'sum'),
    #     # stromertrag_pot_class2up_kwh=('STROMERTRAG', lambda x: x[x['KLASSE'] >= 2].sum()),
    #     # stromertrag_pot_class3up_kwh=('STROMERTRAG', lambda x: x[x['KLASSE'] >= 3].sum()),
    #     # stromertrag_pot_class4up_kwh=('STROMERTRAG', lambda x: x[x['KLASSE'] >= 4].sum()),
    #     # stromertrag_pot_class5up_kwh=('STROMERTRAG', lambda x: x[x['KLASSE'] >= 5].sum()),
    #     # stromertrag_pv_kwh=('STROMERTRAG', lambda x: x[x['xtf_id'].notna()].sum()),
    #     # stromertrag_pv_class2up_kwh=('STROMERTRAG', lambda x: x[(df_agg_pq['KLASSE'] >= 2) & (df_agg_pq['xtf_id'].notna())].sum()),
    #     # stromertrag_pv_class3up_kwh=('STROMERTRAG', lambda x: x[(df_agg_pq['KLASSE'] >= 3) & (df_agg_pq['xtf_id'].notna())].sum()),
    #     # stromertrag_pv_class4up_kwh=('STROMERTRAG', lambda x: x[(df_agg_pq['KLASSE'] >= 4) & (df_agg_pq['xtf_id'].notna())].sum()),
    #     # stromertrag_pv_class5up_kwh=('STROMERTRAG', lambda x: x[(df_agg_pq['KLASSE'] >= 5) & (df_agg_pq['xtf_id'].notna())].sum())
    #     ).reset_index()

    # ---------------------------------------------------------------------------------------------

    df_agg_BY_gm_year = df_agg_pq.groupby(['BFS_NUMMER', 'year']).apply(
        lambda group: pd.Series({
            'SB_UUID': group['SB_UUID'].nunique(),
            'stromertrag_pot_kwh': group['STROMERTRAG'].sum(),
            'stromertrag_class2up_kwh': group.loc[group['KLASSE'] >= 2, 'STROMERTRAG'].sum(),
            'stromertrag_class3up_kwh': group.loc[group['KLASSE'] >= 3, 'STROMERTRAG'].sum(),
            'stromertrag_class4up_kwh': group.loc[group['KLASSE'] >= 4, 'STROMERTRAG'].sum(),
            'stromertrag_class5up_kwh': group.loc[group['KLASSE'] >= 5, 'STROMERTRAG'].sum(),
            'stromertrag_pv_kwh': group.loc[group['xtf_id'].notna(), 'STROMERTRAG'].sum(),
            'stromertrag_pv_class2up_kwh': group.loc[(group['KLASSE'] >= 2) & (group['xtf_id'].notna()), 'STROMERTRAG'].sum(),
            'stromertrag_pv_class3up_kwh': group.loc[(group['KLASSE'] >= 3) & (group['xtf_id'].notna()), 'STROMERTRAG'].sum(),
            'stromertrag_pv_class4up_kwh': group.loc[(group['KLASSE'] >= 4) & (group['xtf_id'].notna()), 'STROMERTRAG'].sum(),
            'stromertrag_pv_class5up_kwh': group.loc[(group['KLASSE'] >= 5) & (group['xtf_id'].notna()), 'STROMERTRAG'].sum(),
        })
    )
    """
    # AGGREGATE
    df_agg_BY_GM = df_agg_pq.groupby(['BFS_NUMMER']).agg(
        stromertrag_pot_kwh=('STROMERTRAG', 'sum'),
        stromertrag_pot_kwh_class2up=('STROMERTRAG', lambda x: x[df_agg_pq.loc[x.index, 'KLASSE'] >= 2].sum()),
        stromertrag_pot_kwh_class3up=('STROMERTRAG', lambda x: x[df_agg_pq.loc[x.index, 'KLASSE'] >= 3].sum()),
        stromertrag_pot_kwh_class4up=('STROMERTRAG', lambda x: x[df_agg_pq.loc[x.index, 'KLASSE'] >= 4].sum()),
        stromertrag_pot_kwh_class5up=('STROMERTRAG', lambda x: x[df_agg_pq.loc[x.index, 'KLASSE'] >= 5].sum()),
        ).reset_index()

    # EXPORT
    df_agg_BY_GM.to_csv(f'{data_path}/Lupien_aggregation/agg_solkat_pv_gm_BY_GM.csv', index=False)
    df_agg_BY_GM.to_parquet(f'{data_path}/Lupien_aggregation/agg_solkat_pv_gm_BY_GM.parquet')
    print(f'* export << agg_solkat_pv_gm_BY_GM >> to parquet and csv, aggregated by gm')
    with open(export_txt_name, 'a') as export_txt:
        export_txt.write(f'\n\n *export << agg_solkat_pv_gm_BY_GM >> to parquet and csv, time: {datetime.now()}')

    # Sanity check 2
    with open(export_txt_name, 'a') as export_txt:
        print(f'** sanity check2: STROMERTRAG - DIFFERENCES IN PERCENT')
        export_txt.write(f'\n\n - df_exported vs raw data in DIFFERENCES IN PERCENT')
        export_txt.write(f'\n total "STROMERTRAG":          {(df_agg_BY_GM["stromertrag_pot_kwh"].sum() - roof_kat["STROMERTRAG"].sum())/roof_kat["STROMERTRAG"].sum()}, (df_agg_BY_GM -roof_kat) / roof_kat')
        export_txt.write(f'\n total "STROMERTRAG_class2up": {(df_agg_BY_GM["stromertrag_pot_kwh_class2up"].sum() - roof_kat.loc[roof_kat["KLASSE"] >= 2, "STROMERTRAG"].sum())/roof_kat.loc[roof_kat["KLASSE"] >= 2, "STROMERTRAG"].sum()}, (df_agg_BY_GM -roof_kat) / roof_kat')
        export_txt.write(f'\n total "STROMERTRAG_class3up": {(df_agg_BY_GM["stromertrag_pot_kwh_class3up"].sum() - roof_kat.loc[roof_kat["KLASSE"] >= 3, "STROMERTRAG"].sum())/roof_kat.loc[roof_kat["KLASSE"] >= 3, "STROMERTRAG"].sum()}, (df_agg_BY_GM -roof_kat) / roof_kat')
        export_txt.write(f'\n total "STROMERTRAG_class4up": {(df_agg_BY_GM["stromertrag_pot_kwh_class4up"].sum() - roof_kat.loc[roof_kat["KLASSE"] >= 4, "STROMERTRAG"].sum())/roof_kat.loc[roof_kat["KLASSE"] >= 4, "STROMERTRAG"].sum()}, (df_agg_BY_GM -roof_kat) / roof_kat')
        export_txt.write(f'\n total "STROMERTRAG_class5up": {(df_agg_BY_GM["stromertrag_pot_kwh_class5up"].sum() - roof_kat.loc[roof_kat["KLASSE"] >= 5, "STROMERTRAG"].sum())/roof_kat.loc[roof_kat["KLASSE"] >= 5, "STROMERTRAG"].sum()}, (df_agg_BY_GM -roof_kat) / roof_kat')



    # EXPORT: aggregate data by municipality AND YEAR ------------------------------------------------

    print(f'* start aggregation <<agg_solkat_pv_gm_gwr_heat_BY_gm_YEAR>>, time: {datetime.now()}')

    # Sanity check
    pv['BeginningOfOperation'] = pd.to_datetime(pv['BeginningOfOperation'])
    with open(export_txt_name, 'a') as export_txt:
        export_txt.write(f'\n\n{10*"-"}\n sanity checks df_agg_pg by municipalities and year;  time: {datetime.now()} \n{10*"-"}')
        
        print(f'** sanity check: number of installations per year')
        export_txt.write(f'\n\n\n- N UNIQUE INSTALLATIONS, by YEAR')
        export_txt.write(f'\n nunique inst 2015+, df_agg_pq: {df_agg_pq.loc[df_agg_pq["year"] >= 2015, "xtf_id"].nunique()}, pv: {pv.loc[pv["BeginningOfOperation"] >= "2015-01-01", "xtf_id"].nunique()}')
        export_txt.write(f'\n nunique inst 2016+, df_agg_pq: {df_agg_pq.loc[df_agg_pq["year"] >= 2016, "xtf_id"].nunique()}, pv: {pv.loc[pv["BeginningOfOperation"] >= "2016-01-01", "xtf_id"].nunique()}')
        export_txt.write(f'\n nunique inst 2017+, df_agg_pq: {df_agg_pq.loc[df_agg_pq["year"] >= 2017, "xtf_id"].nunique()}, pv: {pv.loc[pv["BeginningOfOperation"] >= "2017-01-01", "xtf_id"].nunique()}')
        export_txt.write(f'\n nunique inst 2018+, df_agg_pq: {df_agg_pq.loc[df_agg_pq["year"] >= 2018, "xtf_id"].nunique()}, pv: {pv.loc[pv["BeginningOfOperation"] >= "2018-01-01", "xtf_id"].nunique()}')
        export_txt.write(f'\n nunique inst 2019+, df_agg_pq: {df_agg_pq.loc[df_agg_pq["year"] >= 2019, "xtf_id"].nunique()}, pv: {pv.loc[pv["BeginningOfOperation"] >= "2019-01-01", "xtf_id"].nunique()}')
        export_txt.write(f'\n nunique inst 2020+, df_agg_pq: {df_agg_pq.loc[df_agg_pq["year"] >= 2020, "xtf_id"].nunique()}, pv: {pv.loc[pv["BeginningOfOperation"] >= "2020-01-01", "xtf_id"].nunique()}')
        export_txt.write(f'\n nunique inst 2021+, df_agg_pq: {df_agg_pq.loc[df_agg_pq["year"] >= 2021, "xtf_id"].nunique()}, pv: {pv.loc[pv["BeginningOfOperation"] >= "2021-01-01", "xtf_id"].nunique()}')
        export_txt.write(f'\n nunique inst 2022+, df_agg_pq: {df_agg_pq.loc[df_agg_pq["year"] >= 2022, "xtf_id"].nunique()}, pv: {pv.loc[pv["BeginningOfOperation"] >= "2022-01-01", "xtf_id"].nunique()}')
        export_txt.write(f'\n nunqiue inst 2023+, df_agg_pq: {df_agg_pq.loc[df_agg_pq["year"] >= 2023, "xtf_id"].nunique()}, pv: {pv.loc[pv["BeginningOfOperation"] >= "2023-01-01", "xtf_id"].nunique()}')
        export_txt.write(f'\n -')

        print(f'** sanity check: number of installations per year IN PERCENT')
        export_txt.write(f'\n\n- N UNIQUE INSTALLATIONS, by YEAR IN PERCENT')
        export_txt.write(f'\n df_agg_pq[xtf_id, year] / pv[xtf_id, year]: {df_agg_pq.loc[df_agg_pq["year"] >= 2015, "xtf_id"].nunique() / pv.loc[pv["BeginningOfOperation"] >= "2015-01-01", "xtf_id"].nunique() * 100:.2f} %')
        export_txt.write(f'\n df_agg_pq[xtf_id, year] / pv[xtf_id, year]: {df_agg_pq.loc[df_agg_pq["year"] >= 2016, "xtf_id"].nunique() / pv.loc[pv["BeginningOfOperation"] >= "2016-01-01", "xtf_id"].nunique() * 100:.2f} %')
        export_txt.write(f'\n df_agg_pq[xtf_id, year] / pv[xtf_id, year]: {df_agg_pq.loc[df_agg_pq["year"] >= 2017, "xtf_id"].nunique() / pv.loc[pv["BeginningOfOperation"] >= "2017-01-01", "xtf_id"].nunique() * 100:.2f} %')
        export_txt.write(f'\n df_agg_pq[xtf_id, year] / pv[xtf_id, year]: {df_agg_pq.loc[df_agg_pq["year"] >= 2018, "xtf_id"].nunique() / pv.loc[pv["BeginningOfOperation"] >= "2018-01-01", "xtf_id"].nunique() * 100:.2f} %')
        export_txt.write(f'\n df_agg_pq[xtf_id, year] / pv[xtf_id, year]: {df_agg_pq.loc[df_agg_pq["year"] >= 2019, "xtf_id"].nunique() / pv.loc[pv["BeginningOfOperation"] >= "2019-01-01", "xtf_id"].nunique() * 100:.2f} %')
        export_txt.write(f'\n df_agg_pq[xtf_id, year] / pv[xtf_id, year]: {df_agg_pq.loc[df_agg_pq["year"] >= 2020, "xtf_id"].nunique() / pv.loc[pv["BeginningOfOperation"] >= "2020-01-01", "xtf_id"].nunique() * 100:.2f} %')
        export_txt.write(f'\n df_agg_pq[xtf_id, year] / pv[xtf_id, year]: {df_agg_pq.loc[df_agg_pq["year"] >= 2021, "xtf_id"].nunique() / pv.loc[pv["BeginningOfOperation"] >= "2021-01-01", "xtf_id"].nunique() * 100:.2f} %')
        export_txt.write(f'\n df_agg_pq[xtf_id, year] / pv[xtf_id, year]: {df_agg_pq.loc[df_agg_pq["year"] >= 2022, "xtf_id"].nunique() / pv.loc[pv["BeginningOfOperation"] >= "2022-01-01", "xtf_id"].nunique() * 100:.2f} %')
        export_txt.write(f'\n df_agg_pq[xtf_id, year] / pv[xtf_id, year]: {df_agg_pq.loc[df_agg_pq["year"] >= 2023, "xtf_id"].nunique() / pv.loc[pv["BeginningOfOperation"] >= "2023-01-01", "xtf_id"].nunique() * 100:.2f} %')
        export_txt.write(f'\n -')
        
        print(f'** sanity check: number of installations per year IN PERCENT, BY KLASSE')
        export_txt.write(f'\n\n- STROMERTRAG FOR INSTALLATIONS, for some years and classes')
        export_txt.write(f'\n** sanity check: PRODUCTION POTENTIAL on INSTALLED ROOFS class3up, in 2020+')
        export_txt.write(f'\n "STROMERTRAG" potentail on PVInst, class3 up, 2019:            {df_agg_pq.loc[(df_agg_pq["year"] == 2019) & (df_agg_pq["KLASSE"] >= 3), "STROMERTRAG"].sum()}')
        export_txt.write(f'\n "STROMERTRAG" potentail on PVInst, class3 up, 2019, by xtf_id: {df_agg_pq.loc[(df_agg_pq["year"] == 2019) & (df_agg_pq["KLASSE"] >= 3) & (df_agg_pq["xtf_id"].notna()), "STROMERTRAG"].sum()}')
        export_txt.write(f'\n "STROMERTRAG" potentail on PVInst, class3 up, 2020:            {df_agg_pq.loc[(df_agg_pq["year"] == 2020) & (df_agg_pq["KLASSE"] >= 3), "STROMERTRAG"].sum()}')
        export_txt.write(f'\n "STROMERTRAG" potentail on PVInst, class3 up, 2020, by xtf_id: {df_agg_pq.loc[(df_agg_pq["year"] == 2020) & (df_agg_pq["KLASSE"] >= 3) & (df_agg_pq["xtf_id"].notna()), "STROMERTRAG"].sum()}')
        export_txt.write(f'\n "STROMERTRAG" potentail on PVInst, class4 up, 2019:            {df_agg_pq.loc[(df_agg_pq["year"] == 2019) & (df_agg_pq["KLASSE"] >= 4), "STROMERTRAG"].sum()}')
        export_txt.write(f'\n "STROMERTRAG" potentail on PVInst, class4 up, 2019, by xtf_id: {df_agg_pq.loc[(df_agg_pq["year"] == 2019) & (df_agg_pq["KLASSE"] >= 4) & (df_agg_pq["xtf_id"].notna()), "STROMERTRAG"].sum()}')
        export_txt.write(f'\n "STROMERTRAG" potentail on PVInst, class4 up, 2020:            {df_agg_pq.loc[(df_agg_pq["year"] == 2020) & (df_agg_pq["KLASSE"] >= 4), "STROMERTRAG"].sum()}')
        export_txt.write(f'\n "STROMERTRAG" potentail on PVInst, class4 up, 2020, by xtf_id: {df_agg_pq.loc[(df_agg_pq["year"] == 2020) & (df_agg_pq["KLASSE"] >= 4) & (df_agg_pq["xtf_id"].notna()), "STROMERTRAG"].sum()}')



    # AGGREGATE
    df_agg_BY_gm_YR = df_agg_pq.groupby(['BFS_NUMMER', 'year']).agg(
        nunique_xtf_id=('xtf_id', 'nunique'),
        stromertrag_pot_kwh_PVinst = ('STROMERTRAG', 'sum'),
        stromertrag_pot_kwh_PVinst_class2up = ('STROMERTRAG', lambda x: x[df_agg_pq.loc[x.index, 'KLASSE'] >= 2].sum()),
        stromertrag_pot_kwh_PVinst_class3up = ('STROMERTRAG', lambda x: x[df_agg_pq.loc[x.index, 'KLASSE'] >= 3].sum()),
        stromertrag_pot_kwh_PVinst_class4up = ('STROMERTRAG', lambda x: x[df_agg_pq.loc[x.index, 'KLASSE'] >= 4].sum()),
        stromertrag_pot_kwh_PVinst_class5up = ('STROMERTRAG', lambda x: x[df_agg_pq.loc[x.index, 'KLASSE'] >= 5].sum())
        ).reset_index()

    # EXPORT
    df_agg_BY_gm_YR.to_csv(f'{data_path}/Lupien_aggregation/agg_solkat_pv_gm_BY_gm_YR.csv', index=False)
    df_agg_BY_gm_YR.to_parquet(f'{data_path}/Lupien_aggregation/agg_solkat_pv_gm_BY_gm_YR.parquet')
    print(f'* export << agg_solkat_pv_gm_BY_gm_YR >> to parquet and csv, aggregated by gm and year')
    with open(export_txt_name, 'a') as export_txt:
        export_txt.write(f'\n\n *export << agg_solkat_pv_gm_BY_gm_YR >> to parquet and csv, time: {datetime.now()}')

    print(f'\n\n ***** END SCRIPT ***** \t time: {datetime.now()}')

        
