import os
import pandas as pd
import geopandas as gpd
import datetime
import glob

from datetime import datetime

# ------------------------------------------------------------------------------
# SETUP
# ------------------------------------------------------------------------------

wd_path = 'C:/Models/OptimalPV_RH'
data_path = f'{wd_path}_data'
os.chdir(wd_path)
os.listdir(data_path)

# create a export txt file for summary outputs
print(f'\n\n ***** AGGREGATION roofkat pv munic ***** \t time: {datetime.now()}')
if not os.path.exists(f'{data_path}/Lupien_aggregation'):
    os.makedirs(f'{data_path}/Lupien_aggregation')

export_txt_name = f'{data_path}/Lupien_aggregation/aggregation_roofkat_pv_munic_log.txt'
with open(export_txt_name, 'w') as export_txt:
    export_txt.write(f'\n')
    export_txt.write(f'\n *************************** \n     SANITY CHECK OUTPUT \n *************************** \n')
    export_txt.write(f'\n* start script: time: {datetime.now()}')

# ------------------------------------------------------------------------------
# DATA IMPORT
# ------------------------------------------------------------------------------

# import raw input data
print(f'\n\n* import raw input data: time: {datetime.now()}')
roof_kat = gpd.read_file(f'{data_path}/input/solarenergie-eignung-daecher_2056.gdb/SOLKAT_DACH_20230221.gdb', layer ='SOLKAT_CH_DACH')
print(f'imported roof_kat, time: {datetime.now()}')
elec_prod = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
pv = elec_prod[elec_prod['SubCategory'] == 'subcat_2'].copy()
print(f'imported pv, time: {datetime.now()}')

# import aggregates 
agg_pq_files = glob.glob(f'{data_path}/agg_solkat_pv_gm_gwr_heat_BY_KT/df3_agg_solkat_pv_gm_gwr_heat_KT*.parquet')
agg_pq_files = [f for f in agg_pq_files if 'selected_gm_shp' not in f]

df_agg_pq = pd.DataFrame()
# f = agg_pq_files[0]
for f in agg_pq_files:
    print(f'import: {f.split("df3_")[-1]}')
    df_read = pd.read_parquet(f)
    drop_cols = ['UUID', 'DATUM_AEND', 'DATUM_ERST', 'ERSTELL_J', 'ERSTELL_M', 'REVISION_J',
       'REVISION_M', 'GRUND_AEND', 'HERKUNFT', 'HERKUNFT_J', 'HERKUNFT_M',
       'OBJEKTART', 'SEE_FLAECH', 'REVISION_Q', 'ICC', 'HIST_NR', 'GEM_TEIL',
       'GEM_FLAECH', 'SHN', ]
    drop_cols = [col for col in drop_cols if col in df_read.columns]
    df_read.drop(columns=drop_cols, inplace=True)
    df_agg_pq = df_agg_pq._append(df_read)

with open(export_txt_name, 'a') as export_txt:
    export_txt.write(f'\n\n* imported all aggregates from parquet: time: {datetime.now()}')



# EXPORT: Select Houses with PV installation ----------------------------------------------------
df_agg_pq_non_nan = df_agg_pq[df_agg_pq['xtf_id'].notna()]
len(df_agg_pq_non_nan['xtf_id'].unique()) 
len(pv['xtf_id'].unique())
missing_xtf_id = pv[~pv['xtf_id'].isin(df_agg_pq_non_nan['xtf_id'].unique())]['xtf_id'].unique()

# Sanity check
with open(export_txt_name, 'a') as export_txt:
    export_txt.write(f'\n\n{10*"-"}\n sanity checks pv;  time: {datetime.now()} \n{10*"-"}')
    export_txt.write(f'\n length xtf_id in pv: {len(pv["xtf_id"].unique())} compared to total length in pv: {pv.shape[0]} => xtf_id is a unique identifier') 
    export_txt.write(f'\n length unique "xtf_id": {len(pv["xtf_id"].unique())} in pv ("raw import") | {len(df_agg_pq["xtf_id"].unique())} df_agg_pq ("aggregated") | {len(df_agg_pq_non_nan["xtf_id"].unique())} df_agg_pq_non_nan ("aggregated, by house shapes")')
    export_txt.write(f'\n number of missing installations, not covered by house shape: {len(missing_xtf_id)}')
    export_txt.write(f'\n omitted xtf_ids: {missing_xtf_id}')


# EXPORT:
df_agg_pq_non_nan.to_csv(f'{data_path}/Lupien_aggregation/agg_solkat_pv_gm_BY_INSTALLATION.csv', index=False)
df_agg_pq_non_nan.to_parquet(f'{data_path}/Lupien_aggregation/agg_solkat_pv_gm_BY_INSTALLATION.parquet')
print(f'export << agg_solkat_pv_gm_BY_INSTALLATION >> to parquet and csv, all roof shapes that contain a pv installation, intersected with gm and roof_kat, ')


# EXPORT: aggregate data by municipality and year ------------------------------------------------

# Convert 'BeginningOfOperation' to datetime and to 'year'
df_agg_pq['BeginningOfOperation'] = pd.to_datetime(df_agg_pq['BeginningOfOperation'])
df_agg_pq['year'] = df_agg_pq['BeginningOfOperation'].dt.year

# Sanity check
with open(export_txt_name, 'a') as export_txt:
    export_txt.write(f'\n\n{10*"-"}\n sanity checks df_agg_pg by municipalities and year;  time: {datetime.now()} \n{10*"-"}')
    export_txt.write(f'\n length unique "SB_UUID": {len(roof_kat["SB_UUID"].unique())} in roof_kat ("raw import") | {len(df_agg_pq["SB_UUID"].unique())} df_agg_pq_non_nan ("aggregated, by house shapes")')
    export_txt.write(f'\n total "STROMERTRAG": {roof_kat["STROMERTRAG"].sum()} in roof_kat ("raw import") | {df_agg_pq["STROMERTRAG"].sum()} df_agg_pq_non_nan ("aggregated, by house shapes")')



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
).reset_index()

# Rename columns
df_agg_BY_gm_year.columns = ['bfs', 'year', 'SB_UUID', 
                             'stromertrag_pot_kwh', 'stromertrag_pot_class2up_kwh', 'stromertrag_pot_class3up_kwh', 'stromertrag_pot_class4up_kwh', 'stromertrag_pot_class5up_kwh',
                             'stromertrag_pv_kwh', 'stromertrag_pv_class2up_kwh', 'stromertrag_pv_class3up_kwh', 'stromertrag_pv_class4up_kwh', 'stromertrag_pv_class5up_kwh']
# EXPORT
df_agg_BY_gm_year.to_csv(f'{data_path}/Lupien_aggregation/agg_solkat_pv_gm_gwr_heat_BY_gm_year.csv', index=False)
df_agg_BY_gm_year.to_parquet(f'{data_path}/Lupien_aggregation/agg_solkat_pv_gm_gwr_heat_BY_gm_year.parquet')



# EXPORT: aggregate heatcool_weights by municipality and year ------------------------------------
# heatcool_dem = pd.read_parquet(f'{data_path}/spatial_intersection_by_gm/heatcool_dem_by_gm.parquet')
# heatcool_dem.info()
# heatcool_dem_BY_gm = heatcool_dem.groupby('BFS_NUMMER')['NEEDHOME'].sum().reset_index()

# # EXPORT
# heatcool_dem_BY_gm.to_csv(f'{data_path}/Lupien_aggregation/heatcool_dem_BY_gm.csv', index=False)
# heatcool_dem_BY_gm.to_parquet(f'{data_path}/Lupien_aggregation/heatcool_dem_BY_gm.parquet')



# BOOKMARK: 
# first run spatial to parquet by gm - done
# aggregate again data by kt, df3 and df5 will be exported - done
# run lupien aggregation with df3 
