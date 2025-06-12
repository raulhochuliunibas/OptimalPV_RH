import sys
import os
import pandas as pd
import numpy as np
import json
import winsound
import glob
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

import copy
import plotly.graph_objects as go
import plotly.express as px


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


wd_path = 'C:/Models/OptimalPV_RH'
data_path     = f'{wd_path}/data'
# data_path_def = f'{wd_path}_data'
scen = "pvalloc_BLsml_1roof_extSolkatEGID_12m_meth2.2_rad_dfuid_ind"
name_dir_import = 'preprep_BLSO_22to23_1and2homes'


# CONVERT parquet to csv
if True:

    pq_path = r"C:\Users\hocrau00\Downloads\npv_df_13.parquet"
    file_name = pq_path.split('\\')[-1].split('.parquet')[0]
    csv_path = "\\".join(pq_path.split('\\')[0:-1])

    pq_path.split('\\')[0:-1]

    df  = pd.read_parquet(pq_path)
    # df = df.loc[df['EGID'].isin(['400415',])]
    # df = df.head(8760 * 40)
    df.to_csv(f'{csv_path}/{file_name}.csv')
    print(f'exported {file_name}.csv')



# ------------------------------------------------------------------------------------------------------

if False: 
    # import geopandas as gpd
    import pandas as pd
    import numpy as np
    from shapely.geometry import Point, MultiPoint
    import plotly.express as px
    import json

    # Step 1: Create sample GeoDataFrame with random points in central Europe
    np.random.seed(42)
    lons = np.random.uniform(5.0, 10.0, 10)
    lats = np.random.uniform(46.0, 49.0, 10)
    points = [Point(lon, lat) for lon, lat in zip(lons, lats)]

    gdf = gpd.GeoDataFrame({
        'id': range(1, 11),
        'name': [f"Point {i}" for i in range(1, 11)],
    }, geometry=points, crs='EPSG:4326')

    # Step 2: Create convex hull polygon
    multipoint = MultiPoint(gdf.geometry.tolist())
    hull_polygon = multipoint.convex_hull

    hull_gdf = gpd.GeoDataFrame(geometry=[hull_polygon], crs='EPSG:4326')

    # Step 3: Convert to GeoJSON for plotly
    geojson_hull = json.loads(hull_gdf.to_json())

    # Step 4: Plot hull using px.choropleth_mapbox
    fig = px.choropleth_mapbox(
        hull_gdf,
        geojson=geojson_hull,
        locations=hull_gdf.index.astype(str),
        color_discrete_sequence=["lightblue"],
        center=dict(lat=gdf.geometry.y.mean(), lon=gdf.geometry.x.mean()),
        zoom=6,
        opacity=0.4,
        mapbox_style="carto-positron"
    )

    # Optional: Add the original points for reference
    import plotly.graph_objects as go
    fig.add_trace(go.Scattermapbox(
        lat=gdf.geometry.y,
        lon=gdf.geometry.x,
        mode='markers',
        marker=dict(size=8, color='red'),
        name='Points'
    ))

    # Compute the median longitude
    median_lon = gdf.geometry.x.median()

    # Assign groups based on longitude
    gdf['group'] = np.where(gdf.geometry.x <= median_lon, 'A', 'B')


    # Define color mapping
    group_colors = {'A': 'green', 'B': 'orange'}

    from shapely.geometry import MultiPoint

    # Create two GeoDataFrames by group
    gdf_A = gdf[gdf['group'] == 'A']
    gdf_B = gdf[gdf['group'] == 'B']

    # Generate convex hulls for each group
    hull_A = MultiPoint(gdf_A.geometry.tolist()).convex_hull
    hull_B = MultiPoint(gdf_B.geometry.tolist()).convex_hull

    # Create GeoDataFrames for each hull
    hull_A_gdf = gpd.GeoDataFrame(geometry=[hull_A], crs='EPSG:4326')
    hull_B_gdf = gpd.GeoDataFrame(geometry=[hull_B], crs='EPSG:4326')

    import json

    hull_A_geojson = json.loads(hull_A_gdf.to_json())
    hull_B_geojson = json.loads(hull_B_gdf.to_json())

    # Extract trace from px figure
    hull_A_trace = px.choropleth_mapbox(
        hull_A_gdf,
        geojson=hull_A_geojson,
        locations=hull_A_gdf.index.astype(str),
        color_discrete_sequence=["rgba(0, 255, 0, 0.3)"],
    ).data[0]
    hull_A_trace.name = "Hull A"
    fig.add_trace(hull_A_trace)

    # Same for Hull B
    hull_B_trace = px.choropleth_mapbox(
        hull_B_gdf,
        geojson=hull_B_geojson,
        locations=hull_B_gdf.index.astype(str),
        color_discrete_sequence=["rgba(255, 165, 0, 0.3)"],
    ).data[0]
    hull_B_trace.name = "Hull B"
    fig.add_trace(hull_B_trace)

    fig.show()





# ------------------------------------------------------------------------------------------------------
if False: 
    wd_path = 'C:/Models/OptimalPV_RH'
    data_path     = f'{wd_path}/data'
    # data_path_def = f'{wd_path}_data'
    scen = "pvalloc_BLsml_10y_f2013_1mc_meth2.2_rnd"

    subdf_selected_list = []
    pq_paths = glob.glob(f'{data_path}/pvalloc/{scen}/topo_time_subdf/*.parquet')
    path = pq_paths[0]
    for path in pq_paths:
        subdf = pd.read_parquet(path)
        # subdf = subdf.loc[
        subdf['df_uid']

    pq_path = r"C:\Models\OptimalPV_RH\data\pvalloc\pvalloc_BLsml_10y_f2013_1mc_meth2.2_rnd\topo_time_subdf\topo_subdf_0to399.parquet"
    file_name = pq_path.split('\\')[-1].split('.parquet')[0]
    csv_path = "\\".join(pq_path.split('\\')[0:-1])

    pq_path.split('\\')[0:-1]

    df  = pd.read_parquet(pq_path)
    df = df.head(8760 * 40)
    df.to_csv(f'{csv_path}/{file_name}.csv')
    print(f'exported {file_name}.csv')

# ------------------------------------------------------------------------------------------------------
if False:
    # pv_all_gdf_raw = gpd.read_file(f'{data_path}/input/ch.bfe.elektrizitaetsproduktionsanlagen_gpkg/ch.bfe.elektrizitaetsproduktionsanlagen.gpkg')
    # pv_all_gdf = gpd.read_file(f'{data_path}/input_split_data_geometry/pv_bsblso_geo.geojson')
    def get_bfsnr_name_tuple_list(bfs_number_list=None):
            
        BFS_NUMMER_BL_tuple = [
            (2761, 'Aesch (BL)'),
            (2762, 'Allschwil'),
            (2841, 'Anwil'),
            (2881, 'Arboldswil'),
            (2821, 'Arisdorf'),
            (2763, 'Arlesheim'),
            (2822, 'Augst'),
            (2842, 'Bennwil'),
            (2764, 'Biel-Benken'),
            (2765, 'Binningen'),
            (2766, 'Birsfelden'),
            (2781, 'Blauen'),
            (2842, 'Böckten'),
            (2767, 'Bottmingen'),
            (2883, 'Bretzwil'),
            (2782, 'Brislach'),
            (2823, 'Bubendorf'),
            (2843, 'Buckten'),
            (2783, 'Burg im Leimental'),
            (2844, 'Buus'),
            (2884, 'Diegten'),
            (2845, 'Diepflingen'),
            (2784, 'Dittingen'),
            (2785, 'Duggingen'),
            (2885, 'Eptingen'),
            (2768, 'Ettingen'),
            (2824, 'Frenkendorf'),
            (2825, 'Füllinsdorf'),
            (2846, 'Gelterkinden'),
            (2826, 'Giebenach'),
            (2786, 'Grellingen'),
            (2847, 'Häfelfingen'),
            (2848, 'Hemmiken'),
            (2827, 'Hersberg'),
            (2886, 'Hölstein'),
            (2849, 'Itingen'),
            (2850, 'Känerkinden'),
            (2851, 'Kilchberg (BL)'),
            (2887, 'Lampenberg'),
            (2888, 'Langenbruck'),
            (2852, 'Läufelfingen'),
            (2787, 'Laufen'),
            (2828, 'Lausen'),
            (2889, 'Lauwil'),
            (2890, 'Liedertswil'),
            (2788, 'Liesberg'),
            (2829, 'Liestal'),
            (2830, 'Lupsingen'),
            (2853, 'Maisprach'),
            (2769, 'Münchenstein'),
            (2770, 'Muttenz'),
            (2789, 'Nenzlingen'),
            (2891, 'Niederdorf'),
            (2854, 'Nusshof'),
            (2892, 'Oberdorf (BL)'),
            (2771, 'Oberwil (BL)'),
            (2855, 'Oltingen'),
            (2856, 'Ormalingen'),
            (2772, 'Pfeffingen'),
            (2831, 'Pratteln'),
            (2832, 'Ramlinsburg'),
            (2893, 'Reigoldswil'),
            (2773, 'Reinach (BL)'),
            (2857, 'Rickenbach (BL)'),
            (2790, 'Roggenburg'),
            (2791, 'Röschenz'),
            (2858, 'Rothenfluh'),
            (2859, 'Rümlingen'),
            (2860, 'Rünenberg'),
            (2774, 'Schönenbuch'),
            (2833, 'Seltisberg'),
            (2861, 'Sissach'),
            (2862, 'Tecknau'),
            (2863, 'Tenniken'),
            (2775, 'Therwil'),
            (2864, 'Thürnen'),
            (2894, 'Titterten'),
            (2792, 'Wahlen'),
            (2895, 'Waldenburg'),
            (2865, 'Wenslingen'),
            (2866, 'Wintersingen'),
            (2867, 'Wittinsburg'),
            (2868, 'Zeglingen'),
            (2834, 'Ziefen'),
            (2869, 'Zunzgen'),
            (2793, 'Zwingen'),
        ]
        BFS_NUMMER_AG_tuple = [
            (2421, 'Aedermannsdorf'),
            (2511, 'Aeschi (SO)'),
            (2541, 'Balm bei Günsberg'),
            (2422, 'Balsthal'),
            (2611, 'Bärschwil'),
            (2471, 'Bättwil'),
            (2612, 'Beinwil (SO)'),
            (2542, 'Bellach'),
            (2543, 'Bettlach'),
            (2513, 'Biberist'),
            (2445, 'Biezwil'),
            (2514, 'Bolken'),
            (2571, 'Boningen'),
            (2613, 'Breitenbach'),
            (2465, 'Buchegg'),
            (2472, 'Büren (SO)'),
            (2614, 'Büsserach'),
            (2572, 'Däniken'),
            (2516, 'Deitingen'),
            (2517, 'Derendingen'),
            (2473, 'Dornach'),
            (2535, 'Drei Höfe'),
            (2573, 'Dulliken'),
            (2401, 'Egerkingen'),
            (2574, 'Eppenberg-Wöschnau'),
            (2503, 'Erlinsbach (SO)'),
            (2615, 'Erschwil'),
            (2518, 'Etziken'),
            (2616, 'Fehren'),
            (2544, 'Feldbrunnen-St. Niklaus'),
            (2545, 'Flumenthal'),
            (2575, 'Fulenbach'),
            (2474, 'Gempen'),
            (2519, 'Gerlafingen'),
            (2546, 'Grenchen'),
            (2576, 'Gretzenbach'),
            (2617, 'Grindel'),
            (2547, 'Günsberg'),
            (2578, 'Gunzgen'),
            (2579, 'Hägendorf'),
            (2520, 'Halten'),
            (2402, 'Härkingen'),
            (2491, 'Hauenstein-Ifenthal'),
            (2424, 'Herbetswil'),
            (2618, 'Himmelried'),
            (2475, 'Hochwald'),
            (2476, 'Hofstetten-Flüh'),
            (2425, 'Holderbank (SO)'),
            (2523, 'Horriwil'),
            (2523, 'Horriwil'),
            (2548, 'Hubersdorf'),
            (2524, 'Hüniken'),
            (2549, 'Kammersrohr'),
            (2580, 'Kappel (SO)'),
            (2403, 'Kestenholz'),
            (2492, 'Kienberg'),
            (2619, 'Kleinlützel'),
            (2525, 'Kriegstetten'),
            (2550, 'Langendorf'),
            (2426, 'Laupersdorf'),
            (2526, 'Lohn-Ammannsegg'),
            (2551, 'Lommiswil'),
            (2493, 'Lostorf'),
            (2464, 'Lüsslingen-Nennigkofen'),
            (2527, 'Luterbach'),
            (2455, 'Lüterkofen-Ichertswil'),
            (2427, 'Matzendorf'),
            (2620, 'Meltingen'),
            (2457, 'Messen'),
            (2477, 'Metzerlen-Mariastein'),
            (2428, 'Mümliswil-Ramiswil'),
            (2404, 'Neuendorf'),
            (2405, 'Niederbuchsiten'),
            (2495, 'Niedergösgen'),
            (2478, 'Nuglar-St. Pantaleon'),
            (2621, 'Nunningen'),
            (2406, 'Oberbuchsiten'),
            (2553, 'Oberdorf (SO)'),
            (2528, 'Obergerlafingen'),
            (2497, 'Obergösgen'),
            (2529, 'Oekingen'),
            (2407, 'Oensingen'),
            (2581, 'Olten'),
            (2530, 'Recherswil'),
            (2582, 'Rickenbach (SO)'),
            (2554, 'Riedholz'),
            (2479, 'Rodersdorf'),
            (2555, 'Rüttenen'),
            (2461, 'Schnottwil'),
            (2583, 'Schönenwerd'),
            (2480, 'Seewen'),
            (2556, 'Selzach'),
            (2601, 'Solothurn'),
            (2584, 'Starrkirch-Wil'),
            (2499, 'Stüsslingen'),
            (2532, 'Subingen'),
            (2500, 'Trimbach'),
            (2463, 'Unterramsern'),
            (2585, 'Walterswil (SO)'),
            (2586, 'Wangen bei Olten'),
            (2430, 'Welschenrohr-Gänsbrunnen'),
            (2501, 'Winznau'),
            (2502, 'Wisen (SO)'),
            (2481, 'Witterswil'),
            (2408, 'Wolfwil'),
            (2534, 'Zuchwil'),
            (2622, 'Zullwil')
        ]
        
        BFS_all_tuple = BFS_NUMMER_BL_tuple + BFS_NUMMER_AG_tuple
        if isinstance(bfs_number_list, list):
            bfsnr_name_tuple_list = [x for x in BFS_all_tuple if x[0] in bfs_number_list]
        elif bfs_number_list == None:
            bfsnr_name_tuple_list = BFS_all_tuple

        return bfsnr_name_tuple_list

    def flatten_geometry(geom):
        if geom.has_z:
            if geom.geom_type == 'Polygon':
                exterior = [(x, y) for x, y, z in geom.exterior.coords]
                interiors = [[(x, y) for x, y, z in interior.coords] for interior in geom.interiors]
                return Polygon(exterior, interiors)
            elif geom.geom_type == 'MultiPolygon':
                return MultiPolygon([flatten_geometry(poly) for poly in geom.geoms])
        return geom

    pv_df = pd.read_parquet(f'{data_path}/input_split_data_geometry/pv_pq.parquet')

    gwr_bsblso_pq = pd.read_parquet(f'{data_path}/input_split_data_geometry/gwr_bsblso_pq.parquet')
    Map_egid_dsonode = pd.read_parquet(f'{data_path}/preprep/preprep_BLSO_22to23_extSolkatEGID_DFUIDduplicates/Map_egid_dsonode.parquet')

    gwr_in_primeo = gwr_bsblso_pq.loc[gwr_bsblso_pq['EGID'].isin(Map_egid_dsonode['EGID'].unique())]
    bfs_in_primeo = gwr_in_primeo['GGDENR'].unique()


    pv_df.dtypes
    Map_egid_dsonode.dtypes
    pv_in_primeo = copy.deepcopy(pv_df.loc[pv_df['BFS_NUMMER'].isin(bfs_in_primeo)])

    pv_in_primeo.rename(columns={'BeginningOfOperation': 'BeginOp', }, inplace=True)

    pv_in_primeo['BeginOp'] = pd.to_datetime(pv_in_primeo['BeginOp'], format='%Y-%m-%d')
    pv_in_primeo['BeginOp_year'] = pv_in_primeo['BeginOp'].dt.to_period('Y')
    pv_in_primeo = pv_in_primeo.groupby(['BeginOp_year', 'BFS_NUMMER'])['TotalPower'].sum().reset_index().copy()
    pv_in_primeo['BeginOp_year'] = pv_in_primeo['BeginOp_year'].dt.to_timestamp()

    fig = go.Figure()
    for b in pv_in_primeo['BFS_NUMMER'].unique():
        subdf = pv_in_primeo.loc[pv_in_primeo['BFS_NUMMER'] == b]
        b_name = get_bfsnr_name_tuple_list([int(b),])[0][1]
        fig.add_trace(go.Scatter(x=subdf['BeginOp_year'], y=subdf['TotalPower'], mode='lines+markers', name=f'{b}_{b_name}'))

    fig.update_layout(title='PV Total Power per year (all BFS in primeo grid)', xaxis_title='Year', yaxis_title='Total Power')
    fig.show()
    fig.write_html(f'{data_path}/pvinstCap_in_primeo_BFS.html')



    # add gemeinde mapplot
    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp/swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET.shp')
    gm_shp = gm_shp.to_crs('EPSG:4326')
    gm_shp['geometry'] = gm_shp['geometry'].apply(flatten_geometry)

    gm_shp['BFS_NUMMER'] = gm_shp['BFS_NUMMER'].astype(str)
    date_cols = [col for col in gm_shp.columns if (gm_shp[col].dtype == 'datetime64[ns]') or (gm_shp[col].dtype == 'datetime64[ms]')]
    gm_shp.drop(columns=date_cols, inplace=True)
    gm_shp = gm_shp.loc[gm_shp['BFS_NUMMER'].isin(bfs_in_primeo)]

    pv_mrg = pv_in_primeo.merge(gm_shp, how='left', on='BFS_NUMMER')


    t0_row = []
    for bfs in pv_mrg['BFS_NUMMER'].unique():
        subdf = pv_mrg.loc[pv_mrg['BFS_NUMMER'] == bfs]
        subdf_t0 = subdf.loc[subdf['BeginOp_year'] == subdf['BeginOp_year'].min()]
        t0_row.append(subdf_t0)

    pv_t0_gdf = pd.concat(t0_row)
    pv_mrg_gdf = gpd.GeoDataFrame(pv_t0_gdf, geometry=pv_t0_gdf['geometry'], crs=gm_shp.crs)
    pv_mrg_gdf['BeginOp_year'] = pv_mrg_gdf['BeginOp_year'].astype(str)

    pv_mrg_gdf['hover_text'] = pv_mrg_gdf.apply(lambda row: f'BFS: {row["BFS_NUMMER"]}, {get_bfsnr_name_tuple_list([int(row["BFS_NUMMER"]),])[0][1]}<br>t0_TotalPower: {row["TotalPower"]} kWp', axis=1)

    # pv_mrg_gdf = pv_mrg_gdf.to_crs('EPSG:4326')
    geojson = json.loads(pv_mrg_gdf.to_json())


    map = px.choropleth_mapbox()
    for year in pv_mrg_gdf['BeginOp_year'].unique():
        subdf = pv_mrg_gdf.loc[pv_mrg_gdf['BeginOp_year'] == year]
        map.add_trace(
            go.Choroplethmapbox(
                geojson=geojson,
                locations=subdf['BFS_NUMMER'],
                z=subdf['TotalPower'],
                hoverinfo='text',
                hovertemplate=subdf['hover_text'],
                marker_opacity=0.5,
                marker_line_width=0,
                name=str(year),
            )
        )
    map.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=7,
        mapbox_center={"lat": 47.5, "lon": 7.5},
        title_text="PV Total Power per year (all BFS in primeo grid)",
        title_x=0.5,
    )
    map.show()




# instcomp_year_df['BeginOp'] = pd.to_datetime(instcomp_year_df['BeginOp'], format='%Y-%m-%d')
# instcomp_year_df['BeginOp_year'] = instcomp_year_df['BeginOp'].dt.to_period('Y')
# instcomp_year_df = instcomp_year_df.groupby(['BeginOp_year',])['TotalPower'].sum().reset_index().copy()
# instcomp_year_df['BeginOp_year'] = instcomp_year_df['BeginOp_year'].dt.to_timestamp()
# instcomp_year_df['Cumm_TotalPower'] = instcomp_year_df['TotalPower'].cumsum()
# instcomp_year_df['growth_cumm_TotalPower'] = instcomp_year_df['Cumm_TotalPower'].diff() / instcomp_year_df['Cumm_TotalPower'].shift(1) 
# instcomp_year_df[['Cumm_TotalPower', 'growth_cumm_TotalPower']] 




# ------------------------------------------------------------------------------------------------------
if False:
    egid = '410298'

    topo_subdf_pq_list = glob.glob(f'{data_path}/output/{scen}/topo_time_subdf/*.parquet')
    df_list= []
    for f in topo_subdf_pq_list:
        subdf = pd.read_parquet(f)
        if egid in subdf['EGID'].unique():
            #topo_time_subdf
            subdf = subdf.loc[subdf['EGID'] == egid]
            df_list.append(subdf)

            f_name = f.split(f'{data_path}/output/{scen}/topo_time_subdf\\')[1].split('.parquet')[0]
            print(f'EGID: {f_name} -> export csv')

    df = pd.concat(df_list)
    df.to_csv(f'{data_path}/output/{scen}/topo_time_subdf/{f_name}.csv')

    # solkat_month
    solkat_month = pd.read_parquet(f'{data_path}/output/{scen}/solkat_month.parquet')
    df.columns
    dfuid_ls = df['df_uid'].unique()
    solkat_month.loc[solkat_month['DF_UID'].isin(dfuid_ls)].to_csv(f'{data_path}/output/{scen}/topo_time_subdf/solkat_month_{egid}.csv') 


# ------------------------------------------------------------------------------------------------------
# print directory scheme to txt file
if False:
    from pathlib import Path

    # prefix components:
    space =  '    '
    branch = '│   '
    # pointers:
    tee =    '├── '
    last =   '└── '


    def tree(dir_path: Path, prefix: str='', exclude_list = None):
        """A recursive generator, given a directory Path object
        will yield a visual tree structure line by line
        with each line prefixed by the same characters
        """
        if exclude_list is None:
            exclude_list = []

        contents = [p for p in dir_path.iterdir() if p.name not in exclude_list and not p.name.startswith('.')]
        # contents each get pointers that are ├── with a final └── :
        pointers = ['├── '] * (len(contents) - 1) + ['└── ']
        for pointer, path in zip(pointers, contents):
            yield prefix + pointer + path.name
            if path.is_dir():  # extend the prefix and recurse:
                extension = '│   ' if pointer == '├── ' else '    '
                # i.e. space because last, └── , above so no more |
                yield from tree(path, prefix=prefix+extension, exclude_list=exclude_list)

    # Print the directory tree excluding specified directories and those starting with "."
    txt_header = f'** Directory structure for OptimalPV_RH **\n date: {pd.Timestamp.now()}\n\n'

    with open(f'{wd_path}/OptimalPV_RH_directory_structure.txt', 'w') as f:
        f.write(txt_header)
        for line in tree(Path('C:/Models/OptimalPV_RH'), exclude_list=['archiv_no_longer_used']):
            print(line)
            f.write(line + '\n')





# ------------------------------------------------------------------------------------------------------
"""
solkat_preprep_wo_missingEGID_gdf = gpd.read_file(r"C:\Models\OptimalPV_RH_data\output\preprep_BL_22to23_1and2homes\solkat_gdf.geojson")
solkat_preprep_incl_missingEGID_gdf = gpd.read_file(r"C:\Models\OptimalPV_RH_data\output\preprep_BL_22to23_1and2homes_incl_missingEGID\solkat_gdf.geojson")
gwr = pd.read_parquet(r"C:\Models\OptimalPV_RH_data\output\preprep_BL_22to23_1and2homes_incl_missingEGID\gwr.parquet")
# solkat = pd.read_parquet(r"C:\Models\OptimalPV_RH_data\output\preprep_BL_22to23_1and2homes_incl_missingEGID\solkat.parquet")


cols_to_check = ['391293', '391294', '391295', '391296', '391297', ]
# cols_to_check = ['391291', '391290']
subdf_preprep_incl_missingEGID = solkat_preprep_incl_missingEGID_gdf[solkat_preprep_incl_missingEGID_gdf['EGID'].isin(cols_to_check)]
subdf_preprep_wo_missingEGID = solkat_preprep_wo_missingEGID_gdf[solkat_preprep_wo_missingEGID_gdf['EGID'].isin(cols_to_check)]

subdf_preprep_incl_missingEGID.loc[:, ['EGID', 'geometry', 'FLAECHE', 'DF_UID']]
subdf_preprep_wo_missingEGID.loc[:, ['EGID', 'geometry', 'FLAECHE', 'DF_UID']]

subdf_preprep_wo_missingEGID.buffer(-0.5, resolution=16)

isin_ls = ['245054165', ' 245054165', '245054165 ', ' 245054165 ']
gwr.loc[gwr['EGID'].isin(isin_ls)]['EGID'].unique()

solkat_preprep_incl_missingEGID_gdf.loc[solkat_preprep_incl_missingEGID_gdf['EGID'].isin(isin_ls)]['EGID'].unique()
"""

# ------------------------------------------------------------------------------------------------------
"""
gpd.list_layers(f'{data_path_def}/input\solarenergie-eignung-daecher_2056_monthlydata.gpkg\SOLKAT_DACH_MONAT.gpkg')
solkat_month = gpd.read_file(f'{data_path_def}/input\solarenergie-eignung-daecher_2056_monthlydata.gpkg\SOLKAT_DACH_MONAT.gpkg', layer ='SOLKAT_CH_DACH_MONAT', rows = 100000)
month = gpd.read_file(f'{data_path_def}/input\solarenergie-eignung-daecher_2056_monthlydata.gpkg\SOLKAT_DACH_MONAT.gpkg', layer ='MONAT', rows = 1000)

solkat_month.to_csv(f'{wd_path}/solkat_month.csv')
solkat_month.dtypes
type(solkat_month)
solkat_month.head(20)
month.columns
                             
"""
# ------------------------------------------------------------------------------------------------------
"""
topo = json.load(open(f'{data_path_def}/output/pvalloc_run/topo_egid.json', 'r'))

topo[list(topo.keys())[0]].get('pv_inst').get('info_source')

egid_ls, alloc_algorithm_ls = [], []
for k,v in topo.items():
    egid_ls.append(k)
    alloc_algorithm_ls.append(v.get('pv_inst').get('info_source'))

df = pd.DataFrame({'EGID': egid_ls, 'info_source': alloc_algorithm_ls})
df['info_source'].value_counts()
egids = df.loc[df['info_source'] == 'alloc_algorithm', 'EGID'].to_list()
egids 

# -----------------------
subdf_t0['EGID'].isin(egids).sum()
subdf['EGID'].isin(egids).sum()

aggsubdf_combo['EGID'].isin(egids).sum()

npv_df['EGID'].isin(egids).sum()
"""

# ------------------------------------------------------------------------------------------------------
"""

topo = json.load(open(f'{data_path}/output/{scen}/topo_egid.json', 'r'))

# topo characteristics
topo[list(topo.keys())[0]].keys()
topo[list(topo.keys())[0]]['pv_inst']


all_topo_m = glob.glob(f'{data_path}/output/pvalloc_smallBL_SLCTN_npv_weighted/interim_predictions/topo*.json')

for f in all_topo_m: 

    topo = json.load(open(f, 'r'))
    print(f'\ncounts for {f.split("topo_")[-1].split(".json")[0]}')

    egid_list, gklas_list, inst_tf_list, inst_info_list, inst_id_list, beginop_list, power_list = [], [], [], [], [], [], []
    for k,v in topo.items():
        # print(k)
        egid_list.append(k)
        gklas_list.append(v['gwr_info']['gklas'])
        inst_tf_list.append(v['pv_inst']['inst_TF'])
        inst_info_list.append(v['pv_inst']['info_source'])
        if 'xtf_id' in v['pv_inst']:
            inst_id_list.append(v['pv_inst']['xtf_id'])
        else:   
            inst_id_list.append('')
        beginop_list.append(v['pv_inst']['BeginOp'])
        power_list.append(v['pv_inst']['TotalPower'])
    # for ls in [egid_list, gklas_list, inst_tf_list, inst_info_list, inst_id_list, beginop_list]:
    #     print(len(ls))

    topo_df = pd.DataFrame({'egid': egid_list, 'gklas': gklas_list, 'inst_tf': inst_tf_list, 'inst_info': inst_info_list, 
                            'inst_id': inst_id_list, 'beginop': beginop_list, 'power': power_list})
    
"""

if False:
    topo = json.load(open(f'{data_path}/output\pvalloc_smallBL_1y_npv_weighted/topo_egid.json', 'r'))
    egid_list, gklas_list, inst_tf_list, inst_info_list, inst_id_list, beginop_list, power_list = [], [], [], [], [], [], []
    for k,v in topo.items():
        # print(k)
        egid_list.append(k)
        gklas_list.append(v['gwr_info']['gklas'])
        inst_tf_list.append(v['pv_inst']['inst_TF'])
        inst_info_list.append(v['pv_inst']['info_source'])
        if 'xtf_id' in v['pv_inst']:
            inst_id_list.append(v['pv_inst']['xtf_id'])
        else:   
            inst_id_list.append('')
        beginop_list.append(v['pv_inst']['BeginOp'])
        power_list.append(v['pv_inst']['TotalPower'])

    topo_df = pd.DataFrame({'egid': egid_list, 'gklas': gklas_list, 'inst_tf': inst_tf_list, 'inst_info': inst_info_list,
                            'inst_id': inst_id_list, 'beginop': beginop_list, 'power': power_list})

    # topo_df.to_parquet(f'{data_path}/output/pvalloc_run/topo_df.parquet')

    topo_df.to_csv(f'{wd_path}/topo3_df.csv')
# ------------------------------------------------------------------------------------------------------
# Theoretical change in NPV distribution over multiple allocation algorithms
if False:
        
    import statistics as stats
    import matplotlib.pyplot as plt
    import plotly.figure_factory as ff

    def rand_skew_norm(fAlpha, fLocation, fScale):
        sigma = fAlpha / np.sqrt(1.0 + fAlpha**2) 

        afRN = np.random.randn(2)
        u0 = afRN[0]
        v = afRN[1]
        u1 = sigma*u0 + np.sqrt(1.0 -sigma**2) * v 

        if u0 >= 0:
            return u1*fScale + fLocation 
        return (-u1)*fScale + fLocation 

    def randn_skew(N, skew=0.0):
        return [rand_skew_norm(skew, 0, 1) for x in range(N)]

    n_sample = 10**5
    df_before = pd.DataFrame({'skew3': randn_skew(n_sample, 3), 'skew0': randn_skew(n_sample, 0)})
    df_before['stand'] = (df_before['skew3'] / df_before['skew3'].max())
    df_before['id'] = df_before.index

    df = df_before.copy()
    df_pick_list = []
    draws = 1000
    print('\n\nstart loop')
    for i in range(1, draws+1):
        if (i) % (draws/4)==0:
            print(f'{i/ draws * 100}% done')
        rand_num = np.random.uniform(0, 1)
        df['stand'] = (df['skew3'] / df['skew3'].max())
        df['diff_stand_rand'] = abs(df['stand'] - rand_num)
        df_pick  = df[df['diff_stand_rand'] == min(df['diff_stand_rand'])].copy()

        if df_pick.shape[0] > 1:
            print('more than one row picked')
            rand_row = np.random.randint(0, df_pick.shape[0])
            df_pick = df_pick.iloc[rand_row]
        # adjust df
        df_pick_list.append(df_pick)
        df = df.drop(df_pick.index)
    df_picked = pd.concat(df_pick_list)

    # hist_data = [df_before['skew0'],df['skew0'], df_before['stand'], df['stand'],]
    # labels = ['skew0_before', 'skew0_after', 'stand_before', 'stand_after']

    hist_data = [df_before['skew3'],df['skew3'], df_picked['skew3'], df_before['stand'], df['stand'], ]
    labels = ['skew3_before', 'skew3_after', 'skew3_picked', 'stand_before', 'stand_after']
    df_picked['skew3'].var()
    df_picked['stand'].var()

    df_before.shape, df.shape, df_picked.shape
    print('create fig')
    fig = ff.create_distplot(hist_data, labels, bin_size=0.005)
    fig.show()
    print('end loop')

    winsound.Beep(500, 1000)
    winsound.Beep(500, 1000)

# ------------------------------------------------------------------------------------------------------






