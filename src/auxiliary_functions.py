import os as os
import geopandas as gpd
import polars as pl     
from typing_extensions import List

from datetime import datetime
from dataclasses import replace


# this is required to set the timer properly
time_last_call = None


def chapter_to_logfile(str_def, log_file_name_def, overwrite_file=False):
    """
    Function to write a chapter to the logfile
    """
    check = f'\n\n****************************************\n {str_def} \n timestamp: {datetime.now()} \n****************************************\n'
    print(check)
    
    log_file_exists_TF = os.path.exists(f'{log_file_name_def}')
    if not log_file_exists_TF or overwrite_file:
        with open(f'{log_file_name_def}', 'w') as log_file:
            log_file.write(f'{check}\n')
    else: 
        with open(f'{log_file_name_def}', 'a') as log_file:
            log_file.write(f'{check}\n')


def subchapter_to_logfile(str_def, log_file_name_def):
    """
    Function to write a subchapter to the logfile
    """
    check = f'\n----------------------------------------\n {str_def} > start at: {datetime.now()} \n----------------------------------------\n'
    print(check)
    
    if not os.path.exists(f'{log_file_name_def}'):
        with open(f'{log_file_name_def}', 'w') as log_file:
            log_file.write(f'{check}\n')
    elif os.path.exists(f'{log_file_name_def}'): 
        with open(f'{log_file_name_def}', 'a') as log_file:
            log_file.write(f'{check}\n')


def print_to_logfile(str_def, log_file_name_def):
    """
    Function to write just plain text to logfile
    """
    check = f'{str_def:60}'
    print(check)

    with open(f'{log_file_name_def}', 'a') as log_file:
        log_file.write(f"{check}\n")


def checkpoint_to_logfile(str_def, log_file_name_def, n_tabs_def = 0, show_debug_prints_def = True):
    """
    Function to write a checkpoint to the logfile, Mainly used for debugging because prints can be switched off
    """ 
    global time_last_call
    
    if show_debug_prints_def:
        time_now = datetime.now()
        if time_last_call:
            # total_seconds = (time_now - time_last_call).total_seconds()
            # hours, remainder_min = divmod(total_seconds, 3600)
            # minutes, remainder_sec = divmod(remainder_min, 60)
            # seconds = int(remainder_sec)
            # milliseconds = int((remainder_sec - seconds) * 1000)
            # runtime_str = f"{hours} hr {minutes} min {seconds} sec {milliseconds} ms"

            runtime_str = f'{time_now - time_last_call} (h:m:s.s)'
            
        else:
            runtime_str = 'N/A'
        
        n_tabs_str = '\t' * n_tabs_def
        check = f' * {str_def:60}{n_tabs_str} > runtime: {runtime_str:18};      (stamp: {datetime.now()})'
        print(check)

        with open(f'{log_file_name_def}', 'a') as log_file:
            log_file.write(f"{check}\n")
        
        time_last_call = time_now


def get_bfs_from_ktnr(
        ktnr_list: List[int] , 
        data_path: str,
        log_name: str
    ) -> List[str]:
    """
    Function to get a list of the BFS numbers for all municipalites within a list of canton numbers
    """
    gm_shp = gpd.read_file(f'{data_path}/input/swissboundaries3d_2023-01_2056_5728.shp', layer ='swissBOUNDARIES3D_1_4_TLM_HOHEITSGEBIET')

    if (isinstance(ktnr_list, list)) and (not not ktnr_list):
        gm_shp['BFS_NUMMER'] = gm_shp['BFS_NUMMER'].astype(str)
        gm_shp_sub = gm_shp[gm_shp['KANTONSNUM'].isin(ktnr_list)]
        # gm_shp_sub['BFS_NUMMER'] = gm_shp_sub['BFS_NUMMER'].astype(int).astype(str)
        # gm_shp_sub.loc[:, 'BFS_NUMMER'] = gm_shp_sub['BFS_NUMMER'].astype(int).astype(str)

        bfs_list = gm_shp_sub['BFS_NUMMER'].unique().tolist()
    else: 
        print_to_logfile(' > ERROR: no canton or bfs selection applicables; NOT used any municipality selection', log_name)

    return bfs_list


def format_MASTER_settings(settings_dict):
    formatted_settings = {}
    for key, value in settings_dict.items():
        if isinstance(value, list):
            formatted_settings[key] = ', '.join(map(str, value))
        elif isinstance(value, dict):
            formatted_settings[key] = format_MASTER_settings(value)
        else:
            formatted_settings[key] = value
    return formatted_settings


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



def add_static_topo_data_to_subdf(subdf, topo):
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
    return subdf


def make_scenario(default_scen, name_dir_export, bfs_numbers=None, **overrides):
    kwargs = {'name_dir_export': name_dir_export}
    if bfs_numbers is not None:
        kwargs['bfs_numbers'] = bfs_numbers
    if overrides:
        kwargs.update(overrides)
    return replace(default_scen, **kwargs)


# ----------------------------------------------------------------------
    
def test_functions(text):
    print(f'return: {text}')



# NO LONGER ACTIVENo longer Active ----------------------------------------------------------------------

def print_directory_stucture_to_txtfile(if_True = False):
    dir_exclusion_list = ['archiv_no_longer_used', '__pycache__', '__init__.py', 'poetry.lock', '__poetry.lock', 'pyproject.toml', '__pyproject.toml', 
                          'README.md', 'selection_mechanism_theory.py', 'ToDos_NextSteps.py', 'TRY_OUT.py', 'OptimalPV_RH_directory_structure.txt',
                          'visualization_MASTER_oldcopy.py', 'x_archiv']
    if if_True:
        import os as os
        import pandas as pd
        from pathlib import Path

        # Prefix components
        space = '    '
        branch = '│   '
        # Pointers
        tee = '├── '
        last = '└── '

        def tree(dir_path: Path, prefix: str='', exclude_list=None):
            """A recursive generator, given a directory Path object
            will yield a visual tree structure line by line
            with each line prefixed by the same characters
            """
            if exclude_list is None:
                exclude_list = []

            contents = [p for p in dir_path.iterdir() if p.name not in exclude_list and not p.name.startswith('.')]
            # Contents each get pointers that are ├── with a final └── :
            pointers = ['├── '] * (len(contents) - 1) + ['└── ']
            for pointer, path in zip(pointers, contents):
                yield prefix + pointer + path.name
                if path.is_dir() and path.name not in exclude_list:  # Extend the prefix and recurse:
                    extension = '│   ' if pointer == '├── ' else '    '
                    # i.e. space because last, └── , above so no more |
                    yield from tree(path, prefix=prefix+extension, exclude_list=exclude_list)

        # Print the directory tree excluding specified directories and those starting with "."
        txt_header = f'** Directory structure for OptimalPV_RH **\n date: {pd.Timestamp.now()}\n\n'

        # with open(f'{os.getcwd()}/OptimalPV_RH_directory_structure.txt', 'w', encoding='utf-8') as f:
        #     f.write(txt_header)
        #     for line in tree(Path('C:/Models/OptimalPV_RH'), exclude_list= dir_exclusion_list):
        #         # print(line)
        #         f.write(line + '\n')


