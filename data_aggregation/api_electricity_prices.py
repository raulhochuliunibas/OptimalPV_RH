import sys
import os
import pandas as pd
import geopandas as gpd
import time

import json
import re
import string

# import folium
# import mapclassify
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

from graphly.api_client import SparqlClient

sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile

# ------------------------------------------------------------------------------------------------------
# API DATA IMPORT
# ------------------------------------------------------------------------------------------------------

# SWISS ELECTRICITY PRICES ------------------------------------------------------
#> https://jupyter.zazuko.com/electricity_prices.html
#   copied code from: https://colab.research.google.com/github/zazuko/notebooks/blob/master/notebooks/electricity_prices/electricity_prices.ipynb
#> https://lindas.admin.ch/sparql/#
def api_electricity_prices_data(
        dataagg_settings_def, ):
    '''
    This function imports electricity prices from the Swiss (government) ELCOM API.
    The data is index by municipality and year and tariff type and in the data folder.
    ''' 
    # import settings + setup -------------------
    script_run_on_server_def = dataagg_settings_def['script_run_on_server']
    bfs_number_def = dataagg_settings_def['bfs_numbers']
    year_range_def = dataagg_settings_def['year_range']
    smaller_import_def = dataagg_settings_def['smaller_import']
    show_debug_prints_def = dataagg_settings_def['show_debug_prints']
    log_file_name_def = dataagg_settings_def['log_file_name']
    wd_path_def = dataagg_settings_def['wd_path']
    data_path_def = dataagg_settings_def['data_path']
    print_to_logfile(f'run function: api_electricity_prices.py', log_file_name_def=log_file_name_def)

    # query -------------------
    checkpoint_to_logfile(f'api call yearly electricity prices: {year_range_def[0]} to {year_range_def[1]}', log_file_name_def=log_file_name_def, n_tabs_def = 6)
    year_range_list = list(range(year_range_def[0], year_range_def[1]+1))
    tariffs_agg_list = []

    sparql = SparqlClient("https://lindas.admin.ch/query")
    sparql.add_prefixes({
        "schema": "<http://schema.org/>",
        "cube": "<https://cube.link/>",
        "elcom": "<https://energy.ld.admin.ch/elcom/electricityprice/dimension/>",
        "admin": "<https://schema.ld.admin.ch/>"
    })

    for y in year_range_list:
        query = f"""
        SELECT ?municipality_id ?category ?energy ?grid ?aidfee (?community_fees + ?aidfee as ?taxes) ?fixcosts ?variablecosts 
        FROM <https://lindas.admin.ch/elcom/electricityprice>
        WHERE {{
            <https://energy.ld.admin.ch/elcom/electricityprice/observation/> cube:observation ?observation.
            
            ?observation
            elcom:category/schema:name ?category;
            elcom:municipality ?municipality_id;
            elcom:period "{y}"^^<http://www.w3.org/2001/XMLSchema#gYear>;
            elcom:product <https://energy.ld.admin.ch/elcom/electricityprice/product/standard>;
            elcom:fixcosts ?fixcosts;
            elcom:total ?variablecosts;
            elcom:gridusage ?grid;
            elcom:energy ?energy;
            elcom:charge ?community_fees;
            elcom:aidfee ?aidfee.
            
        }}
        ORDER BY ?municipality_id ?category ?variablecosts
        """

        tariffs_response = sparql.send_query(query)
        tariffs_y = tariffs_response.groupby(["municipality_id", "category"]).first().reset_index()
        tariffs_y["year"] = y
        tariffs_agg_list = tariffs_agg_list + [tariffs_y]
        checkpoint_to_logfile(f'api call year: {y} completed', log_file_name_def=log_file_name_def, n_tabs_def = 3, show_debug_prints_def=show_debug_prints_def)
        
    elecpri = pd.concat(tariffs_agg_list)

    # output transformations 
    elecpri["bfs_number"] = elecpri["municipality_id"].str.extract('https://ld.admin.ch/municipality/(\d+)')

    # subselect only relevant bfs numbers (faster to leave the API call whole for Switzerland and filter later)
    bfs_numbers_str = [str(i) for i in bfs_number_def]     # transform bfs_number_def to string
    elecpri = elecpri[elecpri["bfs_number"].isin(bfs_numbers_str)]

    # export
    elecpri.to_parquet(f'{data_path_def}/output/preprep_data/elecpri.parquet')
    elecpri.to_csv(f'{data_path_def}/output/preprep_data/elecpri.csv', index=False)
    checkpoint_to_logfile(f'exported electricity prices', log_file_name_def=log_file_name_def, n_tabs_def = 5)

