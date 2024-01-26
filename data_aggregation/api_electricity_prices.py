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
from functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile

# ------------------------------------------------------------------------------------------------------
# API DATA IMPORT
# ------------------------------------------------------------------------------------------------------

# SWISS ELECTRICITY PRICES ------------------------------------------------------
#> https://jupyter.zazuko.com/electricity_prices.html
#> https://lindas.admin.ch/sparql/#
def api_electricity_prices(
        script_run_on_server_def = None,
        recreate_parquet_files_def = None,
        smaller_import_def = None,
        log_file_name_def = None,
        wd_path_def = None,
        data_path_def = None,
        show_debug_prints_def = None,
        year_range_def = [2017, 2020],
        ):
    '''
    This function imports electricity prices from the Swiss government API.
    The data is aggregated by municipality and year and saved as parquet file in the data folder.
    ''' 

    # setup -------------------
    wd_path = wd_path_def if script_run_on_server_def else "C:/Models/OptimalPV_RH"
    data_path = f'{wd_path}_data'

    # create directory + log file
    if not os.path.exists(f'{data_path}/output/preprep_data'):
        os.makedirs(f'{data_path}/output/preprep_data')
    checkpoint_to_logfile('run function: api_electricity_prices.py', log_file_name_def=log_file_name_def, n_tabs_def = 5) 



    # create query prompt df -------------------
    # query_df = pd.DataFrame({
    #     "year": list(range(1980, 2024)),
    #     "prompt": [0] * len(range(1980, 2024))
    #     })
    # # write out all prompts as string because f strings are not working
    if False:
        query_df.loc[query_df["year"] == 2020, "prompt"] = """
        SELECT ?municipality_id ?category ?energy ?grid ?aidfee (?community_fees + ?aidfee as ?taxes) ?fixcosts ?variablecosts 
        FROM <https://lindas.admin.ch/elcom/electricityprice>
        WHERE {
            <https://energy.ld.admin.ch/elcom/electricityprice/observation/> cube:observation ?observation.
            
            ?observation
            elcom:category/schema:name ?category;
            elcom:municipality ?municipality_id;
            elcom:period "2020"^^<http://www.w3.org/2001/XMLSchema#gYear>;
            elcom:product <https://energy.ld.admin.ch/elcom/electricityprice/product/standard>;
            elcom:fixcosts ?fixcosts;
            elcom:total ?variablecosts;
            elcom:gridusage ?grid;
            elcom:energy ?energy;
            elcom:charge ?community_fees;
            elcom:aidfee ?aidfee.
            
        }
        ORDER BY ?muncipality ?category ?variablecosts
        """
        query_df.loc[query_df["year"] == 2021, "prompt"] = """
        SELECT ?municipality_id ?category ?energy ?grid ?aidfee (?community_fees + ?aidfee as ?taxes) ?fixcosts ?variablecosts 
        FROM <https://lindas.admin.ch/elcom/electricityprice>
        WHERE {
            <https://energy.ld.admin.ch/elcom/electricityprice/observation/> cube:observation ?observation.
            
            ?observation
            elcom:category/schema:name ?category;
            elcom:municipality ?municipality_id;
            elcom:period "2021"^^<http://www.w3.org/2001/XMLSchema#gYear>;
            elcom:product <https://energy.ld.admin.ch/elcom/electricityprice/product/standard>;
            elcom:fixcosts ?fixcosts;
            elcom:total ?variablecosts;
            elcom:gridusage ?grid;
            elcom:energy ?energy;
            elcom:charge ?community_fees;
            elcom:aidfee ?aidfee.
            
        }
        ORDER BY ?municipality_id ?category ?variablecosts
        """

        query_df.loc[query_df["year"] == 2022, "prompt"] = """
        SELECT ?municipality_id ?category ?energy ?grid ?aidfee (?community_fees + ?aidfee as ?taxes) ?fixcosts ?variablecosts 
        FROM <https://lindas.admin.ch/elcom/electricityprice>
        WHERE {
            <https://energy.ld.admin.ch/elcom/electricityprice/observation/> cube:observation ?observation.
            
            ?observation
            elcom:category/schema:name ?category;
            elcom:municipality ?municipality_id;
            elcom:period "2022"^^<http://www.w3.org/2001/XMLSchema#gYear>;
            elcom:product <https://energy.ld.admin.ch/elcom/electricityprice/product/standard>;
            elcom:fixcosts ?fixcosts;
            elcom:total ?variablecosts;
            elcom:gridusage ?grid;
            elcom:energy ?energy;
            elcom:charge ?community_fees;
            elcom:aidfee ?aidfee.
            
        }
        ORDER BY ?municipality_id ?category ?variablecosts
        """ 

        query_df.loc[query_df["year"] == 2023, "prompt"] = """
        SELECT ?municipality_id ?category ?energy ?grid ?aidfee (?community_fees + ?aidfee as ?taxes) ?fixcosts ?variablecosts 
        FROM <https://lindas.admin.ch/elcom/electricityprice>
        WHERE {
            <https://energy.ld.admin.ch/elcom/electricityprice/observation/> cube:observation ?observation.
            
            ?observation
            elcom:category/schema:name ?category;
            elcom:municipality ?municipality_id;
            elcom:period "2023"^^<http://www.w3.org/2001/XMLSchema#gYear>;
            elcom:product <https://energy.ld.admin.ch/elcom/electricityprice/product/standard>;
            elcom:fixcosts ?fixcosts;
            elcom:total ?variablecosts;
            elcom:gridusage ?grid;
            elcom:energy ?energy;
            elcom:charge ?community_fees;
            elcom:aidfee ?aidfee.
            
        }
        ORDER BY ?municipality_id ?category ?variablecosts
        """

        query_df.loc[query_df["year"] == 2019, "prompt"] = """
        SELECT ?municipality_id ?category ?energy ?grid ?aidfee (?community_fees + ?aidfee as ?taxes) ?fixcosts ?variablecosts 
        FROM <https://lindas.admin.ch/elcom/electricityprice>
        WHERE {
            <https://energy.ld.admin.ch/elcom/electricityprice/observation/> cube:observation ?observation.
            
            ?observation
            elcom:category/schema:name ?category;
            elcom:municipality ?municipality_id;
            elcom:period "2019"^^<http://www.w3.org/2001/XMLSchema#gYear>;
            elcom:product <https://energy.ld.admin.ch/elcom/electricityprice/product/standard>;
            elcom:fixcosts ?fixcosts;
            elcom:total ?variablecosts;
            elcom:gridusage ?grid;
            elcom:energy ?energy;
            elcom:charge ?community_fees;
            elcom:aidfee ?aidfee.
            
        }
        ORDER BY ?municipality_id ?category ?variablecosts
        """

        query_df.loc[query_df["year"] == 2018, "prompt"] = """
        SELECT ?municipality_id ?category ?energy ?grid ?aidfee (?community_fees + ?aidfee as ?taxes) ?fixcosts ?variablecosts 
        FROM <https://lindas.admin.ch/elcom/electricityprice>
        WHERE {
            <https://energy.ld.admin.ch/elcom/electricityprice/observation/> cube:observation ?observation.
            
            ?observation
            elcom:category/schema:name ?category;
            elcom:municipality ?municipality_id;
            elcom:period "2018"^^<http://www.w3.org/2001/XMLSchema#gYear>;
            elcom:product <https://energy.ld.admin.ch/elcom/electricityprice/product/standard>;
            elcom:fixcosts ?fixcosts;
            elcom:total ?variablecosts;
            elcom:gridusage ?grid;
            elcom:energy ?energy;
            elcom:charge ?community_fees;
            elcom:aidfee ?aidfee.
            
        }
        ORDER BY ?municipality_id ?category ?variablecosts
        """

        query_df.loc[query_df["year"] == 2017, "prompt"] = """
        SELECT ?municipality_id ?category ?energy ?grid ?aidfee (?community_fees + ?aidfee as ?taxes) ?fixcosts ?variablecosts 
        FROM <https://lindas.admin.ch/elcom/electricityprice>
        WHERE {
            <https://energy.ld.admin.ch/elcom/electricityprice/observation/> cube:observation ?observation.
            
            ?observation
            elcom:category/schema:name ?category;
            elcom:municipality ?municipality_id;
            elcom:period "2017"^^<http://www.w3.org/2001/XMLSchema#gYear>;
            elcom:product <https://energy.ld.admin.ch/elcom/electricityprice/product/standard>;
            elcom:fixcosts ?fixcosts;
            elcom:total ?variablecosts;
            elcom:gridusage ?grid;
            elcom:energy ?energy;
            elcom:charge ?community_fees;
            elcom:aidfee ?aidfee.
            
        }
        ORDER BY ?municipality_id ?category ?variablecosts
        """

        query_df.loc[query_df["year"] == 2016, "prompt"] = """
        SELECT ?municipality_id ?category ?energy ?grid ?aidfee (?community_fees + ?aidfee as ?taxes) ?fixcosts ?variablecosts 
        FROM <https://lindas.admin.ch/elcom/electricityprice>
        WHERE {
            <https://energy.ld.admin.ch/elcom/electricityprice/observation/> cube:observation ?observation.
            
            ?observation
            elcom:category/schema:name ?category;
            elcom:municipality ?municipality_id;
            elcom:period "2016"^^<http://www.w3.org/2001/XMLSchema#gYear>;
            elcom:product <https://energy.ld.admin.ch/elcom/electricityprice/product/standard>;
            elcom:fixcosts ?fixcosts;
            elcom:total ?variablecosts;
            elcom:gridusage ?grid;
            elcom:energy ?energy;
            elcom:charge ?community_fees;
            elcom:aidfee ?aidfee.
            
        }
        ORDER BY ?municipality_id ?category ?variablecosts
        """

        



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
        # query = query_df.loc[query_df["year"] == y, "prompt"].values[0]
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
        # time.sleep(120)
        checkpoint_to_logfile(f'api call year: {y} completed', log_file_name_def=log_file_name_def, n_tabs_def = 3, show_debug_prints_def=show_debug_prints_def)
        
    elecpri = pd.concat(tariffs_agg_list)

    # output transformations 
    elecpri["bfs_number"] = elecpri["municipality_id"].str.extract('https://ld.admin.ch/municipality/(\d+)')

    # export
    elecpri.to_parquet(f'{data_path}/output/preprep_data/elecpri.parquet')
    checkpoint_to_logfile(f'exported electricity prices', log_file_name_def=log_file_name_def, n_tabs_def = 5)

