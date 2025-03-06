import os
import pandas as pd

# import folium
# import mapclassify
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

from graphly.api_client import SparqlClient


# ------------------------------------------------------------------------------------------------------
# API DATA IMPORT
# ------------------------------------------------------------------------------------------------------

# SWISS ELECTRICITY PRICES ------------------------------------------------------
#> https://jupyter.zazuko.com/electricity_prices.html
#   copied code from: https://colab.research.google.com/github/zazuko/notebooks/blob/master/notebooks/electricity_prices/electricity_prices.ipynb
#> https://lindas.admin.ch/sparql/#


# import settings + setup -------------------
print(f'run function: get_api_electricity_prices.py')

# query -------------------
year_range_def = [2015, 2023]
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
    print(f'api call year: {y} completed')
    
elecpri = pd.concat(tariffs_agg_list)

# output transformations 
elecpri["bfs_number"] = elecpri["municipality_id"].str.extract('https://ld.admin.ch/municipality/(\d+)')

# subselect only relevant bfs numbers (faster to leave the API call whole for Switzerland and filter later)
# bfs_numbers_str = [str(i) for i in bfs_number_def]     # transform bfs_number_def to string
# elecpri = elecpri[elecpri["bfs_number"].isin(bfs_numbers_str)]

# export
output_path = 'C:/Models/OptimalPV_RH_data/input/ElCom_consum_price_api_data'

if not os.path.exists(output_path):
    os.makedirs(output_path)
else: 
    for file in os.listdir(output_path):
        os.remove(os.path.join(output_path, file))

elecpri.to_parquet(f'{output_path}/elecpri.parquet')
elecpri.to_csv(f'{output_path}/elecpri.csv', index=False)
