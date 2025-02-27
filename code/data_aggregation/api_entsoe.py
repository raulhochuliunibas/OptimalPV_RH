import sys
import os
import pandas as pd
import geopandas as gpd
import requests
import xml.etree.ElementTree as ET

from datetime import datetime, timedelta

sys.path.append('..')
from auxiliary_functions import chapter_to_logfile, subchapter_to_logfile, checkpoint_to_logfile, print_to_logfile

sys.path.append(r'C:/Models/OptimalPV_RH_apikeys')
from apikeys import get_entsoe_key

print(get_entsoe_key)

# ------------------------------------------------------------------------------------------------------
# API DATA IMPORT
# ------------------------------------------------------------------------------------------------------

# ENTSO-E, DAY AHEAD PRICES ------------------------------------------------------
#> general guide url: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
#>>> useful codes at the very end of the page
#> implementation guide pdf: https://transparency.entsoe.eu/content/static_content/download?path=/Static%20content/web%20api/RestfulAPI_IG.pdf

def api_entsoe_ahead_elecpri_data(
        dataagg_settings_def, ):
    '''
    Access ENTSO-E API to get day-ahead electricity prices (for Switzerland?).
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
    print_to_logfile(f'run function: api_pvtarif.py', log_file_name_def=log_file_name_def)


    # create directory in input folder -------------------
    if not os.path.exists(f'{data_path_def}/input_api'):
        os.makedirs(f'{data_path_def}/input_api')


    # year loop -------------------
    # necessary because api only ansers 1 year at a time
    market_elecpri = pd.DataFrame() # empty df to be filled with every loop later
    year_list = range(year_range_def[0], year_range_def[1]+1)
    for year in year_list:

        # API query -------------------
        # url parts for whole query
        docType = 'documentType=A44'                 # A44: Day-ahead prices
        in_Domain = 'in_Domain=10YCH-SWISSGRIDZ'  # Switzerland (and other regions / countries ?)
        out_Domain = 'out_Domain=10YCH-SWISSGRIDZ'
        periodStart = f'periodStart={year}01010000' if not smaller_import_def else f'periodStart={year}12290000'
        periodEnd =   f'periodEnd={year}12310000'   if not smaller_import_def else f'periodEnd={year}12310000'
        req_url = f'https://web-api.tp.entsoe.eu/api?securityToken={get_entsoe_key()}&{docType}&{in_Domain}&{out_Domain}&{periodStart}&{periodEnd}'      # req_url = f'https://web-api.tp.entsoe.eu/api?securityToken={get_entsoe_key()}&documentType=A44&in_Domain=10YCZ-CEPS-----N&out_Domain=10YCZ-CEPS-----N&periodStart=201612302300&periodEnd=201612312300'

        # query
        checkpoint_to_logfile(f'start api call ENTSO-E, year: {year}', log_file_name_def=log_file_name_def, n_tabs_def = 2)
        response = requests.get(req_url)
        response.status_code
        checkpoint_to_logfile(f'response.status_code: {response.status_code}', log_file_name_def=log_file_name_def, n_tabs_def = 3, show_debug_prints_def=show_debug_prints_def)

        # export response to check content
        # with open(f'{data_path_def}/output/preprep_data/entsoe_response.txt', 'w') as f:
        with open(f'{data_path_def}/input_api/entsoe_response.txt', 'w') as f:
            f.write(response.text)

        # extract data from xml response
        root = ET.fromstring(response.text)
        ns = {'ns': 'urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0'}     # Namespace

        # Initialize the result dictionary
        results_dict = {}
        currency_list = []
        price_measurement_unit_list = []
        in_domain_list = []

        resp_timestamp_list = []
        resp_prices_list = []

        # Iterate over each TimeSeries
        timeseries_all = root.findall('ns:TimeSeries', ns)
        if not timeseries_all:
            raise Exception("No TimeSeries elements found in the response")
        elif len(timeseries_all) > 1:
            ts = timeseries_all[0]
    
        for ts in root.findall('ns:TimeSeries', ns):
            currency_list.append(ts.find('ns:currency_Unit.name', ns).text)
            price_measurement_unit_list.append(ts.find('ns:price_Measure_Unit.name', ns).text)
            in_domain_list.append(ts.find('ns:in_Domain.mRID', ns).text)

            period = ts.find('ns:Period', ns)
            time_interval = period.find('ns:timeInterval', ns)
            start_time = time_interval.find('ns:start', ns).text
            end_time = time_interval.find('ns:end', ns).text
            
            points_all = period.findall('ns:Point', ns)
            start_time_dt = datetime.fromisoformat(start_time)
            point = points_all[0]

            for point in period.findall('ns:Point', ns):
                position = point.find('ns:position', ns).text
                price_amount = point.find('ns:price.amount', ns).text
                current_timestamp = start_time_dt + timedelta(hours=int(position)-1)

                # time_stamp_prices.append({
                #     'timestamp': current_timestamp,
                #     'price_amount': price_amount})
            # timestamps_prices.append((current_timestamp.isoformat(), price_amount))
            resp_timestamp_list.append(current_timestamp)
            resp_prices_list.append(price_amount)

        # create dataframe
        market_elecpri_year = pd.DataFrame({'timestamp': resp_timestamp_list, 'price': resp_prices_list})    

        # merge with existing data
        market_elecpri = pd.concat([market_elecpri, market_elecpri_year], axis=0)

    col1 = f'{"_".join(list(set(in_domain_list)))}'
    col2 = f'{"_".join(list(set(currency_list)))}_per_{"_".join(list(set(price_measurement_unit_list)))}'
    market_elecpri.columns = [col1, col2]

    # export
    # market_elecpri.to_parquet(f'{data_path_def}/output/preprep_data/market_elecpri.parquet')
    # market_elecpri.to_csv(f'{data_path_def}/output/preprep_data/market_elecpri.csv', index=False)
    market_elecpri.to_parquet(f'{data_path_def}/input_api/market_elecpri.parquet')
    market_elecpri.to_csv(f'{data_path_def}/input_api/market_elecpri.csv')
    checkpoint_to_logfile(f'exported market_elecpri.parquet', log_file_name_def=log_file_name_def, n_tabs_def = 3, show_debug_prints_def=show_debug_prints_def)


