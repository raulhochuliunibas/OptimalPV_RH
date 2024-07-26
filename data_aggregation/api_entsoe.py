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
    tbd
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


    # query -------------------
    # url parts for whole query
    docType = 'documentType=A44'                 # A44: Day-ahead prices
    in_Domain = 'in_Domain=10YCH-SWISSGRIDZ'  # Switzerland (and other regions / countries ?)
    out_Domain = 'out_Domain=10YCH-SWISSGRIDZ'
    periodStart = f'periodStart={year_range_def[0]}01010000' if not smaller_import_def else f'periodStart={year_range_def[0]}12290000'
    periodEnd =   f'periodEnd={year_range_def[1]}12310000'   if not smaller_import_def else f'periodEnd={year_range_def[1]}12310000'
    req_url = f'https://web-api.tp.entsoe.eu/api?securityToken={get_entsoe_key()}&{docType}&{in_Domain}&{out_Domain}&{periodStart}&{periodEnd}'      # req_url = f'https://web-api.tp.entsoe.eu/api?securityToken={get_entsoe_key()}&documentType=A44&in_Domain=10YCZ-CEPS-----N&out_Domain=10YCZ-CEPS-----N&periodStart=201612302300&periodEnd=201612312300'

    # query
    checkpoint_to_logfile(f'start api call ENTSO-E, year: {year_range_def[0]} to {year_range_def[1]}', log_file_name_def=log_file_name_def, n_tabs_def = 6)
    response = requests.get(req_url)
    response.status_code
    checkpoint_to_logfile(f'response.status_code: {response.status_code}', log_file_name_def=log_file_name_def, n_tabs_def = 3, show_debug_prints_def=show_debug_prints_def)

    # ------------------------------------------------------------------------------------------------------
    with open(f'{data_path_def}/output/preprep_data/entsoe_response.txt', 'w') as f:
        f.write(response.text)

    # extract data from xml response
    root = ET.fromstring(response.text)
    # Namespace
    ns = {'ns': 'urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0'}

    # Initialize the result dictionary
    results_dict = {}
    currency_list = []
    price_measurement_unit_list = []
    in_domain_list = []

    timestamps_prices = []

    

    # Iterate over each TimeSeries
    timeseries_all = root.findall('ns:TimeSeries', ns)
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
        timestamps_prices.append((current_timestamp.isoformat(), price_amount))
            
    # write column headers
    col1 = f'{"_".join(list(set(in_domain_list)))}'
    col2 = f'{"_".join(list(set(currency_list)))}_per_{"_".join(list(set(price_measurement_unit_list)))}'
    colnames = [col1, col2]

    # create dataframe
    df = pd.DataFrame(timestamps_prices, columns=colnames)


    

    print('asdf')


"""



        currency = ts.find('ns:currency_Unit.name', ns).text
        price_measurement_unit = ts.find('ns:price_Measure_Unit.name', ns).text
        in_domain = ts.find('ns:in_Domain.mRID', ns).text
        out_domain = ts.find('ns:out_Domain.mRID', ns).text
        
        key = (currency, price_measurement_unit, in_domain, out_domain)
        
        if key not in results_dict:
            results_dict[key] = []
        
        period = ts.find('ns:Period', ns)
        time_interval = period.find('ns:timeInterval', ns)
        start_time = time_interval.find('ns:start', ns).text
        end_time = time_interval.find('ns:end', ns).text
        
        for point in period.findall('ns:Point', ns):
            position = point.find('ns:position', ns).text
            price_amount = point.find('ns:price.amount', ns).text
            results_dict[key].append({
                'position': position,
                'price_amount': price_amount,
                'start_time': start_time,
                'end_time': end_time
            })
        
    # Print the results
    results_dict






    response.status_code == 200 
    response.headers.get('Content-Type')
    # save xml response file to txt file

        
    root = ET.fromstring(response.content)
    for child in root:
        print(child.tag, child.attrib)

    print('asdf')



    # ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------
    
    # year_range_list = [str(year % 100).zfill(2) for year in range(year_range_def[0], year_range_def[1]+1)]
    # ew_id = list(range(1, 200+1)) if smaller_import_def else range(1, 1000) 
    # ew_id_counter = len(ew_id) / 4

    url = "https://opendata.vese.ch/pvtarif/api/getData/evu?"

    for y in year_range_list:
        checkpoint_to_logfile(f'api call pvtarif for year: {y} started', log_file_name_def=log_file_name_def, n_tabs_def = 3, show_debug_prints_def=show_debug_prints_def)

        response_ew_list = []

        for i_ew, ew in enumerate(ew_id):
            req_url = f'{url}evuId={ew}&year={y}&licenseKey={get_pvtarif_key()}'
            response = requests.get(req_url)
            response_json = response.json()
            
            response_ew_list.append(response_json)
            if ew % ew_id_counter == 0:
                checkpoint_to_logfile(f'year: {y}, ew: {ew}, {i_ew+1} of {len(ew_id)} in list', log_file_name_def=log_file_name_def, n_tabs_def = 2, show_debug_prints_def=show_debug_prints_def)

        response_ew_df = pd.DataFrame(response_ew_list)
        response_ew_df['year'] = y
        response_all_df_list.append(response_ew_df)
        checkpoint_to_logfile(f'api call year: {y} completed', log_file_name_def=log_file_name_def, n_tabs_def = 3, show_debug_prints_def=show_debug_prints_def)

    pvtarif_raw = pd.concat(response_all_df_list)

    # output transformations
    pvtarif = pvtarif_raw.loc[pvtarif_raw['valid'] == True].copy()

    # export
    pvtarif.to_parquet(f'{data_path_def}/output/preprep_data/pvtarif.parquet')
    pvtarif.to_csv(f'{data_path_def}/output/preprep_data/pvtarif.csv', index=False)
    checkpoint_to_logfile(f'exported electricity prices', log_file_name_def=log_file_name_def, n_tabs_def = 5)

"""
    

