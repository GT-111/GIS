import yaml
from yaml import Loader
import glob
import os
import pandas as pd
# cudf
import cudf
# cupy
import cupy as cp
import numpy as np

def get_config(config_path:str):
    """
    Parameters:
        config_path: path to the config file.
    Returns:
        config: config file.
    """
    with open(config_path, 'r') as stream:
        config = yaml.load(stream, Loader)

    return config

def files2dict(files):
    """
    Parameters:
        files: list of file paths. Format: '*/aisdk-2023-09-20.*'
    Returns:
        result: dictionary of file paths.
    """
    result = {}
    for file in files:
        yyyy, mm, dd = file.split('/')[-1].split('.')[0].split('-')[1:]
        date = yyyy+'-'+mm+'-'+dd
        result[date] = file
    return result

def get_last_file(path):

    list_of_files = glob.glob(f'{path}/*')
    if len(list_of_files) == 0:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)

    return latest_file

def get_all_files(directory_path):

    files_and_directories = os.listdir(directory_path)
    file_paths = []
    for item in files_and_directories:
        item_path = os.path.join(directory_path, item)  
        if os.path.isfile(item_path):
            file_paths.append(item_path)
    return file_paths

def rename_columns(df, cfg):
    return df.rename(columns=cfg.column_rename_dict)

def filter_missing_value(df, colums_to_drop_na:list):
    """
    Parameters:
        df: AIS data.
        colums_to_drop_na: columns to check and drop null values.

    Returns:
        df: AIS data with missing values filtered.
    """
    return df.dropna(subset=colums_to_drop_na)

def filter_mmsi(df):
    # Note: mmsi start with 2-7 belongs to single vessel, 9 digits
    # Sample for data in 16/09/2023 7191-> 6950
    df['MMSI'] = df['MMSI'].astype(str)

    return df[df['MMSI'].str.match(r'^[2-7]') & (df['MMSI'].str.len() == 9)]

def filter_SOG(df, SOG_threshold:float):
    """
    Parameters:
        df: AIS data.
        SOG_threshold: threshold of maximum speed in knots.
 
    Returns:
        df: AIS data with abnormal speed filtered
    """
    
    return df[df['SOG'] < SOG_threshold]

def filter_minority(df, threshold:int, column='MMSI'):
    """
    Parameters:
        df: AIS data.
        threshold: threshold to filter minority ids.
        column: column name to filter.
 
    Returns:
        df: AIS data with minority ids filtered
    """
    mmsi_counts = df[column].value_counts().reset_index()
    mmsi_counts.columns = [column, 'count']

    # Filter MMSIs that meet the threshold
    valid_id = mmsi_counts[mmsi_counts['count'] >= threshold][column]

    # Use boolean indexing to filter the DataFrame
    df = df[df[column].isin(valid_id)]

    return df

def haversine_distance(group):
    """
    Parameters:
        group: AIS data grouped by id (MMSI/trips_id).
    Returns:
        distance: distance between two consecutive points in nautical miles.
    
    Typical usage example:

        df = df.sort_values(by=['MMSI', 'time']).reset_index(drop=True)

        df['distance'] = df.groupby('MMSI').apply(haversine_distance).reset_index(drop=True)
    """
    lat1 = group['latitude']
    lon1 = group['longitude']
    lat2 = group['latitude'].shift(1).fillna(0)
    lon2 = group['longitude'].shift(1).fillna(0)
    
    # Convert latitude and longitude to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Approximate radius of the Earth in Nautical miles
    earth_radius = 3440.065

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = earth_radius * c

    # Set the first element to 0
    distance[0] = 0
    return pd.Series(distance, index=group.index)


def rmove_outlinears(df, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX):
    """
    Parameters:
        df: AIS data
        LAT_MAX: the max latitude
        LAT_MIN: the min latitude
        LON_MAX: the max longitude
        LON_MIN: the min longitude
    
    Returns:
        df: AIS data with points out of the region filtered.
    """
    # Region Coordinates
    
    # Filter out the points that are outside the region
    idx = (df['latitude'] > LAT_MIN) & (df['latitude'] < LAT_MAX) & (df['longitude'] > LON_MIN) & (df['longitude'] < LON_MAX)
    df = df[idx]
    return df

def drop_duplicates(df):
    """
    Parameters:
        df: AIS data.
    """
    return df.drop_duplicates(subset=['MMSI', 'time'], keep='first')



def trans2cat(df):

    df['ship_type'] = df['ship_type'].astype("category").cat.codes
    df['navigational_status'] = df['navigational_status'].astype("category").cat.codes

    return df

def split_trips(group, cfg):
    if 'total_trips' not in split_trips.__dict__:
        split_trips.total_trips = 0
    
    max_time_diff = cfg.max_time_diff
    max_dis_diff = cfg.max_dis_diff
    # trips_id = ((group['time_diff'].dt.seconds.fillna(0) > max_time_diff)|(group['distance_diff'].fillna(0) > max_dis_diff)).cumsum() + split_trips.total_trips
    trips_id = ((group['time_diff'].dt.seconds.fillna(0) > max_time_diff)).cumsum() + split_trips.total_trips
    split_trips.total_trips = trips_id.nunique() + split_trips.total_trips
    
    return pd.Series(trips_id, index=group.index)



def update_ship_info(group, ships_info_dict):
    mmsi_str = str(group.name)
    group['ship_type'] = ships_info_dict[mmsi_str]['vessel_type']
    group['ship_subtype'] = ships_info_dict[mmsi_str]['vessel_subtype']
    group['length'] = ships_info_dict[mmsi_str]['vessel_length']
    group['width'] = ships_info_dict[mmsi_str]['vessel_width']
    return group

def complete_missing_value(cfg, df):
    with open(cfg.vessels_info_dic_dir) as json_file:
        ships_info_dict = json.load(json_file)

    df_completed = df.groupby('MMSI').apply(update_ship_info, ships_info_dict=ships_info_dict).reset_index(drop=True)
    
    return df_completed




import requests
from bs4 import BeautifulSoup
import json
from typing import List

def get_vessel_info(mmsi:str):
    '''
    Parameters:
        mmsi: MMSI of the vessel

    Returns:
        vessel_type: Pramary type, String,
        vessel_subtype: Pramary scendary type, String
        vessel_length: length (dm), float, Note: 10dm = 1m
        vessel_width: vessel width (dm), float

    Typical usage example:
        type_cal = ['Sailing','Pleasure' ,'Cargo','Fishing','Passenger','Tanker','Tug','SAR','HSC','Dredging','Military']

        mmsi = '538003672'

        vessel_type, vessel_subtype, vessel_length, vessel_width = get_vessel_info(type_cal,mmsi)
        
    '''
    type_cal = ['Sailing','Pleasure' ,'Cargo','Fishing','Passenger','Tanker','Tug','SAR','HSC','Dredging','Military']

    vessel_type = 'Other'
    vessel_subtype = ''
    vessel_length = ''
    vessel_width = ''
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
        "Referer": "https://www.marinetraffic.com/",
        "Accept-Language": "en-US,en;q=0.9",
    }

    data = {'mmsi':mmsi}
    url = f'https://www.marinetraffic.com/en/ais/details/ships/mmsi:{mmsi}'
    with requests.Session() as session:
        response = session.get(url=url, headers=headers)
    try :
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            content = soup.find_all('meta', {'content': lambda x: x and x.startswith('vessel ')})
            for tag in content:
                if tag.get('data-react-helmet') == 'true' and tag.get('name') == 'keywords':
                    keywords = tag.get('content')
                    vessel_type = keywords.split(',')[3][1:]
                    vessel_type = vessel_type.split(' ')[0]
                    vessel_subtype = keywords.split(',')[2][13:]
                    if vessel_type not in type_cal:
                        vessel_type = 'Other'
                    break
    except:
        pass
    response = requests.post("http://www.shipfinder.com/ship/GetShip", data=data)
    try:
        if response.status_code == 200:
            result = json.loads(response.text)['data'][0]
            vessel_length = result['length']
            vessel_width = result['width']
            vessel_length = float(vessel_length)
            vessel_width = float(vessel_width)
    except:
        pass
    return vessel_type, vessel_subtype, vessel_length, vessel_width