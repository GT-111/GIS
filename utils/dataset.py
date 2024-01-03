import numpy as np
import pandas as pd
import geopandas as gpd
from typing import List
from numpy import int64
from . import data_preprocess_utils as pre

class DenmarkDataset():
    def __init__(self, cfg):
        self._cfg = cfg
        self._directories = pre.get_all_subdirectories(self._cfg.data_dir)
        self._raw_files_dict = self._directories2dict(self._directories)
        self._day_range = self._get_day_range()
        self._interval_list = [ (time_interval) for time_interval in self._raw_files_dict.keys()]
        self._adjacency_matrix = self._get_adjacency_matrix()
        self._distance_matrix = self._get_distance_matrix()
        self._width_matrix = self._get_width_matrix()
        self._length_matrix = self._get_length_matrix()
        self._convex_hulls = self._get_convex_hulls()

    def _get_convex_hulls(self):
        """
        Get the convex hull of the dataset
        """
        _convex_hulls = gpd.read_feather(self._cfg.convex_hull_dir)
        _convex_hulls.reset_index(inplace=True, drop=True)
        return _convex_hulls
    
    def _get_adjacency_matrix(self):
        """
        Get the adjacency matrix of the dataset
        """
        return np.load(self._cfg.adjacency_matrix_dir)
    def _get_distance_matrix(self):
        """
        Get the distance matrix of the dataset
        """
        return np.load(self._cfg.distance_matrix_dir)
    def _get_width_matrix(self):
        """
        Get the width matrix of the dataset
        """
        return np.load(self._cfg.width_matrix_dir)
    
    def _get_length_matrix(self):
        """
        Get the length matrix of the dataset
        """
        return np.load(self._cfg.length_matrix_dir)
    
    def _generate_rel_file(self):
        wp_labels = self._convex_hulls['label'].values
        rel_dic = []
        for wp_origin in wp_labels:
            for wp_destination in wp_labels:
                if wp_origin != wp_destination:
                    rel_dic.append({'origin_id': wp_origin, 
                                    'destination_id': wp_destination, 
                                    'cost': self._distance_matrix[wp_origin, wp_destination], 
                                    'weight': self._adjacency_matrix[wp_origin, wp_destination],
                                    'length_max': self._length_matrix[wp_origin, wp_destination],
                                    'width_max': self._width_matrix[wp_origin, wp_destination],
                                    })
        rel_df = pd.DataFrame(rel_dic)
        rel_df['type'] = 'geo'
        rel_df['rel_id'] = rel_df.index.astype(int64)
        new_order = ['rel_id', 'type', 'origin_id', 'destination_id', 'cost', 'weight', 'length_max', 'width_max']
        rel_df = rel_df[new_order]

        return rel_df
    def _generate_geo_file(self):
        
        geo_df = pd.DataFrame()
        geo_df['geo_id'] = self._convex_hulls['label']
        geo_df['type'] = 'Point'
        geo_df['coordinates'] = self._convex_hulls['centroid'].apply(lambda p: [p.x, p.y]) # lon lat
        geo_df.reset_index(drop=True, inplace=True)
        new_order = ['geo_id', 'type', 'coordinates']
        geo_df = geo_df[new_order]

        return geo_df
    
    def _generate_dyna_file(self, start_date, end_date, time_interval='4H', vessel_type='Total'):
        day_range = pd.date_range(start_date, end_date, freq=time_interval)
        range_data_dic = self.get_data_by_range(start_date=start_date, end_date=end_date, time_interval=time_interval)
        waypoints_num = self._convex_hulls['label'].unique().shape[0]
        vessel_type = 'Total'
        dyna_df_list = []
        for wp in range(waypoints_num):
            for time_idx, time in enumerate(day_range):
                day_str = time.strftime('%Y-%m-%dT%H:%M:%SZ')
                dyna_df_list.append({
                    'type': 'state',
                    'time': day_str,
                    'entity_id': wp,
                    'num': range_data_dic[vessel_type][time_idx, wp, 0],
                    'in_flow': range_data_dic[vessel_type][time_idx, wp, 1],
                    'out_flow': range_data_dic[vessel_type][time_idx, wp, 2],
                    'speed': range_data_dic[vessel_type][time_idx, wp, 3]
                })
        dyna_df = pd.DataFrame(dyna_df_list)
        dyna_df['dyna_id'] = dyna_df.index
        new_order = ['dyna_id', 'type', 'time', 'entity_id', 'num', 'in_flow', 'out_flow', 'speed']
        dyna_df = dyna_df[new_order]
        return dyna_df
    
    def _directories2dict(self, directories):
        raw_files_dict = {}
        for directory in directories:
            time_interval = directory.split('/')[-1] + 'H'
            raw_files_dict[time_interval] = self._files2dict(pre.get_all_files(directory))
        return raw_files_dict
    def _files2dict(self, files):
        """
        Map the files to a dictionary with key as the date and value as the file path
        """
        result = {}
        for file in files:
            yyyy, mm, dd = file.split('/')[-1].split('.')[0].split('-')
            date = yyyy+'-'+mm+'-'+dd
            result[date] = file
        return result
    
    def _get_day_data(self, day, time_interval='4H'):
        """
        Get the data of a specific day

        Parameters:
            day: str, the date of the day
            time_interval: str, the time interval of the data, in the format of '4H', '8H', '12H', '24H'
        Returns:
            day_dic: dict, the data of the day, with keys as the vessel type and values as the data
        Typical usage example:
            get_day_data('2019-01-01')
        """
        if time_interval not in self._interval_list:
            raise Exception(f"Time interval {time_interval} not in the dataset, please choose from {self._interval_list}")
        if day not in self._raw_files_dict[time_interval]:
            raise Exception(f"Day {day} not in the dataset")
        day_dic = np.load(self._raw_files_dict[time_interval][day], allow_pickle=True).item()
        for vessel_type in day_dic.keys():
            day_dic[vessel_type] = np.array(day_dic[vessel_type])
        return day_dic
    
    def _get_day_range(self):
        """
        Get the valid day range of the dataset
        """
        dates_str = list(self._raw_files_dict['4H'].keys())
        dates_datetime = [pd.to_datetime(date) for date in dates_str]
        # sort by date
        dates_datetime.sort()
        # get the start and end date
        start_date = dates_datetime[0]
        end_date = dates_datetime[-1]
        # get the day range
        day_range = pd.date_range(start_date, end_date, freq='1D')
        return day_range
    
    def get_data_by_range(self, start_date, end_date, time_interval='4H'):
        """
        Get the data of a specific range
        
        Parameters
            start_date: str, start date of the range
            end_date: str, end date of the range
            time_interval: str, the time interval of the data, in the format of '4H', '8H', '12H', '24H'
        Returns:
            result_dic: dict, the data of the range, with keys as the vessel type and values as the data
        Typical usage example:
            get_data_by_range('2019-01-01', '2019-01-10')
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        if start_date not in self._day_range:
            raise Exception(f"Start date {start_date} not in the dataset")
        if end_date not in self._day_range:
            raise Exception(f"End date {end_date} not in the dataset")
        start_idx = self._day_range.get_loc(start_date)
        end_idx = self._day_range.get_loc(end_date)
        result_dic = self._get_day_data(day=self._day_range[start_idx].strftime("%Y-%m-%d"), time_interval=time_interval)
        for idx in range(start_idx+1, end_idx+1):
            date = self._day_range[idx].strftime("%Y-%m-%d")
            day_dic = self._get_day_data(day=date, time_interval=time_interval)
            for key in result_dic.keys():
                result_dic[key] = np.concatenate((result_dic[key], day_dic[key]))
        return result_dic
    

    def convert2PEMS(self):
        """
        Convert the dataset to the PEMS format
        """
        rel_df = self._generate_rel_file()
        geo_df = self._generate_geo_file()
        dyna_df = self._generate_dyna_file(start_date=self._cfg.PEMS.start_date, 
                                           end_date=self._cfg.PEMS.end_date, 
                                           time_interval=self._cfg.PEMS.time_interval, 
                                           vessel_type=self._cfg.PEMS.vessel_type
                                           )
        path = self._cfg.PEMS.save_dir
        name = self._cfg.PEMS.name
        rel_df.to_csv(path + name + '.rel', index=False)
        geo_df.to_csv(path + name + '.geo', index=False)
        dyna_df.to_csv(path + name + '.dyna', index=False)


    def dic2array(self, result_dic, vessel_types:List[str]=['Passenger', 'Tanker', 'Cargo', 'Other', 'Total'], time_interval='4H'):
        """
        Parameters:
            result_dic: dict, the data of the range, with keys as the vessel type and values as the data
            vessel_types: list, the vessel types to include. 
            Note: 
                1) vessel_types should be a subset of ['Sailing', 'Pleasure', 'Cargo', 'Fishing', 'Passenger', 'Tanker', 'Tug', 'Other', 'Total'].
                2) must include 'Other' and 'Total'.
        Returns:
            feature_arr: np.array, the data in the format of (time_span, time_interval, wp_cnt, vessel_types_cnt, feature_dim)
            type_index_dic: dict, the index of the vessel type in the feature_arr
        """

        if not set(vessel_types).issubset(set(['Sailing', 'Pleasure', 'Cargo', 'Fishing', 'Passenger', 'Tanker', 'Tug', 'Other', 'Total'])):
            raise Exception("vessel_types should be a subset of ['Sailing', 'Pleasure', 'Cargo', 'Fishing', 'Passenger', 'Tanker', 'Tug', 'Other', 'Total']")
        if 'Other' not in vessel_types or 'Total' not in vessel_types:
            raise Exception("vessel_types must include 'Other' and 'Total'")

        time_span, wp_cnt, feature_dim = result_dic['Total'].shape
        vessel_types_cnt = len(vessel_types)
        type_index_dic = {}
        time_interval = int(time_interval[:-1])
        interval_per_day = int(24/time_interval)
        feature_arr = np.zeros((int(time_span/ interval_per_day), interval_per_day , wp_cnt, vessel_types_cnt, feature_dim))
        for _, category in enumerate(result_dic.keys()):
            if category not in vessel_types:
                category = 'Other'
            
            feature_arr[:, :, :, vessel_types.index(category), :] = result_dic[category].reshape((int(time_span/ interval_per_day), interval_per_day , wp_cnt, feature_dim))
            type_index_dic[category] = vessel_types.index(category)
        return feature_arr, type_index_dic
        