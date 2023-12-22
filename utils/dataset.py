import numpy as np
import pandas as pd
from . import data_preprocess_utils as pre

class DenmarkDataset():
    def __init__(self, cfg):
        self._cfg = cfg
        self._raw_files_dict = self._files2dict(pre.get_all_files(self._cfg.data_dir))
        self._day_range = self._get_day_range()
        
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
    
    def _get_day_data(self, day):
        """
        Get the data of a specific day

        Parameters:
            day: str, the date of the day
        Returns:
            day_dic: dict, the data of the day, with keys as the vessel type and values as the data
        Typical usage example:
            get_day_data('2019-01-01')
        """
        if day not in self._raw_files_dict:
            raise Exception(f"Day {day} not in the dataset")
        day_dic = np.load(self._raw_files_dict[day], allow_pickle=True).item()
        for vessel_type in day_dic.keys():
            day_dic[vessel_type] = np.array(day_dic[vessel_type])
        return day_dic
    
    def _get_day_range(self):
        """
        Get the valid day range of the dataset
        """
        dates_str = list(self._raw_files_dict.keys())
        dates_datetime = [pd.to_datetime(date) for date in dates_str]
        # sort by date
        dates_datetime.sort()
        # get the start and end date
        start_date = dates_datetime[0]
        end_date = dates_datetime[-1]
        # get the day range
        day_range = pd.date_range(start_date, end_date, freq='1D')
        return day_range
    
    def get_data_by_range(self, start_date, end_date):
        """
        Get the data of a specific range
        
        Parameters
            start_date: str, start date of the range
            end_date: str, end date of the range
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
        result_dic = self._get_day_data(self._day_range[start_idx].strftime("%Y-%m-%d"))
        for idx in range(start_idx+1, end_idx+1):
            date = self._day_range[idx].strftime("%Y-%m-%d")
            day_dic = self._get_day_data(date)
            for key in result_dic.keys():
                result_dic[key] = np.concatenate((result_dic[key], day_dic[key]))
        return result_dic