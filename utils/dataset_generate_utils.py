import pandas as pd
import geopandas as gpd
import numpy as np
from tqdm import tqdm
def get_traj_point_type(group, cfg):
    
    gdf = gpd.read_feather(cfg.convex_hull_dir) # waypoint hulls
    group.reset_index(inplace=True)
    geometry = gpd.points_from_xy(group['longitude'], group['latitude'])
    result = gpd.sjoin(gdf, gpd.GeoDataFrame(geometry=geometry, crs='EPSG:4326'), how="inner", predicate="intersects")

    if result['label'].nunique() < 2:
        group.set_index('index', inplace=True)
        return group

    dftmp_idx = result.index_right.values
    group_idx = group.index.values

    # Vectorized label assignment
    group.loc[dftmp_idx[:], 'label'] = result.iloc[:]['label'].values

    # Vectorized 'inout' calculation
    mask = (group['label'] != group['label'].shift(1))
    mask_shift = np.roll(mask, -1)
    mask[0] = False
    mask_shift[-1] = False
    group['inout'] = np.where(mask, 1, 0) + np.where(mask_shift, 2, 0)

    group.set_index('index', inplace=True)
    return group

def generate_dataset(day, cfg, df, time_interval:int, keep_unlabeled=False):

    """
    Parameters:
        day: str, the date of the day to be processed.
        cfg: config object.
        df: AIS data, with column 'inout' and 'label' added.
        time_interval: int, the time interval of the data, in terms of hour.
        keep_unlabeled: bool, whether to keep the unlabeled waypoints.
    Returns:
        result_dic: dict, the dictionary of the features of the day.
    """
    periods = int(24 // time_interval)
    freq = str(time_interval) + 'H'
    idx = pd.date_range(day, periods=periods, freq=freq)
    gdf = gpd.read_feather(cfg.convex_hull_dir) # waypoint hulls
    waypoints = gdf['centroid'].apply(lambda p: (p.y, p.x)).values
    labels = gdf['label'].values
    labels = labels[labels != -1]

    df['period'] = -1
    df['time'] = pd.to_datetime(df['time'])
    
    for i in range(len(idx)):
        
        period_idx = df['time'] >= idx[i]
        df.loc[period_idx, 'period'] = i
    
    # num in-flow out-flow speed (4-dimensional feature) 
    feature_dic = {}
    vessels_type_list = ['Sailing', 'Pleasure', 'Cargo', 'Fishing', 'Passenger', 'Tanker', 'Tug', 'Other', 'Total']
    for vessel_type in vessels_type_list:
        feature_dic[vessel_type] = [pd.DataFrame(0, columns=range(-1, len(waypoints)), index=range(len(idx))) for i in range(4)]
    
    N = len(waypoints)
    for vessel_type, features in feature_dic.items():
        vessel_type_mask = df['ship_type'] == vessel_type
        df_vessel_type = df[vessel_type_mask]

        if df_vessel_type['label'].nunique() < 2:
            continue

        df_in = df_vessel_type.loc[df_vessel_type['inout'] == 1, ['period', 'label']]
        df_out = df_vessel_type.loc[df_vessel_type['inout'] == 2, ['period', 'label']]
        df_inout = df_vessel_type.loc[df_vessel_type['inout'] == 3, ['period', 'label']]

        num_in = df_in.groupby(['period', 'label']).size().unstack(fill_value=0)
        num_out = df_out.groupby(['period', 'label']).size().unstack(fill_value=0)
        num_inout = df_inout.groupby(['period', 'label']).size().unstack(fill_value=0)
        num = df_vessel_type.reset_index(drop=True).groupby(['period','MMSI', 'label']).size().unstack(fill_value=0)
        speed = df_vessel_type.groupby(['period','label']).agg({'SOG': 'mean'}).unstack(fill_value=0)['SOG']
        num[num > 0] = 1
        # 
        num = num.groupby('period').sum().reset_index(drop=True)
        features[0] = features[0].add(num, fill_value=0)  # flow
        features[1] = features[1].add(num_in, fill_value=0).add(num_inout, fill_value=0)  # in-flow
        features[2] = features[2].add(num_out, fill_value=0).add(num_inout, fill_value=0)  # out-flow
        features[3] = features[3].add(speed, fill_value=0)  # speed

        # add to total
        feature_dic['Total'][0] = feature_dic['Total'][0].add(num, fill_value=0)
        feature_dic['Total'][1] = feature_dic['Total'][1].add(num_in, fill_value=0).add(num_inout, fill_value=0)
        feature_dic['Total'][2] = feature_dic['Total'][2].add(num_out, fill_value=0).add(num_inout, fill_value=0)
        feature_dic['Total'][3] = feature_dic['Total'][3].add(speed, fill_value=0)

    result_dic = {}  
    for vessel_type, vessel_feature in feature_dic.items():
        vessel_feature[3] = vessel_feature[3].div(vessel_feature[0])  # calc speed
        
        feature_tmp = np.array(vessel_feature)
        feature_tmp[np.isnan(feature_tmp)] = 0
        feature_tmp = feature_tmp.transpose(1, 2, 0)  # 4*node*4
        f_day = pd.DataFrame()
        if keep_unlabeled:
            for i in range(-1, N):
                f_day = f_day.copy()
                f_day[i] = [feature_tmp[v0][i + 1] for v0 in range(periods)]
        else:
            for i in range(N):
                f_day = f_day.copy()
                f_day[i] = [feature_tmp[v0][i + 1] for v0 in range(periods)]

        f_day = f_day.rename(index=dict(zip(f_day.index, idx)))
        result_dic[vessel_type] = np.array(f_day.values.tolist())

    return result_dic
