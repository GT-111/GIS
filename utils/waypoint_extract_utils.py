from easydict import EasyDict
from simplification.cutil import simplify_coords_idx
import numpy as np
import pandas as pd
# hdbscan
import hdbscan


def dicover_maneuver_points(df:pd.DataFrame, cfg:EasyDict):
    """
    Parameters:
        df: AIS data
        cfg: configuration file
    Returns:
        maneuver_points: maneuver points
    """
    
    window_size = cfg.maneuver_points_discovery.window_size
    sog_threshold_multiplier = cfg.maneuver_points_discovery.sog_threshold_multiplier
    cog_threshold_multiplier = cfg.maneuver_points_discovery.cog_threshold_multiplier
    # smoothe the COG_diff with moving average
    df.groupby('MMSI')['COG_diff'].rolling(window_size, min_periods=1).mean().reset_index(inplace=True, drop=True)
    df['COG_diff'] = df['COG_diff'].abs()
    # if the COG_diff is larger than 180, then we need to change it to 360 - COG_diff
    df['COG_diff'] = df['COG_diff'].map(lambda x: 360 - x if x > 180 else x)
    # smoothe the SOG_diff with moving average
    df.groupby('MMSI')['SOG_diff'].rolling(window_size, min_periods=1).mean().reset_index(inplace=True, drop=True)
    df['SOG_diff'] = df['SOG_diff'].abs()
    # Calculate the sog and cog threshold
    sog_threshold = df['SOG_diff'].mean() + sog_threshold_multiplier * df['SOG_diff'].std()
    cog_threshold = df['COG_diff'].mean() + cog_threshold_multiplier * df['COG_diff'].std()
    # Filter out the points that are not maneuver points
    maneuver_points = df[(df['COG_diff'] > cog_threshold) | (df['SOG_diff'] > sog_threshold)]
    

    return maneuver_points

def calculate_geodesic_distance_matrix(matrix1:np.ndarray, matrix2:np.ndarray):
    lat1, lon1 = matrix1.T
    lat2, lon2 = matrix2.T 
    
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    
    # Earth radius in nautical miles
    earth_radius = 3440.065
    
    dlat = lat2[:, np.newaxis] - lat1
    dlon = lon2[:, np.newaxis] - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2[:, np.newaxis]) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance_matrix = earth_radius * c
    
    return distance_matrix

def clustering_hdbscan(maneuver_points, cfg:EasyDict):
    maneuver_points_lat_lon = maneuver_points[['latitude', 'longitude']].to_numpy()
    min_cluster_size = cfg.hdbscan_params.min_cluster_size
    min_samples = cfg.hdbscan_params.min_samples
    cluster_selection_epsilon = cfg.hdbscan_params.cluster_selection_epsilon
    cluster_selection_method = cfg.hdbscan_params.cluster_selection_method
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=cluster_selection_epsilon, cluster_selection_method=cluster_selection_method)
    # distance = calculate_geodesic_distance_matrix(maneuver_points_lat_lon, maneuver_points_lat_lon)
    cluster_labels = clusterer.fit_predict(maneuver_points_lat_lon)
    return cluster_labels

def get_cluster_centers(maneuver_points, cluster_labels):

    unique_labels = np.unique(cluster_labels)
    maneuver_points['cluster_label'] = cluster_labels
    waypoints = []
    waypoints_labels = []
    for _, label in enumerate(unique_labels):
        if label == -1:
            continue
        waypoints_labels.append(label)
        waypoints.append([maneuver_points[maneuver_points['cluster_label'] == label]['latitude'].mean(), maneuver_points[maneuver_points['cluster_label'] == label]['longitude'].mean()])
    return np.array(waypoints), waypoints_labels

def simplify_traj(group:pd.Series) :
    """
    Parameters:
        group: grouped AIS data.
    Returns:
        group: simplified grouped AIS data.
    Usage:
        df.groupby('MMSI').apply(simplify_traj)
    """
    coordinates = np.column_stack((group['longitude'].to_numpy(), group['latitude'].to_numpy()))
    simplified_idx = simplify_coords_idx(coordinates, 0.025)
    return group.iloc[simplified_idx]