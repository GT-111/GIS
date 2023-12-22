import folium
from folium.plugins import MarkerCluster, HeatMap
import random
import numpy as np

def random_color():
    """
    Returns:
        a random color in hex format
    """
    return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


def create_map(LAT_MAX, LAT_MIN, LON_MAX, LON_MIN):
    """
    Parameters:
        LAT_MAX - the max latitude
        LAT_MIN - the min latitude
        LON_MAX - the max longitude
        LON_MIN - the min longitude
 
    Returns:
        a folium map
    """
    m = folium.Map(location=[(LAT_MAX + LAT_MIN) / 2, (LON_MAX + LON_MIN) / 2], zoom_start=8)
    return m



def get_color(label):
    """
    Parameters:
        label- cluster label
    Returns:
        color for the cluster
    """
    # generate a color for a cluster label via its id
    r = (label * 37) % 256
    g = (label * 73) % 256
    b = (label * 101) % 256
    return 'rgb(%d, %d, %d)' % (r, g, b)

def visulize_trajs(df, group_column = 'mmsi', mode='lines', map_ori = None, heatmap = False, save = False):
    # phase: start or end
    # raise exception if phase is not start or end
    if mode not in ['lines', 'scatters']:
        raise Exception('mode should be lines or scatters')
    if map_ori is None:
        m = folium.Map(location=[0, -180], zoom_start=1, )
    else:
        m = map_ori
        
    if mode == 'lines':
        marker_cluster = MarkerCluster().add_to(m)
        for mmsi, group in df.groupby(group_column):
            traj_data = group[['latitude', 'longitude']].values.tolist()
            traj_data = [[point[0], point[1] + 360 if point[1] < -20 else point[1]] for point in traj_data]
            folium.PolyLine(locations=traj_data, color=random_color(), weight=5, opacity=0.7, popup=mmsi).add_to(marker_cluster)
    elif mode == 'scatters':
        for mmsi, group in df.groupby(group_column):
            color = random_color()
            for idx, row in group.iterrows():
    
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude'] + 360 if row['longitude'] < -20 else row['longitude']],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7
                ).add_to(m)
           
    
    heat_data = [[point[0], point[1] + 360 if point[1] < -20 else point[1]] for point in  df[['latitude', 'longitude']].values.tolist()]
    if heatmap is True:
        HeatMap(heat_data).add_to(m)
    if save is True:
        m.save('trajectory_map.html')
    return m
