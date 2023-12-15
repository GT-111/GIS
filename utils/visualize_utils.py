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