import numpy as np
import folium
from branca.colormap import linear

class GeoHashBinary:
    def __init__(self, bounds):
        self.bounds = bounds

    def encode_to_geohash_binary(self,latitude, longitude, precision=12) -> str:
        bounds = self.bounds
        mid_lon = (bounds[1] + bounds[3]) / 2
        mid_lat = (bounds[0] + bounds[2]) / 2
        geohash_binary = ''
        for i in range(precision):
            if latitude < mid_lat:
                geohash_binary += '0'
                bounds = (bounds[0], bounds[1], mid_lat, bounds[3])
            else:
                geohash_binary += '1'
                bounds = (mid_lat, bounds[1], bounds[2], bounds[3])
            if longitude < mid_lon:
                geohash_binary += '0'
                bounds = (bounds[0], bounds[1], bounds[2], mid_lon)
            else:
                geohash_binary += '1'
                bounds = (bounds[0], mid_lon, bounds[2], bounds[3])
            mid_lon = (bounds[1] + bounds[3]) / 2
            mid_lat = (bounds[0] + bounds[2]) / 2
        return geohash_binary

    def decode_geohash_binary(self, geohash_binary):
        bounds = self.bounds
        # decode the geohash binary code to get the latitude and longitude
        mid_lon = (bounds[1] + bounds[3]) / 2
        mid_lat = (bounds[0] + bounds[2]) / 2
        for i in range(0, len(geohash_binary) ,2):
            if geohash_binary[i] == '0':
                bounds = (bounds[0], bounds[1], mid_lat, bounds[3])
            else:
                bounds = (mid_lat, bounds[1], bounds[2], bounds[3])
            if geohash_binary[i+1] == '0':
                bounds = (bounds[0], bounds[1], bounds[2], mid_lon)
            else:
                bounds = (bounds[0], mid_lon, bounds[2], bounds[3])
            mid_lon = (bounds[1] + bounds[3]) / 2
            mid_lat = (bounds[0] + bounds[2]) / 2
        return mid_lat, mid_lon

    def get_geohash_neighbors_from_binary(self, geohash_binary, geohash_precision):
        # geohash_precision = len(geohash_binary) // 2
        bounds = self.bounds
        neighbors = []
        # Determine the step size for latitude and longitude based on geohash precision
        lat_step = (bounds[2] - bounds[0]) / 2 ** (geohash_precision // 2 + 1)
        lon_step = (bounds[3] - bounds[1]) / 2 ** (geohash_precision // 2 + 1)
        # Decode base32 geohash to get the coordinates
        decoded_lat, decoded_lon = self.decode_geohash_binary(geohash_binary)
        # Calculate the neighbors' coordinates
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbor_lat = decoded_lat + dy * lat_step
                neighbor_lon = decoded_lon + dx * lon_step
                neighbors.append(self.encode_to_geohash_binary(neighbor_lat, neighbor_lon, geohash_precision))

        return neighbors
    
class QuadTreeNode:
    def __init__(self, bounds, geohash_code, father=None):
        self.bounds = bounds # format: (lat_min, lon_min, lat_max, lon_max)
        self.children = []  # four sub-grids, in order of left bottom, left top, right bottom, right top
        self.father = father
        self.geohash_code = geohash_code
        self.ship_dic = {}
        self.waypoint_cnt = {} # number of data points belong to each waypoint in the grid
        self.ship_cnt = 0
        self.valid = True
        self.label_assigned = -1

class QuadTree:
    def __init__(self, bounds, precision_max):
        # bitnummax = 2 * precision
        self.root = QuadTreeNode(bounds, '')
        self.precision_max = precision_max
        self.geohash_encoder = GeoHashBinary(bounds)
    
    def init_subgrids(self, node: QuadTreeNode):
        # genrate four sub-grid
        mid_latitude = (node.bounds[0] + node.bounds[2]) / 2
        mid_longitude = (node.bounds[1] + node.bounds[3]) / 2
        current_node_geohash_code = node.geohash_code
        # if the current precision is the maximum precision, then stop
        if len(current_node_geohash_code) == self.precision_max * 2:
            return
        # break the current geohash code into four parts
        node.children.append(QuadTreeNode(bounds=(node.bounds[0], node.bounds[1], mid_latitude, mid_longitude), geohash_code=current_node_geohash_code + '00', father=node)) # left bottom, add 00 to the end
        node.children.append(QuadTreeNode(bounds=(node.bounds[0], mid_longitude, mid_latitude, node.bounds[3]), geohash_code=current_node_geohash_code + '01', father=node))  # left top, add 01 to the end
        node.children.append(QuadTreeNode(bounds=(mid_latitude, node.bounds[1], node.bounds[2], mid_longitude), geohash_code=current_node_geohash_code + '10', father=node))  # right bottom, add 10 to the end
        node.children.append(QuadTreeNode(bounds=(mid_latitude, mid_longitude, node.bounds[2], node.bounds[3]), geohash_code=current_node_geohash_code + '11', father=node)) # right top, add 11 to the end
        # recursively generate sub-grids
        for child in node.children:
            self.init_subgrids(child)
        return
    
    def get_leaf_nodes(self):
        # get the smallest grids which corresponds to the leaf nodes
        leaf_nodes = []
        def _get_leaf_nodes(node: QuadTreeNode):
            if node is None:
                return
            # since after filtering, some of the node wouble be invalid, so we need to check the valid flag
            if len(node.children) == 0:
                leaf_nodes.append(node)
                return
            for child in node.children:
                _get_leaf_nodes(child)
        _get_leaf_nodes(self.root)
        return leaf_nodes
    
    def get_node_by_geohash_code(self, geohash_code):
        # get the node by geohash code
        def _get_node_by_geohash_code(node: QuadTreeNode, geohash_code, depth=0):
            if node is None:
                if node.father is not None:
                    return node.father
                else:
                    return None
            if node.geohash_code == geohash_code:
                return node
            subgrid_suffix = geohash_code[depth:depth+2]
            if subgrid_suffix == '00':
                return _get_node_by_geohash_code(node.children[0], geohash_code, depth+2)
            elif subgrid_suffix == '01':
                return _get_node_by_geohash_code(node.children[1], geohash_code, depth+2)
            elif subgrid_suffix == '10':
                return _get_node_by_geohash_code(node.children[2], geohash_code, depth+2)
            elif subgrid_suffix == '11':
                return _get_node_by_geohash_code(node.children[3], geohash_code, depth+2)
            
        return _get_node_by_geohash_code(self.root, geohash_code, depth=0)
    
    def init_ship_dic_row(self, row):
        lat = row['latitude']
        lon = row['longitude']
        geohash_binary = self.geohash_encoder.encode_to_geohash_binary(lat, lon, self.precision_max)
        corresponding_node = self.get_node_by_geohash_code(geohash_binary)
        # save the ship data points in the ship_dic
        if row['trips_id'] not in corresponding_node.ship_dic:
            corresponding_node.ship_dic[row['trips_id']] = []
            corresponding_node.ship_dic[row['trips_id']].append({'time': row['time'], 'lat': row['latitude'], 'lon': row['longitude']})
        corresponding_node.ship_dic[row['trips_id']].append({'time': row['time'], 'lat': row['latitude'], 'lon': row['longitude']})

        # count the number of data points belong to each waypoint in the grid
        if row['cluster_label'] not in corresponding_node.waypoint_cnt:
            corresponding_node.waypoint_cnt[row['cluster_label']] = 0
        corresponding_node.waypoint_cnt[row['cluster_label']] += 1
    def init_ship_dic_df(self, df):
        df.sort_values(by=['trips_id', 'time'], inplace=True)
        df.apply(self.init_ship_dic_row, axis=1)
    
    def init_ship_cnt(self):
        def _init_ship_cnt(node: QuadTreeNode):
            if node is None:
                return
            if len(node.children) == 0:
                node.ship_cnt = len(node.ship_dic)
                return
            else:
                node.ship_cnt = 0
                for child in node.children:
                    _init_ship_cnt(child)
                    node.ship_cnt += child.ship_cnt
        _init_ship_cnt(self.root)
        
    def filter_nodes(self, alpha):
        # alpha is the threshold multiplier
        leaf_nodes = self.get_leaf_nodes()
        for node in leaf_nodes:
            # get the neighbor geohash codes
            neighbors = self.geohash_encoder.get_geohash_neighbors_from_binary(node.geohash_code, self.precision_max)
            # print(neighbors)
            # get sneighbors nodes
            neighbor_nodes = [self.get_node_by_geohash_code(neighbor) for neighbor in neighbors]
            # print(neighbor_nodes)
            neighbor_ship_numbers = [len(neighbor_node.ship_dic) for neighbor_node in neighbor_nodes]
            # print(neighbor_ship_numbers)
            # if np.sum(neighbor_ship_numbers) > 0:
            #     print(neighbor_ship_numbers)
            neighbor_ship_numbers_mean = np.mean(neighbor_ship_numbers)
            neighbor_ship_numbers_std = np.std(neighbor_ship_numbers)
            # print(neighbor_ship_numbers_mean, neighbor_ship_numbers_std)
            threshold = neighbor_ship_numbers_mean + alpha * neighbor_ship_numbers_std
            if len(node.ship_dic) < threshold  or len(node.ship_dic) <= 4:
                node.valid = False

def assign_label(leaf_nodes):
    for node in leaf_nodes:
        if node.ship_cnt == 0:
            node.label_assigned = -1
        else:
            node.label_assigned = max(node.waypoint_cnt, key=node.waypoint_cnt.get)
    return leaf_nodes

def leaf_nodes2waypoints(leaf_nodes):
    waypoints = {}
    for node in leaf_nodes:
        if not node.valid or node.ship_cnt == 0:
            continue
        if node.label_assigned not in waypoints:
            waypoints[node.label_assigned] = []
        waypoints[node.label_assigned].append((node.bounds[1], node.bounds[0]))
        waypoints[node.label_assigned].append((node.bounds[1], node.bounds[2]))
        waypoints[node.label_assigned].append((node.bounds[3], node.bounds[0]))
        waypoints[node.label_assigned].append((node.bounds[3], node.bounds[2]))
    return waypoints

from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd

def get_convex_hulls(waypoints):
    gdf = gpd.GeoDataFrame()
    for label, points in waypoints.items():
        if label == -1:
            continue
        points = np.array(points)
        if points.shape[0] < 3:
            continue
        hull = ConvexHull(points)
        # add convex hull's vertices to the map
        convex_hull_polygon = Polygon(hull.points[hull.vertices])
        gdf = pd.concat([gdf, gpd.GeoDataFrame({'label': label, 'geometry': convex_hull_polygon}, index=[0], crs='EPSG:4326')])
        gdf['label'] = gdf.reset_index().index
        gdf.crs = 'EPSG:4326'
        gdf['centroid'] = gdf['geometry'].centroid.to_crs(crs='EPSG:4326')
    return gdf

def get_adjacency_shape_matrix(convex_hulls, df):
    waypoints = np.array(convex_hulls['centroid'].apply(lambda p: (p.y, p.x)).to_list())
    labels = convex_hulls['label'].values
    labels = labels[labels != -1]
    adjacency_matrix = np.zeros((labels.shape[0], labels.shape[0]))
    width_matrix = np.zeros((labels.shape[0], labels.shape[0]))
    length_matrix = np.zeros((labels.shape[0], labels.shape[0]))
    for mmsi, group in df.groupby('MMSI'):
        length_mean = group['width'].mean()
        width_mean = group['length'].mean()

        geometry = gpd.points_from_xy(group['longitude'], group['latitude'])
        result = gpd.sjoin(convex_hulls, gpd.GeoDataFrame(geometry=geometry, crs='EPSG:4326', index=group.index), how="inner", predicate="intersects")

        result.sort_values(by=['index_right'], inplace=True)
        
        if result['label'].unique().shape[0] < 2:
            continue
        idx = result['label'].values
        adjacency_matrix[idx[:-1], idx[1:]] += 1
        length_matrix[idx[:-1], idx[1:]] = np.maximum(length_matrix[idx[:-1], idx[1:]], length_mean)
        width_matrix[idx[:-1], idx[1:]] = np.maximum(width_matrix[idx[:-1], idx[1:]], width_mean)
    return adjacency_matrix, length_matrix, width_matrix

import utils.waypoint_extract_utils as weu

def get_edges_list(convex_hulls, adjacency_matrix, flow_threshold=2, dis_threshold=24):
    waypoints = np.array(convex_hulls['centroid'].apply(lambda p: (p.y, p.x)).to_list())
    edges_list = []
    dis_matrix = weu.calculate_geodesic_distance_matrix(waypoints, waypoints)

    for i in range(waypoints.shape[0]):
        for j in range(waypoints.shape[0]):

            if (adjacency_matrix[i,j] > flow_threshold) and (dis_matrix[i,j] < dis_threshold):
                edges_list.append([(waypoints[i][0],waypoints[i][1]),(waypoints[j][0], waypoints[j][1])])
            else:
                adjacency_matrix[i,j] = 0

    return edges_list, adjacency_matrix, dis_matrix

def visualize_node_list(df, node_list: [QuadTreeNode], color='red'):
    # visualize the node list
    m = folium.Map(location=[55, 12], zoom_start=6)
    ship_cnts = [len(node.ship_dic) for node in node_list]
    labels = df['cluster_label'].unique()
    labels = np.sort(labels)
    min_value, max_value = min(labels), max(labels)

    colormap = linear.YlGn_09.scale(min_value, max_value)
    # print(colormap(5.0))
    for node in node_list:
        if not node.valid or node.ship_cnt == 0:
            continue
        label_assigned = node.label_assigned
        if label_assigned == -1:
            continue
        # node.bounds = [lat_min, lon_min, lat_max, lon_max]
        # convert each node to a rectangle, lat come first
        folium.Rectangle(bounds=[node.bounds[0:2], node.bounds[2:4]], 
                         color=colormap(label_assigned), 
                         fill=True, 
                         fill_color=colormap(label_assigned),
                         popup='wp' + str(label_assigned),
                         fill_opacity=1
                         ).add_to(m)
    # add color bar
    colormap.caption = 'Values'
    colormap.add_to(m)
    return m