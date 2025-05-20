from typing import Union
from enum import Enum 
import numpy as np
from types import SimpleNamespace as SN
from scipy.spatial.transform import Rotation as R
import math

def objectID_to_roomID(group_objects_by_rooms):
    objectID2roomID = dict()
    for rm_num, obj_list in group_objects_by_rooms.items():
        for obj in obj_list:
            objectID2roomID[obj['id']]=f'room|{rm_num}'
    return objectID2roomID

def extract_room_polygons(rooms_info):
    room_ranges = {}
    for rm_num, rm_info in rooms_info.items():
        x_range = np.array([1e6,-1e6],dtype=np.float32)
        z_range = np.array([1e6,-1e6],dtype=np.float32)
        for pt in rm_info['floorPolygon']:
            if pt['x']<x_range[0]:
                x_range[0] = pt['x']
            if pt['x']>x_range[1]:
                x_range[1] = pt['x']
            if pt['z']<z_range[0]:
                z_range[0] = pt['z']
            if pt['z']>z_range[1]:
                z_range[1] = pt['z']
        room_ranges[rm_num] = SN(x=x_range, z=z_range)
    return room_ranges


def locate_in_room(rooms_info, room_ranges, x, z):
    for rm_num, rm_info in rooms_info.items():
        within_x = x > room_ranges[rm_num].x[0] and x < room_ranges[rm_num].x[1]
        if within_x:
            within_z = z > room_ranges[rm_num].z[0] and z < room_ranges[rm_num].z[1]
            if within_z:
                return (rm_num, rm_info)
    return (-1, None)

def getCurrentRoom(position, rooms_info, room_ranges):
    # event_data = controller.last_event.metadata
    #(x,_,z) = tuple(event_metadata['agent']['position'].values())   
    (rm_id, rm_info) = locate_in_room(rooms_info, room_ranges, position[0], position[1])
    return (rm_id, rm_info)

def mean_position(position1, position2, weight1 = np.array([1.0, 1.0, 1.0]), weight2 = np.array([1.0, 1.0, 1.0])):
    mean_pos = {}
    sum_of_weights = weight1 + weight2

    mean_pos['x'] = (weight1[0] * position1['x'] + weight2[0] * position2['x']) / sum_of_weights[0]
    mean_pos['y'] = (weight1[1] * position1['y'] + weight2[1] * position2['y']) / sum_of_weights[1]
    mean_pos['z'] = (weight1[2] * position1['z'] + weight2[2] * position2['z']) / sum_of_weights[2]
    return mean_pos


def project_point_on_frame(point, frame, camera_pos, heading, camera_horizon, fov):
    height = frame.shape[0]
    width = frame.shape[1]
    fov_rad = math.radians(fov)

    diag = math.sqrt(width*width + height*height)
    focal_length = diag / (2.0 * math.tan(fov_rad/2.0))
    fx = (focal_length * width) / diag
    fy = (focal_length * height) / diag

    K = np.array([[fx, 0, width/ 2], [0, fy, height / 2], [0, 0, 1]])

    world_pt = np.array([[point['x'], point['y'], point['z']]]).transpose()
    rot_matrix1 = np.array([[math.cos(np.radians(heading)), 0, math.sin(np.radians(heading))], [0, -1, 0], [-math.sin(np.radians(heading)), 0, math.cos(np.radians(heading))]])
    rot_matrix2 = np.array([[1, 0, 0], [0, math.cos(np.radians(camera_horizon)), math.sin(np.radians(camera_horizon))], [0, -math.sin(np.radians(camera_horizon)), math.cos(np.radians(camera_horizon))]])
    rot_matrix = np.matmul(rot_matrix1, rot_matrix2)
    trans = np.array([[camera_pos['x'], camera_pos['y'], camera_pos['z']]]).transpose()
    cam_pt = np.matmul(rot_matrix.transpose(), world_pt) - np.matmul(rot_matrix.transpose(), trans)
    img_pixel = np.matmul(K, cam_pt)
    img_pixel = img_pixel / img_pixel[2][0]
    img_pixel[0][0] = int(round(img_pixel[0][0]))
    if img_pixel[0][0] < 0:
        img_pixel[0][0] = 0
    if img_pixel[0][0] >= width:
        img_pixel[0][0] = width - 1
    img_pixel[1][0] = int(round(img_pixel[1][0]))
    if img_pixel[1][0] < 0:
        img_pixel[1][0] = 0
    if img_pixel[1][0] >= height:
        img_pixel[1][0] = height - 1
    
    return (img_pixel[0][0], img_pixel[1][0]) 

def depth_to_world_coordinates(depth_frame, camera_pos, heading, camera_horizon, fov):
    height = depth_frame.shape[0]
    width = depth_frame.shape[1]
    fov_rad = math.radians(fov)

    diag = math.sqrt(width*width + height*height)
    focal_length = diag / (2.0 * math.tan(fov_rad/2.0))
    fx = (focal_length * width) / diag
    fy = (focal_length * height) / diag

    K = np.array([[fx, 0, width/ 2], [0, fy, height / 2], [0, 0, 1]])
    K_inv = np.linalg.inv(K)

    #AI2Thor coordinate system is LHS. When agent's heading is 0: Z-front, X-right, Y-up
    rot_matrix1 = np.array([[math.cos(np.radians(heading)), 0, math.sin(np.radians(heading))], [0, -1, 0], [-math.sin(np.radians(heading)), 0, math.cos(np.radians(heading))]])
    rot_matrix2 = np.array([[1, 0, 0], [0, math.cos(np.radians(camera_horizon)), math.sin(np.radians(camera_horizon))], [0, -math.sin(np.radians(camera_horizon)), math.cos(np.radians(camera_horizon))]])
    rot_matrix = np.matmul(rot_matrix1, rot_matrix2)
    trans = np.array([[camera_pos['x'], camera_pos['y'], camera_pos['z']] for i in range(height * width)]).transpose()

    pixel_coordinates = np.array([[ depth_frame[r, c] * np.array([c, r, 1]) for c in range(width)] for r in range(height)])
    pixel_coordinates = pixel_coordinates.reshape((height*width, 3)).transpose()
    camera_coordinates = np.matmul(K_inv, pixel_coordinates)
    world_coordinates = np.matmul(rot_matrix, camera_coordinates) + trans
    world_coordinates = world_coordinates.transpose().reshape((height, width, 3))

    return world_coordinates

def is_door_present(instance_masks):
    for key in instance_masks.keys():
        if 'door' in key:
            return True
    return False

def get_all_walls(instance_masks, detections, world_coordinates):
    walls = []
    for key, val in instance_masks.items():
        det_box = detections[key]
        if 'wall' in key:
            wall = {}
            wall_coordinates = [world_coordinates[r][c] for r in range(det_box[1], det_box[3]+1) for c in range(det_box[0], det_box[2]+1) if val[r][c]]
            x_coords = [coord[0] for coord in wall_coordinates]
            min_x = min(x_coords)
            max_x = max(x_coords)
            z_coords = [coord[2] for coord in wall_coordinates]
            min_z = min(z_coords)
            max_z = max(z_coords)
            x_range = max_x - min_x
            z_range = max_z - min_z
            if x_range < z_range:
                wall['x'] = ((min_x+max_x)/2.0, (min_x+max_x)/2.0)
                wall['z'] = (min_z, max_z)
                walls.append(wall)
            else:
                wall['z'] = ((min_z+max_z)/2.0, (min_z+max_z)/2.0)
                wall['x'] = (min_x, max_x)
                walls.append(wall)
    return walls

def match_door_with_walls(door_coordinates, walls, thresh = 0.01, min_matches = 10):
    num_matches = [0 for i in range(len(walls))]
    for coord in door_coordinates:
        for i, wall in enumerate(walls):
            if wall['x'][0] == wall['x'][1]:
                if abs(coord[0] - wall['x'][0]) < thresh:
                    num_matches[i] += 1
            elif wall['z'][0] == wall['z'][1]:
                if abs(coord[2] - wall['z'][0]) < thresh:
                    num_matches[i] += 1

    max_match = max(num_matches)
    if max_match > min_matches:
        return num_matches.index(max_match)
    else:
        return -1

def get_door_opening_extent(door_coordinates, matching_wall, thresh = 0.01):
    matching_coordinates = []
    for coord in door_coordinates:
        if matching_wall['x'][0] == matching_wall['x'][1]:
            if abs(coord[0] - matching_wall['x'][0]) < thresh:
                matching_coordinates.append(coord)
        elif matching_wall['z'][0] == matching_wall['z'][1]:
            if abs(coord[2] - matching_wall['z'][0]) < thresh:
                matching_coordinates.append(coord)
    
    min_x = None
    max_x = None
    min_z = None
    max_z = None
    if matching_wall['x'][0] == matching_wall['x'][1]:
        min_x = matching_wall['x'][0]
        max_x = matching_wall['x'][0]
    else:
        x_coords = [coord[0] for coord in matching_coordinates]
        min_x = min(x_coords)
        max_x = max(x_coords)

    y_coords = [coord[1] for coord in matching_coordinates]
    min_y = min(y_coords)
    max_y = max(y_coords)
    
    if matching_wall['z'][0] == matching_wall['z'][1]:
        min_z = matching_wall['z'][0]
        max_z = matching_wall['z'][0]
    else:
        z_coords = [coord[2] for coord in matching_coordinates]
        min_z = min(z_coords)
        max_z = max(z_coords)
    
    return (min_x, min_y, min_z, max_x, max_y, max_z)
