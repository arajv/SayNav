from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import prior
import numpy as np
import copy
from collections import defaultdict
import math
from statistics import mean
import hl_utils
import pointnav_utils
import random
from aStarPlanner import AStarPlanner
from utils_scenegraph import depth_to_world_coordinates
from utils_scenegraph import is_door_present

import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import arch.pointnav as arch_policy
import util.smartparse as smartparse
import util.db as db

class Agent:

    ### INITIALIZATION

    def __init__(self, controller, params):
        self.controller = controller
        self.reachable_positions = None
        self.picked_object = None
        self.nav_policy_type = params['nav_policy_type']
        self.num_steps = 0
        self.setup_top_down_camera()
        
        if self.nav_policy_type == 1:
            self.reachable_positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
            self.astar_planner = AStarPlanner(self.reachable_positions, params['grid_size'])
        
        elif self.nav_policy_type == 2:
            state=torch.load(params['policy_model'])
            policy=arch_policy.new()
            policy.to(0)
            policy.load_state_dict(state["net"])
            policy.eval()
            self.pointnav_policy=policy
        

    ### SETTERS

    def set_start_pose(self, start_position, start_heading = None):
        y = self.controller.last_event.metadata['agent']['position']['y']        
        position = {'x':start_position[0], 'y': y, 'z':start_position[1]}
        rotation = copy.deepcopy(self.controller.last_event.metadata['agent']['rotation'])        
        if start_heading is not None:
            rotation['y'] = start_heading
        
        self.controller.step(action="Teleport", position=position, rotation=rotation)

    def reset(self):
        self.num_steps = 0

    ### GETTERS

    def get_current_position(self):
        x = self.controller.last_event.metadata['agent']['position']['x']
        z = self.controller.last_event.metadata['agent']['position']['z']
        return (x, z)

    def get_current_heading(self):
        return self.controller.last_event.metadata['agent']['rotation']['y']

    def get_reachable_positions(self):
        if self.reachable_positions is None:
            self.reachable_positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        return self.reachable_positions

    def get_num_steps(self):
        return self.num_steps
    

    ### ACTIONS

    # Look around and return the visible objects around
    def lookAround(self, visuals = defaultdict(list), method = 0, minimum_pixel_size=1000, target_objects = [], display_string = ''):
        visible_objects = set()
        events = []
        step_rotation = 60 # deg, default value = 60
        objs_found = [False for i in range(len(target_objects))]
        

        for n in range(int(360/step_rotation)):
            new_event = self.controller.step(action="RotateRight", degrees=step_rotation)
            frame_size = new_event.depth_frame.size
            events.append(copy.deepcopy(new_event))            
            new_visible_objects = set()

            if method == 0 or method == 1:            
                new_visible_objects = {obj['objectType'] for obj in new_event.metadata["objects"] if obj['visible']}
                visible_objects = visible_objects.union(new_visible_objects)
                minimum_pixel_size = 0                

            elif method == 2:
                masks = new_event.instance_masks
                is_door_detected = is_door_present(new_event.instance_masks)
                largest_door_pixel_size = 0
                for key, val in masks.items():
                    if 'door' in key:
                        largest_door_pixel_size = max(largest_door_pixel_size, np.sum(val))

                    if minimum_pixel_size > 0:
                        if np.sum(val) < minimum_pixel_size:
                            continue 
                    
                    if 'room' in key or 'wall' in key:
                        continue                   

                    tokens = key.split('|')
                    new_visible_objects.add(tokens[0])
                if not is_door_detected or largest_door_pixel_size < frame_size / 8 :
                    visible_objects = visible_objects.union(new_visible_objects)
        
            if len(target_objects) > 0:
                visible_objects_l = {obj.lower() for obj in new_visible_objects}
                objects_found = []
                for i, obj in enumerate(target_objects):
                    if obj.lower() in visible_objects_l:
                        objs_found[i] = True
                        objects_found.append(obj)
                        print("Found: ", obj)                       
                
                hl_utils.get_objects_visuals_from_event(objects_found, new_event, visuals, minimum_pixel_size = minimum_pixel_size, display_string = display_string)                

            else:
                visuals['rgb'].append(copy.deepcopy(new_event.frame))
                visuals['top'].append(copy.deepcopy(new_event.third_party_camera_frames[0][:,:,:-1]))
                visuals['text'].append(display_string)
                visuals['instance'].append(np.array([]))                 
        
        if len(target_objects) > 0:
            return visible_objects, events, objs_found
        else:        
            return visible_objects, events

    # Navigate to a specific position or nearest location to that position with the specified position in the field of view
    def navigateTo(self, position, orientation = None, target_room_walls = None, visuals = defaultdict(list), use_reachable_position = True):
               
        nr_pos = self.get_nearest_reachable_position(position, target_room_walls)
        event = self.controller.last_event

        if self.nav_policy_type == 0: # Teleport to that point
            y = self.controller.last_event.metadata['agent']['position']['y']  
            nr_pos_dict = {'x':nr_pos[0], 'y': y, 'z':nr_pos[1]}        
            event = self.controller.step(action="Teleport", position=nr_pos_dict)
            event = self.orient_towards(position)
            visuals['rgb'].append(copy.deepcopy(event.frame))
            visuals['top'].append(copy.deepcopy(event.third_party_camera_frames[0][:,:,:-1]))

        elif self.nav_policy_type == 1: # Astar            
            path = self.astar_planner.get_path(self.get_current_position(), nr_pos)
            #print('Astar path = ', path)
            event = self.follow_path(path, visuals)
            if orientation is None:
                event = self.orient_towards(position, num_steps = 3, visuals = visuals)
            else:
                event = self.orient_at(orientation, num_steps = 3, visuals = visuals)
            self.num_steps += len(path) - 1

        elif self.nav_policy_type == 2: # Pointnav
            if not use_reachable_position:
                nr_pos = position
            y = self.controller.last_event.metadata['agent']['position']['y']  
            nr_pos_dict = {'x':nr_pos[0], 'y': y, 'z':nr_pos[1]} 
            event=self.pointnav(nr_pos_dict, visuals=visuals)
            if orientation is None:
                event = self.orient_towards(position, num_steps = 3, visuals = visuals)
            else:
                event = self.orient_at(orientation, num_steps = 3, visuals = visuals)         
        
        return event

    # Look for particular objects around you
    def lookFor(self, objects, allow_inplace_rotation = False, visuals = defaultdict(list)):
        visible_objects = {obj['objectType'].lower() for obj in self.controller.last_event.metadata["objects"] if obj['visible']}
        fov = 60 # deg, default value = 60
        objs_found = [False for i in range(len(objects))]
        rotation_angles = [0 for i in range(len(objects))]
        success = True
        any_one_success = False
        for i, obj in enumerate(objects):
            if obj.lower() in visible_objects:
                objs_found[i] = True
                any_one_success = True
            else:
                success = False
                
        if any_one_success:
            hl_utils.get_objects_visuals_from_event(objects, self.controller.last_event, visuals)

        rotation_idx = -1
        if allow_inplace_rotation and success == False:
            for n in range(6):
                new_event = self.controller.step(action="RotateRight", degrees=fov)
                if n != 5:
                    new_visible_objects = {obj['objectType'].lower() for obj in new_event.metadata["objects"] if obj['visible']}
                    for i, obj in enumerate(objects):                    
                        if obj.lower() in new_visible_objects:
                            rotation_idx = n 
                            objs_found[i] = True
                            rotation_angles[i] = (rotation_idx + 1)*fov                                             
                    hl_utils.get_objects_visuals_from_event(objects, new_event, visuals)                 
                    
        return objs_found, rotation_angles
    
    
    def navigateToNeighboringRoom(self, door, neighbor_room_num, curr_room_num, grouped_walls, visuals = defaultdict(list), sg_input_source = 0, minimum_pixel_size=1000):

        use_reachable_position = True
        target_room_walls=grouped_walls[curr_room_num]
        if sg_input_source == 2:
            use_reachable_position = False
            target_room_walls=None

        x_door = 0
        z_door = 0
        orientation = None
        if sg_input_source == 0 or sg_input_source == 1:
            # Navigate to Door
            tokens = door['wall0'].split('|')
            x1 = float(tokens[2])
            x2 = float(tokens[4])
            z1 = float(tokens[3])
            z2 = float(tokens[5])
            if x1 == x2:
                x_door = x1
                z_door = door['assetPosition']['x']
            else:
                z_door = z1
                x_door = door['assetPosition']['x']
        else:
            x_door = door['est_position']['x']
            z_door = door['est_position']['z']
            curr_pos = self.get_current_position()
            if door['est_nav_axis'] == 'x':
                if x_door > curr_pos[0]:
                    orientation = 90
                else:
                    orientation = 270
            elif door['est_nav_axis'] == 'z':
                if z_door > curr_pos[1]:
                    orientation = 0
                else:
                    orientation = 180
        
        event = self.navigateTo((x_door, z_door), orientation=orientation, target_room_walls=target_room_walls, visuals = visuals, use_reachable_position = use_reachable_position)
        
        for n in range(len(visuals['rgb']) - len(visuals['text'])):
            visuals['text'].append("Walking to the door")
        
        # Navigate to the other room
        x_mean = 0
        z_mean = 0
        ready = False
        if sg_input_source == 0 or sg_input_source == 1:
            all_x = []
            all_z = []        
            for wall in grouped_walls[neighbor_room_num]:
                tokens = wall['id'].split('|')
                x1 = float(tokens[2])
                x2 = float(tokens[4])
                z1 = float(tokens[3])
                z2 = float(tokens[5])
                all_x.extend([x1, x2])
                all_z.extend([z1, z2])
            x_mean = mean(all_x)
            z_mean = mean(all_z)
            ready = True
        else:
            detections = event.instance_detections2D
            camera_horizon = event.metadata['agent']['cameraHorizon']
            heading = event.metadata['agent']['rotation']['y']
            fov = event.metadata['fov']
            camera_pos = event.metadata['cameraPosition']
            world_coordinates = depth_to_world_coordinates(event.depth_frame, camera_pos, heading, camera_horizon, fov)
            min_dist = -1
            for key, val in event.instance_masks.items():
                if 'room' in key and str(curr_room_num) not in key:
                    det_box = detections[key]
                    coordinates = [world_coordinates[r][c] for r in range(det_box[1], det_box[3]+1) for c in range(det_box[0], det_box[2]+1) if val[r][c]]
                    x_room = mean([coord[0] for coord in coordinates])
                    z_room = mean([coord[2] for coord in coordinates])
                    dist = math.sqrt((x_room - x_door)*(x_room - x_door) + (z_room - z_door)*(z_room - z_door))
                    if min_dist == -1 or dist < min_dist:
                        min_dist = dist
                        x_mean = x_room
                        z_mean = z_room
                        ready = True
            
            if min_dist == -1:
                print("Didn't see floor through the door")
                for key, val in event.instance_masks.items():
                    if minimum_pixel_size > 0:
                        if np.sum(val) < minimum_pixel_size:
                            continue
                    if str(curr_room_num) not in key:
                        det_box = detections[key]
                        coordinates = [world_coordinates[r][c] for r in range(det_box[1], det_box[3]+1) for c in range(det_box[0], det_box[2]+1) if val[r][c]]
                        x_room = mean([coord[0] for coord in coordinates])
                        z_room = mean([coord[2] for coord in coordinates])
                        dist = math.sqrt((x_room - x_door)*(x_room - x_door) + (z_room - z_door)*(z_room - z_door))
                        if min_dist == -1 or dist < min_dist:
                            min_dist = dist
                            x_mean = x_room
                            z_mean = z_room
                            ready = True

        if ready:
            event = self.navigateTo((x_mean, z_mean), visuals = visuals, use_reachable_position = use_reachable_position)
            for n in range(len(visuals['rgb']) - len(visuals['text'])):
                visuals['text'].append("Walk into the neighboring room")

        return event

    def navigateToRoom(self, room_num, grouped_walls, visuals = defaultdict(list)):
        all_x = []
        all_z = []        
        for wall in grouped_walls[room_num]:
            tokens = wall['id'].split('|')
            x1 = float(tokens[2])
            x2 = float(tokens[4])
            z1 = float(tokens[3])
            z2 = float(tokens[5])
            all_x.extend([x1, x2])
            all_z.extend([z1, z2])
        x_mean = mean(all_x)
        z_mean = mean(all_z)

        nr_pos = self.get_nearest_reachable_position((x_mean, z_mean))
        y = self.controller.last_event.metadata['agent']['position']['y']  
        nr_pos_dict = {'x':nr_pos[0], 'y': y, 'z':nr_pos[1]}        
        event = self.controller.step(action="Teleport", position=nr_pos_dict)
        visuals['rgb'].append(copy.deepcopy(event.frame))
        visuals['top'].append(copy.deepcopy(event.third_party_camera_frames[0][:,:,:-1]))

        return event

    def pick(self, desired_object, visuals = defaultdict(list)):
        
        if self.picked_object is not None:
            print("Another object in hand")
            visuals['rgb'].append(copy.deepcopy(self.controller.last_event.frame))
            visuals['top'].append(copy.deepcopy(self.controller.last_event.third_party_camera_frames[0][:,:,:-1]))
            return None
        
        options = [obj for obj in self.controller.last_event.metadata["objects"] \
                     if obj['visible'] and obj['pickupable'] and not obj['isPickedUp'] \
                     and desired_object.lower() in obj['objectType'].lower()]

        if len(options) == 0:
            print("No such pickable object found")
            visuals['rgb'].append(copy.deepcopy(self.controller.last_event.frame))
            visuals['top'].append(copy.deepcopy(self.controller.last_event.third_party_camera_frames[0][:,:,:-1]))
            return None

        chosen_object = random.choice(options)
        event = self.controller.step(action="PickupObject",  objectId=chosen_object['objectId'])
        
        if event.metadata["lastActionSuccess"]:
            self.picked_object = chosen_object
        else:
            self.picked_object = None

        visuals['rgb'].append(copy.deepcopy(event.frame))
        visuals['top'].append(copy.deepcopy(event.third_party_camera_frames[0][:,:,:-1]))
        return event

    def place(self, receptacle, visuals = defaultdict(list)):
        
        if self.picked_object == None:
            print("Nothing to place")
            visuals['rgb'].append(copy.deepcopy(self.controller.last_event.frame))
            visuals['top'].append(copy.deepcopy(self.controller.last_event.third_party_camera_frames[0][:,:,:-1]))
            return None
        
        options = [obj for obj in self.controller.last_event.metadata["objects"] \
                     if obj['visible'] and obj['receptacle'] \
                     and receptacle.lower() in obj['objectType'].lower()]

        if len(options) == 0:
            print("No such receptacle object found")
            visuals['rgb'].append(copy.deepcopy(self.controller.last_event.frame))
            visuals['top'].append(copy.deepcopy(self.controller.last_event.third_party_camera_frames[0][:,:,:-1]))
            return None

        chosen_receptacle = random.choice(options)
        event = self.controller.step(action="PutObject",  objectId=chosen_receptacle['objectId'])
        
        if event.metadata["lastActionSuccess"]:
            self.picked_object = None
        
        visuals['rgb'].append(copy.deepcopy(event.frame))
        visuals['top'].append(copy.deepcopy(event.third_party_camera_frames[0][:,:,:-1]))
        return event

    def open(self, desired_object, visuals = defaultdict(list)):
        options = [obj for obj in self.controller.last_event.metadata["objects"] \
                     if obj['visible'] and obj['openable'] and not obj['isOpen'] \
                     and desired_object.lower() in obj['objectType'].lower()]

        if len(options) == 0:
            print("No such openable object found")
            visuals['rgb'].append(copy.deepcopy(self.controller.last_event.frame))
            visuals['top'].append(copy.deepcopy(self.controller.last_event.third_party_camera_frames[0][:,:,:-1]))
            return None

        chosen_object = random.choice(options)
        event = self.controller.step(action="OpenObject",  objectId=chosen_object['objectId'])

        visuals['rgb'].append(copy.deepcopy(event.frame))
        visuals['top'].append(copy.deepcopy(event.third_party_camera_frames[0][:,:,:-1]))
        return event

    def close(self, desired_object, visuals = defaultdict(list)):
        options = [obj for obj in self.controller.last_event.metadata["objects"] \
                     if obj['visible'] and obj['openable'] and obj['isOpen'] \
                     and desired_object.lower() in obj['objectType'].lower()]

        if len(options) == 0:
            print("No such opened object found")
            visuals['rgb'].append(copy.deepcopy(self.controller.last_event.frame))
            visuals['top'].append(copy.deepcopy(self.controller.last_event.third_party_camera_frames[0][:,:,:-1]))
            return None

        chosen_object = random.choice(options)
        event = self.controller.step(action="CloseObject",  objectId=chosen_object['objectId'])

        visuals['rgb'].append(copy.deepcopy(event.frame))
        visuals['top'].append(copy.deepcopy(event.third_party_camera_frames[0][:,:,:-1]))
        return event
    
    
    
    ### HELPER / UTIL FUNCTIONS

    def add_rgb_from_latest_event(self, visuals = defaultdict(list)):
        visuals['rgb'].append(copy.deepcopy(self.controller.last_event.frame))
        visuals['top'].append(copy.deepcopy(self.controller.last_event.third_party_camera_frames[0][:,:,:-1]))

    def follow_path(self, path, visuals = defaultdict(list)):
        event = self.controller.last_event
        for i in range(len(path)-1):
            next_wp = path[i+1]
            event = self.orient_towards(next_wp, num_steps = 3, visuals = visuals)
            #print("curr heading = ", self.get_current_heading())
            event = self.controller.step(action="MoveAhead")
            #print("curr position = ", self.get_current_position())
            visuals['rgb'].append(copy.deepcopy(event.frame))
            visuals['top'].append(copy.deepcopy(event.third_party_camera_frames[0][:,:,:-1]))
        return event

    
    def orient_at(self, angle, num_steps = 1, visuals = defaultdict(list)):
        eps = 0.01
        event = self.controller.last_event
        
        desired_heading = angle
        if desired_heading < 0:
            desired_heading = 360 + desired_heading
        curr_heading = self.get_current_heading()
        delta_heading = desired_heading - curr_heading
        #print("Curr Heading = ", curr_heading)
        #print("Desired Heading = ", desired_heading)
        
        if abs(delta_heading) > eps:
            if num_steps == 1:
                event = self.controller.step(action="RotateRight", degrees=delta_heading)
                visuals['rgb'].append(copy.deepcopy(event.frame))
                visuals['top'].append(copy.deepcopy(event.third_party_camera_frames[0][:,:,:-1]))
            else:
                for n in range(num_steps):
                    event = self.controller.step(action="RotateRight", degrees= delta_heading / num_steps)
                    visuals['rgb'].append(copy.deepcopy(event.frame))
                    visuals['top'].append(copy.deepcopy(event.third_party_camera_frames[0][:,:,:-1]))

        return event
    
    def orient_towards(self, point, num_steps = 1, visuals = defaultdict(list)):
        curr_pos = self.get_current_position()
        event = self.controller.last_event
        if point[0] != curr_pos[0] or point[1] != curr_pos[1]:
            view_direction = [point[0] - curr_pos[0], point[1] - curr_pos[1]]
            desired_heading = math.degrees(math.atan2(view_direction[0], view_direction[1]))
            event = self.orient_at(desired_heading, num_steps = num_steps, visuals = visuals)

        return event
        
    def get_nearest_reachable_position(self, position, target_room_walls = None):
        if self.reachable_positions is None:
            self.reachable_positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        
        min_idx = -1
        if target_room_walls is not None:
            #Ensure that the target point is inside the target room
            distances = [ (np.linalg.norm(np.array([ position[0] - pos['x'], position[1] - pos['z'] ])), i) for i, pos in enumerate(self.reachable_positions)]
            distances.sort()
            for tup in distances:
                point = (self.reachable_positions[tup[1]]['x'], self.reachable_positions[tup[1]]['z'])
                if hl_utils.is_point_inside_room(point, target_room_walls):
                    min_idx = tup[1]
                    break

            if min_idx == -1:
                min_idx = distances[0][1]
        else:
            distances = [ np.linalg.norm(np.array([ position[0] - pos['x'], position[1] - pos['z'] ])) for pos in self.reachable_positions ]    
            min_idx = np.argmin(distances)

        return (self.reachable_positions[min_idx]['x'], self.reachable_positions[min_idx]['z'])
    
    def get_farthest_reachable_position(self, position, target_room_walls = None, x = None, z = None):
        if self.reachable_positions is None:
            self.reachable_positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        
        max_idx = -1
        if target_room_walls is not None:
            #Ensure that the target point is inside the target room
            distances = [ (np.linalg.norm(np.array([ position[0] - pos['x'], position[1] - pos['z'] ])), i) for i, pos in enumerate(self.reachable_positions)]
            distances.sort(reverse=True)
            for tup in distances:
                point = (self.reachable_positions[tup[1]]['x'], self.reachable_positions[tup[1]]['z'])
                if hl_utils.is_point_inside_room(point, target_room_walls):
                    if x is not None:
                        if point[0] == x:
                            max_idx = tup[1]
                            break
                    if z is not None:
                        if point[1] == z:
                            max_idx = tup[1]
                            break

            if max_idx == -1:
                max_idx = distances[0][1]
        else:
            distances = [ np.linalg.norm(np.array([ position[0] - pos['x'], position[1] - pos['z'] ])) for pos in self.reachable_positions ]    
            max_idx = np.argmax(distances)

        return (self.reachable_positions[max_idx]['x'], self.reachable_positions[max_idx]['z'])
    
    def pointnav(self,goal_pos,nsteps=500, visuals = defaultdict(list)):
        #First correct initial facing angle to multiples of 90 degrees
        ry=self.get_current_heading()
        if not ry%90==0:
            event = self.controller.step(action="RotateLeft", degrees=ry%90)
        
        prev_agent_pos = self.get_current_position()
        #ry=self.get_current_heading()
        #assert ry%90==0
        data=self.controller.last_event
        with torch.no_grad():
                        
            actions=['Stop','MoveAhead','RotateLeft','RotateRight','LookUp','LookDown']
            h = torch.zeros((1,self.pointnav_policy.num_recurrent_layers,512),device=0)
            action = torch.zeros(1,1,device=0,dtype=torch.long)
            not_done = torch.zeros(1,1,device=0,dtype=torch.bool)
            
            for i in range(nsteps):
                obs=pointnav_utils.extract_obs_pointnav(data,goal_pos)
                obs=db.Table.from_rows([obs]).d
                obs={k:v.to(0) for k,v in obs.items()}
                reward,info=pointnav_utils.extract_info(data,goal_pos)
                
                pred = self.pointnav_policy.act(obs,h,action,not_done,deterministic=False,)
                h=pred.rnn_hidden_states
                action=pred.actions
                step_action = [a.item() for a in pred.actions.cpu()][0]
                not_done[0,0]=1
                
                #print(obs['gps'],obs['pointgoal'],step_action)
                #Get environment outputs
                if step_action==0:
                    break
                elif step_action==1:
                    data=self.controller.step(actions[step_action])
                    curr_agent_pos = self.get_current_position()
                    dist = math.sqrt((curr_agent_pos[0] - prev_agent_pos[0])**2 + (curr_agent_pos[1] - prev_agent_pos[1])**2)
                    if dist > 0:
                        self.num_steps += 1
                    prev_agent_pos = curr_agent_pos
                elif step_action in [2,3,4,5]:
                    data=self.controller.step(actions[step_action],degrees=90)
                else:
                    a=0/0

                visuals['rgb'].append(copy.deepcopy(data.frame))
                visuals['top'].append(copy.deepcopy(data.third_party_camera_frames[0][:,:,:-1]))

        return data
    
    
    def setup_top_down_camera(self):
        event = self.controller.step(action="GetMapViewCameraProperties", raise_for_failure=True)
        pose = copy.deepcopy(event.metadata["actionReturn"])

        bounds = event.metadata["sceneBounds"]["size"]
        max_bound = max(bounds["x"], bounds["z"])

        pose["fieldOfView"] = 50
        pose["position"]["y"] += 1.0 * max_bound
        pose["orthographic"] = False
        pose["farClippingPlane"] = 50
        del pose["orthographicSize"]

        # add the camera to the scene
        event = self.controller.step(
            action="AddThirdPartyCamera",
            **pose,
            skyboxColor="white",
            raise_for_failure=True,
        )
