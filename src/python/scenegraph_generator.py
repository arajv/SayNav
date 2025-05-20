from house import House
from scenegraph import Node, NodeType
from scenegraph import Edge, EdgeType
from scenegraph import SceneGraph
from utils_scenegraph import *

from statistics import mean
import numpy as np
import copy
from collections import defaultdict
import math

class BaseSceneGraphGenerator:
    def __init__(self, scene_id, verbose=False ):
        # create the root of SG
        self._scene_id = scene_id
        root_node = Node(self._scene_id, {'data':[]})
        self.sg = SceneGraph(self._scene_id)
        self.sg.addNode(root_node) # this is the root
        self.verbose = False
        if verbose:
            self.verbose = True
            self.sg.print_()

    def updateGraph(self):
        raise NotImplementedError

class SceneGraphGenerator(BaseSceneGraphGenerator):
    # This version is very much tied to AI2Thor's ground truth info
    def __init__(self, scene_id, house:House, params, verbose=False):
        super().__init__(scene_id, verbose)
        self.params = params

        # create 'cheating' info from gt house info
        housedata = house     
        #housedata.group_objects_by_rooms(include_children_objects=True)   
        self._objects_info = housedata.grouped_objects_including_children
        self._rooms_info = housedata.grouped_room_info
        self._objectID2name={}
        self._objectID2info={}
        for _ , obj_info_list in self._objects_info.items():
            for obj_info in obj_info_list:
                tokens = obj_info['id'].split('|')
                self._objectID2name[obj_info['id']]=tokens[0]
                self._objectID2info[obj_info['id']]=obj_info
        self._objectID2roomID = objectID_to_roomID(self._objects_info)
        self._roomID2name = {rm_info['id']: rm_info['roomName'] for rm_no, rm_info in self._rooms_info.items()}
        self.roomNum2ID = {rm_no:rm_info['id'] for rm_no, rm_info in self._rooms_info.items()}
        self.roomID2Num = {rm_info['id']:rm_no for rm_no, rm_info in self._rooms_info.items()}
        self._room_ranges = extract_room_polygons(self._rooms_info)

        # create 'cheating' door info
        self.door_connections = {}
        self.door_coordinates = {}
        self.est_door_coordinates = {}
        self.est_door_coordinates_weights = {}
        self.est_door_nav_axis = {}        
        self._doors_info =  {door['id']: door for door in housedata.get_all_doors()}
        for door_id, door_info in self._doors_info.items():
            tokens = door_id.split('|')
            room1_num = int(tokens[1])
            room2_num = int(tokens[2])
            # ignore if the door leads to no room (e.g., outside the house)
            if (room1_num not in self._rooms_info) or (room2_num not in self._rooms_info):
                continue            
            self.door_connections[door_id] = (room1_num, room2_num)
            tokens = door_info['wall0'].split('|')
            x1 = float(tokens[2])
            x2 = float(tokens[4])
            x = (x1+x2)/2
            z1 = float(tokens[3])
            z2 = float(tokens[5])
            z = (z1+z2)/2
            self.door_coordinates[door_id] = (x, z)
        
        self.skip_keywords = ['wall', 'Painting']

    def add_keyword_to_filter(self, keyword):
        if isinstance(keyword, list):
            for kword in keyword:
                if kword not in self.skip_keywords:
                    self.skip_keywords.append(kword)
        else:
            if keyword not in self.skip_keywords:
                self.skip_keywords.append(keyword)

    
    def updateGraphFromObservations(self, events, method = 1, minimum_pixel_size=1000, visuals = defaultdict(list)):        
        for event in events:

            agent_position = (event.metadata['agent']['position']['x'], event.metadata['agent']['position']['z'])
            visible_objects_ids = []
            visible_objects_weighted_positions = []
            camera_horizon = event.metadata['agent']['cameraHorizon']
            heading = event.metadata['agent']['rotation']['y']
            fov = event.metadata['fov']
            camera_pos = event.metadata['cameraPosition']

            if method==1:
                visible_objects_ids = [obj['name'] for obj in event.metadata["objects"] if obj['visible']]
            
            elif method==2:            
                
                world_coordinates = depth_to_world_coordinates(event.depth_frame, camera_pos, heading, camera_horizon, fov)

                # Detections are used to avoid searching on whole image
                detections = event.instance_detections2D

                walls = []
                is_door_detected = is_door_present(event.instance_masks)                
                if is_door_detected:
                    walls = get_all_walls(event.instance_masks, detections, world_coordinates)

                for key, val in event.instance_masks.items():                    
                    skip = False
                    for skip_keyword in self.skip_keywords:
                        if skip_keyword in key:
                            skip = True
                            break
                    if skip:
                        continue
                    
                    if minimum_pixel_size > 0:
                        if np.sum(val) < minimum_pixel_size:
                            continue
                    
                    det_box = detections[key]
                    coordinates = [world_coordinates[r][c] for r in range(det_box[1], det_box[3]+1) for c in range(det_box[0], det_box[2]+1) if val[r][c]]
                    
                    if 'door' in key:

                        #x_coords = [coord[0] for coord in coordinates]
                        #y_coords = [coord[1] for coord in coordinates]
                        #z_coords = [coord[2] for coord in coordinates]
                        #mean_x = mean(x_coords)
                        #mean_y = mean(y_coords)
                        #ean_z = mean(z_coords)
                        #print(key, " Agent Position = ", agent_position)
                        #print(key, " Agent Heading = ", heading)
                        #print(key, " Avg Distance = ", math.sqrt((agent_position[0] - mean_x)**2 + (agent_position[1] - mean_z)**2) )

                        matching_wall_idx = -1
                        if len(walls) > 0:
                            matching_wall_idx = match_door_with_walls(coordinates, walls, thresh = self.params['door_wall_matching_thresh_dist'], min_matches = 10)
                        
                        if matching_wall_idx != -1:
                            door_opening_extent = get_door_opening_extent(coordinates, walls[matching_wall_idx], thresh = self.params['door_wall_matching_thresh_dist'])
                            mean_x = (door_opening_extent[0] + door_opening_extent[3])/2.0
                            mean_y = (door_opening_extent[1] + door_opening_extent[4])/2.0
                            mean_z = (door_opening_extent[2] + door_opening_extent[5])/2.0
                            position = {'x':mean_x, 'y':mean_y, 'z':mean_z}
                            range_x = door_opening_extent[3] - door_opening_extent[0]
                            range_y = door_opening_extent[4] - door_opening_extent[1]
                            range_z = door_opening_extent[5] - door_opening_extent[2]
                            weight = np.array([max(0.001, range_x), max(0.001, range_y), max(0.001, range_z)])
                            nav_axis = 'x'
                            if abs(door_opening_extent[5] - door_opening_extent[2]) < abs(door_opening_extent[3] - door_opening_extent[0]):
                                nav_axis = 'z'
                            visible_objects_weighted_positions.append((weight, position, nav_axis))
                        else:
                            continue
                    else:
                        x_coords = [coord[0] for coord in coordinates]
                        y_coords = [coord[1] for coord in coordinates]
                        z_coords = [coord[2] for coord in coordinates]
                        mean_x = mean(x_coords)
                        mean_y = mean(y_coords)
                        mean_z = mean(z_coords)
                        position = {'x':mean_x, 'y':mean_y, 'z':mean_z}
                        range_x = max(x_coords) - min(x_coords)
                        range_y = max(y_coords) - min(y_coords)
                        range_z = max(z_coords) - min(z_coords)
                        weight = np.array([max(0.001, range_x), max(0.001, range_y), max(0.001, range_z)])
                        visible_objects_weighted_positions.append((weight, position))
                                        
                    if '___' in key:
                        # instance segmentation frame can contain multiple segments of the same object instance
                        tokens = key.split('___')
                        visible_objects_ids.append(tokens[0])
                    else:
                        visible_objects_ids.append(key)

            self.updateGraph(agent_position, visible_objects_ids, visible_objects_weighted_positions)
            if len(visible_objects_weighted_positions) > 0:
                for key in visible_objects_ids:
                    if 'door' in key and key in self.est_door_coordinates:
                        rgb_frame = copy.deepcopy(event.frame)
                        door_pos = self.est_door_coordinates[key]
                        door_img_pos = project_point_on_frame(door_pos, event.frame, camera_pos, heading, camera_horizon, fov)
                        visuals['doors'].append((rgb_frame, key, door_img_pos))

            
    def updateGraph(self, agent_position, objects_ids, objects_weighted_positions = []):
        # identify which room the agent is located now
        rm_number, rm_info = getCurrentRoom(agent_position, self._rooms_info, self._room_ranges)  
        if self.verbose:
            number_of_nodes = self.sg.numNodes()
            print(f"[GraphGenerator] before update, there are {number_of_nodes} nodes.")
            print(f"Agent is in {rm_info['roomName']} (id:{rm_number})")
        
        # add the room to SG if not exists
        rm_id = rm_info['id']
        if not self.sg.isExist(rm_id):
            rm_node = Node(rm_id, node_type=NodeType.Room, data=rm_info)
            if self.verbose:
                print(f"[GraphGenerator] {rm_id} is added. (This is where the robot is located now.)")
            self.sg.addNode(rm_node, self._scene_id)
        
        new_edges = []
        for idx, obj_id in enumerate(objects_ids):  
            # handle diffeerent types of objects differently              
            if 'room' in obj_id and 'Ceiling' not in obj_id:
                # if it's a room 
                # add the object to SG
                
                if not self.sg.isExist(obj_id): 
                    if len(objects_weighted_positions) > 0:                        
                        rm_info['est_position'] = objects_weighted_positions[idx][1]
                        rm_info['est_position_weight'] = objects_weighted_positions[idx][0] 

                    obj_node = Node(obj_id, node_type=NodeType.Room, data=rm_info)
                    
                    if self.verbose:
                        print(f"[GraphGenerator] {obj_id} is added.")
                    self.sg.addNode(obj_node, self._scene_id)
                else:
                    if len(objects_weighted_positions) > 0:
                        obj_node = self.sg.getNode(obj_id)
                        if 'est_position' not in obj_node.data:
                            obj_node.data['est_position'] = objects_weighted_positions[idx][1]
                            obj_node.data['est_position_weight'] = objects_weighted_positions[idx][0]
                        else:                            
                            if obj_node is not None:
                                obj_node.data['est_position'] = mean_position(obj_node.data['est_position'], objects_weighted_positions[idx][1], obj_node.data['est_position_weight'], objects_weighted_positions[idx][0])
                                obj_node.data['est_position_weight'] += objects_weighted_positions[idx][0]
                    
                continue

            elif 'door' in obj_id:
                if obj_id not in self.door_connections:
                    continue
                rms = self.door_connections[obj_id]
                node_id_src = self._rooms_info[rms[0]]['id']
                node_id_to = self._rooms_info[rms[1]]['id']
                edge_id = (node_id_src, node_id_to,  EdgeType.Door)
                
                # add room nodes if not exist
                if not self.sg.isExist(node_id_src):
                    rm1_node = Node(node_id_src, node_type=NodeType.Room, data=self._rooms_info[rms[0]])
                    if self.verbose:
                        print(f"[GraphGenerator] {node_id_src} is added.")
                    self.sg.addNode(rm1_node, self._scene_id)
                if not self.sg.isExist(node_id_to):
                    rm2_node = Node(node_id_to, node_type=NodeType.Room, data=self._rooms_info[rms[1]])
                    if self.verbose:
                        print(f"[GraphGenerator] {node_id_to} is added.")
                    self.sg.addNode(rm2_node, self._scene_id)
                if not self.sg.isEdgeExist(edge_id):
                    if len(objects_weighted_positions) > 0:
                        self.est_door_coordinates[obj_id] = objects_weighted_positions[idx][1]
                        self.est_door_coordinates_weights[obj_id] = objects_weighted_positions[idx][0]
                        self.est_door_nav_axis[obj_id] = objects_weighted_positions[idx][2]  
                        if self.verbose:                     
                            print(obj_id, " gt position: ", self.door_coordinates[obj_id])
                            print(obj_id, " est position: ", self.est_door_coordinates[obj_id])
                            print(obj_id, " est position weights: ", self.est_door_coordinates_weights[obj_id])
                        
                    door_edge1 = Edge(node_id_src, node_id_to, EdgeType.Door, edge_data={'id':obj_id})
                    self.sg.addEdge(door_edge1)
                    door_edge2 = Edge(node_id_to, node_id_src, EdgeType.Door, edge_data={'id':obj_id})
                    self.sg.addEdge(door_edge2)
                    if self.verbose:
                        print(f"new door is added: {edge_id}, {obj_id}")
                else:  
                    # make changes to the edge if needed
                    if len(objects_weighted_positions) > 0:                        
                        self.est_door_coordinates[obj_id] = mean_position(self.est_door_coordinates[obj_id], objects_weighted_positions[idx][1], self.est_door_coordinates_weights[obj_id], objects_weighted_positions[idx][0])           
                        self.est_door_coordinates_weights[obj_id] += objects_weighted_positions[idx][0]                        
                        
                    if self.verbose:
                        print(obj_id, " gt position: ", self.door_coordinates[obj_id])
                        print(obj_id, " raw est position: ", objects_weighted_positions[idx][1])
                        print(obj_id, " raw est position weights: ", objects_weighted_positions[idx][0])
                        print(obj_id, " est position: ", self.est_door_coordinates[obj_id])
                        print(obj_id, " est position weights: ", self.est_door_coordinates_weights[obj_id])
                        print(f"{edge_id} already exists and updated.")
                continue
            else:
                # skip un-handled cases
                skip = False
                for skip_keyword in self.skip_keywords:
                    if skip_keyword in obj_id:
                        skip = True
                        break
                if obj_id not in self._objectID2roomID:
                    skip = True
                if skip:
                    if self.verbose:
                        print(f"[GraphGenerator] unable to handle. {obj_id} is skipped.")
                    continue       
                # get rooms where the objects are
                if obj_id not in self._objectID2roomID:
                    print(f"unidentified object {obj_id}")
                    continue
                obj_info = self._objectID2info[obj_id]
                rm_id = self._objectID2roomID[obj_id]
                # add the room
                if not self.sg.isExist(rm_id):
                    rm_node = Node(rm_id, node_type=NodeType.Room, data=rm_info)
                    self.sg.addNode(rm_node, self._scene_id)
                    if self.verbose:
                        print(f"[GraphGenerator] {rm_id} is added.")
                # add the object to SG
                if not self.sg.isExist(obj_id):
                    if len(objects_weighted_positions) > 0:
                        obj_info['est_position'] = objects_weighted_positions[idx][1]
                        obj_info['est_position_weight'] = objects_weighted_positions[idx][0]
                    if 'parent_obj' in obj_info.keys():
                        # this indicates it is derived (child) object
                        obj_node = Node(obj_id, node_type=NodeType.SmallObject, data=obj_info)
                    else:
                        # this indicates it is a larger object (like furniture)
                        obj_node = Node(obj_id, node_type=NodeType.Object, data=obj_info)
                    self.sg.addNode(obj_node, rm_id)    
                    if self.verbose:
                        print(f"[GraphGenerator] {obj_id} is added.")
                else:
                    if len(objects_weighted_positions) > 0:
                        obj_node = self.sg.getNode(obj_id)
                        if obj_node is not None:
                            obj_node.data['est_position'] = mean_position(obj_info['est_position'], objects_weighted_positions[idx][1], obj_info['est_position_weight'], objects_weighted_positions[idx][0])
                            obj_node.data['est_position_weight'] += objects_weighted_positions[idx][0]
                    
        if self.verbose:
            if self.sg.numNodes() ==  number_of_nodes:
                print("[GraphGenerator] no new node is added.")
            else:
                if self.verbose:
                    print(f"[GraphGenerator] {self.sg.numNodes()-number_of_nodes} nodes are added.")
                print(f"<<Scene Graph>>")
                self.sg.print_()

    
    def get_all_doors(self):
        all_doors_info = {}
        rm_count, rm_ids = self.sg.findNodes("room")
        for room_id in rm_ids:
            room_num = self.roomID2Num[room_id]
            all_doors_info[room_num] = self.get_doors(room_id)
        return all_doors_info
    
    def get_doors(self, room_id):
        assert self.sg.isExist(room_id)
        edge_list = self.sg._edge_list_by_node[room_id]
        doors_info = {}
        for edge in edge_list:
            to_room_num = self.roomID2Num[edge[1]]
            door_id = self.sg._edges[edge].data['id']
            door_position = self.est_door_coordinates[door_id]
            door_nav_axis = self.est_door_nav_axis[door_id]
            door = {'est_position': door_position, 'est_nav_axis':door_nav_axis} # Can be used to pass other data as well
            doors_info[door_id] = (door, to_room_num)
        return doors_info

    def get_room_center(self, room_num):
        room_id = self.roomNum2ID[room_num]
        assert self.sg.isExist(room_id)
        room_node = self.sg.getNode(room_id)
        if 'est_position' in room_node.data:
            return room_node.data['est_position']
        else:
            return None

    def get_doors_gt(self, room_id):
        assert self.sg.isExist(room_id)
        edge_list = self.sg._edge_list_by_node[room_id]
        doors_info = {}
        for edge in edge_list:
            if edge[0] == room_id:
                neighbor_rm_name = self._roomID2name[edge[1]]
            else:
                neighbor_rm_name = self._roomID2name[edge[0]]
            door_sim_id = self.sg._edges[edge].data['id']
            door_position = self.door_coordinates[door_sim_id]
            doors_info[neighbor_rm_name] = door_position
        return doors_info

    def update_object_data(self, obj_id, data_key, data_item):
        node = self.sg.getNode(obj_id)
        if node is None:
            print(f"{obj_id} does not exist in the scene graph.")
            return
        node.data[data_key]=data_item

    def _describe_door_from_sg(self, room_id, wordy_description=False):
        description = ""
        doors_info = self.get_doors_gt(room_id)
        #edge_list = self.sg._edge_list_by_node[room_id]
        for neighbor_rm_name, door_position in doors_info.items():
            if wordy_description:
                description += f"There is a door to {neighbor_rm_name} "
                description += "at coordinate ({:0.2f}, {:0.2f}).\n".format(door_position[0],door_position[1])
            else: # temporarily output this way.
                description += "Door to {neighbor_rm_name} "
                description += "({:0.2f}, {:0.2f}).\n".format(door_position[0],door_position[1])
        return description

    def _describe_roomNum_from_sg(self, room_number, wordy_description=False, include_door=False):
        assert room_number in self.roomNum2ID.keys()
        return self._describe_room_from_sg(self.roomNum2ID[room_number], wordy_description, include_door)

    def _describe_room_from_sg(self, room_id=None, wordy_description=False, include_door=False, no_small_object=True):
        description = ""
        if room_id is None:
            rm_count, rm_list = self.sg.findNodes("room")
            assert not rm_count < 0
            if rm_count == 0:
                description += "There is no room found yet.\n"
            else:
                if rm_count == 1:
                    description += "There is one room found:"
                else:
                    description += f"There are {rm_count} rooms found:"
                for n, rm in enumerate(rm_list):
                    if n == 0:
                        description += f" {self._roomID2name[rm]}"
                        if rm_count == 1:
                            description += ".\n"
                    elif n == rm_count-1:
                        if rm_count == 2:
                            description += f" and {self._roomID2name[rm]}.\n"
                        else:
                            description += f", and {self._roomID2name[rm]}.\n"
                    else:
                        description += f", {self._roomID2name[rm]}"
        else:
            assert self.sg.isExist(room_id)
            rm_node = self.sg.getNode(room_id)
            if wordy_description:
                description += f"In {self._roomID2name[room_id]}, "
            num_objects = len(rm_node.children)
            if num_objects==0:
                if wordy_description: # wordy version gives a full sentence.
                    description += "there is no object seen yet.\n"
            else:
                if num_objects==1:
                    child = list(rm_node.children.values())[0]
                    filter_small_object = no_small_object and (child.type == NodeType.SmallObject)
                    if not filter_small_object:
                        coord = (child.data['position']['x'], child.data['position']['z'])
                        if 'est_position' in child.data:
                            coord = (child.data['est_position']['x'], child.data['est_position']['z'])
                           
                        if wordy_description: # wordy version gives a full sentence.
                            description += f"there is a {self._objectID2name[child.id]} seen so far.\n"
                            description += "It is at coordinate ({:0.2f}, {:0.2f}).\n".format(coord[0],coord[1])
                        else: # succint version: <Name> (coordinate)
                            description += f"{self._objectID2name[child.id]} "
                            description += "({:0.2f}, {:0.2f})\n".format(coord[0],coord[1])
                else:
                    if wordy_description:
                        description += f"there are {num_objects} objects seen so far.\n"
                    num_instances = {}
                    for child_id, child_node in rm_node.children.items():
                        obj_name = self._objectID2name[child_id]
                        if obj_name in num_instances:
                            num_instances[obj_name] += 1
                        else:
                            num_instances[obj_name] = 1

                    for child_id, child_node in rm_node.children.items():
                        filter_small_object = no_small_object and (child_node.type == NodeType.SmallObject)
                        if filter_small_object:
                            continue
                        coord = (child_node.data['position']['x'], child_node.data['position']['z'])
                        if 'est_position' in child_node.data:
                            coord = (child_node.data['est_position']['x'], child_node.data['est_position']['z'])

                        if num_instances[self._objectID2name[child_id]] > 1:
                            tokens = child_id.split('|')    
                            description += f"{self._objectID2name[child_id]}" + tokens[-1] + " "
                        else:
                            description += f"{self._objectID2name[child_id]} "
                        if wordy_description:  # wordy version gives a full sentence.
                            description += "is at coordinate ({:0.2f}, {:0.2f}).\n".format(coord[0],coord[1])
                        else:
                            description += "({:0.2f}, {:0.2f})\n".format(coord[0],coord[1])
            if include_door:
                description += self._describe_door_from_sg(room_id)
        return description

    def get_house_description(self):
        return self._describe_room_from_sg()

    def get_current_room_description(self, last_event_metadata, wordy_description=False, include_door=False,  no_small_object=True):
        # identify which room the agent is located now
        (rm_id, rm_info) = getCurrentRoom(last_event_metadata, self._rooms_info, self._room_ranges)
        room_id = rm_info['id']
        if self.verbose:
            print(f"Agent is in {rm_info['roomName']} (id:{rm_id})")
        return  self._describe_room_from_sg(room_id, wordy_description, include_door,  no_small_object)

    def get_all_rooms_description(self, wordy_description=False, include_door=False, no_small_object=True):
        # identify which room the agent is located now
        description = ""
        rm_count, rm_list = self.sg.findNodes("room")
        for rm in rm_list:
            description += self._describe_room_from_sg(rm, wordy_description, include_door, no_small_object)
            description += "\n"
        return description
