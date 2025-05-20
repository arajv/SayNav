# Project Code
from agent import Agent
from house import House
from scenegraph_generator import SceneGraphGenerator
import llm
from llm import LLMPlanner
import hl_utils

# Simulator
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import prior

# Python packages
import pickle
import random
from datetime import datetime
from collections import defaultdict
import os
import yaml
import copy
import numpy as np


def pipeline(house_instance, controller, objects, params):

    # Define the task / goal
    goal_type = 0 #(0: Find objects, 1: Pick and Place objects - Not implemented)
    objects_status = {obj.lower():0 for obj in objects} # 0: Not found, 1: Found
    targets_detection_order = {}
    curr_detection_num = 0

    # Initialize the environment using house instance
    env = House(house_instance)

    # create a Scene Graph Generator instance
    house_id = "House" # assign an ID to house (as the Scene Graph root)
    sgg = SceneGraphGenerator( house_id, env, params['SceneGraph'], verbose=False)
    
    # CHOOSE Source to build SG 
    sg_input_source = params['General']['scene_graph_source']
    # option 0: ground truth room info
    #           This is equivalent to calling env.describe_room(room_num)
    # option 1: SG. using ground truth, but based on metadata visiblity
    #           I don't know what this 'visibility' truely means. 
    # option 2: SG. using ground truth, but based on instance segmentation info 
    #           Slow.

    # Compute reachable position nearest to goal for navigation
    use_reachable_position = True
    if sg_input_source == 2:
        use_reachable_position = False

    # Create a controller and an agent
    robot = Agent(controller, params['Agent'])

    # Initialize an empty list for storing house map
    env_map = {}

    # Initialize an empty dictionary for tracking visited doors and rooms
    visited_doors = defaultdict(set)
    recently_visited_doors = defaultdict(set)
    
    # Initialize empty sets for tracking rooms
    visited_rooms = set()
    come_back_rooms = set()
    num_times_room_entered = defaultdict(int)
    max_entry_limit = 6

    # Initialize an empty dictionary for storing visualizations
    visualizations = defaultdict(list)

    # Set the robot in the starting pose
    start_position = None
    start_heading = None
    start_room_num = None
    if 'start_position' in params['General']:
        start_position = params['General']['start_position']
        start_heading = params['General']['start_heading']
    if 'start_room_num' in params['General']:
        start_room_num = params['General']['start_room_num']
    if start_position is not None:
        robot.set_start_pose(start_position, start_heading)
    elif start_room_num is not None:    
        robot.navigateToRoom(start_room_num, env.grouped_walls)    

    # Initialize the High Level Planner
    llm_planner = LLMPlanner(enable_room_tracking = params['General']['enable_llm_room_tracking'], model_name=params['General']['llm_model_name'], openai_key=params['General']['openai_key'] )
    llm_planner.setup_chat_templates()

    # Initialize an empty dictionary for storing plans
    plan_data = {}

    # Misc Variables
    counter = 0
    task_is_successful = False
    prev_room_num = -1
    last_door_id = ''
    all_doors = None
    history = []

    # Starting and specifying goal
    visuals = defaultdict(list)
    visuals['rgb'].append(copy.deepcopy(controller.last_event.frame))
    visuals['top'].append(copy.deepcopy(controller.last_event.third_party_camera_frames[0][:,:,:-1]))
    visuals['text'].append(f"Goal: Find {', '.join(objects)}")
    hl_utils.update_visualizations(visualizations, visuals)
    
    while task_is_successful is False:
        # Get a list of unfound objects
        unfound_objects = hl_utils.get_unfound_objects(objects, objects_status)
        goal = llm.construct_goal(goal_type, unfound_objects)

        # Get the current room number
        room_num = env.get_room_num(robot.get_current_position())

        if prev_room_num != -1:
            success = hl_utils.update_visited_doors(last_door_id, room_num, prev_room_num, all_doors, visited_doors, recently_visited_doors)
            if not success:
                visuals = defaultdict(list)
                text = "Unable to go through door,\n so going to the room center"
                visuals['text'] = [text]
                room_center = None
                if sg_input_source == 2:
                    room_center_dict = sgg.get_room_center(room_num)
                    if room_center_dict is not None:
                        room_center = (room_center_dict['x'], room_center_dict['z'])
                else:
                    room_center = env.get_room_center(room_num)
                event = robot.navigateTo(room_center, target_room_walls = env.grouped_walls[room_num], visuals = visuals, use_reachable_position = use_reachable_position)
                if len(visuals['rgb']) > 1: # Look can involve multiple controller's step function calls 
                    for n in range(len(visuals['rgb']) - 1):
                        visuals['text'].append(text)
                hl_utils.update_visualizations(visualizations, visuals)
                history.extend(visuals['text'])

        # Since map is empty, then we select LookAround action
        visuals = defaultdict(list)
        visuals['rgb'].append(copy.deepcopy(controller.last_event.frame))
        visuals['top'].append(copy.deepcopy(controller.last_event.third_party_camera_frames[0][:,:,:-1]))
        visuals['text'].append("Entered a new room")
        visuals['instance'].append(np.array([]))        
        
        #visuals['text'].append("Looking around")   
        visible_objects, events, found = robot.lookAround(visuals = visuals, method = sg_input_source, minimum_pixel_size=params['SceneGraph']['min_pixels_per_object'], target_objects = unfound_objects, display_string = "Looking around")
        #if len(visuals['rgb']) > 1: # Look can involve multiple controller's step function calls 
        #    for n in range(len(visuals['rgb']) - len(visuals['text'])):
        #        visuals['text'].append("Looking around")
        
        if any(found):
            curr_detection_num += 1
            for i, f in enumerate(found):
                if f:
                    objects_status[unfound_objects[i].lower()] = 1
                    targets_detection_order[unfound_objects[i]] = curr_detection_num
            if all(list(objects_status.values())):
                print("Found all objects")
                task_is_successful = True
                hl_utils.update_visualizations(visualizations, visuals)
                history.extend(visuals['text'])
                break
            else:
                unfound_objects = hl_utils.get_unfound_objects(objects, objects_status)
                goal = llm.construct_goal(goal_type, unfound_objects)

        if sg_input_source == 0:
            visible_objects = {obj['id'].split('|')[0] for obj in env.grouped_objects[room_num]}
            visible_objects_ids = [obj['id'] for obj in env.grouped_objects[room_num]]
            sgg.updateGraph(robot.get_current_position(), visible_objects_ids)
        else:  
            sgg.updateGraphFromObservations(events, method = sg_input_source, minimum_pixel_size=params['SceneGraph']['min_pixels_per_object'], visuals = visuals)
        
        hl_utils.update_visualizations(visualizations, visuals)
        history.extend(visuals['text'])
        
        # Determine the room type
        room_type = llm_planner.get_room_type(visible_objects)        
        room_type_gt = env.grouped_room_info[room_num]['roomType']
        print("Visible Objects = ", visible_objects)
        print("Ground Truth Room Type = ", room_type_gt)
        print("Detected Room Type = ", room_type)       

        room_name_gt = env.grouped_room_info[room_num]['roomName']
        room_name = room_type + " " + str(room_num)

        if sg_input_source == 0 or sg_input_source == 1:
            room_type = room_type_gt
            room_name = room_name_gt

        num_times_room_entered[room_num]+= 1
        #print(f"Currently in detected room {room_type}\nGround truth room {grouped_room_info[room_num]['roomName']}")
        print(f"Currently in room: {room_name}")

        # Check if a plan needs to be generated for the room        
        plan_for_room, output = check_if_plan_needed(room_num, room_name, room_type, visited_rooms, come_back_rooms, unfound_objects, llm_planner, use_llm_room_tracking = params['General']['enable_llm_room_tracking'])
        if output == 1 or output == 2:
            visuals = defaultdict(list)
            if output == 1:
                print("Marking the room as: come_back_later")
                visuals['text'] = ["It is unlikely to find " + ", ".join(unfound_objects) + " \nin " + room_type + ",\nSo will come back later"]
            elif output == 2:
                print("Skipping the room as it has already been searched")
                visuals['text'] = ["Skipping this room \nas it has already been searched"]
            robot.add_rgb_from_latest_event(visuals = visuals)
            hl_utils.update_visualizations(visualizations, visuals)
            history.extend(visuals['text'])
        
        if plan_for_room:

            # Updating the map
            if sg_input_source == 0:
                env_map[f'{room_type_gt}_{room_num}'] = env.describe_room(room_num)
            else:
                env_map[f'{room_type_gt}_{room_num}'] = sgg._describe_roomNum_from_sg(room_num)
            current_room_description = env_map[f'{room_type_gt}_{room_num}']
            print(f"*** ROOM Description ***")
            print(current_room_description)

            # Ping the llm to find the next step to get to goal            
            plan = llm_planner.get_plan(current_room_description, room_type, goal)
            #plan = hl_utils.update_llm_plan(plan, robot.get_current_position())
            plan_data[counter] = plan
            counter += 1

            for p in plan:
                action = p['action']
                action_arg = p['arg']
                string_curr_action = f"Executing\naction={action}\narg={action_arg}\n{p['comment']}\nCurrently in room: {room_name}"
                print(string_curr_action)
                
                # Execute the low-level actions
                visuals = defaultdict(list)
                visuals['text'] = [string_curr_action]
                
                placed_object = False
                
                if action == "navigate":
                    event = robot.navigateTo(action_arg, target_room_walls = env.grouped_walls[room_num], visuals = visuals, use_reachable_position = use_reachable_position)
                    if len(visuals['rgb']) > 1: # Look can involve multiple controller's step function calls 
                        for n in range(len(visuals['rgb']) - 1):
                            visuals['text'].extend([string_curr_action])
                
                elif action == "look":
                    #objects_to_look = action_arg.split(', ')
                    visuals['text'].clear()
                    _, events, found = robot.lookAround(visuals = visuals, method = sg_input_source, minimum_pixel_size=params['SceneGraph']['min_pixels_per_object'], target_objects = unfound_objects, display_string = string_curr_action)
                    #found, rotations = robot.lookFor(objects_to_look, allow_inplace_rotation=True, visuals = visuals)
                    
                    #if len(visuals['rgb']) > 1: # Look can involve multiple controller's step function calls 
                    #    for n in range(len(visuals['rgb']) - len(visuals['text'])):
                    #        visuals['text'].extend([string_curr_action])

                    if goal_type == 0: 
                        if any(found):
                            curr_detection_num += 1
                            for i, f in enumerate(found):
                                if f:
                                    objects_status[unfound_objects[i].lower()] = 1
                                    targets_detection_order[unfound_objects[i]] = curr_detection_num
                            if all(list(objects_status.values())):
                                print("Found all objects")
                                task_is_successful = True
                                hl_utils.update_visualizations(visualizations, visuals)
                                break
                            unfound_objects = hl_utils.get_unfound_objects(objects, objects_status)

                    if sg_input_source != 0:
                        sgg.updateGraphFromObservations(events, method = sg_input_source, minimum_pixel_size=1000, visuals = visuals)

                elif action == "open":
                    event = robot.open(action_arg, visuals = visuals)
                elif action == "close":
                    event = robot.close(action_arg, visuals = visuals)            
                elif action == "pick":
                    event = robot.pick(action_arg, visuals = visuals)
                elif action == "place":
                    event = robot.place(action_arg, visuals = visuals)
                    placed_object = event.metadata["lastActionSuccess"]
                
                # Update the visualizations with visuals from current action
                hl_utils.update_visualizations(visualizations, visuals)
                history.extend(visuals['text'])                               
                        
                if goal_type == 1 and placed_object:
                    print("Object placed successfully")
                    task_is_successful = True
                    break 
            if not task_is_successful:        
                print("Could not find all objects, so going into exploration mode")
        
        if task_is_successful:
            break
        
        max_limit_reached = False
        for roomNum, num_times in num_times_room_entered.items():
            if num_times > max_entry_limit:
                max_limit_reached = True
                break

        if max_limit_reached:
            print("Task Failed")
            break
        
        door_id = None
        
        if sg_input_source == 0 or sg_input_source == 1:
            all_doors = env.grouped_doors                
        else:
            all_doors = sgg.get_all_doors()
            
        door_id = hl_utils.choose_random_door(room_num, all_doors, visited_doors, recently_visited_doors, come_back_rooms)
        if door_id == None:
            print("Task Failed")
            break
        
        random_door = all_doors[room_num][door_id]

        visuals = defaultdict(list)
        visuals['text'] = ["Could not complete goal, \nso going into exploration mode: \nGo to a Door"]
        event = robot.navigateToNeighboringRoom(random_door[0], random_door[1], room_num, env.grouped_walls, visuals = visuals, sg_input_source = sg_input_source, minimum_pixel_size=params['SceneGraph']['min_pixels_per_object'])
        
        hl_utils.update_visualizations(visualizations, visuals)
        history.extend(visuals['text'])
        last_door_id = door_id
        prev_room_num = room_num
    
    output_folder = ''
    if  'output_folder' not in params['General'] or params['General']['output_folder'] == '':
        # Get the current timestamp
        current_timestamp = datetime.now()

        # Convert the timestamp to a string
        timestamp_string = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")

        output_folder = os.path.join("visual", timestamp_string)
    else:
        output_folder = params['General']['output_folder']
    
    os.makedirs(output_folder, exist_ok=True)
    
    #meta_data = f"Goal: {goal}\nStarting room: {starting_room_num}\nTask successful: {task_is_successful}"
    #with open(os.path.join(output_folder, "meta_data.txt"), "w") as f:
    #    f.write(meta_data)

    results = {}
    results['goals'] = objects_status
    results['path_length'] = robot.get_num_steps() * params['Agent']['grid_size']
    results['targets_detection_order'] = targets_detection_order
    if task_is_successful:
        if results['path_length'] == 0 or 'shortest_path_length' not in params['General']:
            results['spl'] = 1.0
        else:
            shortest_path_length = params['General']['shortest_path_length']
            results['spl'] = shortest_path_length / max(shortest_path_length, results['path_length'])
    else:    
        results['spl'] = 0

    with open(os.path.join(output_folder, "results.yaml"), 'w') as output_file:
        yaml.dump(results, output_file)

    hl_utils.save_visualizations(visualizations, output_folder)
    pickle.dump(plan_data, open(os.path.join(output_folder, "plan_data.pkl"), "wb"))
    pickle.dump(history, open(os.path.join(output_folder, "history_text.pkl"), "wb"))


def check_if_plan_needed(room_num, room_name, room_type, visited_rooms, come_back_rooms, unfound_objects, llm_planner, use_llm_room_tracking = False):
    plan_for_room = False
    output = -1
    if use_llm_room_tracking:
        print("Using LLM for tracking")
        output = llm_planner.get_decision_on_room(room_name, unfound_objects)
        if output == 0:
            plan_for_room = True
            visited_rooms.add(room_num)
            if room_num in come_back_rooms:
                come_back_rooms.remove(room_num)
        elif output == 1:
            come_back_rooms.add(room_num)
        elif output == 2:
            pass
        
    else:
        if room_num not in visited_rooms:
            if room_num in come_back_rooms:
                plan_for_room = True
            else:
                plan_for_room = llm_planner.get_feasibility_of_search(unfound_objects, room_type)
            
            if not plan_for_room:
                come_back_rooms.add(room_num)
                output = 1
            else:
                output = 0
                visited_rooms.add(room_num)
                if room_num in come_back_rooms:
                    come_back_rooms.remove(room_num)
        else:
            output = 2

    return plan_for_room, output

def baseline(house_instance, controller, objects, gt_locations, params):

    objects_status = {obj.lower():0 for obj in objects} # 0: Not found, 1: Found

     # Initialize the environment using house instance
    env = House(house_instance)

    # Create a controller and an agent
    robot = Agent(controller, params['Agent'])

     # Initialize an empty dictionary for storing visualizations
    visualizations = defaultdict(list)

    # Set the robot in the starting pose
    start_position = None
    start_heading = None
    start_room_num = None
    if 'start_position' in params['General']:
        start_position = params['General']['start_position']
        start_heading = params['General']['start_heading']
    if 'start_room_num' in params['General']:
        start_room_num = params['General']['start_room_num']
    if start_position is not None:
        robot.set_start_pose(start_position, start_heading)
    elif start_room_num is not None:    
        robot.navigateToRoom(start_room_num, env.grouped_walls)

    for i in range(len(objects)):
        obj = objects[i]
        gt_loc = gt_locations[i]
        target_room_num = env.get_room_num(gt_loc)

        visuals = defaultdict(list)
        string_curr_action = f"Executing\naction=Navigate to \narg={obj} ({gt_loc})\n"
        
        event = robot.navigateTo(gt_loc, target_room_walls = env.grouped_walls[target_room_num], visuals = visuals)        
        if len(visuals['rgb']) > 1:
            for n in range(len(visuals['rgb'])):
                visuals['text'].append(string_curr_action)

        hl_utils.update_visualizations(visualizations, visuals)

        visuals = defaultdict(list)
        string_curr_action = f"Executing\naction=Look for \narg={obj} ({gt_loc})\n"
        found, rotations = robot.lookFor(objects, allow_inplace_rotation=True, visuals = visuals)
        if len(visuals['rgb']) > 1:
            for n in range(len(visuals['rgb'])):
                visuals['text'].append(string_curr_action)
        
        hl_utils.update_visualizations(visualizations, visuals)
        
        if any(found):
            for n, f in enumerate(found):
                if f:
                    objects_status[objects[n].lower()] = 1

    output_folder = ''
    if  'output_folder' not in params['General'] or params['General']['output_folder'] == '':
        # Get the current timestamp
        current_timestamp = datetime.now()

        # Convert the timestamp to a string
        timestamp_string = current_timestamp.strftime("%Y-%m-%d_%H-%M-%S")

        output_folder = os.path.join("visual", timestamp_string)
    else:
        output_folder = params['General']['output_folder']

    hl_utils.save_visualizations(visualizations, output_folder)
    
    results = {}
    results['goals'] = objects_status
    results['path_length'] = robot.get_num_steps() * params['Agent']['grid_size']
    if all(list(objects_status.values())):
        if results['path_length'] == 0 or 'shortest_path_length' not in params['General']:
            results['spl'] = 1.0
        else:
            shortest_path_length = params['General']['shortest_path_length']
            results['spl'] = shortest_path_length / max(shortest_path_length, results['path_length'])
    else:    
        results['spl'] = 0

    with open(os.path.join(output_folder, "results.yaml"), 'w') as output_file:
        yaml.dump(results, output_file)
            