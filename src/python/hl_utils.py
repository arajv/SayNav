from PIL import Image, ImageDraw, ImageFont
import random
import copy
import os
import numpy as np
import cv2
from collections import defaultdict
import math

def choose_random_door(room_num, grouped_doors, visited_doors, recently_visited_doors, come_back_rooms):
    available_doors_ids = set(grouped_doors[room_num].keys())
    if room_num in visited_doors:
        available_doors_ids = available_doors_ids.difference(visited_doors[room_num])    
    
    if len(available_doors_ids) == 0:

        # Return None if all doors in the house have been visited and there are no rooms to come back later, 
        # else allow to visit the visited doors
        if len(come_back_rooms) > 0 or len(grouped_doors.keys()) != len(visited_doors.keys()):
            available_doors_ids = set(grouped_doors[room_num].keys())
            available_doors_ids = available_doors_ids.difference(recently_visited_doors[room_num])
        else:
            all_doors_visited = True
            for num in list(grouped_doors.keys()):
                if len(grouped_doors[num]) != len(visited_doors[num]):
                    available_doors_ids = set(grouped_doors[room_num].keys())
                    all_doors_visited = False
                    break
            if all_doors_visited:
                return None
    
    print("Available doors = ", available_doors_ids)
    door_id = random.choice(list(available_doors_ids))
    
    '''
    visited_doors[room_num].add(door_id)
    if len(grouped_doors[room_num]) > 1:
        recently_visited_doors[room_num].add(door_id)
    if len(grouped_doors[room_num]) == len(recently_visited_doors[room_num]):
        recently_visited_doors[room_num].clear()
        recently_visited_doors[room_num].add(door_id)

    # Mark the door visited in the destination room too
    dest_room_num = grouped_doors[room_num][door_id][1]
    visited_doors[dest_room_num].add(door_id)
    
    if len(grouped_doors[dest_room_num]) > 1:
        recently_visited_doors[dest_room_num].add(door_id)
    if len(grouped_doors[room_num]) == len(recently_visited_doors[room_num]):
        recently_visited_doors[dest_room_num].clear()
        recently_visited_doors[dest_room_num].add(door_id)
    #else:
    #    visited_doors[dest_room_num].add(door_id)
    #    recently_visited_doors[dest_room_num].add(door_id)
    '''
    return door_id

def update_visited_doors(last_door_id, curr_room_num, prev_room_num, grouped_doors, visited_doors, recently_visited_doors):
    if prev_room_num == -1:
        return None
    if prev_room_num == curr_room_num:
        print("Agent failed to go to the neighboring room")
        return False
    
    visited_doors[prev_room_num].add(last_door_id)
    if len(grouped_doors[prev_room_num]) > 1:
        recently_visited_doors[prev_room_num].add(last_door_id)
    if len(grouped_doors[prev_room_num]) == len(recently_visited_doors[prev_room_num]):
        recently_visited_doors[prev_room_num].clear()
        recently_visited_doors[prev_room_num].add(last_door_id)

    # Mark the door visited in the current room too
    visited_doors[curr_room_num].add(last_door_id)
    
    if len(grouped_doors[curr_room_num]) > 1:
        recently_visited_doors[curr_room_num].add(last_door_id)
    if len(grouped_doors[curr_room_num]) == len(recently_visited_doors[curr_room_num]):
        recently_visited_doors[curr_room_num].clear()
        recently_visited_doors[curr_room_num].add(last_door_id)
    
    return True

def is_point_inside_room(point, room_walls):
    all_x = []
    all_z = []        
    for wall in room_walls:
        tokens = wall['id'].split('|')
        x1 = float(tokens[2])
        x2 = float(tokens[4])
        z1 = float(tokens[3])
        z2 = float(tokens[5])
        all_x.extend([x1, x2])
        all_z.extend([z1, z2])
    min_x = min(all_x)
    max_x = max(all_x)
    min_z = min(all_z)
    max_z = max(all_z)
    if point[0] > min_x and point[0] < max_x and point[1] > min_z and point[1] < max_z:
        #print("Room num = ", room_num)
        return True
    return False 

def get_unfound_objects(objects, objects_status):    
    return [obj for obj in objects if objects_status[obj.lower()] == 0]

def get_objects_visuals_from_event(objects, event, visuals, minimum_pixel_size = 0, display_string = ''):
    visuals['rgb'].extend([copy.deepcopy(event.frame)])
    visuals['top'].append(copy.deepcopy(event.third_party_camera_frames[0][:,:,:-1]))
    instance_seg_frame = np.full(event.frame.shape[:-1], False)
    found = False
    objects_found = []
    for k, v in event.instance_masks.items():
        if minimum_pixel_size > 0:
            if np.sum(v) < minimum_pixel_size:
                continue 
        obj = k.split('|')[0].lower()
        for object in objects:
            if obj == object.lower():
                instance_seg_frame = np.logical_or(instance_seg_frame, v)
                found = True
                objects_found.append(object)
                break
    if found:
        visuals['instance'].append(instance_seg_frame)
        visuals['text'].append("Found: " + ', '.join(objects_found))
    else:
        visuals['instance'].append(np.array([]))
        visuals['text'].append(display_string)
    
def update_visualizations(visualizations, curr_visuals):
    lengths = [0]
    for k in ['rgb', 'top', 'instance', 'text']:
        if k in curr_visuals:
            lengths.append(len(curr_visuals[k]))
    
    max_length = max(lengths)

    for k in ['rgb', 'top', 'instance', 'text']:
        if k in curr_visuals:
            visualizations[k].extend(curr_visuals[k])
        else:
            for i in range(max_length):
                visualizations[k].append(np.array([]))

    if 'doors' in curr_visuals:
        visualizations['doors'].extend(curr_visuals['doors'])

def save_visualizations(visualizations, folder):
    if len(visualizations['rgb']) == 0:
        return
    os.makedirs(folder, exist_ok=True)    
    height = visualizations['rgb'][0].shape[0]
    width = visualizations['rgb'][0].shape[1]
    frame_size = (3*width, height)    
    video = cv2.VideoWriter(folder + '/plan.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 2, frame_size)
    for n in range(len(visualizations['rgb'])):    
        img = visualizations['rgb'][n]
        img = Image.fromarray(img)
        img = add_info_to_image(img, Image.fromarray(visualizations['top'][n]), visualizations['text'][n])
        video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        
        if len(visualizations['instance'][n]) != 0:
            img2 = visualizations['instance'][n]
            img2 = Image.fromarray(img2)
            img2 = add_info_to_image(img2, Image.fromarray(visualizations['top'][n]), visualizations['text'][n])
            video.write(cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2BGR))

    video.release()

    if 'doors' in visualizations:
        door_instance_count = defaultdict(int)
        for door in visualizations['doors']:
            door_instance_count[door[1]] += 1
            frame = Image.fromarray(door[0])
            draw = ImageDraw.Draw(frame)
            pos = door[2]
            radius = 2
            ellipse = (pos[0] - radius, pos[1] - radius, pos[0] + radius, pos[1] + radius)
            draw.ellipse(ellipse, fill = 'red', outline ='red')
            frame.save(folder + '/' + door[1].replace('|', '-') + '_' + str(door_instance_count[door[1]]) + '.png')

def add_info_to_image(image, top_image, info_text):
    # Load the image
    image_width, image_height = image.size

    # Create a blank image with a black box on the right side
    info_box_width = image_width  # Width of the black box
    total_width = 2 * image_width + info_box_width
    result_image = Image.new("RGB", (total_width, image_height), color="white")

    # Paste the original image onto the blank image
    result_image.paste(image, (0, 0))
    result_image.paste(top_image, (image_width, 0))

    # Draw a black box on the right side
    draw = ImageDraw.Draw(result_image)
    draw.rectangle([(2 * image_width, 0), (total_width, image_height)], fill="black")

    # Write the information text inside the black box
    font = ImageFont.truetype("arial.ttf", 16)  # Specify the font and size
    text_color = "white"
    text_position = (2 * image_width + 10, 10)  # Position of the text
    draw.text(text_position, info_text, fill=text_color, font=font)

    return result_image

def min_dist_point(points_dict, curr_pos):
    min_dist = -1
    min_dist_idx = None
    for idx, pt in points_dict.items():
        dist = math.sqrt((pt[0] - curr_pos[0])**2 + (pt[1] - curr_pos[1])**2)
        if min_dist == -1 or dist < min_dist:
            min_dist = dist
            min_dist_idx = idx

    return min_dist_idx

def update_llm_plan(plan, curr_pos):
    plan_target_points = {}
    navigate_action_indices = {}

    counter = 0
    for idx, p in enumerate(plan):
        action = p['action']
        action_arg = p['arg']
        if action != "navigate":
            continue
        navigate_action_indices[counter] = idx
        plan_target_points[counter] = action_arg
        counter += 1

    updated_plan = []

    pos = curr_pos
    while len(plan_target_points) != 0:
        pt_counter = min_dist_point(plan_target_points, pos)
        start_idx = navigate_action_indices[pt_counter]
        if pt_counter + 1 in navigate_action_indices:
            end_idx = navigate_action_indices[pt_counter + 1]
        else:
            end_idx = len(plan)    
        updated_plan.extend(plan[start_idx:end_idx])
        pos = plan_target_points[pt_counter]
        plan_target_points.pop(pt_counter)

    assert len(plan) == len(updated_plan)

    return updated_plan