from collections import defaultdict
from statistics import mean

class House:

    def __init__(self, house):
        self.house = house
        self.group_room_info_by_rooms()
        self.group_objects_by_rooms()
        self.group_walls_by_rooms()
        self.group_doors_by_rooms()

    ### GROUPING THE DATA BY ROOM NUMBERS

    def group_room_info_by_rooms(self):
        self.grouped_room_info = {}
        room_types_count = {}
        for room in self.house['rooms']:
            room_num = int(room['id'].split('|')[1])
            self.grouped_room_info[room_num] = room
            roomType = self.grouped_room_info[room_num]['roomType']
            self.grouped_room_info[room_num]['roomName'] = roomType
            if roomType not in room_types_count:
                room_types_count[roomType] = 1
            else:            
                room_types_count[roomType] += 1

        #Update the room name if there are multiple rooms of same type
        room_types_track = {}
        for room_num in list(self.grouped_room_info.keys()):
            roomType = self.grouped_room_info[room_num]['roomType']
            if room_types_count[roomType] > 1:            
                if roomType not in room_types_track:
                    room_types_track[roomType] = 1
                else:
                    room_types_track[roomType] += 1
                self.grouped_room_info[room_num]['roomName'] = roomType + " " + str(room_types_track[roomType])
                
    def group_objects_by_rooms(self):
        self.grouped_objects = defaultdict(list)
        self.grouped_objects_including_children = defaultdict(list)
        for obj in self.house['objects']:
            room_num = int(obj['id'].split('|')[1])
            self.grouped_objects[room_num].append(obj)
            self.grouped_objects_including_children[room_num].append(obj)
            if 'children'in obj:
                for child_obj in obj['children']:
                    tokens = child_obj['id'].split('|')
                    if tokens[1] == 'surface':
                        room_num = int(child_obj['id'].split('|')[2])
                    else:
                        room_num = int(child_obj['id'].split('|')[1])
                    child_obj['parent_obj'] = obj['id'] 
                    self.grouped_objects_including_children[room_num].append(child_obj)

    def group_walls_by_rooms(self):
        self.grouped_walls = defaultdict(list)
        for wall in self.house['walls']:
            if wall['roomId'] == 'exterior':
                continue
            room_num = int(wall['roomId'].split('|')[1])
            self.grouped_walls[room_num].append(wall)

    def group_doors_by_rooms(self):
        self.grouped_doors = defaultdict(dict)
        for door in self.house['doors']:
            tokens = door['id'].split('|')
            room1_num = int(tokens[1])
            room2_num = int(tokens[2])
            if room1_num not in self.grouped_room_info or room2_num not in self.grouped_room_info:
                continue
            self.grouped_doors[room1_num][door['id']] = (door, room2_num)
            self.grouped_doors[room2_num][door['id']] = (door, room1_num)

    def get_all_doors(self):
        return self.house['doors']

    ### FUNCTIONS TO DESCRIBE THE HOUSE IN TEXT

    def describe_room(self, room_num):
        description = ""
        num_instances = {}
        for obj in self.grouped_objects[room_num]:
            tokens = obj['id'].split('|')
            obj_name = tokens[0]
            if obj_name in num_instances:
                num_instances[obj_name] += 1
            else:
                num_instances[obj_name] = 1
        
        for obj in self.grouped_objects[room_num]:
            tokens = obj['id'].split('|')
            obj_name = tokens[0]
            if 'painting' in obj_name.lower():
                continue
            obj_id = tokens[-1]
            if num_instances[obj_name] > 1:
                obj_name += obj_id
            description += "{} ({:0.2f}, {:0.2f})\n".format(obj_name, obj['position']['x'], obj['position']['z'])
        
        return description

    def describe_doors(self):
        description = ""
        for idx, door in enumerate(self.house['doors']):
            tokens = door['id'].split('|')
            room1_num = int(tokens[1])
            room2_num = int(tokens[2])
            if room1_num not in self.grouped_room_info or room2_num not in self.grouped_room_info:
                continue
            room_name1 = self.grouped_room_info[room1_num]['roomName']
            room_name2 = self.grouped_room_info[room2_num]['roomName']
            tokens = door['wall0'].split('|')
            x1 = float(tokens[2])
            x2 = float(tokens[4])
            x = (x1+x2)/2
            z1 = float(tokens[3])
            z2 = float(tokens[5])
            z = (z1+z2)/2
            description += "Door{} is at coordinate ({:0.2f}, {:0.2f}) which connects {} and {}\n".format(idx, x, z, room_name1, room_name2)

        return description
            
    def describe_house(self):    
        
        description = "This is a top-down view of my house, described with a 2D coordinate system.\n"
        description += "My house has " + str(len(self.grouped_objects)) + " rooms.\n"
        description += "These are the rooms' descriptions.\n\n"
        
        for room_num in list(self.grouped_objects.keys()):
            room_type = self.grouped_room_info[room_num]['roomName']
            description += room_type + ':\n'
            description += self.describe_room(self.grouped_objects[room_num]) + '\n'

        description += "\nThese are the descriptions of different doors in the house.\n"
        description += self.describe_doors()
        
        return description

    ### EXTRACTING INFORMATION

    def get_room_num(self, coordinate_inside_room):
        for (room_num, walls) in self.grouped_walls.items():
            all_x = []
            all_z = []        
            for wall in walls:
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
            if coordinate_inside_room[0] > min_x and coordinate_inside_room[0] < max_x and coordinate_inside_room[1] > min_z and coordinate_inside_room[1] < max_z:
                #print("Room num = ", room_num)
                return room_num
        
        return -1
    
    def get_room_center(self, room_num):
        all_x = []
        all_z = []        
        for wall in self.grouped_walls[room_num]:
            tokens = wall['id'].split('|')
            x1 = float(tokens[2])
            x2 = float(tokens[4])
            z1 = float(tokens[3])
            z2 = float(tokens[5])
            all_x.extend([x1, x2])
            all_z.extend([z1, z2])
        x_mean = mean(all_x)
        z_mean = mean(all_z)
        return (x_mean, z_mean)


    
