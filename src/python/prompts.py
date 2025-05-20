"""
List of prompts to be used
"""

##############
search_plan = {}
search_plan['system'] = """Assume you are provided with a text-based description of a room in a house with objects and their 2D coordinates

Task: I am at a base location. Suggest me a step-wise high-level plan to achieve the goal below. Here are some rules: 
1. I can only perform the actions- (navigate, (x, y)), (look, objects), (open, object), (close, object).
2. I can only localize objects around other objects and not room e.g. apple should be looked for in fridge and not kitchen.
3. Provide the plan in a csv format. I have shown two examples below: 
a) Goal = 'Find a laptop', Base Location = 'LivingRoom' 
- navigate; (3.4, 2.6); Go to table
- look; (laptop); Look for laptop near the table
b) Goal = 'Find apple and bowl', Base Location = 'Kitchen'
- navigate; (3.4, 2.6); Go to DiningTable
- look; (apple, bowl); Look for apple near the DiningTable
- navigate;  (8.04, 6.13); Go to Fridge
- open; (Fridge); Open the Fridge
- look; (apple, bowl); Look for apple in the Fridge
- close; (Fridge); Close the Fridge
4. Start the line with - and no numbers
5. Provide the plan as human would by e.g. based on contextual relationships between objects


"""

search_plan['system-nav'] = """Assume you are provided with a text-based description of a room in a house with objects and their 2D coordinates

Task: I am at a base location. Suggest me a step-wise high-level plan to achieve the goal below. Here are some rules: 
1. I can only perform the actions- (navigate, (x, y)), (look, objects)
2. I can only localize objects around other objects and not room e.g. apple should be looked for on the table and not kitchen.
3. Provide the plan in a csv format. I have shown two examples below: 
a) Goal = 'Find a laptop', Base Location = 'LivingRoom' 
- navigate; (3.4, 2.6); Go to table
- look; (laptop); Look for laptop near the table
b) Goal = 'Find apple and bowl', Base Location = 'Kitchen'
- navigate; (3.4, 2.6); Go to DiningTable
- look; (apple, bowl); Look for apple and bowl near the DiningTable
- navigate;  (8.32, 0.63); Go to CounterTop
- look; (apple, bowl); Look for apple and bowl near the CounterTop
4. Start the line with - and no numbers
5. Provide the plan as human would by e.g. based on contextual relationships between objects


"""

search_plan['human']="""Room description
{room}

Goal: {goal}
Base location: {base_location}
"""
############################################################################################################

room_identification_template = {}
room_identification_template['system'] = """Identify the room based on the list of seen objects from the list below
- Kitchen 
- LivingRoom
- Bathroom
- Bedroom

Output should be the room name as a single word without the -
"""
room_identification_template['human'] = "{objects}"
room_identification_template['possible_outputs'] = ['Kitchen', 'LivingRoom', 'Bathroom', 'Bedroom']

###############

task_feasibility_template = {}

task_feasibility_template['system'] = """
Answer the question as 'Yes' or 'No'
"""

task_feasibility_template['human'] = """
Is it likely to find {object} in a {room}?
"""

####################################################################################################

rooms_tracking_template = {}

rooms_tracking_template['system'] = """
An agent is looking for certain objects in a house. It goes from one room to another room to search the objects.
Given a room name, output one of the following three options:
1) - Search this room;
2) - Come back later;  Reason: It is unlikely to find either of the given objects in this room
3) - Skip this room; Reason: It has already been searched

Follow these rules while selecting an option:
a. If it is likely to find any one of the given objects in the given room, output the first option (Search this room).
b. If it is not likely to find either of the objects in the given room and the room is being mentioned for the first time, output the second option (Come back later)
c. If it is not likely to find either of the objects in the given room but the room was marked to come back before, output the first option (Search this room)
d. If the agent has already searched the room before, output the third option (Skip this room)

Notes:
a. 'Bedroom 1' and 'Bedroom 2' are different rooms.
b. 'Bathroom 1' and 'Bathroom 2' are different rooms.

Here is an example:
Objects: Apple, Laptop
Room: LivingRoom
- Search this room
Objects: Apple, Laptop
Room: Bathroom
- Come back later;  Reason: It is unlikely to find either of the given objects in this room
Objects: Apple
Room: LivingRoom
- Skip this room; Reason: It has already been searched
Objects: Apple
Room: Bathroom
- Search this room
"""

rooms_tracking_template['human'] = """
Objects: {objects}
Room: {room}
"""