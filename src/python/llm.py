import openai
import copy
import prompts
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os

# OpenAI
from langchain.chat_models import ChatOpenAI

class LLMPlanner:

    def __init__(self, enable_room_tracking = False, model_name = 'gpt-3.5-turbo', openai_key = ''):
        print("LLM Model Name = ", model_name)
        self.chatllm = ChatOpenAI(temperature=0.25, max_tokens=512, model_name=model_name, openai_api_key=openai_key)
        self.enable_room_tracking = enable_room_tracking
        if enable_room_tracking:
            self.chatllm_memory = ChatOpenAI(temperature=0.25, max_tokens=512, model_name=model_name, openai_api_key=openai_key)

    def setup_chat_templates(self):
        self.setup_room_identification_template()
        self.setup_search_plan_template()
        self.setup_task_feasibility_template()
        if self.enable_room_tracking:
            self.setup_room_tracking_template()

    def setup_room_identification_template(self):
        human_template = prompts.room_identification_template['human']
        system_template = prompts.room_identification_template['system']
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        self.room_identification_chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    def setup_search_plan_template(self):
        human_template = prompts.search_plan['human']
        system_template = prompts.search_plan['system-nav']
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        self.search_plan_chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    def setup_task_feasibility_template(self):
        human_template = prompts.task_feasibility_template['human']
        system_template = prompts.task_feasibility_template['system']
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        self.task_feasibility_chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    def setup_room_tracking_template(self):
        system_template = prompts.rooms_tracking_template['system']
        human_message_prompt =  HumanMessagePromptTemplate.from_template("{input}")
        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

        room_tracking_chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, MessagesPlaceholder(variable_name="history"), human_message_prompt])
        memory = ConversationBufferMemory(return_messages=True)
        self.room_tracking_conversation = ConversationChain(memory=memory, prompt=room_tracking_chat_prompt, llm=self.chatllm_memory)

    def get_room_type(self, visible_objects):
        if len(visible_objects) == 0:
            return "LivingRoom"
        
        visible_objects = list(visible_objects)
        visible_objects = "\n".join([it.lower() for it in visible_objects])

        # get a chat completion from the formatted messages
        prompt = self.room_identification_chat_prompt.format_prompt(objects=visible_objects).to_messages()

        # Make sure the output is valid
        num_tries = 2
        ii = 0
        flag = False
        while (ii < num_tries):
            output = self.chatllm.predict_messages(prompt).content
            ##TODO: Use parsing from langchain
            if output in prompts.room_identification_template['possible_outputs']:
                flag = True
                break
            ii += 1
        
        if flag == False:
            print("Could not identify room type so defaulting to kitchen")
            output = "LivingRoom"
        
        return output


    def get_plan(self, room_description, room_type, goal):

        # get a chat completion from the formatted messages
        prompt = self.search_plan_chat_prompt.format_prompt(room=room_description, goal=goal, base_location=room_type).to_messages()

        # Make sure the output is valid
        num_tries = 2
        ii = 0
        flag = False
        while (ii < num_tries):
            output = self.chatllm.predict_messages(prompt).content
            ##TODO: Use parsing from langchain
            try:
                plan = self.parse_plan(output)
                flag = True
                break
            except:
                ii += 1
        
        return plan


    def get_decision_on_room(self, room_name, objects):
        objects_str = ', '.join(objects)
        input_prompt = prompts.rooms_tracking_template['human'].format(objects=objects_str, room=room_name)
        output = self.room_tracking_conversation.predict(input=input_prompt)
        if "Search this room" in output:
            return 0
        elif "Come back later" in output:
            return 1
        elif "Skip this room" in output:
            return 2
        else:
            print("Unknown output from Room tracking LLM: ", output)
            return -1


    def parse_plan(self, plan):
        actions = []
        current_action = {}
        
        for line in plan.split('\n'):
            line = line.strip()

            #if "if" in line or "If" in line:
            #    continue
            
            if line.startswith('-'):
                if current_action:
                    actions.append(current_action)
                    
                current_action = {'action': '', 'arg': '', 'comment': ''}
                line = line[1:].strip()
                
                if line.startswith('If'):
                    continue
                
            if line.startswith('navigate'):
                parts = line.split(';')
                current_action['action'] = 'navigate'
                current_action['arg'] = eval(parts[1].strip())
                current_action['comment'] = parts[2].strip()
                
            elif line.startswith('look'):
                parts = line.split(';')
                current_action['action'] = 'look'
                current_action['arg'] = parts[1].strip().replace('(', '').replace(')', '')
                current_action['comment'] = parts[2].strip()

            elif line.startswith('pick'):
                parts = line.split(';')
                current_action['action'] = 'pick'
                current_action['arg'] = parts[1].strip().replace('(', '').replace(')', '')
                current_action['comment'] = parts[2].strip()

            elif line.startswith('place'):
                parts = line.split(';')
                current_action['action'] = 'place'
                current_action['arg'] = parts[1].strip().replace('(', '').replace(')', '')
                current_action['comment'] = parts[2].strip()

            elif line.startswith('open'):
                parts = line.split(';')
                current_action['action'] = 'open'
                current_action['arg'] = parts[1].strip().replace('(', '').replace(')', '')
                current_action['comment'] = parts[2].strip()

            elif line.startswith('close'):
                parts = line.split(';')
                current_action['action'] = 'close'
                current_action['arg'] = parts[1].strip().replace('(', '').replace(')', '')
                current_action['comment'] = parts[2].strip()
                
            elif line.startswith('-'):
                current_action['comment'] = line[1:].strip()
        
        if current_action:
            actions.append(current_action)
        
        return actions


    def get_feasibility_of_search(self, objects, roomType):
        for obj in objects:
            # get a chat completion from the formatted messages
            prompt = self.task_feasibility_chat_prompt.format_prompt(object=obj, room=roomType).to_messages()

            output = self.chatllm.predict_messages(prompt).content
            #print("Feasibility of finding ", object, "in room ", roomType, " = ", output)
            if output == 'Yes' or output == 'yes':
                return True

        return False

def construct_goal(goal_type, objects):
    goal = ""
    if goal_type == 0:
        goal = "Find"
        for i, obj in enumerate(objects):            
            if i == 0:
                goal = goal + " " + obj
            elif i == len(objects) - 1:
                goal = goal + " and " + obj
            else:
                goal = goal + ", " + obj
    return goal
    
