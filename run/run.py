import prior
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from pipeline import pipeline, baseline

if __name__ == "__main__":
    
    dataset = prior.load_dataset("procthor-10k")
    house_idx = 10
    house_instance = dataset["train"][house_idx]
    
    controller = Controller(agentMode="default", renderInstanceSegmentation=True, renderDepthImage=True, width = 320, height = 240, visibilityDistance=2.5, platform=CloudRendering)
    controller.reset(scene=house_instance)
    objects = ["AlarmClock", "Laptop", "CellPhone"]
    
    params = {}
    params['General'] = {}
    params['Agent'] = {}
    params['SceneGraph'] = {}

    ## General Parameters
    params['General']['output_folder'] = ''    
    params['General']['scene_graph_source'] = 0 # 0: From Ground Truth Positions, 2: Visual Observations 
    params['General']['enable_llm_room_tracking'] = False
    params['General']['llm_model_name'] = 'gpt-3.5-turbo'
    #params['General']['llm_model_name'] = 'gpt-4'
    params['General']['openai_key'] = '' # Enter the openai key
    
    ## Low-Level Planner (Agent) Parameters
    params['Agent']['grid_size'] = 0.25
    params['Agent']['nav_policy_type'] = 1 # 0: Jump from one point to another, 1: Oracle Planner, 2: PointNav Planner
    params['Agent']['policy_model'] = 'point_nav.pt'

    ## Scene Graph Parameters
    params['SceneGraph']['min_pixels_per_object'] = 20
    params['SceneGraph']['door_wall_matching_thresh_dist'] = 0.05

    ## Use either this:
    params['General']['start_room_num'] = 2    
    ## Or these:
    #params['General']['start_position'] = (1.0, 3.75) 
    #params['General']['start_heading'] = 180

    ## Provide GT to compute SPL
    #params['General']['shortest_path_length'] = 10.0
    
    ## Run the pipeline
    pipeline(house_instance, controller, objects, params)