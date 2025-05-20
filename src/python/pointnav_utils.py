import torch
import importlib
import math
import copy

def snap(point,grid):
    p=torch.Tensor([[p['x'],p['z']] for p in grid])
    p2=torch.Tensor([point['x'],point['z']])
    d=((p-p2)**2).sum(dim=-1)
    dmin,i=d.min(dim=0)
    return grid[i]

def extract_obs_pointnav(event,goal_pos):
    rgb=copy.deepcopy(event.frame)
    depth=copy.deepcopy(event.depth_frame)
    
    p=event.metadata['agent']['position']
    r=event.metadata['agent']['rotation']
    pos=[p['x'],p['y'],p['z'],r['x'],r['y'],r['z']]
    gps=[p['x'],p['z']]
    compass=[(r['y']+90)/180*math.pi]
    
    #package into a batch 1 query
    rgb=torch.from_numpy(rgb)
    depth=torch.from_numpy(depth).unsqueeze(-1)/20 #Empirical normalization
    gps=torch.Tensor(gps)
    compass=torch.Tensor(compass)
    pointgoal=torch.Tensor([goal_pos['x'],goal_pos['z']])
    
    obs={'rgb':rgb,'depth':depth,'gps':gps,'compass':compass,'pointgoal':pointgoal}
    return obs

def extract_info(event,goal_pos,r=1.5):
    #distance to goal
    p=event.metadata['agent']['position']
    dist=((p['x']-goal_pos['x'])**2+(p['z']-goal_pos['z'])**2)**0.5
    
    success = dist<=r
    info={'success':success,'spl':success,'dist':dist}
    return success,info

#Find shortest path to pgoal
#env is controller instance
#p0 and r0 from event.metadata['agent']['position'] event.metadata['agent']['rotation']
class SP:
    def __init__(self):
        self.forward0=None
        self.forward1=None
        self.failed=0
    
    def act(self,env,p0,r0,pgoal,pos_reachable,forward_dist=0.25,rotate_angle=90.0,success_radius=1.0):
        #Find closest reachable point to pgoal
        #data=env.step(action="GetReachablePositions")
        #pos_reachable=data.metadata['actionReturn']
        
        #pos_reachable=grid
        pgoal_=snap(pgoal,pos_reachable)
        
        if (p0['x']-pgoal_['x'])**2+(p0['z']-pgoal_['z'])**2<success_radius**2:
            return None
        
        if self.forward1==(p0['x'],p0['z'],r0['y']):
            #Last action failed
            self.failed=1
            #Try rotating 90 degrees and move forward
            if torch.rand(1)>0.5:
                action={"action":"RotateLeft","degrees":90}
            else:
                action={"action":"RotateRight","degrees":90}
            return action
        elif self.failed>0:
            action={"action":"MoveAhead","moveMagnitude":forward_dist}
            self.forward0=self.forward1
            self.forward1=(p0['x'],p0['z'],r0['y'])
            self.failed=0
            return action
        
        '''
        p=torch.Tensor([[p['x'],p['z']] for p in pos_reachable])
        p2=torch.Tensor([pgoal['x'],pgoal['z']])
        d=((p-p2)**2).sum(dim=-1)
        dmin,i=d.min(dim=0)
        pgoal_=pos_reachable[i]
        '''
        
        #Find shortest path
        data2=env.step(action="GetShortestPathToPoint",position=p0,target=pgoal_,allowedError=2)
        if data2.metadata['actionReturn'] is None:
            #Shortest path failed
            #Produce a random action or something?
            print('shortest path failed')
            print(data2)
        
        path=data2.metadata['actionReturn']['corners']
        
        path=[snap(x,pos_reachable) for x in path]
        #Search for a waypoint that's not nearby
        for p in path:
            dx=p0['x']-p['x']
            dz=p0['z']-p['z']
            if (dx**2+dz**2)**0.5>forward_dist:
                break
        
        
        #Align facing to first waypoint
        # AI2Thor axis system seems to be
        #   ---->z
        #  /|
        # /x|y
        # AI2Thor rotation along y is from z to x
        # But turn left is z=>-x
        # We first change current rotation to z=>-x
        # Then calculate the rotation difference needed in the z=>-x plane
        # Positive is turn left, negative is turn right
        dx,dz=[p['x']-p0['x'],p['z']-p0['z']]
        if dx==0:
            dx=1e-4
        
        r0=-r0['y']
        r=2*math.atan2(-dx,dz+(dz**2+dx**2)**0.5)
        r=r/math.pi*180
        diff=r-r0
        diff=(diff+179.999)%360-179.999
        if abs(diff)>rotate_angle/2:
            #Rotation needed
            if diff>0:
                action={"action":"RotateLeft","degrees":rotate_angle}
            else:
                action={"action":"RotateRight","degrees":rotate_angle}
        else:
            dx=p0['x']-p['x']
            dz=p0['z']-p['z']
            if (dx**2+dz**2)**0.5>=forward_dist or len(path)>1:
                action={"action":"MoveAhead","moveMagnitude":forward_dist}
                
                self.forward0=self.forward1
                self.forward1=(p0['x'],p0['z'],-r0)
                
            else:
                action=None
        
        print('(%.2f,%.2f) %.2f => (%.2f, %.2f)'%(p0['x'],p0['z'],r0,p['x'],p['z']),action)
        return action