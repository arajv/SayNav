
#ResNet with group norm and adjustable number of input channels
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.models.rnn_state_encoder import GRUStateEncoder
from habitat_baselines.utils.common import CategoricalNet
import torch
import torch.nn.functional as F
import torch.nn as nn

class obj(object):
    def __init__(self,d=None):
        if not(d is None):
            for a, b in d.items():
                if isinstance(b, (list, tuple)):
                    setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
                else:
                    setattr(self, a, obj(b) if isinstance(b, dict) else b)


class new(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder=resnet.resnet50(4,32,16)
        self.visual_fc=nn.Sequential(nn.Conv2d(1024,128,3,padding=1,bias=False),nn.GroupNorm(1, 128),nn.ReLU(),nn.Flatten(),nn.Linear(2048,512),nn.ReLU())
        self.t_target=nn.Linear(3,32) #target embedding
        self.embed_action=nn.Embedding(5,32) #previous action embedding
        
        data_size=512+32+32
        self.rnn_state_encoder=GRUStateEncoder(data_size,512,num_layers=1)
        
        self.action_distribution_head=CategoricalNet(512,4)
        self.critic_head=nn.Linear(512,1)
        
        self.num_recurrent_layers=1
    
    
    def forward(self,obs,h_prev,a_prev,not_done_mask,rnn_build_seq_info=None):
        rgb=obs['rgb'].float()/255.0
        rgb=rgb.transpose(-1,-3).transpose(-1,-2) #HWC -> CHW
        #rgb_mean=torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1).to(rgb.device)
        #rgb_std=torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1).to(rgb.device)
        #rgb=(rgb-rgb_mean)/rgb_std
        
        depth=obs['depth']
        depth=depth.transpose(-1,-3).transpose(-1,-2)
        
        im=torch.cat((rgb,depth),dim=-3)
        im = F.avg_pool2d(im, 2)
        h_vis=self.visual_encoder(im)
        #h_vis=F.adaptive_avg_pool2d(h_vis,(4,4))
        #h_vis=h_vis.view(h_vis.shape[0],-1)
        h_vis=self.visual_fc(h_vis)
        
        gps=obs['pointgoal_with_gps_compass']
        gps=torch.stack((gps[:,0],torch.cos(-gps[:,1]),torch.sin(-gps[:,1])),dim=-1)
        h_gps=self.t_target(gps)
        
        a_prev=torch.where(not_done_mask==True,a_prev+1,0) #reset action on done
        h_action=self.embed_action(a_prev.view(-1))
        
        #print([x.shape for x in (h_prev,obs['rgb'],obs['pointgoal_with_gps_compass'],h_vis,h_gps,h_action)])
        h_input=torch.cat((h_vis,h_gps,h_action),dim=-1)
        
        h_out, h_next = self.rnn_state_encoder(h_input, h_prev, not_done_mask, rnn_build_seq_info)
        return h_out,h_next
    
    def act(self,observations,rnn_hidden_states,prev_actions,masks,deterministic=False):
        h_out, h_next = self.forward(observations, rnn_hidden_states, prev_actions, masks)
        distribution = self.action_distribution_head(h_out)
        value = self.critic_head(h_out)
        
        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()
        
        action_logp = distribution.log_probs(action)
        
        results=obj()
        results.values=value
        results.actions=action
        results.action_log_probs=action_logp
        results.rnn_hidden_states=h_next
        return results
    
    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        h_out, _ = self.forward(observations, rnn_hidden_states, prev_actions, masks)
        value = self.critic_head(h_out)
        return value
    
    def evaluate_actions(self,observations,rnn_hidden_states,prev_actions,masks,action,rnn_build_seq_info,*args,**kwargs):
        h_out, h_next = self.forward(observations, rnn_hidden_states, prev_actions, masks,rnn_build_seq_info)
        distribution = self.action_distribution_head(h_out)
        value = self.critic_head(h_out)
        
        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()
        
        return (value,action_log_probs,distribution_entropy,rnn_hidden_states,{})
    
    def policy_parameters(self):
        return self.parameters()
    
    def aux_loss_parameters(self):
        return {}

