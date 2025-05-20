
#ResNet with group norm and adjustable number of input channels
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.models.rnn_state_encoder import GRUStateEncoder
from habitat_baselines.utils.common import CategoricalNet
import torch.nn.functional as F
import torch.nn as nn
import torch

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
        self.visual_encoder=resnet.resnet18(4,32,16)
        #self.visual_fc=nn.Sequential(nn.Conv2d(256,128,3,padding=1,bias=False),nn.GroupNorm(1, 128),nn.ReLU(),nn.Flatten(),nn.Linear(2048,512),nn.ReLU())
        
        self.visual_fc=nn.Sequential(nn.Conv2d(256,32,3,padding=1,bias=False),nn.GroupNorm(1, 32),nn.ReLU(),nn.Flatten(),nn.Linear(32*80,512),nn.ReLU())
        self.t_target=nn.Linear(6,32) #target embedding
        self.embed_action=nn.Embedding(5,32) #previous action embedding
        
        data_size=512+32+32
        self.rnn_state_encoder=GRUStateEncoder(data_size,512,num_layers=1)
        
        self.action_distribution_head=CategoricalNet(512,4)
        self.critic_head=nn.Linear(512,1)
        
        self.num_recurrent_layers=1
    
    
    def forward_imitation(self,obs,a_prev):
        rgb=obs['rgb'].float()
        depth=obs['depth']
        
        im=torch.cat((rgb,depth),dim=-3)
        h_vis=self.visual_encoder(im)
        h_vis=self.visual_fc(h_vis)
        
        gps=obs['gps']
        goal=obs['pointgoal']
        compass=obs['compass']
        #Xiao: augment a bit
        diff=goal-gps
        r=F.normalize(diff,dim=-1)
        r0=torch.stack((torch.cos(compass[:,0]),torch.sin(compass[:,0])),dim=-1)
        s=torch.stack((r[:,0]*r0[:,0]+r[:,1]*r0[:,1],r[:,0]*r0[:,1]-r[:,1]*r0[:,0]),dim=-1)
        gps=torch.stack((diff[:,0],diff[:,1],s[:,0],s[:,1],r0[:,0],r[:,0]),dim=-1)
        h_gps=self.t_target(gps)
        
        
        a_prev=a_prev+1
        h_action=self.embed_action(a_prev.view(-1))
        
        h_input=torch.cat((h_vis,h_gps,h_action),dim=-1)
        h_prev=torch.zeros(1,512).to(h_input.device)
        
        #h_out, h_next = self.rnn_state_encoder(h_input, h_prev, not_done_mask, rnn_build_seq_info)
        h_out, h_next = self.rnn_state_encoder.rnn(h_input, h_prev)
        
        distribution = self.action_distribution_head(h_out)
        value = self.critic_head(h_out)
        action_logits = distribution.logits
        return action_logits,value
    
    def forward(self,obs,h_prev,a_prev,not_done_mask,rnn_build_seq_info=None):
        rgb=obs['rgb'].float()/255.0
        rgb=rgb.transpose(-1,-3).transpose(-1,-2) #HWC -> CHW
        #rgb_mean=torch.Tensor([0.485, 0.456, 0.406]).view(3,1,1).to(rgb.device)
        #rgb_std=torch.Tensor([0.229, 0.224, 0.225]).view(3,1,1).to(rgb.device)
        #rgb=(rgb-rgb_mean)/rgb_std
        assert rgb.shape[-1]==320
        assert rgb.shape[-2]==240
        
        depth=obs['depth']
        depth=depth.transpose(-1,-3).transpose(-1,-2)
        
        im=torch.cat((rgb,depth),dim=-3)
        #im = F.avg_pool2d(im, 2)
        h_vis=self.visual_encoder(im)
        h_vis=self.visual_fc(h_vis)
        
        
        gps=obs['gps']
        goal=obs['pointgoal']
        compass=obs['compass']
        #Xiao: augment a bit
        diff=goal-gps
        r=F.normalize(diff,dim=-1)
        r0=torch.stack((torch.cos(compass[:,0]),torch.sin(compass[:,0])),dim=-1)
        s=torch.stack((r[:,0]*r0[:,0]+r[:,1]*r0[:,1],r[:,0]*r0[:,1]-r[:,1]*r0[:,0]),dim=-1)
        gps=torch.stack((diff[:,0],diff[:,1],s[:,0],s[:,1],r0[:,0],r0[:,1]),dim=-1)
        h_gps=self.t_target(gps)
        
        a_prev=torch.where(not_done_mask==True,a_prev+1,0) #reset action on done
        h_action=self.embed_action(a_prev.view(-1))
        
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

