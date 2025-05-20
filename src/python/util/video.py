
import time
import os
from pathlib import Path
import torch
import torchvision
from torchvision import transforms
import torchvision.datasets.folder
import torch.nn.functional as F
from multiprocessing.pool import ThreadPool

import PIL.Image
import numpy
import cv2


def get_video_info_images(video_path,format='%04d.png',start_idx=1,fps=30):
    T=len(os.listdir(video_path))
    
    transform=transforms.Compose([lambda image: image.convert("RGB"),transforms.ToTensor()]);
    fname=os.path.join(video_path,format%start_idx);
    frame=torchvision.datasets.folder.default_loader(fname);
    frame=transform(frame);
    C,H,W=frame.shape;
    return T,H,W,fps;

def get_video_info(video_path):
    reader = cv2.VideoCapture(video_path)
    T=int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    W=int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    H=int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=reader.get(cv2.CAP_PROP_FPS)
    return T,H,W,fps


class load_video_stream:
    def __init__(self,video_path,start_idx=None,nframes=None,fps=None):
        self.reader = cv2.VideoCapture(video_path)
        if not start_idx is None:
            self.reader.set(cv2.CAP_PROP_POS_FRAMES,start_idx);
        
        
        self.transform=transforms.Compose([lambda image: image.convert("RGB"),transforms.ToTensor()]);
        self.fps=self.reader.get(cv2.CAP_PROP_FPS)
        self.n=0;
        self.nframes=nframes
    
    
    def get(self):
        if not self.nframes is None and self.n>=self.nframes:
            return None
        
        success, data = self.reader.read()
        if not success:
            return None
        
        self.n=self.n+1
        
        assert data.shape[-1]==3; #BGR->RGB
        data=data[:,:,::-1]
        
        im=PIL.Image.fromarray(data)
        im=self.transform(im)
        return im

class load_video_images_stream:
    def __init__(self,video_path,format='%04d.png',start_idx=1,nframes=None,fps=None):
        N=len(os.listdir(video_path));
        if nframes is None:
            nframes=N;
        
        self.transform=transforms.Compose([lambda image: image.convert("RGB"),transforms.ToTensor()]);
        self.id=start_idx;
        self.last=start_idx+nframes;
        self.video_path=video_path
        self.format=format
        if fps is None:
            self.fps=30;
        else:
            self.fps=fps;
    
    def get(self):
        if self.id>=self.last:
            return None;
        
        fname=os.path.join(self.video_path,self.format%self.id);
        frame=torchvision.datasets.folder.default_loader(fname);
        #frame=PIL.Image.open(fname)
        #frame=cv2.imread(fname);
        #frame=PIL.Image.fromarray(frame)
        frame=self.transform(frame);
        self.id=self.id+1
        return frame

class write_video_stream:
    def __init__(self,video_path,fps=None):
        if video_path.rfind('/')>0:
            Path(video_path[:video_path.rfind('/')]).mkdir(parents=True, exist_ok=True);
        
        self.video_path=video_path
        self.fps=fps
        self.id=0;
        self.transform=transforms.Compose([transforms.ToPILImage(),lambda image: image.convert("RGB")]);
    
    def put(self,frame):
        if self.id==0:
            self.writer=cv2.VideoWriter(self.video_path,cv2.VideoWriter_fourcc('M','J','P','G'), self.fps,(frame.shape[-1],frame.shape[-2]));
            print('W=%d, H=%d'%(frame.shape[-1],frame.shape[-2]))
        

        frame=self.transform(frame)
        frame=numpy.array(frame)
        assert frame.shape[-1]==3; #RGB->BGR
        frame=frame[:,:,::-1]
        self.writer.write(frame)
        self.id=self.id+1;

class write_video_images_stream:
    def __init__(self,video_path,format='%04d.png',start_idx=1,fps=None,force=False):
        Path(video_path).mkdir(parents=True, exist_ok=force); # fool-proof, avoid overriding existing folders
        self.transform=transforms.Compose([transforms.ToPILImage(),lambda image: image.convert("RGB")]);
        self.start_idx=start_idx;
        self.id=start_idx;
        self.video_path=video_path
        self.format=format
        self.pool = ThreadPool(processes=1)
    
    def put(self,frame):
        if self.id==self.start_idx:
            print('W=%d, H=%d'%(frame.shape[-1],frame.shape[-2]))
        
        fname=os.path.join(self.video_path,self.format%self.id);
        frame=frame.cpu();
        def write(fname,frame):
            frame=self.transform(frame);
            frame=numpy.array(frame)
            assert frame.shape[-1]==3; #RGB->BGR
            frame=frame[:,:,::-1]
            cv2.imwrite(fname,frame)
        
        self.pool.apply_async(write, args=(fname,frame))
        #write(fname,frame)
        self.id=self.id+1;


def load_video(*args,**kwargs):
    frames=[];
    loader=load_video_stream(*args,**kwargs);
    while True:
        frame=loader.get();
        if frame is None:
            break;
        else:
            frames.append(frame);
    
    return torch.stack(frames,dim=0),loader.fps;

def load_video_images(*args,**kwargs):
    frames=[];
    loader=load_video_images_stream(*args,**kwargs);
    while True:
        frame=loader.get();
        if frame is None:
            break;
        else:
            frames.append(frame);
    
    return torch.stack(frames,dim=0),loader.fps;


def write_video(vid,*args,**kwargs):
    loader=write_video_stream(*args,**kwargs);
    for i in range(vid.shape[0]):
        loader.put(vid[i]);
    
    return;

def write_video_images(vid,*args,**kwargs):
    loader=write_video_images_stream(*args,**kwargs);
    for i in range(vid.shape[0]):
        loader.put(vid[i]);
    
    return;


def normalize(video,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]):
    if len(video.shape)==3:
        mean=torch.Tensor(mean).view(3,1,1).to(video.device);
        std=torch.Tensor(std).view(3,1,1).to(video.device);
    elif len(video.shape)==4:
        mean=torch.Tensor(mean).view(1,3,1,1).to(video.device);
        std=torch.Tensor(std).view(1,3,1,1).to(video.device);
    elif len(video.shape)==5:
        mean=torch.Tensor(mean).view(1,1,3,1,1).to(video.device);
        std=torch.Tensor(std).view(1,1,3,1,1).to(video.device);
    
    video=(video-mean)/(std+1e-20);
    return video;

def denormalize(video,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]):
    if len(video.shape)==3:
        mean=torch.Tensor(mean).view(3,1,1).to(video.device);
        std=torch.Tensor(std).view(3,1,1).to(video.device);
    elif len(video.shape)==4:
        mean=torch.Tensor(mean).view(1,3,1,1).to(video.device);
        std=torch.Tensor(std).view(1,3,1,1).to(video.device);
    elif len(video.shape)==5:
        mean=torch.Tensor(mean).view(1,1,3,1,1).to(video.device);
        std=torch.Tensor(std).view(1,1,3,1,1).to(video.device);
    
    video=video*(std+1e-20)+mean;
    return video;

def to_windows(video,window_size):
    T=video.shape[-4];
    remainder=T%window_size;
    if remainder>0:
        pad=torch.flip(video[-(window_size-remainder):,:,:,:],[0]);
        video=torch.cat((video,pad),dim=0);
    
    video=video.view(-1,window_size,*video.shape[-3:]);
    return video

def to_2x_windows(video,window_size):
    T=video.shape[-4];
    remainder=T%window_size;
    if remainder>0:
        pad=torch.flip(video[-(window_size-remainder):,:,:,:],[0]);
        video=torch.cat((video,pad),dim=0);
    
    video=video.view(-1,window_size,*video.shape[-3:]);
    video=torch.cat((video[0:-1],video[1:]),dim=1);
    return video
