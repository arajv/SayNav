import os
import os.path
import sys
import torch
import json
from collections import OrderedDict

def find(s,root='sessions'):
    folders=os.listdir(root)
    folders=sorted(folders)
    results=[];
    for folder in folders:
        try:
            f=open(os.path.join(root,folder,'params.json'),'r');
            params=json.load(f)
            v=json.dumps(params)
            if v.find(s)>=0:
                results.append({'session':folder,'params':params})
        except:
            pass
    
    return results;

def create_session(params):
    try:
        session_dir=params.session_dir;
    except:
        session_dir=None;
    
    session=Session(session_dir=session_dir); #Create session
    torch.save({'params':params},session.file('params.pt'));
    pmvs=vars(params);
    pmvs=dict([(k,pmvs[k]) for k in pmvs if not(k=='stuff')]);
    print(pmvs);
    f=open(session.file('params.json'),'w');
    json.dump(pmvs,f); #Write a human-readable parameter json
    f.close();
    session.file('model','dummy');
    return session;

class loss_tracker:
    def __init__(self):
        self.loss=OrderedDict();
        return;
    
    def add(self,**kwargs):
        for k in kwargs:
            if not k in self.loss:
                self.loss[k]=[];
            self.loss[k].append(float(kwargs[k]));
    
    def mean(self):
        data={};
        for k in self.loss:
            data[k]=sum(self.loss[k])/len(self.loss[k]);
        
        return data;
    
    def str(self,format='%.3f'):
        s='';
        for k in self.loss:
            s=s+k+': '+format%(sum(self.loss[k])/len(self.loss[k]));
            s=s+', '
        
        return s;

class Session:
    id=-1;
    def __init__(self,id=-1,root='sessions', name=None, session_dir=None):
        if session_dir is not None:
            if not os.path.isdir(session_dir):
                os.mkdir(session_dir);
            self.id = session_dir
        else:
            self.root=root;
            if not os.path.isdir(root):
                os.mkdir(root);
            if id==-1:
                #New session
                existing_sessions=dict();
                for dir in os.listdir(root):
                    existing_sessions[dir]=1;
                id=0;
                while True:
                    if '%07d'%id in existing_sessions:
                        id=id+1;
                    else:
                        break;
                os.mkdir(os.path.join(root,'%07d'%id));
            elif not os.path.isdir(os.path.join(root,'%07d'%id)):
                os.mkdir(os.path.join(root,'%07d'%id));

            if name:
                os.symlink('%07d'%id, os.path.join(root, name))
            
            self.id=os.path.join(root,'%07d'%id);
            return;
    #
    def file(self,p1='',p2='',p3=''):
        if not os.path.isdir(self.id):
            os.mkdir(self.id);
        if p1=='':
            return self.id;
        elif p2=='':
            return os.path.join(self.id,p1);
        elif p3=='':
            if not os.path.isdir(os.path.join(self.id,p1)):
                os.mkdir(os.path.join(self.id,p1));
            return os.path.join(self.id,p1,p2);
        else:
            if not os.path.isdir(os.path.join(self.id,p1)):
                os.mkdir(os.path.join(self.id,p1));
            if not os.path.isdir(os.path.join(self.id,p1,p2)):
                os.mkdir(os.path.join(self.id,p1,p2));
            return os.path.join(self.id,p1,p2,p3);
    
    def log(self,str,fname='log.txt'):
        print(str);
        sys.stdout.flush()
        f=open(self.file(fname),'a');
        f.write(str+'\n');
        f.close();
    
    def log_test(self,str):
        print(str);
        sys.stdout.flush()
        f=open(self.file('test.txt'),'a');
        f.write(str+'\n');
        f.close();
