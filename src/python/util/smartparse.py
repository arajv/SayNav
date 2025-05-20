from collections import OrderedDict
import copy
import sys

class obj(object):
    def __init__(self,d=None):
        if not(d is None):
            for a, b in d.items():
                if isinstance(b, (list, tuple)):
                    setattr(self, a, [obj(x) if isinstance(x, dict) else x for x in b])
                else:
                    setattr(self, a, obj(b) if isinstance(b, dict) else b)


def obj2dict(o,prefix=''):
    params=vars(o);
    d=OrderedDict([(prefix+k,params[k]) for k in params if not k=='_parent']);
    #Recursively convert sub objects to dict
    for k in params:
        if k=='_parent':
            pass;
        else:
            if isinstance(params[k],type(o)):
                d.pop(prefix+k);
                x=obj2dict(params[k],prefix+k+'.');
                d.update(x);
    
    return d;

def dict2obj(d):
    #Expand input into subdictionaries
    d_exp=OrderedDict();
    for k in d:
        levels=k.split('.');
        current_level=d_exp;
        for x in levels[:-1]:
            if x in current_level:
                current_level=current_level[x];
            else:
                current_level[x]=OrderedDict();
                current_level=current_level[x];
        
        current_level[levels[-1]]=d[k];
    
    #Convert dict with subdicts to obj
    return obj(d_exp);

def merge(x,d_default):
    if x is None:
        return d_default;
    
    if not isinstance(x,dict):
        d=obj2dict(x);
    else:
        d=x
    
    if not isinstance(d_default,dict):
        d_default=obj2dict(d_default);
    
    #Create a shallow copy
    d_new=dict([(k,d_default[k]) for k in d_default]);
    #Type match input with default
    for k in d:
        if k in d_new:
            try:
                d_new[k]=type(d_default[k])(d[k]);
            except:
                d_new[k]=d[k];
        else:
            d_new[k]=d[k]
    
    if not isinstance(x,dict):
        d_new=dict2obj(d_new);
    
    return d_new;

def sub(x,prefix=''):
    if not isinstance(x,dict):
        d=obj2dict(x);
    else:
        d=x
    
    d_new={};
    for k in d:
        if k.find(prefix)==0:
            d_new[k[len(prefix):]]=d[k];
    
    if not isinstance(x,dict):
        d_new=dict2obj(d_new);
    
    return d_new;

def chain(x,prefixes=[]):
    if not isinstance(x,dict):
        d=obj2dict(x);
    else:
        d=x
    
    d_new=obj();
    for prefix in prefixes:
        d_new=merge(d_new,sub(d,prefix));
    
    if isinstance(x,dict):
        d_new=obj2dict(d_new);
    
    return d_new;

def parse(default=None,argv=None):
    if argv is None:
        argv=sys.argv;
    
    d=[];
    k=None;
    for v in argv:
        if v.find('--')==0:
            if not k is None:
                d.append((k,True));
                k=None;
            
            k=v[2:];
            k=k.replace('-','_');
        elif not (k is None):
            d.append((k,v));
            k=None;
    
    if not k is None:
        d.append((k,True));
        k=None;
    
    d=OrderedDict(d);
    d=dict2obj(d);
    if not default is None:
        d=merge(d,default);
    return d;
