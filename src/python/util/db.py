import torch

#Store stuff by column
#So features can be effectively indexed
class Table:
    def __init__(self,d):
        self.d=d;
        self.cache={};
        return;
    
    def from_rows(rows):
        #Get a list of fields
        all_keys=set();
        for r in rows:
            all_keys=all_keys.union(set(r.keys()));
        
        all_keys=list(all_keys);
        d={};
        for k in all_keys:
            #Make sure that tensor fields are present in every row
            if isinstance(rows[0][k],torch.Tensor):
                if not all([k in r for r in rows]):
                    raise ValueError('Tensor field %s not present in all rows'%k);
                
                v=[r[k] for r in rows];
                try:
                    v=torch.stack(v,dim=0);
                except:
                    pass;
                
                d[k]=v;
            else:
                l=[];
                for r in rows:
                    if k in r:
                        l.append(r[k]);
                    else:
                        l.append(None);
                
                d[k]=l;
        
        return Table(d);
    
    def from_csv_file(fname):
        import csv
        with open(fname) as f:
            reader=csv.reader(f);
            keys=None;
            data=[];
            for i,row in enumerate(reader):
                if i==0:
                    keys=row;
                else:
                    data.append(dict(zip(keys,row)));
        
        return Table.from_rows(data);
    
    def sort_by(self,k,reverse=False):
        data=self.d[k];
        #Get sort index
        if isinstance(data,list):
            ind=sorted(range(len(data)),key=data.__getitem__,reverse=reverse);
        else: #Tensors
            _,ind=data.sort(dim=0,descending=reverse);
            ind=ind.cpu().tolist();
        
        for field in self.d:
            data=self.d[field];
            if isinstance(data,list):
                self.d[field]=[data[i] for i in ind];
            else:
                self.d[field]=data[torch.LongTensor(ind)].contiguous();
        
        self.cache={};
        return ind;
    
    def select_by_index(self,ind,fields=None):
        if fields is None:
            fields=self.fields();
        
        d={};
        for field in fields:
            data=self.d[field];
            if isinstance(data,list):
                d[field]=[data[i] for i in ind];
            else:
                d[field]=data[torch.LongTensor(ind)].contiguous();
        
        return Table(d);
    
    def cuda(self):
        for field in self.d:
            data=self.d[field];
            if not isinstance(data,list):
                self.d[field]=data.cuda(non_blocking=True);
            else:
                if len(data)>0 and torch.is_tensor(data[0]):
                    self.d[field]=[x.cuda(non_blocking=True) for x in data]
        
        return self;
    
    def cpu(self):
        for field in self.d:
            data=self.d[field];
            if not isinstance(data,list):
                self.d[field]=data.cpu();
            else:
                if len(data)>0 and torch.is_tensor(data[0]):
                    self.d[field]=[x.cpu() for x in data]
        
        return self;
    
    def fields(self):
        return list(self.d.keys());
    
    def __len__(self):
        return len(next(iter(self.d.values())));
    
    def __getitem__(self,id):
        if isinstance(id,str):
            return self.d[id];
        else:
            return dict([(field,self.d[field][id]) for field in self.d]);
    
    def __setitem__(self,id,data):
        if isinstance(id,str):
            self.d[id]=data;
        else:
            for field in self.d:
                self.d[field][id]=data[field];
        
        self.cache={};
        return data;
    
    def row(self,id):
        return dict([(field,self.d[field][id]) for field in self.d]);
    
    def rows(self):
        for i in range(len(self)):
            yield self.row(i);
    
    def column(self,field):
        return self.d[field]
    
    def rename_column(self,old_key,new_key):
        self.d[new_key]=self.d.pop(old_key);
        return;
    
    def delete_column(self,old_key):
        self.d.pop(old_key);
        self.cache={};
        return;
    
    def add_index(self,name='__id__'):
        N=self.__len__();
        id=list(range(N));
        self.d[name]=id;
        self.cache={};
        return;
    
    def data(self):
        return self.d;

def inner_join(tbl1,tbl2,keyeq1,keyeq2=None,keys1=None,keys2=None):
    if keyeq2 is None:
        keyeq2=keyeq1;
    
    if keys1 is None:
        keys1=tbl1.fields();
    
    if keys2 is None:
        keys2=set(tbl2.fields())
        keys2=keys2.difference(set(tbl1.fields()))
        keys2=list(keys2);
    
    #Generate lookup tables for quick matching
    try:
        tmp=tbl1.cache;
    except:
        tbl1.cache={};
    
    try:
        tmp=tbl2.cache;
    except:
        tbl2.cache={};
    
    if keyeq1 in tbl1.cache:
        lookup1=tbl1.cache[keyeq1];
    else:
        lookup1={};
        for i,v in enumerate(tbl1[keyeq1]):
            if not v in lookup1:
                lookup1[v]=[];
            
            lookup1[v].append(i);
        
        tbl1.cache[keyeq1]=lookup1;
    
    if keyeq2 in tbl2.cache:
        lookup2=tbl2.cache[keyeq2];
    else:
        lookup2={};
        for i,v in enumerate(tbl2[keyeq2]):
            if not v in lookup2:
                lookup2[v]=[];
            
            lookup2[v].append(i);
        
        tbl2.cache[keyeq2]=lookup2;
    
    
    #Find matchings
    ind1=[];
    ind2=[];
    for v in lookup1:
        if v in lookup2:
            for i in lookup1[v]:
                for j in lookup2[v]:
                    ind1.append(i);
                    ind2.append(j);
    
    #Create new table
    d={};
    for field in keys1:
        data=tbl1[field]
        if isinstance(data,list):
            d[field]=[data[i] for i in ind1];
        else:
            d[field]=data[torch.LongTensor(ind1)].contiguous();
    
    for field in keys2:
        data=tbl2[field]
        if isinstance(data,list):
            d[field]=[data[i] for i in ind2];
        else:
            d[field]=data[torch.LongTensor(ind2)].contiguous();
    
    return Table(d);

def left_join(tbl1,tbl2,keyeq1,keyeq2=None,keys1=None,keys2=None):
    if keyeq2 is None:
        keyeq2=keyeq1;
    
    if keys1 is None:
        keys1=tbl1.fields();
    
    if keys2 is None:
        keys2=set(tbl2.fields())
        keys2=keys2.difference(set(tbl1.fields()))
        keys2=list(keys2);
    
    #Generate lookup tables for quick matching
    try:
        tmp=tbl1.cache;
    except:
        tbl1.cache={};
    
    try:
        tmp=tbl2.cache;
    except:
        tbl2.cache={};
    
    if keyeq1 in tbl1.cache:
        lookup1=tbl1.cache[keyeq1];
    else:
        lookup1={};
        for i,v in enumerate(tbl1[keyeq1]):
            if not v in lookup1:
                lookup1[v]=[];
            
            lookup1[v].append(i);
        
        tbl1.cache[keyeq1]=lookup1;
    
    if keyeq2 in tbl2.cache:
        lookup2=tbl2.cache[keyeq2];
    else:
        lookup2={};
        for i,v in enumerate(tbl2[keyeq2]):
            if not v in lookup2:
                lookup2[v]=[];
            
            lookup2[v].append(i);
        
        tbl2.cache[keyeq2]=lookup2;
    
    #Find matchings
    ind1=[];
    ind2=[];
    for v in lookup1:
        if v in lookup2:
            for i in lookup1[v]:
                for j in lookup2[v]:
                    ind1.append(i);
                    ind2.append(j);
        
        else:
            for i in lookup1[v]:
                ind1.append(i);
                ind2.append(None);
    
    #Create new table
    d={};
    for field in keys1:
        data=tbl1[field]
        if isinstance(data,list):
            d[field]=[data[i] if i is not None else None for i in ind1];
        else:
            d[field]=data[torch.LongTensor(ind1)].contiguous();
    
    for field in keys2:
        data=tbl2[field]
        if isinstance(data,list):
            d[field]=[data[i] if i is not None else None for i in ind2];
        else:
            d[field]=data[torch.LongTensor(ind2)].contiguous();
    
    return Table(d);

def unique(tbl,key,keys=None):
    if keys is None:
        keys=tbl.fields();
    
    seen=set();
    ind=[];
    for i,v in enumerate(tbl[key]):
        if not v in seen:
            seen.add(v);
            ind.append(i);
    
    return tbl.select_by_index(ind,keys);

def count(tbl,key,name='__count__'):
    cnt=dict();
    for i,v in enumerate(tbl[key]):
        if not v in cnt:
            cnt[v]=0;
        
        cnt[v]+=1;
    
    d=[{key:v,name:cnt[v]} for v in cnt]
    return Table.from_rows(d);

def union(tbl1,tbl2):
    fields=set(tbl1.fields()).union(set(tbl2.fields()));
    d={};
    for field in fields:
        if field in tbl1.fields() and field in tbl2.fields():
            if isinstance(tbl1[field],list):
                d[field]=tbl1[field]+tbl2[field];
            else:
                d[field]=torch.cat((tbl1[field],tbl2[field]),dim=0);
        elif field in tbl1.fields() and not field in tbl2.fields():
            if isinstance(tbl1[field],list):
                d[field]=tbl1[field]+[None for i in range(len(tbl2))];
            else:
                raise ValueError('Union cannot merge tensor field %s with None'%field);
        elif field in tbl2.fields() and not field in tbl1.fields():
            if isinstance(tbl2[field],list):
                d[field]=[None for i in range(len(tbl1))]+tbl2[field];
            else:
                raise ValueError('Union cannot merge tensor field %s with None'%field);
        else:
            raise ValueError('Field %s did not register'%field);
    
    return Table(d);

def filter_index(table,f):
    ind=[i for i in range(len(table)) if f(table[i])];
    return ind;

def filter(table,f):
    ind=[i for i in range(len(table)) if f(table[i])];
    return table.select_by_index(ind);

def gather(tbl,key_query,key_value,key_into=None):
    if key_into is None:
       key_into=key_value;
    
    data=dict();
    for i,v in enumerate(tbl[key_query]):
        if not tbl[key_value][i] is None:
            if v in data:
                data[v].append(tbl[key_value][i]);
            else:
                data[v]=[tbl[key_value][i]];
    
    data=list(zip(*[(k,data[k]) for k in data]))
    return Table({key_query:list(data[0]),key_into:list(data[1])});


#Makes it easier to name and access a bunch of tables
class DB:
    def __init__(self,d):
        self.d=d;
        return;
    
    def load(fname):
        d=torch.load(fname);
        d=dict([(k,Table(d[k])) for k in d]);
        return DB(d);
    
    def save(self,fname):
        d=dict([(k,self.d[k].data()) for k in self.d]);
        torch.save(d,fname);
        return;
    
    def list_tables(self):
        return self.d.keys();
    
    def __setitem__(self,name,tbl):
        if tbl is None:
            self.d.pop(name);
        else:
            self.d[name]=tbl;
        
        return tbl;
    
    def __getitem__(self,name):
        return self.d[name];
    
    def rename_table(self,old_key,new_key):
        self.d[new_key]=self.d.pop(old_key);
        return;
    
    def delete_table(self,old_key):
        self.d.pop(old_key);
        return;
    
    def cuda(self):
        for tbl in self.d:
            self.d[tbl].cuda();
        
        return self
    
    def cpu(self):
        for tbl in self.d:
            self.d[tbl].cpu();
        
        return self;

class Dataloader:
    #The main interface is to connect to a DB
    #Batch size is provided to allow iterator-based data loading
    def __init__(self,d):
        if isinstance(d,str):
            fname=d;
            self.data=DB.load(fname);
        else:
            self.data=d;
    
    def cuda(self):
        self.data.cuda();
    
    def cpu(self):
        self.data.cpu();
    
    
    