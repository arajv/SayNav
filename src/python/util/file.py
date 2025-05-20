import json

def write_json(fname,data):
	f=open(fname,'w');
	json.dump(data,f);
	return;

def load_json(fname):
	f=open(fname,'r');
	data=json.load(f);
	return data;
