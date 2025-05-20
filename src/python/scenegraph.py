from typing import Union
from enum import Enum 
from types import SimpleNamespace as SN

class NodeType(Enum):
    Undefined = 0
    Place = 1
    Room = 2
    Object = 3
    SmallObject = 4

class EdgeType(Enum):
    Undefined = 0
    Door = 1

class Node:
    def __init__(self, node_id:Union[int, str]=0, node_type:NodeType=NodeType.Undefined, data:dict={}):
        self.id = node_id 
        self.type = node_type
        self.data = data
        self.children = {}
        self.parent = None
        
    def depth(self):
        if self.is_root():
            return 0
        else:
            return 1+self.parent.depth()

    def add_child(self, child):
        if child.id not in self.children:
            child.parent = self
            self.children[child.id] = child
    
    def set_parent(self, parent):
        if self.parent is not None:
            if self.id in self.parent.children:
                del self.parent.children[self.id]
        self.parent = parent
        if self.id not in self.parent.children:
            self.parent.children[self.id] = self
        #return self.parent
            
    def is_root(self):
        return (self.parent == None)
    
    def is_leaf(self):
        return (len(self.children)==0)
    
    def is_id(self, node_id:Union[int, str], response:list):
        for child in self.children.values():
            child.is_id(node_id, response)
        if node_id == self.id:
            response.append(self)
    
    def print_(self):
        s = "  "*self.depth()
        print(f"{s}{self.id}")
        for child in self.children.values():
            child.print_()
    
    def count_node(self, count_list:list):
        for child in self.children.values():
            child.count_node(count_list)
        count_list.append(1)

    # only get id in this search
    def count_node_by_keyword(self, keyword:Union[int, str], count_list:list):
        for child in self.children.values():
            child.count_node_by_keyword(keyword, count_list)
        if keyword in self.id:
            count_list.append(self.id)

    # get dict of {id: self} in this search
    # use this to get the whole node info than just id
    def get_node_by_keyword(self, keyword:Union[int, str], count_dict:dict):
        for child in self.children.values():
            child.get_node_by_keyword(keyword, count_dict)
        if keyword in self.id:
            count_dict[self.id]=self

    def count_node_by_type(self, node_type:NodeType, count_list:list):
        for child in self.children.values():
            child.count_node_by_type(node_type, count_list)
        if node_type == self.type:
            count_list.append(self.id)

    def get_node_by_type(self, node_type:NodeType, count_dict:dict):
        for child in self.children.values():
            child.get_node_by_type(node_type, count_dict)
        if node_type == self.type:
            count_dict[self.id]=self
                
    def __repr__(self):
        if self.parent is None:
            return f"<Node (id:{self.id}, depth:{self.depth()}, num_children:{len(self.children)}, no parent)>"
        else:
            return f"<Node (id:{self.id}, depth:{self.depth()}, num_children:{len(self.children)}, parent:{self.parent.id})>"

class Edge:
    # Directional edge
    def __init__(self, 
            node_id_src:Union[int, str], 
            node_id_to:Union[int, str], 
            edge_type:EdgeType=EdgeType.Undefined, 
            edge_data:dict={}):
        self.id = (node_id_src, node_id_to, edge_type) # tuple is hashable
        self.type = edge_type
        self.data = edge_data
        self.node_ids = SN(src=node_id_src, to=node_id_to)

    def updateData(self, edge_data:dict):
        self.data = edge_data

    def __repr__(self):
        return f"<Edge (TYPE {self.type}: {self.node_ids.src} --> {self.node_ids.to})>"
 
class SceneGraph:
    # For now, it comes with limited support
    #  1) It is more like a tree (An edge can be added yet limited to nodes of the same depth (or level))
    #  2) the root cannot be changed (so initialize the graph with the root defined.)
    #  3) once a node is added, its parent should not be changed.
    def __init__(self, name: str = None, node:Node=None):
        self.name = name
        if node is None:
            self.root = None 
        else:
            self.root = Node(node.id, node.type, node.data)
        self._node_ids = set()
        self._edges = dict() # key: id 
                             # value: edge instance
        self._edge_list_by_node = dict() # This is similar to an adjacent list 
                                         # implmented as a dictionary
                                         # for which the keys are node ids and values are edge ids
        self._edge_list_by_edgetype = {etype: [] for etype in EdgeType} 
                                         # This is a dictionary
                                         # for which the keys are edge types and values are edge ids
    
    def isExist(self, node_id:Union[int, str]):
        if node_id in self._node_ids:
            return True
        return False

    def isEdgeExist(self, edge_id):
        if edge_id in self._edges.keys():
            return True
        return False


    # TODO
    """
    def queryEdges(self, node1_id:Union[int, str], node2_id:Union[int, str]=None, edge_type:EdgeType=None):
        if node2_id is None and edge_type is None:
            return dict()
        assert self.isExist(node1_id) and (node1_id in self._edge_list)
        adj_lists = self._edge_list[node1_id]
        if node2_id is not None and edge_type is None:
            edge_list = dict()
            for etype, elist in adj_lists.items():
                for adj_pair in elist:
                    if node2_id == adj_pair[1]:
                        if etype in edge_list:
                            edge_list[etype].append(adj_pair)
            return False
        elif edge_type is not None and node2_id is None:
            for etype, elist in adj_lists.items():
                if etype == edge_type:
                    return True
            return False
        else: # both node2_id and edge_type are specified 
            for etype, elist in adj_lists.items():
                if etype == edge_type:
                    for adj_pair in elist:
                        if node2_id == adj_pair[1]:
                            return True
            return False
    """
    
    def addNode(self, node: Node, parent_id:Union[int, str]=None): 
        # The first node will be the root node.
        # Currently, changing the root is not supported.
        if self.isExist(node):
            print(f"node {node.id} already exists.")
            return
        if parent_id is None:
            if self.root is None:
                self.root = Node(node.id, node.type, node.data)
                self._node_ids.add(node.id)
                return
            else:
                print(f"cannot add to SG. It requires parent_id to be specified. (It does not support an isolated subgraph)")
                return
        p = self.getNode(parent_id)
        if p is not None:
            #print(f"[SceneGraph] New node is Added. (id:{node.id})")
            p.add_child(node) 
            self._node_ids.add(node.id)
            self._edge_list_by_node[node.id] = []

    def addEdge(self, edge:Edge): 
        # nodes must exist in the graph to add an edge
        if not self.isExist(edge.node_ids.src):
            print(f"node {edge.node_ids.src} does not exist. cannot create an edge.")
            return
        if not self.isExist(edge.node_ids.to):
            print(f"node {edge.node_ids.to} does not exist. cannot create an edge.")
            return
        if self.isEdgeExist(edge.id):
            print(f"edge {edge.id} already exists.")
            return
        self._edges[edge.id] = edge
        # add to edge lists as well for fast lookup
        assert edge.node_ids.src in self._edge_list_by_node
        assert edge.node_ids.to in self._edge_list_by_node
        if edge.id not in self._edge_list_by_node[edge.node_ids.src]:
            self._edge_list_by_node[edge.node_ids.src].append(edge.id)
        #if edge.id not in self._edge_list_by_node[edge.node_ids.to]:
        #    self._edge_list_by_node[edge.node_ids.to].append(edge.id)
        if edge.id not in self._edge_list_by_edgetype[edge.type]:
            self._edge_list_by_edgetype[edge.type].append(edge.id)


    def numNodes(self):
        # proper version
        """
        count_list = []
        if self.root is not None:
            self.root.count_node(count_list)
        return len(count_list)
        """
        # fast version
        return len(self._node_ids)

    def findNodes(self, keyword_in_id:Union[int, str], return_id_only=True):        
        if self.root is not None:
            if return_id_only:
                count_list = []
                self.root.count_node_by_keyword(keyword_in_id,count_list)
                return len(count_list), count_list
            else:
                count_dict = {}
                self.root.get_node_by_keyword(keyword_in_id,count_dict)
                return len(count_dict), count_dict
        return 0, []

    def getNode(self, node_id:Union[int, str]):
        node_list = []
        if self.root is not None:
            self.root.is_id(node_id,node_list)   
        if len(node_list)==0:
            print(f"no node with id {node_id} exists. returning None")
            return None
        elif len(node_list)>1:
            print(f"Multiple nodes with id {node_id} exist. returning the first one encountered.")      
        return node_list[0]

    def findNodeType(self, node_type:NodeType, room_id=None, return_id_only=True):
        #print(f"{return_id_only=}")    
        count_list = []    
        if room_id is None:
            if self.root is not None:
                if return_id_only:
                    self.root.count_node_by_type(node_type,count_list)
                else:
                    count_list = {}
                    self.root.get_node_by_type(node_type,count_list)
        else:
            room_node = self.getNode(room_id)
            # returns a list of nodes of id
            # ideally, there must be one element in the list
            # if not valid id, it returns None
            if room_node is not None:
                if return_id_only:
                    room_node.count_node_by_type(node_type,count_list)
                else:
                    count_list = {}
                    room_node.get_node_by_type(node_type,count_list)
        return len(count_list), count_list

    def getEdge(self, edge_id):
        if edge_id in self._edges.keys():
            return self._edges[edge_id]
        return None

    def print_(self):
        if self.root is not None:
            self.root.print_()
        for etype, elist in self._edge_list_by_edgetype.items():
            if len(elist)>0:
                print(f"  {etype}: {elist}")
    
    def __repr__(self):
        return f"<SceneGraph object ({self.name}) with {self.numNodes()} nodes.>"
