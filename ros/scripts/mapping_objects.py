#!/usr/bin/env python3
import rospy
import numpy as np
from hydra_msgs.msg import ObjectInfo
import networkx as nx
import matplotlib.pyplot as plt
import csv
import tf
from geometry_msgs.msg import PoseStamped
import tf2_geometry_msgs
import tf2_ros
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import dill
import signal
import argparse

plt.ion()
plt.show()

class connection:
    def __init__(self,pair):
        self.pair = pair
        self.time_history = [0]*10
        self.back_in_view = False
        self.no_of_det = 0
        self.back_in_view_time = 0.0

class household_object:
    def __init__(self,label_id,pose,bbox_min,bbox_max):
        self.label_id = label_id
        self.pose = pose
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

class mapping_objects:
    def __init__(self,file_name):
        path = "/home/aswingururaj/catkin_ws/src/kimera_semantics/kimera_semantics_ros/cfg/mask_rcnn_mapping.csv"
        self.object_maps = {}
        self.color_maps = {}
        self.static_objects = []
        self.tracking_objects = []
        self.connections = []
        self.read_pkl(file_name)
        with open(path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            next(reader,None)
            for row in reader:
                self.object_maps[row['id']] = row['name']
                self.color_maps[row['id']] = [row['blue'],row['green'],row['red']]
        self.sub = rospy.Subscriber("/incremental_dsg_builder_node/object_info", ObjectInfo, self.callback)
        signal.signal(signal.SIGINT, self.handler)

    def callback(self,data):
        if len(data.labels) < 2:
            return
        centroids = []
        for i in range(len(data.labels)):
            centroids.append([(data.bbox_min[i].x + data.bbox_max[i].x)/2, (data.bbox_min[i].y + data.bbox_max[i].y)/2, (data.bbox_min[i].z + data.bbox_max[i].z)/2])
        for i in range(len(data.labels)):
            for j in range(len(data.labels)):
                if self.isPointAboveBbox(data.bbox_min[i],data.bbox_max[i],centroids[j]) and i!=j:
                    new_object = household_object(data.labels[j],data.positions[j],data.bbox_min[j],data.bbox_max[j])
                    new_connection = connection([data.labels[j],data.labels[i]])
                    count = 0
                    for x in self.connections:
                        if x.pair != [data.labels[j],data.labels[i]]:
                            count = count + 1
                    if count == len(self.connections):
                        print("Adding new connection ", data.labels[j], "  ", data.labels[i])
                        self.connections.append(new_connection)
                        self.tracking_objects.append(new_object)
    
    def read_pkl(self,file_name):
        with open('/home/aswingururaj/catkin_ws/src/orb_slam_2_ros/ros/scripts/'+file_name,'rb') as handle:
                count = dill.load(handle)
                for i in range(count):
                    obj = dill.load(handle)
                    self.static_objects.append(obj)

    def hierarchy_pos(self,G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

        '''
        From Joel's answer at https://stackoverflow.com/a/29597209/2966723.  
        Licensed under Creative Commons Attribution-Share Alike 
        
        If the graph is a tree this will return the positions to plot this in a 
        hierarchical layout.
        
        G: the graph (must be a tree)
        
        root: the root node of current branch 
        - if the tree is directed and this is not given, 
        the root will be found and used
        - if the tree is directed and this is given, then 
        the positions will be just for the descendants of this node.
        - if the tree is undirected and not given, 
        then a random choice will be used.
        
        width: horizontal space allocated for this branch - avoids overlap with other branches
        
        vert_gap: gap between levels of hierarchy
        
        vert_loc: vertical location of root
        
        xcenter: horizontal location of root
        '''
        if not nx.is_tree(G):
            raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

        if root is None:
            if isinstance(G, nx.DiGraph):
                root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
            else:
                root = random.choice(list(G.nodes))

        def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
            '''
            see hierarchy_pos docstring for most arguments

            pos: a dict saying where all nodes go if they have been assigned
            parent: parent of this branch. - only affects it if non-directed

            '''
        
            if pos is None:
                pos = {root:(xcenter,vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)
            children = list(G.neighbors(root))
            if not isinstance(G, nx.DiGraph) and parent is not None:
                children.remove(parent)  
            if len(children)!=0:
                dx = width/len(children) 
                nextx = xcenter - width/2 - dx/2
                for child in children:
                    nextx += dx
                    pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                        vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                        pos=pos, parent = root)
            return pos

        return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

    def visualize_scene_graphs(self,objects):
        if len(objects)>0:
            self.G = nx.DiGraph()
            for x in objects:
                obj = self.object_maps[str(x.label_id)]
                self.G.add_edge("kitchen",obj,label='in')
            for x in self.connections:
                obj_2 = self.object_maps[str(x.pair[0])]
                obj_1 = self.object_maps[str(x.pair[1])]
                self.G.add_edge(obj_1,obj_2,label='on')
            layout = self.hierarchy_pos(self.G,"kitchen")
            nx.draw(
            self.G, pos=layout, edge_color='black', width=1, linewidths=1,
            node_size=10000, node_color='red', alpha=0.9,
            labels={node: node for node in self.G.nodes()}
            )
            nx.draw_networkx_edge_labels(
                self.G, pos=layout,edge_labels=nx.get_edge_attributes(self.G,'label'),
                font_color='black'
            )
            plt.axis('off')
            plt.pause(1e-10)
            plt.clf()

    def start(self):
        rate = rospy.Rate(15.0)
        while not rospy.is_shutdown():
            self.visualize_scene_graphs(self.static_objects)
            rate.sleep()

    def isPointAboveBbox(self,bbox_min,bbox_max,point):
        if (bbox_min.x <= point[0] <= bbox_max.x) and (bbox_min.y <= point[1] <= bbox_max.y) and (0.8*bbox_max.z <= point[2] <= bbox_max.z + 2):
            return True

    def handler(self,signum,frame):
        if len(self.connections)>0:
            with open('/home/aswingururaj/catkin_ws/src/orb_slam_2_ros/ros/scripts/current_scene_graph.pkl', 'wb') as outp:
                    print("DUMPING")
                    dill.dump(len(self.static_objects),outp)
                    for i in range(len(self.static_objects)):
                        dill.dump(self.static_objects[i], outp)
                    dill.dump(len(self.tracking_objects),outp)
                    for i in range(len(self.tracking_objects)):
                        dill.dump(self.tracking_objects[i], outp)
                    dill.dump(len(self.connections),outp)
                    for i in range(len(self.connections)):
                        dill.dump(self.connections[i], outp)
        exit(1)

if __name__ == '__main__':
    rospy.init_node('mapping_objects', anonymous=True)
    file_name = 'static_objects.pkl'
    mapping_objects = mapping_objects(file_name)
    mapping_objects.start()