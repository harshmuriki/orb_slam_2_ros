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
        self.time_stamp = 0.0

class household_object:
    def __init__(self,label_id,pose,bbox_min,bbox_max):
        self.label_id = label_id
        self.pose = pose
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

class update_scene_graph:
    def __init__(self,file_name):
        self.br = CvBridge()
        path = "/home/aswingururaj/catkin_ws/src/kimera_semantics/kimera_semantics_ros/cfg/mask_rcnn_mapping.csv"
        self.image = None
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
        self.sub = rospy.Subscriber("/sparseinst_seg", Image, self.image_callback)
        signal.signal(signal.SIGINT, self.handler)
    
    def image_callback(self,msg):
        self.image = self.br.imgmsg_to_cv2(msg,desired_encoding='bgr8')
    
    def callback(self,data):
        print("Length of connections ", len(self.connections))
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
                    flag = 0
                    for x in self.connections:
                        if x.pair != [data.labels[j],data.labels[i]]:
                            count = count + 1
                    for index,v in enumerate(self.connections):
                        if v.pair[0]==data.labels[j] and v.pair[1]!=data.labels[i]:
                            del self.connections[index]
                            break
                    if count == len(self.connections):
                        print("Adding new connection ", data.labels[j], "  ", data.labels[i])
                        new_connection.time_stamp = rospy.Time.now().secs
                        self.connections.append(new_connection)
                        for j in range(len(self.tracking_objects)):
                            if self.tracking_objects[j].label_id == new_object.label_id:
                                self.tracking_objects[j].pose = new_object.pose
                                self.tracking_objects[j].bbox_min = new_object.bbox_min
                                self.tracking_objects[j].bbox_max = new_object.bbox_max
                                flag = 1
                        if flag==0:
                            self.tracking_objects.append(new_object)

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
            objects_in_connections = []
            for x in self.connections:
                obj_2 = self.object_maps[str(x.pair[0])]
                obj_1 = self.object_maps[str(x.pair[1])]
                self.G.add_edge(obj_1,obj_2,label='on')
                objects_in_connections.append(x.pair[0])
            objects_tracked = []
            for x in self.tracking_objects:
                objects_tracked.append(x.label_id)
            unknown_objects = set(objects_tracked).difference(objects_in_connections)
            if len(unknown_objects) > 0:
                self.G.add_edge("kitchen","unknown")
                for x in unknown_objects:
                    self.G.add_edge("unknown",self.object_maps[str(x)])
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
        rate = rospy.Rate(30.0)
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        pattern = [0,1,1,1,1,1,1,1,1,1]
        while not rospy.is_shutdown():
            print("Length of connections ", len(self.connections))
            try:
                trans = tfBuffer.lookup_transform('world', 'camera_color_optical_frame', rospy.Time.now(),rospy.Duration(1.0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                rate.sleep()
                continue
            for i in range(len(self.connections)):
                if rospy.Time.now().secs - self.connections[i].time_stamp < 150:
                    continue
                print("HELLO")
                object_in_world = PoseStamped()
                object_in_world.header.frame_id = 'world'
                object_in_world.pose.orientation.x = 0.0
                object_in_world.pose.orientation.y = 0.0
                object_in_world.pose.orientation.z = 0.0
                object_in_world.pose.orientation.w = 1.0
                object_in_world.header.stamp = rospy.Time(0)
                
                for j in range(len(self.tracking_objects)):
                    if self.tracking_objects[j].label_id == self.connections[i].pair[0]:
                        object_in_world.pose.position = self.tracking_objects[j].pose
                        break

                object_in_camera = tfBuffer.transform(object_in_world, "camera_color_optical_frame", timeout=rospy.Duration(10.0)) #point on camera_color_optical_frame
            
                if object_in_camera.pose.position.z < 0:
                    print("Negative z")

                yangle = np.arctan(object_in_camera.pose.position.y/object_in_camera.pose.position.z) #rads
                xangle = np.arctan(object_in_camera.pose.position.x/object_in_camera.pose.position.z) #rads

                fovx = 55.8 #horizontal field of view in deg
                fovy = 43.4 #vertical field of view in deg
                if object_in_camera.pose.position.z > 0:
                    if abs(xangle) < abs(fovx/2*math.pi/180) and abs(yangle) < abs(fovy/2*math.pi/180): #57/43
                        if len(self.connections[i].time_history) == 10:
                            self.connections[i].time_history.pop(0)
                        self.connections[i].time_history.append(1)
                        print("In my view")
                    else:
                        if len(self.connections[i].time_history) == 10:
                            self.connections[i].time_history.pop(0)
                        self.connections[i].time_history.append(0)
                    print(self.connections[i].time_history)
                    if self.connections[i].time_history == pattern:
                        self.connections[i].back_in_view_time = rospy.get_rostime().secs
                        self.connections[i].back_in_view = True
                        if ([int(x) for x in self.color_maps[str(self.connections[i].pair[0])]]==self.image).all(axis=2).sum() != 0:
                            self.connections[i].no_of_det = self.connections[i].no_of_det + 1
                    if self.connections[i].back_in_view:
                        if (rospy.get_rostime().secs - self.connections[i].back_in_view_time)<3:    
                        #print(([int(x) for x in self.color_maps[str(self.connections[i].pair[0])]]==self.image).all(axis=2).sum())
                            if ([int(x) for x in self.color_maps[str(self.connections[i].pair[0])]]==self.image).all(axis=2).sum() != 0:
                                self.connections[i].no_of_det = self.connections[i].no_of_det + 1
                            print("Number of detections ", self.connections[i].no_of_det)
                        else:
                            if self.connections[i].no_of_det < 2:
                                print(self.object_maps[str(self.connections[i].pair[0])], " is not on ",self.object_maps[str(self.connections[i].pair[1])])
                                print("REMOVEDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
                                del self.connections[i]
                                break

    
            self.visualize_scene_graphs(self.static_objects)
            rate.sleep()

    def read_pkl(self,file_name):
        with open('/home/aswingururaj/catkin_ws/src/orb_slam_2_ros/ros/scripts/'+file_name,'rb') as handle:
                count = dill.load(handle)
                for i in range(count):
                    obj = dill.load(handle)
                    self.static_objects.append(obj)
                count = dill.load(handle)
                for i in range(count):
                    obj = dill.load(handle)
                    self.tracking_objects.append(obj)
                count = dill.load(handle)
                for i in range(count):
                    obj = dill.load(handle)
                    old_connection = connection(obj.pair)
                    self.connections.append(old_connection)

    def handler(self,signum,frame):
        if len(self.connections)>0:
            with open('/home/aswingururaj/catkin_ws/src/orb_slam_2_ros/ros/scripts/updated_scene_graph.pkl', 'wb') as outp:
                    print("DUMPING")
                    dill.dump(len(self.static_objects),outp)
                    for i in range(len(self.static_objects)):
                        dill.dump(self.static_objects[i], outp)
                        print("Label : ",self.object_maps[str(self.static_objects[i].label_id)])
                        print("Pose : ",self.static_objects[i].pose)
                    dill.dump(len(self.tracking_objects),outp)
                    for i in range(len(self.tracking_objects)):
                        dill.dump(self.tracking_objects[i], outp)
                        print("Label : ",self.object_maps[str(self.tracking_objects[i].label_id)])
                        print("Pose : ",self.tracking_objects[i].pose)
                    dill.dump(len(self.connections),outp)
                    for i in range(len(self.connections)):
                        dill.dump(self.connections[i], outp)
        exit(1)
    
    def isPointAboveBbox(self,bbox_min,bbox_max,point):
        if (bbox_min.x <= point[0] <= bbox_max.x) and (bbox_min.y <= point[1] <= bbox_max.y ) and (0.5*bbox_max.z <= point[2] <= bbox_max.z + 2):
            return True

if __name__ == '__main__':
    rospy.init_node('update_scene_graph', anonymous=True)
    file_name = 'static_objects.pkl'
    update_scene_graph = update_scene_graph(file_name)
    update_scene_graph.start()