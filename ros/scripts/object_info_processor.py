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
    def __init__(self,pair,removed,pose,bbox_min,bbox_max,ref):
        self.pair = pair
        self.removed = removed
        self.pose = pose
        self.time_history = [0]*10
        self.prior = [0.5,0.5] # number of classes to consider
        self.back_in_view = False
        self.no_of_det = 0
        self.back_in_view_time = 0.0
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.ref = ref
    
    def return_pair(self):
        return self.pair

class object_processor:
    def __init__(self,file_name):
        self.br = CvBridge()
        path = "/home/aswingururaj/catkin_ws/src/kimera_semantics/kimera_semantics_ros/cfg/mask_rcnn_mapping.csv"
        self.object_maps = {}
        self.color_maps = {}
        self.image = None
        self.flag = 0
        self.pose_set = False
        self.removed = False
        self.connections = []
        self.prior_connections = []
        self.prior_connections_set = False
        if file_name is None:
            signal.signal(signal.SIGINT, self.handler)
        else:
            self.read_pkl(file_name)
            self.prior_connections_set = True
            self.pose_set = True
        with open(path, newline='', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            next(reader,None)
            for row in reader:
                self.object_maps[row['id']] = row['name']
                self.color_maps[row['id']] = [row['blue'],row['green'],row['red']]
        #print(type(self.color_maps[str(19)]))
        self.sub = rospy.Subscriber("/incremental_dsg_builder_node/object_info", ObjectInfo, self.callback)
        self.sub = rospy.Subscriber("/sparseinst_seg", Image, self.image_callback)

    def callback(self,data):
        if len(data.labels) < 2:
            return
        centroids = []
        for i in range(len(data.labels)):
            centroids.append([(data.bbox_min[i].x + data.bbox_max[i].x)/2, (data.bbox_min[i].y + data.bbox_max[i].y)/2, (data.bbox_min[i].z + data.bbox_max[i].z)/2])
        if not self.prior_connections_set:
            for i in range(len(data.labels)):
                for j in range(len(data.labels)):
                    if self.isPointAboveBbox(data.bbox_min[i],data.bbox_max[i],centroids[j]) and i!=j:
                        if data.labels[j]==1:
                            ref = 0.0018435
                        if data.labels[j]==2:
                            ref = 0.009963
                        new_connection = connection([data.labels[j],data.labels[i]],False,data.positions[j],data.bbox_min[j],data.bbox_max[j],ref)
                        #print("Volume ", self.findVolume(data.bbox_min[j],data.bbox_max[j]))
                        #print("Pairs",new_connection.pair)
                        count = 0
                        for x in self.connections:
                            if x.pair != [data.labels[j],data.labels[i]] and not x.removed:
                                count = count + 1
                            if x.pair == [data.labels[j],data.labels[i]] and not x.removed:
                                x.bbox_min = data.bbox_min[j]
                                x.bbox_max = data.bbox_max[j]
                                #print(x)
                        if count == len(self.connections):
                            print("Adding new connection ", new_connection.pair)
                            self.connections.append(new_connection)
                            #self.pose_set = True
        else:
            for i in range(len(data.labels)):
                for j in range(len(data.labels)):
                    if self.isPointAboveBbox(data.bbox_min[i],data.bbox_max[i],centroids[j]) and i!=j:
                        new_connection = connection([data.labels[j],data.labels[i]],False,data.positions[j])
                        #print("Pairs",new_connection.pair)
                        count = 0
                        replaced = False
                        # for x in self.prior_connections:
                        #     if x.pair[0] != data.labels[j] and x.pair[1] == data.labels[i] and not x.removed:
                        #         x.pair[0] = data.labels[j]
                        #         replaced = True
                        for y in self.prior_connections:
                            if y.pair != [data.labels[j],data.labels[i]] and not y.removed:
                                count = count + 1
                                #print(x)
                        if count == len(self.prior_connections):
                            print("Adding new connection ", new_connection.pair)
                            self.prior_connections.append(new_connection)
                            #self.pose_set = True
                        for index,v in enumerate(self.prior_connections):
                            if v.pair[0]==data.labels[j] and v.pair[1]!=data.labels[i]:
                                del self.prior_connections[index]
                                print("HELLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
                                break

    def visualize_scene_graphs(self,connections):
        if len(connections)>0:
            layout = {}
            self.G = nx.DiGraph()
            i = 0
            for x in connections:
                if not x.removed:
                    obj_2 = self.object_maps[str(x.pair[0])]#+ "\n" + str(data.header.stamp.to_sec())
                    obj_1 = self.object_maps[str(x.pair[1])]#+ "\n" + str(data.header.stamp.to_sec())
                    #self.G.add_edge("Kitchen",obj_2)
                    self.G.add_edge(obj_2,obj_1,label='on')
                    layout[obj_1] = [1+(5*i),11]
                    layout[obj_2] = [1+(5*i),6]
                    #layout["Kitchen"] = [1,1]
                    i = i + 1
            # if(len(self.furniture)>0):
            #     print("Hello")
            #     for i in self.furniture:
            #         rem = self.object_maps[str(i)]
            #         print(rem)
            #         self.G.add_edge("Kitchen",rem)
            #         layout[rem] = [11,6]
            # if(len(self.object)>0):
            #     for i in self.object:
            #         rem = self.object_maps[str(i)]
            #         self.G.add_edge(rem,"unknown")
            #         layout[rem] = [11,11]
            #         layout["unknown"] = [11,20]
            #pos = nx.spring_layout(self.G)
            nx.draw(
            self.G, pos=layout, edge_color='black', width=1, linewidths=1,
            node_size=10000, node_color='red', alpha=0.9,
            labels={node: node for node in self.G.nodes()}
            )
            edge_list = {((u,v) for u, v, d in self.G.edges(data=True)):'on'}
            nx.draw_networkx_edge_labels(
                self.G, pos=layout,edge_labels=nx.get_edge_attributes(self.G,'label'),
                #edge_labels={edge for edge in edge_list: 'on'},
                font_color='black'
            )
            plt.axis('off')
            plt.pause(1e-10)
            plt.clf()

    def image_callback(self,msg):
        self.image = self.br.imgmsg_to_cv2(msg,desired_encoding='bgr8')
    
    def read_pkl(self,file_name):
        with open('/home/aswingururaj/catkin_ws/src/orb_slam_2_ros/ros/scripts/'+file_name,'rb') as handle:
            count = dill.load(handle)
            for i in range(count):
                obj = dill.load(handle)
                obj.time_history = [0]*10
                self.prior_connections.append(obj)
    
    def handler(self,signum,frame):
        if len(self.connections)>0:
            with open('/home/aswingururaj/catkin_ws/src/orb_slam_2_ros/ros/scripts/connections_data.pkl', 'wb') as outp:
                    print("DUMPING")
                    dill.dump(len(self.connections),outp)
                    for i in range(len(self.connections)):
                        dill.dump(self.connections[i], outp)
        exit(1)

    def start(self):        
        rate = rospy.Rate(15.0)
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        # listener.waitForTransform('camera_color_optical_frame', 'base_link', rospy.Time(0), rospy.Duration(4.0))
        pattern = [0,1,1,1,1,1,1,1,1,1]
        while not rospy.is_shutdown():
            # print("Timer callback called")
            # try:
            #     trans = tfBuffer.lookupTransform('camera_color_optical_frame', 'world', rospy.Time(0))
            # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            #     print("Nope")
            #     continue
            if not self.prior_connections_set:
                print("Length of connections ", len(self.connections))
                if len(self.connections) == 0:
                    print("No connections to show")
                for i in range(len(self.connections)):
                    # if not self.connections[i].removed:
                    #     print(self.object_maps[str(self.connections[i].pair[0])]," is on top of ", self.object_maps[str(self.connections[i].pair[1])])
                    # else:
                    #     print(self.object_maps[str(self.connections[i].pair[0])]," is not on top of ", self.object_maps[str(self.connections[i].pair[1])])
                    object_in_world = PoseStamped()
                    object_in_world.header.frame_id = 'world'
                    object_in_world.pose.orientation.x = 0.0
                    object_in_world.pose.orientation.y = 0.0
                    object_in_world.pose.orientation.z = 0.0
                    object_in_world.pose.orientation.w = 1.0
                    object_in_world.header.stamp = rospy.Time(0)
                    object_in_world.pose.position = self.connections[i].pose
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
                            #print("In my view")
                        else:
                            if len(self.connections[i].time_history) == 10:
                                self.connections[i].time_history.pop(0)
                            self.connections[i].time_history.append(0)
                        if self.connections[i].time_history == pattern:
                            self.connections[i].back_in_view_time = rospy.get_rostime().secs
                            self.connections[i].back_in_view = True
                            if ([int(x) for x in self.color_maps[str(self.connections[i].pair[0])]]==self.image).all(axis=2).sum() != 0:
                                self.connections[i].no_of_det = self.connections[i].no_of_det + 1
                            self.connections[i].time_history.pop(0)
                            self.connections[i].time_history.append(1)
                        if self.connections[i].time_history != [1]*10:
                            self.connections[i].back_in_view = False
                        if self.connections[i].back_in_view:
                            if (rospy.get_rostime().secs - self.connections[i].back_in_view_time)<2:    
                            #print(([int(x) for x in self.color_maps[str(self.connections[i].pair[0])]]==self.image).all(axis=2).sum())
                                if ([int(x) for x in self.color_maps[str(self.connections[i].pair[0])]]==self.image).all(axis=2).sum() != 0:
                                    self.connections[i].no_of_det = self.connections[i].no_of_det + 1
                                #print("Number of detections ", self.connections[i].no_of_det)
                            else:
                                if self.connections[i].no_of_det < 2:
                                    print(self.object_maps[str(self.connections[i].pair[0])], " is not on ",self.object_maps[str(self.connections[i].pair[1])])
                                    # print("Removed a connection")
                                    # self.connections[i].removed = True
                                else:
                                    vol = self.findVolume(self.connections[i].bbox_min,self.connections[i].bbox_max)
                                    print("vol ", vol)
                                    temp = self.distance_observation(object_in_camera.pose.position.z)*self.Gaussian(yangle,0,10)
                                    if self.connections[i].pair[0] == 2:
                                        thresh = 0.8
                                    else:
                                        thresh = 0.2
                                    for k in range(2):
                                        if k==0:
                                            self.connections[i].prior[k] = temp*thresh*self.connections[i].prior[k]*self.Gaussian(vol,0.068,0.01)
                                        else:
                                            self.connections[i].prior[k] = temp*(1-thresh)*self.connections[i].prior[k]*self.Gaussian(vol,0.022,0.001)
                                    
                                    self.connections[i].prior[0] = self.connections[i].prior[0]/sum(self.connections[i].prior)
                                    self.connections[i].prior[1] = self.connections[i].prior[1]/sum(self.connections[i].prior)
                                    print("Prior ", self.connections[i].prior[0], " ", self.connections[i].prior[1])
                    

                self.visualize_scene_graphs(self.connections)
            else:
                print("Length of connections ", len(self.prior_connections))
                if len(self.prior_connections) == 0:
                    print("No connections to show")
                for i in range(len(self.prior_connections)):
                    if not self.prior_connections[i].removed:
                        print(self.object_maps[str(self.prior_connections[i].pair[0])]," is on top of ", self.object_maps[str(self.prior_connections[i].pair[1])])
                    else:
                        print(self.object_maps[str(self.prior_connections[i].pair[0])]," is not on top of ", self.object_maps[str(self.prior_connections[i].pair[1])])
                self.visualize_scene_graphs(self.prior_connections)
            
            if self.pose_set:
                # print("Length of connections ", len(self.connections))
                # if len(self.connections) == 0:
                #     print("No connections to show")
                for i in range(len(self.prior_connections)):
                    # if not self.connections[i].removed:
                    #     print(self.object_maps[str(self.connections[i].pair[0])]," is on top of ", self.object_maps[str(self.connections[i].pair[1])])
                    # else:
                    #     print(self.object_maps[str(self.connections[i].pair[0])]," is not on top of ", self.object_maps[str(self.connections[i].pair[1])])
                    object_in_world = PoseStamped()
                    object_in_world.header.frame_id = 'world'
                    object_in_world.pose.orientation.x = 0.0
                    object_in_world.pose.orientation.y = 0.0
                    object_in_world.pose.orientation.z = 0.0
                    object_in_world.pose.orientation.w = 1.0
                    object_in_world.header.stamp = rospy.Time(0)
                    # print("Index value ", i)
                    print(" length of prior ",len(self.prior_connections))
                    object_in_world.pose.position = self.prior_connections[i].pose
                
                    object_in_camera = tfBuffer.transform(object_in_world, "camera_color_optical_frame", timeout=rospy.Duration(10.0)) #point on camera_color_optical_frame
            
                    if object_in_camera.pose.position.z < 0:
                        print("Negative z")

                    yangle = np.arctan(object_in_camera.pose.position.y/object_in_camera.pose.position.z) #rads
                    xangle = np.arctan(object_in_camera.pose.position.x/object_in_camera.pose.position.z) #rads

                    fovx = 55.8 #horizontal field of view in deg
                    fovy = 43.4 #vertical field of view in deg
                    if object_in_camera.pose.position.z > 0:
                        if abs(xangle) < abs(fovx/2*math.pi/180) and abs(yangle) < abs(fovy/2*math.pi/180): #57/43
                            if len(self.prior_connections[i].time_history) == 10:
                                self.prior_connections[i].time_history.pop(0)
                            self.prior_connections[i].time_history.append(1)
                            #print("In my view")
                        else:
                            if len(self.prior_connections[i].time_history) == 10:
                                self.prior_connections[i].time_history.pop(0)
                            self.prior_connections[i].time_history.append(0)
                        print(self.prior_connections[i].time_history)
                        if self.prior_connections[i].time_history == pattern:
                            self.prior_connections[i].back_in_view_time = rospy.get_rostime().secs
                            self.prior_connections[i].back_in_view = True
                            if ([int(x) for x in self.color_maps[str(self.prior_connections[i].pair[0])]]==self.image).all(axis=2).sum() != 0:
                                self.prior_connections[i].no_of_det = self.prior_connections[i].no_of_det + 1
                        if self.prior_connections[i].back_in_view:
                            if (rospy.get_rostime().secs - self.prior_connections[i].back_in_view_time)<3:    
                            #print(([int(x) for x in self.color_maps[str(self.connections[i].pair[0])]]==self.image).all(axis=2).sum())
                                if ([int(x) for x in self.color_maps[str(self.prior_connections[i].pair[0])]]==self.image).all(axis=2).sum() != 0:
                                    self.prior_connections[i].no_of_det = self.prior_connections[i].no_of_det + 1
                                print("Number of detections ", self.prior_connections[i].no_of_det)
                            else:
                                if self.prior_connections[i].no_of_det < 2:
                                    print(self.object_maps[str(self.prior_connections[i].pair[0])], " is not on ",self.object_maps[str(self.prior_connections[i].pair[1])])
                                    print("Removed a connection")
                                    self.prior_connections[i].removed = True
                                    print("REMOVEDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
                                    del self.prior_connections[i]
                                    break

                                # self.prior_connections[i].back_in_view = False
                                # self.prior_connections[i].back_in_view_time = 0.0
                                # self.prior_connections[i].no_of_det = 0
                    # for index,value in enumerate(self.prior_connections):
                    #     if value.removed:
                    #         del self.prior_connections[index]
            # if self.prior_connections_set:
            #     self.visualize_scene_graphs(self.prior_connections)25
            # else:
            #     self.visualize_scene_graphs(self.connections)
            
            rate.sleep()

    def isPointAboveBbox(self,bbox_min,bbox_max,point):
        if (bbox_min.x <= point[0] <= bbox_max.x) and (bbox_min.y <= point[1] <= bbox_max.y) and (0.8*bbox_max.z <= point[2] <= bbox_max.z + 2):
            return True

    def findVolume(self,bbox_min,bbox_max):
        return (bbox_max.x-bbox_min.x)*(bbox_max.y-bbox_min.y)*(bbox_max.z-bbox_min.z)
    
    def distance_observation(self,z):
        if (0.3 <= z <= 3):
            return (10/27.0)
        else:
            return 0
    
    def Gaussian(self, x, mu=0.0, sigma=1.0):
        return np.exp(-0.5*(x-mu)**2/sigma**2)/np.sqrt(2*np.pi*sigma**2)


if __name__ == '__main__':
    rospy.init_node('object_info_processor', anonymous=True)
    # argParser = argparse.ArgumentParser()
    # argParser.add_argument("--file-name", help="name of the pkl file",action='store_true')
    # args = argParser.parse_args()
    # print(args.file_name)
    file_name = 'connections_data.pkl'
    file_name = None
    object_processor = object_processor(file_name)
    object_processor.start()