#!/usr/bin/env python3  
import rospy
import math
import tf
from geometry_msgs.msg import Quaternion 
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_multiply
import numpy

if __name__ == '__main__':
    rospy.init_node('correct_tf')

    listener = tf.TransformListener()

    listener.waitForTransform("/map", "/odom", rospy.Time(), rospy.Duration(4.0))
    rate = rospy.Rate(15.0)
    # trans = [0.042, -0.004, 1.322] # where camera is supposed to be in the world frame (aux)
    # q = [-0.003, 0.707, -0.004, 0.707] # where camera is supposed to be in the world frame (aux)
    trans = [0.046, -0.003, 1.314]
    q = [0.001, 0.777, -0.001, 0.630]
    transinitialodom2cam_mat = tf.transformations.translation_matrix(trans)
    rotinitialodom2cam_mat   = tf.transformations.quaternion_matrix(q)
    tf_initialodom2cam = numpy.dot(transinitialodom2cam_mat, rotinitialodom2cam_mat)
    # trans = [-0.736, -0.032, 1.304]                  # Where camera is supposed to be in the world frame (map)
    # q = quaternion_from_euler(-3.123, 1.152, 2.316)  # Where camera is supposed to be in the world frame (map)
    # transinitialmap2cam_mat = tf.transformations.translation_matrix(trans)
    # rotinitialmap2cam_mat   = tf.transformations.quaternion_matrix(q)
    # tf_initialmap2cam = numpy.dot(transinitialmap2cam_mat, rotinitialmap2cam_mat)
    while not rospy.is_shutdown():
        try:
            (transaux2cam,rotaux2cam) = listener.lookupTransform('/map', '/odom', rospy.Time(0))
            # (transcam2tar,rotcam2tar) = listener.lookupTransform('/camera_color_optical_frame', '/odom', rospy.Time(0))
            # trans = [-0.736, -0.032, 1.304]                  # Where camera is supposed to be in the world frame (map)
            # q = quaternion_from_euler(-3.123, 1.152, 2.316)  # Where camera is supposed to be in the world frame (map)
            # transinitialmap2cam_mat = tf.transformations.translation_matrix(trans)
            # rotinitialmap2cam_mat   = tf.transformations.quaternion_matrix(q)
            # tf_initialmap2cam = numpy.dot(transinitialmap2cam_mat, rotinitialmap2cam_mat)
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("Nope")
            continue
        transaux2cam_mat = tf.transformations.translation_matrix(transaux2cam)
        rotaux2cam_mat   = tf.transformations.quaternion_matrix(rotaux2cam)
        tf_aux2cam = numpy.dot(transaux2cam_mat, rotaux2cam_mat)
        tf_cam2aux = tf.transformations.inverse_matrix(tf_aux2cam)

        tf_world2aux = numpy.dot(tf_initialodom2cam,tf_cam2aux)

        # transcam2tar_mat = tf.transformations.translation_matrix(transcam2tar)
        # rotcam2tar_mat   = tf.transformations.quaternion_matrix(rotcam2tar)
        # tf_cam2tar = numpy.dot(transcam2tar_mat, rotcam2tar_mat)
        
        # tf_map2tar = numpy.dot(tf_map2cam, tf_cam2tar)
        # trans3 = tf.transformations.translation_from_matrix(tf_map2tar)
        # rot3 = tf.transformations.translation_from_matrix(tf_map2tar)
        
        
        correct_world_rot = quaternion_from_euler(0,1.5708,0,'rxyz')
        final_rot = quaternion_from_euler(0,0,0,'rxyz')
        #tf_new = numpy.dot(tf_correction,tf_aux2cam)

        correct_trans = [0, 0, 0]
        correct_rot = quaternion_from_euler(0,0,0,'rxyz')
        correct_trans_mat = tf.transformations.translation_matrix(correct_trans)
        correct_rot_mat = tf.transformations.quaternion_matrix(correct_rot)
        tf_correction = numpy.dot(correct_trans_mat, correct_rot_mat)
        tf_world2aux = numpy.dot(tf_correction,tf_cam2aux)

        one_mat = tf.transformations.translation_matrix(trans)
        two_mat   = tf.transformations.quaternion_matrix(correct_world_rot)
        one = numpy.dot(one_mat, two_mat)

        three_mat = tf.transformations.translation_matrix(correct_trans)
        four_mat   = tf.transformations.quaternion_matrix(correct_rot)
        two = numpy.dot(three_mat, four_mat)

        stup = numpy.dot(one,two)
        new = tf.transformations.inverse_matrix(stup)
        final_new = numpy.dot(new,tf_initialodom2cam)
        # identity_trans = [0,0,0]
        # correct_q = quaternion_from_euler(1.57,0.3975,0)
        # #correct_q = quaternion_from_euler(0,0,0)
        # new_rot3 = quaternion_multiply(rot3,correct_q)
        # trans[0] = trans[0]+0.027
        # trans[1] = trans[1]+0.003 
        # trans[2] = trans[2]-0.010 
        # q_rot = quaternion_from_euler(3.120, 1.189, 3.099)
        # q_new = quaternion_multiply(rot, q_rot)
        # identity_trans = [0,0,0]
        # identity_rot = quaternion_from_euler(1.57,0.3975,0)
        # final_rot = quaternion_multiply(identity_rot,q_new)
        br = tf.TransformBroadcaster()
        # br.sendTransform(trans3,
        #                  new_rot3,
        #                  rospy.Time.now(),
        #                  "test",
        #                  "world")
        # br.sendTransform(trans3,
        #                  rot3,
        #                  rospy.Time.now(),
        #                  "odom",
        #                  "world")
        # br.sendTransform(trans3,
        #                  rot3,
        #                  rospy.Time.now(),
        #                  "odom",
        #                  "test")
        br.sendTransform(
                        # tf.transformations.translation_from_matrix(tf_cam2aux),
                        #  tf.transformations.quaternion_from_matrix(tf_cam2aux),
                         tf.transformations.translation_from_matrix(tf_world2aux),
                         tf.transformations.quaternion_from_matrix(tf_world2aux),
                         rospy.Time.now(),
                         "map",
                         "world")

        rate.sleep()