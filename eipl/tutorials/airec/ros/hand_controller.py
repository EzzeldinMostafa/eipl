#!/usr/bin/env python3

import numpy as np

# ROS
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory

class HandClient:
    def __init__(self, namespace, side):
        self.side = side

        TOPIC_NAME = namespace + "/" + self.side + '_hand_controller/command'
        self.publisher = rospy.Publisher(TOPIC_NAME, JointTrajectory, queue_size=1)
            
        self.hand_min  = np.deg2rad([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0])
        self.hand_max =  np.deg2rad([30.0, 48.0, 52.5, 34.0, 15.0, 18.0])
        self.scale = 0.05
        self.max_effort = 0.5

        # Create JointTrajectory message
        self.trajectory = self.__create_joint_trajectory_base()
        self.trajectory.joint_names = [
                self.side + '_hand_first_finger/joint_1',
                self.side + '_hand_second_finger/joint_1',
                self.side + '_hand_third_finger/joint_1',
                self.side + '_hand_thumb/joint_1',
                self.side + '_hand_thumb/joint_2',
                self.side + '_hand_thumb/joint_3'
                ]

        rospy.Subscriber('/torobo/joint_states', JointState, self.__joint_callback)
        print("wait for message")
        rospy.wait_for_message('/torobo/joint_states', JointState, timeout=None)

    def __joint_callback(self, msg):
        if self.side == "left":
            self.hand_t = np.array(msg.position)[np.array([17, 19, 21, 23, 24, 25])]
        elif self.side == "right":
            self.hand_t = np.array(msg.position)[np.array([34, 36, 38, 40, 41, 42])]

    def __command(self, pose):
        self.trajectory.header.stamp = rospy.Time.now()
        self.trajectory.points[0].positions = pose.tolist()
        self.trajectory.points[0].effort = [self.max_effort] * len(pose)
        self.publisher.publish(self.trajectory)

    def __create_joint_trajectory_base(self):
        # Creates a message.
        trajectory = JointTrajectory()
        trajectory.header.stamp = rospy.Time.now()
        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration(0)
        trajectory.points.append(point)

        return trajectory 
    
    def set_open(self):
        self.__command(self.hand_min)
        
    def set_close(self):
        self.__command(self.hand_max)