#!/usr/bin/env python3

import numpy as np

# ROS
import rospy
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory

class HandClient:
    def __init__(self, namespace, side):
        self.side = side
        self.hand_t = None

        TOPIC_NAME = namespace + "/" + self.side + '_hand_controller/command'
        self.publisher = rospy.Publisher(TOPIC_NAME, JointTrajectory, queue_size=1)
            
        self.hand_min  = np.deg2rad([ 0.0,  0.0,  0.0,  0.0,  0.0,  0.0])
        self.hand_max =  np.deg2rad([76.0, 16.10, 50.0, 45.9, 58.5, 52.5])
        self.scale = 0.05

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
        self.publisher.publish(self.trajectory)

    def __create_joint_trajectory_base(self):
        # Creates a message.
        trajectory = JointTrajectory()
        trajectory.header.stamp = rospy.Time.now()
        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Time(0)
        trajectory.points.append(point)

        return trajectory 
    
    def update(self, diff):
        if self.hand_t is None or diff == 0.0:
            return
        pose = np.clip(self.hand_t + diff * self.scale * (self.hand_max - self.hand_min), self.hand_min, self.hand_max)
        self.__command(pose)

if __name__ == "__main__":    

    rospy.init_node('hand_controller', anonymous=True, disable_signals=True)