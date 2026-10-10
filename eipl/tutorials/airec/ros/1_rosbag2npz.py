#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import os
import cv2
import glob
import rospy
import rosbag
import argparse
import numpy as np
from cv_bridge import CvBridge


parser = argparse.ArgumentParser()
parser.add_argument("--bag_dir", type=str)
parser.add_argument("--freq", type=float, default=10)
args = parser.parse_args()
bridge = CvBridge()


files = glob.glob(os.path.join(args.bag_dir, "*.bag"))
files.sort()
for file in files:
    print(file)
    savename = file.split(".bag")[0] + ".npz"

    # Open the rosbag file
    bag = rosbag.Bag(file)

    # Get the start and end times of the rosbag file
    start_time = bag.get_start_time()
    end_time = bag.get_end_time()

    # Get the topics in the rosbag file
    # topics = bag.get_type_and_topic_info()[1].keys()
    topics = [
        "/torobo/joint_states",
        "/torobo/right_hand_controller/state",
        #"/torobo/head/sr300/camera/aligned_depth_to_color/image_raw/compressed",
        #"/torobo/head/sr300/camera/depth/image_rect_raw",
        "/torobo/head/sr300/camera/color/image_rect_color/compressed",
        
    ]

    # Create a rospy.Time object to represent the current time
    current_time = rospy.Time.from_sec(start_time)

    joint_list = []
    finger_list = []
    image_list = []
    depth_list = []
    finger_state_list = []

    prev_finger = None
    finger_state = 0

    # Loop through the rosbag file at regular intervals (args.freq)
    freq = 1.0 / float(args.freq)
    while current_time.to_sec() < end_time:
        print(current_time.to_sec())

        # Get the messages for each topic at the current time
        for topic in topics:
            print(f"topic{topic}")
            for topic_msg, msg, time in bag.read_messages(
                topic, start_time=current_time
            ):
                if time >= current_time:
                    if topic == "/torobo/joint_states":
                        joint_indices = [
                            msg.name.index(f"right_arm/joint_{i}")
                            for i in range(1, 8)
                        ]
                        joint_list.append([msg.position[i] for i in joint_indices])
                    if (
                        topic
                        == "/torobo/head/sr300/camera/color/image_rect_color/compressed"
                    ):
                        np_img = bridge.compressed_imgmsg_to_cv2(
                            msg, desired_encoding="bgr8"
                        )                        
                        image_list.append(np_img.astype(np.uint8))
                    if topic == "/torobo/head/sr300/camera/depth/image_rect_raw":
                        np_depth = bridge.imgmsg_to_cv2(
                            msg, desired_encoding="passthrough"
                        ) 
                        depth_list.append(np_depth.copy())
                    if topic == "/torobo/right_hand_controller/state":
                        finger_index = msg.joint_names.index("right_hand_first_finger/joint_1")
                        finger = np.array(msg.desired.positions[finger_index])
                        if prev_finger is None:
                            prev_finger = finger

                        if finger - prev_finger > 0.005 and finger_state == 0:
                            finger_state = 1
                        elif prev_finger - finger > 0.005 and finger_state == 1:
                            finger_state = 0
                        prev_finger = finger

                        finger_list.append(finger)
                        finger_state_list.append(finger_state)

                    break

        # Wait for the next interval
        current_time += rospy.Duration.from_sec(freq)

    # Close the rosbag file
    bag.close()

    # Convert list to array
    joints = np.array(joint_list, dtype=np.float32)
    finger = np.array(finger_list, dtype=np.float32)
    finger_state = np.array(finger_state_list, dtype=np.float32)
    images = np.array(image_list, dtype=np.uint8)
    depths = np.array(depth_list)
    # Get shorter lenght
    shorter_length = min(len(joints), len(images), len(finger), len(finger_state))

    print(f"images shape is {images.shape}")
    print(f"joints shape is {joints.shape}")
    print(f"finger shape is {finger.shape}")
    print(f"depths shape is {depths.shape}")
    print(f"finger_state shape is {finger_state.shape}")
    
    # Trim
    joints = joints[:shorter_length]
    finger = finger[:shorter_length]
    images = images[:shorter_length]
    depths = depths[:shorter_length]
    finger_state = finger_state[:shorter_length]

    # Save
    np.savez(
        savename, joints=joints, finger=finger, finger_state=finger_state, images=images, depths =depths,
    )
