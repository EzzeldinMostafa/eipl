#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch
import argparse
import numpy as np
import numpy.core as _core

import cv2
import json
import sys
import importlib
# Alias every numpy.core submodule under the numpy._core name
for name in ["multiarray", "numeric", "umath", "fromnumeric", "_internal",
             "_multiarray_umath", "_methods", "records", "shape_base"]:
    try:
        sys.modules[f"numpy._core.{name}"] = importlib.import_module(f"numpy.core.{name}")
    except ImportError:
        pass
sys.modules["numpy._core"] = _core

from typing import List

# ROS packages
import rospy
from sensor_msgs.msg import Image, JointState
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory
from cv_bridge import CvBridge

from eipl.model.SARNN import SARNN
from eipl.utils.data import normalization
from eipl.utils.data import tensor2numpy, deprocess_img
from eipl.data import LocalSampleDownloader
from eipl.utils import resize_img
from hand_controller import HandClient

# GLOBAL VARS
bridge = CvBridge()
TOPIC_NAME_position = '/torobo/online_joint_trajectory_controller/command'
TOPIC_NAME_image = '/torobo/head/sr300/camera/color/image_rect_color'
TOPIC_NAME_joint_state = '/torobo/joint_states'
TOPIC_NAME_hand_state = '/torobo/right_hand_controller/state'

JOINTS = np.array([])
LAST_HAND_STATE = 0
LAST_HAND_VALUE = None
HAND_VALUE = None
IMAGE_RGB = None

JOINT_NAMES = [
    f"right_arm/joint_{i}" for i in range(1,8)
]
HAND_NAME = 'right_hand_first_finger/joint_1'



def publish_joint_trajectory(publisher: rospy.Publisher, joint_names: List[str], positions:List[float], time_from_start:float):

    # Creates a message.
    trajectory = JointTrajectory()
    trajectory.header.stamp = rospy.Time.now()
    trajectory.joint_names = joint_names
    point = JointTrajectoryPoint()
    point.positions = positions
    point.velocities = [0.0] * len(joint_names)
    point.accelerations = [0.0] * len(joint_names)
    point.effort = [0.0] * len(joint_names)
    point.time_from_start = rospy.Duration(time_from_start)
    trajectory.points.append(point)

    # Publish the message.
    publisher.publish(trajectory)

def publish_position(q_target, position_publisher, hand_pub):
    msg = JointTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.joint_names = JOINT_NAMES

    point = JointTrajectoryPoint()
    point.positions = q_target[:-1].tolist()
    point.velocities = [0.0] * len(JOINT_NAMES)
    point.accelerations = [0.0] * len(JOINT_NAMES)
    point.effort = [0.0] * len(JOINT_NAMES)
    point.time_from_start = rospy.Duration(0,0)
    msg.points.append(point)
    position_publisher.publish(msg)
    rospy.loginfo("Published JointTrajectoryPoint to robot")
    print(msg)
    
    # Publish hand as well, simple threshold to set hand pose
    if q_target[-1] > 0.1:
        hand_pub.update(1)
        rospy.loginfo("Hand should be closing")
    else:
        hand_pub.update(-1)
        rospy.loginfo("Hand should be opening")

def rosimg_to_bgr(msg: Image) -> np.ndarray:
    return bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

def check_cuda():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            props = torch.cuda.get_device_properties(0)
            if getattr(props, "major", 0) >= 8:
                torch.backends.cudnn.conv.fp32_precision = "tf32"
                torch.backends.cuda.matmul.fp32_precision = "tf32"
        except Exception:
            pass

def compute_hand_state(current_joint_value):
    global LAST_HAND_STATE
    global LAST_HAND_VALUE
    if LAST_HAND_VALUE is None:
        LAST_HAND_VALUE = current_joint_value

    if current_joint_value - LAST_HAND_VALUE > 0.005 and LAST_HAND_STATE == 0:
        LAST_HAND_STATE = 1
    elif LAST_HAND_VALUE - current_joint_value > 0.005 and LAST_HAND_STATE == 1:
        LAST_HAND_STATE = 0
    LAST_HAND_VALUE = current_joint_value

def angle_sensor_callback(msg):
    global JOINTS
    global LAST_HAND_STATE

    joint_map = dict(zip(msg.name, msg.position))
    if all([True for key in list(joint_map.keys()) if key in JOINT_NAMES]):
        rospy.logdebug("All joints received and available")
        JOINTS = np.array([joint_map[name] for name in JOINT_NAMES])      
        assert(JOINTS.size == 7)
        return
    rospy.logdebug(msg.position)
    rospy.logdebug(msg)
    rospy.logwarn("Unexpected message format received in joint callback")

def hand_state_callback(msg):
    global LAST_HAND_VALUE
    global HAND_VALUE
    finger_index = msg.joint_names.index(HAND_NAME)
    HAND_VALUE = np.array(msg.desired.positions[finger_index])
    compute_hand_state(HAND_VALUE)  # updates LAST_HAND_STATE


def camera_right_callback(msg):
    global IMAGE_RGB
    try:
        IMAGE_RGB = rosimg_to_bgr(msg)
        if IMAGE_RGB is None:
            rospy.logerr("Failed to capture image from camera.")

    except Exception as err:
        rospy.logerr(err)


def run_offline_inference(model, images, joint_bounds, minmax, position_publisher, step_by_step=False, hand_publisher=None):
    global JOINTS
    global IMAGE_RGB
    global LAST_HAND_STATE
    global LAST_HAND_VALUE
    global HAND_VALUE
    
    rospy.loginfo("Starting inference in offline mode, using images from stored data")
    _, joint_list = [], []
    state = None 
    r = rospy.Rate(5)

    # Inference
    nloop = len(images)
    for loop_ct in range(nloop):
        # load data and normalization
        img_t = images[loop_ct].transpose(2, 0, 1)
        img_t = torch.Tensor(np.expand_dims(img_t, 0))
        img_t = normalization(img_t, (0, 255), minmax)
        #joint_t = torch.Tensor(np.expand_dims(joints[loop_ct], 0))
        #joint_t = normalization(joint_t, joint_bounds, minmax)
        while JOINTS.size == 0 and HAND_VALUE is None:
            rospy.logwarn(f"Waiting for joint state informatio from {TOPIC_NAME_joint_state} and {TOPIC_NAME_hand_state}")
            rospy.sleep(0.1)
        assert(LAST_HAND_STATE == 0 or LAST_HAND_STATE == 1)
        joints_and_hand = np.append(JOINTS, LAST_HAND_STATE)
        _joint_t = torch.Tensor(np.expand_dims(joints_and_hand, 0))
        joint_t = normalization(_joint_t, joint_bounds, minmax)

        # predict rnn
        print(f"input joints {joint_t}")
        y_image, y_joint, _enc_pts, _dec_pts, state = model.forward(img_t, joint_t, state)

        # denormalization
        pred_image = tensor2numpy(y_image[0])
        pred_image = deprocess_img(pred_image, minmax[0], minmax[1])
        pred_image = pred_image.transpose(1, 2, 0)
        pred_joint = tensor2numpy(y_joint[0]) #first prediction in time
        print(f"Predicted value hand not normalized {pred_joint[-1]}")
        pred_joint = normalization(pred_joint, minmax, joint_bounds)
    
        if loop_ct:
            if(pred_joint.shape!=(8,)) or not np.all(np.isfinite(pred_joint)):
                rospy.logerr("Invalid EIPL prediction")
                raise RuntimeError(f"pred_joints invalid: {pred_joint}")
            # conservative example: do not allow jumps
            max_step = 0.02
            print(f"Predicted value hand {pred_joint[-1]}")
            q_safe = JOINTS + np.clip(pred_joint[:-1]-JOINTS, -max_step, max_step)
            q_safe= np.append(q_safe, pred_joint[-1])
            rospy.loginfo(f"Clipped joint values delta within range {max_step} to\n {q_safe}")
            cv2.imshow("predicted image", pred_image)
            cv2.waitKey(1)
            if step_by_step:  
                input("Send to robot? <Press Enter>")
            publish_position(q_safe, position_publisher=position_publisher, hand_pub=hand_publisher)

            
            r.sleep()
    JOINTS = np.array([])
    HAND_VALUE = None


def run_online_inference(model, joint_bounds, minmax, position_publisher, img_resize, step_by_step=False, hand_publisher=None):
    global JOINTS
    global IMAGE_RGB
    global LAST_HAND_STATE
    global LAST_HAND_VALUE
    global HAND_VALUE
    
    rospy.loginfo("Starting inference in ONLINE mode: DANGER!!!")
    state = None 
    r = rospy.Rate(1)

    while not rospy.is_shutdown():
        # load data and normalization
        if IMAGE_RGB is None: 
            rospy.logwarn(f"Wait for image from {TOPIC_NAME_image}")
            continue
        
        img_t = cv2.resize(IMAGE_RGB, (img_resize, img_resize))
        img_t = img_t.transpose(2, 0, 1)
        img_t = torch.Tensor(np.expand_dims(img_t, 0))
        img_t = normalization(img_t, (0, 255), minmax)
        #joint_t = torch.Tensor(np.expand_dims(joints[loop_ct], 0))
        #joint_t = normalization(joint_t, joint_bounds, minmax)
        while JOINTS.size == 0 and HAND_VALUE is None:
            rospy.logwarn(f"Waiting for joint state information from {TOPIC_NAME_joint_state} and {TOPIC_NAME_hand_state}")
            rospy.sleep(0.1)
        assert(LAST_HAND_STATE == 0 or LAST_HAND_STATE == 1)
        joints_and_hand = np.append(JOINTS, LAST_HAND_STATE)
        _joint_t = torch.Tensor(np.expand_dims(joints_and_hand, 0))
        joint_t = normalization(_joint_t, joint_bounds, minmax)
        # predict rnn
        y_image, y_joint, _enc_pts, _dec_pts, state = model.forward(img_t, joint_t, state)

        # denormalization
        pred_image = tensor2numpy(y_image[0])
        pred_image = deprocess_img(pred_image, minmax[0], minmax[1])
        pred_image = pred_image.transpose(1, 2, 0)
        pred_joint = tensor2numpy(y_joint[0])
        pred_joint = normalization(pred_joint, minmax, joint_bounds)
    
        if(pred_joint.shape!=(8,)) or not np.all(np.isfinite(pred_joint)):
                rospy.logerr("Invalid EIPL prediction")
                raise RuntimeError(f"pred_joints invalid: {pred_joint}")
        # conservative example: do not allow jumps
        max_step = 0.02
        print(f"Predicted value hand {pred_joint[-1]}")
        q_safe = JOINTS + np.clip(pred_joint[:-1]-JOINTS, -max_step, max_step)
        q_safe= np.append(q_safe, pred_joint[-1])
        rospy.loginfo(f"Clipped joint values delta within range {max_step} to\n {q_safe}")
        cv2.imshow("predicted image", pred_image)
        cv2.waitKey(1)
        if step_by_step:  
            input("Send to robot? <Press Enter>")
        publish_position(q_safe, position_publisher=position_publisher, hand_pub=hand_publisher)

        # Reset global vars
        JOINTS = np.array([])
        IMAGE_RGB = None
        HAND_VALUE = None
        r.sleep()
  
def main():
    # argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to EIPL checkpoint")
    parser.add_argument("--idx", type=int, default=0, help="Select the test data sequence")
    parser.add_argument("--img-size", type=int, default=128, help="Image size :D ")
    parser.add_argument("--joint-dims", type=int, default=8, help="Number of joints of the robot")
    parser.add_argument("--offline", default=False, action="store_true", help="If set to true a path to offline recorded data is needed and used instead of real camera data.")
    parser.add_argument("--debug", default=False, action="store_true", help="Press enter before sending data to robot.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to offline data")
    args = parser.parse_args()

    try:
        check_cuda()
    except Exception as e:
        print(e)
        exit(-1)


    # EIPL INFERENCE SCRIPT
    # restore parameters
    dir_name = os.path.split(args.checkpoint_path)[0] # get parent path of checkpoint file
    hyperparam_filename = os.path.join(dir_name, "args.json") # data.json is generated when training
    with open(hyperparam_filename) as f:
        params = json.load(f) 


    # Load model
    model_sarnn = SARNN(
        rec_dim=params["rec_dim"],
        joint_dim=args.joint_dims,
        k_dim=params["k_dim"],
        heatmap_size=params["heatmap_size"],
        temperature=params["temperature"],
        im_size=[args.img_size, args.img_size]
    )
    minmax = [params["vmin"], params["vmax"]]

    # load weight
    print("Loading model to CPU")
    ckpt = torch.load(args.checkpoint_path, map_location=torch.device("cpu"), weights_only=False) # CPU
    model_sarnn.load_state_dict(ckpt["model_state_dict"])
    model_sarnn.eval() 
    model_sarnn.to("cpu")   
    print("Loaded model") 

    # ROS Setup
    try:
        rospy.init_node('eipl_inference_node', anonymous=False)
        rospy.Subscriber(TOPIC_NAME_image, Image, camera_right_callback)
        rospy.Subscriber(TOPIC_NAME_joint_state, JointState, angle_sensor_callback)
        rospy.Subscriber(TOPIC_NAME_hand_state, JointTrajectoryControllerState, hand_state_callback)
        pub_pos = rospy.Publisher(TOPIC_NAME_position, JointTrajectory, queue_size=10)
        hand_client = HandClient("/torobo", "right")

    except rospy.ROSInterruptException as e:
        print(e)
        raise RuntimeError("Could not setup ROS env")

    while IMAGE_RGB is None:
        # wait for next image from AIREC
        rospy.logwarn(f"Waiting for first image from {TOPIC_NAME_image}")
        rospy.sleep(0.5)
    rospy.loginfo("Received first image")

    if args.offline:
        # OFFLINE mode
        if not args.data_path:
            raise RuntimeError("Path to offline dataset is needed in offline mode.")
        
        # load dataset
        grasp_data = LocalSampleDownloader(data_path=args.data_path, img_format="HWC")
        _images, _joints, _masks, _poses = grasp_data.load_raw_data("test")
        _images = resize_img(_images, (args.img_size, args.img_size))
        images = _images[args.idx]
        _joints = _joints[args.idx]
        masks = _masks[args.idx]
        poses = _poses[args.idx]
        joint_bounds = grasp_data._load_bounds()
        print("Enter offline inference")
        run_offline_inference(model=model_sarnn, images=images, joint_bounds=joint_bounds, minmax=minmax, step_by_step=args.debug, position_publisher=pub_pos, hand_publisher=hand_client)
    else:
        # ONLINE MODE!!
        # TODO joint_bounds are magic numbers!
        run_online_inference(model=model_sarnn, joint_bounds=np.array([0.1,0.9]), minmax=minmax, position_publisher=pub_pos, step_by_step=args.debug, img_resize=args.img_size, hand_publisher=hand_client)

if __name__ == "__main__":
    main()
