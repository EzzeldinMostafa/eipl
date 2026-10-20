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


def check_cuda():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            props = torch.cuda.get_device_properties(0)
            if getattr(props, "major", 0) >= 8:
                torch.backends.cudnn.conv.fp32_precision = "tf32"
                torch.backends.cuda.matmul.fp32_precision = "tf32"
        except Exception as e:
            print(e)


class EiplInferenceHandler:

    def __init__(self, loaded_model: SARNN, step_by_step: bool, img_resize: int, minmax,
                 joint_bounds: np.ndarray = np.array([0.1, 0.9])):
        self.cv_bridge = CvBridge()

        # model configuration
        self.eipl_model = loaded_model
        self.step_by_step = step_by_step
        self.img_resize = img_resize
        self.minmax = minmax
        self.joint_bounds = joint_bounds  # TODO joint_bounds are magic numbers for online mode!
        self.eipl_model_state = None

        # topics
        self.topic_name_arm_position = '/torobo/online_joint_trajectory_controller/command'
        self.topic_name_image = '/torobo/head/sr300/camera/color/image_rect_color'
        self.topic_name_joint_state = '/torobo/joint_states'
        self.topic_name_hand_state = '/torobo/right_hand_controller/state'
        self.joint_names_right_arm = [
            f"right_arm/joint_{i}" for i in range(1, 8)
        ]
        self.joint_name_index_finger = 'right_hand_first_finger/joint_1'

        # Robot State
        self.current_joint_values_arm = np.array([])
        self.current_hand_state = 0  # open: 1, close: 0
        self.current_joint_value_hand: None | float = None
        self.current_rgb_image = None

        # ROS
        self.arm_position_publisher = None
        self.hand_position_publisher = None
        self.ros_setup()

    def ros_image_to_bgr(self, msg: Image) -> np.ndarray:
        return self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

    def reset_robot_state(self):
        self.current_joint_values_arm = np.array([])
        self.current_joint_value_hand = None
        self.current_rgb_image = None

    def ros_setup(self):
        try:
            rospy.init_node('eipl_inference_node', anonymous=False)
            rospy.Subscriber(self.topic_name_image, Image, self.camera_right_callback)
            rospy.Subscriber(self.topic_name_joint_state, JointState, self.arm_state_callback)
            rospy.Subscriber(self.topic_name_hand_state, JointTrajectoryControllerState, self.hand_state_callback)
            self.arm_position_publisher = rospy.Publisher(self.topic_name_arm_position, JointTrajectory, queue_size=10)
            self.hand_position_publisher = HandClient("/torobo", "right")

        except rospy.ROSInterruptException as e:
            print(e)
            raise RuntimeError("Could not setup ROS publisher and subscriber")

    def publish_position(self, q_target: np.ndarray):
        # Publish arm joints
        msg = JointTrajectory()
        msg.header.stamp = rospy.Time.now()
        msg.joint_names = self.joint_names_right_arm

        point = JointTrajectoryPoint()
        point.positions = q_target[:-1].tolist()
        point.velocities = [0.0] * len(self.joint_names_right_arm)
        point.accelerations = [0.0] * len(self.joint_names_right_arm)
        point.effort = [0.0] * len(self.joint_names_right_arm)
        point.time_from_start = rospy.Duration(0, 0)
        msg.points.append(point)
        self.arm_position_publisher.publish(msg)
        rospy.loginfo("Published JointTrajectoryPoint to robot")
        rospy.logdebug(msg)

        # Publish hand position, simple threshold to set hand pose
        if q_target[-1] > 0.1:
            self.hand_position_publisher.update(1)
            rospy.loginfo("Hand should be closing")
        else:
            self.hand_position_publisher.update(-1)
            rospy.loginfo("Hand should be opening")

    def arm_state_callback(self, msg: JointState):
        joint_map = dict(zip(msg.name, msg.position))
        if all([True for key in list(joint_map.keys()) if key in self.joint_names_right_arm]):
            rospy.logdebug("All joints received and available")
            self.current_joint_values_arm = np.array(
                [joint_map[joint_name] for joint_name in self.joint_names_right_arm])
            assert (self.current_joint_values_arm.size == 7)
            return
        rospy.logdebug(msg)
        rospy.logwarn("Unexpected message format received in joint callback")

    def compute_hand_state(self, previous_joint_value_hand: float):
        if self.current_joint_value_hand - previous_joint_value_hand > 0.005 and self.current_hand_state == 0:
            self.current_hand_state = 1
        elif previous_joint_value_hand - self.current_joint_value_hand > 0.005 and self.current_hand_state == 1:
            self.current_hand_state = 0

    def hand_state_callback(self, msg: JointTrajectoryControllerState):
        finger_index = msg.joint_names.index(self.joint_name_index_finger)
        previous_joint_value_hand = self.current_joint_value_hand
        self.current_joint_value_hand = np.array(msg.desired.positions[finger_index])
        if previous_joint_value_hand is None:
            previous_joint_value_hand = self.current_joint_value_hand

        # Check hand state
        self.compute_hand_state(previous_joint_value_hand)

    def camera_right_callback(self, msg: Image):
        try:
            self.current_rgb_image = self.ros_image_to_bgr(msg)
            if self.current_rgb_image is None:
                rospy.logerr("Failed to capture image from camera.")

        except Exception as err:
            rospy.logerr(err)

    def run_inference(self, img_t: np.ndarray, joint_t: np.ndarray):
        # run the actual inference on the model, handle en- and de-coding
        # takes the (resized) image and the current joint values (len == 8)

        # publishing rate
        r = rospy.Rate(5)

        # image
        img_t = img_t.transpose(2, 0, 1)
        img_t = torch.Tensor(np.expand_dims(img_t, 0))
        img_t = normalization(img_t, (0, 255), self.minmax)

        # joints
        joint_t = torch.Tensor(np.expand_dims(joint_t, 0))
        joint_t = normalization(joint_t, self.joint_bounds, self.minmax)

        # predict rnn
        print(f"input joints {joint_t}")
        y_image, y_joint, _enc_pts, _dec_pts, state = self.eipl_model.forward(img_t, joint_t, self.eipl_model_state)

        # denormalization
        pred_image = tensor2numpy(y_image[0])  # first prediction in time
        pred_image = deprocess_img(pred_image, self.minmax[0], self.minmax[1])
        pred_image = pred_image.transpose(1, 2, 0)
        pred_joint = tensor2numpy(y_joint[0])  # first prediction in time
        rospy.logdebug(f"Predicted value hand joint (not normalized):\n {pred_joint[-1]} \n")
        pred_joint = normalization(pred_joint, self.minmax, self.joint_bounds)
        rospy.logdebug(f"Predicted value hand joint (normalized):\n {pred_joint[-1]} \n")

        if (pred_joint.shape != (8,)) or not np.all(np.isfinite(pred_joint)):
            rospy.logerr("Invalid EIPL prediction")
            raise RuntimeError(f"pred_joints invalid: {pred_joint}")

        # conservative example: do not allow jumps
        max_step = 0.02
        print(f"Predicted value hand {pred_joint[-1]}")
        q_safe = self.current_joint_values_arm + np.clip(pred_joint[:-1] - self.current_joint_values_arm, -max_step,
                                                         max_step)
        q_safe = np.append(q_safe, pred_joint[-1])
        rospy.loginfo(f"Clipped joint values delta within range {max_step} to\n {q_safe}")
        cv2.imshow("[EIPL] Predicted Image", pred_image)
        cv2.waitKey(1)
        if self.step_by_step:
            input("Send to robot? <Press Enter>")
        self.publish_position(q_safe)

        r.sleep()
        self.reset_robot_state()

    def run_offline_inference(self, data_path: str, load_index: int):
        rospy.loginfo("Starting inference in offline mode, using images from stored data")

        # Load dataset
        grasp_data = LocalSampleDownloader(data_path=data_path, img_format="HWC")
        _images, _joints, _masks, _poses = grasp_data.load_raw_data("test")
        _images = resize_img(_images, (self.img_resize, self.img_resize))
        images = _images[load_index]
        _joints = _joints[load_index]
        masks = _masks[load_index]
        poses = _poses[load_index]
        self.joint_bounds = grasp_data._load_bounds()

        # Inference
        for loop_ct in range(images):
            # load data and normalization
            img_t = images[loop_ct]

            while self.current_joint_values_arm.size == 0 and self.current_joint_value_hand is None:
                rospy.logwarn(
                    f"Waiting for joint state information from {self.topic_name_joint_state} and {self.topic_name_hand_state}")
                rospy.sleep(0.1)
            assert (self.current_hand_state == 0 or self.current_hand_state == 1)
            joints_and_hand = np.append(self.current_joint_values_arm, self.current_hand_state)

            # predict rnn
            self.run_inference(img_t=img_t, joint_t=joints_and_hand)

        self.reset_robot_state()

    def run_online_inference(self):
        rospy.loginfo("Starting inference in ONLINE mode: DANGER!!!")

        while not rospy.is_shutdown():
            # wait for image and joint data
            if self.current_rgb_image is None:
                rospy.logwarn(f"Wait for image from {self.topic_name_image}")
                rospy.sleep(0.1)
                continue
            img_t = cv2.resize(self.current_rgb_image, (self.img_resize, self.img_resize))

            if self.current_joint_values_arm.size == 0 and self.current_joint_value_hand is None:
                rospy.logwarn(
                    f"Waiting for joint state information from {self.topic_name_joint_state} and {self.topic_name_hand_state}")
                rospy.sleep(0.1)
                continue
            joints_and_hand = np.append(self.current_joint_values_arm, self.current_hand_state)

            # predict with EIPL
            self.run_inference(img_t=img_t, joint_t=joints_and_hand)


def main():
    # argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to EIPL checkpoint")
    parser.add_argument("--idx", type=int, default=0, help="Select the test data sequence")
    parser.add_argument("--img-size", type=int, default=128, help="Image size :D ")
    parser.add_argument("--joint-dims", type=int, default=8, help="Number of joints of the robot")
    parser.add_argument("--offline", default=False, action="store_true",
                        help="If set to true a path to offline recorded data is needed and used instead of real camera data.")
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
    dir_name = os.path.split(args.checkpoint_path)[0]  # get parent path of checkpoint file
    hyperparam_filename = os.path.join(dir_name, "args.json")  # data.json is generated when training
    with open(hyperparam_filename) as f:
        params = json.load(f)

    # Init model
    model_sarnn = SARNN(
        rec_dim=params["rec_dim"],
        joint_dim=args.joint_dims,
        k_dim=params["k_dim"],
        heatmap_size=params["heatmap_size"],
        temperature=params["temperature"],
        im_size=[args.img_size, args.img_size]
    )
    minmax = [params["vmin"], params["vmax"]]

    # Load weights
    print("Loading model to CPU")
    ckpt = torch.load(args.checkpoint_path, map_location=torch.device("cpu"), weights_only=False)  # CPU
    model_sarnn.load_state_dict(ckpt["model_state_dict"])
    model_sarnn.eval()
    model_sarnn.to("cpu")
    print("Loaded model")

    eipl_inference_handler = EiplInferenceHandler(loaded_model=model_sarnn, step_by_step=args.debug,
                                                  img_resize=args.img_size,
                                                  minmax=minmax, joint_bounds=np.array([0.1, 0.9]))

    if args.offline:
        # OFFLINE mode, overrides joint_bounds
        if not args.data_path:
            raise RuntimeError("Path to offline dataset is needed in offline mode.")
        eipl_inference_handler.run_offline_inference(data_path=args.data_path, load_index=args.idx)
    else:
        # ONLINE MODE!!
        eipl_inference_handler.run_online_inference()


if __name__ == "__main__":
    main()
