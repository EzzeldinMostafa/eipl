#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import torch
import argparse
import numpy as np
import numpy.core as _core

import cv2
import time
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
from PIL import Image as PILImage

# ROS packages
import rospy
from torobo_sensor_msgs.msg import ToroboState
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, Bool
from trajectory_msgs.msg import JointTrajectoryPoint, JointTrajectory

from eipl.model.SARNN import SARNN
from eipl.utils.data import normalization
from eipl.utils.data import tensor2numpy, deprocess_img
from eipl.data import LocalSampleDownloader
from eipl.utils import resize_img

# GLOBAL VARS
TOPIC_NAME_position = '/torobo/online_joint_trajectory_controller/command'
TOPIC_NAME_impedance = '/torobo/online_joint_impedance_controller/command'
TOPIC_NAME_image = '/torobo/head/sr300/camera/color/image_rect_color'
TOPIC_NAME_joint_state = '/torobo/joint_states'
JOINTS = np.array([])
TORQUE = np.array([])
IMAGE_RGB = None

JOINT_NAMES = [
    f"right_arm/joint_{i}" for i in range(1,8)
]


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

def publish_position(q_target, position_publisher):
    msg = JointTrajectory()
    msg.header.stamp = rospy.Time.now()
    msg.joint_names = JOINT_NAMES

    point = JointTrajectoryPoint()
    point.positions = q_target.tolist()
    point.time_from_start = rospy.Duration(0,0)
    
    msg.points = [point]
    position_publisher.publish(msg)
    rospy.loginfo("Published JointTrajectoryPoint to robot")

def rosimg_to_bgr(msg: Image) -> np.ndarray:
    """
    ROSの画像メッセージ(sensor_msgs/Image or sensor_msgs/CompressedImage)を
    OpenCVのBGR画像(numpy.ndarray)に変換
    """
    if isinstance(msg, Image):
        # 非圧縮の生画素列
        h, w = msg.height, msg.width
        enc = msg.encoding.lower()
        # 8/16bit 判定
        if enc.endswith("16"):
            dtype = np.uint16
        else:
            dtype = np.uint8
        buf = np.frombuffer(msg.data, dtype=dtype)

        # エンディアン調整（16bitでbig-endianのとき）
        if getattr(msg, "is_bigendian", 0) and dtype == np.uint16:
            buf = buf.byteswap().newbyteorder()

        if enc in ("bgr8", "rgb8"):
            img = buf.reshape(h, w, 3)
            if enc == "rgb8":
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        elif enc in ("bgra8", "rgba8"):
            img = buf.reshape(h, w, 4)
            code = cv2.COLOR_BGRA2BGR if enc == "bgra8" else cv2.COLOR_RGBA2BGR
            return cv2.cvtColor(img, code)
        elif enc == "mono8":
            img = buf.reshape(h, w)
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif enc == "mono16":
            # 16bitグレースケール → 8bitへ正規化してBGR
            img16 = buf.reshape(h, w)
            img8 = cv2.convertScaleAbs(img16, alpha=(255.0 / 65535.0))
            return cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f"Unsupported Image encoding: {msg.encoding}")

    elif isinstance(msg, CompressedImage):
        # JPEG/PNGなどの圧縮データ
        arr = np.frombuffer(msg.data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode CompressedImage")
        return img

    else:
        raise TypeError(f"Unsupported message type: {type(msg)}")


def check_cuda():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        try:
            props = torch.cuda.get_device_properties(0)
            if getattr(props, "major", 0) >= 8:
                torch.backends.cudnn.conv.fp32_precision = "tf32"
                torch.backends.cuda.matmul.fp32_precision = "tf32"
        except Exception:
            pass


def visualize_cv(input_img, predict_img, loop_id=None):

    inp = input_img
    pred = predict_img
    if pred.dtype != np.uint8:
        pred = np.clip(pred, 0, 255).astype(np.uint8)
    if pred.ndim == 3 and pred.shape[2] == 3:
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
    elif pred.ndim == 2:
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

    top = np.hstack([inp, pred])
    grid = np.vstack([top])

    win = "Inference Visualization (OpenCV)"
    if not hasattr(visualize_cv, "_inited"):
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        visualize_cv._inited = True

    cv2.imshow(win, grid)
    cv2.waitKey(1)

def generate_angle_ranges(joint_list):
    return [[(angle - 5, angle + 5) for angle in joint_set] for joint_set in joint_list]

def deg2rad_converter(limit_angles):
    limit_angle_list = []
    for limit_angle in limit_angles:
        limit_angle_list.append([np.deg2rad(angle) for angle in limit_angle])
    return limit_angle_list

def check_joint_in_range(pred_joint, limit_joint_ranges):
    limit_angle = deg2rad_converter(limit_joint_ranges)
    
    for joint_range_set in limit_angle:
        for i, joint_range in enumerate(joint_range_set):
            if joint_range[0] <= pred_joint[i] <= joint_range[1]:
                continue 
            else:
                rospy.logwarn("joint_false")
                return False 
    return True


def angle_sensor_callback(msg):
    global JOINTS
    global TORQUE
    joint_map = dict(zip(msg.name, msg.position))
    if all([True for key in list(joint_map.keys()) if key in JOINT_NAMES]):
        rospy.logdebug("All joints received and available")
        JOINTS = np.array([joint_map[name] for name in JOINT_NAMES])
        JOINTS = np.concatenate((JOINTS, np.array([0.0])), axis=0)
        assert(JOINTS.size == 8 and JOINTS[-1]==0)
        return
    rospy.logdebug(msg.position)
    rospy.logdebug(msg)
    rospy.logwarn("Unexpected message format received in joint callback")

 
def camera_right_callback(msg):
    global IMAGE_RGB
    try:
        IMAGE_RGB = rosimg_to_bgr(msg)
        IMAGE_RGB = IMAGE_RGB[0:960, 0:1280]  # TODO check image resolution, currentlx 960x1280
        if IMAGE_RGB is None:
            rospy.logerr("Failed to capture image from camera.")

    except Exception as err:
        rospy.logerr(err)


def run_offline_inference(model, images, joints, joint_bounds, minmax, impedance_publisher, position_publisher, step_by_step=False):
    global JOINTS
    global TORQUE
    global IMAGE_RGB
    
    rospy.loginfo("Starting inference in offline mode, using images from stored data")
    ########## Loading data ##########
    original_images_multiple = []

    image_list, joint_list = [], []
    ect_pts_list, dec_pts_list = [], []
    state_list = []
    attn_list = []
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
        while JOINTS.size == 0:
            #print("Waiting for ROS callback")
            rospy.sleep(0.1)
        #_joint_t = np.concatenate([JOINTS, TORQUE], axis=0)
        _joint_t = torch.Tensor(np.expand_dims(JOINTS, 0))
        joint_t = normalization(_joint_t, joint_bounds, minmax)

        # predict rnn
        y_image, y_joint, _enc_pts, _dec_pts, state = model.forward(img_t, joint_t, state)

        # denormalization
        pred_image = tensor2numpy(y_image[0])
        pred_image = deprocess_img(pred_image, minmax[0], minmax[1])
        pred_image = pred_image.transpose(1, 2, 0)
        pred_joint = tensor2numpy(y_joint[0])
        pred_joint = normalization(pred_joint, minmax, joint_bounds)
    
        if loop_ct:
            if(pred_joint.shape!=(8,)) or not np.all(np.isfinite(pred_joint)):
                rospy.logerr("Invalid EIPL prediction")
                raise RuntimeError(f"pred_joints invalid: {pred_joint}")
            # conservative example: do not allow jumps
            max_step = 0.02
            q_safe = JOINTS + np.clip(pred_joint-JOINTS, -max_step, max_step)
            rospy.loginfo(f"Clipped joint values delta within range {max_step} to\n {q_safe}")
            cv2.imshow("predicted image", pred_image)
            cv2.waitKey(1)
            if step_by_step:
                input("Send to robot? <Press Enter>")

            #publish_joint_trajectory(
            #            publisher = impedance_publisher,
            #            joint_names = JOINT_NAMES,
            #            positions = [pred_joint[6], pred_joint[13]],
            #           time_from_start = 1
            #        )
            publish_position(q_safe, position_publisher=position_publisher)

            
            r.sleep()
    JOINTS = np.array([])
    TORQUE = np.array(joint_list)

  
def main():
    global IMAGE_RGB

    # argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to EIPL checkpoint")
    parser.add_argument("--idx", type=int, default=0, help="Select the test data sequence")
    parser.add_argument("--img-size", type=int, default=128, help="Image size :D ")
    parser.add_argument("--joint-dims", type=int, default=8, help="Number of joints of the robot")
    parser.add_argument("--offline", default=True, action="store_true", help="If set to true a path to offline recorded data is needed and used instead of real camera data.")
    parser.add_argument("--debug", default=True, action="store_true", help="Press enter before sending data to robot.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to offline data")
    args = parser.parse_args()

    # Check CUDA
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

    # load weight
    ckpt = torch.load(args.checkpoint_path, map_location=torch.device("cpu"), weights_only=False) # CPU
    rospy.loginfo("model is loaded")
    model_sarnn.load_state_dict(ckpt["model_state_dict"])
    model_sarnn.eval() 
    model_sarnn.to("cpu")    

    # ROS Setup
    try:
        rospy.init_node('eipl_inference_node', anonymous=True)
        rospy.Subscriber(TOPIC_NAME_image, Image, camera_right_callback)
        rospy.Subscriber(TOPIC_NAME_joint_state, JointState, angle_sensor_callback)

    except rospy.ROSInterruptException as e:
        print(e)
        raise RuntimeError("Could not setup ROS env")

    pub_pos = rospy.Publisher(TOPIC_NAME_position, JointTrajectory, queue_size=1)
    pub_imp = rospy.Publisher(TOPIC_NAME_impedance, JointTrajectory, queue_size=1)

    while IMAGE_RGB is None:
        # wait for next image from AIREC
        rospy.logwarn(f"Waiting for first image from {TOPIC_NAME_image}")
        rospy.sleep(0.5)
    rospy.loginfo("Receive first image")

    # OFFLINE mode
    if args.offline:
        if not args.data_path:
            raise RuntimeError("Path to offline dataset is needed in offline mode.")
        
        # load dataset
        minmax = [params["vmin"], params["vmax"]]
        grasp_data = LocalSampleDownloader(data_path=args.data_path, img_format="HWC")
        _images, _joints, _masks, _poses = grasp_data.load_raw_data("test")
        _images = resize_img(_images, (args.img_size, args.img_size))
        images = _images[args.idx]
        joints = _joints[args.idx]
        masks = _masks[args.idx]
        poses = _poses[args.idx]
        joint_bounds = grasp_data._load_bounds()

        run_offline_inference(model_sarnn, images, joints, joint_bounds, minmax, impedance_publisher=pub_imp, step_by_step=args.debug, position_publisher=pub_pos)


if __name__ == "__main__":
    main()
