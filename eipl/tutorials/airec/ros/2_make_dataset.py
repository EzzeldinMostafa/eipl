#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import os
import cv2
import glob
import argparse
import numpy as np
import matplotlib.pylab as plt
from datetime import datetime
from eipl.utils import resize_img, calc_minmax, list_to_numpy, cos_interpolation


def load_data(dir):
    joints = []
    images = []
    seq_length = []
    masks = []
    poses = []

    files = glob.glob(os.path.join(dir, "*.npz"))
    files.sort()
    for filename in files:
        print(filename)
        npz_data = np.load(filename, allow_pickle=True)
        images.append(resize_img(npz_data["images"], (128, 128)))
        finger_state = cos_interpolation(npz_data["finger_state"], expand_dims=True)
        # shape 8 for AIREC 7 arm + one for the gripper like hand
        _joints = np.concatenate((npz_data["joints"], finger_state), axis=-1)
        joints.append(_joints)
        seq_length.append(len(_joints))

        if npz_data["masks"].shape[1] != 0: 
            masks.append(resize_img(npz_data["masks"], (128, 128)))
        if npz_data["poses"].shape[1] != 0:
            poses.append(npz_data["poses"]) 
    
    max_seq = max(seq_length)
    images = list_to_numpy(images, max_seq)
    joints = list_to_numpy(joints, max_seq)

    if len(masks) > 0:
        masks = list_to_numpy(masks, max_seq)
    else:
        masks = None

    if len(poses) > 0:
        poses = list_to_numpy(poses, max_seq)
    else:
        poses = None 
        
    return images, joints, masks, poses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag_dir", type=str)
    args = parser.parse_args()

    # load data
    images, joints, masks, poses = load_data(args.bag_dir)

    
    # dataset index
    seq_len = images.shape[0]
    train_list = list(range(1, seq_len-1)) 
    test_list = [0, seq_len-1]# hard coded first and last sample
    
    # format as Year-Month-Day Hour:Minute:Second
    now = datetime.now()
    formatted_timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    # check and mkdir
    os.makedirs(f"./data/{formatted_timestamp}/train/", exist_ok=True,)
    os.makedirs(f"./data/{formatted_timestamp}/test/", exist_ok=True,)
    
    # save images and joints
    np.save(f"./data/{formatted_timestamp}/train/images.npy", images[train_list].astype(np.uint8))
    np.save(f"./data/{formatted_timestamp}/train/joints.npy", joints[train_list].astype(np.float32))
    np.save(f"./data/{formatted_timestamp}/test/images.npy", images[test_list].astype(np.uint8))
    np.save(f"./data/{formatted_timestamp}/test/joints.npy", joints[test_list].astype(np.float32))

    if masks is not None:
        np.save(f"./data/{formatted_timestamp}/train/masks.npy", masks[train_list].astype(np.uint8))
        np.save(f"./data/{formatted_timestamp}/test/masks.npy", masks[test_list].astype(np.uint8))

    if poses is not None:
        np.save(f"./data/{formatted_timestamp}/train/poses.npy", poses[train_list].astype(np.float32))
        np.save(f"./data/{formatted_timestamp}/test/poses.npy", poses[test_list].astype(np.float32))

    # save joint bounds
    joint_bounds = calc_minmax(joints)
    np.save(f"./data/{formatted_timestamp}/joint_bounds.npy", joint_bounds)
    np.save(f"./data/{formatted_timestamp}/joint_bounds.npy", joint_bounds)

    print(f"images shape is {images[train_list].astype(np.uint8).shape}")
    print(f"joints shape is {joints[train_list].astype(np.float32).shape}")

    if masks is not None:
        print(f"masks shape is {masks[train_list].astype(np.uint8).shape}")
    
    if poses is not None:
        print(f"poses shape is {poses[train_list].astype(np.float32).shape}")
