#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#
import os
import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
from eipl.utils import normalization
import time
from pathlib import Path
from eipl.utils import resize_img

parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=-1)
parser.add_argument("--seq_len", type=int, default=5)
parser.add_argument("--path", type=str, help="path to folder containing final npy files")
parser.add_argument("--img_size", type=int, default=128)
args = parser.parse_args()

img_size = args.img_size
resize = False

n_sub = 3
if "masks.npy" in os.listdir(args.path):
    n_sub =4 
    masks = np.load(os.path.join(args.path, "masks.npy"))
    if resize:
        masks = resize_img(masks, (args.img_size, args.img_size))

idx = int(args.idx)
    
joints = np.load(os.path.join(args.path, "joints.npy"))
joint_bounds = np.load(os.path.join(args.path,"../joint_bounds.npy"))
images = np.load(os.path.join(args.path,"images.npy"))

#optionally resize images
if resize:
    images = resize_img(images, (args.img_size, args.img_size))


N = images.shape[1] #time length
n_trials = images.shape[0] #how many experiments were done


# normalized joints
minmax = [0.1, 0.9]
norm_joints = normalization(joints, joint_bounds, minmax)

# print data information
print("load test data, index number is {}".format(idx))
print(
    "Joint: shape={}, min={:.3g}, max={:.3g}".format(
        joints.shape, joints.min(), joints.max()
    )
)
print(
    "Norm joint: shape={}, min={:.3g}, max={:.3g}".format(
        norm_joints.shape, norm_joints.min(), norm_joints.max()
    )
)



def plot_sequence(time_step, plot_ind=None, i_seq=0, img_size=128):
    if plot_ind is None:
        plot_ind = i_seq
    # plot image
    ax[plot_ind][0].imshow(images[i_seq, time_step, :, :, ::-1])
    ax[plot_ind][0].axis("off")
    ax[plot_ind][0].set_title("Image")

    # plot joint angle
    ax[plot_ind][1].set_ylim(-1.0, 2.0)
    ax[plot_ind][1].set_xlim(0, N)
    ax[plot_ind][1].plot(joints[i_seq], linestyle="dashed", c="k")

    for joint_idx in range(8):
        ax[plot_ind][1].plot(np.arange(time_step + 1), joints[i_seq, : time_step + 1, joint_idx])
    ax[plot_ind][1].set_xlabel("Step")
    ax[plot_ind][1].set_title("Joint angles")

    # plot normalized joint angle
    ax[plot_ind][2].set_ylim(0.0, 1.0)
    ax[plot_ind][2].set_xlim(0, N)
    ax[plot_ind][2].plot(norm_joints[i_seq], linestyle="dashed", c="k")

    for joint_idx in range(8):
        ax[plot_ind][2].plot(np.arange(time_step + 1), norm_joints[i_seq, : time_step + 1, joint_idx])
    ax[plot_ind][2].set_xlabel("Step")
    ax[plot_ind][2].set_title("Normalized joint angles")

    if n_sub == 4:
        # plot masks
        ax[plot_ind][3].imshow(masks[i_seq, time_step])
        ax[plot_ind][3].axis("off")
        ax[plot_ind][3].set_title("Masks")

def anim_update(i, idx, img_size=128, seq_len=1, run_indices=[0]):
    if idx!=-1:
        #only show the selected idx not all data
        for j in range(n_sub):
            ax[0][j].cla()
        plot_sequence(time_step=i, i_seq=0, img_size=img_size)
        return
    
    #if idx is -1, we want to show all data
    for plot_ind, k in enumerate(run_indices):
        for j in range(n_sub):
            ax[plot_ind][j].cla()
        plot_sequence(time_step=i, plot_ind=plot_ind, i_seq=k, img_size=img_size)

seq_len = min(args.seq_len, n_trials)
n_plots = int(np.ceil(n_trials/seq_len))

input_folder_name = Path(args.path).stem

if n_plots == 1:
    #create only 1 file

    # plot images and normalized joints
    fig, ax = plt.subplots(seq_len, n_sub, figsize=(14, 5*seq_len), dpi=60, squeeze=False)
    start = time.time()
    ani = anim.FuncAnimation(fig, anim_update, interval=int(N / 10), frames=N,
                         fargs=[idx, img_size, seq_len, range(seq_len)])
    os.makedirs("./output", exist_ok=True)
    ani.save(f"./output/check_data_{input_folder_name}_{img_size}_{idx}.gif")
    end = time.time()
else:
    os.makedirs(f"./output/{input_folder_name}/", exist_ok=True)
    start_ind = 0
    end_ind = seq_len
    for i in range(n_plots):
        run_indices = list(range(start_ind, end_ind))
        # plot images and normalized joints
        fig, ax = plt.subplots(seq_len, n_sub, figsize=(14, 5*seq_len), dpi=60, squeeze=False)
        start = time.time()
        ani = anim.FuncAnimation(fig, anim_update, interval=int(N / 10), frames=N,
                            fargs=[idx, img_size, seq_len, run_indices])
        ani.save(f"./output/{input_folder_name}/check_data_{img_size}_{start_ind}_{end_ind}.gif")
        end = time.time()
        print(f"At index {i} out ot {n_plots}. Anne, it took {end-start} seconds to render this gif")
        start_ind += seq_len
        end_ind += seq_len
        end_ind = min(end_ind, n_trials)