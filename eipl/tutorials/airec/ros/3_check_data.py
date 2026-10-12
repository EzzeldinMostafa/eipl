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


parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=0)
parser.add_argument("--seq_len", type=int, default=1)
parser.add_argument("--path", type=str, help="path to folder containing final npy files")
args = parser.parse_args()

n_sub = 3
if "masks.npy" in os.listdir(args.path):
    n_sub =4 
    masks = np.load(os.path.join(args.path, "masks.npy"))


idx = int(args.idx)
if idx < 0:
    idx = 0
    
joints = np.load(os.path.join(args.path, "joints.npy"))
joint_bounds = np.load(os.path.join(args.path,"../joint_bounds.npy"))
images = np.load(os.path.join(args.path,"images.npy"))
N = images.shape[1]


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

# plot images and normalized joints
fig, ax = plt.subplots(args.seq_len, n_sub, figsize=(14, 5*args.seq_len), dpi=60)

def anim_update(i):
    idx = 0
    for k in range(args.seq_len):
        for j in range(n_sub):
            ax[k][j].cla()

        # plot image
        ax[k][0].imshow(images[idx, i, :, :, ::-1])
        ax[k][0].axis("off")
        ax[k][0].set_title("Image")

        # plot joint angle
        ax[k][1].set_ylim(-1.0, 2.0)
        ax[k][1].set_xlim(0, N)
        ax[k][1].plot(joints[idx], linestyle="dashed", c="k")

        for joint_idx in range(8):
            ax[k][1].plot(np.arange(i + 1), joints[idx, : i + 1, joint_idx])
        ax[k][1].set_xlabel("Step")
        ax[k][1].set_title("Joint angles")

        # plot normalized joint angle
        ax[k][2].set_ylim(0.0, 1.0)
        ax[k][2].set_xlim(0, N)
        ax[k][2].plot(norm_joints[idx], linestyle="dashed", c="k")

        for joint_idx in range(8):
            ax[k][2].plot(np.arange(i + 1), norm_joints[idx, : i + 1, joint_idx])
        ax[k][2].set_xlabel("Step")
        ax[k][2].set_title("Normalized joint angles")

        if n_sub == 4:
            # plot masks
            ax[k][3].imshow(masks[idx, i])
            ax[k][3].axis("off")
            ax[k][3].set_title("Masks")
        idx += 1
    

start = time.time()
ani = anim.FuncAnimation(fig, anim_update, interval=int(N / 10), frames=N)
os.makedirs("./output", exist_ok=True)
ani.save("./output/check_data_{}.gif".format(idx))
end = time.time()
print(f"Anne, it took {end-start} seconds to render this gif")
