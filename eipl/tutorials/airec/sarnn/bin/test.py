#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
# Released under the MIT License.
#

import os
import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
import torch
import torch.nn as nn
from eipl.data import SampleDownloader, WeightDownloader, LocalSampleDownloader
from eipl.model import SARNN
from eipl.utils import normalization
from eipl.utils import restore_args, tensor2numpy, deprocess_img, resize_img

from eipl.tutorials.airec.sarnn.libs.fullBPTT import fullBPTTtrainer

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--path-to-checkpoint", type=str, default=None, help="Path to the model checkpoint")
parser.add_argument("--path-to-data-dir", type=str, default=None, help="Path to test data")
# should end with <your-path>/data/test/
parser.add_argument("--path-to-save-dir", type=str, default=None, help="Path to save data")
parser.add_argument("--idx", type=int, default=0, help="Load images and joint values at index")
parser.add_argument("--input_param", type=float, default=1.0, help="This parameter does nothing.")
parser.add_argument("--pretrained", action="store_true",
                    help="This searches for a model path at '~/.eipl/airec/grasp_bottle/weights/SARNN/model.pth'")
parser.add_argument("--pos_as_mask", action="store_true", help="Input the pose through the mask")
parser.add_argument("--pos_as_state", action="store_true", help="Input the pose through the state encoder")
parser.add_argument("--img_resize", type=int, help="additional image resize", default=64)
parser.add_argument("--loss_weights", type=tuple, help="weights for image and joint loss in this order", default=[1., 1.])
args = parser.parse_args()

# check args
assert args.path_to_checkpoint or args.pretrained, "Please set filename or pretrained"
# load pretrained weight
if args.pretrained:

    WeightDownloader("airec", "grasp_bottle")
    args.filename = os.path.join(
        os.path.expanduser("~"), ".eipl/airec/grasp_bottle/weights/SARNN/model.pth"
    )

# restore parameters
dir_name = os.path.split(args.path_to_checkpoint)[0]  # get directory up to file
params = restore_args(os.path.join(dir_name, "args.json"))
idx = args.idx

# load dataset
minmax = [params["vmin"], params["vmax"]]
grasp_data = LocalSampleDownloader(data_path=args.path_to_data_dir, img_format="HWC")
_images, _joints, masks, poses = grasp_data.load_raw_data("test")
_images = resize_img(_images, (args.img_resize, args.img_resize))
images = _images[idx]
joints = _joints[idx]
joint_bounds = grasp_data.joint_bounds
print(
    "images shape:{}, min={}, max={}".format(images.shape, images.min(), images.max())
)
print(
    "joints shape:{}, min={}, max={}".format(joints.shape, joints.min(), joints.max())
)

# define model
model = SARNN(
    rec_dim=params["rec_dim"],
    joint_dim=8,  # TODO check how many joints we have!
    k_dim=params["k_dim"],
    im_size=[args.img_resize, args.img_resize],
    heatmap_size=params["heatmap_size"],
    temperature=params["temperature"],
)

if params["compile"]:
    model = torch.compile(model)


device = "cuda" if torch.cuda.is_available() else "cpu"

# load weight
ckpt = torch.load(args.path_to_checkpoint, map_location=torch.device("cpu"), weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference
img_size = args.img_resize
image_list, joint_list = [], []
enc_pts_list, dec_pts_list = [], []
loss_list, img_loss_list, joint_loss_list = [], [], []
state = None
mse_loss = nn.MSELoss()
for loop_ct in range(len(images)):
    print(f"{loop_ct} from {len(images)}")
    # load data and normalization
    img_t = images[loop_ct].transpose(2, 0, 1)
    img_t = torch.Tensor(np.expand_dims(img_t, 0))
    img_t = normalization(img_t, (0, 255), minmax)
    joint_t = torch.Tensor(np.expand_dims(joints[loop_ct], 0))
    joint_t = normalization(joint_t, joint_bounds, minmax)

    # closed loop
    if loop_ct > 0:  # not for first image!! y_image is undefined & input_param = 1
        # this function does exactly nothing and cannot work
        img_t = args.input_param * img_t + (1.0 - args.input_param) * y_image
        joint_t = args.input_param * joint_t + (1.0 - args.input_param) * y_joint

    # INFERENCE predict rnn
    y_image, y_joint, enc_pts, dec_pts, state = model.forward(img_t, joint_t, state)

    # Add the loss for inference
    if loop_ct < len(images) - 1: #  we want to stop one before the last
       # Prepare target image: HWC -> BCHW
            target_img = torch.tensor(
                images[loop_ct + 1].transpose(2, 0, 1)[None, ...],
                dtype=torch.float32,
            )
            target_img = normalization(target_img, (0, 255), minmax)

            # Prepare target joints in the same normalized space as y_joint
            target_joint = torch.tensor(
                joints[loop_ct + 1][None, ...],
                dtype=torch.float32,
            )
            target_joint = normalization(target_joint, joint_bounds, minmax)

            # Weighted MSE losses
            img_loss = mse_loss(y_image, target_img) * args.loss_weights[0]
            joint_loss = mse_loss(y_joint, target_joint) * args.loss_weights[1]
            total_loss = img_loss + joint_loss

            # Store scalar values for plotting
            img_loss_list.append(img_loss.item())
            joint_loss_list.append(joint_loss.item())
            loss_list.append(total_loss.item()) 

    # denormalization
    pred_image = tensor2numpy(y_image[0])
    pred_image = deprocess_img(pred_image, params["vmin"], params["vmax"])
    pred_image = pred_image.transpose(1, 2, 0)
    pred_joint = tensor2numpy(y_joint[0])
    pred_joint = normalization(pred_joint, minmax, joint_bounds)

    # send pred_joint to robot
    # send_command(pred_joint)
    # pub.publish(pred_joint)

    # append data
    image_list.append(pred_image)
    joint_list.append(pred_joint)
    enc_pts_list.append(tensor2numpy(enc_pts[0]))
    dec_pts_list.append(tensor2numpy(dec_pts[0]))

    print("loop_ct:{}, joint:{}".format(loop_ct, pred_joint))

pred_image = np.array(image_list)
pred_joint = np.array(joint_list)

# split key points
enc_pts = np.array(enc_pts_list)
dec_pts = np.array(dec_pts_list)
enc_pts = enc_pts.reshape(-1, params["k_dim"], 2) * img_size
dec_pts = dec_pts.reshape(-1, params["k_dim"], 2) * img_size
enc_pts = np.clip(enc_pts, 0, img_size)
dec_pts = np.clip(dec_pts, 0, img_size)

# plot images
T = len(images)
fig, ax = plt.subplots(2, 3, figsize=(18, 10), dpi=100)
ax = ax.flatten()

def anim_update(i):
    for j in range(6):
        ax[j].cla()

    # plot camera image
    ax[0].imshow(images[i, :, :, ::-1])
    for j in range(params["k_dim"]):
        ax[0].plot(enc_pts[i, j, 0], enc_pts[i, j, 1], "bo", markersize=6)  # encoder
        ax[0].plot(
            dec_pts[i, j, 0], dec_pts[i, j, 1], "rx", markersize=6, markeredgewidth=2
        )  # decoder
    ax[0].axis("off")
    ax[0].set_title("Input image")

    # plot predicted image
    ax[1].imshow(pred_image[i, :, :, ::-1])
    ax[1].axis("off")
    ax[1].set_title("Predicted image")

    # plot joint angle
    ax[2].set_ylim(-1.0, 2.0)
    ax[2].set_xlim(0, T)
    ax[2].plot(joints[1:], linestyle="dashed", c="k")
    for joint_idx in range(8):
        ax[2].plot(np.arange(i + 1), pred_joint[: i + 1, joint_idx])
    ax[2].set_xlabel("Step")
    ax[2].set_title("Joint angles")

     # 4. Overall loss
    loss_steps = np.arange(len(loss_list))
    ax[3].plot(loss_steps[:i+1], loss_list[:i+1], c="purple")
    ax[3].set_xlim(0, max(T - 1, 1))
    ax[3].set_title("Overall loss")
    ax[3].set_xlabel("Step")
    ax[3].set_ylabel("Weighted MSE")
    ax[3].grid(True, alpha=0.3)

    # 5. Image loss
    img_steps = np.arange(len(img_loss_list))
    ax[4].plot(img_steps[:i+1], img_loss_list[:i+1], c="blue")
    ax[4].set_xlim(0, max(T - 1, 1))
    ax[4].set_title("Image loss")
    ax[4].set_xlabel("Step")
    ax[4].set_ylabel("Weighted MSE")
    ax[4].grid(True, alpha=0.3)

    # 6. Joint loss
    joint_steps = np.arange(len(joint_loss_list))
    ax[5].plot(joint_steps[:i+1], joint_loss_list[:i+1], c="red")
    ax[5].set_xlim(0, max(T - 1, 1))
    ax[5].set_title("Joint loss")
    ax[5].set_xlabel("Step")
    ax[5].set_ylabel("Weighted MSE")
    ax[5].grid(True, alpha=0.3)

    plt.tight_layout() 


ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
if not os.path.exists(args.path_to_save_dir):
    os.makedirs(args.path_to_save_dir)

try:
    ani.save(os.path.join(args.path_to_save_dir, "SARNN_{}_{}_{}.gif").format(params["tag"], idx, args.input_param),
             writer="ffmpeg")
except Exception as e:
    print(e)
    print("""
    If an error occurs in generating the gif animation or mp4, change the writer (imagemagick/ffmpeg).
    ani.save("./output/PCA_SARNN_{}.gif".format(params["tag"]), writer="imagemagick")
    ani.save("./output/PCA_SARNN_{}.mp4".format(params["tag"]), writer="ffmpeg")
    """)
