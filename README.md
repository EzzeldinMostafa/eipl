
<div align="center">
<img src="https://raw.githubusercontent.com/ogata-lab/eipl-docs/master/top/resources/logo.svg" width="100%">
</div>

## What's EIPL?

EIPL (Embodied Intelligence with Deep Predictive Learning) is a library for robot motion generation using deep predictive learning developed at [Ogata Laboratory](https://ogata-lab.jp/), [Waseda University](https://www.waseda.jp/top/en).
Highlighted features include:

- [**Full documentation**](https://ogata-lab.github.io/eipl-docs) for the systematic understanding of deep predictive learning
- **Easy model training:** Includes sample datasets, source code, and pre-trained weights
- **Applicable to real robots:** Generalized motion can be acquired with small data sets

## Install

```sh
pip install -r requirements.txt
pip install -e .
```

## Citation

```
@article{suzuki2023deep,
  author    = {Kanata Suzuki and Hiroshi Ito and Tatsuro Yamada and Kei Kase and Tetsuya Ogata},
  title     = {Deep Predictive Learning : Motion Learning Concept inspired by Cognitive Robotics}, 
  booktitle = {arXiv preprint arXiv:2306.14714},
  year      = {2023},
}
```

## LICENSE

MIT License

## Eipl in MOTH

### General Overview of EIPL and MOTH
In moth, we want to feed the output of M3T tracker to eipl.

We start with a vanilla eipl approach using only right arm joint angles, 1D hand state and images

Input:
- joints (8: `n_joints`)
- images (128x128/64x64x3/`img_size`x`img_size`x`n_channels`)

Ouput:
- joints (8: `n_joints`)
- images (128x128/64x64x3/`img_size`x`img_size`x`n_channels`)

We will from now on refer to these shapes as `img_size` and `n_joints` and `n_channels`

The idea is to add the information available by the M3T tracker to eipl to improve training speed/performance. There are two options:

#### 1. tracked mask concatinated to image input
One option is to add the mask (`img_size`x`img_size`x1) as a channel to the images.

Input:
- joints (8: `n_joints`)
- images (128x128x4/64x64x4/`img_size`x`img_size`x`n_channels`)

Ouput:
- joints (8: `n_joints`)
- images (128x128x4/64x64x3/`img_size`x`img_size`x`n_channels`)

This changes `n_channels` from 3 to 4 for the input, so far we did only add it as an input but not to the prediction such that `n_channels` stays at 3 for the Ouput, we can discuss whether changin this makes sense.

#### 2. tracked pose added to joints
Another option is to add the pose (7x1 quaternion) to the joint vector

Input:
- joints (15: `n_joints`)
- images (128x128/64x64x3/`img_size`x`img_size`x`n_channels`)

Ouput:
- joints (8: `n_joints`)
- images (128x128/64x64x3/`img_size`x`img_size`x`n_channels`)

This changes `n_joint` from 8 to 15. I am not sure yet whether we will also predict the poses or not.
Another discussion was whether it makes sense to normalize the data to (0,1) as this is done for all the other joints, and also whether it makes sense to make the pose relative to the start pose or keep it relative to the camera.

### Overview what we will need from this repo
#### 1. Data Preprocessing
Inside `eipl/eipl/tutorials/airec/ros` there are three scripts used to convert the bag files from the experiments into npy matrices used for training eipl and visualizing the collected training data

#### 2. Training and Testing
Inside `eipl/eipl/tutorials/airec/sarnn/bin` there is a training script to train the model and a test script to visualize the predicitons on the test dataset.

### Getting started
If you want to train a model, inspect the data and test it, we recommend using pixi to run eipl, instead of cloning this repo clone the pixi environment (id not already done anyways), this contains the eipl repo as a submodule. If you want to run inference we did not yet find a way to use the pixi environment but set it up globally, we will add sintructions on this later.

#### 1. clone pixi environment (if not already done)
```
git clone https://github.com/AnneRei/pixi_ros_noetic_ws
```
make sure to update the submodules
```
cd pixi_ros_noetic_ws
```
```
git submodule update --init --recursive
```
#### 2. make sure to have the collected bag-files available.
Create a folder and put all bag-files inside. We will from now on refer to the folder containing the bag-files as the `<bag-file-folder>`

#### 3. Activate pixi to use the environment
Go to the pixi foldere
```
cd /path/to/pixi_ros_noetic_ws
```
activate a pixi shell
```
pixi shell
```

### Preprocessing and visualizing the recorded data
`eipl/eipl/tutorials/airec/ros` contains all scipts needed for data preparation.
#### 1. 1_rosbag2npz.py
This script reads all bag-files inside the `--bag_dir` and creates one npz-file.
It uncompressed the images and converts all data into numpy arrays and cuts the topics that have by chance an element more than others to one common size. The npz files will be written inside the `--bag-dir`

Input: N bag-files
Output: One npz file per bag-file containing all topics specified inside the script (N npz files)

You can change the topcis that should be included inside the script. Use this with a grain of salt though, as not all topics are supported yet and there is quite some hard-coding still, so if you change the topics you might need to adapt the code as well.


To run it use:
Activate pixi (if not already done)

Go to the pixi folder
```
cd /path/to/pixi_ros_noetic_ws
```
activate a pixi shell
```
pixi shell
```

change into directory (if not aslready done)
```
cd eipl/eipl/tutorials/airec/ros
```
run the code
```
python3 1_rosbag2npz.py --bag_dir <bag-file-folder>
```

#### 2. 2_make_dataset.py
This script reads npz-file created above and creates one npy-file per topic for training data and one npy-file per topic for test data. It reshapes the images to 128x128 and pads all logs to the longest one such that we can now put the data from all experiments into one big numpy tensor.

Input: npz file
Output: 2*M npy files (where M is the number of topics, 1 for training data and 1 for test data)

To run it use:

Activate pixi (if not already done)

Go to the pixi folder
```
cd /path/to/pixi_ros_noetic_ws
```
activate a pixi shell
```
pixi shell
```
change into directory (if not aslready done)
```
cd eipl/eipl/tutorials/airec/ros
```
run the code
```
python3 2_make_dataset.py --bag_dir <bag-file-folder>
```

The data is created inside `./data/{formatted_timestamp}/train/` and `./data/{formatted_timestamp}/test` respecively so it will be located relative to the current directory (path inside eipl: `eipl/eipl/tutorials/airec/ros/data/{formatted_timestamp}/train` and `eipl/eipl/tutorials/airec/ros/data/{formatted_timestamp}/test`)

#### 3. 3_check_data.py
This script can be used to generate gifs that visualize the training/test data.

Input: npy files
Output: .gifs

To run it use:

Activate pixi (if not already done)

Go to the pixi folder
```
cd /path/to/pixi_ros_noetic_ws
```
activate a pixi shell
```
pixi shell
```

change into directory (if not aslready done)
```
cd eipl/eipl/tutorials/airec/ros
```
run the code
```
python3 3_check_data.py 
```



