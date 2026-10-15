
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

### Overview what we will need from this repo
#### 1. Data Preprocessing
Inside `eipl/eipl/tutorials/airec/ros` there are three scripts used to convert the bag files from the experiments into npy matrices used for training eipl and visualizing the collected training data

#### 2. Training and Testing
Inside `eipl/eipl/tutorials/airec/sarnn/bin` there is a training script to train the model and a test script to visualize the predicitons on the test dataset.

### Getting started
1. clone this repo
```
git clone https://github.com/EzzeldinMostafa/eipl
```
2. make sure to have the collected bag-files available. Create a folder and put all bag-files inside. We will form now on refer to the folder containing the bag-files as the `<bag-file-folder>`
3. add the local package to pixi
Go to the pixi foldere
```
cd /path/to/pixi_ros_noetic_ws
```
add this package to the pixi env
```
pixi add --pypi "my_package @ file:///path/to/this/repo"
```
activate a pixi shell
```
pixi shell
```

