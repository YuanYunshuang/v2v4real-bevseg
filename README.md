# V2V4real with BEV map segmentation

## Introduction
This repo is modified from the project [V2V4Real](https://github.com/ucla-mobility/V2V4Real) of [UCLA Mobility Lab](https://mobility-lab.seas.ucla.edu/).
Check the original version on the [main branch](https://github.com/YuanYunshuang/v2v4real-bevseg/tree/main) or their official [page](https://research.seas.ucla.edu/mobility-lab/v2v4real/).


## Data Download
### OPV2V
Download our [augmented OPV2V dataset](https://seafile.cloud.uni-hannover.de/d/c88d1cc85e7e4cae929f/) for lidar-based BEV map segmentation. 
Unzip with 
```shell
cat train.part.* > train.zip
cat test.part.* > test.zip
unzip train.zip
unzip test.zip
```
The unzipped files should have the following structure:
```shell
├── opv2v
│   ├── train
|      |── 2021_08_16_22_26_54
|      |── ...
│   ├── test
```
### V2V4Real
Please check the official [website](https://research.seas.ucla.edu/mobility-lab/v2v4real/) to download the V2V4Real dataset (OPV2V format).
The unzipped files should have the following structure:
```shell
├── v2v4real
│   ├── train
|      |── testoutput_CAV_data_2022-03-15-09-54-40_1
│   ├── validate
│   ├── test
```
## Installation
To set up the codebase environment, do the following steps:
#### 1. Create conda environment (python >= 3.7)
```shell
conda create -n v2v4real python=3.7
conda activate v2v4real
```
#### 2. Pytorch Installation (>= 1.12.0 Required)
Take pytorch 1.12.0 as an example:
```shell
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```
#### 3. spconv 2.x Installation
```shell
pip install spconv-cu113
```
#### 4. Install other dependencies
```shell
pip install -r requirements.txt
python setup.py develop
```
#### 5.Install bbx nms calculation cuda version
```shell
python opencood/utils/setup.py build_ext --inplace
```

## Quick Start
### Data sequence visualization
To quickly visualize the LiDAR stream in the OPV2V dataset, first modify the `validate_dir`
in your `opencood/hypes_yaml/visualization.yaml` to the opv2v data path on your local machine, e.g. `opv2v/validate`,
and the run the following commond:
```shell
cd ~/v2v4real-bevseg
python opencood/visualization/vis_data_sequence.py [--color_mode ${COLOR_RENDERING_MODE} --isSim]
```
Arguments Explanation:
- `color_mode` : str type, indicating the lidar color rendering mode. You can choose from 'v2vreal', 'constant', 'intensity' or 'z-value'.
- `isSim` : bool type, if you are visualizing the simulation data, then claim this argument.

### Train your model
OpenCOOD uses yaml file to configure all the parameters for training. To train your own model
from scratch or a continued checkpoint, run the following commonds:
```shell
python opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER} --half]
```
Arguments Explanation:
- `hypes_yaml`: the path of the training configuration file, e.g. `opencood/hypes_yaml/v2vreal/point_pillar_fax.yaml`, meaning you want to train
CoBEVT with pointpillar backbone on V2V4Real dataset. See [Tutorial 1: Config System](https://opencood.readthedocs.io/en/latest/md_files/config_tutorial.html) to learn more about the rules of the yaml files.
- `model_dir` (optional) : the path of the checkpoints. This is used to fine-tune the trained models. When the `model_dir` is
given, the trainer will discard the `hypes_yaml` and load the `config.yaml` in the checkpoint folder.
- `half` (optional): If set, the model will be trained with half precision. It cannot be set with multi-gpu training togetger.

To train on **multiple gpus**, run the following command:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --use_env opencood/tools/train.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}]
```
**_For more details, please check the original version of this project_

### Test the model
Before you run the following command, first make sure the `validation_dir` in config.yaml under your checkpoint folder
refers to the testing dataset path, e.g. `v2v4real/test`.

```shell
python opencood/tools/inference.py --model_dir ${CHECKPOINT_FOLDER} --fusion_method ${FUSION_STRATEGY} [--show_vis] [--show_sequence] [--save_evibev]
```
Arguments Explanation:
- `model_dir`: the path to your saved model.
- `fusion_method`: indicate the fusion strategy, currently support 'nofusion', 'early', 'late', and 'intermediate'.
- `show_vis`: whether to visualize the detection overlay with point cloud.
- `show_sequence` : the detection results will visualized in a video stream. It can NOT be set with `show_vis` at the same time.
- `save_evibev` : whether to save the test output for later evaluation in evibev project.

### BEV segmentation result

| Method   | OPV2V-road | OPV2V-object | V2V4Real-object | OPV2V ckpt                                                                                                                            | V2V4Real ckpt |
|----------|------------|--------------|-----------------|---------------------------------------------------------------------------------------------------------------------------------------|---------------|
| Fcooper  | 70.3       | 52.06        | 25.87           | [<img src="./imgs/download.png" alt="drawing" width="20"/>](https://seafile.cloud.uni-hannover.de/d/e6bcab88954443bca0cc/) | [<img src="./imgs/download.png" alt="drawing" width="20"/>](https://seafile.cloud.uni-hannover.de/d/c4b038c7ff014d058d1f/) |
| AttnFuse | 75.32      | 52.34        | 25.47           | [<img src="./imgs/download.png" alt="drawing" width="20"/>](https://seafile.cloud.uni-hannover.de/d/82cec5d502ef4e4f8aba/) | [<img src="./imgs/download.png" alt="drawing" width="20"/>](https://seafile.cloud.uni-hannover.de/d/a880908e443d4ccbb43c/) |
| V2X-ViT  | 75.03      | 50.41        | 29.87           | [<img src="./imgs/download.png" alt="drawing" width="20"/>](https://seafile.cloud.uni-hannover.de/d/12bfe53fd82d42e583a2/) | [<img src="./imgs/download.png" alt="drawing" width="20"/>](https://seafile.cloud.uni-hannover.de/d/229f8268a0924d5e89a9/) |
| CoBEVT   | 75.89      | 53.34        | 29.62           | [<img src="./imgs/download.png" alt="drawing" width="20"/>](https://seafile.cloud.uni-hannover.de/d/9d8db83dc2c54646b150/) | [<img src="./imgs/download.png" alt="drawing" width="20"/>](https://seafile.cloud.uni-hannover.de/d/b3780d91640a4d8dbf8a/) |


## Citation
```shell
@inproceedings{xu2023v2v4real,
  title={V2V4Real: A Real-world Large-scale Dataset for Vehicle-to-Vehicle Cooperative Perception},
  author={Xu, Runsheng and Xia, Xin and Li, Jinlong and Li, Hanzhao and Zhang, Shuo and Tu, Zhengzhong and Meng, Zonglin and Xiang, Hao and Dong, Xiaoyu and Song, Rui and Yu, Hongkai and Zhou, Bolei and Ma, Jiaqi},
  booktitle={The IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
  year={2023}
}
```

## Acknowledgment
This dataset belongs to the [OpenCDA ecosystem](https://arxiv.org/abs/2301.07325) family. The codebase is build upon [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD), which is the first Open Cooperative Detection framework for autonomous driving.