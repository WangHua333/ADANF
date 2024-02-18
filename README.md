# ADANF
This repo is the implementation of ADANF. 

### Install Dependencies
- Python 3.8
- Pytorch 1.13.1

```
pip install -r ./code/requirements.txt
```

### Dataset
The datasets
[LOL](https://daooshee.github.io/BMVC2018website/), and
[LOL-v2](https://github.com/flyywh/CVPR-2020-Semi-Low-Light)
can be downloaded form their offical website. 

### Pretrained Model
We provide the pre-trained models on [ [Baidu drive](https://pan.baidu.com/s/12ZLBIUS9t87mJkWIWZhXZA), Extraction code: s2a7 ] with the following settings:
+ LOLv1 with training config file `./confs/LOL-pc.yml`.
+ LOLV2-real with training config file `./confs/LOLv2-pc-real.yml`.
+ LOLV2-syn with training config file `./confs/LOLv2-pc-syn.yml`.

### Config Setting 
Before training and testing, you should change the paths to the dataset and pretrained model in the corresponding configuration files.

```python
#### datasets
datasets:
  train:
    root: # path to the dataset path

#### Test Settings
dataroot_unpaired # needed for testing with unpaired data
dataroot_GT # needed for testing with paired data
dataroot_LR # needed for testing with paired data
model_path # path to your checkpoint
```


### Test
To test the model with paired data
```bash
python test.py --opt your_config_path
```

To test the model with unpaired data
```bash
python test_unpaired.py --opt your_config_path -n results_folder_name
```
You can check the output in `../results`.
### Train
```bash
python train.py --opt your_config_path
```
You can check All logging files in the training process in  `../experiments`.

### Acknowledgments
This source code is mainly based on [LLFlow](https://github.com/wyf0912/LLFlow).