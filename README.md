# BYU ML Lab Deep Integration of LM into HWR

## Prerequisites

To work with this project effectively, supercomputer access is highly
recommended.  Sign up [here](https://rc.byu.edu/account/create/).

Next, request group access from Taylor Archibald.

After logging in, install Anaconda 3:

``` sh
cd /tmp
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-5.2.0-Linux-x86_64.sh
```

## Environment

Install this from the repo:
https://github.com/SeanNaren/warp-ctc

### Configuration

The configuration files located in `config/` are used to create environments
within which to 

### Activating Environment

``` sh
conda env create -f environment.yaml
conda activate hwr
```

## Execution

### Downloading/Preparing Datasets

Ensure that you have an IAM Database access account ([register](http://www.fki.inf.unibe.ch/DBs/iamDB/iLogin/index.php)), then:

``` bash
cd data
sh generate-all-datasets.sh
```

### Train

To train, run `train.py` with one of the configurations:

``` sh
python train.py sample_config.json
```
or if you want to run a lot of epochs (kill whenever)
``` sh
python train.py sample_config_iam.json
```

### Recognize

``` sh
python recognize.py sample_config.json prepare_font_data/output/0.png
```
or 

``` sh
python recognize.py sample_config_iam.json prepare_IAM_Lines/lines/r06/r06-000/r06-000-00.png
```
