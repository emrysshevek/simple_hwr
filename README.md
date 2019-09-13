# BYU ML Lab Deep Integration of LM into HWR

## 


### Dependances

numpy  
opencv 3  
pytorch  
cffi  
editdistance

Install this from the repo:
https://github.com/SeanNaren/warp-ctc


## Execution

### Prepare Font Data

``` bash
git clone https://github.com/cwig/prepare_font_data  
cd prepare_font_data  
bash run.sh  
cd ..  
python character_set.py prepare_font_data/training.json prepare_font_data/char_set.json
```

### Prepare IAM Data

After signing up for an IAM Database access account:

``` bash
git clone https://github.com/cwig/prepare_IAM_Lines   
cd prepare_IAM_Lines
sh download_IAM_data.sh  
python extract_all_words_lines.py  
cd ..  
python character_set.py prepare_IAM_Lines/raw_gts/lines/txt/training.json prepare_IAM_Lines/char_set.json  
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
