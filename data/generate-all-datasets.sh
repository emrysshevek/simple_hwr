#!/bin/bash

cd prepare_font_data
bash download_and_setup.sh

# No need to create the char_set because it is not used when training

cd ../prepare_IAM_Lines
bash run.sh

cd ../prepare_online_data
bash download.sh
