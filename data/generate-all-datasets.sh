#!/bin/bash

cd prepare_font_data
bash run.sh

# No need to create the char_set because it is not used when training

cd ..
cd prepare_IAM_Lines
bash run.sh
