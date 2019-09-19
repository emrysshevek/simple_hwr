#!/bin/bash
read -r -p "This will generate all datasets. Make sure you are using the hwr Anaconda environment before you begin.  If there is an error with the script, run clear-all-datasets.sh.  Are you sure you want to continue? [y/N] " response

cd prepare_font_data
bash download_and_setup.sh

# No need to create the char_set because it is not used when training

cd ../prepare_IAM_Lines
bash run.sh

cd ../prepare_online_data
bash download.sh
