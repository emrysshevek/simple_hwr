read -r -p "This will clear all datasets and require you to run generate-all-datasets.sh to recreate them. Are you sure? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
  cd prepare_font_data
  sh clear-data.sh
  cd ../prepare_IAM_Lines
  sh clear-data.sh
  cd ../prepare_online_data
  sh clear-data.sh
fi
