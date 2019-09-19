#!/bin/bash

read -r -p "This will clear all datasets and require you to run generate-all-datasets.sh to recreate them. Are you sure? [y/N] " response
if [[ "$response" =~ ^([yY][eE][sS]|[yY])+$ ]]
then
  cd ..
  git clean -xfd data/
fi
