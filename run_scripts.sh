#!/bin/bash


for file in slurm_scripts/*; do
    if [[ -f $file ]]; then
        echo "Putting $file on the queue"
        sbatch $file
    fi
done

#i=0
#while read line
#do
#    array[ $i ]="$line"
#    (( i++ ))
#done < <(ls -ls)
#
#echo ${array[1]}