#!/bin/bash

declare -a arr=("D0117-5755035" "D0117-5755036" "D5005-5028052" "D5005-5028054" "D5005-5028097" "D5005-5028100" "D5005-5028102" "D5005-5028149")

## now loop through the above array
for i in "${arr[@]}"
do
   echo "$i"
   ./run2.sh "$i"
   #sbatch -p titanx-long --gres=gpu:2 --mem=200000 ./run2.sh "$i"
   # or do whatever with individual element of the array
done
