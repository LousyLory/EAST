#!/bin/bash

for i in {1..5}
do
	source activate EAST
	sbatch -p 1080ti-long --gres=gpu:2 --mem=200000 run_4_channels_${i}.sh
done
