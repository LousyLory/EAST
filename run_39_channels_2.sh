#!/bin/bash
#SBATCH --mem=250000

#i="5"
reg_constant=2.0
#rm -r tmp/east_icdar2015_resnet_v1_50_rbox${i}
#cp -r tmp/east_icdar2015_resnet_v1_50_rbox tmp/east_icdar2015_resnet_v1_50_rbox${i}
#for idx in {1..20}
#do
#python -W ignore evalEAST.py --model_name=${i}
python -u multigpu_train_39_2.py --regC=${reg_constant} --gpu_list=0,1 --input_size=512 --batch_size_per_gpu=16 --checkpoint_path=tmp/east_icdar2015_resnet_v1_39_channel_2/ \
--text_scale=512 --restore=true --training_data_path=Data/cropped_img_train/ --geometry=RBOX --learning_rate=0.0005 --num_readers=24 \
--pretrained_model_path=tmp/resnet_v1_50.ckpt 2>&1 | tee outfile.txt
#done
