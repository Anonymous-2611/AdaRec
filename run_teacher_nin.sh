# dataset = ml2k | 30music | retailrocket

dataset=ml2k

name=c256-b8

gpu=6

nohup python teacher_nin.py \
  --name $name \
  --gpu $gpu \
  --dataset $dataset \
  --dataset_type seq \
  --nin_hidden_units 256 \
  --nin_num_blocks 8 \
  --preset nin-train \
  --aux_store_root store/pre-nin/$dataset \
  > out/pre-nin-$dataset.log &


