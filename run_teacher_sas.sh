# dataset = ml2k | 30music | retailrocket

dataset=ml2k

name=c128-b8

gpu=2

nohup python teacher_sas.py \
  --name $name \
  --gpu $gpu \
  --dataset $dataset \
  --dataset_type seq \
  --sas_hidden_units 128 \
  --sas_num_blocks 8 \
  --preset sas-train \
  --aux_store_root store/pre-sas/$dataset \
  > out/pre-sas-$dataset.log &


