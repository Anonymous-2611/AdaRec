# dataset = ml2k | 30music | retailrocket

dataset=ml2k

name=b8

gpu=1

nohup python teacher_bert.py \
  --name $name \
  --gpu $gpu \
  --dataset $dataset \
  --dataset_type mask \
  --train_iter 500 \
  --preset bert-train \
  --aux_store_root store/pre-bert/$dataset \
  > out/pre-bert-$dataset.log &


