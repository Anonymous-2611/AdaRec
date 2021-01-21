dataset=ml2k
masked_model_path=/path/to/MLM-bert

name=seq
gpu=0

nohup python teacher_bert_finetune_seq.py \
  --name $name \
  --gpu $gpu \
  --dataset $dataset \
  --dataset_type seq \
  --loader_train_batch_size 64 \
  --resume $masked_model_path \
  --train_iter 200 \
  --train_lr 2e-5 \
  --train_wd 1e-7 \
  --preset bert-train \
  --aux_store_root store/seq-bert/$dataset \
  > out/transfer-bert-$dataset.log &


