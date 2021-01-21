# dataset = ml2k | 30music | retailrocket
# teacher_type = nin | sas | bert

pre_path=/path/to/nas/search/folder

teacher_type=nin
dataset=ml2k

name=c32-emd
gpu=0

nohup python student_finetune.py \
  --name $name \
  --gpu $gpu \
  --preset finetune \
  --search_folder $pre_path \
  --search_teacher_type $teacher_type \
  --model_num_hidden 64 \
  --dataset $dataset \
  --dataset_type seq \
  --aux_store_root store/nas-$teacher_type/finetune-$dataset \
  > out/ft-$teacher_type-$dataset.log &