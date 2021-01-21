# dataset = ml2k | 30music | retailrocket
# teacher_type = nin | sas | bert

teacher_type=nin
dataset=ml2k
seq_teacher_folder=/path/to/teacher/folder

gpu_teacher=3
gpu_student=3

name=p-emb

gamma=0.5
beta=8.0

num_cell=4
num_node=3

train_epoch=200

nohup python student_search.py \
  -T $teacher_type -D $dataset \
  --name $name \
  --gpu_teacher $gpu_teacher \
  --gpu_student $gpu_student \
  --preset $teacher_type-search \
  --dataset_type seq \
  --dataset $dataset \
  --train_iter $train_epoch \
  --loader_train_batch_size 64 \
  --model_num_node $num_node \
  --model_num_cell $num_cell \
  --search_loss_gamma $gamma \
  --search_loss_beta $beta \
  --search_distill_loss emd \
  --search_teacher_folder $seq_teacher_folder \
  --aux_store_root store/nas-$teacher_type/search-$dataset \
  > out/$teacher_type-$dataset.log &
