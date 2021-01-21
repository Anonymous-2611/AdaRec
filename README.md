# AdaRec: Scene-adaptive Knowledge Distillation for Sequential Recommendation via Differentiable Architecture Search

## Requirements

- python 3.8
- pytorch 1.7.1
- pyemd 0.5.1
- pandas 1.1.5
- numpy 1.19.2

## Usage

**Note-1**: To run directly, disable `nohup` and remove output redirection in `.sh` files.

**Note-2**: Before running the `.sh` file, please fill in the blank path parameters in some `.sh` files.

### step.0 - prepare datasets

1. Unzip datasets in `./datasets/raw-datas`
2. Run python files under `./datasets`

### step.1 - train teacher network

- Teacher-NextItNet

```sh
sh run_teacher_nin.sh
```

- Teacher-SASRec

```sh
sh run_teacher_sas.sh
```

- Teacher-BERT

```sh
sh run_teacher_bert.sh
```

### step.1.2 - finetune teacher-bert

Note that *Teacher-BERT* is trained by masked language model method, which is hard to distill, so we finetune *Teacher-BERT* model using auto-regressive method.

```sh
sh run_teacher_bert_finetune_seq.sh
```

### step.2 - do NAS search

```sh
sh run_nas_search.sh
```

### step.3 - do NAS finetune

```sh
sh run_nas_finetune.sh
```
