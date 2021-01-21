import argparse


"""
==========================
  All options' parsers.
==========================

Except for global setting (e.g. name/gpu/seed/seed/etc.), options are designed like `{scope}_{option}`.

Note: `scope` is not something related to real code, they just make 
options easy to arrange and look pretty when printing.
"""


def add_common_args(parser):
    parser.add_argument("--name", type=str, default="NoName")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", type=str, default=None, help="Use preset template, see presets.py for detail.")

    parser.add_argument("--resume", type=str, default=None, help="path/to/FOLDER/from/where/you/want/to/resume")

    parser.add_argument("--aux_store_root", type=str, help="/path/to/store", default=None)
    parser.add_argument("--aux_console_output", action="store_true", help="Print logger info to console")
    parser.add_argument("--aux_eval_ks", nargs="+", type=int, default=None, help="ks for Metric@k")
    return parser


def add_dataset_args(parser):
    parser.add_argument("--dataset", type=str, default=None, help="Preset for dataset and dataloader.")
    # mask : masked language model data loader
    # seq  : normal next item data loader
    parser.add_argument("--dataset_type", type=str, default="mask", choices=["mask", "seq"])
    parser.add_argument("--data_folder", type=str, default=None)
    parser.add_argument("--data_main", type=str, default=None)
    parser.add_argument("--data_neg", type=str, default=None)

    parser.add_argument("--loader_generate_sub_session", action="store_true", default=None)
    parser.add_argument("--loader_train_batch_size", type=int, default=None)
    parser.add_argument("--loader_val_batch_size", type=int, default=None)
    parser.add_argument("--loader_test_batch_size", type=int, default=None)
    parser.add_argument("--loader_mask_prob", type=float, default=None)
    parser.add_argument("--loader_max_len", type=int, default=None)
    parser.add_argument("--loader_num_items", type=int, default=None, help="Number of real items, without PAD and MASK")
    parser.add_argument("--loader_num_aux_vocabs", type=int, default=None, help="+1 when seq, +2 when mask")
    return parser


def add_bert_args(parser):
    parser.add_argument("--bert_hidden_units", type=int, default=None, help="Size of hidden vectors (d_model)")
    parser.add_argument("--bert_num_blocks", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--bert_num_heads", type=int, default=None, help="Number of heads for multi-attention")
    parser.add_argument("--bert_dropout", type=float, default=None, help="Dropout probability")
    parser.add_argument("--bert_use_eps", action="store_true", default=None, help="Use x_{i+1} = x_{i} + eps*F(x_{i})")
    return parser


def add_nin_args(parser):
    parser.add_argument("--nin_num_blocks", type=int, default=None)
    parser.add_argument("--nin_block_dilations", nargs="+", type=int, default=None)
    parser.add_argument("--nin_hidden_units", type=int, default=None)
    parser.add_argument("--nin_kernel_size", type=int, default=None)
    parser.add_argument("--nin_use_eps", action="store_true", default=None)

    return parser


def add_sas_args(parser):
    parser.add_argument("--sas_num_blocks", type=int, default=None)
    parser.add_argument("--sas_hidden_units", type=int, default=None)
    parser.add_argument("--sas_num_heads", type=int, default=None)
    parser.add_argument("--sas_dropout", type=float, default=None)
    parser.add_argument("--sas_use_eps", action="store_true", default=None)
    return parser


def add_student_model_args(parser):
    parser.add_argument("--model_dropout", type=float, default=None, help="Dropout probability in cells.")
    parser.add_argument("--model_num_hidden", type=int, default=None, help="Hidden in cells.")
    parser.add_argument("--model_num_cell", type=int, default=None, help="Number of cells.")
    parser.add_argument("--model_num_node", type=int, default=None, help="Number of intermediate node in a cell.")
    return parser


def add_training_args(parser, is_search=False):
    parser.add_argument("--train_iter", type=int, default=None, help="Number of epochs for training")
    parser.add_argument("--train_log_every", type=int, default=None, help="Log every T*b.")
    parser.add_argument("--train_grad_clip_norm", type=float, default=None, help="Clip gradient by norm.")

    if not is_search:  # single model training, maybe teacher model or finetune student model
        parser.add_argument("--train_lr", type=float, default=None, help="Learning rate")
        parser.add_argument("--train_lr_decay_step", type=int, default=None, help="Decay step for StepLR")
        parser.add_argument("--train_lr_decay_gamma", type=float, default=None, help="Gamma for StepLR")
        parser.add_argument("--train_wd", type=float, default=None, help="l2 regularization")
    else:
        parser.add_argument("--train_model_lr", type=float, default=None, help="Initial learning rate for model")
        parser.add_argument("--train_model_lr_decay_step", type=int, default=None)
        parser.add_argument("--train_model_lr_decay_gamma", type=float, default=None)
        parser.add_argument("--train_model_wd", type=float, default=None, help="l2 regularization for model")

        parser.add_argument("--train_alpha_lr", type=float, default=None, help="Initial learning rate for alpha")
        parser.add_argument("--train_alpha_lr_decay_step", type=int, default=None)
        parser.add_argument("--train_alpha_lr_decay_gamma", type=float, default=None)
        parser.add_argument("--train_alpha_wd", type=float, default=None, help="l2 regularization for alpha")
    return parser


def add_gru4rec_args(parser):
    parser.add_argument("--gru_num_layers", type=int, default=None)
    parser.add_argument("--gru_hidden_units", type=int, default=None)
    parser.add_argument("--gru_dropout", type=float, default=None)
    return parser


def add_caser_args(parser):
    parser.add_argument("--caser_hidden_units", type=int, default=None)
    parser.add_argument("--caser_dropout", type=float, default=None)
    parser.add_argument("--caser_num_hf", type=int, default=None)
    parser.add_argument("--caser_num_vf", type=int, default=None)
    parser.add_argument("--caser_hf_size", type=int, nargs="+", default=None)
    return parser


def gru4rec_parser():
    # Baseline: GRU4Rec
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)

    parser = add_common_args(parser)
    parser = add_dataset_args(parser)
    parser = add_gru4rec_args(parser)
    parser = add_training_args(parser, is_search=False)

    return parser


def caser_parser():
    # Baseline: Caser
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)

    parser = add_common_args(parser)
    parser = add_dataset_args(parser)
    parser = add_caser_args(parser)
    parser = add_training_args(parser, is_search=False)

    return parser


def bert4rec_parser():
    # Train Teacher-BERT network
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)

    parser = add_common_args(parser)
    parser = add_dataset_args(parser)
    parser = add_bert_args(parser)
    parser = add_training_args(parser, is_search=False)

    return parser


def nextitnet_parser():
    # Train Teacher-NextItNet network
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)

    parser = add_common_args(parser)
    parser = add_dataset_args(parser)
    parser = add_nin_args(parser)
    parser = add_training_args(parser, is_search=False)

    return parser


def nextitnet_distill_parser():
    # Distill NextItNet into NextItNet
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--nin_student_hidden_units", type=int, default=None)
    parser.add_argument("--nin_student_num_blocks", type=int, default=None)
    parser.add_argument("--nin_student_block_dilations", nargs="+", type=int, default=None)

    parser = add_common_args(parser)
    parser = add_dataset_args(parser)
    parser = add_nin_args(parser)

    parser = add_training_args(parser, is_search=False)
    parser.add_argument("--distill_loss_gamma", type=float, default=None, help="Trade off between CE and KD.")
    parser.add_argument("--distill_loss_gamma_decay", type=float, default=None, help="Gamma decay every.")
    parser.add_argument("--distill_teacher_folder", type=str, default=None)

    # use EMD distillation method
    return parser


def sasrec_distill_parser():
    # Distill SASRec into SASRec
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sas_student_num_heads", type=int, default=None)
    parser.add_argument("--sas_student_hidden_units", type=int, default=None)
    parser.add_argument("--sas_student_num_blocks", type=int, default=None)

    parser = add_common_args(parser)
    parser = add_dataset_args(parser)
    parser = add_sas_args(parser)

    parser = add_training_args(parser, is_search=False)
    parser.add_argument("--distill_loss_gamma", type=float, default=None, help="Trade off between CE and KD.")
    parser.add_argument("--distill_loss_gamma_decay", type=float, default=None, help="Gamma decay every.")
    parser.add_argument("--distill_teacher_folder", type=str, default=None)

    # use EMD distillation method
    return parser


def bert4rec_distill_parser():
    # Distill Bert4rec into Bert4rec
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--bert_student_num_heads", type=int, default=None)
    parser.add_argument("--bert_student_hidden_units", type=int, default=None)
    parser.add_argument("--bert_student_num_blocks", type=int, default=None)

    parser = add_common_args(parser)
    parser = add_dataset_args(parser)
    parser = add_bert_args(parser)

    parser = add_training_args(parser, is_search=False)
    parser.add_argument("--distill_loss_gamma", type=float, default=None, help="Trade off between CE and KD.")
    parser.add_argument("--distill_loss_gamma_decay", type=float, default=None, help="Gamma decay every.")
    parser.add_argument("--distill_teacher_folder", type=str, default=None)

    # use EMD distillation method
    return parser


def sasrec_parser():
    # Train Teacher-SASRec network
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)

    parser = add_common_args(parser)
    parser = add_dataset_args(parser)
    parser = add_sas_args(parser)
    parser = add_training_args(parser, is_search=False)

    return parser


def student_search_preset_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", type=str, required=True, help="teacher's type")
    parser.add_argument("-D", type=str, required=True, help="dataset's name")
    return parser


def student_search_parser(teacher_type):
    # Search student network architecture using pretrained Teacher model

    teacher_type = teacher_type.strip().lower()
    assert teacher_type in ["bert", "nin", "sas"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_teacher", type=int, default=0)
    parser.add_argument("--gpu_student", type=int, default=0)

    # Student-Search
    parser.add_argument("--search_temperature", type=float, default=None, help="Initial gumbel sampling temperature.")
    parser.add_argument("--search_temperature_decay_rate", type=float, default=None, help="Temperature decay rate.")
    parser.add_argument("--search_temperature_decay_epochs", type=float, default=None, help="Temperature decay every.")

    parser.add_argument("--search_teacher_folder", type=str, default=None)
    parser.add_argument("--search_teacher_layers", type=int, default=None, help="Number of layers in teacher network.")
    parser.add_argument("--search_teacher_hidden", type=int, default=None, help="Hidden units in teacher network.")
    parser.add_argument("--search_distill_loss", type=str, default=None, help="KD loss type.")  # hierarchical|emd|ada
    parser.add_argument("--search_hierarchical_select_method", type=str, default=None, help="Hierarchical KD method.")

    parser.add_argument("--search_loss_gamma", type=float, default=None, help="Trade off between CE and KD.")
    parser.add_argument("--search_loss_gamma_decay", type=float, default=None, help="Gamma decay every.")
    parser.add_argument("--search_loss_beta", type=float, default=None, help="Loss factor for model efficiency.")

    parser = add_common_args(parser)
    parser = add_dataset_args(parser)
    parser = add_training_args(parser, is_search=True)
    parser = add_student_model_args(parser)

    if teacher_type == "bert":
        parser = add_bert_args(parser)
    elif teacher_type == "nin":
        parser = add_nin_args(parser)
    elif teacher_type == "sas":
        parser = add_sas_args(parser)

    return parser


def student_finetune_parser():
    # Finetune student network architecture after architecture is generated
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--search_folder", type=str, default=None)
    parser.add_argument("--search_teacher_type", type=str, default=None)

    parser = add_common_args(parser)
    parser = add_dataset_args(parser)
    parser = add_student_model_args(parser)
    parser = add_training_args(parser, is_search=False)

    return parser


def student_augment_parser():
    # Augment student network
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=int, default=0)

    # e.g. Using 30music's alpha to train ml2k
    parser.add_argument("--augment_source_folder", type=str, default=None)  # get 30music's alpha (arch)
    parser.add_argument("--augment_target_folder", type=str, default=None)  # get ml2k's embedding and linear
    parser = add_common_args(parser)
    parser = add_dataset_args(parser)
    parser = add_student_model_args(parser)
    parser = add_training_args(parser, is_search=False)

    return parser


def student_augment_preset_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", type=str, required=True, help="teacher's type")
    parser.add_argument("-D_src", type=str, required=True, help="from which alpha file is searched")
    return parser


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def empty_parser():
    return argparse.ArgumentParser()
