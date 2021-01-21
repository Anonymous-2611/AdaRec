def if_none(src, v):
    if src is None:
        return v
    return src


def load_preset(args):
    # 1. load model preset
    # 2. load dataset preset
    args.seed = if_none(args.seed, 42)

    load_model_args(args)
    load_data_args(args)


def load_data_args(args):
    if args.dataset is None:
        return

    args.loader_generate_sub_session = if_none(args.loader_generate_sub_session, False)
    args.loader_train_batch_size = if_none(args.loader_train_batch_size, 128)
    args.loader_val_batch_size = if_none(args.loader_val_batch_size, 128)
    args.loader_test_batch_size = if_none(args.loader_test_batch_size, 128)
    # args.loader_num_items will be set by `dataloader` later

    if args.dataset_type == "mask":
        args.loader_num_aux_vocabs = 2  # + [PAD=0] + [MASK]
    else:  # args.dataset_type == "seq":
        args.loader_num_aux_vocabs = 1  # + [PAD=0]

    if args.dataset == "ml2k":
        args.data_folder = if_none(args.data_folder, "datasets/raw-datas/ml-2k/prepare")
        args.data_main = if_none(args.data_main, "dataset-len4-num4-min3.pkl")
        args.data_neg = if_none(args.data_neg, "neg-random-n100.pkl")

        args.loader_max_len = if_none(args.loader_max_len, 50)
        if args.dataset_type == "mask":
            args.loader_mask_prob = if_none(args.loader_mask_prob, 0.5)

        if args.loader_generate_sub_session:
            args.train_log_every = if_none(args.train_log_every, 200)
        else:
            args.train_log_every = if_none(args.train_log_every, 20)
        return

    if args.dataset == "30music":
        args.data_folder = if_none(args.data_folder, "datasets/raw-datas/30music/prepare")
        args.data_main = if_none(args.data_main, "dataset-len3-num5.pkl")
        args.data_neg = if_none(args.data_neg, "neg-random-n100.pkl")

        args.loader_max_len = if_none(args.loader_max_len, 20)
        if args.dataset_type == "mask":
            args.loader_mask_prob = if_none(args.loader_mask_prob, 0.4)

        if args.loader_generate_sub_session:
            args.train_log_every = if_none(args.train_log_every, 2000)
        else:
            args.train_log_every = if_none(args.train_log_every, 350)
        return

    if args.dataset == "retailrocket":
        args.data_folder = if_none(args.data_folder, "datasets/raw-datas/retailrocket/prepare")
        args.data_main = if_none(args.data_main, "dataset-len4-num5.pkl")
        args.data_neg = if_none(args.data_neg, "neg-random-n100.pkl")

        args.loader_max_len = if_none(args.loader_max_len, 10)
        if args.dataset_type == "mask":
            args.loader_mask_prob = if_none(args.loader_mask_prob, 0.4)  # [to be filled]

        if args.loader_generate_sub_session:
            args.train_log_every = if_none(args.train_log_every, 1000)
        else:
            args.train_log_every = if_none(args.train_log_every, 200)
        return


def load_model_args(args):
    # Load args from preset is a two-phase procedure.
    #
    #   1. Load main model schema.
    #   2. Load task.
    #
    # args.preset is something like `{schema}-{task}` or `{task}`
    #
    # `{schema}` is namely the teacher model
    #   - BERT
    #   - nin -> NextItNet
    #   - sas -> SASRec
    # `{task}` is what the job to do now
    #   - train    / Train teacher model (or baseline model) from scratch.
    #   - search   / Using NAS and teacher model to search (sub-)optimal student model.
    #   - finetune / [EXCLUSIVE] Load student model and finetune till converge, searched architecture is also extracted.
    #   - distill  / Without NAS, using normal distillation method to train small but isomorphic student model.
    #   > [EXCLUSIVE] means `{task}`, see examples below.
    #
    # e.g.  args.preset = "bert-train"
    #       args.preset = "bert-search"
    #       args.preset = "bert-distill"
    #
    #       args.preset = "nin-train"
    #       args.preset = "nin-search"
    #       args.preset = "nin-distill"
    #
    #       args.preset = "sas-train"
    #       args.preset = "sas-search"
    #       args.preset = "sas-distill"
    #
    #       args.preset = "finetune"
    #       args.preset = "augment"
    #
    #

    args.aux_eval_ks = if_none(args.aux_eval_ks, [1, 5, 10, 20])
    args.train_grad_clip_norm = if_none(args.train_grad_clip_norm, 5.0)  # DO WE NEED GRAD CLIP FOR TEACHER MODEL?

    if args.preset is None:
        return

    # ---------------- [MODEL SCHEMA SETTINGS] ----------------
    if args.preset.startswith("bert"):
        args.bert_dropout = if_none(args.bert_dropout, 0.1)
        args.bert_use_eps = if_none(args.bert_use_eps, False)  # x = x + F(x) / x = x + eps * F(x)
        args.bert_num_blocks = if_none(args.bert_num_blocks, 8)
        args.bert_hidden_units = if_none(args.bert_hidden_units, 128)
        args.bert_num_heads = if_none(args.bert_num_heads, 4)
        # let d <= 32 for each head, d_{each} = args.bert_hidden_units / args.bert_num_heads

    if args.preset.startswith("nin"):
        args.nin_num_blocks = if_none(args.nin_num_blocks, 8)
        args.nin_block_dilations = if_none(args.nin_block_dilations, [1, 4])
        # 8 * [1, 4] => 16 blocks
        # using `4`s' output as teacher intermediate representation
        # that's 8 hidden output
        args.nin_hidden_units = if_none(args.nin_hidden_units, 256)
        args.nin_kernel_size = if_none(args.nin_kernel_size, 3)
        args.nin_use_eps = if_none(args.nin_use_eps, True)

    if args.preset.startswith("sas"):
        args.sas_num_blocks = if_none(args.sas_num_blocks, 8)
        # using EVERY layers' output as teacher intermediate representation
        args.sas_hidden_units = if_none(args.sas_hidden_units, 128)
        args.sas_num_heads = if_none(args.sas_num_heads, 4)
        args.sas_dropout = if_none(args.sas_dropout, 0.2)
        args.sas_use_eps = if_none(args.sas_use_eps, True)

    # ---------------- [MODEL TASK SETTINGS] ----------------
    # Task - 1: train
    if args.preset.endswith("train"):  # train Teacher net
        if args.preset.startswith("bert"):
            args.train_lr = if_none(args.train_lr, 0.005)  # Lamb
            args.train_lr_decay_step = if_none(args.train_lr_decay_step, 10)
            args.train_lr_decay_gamma = if_none(args.train_lr_decay_gamma, 1.0)
            args.train_wd = if_none(args.train_wd, 0.01)
            args.train_iter = if_none(args.train_iter, 200)
        if args.preset.startswith("sas"):
            args.train_lr = if_none(args.train_lr, 0.005)  # Lamb
            args.train_lr_decay_step = if_none(args.train_lr_decay_step, 10)
            args.train_lr_decay_gamma = if_none(args.train_lr_decay_gamma, 1.0)
            args.train_wd = if_none(args.train_wd, 0.01)
            args.train_iter = if_none(args.train_iter, 50)
        if args.preset.startswith("nin"):
            args.train_lr = if_none(args.train_lr, 0.001)  # Adam
            args.train_lr_decay_step = if_none(args.train_lr_decay_step, 10)
            args.train_lr_decay_gamma = if_none(args.train_lr_decay_gamma, 0.9)
            args.train_wd = if_none(args.train_wd, 1e-7)
            args.train_iter = if_none(args.train_iter, 50)

    # Task - 2: [NAS] search student network architecture
    if args.preset.endswith("search"):  # use teacher model config
        if args.preset.startswith("bert"):
            args.search_teacher_layers = args.bert_num_blocks
            args.search_teacher_hidden = args.bert_hidden_units
        if args.preset.startswith("nin"):
            args.search_teacher_layers = args.nin_num_blocks
            args.search_teacher_hidden = args.nin_hidden_units
        if args.preset.startswith("sas"):
            args.search_teacher_layers = args.sas_num_blocks
            args.search_teacher_hidden = args.sas_hidden_units

        args.search_loss_gamma = if_none(args.search_loss_gamma, 0.5)
        args.search_loss_gamma_decay = if_none(args.search_loss_gamma_decay, 0.00)
        args.search_loss_beta = if_none(args.search_loss_beta, 8.0)

        args.search_temperature = if_none(args.search_temperature, 1)
        args.search_temperature_decay_rate = if_none(args.search_temperature_decay_rate, 0.8)
        args.search_temperature_decay_epochs = if_none(args.search_temperature_decay_epochs, 5)

        args.search_distill_loss = if_none(args.search_distill_loss, "hierarchical")  # hierarchical | emd | ada
        args.search_hierarchical_select_method = if_none(args.search_hierarchical_select_method, "left-jump")

        args.model_num_cell = if_none(args.model_num_cell, 4)
        args.model_num_node = if_none(args.model_num_node, 3)
        args.model_dropout = if_none(args.model_dropout, 0.1)
        if args.preset.startswith("nin"):
            args.model_num_hidden = if_none(args.model_num_hidden, 64)
        if args.preset.startswith("sas"):
            args.model_num_hidden = if_none(args.model_num_hidden, 32)
        if args.preset.startswith("bert"):
            args.model_num_hidden = if_none(args.model_num_hidden, 32)

        args.train_model_lr = if_none(args.train_model_lr, 5e-3)
        args.train_model_lr_decay_step = if_none(args.train_model_lr_decay_step, 10)
        args.train_model_lr_decay_gamma = if_none(args.train_model_lr_decay_gamma, 1.0)  # no decay
        args.train_model_wd = if_none(args.train_model_wd, 5e-4)

        args.train_alpha_lr = if_none(args.train_alpha_lr, 2e-5)
        args.train_alpha_lr_decay_step = if_none(args.train_alpha_lr_decay_step, 10)
        args.train_alpha_lr_decay_gamma = if_none(args.train_alpha_lr_decay_gamma, 1.0)  # no decay
        args.train_alpha_wd = if_none(args.train_alpha_wd, 1e-4)

        args.train_iter = if_none(args.train_iter, 200)

    # Task - 3: [Distillation] Big model into small model, normal distillation
    if args.preset.endswith("distill"):
        if args.preset.startswith("nin"):
            args.nin_student_hidden_units = if_none(args.nin_student_hidden_units, 64)
            args.nin_student_num_blocks = if_none(args.nin_student_num_blocks, 4)
            args.nin_student_block_dilations = if_none(args.nin_student_block_dilations, [1, 4])
        if args.preset.startswith("sas"):
            args.sas_student_hidden_units = if_none(args.sas_student_hidden_units, 32)
            args.sas_student_num_heads = if_none(args.sas_student_num_heads, 1)
            args.sas_student_num_blocks = if_none(args.sas_student_num_blocks, 4)
        if args.preset.startswith("bert"):
            args.bert_student_hidden_units = if_none(args.bert_student_hidden_units, 32)
            args.bert_student_num_heads = if_none(args.bert_student_num_heads, 1)
            args.bert_student_num_blocks = if_none(args.bert_student_num_blocks, 4)

        args.distill_loss_gamma = if_none(args.distill_loss_gamma, 0.5)
        args.distill_loss_gamma_decay = if_none(args.distill_loss_gamma_decay, 0.0)  # no decay [preset]

        if args.preset.startswith("nin"):
            args.train_lr = if_none(args.train_lr, 0.001)  # Adam
            args.train_lr_decay_step = if_none(args.train_lr_decay_step, 10)
            args.train_lr_decay_gamma = if_none(args.train_lr_decay_gamma, 0.9)
            args.train_wd = if_none(args.train_wd, 1e-7)
            args.train_iter = if_none(args.train_iter, 50)
        if args.preset.startswith("sas"):
            args.train_lr = if_none(args.train_lr, 0.005)  # Lamb
            args.train_lr_decay_step = if_none(args.train_lr_decay_step, 10)
            args.train_lr_decay_gamma = if_none(args.train_lr_decay_gamma, 0.9)
            args.train_wd = if_none(args.train_wd, 0.01)
            args.train_iter = if_none(args.train_iter, 50)
        if args.preset.startswith("bert"):
            args.train_lr = if_none(args.train_lr, 0.005)  # Lamb
            args.train_lr_decay_step = if_none(args.train_lr_decay_step, 10)
            args.train_lr_decay_gamma = if_none(args.train_lr_decay_gamma, 0.9)
            args.train_wd = if_none(args.train_wd, 0.01)
            args.train_iter = if_none(args.train_iter, 200)

    # Task - 4: [NAS] finetune student network
    if args.preset.endswith("finetune"):
        args.model_num_hidden = if_none(args.model_num_hidden, 64) # may change

        args.model_num_cell = if_none(args.model_num_cell, 4)
        args.model_num_node = if_none(args.model_num_node, 3)
        args.model_dropout = if_none(args.model_dropout, 0.1)

        args.train_lr = if_none(args.train_lr, 0.0001)
        args.train_lr_decay_step = if_none(args.train_lr_decay_step, 10)
        args.train_lr_decay_gamma = if_none(args.train_lr_decay_gamma, 0.9)
        args.train_wd = if_none(args.train_wd, 5e-4)
        args.train_iter = if_none(args.train_iter, 100)

    # Task - 5: [NAS] augment student network
    if args.preset.endswith("augment"):
        args.model_num_hidden = if_none(args.model_num_hidden, 64) # may change

        args.model_num_cell = if_none(args.model_num_cell, 4)
        args.model_num_node = if_none(args.model_num_node, 3)
        args.model_dropout = if_none(args.model_dropout, 0.1)

        args.train_lr = if_none(args.train_lr, 0.001)
        args.train_lr_decay_step = if_none(args.train_lr_decay_step, 10)
        args.train_lr_decay_gamma = if_none(args.train_lr_decay_gamma, 1.0)
        args.train_wd = if_none(args.train_wd, 5e-4)
        args.train_iter = if_none(args.train_iter, 100)
