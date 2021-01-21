import logging
import pickle
import shutil
import time
from pathlib import Path

import torch
from torch import optim, nn

from common import setup_folder, setup_env
from options import student_finetune_parser
from presets import load_preset

from loader.sequence_loader import SequenceLoaderProvider

from models.nas.kd_finetune import StudentFinetune
from models.others.lamb_optim import Lamb

from utils.logger import setup_logger
from utils.metrics import SRSMetric, AverageMeter
from utils.model_stat import get_model_complexity_info
from utils.others import load_state_dict
from utils.printer import dict_to_logger, output_alpha_to_logger


def create_state_dict():
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict()}


def need_to_log(num_iter):
    return (num_iter + 1) % args.train_log_every == 0


def create_model_and_load_parameter():
    pth = load_state_dict(Path(args.search_folder).joinpath("best-model.pth"))

    epoch = pth["epoch"]
    alpha = load_alpha_from_file(epoch)
    alpha_arch = alpha["arch"]
    output_alpha_to_logger(alpha_arch)
    _model = StudentFinetune(args, alpha_arch)

    pretrained_full = pth["state_dict"]["model"]
    param_names = _model.state_dict().keys()

    filtered_params = {k: v for k, v in pretrained_full.items() if k in param_names}
    emb_params = {k: v for k, v in filtered_params.items() if "embedding" in k}
    net_params = {k: v for k, v in filtered_params.items() if "cell" in k}
    out_params = {k: v for k, v in filtered_params.items() if "out" in k}

    # _model.load_state_dict(filtered_params)
    _model.load_state_dict(emb_params, strict=False)
    _model.load_state_dict(net_params, strict=False)
    _model.load_state_dict(out_params, strict=False)

    return _model


def load_alpha_from_file(epoch):
    search_folder = Path(args.search_folder)
    alpha_file = search_folder.joinpath("alpha-final").joinpath("epoch-{:05d}.pickle".format(epoch))

    with alpha_file.open("rb") as f:
        alpha_dict = pickle.load(f)
    logging.info("==> Alpha loaded from `{}`".format(alpha_file))

    to_save = folder.joinpath("alpha.pickle")
    shutil.copy(alpha_file, to_save)
    logging.info("==> Alpha saved to `alpha.pickle`")

    return alpha_dict


def do_validate(epoch=None):
    logging.info("[VALIDATE]")
    model.eval()

    meter = SRSMetric(k_list=args.aux_eval_ks)
    meter.setup_and_clean()

    with torch.no_grad():
        for batch in iter(val_loader):
            batch = [x.cuda() for x in batch]
            seqs, candidates = batch

            scores = model(seqs)  # [B, L, num_item]
            scores = scores[:, -1, :]  # [B, num_item]
            scores = scores.gather(1, candidates)  # [B, 1+num_sample]

            meter.submit(scores.cpu(), [[0]] * len(scores))

    meter.calc()
    meter.output_to_logger()

    if epoch is not None:
        save_data = {"state_dict": create_state_dict(), "epoch": epoch}
        torch.save(save_data, folder.joinpath("checkpoint.pth"))

    return round(meter.mrr[5], 4), round(meter.hit[5], 4)


def do_train(epoch):
    logging.info("-" * 30)
    logging.info(f"[TRAIN] Epoch: {epoch + 1} / {args.train_iter}")
    logging.info("\tLearning rate: {:.10f}".format(optimizer.param_groups[0]["lr"]))

    model.train()
    loss_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # non-zero elements in `labels` are learning targets

    for batch_idx, batch in enumerate(train_loader):
        batch = [x.cuda() for x in batch]
        training_seq, label_seq = batch

        optimizer.zero_grad()
        logits = model(training_seq)  # [B, L, #item]
        logits = logits.view(-1, logits.size(-1))  # [B*L, #item]

        label_seq = label_seq.view(-1)  # [B*L]
        loss = criterion(logits, label_seq)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip_norm)
        optimizer.step()

        loss_meter.update(loss.item())

        if need_to_log(batch_idx):
            cur, total, loss = batch_idx + 1, len(train_loader), loss_meter.avg
            logging.info("\tStep: {:5d} / {:5d}\tLoss: {:.4f}".format(cur, total, loss))


def start():
    best_mrr5, best_hit5 = do_validate()
    logging.info(">>>>> at epoch O better (MRR@5, HIT@5) = ({:.4f}, {:.4f}) <<<<< ".format(best_mrr5, best_hit5))
    save_data = {"state_dict": create_state_dict(), "epoch": 0}
    torch.save(save_data, folder.joinpath("best-model.pth"))

    for epoch in range(args.train_iter):
        tic = time.time()

        do_train(epoch)
        cur_mrr5, cur_hit5 = do_validate(epoch)
        lr_scheduler.step()

        if cur_mrr5 > best_mrr5 or cur_hit5 > best_hit5:
            best_mrr5 = max(best_mrr5, cur_mrr5)
            best_hit5 = max(best_hit5, cur_hit5)

            s = ">>>>> at epoch {:5d} better ".format(epoch)
            s += "(MRR@5, HIT@5) = ({:.4f}, {:.4f}) <<<<< ".format(cur_mrr5, cur_hit5)
            logging.info(s)

            save_data = {"state_dict": create_state_dict(), "epoch": epoch}
            torch.save(save_data, folder.joinpath("best-model.pth"))
        toc = time.time()
        logging.info("Iter: {} / {} finish. Time: {:.2f} min".format(epoch + 1, args.train_iter, (toc - tic) / 60))


def log_model_info():
    def input_maker(shape):
        return {"input_seqs": torch.randint(1, data_provider.num_token, (1, *shape), device="cuda")}

    input_shape = (data_provider.max_len,)
    MACs, params = get_model_complexity_info(
        model, input_shape, input_constructor=input_maker, print_per_layer_stat=False, as_strings=False
    )
    logging.info("Model MACs: {:.2f} M".format(MACs / 1024.0 / 1024.0))
    logging.info("Model params: {:.2f} M".format(params / 1024.0 / 1024.0))

    embedding_part = model.net.embedding
    emb_MACs, emb_params = get_model_complexity_info(
        embedding_part, input_shape, input_constructor=input_maker, print_per_layer_stat=False, as_strings=False
    )

    linear_part = model.net.out
    hidden_shape = (data_provider.max_len, args.model_num_hidden)
    linear_MACs, linear_params = get_model_complexity_info(
        linear_part, hidden_shape, print_per_layer_stat=False, as_strings=False
    )

    logging.info("Just net MACs: {:.2f} K".format((MACs - emb_MACs - linear_MACs) / 1024.0))
    logging.info("Just net params: {:.2f} K".format((params - emb_params - linear_params) / 1024.0))


if __name__ == "__main__":
    is_testing = False
    if is_testing:
        teacher_type, dataset = "nin", "ml2k"
        pre_path = "/path/to/nas/search/folder"
        parser = student_finetune_parser()
        cmd = (
            # -----[GLOBAL]-----
            "--name ft-ml2k "
            "--gpu 6 "
            f"--search_folder {pre_path} "
            f"--search_teacher_type {teacher_type} "
            f"--dataset {dataset} "
            "--dataset_type seq "
            "--model_num_hidden 64 "
            # -----[AUXILIARY]-----
            "--preset finetune "
            "--aux_store_root store/1.TEST "
            "--aux_console_output"
        )
        cmd = cmd.strip()
        args = parser.parse_args(cmd.split(" "))
    else:  # not testing
        parser = student_finetune_parser()
        args = parser.parse_args()

    store_folder = setup_folder(store_root=args.aux_store_root, project_name=args.name)
    folder = Path(store_folder)

    setup_logger(folder_path=folder, console_output=args.aux_console_output)
    setup_env(gpu_s=args.gpu, seed=args.seed)

    load_preset(args)

    data_provider = SequenceLoaderProvider(args)
    if args.search_teacher_type == "bert":
        args.loader_num_aux_vocabs = 2

    args.loader_num_items = data_provider.num_item
    logging.info(data_provider.report_info())

    dict_to_logger(vars(args), exclude_list=["name", "seed", "gpu"])

    model = create_model_and_load_parameter()
    model.cuda()

    train_loader = data_provider.training_loader()
    val_loader = data_provider.validation_loader()

    pack = {"params": model.parameters(), "lr": args.train_lr, "weight_decay": args.train_wd}
    # optimizer = optim.Adam(**pack)
    optimizer = optim.AdamW(**pack)
    # optimizer = Lamb(**pack)
    # optimizer = optim.SGD(**pack, momentum=0.9)

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.train_lr_decay_step, gamma=args.train_lr_decay_gamma
    )
    log_model_info()
    # exit(0)
    start()
