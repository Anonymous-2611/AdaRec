import logging
import time
from pathlib import Path
import numpy as np

import torch
from torch import optim, nn

from common import setup_folder, setup_env
from loader.sequence_loader import SequenceLoaderProvider
from models.sasrec.model import SASRec
from models.others.lamb_optim import Lamb

from options import sasrec_parser
from presets import load_preset

from utils.logger import setup_logger
from utils.metrics import SRSMetric, AverageMeter
from utils.printer import dict_to_logger

EVAL_NEG_SAMPLE = True


def create_state_dict():
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict()}


def need_to_log(num_iter):
    return (num_iter + 1) % args.train_log_every == 0


def do_validate(epoch):
    logging.info("[VALIDATE]")
    model.eval()

    meter = SRSMetric(k_list=args.aux_eval_ks)
    meter.setup_and_clean()

    def do_metric(logits, other):
        if EVAL_NEG_SAMPLE:
            chosen_from = other
            logits = logits.gather(1, chosen_from)  # [B, 1+num_neg_sample]
            meter.submit(logits.cpu(), [[0]] * len(logits))
        else:  # EVAL_ALL_SAMPLE
            ground_truth = np.array(other[:, 0:1].cpu())  # [B, 1]
            logits = np.array(logits.cpu())
            meter.submit(logits, ground_truth)

    with torch.no_grad():
        for batch in iter(val_loader):
            batch = [x.cuda() for x in batch]
            seqs, candidates = batch

            scores = model(seqs)  # [B, L, #item]
            scores = scores[:, -1, :]  # [B, #item]

            do_metric(scores, candidates)

    meter.calc()
    meter.output_to_logger()

    save_data = {"state_dict": create_state_dict(), "epoch": epoch}
    torch.save(save_data, folder.joinpath("checkpoint.pth"))

    return round(meter.mrr[5], 4), round(meter.hit[5], 4)


def do_train(epoch):
    logging.info("-" * 30)
    logging.info(f"[TRAIN] Epoch: {epoch + 1} / {args.train_iter}")
    logging.info("\tLearning rate: {:.5f}".format(optimizer.param_groups[0]["lr"]))

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
        optimizer.step()

        loss_meter.update(loss.item())

        if need_to_log(batch_idx):
            cur, total, loss = batch_idx + 1, len(train_loader), loss_meter.avg
            logging.info("\tStep: {:5d} / {:5d}\tLoss: {:.4f}".format(cur, total, loss))


def start():
    best_mrr5, best_hit5 = 0.0, 0.0

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


if __name__ == "__main__":
    is_testing = False
    if is_testing:
        dataset = "ml2k"
        cmd = (
            "--name b8 "
            "--gpu 1 "
            "--preset sas-train "
            f"--dataset {dataset} "
            "--dataset_type seq "
            "--aux_store_root store/1.TEST "
            "--aux_console_output"
        )
        cmd = cmd.strip()
        parser = sasrec_parser()
        args = parser.parse_args(cmd.split(" "))
    else:
        parser = sasrec_parser()
        args = parser.parse_args()

    store_folder = setup_folder(store_root=args.aux_store_root, project_name=args.name)
    folder = Path(store_folder)

    setup_logger(folder_path=folder, console_output=args.aux_console_output)
    setup_env(gpu_s=args.gpu, seed=args.seed)

    load_preset(args)

    data_provider = SequenceLoaderProvider(args)
    logging.info(data_provider.report_info())
    args.loader_num_items = data_provider.num_item

    dict_to_logger(vars(args), exclude_list=["name", "seed", "gpu"])

    model = SASRec(args).cuda()

    pack = {"params": model.parameters(), "lr": args.train_lr, "weight_decay": args.train_wd}

    optimizer = Lamb(**pack)
    # optimizer = optim.AdamW(**pack)
    # optimizer = optim.Adam(**pack, betas=(0.9, 0.98))
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.train_lr_decay_step, gamma=args.train_lr_decay_gamma
    )

    train_loader = data_provider.training_loader()
    val_loader = data_provider.validation_loader()

    start()
