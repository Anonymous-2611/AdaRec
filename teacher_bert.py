import logging
import time
from pathlib import Path

import torch
from torch import optim, nn

from options import bert4rec_parser
from presets import load_preset

from common import setup_folder, setup_env
from loader.masked_loader import MaskedLoaderProvider
from models.bert.model import BertModel
from models.others.lamb_optim import Lamb

from utils.logger import setup_logger
from utils.metrics import SRSMetric, AverageMeter
from utils.others import load_state_dict
from utils.printer import dict_to_logger


def create_state_dict():
    return {"model": model.state_dict(), "optimizer": optimizer.state_dict()}


def need_to_log(num_iter):
    return (num_iter + 1) % args.train_log_every == 0


def do_validate(epoch=None):
    logging.info("[VALIDATE]")
    model.eval()

    meter = SRSMetric(k_list=args.aux_eval_ks)
    meter.setup_and_clean()

    with torch.no_grad():
        for batch in iter(val_loader):
            batch = [x.cuda() for x in batch]
            seqs, candidates = batch
            scores = model(seqs)  # [batch_size, session_len, num_item]
            scores = scores[:, -1, :]  # [batch_size, num_item]
            scores = scores.gather(1, candidates)  # [batch_size, 1+num_sample]

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
    logging.info("\tLearning rate: {:.5f}".format(optimizer.param_groups[0]["lr"]))

    model.train()
    loss_meter = AverageMeter()

    criterion = nn.CrossEntropyLoss(ignore_index=0)  # non-zero elements in `labels` are learning targets

    for batch_idx, batch in enumerate(train_loader):
        batch = [x.cuda() for x in batch]

        masked_seqs, labels_of_mask = batch
        logits = model(masked_seqs)  # [B, L, num_classes]

        logits = logits.view(-1, logits.size(-1))  # [B*L, num_classes]
        labels_of_mask = labels_of_mask.view(-1)  # [B*L]

        loss = criterion(logits, labels_of_mask)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.train_grad_clip_norm)
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
        cmd = (
            "--name test-ml2k "
            "--gpu 0 "
            "--dataset ml2k "
            "--dataset_type mask "
            "--train_lr 0.005 "
            "--preset bert-train "
            "--aux_store_root store/1.TEST "
            "--aux_console_output"
        )
        cmd = cmd.strip()
        parser = bert4rec_parser()
        args = parser.parse_args(cmd.split(" "))
    else:
        parser = bert4rec_parser()
        args = parser.parse_args()

    store_folder = setup_folder(store_root=args.aux_store_root, project_name=args.name)
    folder = Path(store_folder)

    setup_logger(folder_path=folder, console_output=args.aux_console_output)
    setup_env(gpu_s=args.gpu, seed=args.seed)

    load_preset(args)

    data_provider = MaskedLoaderProvider(args)
    logging.info(data_provider.report_info())
    args.loader_num_items = data_provider.num_item

    dict_to_logger(vars(args), exclude_list=["name", "seed", "gpu"])

    model = BertModel(args).cuda()

    optimizer = Lamb(model.parameters(), lr=args.train_lr, weight_decay=args.train_wd)
    # optimizer = optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.train_wd)

    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.train_lr_decay_step, gamma=args.train_lr_decay_gamma
    )

    train_loader = data_provider.training_loader()
    val_loader = data_provider.validation_loader()

    start()
