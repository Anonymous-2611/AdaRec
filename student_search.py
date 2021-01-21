import logging
import pickle
import time
from pathlib import Path

import torch
from torch import optim

from common import setup_folder, setup_seed, setup_gpu
from options import student_search_parser, student_search_preset_parser
from presets import load_preset

from loader.sequence_loader import SequenceLoaderProvider

from models.bert.model import BertModel
from models.nextitnet.model import NextItNet
from models.sasrec.model import SASRec
from models.nas.kd_search import StudentSearch

from utils.logger import setup_logger
from utils.metrics import SRSMetric, AverageMeterSet
from utils.others import load_state_dict
from utils.printer import dict_to_logger, output_alpha_to_logger

# ablation study
NO_TASK_CE_LOSS = False
NO_LOGITS_KD = False
NO_HIDDEN_KD = False
NO_EMBEDDING_KD = False


def create_state_dict():
    return {
        "model": model_search.state_dict(),
        "optimizer_alpha": optim_alpha.state_dict(),
        "optimizer_model": optim_model.state_dict(),
        "scheduler_alpha": lr_scheduler_alpha.state_dict(),
        "scheduler_model": lr_scheduler_model.state_dict(),
    }


def need_to_log(num_iter):
    return (num_iter + 1) % args.train_log_every == 0


def do_validate(epoch):
    logging.info("[VALIDATE]")
    model_search.eval()

    meter = SRSMetric(k_list=args.aux_eval_ks)
    meter.setup_and_clean()

    with torch.no_grad():
        for batch in iter(val_loader):
            batch = [x.to(device_student) for x in batch]
            seqs, candidates = batch

            scores = model_search.predict(seqs)  # [B, L, num_item]
            scores = scores[:, -1, :]  # [B, num_item]
            scores = scores.gather(1, candidates)  # [B, 1+num_sample]

            meter.submit(scores.cpu(), [[0]] * len(scores))

    meter.calc()
    meter.output_to_logger()

    if epoch is not None:
        save_data = {"state_dict": create_state_dict(), "epoch": epoch}
        torch.save(save_data, folder.joinpath("checkpoint.pth"))

    return round(meter.mrr[5], 4), round(meter.hit[5], 4)


def save_alpha(file, ret=False):
    alpha_arch = [cell.data.cpu().clone().numpy().tolist() for cell in model_search.alpha_arch]

    assert not file.exists(), f"Error: file `{file}` already exist."
    with file.open("wb") as f:
        data = {"arch": alpha_arch}
        pickle.dump(data, f)

    if ret:
        return alpha_arch


def do_train(epoch):
    logging.info("-" * 30)
    logging.info(f"[TRAIN] Epoch: {epoch + 1} / {args.train_iter}")
    logging.info("\tModel `temperature` = {:.4f}".format(model_search.temperature))

    lr_model, lr_alpha = optim_model.param_groups[0]["lr"], optim_alpha.param_groups[0]["lr"]
    logging.info("\t`lr`-(model, alpha) = {:.5f}, {:.5f}".format(lr_model, lr_alpha))

    model_search.train()
    teacher.eval()

    losses = AverageMeterSet()

    epoch_folder_name = "epoch-{:05d}".format(epoch)
    epoch_arch_folder = all_alpha_folder.joinpath(epoch_folder_name)
    if not epoch_arch_folder.is_dir():
        epoch_arch_folder.mkdir()

    cur_gamma = args.search_loss_gamma * (1 - args.search_loss_gamma_decay) ** epoch

    if NO_TASK_CE_LOSS:
        weight_ce = 0
    else:
        weight_ce = 1 - cur_gamma
    weight_kd = cur_gamma
    weight_eff = args.search_loss_beta

    w_kd_emb, w_kd_hid, w_kd_logits = 1, 1, 1
    if NO_EMBEDDING_KD:
        w_kd_emb = 0
    if NO_HIDDEN_KD:
        w_kd_hid = 0
    if NO_LOGITS_KD:
        w_kd_logits = 0

    weight_embedding = weight_kd * w_kd_emb
    weight_hidden = weight_kd * w_kd_hid
    weight_logits = weight_kd * w_kd_logits

    s = "\tw_CE, w_KD_e, w_KD_h, w_KD_l, w_E = {:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
        weight_ce, weight_embedding, weight_hidden, weight_logits, weight_eff
    )
    logging.info(s)

    for batch_idx, batch in enumerate(train_loader):
        # 1. teacher output layer-wise hidden and logits
        batch_t = [x.to(device_teacher) for x in batch]
        training_seq, label_seq = batch_t
        emb_t, hidden_t, out_t = teacher.emb_hidden_logits(training_seq)

        emb_t = emb_t.to(device_student)  # [B, L, C_t]
        hidden_t = torch.stack(hidden_t, 0).to(device_student)  # [|T|, B, L ,C_t]
        out_t = out_t.to(device_student)  # [B, L, num_classes]

        optim_model.zero_grad()
        optim_alpha.zero_grad()

        # 2. student learn
        batch_s = [x.to(device_student) for x in batch]
        training_seq, label_seq = batch_s
        loss_pack = model_search(training_seq, label_seq, emb_t, hidden_t, out_t)

        loss_ce, loss_kd_pack, loss_e = loss_pack
        loss_embedding, loss_hidden, loss_logits = loss_kd_pack

        # record raw losses
        losses.update("ce", loss_ce.item())
        losses.update("kd-emb", loss_embedding.item())
        losses.update("kd-hid", loss_hidden.item())
        losses.update("kd-out", loss_logits.item())
        losses.update("e", loss_e.item())

        loss_kd = weight_embedding * loss_embedding
        loss_kd += weight_hidden * loss_hidden
        loss_kd += weight_logits * loss_logits
        losses.update("mix", loss_kd.item())

        loss_ce = weight_ce * loss_ce
        loss_e = weight_eff * loss_e

        loss = loss_ce + loss_kd + loss_e
        loss.backward()

        optim_model.step()
        optim_alpha.step()

        if need_to_log(batch_idx):
            cur, total = batch_idx + 1, len(train_loader)
            alpha_file_name = "alpha-{:05d}.pickle".format(cur)
            alpha_save_path = epoch_arch_folder.joinpath(alpha_file_name)
            save_alpha(alpha_save_path)

            s = "\tStep: {:5d}/{:5d}".format(cur, total)
            s += " L_CE: {:.4f}".format(losses["ce"].avg)
            s += " | ( L_KD_e: {:.4f}".format(losses["kd-emb"].avg)
            s += " | L_KD_h: {:.4f}".format(losses["kd-hid"].avg)
            s += " | L_KD_l: {:.4f}".format(losses["kd-out"].avg)
            s += " | L_mix: {:.4f} )".format(losses["mix"].avg)
            s += " | L_E: {:.4f}".format(losses["e"].avg)
            s += " => {}/{}".format(epoch_folder_name, alpha_file_name)

            logging.info(s)

    final_alpha_file_name = "epoch-{:05d}.pickle".format(epoch)
    final_alpha_save_path = final_alpha_folder.joinpath(final_alpha_file_name)

    alpha_arch = save_alpha(final_alpha_save_path, ret=True)
    logging.info(f"==> Final alpha saved at: `final-alpha/{final_alpha_file_name}`")
    output_alpha_to_logger(alpha_arch)

    if args.search_distill_loss.lower() == "emd":
        logging.info("Teacher weights: {}".format(" ".join(["{:.4f}".format(e) for e in model_search.teacher_weights])))
        logging.info("Student weights: {}".format(" ".join(["{:.4f}".format(e) for e in model_search.student_weights])))


def start():
    best_mrr5, best_hit5 = 0.0, 0.0
    for epoch in range(args.train_iter):
        tic = time.time()

        do_train(epoch)
        cur_mrr5, cur_hit5 = do_validate(epoch)

        lr_scheduler_model.step()
        lr_scheduler_alpha.step()
        model_search.temperature_step()  # Temperature in gumbel softmax sampling

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

    if is_testing:  # 1. Testing
        teacher_type, dataset = "bert", "ml2k"
        teacher_folder = "/path/to/teacher/folder"
        cmd = (
            # -----[GLOBAL]-----
            "--name test "
            "--gpu_teacher 0 --gpu_student 0 "
            f"--preset {teacher_type}-search "
            f"--dataset {dataset} "
            "--dataset_type seq "
            "--loader_train_batch_size 64 "
            # -----[TEACHER]-----
            f"--search_teacher_folder {teacher_folder} "
            # -----[AUXILIARY]-----
            "--aux_store_root store/1.TEST "
            "--aux_console_output"
        )
        cmd = cmd.strip()
        parser = student_search_parser(teacher_type=teacher_type)
        args = parser.parse_args(cmd.split(" "))
    else:  # 2. Running
        preset_parser = student_search_preset_parser()
        preset_args, unk = preset_parser.parse_known_args()
        teacher_type, dataset = preset_args.T, preset_args.D
        parser = student_search_parser(teacher_type=teacher_type)
        args = parser.parse_args(unk)
        assert args.search_teacher_folder is not None

    store_folder = setup_folder(store_root=args.aux_store_root, project_name=args.name)
    folder = Path(store_folder)

    setup_logger(folder_path=folder, console_output=args.aux_console_output)
    setup_seed(args.seed)

    if args.gpu_teacher == args.gpu_student:
        gpu_str = args.gpu_teacher
        device_student = torch.device("cuda:0")
        device_teacher = torch.device("cuda:0")
    else:  # args.gpu_teacher != args.gpu_student:
        if args.gpu_teacher > args.gpu_student:
            gpu_str = f"{args.gpu_student},{args.gpu_teacher}"
            device_student = torch.device("cuda:0")
            device_teacher = torch.device("cuda:1")
        else:  # args.gpu_teacher < args.gpu_student:
            gpu_str = f"{args.gpu_teacher},{args.gpu_student}"
            device_student = torch.device("cuda:1")
            device_teacher = torch.device("cuda:0")
    setup_gpu(gpu_str)

    logging.info(f"Teacher Model using device - <{device_teacher}>")
    logging.info(f"Student Model using device - <{device_student}>")

    load_preset(args)

    data_provider = SequenceLoaderProvider(args)
    if teacher_type == "bert":
        args.loader_num_aux_vocabs = 2

    args.loader_num_items = data_provider.num_item
    logging.info(data_provider.report_info())

    dict_to_logger(vars(args), exclude_list=["name", "seed"])

    if teacher_type == "bert":
        teacher = BertModel(args).to(device_teacher)
    elif teacher_type == "nin":
        teacher = NextItNet(args).to(device_teacher)
    else:  # teacher_type == "sas":
        teacher = SASRec(args).to(device_teacher)

    # load pre-trained teacher model
    teacher_model_path = Path(args.search_teacher_folder).joinpath("best-model.pth")
    logging.info("Loading teacher model from `{}`".format(teacher_model_path))
    state_dict = load_state_dict(teacher_model_path, device=device_teacher)
    teacher.load_state_dict(state_dict["state_dict"]["model"])
    teacher.to(device_teacher).eval()  # freeze
    logging.info("...Loaded.")

    # student network architecture nas
    model_search = StudentSearch(args).to(device_student)

    pack = {"params": model_search.net_parameters(), "lr": args.train_model_lr, "weight_decay": args.train_model_wd}
    optim_model = optim.AdamW(**pack)
    lr_scheduler_model = optim.lr_scheduler.StepLR(
        optim_model, step_size=args.train_model_lr_decay_step, gamma=args.train_model_lr_decay_gamma
    )

    pack = {"params": model_search.alpha_parameters(), "lr": args.train_alpha_lr, "weight_decay": args.train_alpha_wd}
    optim_alpha = optim.AdamW(**pack)
    lr_scheduler_alpha = optim.lr_scheduler.StepLR(
        optim_alpha, step_size=args.train_alpha_lr_decay_step, gamma=args.train_alpha_lr_decay_gamma
    )

    all_alpha_folder = folder.joinpath("alpha")  # save intermediate alpha
    if not all_alpha_folder.is_dir():
        all_alpha_folder.mkdir()

    final_alpha_folder = folder.joinpath("alpha-final")  # save alpha after epoch finish
    if not final_alpha_folder.is_dir():
        final_alpha_folder.mkdir()

    train_loader = data_provider.training_loader()
    val_loader = data_provider.validation_loader()

    start()
