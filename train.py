import os
import argparse
import datetime
import time
import pprint
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from src.dataset import HRTFDataset, MergedHRTFDataset
from src.model import DACD_HRTF
from src.trainer import Trainer
from src.losses import LSD, LSD_before_mean, CosDistIntra, ILD_Loss,Ortho_Loss
from src.configs import *
from torch.utils.tensorboard import SummaryWriter


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-datadir", "--dataset_directory",
                        type=str,
                        default='./preprocessed_data/HRIR',
                        help="path to the train data")
    parser.add_argument('-n', '--training_dataset_names', nargs='+',
                        default=['ari','bili','cipic','hutubs','listen','sonicom'])
    parser.add_argument("-a", "--artifacts_directory",
                        type=str,
                        default="",
                        help="directory to write model files to")
    parser.add_argument("-c", "--type_config",
                        type=str,
                        default='train',
                        help="idx of config you use")
    args = parser.parse_args()
    return args


def train(trainer):
    BEST_LOSS = 1e+16
    LAST_SAVED = -1
    for epoch in range(1, 1 + config["epochs"]):
        trainer.net.cuda()
        trainer.net.train()
        # ========== Train ==========
        loss_train = trainer.train_1ep(epoch, config)
        # -- logging(TensorBoard) ---
        for k, v in loss_train.items():
            writer.add_scalar(k + " / train", v, epoch)
        # ======= Validation ========
        print("----------")
        print(f"epoch {epoch} (valid)")
        t_start = time.time()
        use_cuda = True

        loss_valid = test(trainer.net, BEST_LOSS, valid_dataset, use_cuda=use_cuda)
        t_end = time.time()
        time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end - t_start))})"
        print(time_str)
        # -- Logging(TensorBoard) ---
        for k, v in loss_valid.items():
            writer.add_scalar(k + " / valid", v, epoch)
        # ===========================

        if loss_valid["loss"] < BEST_LOSS:
            BEST_LOSS = loss_valid["loss"]
            LAST_SAVED = epoch
            print("Best Loss. Saving model!")
            trainer.save(suffix="best")
        elif config["save_frequency"] > 0 and epoch % config["save_frequency"] == 0:
            print("Saving model!")
            trainer.save(suffix="log_" + f'{epoch:03}' + "ep")
        else:
            print("Not saving model! Last saved: {}".format(LAST_SAVED))
        print("---------------------")

    # --- Save final model ----
    trainer.save(suffix="final_" + f'{epoch:03}' + "ep")


def test(net, best_loss, data, use_cuda=True):
    device = 'cuda' if use_cuda else 'cpu'
    net.to(device).eval()

    srcpos, hrtf_gt = data
    srcpos, hrtf_gt = srcpos.to(device), hrtf_gt.to(device)
    masks = th.ones((srcpos.shape[0], srcpos.shape[1]), dtype=th.float32).to(device)
    returns = {}
    lsd_loss = LSD()
    ild_loss = ILD_Loss()
    keys = ["lsd", "ild"]
    for k in keys:
        returns.setdefault(k, 0)
    subject_list = np.arange(0, srcpos.shape[0])
    for sub_id in subject_list:
        mask_sub = masks[sub_id:sub_id + 1, :]
        hrtf_sub = hrtf_gt[sub_id:sub_id + 1, :, :, :]
        input = th.cat((th.real(hrtf_sub[:, :, 0, :]), th.imag(hrtf_sub[:, :, 0, :]), th.real(hrtf_sub[:, :, 1, :]),
                        th.imag(hrtf_sub[:, :, 1, :])), dim=-1)
        srcpos_sub = srcpos[sub_id:sub_id + 1, :, :]
        prediction = net.forward(input=input, srcpos=srcpos_sub, mask=mask_sub)
        hrtf_est = prediction["output"].detach().clone()
        eval_range = config.get("evaluation_frequency_range", (20, 20000))
        freq_axis = np.linspace(0, config["max_frequency"], hrtf_est.shape[-1] + 1)[1:]
        freq_mask = (freq_axis >= eval_range[0]) & (freq_axis <= eval_range[1])
        freq_idx = th.as_tensor(np.nonzero(freq_mask)[0], device=hrtf_est.device)
        hrtf_est_eval = hrtf_est.index_select(-1, freq_idx)
        hrtf_sub_eval = hrtf_sub.index_select(-1, freq_idx)
        returns["lsd"] += lsd_loss(hrtf_est_eval, hrtf_sub_eval)
        returns["ild"] += ild_loss(hrtf_est_eval, hrtf_sub_eval)
    loss = returns["lsd"]
    returns["loss"] = loss.detach().clone()
    for k in returns:
        returns[k] /= len(subject_list)
    loss_str = "    ".join([f"{k}:{returns[k]:.4}" for k in sorted(returns.keys())])
    print(loss_str)
    if returns["loss"] < best_loss:
        print("Best Loss.")

    return returns


if __name__ == "__main__":
    # Set default GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # ========== set random seed ===========
    seed = 0
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)

    # ========== time stamp =================
    dt_now = datetime.datetime.now() + datetime.timedelta() # China timezone
    datestamp = dt_now.strftime('_%Y%m%d')
    timestamp = dt_now.strftime('_%m%d_%H%M')
    # ========== args =================
    args = arg_parse()
    print("=====================")
    print("args: ")
    print(args)

    # ========== load config from src/configs.py =================
    config = {}
    config_name = f"config_{args.type_config}"
    config.update(eval(config_name))
    print("=====================")
    print("config: ")
    pprint.pprint(config)
    print("=====================")

    # ========== init model ===========
    net = DACD_HRTF(config=config)
    # #========== make dir contains model ===========
    if args.artifacts_directory == "":
        config["artifacts_dir"] = "outputs/" + datestamp + '_' + config[
            "model"] + '_' + f'{str(args.training_dataset_names)}'
    else:
        config["artifacts_dir"] = args.artifacts_directory
    print("artifacts_dir: " + config["artifacts_dir"])
    os.makedirs(config["artifacts_dir"], exist_ok=True)
    print("---------------------")

    # ========== load dataset ===========
    train_dataset = MergedHRTFDataset(args.training_dataset_names,
                                      hrir_path=args.dataset_directory,
                                      norm_way=0)
    dataset_hutubs = HRTFDataset(hrir_path=args.dataset_directory,
                                 dataset="hutubs",
                                 norm_way=0)
    valid_dataset = dataset_hutubs.validitem()
    # ========== train model ===========
    print("=====================")
    print(net)
    print("=====================")
    print("Train model.")
    print(f"number of trainable parameters: {net.num_trainable_parameters()}")
    print("---------------------")
    trainer = Trainer(config, net, train_dataset)
    # ---- TensorBoard -----
    log_dir = config["artifacts_dir"] + "/logs/" + config["model"] + timestamp
    writer = SummaryWriter(log_dir)
    print("logdir: " + log_dir)
    print("---------------------")
    train(trainer=trainer)
    writer.close()
