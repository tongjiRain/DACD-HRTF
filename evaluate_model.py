import os
import argparse
import pprint
import datetime
from src.configs import *
import numpy as np
import torch as th
from src.model import DACD_HRTF
from src.dataset import HRTFDataset, MergedHRTFDataset
from src.losses import LSD, LSD_before_mean, CosDistIntra,ILD_Loss
from src.utils import get_plane_indices
from src.utils import plotmaghrtf, plothrir, hrir2itd, minphase_recon, assign_itd, posTF2IR_dim4, plotazimzeni




def initParams():
    parser = argparse.ArgumentParser(description=__doc__)
    # Dataset parameters
    parser.add_argument("-d", "--dataset_directory", type=str, default="./preprocessed_data_few/HRIR")
    parser.add_argument("-a", "--artifacts_directory",type=str,default="",
                        help="directory to write test figures to")
    parser.add_argument('-t', '--testing_dataset_names', nargs='+',
                        default=["riec","crossmod"])

    # Model parameters
    parser.add_argument("-lf", "--model_file",
                        type=str,
                        default="./checkpoint/dacd_hrtf.best.net",
                        help="model file containing the trained network weights")
    # inference configuration
    parser.add_argument("-c", "--type_config",type=str,default='test')
    parser.add_argument("--gpu", type=str, help="GPU index", default="1")
    args = parser.parse_args()
    # Change this to specify GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args

if __name__ == '__main__':
    # ========== args =================
    args = initParams()
    # ========== load config from src/configs.py =================
    config = {}
    config_name = f"config_{args.type_config}"
    config.update(eval(config_name))
    print("=====================")
    print("config: ")
    pprint.pprint(config)
    print("=====================")
    # ========== make dir contains test figure ===========
    dt_now = datetime.datetime.now() + datetime.timedelta()
    if args.artifacts_directory == "":
        config["artifacts_dir"] = "outputs-test/"
    else:
        config["artifacts_dir"] = args.artifacts_directory
    print("artifacts_dir: " + config["artifacts_dir"])
    os.makedirs(config["artifacts_dir"], exist_ok=True)
    print("---------------------")
    # ========== load model =================
    net = DACD_HRTF(config=config)
    net.load_from_file(args.model_file)
    net.cuda().eval()
    # ========== load testing data  =================
    test_dataset_names = args.testing_dataset_names
    test_datasets = {} # key: dataset_name, value: dataset_test
    for dataset_name in test_dataset_names:
        dataset = HRTFDataset(hrir_path=args.dataset_directory,
                              dataset=dataset_name,
                              norm_way=0)
        test_dataset = dataset.allitem()  # locations_test, HRTFs_test
        test_datasets[dataset_name] = test_dataset
    # ========== inference =================
    num_pts_list = config.get("lap_sparse_levels", [3, 5, 19, 100])
    for num_pts in num_pts_list:
        config["num_pts"] = num_pts
        print("---------------")
        print(f'{config["num_pts"]} pts (SONICOM/LAP sparse layout, nearest-neighbor matched)')
        print("- - - - - - - -")

        for dataset_name in test_datasets.keys():
            print(f"Testing on {dataset_name}")
            srcpos, hrtf_gt = test_datasets[dataset_name]
            srcpos, hrtf_gt = srcpos.cuda(), hrtf_gt.cuda()
            masks = th.ones((srcpos.shape[0], srcpos.shape[1]), dtype=th.float32).to(srcpos.device)
            returns = {}
            lsd_loss = LSD()
            ild_loss = ILD_Loss()
            keys = ["lsd", "ild"]
            for k in keys:
                returns.setdefault(k, 0)
            subject_list = np.arange(0, srcpos.shape[0])
            # Determine plot indices: select points on the horizontal and median planes for visualization
            srcpos_np = srcpos[0, :, :].to('cpu').detach().numpy().copy()
            median_indices = get_plane_indices(srcpos_np, plane="midplane", num_samples=6)
            horizontal_indices = get_plane_indices(srcpos_np, plane="horizontal", num_samples=6)
            # Perform testing for each subject in unseen datasets
            for sub_id in subject_list:
                mask_sub = masks[sub_id:sub_id + 1, :]
                srcpos_sub = srcpos[sub_id:sub_id + 1, :, :]
                hrtf_sub = hrtf_gt[sub_id:sub_id + 1, :, :, :]
                input = th.cat((th.real(hrtf_sub[:, :, 0, :]), th.imag(hrtf_sub[:, :, 0, :]), th.real(hrtf_sub[:, :, 1, :]),
                                th.imag(hrtf_sub[:, :, 1, :])), dim=-1)
                prediction = net.forward(input=input, srcpos=srcpos_sub,mask=mask_sub)
                hrtf_est = prediction["output"].detach().clone()

                eval_range = config.get("evaluation_frequency_range", (20, 20000))
                freq_axis = np.linspace(0, config["max_frequency"], hrtf_est.shape[-1] + 1)[1:]
                freq_mask = (freq_axis >= eval_range[0]) & (freq_axis <= eval_range[1])
                freq_idx = th.as_tensor(np.nonzero(freq_mask)[0], device=hrtf_est.device)
                hrtf_est_eval = hrtf_est.index_select(-1, freq_idx)
                hrtf_sub_eval = hrtf_sub.index_select(-1, freq_idx)

                sub_lsd = lsd_loss(hrtf_est_eval, hrtf_sub_eval)
                sub_ild = ild_loss(hrtf_est_eval, hrtf_sub_eval)
                print(f"{dataset_name} sub-{sub_id + 1} lsd: {sub_lsd:.4f} ild: {sub_ild:.4f}")

                returns["lsd"] += sub_lsd
                returns["ild"] += sub_ild
                if sub_id == 0:
                    plotmaghrtf(srcpos=srcpos_sub[0, :, :],
                                hrtf_gt=hrtf_sub[0, :, :, :],
                                hrtf_est=hrtf_est[0, :, :, :],
                                mode=f'{dataset_name}_sub-{sub_id + 1}',
                                idx_plot_list=horizontal_indices,
                                config=config)
                #======== Obtain HRIRs =========
                fs = config.get('sampling_rate', 44100) # HRIR sampling rate
                f_us = config['fs_upsampling'] # frequency to calc ITD
                # ------- Obtain true ITDs ------
                hrir_sub = posTF2IR_dim4(hrtf_sub)   # S x B x 2 x 2L
                _, hrir_min = minphase_recon(tf=hrtf_est)  # S x B x 2 x 2L
                if config["use_itd_gt"]: # Use true ITDs
                    #--------- Assign ITD ----------
                    itd_des = hrir2itd(hrir=hrir_sub, fs=fs, f_us=f_us)
                    itd_ori = hrir2itd(hrir=hrir_min, fs=fs, f_us=f_us)  # original ITD
                    hrir_est = assign_itd(hrir_ori=hrir_min, itd_ori=itd_ori, itd_des=itd_des,
                                              fs=fs)  # HRIR with minimum phase & true ITD
                else:
                    hrir_est = hrir_min  # HRIR with minimum phase
                # ========= Plot HRIRs ==========
                if sub_id == 0:
                    plothrir(srcpos=srcpos_sub[0, :, :],
                             mode=f'{dataset_name}_sub-{sub_id + 1}',
                             idx_plot_list=horizontal_indices,
                             config=config,
                             hrir_gt=hrir_sub[0, :, :, :],
                             hrir_est=hrir_est[0, :, :, :])
                if sub_id == 0:
                    hrir_gt_all = th.zeros(len(subject_list), hrir_est.shape[1],hrir_est.shape[2],hrir_est.shape[3])
                    hrir_est_all = th.zeros(len(subject_list), hrir_est.shape[1],hrir_est.shape[2],hrir_est.shape[3])
                    lsd_bm_all = th.zeros(len(subject_list),hrir_est.shape[1],hrir_est.shape[2]) # B,2,S
                hrir_gt_all[sub_id,:,:,:] = hrir_sub[0,:,:,:]
                hrir_est_all[sub_id,:,:,:] = hrir_est[0,:,:,:,]

            for k in returns:
                returns[k] /= len(subject_list)
            loss_str = "    ".join([f"{k}:{returns[k]:.4}" for k in sorted(returns.keys())])
            print(f"{dataset_name} all subjects "+loss_str)


