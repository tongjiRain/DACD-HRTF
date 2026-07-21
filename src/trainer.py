import time 
import numpy as np
import torch as th
from torch.utils.data import DataLoader
from src.utils import plotmaghrtf, plothrir, hrir2itd, minphase_recon, assign_itd, posTF2IR_dim4
from src.losses import LSD, CosDistIntra,ILD_Loss,Triplet_loss,Ortho_Loss


class Trainer:
    def __init__(self, config, net, dataset):
        '''
        args
        -----
        config: a dict containing parameters
        net: the network to be trained, must be of type src.utils.Net
        dataset: the dataset to be trained on
        '''
        self.config = config
        self.dataset = dataset
        gpus = [i for i in range(config["num_gpus"])]
        self.dataloader = DataLoader(dataset,
                                     batch_size=config["batch_size"],
                                     shuffle=True,
                                     collate_fn=dataset.collate_fn,
                                     num_workers=0)
        self.net = th.nn.DataParallel(net, gpus)
        weights = filter(lambda x: x.requires_grad, net.parameters())
        self.optimizer = th.optim.Adam(weights, lr=config["learning_rate"], eps=1e-8)
        self.lsd_loss = LSD()
        self.cosdist = CosDistIntra()
        self.ild_loss = ILD_Loss()
        self.triplet_loss = Triplet_loss()
        self.ori_loss = Ortho_Loss()

        self.total_iters = 0
        self.net.train()

    def keras_decay(self, step, decay=0.01):
        # Keras learning rate decay schedule
        return 1.0 / (1.0 + decay * step)

    def save(self, suffix=""):
        self.net.module.save(self.config["artifacts_dir"], suffix)

    def train_1ep(self, epoch,config):
        '''1 training epoch
        '''
        loss_stats = {}
        t_start = time.time()
        num_sub = len(self.dataset)
        bs  = self.config["batch_size"]
        num_batch = np.ceil(num_sub/self.config["batch_size"])
        if epoch == 1:
            print(f'num_sub: {num_sub}, batch_size: {self.config["batch_size"]}, num_batch:{num_batch}')
        for itr, data in enumerate(self.dataloader):
            loss_new = self.train_iteration(data,itr=itr,config=config)
            # logging
            for k, v in loss_new.items():
                loss_stats[k] = loss_stats[k]+v if k in loss_stats else v
            #===== progress bar ======

            prog_step = 1
            if round(num_batch/prog_step) != 0:
                if itr == 0:
                    print('[',end='')
                elif itr == num_batch-1:
                    print('#]')
                elif itr // round(num_batch/prog_step) == 0:
                    print('#',end='')
            #=========================
        for k in loss_stats:
            loss_stats[k] /= num_batch

        # self.lr_scheduler.step()

        t_end = time.time()
        loss_str = "    ".join([f"{k}:{loss_stats[k]:.4}" for k in sorted(loss_stats.keys())])
        time_str = f"({time.strftime('%H:%M:%S', time.gmtime(t_end-t_start))})"
        print(f"epoch {epoch} (train) ")
        print(loss_str + "        " + time_str)
        return loss_stats

    def train_iteration(self, data,itr=0,config=''):
        '''
        one optimization step
        args
        -----
        data: tuple of tensors, [source position, hrtf]
        sub: (list) indexes of subjects
        itr: (int) iteration number
        returns
        -----

        '''
        returns = {}
        #=== forward ====
        self.optimizer.zero_grad()
        anchor_data, positive_data, negative_data = data
        a_srcpos, a_hrtf_gt, a_mask, a_names = anchor_data  # srcpos: S x B x 3, hrtf_gt: S x B x 2 x L
        p_srcpos, p_hrtf_gt, p_mask, p_names = positive_data
        n_srcpos, n_hrtf_gt, n_mask, n_names = negative_data
        a_srcpos, a_hrtf_gt,a_mask = a_srcpos.cuda(), a_hrtf_gt.cuda(), a_mask.cuda()
        p_srcpos, p_hrtf_gt,p_mask = p_srcpos.cuda(), p_hrtf_gt.cuda(), p_mask.cuda()
        n_srcpos, n_hrtf_gt,n_mask = n_srcpos.cuda(), n_hrtf_gt.cuda(), n_mask.cuda()


        a_hrtf_gt_l = a_hrtf_gt[:, :, 0, :]
        a_hrtf_gt_r = a_hrtf_gt[:, :, 1, :]
        # input S x B x 2 x L -> S x B x (4L)
        a_input = th.cat((th.real(a_hrtf_gt_l), th.imag(a_hrtf_gt_l), th.real(a_hrtf_gt_r), th.imag(a_hrtf_gt_r)), dim=-1)
        a_prediction = self.net.forward(input=a_input, srcpos=a_srcpos,mask=a_mask)
        hrtf_est = a_prediction["output"] # S x B x 2 x L
        z_person_est = a_prediction["z_person"]  # S x B x z_dim
        z_db_est = a_prediction["z_db"]  # S x B x z_dim
        z_db_a = a_prediction["z_db_avg"] # S x z_dim_db
        z_person_a = a_prediction["z_person_avg"] # S x z_dim_person

        # positive data model forward
        p_hrtf_gt_l = p_hrtf_gt[:, :, 0, :]
        p_hrtf_gt_r = p_hrtf_gt[:, :, 1, :]
        # input S x B x 2 x L -> S x B x (4L)
        p_input = th.cat((th.real(p_hrtf_gt_l), th.imag(p_hrtf_gt_l), th.real(p_hrtf_gt_r), th.imag(p_hrtf_gt_r)), dim=-1)
        p_prediction = self.net.forward(input=p_input, srcpos=p_srcpos,mask=p_mask) # dict key: "output", "z", "idx_mes_pos"
        z_db_p = p_prediction["z_db_avg"] # S x z_dim_db

        # negative data model forward
        n_hrtf_gt_l = n_hrtf_gt[:, :, 0, :]
        n_hrtf_gt_r = n_hrtf_gt[:, :, 1, :]
        # input S x B x 2 x L -> S x B x (4L)
        n_input = th.cat((th.real(n_hrtf_gt_l), th.imag(n_hrtf_gt_l), th.real(n_hrtf_gt_r), th.imag(n_hrtf_gt_r)), dim=-1)
        n_prediction = self.net.forward(input=n_input, srcpos=n_srcpos,mask=n_mask) # dict key: "output", "z", "idx_mes_pos"
        z_db_n = n_prediction["z_db_avg"] # S x z_dim_db

        #==================
        loss_dict = {}
        loss_dict["cdintra_z_person"] = self.cosdist(z_person_est,a_mask)
        loss_dict["cdintra_z_db"] = self.cosdist(z_db_est, a_mask)
        loss_dict["lsd"] = self.lsd_loss(hrtf_est, a_hrtf_gt, a_mask)
        loss_dict["ild"] = self.ild_loss(hrtf_est, a_hrtf_gt, a_mask)
        loss_dict["triplet_loss"] = self.triplet_loss(z_db_a, z_db_p, z_db_n)
        loss_dict['ori_loss'] = self.ori_loss(z_person_a, z_db_a)  # S x dim_person tensor
        loss = self.config["loss_weights"]["lsd"] * loss_dict["lsd"] + \
               self.config["loss_weights"]["triplet"] * loss_dict["triplet_loss"] + \
               self.config["loss_weights"]["ild"] * loss_dict["ild"] + \
               self.config["loss_weights"]["cdintra_z_person"] * loss_dict["cdintra_z_person"] + \
               self.config["loss_weights"]["cdintra_z_db"] * loss_dict["cdintra_z_db"]


        # update model parameters
        loss.backward()
        self.optimizer.step()
        self.total_iters += 1
        #==================
        returns["loss"] = loss.detach().clone()
        # update model paramet
        return {
            "loss": loss.detach().clone(),
            "lsd": loss_dict["lsd"].detach().clone(),
            "ild": loss_dict["ild"].detach().clone(),
            "cdintra_z_person": loss_dict["cdintra_z_person"].detach().clone(),
            "cdintra_z_db": loss_dict["cdintra_z_db"].detach().clone(),
            "triplet_loss": loss_dict["triplet_loss"].detach().clone(),
            "ori_loss": loss_dict["ori_loss"].detach().clone(),
        }