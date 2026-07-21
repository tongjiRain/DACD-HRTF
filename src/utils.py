import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import torchaudio as ta   
import torch.nn.functional as F
import os
from scipy.spatial import KDTree
from sparse_sampling.lap_sparse import build_lap_target_grid

class Net(nn.Module):

    def __init__(self, model_name="network", use_cuda=True):
        super().__init__()
        self.use_cuda = use_cuda
        if th.cuda.is_available(): 
            self.use_cuda = True
        self.model_name = model_name

    def save(self, model_dir, suffix=''):
        '''
        save the network to model_dir/model_name.suffix.net
        args
        -----
        model_dir: directory to save the model to

        returns
        -----
        suffix: suffix to append after model name
        '''
        if self.use_cuda:
            self.cpu()

        if suffix == "":
            fname = f"{model_dir}/{self.model_name}.net"
        else:
            fname = f"{model_dir}/{self.model_name}.{suffix}.net"

        th.save(self.state_dict(), fname)
        if self.use_cuda:
            self.cuda()

    def load_from_file(self, model_file):
        '''load network parameters from model_file

        args
        -----
        model_file: file containing the model parameters
        '''
        if self.use_cuda:
            self.cpu()

        states = th.load(model_file)
        self.load_state_dict(states)

        if self.use_cuda:
            self.cuda()
        print(f"Loaded: {model_file}")

    def load(self, model_dir, suffix=''):
        '''
        load network parameters from model_dir/model_name.suffix.net

        args
        -----
        model_dir: directory to load the model from
        suffix: suffix to append after model name
        '''
        if suffix == "":
            fname = f"{model_dir}/{self.model_name}.net"
        else:
            fname = f"{model_dir}/{self.model_name}.{suffix}.net"
        self.load_from_file(fname)

    def num_trainable_parameters(self):
        '''
        returns
        -----
        the number of trainable parameters in the model
        '''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_plane_indices(srcpos_np, plane="midplane", num_samples=5):
    """
    Get the sampling indices for a specified anatomical plane.

    Args:
        srcpos_np (np.ndarray): Array of shape (N, 3), representing source positions
                            in spherical coordinates (azimuth, elevation, distance).
        plane (str): Sampling plane. Must be either "midplane" (median plane) or "horizontal".
        num_samples (int): Number of samples to select from the plane (no more than available points). Default is 5.

    Returns:
        idx_plot_list (List[int]): List of indices corresponding to sampled positions on the selected plane.
    """

    if plane == "midplane":
        indices = np.where(np.abs(srcpos_np[:, 0]) == 0)[0]  # azimuth = 0
        values = srcpos_np[indices, 1]  # elevation
    elif plane == "horizontal":
        indices = np.where(np.abs(srcpos_np[:, 1]) == 0)[0]  # elevation = 0
        values = srcpos_np[indices, 0]  # azimuth
    else:
        raise ValueError("plane must be  'midplane' 或 'horizontal'")
    sorted_indices = indices[np.argsort(values)]
    num_samples = min(num_samples, len(sorted_indices))
    idx_plot_list = sorted_indices[np.linspace(0, len(sorted_indices) - 1, num_samples).astype(int)]

    return idx_plot_list

def plotmaghrtf(srcpos,hrtf_gt,hrtf_est,idx_plot_list,mode, config):
    '''plot magnitude frequency response
    args
    -----
    srcpos:   B x 3 tensor
    hrtf_gt:  B x 2 x L tensor, true HRTF
    hrtf_est: B x 2 x L tensor, estimated HRTF
    idx_plot_list: np.array, indexes of source position for which you want to plot HRIRs
    mode: (str)
    config: (dict)
    '''
    mag2db = ta.transforms.AmplitudeToDB(stype = 'magnitude')     
    f_bin = np.linspace(0,config["max_frequency"],round(config["fft_length"]/2)+1)[1:]
    plt.figure(figsize=(12,round(6*len(idx_plot_list))))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)

    for itr, idx_plot in enumerate(idx_plot_list):
        plt.subplot(len(idx_plot_list),1, itr+1)
        plt.plot(f_bin, mag2db(th.abs(hrtf_gt[idx_plot,0,:])).to('cpu').detach().numpy().copy(), label="Left (Ground Truth)", color='b',linestyle=':')
        plt.plot(f_bin, mag2db(th.abs(hrtf_gt[idx_plot,1,:])).to('cpu').detach().numpy().copy(), label="Right (Ground Truth)", color='r',linestyle=':')
        plt.plot(f_bin, mag2db(th.abs(hrtf_est[idx_plot,0,:])).to('cpu').detach().numpy().copy(), label="Left (Estimated)", color='b')
        plt.plot(f_bin, mag2db(th.abs(hrtf_est[idx_plot,1,:])).to('cpu').detach().numpy().copy(), label="Right (Estimated)", color='r')
        plt.grid()
        plt.legend()
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        # ylim = [-80,0] if config["green"] else [-50,30]
        ylim =  [-50, 30]
        plt.ylim(ylim)
        plt.xlim([0,config["max_frequency"]])
        srcpos_np = srcpos.to("cpu").detach().numpy().copy()
        plt.title(f"HRTF (radius={srcpos_np[idx_plot,2]:.2f} m, azimuth={srcpos_np[idx_plot,0]:.1f} degree, zenith={srcpos_np[idx_plot,1]:.1f} degree)")

    figure_dir = config["artifacts_dir"] + "/figure/HRTF/"
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(figure_dir + "HRTF_mag_"+mode+".png", dpi=300)
    plt.close()

def plothrir(srcpos,mode,config,idx_plot_list=np.arange(5), hrir_gt=None, hrir_est=None):
    '''plot HRIRs

    args
    -----
    srcpos:   B x 3 tensor
    mode: (str)
    config: (dict)
    idx_plot_list: np.array, indexes of source position for which you want to plot HRIRs
    hrir_gt:  B x 2 x 2L tensor, true HRIR
    hrir_est: B x 2 x 2L tensor, estimated HRIR
    '''
    t_bin = np.linspace(0, config["fft_length"]/(2*config["max_frequency"]), config["fft_length"])
    plt.figure(figsize=(12,round(6*len(idx_plot_list))))
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    for itr, idx_plot in enumerate(idx_plot_list):
        plt.subplot(len(idx_plot_list),1, itr+1)
        plt.plot(t_bin, hrir_gt[idx_plot,0,:].to('cpu').detach().numpy().copy(), label="Left (Ground Truth)", color='b',linestyle=':')
        plt.plot(t_bin, hrir_gt[idx_plot,1,:].to('cpu').detach().numpy().copy(), label="Right (Ground Truth)", color='r',linestyle=':')
        plt.plot(t_bin, hrir_est[idx_plot,0,:].to('cpu').detach().numpy().copy(), label="Left (Estimated)", color='b')
        plt.plot(t_bin, hrir_est[idx_plot,1,:].to('cpu').detach().numpy().copy(), label="Right (Estimated)", color='r')
        plt.grid()
        plt.legend()
        plt.xlabel("Time [s]")
        plt.ylabel("Magnitude")
        plt.xlim([t_bin[0], t_bin[-1]])
        srcpos_np = srcpos.to("cpu").detach().numpy().copy()
        plt.title(f"HRIR (radius={srcpos_np[idx_plot,2]:.2f} m, azimuth={srcpos_np[idx_plot,0]:.1f} degree, zenith={srcpos_np[idx_plot,1]:.1f} degree)")

    figure_dir = config["artifacts_dir"] + "/figure/HRIR/"
    os.makedirs(figure_dir, exist_ok=True)
    plt.savefig(figure_dir + "HRIR_"+mode+".png", dpi=300)
    plt.close()

def sph2cart( theta, phi, r):
    """Conversion from spherical to Cartesian coordinates

    Parameters
    ------
    theta, phi, r: Azimuth angle, zenith angle, distance
    theta [0,360]
    phi [-90,90]
    Returns
    ------
    x, y, z : Position in Cartesian coordinates
    """
    theta = theta / 180 * np.pi
    phi = phi / 180 * np.pi
    x = r * th.cos(phi) * th.cos(theta)
    y = r * th.cos(phi) * th.sin(theta)
    z = r * th.sin(phi)
    return th.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)),dim=-1)



def aprox_lap_sparse(pts, level, return_target=False, verbose=True):
    """Select measured directions nearest to the SONICOM/LAP sparse layout."""
    if isinstance(pts, th.Tensor):
        pts_np = pts.detach().cpu().numpy()
    else:
        pts_np = np.asarray(pts, dtype=np.float64)

    pts_np = pts_np / np.linalg.norm(pts_np, axis=1, keepdims=True)
    target_xyz = build_lap_target_grid(level=level)

    _, idx = KDTree(pts_np).query(target_xyz)
    idx_prev = sorted(idx.tolist())
    idx = sorted(set(idx_prev))

    if verbose and len(idx_prev) > len(idx):
        print(f"[aprox_lap_sparse] detected duplication. {len(idx_prev)} -> {len(idx)} pts")

    if return_target:
        return idx, target_xyz
    return idx

def posTF2IR_dim4(tf):
    '''Transfer function (positive freq. bins) -> Impulse response
    args
    -----
    tf: S x B x 2 x L complex tensor

    returns
    -----
    ir: S x B x 2 x 2L float tensor
    '''
    # 全零Tensor,用于填充DC分量（直流分量）
    zeros = th.zeros([tf.shape[0],tf.shape[1],tf.shape[2],1]).to(tf.device).to(tf.dtype)
    # 翻转取共轭,构建负频率部分
    tf_fc = th.conj(th.flip(tf[:,:,:,:-1],dims=(-1,)))
    # 大体上没问题，但是正负频率包括共遏我得研究一下
    tf = th.cat((zeros,tf,tf_fc), dim=-1) # DC, positive freq. bins, negative freq. bins
    ir = th.fft.ifft(th.conj(tf), dim=-1)
    ir = th.real(ir)

    return ir

def hrir2itd(hrir,fs,f_us=8*44100,thrsh_ms=1000,lpf=True,upsample_via_cpu=True):
    '''calculate ITD from HRIR

    args
    -----
        hrir: (S,B,2,L) tensor
        fs: (int) sampling freq. of original hrir
        f_us: (int) sampling freq. after upsampling
        thrsh_ms: (float) threshold [ms]. (computed ITD is forced to be in [-thrsh_ms, +thrsh_ms] )
        lpf: (bool) If True, Low-pass filter is filtered to hrir.
    returns
    -----
        ITD: (S,B) tensor, interaural time difference [s] (-:src@left, +:src@right)
    '''
    if lpf:
        hrir = ta.functional.lowpass_biquad(waveform=hrir, sample_rate=fs, cutoff_freq=1600)
    else:
        pass
    if upsample_via_cpu:
        hrir = hrir.cpu()
    upsampler = ta.transforms.Resample(fs,f_us)
    hrir_us = upsampler(hrir.contiguous())
    if upsample_via_cpu:
        hrir_us = hrir_us.cuda()
    S, B, _, L = hrir_us.shape
    thrsh_idx = round(f_us/thrsh_ms)
    #===============================
    HRIR_l = hrir_us[:,:,0,:]
    HRIR_r = hrir_us[:,:,1,:]
    HRIR_l_pad = F.pad(HRIR_l,(L,L))
    HRIR_l_pad_in = HRIR_l_pad.reshape(1, S*B, -1)
    HRIR_r_wt = HRIR_r.reshape(S*B, 1, -1)
    crs_cor = F.conv1d(HRIR_l_pad_in, HRIR_r_wt, groups=S*B)
    crs_cor = crs_cor.reshape(S, B, -1)
    idx_beg = L - thrsh_idx
    idx_end = L + thrsh_idx + 1
    idx_max = th.argmax(crs_cor[:,:,idx_beg:idx_end], dim=-1) - thrsh_idx
    ITD = idx_max/f_us

    return ITD

def HilbertTransform(data, detach=False):
    '''perform Hilbert transformation换
    '''
    assert data.dim()==4
    N = data.shape[-1]
    # Allocates memory on GPU with size/dimensions of signal
    if detach:
        transforms = data.clone().detach()
    else:
        transforms = data.clone()
    transforms = th.fft.fft(transforms, axis=-1)
    transforms[:,:,:,1:N//2]      *= -1j      # positive frequency
    transforms[:,:,:,(N+2)//2 + 1: N] *= +1j  # negative frequency
    transforms[:,:,:,0] = 0; # DC signal
    if N % 2 == 0:
        transforms[:,:,:,N//2] = 0; # the (-1)**n term
    # Do IFFT on GPU: in place (same memory)
    return th.fft.ifft(transforms, axis=-1)

def minphase_recon(tf, contain_neg_fbin=False):
    '''minimum phase reconstruction
    args
    -----
        tf: (S,B,2,L) or (S,B,2,2L) tensor (L: # of freq. bins)
        contain_neg_fbin: bool. If True, mag.shape == (S,B,2,2L)
        conj: bool. 
    return
    -----
        phase_min: (S,B,2,2L) tensor
        ir_min:  (S,B,2,2L) tensor. Impulse response with minimum phase.
    '''
    if contain_neg_fbin:
        tf_pm = tf
    else:
        tf_nf = th.conj(th.flip(tf[:,:,:,:-1],dims=(-1,))) # negatibe freq.
        tf_pm = th.cat((th.ones_like(tf)[:,:,:,0:1], tf, tf_nf), dim=-1) # [1,pos,neg]
    mag_pm_log = th.log(th.abs(tf_pm)) # magnitude
    phase_min =  - HilbertTransform(mag_pm_log)
    ir_min = th.real(th.fft.ifft(th.abs(tf_pm)*th.exp(1j * phase_min), axis=-1))

    return phase_min, ir_min

def assign_itd(hrir_ori,itd_ori,itd_des,fs,shift_s=1e-3):
    '''assign ITD

    args
    -----
        hrir_ori: (S,B,2,L) tensor. (L: filter length)
        itd_ori: (S,B) tensor. ITD [s] of hrir_ori
        itd_des: (S,B) tensor. desired ITD [s]
        fs: (int) [Hz]. Sampling Frequency.
        shift_lr: (float) [s]. offset when ITD==0.
    return
    -----
        ir_itd_des:  (S,B,2,L) tensor. Impulse response with desired ITD.
    '''
    S, B = itd_ori.shape
    L = hrir_ori.shape[-1]
    shift_idx = shift_s * fs
    ITD_idx_fs_half = (itd_des-itd_ori) * fs / 2
    offset = th.ones(S,B,2).to(ITD_idx_fs_half.device) * shift_idx
    offset[:,:,0] += ITD_idx_fs_half # left
    offset[:,:,1] -= ITD_idx_fs_half # right
    offset = th.round(offset).to(int)

    arange = th.arange(L).reshape(1,1,1,L).tile(S,B,2,1).to(ITD_idx_fs_half.device)
    arange = (arange - offset[:,:,:,None]) % L

    # square window to remove pre-echo
    window_length = int(L - shift_idx)
    window_sq = th.cat((th.ones(window_length), th.zeros(L-window_length))).to(hrir_ori.device)
    hrir_ori_w = hrir_ori * window_sq[None,None,None,:]
    ir_itd_des = th.gather(hrir_ori_w, -1, arange)

    return ir_itd_des

def vhlines(ax, linestyle='-', color='gray', zorder=1, alpha=0.8, lw=0.75):
    ax.axhline(y=np.pi/2, linestyle=linestyle, color=color,zorder=zorder, alpha=alpha, lw=lw)
    ax.axvline(x=np.pi/2, linestyle=linestyle, color=color,zorder=zorder, alpha=alpha, lw=lw)
    ax.axvline(x=np.pi, linestyle=linestyle, color=color,zorder=zorder, alpha=alpha, lw=lw)
    ax.axvline(x=np.pi*3/2, linestyle=linestyle, color=color,zorder=zorder, alpha=alpha, lw=lw)
    ax.text(np.pi/2, np.pi+0.05, "Left", ha='center')
    ax.text(np.pi*3/2, np.pi+0.05, "Right", ha='center')
    ax.text(np.pi, np.pi+0.05, "Back", ha='center')

def plotazimzeni(pos,c,fname,title,cblabel,cmap='gist_heat',figsize=(10.5,5),dpi=300, emphasize_mes_pos=False, idx_mes_pos=None, vmin=None, vmax=None, save=True, clf=True):
    '''

    args
    -----
        pos: (B,*>3) tensor. (:,1):azimuth, (:,2):zenith
        c: (B) tensor.
        fname: str. filename
        title: str. title.
        cblabel: str. label of colorbar.
        cmap: colormap.
        figsie: (*,*) tuple.
        dpi: scalar.
    '''
    fig, ax = plt.subplots(figsize=figsize)
    vhlines(ax)
    if vmin==None:
        vmin=th.min(c)
    if vmax == None:
        vmax=th.max(c)
    mappable = ax.scatter(pos[:,1], pos[:,2], c=c, cmap=cmap, s=60, lw=0.3, ec="gray", zorder=2, vmin=vmin, vmax=vmax)
    fig.colorbar(mappable=mappable,label=cblabel)
    if emphasize_mes_pos:
        ax.scatter(pos[idx_mes_pos,1], pos[idx_mes_pos,2], s=120, lw=0.5, c="None", marker="o", ec="k", zorder=1)
    ds = 0.1
    xlim = [0-ds, 2*np.pi]
    ylim = [0-ds, ds+np.pi]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()
    ax.set_xlabel('Azimuth (rad)')
    ax.set_ylabel('Zenith (rad)')
    ax.set_title(title)
    if save:
        fig.savefig(f'{fname}.png', dpi=dpi)

    if clf:
        fig.clf()
        plt.close()


if __name__ == '__main__':
    pass
