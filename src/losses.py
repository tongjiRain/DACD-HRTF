import torch as th
import torch.nn.functional as F
import torchaudio as ta

class LSD(th.nn.Module):
    """
    Log-Spectral Distance loss, computed in decibel scale.
    Supports optional per-sample masking.
    """
    def __init__(self):
        super().__init__()

    def forward(self, data, target, mask=None):
        return self._loss(data, target, mask)

    def _loss(self, data, target, mask):
        """
        Args:
            data:   (S, B, 2, L) complex or float tensor, predicted HRTFs
            target: (S, B, 2, L) complex or float tensor, ground-truth HRTFs
            mask:   (S, B) float tensor indicating valid samples

        Returns:
            A scalar LSD loss value
        """
        mag2db = ta.transforms.AmplitudeToDB(stype='magnitude')
        data_db = mag2db(th.abs(data))
        target_db = mag2db(th.abs(target))

        if mask is not None:
            lsd_elements = (data_db - target_db).pow(2)          # (S, B, 2, L)
            lsd_mean = th.mean(lsd_elements, dim=-1)             # (S, B, 2)
            mask_expand = mask[:, :, None].expand(-1, -1, 2)     # (S, B, 2)
            weighted_lsd = th.sqrt(lsd_mean) * mask_expand       # masked LSD (S, B, 2)
            return th.sum(weighted_lsd) / th.sum(mask_expand)    # mean over valid samples
        else:
            return th.mean(th.sqrt(th.mean((data_db - target_db).pow(2), dim=-1)))  # scalar

class LSD_before_mean(th.nn.Module):
    """
    Log-Spectral Distance before averaging: returns per-sample LSD without reduction.
    """
    def __init__(self):
        super().__init__()

    def forward(self, data, target):
        return self._loss(data, target)

    def _loss(self, data, target):
        """
        Args:
            data:   (S, B, 2, L) complex or float tensor
            target: (S, B, 2, L) complex or float tensor

        Returns:
            LSD: (S, B, 2) tensor of per-ear per-sample LSD
        """
        mag2db = ta.transforms.AmplitudeToDB(stype='magnitude')
        data_db = mag2db(th.abs(data))
        target_db = mag2db(th.abs(target))
        return th.sqrt(th.mean((data_db - target_db).pow(2), dim=-1))


class ILD_Loss(th.nn.Module):
    """
    Interaural Level Difference loss.
    Compares ILD (L - R) in dB scale between predicted and target HRTFs.
    """
    def __init__(self):
        super().__init__()

    def forward(self, data, target, mask=None):
        return self._loss(data, target, mask)

    def _loss(self, data, target, mask):
        """
        Args:
            data:   (S, B, 2, L) complex or float tensor, predicted HRTFs
            target: (S, B, 2, L) complex or float tensor, ground-truth HRTFs
            mask:   (S, B) float tensor indicating valid samples

        Returns:
            A scalar ILD loss value
        """
        eps = th.tensor(1e-10, device=data.device)
        mag_data = th.abs(data)
        mag_target = th.abs(target)

        ild_data = 20 * th.log10((mag_data[:, :, 0, :] + eps) / (mag_data[:, :, 1, :] + eps))      # (S, B, L)
        ild_target = 20 * th.log10((mag_target[:, :, 0, :] + eps) / (mag_target[:, :, 1, :] + eps))  # (S, B, L)

        ild_diff = th.abs(ild_data - ild_target)        # (S, B, L)
        ild_loss_mean = th.mean(ild_diff, dim=-1)       # (S, B)

        if mask is not None:
            weighted_loss = ild_loss_mean * mask        # apply mask
            return th.sum(weighted_loss) / th.sum(mask) # mean over valid entries

        return th.mean(ild_loss_mean)


class CosDistIntra(th.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, z,mask = None):
        return self._loss(z,mask)
    def _loss(self, z,mask):
        '''
        args
        -----
        z: (S,B,dimz) tensor
        mask: (S,B) float tensor
        return
        -----
        a scalar loss value
        '''
        c = th.mean(z, dim=1)  # centroid; prototype
        cs = F.cosine_similarity(z, c[:, None, :], dim=2)
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (S,B) -> (S,B,1)

            mask_z = z*mask            # (S,B,dimz)
            mask_z_sum = mask_z.sum(dim=1)  # (S,dimz)
            valid_z = mask.sum(dim=1)  # (S,1)
            c = mask_z_sum / valid_z       #   (S,dimz)
            cs = F.cosine_similarity(z, c[:, None, :], dim=2)
            cosdist = (1 - cs).pow(2)
            cosdist_mask = cosdist * mask.squeeze(-1)   # (S,B)
            mask_sum = th.sum(mask)
            return th.sqrt(th.sum(cosdist_mask) / mask_sum)

        return th.mean((1-cs).pow(2))**0.5

class Triplet_loss(th.nn.Module):
    """
    Standard triplet loss with margin.
    Encourages anchor-positive pairs to be closer than anchor-negative by at least margin.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        return self._loss(anchor, positive, negative)

    def _loss(self, anchor, positive, negative):
        """
        Args:
            anchor:   (S, Dimz) tensor
            positive: (S, Dimz) tensor
            negative: (S, Dimz) tensor

        Returns:
            A scalar margin-based triplet loss
        """
        pos_dist = F.pairwise_distance(anchor, positive)  # (S,)
        neg_dist = F.pairwise_distance(anchor, negative)  # (S,)
        loss = F.relu(pos_dist - neg_dist + self.margin)  # (S,)
        return loss.mean()


class Ortho_Loss(th.nn.Module):
    """
    Orthogonality loss between two sets of latent embeddings.
    Minimizes the dot product between z_person and z_db.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z_person_avg, z_db_avg):
        return self._loss(z_person_avg, z_db_avg)

    def _loss(self, z_person_avg, z_db_avg):
        """
        Args:
            z_person_avg: (S, Dim1) tensor
            z_db_avg:     (S, Dim2) tensor

        Returns:
            A scalar orthogonality loss
        """
        dot_product = (z_person_avg * z_db_avg).sum(dim=1)  # (S,)
        return th.mean(th.abs(dot_product))                 # scalar


