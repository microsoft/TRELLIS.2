from typing import *
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from easydict import EasyDict as edict

from ..basic import BasicTrainer


class SparseStructureVaeTrainer(BasicTrainer):
    """
    Trainer for Sparse Structure VAE.
    
    Args:
        models (dict[str, nn.Module]): Models to train.
        dataset (torch.utils.data.Dataset): Dataset.
        output_dir (str): Output directory.
        load_dir (str): Load directory.
        step (int): Step to load.
        batch_size (int): Batch size.
        batch_size_per_gpu (int): Batch size per GPU. If specified, batch_size will be ignored.
        batch_split (int): Split batch with gradient accumulation.
        max_steps (int): Max steps.
        optimizer (dict): Optimizer config.
        lr_scheduler (dict): Learning rate scheduler config.
        elastic (dict): Elastic memory management config.
        grad_clip (float or dict): Gradient clip config.
        ema_rate (float or list): Exponential moving average rates.
        fp16_mode (str): FP16 mode.
            - None: No FP16.
            - 'inflat_all': Hold a inflated fp32 master param for all params.
            - 'amp': Automatic mixed precision.
        fp16_scale_growth (float): Scale growth for FP16 gradient backpropagation.
        finetune_ckpt (dict): Finetune checkpoint.
        log_param_stats (bool): Log parameter stats.
        i_print (int): Print interval.
        i_log (int): Log interval.
        i_sample (int): Sample interval.
        i_save (int): Save interval.
        i_ddpcheck (int): DDP check interval.
        
        loss_type (str): Loss type. 'bce' for binary cross entropy, 'l1' for L1 loss, 'dice' for Dice loss.
        lambda_kl (float): KL divergence loss weight.
    """
    
    def __init__(
        self,
        *args,
        loss_type='bce',
        lambda_kl=1e-6,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.lambda_kl = lambda_kl
    
    def training_losses(
        self,
        ss: torch.Tensor,
        **kwargs
    ) -> Tuple[Dict, Dict]:
        """
        Compute training losses.

        Args:
            ss: The [N x 1 x H x W x D] tensor of binary sparse structure.

        Returns:
            a dict with the key "loss" containing a scalar tensor.
            may also contain other keys for different terms.
        """
        z, mean, logvar = self.training_models['encoder'](ss.float(), sample_posterior=True, return_raw=True)
        logits = self.training_models['decoder'](z)

        terms = edict(loss = 0.0)
        if self.loss_type == 'bce':
            terms["bce"] = F.binary_cross_entropy_with_logits(logits, ss.float(), reduction='mean')
            terms["loss"] = terms["loss"] + terms["bce"]
        elif self.loss_type == 'l1':
            terms["l1"] = F.l1_loss(F.sigmoid(logits), ss.float(), reduction='mean')
            terms["loss"] = terms["loss"] + terms["l1"]
        elif self.loss_type == 'dice':
            logits = F.sigmoid(logits)
            terms["dice"] = 1 - (2 * (logits * ss.float()).sum() + 1) / (logits.sum() + ss.float().sum() + 1)
            terms["loss"] = terms["loss"] + terms["dice"]
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')
        terms["kl"] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        terms["loss"] = terms["loss"] + self.lamda_kl * terms["kl"]
            
        return terms, {}
    
    @torch.no_grad()
    def snapshot(self, suffix=None, num_samples=64, batch_size=1, verbose=False):
        super().snapshot(suffix=suffix, num_samples=num_samples, batch_size=batch_size, verbose=verbose)
    
    @torch.no_grad()
    def run_snapshot(
        self,
        num_samples: int,
        batch_size: int,
        verbose: bool = False,
    ) -> Dict:
        dataloader = DataLoader(
            copy.deepcopy(self.dataset),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.dataset.collate_fn if hasattr(self.dataset, 'collate_fn') else None,
        )

        # inference
        gts = []
        recons = []
        for i in range(0, num_samples, batch_size):
            batch = min(batch_size, num_samples - i)
            data = next(iter(dataloader))
            args = {k: v[:batch].cuda() if isinstance(v, torch.Tensor) else v[:batch] for k, v in data.items()}
            z = self.models['encoder'](args['ss'].float(), sample_posterior=False)
            logits = self.models['decoder'](z)
            recon = (logits > 0).long()
            gts.append(args['ss'])
            recons.append(recon)

        sample_dict = {
            'gt': {'value': torch.cat(gts, dim=0), 'type': 'sample'},
            'recon': {'value': torch.cat(recons, dim=0), 'type': 'sample'},
        }
        return sample_dict
