"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
from tkinter import Y
import fastmri

import torch
from torch import nn
from fastmri.models.unet import Inet, Knet
from torch.nn import functional as F
from fastmri.data import transforms
from fastmri.losses import SSIMLoss

from .mri_module import MriModule


class NormInet(nn.Module):
    def __init__(self, inp, out, chans, num_pools):
        super().__init__()
        self.unet = Inet(inp, out, chans, num_pools)
        
    def norm(self, x):
        # # Group norm
        b, c, h, w = x.shape
        x = x.contiguous().view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1, 1).expand(
            b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1, 1).expand(
            b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        x = x.view(b, c, h, w)
        return (x - mean) / std, mean, std
    
    def unnorm(self, x, mean, std):
        # if(x.shape[1]==mean.shape[1]):
        return x * std + mean
        # if(x.shape[1]!=mean.shape[1]):
        #     return x * std[:,0:2,:,:] + mean[:,0:2,:,:]

    def forward(self, x, y):
        # y = y.permute(0,3,1,2)
        x, mean, std = self.norm(x)
        y, mean_high, std_high = self.norm(y)
        x = self.unet(x, y)
        x = self.unnorm(x, mean, std)
        # y = self.unnorm(y, mean_high, std_high)
        return x

class NormKnet(nn.Module):
    def __init__(self, inp, out, chans, num_pools):
        super().__init__()
        self.unet = Knet(inp, out, chans, num_pools)

    def norm(self, x):
        # Group norm
        b, c, h, w = x.shape
        x = x.contiguous().view(b, 2, c // 2 * h * w)
        mean = x.mean(dim=2).view(b, 2, 1, 1, 1).expand(
            b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        std = x.std(dim=2).view(b, 2, 1, 1, 1).expand(
            b, 2, c // 2, 1, 1).contiguous().view(b, c, 1, 1)
        x = x.view(b, c, h, w)

        return (x - mean) / std, mean, std

    def unnorm(self, x, mean, std):
        # if(x.shape[1]==mean.shape[1]):
        return x * std + mean
        # if(x.shape[1]!=mean.shape[1]):
        #     return x * std[:,0:2,:,:] + mean[:,0:2,:,:]        

    def forward(self, x):
        x, mean, std = self.norm(x)
        x = self.unet(x)
        x = self.unnorm(x, mean, std)
        return x

class DualUNetSimpleModule(MriModule):
    """
    Unet training module.

    This can be used to train baseline U-Nets from the paper:

    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """

    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        **kwargs,
    ):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

        self.dtrans_i = nn.ModuleList()
        self.dtrans_k = nn.ModuleList()

        self.dc_weight_i = nn.ParameterList()
        self.dc_weight_k = nn.ParameterList()
        self.fuse_weight_i = nn.ParameterList()

        for i in range(8):
            self.dc_weight_i.append(nn.Parameter(torch.ones(1)))
            self.dc_weight_k.append(nn.Parameter(torch.ones(1)))
            self.fuse_weight_i.append(nn.Parameter(torch.ones(1)))
            self.dtrans_i.append(NormInet(4, 2, 32, 3))
            self.dtrans_k.append(NormKnet(4, 2, 12, 3))

        self.loss = SSIMLoss()


    def forward(self, image, mask, masked_kspace,image_pocs, masked_kspace_pocs):
    # def forward(self, image, mask, masked_kspace):
        # kspace = masked_kspace.clone().permute(0,3,1,2)
        # kspace = masked_kspace_pocs.clone()
        zero = torch.zeros(1, 1, 1, 1).to(masked_kspace)
        # zero = torch.zeros(1, 1, 1, 1).to(masked_kspace_pocs)
        pad, num_low_freqs = self.get_pad_and_num_low_freqs(mask)

        masked_kspace_high = self.batched_mask_center_high(
            masked_kspace.permute(0,3,1,2), pad - pad, pad, pad + num_low_freqs, 2 * pad + num_low_freqs).permute(0,2,3,1)
        
        masked_kspace_high_pocs = self.batched_mask_center_high(masked_kspace_pocs, pad - pad, pad, pad + num_low_freqs, 2 * pad + num_low_freqs).permute(0,2,3,1)

        image_high = fastmri.ifft2c(masked_kspace_high).permute(0,3,1,2)
        image_high_copy = image_high
        image_high_pocs = fastmri.ifft2c(masked_kspace_high_pocs).permute(0,3,1,2)

        kspace = torch.cat([masked_kspace.permute(0,3,1,2), masked_kspace_pocs], dim=1)
        image = torch.cat([image, image_pocs], dim=1)
        image_high = torch.cat([image_high, image_high_pocs], dim = 1)
        i = 0
        # kspace = torch.cat([masked_kspace.permute(0,3,1,2), kspace], dim=1)

        for li, lk, wi, wk, wfi in zip(self.dtrans_i, self.dtrans_k, self.dc_weight_i, self.dc_weight_k, self.fuse_weight_i):
            # ikik
            # image = li(image, image_high)
            # image_k = fastmri.fft2c(image.permute(0,2,3,1))
            # image_k_dc = image_k - torch.where(mask, image_k - masked_kspace, zero) * wi
            # kspace = image_k_dc.permute(0,3,1,2)
            # kspace = lk(kspace) + kspace
            # kspace_k = kspace.permute(0,2,3,1)
            # kspace_k_dc = kspace_k - torch.where(mask, kspace_k - masked_kspace, zero) * wk

            # image = fastmri.ifft2c(kspace_k_dc).permute(0,3,1,2)
            # kspace_k_dc_high = self.batched_mask_center_high(
            # kspace_k_dc.permute(0,3,1,2), pad - pad, pad, pad + num_low_freqs, 2 * pad + num_low_freqs
            # ).permute(0,2,3,1)
            # image_high = fastmri.ifft2c(kspace_k_dc_high).permute(0,3,1,2)

            # ikik 601version_2
            image = li(image, image_high)
            image_k = fastmri.fft2c(image.permute(0,2,3,1))
            image_k_dc = image_k - torch.where(mask, image_k - masked_kspace, zero) * wi
            kspace = image_k_dc.permute(0,3,1,2)

            kspace = torch.cat([kspace, masked_kspace_pocs], dim=1)
            kspace = lk(kspace)
            kspace_k = kspace.permute(0,2,3,1)
            kspace_k_dc = kspace_k - torch.where(mask, kspace_k - masked_kspace, zero) * wk
            kspace_k_dc_high = self.batched_mask_center_high(kspace_k_dc.permute(0,3,1,2), pad - pad, pad, pad + num_low_freqs, 2 * pad + num_low_freqs).permute(0,2,3,1)


            image = fastmri.ifft2c(kspace_k_dc).permute(0,3,1,2)
            image_high = fastmri.ifft2c(kspace_k_dc_high).permute(0,3,1,2)


        unet_out_abs = fastmri.complex_abs(fastmri.ifft2c(kspace_k_dc))
        return unet_out_abs

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor):
        # get low frequency line locations and mask them out
        # mask = mask.unsqueeze(-1)
        # print(mask.shape)torch.Size([1, 1, 1, 368, 1])
        squeezed_mask = mask[:, 0, :, 0]
        # print(squeezed_mask.shape)torch.Size([1, 368])
        cent = squeezed_mask.shape[1] // 2
        # print(cent)
        # print(squeezed_mask[:, :cent].shape)


        # running argmin returns the first non-zero
        left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
        right = torch.argmin(squeezed_mask[:, cent:], dim=1)
        num_low_freqs = torch.max(
            2 * torch.min(left, right), torch.ones_like(left)
        )  # force a symmetric center unless 1

        # if self.num_sense_lines is not None:  # Use prespecified number instead
        #     if (num_low_freqs < num_sense_lines).all():
        #         raise RuntimeError(
        #             "`num_sense_lines` cannot be greater than the actual number of "
        #             "low-frequency lines in the mask: {}".format(num_low_freqs)
        #         )
        #     num_low_freqs = num_sense_lines * torch.ones(
        #         mask.shape[0], dtype=mask.dtype, device=mask.device
        #     )0

        pad = (mask.shape[-2] - num_low_freqs + 1) // 2
        return pad, num_low_freqs

    def batched_mask_center_high(self, x: torch.Tensor, mask_from_left: torch.Tensor, mask_to_left: torch.Tensor,mask_from_right: torch.Tensor, mask_to_right: torch.Tensor,) -> torch.Tensor:
        """
        Initializes a mask with the center filled in.

        Can operate with different masks for each batch element.

        Args:
            mask_from: Part of center to start filling.
            mask_to: Part of center to end filling.

        Returns:
            A mask with the center filled.
        """
        if not mask_from_left.shape == mask_to_left.shape:
            raise ValueError("mask_from and mask_to must match shapes.")
        if not mask_from_left.ndim == 1:
            raise ValueError("mask_from and mask_to must have 1 dimension.")
        if not mask_from_left.shape[0] == 1:
            if (not x.shape[0] == mask_from_left.shape[0]) or (
                not x.shape[0] == mask_to_left.shape[0]
            ):
                raise ValueError("mask_from and mask_to must have batch_size length.")

        if mask_from_left.shape[0] == 1:
            mask = torch.zeros_like(x)
            mask[:, :, int(mask_from_left):int(mask_to_left), :] = x[:, :, int(mask_from_left):int(mask_to_left), :]
            mask[:, :, int(mask_from_right):int(mask_to_right), :] = x[:, :, int(mask_from_right):int(mask_to_right), :]
        else:
            mask = torch.zeros_like(x)
            for i, (start, end) in enumerate(zip(mask_from_left, mask_to_left)):
                mask[i, :, :, start:end] = x[i, :, :, start:end]

        return mask

    def training_step(self, batch, batch_idx):
        image, target, mean, std, fname, slice_num, max_value, mask, masked_kspace = batch
        output = self(image, mask, masked_kspace)
        output, target = transforms.center_crop_to_smallest(output.unsqueeze(0), target.unsqueeze(0))
        loss = self.loss(output, target, max_value)

        self.log("loss", loss.detach())

        return loss

    def validation_step(self, batch, batch_idx):
        image, target, mean, std, fname, slice_num, max_value, mask, masked_kspace = batch
        output = self(image, mask, masked_kspace)
        output, target = transforms.center_crop_to_smallest(output.unsqueeze(0), target.unsqueeze(0))

        return {
            "batch_idx": batch_idx,
            "fname": fname,
            "slice_num": slice_num,
            "max_value": max_value,
            "output": output[0][0],
            "target": target[0][0],
            "val_loss": self.loss(output, target, max_value),
        }

    def test_step(self, batch, batch_idx):
        # image, target, mean, std, fname, slice_num, max_value, mask, masked_kspace , image_pocs = batch
        # output = self(image, mask, masked_kspace,image_pocs)
        # output = transforms.center_crop(output, [320,320])
        image, target, mean, std, fname, slice_num, max_value, mask, masked_kspace= batch
        output = self(image, mask, masked_kspace)
        output = transforms.center_crop(output, [320,320])

        return {
            "fname": fname,
            "slice": slice_num,
            "output": output.cpu().numpy(),
        }

    def configure_optimizers(self):
        optim = torch.optim.RMSprop(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, self.lr_step_size, self.lr_gamma
        )

        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
        """
        Define parameters that only apply to this model
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--in_chans", default=1, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=1, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=1, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.0, type=float, help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", default=0.1, type=float, help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser
