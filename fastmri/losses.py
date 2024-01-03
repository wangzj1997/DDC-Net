"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X: torch.Tensor, Y: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)  # typing: ignore
        uy = F.conv2d(Y, self.w)  #
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()


class PSNRLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        fenzi = torch.sum((target - output) ** 2)
        fenmu = torch.sum(target ** 2)
        loss = torch.log(fenzi / fenmu + 1) * 10
        # print('')
        return loss


class LogLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        loss = torch.sum(torch.log(torch.abs(target - output) + 1)) / 320 / 2 / 240
        return loss


class GuiyiLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        loss = torch.sum(((target - output) ** 2) / (target ** 2 + 0.0000001)) / 320 / 2 / 240
        return loss


class Guiyi1Loss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        loss = torch.sum(torch.abs(target - output) / (torch.abs(target) + 0.0000001)) / 320 / 2 / 240
        return loss


class WeightLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.weight = [1, 1, 1, 1]
        L_wid = 40
        M_wid = 40
        H_wid = 40
        R_wid = 160 - L_wid - M_wid - H_wid
        # L
        self.L = torch.zeros(1,320,320,2)
        self.L[:,160-L_wid:160+L_wid,160-L_wid:160+L_wid,:] = 1
        # M
        self.M = torch.zeros(1,320,320,2)
        self.M[:,160-(M_wid + L_wid):160+(M_wid + L_wid),160-(M_wid + L_wid):160+(M_wid + L_wid),:] = 1
        self.M[:,160-L_wid:160+L_wid,160-L_wid:160+L_wid,:] = 0
        # H
        self.H = torch.zeros(1,320,320,2)
        self.H[:,160-(M_wid + L_wid + H_wid):160+(M_wid + L_wid + H_wid),160-(M_wid + L_wid + H_wid):160+(M_wid + L_wid + H_wid),:] = 1
        self.H[:,160-(M_wid + L_wid):160+(M_wid + L_wid),160-(M_wid + L_wid):160+(M_wid + L_wid),:] = 0
        # T 
        self.T = torch.ones(1,320,320,2)
        self.T[:,160-(M_wid + L_wid + H_wid):160+(M_wid + L_wid + H_wid),160-(M_wid + L_wid + H_wid):160+(M_wid + L_wid + H_wid),:] = 0

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        # junzhi
        L_mean = torch.sum(target * self.L) / torch.sum(self.L)
        M_mean = torch.sum(target * self.M) / torch.sum(self.M)
        H_mean = torch.sum(target * self.H) / torch.sum(self.H)
        T_mean = torch.sum(target * self.T) / torch.sum(self.T)
        s_L = torch.sum((target - L_mean) ** 2 * self.L)
        s_M = torch.sum((target - M_mean) ** 2 * self.M)
        s_H = torch.sum((target - H_mean) ** 2 * self.H)
        s_T = torch.sum((target - T_mean) ** 2 * self.T)
        loss_all = (target - output) ** 2
        loss_L  = torch.sum(loss_all * self.L) / s_L
        loss_M  = torch.sum(loss_all * self.M) / s_M
        loss_H  = torch.sum(loss_all * self.H) / s_H
        loss_T  = torch.sum(loss_all * self.T) / s_T
        return self.weight[0] * loss_L + self.weight[1] * loss_M + self.weight[2] * loss_H + self.weight[3] * loss_T


class l2_Loss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()

    def forward(self, output: torch.Tensor, target: torch.Tensor):
        batch = output.shape[0]
        loss = torch.sum((target - output) ** 2) / batch / 64 / 64
        return loss