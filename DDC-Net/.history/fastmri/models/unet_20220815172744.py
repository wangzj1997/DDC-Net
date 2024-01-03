"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from fastmri import data
from fastmri.fftc import fft2c_new, ifft2c_new
import torch
from torch import nn
from torch.nn import functional as F
import fastmri


class Unet(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.conv(output)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        return output


class Unet_Kimgsampling(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.weight_k = nn.ParameterList([nn.Parameter(torch.ones(1))])

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        ch = chans
        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2
            self.weight_k.append(nn.Parameter(torch.ones(1)))
        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            # self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            self.up_conv.append(ConvBlock(ch , ch, drop_prob))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_conv.append(
            nn.Sequential(
                # ConvBlock(ch * 2, ch, drop_prob),
                ConvBlock(ch, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        output = image

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            # stack.append(output)
            output = self.complex_to_chan_dim(fastmri.ifft2c(self.chan_complex_to_last_dim(output)))
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
            output = self.complex_to_chan_dim(fastmri.fft2c(self.chan_complex_to_last_dim(output)))

        output = self.conv(output)
        output = self.complex_to_chan_dim(fastmri.ifft2c(self.chan_complex_to_last_dim(output)))

        # apply up-sampling layers
        for transpose_conv, conv, wfi in zip(self.up_transpose_conv, self.up_conv, self.weight_k):
            downsample_layer = stack.pop()
            # output = self.complex_to_chan_dim(fastmri.ifft2c(self.chan_complex_to_last_dim(output)))
            output = transpose_conv(output)
            # output = self.complex_to_chan_dim(fastmri.fft2c(self.chan_complex_to_last_dim(output)))

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            # output = torch.cat([output, downsample_layer], dim=1)
            output = (output + downsample_layer * wfi) / (1 + wfi)
            # output = output + downsample_layer
            # output = self.complex_to_chan_dim(fastmri.fft2c(self.chan_complex_to_last_dim(output)))
            output = conv(output)

        output = self.complex_to_chan_dim(fastmri.fft2c(self.chan_complex_to_last_dim(output)))
        
        return output

    def complex_to_chan_dim(self, x):
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).contiguous().view(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x):
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 第一个全连接层 (降维)
            nn.ReLU(inplace=True),                                 # ReLU 非线性激活函数
            nn.Linear(channel // reduction, channel, bias=False),  # 第二个全连接层 (升维)
            nn.Sigmoid()                                           # 非线性激活函数 + 数值范围约束 (0, 1)
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # 即上文所述的 U
        y = self.fc(y).view(b, c, 1, 1)  # reshape 张量以便于进行通道重要性加权的乘法操作
 
        return x * y.expand_as(x)  # 按元素一一对应相乘

class Unet_yent(nn.Module):
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 2,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        self.down_sample_layers_high = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])

        ch = chans

        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            self.down_sample_layers_high.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2

        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.fusion = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()
        for _ in range(num_pool_layers - 1):
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            # self.fusion.append(BiFusion_block(ch_1=ch, ch_2=ch, r_2=4, ch_int=ch, ch_out=ch))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        # self.fusion.append(BiFusion_block(ch_1=ch, ch_2=ch, r_2=4, ch_int=ch, ch_out=ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor, image_high: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        stack_high = []
        output = image
        output_high = image_high

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        for layer in self.down_sample_layers_high:
            output_high = layer(output_high)
            stack_high.append(output_high)
            output_high = F.avg_pool2d(output_high, kernel_size=2, stride=2, padding=0)

        output = self.conv(output+output_high)

        # apply up-sampling layers
        for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
            downsample_layer = stack.pop()
            # downsample_layer_high = stack_high.pop()
            output = transpose_conv(output)

            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            # fushion_map = fuse(downsample_layer,downsample_layer_high)
            output = torch.cat([output, downsample_layer], dim=1)
            output = conv(output)

        # image = output.permute(0, 2, 3, 1)
        # output = fastmri.fft2c(output)
        # image = torch.where(masked_kspace.to(bool), kspace, output)
        # image = fastmri.ifft2c(image)

        return output

class Unet_add_channel_attention_add_res_addcat(nn.Module):
    # """
    # PyTorch implementation of a U-Net model.

    # O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    # for biomedical image segmentation. In International Conference on Medical
    # image computing and computer-assisted intervention, pages 234–241.
    # Springer, 2015.
    # """

    # def __init__(
    #     self,
    #     in_chans: int,
    #     out_chans: int,
    #     chans: int = 32,
    #     num_pool_layers: int = 4,
    #     drop_prob: float = 0.0,
    # ):
    #     """
    #     Args:
    #         in_chans: Number of channels in the input to the U-Net model.
    #         out_chans: Number of channels in the output to the U-Net model.
    #         chans: Number of output channels of the first convolution layer.
    #         num_pool_layers: Number of down-sampling and up-sampling layers.
    #         drop_prob: Dropout probability.
    #     """
    #     super().__init__()

    #     self.in_chans = in_chans
    #     self.out_chans = out_chans
    #     self.chans = chans
    #     self.num_pool_layers = num_pool_layers
    #     self.drop_prob = drop_prob

    #     self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
    #     ch = chans
    #     for _ in range(num_pool_layers - 1):
    #         self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
    #         ch *= 2
    #     self.conv = ConvBlock_DGD(ch, ch, drop_prob)

    #     self.up_conv = nn.ModuleList()
    #     self.up_transpose_conv = nn.ModuleList()
    #     self.c_a_fc = nn.ModuleList()
    #     for _ in range(num_pool_layers - 1):
    #         self.up_transpose_conv.append(TransposeConvBlock(ch, ch))
    #         self.c_a_fc.append(SELayer(ch, 8))
    #         self.up_conv.append(ConvBlock_BBD(ch, ch // 2, drop_prob))

    #         ch //= 2

    #     self.up_transpose_conv.append(TransposeConvBlock(ch, ch))
    #     self.c_a_fc.append(SELayer(ch, 4))
    #     self.up_conv.append(
    #         nn.Sequential(
    #             ConvBlock(ch, ch, drop_prob),
    #             nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
    #         )
    #     )

    # def forward(self, image: torch.Tensor) -> torch.Tensor:
    #     """
    #     Args:
    #         image: Input 4D tensor of shape `(N, in_chans, H, W)`.

    #     Returns:
    #         Output tensor of shape `(N, out_chans, H, W)`.
    #     """
    #     stack = []
    #     stack_res = []
    #     output = image

    #     # apply down-sampling layers
    #     for layer in self.down_sample_layers:
    #         output = layer(output)
    #         stack.append(output)
    #         output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)
    #         stack_res.append(output)

    #     output = self.conv(output)

    #     # apply up-sampling layers
    #     for transpose_conv, conv, attention_fc in zip(self.up_transpose_conv, self.up_conv, self.c_a_fc):
    #         downsample_res = stack_res.pop()
    #         output = output + downsample_res
    #         downsample_layer = stack.pop()
    #         output = transpose_conv(output)

    #         # reflect pad on the right/botton if needed to handle odd input dimensions
    #         padding = [0, 0, 0, 0]
    #         if output.shape[-1] != downsample_layer.shape[-1]:
    #             padding[1] = 1  # padding right
    #         if output.shape[-2] != downsample_layer.shape[-2]:
    #             padding[3] = 1  # padding bottom
    #         if torch.sum(torch.tensor(padding)) != 0:
    #             output = F.pad(output, padding, "reflect")

    #         output = output + downsample_layer
    #         # output = torch.cat([output, downsample_layer], dim=1)
    #         output = attention_fc(output)
    #         output = conv(output)

    #     return output
    """
    PyTorch implementation of a U-Net model.

    O. Ronneberger, P. Fischer, and Thomas Brox. U-net: Convolutional networks
    for biomedical image segmentation. In International Conference on Medical
    image computing and computer-assisted intervention, pages 234–241.
    Springer, 2015.
    """

    def __init__(
        self,
        in_chans: int = 2,
        out_chans: int = 2,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
    ):
        """
        Args:
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            chans: Number of output channels of the first convolution layer.
            num_pool_layers: Number of down-sampling and up-sampling layers.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob

        self.down_sample_layers = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])
        self.down_sample_layers_high = nn.ModuleList([ConvBlock(in_chans, chans, drop_prob)])

        ch = chans

        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(ConvBlock(ch, ch * 2, drop_prob))
            self.down_sample_layers_high.append(ConvBlock(ch, ch * 2, drop_prob))
            ch *= 2

        self.conv = ConvBlock(ch, ch * 2, drop_prob)

        self.up_conv = nn.ModuleList()
        self.up_transpose_conv = nn.ModuleList()

        self.up_conv_high = nn.ModuleList()
        self.up_transpose_conv_high = nn.ModuleList()


        for _ in range(num_pool_layers - 1):
            # self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            # self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
            self.up_transpose_conv_high.append(TransposeConvBlock(ch * 2, ch))
            self.up_conv.append(ConvBlock(ch * 2, ch, drop_prob))
            self.up_conv_high.append(ConvBlock(ch * 2, ch, drop_prob))
            # self.fusion.append(BiFusion_block(ch_1=ch, ch_2=ch, r_2=4, ch_int=ch, ch_out=ch))
            ch //= 2

        self.up_transpose_conv.append(TransposeConvBlock(ch * 2, ch))
        self.up_transpose_conv_high.append(TransposeConvBlock(ch * 2, ch))

        # self.fusion.append(BiFusion_block(ch_1=ch, ch_2=ch, r_2=4, ch_int=ch, ch_out=ch))
        self.up_conv.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )
        self.up_conv_high.append(
            nn.Sequential(
                ConvBlock(ch * 2, ch, drop_prob),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def forward(self, image: torch.Tensor, image_high: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        stack = []
        stack_high = []
        output = image
        output_high = image_high

        # apply down-sampling layers
        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        for layer in self.down_sample_layers_high:
            output_high = layer(output_high)
            stack_high.append(output_high)
            output_high = F.avg_pool2d(output_high, kernel_size=2, stride=2, padding=0)

        output = self.conv(output+output_high)
        output_high = output
        # apply up-sampling layers
        # for transpose_conv, conv in zip(self.up_transpose_conv, self.up_conv):
        for transpose_conv, conv, transpose_conv_high, conv_high in zip(self.up_transpose_conv, self.up_conv, self.up_transpose_conv_high, self.up_conv_high):
            downsample_layer = stack.pop()
            downsample_layer_high = stack_high.pop()
            output = transpose_conv(output)
            output_high = transpose_conv_high(output_high)
            # reflect pad on the right/botton if needed to handle odd input dimensions
            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = 1  # padding right
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding)) != 0:
                output = F.pad(output, padding, "reflect")

            padding_high = [0, 0, 0, 0]
            if output_high.shape[-1] != downsample_layer_high.shape[-1]:
                padding_high[1] = 1  # padding right
            if output_high.shape[-2] != downsample_layer_high.shape[-2]:
                padding_high[3] = 1  # padding bottom
            if torch.sum(torch.tensor(padding_high)) != 0:
                output_high = F.pad(output_high, padding_high, "reflect")

            output = torch.cat([output, downsample_layer + downsample_layer_high], dim=1)
            output_high = torch.cat([output_high, downsample_layer_high], dim=1)

            output = conv(output)
            output_high = conv_high(output_high)
        # image = output.permute(0, 2, 3, 1)
        # output = fastmri.fft2c(output)
        # image = torch.where(masked_kspace.to(bool), kspace, output)
        # image = fastmri.ifft2c(image)

        return output, output_high

class ConvBlock(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class ConvBlock_BBD(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, in_chans // 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(in_chans // 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(in_chans // 2, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class ConvBlock_DGD(nn.Module):
    """
    A Convolutional Block that consists of two convolution layers each followed by
    instance normalization, LeakyReLU activation and dropout.
    """

    def __init__(self, in_chans: int, out_chans: int, drop_prob: float):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
            drop_prob: Dropout probability.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.drop_prob = drop_prob

        self.layers = nn.Sequential(
            nn.Conv2d(in_chans, in_chans * 2, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(in_chans * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            nn.Conv2d(in_chans * 2, out_chans, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H, W)`.
        """
        return self.layers(image)


class TransposeConvBlock(nn.Module):
    """
    A Transpose Convolutional Block that consists of one convolution transpose
    layers followed by instance normalization and LeakyReLU activation.
    """

    def __init__(self, in_chans: int, out_chans: int):
        """
        Args:
            in_chans: Number of channels in the input.
            out_chans: Number of channels in the output.
        """
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_chans, out_chans, kernel_size=2, stride=2, bias=False
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: Input 4D tensor of shape `(N, in_chans, H, W)`.

        Returns:
            Output tensor of shape `(N, out_chans, H*2, W*2)`.
        """
        return self.layers(image)
