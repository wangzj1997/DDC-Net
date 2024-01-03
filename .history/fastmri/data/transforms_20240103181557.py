"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Dict, Optional, Sequence, Tuple, Union
from matplotlib import pyplot as plt

from torch._C import dtype
from torch.nn.functional import instance_norm

import fastmri
import numpy as np
import torch
# import cv2
from torch.nn import functional as F
import copy
from .subsample import MaskFunc
from fastmri.data.subsample import create_mask_for_mask_type
# import pywt
# from pytorch_wavelets import DWTForward, DWTInverse


def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.

    Args:
        data: Input numpy array.

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def tensor_to_complex_np(data: torch.Tensor) -> np.ndarray:
    """
    Converts a complex torch tensor to numpy array.

    Args:
        data: Input data to be converted to numpy.

    Returns:
        Complex numpy version of data.
    """
    data = data.numpy()

    return data[..., 0] + 1j * data[..., 1]


def apply_mask(
    data: torch.Tensor,
    mask_func: MaskFunc,
    seed: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values).
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Padding value to apply for mask.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data
            mask: The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    if padding is not None:
        mask[:, :, : padding[0]] = 0
        mask[:, :, padding[1] :] = 0  # padding value inclusive on right of zeros

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask


def mask_center(x: torch.Tensor, mask_from: int, mask_to: int) -> torch.Tensor:
    """
    Initializes a mask with the center filled in.

    Args:
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[:, mask_from:mask_to,:] = x[:,  mask_from:mask_to,:]

    return mask


def center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data: torch.Tensor, shape: Tuple[int, int]) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """
    if not (0 < shape[0] <= data.shape[-3] and 0 < shape[1] <= data.shape[-2]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to, :]

def i_complex_center_crop(data: torch.Tensor, shape: Tuple[int, int], data_croped: torch.Tensor) -> torch.Tensor:
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data: The complex input tensor to be center cropped. It should have at
            least 3 dimensions and the cropping is applied along dimensions -3
            and -2 and the last dimensions should have a size of 2.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image
    """

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    data[..., :, w_from:w_to, h_from:h_to] = data_croped[...]

    return data

def center_crop_to_smallest(
    x: torch.Tensor, y: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over dim=-1 and dim=-2. If x is smaller than y at
    dim=-1 and y is smaller than x at dim=-2, then the returned dimension will
    be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.

    Returns:
        tuple of tensors x and y, each cropped to the minimim size.
    """
    smallest_width = min(x.shape[-1], y.shape[-1])
    smallest_height = min(x.shape[-2], y.shape[-2])
    x = center_crop(x, (smallest_height, smallest_width))
    y = center_crop(y, (smallest_height, smallest_width))

    return x, y


def normalize(
    data: torch.Tensor,
    mean: Union[float, torch.Tensor],
    stddev: Union[float, torch.Tensor],
    eps: Union[float, torch.Tensor] = 0.0,
) -> torch.Tensor:
    """
    Normalize the given tensor.

    Applies the formula (data - mean) / (stddev + eps).

    Args:
        data: Input data to be normalized.
        mean: Mean value.
        stddev: Standard deviation.
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        Normalized tensor.
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(
    data: torch.Tensor, eps: Union[float, torch.Tensor] = 0.0
) -> Tuple[torch.Tensor, Union[torch.Tensor], Union[torch.Tensor]]:
    """
    Normalize the given tensor  with instance norm/

    Applies the formula (data - mean) / (stddev + eps), where mean and stddev
    are computed from the data itself.

    Args:
        data: Input data to be normalized
        eps: Added to stddev to prevent dividing by zero.

    Returns:
        torch.Tensor: Normalized tensor
    """
    mean = data.mean()
    std = data.std()

    return normalize(data, mean, std, eps), mean, std


class UnetDataTargetTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        import h5py
        hf = h5py.File('/home/biit/fastmri_dataset/singlecoil_knee1/singlecoil_val/file1000277.h5')
        target = hf['reconstruction_rss'][()][11]

        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)
        # image1 = fastmri.complex_abs(image)
        # from matplotlib import pyplot as plt
        # plt.imshow(image1,'gray')
        # plt.savefig('./test.png')
        # plt.imshow(target,'gray')
        # plt.savefig('./test1.png')

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = image.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            target, mean, std = normalize_instance(target, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return target, target, mean, std, fname, slice_num, max_value


class UnetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
            mask = torch.from_numpy(mask).unsqueeze(1).unsqueeze(0)

        # inverse Fourier transform to get zero filled solution
        # image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = masked_kspace.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, mask.permute(2,0,1).byte(), kspace.permute(2,0,1)


class UnetDataTransform_MC:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
            mask = torch.from_numpy(mask).unsqueeze(1).unsqueeze(0)

        # inverse Fourier transform to get zero filled solution
        # image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        # image = masked_kspace.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return masked_kspace, target, 0, 0, fname, slice_num, max_value, mask.byte()


class UnetDataRawTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
            mask = torch.from_numpy(mask).unsqueeze(1).unsqueeze(0)

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = image.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, mask.permute(2,0,1).byte()


class XiaoboTargetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

        self.xfm = DWTForward(J=1, mode='zero', wave='haar')
        self.ifm = DWTInverse(mode='zero', wave='haar')

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # normalize input
        image = image.permute(2,0,1)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            ll, hs = self.xfm(target.unsqueeze(0).unsqueeze(0))
            hs = hs[0][0]
            haar_image = torch.cat([ll, hs], 1)[0]
            # print(0)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, mask.permute(2,0,1).byte(), haar_image


class UnetDataAddGrappaTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        sx, sy, ncoils = masked_kspace.shape[:] # center 20 lines are ACS
        ctr, pd = int(sy/2), int(sy*0.04)
        calib = masked_kspace[:, ctr-pd:ctr+pd, :].numpy().copy()
        import time
        import matplotlib.pyplot as plt
        start_time = time.time()
        res = grappa(masked_kspace.numpy(), calib, kernel_size=(5, 5))
        print(time.time() - start_time)
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(1,2,1)
        ax.imshow(np.sqrt(np.sum(fastmri.ifft2c(masked_kspace).cpu().detach().numpy() ** 2, -1)), 'gray')
        ax = fig.add_subplot(1,2,2)
        ax.imshow(np.sqrt(np.sum(fastmri.ifft2c(torch.from_numpy(res)).cpu().detach().numpy() ** 2, -1)), 'gray')
        # ax = fig.add_subplot(3,1,3)
        # ax.imshow(np.log(np.sqrt(np.sum(error ** 2, -1)) + 0.000000000000000001), 'gray')
        plt.savefig('./test.png')

        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = image.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value


class KspaceDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        # image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        masked_kspace = masked_kspace.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return masked_kspace, target, 0, 0, fname, slice_num, max_value


class FcData32Transform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)
        image = center_crop(image.permute(2,0,1), [320, 320])
        image_target = fastmri.ifft2c(kspace)
        image_target = center_crop(image_target.permute(2,0,1), [320, 320])

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        # image = image.permute(2,0,1)
        # image_target = image_target.permute(2,0,1)
        image = F.interpolate(image.unsqueeze(0), (32,32), mode='bilinear').squeeze(0)
        # image = fastmri.fft2c(image.permute(1,2,0)).permute(2,0,1)
        image_target = F.interpolate(image_target.unsqueeze(0), (32,32), mode='bilinear').squeeze(0)
        image_target = fastmri.fft2c(image_target.permute(1,2,0)).permute(2,0,1)
        # image = fastmri.complex_abs(image.permute(0,2,3,1))
        # from matplotlib import pyplot as plt
        # plt.imshow(image[0], 'gray')
        # plt.savefig('./test.png')
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            target = F.interpolate(target.unsqueeze(0).unsqueeze(0), (32,32), mode='bilinear').squeeze(0)
            # from matplotlib import pyplot as plt
            # plt.imshow(target[0], 'gray')
            # plt.savefig('./test.png')
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return target, target, 0, 0, fname, slice_num, max_value, image_target


def guideFilter(I, p, winSize, eps, s):

    #输入图像的高、宽
    h, w = I.shape[:2]

    #缩小图像
    size = (int(round(w*s)), int(round(h*s)))

    small_I = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)
    small_p = cv2.resize(I, size, interpolation=cv2.INTER_CUBIC)

    #缩小滑动窗口
    X = winSize[0]
    small_winSize = (int(round(X*s)), int(round(X*s)))

    #I的均值平滑
    mean_small_I = cv2.blur(small_I, small_winSize)

    #p的均值平滑
    mean_small_p = cv2.blur(small_p, small_winSize)

    #I*I和I*p的均值平滑
    mean_small_II = cv2.blur(small_I*small_I, small_winSize)

    mean_small_Ip = cv2.blur(small_I*small_p, small_winSize)

    #方差
    var_small_I = mean_small_II - mean_small_I * mean_small_I #方差公式

    #协方差
    cov_small_Ip = mean_small_Ip - mean_small_I * mean_small_p

    small_a = cov_small_Ip / (var_small_I + eps)
    small_b = mean_small_p - small_a*mean_small_I

    #对a、b进行均值平滑
    mean_small_a = cv2.blur(small_a, small_winSize)
    mean_small_b = cv2.blur(small_b, small_winSize)

    #放大
    size1 = (w, h)
    mean_a = cv2.resize(mean_small_a, size1, interpolation=cv2.INTER_LINEAR)
    mean_b = cv2.resize(mean_small_b, size1, interpolation=cv2.INTER_LINEAR)

    q = mean_a*I + mean_b

    return q


class UnetDataWithLowFreqTargetTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace).permute(2,0,1)

        # normalize input
        # image = fastmri.complex_abs(image).unsqueeze(0)
        # image, mean, std = normalize_instance(image, eps=1e-11)
        # image = image.clamp(-6, 6)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            target_max = torch.max(target)
            target_min = torch.min(target)
            target_guiyi = ((target - target_min) / target_max).numpy()
            target_guiyi = torch.from_numpy(guideFilter(target_guiyi, target_guiyi, (16,16), 0.01, 0.5)).unsqueeze(0) * target_max + target_min
            # from matplotlib import pyplot as plt
            # plt.imshow(target_guiyi, 'gray')
            # plt.savefig('./test.png')
            # target = normalize(target.unsqueeze(0), mean, std, eps=1e-11)
            # target = target.clamp(-6, 6)
            # target_guiyi = normalize(target_guiyi, mean, std, eps=1e-11)
            # target_guiyi = target_guiyi.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target.unsqueeze(0), 0, 0, fname, slice_num, max_value, target_guiyi, mask


class UnetAbsDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = fastmri.complex_abs(image).unsqueeze(0)
        image = center_crop(image, [320,320])
        image, mean, std = normalize_instance(image, eps=1e-11)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target).unsqueeze(0)
            # target = center_crop(target, crop_size)
            target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, mean, std, fname, slice_num, max_value


class UnetDataTransform_320:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = complex_center_crop(image, [320,320])
        image = image.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return target.unsqueeze(0), target.unsqueeze(0), 0, 0, fname, slice_num, max_value, mask


class VarNetDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, int, float, torch.Tensor]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                masked_kspace: k-space after applying sampling mask.
                mask: The applied sampling mask
                target: The target image (if applicable).
                fname: File name.
                slice_num: The slice index.
                max_value: Maximum image value.
                crop_size: The size to crop the final image.
        """
        if target is not None:
            target = to_tensor(target)
            max_value = attrs["max"]
        else:
            target = torch.tensor(0)
            max_value = 0.0

        kspace = to_tensor(kspace)
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = torch.tensor([attrs["recon_size"][0], attrs["recon_size"][1]])

        if self.mask_func:
            masked_kspace, mask = apply_mask(
                kspace, self.mask_func, seed, (acq_start, acq_end)
            )
        else:
            masked_kspace = kspace
            shape = np.array(kspace.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask = mask.reshape(*mask_shape)
            mask[:, :, :acq_start] = 0
            mask[:, :, acq_end:] = 0

        return (
            masked_kspace,
            mask.byte(),
            target,
            fname,
            slice_num,
            max_value,
            crop_size,
        )


class LieFcDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.shape = [320, 320, 2]
        mask_func_ = create_mask_for_mask_type('equispaced', [0.08], [4])
        seed_ = (102, 105, 108, 101, 49, 48, 48, 50, 51, 56, 48, 46, 104, 53)
        self.mask = mask_func_(self.shape, seed_)
        # print('')

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # # apply mask
        # if self.mask_func:
        #     seed = None if not self.use_seed else tuple(map(ord, fname))
        #     masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        # else:
        #     masked_kspace = kspace

        # # inverse Fourier transform to get zero filled solution
        # image = fastmri.ifft2c(masked_kspace)

        full_image = fastmri.ifft2c(kspace)
        full_image_320 = complex_center_crop(full_image, (320,320))
        kspace = fastmri.fft2c(full_image_320)

        # apply mask
        masked_kspace = kspace * self.mask + 0.0

        # visual code.
        # from matplotlib import pyplot as plt
        # fade_kspace = torch.ones((320,320,1))
        # fade_kspace = fade_kspace * self.mask + 0.0
        # plt.imshow(fade_kspace[...,0].numpy(),'gray')
        # plt.savefig('/home/vpa/test.png')

        # normalize input
        kspace = kspace.permute(2,0,1)
        masked_kspace = masked_kspace.permute(2,0,1)
        kspace, mean, std = normalize_instance(kspace, eps=1e-11)
        masked_kspace = normalize(masked_kspace, mean, std, eps=1e-11)

        return masked_kspace, kspace, mean, std, fname, slice_num, max_value


class LieFc_With_Unet_DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.shape = [320, 320, 2]
        mask_func_ = create_mask_for_mask_type('equispaced', [0.08], [4])
        seed_ = (102, 105, 108, 101, 49, 48, 48, 50, 51, 56, 48, 46, 104, 53)
        self.mask = mask_func_(self.shape, seed_)
        # print('')

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # # apply mask
        # if self.mask_func:
        #     seed = None if not self.use_seed else tuple(map(ord, fname))
        #     masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        # else:
        #     masked_kspace = kspace

        # # inverse Fourier transform to get zero filled solution
        # image = fastmri.ifft2c(masked_kspace)

        full_image = fastmri.ifft2c(kspace)
        full_image_320 = complex_center_crop(full_image, (320,320))
        kspace = fastmri.fft2c(full_image_320)

        # apply mask
        masked_kspace = kspace * self.mask + 0.0

        # visual code.
        # from matplotlib import pyplot as plt
        # fade_kspace = torch.ones((320,320,1))
        # fade_kspace = fade_kspace * self.mask + 0.0
        # plt.imshow(fade_kspace[...,0].numpy(),'gray')
        # plt.savefig('/home/vpa/test.png')

        # normalize input
        kspace = kspace.permute(2,0,1)
        masked_kspace = masked_kspace.permute(2,0,1)
        kspace, mean, std = normalize_instance(kspace, eps=1e-11)
        masked_kspace = normalize(masked_kspace, mean, std, eps=1e-11)
        # normalize target
        if target is not None:
            target = to_tensor(target)
            target = center_crop(target, (320,320))
            target, mean_img, std_img = normalize_instance(target, eps=1e-11)
            # target = normalize(target, mean, std, eps=1e-11)
            target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return masked_kspace, kspace, mean, std, fname, slice_num, max_value, target, mean_img, std_img

class Unet_With_High_Freq_DataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # normalize input
        image = image.permute(2,0,1)

        # normalize target
        if target is not None:
            target = to_tensor(target)
        else:
            target = torch.Tensor([0])
        d_matrix = self.make_transform_matrix(20, (image.shape))
        high_pass_kspace = kspace * d_matrix
        high_pass_img = center_crop(fastmri.complex_abs(fastmri.ifft2c(high_pass_kspace)), (320,320))
        # import matplotlib.pyplot as plt
        # plt.imshow(high_pass_img, 'gray')
        # plt.savefig('/raid/MRI_group/test.png')
        return image, target, 0, 0, fname, slice_num, max_value, high_pass_img

    def make_transform_matrix(self, d, img_size):
        img_temp = torch.zeros(img_size)
        hangshu = torch.arange(0, img_size[1], 1)
        lieshu = torch.arange(0, img_size[2], 1)
        for i in range(img_size[2]):
            img_temp[0, :, i] = hangshu
        for i in range(img_size[1]):
            img_temp[1, i, :] = lieshu
        hangshu_mid = (img_size[1] - 1) / 2
        lieshu_mid = (img_size[2] - 1) / 2
        img_temp[0] -= hangshu_mid
        img_temp[1] -= lieshu_mid
        dis = torch.sqrt(img_temp[0] ** 2 + img_temp[1] ** 2)
        transfor_matrix = (dis >= d)
        img_temp[0] = transfor_matrix
        img_temp[1] = transfor_matrix
        return img_temp.permute(1,2,0)


class K_UnetDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        # image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        masked_kspace = masked_kspace.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            target_max = torch.max(target)
            target_min = torch.min(target)
            target_guiyi = ((target - target_min) / target_max).numpy()
            target_guiyi = torch.from_numpy(guideFilter(target_guiyi, target_guiyi, (16,16), 0.01, 0.5)).unsqueeze(0) * target_max + target_min
        else:
            target = torch.Tensor([0])

        return masked_kspace, target, 0, 0, fname, slice_num, max_value, mask.permute(2,0,1).byte(), target_guiyi

class ImageUnetWithTargetKDataTransform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = image.permute(2,0,1)
        # kspace = kspace.permute(2,0,1)
        # mask = mask.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, kspace, mask

    
class ImageUnetWithTargetKData320320Transform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.shape = [320, 320, 2]
        mask_func_ = create_mask_for_mask_type('equispaced', [0.08], [4])
        seed_ = (102, 105, 108, 101, 49, 48, 48, 50, 51, 56, 48, 46, 104, 53)
        self.mask = mask_func_(self.shape, seed_)

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0
        full_image = fastmri.ifft2c(kspace)
        full_image_320 = complex_center_crop(full_image, (320,320))
        kspace = fastmri.fft2c(full_image_320)

        # apply mask
        # if self.mask_func:
        #     seed = None if not self.use_seed else tuple(map(ord, fname))
        #     masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        # else:
        #     masked_kspace = kspace
        masked_kspace = kspace * self.mask + 0.0

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = image.permute(2,0,1)
        # kspace = kspace.permute(2,0,1)
        # mask = mask.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, kspace, self.mask


class ImageUnetWithTargetKData41_320320Transform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.shape = [320, 320, 2]
        mask_func_ = create_mask_for_mask_type('equispaced', [0.08], [4])
        seed_ = (102, 105, 108, 101, 49, 48, 48, 50, 51, 56, 48, 46, 104, 53)
        self.mask = 1 - mask_func_(self.shape, seed_)
        self.mask[:,160-13:160+13, :] = 1

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0
        full_image = fastmri.ifft2c(kspace)
        full_image_320 = complex_center_crop(full_image, (320,320))
        kspace = fastmri.fft2c(full_image_320)

        # apply mask
        # if self.mask_func:
        #     seed = None if not self.use_seed else tuple(map(ord, fname))
        #     masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        # else:
        #     masked_kspace = kspace
        masked_kspace = kspace * self.mask + 0.0

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = image.permute(2,0,1)
        # kspace = kspace.permute(2,0,1)
        # mask = mask.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, kspace, self.mask


class ImageUnetWithTargetKData41_320320Transform_multicoils:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed
        self.shape = [22, 640, 320, 2]
        mask_func_ = create_mask_for_mask_type('equispaced', [0.08], [4])
        seed_ = (102, 105, 108, 101, 49, 48, 48, 50, 51, 56, 48, 46, 104, 53)
        self.mask = mask_func_(self.shape, seed_)
        self.except_mid_mask = torch.zeros_like(self.mask)
        self.except_mid_mask[:, :, 160-13:160+13, :] = 1
        self.max_coils = 22

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        coils = kspace.shape[0]
        if coils > self.max_coils:
            self.max_coils = coils
            # print(self.max_coils)
        kspace = to_tensor(kspace)
        if coils < self.max_coils:
            gap = self.max_coils - coils
            if gap <= coils:
                kspace = torch.cat([kspace, kspace[:gap]], 0)
            else:
                while(kspace.shape[0]< self.max_coils):
                    kspace = torch.cat([kspace, kspace], 0)
                kspace = kspace[:self.max_coils]

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0
        # full_image = fastmri.ifft2c(kspace)
        # full_image_320 = complex_center_crop(full_image, (320,320))
        # kspace = fastmri.fft2c(full_image_320)

        # apply mask
        # if self.mask_func:
        #     seed = None if not self.use_seed else tuple(map(ord, fname))
        #     masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        # else:
        #     masked_kspace = kspace
        masked_kspace = kspace * self.mask + 0.0

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace)

        # crop input to correct size
        # if target is not None:
        #     crop_size = (target.shape[-2], target.shape[-1])
        # else:
        #     crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        # check for FLAIR 203
        # if image.shape[-2] < crop_size[1]:
        #     crop_size = (image.shape[-2], image.shape[-2])

        # image = complex_center_crop(image, crop_size)

        # absolute value
        # image = fastmri.complex_abs(image)

        # apply Root-Sum-of-Squares if multicoil data
        # if self.which_challenge == "multicoil":
        #     image = fastmri.rss(image)

        # normalize input
        image = self.complex_to_chan_dim(image.unsqueeze(0))[0]
        # kspace = kspace.permute(2,0,1)
        # mask = mask.permute(2,0,1)
        # image, mean, std = self.norm(image)

        # normalize target
        # if target is not None:
        #     target = to_tensor(target)
            # target = center_crop(target, crop_size)
            # target = normalize(target, mean, std, eps=1e-11)
            # target, mean_target, std_target = normalize_instance(target, eps=1e-11)
            # target = target.clamp(-6, 6)
        # else:
        #     target = torch.Tensor([0])
        # if image.shape[0] != 44:
        #     print('')
        target = center_crop(fastmri.complex_abs((fastmri.ifft2c(kspace))), [320, 320])
        target = fastmri.rss(target)

        return image, target, 0, 0, fname, slice_num, max_value, kspace, self.mask, self.except_mid_mask
    def complex_to_chan_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w, two = x.shape
        assert two == 2
        return x.permute(0, 4, 1, 2, 3).reshape(b, 2 * c, h, w)

    def chan_complex_to_last_dim(self, x: torch.Tensor) -> torch.Tensor:
        b, c2, h, w = x.shape
        assert c2 % 2 == 0
        c = c2 // 2
        return x.view(b, 2, c, h, w).permute(0, 2, 3, 4, 1).contiguous()

class UnetData_2channel_allresolution_Transform:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace).permute(2,0,1)

        # normalize target
        if target is not None:
            target = to_tensor(target)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, mask.byte(), masked_kspace


class UnetData_2channel_allresolution_Transform_visual:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace
            mask = torch.from_numpy(mask).float().unsqueeze(1).unsqueeze(0)

        masked_kspace_origin = copy.deepcopy(masked_kspace)
        image = fastmri.ifft2c(masked_kspace_origin).permute(2,0,1)



        masked_kspace_copy = masked_kspace_copy.permute(2, 0, 1)

        num_cols = kspace.shape[-2]
        num_low_freqs = int(round(num_cols * 0.08))
        pad = (num_cols - num_low_freqs + 1) // 2
        start = num_cols - 2 - pad
        masked_kspace_center = mask_center(masked_kspace, pad, pad + num_low_freqs)  # 640,368,2
        masked_kspace_center_img = fastmri.ifft2c(masked_kspace_center)



        masked_kspace_img_copy = fastmri.ifft2c(masked_kspace)
        masked_kspace_img = copy.deepcopy(masked_kspace_img_copy)

        for i in range(5):
            masked_kspace_img = torch.exp(1j * (torch.angle(masked_kspace_center_img))) * torch.abs(masked_kspace_img)
            masked_kspace = fastmri.fft2c(masked_kspace_img.float())
            masked_kspace = masked_kspace.permute(2, 0, 1)  # 2,640,368
            masked_kspace = (torch.rot90(masked_kspace, k=2, dims=[1, 2]))

            masked_kspace = masked_kspace.permute(2,0,1)
            masked_kspace[:, :, 0:start] = masked_kspace_copy[:, :, 0:start]
            masked_kspace = masked_kspace * (masked_kspace_copy == 0) + masked_kspace_copy
            masked_kspace_img = fastmri.ifft2c(masked_kspace.permute(1, 2, 0))

        # inverse Fourier transform to get zero filled solution
        masked_kspace_pocs = masked_kspace
        image_pocs = masked_kspace_img.permute(2,0,1)
        
        if target is not None:
            target = to_tensor(target)
        else:
            target = torch.Tensor([0])

        return image, target, 0, 0, fname, slice_num, max_value, mask.byte(), masked_kspace_copyy, image_pocs, masked_kspace_pocs

class UnetData_2channel_allresolution_Transform_md:
    """
    Data Transformer for training U-Net models.
    """

    def __init__(
        self,
        which_challenge: str,
        mask_func: Optional[MaskFunc] = None,
        use_seed: bool = True,
    ):
        """
        Args:
            which_challenge: Challenge from ("singlecoil", "multicoil").
            mask_func: Optional; A function that can create a mask of
                appropriate shape.
            use_seed: If true, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        if which_challenge not in ("singlecoil", "multicoil"):
            raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

        self.mask_func = mask_func
        self.which_challenge = which_challenge
        self.use_seed = use_seed

    def __call__(
        self,
        kspace: np.ndarray,
        mask: np.ndarray,
        target: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data or (rows, cols) for single coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            tuple containing:
                image: Zero-filled input image.
                target: Target image converted to a torch.Tensor.
                mean: Mean value used for normalization.
                std: Standard deviation value used for normalization.
                fname: File name.
                slice_num: Serial number of the slice.
        """
        kspace = to_tensor(kspace)
        image = fastmri.ifft2c(kspace)
        image_j = image[..., 0] + 1j * image[..., 1]

        norm = torch.abs(image_j)
        min = torch.min(norm)
        max = torch.max(norm)

        kspace = fastmri.fft2c(to_tensor((image_j - min) / (max - min) * 255))

        # check for max value
        max_value = attrs["max"] if "max" in attrs.keys() else 0.0

        # apply mask
        if self.mask_func:
            seed = None if not self.use_seed else tuple(map(ord, fname))
            masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
        else:
            masked_kspace = kspace

        # inverse Fourier transform to get zero filled solution
        image = fastmri.ifft2c(masked_kspace).permute(2,0,1)

        # normalize target
        if target is not None:
            target = to_tensor(target)
            target = (target - min) / (max - min) * 255
        else:
            target = torch.Tensor([0])

        return image, target, min, max, fname, slice_num, max_value, mask.byte(), masked_kspace

#     def batched_mask_center_high(
# x: torch.Tensor, mask_from_left: torch.Tensor, mask_to_left: torch.Tensor,mask_from_right: torch.Tensor, mask_to_right: torch.Tensor,
# ) -> torch.Tensor:
#         """
#         Initializes a mask with the center filled in.

#         Can operate with different masks for each batch element.

#         Args:
#             mask_from: Part of center to start filling.
#             mask_to: Part of center to end filling.

#         Returns:
#             A mask with the center filled.
#         """
#         if not mask_from_left.shape == mask_to_left.shape:
#             raise ValueError("mask_from and mask_to must match shapes.")
#         if not mask_from_left.ndim == 1:
#             raise ValueError("mask_from and mask_to must have 1 dimension.")
#         if not mask_from_left.shape[0] == 1:
#             if (not x.shape[0] == mask_from_left.shape[0]) or (
#                 not x.shape[0] == mask_to_left.shape[0]
#             ):
#                 raise ValueError("mask_from and mask_to must have batch_size length.")

#         if mask_from_left.shape[0] == 1:
#             mask = mask_center_high(x, int(mask_from_left), int(mask_to_left), int(mask_from_right), int(mask_to_right))
#         else:
#             mask = torch.zeros_like(x)
#             for i, (start, end) in enumerate(zip(mask_from_left, mask_to_left)):
#                 mask[i, :, :, start:end] = x[i, :, :, start:end]

#         return mask

# class UnetData_2channel_allresolution_Transform_visual:
# POCS
#     """
#     Data Transformer for training U-Net models.
#     """

#     def __init__(
#         self,
#         which_challenge: str,
#         mask_func: Optional[MaskFunc] = None,
#         use_seed: bool = True,
#     ):
#         """
#         Args:
#             which_challenge: Challenge from ("singlecoil", "multicoil").
#             mask_func: Optional; A function that can create a mask of
#                 appropriate shape.
#             use_seed: If true, this class computes a pseudo random number
#                 generator seed from the filename. This ensures that the same
#                 mask is used for all the slices of a given volume every time.
#         """
#         if which_challenge not in ("singlecoil", "multicoil"):
#             raise ValueError("Challenge should either be 'singlecoil' or 'multicoil'")

#         self.mask_func = mask_func
#         self.which_challenge = which_challenge
#         self.use_seed = use_seed

#     def __call__(
#         self,
#         kspace: np.ndarray,
#         mask: np.ndarray,
#         target: np.ndarray,
#         attrs: Dict,
#         fname: str,
#         slice_num: int,
#     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str, int, float]:
#         """
#         Args:
#             kspace: Input k-space of shape (num_coils, rows, cols) for
#                 multi-coil data or (rows, cols) for single coil data.
#             mask: Mask from the test dataset.
#             target: Target image.
#             attrs: Acquisition related information stored in the HDF5 object.
#             fname: File name.
#             slice_num: Serial number of the slice.

#         Returns:
#             tuple containing:
#                 image: Zero-filled input image.
#                 target: Target image converted to a torch.Tensor.
#                 mean: Mean value used for normalization.
#                 std: Standard deviation value used for normalization.
#                 fname: File name.
#                 slice_num: Serial number of the slice.
#         """
#         # import h5py
#         # hf = h5py.File('/raid/dataset/singlecoil_knee_wyz/singlecoil_val/file1001344.h5')
#         # kspace = hf['kspace'][15]
#         # # print(list(hf.keys()))
#         # target = hf['reconstruction_esc'][15]
#         kspace = to_tensor(kspace)

#         # check for max value
#         max_value = attrs["max"] if "max" in attrs.keys() else 0.0

#         # apply mask
#         if self.mask_func:
#             seed = None if not self.use_seed else tuple(map(ord, fname))
#             masked_kspace, mask = apply_mask(kspace, self.mask_func, seed)
#         else:
#             masked_kspace = kspace
#             mask = torch.from_numpy(mask).float().unsqueeze(1).unsqueeze(0)

#         masked_kspace_copyy = copy.deepcopy(masked_kspace)
#         masked_kspace_copy = copy.deepcopy(masked_kspace)
#         masked_kspace_copy = masked_kspace_copy.permute(2, 0, 1)

#         num_cols = kspace.shape[-2]
#         num_low_freqs = int(round(num_cols * 0.08))
#         pad = (num_cols - num_low_freqs + 1) // 2
#         start = num_cols - 2 - pad
#         masked_kspace_center = mask_center(masked_kspace, pad, pad + num_low_freqs)  # 640,368,2
#         masked_kspace_center_img = fastmri.ifft2c(masked_kspace_center)



#         masked_kspace_img_copy = fastmri.ifft2c(masked_kspace)
#         masked_kspace_img = copy.deepcopy(masked_kspace_img_copy)

#         for i in range(5):
#             masked_kspace_img = torch.exp(1j * (torch.angle(masked_kspace_center_img))) * torch.abs(masked_kspace_img)
#             masked_kspace = fastmri.fft2c(masked_kspace_img.float())
#             masked_kspace = masked_kspace.permute(2, 0, 1)  # 2,640,368
#             masked_kspace = (torch.rot90(masked_kspace, k=2, dims=[1, 2]))

#             # masked_kspace = masked_kspace.permute(2,0,1)
#             # masked_kspace[:, :, 0:start] = masked_kspace_copy[:, :, 0:start]
#             # masked_kspace = masked_kspace * (masked_kspace_copy == 0) + masked_kspace_copy
#             masked_kspace_img = fastmri.ifft2c(masked_kspace.permute(1, 2, 0))

#         # inverse Fourier transform to get zero filled solution
#         # plt.imshow(torch.log(fastmri.complex_abs((masked_kspace.permute(1,2,0).float()))), cmap='gray')
#         # plt.show()
#         masked_kspace_pocs = masked_kspace
#         image = fastmri.ifft2c(masked_kspace_copyy).permute(2,0,1)
#         image_pocs = masked_kspace_img.permute(2,0,1)
#         # normalize target
#         if target is not None:
#             target = to_tensor(target)
#         else:
#             target = torch.Tensor([0])

#         return image, target, 0, 0, fname, slice_num, max_value, mask.byte(), masked_kspace_copyy, image_pocs, masked_kspace_pocs