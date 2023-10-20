from torchvision import transforms
import torch
import torchvision.transforms.functional as F
from model import INPUT_SIZE
import random
import torch.nn as nn
from typing import Optional
from torch import Tensor
from typing import List, Any
from PIL import Image
import h5py
import os
import io
import numpy as np
from tqdm import tqdm

SQUEEZE_RATIO = 2.5


class GaussianNoise(torch.nn.Module):
    """Add Gaussian noise to a tensor.

    Args:
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    """

    def __init__(self, mean: int = 0.0, std: int = 3.0):
        super(GaussianNoise, self).__init__()
        # The scale factor is used to scale the noise tensor to the same range as the input tensor
        # Since torch transforms work with PIL images, the input tensor is expected to be in the range [0, 1]
        assert isinstance(mean, int), 'mean must be an integer.'
        assert isinstance(std, int), 'std must be an integer.'
        self.scale_factor = 255.0
        self.mean = mean / self.scale_factor
        self.std = std / self.scale_factor

    def forward(self, tensor: Tensor) -> Tensor:
        """Apply Gaussian noise to the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor with added Gaussian noise.
        """
        assert isinstance(tensor, Tensor), 'input must be a tensor.'
        noise = torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        noisy_tensor = tensor + noise
        # Clip the values to be in [0, 1] range
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"


class GradientFill(torch.nn.Module):
    def __init__(self, scale: float = 0.1, orientation: Optional[str] = None):
        super(GradientFill, self).__init__()
        assert isinstance(scale, float), 'scale must be a float.'
        self.scale = scale
        if orientation is None:
            self.orientation = random.choice(['horizontal', 'vertical', 'radial'])
        else:
            self.orientation = orientation

    def forward(self, img_tensor: Tensor) -> Tensor:
        assert isinstance(img_tensor, Tensor), 'input must be a tensor.'
        shading_tensor = self._create_shading_tensor(
            img_tensor.shape[-2:], self.orientation
        )
        # Only apply shading to background (where pixel values are close to 1)
        background_mask = (img_tensor > 0.9).float()
        img_tensor = (
            img_tensor * (1 - background_mask) + (shading_tensor) * background_mask
        )
        img_tensor = torch.clamp(img_tensor, 0, 1)  # Ensure values are within [0, 1]

        return img_tensor

    def _create_shading_tensor(
        self, shape: Tensor, orientation: str = 'horizontal'
    ) -> Tensor:
        if orientation == 'horizontal':
            shading_tensor = torch.linspace(-1, 1, steps=shape[1])[None, :].repeat(
                shape[0], 1
            )
        elif orientation == 'vertical':
            shading_tensor = torch.linspace(-1, 1, steps=shape[0])[:, None].repeat(
                1, shape[1]
            )
        elif orientation == 'radial':
            x = torch.linspace(-1, 1, steps=shape[1])[None, :].repeat(shape[0], 1)
            y = torch.linspace(-1, 1, steps=shape[0])[:, None].repeat(1, shape[1])
            shading_tensor = torch.sqrt(x**2 + y**2)
        else:
            raise ValueError(f"Unsupported orientation: {orientation}")

        # Normalize the tensor to be between 0 and 1
        shading_tensor = (shading_tensor - shading_tensor.min()) / (
            shading_tensor.max() - shading_tensor.min()
        )
        shading_tensor = (
            1 - self.scale
        ) + self.scale * shading_tensor  # Scale the values to be in [0.9, 1] range
        return shading_tensor[None, :, :]  # Shape: [1, H, W]

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(scale={self.scale},orientation={self.orientation})"
        )


class VariableAspectRatio(nn.Module):
    def __init__(
        self,
        fixed_height: int = INPUT_SIZE,
        min_ratio: float = 5 / 6,
        max_ratio: float = 7 / 6,
    ):
        super(VariableAspectRatio, self).__init__()
        assert isinstance(fixed_height, int), 'fixed_height must be an integer.'
        assert isinstance(
            [min_ratio, max_ratio], float
        ), 'min_ratio and max_ratio must be floats.'
        self.fixed_height = fixed_height
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def forward(self, img: Tensor) -> Tensor:
        assert isinstance(img, Tensor), 'input must be a tensor.'
        # Get a random width ratio from the specified range
        width_ratio = (
            torch.rand(1).item() * (self.max_ratio - self.min_ratio) + self.min_ratio
        )

        # Calculate new width
        new_width = int(img.shape[-1] * width_ratio)

        # Resize the image
        return F.resize(img, (self.fixed_height, new_width), antialias=True)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + '(fixed_height={0}, min_ratio={1}, max_ratio={2})'.format(
                self.fixed_height, self.min_ratio, self.max_ratio
            )
        )


class SqueezingCrop(torch.nn.Module):
    """Squeezing operation for image tensors.

    The operation resizes the image to a given height and adjusts the width based on a specified aspect ratio.

    Args:
        height (int): The desired height of the output image.
        aspect_ratio (float): The desired width-to-height aspect ratio for the output image.
    """

    def __init__(self, height: int = INPUT_SIZE, aspect_ratio: float = SQUEEZE_RATIO):
        super(SqueezingCrop, self).__init__()
        assert isinstance(height, int), 'height must be an integer.'
        assert isinstance(aspect_ratio, float), 'aspect_ratio must be a float.'
        self.height = height
        self.aspect_ratio = aspect_ratio

    def forward(self, tensor_img: Tensor) -> Tensor:
        """
        Args:
            tensor_img (torch.Tensor): The input image tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: The squeezed image tensor.
        """
        assert isinstance(tensor_img, Tensor), 'input must be a tensor.'
        new_width = int(self.height * self.aspect_ratio)

        # Resize the tensor to the desired height and computed width
        resized_tensor = F.resize(tensor_img, (self.height, new_width), antialias=True)
        cropped_tensor = F.center_crop(resized_tensor, self.height)
        return cropped_tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(height={self.height}, aspect_ratio={self.aspect_ratio})"


class RandomTransforms(torch.nn.Module):
    """Applies a list of transformations with a given probability."""

    def __init__(self, transforms: List[Any], p: float = 1.0):
        super().__init__()
        assert isinstance(transforms, list), 'transforms must be a list.'
        assert isinstance(p, float), 'p must be a float.'
        self.transforms = transforms
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        assert isinstance(x, Tensor), 'input must be a tensor.'
        if random.random() < self.p:
            for t in self.transforms:
                x = t(x)
        return x


def transform_pipeline(squeeze_ratio: float = SQUEEZE_RATIO) -> transforms.Compose:
    assert isinstance(squeeze_ratio, float), 'squeeze_ratio must be a float.'
    all_transforms = [
        # Add Gaussian noise
        GaussianNoise(0.0, 3),
        # Gaussian blur
        transforms.GaussianBlur(3, sigma=(2.5, 3.5)),
        # Perspective Rotation
        transforms.RandomPerspective(fill=1.0),
        # Gradient fill
        GradientFill(0.5),
        # Adjust character spacing
        # Not implemented, might be handled during text generation
        # Adjust aspect ratio (squeezing)
        VariableAspectRatio(INPUT_SIZE),
    ]

    # Randomly choose 1, 2, or 3 transformations
    num_transforms = random.choice([1, 2, 3])
    chosen_transforms = random.sample(all_transforms, num_transforms)

    transformations = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
            RandomTransforms(chosen_transforms),
            SqueezingCrop(INPUT_SIZE, squeeze_ratio),
        ]
    )
    return transformations


def pad_to_square(img: Image.Image) -> Any:
    assert isinstance(img, Image.Image), 'input must be an image.'
    w, h = img.size
    max_dim = max(w, h)
    hp = (max_dim - w) // 2
    vp = (max_dim - h) // 2
    padding = (hp, vp, hp, vp)  # left, top, right, bottom
    # 255 for white padding.
    return transforms.functional.pad(img, padding, 255, 'constant')


TRANSFORMS_PAD = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Lambda(pad_to_square),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE), antialias=True),
        transforms.ToTensor(),
    ]
)

TRANSFORMS_CROP = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.Resize(
            (INPUT_SIZE,),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
    ]
)

TRANSFORMS_SQUEEZE = transforms.Compose(
    [
        transforms.Grayscale(),
        transforms.ToTensor(),
        SqueezingCrop(INPUT_SIZE, SQUEEZE_RATIO),
    ]
)

# TRANSFORMS_CNN = transform_pipeline()

# img = Image.open('./test_images/syn/2123129.png')
# img_crop = TRANSFORMS_CROP(img)
# img_crop.save('./test_images/syn/2123129_crop.png')


def images_to_hdf5(root_dir, hdf5_file_path):
    total_images = sum(
        [
            len(os.listdir(os.path.join(root_dir, d)))
            for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ]
    )

    with h5py.File(hdf5_file_path, 'w') as hf:
        img_dtype = h5py.vlen_dtype(
            np.dtype('uint8')
        )  # Variable length byte arrays for images
        img_dset = hf.create_dataset("images", shape=(total_images,), dtype=img_dtype)
        label_dset = hf.create_dataset(
            "labels", shape=(total_images,), dtype=h5py.string_dtype(encoding='utf-8')
        )

        idx = 0  # Index of the current image
        font_dirs = [
            d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
        ]
        for font_dir in tqdm(font_dirs):
            font_path = os.path.join(root_dir, font_dir)
            images = [img for img in os.listdir(font_path) if img.endswith('.png')]
            for img_name in images:
                img_path = os.path.join(font_path, img_name)

                with Image.open(img_path) as img:
                    with io.BytesIO() as buffer:
                        img.save(buffer, format="PNG")
                        img_byte_array = buffer.getvalue()

                img_dset[idx] = np.frombuffer(img_byte_array, dtype='uint8')
                label_dset[idx] = font_dir 
                idx += 1


# root_dir = '/public/dataset/AdobeVFR/Raw Image/VFR_syn_train'
# hdf5_file_path = '/public/dataset/AdobeVFR/hdf5/VFR_syn_train_bk.hdf5'
# images_to_hdf5(root_dir, hdf5_file_path)
