from torchvision import transforms
import torch
import torchvision.transforms.functional as F
import random
import torch.nn as nn
from typing import Optional
from torch import Tensor
from typing import List, Any
from PIL import Image, ImageEnhance
import numpy as np
from .config import SQUEEZE_RATIO, INPUT_SIZE, NUM_RANDOM_CROP


class GaussianNoise(torch.nn.Module):
    """Add Gaussian noise to a PIL image.

    Args:
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    """

    def __init__(self, mean: float = 0.0, std: float = 10.0):
        super(GaussianNoise, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, pil_image: Image.Image) -> Image.Image:
        """Apply Gaussian noise to the input PIL image.

        Args:
            pil_image (PIL.Image): Input PIL image.

        Returns:
            PIL.Image: PIL image with added Gaussian noise.
        """
        np_image = np.asarray(pil_image)
        noise = np.random.normal(self.mean, self.std, np_image.shape)
        noisy_np_image = np.clip(np_image + noise, 0, 255)
        noisy_pil_image = Image.fromarray(np.uint8(noisy_np_image))
        return noisy_pil_image

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
        assert isinstance(min_ratio, float) and isinstance(
            max_ratio, float
        ), 'min_ratio and max_ratio must be floats.'
        self.fixed_height = fixed_height
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def forward(self, img: Image.Image) -> Image.Image:
        assert isinstance(img, Image.Image), 'input must be a PIL Image.'

        # Get a random width ratio from the specified range
        width_ratio = (
            torch.rand(1).item() * (self.max_ratio - self.min_ratio) + self.min_ratio
        )

        # Calculate new width
        current_width, _ = img.size
        new_width = int(current_width * width_ratio)

        # Resize the image
        return F.resize(img, (self.fixed_height, new_width), antialias=True)

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + '(fixed_height={0}, min_ratio={1}, max_ratio={2})'.format(
                self.fixed_height, self.min_ratio, self.max_ratio
            )
        )


class Squeezing(torch.nn.Module):
    """Squeezing operation for image tensors.

    The operation resizes the image to a given height and adjusts the width based on a specified aspect ratio.

    Args:
        height (int): The desired height of the output image.
        aspect_ratio (float): The desired width-to-height aspect ratio for the output image.
    """

    def __init__(self, height: int = INPUT_SIZE, aspect_ratio: float = SQUEEZE_RATIO):
        super(Squeezing, self).__init__()
        assert isinstance(height, int), 'height must be an integer.'
        assert isinstance(aspect_ratio, float), 'aspect_ratio must be a float.'
        self.height = height
        self.aspect_ratio = aspect_ratio

    def forward(self, img: Image) -> Image:
        """
        Args:
            tensor_img (torch.Tensor): The input image tensor of shape [B, C, H, W].

        Returns:
            torch.Tensor: The squeezed image tensor.
        """
        assert isinstance(img, Image.Image), 'input must be a PIL Image.'
        new_width = int(self.height * self.aspect_ratio)
        # Resize the tensor to the desired height and computed width
        squeezed_img = F.resize(img, (self.height, new_width), antialias=True)
        return squeezed_img

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(height={self.height}, aspect_ratio={self.aspect_ratio})"


class SqueezingMultiCrop(torch.nn.Module):
    """Squeezing operation for image tensors.

    The operation resizes the image to a given height and adjusts the width based on a specified aspect ratio.

    Args:
        height (int): The desired height of the output image.
        aspect_ratio (float): The desired width-to-height aspect ratio for the output image.
        num_crops (int): The number of random crops to generate.
    """

    def __init__(
        self,
        height: int,
        aspect_ratio: float,
        num_crops: int,
    ):
        super(SqueezingMultiCrop, self).__init__()
        assert isinstance(height, int), 'height must be an integer.'
        assert isinstance(aspect_ratio, float), 'aspect_ratio must be a float.'
        self.height = height
        self.aspect_ratio = aspect_ratio
        self.num_crops = num_crops

    def forward(self, tensor_img: Tensor) -> Tensor:
        """
        Args:
            tensor_img (torch.Tensor): The input image tensor of shape [C, H, W].

        Returns:
            torch.Tensor: The squeezed image tensor.
        """
        assert isinstance(tensor_img, Tensor), 'input must be a tensor.'
        assert tensor_img.dim() == 3, 'input must be a 3D tensor(grey-scale).'
        C, H, W = tensor_img.shape
        new_width = int(self.height * self.aspect_ratio)

        # Resize the tensor to the desired height and computed width
        resized_tensor = F.resize(tensor_img, (self.height, new_width), antialias=True)

        # Randomly crop the tensor multiple times
        crops = []
        for _ in range(self.num_crops):
            left = random.randint(0, new_width - self.height)
            cropped_tensor = resized_tensor[:, : self.height, left : left + self.height]
            crops.append(cropped_tensor)

        return crops

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(height={self.height}, aspect_ratio={self.aspect_ratio})"


class Shading(torch.nn.Module):
    def __init__(self, brightness_factor: float = 0.75, contrast_factor: float = 0.8):
        """
        Initialize the Shading transform.

        Args:
        - brightness_factor (float): Factor to adjust the brightness. 1 is original image,
                                     < 1 will darken the image, and > 1 will brighten the image.
        - contrast_factor (float): Factor to adjust the contrast. 1 is original image,
                                   < 1 will decrease the contrast, and > 1 will increase the contrast.
        """
        super(Shading, self).__init__()
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor

    def forward(self, pil_image: Image.Image) -> Image.Image:
        """
        Apply shading to the input PIL image.

        Args:
        - pil_image (PIL.Image): Input PIL image.

        Returns:
        - PIL.Image: Shaded PIL image.
        """
        enhancer = ImageEnhance.Brightness(pil_image)
        shaded_img = enhancer.enhance(self.brightness_factor)

        enhancer = ImageEnhance.Contrast(shaded_img)
        shaded_img = enhancer.enhance(self.contrast_factor)

        return shaded_img

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + f"(brightness_factor={self.brightness_factor}, contrast_factor={self.contrast_factor})"
        )


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
            Squeezing(INPUT_SIZE, squeeze_ratio),
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


def custom_collate_fn(batch):
    images, labels = zip(*batch)
    # Flatten the list of image patches
    images = [img for sublist in images for img in sublist]
    # Repeat labels for each patch
    labels = [label for label in labels for _ in range(NUM_RANDOM_CROP)]
    return torch.stack(images), torch.tensor(labels)


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

TRANSFORMS_EVAL = transforms.Compose(
    # more transform will implemented in trainer.eval()
    [
        transforms.Grayscale(),
    ]
)

TRANSFORMS_TRAIN = transforms.Compose(
    [
        transforms.Grayscale(),
        GaussianNoise(),
        transforms.GaussianBlur(3, sigma=(2.5, 3.5)),
        transforms.RandomPerspective(fill=255, distortion_scale=0.1),
        Shading(),
        VariableAspectRatio(INPUT_SIZE),
        Squeezing(INPUT_SIZE, SQUEEZE_RATIO),
        transforms.RandomCrop(INPUT_SIZE),
        transforms.ToTensor(),
    ]
)

AUGMENTATION_LIST = [
    GaussianNoise(),
    transforms.GaussianBlur(3, sigma=(2.5, 3.5)),
    transforms.RandomPerspective(fill=255, distortion_scale=0.1),
    Shading(),
    VariableAspectRatio(INPUT_SIZE),
]

