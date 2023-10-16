from torchvision import transforms
import torch
import torchvision.transforms.functional as F
from model import INPUT_SIZE
import random
import torch.nn as nn
from typing import Optional
from PIL import Image
SQUEEZE_RATIO = 2.5



class GaussianNoise(torch.nn.Module):
    """Add Gaussian noise to a tensor.
    
    Args:
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    """
    
    def __init__(self, mean=0., std=3.0):
        super(GaussianNoise, self).__init__()
        # The scale factor is used to scale the noise tensor to the same range as the input tensor
        # Since torch transforms work with PIL images, the input tensor is expected to be in the range [0, 1]
        self.scale_factor = 255.0
        self.mean = mean/self.scale_factor
        self.std = std/self.scale_factor
        

    def forward(self, tensor):
        """Apply Gaussian noise to the input tensor.
        
        Args:
            tensor (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Tensor with added Gaussian noise.
        """
        noise = torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean
        noisy_tensor = tensor + noise
        # Clip the values to be in [0, 1] range
        noisy_tensor = torch.clamp(noisy_tensor, 0, 1)
        return noisy_tensor


    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"

class GradientFill(torch.nn.Module):
    def __init__(self, scale: float = 0.1, orientation: Optional[str] = None):
        super(GradientFill, self).__init__()
        self.scale = scale
        if orientation is None:
            self.orientation = random.choice(['horizontal', 'vertical', 'radial'])
        else:
            self.orientation = orientation

    def forward(self, img_tensor):
        shading_tensor = self._create_shading_tensor(img_tensor.shape[-2:], self.orientation)
        # Only apply shading to background (where pixel values are close to 1)
        background_mask = (img_tensor > 0.9).float()
        img_tensor = img_tensor * (1 - background_mask) + (shading_tensor) * background_mask
        img_tensor = torch.clamp(img_tensor, 0, 1)  # Ensure values are within [0, 1]
        
        return img_tensor

    def _create_shading_tensor(self, shape, orientation):
        if orientation == 'horizontal':
            shading_tensor = torch.linspace(-1, 1, steps=shape[1])[None, :].repeat(shape[0], 1)
        elif orientation == 'vertical':
            shading_tensor = torch.linspace(-1, 1, steps=shape[0])[:, None].repeat(1, shape[1])
        elif orientation == 'radial':
            x = torch.linspace(-1, 1, steps=shape[1])[None, :].repeat(shape[0], 1)
            y = torch.linspace(-1, 1, steps=shape[0])[:, None].repeat(1, shape[1])
            shading_tensor = torch.sqrt(x**2 + y**2)
        else:
            raise ValueError(f"Unsupported orientation: {orientation}")

        
        
        # Normalize the tensor to be between 0 and 1
        shading_tensor = (shading_tensor - shading_tensor.min()) / (shading_tensor.max() - shading_tensor.min())
        shading_tensor = (1-self.scale) + self.scale * shading_tensor  # Scale the values to be in [0.9, 1] range
        return shading_tensor[None, :, :]  # Shape: [1, H, W]
    
    def __repr__(self):
        return self.__class__.__name__ + f"(scale={self.scale},orientation={self.orientation})"

class VariableAspectRatio(nn.Module):
    def __init__(self, fixed_height=INPUT_SIZE, min_ratio=5/6, max_ratio=7/6):
        super(VariableAspectRatio, self).__init__()
        self.fixed_height = fixed_height
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def forward(self, img):
        # Get a random width ratio from the specified range
        width_ratio = torch.rand(1).item() * (self.max_ratio - self.min_ratio) + self.min_ratio
        
        # Calculate new width
        new_width = int(img.shape[-1] * width_ratio)
        
        # Resize the image
        return F.resize(img, (self.fixed_height, new_width),antialias=True)

    def __repr__(self):
        return self.__class__.__name__ + '(fixed_height={0}, min_ratio={1}, max_ratio={2})'.format(self.fixed_height, self.min_ratio, self.max_ratio)


class SqueezingCrop(torch.nn.Module):
    """Squeezing operation for image tensors.
    
    The operation resizes the image to a given height and adjusts the width based on a specified aspect ratio.
    
    Args:
        height (int): The desired height of the output image.
        aspect_ratio (float): The desired width-to-height aspect ratio for the output image.
    """
    
    def __init__(self, height=INPUT_SIZE, aspect_ratio=SQUEEZE_RATIO):
        super(SqueezingCrop, self).__init__()
        self.height = height
        self.aspect_ratio = aspect_ratio
    
    def forward(self, tensor_img):
        """
        Args:
            tensor_img (torch.Tensor): The input image tensor of shape [B, C, H, W].
            
        Returns:
            torch.Tensor: The squeezed image tensor.
        """
        new_width = int(self.height * self.aspect_ratio)

        # Resize the tensor to the desired height and computed width
        resized_tensor = F.resize(tensor_img, (self.height, new_width),antialias=True)
        cropped_tensor = F.center_crop(resized_tensor, self.height)
        return cropped_tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(height={self.height}, aspect_ratio={self.aspect_ratio})"


class RandomTransforms(torch.nn.Module):
    """Applies a list of transformations with a given probability."""
    def __init__(self, transforms, p=1.0):
        super().__init__()
        self.transforms = transforms
        self.p = p

    def forward(self, x):
        if random.random() < self.p:
            for t in self.transforms:
                x = t(x)
        return x

def transform_pipeline(squeeze_ratio=SQUEEZE_RATIO):
    all_transforms = [
        # Add Gaussian noise
        GaussianNoise(0., 3),
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

    transformations = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        RandomTransforms(chosen_transforms),
        SqueezingCrop(INPUT_SIZE, squeeze_ratio),
    ])
    return transformations

def pad_to_square(img):
    w, h = img.size
    max_dim = max(w, h)
    hp = (max_dim - w) // 2
    vp = (max_dim - h) // 2
    padding = (hp, vp, hp, vp)  # left, top, right, bottom
    # 255 for white padding.
    return transforms.functional.pad(img, padding, 255, 'constant')

TRANSFORMS_PAD = transforms.Compose([
    transforms.Grayscale(),
    transforms.Lambda(pad_to_square),
    transforms.Resize((INPUT_SIZE, INPUT_SIZE),antialias=True),
    transforms.ToTensor()
])

TRANSFORMS_CROP = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((INPUT_SIZE,), interpolation=transforms.InterpolationMode.BICUBIC,antialias=True),
    transforms.CenterCrop(INPUT_SIZE),
    transforms.ToTensor()
])

TRANSFORMS_SQUEEZE = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    SqueezingCrop(INPUT_SIZE, SQUEEZE_RATIO),
])

TRANSFORMS_CNN = transform_pipeline()

# img = Image.open('./test_images/syn/2123129.png')
# img_crop = TRANSFORMS_CROP(img)
# img_crop.save('./test_images/syn/2123129_crop.png')