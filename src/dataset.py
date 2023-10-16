import os
from torch.utils.data import Dataset
from PIL import Image
import random
from abc import ABC, abstractmethod
import torchvision.transforms as transforms
from typing import Tuple, Optional


class AdobeVFRAbstractDataset(Dataset, ABC):
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.transform = transform

    @abstractmethod
    def _load_image(self, file_name: str):
        pass


class VFRRealUDataset(AdobeVFRAbstractDataset):
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        super().__init__(root_dir, transform)
        self.image_files = [
            f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpeg'))
        ]
        self.error_count = 0

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        # when some images are corrupted, we just skip them and randomly select another image.
        image = self._load_image(self.image_files[idx])
        while image is None:
            idx = random.randint(0, len(self.image_files) - 1)
            image = self._load_image(self.image_files[idx])
        return image, -1  # Return a dummy label with the image

    def _load_image(self, file_name: str) -> Optional[Image.Image]:
        img_path = os.path.join(self.root_dir, file_name)
        try:
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            return image
        except Exception:
            # print(f"Error loading image {img_path}: {e}")
            return None


class VFRSynDataset(AdobeVFRAbstractDataset):
    def __init__(
        self,
        root_dir: str,
        font_list_path: str,
        transform: Optional[transforms.Compose],
    ):
        super().__init__(root_dir, transform)
        with open(font_list_path, 'r') as f:
            self.font_families = f.read().splitlines()

    def __len__(self) -> int:
        # 1000 images per font family
        return len(self.font_families) * 1000

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        label = idx // 1000
        font_family = self.font_families[idx // 1000]
        image_name = f"{font_family}_{idx % 1000}.png"
        image_filename = os.path.join(self.root_dir, font_family, image_name)
        image = self._load_image(image_filename)
        return image, label
    
    def _check(self,idx: int) -> Tuple[Image.Image, int]:
        label = idx // 1000
        font_family = self.font_families[idx // 1000]
        image_name = f"{font_family}_{idx % 1000}.png"
        image_filename = os.path.join(self.root_dir, font_family, image_name)
        return image_filename

    def _load_image(self, file_name: str) -> None:
        img_path = os.path.join(self.root_dir, file_name)
        try:
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            return image
        except Exception:
            raise Exception(f"Error loading image {img_path}")

    def _label2font(self, label: int) -> str:
        return self.font_families[label]
