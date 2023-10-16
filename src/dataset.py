import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import random
from preprocess import transform_pipeline
class AdobeVFRAbstractDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def load_image(self, file_name):
        img_path = os.path.join(self.root_dir, file_name)
        try:
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            # print(f"Error loading image {img_path}: {e}")
            return None

class VFRRealUDataset(AdobeVFRAbstractDataset):
    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpeg'))]
        self.error_count = 0
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # when some images are corrupted, we just skip them and randomly select another image.
        image = self.load_image(self.image_files[idx])
        while image is None:
            idx = random.randint(0, len(self.image_files) - 1)
            image = self.load_image(self.image_files[idx])
        return image,-1  # Return a dummy label with the image


class VFRSynDataset(AdobeVFRAbstractDataset):
    def __init__(self, root_dir, csv_file, transform=None, max_label=2382):
        super().__init__(root_dir, transform)
        self.labels_frame = pd.read_csv(csv_file)
        
        # Filter labels that are out of range
        self.labels_frame = self.labels_frame[self.labels_frame.iloc[:, 1].between(0, max_label)]
        print(f'Syn Transforms: {self.transform}')

    def __len__(self):
        return len(self.labels_frame)
    
    def __getitem__(self, idx):
        file_name = f"{self.labels_frame.iloc[idx, 0]}.png"
        label = int(self.labels_frame.iloc[idx, 1])
        image = self.load_image(file_name)
        while image is None or label > 2382:
            idx = random.randint(0, len(self.labels_frame) - 1)
            file_name = f"{self.labels_frame.iloc[idx, 0]}.png"
            image = self.load_image(file_name)
            label = int(self.labels_frame.iloc[idx, 1])
        return image, label
    
    def load_image(self, file_name):
        # transform = self.transform if self.transform else transform_pipeline()
        assert self.transform is not None, "Transform is None."
        transform = self.transform
        img_path = os.path.join(self.root_dir, file_name)
        try:
            image = Image.open(img_path)
            if transform:
                image = transform(image)
            return image
        except Exception as e:
            # print(f"Error loading image {img_path}: {e}")
            return None
