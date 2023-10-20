import os
from torch.utils.data import Dataset
from PIL import Image
import random
from abc import ABC, abstractmethod
import torchvision.transforms as transforms
from typing import Tuple, Optional
import h5py
import io

VFR_FONTS_NUM = 2383
SYN_DATA_COUNT_PER_FONT = 1000

# with h5py.File(hdf5_file_path, 'w') as f:
#     for idx in range(len(your_dataset)):
#         image, label = your_dataset[idx]
#         image_np = np.array(image)
#         # Create a dataset for each image
#         dset = f.create_dataset(str(idx), data=image_np)
#         # Store the label as an attribute of the dataset
#         dset.attrs['label'] = label
ROOT_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'result')


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


class VFRAlignHDF5Dataset(Dataset):
    def __init__(
        self,
        hdf5_file_path: str,
        transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        # load hdf5 file in __init__ will lead error when num_workers > 1
        # self.hf = h5py.File(hdf5_file_path, 'r')
        # self._images = self.hf['images']
        # self._labels = self.hf['labels']
        self._hdf5_file_path = hdf5_file_path
        self.transform = transform
        with h5py.File(self._hdf5_file_path, 'r') as f:
            self._length = len(f['images'])
    
    def _set_transform(self, transform: transforms.Compose):
        self.transform = transform
    
    def _open_hdf5(self):
        self.hf = h5py.File(self._hdf5_file_path, 'r')
        try:
            self._images = self.hf['images']
        except Exception:
            raise Exception(f"Error loading images from {self._hdf5_file_path}")
        try:
            self._labels = self.hf['labels']
        except Exception:
            raise Exception(f"Error loading labels from {self._hdf5_file_path}")
        assert len(self._images) == len(
            self._labels
        ), 'images and labels have different length.'
        if self._num_classes is not None:
            pass
        else:
            # check for full dataset.
            assert (
                len(self._labels) == self._length
            ), 'hd5f file did not contain all the data of VFR_syn dataset.'

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        if not hasattr(self, 'hf'):
            self._open_hdf5()
        assert hasattr(self, '_images'), 'images not found in hdf5 file.'
        image_byte_array = self._images[idx]
        image = Image.open(io.BytesIO(image_byte_array))

        if self.transform:
            image = self.transform(image)
        return image


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
            
    def _set_transform(self, transform: transforms.Compose):
        self.transform = transform

    def __len__(self) -> int:
        # 1000 images per font family
        return len(self.font_families) * SYN_DATA_COUNT_PER_FONT

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        label = idx // SYN_DATA_COUNT_PER_FONT
        font_family = self.font_families[idx // SYN_DATA_COUNT_PER_FONT]
        image_name = f"{font_family}_{idx % SYN_DATA_COUNT_PER_FONT}.png"
        image_filename = os.path.join(self.root_dir, font_family, image_name)
        image = self._load_image(image_filename)
        # DEBUG ONLY,REMOVE LATER.
        self._check(image, idx, ROOT_CACHE_DIR)
        return image, label

    def _check(self, image, idx, cache_root) -> None:
        if idx < 1000:
            label = idx // SYN_DATA_COUNT_PER_FONT
            font = self._label2font(label)
            cache_path = os.path.join(cache_root, font)
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
            image_path = os.path.join(
                cache_path, f"{font}_{idx % SYN_DATA_COUNT_PER_FONT}_trans.png"
            )
            image_pil = transforms.ToPILImage()(image)
            image_pil.save(image_path)
        else:
            return None

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


class VFRSynHDF5Dataset(Dataset):
    def __init__(
        self,
        hdf5_file_path: str,
        font_list_path: str,
        num_classes: int,
        transform: Optional[transforms.Compose] = None,
        
    ):
        super().__init__()
        # load hdf5 file in __init__ will lead error when num_workers > 1
        self.hf = h5py.File(hdf5_file_path, 'r')
        self._images = self.hf['images']
        self._labels = self.hf['labels']
        self._hdf5_file_path = hdf5_file_path
        self.transform = transform
        with open(font_list_path, 'r') as f:
            self._font_families = f.read().splitlines()
        # with h5py.File(self._hdf5_file_path, 'r') as f:
        #     self._length = len(f['labels'])
        self._num_classes = num_classes
        self._length = num_classes * SYN_DATA_COUNT_PER_FONT

    def _open_hdf5(self):
        self.hf = h5py.File(self._hdf5_file_path, 'r')
        try:
            self._images = self.hf['images']
        except Exception:
            raise Exception(f"Error loading images from {self._hdf5_file_path}")
        try:
            self._labels = self.hf['labels']
        except Exception:
            raise Exception(f"Error loading labels from {self._hdf5_file_path}")
        assert len(self._images) == len(
            self._labels
        ), 'images and labels have different length.'
        if self._num_classes is not None:
            pass
        else:
            # check for full dataset.
            assert (
                len(self._labels) == self._length
            ), 'hd5f file did not contain all the data of VFR_syn dataset.'

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        if not hasattr(self, 'hf'):
            self._open_hdf5()
        assert hasattr(self, '_images') and hasattr(
            self, '_labels'
        ), 'images or labels not found in hdf5 file.'
        if self._num_classes != VFR_FONTS_NUM:
            # self._images and self._labels will change after calling _load_partial_data().
            self._load_partial_data()
            # now idx is the index of the image in the partial dataset
            # idx = idx % (self._num_classes * SYN_DATA_COUNT_PER_FONT)
        image_byte_array = self._images[idx]
        image = Image.open(io.BytesIO(image_byte_array))

        if self.transform:
            image = self.transform(image)
        # convert font-family name from byte to str
        font = self._labels[idx].decode("utf-8")
        # print(font)

        label = self._font2label(font)

        return image, label

    def _label2font(self, label: int) -> str:
        return self._font_families[label]

    def _font2label(self, font: str) -> int:
        return self._font_families.index(font)

    def _load_partial_data(self):
        """
        Load a subset of the data for quick testing.

        Parameters:
        - num_classes: Number of classes to load. Default is 20.

        Returns:
        - partial_images: A list of loaded images.
        - partial_labels: A list of corresponding labels.
        """
        partial_images = []
        partial_labels = []
        # partial_labels_ = []
        assert hasattr(self, '_num_classes'), 'num_classes not given.'
        # Assuming images are stored in an array format in HDF5
        for i in range(self._num_classes):
            for j in range(SYN_DATA_COUNT_PER_FONT):  # Assuming 1000 images per class
                idx = i * SYN_DATA_COUNT_PER_FONT + j
                image = self._images[idx]
                label = self._labels[idx]
                partial_images.append(image)
                partial_labels.append(label)
                # partial_labels_.append(label.decode("utf-8"))

        self._images = partial_images
        self._labels = partial_labels
        # replace full font list with partial one.
        # self._font_families = list(set(partial_labels_))
