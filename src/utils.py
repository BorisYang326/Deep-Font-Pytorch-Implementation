import h5py
import os
import io
import numpy as np
from tqdm import tqdm
from PIL import Image
import itertools
from torchvision.transforms import transforms
from typing import List
import torch.nn as nn
from .preprocess import Squeezing
from .config import INPUT_SIZE, SQUEEZE_RATIO, COMB_PICK_NUM
import random


def label2font(font_book_path: str, label: int) -> str:
    with open(font_book_path, 'r') as f:
        font_families = f.read().splitlines()
    return font_families[label]


def syn_images_to_hdf5(root_dir, hdf5_file_path):
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


def align_images_to_hdf5(dir_list, hdf5_file_path):
    total_images = 0
    for path in tqdm(dir_list):
        for d in os.listdir(path):
            full_path = os.path.join(path, d)
            if os.path.isdir(full_path):
                total_images += len(
                    [f for f in os.listdir(full_path) if f.endswith(('.png', '.jpeg'))]
                )
            else:
                if d.endswith(('.png', '.jpeg')):
                    total_images += 1

    print('Total images: ', total_images)

    with h5py.File(hdf5_file_path, 'w') as hf:
        img_dtype = h5py.vlen_dtype(np.dtype('uint8'))
        img_dset = hf.create_dataset("images", shape=(total_images,), dtype=img_dtype)
        idx = 0
        for root_dir in dir_list:
            entries = os.listdir(root_dir)
            for entry in tqdm(entries):
                if os.path.isdir(os.path.join(root_dir, entry)):
                    images = [
                        img
                        for img in os.listdir(os.path.join(root_dir, entry))
                        if img.endswith('.png')
                    ]
                    for img_name in images:
                        img_path = os.path.join(root_dir, entry, img_name)
                        with Image.open(img_path) as img:
                            with io.BytesIO() as buffer:
                                img.save(buffer, format="PNG")
                                img_byte_array = buffer.getvalue()

                        img_dset[idx] = np.frombuffer(img_byte_array, dtype='uint8')
                        idx += 1
                else:
                    if entry.endswith(('.png', '.jpeg')):
                        img_path = os.path.join(root_dir, entry)
                        try:
                            with Image.open(img_path) as img:
                                with io.BytesIO() as buffer:
                                    img.save(buffer, format="PNG")
                                    img_byte_array = buffer.getvalue()

                            img_dset[idx] = np.frombuffer(img_byte_array, dtype='uint8')
                            idx += 1
                        except Exception:
                            # print(f"Error reading image {img_path}: {e}")
                            continue


def augment_hdf5_preprocess(
    orign_hdf5_path: str,
    transform_list: List[nn.Module],
    aug_hdf5_path: str,
    batch_size: int = 1024,
) -> None:
    fixed_prefix = [transforms.Grayscale()]
    fixed_suffix = [
        Squeezing(INPUT_SIZE, SQUEEZE_RATIO),
        transforms.RandomCrop(INPUT_SIZE),
        # transforms.ToTensor(),
    ]
    img_dtype = h5py.vlen_dtype(np.dtype('uint8'))
    label_dtype = h5py.string_dtype(encoding='utf-8')
    all_combinations = [tuple()]  # This represents the '0' or no additional transform
    for l in range(1, len(transform_list) + 1):
        all_combinations.extend(itertools.combinations(transform_list, l))

    with h5py.File(orign_hdf5_path, 'r') as original_file, h5py.File(
        aug_hdf5_path, 'a'
    ) as augmented_file:
        # Create or get the datasets in HDF5 file
        if 'images' not in augmented_file:
            dset_images = augmented_file.create_dataset(
                'images',
                shape=(0,),
                maxshape=(None,),
                dtype=img_dtype,
            )
            dset_labels = augmented_file.create_dataset(
                'labels', shape=(0,), maxshape=(None,), dtype=label_dtype
            )
        else:
            dset_images = augmented_file['images']
            dset_labels = augmented_file['labels']

        total_images = len(original_file['images'])
        global_idx = 0
        for idx in tqdm(
            range(0, total_images, batch_size), desc="Batch Loop", leave=False
        ):
            augmented_images_byte = []
            augmented_labels_byte = []

            end_idx = min(idx + batch_size, total_images)
            for index in tqdm(range(idx, end_idx), desc="Image Loop", leave=False):
                image_data = original_file['images'][index]
                label = original_file['labels'][index]
                pil_img = Image.open(io.BytesIO(image_data))

                selected_combinations = random.sample(
                    all_combinations, min(COMB_PICK_NUM, len(all_combinations))
                )
                for combination in selected_combinations:
                    transform = transforms.Compose(
                        fixed_prefix + list(combination) + fixed_suffix
                    )
                    augmented_image = transform(pil_img)

                    # Convert the augmented PIL image to byte array
                    with io.BytesIO() as buffer:
                        augmented_image.save(buffer, format="PNG")
                        img_byte_array = buffer.getvalue()
                    augmented_images_byte.append(
                        np.frombuffer(img_byte_array, dtype='uint8')
                    )
                    augmented_labels_byte.append(label)

            # extend the dataset
            dset_images.resize(global_idx + len(augmented_images_byte), axis=0)
            dset_labels.resize(global_idx + len(augmented_labels_byte), axis=0)

            # batch write the data to HDF5 file
            dset_images[
                global_idx : global_idx + len(augmented_images_byte)
            ] = augmented_images_byte
            dset_labels[
                global_idx : global_idx + len(augmented_labels_byte)
            ] = augmented_labels_byte

            global_idx += len(augmented_images_byte)


def split_hdf5(
    input_hdf5_path: str,
    train_hdf5_path: str,
    test_hdf5_path: str,
    split_ratio: float = 0.9,
):
    """Split the input HDF5 file into train and test HDF5 files.

    Args:
        input_hdf5_path (str): origin HDF5 file path
        train_hdf5_path (str): train HDF5 file path
        test_hdf5_path (str): test HDF5 file path
        split_ratio (float, optional): split ratio. Defaults to 0.9.
    """
    with h5py.File(input_hdf5_path, 'r') as f:
        images = f['images'][:]
        labels = f['labels'][:]

    # Shuffle the data
    indices = np.arange(len(images))
    np.random.shuffle(indices)

    # Split indices
    split_idx = int(len(indices) * split_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    # Extract train and test data
    train_images = images[train_indices]
    train_labels = labels[train_indices]

    test_images = images[test_indices]
    test_labels = labels[test_indices]

    # Write train data to new HDF5 file
    with h5py.File(train_hdf5_path, 'w') as f:
        f.create_dataset('images', data=train_images)
        f.create_dataset('labels', data=train_labels)

    # Write test data to new HDF5 file
    with h5py.File(test_hdf5_path, 'w') as f:
        f.create_dataset('images', data=test_images)
        f.create_dataset('labels', data=test_labels)


def split_and_augment_hdf5(
    origin_hdf5_path: str,
    origin_aug_hdf5_path: str,
    new_train_hdf5_path: str,
    new_test_hdf5_path: str,
    split_ratio: float = 0.9,
)->None:
    """Split the origin HDF5 file into train and test HDF5 files,
    and split the augmented train HDF5 file into new train HDF5 files.
    Finally, the train data will be augmented and the test data will be the same as origin HDF5 file.

    Args:
        origin_hdf5_path (str): origin HDF5 file path
        origin_aug_hdf5_path (str): origin augmented HDF5 file path
        new_train_hdf5_path (str): splitted and augmented train HDF5 file path
        new_test_hdf5_path (str): splitted test HDF5 file path
        split_ratio (float, optional): split ratio. Defaults to 0.9.
    """
    # Determine the split index
    with h5py.File(origin_hdf5_path, 'r') as f:
        total_images = len(f['images'])
    split_idx = int(total_images * split_ratio)

    # Extract training data from train_aug_hdf5
    with h5py.File(origin_aug_hdf5_path, 'r') as f:
        #  Each image in current augmented dataset is augmented 3 times
        augmented_train_images = f['images'][
            : split_idx * 3
        ]  # Assuming each image is augmented 3 times
        augmented_train_labels = f['labels'][: split_idx * 3]

    # Shuffle the train data
    train_indices = np.arange(len(augmented_train_images))
    np.random.shuffle(train_indices)
    augmented_train_images = augmented_train_images[train_indices]
    augmented_train_labels = augmented_train_labels[train_indices]

    # Write augmented training data to new_train_hdf5
    with h5py.File(new_train_hdf5_path, 'w') as f:
        f.create_dataset('images', data=augmented_train_images)
        f.create_dataset('labels', data=augmented_train_labels)

    # Extract test data from origin_hdf5
    with h5py.File(origin_hdf5_path, 'r') as f:
        test_images = f['images'][split_idx:]
        test_labels = f['labels'][split_idx:]

    # Shuffle the test data
    test_indices = np.arange(len(test_images))
    np.random.shuffle(test_indices)
    test_images = test_images[test_indices]
    test_labels = test_labels[test_indices]

    # Write test data to new_test_hdf5
    with h5py.File(new_test_hdf5_path, 'w') as f:
        f.create_dataset('images', data=test_images)
        f.create_dataset('labels', data=test_labels)


# root_dir = '/public/dataset/AdobeVFR/Raw Image/VFR_syn_train'
# hdf5_file_path = '/public/dataset/AdobeVFR/hdf5/VFR_syn_train_bk.hdf5'
# images_to_hdf5(root_dir, hdf5_file_path)

# dir_list = [
#     '/public/dataset/AdobeVFR/Raw Image/VFR_real_u/scrape-wtf-new',
#     '/public/dataset/AdobeVFR/Raw Image/VFR_syn_train'
# ]
# hdf5_file_path = '/public/dataset/AdobeVFR/hdf5/VFR_syn_real_align_bk.hdf5'
# align_images_to_hdf5(dir_list, hdf5_file_path)

# root_dir = '/public/dataset/AdobeVFR/Raw Image/VFR_real_test_fromzip'
# hdf5_file_path = '/public/dataset/AdobeVFR/hdf5/VFR_real_test_bk.hdf5'
# syn_images_to_hdf5(root_dir, hdf5_file_path)
