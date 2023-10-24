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
                    augmented_images_byte.append(np.frombuffer(img_byte_array, dtype='uint8'))
                    augmented_labels_byte.append(label)

            # extend the dataset   
            dset_images.resize(global_idx + len(augmented_images_byte), axis=0)
            dset_labels.resize(global_idx + len(augmented_labels_byte), axis=0)

            # batch write the data to HDF5 file
            dset_images[global_idx:global_idx + len(augmented_images_byte)] = augmented_images_byte
            dset_labels[global_idx:global_idx + len(augmented_labels_byte)] = augmented_labels_byte

            global_idx += len(augmented_images_byte)


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
