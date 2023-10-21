import h5py
import os
import io
import numpy as np
from tqdm import tqdm
from PIL import Image

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
                total_images += len([f for f in os.listdir(full_path) if f.endswith(('.png', '.jpeg'))])
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
                    images = [img for img in os.listdir(os.path.join(root_dir, entry)) if img.endswith('.png')]
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
                        except Exception as e:
                            # print(f"Error reading image {img_path}: {e}")
                            continue



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