import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

from imgaug import augmenters as iaa
import imgaug.augmenters as iaa  # noqa
from tqdm import tqdm
import sys
import random
import imageio 
import matplotlib.pyplot as plt

class NOCSBase(Dataset):
    def __init__(self, data_root, size, obj_ids=None, fraction=1.0, crop_object=False, augment=False, interpolation="bicubic"):
        self.rgb_paths, self.nocs_paths, self.mask_paths = [], [], []

        self.size = size
        self.fraction = fraction
        self.crop_object  = crop_object
        self.augment = augment    

        if obj_ids is None:
            obj_ids = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        
        for obj_id in obj_ids:
            obj_data_root = os.path.join(data_root, str(obj_id))
            rgb_root = os.path.join(obj_data_root, "rgb")
            nocs_root = os.path.join(obj_data_root, "nocs")
            mask_root = os.path.join(obj_data_root, "mask")

            if not os.path.exists(rgb_root) or not os.path.exists(nocs_root):
                continue

            rgb_files = [file for file in sorted(os.listdir(rgb_root)) if file.endswith(".png")]
            nocs_files = [file for file in sorted(os.listdir(nocs_root)) if file.endswith(".png")]
            mask_files = [file for file in sorted(os.listdir(mask_root)) if file.endswith(".png")]

            # Ensure RGB and NOCS files match in count
            if len(rgb_files) != len(nocs_files):
                print(f"Warning: Mismatched number of files for {obj_id}")
                continue

            # Determine how many samples to load based on fraction
            num_samples = int(len(rgb_files) * self.fraction)
            selected_indices = np.random.choice(len(rgb_files), num_samples, replace=False)

            selected_rgb_files = [rgb_files[i] for i in selected_indices]
            selected_nocs_files = [nocs_files[i] for i in selected_indices]
            selected_mask_files = [mask_files[i] for i in selected_indices]

            self.rgb_paths.extend([os.path.join(rgb_root, l) for l in selected_rgb_files])
            self.nocs_paths.extend([os.path.join(nocs_root, l) for l in selected_nocs_files])
            self.mask_paths.extend([os.path.join(mask_root, l) for l in selected_mask_files])

        self.color_augmentation = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.CoarseDropout(p=0.2, size_percent=0.1))
        ], random_order=True)

        self._length = len(self.rgb_paths)

        self.labels = {
            "rgb_file_path_": [l for l in self.rgb_paths],
            "nocs_file_path_": [l for l in self.nocs_paths],
            "mask_file_path_": [l for l in self.mask_paths],
        }

    def __len__(self):
        return self._length

    def process_image(self, file_path, size, image_type, interpolation):
        if image_type == "RGB":
            image = Image.open(file_path).convert("RGB")
        elif image_type == "mask":
            image = Image.open(file_path).convert("1")  # Convert to binary mode
        else:
            raise ValueError("Invalid image type. Supported types are 'RGB' and 'mask'.")

        img_array = np.array(image).astype(np.uint8)
        crop = min(img_array.shape[0], img_array.shape[1])
        h, w = img_array.shape[0], img_array.shape[1]
        img_array = img_array[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
        image = Image.fromarray(img_array)
        image = image.resize((size, size), resample=interpolation)
        img_array = np.array(image).astype(np.uint8)
        return img_array

    def __getitem__(self, i):
        nocs_img_array = (self.process_image(self.labels["nocs_file_path_"][i], size=self.size, image_type="RGB", interpolation=Image.NEAREST) / 127.5 - 1.0).astype(np.float32)
        rgb_img_raw = self.process_image(self.labels["rgb_file_path_"][i], size=self.size, image_type="RGB", interpolation=Image.BICUBIC)
        mask_img_raw = self.process_image(self.labels["mask_file_path_"][i], size=self.size, image_type="mask", interpolation=Image.NEAREST)

        rgb_img_raw = rgb_img_raw * np.expand_dims(mask_img_raw, axis=-1)

        rgb_img_array = (rgb_img_raw / 127.5 - 1.0).astype(np.float32)
        rgb_img_array = np.transpose(rgb_img_array, (2, 0, 1))

        nocs_img_array = np.transpose(nocs_img_array, (2, 0, 1))

        example = {
            "rgb": rgb_img_array,
            "nocs": nocs_img_array,
        }
        return example

class NOCSTrain(NOCSBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
