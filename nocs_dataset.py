import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

from transformers import AutoImageProcessor, AutoModel
from imgaug import augmenters as iaa
import imgaug.augmenters as iaa  # noqa
from tqdm import tqdm
import sys
import random
import imageio 


class NOCSBase(Dataset):
    def __init__(self, data_root, size, obj_id, fraction=1.0, crop_object=False, augment=False, interpolation="bicubic"):
        self.rgb_paths, self.nocs_paths = [], []

        self.size = size
        self.fraction = fraction
        self.crop_object  = crop_object
        self.augment = augment    

        data_root = data_root + "/" + str(obj_id)
        
        rgb_root = os.path.join(data_root, "rgb")
        nocs_root = os.path.join(data_root, "nocs")

        rgb_files = [file for file in sorted(os.listdir(rgb_root)) if file.endswith(".png")]
        nocs_files = [file for file in sorted(os.listdir(nocs_root)) if file.endswith(".png")]

        # Determine how many samples to load based on fraction
        num_samples = int(len(rgb_files) * self.fraction)
        selected_indices = np.random.choice(len(rgb_files), num_samples, replace=False)

        selected_rgb_files = [rgb_files[i] for i in selected_indices]
        selected_nocs_files = [nocs_files[i] for i in selected_indices]
        
        self.rgb_paths.extend([os.path.join(rgb_root, l) for l in selected_rgb_files])
        self.nocs_paths.extend([os.path.join(nocs_root, l) for l in selected_nocs_files])

        self.interpolation = {"bilinear": Image.BILINEAR,
                              "bicubic": Image.BICUBIC,
                              "lanczos": Image.LANCZOS,
                              }[interpolation]

        self.color_augmentation = iaa.Sequential([
            iaa.Sometimes(0.5, iaa.CoarseDropout( p=0.2, size_percent=0.1) )
        ], random_order=True)

        self._length = len(self.rgb_paths)

        self.labels = {
            "rgb_file_path_": [l for l in self.rgb_paths],
            "nocs_file_path_": [l for l in self.nocs_paths],
        }

    def __len__(self):
        return self._length

    def process_image(self, file_path, size, image_type):
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
        image = image.resize((size, size), resample=self.interpolation)
        img_array = np.array(image).astype(np.uint8)
        return img_array


    def apply_cutouts(self, image, prob=0.5, size=20, min_cutouts=1, max_cutouts=10):
        """
        Apply multiple Cutout augmentations to an image with random shapes.

        Parameters:
        - image (np.ndarray): The input image (depth map or RGB).
        - prob (float): The probability of applying the augmentation.
        - size (int): The size parameter for the shapes.
        - min_cutouts (int): Minimum number of cutouts.
        - max_cutouts (int): Maximum number of cutouts.

        Returns:
        - np.ndarray: The augmented image.
        """
        if np.random.rand() > prob:
            return image

        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        num_cutouts = random.randint(min_cutouts, max_cutouts)
        
        mask = np.ones((height, width), dtype=np.float32)

        for _ in range(num_cutouts):
            # Randomly choose a shape
            shapes = ['rectangle', 'circle', 'triangle', 'ellipse']
            shape = random.choice(shapes)

            # Random position
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)

            if shape == 'rectangle':
                x1 = max(0, x - size // 2)
                y1 = max(0, y - size // 2)
                x2 = min(width, x + size // 2)
                y2 = min(height, y + size // 2)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 0, thickness=-1)
            
            elif shape == 'circle':
                radius = size // 2
                cv2.circle(mask, (x, y), radius, 0, thickness=-1)
            
            elif shape == 'triangle':
                points = np.array([
                    [x, y - size // 2],
                    [x - size // 2, y + size // 2],
                    [x + size // 2, y + size // 2]
                ], dtype=np.int32)
                cv2.fillPoly(mask, [points], 0)
            
            elif shape == 'ellipse':
                axes = (size // 2, size // 3)
                cv2.ellipse(mask, (x, y), axes, angle=0, startAngle=0, endAngle=360, color=0, thickness=-1)

        # Expand mask dimensions if image has multiple channels
        if channels > 1:
            mask = np.expand_dims(mask, axis=-1)
            mask = np.repeat(mask, channels, axis=-1)

        # Apply the mask to the image
        image_augmented = image * mask
        return image_augmented

    def __getitem__(self, i):
        nocs_img_array = (self.process_image(self.labels["nocs_file_path_"][i], size=self.size, image_type="RGB") / 127.5 - 1.0).astype(np.float32)
        rgb_img_raw = self.process_image(self.labels["rgb_file_path_"][i], size=self.size, image_type="RGB")

        if self.augment == True:
            #rgb_img_raw = self.color_augmentation(image=rgb_img_raw)
            rgb_img_raw = self.apply_cutouts(rgb_img_raw, prob=0.5, size=int(self.size / 6), min_cutouts=5, max_cutouts=12)

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
