import sys, os

from torch.utils.data import Dataset, DataLoader

from imgaug import augmenters as iaa
import imgaug.augmenters as iaa  # noqa

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch 
import webdataset as wds

import config

def make_log_dirs(weight_dir, val_img_dir):
    weight_dir = weight_dir
    if not(os.path.exists(weight_dir)):
            os.makedirs(weight_dir)

    val_img_dir = val_img_dir
    if not(os.path.exists(val_img_dir)):
        os.makedirs(val_img_dir)

def custom_collate_fn(batch):
    rgb_batch = torch.stack([torch.tensor(item[0]) for item in batch])
    mask_batch = torch.stack([torch.tensor(item[1]) for item in batch])
    nocs_batch = torch.stack([torch.tensor(item[2]) for item in batch])
    info_batch = [item[3] for item in batch]
    # info_batch = {i: item[3] for i, item in enumerate(batch)}

    return {
        'rgb': rgb_batch,
        'mask': mask_batch,
        'nocs': nocs_batch,
        'info': info_batch,
    }

class WebDatasetWrapper(Dataset):
    def __init__(self, dataset):
        # Initialize the WebDataset
        self.dataset = dataset
        
        # Prepare a list to hold the data
        self.data = []
        
        # Load all data into memory (or a subset if the dataset is large)
        for sample in self.dataset:
            self.data.append(sample)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Retrieve an item by index
        return self.data[idx]

def create_webdataset(dataset_paths, size, shuffle_buffer, augment=False):
    dataset = wds.WebDataset(dataset_paths, shardshuffle=True) \
        .decode("pil") \
        .shuffle(shuffle_buffer, initial=size) \
        .to_tuple("rgb.png", "mask.png", "nocs.png", "info.json") \
        .map_tuple( lambda rgb: preprocess(rgb, size, Image.BICUBIC, augment=augment), \
                    lambda mask: preprocess(mask, size, Image.NEAREST), \
                    lambda nocs: preprocess(nocs, size, Image.NEAREST),
                    lambda info: info)
    
    return dataset

def preprocess(image, size, interpolation, augment=False):
    img_array = np.array(image).astype(np.uint8)
    crop = min(img_array.shape[0], img_array.shape[1])
    h, w = img_array.shape[0], img_array.shape[1]
    img_array = img_array[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]
    
    if augment:
        prob = 0.8
        seq_syn = iaa.Sequential([
                                    iaa.Sometimes(0.5 * prob, iaa.GaussianBlur(1.2*np.random.rand())),
                                    iaa.Sometimes(0.5 * prob, iaa.Add((-25, 25), per_channel=0.3)),
                                    iaa.Sometimes(0.3 * prob, iaa.Invert(0.2, per_channel=True)),
                                    iaa.Sometimes(0.5 * prob, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
                                    iaa.Sometimes(0.5 * prob, iaa.Multiply((0.6, 1.4))),
                                    iaa.Sometimes(0.5 * prob, iaa.LinearContrast((0.5, 2.2), per_channel=0.3))
                                    ], random_order = False)
        # seq_syn = iaa.Sequential([        
        #                             iaa.Sometimes(0.5 * prob, iaa.CoarseDropout( p=0.2, size_percent=0.05) ),
        #                             ], random_order = False)
        img_array = seq_syn.augment_image(img_array)

    image = Image.fromarray(img_array)
    image = image.resize((size, size), resample=interpolation)
    img_array = np.array(image).astype(np.uint8)
    return img_array

def normalize_quaternion(q):
    # Normalize the quaternion across the batch
    norm = torch.norm(q, dim=1, keepdim=True)
    return q / norm  # Normalize quaternion

def setup_environment(gpu_id):
    if len(sys.argv) != 2:
        print("Usage: python3 train.py <gpu_id>")
        sys.exit()

    if gpu_id == '-1':
        gpu_id = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

def plot_progress_imgs(imgfn, rgb_images, nocs_images_normalized_gt, nocs_estimated, mask_images, masks_estimated, binary_masks, rot_estimated_R):
    
    _,ax = plt.subplots(10,6,figsize=(10,20))
    
    # Define column titles
    col_titles = ['RGB Image', 'NOCS GT', 'NOCS Estimated', 'Mask GT', 'Mask Estimated', 'Mask Estimated Binary']
    
    # Add column titles
    for i, title in enumerate(col_titles):
        ax[0, i].set_title(title, fontsize=12)

    for i in range(10):
        ax[i, 0].imshow(((rgb_images[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 1].imshow(((nocs_images_normalized_gt[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 2].imshow(((nocs_estimated[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 3].imshow(mask_images[i])
        ax[i, 4].imshow((((masks_estimated[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        ax[i, 5].imshow((((binary_masks[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))

        # Plot rotation arrows
        rotation_matrix = rot_estimated_R[i].detach().cpu().numpy()  # Get the rotation matrix for the current image

        # Define arrow directions for x, y, z axes
        arrow_directions = {
            'x': np.array([1, 0, 0]),  # X-axis direction
            'y': np.array([0, 1, 0]),  # Y-axis direction
            'z': np.array([0, 0, 1])   # Z-axis direction
        }

        # Get the start point (e.g., center of the image in normalized coordinates)
        start_point = np.array([0.5, 0.5])  # Center of the image

        # Iterate over each arrow direction and plot
        for key, direction in arrow_directions.items():
            # Transform the arrow direction using the rotation matrix
            transformed_arrow = rotation_matrix @ direction
            
            # Calculate end point based on the transformed arrow
            end_point = start_point + (transformed_arrow[:2] * config.arrow_length)  # Only use x and y for 2D
            
            # Plot the arrow
            ax[i, 0].quiver(
                start_point[0] * rgb_images[i].shape[2], start_point[1] * rgb_images[i].shape[1],
                (end_point[0] - start_point[0]) * rgb_images[i].shape[2], 
                (end_point[1] - start_point[1]) * rgb_images[i].shape[1],
                angles='xy', scale_units='xy', scale=1, color=config.arrow_colors[key], width=config.arrow_width
            )

    plt.tight_layout()
    plt.savefig(imgfn, dpi=300)
    plt.close()