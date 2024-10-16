import sys, os

from torch.utils.data import Dataset, DataLoader

from imgaug import augmenters as iaa
import imgaug.augmenters as iaa  # noqa

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch 
import webdataset as wds
import open3d as o3d

import config2 as config

def preload_pointclouds(models_root):
    pointclouds = {}
    
    for category in range(1, 11):  # Subfolders 1 to 10
        category_path = os.path.join(models_root, str(category))
        for obj_file in os.listdir(category_path):
            if obj_file.endswith(".ply"):
                obj_name = os.path.splitext(obj_file)[0]
                pointcloud_file = os.path.join(category_path, obj_file)
                
                # Load the point cloud
                pc_gt = o3d.io.read_point_cloud(pointcloud_file)
                gt_points = np.asarray(pc_gt.points)
                
                # Store point cloud in the dictionary using category and name
                pointclouds[(str(category), obj_name)] = gt_points
                
    return pointclouds

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

def create_webdataset(dataset_paths, size=128, shuffle_buffer=1000, augment=False, center_crop=False, class_name=None):

    dataset = wds.WebDataset(dataset_paths, shardshuffle=True) \
        .decode("pil") \
        .shuffle(shuffle_buffer, initial=size) \
        .to_tuple("rgb.png", "visib_mask.png", "nocs.png", "info.json") \
        .map_tuple( 
            lambda rgb: preprocess(rgb, size, Image.BICUBIC, augment=augment, center_crop=center_crop), 
            lambda mask: preprocess(mask, size, Image.NEAREST, center_crop=center_crop), 
            lambda nocs: preprocess(nocs, size, Image.NEAREST, center_crop=center_crop), 
            lambda info: info) \
        .select(lambda sample: (class_name is None) or (sample[3].get('category_id') == class_name))  # Ensure all four elements are included

    return dataset

def preprocess(image, size, interpolation, augment=False, center_crop=False):

    img_array = np.array(image).astype(np.uint8)
    h, w = img_array.shape[0], img_array.shape[1]

    if center_crop:
        # Undo the enlargement by dividing the current image size by 1.5
        original_size = int(min(h, w) / 1.5)

        # Calculate the crop dimensions to get the center crop of original size
        crop_xmin = (w - original_size) // 2
        crop_xmax = crop_xmin + original_size
        crop_ymin = (h - original_size) // 2
        crop_ymax = crop_ymin + original_size

        # Add 5 pixels of padding around the crop
        crop_xmin = max(crop_xmin - 5, 0)  # Ensure xmin doesn't go below 0
        crop_xmax = min(crop_xmax + 5, w)  # Ensure xmax doesn't exceed image width
        crop_ymin = max(crop_ymin - 5, 0)  # Ensure ymin doesn't go below 0
        crop_ymax = min(crop_ymax + 5, h)  # Ensure ymax doesn't exceed image height

        # Crop the image to the original size with padding
        img_array = img_array[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    else:
        # Default center crop
        crop = min(h, w)
        img_array = img_array[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]

    if augment:
        prob = 0.8
        # seq_syn = iaa.Sequential([
        #                             iaa.Sometimes(0.3 * prob, iaa.CoarseDropout( p=0.2, size_percent=0.05) ),
        #                             iaa.Sometimes(0.5 * prob, iaa.GaussianBlur(1.2*np.random.rand())),
        #                             iaa.Sometimes(0.5 * prob, iaa.Add((-25, 25), per_channel=0.3)),
        #                             iaa.Sometimes(0.3 * prob, iaa.Invert(0.2, per_channel=True)),
        #                             iaa.Sometimes(0.5 * prob, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
        #                             iaa.Sometimes(0.5 * prob, iaa.Multiply((0.6, 1.4))),
        #                             iaa.Sometimes(0.5 * prob, iaa.LinearContrast((0.5, 2.2), per_channel=0.3))
        #                             ], random_order = True)

        seq_syn = iaa.Sequential([
                                    iaa.Sometimes(0.3 * prob, iaa.CoarseDropout( p=0.2, size_percent=0.05) ),
                                    iaa.Sometimes(0.5 * prob, iaa.GaussianBlur((0., 3.))),
                                    iaa.Sometimes(0.3 * prob, iaa.pillike.EnhanceSharpness(factor=(0., 50.))),
                                    iaa.Sometimes(0.3 * prob, iaa.pillike.EnhanceContrast(factor=(0.2, 50.))),
                                    iaa.Sometimes(0.5 * prob, iaa.pillike.EnhanceBrightness(factor=(0.1, 6.))),
                                    iaa.Sometimes(0.3 * prob, iaa.pillike.EnhanceColor(factor=(0., 20.))),
                                    iaa.Sometimes(0.5 * prob, iaa.Add((-25, 25), per_channel=0.3)),
                                    iaa.Sometimes(0.3 * prob, iaa.Invert(0.2, per_channel=True)),
                                    iaa.Sometimes(0.5 * prob, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
                                    iaa.Sometimes(0.5 * prob, iaa.Multiply((0.6, 1.4))),
                                    iaa.Sometimes(0.1 * prob, iaa.AdditiveGaussianNoise(scale=10, per_channel=True)),
                                    iaa.Sometimes(0.5 * prob, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),
                                    iaa.Sometimes(0.5 * prob, iaa.LinearContrast((0.5, 2.2), per_channel=0.3))
                                    ], random_order = True)

        # seq_syn = iaa.Sequential([        
        #                             iaa.Sometimes(0.3 * prob, iaa.CoarseDropout( p=0.2, size_percent=0.05) ),
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

def plot_progress_imgs(imgfn, rgb_images, nocs_images_normalized_gt, nocs_estimated, mask_images, mask_images_binary, mask_images_gt, rot_estimated_R):


    # Print value ranges for each input
    # print(f"RGB Images - Min: {torch.min(rgb_images):.4f}, Max: {torch.max(rgb_images):.4f}")
    # print(f"NOCS GT - Min: {torch.min(nocs_images_normalized_gt):.4f}, Max: {torch.max(nocs_images_normalized_gt):.4f}")
    # print(f"NOCS Estimated - Min: {torch.min(nocs_estimated):.4f}, Max: {torch.max(nocs_estimated):.4f}")
    # print(f"Mask Images - Min: {torch.min(mask_images):.4f}, Max: {torch.max(mask_images):.4f}")
    # print(f"Binary Mask Images - Min: {torch.min(mask_images_binary):.4f}, Max: {torch.max(mask_images_binary):.4f}")
    # print(f"Ground Truth Mask Images - Min: {torch.min(mask_images_gt):.4f}, Max: {torch.max(mask_images_gt):.4f}")

    # print()
    _,ax = plt.subplots(config.num_imgs_log,6,figsize=(10,20))
    # rgb_images, nocs_images_normalized_gt, nocs_estimated, mask_images, masks_estimated, binary_masks
    # Define column titles
    col_titles = ['RGB Image', 'NOCS GT', 'NOCS Estimated', 'Mask GT', 'Mask Estimated', 'Mask Binary Estimated']
    
    # Add column titles
    for i, title in enumerate(col_titles):
        ax[0, i].set_title(title, fontsize=12)

    for i in range(config.num_imgs_log):
        ax[i, 0].imshow(((rgb_images[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 1].imshow(((nocs_images_normalized_gt[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 2].imshow(((nocs_estimated[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 3].imshow(((mask_images[i])).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 4].imshow(((mask_images_binary[i])).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 5].imshow(((mask_images_gt[i])).detach().cpu().numpy().transpose(1, 2, 0))

        # # Plot rotation arrows
        # rotation_matrix = rot_estimated_R[i].detach().cpu().numpy()  # Get the rotation matrix for the current image

        # # Define arrow directions for x, y, z axes
        # arrow_directions = {
        #     'x': np.array([1, 0, 0]),  # X-axis direction
        #     'y': np.array([0, 1, 0]),  # Y-axis direction
        #     'z': np.array([0, 0, 1])   # Z-axis direction
        # }

        # # Get the start point (e.g., center of the image in normalized coordinates)
        # start_point = np.array([0.5, 0.5])  # Center of the image

        # # Iterate over each arrow direction and plot
        # for key, direction in arrow_directions.items():
        #     # Transform the arrow direction using the rotation matrix
        #     transformed_arrow = rotation_matrix @ direction
            
        #     # Calculate end point based on the transformed arrow
        #     end_point = start_point + (transformed_arrow[:2] * config.arrow_length)  # Only use x and y for 2D
            
        #     # Plot the arrow
        #     ax[i, 0].quiver(
        #         start_point[0] * rgb_images[i].shape[2], start_point[1] * rgb_images[i].shape[1],
        #         (end_point[0] - start_point[0]) * rgb_images[i].shape[2], 
        #         (end_point[1] - start_point[1]) * rgb_images[i].shape[1],
        #         angles='xy', scale_units='xy', scale=1, color=config.arrow_colors[key], width=config.arrow_width
        #     )

    plt.tight_layout()
    plt.savefig(imgfn, dpi=300)
    plt.close()

def plot_single_image(output_dir, iteration, nocs_estimated):

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare filename with padded iteration (e.g., 00000001.png)
    imgfn = os.path.join(output_dir, f'{iteration:08d}.png')

    # Plot NOCS estimated map (scale to 0-1 for visualization)
    plt.imshow(((nocs_estimated[0] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
    plt.axis('off')  # Turn off axis for cleaner output

    # Save the figure
    plt.savefig(imgfn, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()