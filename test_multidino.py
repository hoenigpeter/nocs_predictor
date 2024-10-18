import os
import sys

import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import pytorch3d
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from multidino_model import MultiDINO as multidino
from transformers import CLIPProcessor, CLIPModel
import open3d as o3d
from torch.optim.lr_scheduler import CosineAnnealingLR

import json
import webdataset as wds

from utils import WebDatasetWrapper, preprocess, normalize_quaternion, setup_environment, \
                    create_webdataset, custom_collate_fn, make_log_dirs, plot_progress_imgs, \
                    preload_pointclouds, plot_single_image

import argparse
import importlib.util

# Function to load config from the passed file
def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('config', type=str, help="Path to the config file")
    return parser.parse_args()

def main(config):
    setup_environment(str(config.gpu_id))
    make_log_dirs(config.weight_dir, config.val_img_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model instantiation and compilation
    generator = multidino(input_resolution=config.size, num_bins=config.num_bins, freeze_backbone=config.freeze_backbone)
    generator.to(device)

    # Load pre-trained model for inference
    model_path = os.path.join('./weights_bottle', 'generator_epoch_50.pth')
    generator.load_state_dict(torch.load(model_path, map_location=device))

    # Set model to evaluation mode for inference
    generator.eval()

    # Instantiate train and val dataset + dataloaders
    test_dataset = create_webdataset(config.test_data_root, config.size, config.shuffle_buffer, augment=False, center_crop=False, class_name=config.class_name)

    test_dataloader = wds.WebLoader(
            test_dataset, batch_size=config.test_batch_size, shuffle=False, num_workers=config.val_num_workers, drop_last=True, collate_fn=custom_collate_fn,
    )

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            print("Step: ", step)
            # unwrap the batch
            rgb_images = batch['rgb']
            mask_images = batch['mask']
            nocs_images = batch['nocs']
            infos = batch['info']

            # RGB processing
            rgb_images = torch.clamp(rgb_images.float(), min=0.0, max=255.0)
            rgb_images = (rgb_images.float() / 127.5) - 1
            rgb_images = rgb_images.permute(0, 3, 1, 2)
            rgb_images_gt = rgb_images.to(device)

            # # Normalize mask to be binary (0 or 1)
            # binary_mask = (mask_images > 0).float()  # Converts mask to 0 or 1
            # binary_mask = binary_mask.permute(0, 3, 1, 2).to(device)  # Make sure mask has same shape
            # rgb_images_gt = rgb_images_gt * binary_mask
            
            # MASK processing
            mask_images_gt = mask_images.float() / 255.0
            mask_images_gt = mask_images_gt.permute(0, 3, 1, 2)
            mask_images_gt = mask_images_gt.to(device)

            # NOCS processing
            nocs_images_normalized = (nocs_images.float() / 127.5) - 1
            nocs_images_normalized = nocs_images_normalized.permute(0, 3, 1, 2)
            nocs_images_normalized_gt = nocs_images_normalized.to(device)

            # ROTATION processing
            rotations = [torch.tensor(entry["rotation"], dtype=torch.float32) for entry in infos]
            rotation_gt = torch.stack(rotations).to(device)

            # forward pass through generator
            x_logits, y_logits, z_logits, masks_estimated = generator(rgb_images_gt)

            binary_masks = (masks_estimated > 0.5).float()  # Convert to float for multiplication

            scaled_nocs = (nocs_images_normalized_gt + 1) * (config.num_bins - 1) / 2.0
            target_bins = scaled_nocs.long().clamp(0, config.num_bins - 1)  # Ensure values are within bin range

            x_bins = torch.softmax(x_logits, dim=1)  # Softmax over the bin dimension
            y_bins = torch.softmax(y_logits, dim=1)
            z_bins = torch.softmax(z_logits, dim=1)

            # Bin centers (shared for x, y, z dimensions)
            bin_centers = torch.linspace(-1, 1, config.num_bins).to(x_logits.device)  # Bin centers

            # Compute the estimated NOCS map for each dimension by multiplying with bin centers and summing over bins
            nocs_x_estimated = torch.sum(x_bins * bin_centers.view(1, config.num_bins, 1, 1), dim=1)
            nocs_y_estimated = torch.sum(y_bins * bin_centers.view(1, config.num_bins, 1, 1), dim=1)
            nocs_z_estimated = torch.sum(z_bins * bin_centers.view(1, config.num_bins, 1, 1), dim=1)

            # Combine the estimated NOCS map from x, y, and z dimensions
            nocs_estimated = torch.stack([nocs_x_estimated, nocs_y_estimated, nocs_z_estimated], dim=1)

            plot_single_image(config.test_img_dir, step, nocs_estimated)

if __name__ == "__main__":
    args = parse_args()
    
    # Load the config file passed as argument
    config = load_config(args.config)
    
    # Call main with the loaded config
    main(config)