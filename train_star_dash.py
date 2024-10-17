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

from star_dash_model import MultiDINO as multidino

from transformers import CLIPProcessor, CLIPModel
import open3d as o3d

from sklearn.decomposition import PCA

import json
import webdataset as wds

from star_dash_utils import WebDatasetWrapper, preprocess, normalize_quaternion, setup_environment, create_webdataset, custom_collate_fn, make_log_dirs, plot_progress_imgs, preload_pointclouds

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

def add_loss(est_points_batch, gt_points_batch):
    """Calculate the ADD score for a batch of point clouds using GPU."""

    distances = torch.cdist(est_points_batch, gt_points_batch, p=2)
    min_distances, _ = distances.min(dim=2)
    add_scores = min_distances.mean(dim=1)

    return add_scores.mean()

def symmetry_aware_loss(destarized_points, symmetry_order):
    """
    Symmetry-aware loss: Enforces periodicity in the destarized points.
    Symmetrically equivalent points should map to the same values.
    Args:
    - destarized_points: Points after destarization
    - symmetry_order: Predicted symmetry order n
    
    Returns:
    - loss: Symmetry-aware loss
    """
    n = symmetry_order.view(-1, 1, 1, 1)  # Reshape to match spatial dimensions
    
    # Ensure that the destarized points repeat periodically with the symmetry order
    # This ensures that rotating by a symmetry step does not change the points
    periodicity_constraint = torch.abs(destarized_points[:, 1] % (2 * torch.pi / n))
    
    return torch.mean(periodicity_constraint)

def consistency_loss(star_output, dash_output):
    """
    Consistency loss between star and dash representations.
    Args:
    - star_output: Output of StarHead
    - dash_output: Output of DashHead
    
    Returns:
    - loss: Consistency loss
    """
    return torch.mean((star_output - dash_output) ** 2)

def apply_rotation(points, rotation_matrix):
    """
    Apply a rotation matrix to a batch of points.
    
    :param points: Tensor of shape (batch_size, 1000, 3)
    :param rotation_matrix: Tensor of shape (batch_size, 3, 3)
    :return: Transformed points of shape (batch_size, 1000, 3)
    """
    # Ensure the correct shapes for batch matrix multiplication
    # points: (batch_size, 1000, 3) -> (batch_size, 3, 1000)
    points_transposed = points.transpose(1, 2)
    
    # Perform batch matrix multiplication: rotated_points = rotation_matrix @ points_transposed
    rotated_points = torch.bmm(rotation_matrix, points_transposed)  # (batch_size, 3, 1000)
    
    # Transpose back to (batch_size, 1000, 3)
    return rotated_points.transpose(1, 2)

def mask_penalty_loss(nocs_output, ground_truth_mask):
    """
    Penalizes the NOCS output values that are outside the ground truth mask.
    
    Args:
    - nocs_output: Predicted NOCS map (batch_size, 3, H, W), values in range [-1, 1]
    - ground_truth_mask: Ground truth mask (batch_size, 3, H, W), binary mask (0 or 1)
    
    Returns:
    - loss: Mask penalty loss
    """
    # Create a mask to identify where the ground truth mask is 0
    # We use `ground_truth_mask == 0` to find the areas where we want to penalize
    mask = ground_truth_mask == 0  # Shape: (batch_size, 3, H, W)
    nocs_normalized = (nocs_output + 1) / 2

    # Calculate the penalty: only consider NOCS values where mask is True 
    # (i.e., ground truth is 0)
    penalty_values = nocs_normalized[mask]  # Get NOCS values outside the mask
    
    # Loss is the mean squared error for these penalty values
    loss = torch.mean(penalty_values**2)  # Penalize the squared values
    
    return loss

def mask_penalty_stardash(nocs_output, ground_truth_mask):
    """
    Penalizes the NOCS output values that are outside the ground truth mask.
    
    Args:
    - nocs_output: Predicted NOCS map (batch_size, 3, H, W), values in range [-1, 1]
    - ground_truth_mask: Ground truth mask (batch_size, 3, H, W), binary mask (0 or 1)
    
    Returns:
    - loss: Mask penalty loss
    """
    # Create a mask to identify where the ground truth mask is 0
    # We use `ground_truth_mask == 0` to find the areas where we want to penalize
    mask = ground_truth_mask == 0  # Shape: (batch_size, 3, H, W)
    nocs_normalized = (nocs_output - torch.min(nocs_output)) / (torch.max(nocs_output) - torch.min(nocs_output))

    # Calculate the penalty: only consider NOCS values where mask is True 
    # (i.e., ground truth is 0)
    penalty_values = nocs_normalized[mask]  # Get NOCS values outside the mask
    
    # Loss is the mean squared error for these penalty values
    loss = torch.mean(penalty_values**2)  # Penalize the squared values
    
    return loss

def main(config):
    setup_environment(str(config.gpu_id))
    make_log_dirs(config.weight_dir, config.val_img_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model instantiation and compilation
    generator = multidino(input_resolution=config.size, num_bins=config.num_bins, num_labels=config.num_labels, freeze_backbone=config.freeze_backbone)
    generator.to(device)

    # Optimizer instantiation
    optimizer_generator = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.epsilon)
    #optimizer_generator = optim.Adam(generator.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Instantiate train and val dataset + dataloaders
    train_dataset = create_webdataset(config.train_data_root, config.size, config.shuffle_buffer, augment=config.augmentation, center_crop=False)
    val_dataset = create_webdataset(config.val_data_root, config.size, config.shuffle_buffer, augment=False, center_crop=False)

    train_dataloader = wds.WebLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.train_num_workers, drop_last=True, collate_fn=custom_collate_fn,
    )
    val_dataloader = wds.WebLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.val_num_workers, drop_last=True, collate_fn=custom_collate_fn,
    )

    pointclouds = preload_pointclouds(config.models_root)

    # Training Loop
    epoch = 0
    iteration = 0
    loss_log = []

    for epoch in range(config.max_epochs):

        start_time_epoch = time.time()
        running_loss = 0.0 
        running_cons_loss = 0.0 
        running_sym_loss = 0.0 
        running_rot_loss = 0.0 
        generator.train()

        # Shuffle before epoch
        train_dataloader.unbatched().shuffle(1000).batched(config.batch_size)
        val_dataloader.unbatched().shuffle(1000).batched(config.batch_size)
       
        for step, batch in enumerate(train_dataloader):
            start_time_iteration = time.time()

            # Update learning rate with linear warmup
            if step < config.warmup_steps and epoch == 0:
                lr = config.lr * (step / config.warmup_steps)
            else:
                lr = config.lr  # Use target learning rate after warmup
            
            # Update the optimizer learning rate
            for param_group in optimizer_generator.param_groups:
                param_group['lr'] = lr

            # unwrap the batch
            rgb_images = batch['rgb']
            mask_images = batch['mask']
            nocs_images = batch['nocs']
            infos = batch['info']

            pcs = []
            cls_gt = []
            for entry in infos:
                obj_name = entry["obj_name"]
                obj_cat = entry["category_id"]
                
                gt_points = pointclouds[(str(obj_cat), obj_name)]
                pcs.append(gt_points)
                cls_gt.append(obj_cat-1)

            pcs_np = np.array(pcs)
            pcs_gt = torch.tensor(pcs_np, dtype=torch.float32)
            pcs_gt = pcs_gt.to(device)  # 16, 10000, 3

            cls_gt = np.array(cls_gt)
            cls_gt = torch.tensor(cls_gt, dtype=torch.long)
            cls_gt = cls_gt.to(device)  # 16, 10000, 3

            # Normalize mask to be binary (0 or 1)
            binary_mask = (mask_images > 0).float()  # Converts mask to 0 or 1
            rgb_images_masked = rgb_images.permute(0, 3, 1, 2).float()  # Convert to float to prevent data type issues
            binary_mask = binary_mask.permute(0, 3, 1, 2)  # Make sure mask has same shape
            rgb_images_masked = rgb_images_masked * binary_mask
            rgb_images_masked = rgb_images_masked.permute(0, 2, 3, 1)

            # RGB processing
            rgb_images_masked = torch.clamp(rgb_images_masked.float(), min=0.0, max=255.0)
            rgb_images_masked = (rgb_images_masked.float() / 127.5) - 1
            rgb_images_masked = rgb_images_masked.permute(0, 3, 1, 2)
            rgb_images_masked = rgb_images_masked.to(device)

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
            nocs_estimated, star_output, dash_output, symmetry_order, quaternion_estimated = generator(rgb_images_masked)

            # print("NOCS Estimated Range: ", nocs_estimated.min().item(), nocs_estimated.max().item())
            # print("Star Output Range: ", star_output.min().item(), star_output.max().item())
            # print("Dash Output Range: ", dash_output.min().item(), dash_output.max().item())
            # print()

            #cons_loss = consistency_loss(star_output, dash_output)  # Consistency between star and dash
            sym_loss = symmetry_aware_loss(generator.destarize(star_output, symmetry_order), symmetry_order)  # Symmetry enforcement

            # LOSSES
            # 1.) Rotation loss - from R to quat
            quaternion_est_normalized = normalize_quaternion(quaternion_estimated)
            rotation_est = quaternion_to_matrix(quaternion_est_normalized)

            gt_points_transformed = apply_rotation(pcs_gt, rotation_gt)  # Shape: (batch_size, 1000, 3)
            est_points_transformed = apply_rotation(pcs_gt, rotation_est)  # Shape: (batch_size, 1000, 3)

            #rot_loss = F.l1_loss(quaternion_est_normalized, quaternion_rotation_gt)
            rot_loss = add_loss(est_points_transformed, gt_points_transformed) * 1000

            cons_loss = mask_penalty_loss(nocs_estimated, binary_mask) + mask_penalty_stardash(star_output, binary_mask) + mask_penalty_stardash(dash_output, binary_mask)
            regression_nocs_loss = F.mse_loss(nocs_estimated, nocs_images_normalized_gt)

            # LOSSES Summation
            loss = 1.0 * cons_loss + 1.0 * sym_loss + 1.0 * rot_loss

            # Loss backpropagation
            optimizer_generator.zero_grad()
            loss.backward()

            # Optimizer gradient update
            optimizer_generator.step()
            elapsed_time_iteration = time.time() - start_time_iteration

            running_cons_loss += cons_loss.item()
            running_sym_loss += sym_loss.item()
            running_rot_loss += rot_loss.item()

            running_loss += loss.item()
            iteration += 1

            if (step + 1) % 10 == 0:
                avg_loss = running_loss / 100
                avg_running_cons_loss = running_cons_loss / 100
                avg_running_rot_loss = running_rot_loss / 100
                avg_running_sym_loss = running_sym_loss / 100

                elapsed_time_iteration = time.time() - start_time_iteration
                lr_current = optimizer_generator.param_groups[0]['lr']
                print("Epoch {:02d}, Iteration {:03d}, Loss: {:.4f}, Consistency Loss: {:.4f}, Sym Loss: {:.4f}, Rot Loss: {:.4f}, lr_gen: {:.6f}, Time per 100 Iterations: {:.4f} seconds".format(
                    epoch, iteration, avg_loss, avg_running_cons_loss, avg_running_sym_loss, avg_running_rot_loss, lr_current, elapsed_time_iteration))

                # Log to JSON
                loss_log.append({
                    "epoch": epoch,
                    "iteration": iteration,
                    "consistency_loss": avg_running_cons_loss,
                    "sym_loss": avg_running_sym_loss,
                    "rot_loss": avg_running_rot_loss,
                    "learning_rate": lr_current,
                    "time_per_100_iterations": elapsed_time_iteration
                })

                running_loss = 0.0 
                running_cons_loss = 0.0 
                running_sym_loss = 0.0 
                running_rot_loss = 0.0 

                imgfn = config.val_img_dir + "/{:03d}_{:03d}.jpg".format(epoch, iteration)
                plot_progress_imgs(imgfn, rgb_images_masked, nocs_images_normalized_gt, nocs_estimated, star_output, dash_output, rotation_est)

        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the whole epoch: {:.4f} seconds".format(elapsed_time_epoch))

        epoch += 1
        iteration = 0   

if __name__ == "__main__":
    args = parse_args()
    
    # Load the config file passed as argument
    config = load_config(args.config)
    
    # Call main with the loaded config
    main(config)