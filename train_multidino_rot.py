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

import json
import webdataset as wds

from utils import WebDatasetWrapper, preprocess, normalize_quaternion, setup_environment, create_webdataset, custom_collate_fn, make_log_dirs, plot_progress_imgs, preload_pointclouds

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

def main(config):
    setup_environment(str(config.gpu_id))
    make_log_dirs(config.weight_dir, config.val_img_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model instantiation and compilation
    generator = multidino(input_resolution=config.size, num_bins=config.num_bins, freeze_backbone=config.freeze_backbone)
    generator.to(device)

    # Optimizer instantiation
    optimizer_generator = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.epsilon)
    #optimizer_generator = optim.Adam(generator.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Instantiate train and val dataset + dataloaders
    train_dataset = create_webdataset(config.train_data_root, config.size, config.shuffle_buffer, augment=config.augmentation)
    val_dataset = create_webdataset(config.val_data_root, config.size, config.shuffle_buffer, augment=False)

    train_dataloader = wds.WebLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.train_num_workers, drop_last=True, collate_fn=custom_collate_fn,
    )
    val_dataloader = wds.WebLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.val_num_workers, drop_last=True, collate_fn=custom_collate_fn,
    )

    # train_dataloader = DataLoader(
    #     train_dataset, 
    #     batch_size=config.batch_size, 
    #     shuffle=True,  # Shuffle should typically be True for training
    #     num_workers=config.train_num_workers, 
    #     drop_last=True, 
    #     collate_fn=custom_collate_fn
    # )

    # # Validation DataLoader
    # val_dataloader = DataLoader(
    #     val_dataset, 
    #     batch_size=config.batch_size, 
    #     shuffle=False,  # Shuffle should be False for validation
    #     num_workers=config.val_num_workers, 
    #     drop_last=True, 
    #     collate_fn=custom_collate_fn
    # )

    pointclouds = preload_pointclouds(config.models_root)

    # Training Loop
    epoch = 0
    iteration = 0
    loss_log = []

    for epoch in range(config.max_epochs):

        start_time_epoch = time.time()
        running_loss = 0.0 
        running_binary_nocs_loss = 0.0 
        running_regression_nocs_loss = 0.0 
        running_masked_nocs_loss = 0.0 
        running_seg_loss = 0.0 
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
            for entry in infos:
                obj_name = entry["obj_name"]
                obj_cat = entry["category_id"]
                
                gt_points = pointclouds[(str(obj_cat), obj_name)]
                pcs.append(gt_points)

            pcs_np = np.array(pcs)
            pcs_gt = torch.tensor(pcs_np, dtype=torch.float32)
            pcs_gt = pcs_gt.to(device)  # 16, 10000, 3

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
            x_logits, y_logits, z_logits, quaternion_estimated = generator(rgb_images_masked)

            # LOSSES
            # 1.) Rotation loss - from R to quat
            quaternion_est_normalized = normalize_quaternion(quaternion_estimated)
            rotation_est = quaternion_to_matrix(quaternion_est_normalized)

            gt_points_transformed = apply_rotation(pcs_gt, rotation_gt)  # Shape: (batch_size, 1000, 3)
            est_points_transformed = apply_rotation(pcs_gt, rotation_est)  # Shape: (batch_size, 1000, 3)

            #rot_loss = F.l1_loss(quaternion_est_normalized, quaternion_rotation_gt)
            rot_loss = add_loss(est_points_transformed, gt_points_transformed) * 100

            # 3.) NOCS Logits loss
            scaled_nocs = (nocs_images_normalized_gt + 1) * (config.num_bins - 1) / 2.0
            target_bins = scaled_nocs.long().clamp(0, config.num_bins - 1)  # Ensure values are within bin range
            
            binary_nocs_loss = 0
            binary_nocs_loss += F.cross_entropy(x_logits, target_bins[:, 0])
            binary_nocs_loss += F.cross_entropy(y_logits, target_bins[:, 1])
            binary_nocs_loss += F.cross_entropy(z_logits, target_bins[:, 2])

            # 3.) NOCS Regression loss for each dimension

            # Softmax over the bin dimension for x, y, z logits
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

            # Calculate the L1 regression loss between the estimated NOCS map and ground truth
            regression_nocs_loss = F.mse_loss(nocs_estimated, nocs_images_normalized_gt)

            # LOSSES Summation
            loss = 1.0 * binary_nocs_loss + 0 * regression_nocs_loss + 1.0 * rot_loss

            # Loss backpropagation
            optimizer_generator.zero_grad()
            loss.backward()

            # Optimizer gradient update
            optimizer_generator.step()
            elapsed_time_iteration = time.time() - start_time_iteration

            running_binary_nocs_loss += binary_nocs_loss.item()
            running_regression_nocs_loss += regression_nocs_loss.item()
            running_rot_loss += rot_loss.item()

            running_loss += loss.item()
            iteration += 1

            if (step + 1) % 100 == 0:
                avg_loss = running_loss / 100
                avg_running_binary_nocs_loss = running_binary_nocs_loss / 100
                avg_running_regression_nocs_loss = running_regression_nocs_loss / 100
                avg_running_rot_loss = running_rot_loss / 100

                elapsed_time_iteration = time.time() - start_time_iteration
                lr_current = optimizer_generator.param_groups[0]['lr']
                print("Epoch {:02d}, Iteration {:03d}, Loss: {:.4f}, Binary NOCS Loss: {:.4f}, Reg NOCS Loss: {:.4f}, Rot Loss: {:.4f}, lr_gen: {:.6f}, Time per 100 Iterations: {:.4f} seconds".format(
                    epoch, iteration, avg_loss, avg_running_binary_nocs_loss, avg_running_regression_nocs_loss, avg_running_rot_loss, lr_current, elapsed_time_iteration))

                # Log to JSON
                loss_log.append({
                    "epoch": epoch,
                    "iteration": iteration,
                    "binary_nocs_loss": avg_running_binary_nocs_loss,
                    "regression_nocs_loss": avg_running_regression_nocs_loss,
                    "rot_loss": avg_running_rot_loss,
                    "learning_rate": lr_current,
                    "time_per_100_iterations": elapsed_time_iteration
                })

                running_loss = 0
                running_binary_nocs_loss = 0
                running_regression_nocs_loss = 0
                running_rot_loss = 0

                imgfn = config.val_img_dir + "/{:03d}_{:03d}.jpg".format(epoch, iteration)
                plot_progress_imgs(imgfn, rgb_images, rgb_images_masked, nocs_images_normalized_gt, nocs_estimated, mask_images, rotation_est)

        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the whole epoch: {:.4f} seconds".format(elapsed_time_epoch))

        generator.eval()
        running_loss = 0.0
        running_binary_nocs_loss = 0
        running_regression_nocs_loss = 0
        running_rot_loss = 0

        val_iter = 0
        start_time_epoch = time.time()

        with torch.no_grad():
            for step, batch in enumerate(val_dataloader):
                start_time_iteration = time.time()

                # unwrap the batch
                rgb_images = batch['rgb']
                mask_images = batch['mask']
                nocs_images = batch['nocs']
                infos = batch['info']

                pcs = []
                for entry in infos:
                    obj_name = entry["obj_name"]
                    obj_cat = entry["category_id"]
                    
                    gt_points = pointclouds[(str(obj_cat), obj_name)]
                    pcs.append(gt_points)

                pcs_np = np.array(pcs)
                pcs_gt = torch.tensor(pcs_np, dtype=torch.float32)
                pcs_gt = pcs_gt.to(device)  # 16, 10000, 3

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
                x_logits, y_logits, z_logits, quaternion_estimated = generator(rgb_images_masked)

                # LOSSES
                # 1.) Rotation loss - from R to quat
                quaternion_est_normalized = normalize_quaternion(quaternion_estimated)
                rotation_est = quaternion_to_matrix(quaternion_est_normalized)

                gt_points_transformed = apply_rotation(pcs_gt, rotation_gt)  # Shape: (batch_size, 1000, 3)
                est_points_transformed = apply_rotation(pcs_gt, rotation_est)  # Shape: (batch_size, 1000, 3)

                #rot_loss = F.l1_loss(quaternion_est_normalized, quaternion_rotation_gt)
                rot_loss = add_loss(est_points_transformed, gt_points_transformed) * 100

                # 3.) NOCS Logits loss
                scaled_nocs = (nocs_images_normalized_gt + 1) * (config.num_bins - 1) / 2.0
                target_bins = scaled_nocs.long().clamp(0, config.num_bins - 1)  # Ensure values are within bin range
                
                binary_nocs_loss = 0
                binary_nocs_loss += F.cross_entropy(x_logits, target_bins[:, 0])
                binary_nocs_loss += F.cross_entropy(y_logits, target_bins[:, 1])
                binary_nocs_loss += F.cross_entropy(z_logits, target_bins[:, 2])

                # 3.) NOCS Regression loss for each dimension

                # Softmax over the bin dimension for x, y, z logits
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

                # Calculate the L1 regression loss between the estimated NOCS map and ground truth
                regression_nocs_loss = F.mse_loss(nocs_estimated, nocs_images_normalized_gt)

                # LOSSES Summation
                loss = 1.0 * binary_nocs_loss + 0 * regression_nocs_loss + 1 * rot_loss

                elapsed_time_iteration = time.time() - start_time_iteration  # Calculate elapsed time for the current iteration

                running_binary_nocs_loss += binary_nocs_loss.item()  # Accumulate loss
                running_regression_nocs_loss += regression_nocs_loss.item()  # Accumulate loss
                running_rot_loss += rot_loss.item()

                running_loss += loss.item()  # Accumulate loss

                val_iter+=1

        avg_loss = running_loss / val_iter
        avg_running_binary_nocs_loss = running_binary_nocs_loss / val_iter
        avg_running_regression_nocs_loss = running_regression_nocs_loss / val_iter
        avg_running_rot_loss = running_rot_loss / val_iter
        
        loss_log.append({
            "epoch": epoch,
            "val_binary_nocs_loss": avg_running_binary_nocs_loss,
            "val_regression_nocs_loss": avg_running_regression_nocs_loss,
            "val_rot_loss": avg_running_rot_loss,
            "val_learning_rate": lr_current,
            "val_time_per_100_iterations": elapsed_time_iteration
        })
        
        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the validation: {:.4f} seconds".format(elapsed_time_epoch))
        print("Val Loss: {:.4f}".format(avg_loss))

        imgfn = config.val_img_dir + "/val_{:03d}.jpg".format(epoch)
        plot_progress_imgs(imgfn, rgb_images, rgb_images_masked, nocs_images_normalized_gt, nocs_estimated, mask_images, rotation_est)
        
        if epoch % config.save_epoch_interval == 0:
            torch.save(generator.state_dict(), os.path.join(config.weight_dir, f'generator_epoch_{epoch}.pth'))

        # Save loss log to JSON after each epoch
        with open("loss_log.json", "w") as f:
            json.dump(loss_log, f, indent=4)

        epoch += 1
        iteration = 0   

if __name__ == "__main__":
    args = parse_args()
    
    # Load the config file passed as argument
    config = load_config(args.config)
    
    # Call main with the loaded config
    main(config)