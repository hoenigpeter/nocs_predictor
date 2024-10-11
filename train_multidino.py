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

import json
import webdataset as wds

from utils import WebDatasetWrapper, preprocess, normalize_quaternion, setup_environment, create_webdataset, custom_collate_fn, make_log_dirs, plot_progress_imgs
import config

def main():
    setup_environment(sys.argv[1])
    make_log_dirs(config.weight_dir, config.val_img_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model instantiation and compilation
    generator = multidino(input_resolution=config.size, num_bins=config.num_bins)
    generator.to(device)

    # for name, param in generator.named_parameters():
    #     if not param.requires_grad:
    #         print(f"Layer {name} is frozen.")
    #     else:
    #         print(f"Layer {name} is trainable.")


    # Optimizer instantiation
    #optimizer_generator = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.epsilon, weight_decay=config.weight_decay)
    optimizer_generator = optim.Adam(generator.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Instantiate train and val dataset + dataloaders
    train_dataset = create_webdataset(config.train_data_root, config.size, config.shuffle_buffer, augment=True)
    val_dataset = create_webdataset(config.val_data_root, config.size, config.shuffle_buffer, augment=False)

    train_dataloader = wds.WebLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.train_num_workers, drop_last=True, collate_fn=custom_collate_fn,
    )
    val_dataloader = wds.WebLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.val_num_workers, drop_last=True, collate_fn=custom_collate_fn,
    )

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
        train_dataloader.unbatched().shuffle(10000).batched(config.batch_size)
        val_dataloader.unbatched().shuffle(10000).batched(config.batch_size)
       
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

            # RGB processing
            rgb_images = torch.clamp(rgb_images.float(), min=0.0, max=255.0)
            rgb_images = (rgb_images.float() / 127.5) - 1
            rgb_images = rgb_images.permute(0, 3, 1, 2)
            rgb_images_gt = rgb_images.to(device)

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
            quaternion_rotation_gt = matrix_to_quaternion(rotation_gt)

            # forward pass through generator
            nocs_logits, masks_estimated, quaternion_estimated = generator(rgb_images_gt)

            # LOSSES
            # 1.) Rotation loss - from R to quat
            quaternion_est_normalized = normalize_quaternion(quaternion_estimated)
            rot_estimated_R = quaternion_to_matrix(quaternion_estimated)

            rot_loss = F.l1_loss(quaternion_est_normalized, quaternion_rotation_gt)

            # 2.) Mask loss
            binary_masks = (masks_estimated > 0.5).float()  # Convert to float for multiplication

            seg_loss = F.mse_loss(masks_estimated, mask_images_gt[:, 0, :, :].unsqueeze(1))

            # 3.) NOCS Logits loss
            scaled_nocs = (nocs_images_normalized_gt + 1) * (config.num_bins - 1) / 2.0
            target_bins = scaled_nocs.long().clamp(0, config.num_bins - 1)  # Ensure values are within bin range
            
            binary_nocs_loss = 0
            for i in range(3):  # Iterate over x, y, z coordinates
                binary_nocs_loss += F.cross_entropy(nocs_logits[:, i], target_bins[:, i])
            
            # 3.) NOCS Regression loss
            nocs_bins = torch.softmax(nocs_logits, dim=2)
            bin_centers = torch.linspace(-1, 1, config.num_bins).to(nocs_logits.device)  # Bin centers
            nocs_estimated = torch.sum(nocs_bins * bin_centers.view(1, 1, config.num_bins, 1, 1), dim=2)
            regression_nocs_loss = F.l1_loss(nocs_estimated, nocs_images_normalized_gt)

            # 4.) NOCS self-supervised loss
            binary_masks_expanded = binary_masks.expand_as(nocs_estimated)
            masked_nocs_estimated = nocs_estimated * binary_masks_expanded
            masked_nocs_gt = nocs_images_normalized_gt * binary_masks_expanded

            masked_nocs_loss = F.mse_loss(masked_nocs_estimated, masked_nocs_gt)

            # LOSSES Summation
            loss = 1.0 * binary_nocs_loss + 1.0 * regression_nocs_loss + 1.0 * masked_nocs_loss + 1.0 * seg_loss + 0.0 * rot_loss

            # Loss backpropagation
            optimizer_generator.zero_grad()
            loss.backward()

            # Optimizer gradient update
            optimizer_generator.step()
            elapsed_time_iteration = time.time() - start_time_iteration

            running_binary_nocs_loss += binary_nocs_loss.item()
            running_regression_nocs_loss += regression_nocs_loss.item()
            running_masked_nocs_loss += masked_nocs_loss.item()
            running_seg_loss += seg_loss.item()
            running_rot_loss += rot_loss.item()

            running_loss += loss.item()
            iteration += 1

            if (step + 1) % 100 == 0:
                avg_loss = running_loss / 100
                avg_running_binary_nocs_loss = running_binary_nocs_loss / 100
                avg_running_regression_nocs_loss = running_regression_nocs_loss / 100
                avg_running_masked_nocs_loss = running_masked_nocs_loss / 100
                avg_running_seg_loss = running_seg_loss / 100
                avg_running_rot_loss = running_rot_loss / 100

                elapsed_time_iteration = time.time() - start_time_iteration
                lr_current = optimizer_generator.param_groups[0]['lr']
                print("Epoch {:02d}, Iteration {:03d}, Loss: {:.4f}, Binary NOCS Loss: {:.4f}, Reg NOCS Loss: {:.4f}, Masked NOCS Loss: {:.4f}, Seg Loss: {:.4f}, Rot Loss: {:.4f}, lr_gen: {:.6f}, Time per 100 Iterations: {:.4f} seconds".format(
                    epoch, iteration, avg_loss, avg_running_binary_nocs_loss, avg_running_regression_nocs_loss, avg_running_masked_nocs_loss, avg_running_seg_loss, avg_running_rot_loss, lr_current, elapsed_time_iteration))

                # Log to JSON
                loss_log.append({
                    "epoch": epoch,
                    "iteration": iteration,
                    "binary_nocs_loss": avg_running_binary_nocs_loss,
                    "regression_nocs_loss": avg_running_regression_nocs_loss,
                    "masked_nocs_loss": avg_running_masked_nocs_loss,
                    "seg_loss": avg_running_seg_loss,
                    "rot_loss": avg_running_rot_loss,
                    "learning_rate": lr_current,
                    "time_per_100_iterations": elapsed_time_iteration
                })

                running_loss = 0
                running_binary_nocs_loss = 0
                running_regression_nocs_loss = 0
                running_masked_nocs_loss = 0
                running_seg_loss = 0
                running_rot_loss = 0

                imgfn = config.val_img_dir + "/{:03d}_{:03d}.jpg".format(epoch, iteration)
                plot_progress_imgs(imgfn, rgb_images, nocs_images_normalized_gt, nocs_estimated, mask_images, masks_estimated, binary_masks, rot_estimated_R)

        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the whole epoch: {:.4f} seconds".format(elapsed_time_epoch))

        generator.eval()
        running_loss = 0.0
        running_binary_nocs_loss = 0
        running_regression_nocs_loss = 0
        running_masked_nocs_loss = 0
        running_seg_loss = 0

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

                # RGB processing
                rgb_images = torch.clamp(rgb_images.float(), min=0.0, max=255.0)
                rgb_images = (rgb_images.float() / 127.5) - 1
                rgb_images = rgb_images.permute(0, 3, 1, 2)
                rgb_images_gt = rgb_images.to(device)

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
                quaternion_rotation_gt = matrix_to_quaternion(rotation_gt)

                # forward pass through generator
                nocs_logits, masks_estimated, quaternion_estimated = generator(rgb_images_gt)

                # LOSSES
                # 1.) Rotation loss - from R to quat
                quaternion_est_normalized = normalize_quaternion(quaternion_estimated)
                rot_estimated_R = quaternion_to_matrix(quaternion_estimated)

                rot_loss = F.l1_loss(quaternion_est_normalized, quaternion_rotation_gt)

                # 2.) Mask loss
                binary_masks = (masks_estimated > 0.5).float()  # Convert to float for multiplication

                seg_loss = F.mse_loss(masks_estimated, mask_images_gt[:, 0, :, :].unsqueeze(1))

                # 3.) NOCS Logits loss
                scaled_nocs = (nocs_images_normalized_gt + 1) * (config.num_bins - 1) / 2.0
                target_bins = scaled_nocs.long().clamp(0, config.num_bins - 1)  # Ensure values are within bin range
                
                binary_nocs_loss = 0
                for i in range(3):  # Iterate over x, y, z coordinates
                    binary_nocs_loss += F.cross_entropy(nocs_logits[:, i], target_bins[:, i])
                
                # 3.) NOCS Regression loss
                nocs_bins = torch.softmax(nocs_logits, dim=2)
                bin_centers = torch.linspace(-1, 1, config.num_bins).to(nocs_logits.device)  # Bin centers
                nocs_estimated = torch.sum(nocs_bins * bin_centers.view(1, 1, config.num_bins, 1, 1), dim=2)
                regression_nocs_loss = F.l1_loss(nocs_estimated, nocs_images_normalized_gt)

                # 4.) NOCS self-supervised loss
                binary_masks_expanded = binary_masks.expand_as(nocs_estimated)
                masked_nocs_estimated = nocs_estimated * binary_masks_expanded
                masked_nocs_gt = nocs_images_normalized_gt * binary_masks_expanded

                masked_nocs_loss = F.mse_loss(masked_nocs_estimated, masked_nocs_gt)

                # LOSSES Summation
                #loss = binary_nocs_loss + regression_nocs_loss + masked_nocs_loss + seg_loss + rot_loss
                loss = 1.0 * binary_nocs_loss + 0.5 * regression_nocs_loss + 0.5 * masked_nocs_loss + 1.0 * seg_loss + 1.0 * rot_loss

                elapsed_time_iteration = time.time() - start_time_iteration  # Calculate elapsed time for the current iteration

                running_binary_nocs_loss += binary_nocs_loss.item()  # Accumulate loss
                running_regression_nocs_loss += regression_nocs_loss.item()  # Accumulate loss
                running_masked_nocs_loss += masked_nocs_loss.item()  # Accumulate loss
                running_seg_loss += seg_loss.item()  # Accumulate loss
                running_rot_loss += rot_loss.item()  # Accumulate loss

                running_loss += loss.item()  # Accumulate loss

                val_iter+=1

        avg_loss = running_loss / val_iter
        avg_running_binary_nocs_loss = running_binary_nocs_loss / val_iter
        avg_running_regression_nocs_loss = running_regression_nocs_loss / val_iter
        avg_running_masked_nocs_loss = running_masked_nocs_loss / val_iter
        avg_running_seg_loss = running_seg_loss / val_iter
        avg_running_rot_loss = running_rot_loss / val_iter
        
        loss_log.append({
            "epoch": epoch,
            "val_binary_nocs_loss": avg_running_binary_nocs_loss,
            "val_regression_nocs_loss": avg_running_regression_nocs_loss,
            "val_masked_nocs_loss": avg_running_masked_nocs_loss,
            "val_seg_loss": avg_running_seg_loss,
            "val_rot_loss": avg_running_rot_loss,
            "val_learning_rate": lr_current,
            "val_time_per_100_iterations": elapsed_time_iteration
        })
        
        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the validation: {:.4f} seconds".format(elapsed_time_epoch))
        print("Val Loss: {:.4f}".format(avg_loss))

        imgfn = config.val_img_dir + "/val_{:03d}.jpg".format(epoch)
        plot_progress_imgs(imgfn, rgb_images, nocs_images_normalized_gt, nocs_estimated, mask_images, masks_estimated, binary_masks, rot_estimated_R)

        if epoch % 2 == 0:
            torch.save(generator.state_dict(), os.path.join(config.weight_dir, f'generator_epoch_{epoch}.pth'))

        # Save loss log to JSON after each epoch
        with open("loss_log.json", "w") as f:
            json.dump(loss_log, f, indent=4)

        epoch += 1
        iteration = 0   

# Run the main function
if __name__ == "__main__":
    main()