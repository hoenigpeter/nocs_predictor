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
                    preload_pointclouds, plot_single_image, apply_rotation, parse_args, load_config, \
                    add_loss

from networks import UnetGeneratorMultiHead, UnetGeneratoNOCSHead, UnetGeneratoNOCSBinHead
import torch
import torch.nn.functional as F

def transformer_loss(x_channel_est, y_channel_est, z_channel_est, gt_nocs_images, transformations_list, cross_entropy_loss):
    batch_size, _, H, W = gt_nocs_images.shape
    min_losses = []

    for i in range(batch_size):
        gt_image = gt_nocs_images[i]  # Shape: [3, H, W]
        
        # Collect losses for all transformations
        losses = []
        for transform in transformations_list[i]:
            # Apply transformation to NOCS image
            nocs_transformed = torch.einsum("ij,jhw->ihw", transform, gt_image)
            nocs_transformed = nocs_transformed.unsqueeze(0)

            nocs_images_split = (((nocs_transformed + 1) / 2) * 255).long()

            x_channel_gt = torch.zeros(nocs_transformed.shape[0], 256, config.size, config.size, device=nocs_transformed.device, dtype=torch.float32)
            y_channel_gt = torch.zeros_like(x_channel_gt)
            z_channel_gt = torch.zeros_like(x_channel_gt)

            x_channel_gt.scatter_(1, nocs_images_split[:,0].unsqueeze(1), 1.0)
            y_channel_gt.scatter_(1, nocs_images_split[:,1].unsqueeze(1), 1.0)
            z_channel_gt.scatter_(1, nocs_images_split[:,2].unsqueeze(1), 1.0)

            loss_x = cross_entropy_loss(x_channel_est[i].unsqueeze(0), x_channel_gt)
            loss_y = cross_entropy_loss(y_channel_est[i].unsqueeze(0), y_channel_gt)
            loss_z = cross_entropy_loss(z_channel_est[i].unsqueeze(0), z_channel_gt)
            regression_nocs_loss = loss_x + loss_y + loss_z

            losses.append(regression_nocs_loss)
        
        # Take the minimum loss over all transformations for this item
        min_loss = torch.min(torch.stack(losses))
        min_losses.append(min_loss)

    # Mean minimum loss across the batch
    return torch.stack(min_losses).mean()

def generate_symmetry_transformations(symmetries, device):
    transformations = []

    # Predefine rotation matrix generator in PyTorch
    def rotation_matrix(axis, angle):
        c, s = torch.cos(angle), torch.sin(angle)
        if axis == 'x':
            return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=torch.float32)
        elif axis == 'y':
            return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=torch.float32)
        elif axis == 'z':
            return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=torch.float32)

    for symmetry in symmetries:
        x_sym, y_sym, z_sym = symmetry

        # Generate rotations based on symmetry order per axis
        x_rotations = [rotation_matrix('x', torch.tensor(2 * np.pi / x_sym * i)) for i in range(x_sym)]
        y_rotations = [rotation_matrix('y', torch.tensor(2 * np.pi / y_sym * i)) for i in range(y_sym)]
        z_rotations = [rotation_matrix('z', torch.tensor(2 * np.pi / z_sym * i)) for i in range(z_sym)]

        # Combine all rotations to get the final transformations
        final_rotations = []
        for x_rot in x_rotations:
            for y_rot in y_rotations:
                for z_rot in z_rotations:
                    final_rotations.append(x_rot @ y_rot @ z_rot)  # Use @ for matrix multiplication

        transformations.append(torch.stack(final_rotations).to(device))

    return transformations

def main(config):
    setup_environment(str(config.gpu_id))
    make_log_dirs(config.weight_dir, config.val_img_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model instantiation and compilation
    generator = UnetGeneratoNOCSBinHead(input_nc=3, output_nc=256, num_heads=3, num_downs=5, ngf=64)
    generator.to(device)
    print(generator)

    # generator = multidino(input_resolution=config.size, num_bins=config.num_bins, freeze_backbone=config.freeze_backbone)
    # generator.to(device)

    # Optimizer instantiation
    optimizer_generator = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.epsilon)
    #optimizer_generator = optim.Adam(generator.parameters(), lr=config.lr)

    class_weights_uvw = torch.ones(256)
    class_weights_uvw[0] = 0.01
    cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=class_weights_uvw).to(device)


    # Instantiate train and val dataset + dataloaders
    train_dataset = create_webdataset(config.train_data_root, config.size, config.shuffle_buffer, augment=config.augmentation, center_crop=config.center_crop, class_name=config.class_name)
    val_dataset = create_webdataset(config.val_data_root, config.size, config.shuffle_buffer, augment=False, center_crop=config.center_crop, class_name=config.class_name)

    train_dataloader = wds.WebLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.train_num_workers, drop_last=True, collate_fn=custom_collate_fn,
    )
    val_dataloader = wds.WebLoader(
            val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.val_num_workers, drop_last=True, collate_fn=custom_collate_fn,
    )

    pointclouds = preload_pointclouds(config.models_root, num_categories=config.num_categories)

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
    #     shuffle=True,  # Shuffle should be False for validation
    #     num_workers=config.val_num_workers, 
    #     drop_last=True, 
    #     collate_fn=custom_collate_fn
    # )

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
        running_bg_loss = 0.0

        generator.train()

        # # Shuffle before epoch
        train_dataloader.unbatched().shuffle(1000).batched(config.batch_size)
        val_dataloader.unbatched().shuffle(1000).batched(config.batch_size)
       
        for step, batch in enumerate(train_dataloader):
            start_time_iteration = time.time()

            # Update learning rate with linear warmup
            if iteration < config.warmup_steps:
                lr = config.lr * (iteration / config.warmup_steps)
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
            symmetries = []

            for entry in infos:
                obj_name = entry["obj_name"]
                obj_cat = entry["category_id"]
                obj_sym = entry[config.symmetry_type]
                
                gt_points = pointclouds[(str(obj_cat), str(obj_name))]
                pcs.append(gt_points)
                if config.with_transformer_loss == True: symmetries.append(np.array(obj_sym))

            pcs_np = np.array(pcs)
            pcs_gt = torch.tensor(pcs_np, dtype=torch.float32)
            pcs_gt = pcs_gt.to(device)

            #print(symmetries)
            if config.with_transformer_loss == True: transformations = generate_symmetry_transformations(symmetries, device)

            # Normalize mask to be binary (0 or 1)
            binary_mask = (mask_images > 0).float()  # Converts mask to 0 or 1
            binary_mask = binary_mask.permute(0, 3, 1, 2).to(device)  # Make sure mask has same shape
            
            # RGB processing
            rgb_images = torch.clamp(rgb_images.float(), min=0.0, max=255.0)
            rgb_images = rgb_images.permute(0, 3, 1, 2)
            rgb_images = rgb_images.to(device)
            rgb_images = rgb_images * binary_mask

            rgb_images_gt = (rgb_images.float() / 127.5) - 1

            # MASK processing
            mask_images_gt = mask_images.float() / 255.0
            mask_images_gt = mask_images_gt.permute(0, 3, 1, 2)
            mask_images_gt = mask_images_gt.to(device)

            # NOCS processing
            nocs_images_float = nocs_images.float()
            # Step 1: Adjust background pixels (where all three channels are 127)
            background_mask = (nocs_images_float[:, :, :, 0] == 127) & (nocs_images_float[:, :, :, 1] == 127) & (nocs_images_float[:, :, :, 2] == 127)
            nocs_images_float[background_mask] += 0.5  # Add 0.5 to the background pixels

            nocs_images_normalized = (nocs_images_float / 127.5) - 1
            nocs_images_normalized = nocs_images_normalized.permute(0, 3, 1, 2)
            nocs_images_normalized_gt = nocs_images_normalized.to(device)

            # ROTATION processing
            rotations = [torch.tensor(entry["rotation"], dtype=torch.float32) for entry in infos]
            rotation_gt = torch.stack(rotations).to(device)

            x_logits, y_logits, z_logits = generator(rgb_images_gt)
            x_channel = torch.max(x_logits, dim=1)[1] 
            y_channel = torch.max(y_logits, dim=1)[1] 
            z_channel = torch.max(z_logits, dim=1)[1] 
            nocs_estimated = torch.stack([x_channel, y_channel, z_channel], dim=1)
            nocs_estimated = (nocs_estimated / 127.5) - 1

            nocs_images_split = (((nocs_images_normalized_gt + 1) / 2) * 255).long()

            x_channel_gt = torch.zeros(rgb_images_gt.shape[0], 256, config.size, config.size, device=rgb_images_gt.device, dtype=torch.float32)
            x_channel_gt.scatter_(1, nocs_images_split[:,0].unsqueeze(1), 1.0)

            y_channel_gt = torch.zeros(rgb_images_gt.shape[0], 256, config.size, config.size, device=rgb_images_gt.device)
            y_channel_gt.scatter_(1, nocs_images_split[:,1].unsqueeze(1), 1.0)

            z_channel_gt = torch.zeros(rgb_images_gt.shape[0], 256, config.size, config.size, device=rgb_images_gt.device)
            z_channel_gt.scatter_(1, nocs_images_split[:,2].unsqueeze(1), 1.0)

            # 3.) NOCS Regression loss for each dimension
            if config.with_transformer_loss:
                regression_nocs_loss = transformer_loss(x_logits, y_logits, z_logits, nocs_images_normalized_gt, transformations, cross_entropy_loss)
            else:
                loss_x = cross_entropy_loss(x_logits, x_channel_gt)
                loss_y = cross_entropy_loss(y_logits, y_channel_gt)
                loss_z = cross_entropy_loss(z_logits, z_channel_gt)
                regression_nocs_loss = loss_x + loss_y + loss_z

            # LOSSES Summation
            loss = 0
            loss += config.w_NOCS_bins * 0
            loss += config.w_NOCS_cont * regression_nocs_loss
            loss += config.w_NOCS_ss * 0
            loss += config.w_seg * 0
            loss += config.w_Rot * 0
            loss += config.w_bg * 0
            
            # Loss backpropagation
            optimizer_generator.zero_grad()
            loss.backward()

            # Optimizer gradient update
            optimizer_generator.step()
            elapsed_time_iteration = time.time() - start_time_iteration

            running_binary_nocs_loss += 0
            running_regression_nocs_loss += regression_nocs_loss.item()
            running_masked_nocs_loss += 0
            running_seg_loss += 0
            running_rot_loss += 0
            running_bg_loss += 0

            running_loss += loss.item()
            iteration += 1

            if (step + 1) % config.iter_cnt == 0:
                avg_loss = running_loss / config.iter_cnt
                avg_running_binary_nocs_loss = running_binary_nocs_loss / config.iter_cnt
                avg_running_regression_nocs_loss = running_regression_nocs_loss / config.iter_cnt
                avg_running_masked_nocs_loss = running_masked_nocs_loss / config.iter_cnt
                avg_running_seg_loss = running_seg_loss / config.iter_cnt
                avg_running_rot_loss = running_rot_loss / config.iter_cnt
                avg_running_bg_loss = running_bg_loss / config.iter_cnt

                elapsed_time_iteration = time.time() - start_time_iteration
                lr_current = optimizer_generator.param_groups[0]['lr']
                print("Epoch {:02d}, Iter {:03d}, Loss: {:.4f}, Binary NOCS Loss: {:.4f}, Reg NOCS Loss: {:.4f}, Masked NOCS Loss: {:.4f}, Seg Loss: {:.4f}, Rot Loss: {:.4f}, BG Loss: {:.4f}, lr_gen: {:.6f}, Time: {:.4f} seconds".format(
                    epoch, step, avg_loss, avg_running_binary_nocs_loss, avg_running_regression_nocs_loss, \
                        avg_running_masked_nocs_loss, avg_running_seg_loss, avg_running_rot_loss, avg_running_bg_loss, lr_current, elapsed_time_iteration))

                # Log to JSON
                loss_log.append({
                    "epoch": epoch,
                    "iteration": iteration,
                    "binary_nocs_loss": avg_running_binary_nocs_loss,
                    "regression_nocs_loss": avg_running_regression_nocs_loss,
                    "masked_nocs_loss": avg_running_masked_nocs_loss,
                    "seg_loss": avg_running_seg_loss,
                    "rot_loss": avg_running_rot_loss,
                    "bg_loss": avg_running_bg_loss,
                    "learning_rate": lr_current,
                    "time_per_100_iterations": elapsed_time_iteration
                })

                running_loss = 0
                running_binary_nocs_loss = 0
                running_regression_nocs_loss = 0
                running_masked_nocs_loss = 0
                running_seg_loss = 0
                running_rot_loss = 0
                running_bg_loss = 0

                imgfn = config.val_img_dir + "/{:03d}_{:03d}.jpg".format(epoch, iteration)
                plot_progress_imgs(imgfn, rgb_images_gt, nocs_images_normalized_gt, nocs_estimated, mask_images_gt, mask_images_gt, mask_images_gt, rotation_gt)
        
        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the whole epoch: {:.4f} seconds".format(elapsed_time_epoch))

        generator.eval()
        running_loss = 0.0
        running_binary_nocs_loss = 0
        running_regression_nocs_loss = 0
        running_masked_nocs_loss = 0
        running_seg_loss = 0
        running_rot_loss = 0
        running_bg_loss = 0

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
                symmetries = []

                for entry in infos:
                    obj_name = entry["obj_name"]
                    obj_cat = entry["category_id"]
                    obj_sym = entry[config.symmetry_type]
                    
                    gt_points = pointclouds[(str(obj_cat), str(obj_name))]
                    pcs.append(gt_points)
                    if config.with_transformer_loss == True: symmetries.append(np.array(obj_sym))

                pcs_np = np.array(pcs)
                pcs_gt = torch.tensor(pcs_np, dtype=torch.float32)
                pcs_gt = pcs_gt.to(device)
                
                # Normalize mask to be binary (0 or 1)
                binary_mask = (mask_images > 0).float()  # Converts mask to 0 or 1
                binary_mask = binary_mask.permute(0, 3, 1, 2).to(device)  # Make sure mask has same shape
                
                # RGB processing
                rgb_images = torch.clamp(rgb_images.float(), min=0.0, max=255.0)
                rgb_images = rgb_images.permute(0, 3, 1, 2)
                rgb_images = rgb_images.to(device)
                rgb_images = rgb_images * binary_mask

                rgb_images_gt = (rgb_images.float() / 127.5) - 1

                # MASK processing
                mask_images_gt = mask_images.float() / 255.0
                mask_images_gt = mask_images_gt.permute(0, 3, 1, 2)
                mask_images_gt = mask_images_gt.to(device)

                # NOCS processing
                nocs_images_float = nocs_images.float()
                # Step 1: Adjust background pixels (where all three channels are 127)
                background_mask = (nocs_images_float[:, :, :, 0] == 127) & (nocs_images_float[:, :, :, 1] == 127) & (nocs_images_float[:, :, :, 2] == 127)
                nocs_images_float[background_mask] += 0.5  # Add 0.5 to the background pixels

                nocs_images_normalized = (nocs_images_float / 127.5) - 1
                nocs_images_normalized = nocs_images_normalized.permute(0, 3, 1, 2)
                nocs_images_normalized_gt = nocs_images_normalized.to(device)

                # ROTATION processing
                rotations = [torch.tensor(entry["rotation"], dtype=torch.float32) for entry in infos]
                rotation_gt = torch.stack(rotations).to(device)

                x_logits, y_logits, z_logits = generator(rgb_images_gt)
                x_channel = torch.max(x_logits, dim=1)[1] 
                y_channel = torch.max(y_logits, dim=1)[1] 
                z_channel = torch.max(z_logits, dim=1)[1] 
                nocs_estimated = torch.stack([x_channel, y_channel, z_channel], dim=1)
                nocs_estimated = (nocs_estimated / 127.5) - 1

                nocs_images_split = (((nocs_images_normalized_gt + 1) / 2) * 255).long()

                x_channel_gt = torch.zeros(rgb_images_gt.shape[0], 256, config.size, config.size, device=rgb_images_gt.device)
                x_channel_gt.scatter_(1, nocs_images_split[:,0].unsqueeze(1), 1.0)

                y_channel_gt = torch.zeros(rgb_images_gt.shape[0], 256, config.size, config.size, device=rgb_images_gt.device)
                y_channel_gt.scatter_(1, nocs_images_split[:,1].unsqueeze(1), 1.0)

                z_channel_gt = torch.zeros(rgb_images_gt.shape[0], 256, config.size, config.size, device=rgb_images_gt.device)
                z_channel_gt.scatter_(1, nocs_images_split[:,2].unsqueeze(1), 1.0)

                # 3.) NOCS Regression loss for each dimension
                if config.with_transformer_loss:
                    regression_nocs_loss = transformer_loss(x_logits, y_logits, z_logits, nocs_images_normalized_gt, transformations, cross_entropy_loss)
                else:
                    loss_x = cross_entropy_loss(x_logits, x_channel_gt)
                    loss_y = cross_entropy_loss(y_logits, y_channel_gt)
                    loss_z = cross_entropy_loss(z_logits, z_channel_gt)
                    regression_nocs_loss = loss_x + loss_y + loss_z

                # LOSSES Summation
                loss = 0
                loss += config.w_NOCS_bins * 0
                loss += config.w_NOCS_cont * regression_nocs_loss
                loss += config.w_NOCS_ss * 0
                loss += config.w_seg * 0
                loss += config.w_Rot * 0
                loss += config.w_bg * 0

                elapsed_time_iteration = time.time() - start_time_iteration  # Calculate elapsed time for the current iteration

                running_binary_nocs_loss += 0
                running_regression_nocs_loss += regression_nocs_loss.item()
                running_masked_nocs_loss += 0
                running_seg_loss += 0
                running_rot_loss += 0
                running_bg_loss += 0

                running_loss += loss.item()

                val_iter+=1

        avg_loss = running_loss / val_iter
        avg_running_binary_nocs_loss = running_binary_nocs_loss / val_iter
        avg_running_regression_nocs_loss = running_regression_nocs_loss / val_iter
        avg_running_masked_nocs_loss = running_masked_nocs_loss / val_iter
        avg_running_seg_loss = running_seg_loss / val_iter
        avg_running_rot_loss = running_rot_loss / val_iter
        avg_running_bg_loss = running_bg_loss / val_iter

        loss_log.append({
            "epoch": epoch,
            "val_binary_nocs_loss": avg_running_binary_nocs_loss,
            "val_regression_nocs_loss": avg_running_regression_nocs_loss,
            "val_masked_nocs_loss": avg_running_masked_nocs_loss,
            "val_seg_loss": avg_running_seg_loss,
            "val_rot_loss": avg_running_rot_loss,
            "val_bg_loss": avg_running_bg_loss,
            "val_learning_rate": lr_current,
            "val_time_per_100_iterations": elapsed_time_iteration
        })
        
        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the validation: {:.4f} seconds".format(elapsed_time_epoch))
        print("Val Loss: {:.4f}".format(avg_loss))

        imgfn = config.val_img_dir + "/val_{:03d}.jpg".format(epoch)
        plot_progress_imgs(imgfn, rgb_images_gt, nocs_images_normalized_gt, nocs_estimated, mask_images_gt, mask_images_gt, mask_images_gt, rotation_gt)
        
        if epoch % config.save_epoch_interval == 0:
            # Save the entire model
            torch.save(generator, os.path.join(config.weight_dir, f'generator_epoch_{epoch}.pth'))
            #torch.save(generator.state_dict(), os.path.join(config.weight_dir, f'generator_epoch_{epoch}.pth'))

        # Save loss log to JSON after each epoch
        with open(config.weight_dir + "/loss_log.json", "w") as f:
            json.dump(loss_log, f, indent=4)

        epoch += 1

if __name__ == "__main__":
    args = parse_args()
    
    # Load the config file passed as argument
    config = load_config(args.config)
    
    # Call main with the loaded config
    main(config)