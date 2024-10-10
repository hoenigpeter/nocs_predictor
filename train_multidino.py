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

from imgaug import augmenters as iaa
import imgaug.augmenters as iaa  # noqa


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

def get_batch_count(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count

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

def l1_rotation_loss(R_pred, R_gt):
    if torch.isnan(R_pred).any() or torch.isnan(R_gt).any():
        print("NaN detected in rotation matrices!")
        print(f"R_pred: {R_pred}")
        print(f"R_gt: {R_gt}")
        raise ValueError("NaN values found in rotation matrices.")

    if torch.isinf(R_pred).any() or torch.isinf(R_gt).any():
        print("Inf detected in rotation matrices!")
        print(f"R_pred: {R_pred}")
        print(f"R_gt: {R_gt}")
        raise ValueError("Inf values found in rotation matrices.")

    return F.l1_loss(R_pred, R_gt)

def main():
    max_epochs = 100
    batch_size = 16
    shuffle_buffer = 1000

    train_data_root = "/ssd3/datasets_bop/housecat6d_nocs_train_with_rotation/scene{01..34}.tar"
    val_data_root = "/ssd3/datasets_bop/housecat6d_nocs_val_with_info/val_scene{1..2}.tar"

    size = 224
    num_bins = 50

    lr = 1e-4
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    setup_environment(sys.argv[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_dir = "./weights_custom_mse"
    if not(os.path.exists(weight_dir)):
            os.makedirs(weight_dir)

    val_img_dir = "./val_img_custom_mse"
    if not(os.path.exists(val_img_dir)):
        os.makedirs(val_img_dir)

    generator = multidino(input_resolution=size, num_bins=num_bins)
    generator.to(device)

    # Move the model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nocs_logits_loss = nn.CrossEntropyLoss()

    optimizer_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon)
    scheduler_generator = optim.lr_scheduler.CosineAnnealingLR(optimizer_generator, max_epochs, eta_min=1e-7)

    epoch = 0
    iteration = 0

    loss_log = []

    train_dataset = create_webdataset(train_data_root, size, shuffle_buffer, augment=True)
    val_dataset = create_webdataset(val_data_root, size, shuffle_buffer, augment=False)
    

    train_dataloader = wds.WebLoader(
        train_dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True, collate_fn=custom_collate_fn,
    )
    val_dataloader = wds.WebLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=16, drop_last=True, collate_fn=custom_collate_fn,
    )

    for epoch in range(max_epochs):
        start_time_epoch = time.time()
        running_loss = 0.0 
        running_binary_nocs_loss = 0.0 
        running_regression_nocs_loss = 0.0 
        running_masked_nocs_loss = 0.0 
        running_seg_loss = 0.0 
        running_rot_loss = 0.0 

        generator.train()

        train_dataloader.unbatched().shuffle(10000).batched(batch_size)
        val_dataloader.unbatched().shuffle(10000).batched(batch_size)
       
        for step, batch in enumerate(train_dataloader):
            start_time_iteration = time.time()

            rgb_images = batch['rgb']
            mask_images = batch['mask']
            nocs_images = batch['nocs']
            infos = batch['info']

            rotations = [torch.tensor(entry["rotation"], dtype=torch.float32) for entry in infos]
            rotation_gt = torch.stack(rotations).to(device)

            mask_images_gt = mask_images.float() / 255.0
            mask_images_gt = mask_images_gt.permute(0, 3, 1, 2)

            #rgb_images = rgb_images.float() * mask_images
            rgb_images = torch.clamp(rgb_images.float(), min=0.0, max=255.0)

            rgb_images = (rgb_images.float() / 127.5) - 1
            nocs_images_normalized = (nocs_images.float() / 127.5) - 1

            rgb_images = rgb_images.permute(0, 3, 1, 2)
 
            nocs_images_normalized = nocs_images_normalized.permute(0, 3, 1, 2)

            rgb_images_gt = rgb_images.to(device)
            nocs_images_normalized_gt = nocs_images_normalized.to(device)
            mask_images_gt = mask_images_gt.to(device)

            nocs_logits, masks_estimated, quaternion_estimated = generator(rgb_images_gt)

            quaternion_est_normalized = normalize_quaternion(quaternion_estimated)
            quaternion_rotation_gt = matrix_to_quaternion(rotation_gt)

            rot_loss = F.l1_loss(quaternion_est_normalized, quaternion_rotation_gt)
            rot_estimated_R = quaternion_to_matrix(quaternion_estimated)

            scaled_nocs = (nocs_images_normalized_gt + 1) * (num_bins - 1) / 2.0
            target_bins = scaled_nocs.long().clamp(0, num_bins - 1)  # Ensure values are within bin range
            
            # Calculate cross-entropy loss using the bin logits
            binary_nocs_loss = 0
            for i in range(3):  # Iterate over x, y, z coordinates
                binary_nocs_loss += nocs_logits_loss(nocs_logits[:, i], target_bins[:, i])
            
            # Apply softmax to logits to get final NOCS map in the range [0, num_bins-1]
            nocs_bins = torch.softmax(nocs_logits, dim=2)  # Softmax over the bin dimension
            # Convert the binned values to continuous NOCS values
            bin_centers = torch.linspace(-1, 1, num_bins).to(nocs_logits.device)  # Bin centers
            nocs_estimated = torch.sum(nocs_bins * bin_centers.view(1, 1, num_bins, 1, 1), dim=2)  # Shape [batch_size, 3, 224, 224]
        
            regression_nocs_loss = F.l1_loss(nocs_estimated, nocs_images_normalized_gt)
            # Threshold the masks to get binary values
            binary_masks = (masks_estimated > 0.5).float()  # Convert to float for multiplication

            # Reshape binary_masks to match the shape of nocs_estimated (broadcasting mask)
            binary_masks_expanded = binary_masks.expand_as(nocs_estimated)

            masked_nocs_estimated = nocs_estimated * binary_masks_expanded
            masked_nocs_gt = nocs_images_normalized_gt * binary_masks_expanded

            # Compute MSE loss only on masked pixels
            masked_nocs_loss = F.mse_loss(masked_nocs_estimated, masked_nocs_gt)

            seg_loss = F.mse_loss(masks_estimated, mask_images_gt[:, 0, :, :].unsqueeze(1)) 

            #loss = seg_loss + rot_loss + nocs_loss
            loss = binary_nocs_loss + regression_nocs_loss + masked_nocs_loss + seg_loss + rot_loss

            optimizer_generator.zero_grad()
            loss.backward()
            optimizer_generator.step()
            elapsed_time_iteration = time.time() - start_time_iteration  # Calculate elapsed time for the current iteration

            running_binary_nocs_loss += binary_nocs_loss.item()  # Accumulate loss
            running_regression_nocs_loss += regression_nocs_loss.item()  # Accumulate loss
            running_masked_nocs_loss += masked_nocs_loss.item()  # Accumulate loss
            running_seg_loss += seg_loss.item()  # Accumulate loss
            running_rot_loss += rot_loss.item()  # Accumulate loss

            running_loss += loss.item()  # Accumulate loss
            iteration += 1

            # Print average loss every 100 iterations
            if (step + 1) % 100 == 0:
                avg_loss = running_loss / 100
                avg_running_binary_nocs_loss = running_binary_nocs_loss / 100
                avg_running_regression_nocs_loss = running_regression_nocs_loss / 100
                avg_running_masked_nocs_loss = running_masked_nocs_loss / 100
                avg_running_seg_loss = running_seg_loss / 100
                avg_running_rot_loss = running_rot_loss / 100

                elapsed_time_iteration = time.time() - start_time_iteration  # Calculate elapsed time for the last 100 iterations
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

                f,ax = plt.subplots(10,6,figsize=(10,20))
                # Define arrow parameters
                arrow_length = 0.3  # Length of the arrows
                arrow_width = 0.03 
                arrow_color = 'red'  # Color of the arrows
                imgfn = val_img_dir + "/{:03d}_{:03d}.jpg".format(epoch, iteration)
                
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

                    # Define colors for the arrows
                    arrow_colors = {
                        'x': 'red',
                        'y': 'green',
                        'z': 'blue'
                    }

                    # Get the start point (e.g., center of the image in normalized coordinates)
                    start_point = np.array([0.5, 0.5])  # Center of the image

                    # Iterate over each arrow direction and plot
                    for key, direction in arrow_directions.items():
                        # Transform the arrow direction using the rotation matrix
                        transformed_arrow = rotation_matrix @ direction
                        
                        # Calculate end point based on the transformed arrow
                        end_point = start_point + (transformed_arrow[:2] * arrow_length)  # Only use x and y for 2D
                        
                        # Plot the arrow
                        ax[i, 0].quiver(
                            start_point[0] * rgb_images[i].shape[2], start_point[1] * rgb_images[i].shape[1],
                            (end_point[0] - start_point[0]) * rgb_images[i].shape[2], 
                            (end_point[1] - start_point[1]) * rgb_images[i].shape[1],
                            angles='xy', scale_units='xy', scale=1, color=arrow_colors[key], width=arrow_width
                        )

                plt.savefig(imgfn, dpi=300)
                plt.close()

        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the whole epoch: {:.4f} seconds".format(elapsed_time_epoch))

        imgfn = val_img_dir + "/val_{:03d}.jpg".format(epoch)

        generator.eval()
        running_loss = 0.0
        running_binary_nocs_loss = 0
        running_regression_nocs_loss = 0
        running_masked_nocs_loss = 0
        running_seg_loss = 0

        val_iter = 0
        start_time_epoch = time.time()

        for step, batch in enumerate(val_dataloader):
            rgb_images = batch['rgb']
            mask_images = batch['mask']
            nocs_images = batch['nocs']
            infos = batch['info']

            rotations = [torch.tensor(entry["rotation"], dtype=torch.float32) for entry in infos]
            rotation_gt = torch.stack(rotations).to(device)

            mask_images_gt = mask_images.float() / 255.0
            mask_images_gt = mask_images_gt.permute(0, 3, 1, 2)

            #rgb_images = rgb_images.float() * mask_images
            rgb_images = torch.clamp(rgb_images.float(), min=0.0, max=255.0)

            rgb_images = (rgb_images.float() / 127.5) - 1
            nocs_images_normalized = (nocs_images.float() / 127.5) - 1

            rgb_images = rgb_images.permute(0, 3, 1, 2)
 
            nocs_images_normalized = nocs_images_normalized.permute(0, 3, 1, 2)

            rgb_images_gt = rgb_images.to(device)
            nocs_images_normalized_gt = nocs_images_normalized.to(device)
            mask_images_gt = mask_images_gt.to(device)

            nocs_logits, masks_estimated, quaternion_estimated = generator(rgb_images_gt)

            quaternion_est_normalized = normalize_quaternion(quaternion_estimated)
            quaternion_rotation_gt = matrix_to_quaternion(rotation_gt)

            rot_loss = F.l1_loss(quaternion_est_normalized, quaternion_rotation_gt)
            rot_estimated_R = quaternion_to_matrix(quaternion_estimated)

            scaled_nocs = (nocs_images_normalized_gt + 1) * (num_bins - 1) / 2.0
            target_bins = scaled_nocs.long().clamp(0, num_bins - 1)  # Ensure values are within bin range
            
            # Calculate cross-entropy loss using the bin logits
            binary_nocs_loss = 0
            for i in range(3):  # Iterate over x, y, z coordinates
                binary_nocs_loss += nocs_logits_loss(nocs_logits[:, i], target_bins[:, i])
            
            # Apply softmax to logits to get final NOCS map in the range [0, num_bins-1]
            nocs_bins = torch.softmax(nocs_logits, dim=2)  # Softmax over the bin dimension
            # Convert the binned values to continuous NOCS values
            bin_centers = torch.linspace(-1, 1, num_bins).to(nocs_logits.device)  # Bin centers
            nocs_estimated = torch.sum(nocs_bins * bin_centers.view(1, 1, num_bins, 1, 1), dim=2)  # Shape [batch_size, 3, 224, 224]
        
            regression_nocs_loss = F.l1_loss(nocs_estimated, nocs_images_normalized_gt)
            # Threshold the masks to get binary values
            binary_masks = (masks_estimated > 0.5).float()  # Convert to float for multiplication

            # Reshape binary_masks to match the shape of nocs_estimated (broadcasting mask)
            binary_masks_expanded = binary_masks.expand_as(nocs_estimated)

            masked_nocs_estimated = nocs_estimated * binary_masks_expanded
            masked_nocs_gt = nocs_images_normalized_gt * binary_masks_expanded

            # Compute MSE loss only on masked pixels
            masked_nocs_loss = F.mse_loss(masked_nocs_estimated, masked_nocs_gt)

            seg_loss = F.mse_loss(masks_estimated, mask_images_gt[:, 0, :, :].unsqueeze(1)) 

            #loss = seg_loss + rot_loss + nocs_loss
            loss = binary_nocs_loss + regression_nocs_loss + masked_nocs_loss + seg_loss + rot_loss

            optimizer_generator.zero_grad()
            loss.backward()
            optimizer_generator.step()
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

        f,ax = plt.subplots(10,6,figsize=(10,20))
        # Define arrow parameters
        arrow_length = 0.3  # Length of the arrows
        arrow_width = 0.03 
        arrow_color = 'red'  # Color of the arrows

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

            # Define colors for the arrows
            arrow_colors = {
                'x': 'red',
                'y': 'green',
                'z': 'blue'
            }

            # Get the start point (e.g., center of the image in normalized coordinates)
            start_point = np.array([0.5, 0.5])  # Center of the image

            # Iterate over each arrow direction and plot
            for key, direction in arrow_directions.items():
                # Transform the arrow direction using the rotation matrix
                transformed_arrow = rotation_matrix @ direction
                
                # Calculate end point based on the transformed arrow
                end_point = start_point + (transformed_arrow[:2] * arrow_length)  # Only use x and y for 2D
                
                # Plot the arrow
                ax[i, 0].quiver(
                    start_point[0] * rgb_images[i].shape[2], start_point[1] * rgb_images[i].shape[1],
                    (end_point[0] - start_point[0]) * rgb_images[i].shape[2], 
                    (end_point[1] - start_point[1]) * rgb_images[i].shape[1],
                    angles='xy', scale_units='xy', scale=1, color=arrow_colors[key], width=arrow_width
                )

        plt.savefig(imgfn, dpi=300)
        plt.close()
        
        #scheduler_generator.step()

        if epoch % 10 == 0:
            torch.save(generator.state_dict(), os.path.join(weight_dir, f'generator_epoch_{epoch}.pth'))

        # Save loss log to JSON after each epoch
        with open("loss_log.json", "w") as f:
            json.dump(loss_log, f, indent=4)

        epoch += 1
        iteration = 0   

# Run the main function
if __name__ == "__main__":
    main()