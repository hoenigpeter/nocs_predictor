# code adapted from: https://github.com/kirumang/Pix2Pose

import os
import sys

import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dino_model import Autoencoder as ae

import json
import webdataset as wds

from imgaug import augmenters as iaa
import imgaug.augmenters as iaa  # noqa

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
        # seq_syn = iaa.Sequential([        
        #                             iaa.Sometimes(0.5 * prob, iaa.CoarseDropout( p=0.2, size_percent=0.05) ),
        #                             iaa.Sometimes(0.5 * prob, iaa.GaussianBlur(1.2*np.random.rand())),
        #                             iaa.Sometimes(0.5 * prob, iaa.Add((-25, 25), per_channel=0.3)),
        #                             iaa.Sometimes(0.3 * prob, iaa.Invert(0.2, per_channel=True)),
        #                             iaa.Sometimes(0.5 * prob, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
        #                             iaa.Sometimes(0.5 * prob, iaa.Multiply((0.6, 1.4))),
        #                             iaa.Sometimes(0.5 * prob, iaa.LinearContrast((0.5, 2.2), per_channel=0.3))
        #                             ], random_order = False)
        seq_syn = iaa.Sequential([        
                                    iaa.Sometimes(0.5 * prob, iaa.CoarseDropout( p=0.2, size_percent=0.05) ),
                                    ], random_order = False)
        img_array = seq_syn.augment_image(img_array)

    image = Image.fromarray(img_array)
    image = image.resize((size, size), resample=interpolation)
    img_array = np.array(image).astype(np.uint8)
    return img_array

def create_webdataset(dataset_paths, size, shuffle_buffer, augment=False):
    dataset = wds.WebDataset(dataset_paths, shardshuffle=True) \
        .decode("pil") \
        .shuffle(shuffle_buffer) \
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

def generate_rotation_matrices(rotation_vector, device):
    x_rotations = int(rotation_vector[0])
    y_rotations = int(rotation_vector[1])
    z_rotations = int(rotation_vector[2])
    
    rotation_matrices = []
    
    # Helper function to create a rotation matrix for a given axis and angle
    def rotation_matrix(axis, angle):
        if axis == 'x':
            return np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
        elif axis == 'y':
            return np.array([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ])
        elif axis == 'z':
            return np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1]
            ])
    
    # Generate rotation matrices for x-axis
    if x_rotations > 1:
        for i in range(x_rotations):
            angle_x = 2 * np.pi * i / x_rotations  # Incremental angle
            rotation_matrices.append(torch.tensor(rotation_matrix('x', angle_x)).to(device))
    
    # Generate rotation matrices for y-axis
    if y_rotations > 1:
        for i in range(y_rotations):
            angle_y = 2 * np.pi * i / y_rotations  # Incremental angle
            rotation_matrices.append(torch.tensor(rotation_matrix('y', angle_y)).to(device))
    
    # Generate rotation matrices for z-axis
    if z_rotations > 1:
        for i in range(z_rotations):
            angle_z = 2 * np.pi * i / z_rotations  # Incremental angle
            rotation_matrices.append(torch.tensor(rotation_matrix('z', angle_z)).to(device))
    
    return rotation_matrices

def rotate_tensor(tensor, rot_matrices):
    """Rotate 3D tensors based on the provided angles around the y-axis."""
    channels, height, width = tensor.shape

    # Flatten tensor for matrix multiplication
    tensor_flat = tensor.view(channels, -1).T  # Shape: [batch_size, height*width, channels]

    rotated_tensors = []
    for rot_matrix in rot_matrices:

        # Ensure rotation matrix is of shape [channels, channels]
        assert rot_matrix.shape == (channels, channels), "Rotation matrix must be of shape [3, 3]"

        # Apply rotation
        rotated = torch.mm(tensor_flat, rot_matrix.float())  # Matrix multiplication
        rotated = rotated.T.view(channels, height, width)  # Reshape to original shape
        rotated = torch.clamp(rotated, min=-1.0, max=1.0)

        # Create a mask where any dimension is exactly -1 or 1
        mask = (rotated.abs() == 1).any(dim=0, keepdim=True)  # Shape: [1, height, width]
        mask = mask.expand(channels, -1, -1)  # Expand to match the shape of [channels, height, width]

        # Set all points with -1 or 1 in any dimension to -1
        rotated[mask] = -1

        rotated_tensors.append(rotated)

    return torch.stack(rotated_tensors, dim=0)  # Shape: [batch_size, num_angles, channels, height, width]

def main():
    max_epochs = 300
    batch_size = 16
    num_workers = 8
    shuffle_buffer = 1000

    train_data_root = "/ssd3/datasets_bop/housecat6d_nocs_train_with_info/scene{01..34}.tar"
    val_data_root = "/ssd3/datasets_bop/housecat6d_nocs_val_with_info/val_scene{1..2}.tar"

    size = 224

    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    setup_environment(sys.argv[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    symmetry_path = "/ssd3/datasets_bop/housecat6d/obj_models/all_categories_symmetries.json"

    # Check the type of the loaded object
    with open(symmetry_path, 'r') as f:
        symmetries = json.load(f)

    print(list(symmetries.values()))
    symmetries = list(symmetries.values())
    sym_list = []
    for sym in symmetries:
        symmetry_Rs = generate_rotation_matrices(sym, device=device)
        sym_list.append(symmetry_Rs)

    weight_dir = "./weights_mse"
    if not(os.path.exists(weight_dir)):
            os.makedirs(weight_dir)

    val_img_dir = "./val_img_mse"
    if not(os.path.exists(val_img_dir)):
        os.makedirs(val_img_dir)

    generator = ae(input_resolution=224)
    generator.to(device)

    mse_loss = torch.nn.MSELoss()

    #optimizer_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon)
    optimizer_generator = optim.Adam(generator.parameters(), weight_decay=0.00004, lr=lr)
    #scheduler_generator = optim.lr_scheduler.CosineAnnealingLR(optimizer_generator, max_epochs, eta_min=1e-7)

    epoch = 0
    iteration = 0

    loss_log = []

    for epoch in range(max_epochs):
        train_dataset = create_webdataset(train_data_root, size, shuffle_buffer, augment=False)
        val_dataset = create_webdataset(val_data_root, size, shuffle_buffer, augment=False)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=custom_collate_fn)

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            collate_fn=custom_collate_fn)

        start_time_epoch = time.time()
        running_loss = 0.0 

        generator.train()
        for step, batch in enumerate(train_dataloader):
            start_time_iteration = time.time()

            rgb_images = batch['rgb']
            mask_images = batch['mask']
            nocs_images = batch['nocs']
            infos = batch['info']

            mask_images = mask_images.float() / 255.0

            rgb_images_segmented = rgb_images.float() * mask_images
            rgb_images_segmented = torch.clamp(rgb_images_segmented, min=0.0, max=255.0)

            rgb_images_segmented = (rgb_images_segmented.float() / 127.5) - 1

            # print("nocs_images: ", torch.min(nocs_images))
            # print("nocs_images: ", torch.max(nocs_images))
            nocs_images_normalized = (nocs_images.float() / 127.5) - 1

            rgb_images_segmented = rgb_images_segmented.permute(0, 3, 1, 2)
            nocs_images_normalized = nocs_images_normalized.permute(0, 3, 1, 2)

            rgb_images_segmented_gt = rgb_images_segmented.to(device)
            nocs_images_normalized_gt = nocs_images_normalized.to(device)

            nocs_images_estimated = generator(rgb_images_segmented_gt)
            #nocs_images_estimated = (nocs_images_estimated + 1) / 2

            # print("nocs_images_estimated: ", torch.min(nocs_images_estimated))
            # print("nocs_images_estimated: ", torch.max(nocs_images_estimated))

            # print("nocs_images_normalized_gt: ", torch.min(nocs_images_normalized_gt))
            # print("nocs_images_normalized_gt: ", torch.max(nocs_images_normalized_gt))

            # mask_weight = mask_images.permute(0, 3, 1, 2).to(device)
            # mask_weight = torch.where(mask_weight == 1, 1.0, 0.1)

            #loss = mse_loss(nocs_images_estimated, nocs_images_normalized_gt)
            #loss = loss * mask_weight

            # Sum the losses for each channel and take the mean

            losses = []

            for idx, info in enumerate(infos):
                category_id = info['category_id']
                sym = sym_list[category_id - 1]
                nocs = nocs_images_normalized_gt[0]
                
                min_loss = mse_loss(nocs_images_estimated[idx], nocs_images_normalized_gt[idx])

                if len(sym) > 0:
                    nocs_rot = rotate_tensor(nocs, sym)

                    for Rs, nocs_rotated in zip(sym, nocs_rot):
                        loss = mse_loss(nocs_images_estimated[idx], nocs_rotated)

                        if loss < min_loss:
                            min_loss = loss
                
                losses.append(min_loss)
            
            losses_tensor = torch.stack(losses) # Ensure the tensor is on the same device as your model
            loss = losses_tensor.mean()            

            optimizer_generator.zero_grad()
            loss.backward()
            optimizer_generator.step()
            elapsed_time_iteration = time.time() - start_time_iteration  # Calculate elapsed time for the current iteration

            running_loss += loss.item()  # Accumulate loss
            iteration += 1

            # Print average loss every 100 iterations
            if (step + 1) % 100 == 0:
                avg_loss = running_loss / 100
                elapsed_time_iteration = time.time() - start_time_iteration  # Calculate elapsed time for the last 100 iterations
                lr_current = optimizer_generator.param_groups[0]['lr']
                print("Epoch {:02d}, Iteration {:03d}, Average Recon Loss: {:.4f}, lr_gen: {:.6f}, Time per 100 Iterations: {:.4f} seconds".format(
                    epoch, iteration, avg_loss, lr_current, elapsed_time_iteration))

                # Log to JSON
                loss_log.append({
                    "epoch": epoch,
                    "iteration": iteration,
                    "average_loss": avg_loss,
                    "learning_rate": lr_current,
                    "time_per_100_iterations": elapsed_time_iteration
                })

                running_loss = 0.0  # Reset running loss for the next span

                f,ax = plt.subplots(10,5,figsize=(10,20))
                imgfn = val_img_dir + "/{:03d}_{:03d}.jpg".format(epoch, iteration)
                for i in range(10):
                    ax[i,0].imshow( rgb_images[i] )
                    ax[i,1].imshow( ( (rgb_images_segmented_gt[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0)  )
                    ax[i,2].imshow( mask_images[i] )
                    ax[i,3].imshow(  nocs_images[i] )
                    ax[i,4].imshow( ( (nocs_images_estimated[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0) )
                plt.savefig(imgfn, dpi=300)
                plt.close()

        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the whole epoch: {:.4f} seconds".format(elapsed_time_epoch))

        imgfn = val_img_dir + "/val_{:03d}.jpg".format(epoch)

        generator.eval()
        running_loss = 0.0 
        val_iter = 0
        start_time_epoch = time.time()
        for step, batch in enumerate(val_dataloader):
            rgb_images = batch['rgb']
            mask_images = batch['mask']
            nocs_images = batch['nocs']
            info = batch['info']

            mask_images = mask_images.float() / 255.0
            rgb_images_segmented = rgb_images.float() * mask_images
            rgb_images_segmented = torch.clamp(rgb_images_segmented, min=0.0, max=255.0)
            rgb_images_segmented = (rgb_images_segmented / 255)
            nocs_images = (nocs_images.float() / 255)
            rgb_images_segmented = rgb_images_segmented.permute(0, 3, 1, 2)
            nocs_images = nocs_images.permute(0, 3, 1, 2)
            rgb_images_segmented_gt = rgb_images_segmented.to(device)
            nocs_images_gt = nocs_images.to(device)
            nocs_images_estimated = generator(rgb_images_segmented_gt)

            mask_weight = mask_images.permute(0, 3, 1, 2).to(device)
            mask_weight = torch.where(mask_weight == 1, 1.0, 0.1)

            loss = mse_loss(nocs_images_estimated, nocs_images_gt)

            loss = loss * mask_weight
            loss = loss.mean()

            running_loss += loss.item()
            val_iter+=1

        avg_val_loss = running_loss / val_iter
        loss_log.append({
            "epoch": epoch,
            "average_val_loss": avg_val_loss,
        })

        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the validation: {:.4f} seconds".format(elapsed_time_epoch))
        print("Val Loss: {:.4f}".format(avg_val_loss))

        f,ax = plt.subplots(10,5,figsize=(10,20))

        for i in range(10):
            ax[i,0].imshow( rgb_images[i] )
            ax[i,1].imshow( ( (rgb_images_segmented_gt[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0)  )
            ax[i,2].imshow( mask_images[i] )
            ax[i,3].imshow( ( nocs_images_gt[i]).detach().cpu().numpy().transpose(1, 2, 0) )
            ax[i,4].imshow( ( (nocs_images_estimated[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0) )
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