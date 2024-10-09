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

from dino_model import AutoencoderXYZHead as ae

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

    weight_dir = "./weights_cross_entropy"
    if not(os.path.exists(weight_dir)):
            os.makedirs(weight_dir)

    val_img_dir = "./val_img_cross_entropy"
    if not(os.path.exists(val_img_dir)):
        os.makedirs(val_img_dir)

    generator = ae(input_resolution=224)
    generator.to(device)

    class_weights_uvw = torch.ones(256)
    class_weights_uvw[0] = 0.01
    cross_entropy_loss = torch.nn.CrossEntropyLoss(weight=class_weights_uvw, reduction='none').to(device)

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
            info = batch['info']

            mask_images = mask_images.float() / 255.0

            rgb_images_segmented = rgb_images.float() * mask_images
            rgb_images_segmented = torch.clamp(rgb_images_segmented, min=0.0, max=255.0)

            # rgb_images_segmented = (rgb_images_segmented / 127.5) - 1.0
            # nocs_images_normalized = (nocs_images.float() / 127.5) - 1.0

            rgb_images_segmented = (rgb_images_segmented.float() / 255)
            nocs_images_normalized = (nocs_images.float() / 255)

            rgb_images_segmented = rgb_images_segmented.permute(0, 3, 1, 2)
            nocs_images_normalized = nocs_images_normalized.permute(0, 3, 1, 2)
            #mask_images = mask_images.permute(0, 3, 1, 2)

            rgb_images_segmented_gt = rgb_images_segmented.to(device)
            nocs_images_normalized_gt = nocs_images_normalized.to(device)

            x_out, y_out, z_out = generator(rgb_images_segmented_gt)

            x_channel = torch.max(x_out, dim=1)[1] 
            y_channel = torch.max(y_out, dim=1)[1] 
            z_channel = torch.max(z_out, dim=1)[1] 

            nocs_images_estimated = torch.stack([x_channel, y_channel, z_channel], dim=1)

            mask_weight = mask_images.permute(0, 3, 1, 2).to(device)
            mask_weight = torch.where(mask_weight[:, 0] == 1, 1.0, 0.1)

            nocs_images_split = batch['nocs'].permute(0, 3, 1, 2).to(device).long()

            x_channel_gt = torch.zeros(batch_size, 256, 224, 224, device=device, dtype=torch.float32)
            x_channel_gt.scatter_(1, nocs_images_split[:,0].unsqueeze(1), 1.0)

            y_channel_gt = torch.zeros(batch_size, 256, 224, 224, device=device)
            y_channel_gt.scatter_(1, nocs_images_split[:,1].unsqueeze(1), 1.0)

            z_channel_gt = torch.zeros(batch_size, 256, 224, 224, device=device)
            z_channel_gt.scatter_(1, nocs_images_split[:,2].unsqueeze(1), 1.0)

            nocs_images_estimated = torch.stack([x_channel, y_channel, z_channel], dim=1)

            r_loss = cross_entropy_loss(x_out, x_channel_gt)
            g_loss = cross_entropy_loss(y_out, y_channel_gt)
            b_loss = cross_entropy_loss(z_out, z_channel_gt)

            # r_loss = r_loss * mask_weight
            # g_loss = g_loss * mask_weight
            # b_loss = b_loss * mask_weight

            # Sum the losses for each channel and take the mean
            loss = (r_loss.sum() + g_loss.sum() + b_loss.sum()) / mask_weight.sum()

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
                    ax[i,1].imshow( ( rgb_images_segmented_gt[i]).detach().cpu().numpy().transpose(1, 2, 0)  )
                    ax[i,2].imshow( mask_images[i] )
                    ax[i,3].imshow( nocs_images[i] )
                    ax[i,4].imshow( ( nocs_images_estimated[i]).detach().cpu().numpy().transpose(1, 2, 0) )
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
            x_out, y_out, z_out = generator(rgb_images_segmented_gt)

            x_channel = torch.max(x_out, dim=1)[1] 
            y_channel = torch.max(y_out, dim=1)[1] 
            z_channel = torch.max(z_out, dim=1)[1] 

            nocs_images_estimated = torch.stack([x_channel, y_channel, z_channel], dim=1)
            mask_weight = mask_images.permute(0, 3, 1, 2).to(device)
            mask_weight = torch.where(mask_weight[:, 0] == 1, 1.0, 0.1)

            nocs_images_split = batch['nocs'].permute(0, 3, 1, 2).to(device).long()

            x_channel_gt = torch.zeros(batch_size, 256, 224, 224, device=device, dtype=torch.float32)
            x_channel_gt.scatter_(1, nocs_images_split[:,0].unsqueeze(1), 1.0)
            y_channel_gt = torch.zeros(batch_size, 256, 224, 224, device=device)
            y_channel_gt.scatter_(1, nocs_images_split[:,1].unsqueeze(1), 1.0)
            z_channel_gt = torch.zeros(batch_size, 256, 224, 224, device=device)
            z_channel_gt.scatter_(1, nocs_images_split[:,2].unsqueeze(1), 1.0)

            nocs_images_estimated = torch.stack([x_channel, y_channel, z_channel], dim=1)

            r_loss = cross_entropy_loss(x_out, x_channel_gt)
            g_loss = cross_entropy_loss(y_out, y_channel_gt)
            b_loss = cross_entropy_loss(z_out, z_channel_gt)

            r_loss = r_loss * mask_weight
            g_loss = g_loss * mask_weight
            b_loss = b_loss * mask_weight

            # Sum the losses for each channel and take the mean
            loss = (r_loss.sum() + g_loss.sum() + b_loss.sum()) / mask_weight.sum()

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
            ax[i,1].imshow( ( rgb_images_segmented_gt[i]).detach().cpu().numpy().transpose(1, 2, 0)  )
            ax[i,2].imshow( mask_images[i] )
            ax[i,3].imshow( ( nocs_images_gt[i]).detach().cpu().numpy().transpose(1, 2, 0) )
            ax[i,4].imshow( ( nocs_images_estimated[i]).detach().cpu().numpy().transpose(1, 2, 0) )
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