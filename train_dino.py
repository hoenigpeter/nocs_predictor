# code adapted from: https://github.com/kirumang/Pix2Pose

import os
import sys

ROOT_DIR = os.path.abspath(".")
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append("./bop_toolkit")

import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dino_model import Autoencoder as ae

from nocs_dataset_multi_object import NOCSTrain

def setup_environment(gpu_id):
    if len(sys.argv) != 2:
        print("Usage: python3 train.py <gpu_id>")
        sys.exit()

    if gpu_id == '-1':
        gpu_id = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
     
def main():
    max_epochs = 300
    batch_size = 16

    augmentation_prob=1.0
    imsize = 128
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8

    setup_environment(sys.argv[1])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_train = NOCSTrain(data_root = "/ssd3/datasets_bop/megapose_nocs", size=224, obj_ids=None, crop_object = True, fraction=1.0, augment=True)
    
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        drop_last=True
    )

    weight_dir = "./weights_big"
    if not(os.path.exists(weight_dir)):
            os.makedirs(weight_dir)

    val_img_dir = "./val_img_big"
    if not(os.path.exists(val_img_dir)):
        os.makedirs(val_img_dir)


    generator = ae(input_resolution=224)
    generator.to(device)

    mse_loss = torch.nn.MSELoss()

    optimizer_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon)
    scheduler_generator = optim.lr_scheduler.CosineAnnealingLR(optimizer_generator, max_epochs, eta_min=1e-7)

    generator.train()

    epoch = 0
    iteration = 0

    total_iterations = len(train_dataloader)
    print("total iterations per epoch: ", total_iterations)

    for epoch in range(max_epochs):
        start_time_epoch = time.time()
        for step, batch in enumerate(train_dataloader):
            start_time_iteration = time.time()
            rgb_images_gt = batch["rgb"].to(device)
            xyz_images_gt = batch["nocs"].to(device)

            xyz_images_estimated = generator(rgb_images_gt)
            output = mse_loss(xyz_images_estimated, xyz_images_gt)
            #loss_transformer = transformer_loss([input, target])   -> needs to be modified

            optimizer_generator.zero_grad()
            output.backward()
            optimizer_generator.step()
            elapsed_time_iteration = time.time() - start_time_iteration  # Calculate elapsed time for the current iteration

            lr_current = optimizer_generator.param_groups[0]['lr']
            print("Epoch {:02d}, Iteration {:03d}/{:03d}, Recon Loss: {:.4f}, lr_gen: {:.6f}, Time per Iteration: {:.4f} seconds".format(epoch, iteration + 1, total_iterations, output, lr_current, elapsed_time_iteration))

            iteration += 1

        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the whole epoch: {:.4f} seconds".format(elapsed_time_epoch))

        imgfn = val_img_dir + "/{:03d}.jpg".format(epoch)

        gen_images = generator(rgb_images_gt)
        f,ax = plt.subplots(10,3,figsize=(10,20))

        for i in range(10):
            ax[i,0].imshow( ( (xyz_images_gt[i]+1)/2).detach().cpu().numpy().transpose(1, 2, 0) )
            ax[i,1].imshow( ( (rgb_images_gt[i]+1)/2).detach().cpu().numpy().transpose(1, 2, 0)  )
            ax[i,2].imshow( ( (gen_images[i]+1)/2).detach().cpu().numpy().transpose(1, 2, 0) )
        plt.savefig(imgfn, dpi=300)
        plt.close()
        
        scheduler_generator.step()

        if epoch % 5 == 0:
            torch.save(generator.state_dict(), os.path.join(weight_dir, f'generator_epoch_{epoch}.pth'))
        epoch += 1
        iteration = 0

# Run the main function
if __name__ == "__main__":
    main()