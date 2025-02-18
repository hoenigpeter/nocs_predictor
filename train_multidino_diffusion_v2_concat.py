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
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import json
import webdataset as wds

from utils import WebDatasetWrapper, preprocess, normalize_quaternion, setup_environment, \
                    create_webdataset, custom_collate_fn, make_log_dirs, plot_progress_imgs, \
                    plot_single_image, apply_rotation, parse_args, load_config, plot_progress_imgs_v2, \
                    add_loss, create_webdataset_v2, scope_collate_fn

from diffusion_model import DiffusionNOCS, SCOPE

# Create a lookup function
def lookup(category_id, objects):
    obj_dict = {obj["id"]: obj["name"] for obj in objects}
    return obj_dict.get(category_id, "Unknown")  # Default to "Unknown" if ID is not found

def main(config):

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running basic DDP example on rank {rank}.")
    # create model and move it to GPU with id rank
    device = rank % torch.cuda.device_count()
    print("device_id: ", device)

    objects = [
        {"id": 0, "name": "bottle"},
        {"id": 1, "name": "bowl"},
        {"id": 2, "name": "camera"},
        {"id": 3, "name": "can"},
        {"id": 4, "name": "laptop"},
        {"id": 5, "name": "mug"}
    ]

    setup_environment(str(config.gpu_id))
    make_log_dirs([config.weight_dir, config.val_img_dir])

    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model instantiation and compilation
    if config.with_dino_concat == True:
        input_nc = 15
    else:
        input_nc = 9

    model = SCOPE(
                        input_nc = input_nc,
                        output_nc = 3,
                        with_dino_feat=config.with_dino_feat,
                        with_bart_feat=config.with_bart_feat,
                        cls_embedding=config.with_cls_embedding,
                        num_class_embeds=config.num_categories,
                        image_size=config.image_size,
                        num_training_steps=config.num_training_steps,
                        num_inference_steps=config.num_inference_steps
                    )

    model.to(device)
    generator = DDP(model, device_ids=[device])

    print(generator)

    # Optimizer instantiation
    optimizer_generator = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.epsilon)

    # Instantiate train and val dataset + dataloaders
    train_dataset = create_webdataset_v2(
                        dataset_paths=config.train_data_root,
                        size=config.image_size,
                        shuffle_buffer=config.shuffle_buffer,
                        dino_mode=config.dino_mode,
                        class_name=None
                    )

    val_dataset = create_webdataset_v2(
                        dataset_paths=config.val_data_root,
                        size=config.image_size,
                        shuffle_buffer=config.shuffle_buffer,
                        dino_mode=config.dino_mode,
                        class_name=None
                    )

    train_dataloader = wds.WebLoader(
                        train_dataset,
                        batch_size=config.batch_size,
                        shuffle=False,
                        num_workers=config.train_num_workers,
                        drop_last=True,
                        collate_fn=scope_collate_fn,
                    )
    val_dataloader = wds.WebLoader(
                        val_dataset,
                        batch_size=config.batch_size,
                        shuffle=False,
                        num_workers=config.val_num_workers,
                        drop_last=True,
                        collate_fn=scope_collate_fn,
                    )

    # Training Loop
    epoch = 0
    iteration = 0
    loss_log = []

    for epoch in range(config.max_epochs):

        start_time_epoch = time.time()
        running_loss = 0.0 

        generator.train()

        # Shuffle before epoch
        train_dataloader.unbatched().shuffle(1000).batched(config.batch_size)
        val_dataloader.unbatched().shuffle(1000).batched(config.batch_size)

        for step, batch in enumerate(train_dataloader):
            start_time_iteration = time.time()

            if iteration < config.warmup_steps:
                lr = config.lr * (iteration / config.warmup_steps)
            else:
                lr = config.lr
            
            # Update the optimizer learning rate
            for param_group in optimizer_generator.param_groups:
                param_group['lr'] = lr

            # unwrap the batch
            rgb_images = batch['rgb']
            nocs_images = batch['nocs']

            if config.noisy_normals == True:
                normal_images = batch['normals_with_aug']
            else:
                normal_images = batch['normals_no_aug']

            mask_images = batch['mask']
            dino_images = batch['dino_pca']
            infos = batch['info']

            obj_names = []
            obj_cats = []

            for entry in infos:
                obj_cat = int(entry["category_id"])
                obj_cats.append(obj_cat)
                obj_name = lookup(obj_cat, objects)
                obj_names.append(obj_name)

            obj_cats_tensor = torch.tensor(obj_cats, dtype=torch.int).to(device)

            # Normalize mask to be binary (0 or 1)
            binary_mask = (mask_images > 0).float()  # Converts mask to 0 or 1
            binary_mask = binary_mask.permute(0, 3, 1, 2).to(device)  # Make sure mask has same shape
            
            # RGB processing
            rgb_images = torch.clamp(rgb_images.float(), min=0.0, max=255.0)
            rgb_images = rgb_images.permute(0, 3, 1, 2)
            rgb_images = rgb_images.to(device)
            rgb_images = rgb_images * binary_mask
            rgb_images_gt = (rgb_images.float() / 127.5) - 1

            # Normals processing
            normal_images = torch.clamp(normal_images.float(), min=0.0, max=255.0)
            normal_images = normal_images.permute(0, 3, 1, 2)
            normal_images = normal_images.to(device)
            normal_images = normal_images * binary_mask
            normal_images_gt = (normal_images.float() / 127.5) - 1

            # DINO images processing
            dino_images = torch.clamp(dino_images.float(), min=0.0, max=255.0)
            dino_images = dino_images.permute(0, 3, 1, 2)
            dino_images = dino_images.to(device)
            #normal_images = normal_images * binary_mask
            dino_images_gt = (dino_images.float() / 127.5) - 1

            # MASK processing
            mask_images_gt = mask_images.float() / 255.0
            mask_images_gt = mask_images_gt.permute(0, 3, 1, 2)
            mask_images_gt = mask_images_gt.to(device)

            # NOCS processing
            nocs_images_float = nocs_images.float()
            nocs_images_normalized = (nocs_images_float / 127.5) - 1
            nocs_images_normalized = nocs_images_normalized.permute(0, 3, 1, 2)
            nocs_images_normalized_gt = nocs_images_normalized.to(device)

            # # Plot the first image with the first three channels as one RGB image
            # plt.figure(figsize=(6, 6))
            # plt.imshow(((dino_images_gt+1)/2)[:, :3, :, :][0].permute(1, 2, 0).cpu().numpy())
            # plt.axis('off')
            # plt.title("RGB Image (First Three Channels)")
            # plt.show()

            # forward pass through generator
            if config.with_cls_embedding == False:
                embeddings = generator.get_embeddings(rgb_images_gt, obj_names)
            else:
                embeddings = None

            if config.with_dino_concat == True:
                inputs = torch.cat([rgb_images_gt, normal_images_gt, dino_images_gt], dim=1)
            else:
                inputs = torch.cat([rgb_images_gt, normal_images_gt], dim=1)

            loss = generator(inputs, nocs_images_normalized_gt, embeddings, obj_cats_tensor)
            
            # Loss backpropagation
            optimizer_generator.zero_grad()
            loss.backward()

            # Optimizer gradient update
            optimizer_generator.step()
            elapsed_time_iteration = time.time() - start_time_iteration

            running_loss += loss.item()
            iteration += 1

            if (step + 1) % config.iter_cnt == 0:
                avg_loss = running_loss / config.iter_cnt

                elapsed_time_iteration = time.time() - start_time_iteration
                lr_current = optimizer_generator.param_groups[0]['lr']
                print("Epoch {:02d}, Iter {:03d}, Loss: {:.4f}, lr_gen: {:.6f}, Time: {:.4f} seconds".format(
                    epoch, step, avg_loss, lr_current, elapsed_time_iteration))

                # Log to JSON
                loss_log.append({
                    "epoch": epoch,
                    "iteration": iteration,
                    "regression_nocs_loss": avg_loss,
                    "learning_rate": lr_current,
                    "time_per_100_iterations": elapsed_time_iteration
                })

                running_loss = 0

                if config.with_cls_embedding == False:
                    embeddings = generator.get_embeddings(rgb_images_gt, obj_names)
                else:
                    embeddings = None

                if config.with_dino_concat == True:
                    inputs = torch.cat([rgb_images_gt, normal_images_gt, dino_images_gt], dim=1)
                else:
                    inputs = torch.cat([rgb_images_gt, normal_images_gt], dim=1)

                with torch.no_grad():
                    nocs_estimated = generator.module.inference(inputs, embeddings, obj_cats_tensor)

                imgfn = config.val_img_dir + "/{:03d}_{:03d}.jpg".format(epoch, iteration)
                plot_progress_imgs_v2(imgfn, rgb_images_gt, normal_images_gt, nocs_images_normalized_gt, nocs_estimated, mask_images_gt, dino_images_gt[:, :3, :, :], config)
        
                break

        elapsed_time_epoch = time.time() - start_time_epoch
        print("Time for the whole epoch: {:.4f} seconds".format(elapsed_time_epoch))

        generator.eval()
        running_loss = 0.0

        val_iter = 0
        start_time_epoch = time.time()

        # with torch.no_grad():
        #     for step, batch in enumerate(val_dataloader):
        #         start_time_iteration = time.time()

        #         # unwrap the batch
        #         rgb_images = batch['rgb']
        #         nocs_images = batch['nocs']

        #         if config.noisy_normals == True:
        #             normal_images = batch['normals_with_aug']
        #         else:
        #             normal_images = batch['normals_no_aug']

        #         mask_images = batch['mask']
        #         dino_images = batch['dino_pca']
        #         infos = batch['info']

        #         obj_names = []
        #         obj_cats = []

        #         for entry in infos:
        #             obj_cat = int(entry["category_id"])
        #             obj_cats.append(obj_cat)
        #             obj_name = lookup(obj_cat, objects)
        #             obj_names.append(obj_name)

        #         obj_cats_tensor = torch.tensor(obj_cats, dtype=torch.int).to(device)

        #         # Normalize mask to be binary (0 or 1)
        #         binary_mask = (mask_images > 0).float()  # Converts mask to 0 or 1
        #         binary_mask = binary_mask.permute(0, 3, 1, 2).to(device)  # Make sure mask has same shape
                
        #         # RGB processing
        #         rgb_images = torch.clamp(rgb_images.float(), min=0.0, max=255.0)
        #         rgb_images = rgb_images.permute(0, 3, 1, 2)
        #         rgb_images = rgb_images.to(device)
        #         rgb_images = rgb_images * binary_mask
        #         rgb_images_gt = (rgb_images.float() / 127.5) - 1

        #         # Normals processing
        #         normal_images = torch.clamp(normal_images.float(), min=0.0, max=255.0)
        #         normal_images = normal_images.permute(0, 3, 1, 2)
        #         normal_images = normal_images.to(device)
        #         normal_images = normal_images * binary_mask
        #         normal_images_gt = (normal_images.float() / 127.5) - 1

        #         # DINO images processing
        #         dino_images = torch.clamp(dino_images.float(), min=0.0, max=255.0)
        #         dino_images = dino_images.permute(0, 3, 1, 2)
        #         dino_images = dino_images.to(device)
        #         #normal_images = normal_images * binary_mask
        #         dino_images_gt = (dino_images.float() / 127.5) - 1

        #         # MASK processing
        #         mask_images_gt = mask_images.float() / 255.0
        #         mask_images_gt = mask_images_gt.permute(0, 3, 1, 2)
        #         mask_images_gt = mask_images_gt.to(device)

        #         # NOCS processing
        #         nocs_images_float = nocs_images.float()
        #         nocs_images_normalized = (nocs_images_float / 127.5) - 1
        #         nocs_images_normalized = nocs_images_normalized.permute(0, 3, 1, 2)
        #         nocs_images_normalized_gt = nocs_images_normalized.to(device)

        #         if config.with_cls_embedding == False:
        #             embeddings = generator.get_embeddings(rgb_images_gt, obj_names)
        #         else:
        #             embeddings = None

        #         if config.with_dino_concat == True:
        #             inputs = torch.cat([rgb_images_gt, normal_images_gt, dino_images_gt], dim=1)
        #         else:
        #             inputs = torch.cat([rgb_images_gt, normal_images_gt], dim=1)

        #         loss = generator(inputs, nocs_images_normalized_gt, embeddings, obj_cats_tensor)
                
        #         elapsed_time_iteration = time.time() - start_time_iteration

        #         running_loss += loss.item()

        #         val_iter+=1

        # avg_loss = running_loss / val_iter

        # loss_log.append({
        #     "epoch": epoch,
        #     "val_regression_nocs_loss": avg_loss,
        #     "val_learning_rate": lr_current,
        #     "val_time_per_100_iterations": elapsed_time_iteration
        # })
        
        # elapsed_time_epoch = time.time() - start_time_epoch
        # print("Time for the validation: {:.4f} seconds".format(elapsed_time_epoch))
        # print("Val Loss: {:.4f}".format(avg_loss))

        if config.with_cls_embedding == False:
            embeddings = generator.get_embeddings(rgb_images_gt, obj_names)
        else:
            embeddings = None

        if config.with_dino_concat == True:
            inputs = torch.cat([rgb_images_gt, normal_images_gt, dino_images_gt], dim=1)
        else:
            inputs = torch.cat([rgb_images_gt, normal_images_gt], dim=1)

        with torch.no_grad():
            nocs_estimated = generator.module.inference(inputs, embeddings, obj_cats_tensor)

        imgfn = config.val_img_dir + "/val_{:03d}.jpg".format(epoch)
        plot_progress_imgs_v2(imgfn, rgb_images_gt, normal_images_gt, nocs_images_normalized_gt, nocs_estimated, mask_images_gt, dino_images_gt[:, :3, :, :], config)
        
        if epoch % config.save_epoch_interval == 0:
            torch.save(generator.state_dict(), os.path.join(config.weight_dir, f'generator_epoch_{epoch}.pth'))

        # Save loss log to JSON after each epoch
        with open(config.weight_dir + "/loss_log.json", "w") as f:
            json.dump(loss_log, f, indent=4)

        epoch += 1

    torch.save(generator.state_dict(), os.path.join(config.weight_dir, f'generator_epoch_{epoch}.pth'))

    dist.destroy_process_group()
    print(f"Finished running basic DDP example on rank {rank}.")

if __name__ == "__main__":
    args = parse_args()
    
    # Load the config file passed as argument
    config = load_config(args.config)
    
    # Call main with the loaded config
    main(config)