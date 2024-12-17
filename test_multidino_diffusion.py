import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

from utils import WebDatasetWrapper, preprocess, normalize_quaternion, setup_environment, \
                    create_webdataset_test, custom_collate_fn_test, make_log_dirs, plot_progress_imgs, \
                    preload_pointclouds, plot_single_image, COCODataset, \
                    collate_fn, restore_original_bbox_crop, overlay_nocs_on_rgb,\
                    paste_mask_on_black_canvas, paste_nocs_on_black_canvas, teaserpp_solve, \
                    backproject, sample_point_cloud, create_open3d_point_cloud, load_config, parse_args, \
                    create_line_set, show_pointcloud
                    
from diffusion_model import DiffusionNOCSDino, DiffusionNOCSDinoBART

def main(config):
    setup_environment(str(config.gpu_id))
    make_log_dirs(config.weight_dir, config.val_img_dir)

    camera_intrinsics = {
        'fx': config.fx,
        'fy': config.fy,
        'cx': config.cx,
        'cy': config.cy,
        'width': config.width,
        'height': config.height
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = DiffusionNOCSDinoBART(input_nc = 6, output_nc = 3, image_size=128, num_training_steps=1000, num_inference_steps=100)
    
    model_path = os.path.join(config.weight_dir, 'generator_epoch_15.pth')
    state_dict = torch.load(model_path, map_location=device)

    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval() 

    dataset = COCODataset(config.coco_json_path, config.test_images_root, image_size=config.size, augment=False)
    test_dataloader = DataLoader(dataset, batch_size=config.test_batch_size, shuffle=True, collate_fn=collate_fn)
    print(len(test_dataloader))

    results = []

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            print("Step: ", step)
            # unwrap the batch
            rgb_images = batch['rgb']
            depth_images = batch['depth']
            mask_full_images = batch['mask']
            rgb_cropped =  batch['rgb_crop']
            mask_images = batch['mask_crop']
            bboxes = batch['bbox'][0].cpu().numpy()
            metadata = batch['metadata'][0]
            category_name = batch['category_name']
            category_id = batch['category_id']
            gts = batch['gts']
            print(category_name)
            print(gts)

            print(metadata)

            rgb_np = rgb_images.squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to HWC
            rgb_cropped_np = rgb_cropped.squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to HWC
            mask_np = mask_images.squeeze().cpu().numpy()  # No need to permute, just remove batch dimension

            # Normalize mask to be binary (0 or 1)
            mask_images = mask_images.unsqueeze(1)
            binary_mask = (mask_images > 0).float()  # Converts mask to 0 or 1
            binary_mask = binary_mask.to(device)  # Make sure mask has same shape

            mask_full_images = mask_full_images.unsqueeze(1)
            binary_mask_full_images = (mask_full_images > 0).float()  # Converts mask to 0 or 1
            binary_mask_full_images = binary_mask_full_images.to(device)  # Make sure mask has same shape

            # RGB processing
            rgb_images = torch.clamp(rgb_cropped.float(), min=0.0, max=255.0)
            rgb_images = rgb_images.to(device)
            rgb_images = rgb_images * binary_mask

            rgb_images_gt = (rgb_images.float() / 127.5) - 1

            # MASK processing
            mask_images_gt = mask_images.float() / 255.0
            mask_images_gt = mask_images_gt.permute(0, 3, 1, 2)
            mask_images_gt = mask_images_gt.to(device)

            # # NOCS processing
            # nocs_images_normalized = (nocs_images.float() / 127.5) - 1
            # nocs_images_normalized = nocs_images_normalized.permute(0, 3, 1, 2)
            # nocs_images_normalized_gt = nocs_images_normalized.to(device)

            # # ROTATION processing
            # rotations = [entry["rotation"] for entry in infos]
            # translations = [entry["translation"] for entry in infos]
            # large_bboxes = [entry["large_bbox"] for entry in infos]
            # category_names = [entry["category_name"] for entry in infos]
            # category_ids = [entry["category_id"] for entry in infos]

            # print(large_bboxes)

            # meshes_path =  '/ssd3/datasets_bop/housecat6d/obj_models/obj_models_small_size_final'
            # mesh_folders = sorted(os.listdir(meshes_path))
            # mesh_folder = mesh_folders[category_ids[0] - 1]

            # mesh_path = meshes_path + "/" + mesh_folder + "/" + category_names[0] + ".obj"
            # mesh = trimesh.load(mesh_path, force='mesh')
            # canonical_bbox = mesh.bounding_box_oriented.vertices
            # ground_truth_bbox = np.dot(rotations[0], canonical_bbox.T).T + translations[0]
    
            # forward pass through generator
            nocs_estimated = generator.inference(rgb_images_gt, category_name)
            #nocs_estimated = ((nocs_estimated + 1 ) /2)

            # x_bins = torch.softmax(x_logits, dim=1)  # Softmax over the bin dimension
            # y_bins = torch.softmax(y_logits, dim=1)
            # z_bins = torch.softmax(z_logits, dim=1)

            # # Bin centers (shared for x, y, z dimensions)
            # bin_centers = torch.linspace(-1, 1, config.num_bins).to(x_logits.device)  # Bin centers

            # # Compute the estimated NOCS map for each dimension by multiplying with bin centers and summing over bins
            # nocs_x_estimated = torch.sum(x_bins * bin_centers.view(1, config.num_bins, 1, 1), dim=1)
            # nocs_y_estimated = torch.sum(y_bins * bin_centers.view(1, config.num_bins, 1, 1), dim=1)
            # nocs_z_estimated = torch.sum(z_bins * bin_centers.view(1, config.num_bins, 1, 1), dim=1)

            # # Combine the estimated NOCS map from x, y, and z dimensions
            # nocs_estimated = torch.stack([nocs_x_estimated, nocs_y_estimated, nocs_z_estimated], dim=1)

            #plot_nocs_image((nocs_estimated + 1 ) /2, min_neighbors=5, neighborhood_size=0.05, eps=0.05, min_samples=10)

            #nocs_estimated = nocs_estimated * binary_mask
            nocs_estimated = ((nocs_estimated + 1 ) / 2)

            nocs_estimated_np = (nocs_estimated).squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to HWC

            nocs_estimated_resized = restore_original_bbox_crop((nocs_estimated_np * 255).astype(np.uint8), metadata)
            mask_resized = restore_original_bbox_crop((mask_np * 255).astype(np.uint8), metadata)

            rgb_nocs_overlay_image = overlay_nocs_on_rgb((rgb_np * 255).astype(np.uint8), nocs_estimated_resized, (mask_resized).astype(np.uint8), bboxes)

            binary_mask = (mask_resized > 0).astype(np.uint8)
            binary_mask_full_images = (binary_mask_full_images.squeeze(0).squeeze(0).cpu().numpy() > 0).astype(np.uint8)
            nocs_estimated_resized_masked = nocs_estimated_resized.copy()
            nocs_estimated_resized_masked[binary_mask == 0] = 0

            depth_images = depth_images.squeeze(0).squeeze(0).cpu().numpy()

            depth_images[binary_mask_full_images == 0] = 0

            rgb_images_gt_np = (((rgb_images_gt + 1) /2)[0].permute(1, 2, 0).cpu().numpy() * 255 ).astype(np.uint8)

            print(rgb_np.shape)

            mask_full_np = paste_mask_on_black_canvas((rgb_np * 255).astype(np.uint8), (mask_resized).astype(np.uint8), bboxes)
            nocs_full_np = paste_nocs_on_black_canvas((rgb_np * 255).astype(np.uint8), (nocs_estimated_resized_masked).astype(np.uint8), bboxes)

            # Plotting
            fig, axs = plt.subplots(1, 8, figsize=(15, 5))
            
            # Plot RGB image
            axs[0].imshow(rgb_np)
            axs[0].set_title("RGB Image")
            axs[0].axis('off')

            # Plot cropped RGB image
            axs[1].imshow(rgb_cropped_np)
            axs[1].set_title("Cropped RGB Image")
            axs[1].axis('off')

            # Plot mask
            axs[2].imshow(mask_np)
            axs[2].set_title("Mask")
            axs[2].axis('off')

            # Plot mask
            axs[3].imshow(nocs_full_np)
            axs[3].set_title("NOCS Estimated")
            axs[3].axis('off')

            # Plot mask
            axs[4].imshow(nocs_estimated_resized_masked)
            axs[4].set_title("NOCS Resized")
            axs[4].axis('off')

            # Plot mask
            axs[5].imshow(mask_full_np)
            axs[5].set_title("Mask Resized")
            axs[5].axis('off')

            # Plot mask
            axs[6].imshow(rgb_nocs_overlay_image)
            axs[6].set_title("Overlay")
            axs[6].axis('off')

            # Plot mask
            axs[7].imshow(depth_images)
            axs[7].set_title("depth cropped")
            axs[7].axis('off')

            plt.show()

            plot_single_image(config.test_img_dir + "/rgb_images_gt", step, rgb_np, False)
            plot_single_image(config.test_img_dir + "/rgb_cropped_np", step, rgb_cropped_np, False)
            plot_single_image(config.test_img_dir + "/rgb_images_gt_np", step, rgb_images_gt_np, False)
            plot_single_image(config.test_img_dir + "/mask_np", step, mask_np, False)
            plot_single_image(config.test_img_dir + "/nocs_estimated", step, nocs_estimated_np, False)
            plot_single_image(config.test_img_dir + "/overlay", step, rgb_nocs_overlay_image, False)

            print("nocs_estimated: ", nocs_estimated.shape)
            print("depth_images: ", depth_images.shape)

            nocs_estimated_resized_masked = nocs_estimated_resized_masked.astype(np.float32) / 255

            #depth_images = depth_images / 100

            #dst, idxs = backproject(depth_images, camera_intrinsics, mask_full_np)
            dst, idxs = backproject(depth_images, camera_intrinsics, mask_full_np)
            dst = dst.T
            print(dst.shape)
            show_pointcloud(dst.T)

            #dst = generate_point_cloud_from_depth(depth_images, camera_intrinsics, bboxes).T
            #src = generate_point_cloud_from_nocs(nocs_estimated_resized_masked).T
            nocs_full_np = (nocs_full_np.astype(float) / 127.5) - 1
            src = nocs_full_np[idxs[0], idxs[1], :].T

            print(src)

            print("src: ", src.shape)
            print("dst: ", dst.shape)

            #show_pointcloud(dst.T)
            show_pointcloud(src.T)

            #src, dst, _ = filter_zero_points(src, dst)
            #dst, src, _ = filter_zero_points(dst, src)

            #src, dst, _ = filter_points_within_limits(src, dst, center=0.5, tolerance=0.05)

            num_points_to_sample = config.num_points_to_sample
            if num_points_to_sample > src.shape[1]:
                src_sampled, dst_sampled = src, dst
            else:
                src_sampled, dst_sampled = sample_point_cloud(src, dst, num_points_to_sample)

            print("src_sampled: ", src_sampled.shape)
            print("dst_sampled: ", dst_sampled.shape)

            pcd_src = create_open3d_point_cloud(src, [0, 0, 1])  # Blue for source
            pcd_dst = create_open3d_point_cloud(dst, [1, 0, 0])  # Red for destination

            # show_pointcloud(src_sampled.T)
            # show_pointcloud(dst_sampled.T)

            R, t, s = teaserpp_solve(src_sampled, dst_sampled)

            # Apply the estimated transformation to the source point cloud
            src_transformed = s * np.matmul(R, src) + t.reshape(3, 1)

            # Create Open3D point clouds for visualization
            #src_transformed = ransac_registration(src_sampled, dst_sampled)
            pcd_src_transformed = create_open3d_point_cloud(src_transformed, [0, 1, 0])  # Green

            # Create line set to visualize correspondences
            line_set = create_line_set(src_sampled, dst_sampled, [0, 1, 0])  # Green lines for correspondences
        
            # Visualize the point clouds using Open3D
            # o3d.visualization.draw_geometries([pcd_src, pcd_dst, pcd_src_transformed, line_set],
            #                                 window_name="TEASER++ Registration with PLY and NOCS",
            #                                 width=1920, height=1080,
            #                                 left=50, top=50,
            #                                 point_show_normal=False)
            
            o3d.visualization.draw_geometries([pcd_src, pcd_dst, pcd_src_transformed],
                                            window_name="TEASER++ Registration with PLY and NOCS",
                                            width=1920, height=1080,
                                            left=50, top=50,
                                            point_show_normal=False)

if __name__ == "__main__":
    args = parse_args()
    
    # Load the config file passed as argument
    config = load_config(args.config)
    
    # Call main with the loaded config
    main(config)