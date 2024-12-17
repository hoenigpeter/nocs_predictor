import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import open3d as o3d
import numpy as np

from utils import WebDatasetWrapper, preprocess, normalize_quaternion, setup_environment, \
                    create_webdataset_test, custom_collate_fn_test, make_log_dirs, plot_progress_imgs, \
                    preload_pointclouds, plot_single_image, COCODataset, CustomDataset, \
                    collate_fn, collate_fn_val, restore_original_bbox_crop, overlay_nocs_on_rgb,\
                    paste_mask_on_black_canvas, paste_nocs_on_black_canvas, teaserpp_solve, \
                    backproject, sample_point_cloud, create_open3d_point_cloud, load_config, parse_args, \
                    create_line_set, show_pointcloud
                    

from nocs_paper_utils import compute_degree_cm_mAP
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

    dataset = CustomDataset(config.coco_json_path, config.test_images_root, image_size=config.size, augment=False)
    test_dataloader = DataLoader(dataset, batch_size=config.test_batch_size, shuffle=True, collate_fn=collate_fn_val)
    print("number of test images: ", len(test_dataloader))

    results = []

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            print("Step: ", step)

            rgb_images = batch['rgb']
            depth_images = batch['depth']
            mask_full_images = batch['mask'][0]
            rgb_crops =  batch['rgb_crops'][0]
            mask_images = batch['mask_crops'][0]
            bboxes = batch['bboxes'][0]
            metadatas = batch['metadatas'][0]
            category_names = batch['category_names'][0]
            category_ids = np.array(batch['category_ids'])
            gts = batch['gts'][0]
            gt_bboxes = gts['gt_bboxes']
            gt_scales = gts['gt_scales']
            gt_RTs = gts['gt_RTs']
            gt_categories = gts['gt_categories']

            rgb_np = rgb_images.squeeze().permute(1, 2, 0).cpu().numpy()

            depth_images = depth_images.squeeze(0).squeeze(0).cpu().numpy()
            
            pred_scales = []
            pred_RTs = []

            print(len(mask_images))
            print(len(rgb_crops))

            for idx in range(len(rgb_crops)):
                mask_np = mask_images[idx].squeeze().cpu().numpy()
                mask_image = mask_images[idx].unsqueeze(0).unsqueeze(0)

                binary_mask = (mask_image > 0).float()
                binary_mask = binary_mask.to(device)

                mask_full_image = mask_full_images[idx].unsqueeze(0)
                binary_mask_full_images = (mask_full_image > 0).float()
                binary_mask_full_images = binary_mask_full_images.to(device)
                print(binary_mask_full_images.shape)

                mask_images_gt = mask_image.float() / 255.0
                mask_images_gt = mask_images_gt.permute(0, 3, 1, 2)
                mask_images_gt = mask_images_gt.to(device)

                rgb_cropped = rgb_crops[idx]
                rgb_cropped_np = rgb_cropped.squeeze().permute(1, 2, 0).cpu().numpy()

                rgb_images = torch.clamp(rgb_cropped.float(), min=0.0, max=255.0).unsqueeze(0)
                rgb_images = rgb_images.to(device)

                print(binary_mask.shape)
                print(rgb_images.shape)
                rgb_images = rgb_images * binary_mask

                rgb_images_gt = (rgb_images.float() / 127.5) - 1

                nocs_estimated = generator.inference(rgb_images_gt, category_names[idx])

                nocs_estimated = ((nocs_estimated + 1 ) / 2)

                nocs_estimated_np = (nocs_estimated).squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to HWC

                nocs_estimated_resized = restore_original_bbox_crop((nocs_estimated_np * 255).astype(np.uint8), metadatas[idx])
                mask_resized = restore_original_bbox_crop((mask_np * 255).astype(np.uint8), metadatas[idx])

                bbox = bboxes[idx].cpu().numpy()
                print(bbox)

                rgb_nocs_overlay_image = overlay_nocs_on_rgb((rgb_np * 255).astype(np.uint8), nocs_estimated_resized, (mask_resized).astype(np.uint8), bbox)

                # AB HIER WEITER DEBUGGEN -> UMSTELLUNG NEUER DATALOADER!
                binary_mask = (mask_resized > 0).astype(np.uint8)
                binary_mask_full_images = (binary_mask_full_images.squeeze(0).squeeze(0).cpu().numpy() > 0).astype(np.uint8)
                print(binary_mask_full_images.shape)
                nocs_estimated_resized_masked = nocs_estimated_resized.copy()
                nocs_estimated_resized_masked[binary_mask == 0] = 0

                depth_images[binary_mask_full_images == 0] = 0

                rgb_images_gt_np = (((rgb_images_gt + 1) /2)[0].permute(1, 2, 0).cpu().numpy() * 255 ).astype(np.uint8)

                mask_full_np = paste_mask_on_black_canvas((rgb_np * 255).astype(np.uint8), (mask_resized).astype(np.uint8), bbox)
                nocs_full_np = paste_nocs_on_black_canvas((rgb_np * 255).astype(np.uint8), (nocs_estimated_resized_masked).astype(np.uint8), bbox)

                nocs_estimated_resized_masked = nocs_estimated_resized_masked.astype(np.float32) / 255

                dst, idxs = backproject(depth_images, camera_intrinsics, mask_full_np)
                dst = dst.T
                #show_pointcloud(dst.T)

                nocs_full_np = (nocs_full_np.astype(float) / 127.5) - 1
                src = nocs_full_np[idxs[0], idxs[1], :].T
                #show_pointcloud(src.T)

                num_points_to_sample = config.num_points_to_sample
                if num_points_to_sample > src.shape[1]:
                    src_sampled, dst_sampled = src, dst
                else:
                    src_sampled, dst_sampled = sample_point_cloud(src, dst, num_points_to_sample)

                pcd_src = create_open3d_point_cloud(src, [0, 0, 1])
                pcd_dst = create_open3d_point_cloud(dst, [1, 0, 0])

                print(src_sampled.shape)
                print(dst_sampled.shape)
                print()
                R, t, s = teaserpp_solve(src_sampled, dst_sampled)
                pred_RTs = np.eye(4)
                pred_RTs[:3, 3] = t
                pred_RTs[:3, :3] = R

                src_transformed = s * np.matmul(R, src) + t.reshape(3, 1)
                pcd_src_transformed = create_open3d_point_cloud(src_transformed, [0, 1, 0])  # Green

                line_set = create_line_set(src_sampled, dst_sampled, [0, 1, 0])  # Green lines for correspondences

                pred_scales.append(s)
            
                # Visualize the point clouds using Open3D
                # o3d.visualization.draw_geometries([pcd_src, pcd_dst, pcd_src_transformed, line_set],
                #                                 window_name="TEASER++ Registration with PLY and NOCS",
                #                                 width=1920, height=1080,
                #                                 left=50, top=50,
                #                                 point_show_normal=False)
                
                # o3d.visualization.draw_geometries([pcd_src, pcd_dst, pcd_src_transformed],
                #                                 window_name="TEASER++ Registration with PLY and NOCS",
                #                                 width=1920, height=1080,
                #                                 left=50, top=50,
                #                                 point_show_normal=False)
                
                # OKAY WIR MÜSSEN HIER DIE BOUNDING BOXEN BEREITS IM PREPARE SCRIPT BERECHNEN
                # SELBIGES FÜR DIE GT SCALES
                # ABER WIR BRAUCHEN DAS MESH DANN NICHT MEHR IN DER compute_degree_cm_mAP funktion, DA WIR KEIN ADD BERECHNEN

            result = {}
            result['image_id'] = 0
            result['image_path'] = "."

            result['gt_class_ids'] = category_ids
            result['gt_bboxes'] = gt_bboxes
            result['gt_RTs'] = gt_RTs            
            result['gt_scales'] = gt_scales
            result['gt_handle_visibility'] = 0

            result['pred_bboxes'] = bboxes
            result['pred_class_ids'] = category_ids
            result['pred_scales'] = pred_scales
            result['pred_RTs'] = pred_RTs

            #result['gt_categories'] = gt_categories

            results.append(result)
            
            aps = compute_degree_cm_mAP(results, gt_categories, "./log",
                                                            degree_thresholds = [5, 10, 15],#range(0, 61, 1), 
                                                            shift_thresholds= [5, 10, 15], #np.linspace(0, 1, 31)*15, 
                                                            iou_3d_thresholds=np.linspace(0, 1, 101),
                                                            iou_pose_thres=0.1,
                                                            use_matches_for_pose=True)

if __name__ == "__main__":
    args = parse_args()

    config = load_config(args.config)
    
    main(config)