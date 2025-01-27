import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from utils import WebDatasetWrapper, preprocess, normalize_quaternion, setup_environment, \
                    create_webdataset_test, custom_collate_fn_test, make_log_dirs, plot_progress_imgs, \
                    preload_pointclouds, plot_single_image, COCODataset, CustomDataset, \
                    collate_fn, collate_fn_val, restore_original_bbox_crop, overlay_nocs_on_rgb,\
                    paste_mask_on_black_canvas, paste_nocs_on_black_canvas, teaserpp_solve, \
                    backproject, sample_point_cloud, create_open3d_point_cloud, load_config, parse_args, \
                    create_line_set, show_pointcloud, filter_points, rotate_transform_matrix_180_z, combine_images_overlapping, \
                    remove_duplicate_pixels
                    
from nocs_paper_utils import compute_degree_cm_mAP
from nocs_paper_aligning import estimateSimilarityTransform
from diffusion_model import DiffusionNOCSDino, DiffusionNOCSDinoBART



def main(config):
    setup_environment(str(config.gpu_id))
    make_log_dirs([config.weight_dir, config.val_img_dir, config.ply_output_dir])

    camera_intrinsics = {
        'fx': config.fx,
        'fy': config.fy,
        'cx': config.cx,
        'cy': config.cy,
        'width': config.width,
        'height': config.height
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = DiffusionNOCSDinoBART(input_nc = 6, output_nc = 3, image_size=config.image_size, num_training_steps=config.num_training_steps, num_inference_steps=config.num_inference_steps)
    
    model_path = os.path.join(config.weight_dir, config.weight_file)
    state_dict = torch.load(model_path, map_location=device)

    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval() 

    dataset = CustomDataset(config.coco_json_path, config.test_images_root, image_size=config.image_size, augment=False)
    test_dataloader = DataLoader(dataset, batch_size=config.test_batch_size, shuffle=False, collate_fn=collate_fn_val)
    print("number of test images: ", len(test_dataloader))

    results = []
    coords = []

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            print("Step: ", step)

            frame_id = batch['frame_id'][0]
            scene_id = batch['scene_id'][0]
            rgb_images = batch['rgb']
            depth_images = batch['depth']
            mask_full_images = batch['mask'][0]
            rgb_crops =  batch['rgb_crops'][0]
            mask_images = batch['mask_crops'][0]
            bboxes = batch['bboxes'][0]
            metadatas = batch['metadatas'][0]
            category_names = batch['category_names'][0]
            category_ids = batch['category_ids'][0]
            gts = batch['gts'][0]
            gt_bboxes = gts['gt_bboxes']
            gt_scales = np.array(gts['gt_scales'])/100
            gt_RTs = gts['gt_RTs']
            gt_category_ids = np.array(gts['gt_category_ids'])

            rgb_np = rgb_images.squeeze().permute(1, 2, 0).cpu().numpy()

            depth_images = depth_images.squeeze(0).squeeze(0).cpu().numpy()
            
            pred_scales = []
            pred_RTs = []
            pred_scores = []
            pred_bboxes = []
            pred_category_ids = []

            gt_handle_visibility = []
            combined_pcds = []

            for idx in range(len(rgb_crops)):
                mask_np = mask_images[idx].squeeze().cpu().numpy()
                mask_image = mask_images[idx].unsqueeze(0).unsqueeze(0)

                binary_mask = (mask_image > 0).float()
                binary_mask = binary_mask.to(device)

                mask_full_image = mask_full_images[idx].unsqueeze(0)
                binary_mask_full_images = (mask_full_image > 0).float()
                binary_mask_full_images = binary_mask_full_images.to(device)

                mask_images_gt = mask_image.float() / 255.0
                mask_images_gt = mask_images_gt.permute(0, 3, 1, 2)
                mask_images_gt = mask_images_gt.to(device)

                rgb_cropped = rgb_crops[idx]
                rgb_cropped_np = rgb_cropped.squeeze().permute(1, 2, 0).cpu().numpy()

                rgb_images = torch.clamp(rgb_cropped.float(), min=0.0, max=255.0).unsqueeze(0)
                rgb_images = rgb_images.to(device)

                rgb_images = rgb_images * binary_mask

                rgb_images_gt = (rgb_images.float() / 127.5) - 1

                # inference with diffusion models
                nocs_estimated = generator.inference(rgb_images_gt, category_names[idx])

                nocs_estimated = ((nocs_estimated + 1 ) / 2)

                nocs_estimated_np = (nocs_estimated).squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to HWC

                nocs_estimated_resized = restore_original_bbox_crop((nocs_estimated_np * 255).astype(np.uint8), metadatas[idx], interpolation=Image.NEAREST)

                nocs_estimated_resized_holes = remove_duplicate_pixels(nocs_estimated_resized)
                
                mask_resized = restore_original_bbox_crop((mask_np * 255).astype(np.uint8), metadatas[idx])

                rgb_nocs_overlay_image = overlay_nocs_on_rgb((rgb_np * 255).astype(np.uint8), nocs_estimated_resized, (mask_resized).astype(np.uint8), bboxes[idx])

                binary_mask = (mask_resized > 0).astype(np.uint8)
                binary_mask_full_images = (binary_mask_full_images.squeeze(0).squeeze(0).cpu().numpy() > 0).astype(np.uint8)
                nocs_estimated_resized_masked = nocs_estimated_resized_holes.copy()
                nocs_estimated_resized_masked[binary_mask == 0] = 0

                rgb_images_gt_np = (((rgb_images_gt + 1) /2)[0].permute(1, 2, 0).cpu().numpy() * 255 ).astype(np.uint8)

                mask_full_np = paste_mask_on_black_canvas((rgb_np * 255).astype(np.uint8), (mask_resized).astype(np.uint8), bboxes[idx])
                nocs_full_np = paste_nocs_on_black_canvas((rgb_np * 255).astype(np.uint8), (nocs_estimated_resized_masked).astype(np.uint8), bboxes[idx])

                coords.append(nocs_full_np)

                nocs_estimated_resized_masked = nocs_estimated_resized_masked.astype(np.float32) / 255

                dst, idxs = backproject(depth_images, camera_intrinsics, mask_full_np)
                dst = dst.T

                nocs_full_np = (nocs_full_np.astype(float) / 127.5) - 1
                src = nocs_full_np[idxs[0], idxs[1], :].T

                print("shape before filter: ", src.shape)
                src_filtered, dst_filtered, _ = filter_points(src, dst, filter_value=-1.0, tolerance=2/255)
                print("shape after filter: ", src_filtered.shape)

                # minimum points for pointcloud registration
                if src_filtered.shape[1] > 4:

                    num_points_to_sample = config.num_points_to_sample

                    if num_points_to_sample > src_filtered.shape[1]:
                        src_filtered, dst_filtered = src_filtered, dst_filtered
                    else:
                        src_filtered, dst_filtered = sample_point_cloud(src_filtered, dst_filtered, num_points_to_sample)

                    pcd_src = create_open3d_point_cloud(src_filtered, [0, 0, 1])
                    pcd_dst = create_open3d_point_cloud(dst_filtered, [1, 0, 0])

                    R, t, s = teaserpp_solve(src_filtered, dst_filtered, config)
                    print("###########################################################")
                    print("TEASER++")
                    print(R)
                    print(t)
                    print(s)

                    pred_RT = np.eye(4)
                    pred_RT[:3, 3] = t
                    pred_RT[:3, :3] = R
                    pred_RTs_transformed = rotate_transform_matrix_180_z(pred_RT)
                    pred_RTs.append(pred_RTs_transformed)

                    src_transformed = s * np.matmul(R, src_filtered) + t.reshape(3, 1)
                    pcd_src_transformed = create_open3d_point_cloud(src_transformed, [0, 1, 0])  # Green

                    line_set = create_line_set(src_filtered, dst_filtered, [0, 1, 0])  # Green lines for correspondences
                    s = np.array([s, s, s])

                    pred_scales.append(s)
                    pred_scores.append(1.0)
                    pred_bboxes.append(bboxes[idx])
                    pred_category_ids.append(category_ids[idx])
               
                    combined_pcds.append(pcd_dst)
                    combined_pcds.append(pcd_src_transformed)


            combined_pcd = o3d.geometry.PointCloud()
            for pcd in combined_pcds:
                combined_pcd += pcd

            o3d.io.write_point_cloud(config.ply_output_dir + f"/scene_{scene_id}_{str(frame_id).zfill(4)}_pcl.ply", combined_pcd)

            result = {}
            #result['image_id'] = 0
            #result['image_path'] = "."

            result['gt_class_ids'] = gt_category_ids
            result['gt_bboxes'] = gt_bboxes
            result['gt_RTs'] = gt_RTs            
            result['gt_scales'] = gt_scales
            result['gt_handle_visibility'] = [1] * len(gt_scales)

            result['pred_bboxes'] = pred_bboxes
            result['pred_class_ids'] = pred_category_ids
            result['pred_scales'] = pred_scales
            result['pred_RTs'] = pred_RTs
            result['pred_scores'] = pred_scores

            results.append(result)
            
            synset_names = [
                'BG', #0
                'bottle', #1
                'bowl', #2
                'camera', #3
                'can',  #4
                'laptop',#5
                'mug'#6
                ]
            
            break
            
        aps = compute_degree_cm_mAP(results, synset_names, log_dir=config.weight_dir,
                                                        degree_thresholds = [5, 10, 15],#range(0, 61, 1), 
                                                        shift_thresholds= [5, 10, 15], #np.linspace(0, 1, 31)*15, 
                                                        #iou_3d_thresholds=np.linspace(0, 1, 101),
                                                        iou_3d_thresholds=np.linspace(0, 1, 101),
                                                        iou_pose_thres=0.1,
                                                        use_matches_for_pose=True)

if __name__ == "__main__":
    args = parse_args()

    config = load_config(args.config)
    
    main(config)