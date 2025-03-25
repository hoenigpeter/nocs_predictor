import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import time
import itertools

from utils import preprocess, setup_environment, \
                    custom_collate_fn_test, make_log_dirs, plot_progress_imgs, \
                    preload_pointclouds, plot_single_image, COCODataset, CustomDataset, \
                    collate_fn, collate_fn_val, restore_original_bbox_crop, overlay_nocs_on_rgb,\
                    paste_mask_on_black_canvas, paste_nocs_on_black_canvas, teaserpp_solve, \
                    backproject, sample_point_cloud, create_open3d_point_cloud, load_config, parse_args, \
                    create_line_set, show_pointcloud, filter_points, rotate_transform_matrix_180_z, combine_images_overlapping, \
                    remove_duplicate_pixels, get_enlarged_bbox, crop_and_resize, project_pointcloud_to_image, create_pointnormals
                    
from nocs_paper_utils import compute_degree_cm_mAP, draw_detections, draw_3d_bbox
from nocs_paper_aligning import estimateSimilarityTransform
from diffusion_model import DiffusionNOCS

def main(config):
    setup_environment(str(config.gpu_id))
    make_log_dirs([config.weight_dir, config.val_img_dir, config.ply_output_dir, config.pkl_output_dir, config.bboxes_output_dir, config.png_output_dir])

    camera_intrinsics = {
        'fx': config.fx,
        'fy': config.fy,
        'cx': config.cx,
        'cy': config.cy,
        'width': config.width,
        'height': config.height
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = DiffusionNOCS(input_nc = 9, output_nc = 3, with_dino_feat = config.with_dino_feat, with_bart_feat = config.with_bart_feat, image_size=config.image_size, num_training_steps=config.num_training_steps, num_inference_steps=config.num_inference_steps)
    
    model_path = os.path.join(config.weight_dir, config.weight_file)
    state_dict = torch.load(model_path, map_location=device)

    generator.load_state_dict(state_dict)
    generator.to(device)
    generator.eval() 

    dataset = CustomDataset(config.coco_json_path, config.test_images_root, image_size=config.image_size, augment=False)
    test_dataloader = DataLoader(dataset, batch_size=config.test_batch_size, shuffle=False, collate_fn=collate_fn_val)
    print("number of test images: ", len(test_dataloader))

    results = []

    with torch.no_grad():
        
        for step, batch in enumerate(test_dataloader):
            # if step < 3000:
            #     print("Step: ", step)
            #     continue  # Skip first 1000 batches

            print("Step: ", step)

            frame_id = batch['frame_id'][0]
            scene_id = batch['scene_id'][0]
            rgb_images = batch['rgb']
            depth_images = batch['depth']
            #mask_full_images = batch['mask'][0]
            rgb_crops =  batch['rgb_crops'][0]
            mask_images = batch['mask_crops'][0]
            bboxes = batch['bboxes'][0]
            metadatas = batch['metadatas'][0]
            category_names = batch['category_names'][0]
            category_ids = batch['category_ids'][0]
            scores = batch['scores'][0]

            gts = batch['gts'][0]
            gt_bboxes = gts['gt_bboxes']
            gt_bboxes_list = []
            for bbox in gt_bboxes:
                gt_bboxes_list.append([bbox[1], bbox[0], bbox[3], bbox[2]])
            gt_bboxes = np.array(gt_bboxes_list, dtype=np.int32)

            gt_scales = np.array(gts['gt_scales'], dtype=np.float32)
            gt_RTs = gts['gt_RTs']
            gt_category_ids = np.array(gts['gt_category_ids'])

            rgb_np = rgb_images.squeeze().permute(1, 2, 0).cpu().numpy()

            depth_images = depth_images.squeeze(0).squeeze(0).cpu().numpy()

            coords_list = []
            normals_list = []

            pred_scales = []
            pred_RTs = []
            pred_scores = []
            pred_bboxes = []
            pred_category_ids = []

            gt_handle_visibility = []
            combined_pcds = []

            for idx in range(len(rgb_crops)):
                t0 = time.time()

                mask_np = mask_images[idx].squeeze().cpu().numpy()
                mask_image = mask_images[idx].unsqueeze(0).unsqueeze(0)

                binary_mask = (mask_image > 0).float()
                binary_mask = binary_mask.to(device)

                mask_images_gt = mask_image.float() / 255.0
                mask_images_gt = mask_images_gt.permute(0, 3, 1, 2)
                mask_images_gt = mask_images_gt.to(device)

                rgb_cropped = rgb_crops[idx]
                rgb_images = torch.clamp(rgb_cropped.float(), min=0.0, max=255.0).unsqueeze(0)
                rgb_images = rgb_images.to(device)
                rgb_images = rgb_images * binary_mask
                rgb_images_gt = (rgb_images.float() / 127.5) - 1

                mask_resized = restore_original_bbox_crop((mask_np * 255).astype(np.uint8), metadatas[idx])
                mask_full_np = paste_mask_on_black_canvas((rgb_np * 255).astype(np.uint8), (mask_resized).astype(np.uint8), bboxes[idx])

                dst, idxs = backproject(depth_images, camera_intrinsics, mask_full_np)
                dst = dst.T

                normals = create_pointnormals(dst)
                normals = project_pointcloud_to_image(dst.T, normals, config.fx, config.fy, config.cx, config.cy, (config.height, config.width, 3))

                mask_normals = np.all(normals == [0, 0, 0], axis=-1)
                normals = ((normals + 1) * 127.5).clip(0, 255).astype(np.uint8)
                normals[mask_normals] = [0, 0, 0]

                enlarged_bbox = get_enlarged_bbox(bboxes[idx], rgb_np.shape, bbox_scaler=1.5)
                normal_images_crop, _ = crop_and_resize(normals, enlarged_bbox, bboxes[idx], target_size=config.image_size, interpolation=Image.NEAREST)
                normal_estimated_resized = restore_original_bbox_crop(np.transpose(normal_images_crop, (1, 2, 0)), metadatas[idx], interpolation=Image.NEAREST)

                normal_images = torch.tensor(normal_images_crop, dtype=torch.float32)
                normal_images = torch.clamp(normal_images.float(), min=0.0, max=255.0).unsqueeze(0)
                normal_images = normal_images.to(device)

                normal_images = normal_images * binary_mask
                normal_images_gt = (normal_images.float() / 127.5) - 1

                binary_mask = (mask_resized > 0).astype(np.uint8)

                max_inliers = 0.0
                best_Rt = None
                best_R = None
                best_t = None
                best_s = None
                best_src_filtered = None
                best_dst_filtered = None
                best_nocs_full_np_save = None
                best_normals_full_np_save = None

                dst = dst.T
                dst[:, 0] = -dst[:, 0]
                dst[:, 1] = -dst[:, 1]
                dst = dst.T
                
                embeddings = generator.get_embeddings(rgb_images_gt, category_names[idx])

                for ref_step in range(config.num_refinement_steps):
                    print("Refinement step: ", ref_step)

                    nocs_estimated = generator.inference(rgb_images_gt, normal_images_gt, embeddings)
                    nocs_estimated = ((nocs_estimated + 1 ) / 2)

                    nocs_estimated_np = (nocs_estimated).squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to HWC
                    nocs_estimated_resized = restore_original_bbox_crop((nocs_estimated_np * 255).astype(np.uint8), metadatas[idx], interpolation=Image.NEAREST)

                    nocs_estimated_resized_holes = remove_duplicate_pixels(nocs_estimated_resized)
                    nocs_estimated_resized_masked = nocs_estimated_resized_holes.copy()
                    nocs_estimated_resized_masked[binary_mask == 0] = 0

                    nocs_full_np = paste_nocs_on_black_canvas((rgb_np * 255).astype(np.uint8), nocs_estimated_resized_masked.astype(np.uint8), bboxes[idx])
                    nocs_full_np_save = paste_nocs_on_black_canvas((rgb_np * 255).astype(np.uint8), nocs_estimated_resized.astype(np.uint8), bboxes[idx])
                    normals_full_np_save = paste_nocs_on_black_canvas((rgb_np * 255).astype(np.uint8), normal_estimated_resized.astype(np.uint8), bboxes[idx])
                    
                    nocs_full_np = (nocs_full_np.astype(float) / 127.5) - 1
                    src = nocs_full_np[idxs[0], idxs[1], :].T

                    src_filtered, dst_filtered, _ = filter_points(src, dst, filter_value=-1.0, tolerance=5/255)

                    if src_filtered.shape[1] > 4:

                        num_points_to_sample = config.num_points_to_sample

                        if num_points_to_sample > src_filtered.shape[1]:
                            src_filtered, dst_filtered = src_filtered, dst_filtered
                        else:
                            src_filtered, dst_filtered = sample_point_cloud(src_filtered, dst_filtered, num_points_to_sample)

                        R, t, s, R_inliers, t_inliers, s_inliers = teaserpp_solve(src_filtered, dst_filtered, config)
                        R_inliers = float(R_inliers)/float(config.num_points_to_sample)
                        t_inliers = float(t_inliers)/float(config.num_points_to_sample)
                        num_total_pairs = (float(config.num_points_to_sample) * (float(config.num_points_to_sample) - 1)) / 2
                        s_inliers = float(s_inliers)/num_total_pairs
                        print("R_inliers: ", R_inliers)
                        print("t_inliers: ", t_inliers)
                        print("s_inliers: ", s_inliers)

                        #inliers = (R_inliers + t_inliers + s_inliers) / 3
                        inliers = R_inliers

                        # print("###########################################################")
                        # print("TEASER++")
                        # print("R: ", R)
                        # print("t: ", t)
                        # print("s: ", s)
                        print("inliers: ", inliers)
                        print()

                        pred_RT = np.eye((4), dtype=np.float32) 
                        pred_RT[:3, :3] = np.matmul(np.diag(np.array([s,s,s])), R)
                        pred_RT[:3, 3] = t

                        if inliers > max_inliers:
                            print("new best inlier!")
                            max_inliers = inliers
                            best_Rt = pred_RT
                            best_R = R
                            best_t = t
                            best_s = s
                            best_src_filtered = src_filtered
                            best_dst_filtered = dst_filtered
                            best_nocs_full_np_save = nocs_full_np_save
                            best_normals_full_np_save = normals_full_np_save

                    if config.refinement == False:
                        break 

                if best_Rt is not None:
                    pred_RTs_transformed = rotate_transform_matrix_180_z(best_Rt)
                    src_transformed = best_s * np.matmul(best_R, best_src_filtered) + best_t.reshape(3, 1)
                    pcd_src_transformed = create_open3d_point_cloud(src_transformed, [0, 1, 0])  # Green

                    min_coords = np.min(best_src_filtered, axis=1)
                    max_coords = np.max(best_src_filtered, axis=1)
                    size = max_coords - min_coords
                    bbox = [bboxes[idx][1], bboxes[idx][0], bboxes[idx][3], bboxes[idx][2]]
                    # abs_coord_pts = np.abs(s * np.matmul(R, src_filtered))
                    # size = 2 * np.amax(abs_coord_pts, axis=1)

                    # saved for later
                    pred_RTs.append(pred_RTs_transformed.tolist())
                    pred_scales.append(size.tolist())
                    pred_scores.append(scores[idx])
                    pred_bboxes.append(bbox)
                    pred_category_ids.append(category_ids[idx])

                    pcd_dst = create_open3d_point_cloud(best_dst_filtered, [1, 0, 0])
                    combined_pcds.append(pcd_dst)
                    combined_pcds.append(pcd_src_transformed)

                    coords_list.append(best_nocs_full_np_save)
                    normals_list.append(best_normals_full_np_save)

                    t1 = time.time()
                    print(f"Elapsed time: {t1 - t0:.4f} seconds")

            if len(pred_RTs) > 0:
                combined_pcd = o3d.geometry.PointCloud()
                for pcd in combined_pcds:
                    combined_pcd += pcd

                coord_image_save_np = np.zeros((480, 640, 3), dtype=np.uint8)
                normals_image_save_np = np.zeros((480, 640, 3), dtype=np.uint8)

                for img in coords_list:
                    mask_temp = np.any(img != [0, 0, 0], axis=-1)
                    coord_image_save_np[mask_temp] = img[mask_temp]

                for img in normals_list:
                    mask_temp = np.any(img != [0, 0, 0], axis=-1)
                    normals_image_save_np[mask_temp] = img[mask_temp]

                coord_image_save = Image.fromarray(coord_image_save_np)
                normals_image_save = Image.fromarray(normals_image_save_np)

                coord_image_save.save(config.png_output_dir + f"/scene_{scene_id}_{str(frame_id).zfill(4)}_coords.png")
                normals_image_save.save(config.png_output_dir + f"/scene_{scene_id}_{str(frame_id).zfill(4)}_normals.png")

                o3d.io.write_point_cloud(config.ply_output_dir + f"/scene_{scene_id}_{str(frame_id).zfill(4)}_pcl.ply", combined_pcd)

                result = {}
                #result['image_id'] = 0
                #result['image_path'] = "."

                result['image_id'] = 0
                result['image_path'] = 'datasets/real_test/scene_1/0000'
                result['gt_class_ids'] = gt_category_ids
                result['gt_bboxes'] = np.array(gt_bboxes, dtype=np.int32)
                result['gt_RTs'] = np.array(gt_RTs)            
                result['gt_scales'] = gt_scales
                
                result['gt_handle_visibility'] = np.array([1] * len(gt_scales))

                result['pred_class_ids'] = np.array(pred_category_ids, dtype=np.int32)
                result['pred_bboxes'] = np.array(pred_bboxes, dtype=np.int32)
                result['pred_RTs'] = np.array(pred_RTs)
                result['pred_scores'] = np.array(pred_scores, dtype=np.float32)
                result['pred_scales'] = np.array(pred_scales)

                # print("gt scales: ", result['gt_scales'])
                # print("pred scales: ", result['pred_scales'] )
                # print("gt_RTs: ", result['gt_RTs'])

                results.append(result)

                draw_3d_bbox(rgb_np, save_dir=config.bboxes_output_dir, data_name=scene_id, image_id=frame_id, intrinsics=camera_intrinsics,
                                gt_RTs=result['gt_RTs'], gt_scales=result['gt_scales'], pred_RTs=result['pred_RTs'], pred_scales=result['pred_scales'])
                
                synset_names = [
                    'BG', #0
                    'bottle', #1
                    'bowl', #2
                    'camera', #3
                    'can',  #4
                    'laptop',#5
                    'mug'#6
                    ]
                
                pickle_file = config.pkl_output_dir + f"/results_real_test_scene_{int(scene_id)}_{int(frame_id):04d}.pkl"

                with open(pickle_file, 'wb') as f:
                    pickle.dump(result, f)

                print(f"Results saved to {pickle_file}")

        aps = compute_degree_cm_mAP(results, synset_names, log_dir=config.weight_dir,
                                                        degree_thresholds = [5, 10, 15],#range(0, 61, 1), 
                                                        shift_thresholds= [5, 10, 15], #np.linspace(0, 1, 31)*15, 
                                                        iou_3d_thresholds=np.linspace(0, 1, 101),
                                                        iou_pose_thres=0.1,
                                                        use_matches_for_pose=True)

if __name__ == "__main__":
    args = parse_args()

    config = load_config(args.config)
    
    main(config)