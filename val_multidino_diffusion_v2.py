import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pickle

from utils import WebDatasetWrapper, preprocess, normalize_quaternion, setup_environment, \
                    create_webdataset_test, custom_collate_fn_test, make_log_dirs, plot_progress_imgs, \
                    preload_pointclouds, plot_single_image, COCODataset, CustomDataset, \
                    collate_fn, collate_fn_val, restore_original_bbox_crop, overlay_nocs_on_rgb,\
                    paste_mask_on_black_canvas, paste_nocs_on_black_canvas, teaserpp_solve, \
                    backproject, sample_point_cloud, create_open3d_point_cloud, load_config, parse_args, \
                    create_line_set, show_pointcloud, filter_points, rotate_transform_matrix_180_z, combine_images_overlapping, \
                    remove_duplicate_pixels
                    
from nocs_paper_utils import compute_degree_cm_mAP, draw_detections, draw_3d_bbox
from nocs_paper_aligning import estimateSimilarityTransform
from diffusion_model import DiffusionNOCSDino, DiffusionNOCSDinoBART, DiffusionNOCSDinoBARTNormals

def project_pointcloud_to_image(pointcloud, pointnormals, fx, fy, cx, cy, image_shape):
    """
    Projects a 3D point cloud to a 2D image plane using the given camera intrinsics,
    with pixel values set to the corresponding x, y, z values of the point normals.

    Args:
        pointcloud (np.ndarray): Array of shape (num_points, 3) containing 3D points (x, y, z).
        pointnormals (np.ndarray): Array of shape (num_points, 3) containing point normals (nx, ny, nz).
        intrinsics (np.ndarray): Camera intrinsics matrix of shape (3, 3).
        image_shape (tuple): Shape of the output image (height, width, 3).

    Returns:
        np.ndarray: 2D image of shape (height, width, 3) with projected points.
    """

    # Extract 3D points
    x, y, z = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]

    # Avoid division by zero
    z = np.where(z == 0, 1e-6, z)

    # Project points to the image plane
    u = (x * fx / z) + cx
    v = (y * fy / z) + cy

    # Round to nearest integer and convert to pixel indices
    u = np.round(u).astype(int)
    v = np.round(v).astype(int)

    # Create an empty image
    height, width, _ = image_shape
    image = np.zeros(image_shape, dtype=np.float32)

    # Keep points within image bounds
    valid_indices = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[valid_indices]
    v = v[valid_indices]
    normals = pointnormals[valid_indices]

    # Project valid points onto the image
    image[v, u] = normals  # Set pixel values to the point normals (nx, ny, nz)

    return image


def main(config):
    setup_environment(str(config.gpu_id))
    make_log_dirs([config.weight_dir, config.val_img_dir, config.ply_output_dir, config.pkl_output_dir, config.bboxes_output_dir, config.png_output.dir])

    camera_intrinsics = {
        'fx': config.fx,
        'fy': config.fy,
        'cx': config.cx,
        'cy': config.cy,
        'width': config.width,
        'height': config.height
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = DiffusionNOCSDinoBARTNormals(input_nc = 9, output_nc = 3, image_size=config.image_size, num_training_steps=config.num_training_steps, num_inference_steps=config.num_inference_steps)
    
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
                rgb_images = torch.clamp(rgb_cropped.float(), min=0.0, max=255.0).unsqueeze(0)
                rgb_images = rgb_images.to(device)
                rgb_images = rgb_images * binary_mask
                rgb_images_gt = (rgb_images.float() / 127.5) - 1

                mask_resized = restore_original_bbox_crop((mask_np * 255).astype(np.uint8), metadatas[idx])
                binary_mask = (mask_resized > 0).astype(np.uint8)
                binary_mask_full_images = (binary_mask_full_images.squeeze(0).squeeze(0).cpu().numpy() > 0).astype(np.uint8)

                mask_full_np = paste_mask_on_black_canvas((rgb_np * 255).astype(np.uint8), (mask_resized).astype(np.uint8), bboxes[idx])

                dst, idxs = backproject(depth_images, camera_intrinsics, mask_full_np)
                dst = dst.T
                pcd_normals = create_open3d_point_cloud(dst, [1, 0, 0])

                o3d.geometry.PointCloud.estimate_normals(pcd_normals)
                pcd_normals.normalize_normals()
                pcd_normals.orient_normals_towards_camera_location()
                normals = np.asarray(pcd_normals.normals)

                print("dst: ", dst.shape)
                normals = project_pointcloud_to_image(dst.T, normals, config.fx, config.fy, config.cx, config.cy, (480, 640, 3))
                normals_uint8 = ((normals + 1) * 127.5).clip(0, 255).astype(np.uint8)
                normal_images = np.expand_dims(normals_uint8, axis=0)

                plt.imshow(normals_uint8)
                plt.axis('off')  # Turn off axis for cleaner output
                plt.show()

                normal_images = torch.tensor(normal_images, dtype=torch.float32)
                normal_images = torch.clamp(normal_images.float(), min=0.0, max=255.0)
                normal_images = normal_images.permute(0, 3, 1, 2)
                normal_images = normal_images.to(device)
                normal_images = normal_images * binary_mask
                normal_images_gt = (normal_images.float() / 127.5) - 1

                nocs_estimated = generator.inference(rgb_images_gt, normal_images_gt, category_names[idx])

                nocs_estimated = ((nocs_estimated + 1 ) / 2)

                nocs_estimated_np = (nocs_estimated).squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to HWC

                nocs_estimated_resized = restore_original_bbox_crop((nocs_estimated_np * 255).astype(np.uint8), metadatas[idx], interpolation=Image.NEAREST)

                nocs_estimated_resized_holes = remove_duplicate_pixels(nocs_estimated_resized)
                
                nocs_estimated_resized_masked = nocs_estimated_resized_holes.copy()
                nocs_estimated_resized_masked[binary_mask == 0] = 0
                nocs_estimated_resized_masked = nocs_estimated_resized_masked.astype(np.float32) / 255
                nocs_full_np = paste_nocs_on_black_canvas((rgb_np * 255).astype(np.uint8), (nocs_estimated_resized_masked).astype(np.uint8), bboxes[idx])

                coords.append(nocs_full_np)

                nocs_full_np = (nocs_full_np.astype(float) / 127.5) - 1
                src = nocs_full_np[idxs[0], idxs[1], :].T

                src_filtered, dst_filtered, _ = filter_points(src, dst, filter_value=-1.0, tolerance=5/255)

                # minimum points for pointcloud registration
                if src_filtered.shape[1] > 4:

                    num_points_to_sample = config.num_points_to_sample

                    if num_points_to_sample > src_filtered.shape[1]:
                        src_filtered, dst_filtered = src_filtered, dst_filtered
                    else:
                        src_filtered, dst_filtered = sample_point_cloud(src_filtered, dst_filtered, num_points_to_sample)

                    pcd_src = create_open3d_point_cloud(src_filtered, [0, 0, 1])
                    pcd_dst = create_open3d_point_cloud(dst_filtered, [1, 0, 0])
                    #pcd_normals = create_open3d_point_cloud(dst, [1, 0, 0])

                    # o3d.geometry.PointCloud.estimate_normals(pcd_normals)
                    # pcd_normals.normalize_normals()
                    # pcd_normals.orient_normals_towards_camera_location()
                    # normals = np.asarray(pcd_normals.normals)

                    # # Visualize the point cloud
                    # #o3d.visualization.draw_geometries([pcd])
                    # print("dst: ", dst.shape)
                    # normals = project_pointcloud_to_image(dst.T, normals, config.fx, config.fy, config.cx, config.cy, (480, 640, 3))
                    # normals_uint8 = ((normals + 1) * 127.5).clip(0, 255).astype(np.uint8)

                    # plt.imshow(normals_uint8)
                    # plt.axis('off')  # Turn off axis for cleaner output
                    # plt.show()

                    R, t, s = teaserpp_solve(src_filtered, dst_filtered, config)
                    print("###########################################################")
                    print("TEASER++")
                    print(R)
                    print(t)
                    print(s)

                    # pred_RT[:3, 3] = t
                    # pred_RT[:3, :3] = R
                    pred_RT = np.eye((4), dtype=np.float32) 
                    pred_RT[:3, :3] = np.matmul(np.diag(np.array([s,s,s])), R)
                    pred_RT[:3, 3] = t

                    print("pred_RT: ", pred_RT)

                    pred_RTs_transformed = rotate_transform_matrix_180_z(pred_RT)
                    pred_RTs.append(pred_RTs_transformed.tolist())

                    src_transformed = s * np.matmul(R, src_filtered) + t.reshape(3, 1)
                    pcd_src_transformed = create_open3d_point_cloud(src_transformed, [0, 1, 0])  # Green

                    min_coords = np.min(src_filtered, axis=1)
                    max_coords = np.max(src_filtered, axis=1)
                    size = max_coords - min_coords

                    # abs_coord_pts = np.abs(s * np.matmul(R, src_filtered))
                    # size = 2 * np.amax(abs_coord_pts, axis=1)

                    bbox = [bboxes[idx][1], bboxes[idx][0], bboxes[idx][3], bboxes[idx][2]]

                    pred_scales.append(size.tolist())
                    pred_scores.append(scores[idx])
                    pred_bboxes.append(bbox)
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

            print("gt scales: ", result['gt_scales'])
            print("pred scales: ", result['pred_scales'] )
            print("gt_RTs: ", result['gt_RTs'])

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