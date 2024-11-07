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
from transformers import CLIPProcessor, CLIPModel
import open3d as o3d
from torch.optim.lr_scheduler import CosineAnnealingLR
import trimesh
from scipy.spatial import cKDTree

import json
import webdataset as wds

from utils import WebDatasetWrapper, preprocess, normalize_quaternion, setup_environment, \
                    create_webdataset_test, custom_collate_fn_test, make_log_dirs, plot_progress_imgs, \
                    preload_pointclouds, plot_single_image, COCODataset, collate_fn

import teaserpp_python

import argparse
import importlib.util

from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

def plot_nocs_image(nocs_tensor, min_neighbors=400, neighborhood_size=0.2, eps=0.05, min_samples=10):
    """
    Plots a cleaned NOCS image as 3D points using Matplotlib, applying k-NN-based filtering 
    and removing the smallest cluster using DBSCAN.

    Args:
    - nocs_tensor (torch.Tensor): Tensor of shape (batch_size, 3, height, width) representing the NOCS coordinates.
    - min_neighbors (int): Minimum number of neighbors a point needs to be considered non-isolated.
    - neighborhood_size (float): Maximum distance within which to search for neighbors.
    - eps (float): The maximum distance between two points for them to be considered in the same neighborhood (DBSCAN).
    - min_samples (int): Minimum number of points required to form a dense region in DBSCAN.
    """
    # Ensure input is a 4D tensor [batch_size, 3, height, width]
    assert nocs_tensor.ndim == 4, "Input tensor must have shape [batch_size, 3, height, width]"
    
    batch_size, _, height, width = nocs_tensor.shape
    
    # Plot only the first element of the batch
    nocs_image = nocs_tensor[0]  # Take first NOCS image from the batch, shape is [3, height, width]
    
    # Extract x, y, z components from the NOCS image
    x_coords = nocs_image[0].flatten().cpu().numpy()
    y_coords = nocs_image[1].flatten().cpu().numpy()
    z_coords = nocs_image[2].flatten().cpu().numpy()
    
    # Stack into a single array of points [num_points, 3]
    points = np.vstack((x_coords, y_coords, z_coords)).T
    
    valid_mask = ~(np.all(points < 0.3, axis=1))
    points = points[valid_mask]

    # Apply k-NN-based filtering to remove isolated points
    nbrs = NearestNeighbors(n_neighbors=min_neighbors + 1, radius=neighborhood_size).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Count how many neighbors are within the neighborhood_size
    neighbor_counts = (distances <= neighborhood_size).sum(axis=1) - 1  # Exclude the point itself
    
    # Only keep points with at least `min_neighbors` neighbors
    valid_mask = neighbor_counts >= min_neighbors
    points_cleaned = points[valid_mask]
    
    # Extract cleaned x, y, z coordinates
    x_coords_cleaned = points_cleaned[:, 0]
    y_coords_cleaned = points_cleaned[:, 1]
    z_coords_cleaned = points_cleaned[:, 2]
    
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot of cleaned 3D points
    ax.scatter(x_coords_cleaned, y_coords_cleaned, z_coords_cleaned, c=z_coords_cleaned, cmap='viridis', marker='o', s=0.5)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Cleaned NOCS Image 3D Plot (k-NN + Cluster Removal)')

    # Set equal scaling for all axes
    x_limits = [x_coords_cleaned.min(), x_coords_cleaned.max()]
    y_limits = [y_coords_cleaned.min(), y_coords_cleaned.max()]
    z_limits = [z_coords_cleaned.min(), z_coords_cleaned.max()]

    # Create an equal aspect ratio for the plot by setting the limits equally
    max_range = np.array([x_limits, y_limits, z_limits]).ptp().max() / 2.0  # ptp() gives range of values

    mid_x = np.mean(x_limits)
    mid_y = np.mean(y_limits)
    mid_z = np.mean(z_limits)

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

def calculate_3d_iou(bbox_pred, bbox_gt):
    """
    Calculate the 3D IoU between a predicted and ground truth bounding box.

    Args:
        bbox_pred (numpy array): Predicted 3D bounding box corners, shape (8, 3).
        bbox_gt (numpy array): Ground truth 3D bounding box corners, shape (8, 3).

    Returns:
        float: 3D IoU score between 0 and 1.
    """

    # Compute the min and max corners along each axis for both predicted and ground truth boxes
    pred_min = np.min(bbox_pred, axis=0)
    pred_max = np.max(bbox_pred, axis=0)
    gt_min = np.min(bbox_gt, axis=0)
    gt_max = np.max(bbox_gt, axis=0)

    # Compute intersection along each axis
    inter_min = np.maximum(pred_min, gt_min)
    inter_max = np.minimum(pred_max, gt_max)
    inter_dim = np.maximum(0, inter_max - inter_min)  # Ensure no negative dimensions

    # Intersection volume
    intersection_volume = np.prod(inter_dim)

    # Volumes of the predicted and ground truth boxes
    pred_volume = np.prod(pred_max - pred_min)
    gt_volume = np.prod(gt_max - gt_min)

    # Union volume
    union_volume = pred_volume + gt_volume - intersection_volume

    # IoU
    iou = intersection_volume / union_volume if union_volume > 0 else 0

    return iou

def compute_3d_bounding_box(point_cloud):
    """
    Compute the axis-aligned 3D bounding box (AABB) of a point cloud.

    Args:
        point_cloud (numpy array): Transformed point cloud in world coordinates (N, 3).

    Returns:
        numpy array: Bounding box corner coordinates (8, 3).
    """
    # Get min and max corners
    bbox_min = np.min(point_cloud, axis=0)
    bbox_max = np.max(point_cloud, axis=0)
    
    # Create the 8 corner points of the bounding box
    bbox_corners = np.array([
        [bbox_min[0], bbox_min[1], bbox_min[2]],  # (min, min, min)
        [bbox_min[0], bbox_min[1], bbox_max[2]],  # (min, min, max)
        [bbox_min[0], bbox_max[1], bbox_min[2]],  # (min, max, min)
        [bbox_min[0], bbox_max[1], bbox_max[2]],  # (min, max, max)
        [bbox_max[0], bbox_min[1], bbox_min[2]],  # (max, min, min)
        [bbox_max[0], bbox_min[1], bbox_max[2]],  # (max, min, max)
        [bbox_max[0], bbox_max[1], bbox_min[2]],  # (max, max, min)
        [bbox_max[0], bbox_max[1], bbox_max[2]]   # (max, max, max)
    ])
    
    return bbox_corners

def generate_point_cloud_from_depth(depth_image, enlarged_bbox, camera_intrinsics):
    """
    Generate a point cloud from a cropped depth image based on the enlarged bounding box and camera intrinsics.

    Args:
        depth_image (numpy array): Cropped depth image (H, W).
        enlarged_bbox (numpy array): Coordinates of the enlarged bounding box (crop_ymin, crop_xmin, crop_ymax, crop_xmax).
        camera_intrinsics (dict): Camera intrinsics containing 'fx', 'fy', 'cx', 'cy', 'width', 'height'.

    Returns:
        numpy array: A point cloud of shape (H * W, 3) where H, W are the dimensions of the input depth image.
                     Invalid points will have values set to (NaN, NaN, NaN) or zero.
    """
    crop_ymin, crop_xmin, crop_ymax, crop_xmax = enlarged_bbox

    # Get camera intrinsics
    fx = camera_intrinsics['fx']
    fy = camera_intrinsics['fy']
    cx = camera_intrinsics['cx']
    cy = camera_intrinsics['cy']

    # Adjust optical center (cx, cy) for the cropped depth image
    adjusted_cx = cx - crop_xmin
    adjusted_cy = cy - crop_ymin

    # Create a meshgrid of pixel coordinates for the cropped depth image
    h, w = depth_image.shape
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # Flatten the arrays to 1D
    x_coords_flat = x_coords.flatten()
    y_coords_flat = y_coords.flatten()
    depth_values_flat = depth_image.flatten()

    # Calculate 3D coordinates from depth values
    Z = depth_values_flat  # No filtering out invalid depth values here
    X = (x_coords_flat - adjusted_cx) * Z / fx
    Y = (y_coords_flat - adjusted_cy) * Z / fy

    # Mask invalid points (where depth == 0 or negative), and assign NaN or zero to those points
    invalid_mask = Z <= 0  # This mask finds invalid points
    X[invalid_mask] = 0  # Optionally use 0 instead of NaN if needed
    Y[invalid_mask] = 0
    Z[invalid_mask] = 0

    # Stack to create the point cloud (X, Y, Z), maintaining the same number of points as pixels
    point_cloud = np.vstack((X, Y, Z)).T  # Shape: (H * W, 3)

    return point_cloud

def filter_zero_points(src, dst):
    """
    Removes points from src and dst where all coordinates in src (x, y, z) are zero.
    
    Parameters:
        src (np.ndarray): Source point cloud (shape: 3, N).
        dst (np.ndarray): Destination point cloud (shape: 3, N).
        
    Returns:
        src_filtered (np.ndarray): Filtered source point cloud.
        dst_filtered (np.ndarray): Filtered destination point cloud.
        indices_removed (np.ndarray): Indices of points removed from src and dst.
    """
    # Step 1: Create a boolean mask to identify non-zero points in src
    mask = ~np.all(src == 0, axis=0)
    
    # Step 2: Retrieve the indices of points to remove (optional)
    indices_removed = np.where(~mask)[0]
    
    # Step 3: Apply the mask to filter out unwanted points
    src_filtered = src[:, mask]
    dst_filtered = dst[:, mask]
    
    return src_filtered, dst_filtered, indices_removed

def sample_point_cloud(src, dst, num_samples):
    if src.shape[1] < num_samples:
        raise ValueError("The number of samples exceeds the number of available points.")
    
    # Randomly choose indices to sample
    indices = np.random.choice(src.shape[1], num_samples, replace=False)
    
    # Return the sampled points
    return src[:, indices], dst[:, indices]

def create_line_set(src_points, dst_points, color=[0, 1, 0]):
    # Convert src_points and dst_points to numpy arrays and ensure they are float64
    src_points = np.asarray(src_points, dtype=np.float64).T
    dst_points = np.asarray(dst_points, dtype=np.float64).T
    
    # Check if shapes are correct
    if src_points.shape[1] != 3 or dst_points.shape[1] != 3:
        raise ValueError("Points must have a shape of (N, 3)")
    
    # Create lines connecting each pair of corresponding points
    lines = [[i, i + len(src_points)] for i in range(len(src_points))]
    
    # Create Open3D LineSet object
    line_set = o3d.geometry.LineSet()
    
    # Concatenate the points and set the points and lines
    all_points = np.concatenate((src_points, dst_points), axis=0)
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    # Set the color for the lines
    colors = [color] * len(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def create_open3d_point_cloud(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.T)  # Open3D expects points as Nx3
    pcd.paint_uniform_color(color)  # Set color of the point cloud
    return pcd

def generate_and_visualize_point_cloud(image_tensor):
    """
    Generates and visualizes a point cloud from a NOCS or depth image tensor.
    
    Args:
        image_tensor (torch.Tensor): Input tensor of shape:
                                     - [1, 3, H, W] for NOCS images
                                     - [1, H, W] for depth images
    """
    # Check if the input is a NOCS or depth image
    if image_tensor.dim() == 4 and image_tensor.size(1) == 3:
        # NOCS image case (shape [1, 3, H, W])
        nocs_image = image_tensor.squeeze(0).cpu().numpy()  # Remove batch dim
        nocs_image = np.transpose(nocs_image, (1, 2, 0))    # Shape becomes (H, W, 3)
        
        # Flatten to get the point cloud (flattening HxWx3 to Nx3)
        points = nocs_image.reshape(-1, 3)  # Shape becomes (H*W, 3)
        
    elif image_tensor.dim() == 3: 
        # Depth image case (shape [1, H, W])
        depth_image = image_tensor.squeeze(0).cpu().numpy()  # Remove batch dim
        H, W = depth_image.shape

        # Generate the 3D point cloud from the depth values (assuming depth values are in metric units)
        # Create meshgrid for pixel coordinates
        xx, yy = np.meshgrid(np.arange(W), np.arange(H))
        points = np.stack((xx, yy, depth_image), axis=-1).reshape(-1, 3)  # Shape becomes (H*W, 3)
    
    else:
        raise ValueError("Input tensor should be of shape [1, 3, H, W] for NOCS or [1, H, W] for depth images.")
    
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

def generate_point_cloud_from_nocs(image_tensor):
    """
    Generates and visualizes a point cloud from a NOCS or depth image tensor.
    
    Args:
        image_tensor (torch.Tensor): Input tensor of shape:
                                     - [1, 3, H, W] for NOCS images
                                     - [1, H, W] for depth images
    """

    # NOCS image case (shape [1, 3, H, W])
    nocs_image = image_tensor.squeeze(0).cpu().numpy()  # Remove batch dim
    nocs_image = np.transpose(nocs_image, (1, 2, 0))    # Shape becomes (H, W, 3)
    
    # Flatten to get the point cloud (flattening HxWx3 to Nx3)
    points = nocs_image.reshape(-1, 3)  # Shape becomes (H*W, 3)

    return points

def show_pointcloud(points):
    # Create Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

# Function to load config from the passed file
def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('config', type=str, help="Path to the config file")
    return parser.parse_args()

def main(config):
    setup_environment(str(config.gpu_id))
    make_log_dirs(config.weight_dir, config.val_img_dir)

    fx = config.fx
    fy = config.fy
    cx = config.cx
    cy = config.cy

    width = config.width
    height = config.height

    camera_intrinsics = {
        'fx': fx,  # Focal length in x
        'fy': fy,  # Focal length in y
        'cx': cx,   # Optical center x-coordinate
        'cy': cy,   # Optical center y-coordinate
        'width': width,
        'height': height
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained model for inference
    model_path = os.path.join(config.weight_dir, 'generator_epoch_95.pth')
    generator = torch.load(model_path, map_location=device)
    generator.to(device)

    # Set model to evaluation mode for inference
    generator.eval()

    dataset = COCODataset(config.coco_json_path, config.test_images_root, image_size=config.size, augment=False)
    test_dataloader = DataLoader(dataset, batch_size=config.test_batch_size, shuffle=True, collate_fn=collate_fn)
    print(len(test_dataloader))

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            print("Step: ", step)
            # unwrap the batch
            rgb_images = batch['rgb']
            rgb_cropped =  batch['rgb_crop']
            mask_images = batch['mask_crop']
            bboxes = batch['bbox']

            print("rgb: ", rgb_images.shape)
            print("rgb_cropped: ", rgb_cropped.shape)
            print("mask: ", mask_images.shape)
            print("rgb_cropped: ", rgb_cropped.shape)

            rgb_np = rgb_images.squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to HWC
            rgb_cropped_np = rgb_cropped.squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to HWC
            mask_np = mask_images.squeeze().cpu().numpy()  # No need to permute, just remove batch dimension

            # Normalize mask to be binary (0 or 1)
            mask_images = mask_images.unsqueeze(1)
            binary_mask = (mask_images > 0).float()  # Converts mask to 0 or 1
            print("binary_mask: ", binary_mask.shape)
            binary_mask = binary_mask.to(device)  # Make sure mask has same shape
            print("binary_mask: ", binary_mask.shape)
            # RGB processing
            rgb_images = torch.clamp(rgb_cropped.float(), min=0.0, max=255.0)
            rgb_images = rgb_images.to(device)
            rgb_images = rgb_images * binary_mask

            rgb_images_gt = (rgb_images.float() / 127.5) - 1

            # MASK processing
            mask_images_gt = mask_images.float() / 255.0
            mask_images_gt = mask_images_gt.permute(0, 3, 1, 2)
            mask_images_gt = mask_images_gt.to(device)

            print(rgb_images_gt.shape)

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
            x_logits, y_logits, z_logits, nocs_estimated, mask_estimated = generator(rgb_images_gt)
            nocs_estimated = ((nocs_estimated + 1 ) /2)

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

            nocs_estimated = nocs_estimated * binary_mask
            nocs_estimated_np = (nocs_estimated).squeeze().permute(1, 2, 0).cpu().numpy()  # Convert to HWC
            # Plotting
            fig, axs = plt.subplots(1, 4, figsize=(15, 5))
            
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
            axs[3].imshow(nocs_estimated_np)
            axs[3].set_title("NOCS Estimated")
            axs[3].axis('off')


            plt.show()

            # Optionally, plot the NOCS image using matplotlib
            plot_single_image(config.test_img_dir + "/rgb_images_gt", step, rgb_images_gt)
            plot_single_image(config.test_img_dir + "/nocs_estimated", step, nocs_estimated)
            # plot_single_image(config.test_img_dir + "/nocs_gt", step, nocs_images_normalized_gt)

            generate_and_visualize_point_cloud(nocs_estimated)
#             # generate_and_visualize_point_cloud(depth_images)

#             print("nocs_estimated: ", nocs_estimated.shape)
#             print("depth_images: ", depth_images.shape)

#             depth_images = depth_images/1000

#             dst = generate_point_cloud_from_depth(depth_images[0].numpy(), large_bboxes[0], camera_intrinsics).T
#             src = generate_point_cloud_from_nocs(nocs_estimated).T

#             print("src: ", src.shape)
#             print("dst: ", dst.shape)

#             src, dst, _ = filter_zero_points(src, dst)
#             dst, src, _ = filter_zero_points(dst, src)

#             # src =  src / 127.5 - 1

#             num_points_to_sample = 100

#             src_sampled, dst_sampled = sample_point_cloud(src, dst, num_points_to_sample)

#             pcd_src = create_open3d_point_cloud(src, [0, 0, 1])  # Blue for source
#             pcd_dst = create_open3d_point_cloud(dst, [1, 0, 0])  # Red for destination

#             pcd_src_sampled = create_open3d_point_cloud(src_sampled, [0, 0, 1])  # Blue for source
#             pcd_dst_sampled = create_open3d_point_cloud(dst_sampled, [1, 0, 0])  # Red for destination

#             print("Sampled src shape: ", src_sampled.shape)
#             print("Sampled dst shape: ", dst_sampled.shape)

#             # Populate the parameters
#             solver_params = teaserpp_python.RobustRegistrationSolver.Params()
#             solver_params.cbar2 = 1
#             solver_params.noise_bound = 0.01
#             solver_params.estimate_scaling = True
#             solver_params.rotation_estimation_algorithm = (
#                 teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
#             )
#             solver_params.rotation_gnc_factor = 1.4
#             solver_params.rotation_max_iterations = 100
#             solver_params.rotation_cost_threshold = 1e-12

#             # Create the TEASER++ solver and solve the registration problem
#             teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

#             teaserpp_solver.solve(src_sampled, dst_sampled)

#             # Get the solution
#             solution = teaserpp_solver.getSolution()
#             print("Solution is:", solution)

#             # Extract rotation, translation, and scale from the solution
#             R = solution.rotation
#             t = solution.translation
#             s = solution.scale

#             # Apply the estimated transformation to the source point cloud
#             src_transformed = s * np.matmul(R, src) + t.reshape(3, 1)

#             # Create Open3D point clouds for visualization
#             #src_transformed = ransac_registration(src_sampled, dst_sampled)
#             pcd_src_transformed = create_open3d_point_cloud(src_transformed, [0, 1, 0])  # Green

#             # Create line set to visualize correspondences
#             line_set = create_line_set(src_sampled, dst_sampled, [0, 1, 0])  # Green lines for correspondences
        
#             # Visualize the point clouds using Open3D
#             o3d.visualization.draw_geometries([pcd_src, pcd_dst, pcd_src_transformed, line_set],
#                                             window_name="TEASER++ Registration with PLY and NOCS",
#                                             width=1920, height=1080,
#                                             left=50, top=50,
#                                             point_show_normal=False)
            
#             estimated_bbox_corners = compute_3d_bounding_box(pcd_src_transformed.points)
#             print("estimated bbox corners: ", estimated_bbox_corners)
#             print()

#             iou = calculate_3d_iou(estimated_bbox_corners, ground_truth_bbox)
#             print("IOU BAAAABYYYYY: ", iou)

if __name__ == "__main__":
    args = parse_args()
    
    # Load the config file passed as argument
    config = load_config(args.config)
    
    # Call main with the loaded config
    main(config)