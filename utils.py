# parts of the code from: https://github.com/hughw19/NOCS_CVPR2019/blob/master/detect_eval.py

import sys, os

from torch.utils.data import Dataset, DataLoader

from imgaug import augmenters as iaa
import imgaug.augmenters as iaa  # noqa

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch 
import webdataset as wds
import open3d as o3d
import io
import PIL.Image

import argparse
import importlib.util
from torchvision import transforms
import json
import imageio

from skimage.transform import SimilarityTransform

class CustomDataset(Dataset):
    def __init__(self, coco_json_path, root_dir, image_size=128, augment=False, center_crop=True, is_depth=False):

        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment
        self.center_crop = center_crop
        self.is_depth = is_depth

        self.enlarge_factor = 1.5

        #self.images = {img['id']: img for img in self.coco_data['images']}
        self.data = self.coco_data['data']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_data = self.data[idx]
        frame_id = image_data['frame_id']
        scene_id = image_data['scene_id']

        predictions = image_data['predictions']
        gts = image_data['gts']

        rgb_filename = image_data['color_file_name']
        depth_filename = image_data['depth_file_name']

        # rgb_filename = img_info['file_name']
        # depth_filename = rgb_filename[:-4] + ".png"

        img_path = os.path.join(self.root_dir, rgb_filename)
        depth_path = os.path.join(self.root_dir, depth_filename)

        rgb_image = Image.open(img_path).convert("RGB")
        rgb_image = np.array(rgb_image)

        with open(depth_path, 'rb') as f:
            depth_bytes = f.read()
        
        depth_image = imageio.imread(depth_bytes).astype(np.float32) / 1000

        rgb_crops = []
        mask_crops = []
        bboxes = []
        metadatas = []
        category_names = []
        category_ids = []
        masks = []
        scores = []

        for prediction in predictions:
            # ab hier dann loopy loopes
            category_names.append(self.categories[prediction['category_id']])
            category_ids.append(prediction['category_id'])
            scores.append(prediction['score'])

            bbox = np.array(prediction['bbox'], dtype=int)

            mask = self.decode_segmentation(prediction['segmentation'], image_data['width'], image_data['height'])
            mask = 1 - mask

            # Apply cropping and preprocessing
            enlarged_bbox = get_enlarged_bbox(bbox, rgb_image.shape, bbox_scaler=self.enlarge_factor)

            rgb_crop, metadata = crop_and_resize(rgb_image, enlarged_bbox, bbox, target_size=self.image_size, interpolation=Image.BILINEAR)
            mask_crop, metadata = crop_and_resize(mask, enlarged_bbox, bbox, target_size=self.image_size, interpolation=Image.NEAREST)
            
            rgb_crops.append(torch.tensor(rgb_crop, dtype=torch.uint8))
            masks.append(torch.tensor(mask, dtype=torch.uint8))
            mask_crops.append(torch.tensor(mask_crop, dtype=torch.uint8))
            metadatas.append(metadata)
            bboxes.append(bbox)

        return {
            "frame_id": frame_id,
            "scene_id": scene_id,
            "rgb": transforms.ToTensor()(rgb_image),  # For later post-processing
            "depth": transforms.ToTensor()(depth_image),
            "masks": masks,
            "rgb_crops": rgb_crops,
            "mask_crops": mask_crops,
            "bboxes": bboxes,
            "metadatas": metadatas,
            "category_names": category_names,
            "category_ids": category_ids,
            "scores": scores,
            "gts": gts,
        }

    def decode_segmentation(self, rle, width, height):
        """
        Decodes COCO-style RLE to a binary mask.
        
        :param rle: Dictionary with 'counts' (RLE-encoded values) and 'size' (height, width).
        :param width: Width of the output binary mask.
        :param height: Height of the output binary mask.
        :return: Decoded binary mask as a numpy array.
        """
        # Initialize an empty 1D mask with all zeros
        mask = np.zeros(width * height, dtype=np.uint8)

        # Unroll the RLE counts into mask positions
        rle_counts = rle['counts']
        current_position = 0

        # Apply the counts as mask segments
        for i in range(len(rle_counts)):
            run_length = rle_counts[i]
            if i % 2 == 0:
                # For even indices, set the corresponding pixels to 1 (foreground)
                mask[current_position:current_position + run_length] = 1
            # Update position
            current_position += run_length

        # Reshape the flat mask into the 2D binary mask (height, width) and return
        return mask.reshape((height, width), order="F")  # Order "F" to match column-wise storage
    
class COCODataset(Dataset):
    def __init__(self, coco_json_path, root_dir, image_size=128, augment=False, center_crop=True, is_depth=False):

        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)
        
        self.root_dir = root_dir
        self.image_size = image_size
        self.augment = augment
        self.center_crop = center_crop
        self.is_depth = is_depth

        self.enlarge_factor = 1.5

        self.images = {img['id']: img for img in self.coco_data['images']}
        self.annotations = self.coco_data['annotations']
        self.gts = self.coco_data['gts']
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_info = self.images[annotation['image_id']]
        category_name = self.categories[annotation['category_id']]
        category_id = annotation['category_id']
        gts = self.gts[annotation['image_id']]

        # rgb_filename = img_info['file_name']
        # depth_filename = "depth" + rgb_filename[3:]

        # rgb_filename = img_info['file_name']
        # depth_filename = rgb_filename

        rgb_filename = img_info['color_file_name']
        depth_filename = img_info['depth_file_name']

        # rgb_filename = img_info['file_name']
        # depth_filename = rgb_filename[:-4] + ".png"

        img_path = os.path.join(self.root_dir, rgb_filename)
        depth_path = os.path.join(self.root_dir, depth_filename)

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        with open(depth_path, 'rb') as f:
            depth_bytes = f.read()
        
        depth_image = imageio.imread(depth_bytes).astype(np.float32) / 1000

        bbox = np.array(annotation['bbox'], dtype=int)

        mask = self.decode_segmentation(annotation['segmentation'], img_info['width'], img_info['height'])
        mask = 1 - mask

        # Apply cropping and preprocessing
        enlarged_bbox = get_enlarged_bbox(bbox, image.shape, bbox_scaler=self.enlarge_factor)

        cropped_image, metadata = crop_and_resize(image, enlarged_bbox, bbox, target_size=self.image_size, interpolation=Image.BILINEAR)
        cropped_mask, metadata = crop_and_resize(mask, enlarged_bbox, bbox, target_size=self.image_size, interpolation=Image.NEAREST)
        
        return {
            "rgb_crop": cropped_image,
            "mask_crop": cropped_mask,
            "rgb": transforms.ToTensor()(image),  # For later post-processing
            "depth": transforms.ToTensor()(depth_image),
            "bbox": bbox,
            "mask": torch.tensor(mask, dtype=torch.uint8),
            "metadata": metadata,
            "category_name": category_name,
            "category_id": category_id,
            "gts": gts,
        }

    def decode_segmentation(self, rle, width, height):
        """
        Decodes COCO-style RLE to a binary mask.
        
        :param rle: Dictionary with 'counts' (RLE-encoded values) and 'size' (height, width).
        :param width: Width of the output binary mask.
        :param height: Height of the output binary mask.
        :return: Decoded binary mask as a numpy array.
        """
        # Initialize an empty 1D mask with all zeros
        mask = np.zeros(width * height, dtype=np.uint8)

        # Unroll the RLE counts into mask positions
        rle_counts = rle['counts']
        current_position = 0

        # Apply the counts as mask segments
        for i in range(len(rle_counts)):
            run_length = rle_counts[i]
            if i % 2 == 0:
                # For even indices, set the corresponding pixels to 1 (foreground)
                mask[current_position:current_position + run_length] = 1
            # Update position
            current_position += run_length

        # Reshape the flat mask into the 2D binary mask (height, width) and return
        return mask.reshape((height, width), order="F")  # Order "F" to match column-wise storage
    
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

def add_loss(est_points_batch, gt_points_batch):
    """Calculate the ADD score for a batch of point clouds using GPU."""

    distances = torch.cdist(est_points_batch, gt_points_batch, p=2)
    min_distances, _ = distances.min(dim=2)
    add_scores = min_distances.mean(dim=1)

    return add_scores.mean()

def apply_rotation(points, rotation_matrix):
    """
    Apply a rotation matrix to a batch of points.
    
    :param points: Tensor of shape (batch_size, 1000, 3)
    :param rotation_matrix: Tensor of shape (batch_size, 3, 3)
    :return: Transformed points of shape (batch_size, 1000, 3)
    """
    # Ensure the correct shapes for batch matrix multiplication
    # points: (batch_size, 1000, 3) -> (batch_size, 3, 1000)
    points_transposed = points.transpose(1, 2)
    
    # Perform batch matrix multiplication: rotated_points = rotation_matrix @ points_transposed
    rotated_points = torch.bmm(rotation_matrix, points_transposed)  # (batch_size, 3, 1000)
    
    # Transpose back to (batch_size, 1000, 3)
    return rotated_points.transpose(1, 2)

def preload_pointclouds(models_root, num_categories):
    pointclouds = {}

    for category in range(1, num_categories + 1):  # Subfolders 1 to 10
        category_path = os.path.join(models_root, str(category))

        for obj_file in os.listdir(category_path):
            if obj_file.endswith(".ply"):
                obj_name = os.path.splitext(obj_file)[0]
                pointcloud_file = os.path.join(category_path, obj_file)
                
                # Load the point cloud
                pc_gt = o3d.io.read_point_cloud(pointcloud_file)
                gt_points = np.asarray(pc_gt.points)
                
                # Store point cloud in the dictionary using category and name
                pointclouds[(str(category), obj_name)] = gt_points
                
    return pointclouds

def make_log_dirs(dir_list):
    for directory in dir_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

def custom_collate_fn(batch):
    rgb_batch = torch.stack([torch.tensor(item[0]) for item in batch])
    normals_batch = torch.stack([torch.tensor(item[1]) for item in batch])
    mask_batch = torch.stack([torch.tensor(item[2]) for item in batch])
    nocs_batch = torch.stack([torch.tensor(item[3]) for item in batch])
    info_batch = [item[4] for item in batch]

    return {
        'rgb': rgb_batch,
        'normals': normals_batch,
        'mask': mask_batch,
        'nocs': nocs_batch,
        'info': info_batch,
    }

def collate_fn_val(batch):
    frame_id = [(item['frame_id']) for item in batch]
    scene_id = [(item['scene_id']) for item in batch]
    rgb_images = torch.stack([torch.tensor(item['rgb']) for item in batch])
    depth_images = torch.stack([torch.tensor(item['depth']) for item in batch])
    mask_images = [(item['masks']) for item in batch]
    rgb_crops = [(item['rgb_crops']) for item in batch]
    mask_crops = [(item['mask_crops']) for item in batch]
    bboxes = [(item['bboxes']) for item in batch]
    metadatas = [(item['metadatas']) for item in batch]
    category_names = [(item['category_names']) for item in batch]
    category_ids = [(item['category_ids']) for item in batch]
    scores = [(item['scores']) for item in batch]    
    gts = [(item['gts']) for item in batch]

    return {
        "frame_id": frame_id,
        "scene_id": scene_id,
        "rgb": rgb_images,
        "depth": depth_images,
        "mask": mask_images,
        "rgb_crops": rgb_crops,
        "mask_crops": mask_crops,
        "bboxes": bboxes,
        "metadatas": metadatas,
        "category_names": category_names,
        "category_ids": category_ids,
        "scores": scores,
        "gts": gts,
    }

def collate_fn(batch):
    rgb_images = torch.stack([torch.tensor(item['rgb']) for item in batch])
    depth_images = torch.stack([torch.tensor(item['depth']) for item in batch])
    rgb_cropped = torch.stack([torch.tensor(item['rgb_crop']) for item in batch])
    mask_cropped = torch.stack([torch.tensor(item['mask_crop']) for item in batch])
    mask_images = torch.stack([torch.tensor(item['mask']) for item in batch])
    bboxes = torch.stack([torch.tensor(item['bbox']) for item in batch])
    metadata = [(item['metadata']) for item in batch]
    category_name = [(item['category_name']) for item in batch]
    category_id = [(item['category_id']) for item in batch]
    gts = [(item['gts']) for item in batch]

    return {
        "rgb": rgb_images,
        "depth": depth_images,
        "rgb_crop": rgb_cropped,
        "mask_crop": mask_cropped,
        "mask": mask_images,
        "bbox": bboxes,
        "metadata": metadata,
        "category_name": category_name,
        "category_id": category_id,
        "gts": gts,
    }

def post_process_crop_to_original(crop, original_image, bbox, image_size):
    # Resize crop back to bounding box size
    x_min, y_min, width, height = map(int, bbox)
    crop_resized = Image.fromarray(crop).resize((width, height), Image.BILINEAR)

    # Place resized crop back on original image
    original_image = np.array(original_image)
    original_image[y_min:y_min + height, x_min:x_min + width] = np.array(crop_resized)
    return original_image

def custom_collate_fn_test(batch):
    rgb_batch = torch.stack([torch.tensor(item[0]) for item in batch])
    mask_batch = torch.stack([torch.tensor(item[1]) for item in batch])
    nocs_batch = torch.stack([torch.tensor(item[2]) for item in batch])
    depth_batch = torch.stack([torch.tensor(item[3]) for item in batch])
    info_batch = [item[4] for item in batch]
    # info_batch = {i: item[3] for i, item in enumerate(batch)}

    return {
        'rgb': rgb_batch,
        'mask': mask_batch,
        'nocs': nocs_batch,
        'metric_depth': depth_batch,
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
    
def load_depth_image(depth_bytes):
    # Custom logic to load depth images from raw bytes
    # Example: assuming depth is stored as 16-bit grayscale PNG, modify based on your format
    depth_image = Image.open(io.BytesIO(depth_bytes)).convert("I;16")  # Load as 16-bit integer
    depth_array = np.array(depth_image, dtype=np.float32)
    return depth_array  # Return depth in the desired format (e.g., float32 metric depth)

def load_image(data):
    with io.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.convert("rgb".upper())
        img = img.convert("RGB")
        return img

def create_webdataset(dataset_paths, size=128, shuffle_buffer=1000, augment=False, center_crop=False, class_name=None):

    dataset = wds.WebDataset(dataset_paths, shardshuffle=True) \
        .decode() \
        .shuffle(shuffle_buffer, initial=size) \
        .to_tuple("rgb.png", "normals.png", "mask_visib.png", "nocs.png", "info.json") \
        .map_tuple( 
            lambda rgb: preprocess(load_image(rgb), size, Image.BICUBIC, augment=augment, center_crop=center_crop), 
            lambda normals: preprocess(load_image(normals), size, Image.NEAREST, center_crop=center_crop), 
            lambda mask: preprocess(load_image(mask), size, Image.NEAREST, center_crop=center_crop), 
            lambda nocs: preprocess(load_image(nocs), size, Image.NEAREST, center_crop=center_crop), 
            lambda info: info) \
        .select(lambda sample: (class_name is None) or (sample[3].get('category_id') == class_name)) # Adjust index for 'info.json'

    return dataset

def create_webdataset_test(dataset_paths, size=128, shuffle_buffer=1000, augment=False, center_crop=False, class_name=None):

    dataset = wds.WebDataset(dataset_paths, shardshuffle=True) \
        .decode() \
        .shuffle(shuffle_buffer, initial=size) \
        .to_tuple("rgb.png", "mask_visib.png", "nocs.png", "metric_depth.png", "info.json") \
        .map_tuple( 
            lambda rgb: preprocess(load_image(rgb), size, Image.BICUBIC, augment=augment, center_crop=center_crop), 
            lambda mask: preprocess(load_image(mask), size, Image.NEAREST, center_crop=center_crop), 
            lambda nocs: preprocess(load_image(nocs), size, Image.NEAREST, center_crop=center_crop), 
            lambda metric_depth: preprocess(load_depth_image(metric_depth), size, Image.NEAREST, center_crop=center_crop, is_depth=True),  # Preprocess depth after loading
            lambda info: info) \
        .select(lambda sample: (class_name is None) or (sample[4].get('category_id') == class_name))  # Adjust index for 'info.json'

    return dataset

def preprocess(image, size, interpolation, augment=False, center_crop=False, is_depth=False):
    
    if is_depth:
        # Depth image (float32) processing
        img_array = image.astype(np.float32)  # Ensure it's a float32 array
    else:
        # Regular image (uint8) processing
        img_array = np.array(image).astype(np.uint8)

    h, w = img_array.shape[0], img_array.shape[1]

    # Center crop logic
    if center_crop:
        original_size = int(min(h, w) / 1.5)
        crop_xmin = (w - original_size) // 2
        crop_xmax = crop_xmin + original_size
        crop_ymin = (h - original_size) // 2
        crop_ymax = crop_ymin + original_size
        crop_xmin = max(crop_xmin - 5, 0)
        crop_xmax = min(crop_xmax + 5, w)
        crop_ymin = max(crop_ymin - 5, 0)
        crop_ymax = min(crop_ymax + 5, h)
        img_array = img_array[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
    else:
        crop = min(h, w)
        img_array = img_array[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]

    # Augmentation logic (only for non-depth images)
    if augment and not is_depth:
        prob = 0.8
        seq_syn = iaa.Sequential([
                                    iaa.Sometimes(0.3 * prob, iaa.CoarseDropout(p=0.2, size_percent=0.05)),
                                    iaa.Sometimes(0.5 * prob, iaa.GaussianBlur((0., 3.))),
                                    iaa.Sometimes(0.3 * prob, iaa.pillike.EnhanceSharpness(factor=(0., 50.))),
                                    iaa.Sometimes(0.3 * prob, iaa.pillike.EnhanceContrast(factor=(0.2, 50.))),
                                    iaa.Sometimes(0.5 * prob, iaa.pillike.EnhanceBrightness(factor=(0.1, 6.))),
                                    iaa.Sometimes(0.3 * prob, iaa.pillike.EnhanceColor(factor=(0., 20.))),
                                    iaa.Sometimes(0.5 * prob, iaa.Add((-25, 25), per_channel=0.3)),
                                    iaa.Sometimes(0.3 * prob, iaa.Invert(0.2, per_channel=True)),
                                    iaa.Sometimes(0.5 * prob, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
                                    iaa.Sometimes(0.5 * prob, iaa.Multiply((0.6, 1.4))),
                                    iaa.Sometimes(0.1 * prob, iaa.AdditiveGaussianNoise(scale=10, per_channel=True)),
                                    iaa.Sometimes(0.5 * prob, iaa.contrast.LinearContrast((0.5, 2.2), per_channel=0.3)),
                                    iaa.Sometimes(0.5 * prob, iaa.LinearContrast((0.5, 2.2), per_channel=0.3))
                                    ], random_order=True)
        img_array = seq_syn.augment_image(img_array)

    image = Image.fromarray(img_array)
    image = image.resize((size, size), resample=interpolation)

    if is_depth:
        img_array = np.array(image).astype(np.float32)
    else:
        img_array = np.array(image).astype(np.uint8)   

    return img_array

def normalize_quaternion(q):
    # Normalize the quaternion across the batch
    norm = torch.norm(q, dim=1, keepdim=True)
    return q / norm  # Normalize quaternion

def setup_environment(gpu_id):
    if len(sys.argv) != 2:
        print("Usage: python3 train.py <gpu_id>")
        sys.exit()

    if gpu_id == '-1':
        gpu_id = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

def plot_progress_imgs(imgfn, rgb_images, normal_images, nocs_images_normalized_gt, nocs_estimated, mask_images, config):
    # Print value ranges for each input
    # print(f"RGB Images - Min: {torch.min(rgb_images):.4f}, Max: {torch.max(rgb_images):.4f}")
    # print(f"NOCS GT - Min: {torch.min(nocs_images_normalized_gt):.4f}, Max: {torch.max(nocs_images_normalized_gt):.4f}")
    # print(f"NOCS Estimated - Min: {torch.min(nocs_estimated):.4f}, Max: {torch.max(nocs_estimated):.4f}")
    # print(f"Mask Images - Min: {torch.min(mask_images):.4f}, Max: {torch.max(mask_images):.4f}")
    # print(f"Binary Mask Images - Min: {torch.min(mask_images_binary):.4f}, Max: {torch.max(mask_images_binary):.4f}")
    # print(f"Ground Truth Mask Images - Min: {torch.min(mask_images_gt):.4f}, Max: {torch.max(mask_images_gt):.4f}")

    # print()
    _,ax = plt.subplots(config.num_imgs_log,6,figsize=(10,20))
    # rgb_images, nocs_images_normalized_gt, nocs_estimated, mask_images, masks_estimated, binary_masks
    # Define column titles
    col_titles = ['RGB Image', 'Normals GT', 'NOCS GT', 'NOCS Estimated', 'Mask GT']
    
    # Add column titles
    for i, title in enumerate(col_titles):
        ax[0, i].set_title(title, fontsize=12)

    for i in range(config.num_imgs_log):
        ax[i, 0].imshow(((rgb_images[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 1].imshow(((normal_images[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 2].imshow(((nocs_images_normalized_gt[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 3].imshow(((nocs_estimated[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 4].imshow(((mask_images[i])).detach().cpu().numpy().transpose(1, 2, 0))

        # # Plot rotation arrows
        # rotation_matrix = rot_estimated_R[i].detach().cpu().numpy()  # Get the rotation matrix for the current image

        # # Define arrow directions for x, y, z axes
        # arrow_directions = {
        #     'x': np.array([1, 0, 0]),  # X-axis direction
        #     'y': np.array([0, 1, 0]),  # Y-axis direction
        #     'z': np.array([0, 0, 1])   # Z-axis direction
        # }

        # # Get the start point (e.g., center of the image in normalized coordinates)
        # start_point = np.array([0.5, 0.5])  # Center of the image

        # # Iterate over each arrow direction and plot
        # for key, direction in arrow_directions.items():
        #     # Transform the arrow direction using the rotation matrix
        #     transformed_arrow = rotation_matrix @ direction
            
        #     # Calculate end point based on the transformed arrow
        #     end_point = start_point + (transformed_arrow[:2] * config.arrow_length)  # Only use x and y for 2D
            
        #     # Plot the arrow
        #     ax[i, 0].quiver(
        #         start_point[0] * rgb_images[i].shape[2], start_point[1] * rgb_images[i].shape[1],
        #         (end_point[0] - start_point[0]) * rgb_images[i].shape[2], 
        #         (end_point[1] - start_point[1]) * rgb_images[i].shape[1],
        #         angles='xy', scale_units='xy', scale=1, color=config.arrow_colors[key], width=config.arrow_width
        #     )

    plt.tight_layout()
    plt.savefig(imgfn, dpi=300)
    plt.close()

def plot_single_image(output_dir, iteration, nocs_estimated, plot_image=False):

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare filename with padded iteration (e.g., 00000001.png)
    imgfn = os.path.join(output_dir, f'{iteration:08d}.png')

    # Plot NOCS estimated map (scale to 0-1 for visualization)
    #plt.imshow(((nocs_estimated[0] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
    plt.imshow(nocs_estimated)
    plt.axis('off')  # Turn off axis for cleaner output
    if plot_image: plt.show()

    # Save the figure
    plt.savefig(imgfn, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

# ACHTUNG!!!! hier unbedingt bbox sache in ordnung bringen!!!!
def crop_and_resize(img, enlarged_bbox, original_bbox, target_size=128, interpolation=Image.NEAREST):
    """
    Crop, pad, and resize the image based on the provided enlarged bounding box.

    Args:
        img (numpy array): Input image.
        enlarged_bbox (tuple): Coordinates of the enlarged bounding box (crop_ymin, crop_xmin, crop_ymax, crop_xmax).
        target_size (int): The size to resize the cropped image to.
        interpolation (Image interpolation method): Interpolation method to be used for resizing.

    Returns:
        numpy array: Cropped and resized image.
    """
    crop_xmin, crop_ymin, crop_xmax, crop_ymax = enlarged_bbox
    orig_xmin, orig_ymin, orig_xmax, orig_ymax = original_bbox

    enlarged_size = max(crop_ymax - crop_ymin, crop_xmax - crop_xmin)
    
    # Initialize the cropped image with zeros
    if img.ndim == 3:
        cropped_img = np.zeros((enlarged_size, enlarged_size, img.shape[2]), dtype=img.dtype)
    else:
        cropped_img = np.zeros((enlarged_size, enlarged_size), dtype=img.dtype)
    
    # Calculate offsets to center the cropped area
    y_offset = (enlarged_size - (crop_ymax - crop_ymin)) // 2
    x_offset = (enlarged_size - (crop_xmax - crop_xmin)) // 2
    
    # Crop and pad the image
    if img.ndim == 3:
        cropped_img[y_offset:y_offset + (crop_ymax - crop_ymin), x_offset:x_offset + (crop_xmax - crop_xmin)] = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax, :]
    else:
        cropped_img[y_offset:y_offset + (crop_ymax - crop_ymin), x_offset:x_offset + (crop_xmax - crop_xmin)] = img[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

    # Resize the image to the target size
    cropped_img_pil = Image.fromarray(cropped_img)
    cropped_img_pil = cropped_img_pil.resize((target_size, target_size), interpolation)
    cropped_img = np.array(cropped_img_pil)
    
    # Store metadata for restoring the original bounding box
    scale_factor = target_size / enlarged_size
    original_offset_x = orig_xmin - crop_xmin + x_offset
    original_offset_y = orig_ymin - crop_ymin + y_offset
    metadata = {
        "enlarged_bbox": enlarged_bbox,
        "scale_factor": scale_factor,
        "original_bbox_size": (orig_xmax - orig_xmin, orig_ymax - orig_ymin),
        "original_offset": (original_offset_x, original_offset_y),
    }

    # If the image has multiple channels, transpose for compatibility if needed
    if cropped_img.ndim == 3:
        cropped_img = np.transpose(cropped_img, (2, 0, 1))

    return cropped_img, metadata

def get_enlarged_bbox(bbox, img_shape, bbox_scaler=1.5):
    """
    Calculate enlarged bounding box coordinates based on the input bounding box and scaling factor.
    
    Args:
        bbox (tuple or list): Original bounding box coordinates (xmin, ymin, xmax, ymax).
        img_shape (tuple): Shape of the image (height, width, channels).
        bbox_scaler (float): Scaling factor for enlarging the bounding box.

    Returns:
        tuple: Coordinates of the enlarged bounding box (crop_xmin, crop_ymin, crop_xmax, crop_ymax).
    """
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    center_x = (bbox[2] + bbox[0]) // 2
    center_y = (bbox[3] + bbox[1]) // 2
        
    # Determine the size of the enlarged square bounding box
    enlarged_size = int(max(bbox_width, bbox_height) * bbox_scaler)
    print("enlarged_size: ", enlarged_size)

    # Calculate the coordinates of the enlarged bounding box, clamping within image boundaries
    crop_xmin = max(center_x - enlarged_size // 2, 0)
    crop_xmax = min(center_x + enlarged_size // 2, img_shape[1])
    crop_ymin = max(center_y - enlarged_size // 2, 0)
    crop_ymax = min(center_y + enlarged_size // 2, img_shape[0])

    return np.array([crop_xmin, crop_ymin, crop_xmax, crop_ymax])

def restore_original_bbox_crop(cropped_resized_img, metadata, interpolation=Image.NEAREST):
    """
    Restore the original bounding box crop from the resized cropped image and metadata.
    
    Args:
        cropped_resized_img (numpy array): Cropped and resized image.
        metadata (dict): Metadata containing 'enlarged_bbox', 'scale_factor', 'original_bbox_size', and 'original_offset'.

    Returns:
        numpy array: Cropped image at the original bounding box size.
    """
    scale_factor = metadata['scale_factor']
    original_bbox_size = metadata['original_bbox_size']
    original_offset = metadata['original_offset']

    # Resize the cropped image back to the enlarged bounding box dimensions
    enlarged_size = int(cropped_resized_img.shape[1] / scale_factor)

    cropped_img_pil = Image.fromarray(cropped_resized_img)
    restored_enlarged_img = cropped_img_pil.resize((enlarged_size, enlarged_size), interpolation)
    restored_enlarged_img = np.array(restored_enlarged_img)

    # Extract the original bounding box area using offsets
    original_offset_x, original_offset_y = original_offset
    if restored_enlarged_img.ndim == 3:
        original_bbox_crop = restored_enlarged_img[
            original_offset_y:original_offset_y + original_bbox_size[1], 
            original_offset_x:original_offset_x + original_bbox_size[0], 
            :
        ]
    else:
        original_bbox_crop = restored_enlarged_img[
            original_offset_y:original_offset_y + original_bbox_size[1], 
            original_offset_x:original_offset_x + original_bbox_size[0]
        ]

    return original_bbox_crop

def overlay_nocs_on_rgb(full_scale_rgb, nocs_image, mask_image, bbox):
    """
    Overlays the masked NOCS image onto the full-scale RGB image.

    Args:
        full_scale_rgb (numpy array): Full-scale RGB image (H, W, C).
        nocs_image (numpy array): NOCS image already cropped and resized to the original bounding box size (h, w, 3).
        mask_image (numpy array): Mask image matching the size of the NOCS image (h, w), binary mask.
        metadata (dict): Metadata containing the position (x, y) to place the NOCS image on the full RGB.

    Returns:
        numpy array: Full-scale RGB image with the masked NOCS overlay.
    """
    # Step 1: Mask the NOCS image using the mask image to remove the background
    binary_mask = (mask_image > 0).astype(np.uint8)
    masked_nocs = nocs_image.copy()
    masked_nocs[binary_mask == 0] = 0

    # Step 2: Get the original bounding box location on the full-scale RGB image from metadata
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

    overlay_image = full_scale_rgb.copy()
    
    # Only update pixels where mask is non-zero
    overlay_region = overlay_image[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]
    overlay_image[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax] = np.where(
        mask_image[:, :, np.newaxis] > 0, masked_nocs, overlay_region
    )

    return overlay_image

def paste_mask_on_black_canvas(base_image, mask_image, bbox):

    canvas_shape = base_image.shape
    if base_image.ndim == 2:  # Single-channel image
        canvas_shape = (*canvas_shape, 1)  # Add a channel dimension
    canvas = np.zeros((canvas_shape[0], canvas_shape[1],1), dtype=base_image.dtype)

    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

    if mask_image.ndim == 2:  # Single-channel mask
        mask_image = np.expand_dims(mask_image, axis=-1)  # Match dimensions

    # Only update pixels where mask is non-zero
    canvas[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax] = mask_image

    if base_image.ndim == 2:
        canvas = np.squeeze(canvas, axis=-1)

    return canvas

def paste_nocs_on_black_canvas(base_image, mask_image, bbox):

    canvas_shape = base_image.shape
    canvas = np.zeros((canvas_shape[0], canvas_shape[1], 3), dtype=base_image.dtype)

    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

    canvas[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax] = mask_image

    return canvas

def combine_images_overlapping(images):
    if not images:
        raise ValueError("The list of images is empty.")
    
    # Ensure all images have the same shape
    img_height, img_width, img_channels = images[0].shape
    for img in images:
        if img.shape != (img_height, img_width, img_channels):
            raise ValueError("All images must have the same dimensions.")
    
    # Create a blank black canvas
    combined_image = np.zeros_like(images[0], dtype=np.uint8)
    
    # Overlay images by copying non-black pixels
    for img in images:
        mask = np.any(img != [0, 0, 0], axis=-1)  # Find where the image is not black
        combined_image[mask] = img[mask]
    
    return combined_image

def teaserpp_solve(src, dst, config):
    import teaserpp_python

    # Populate the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = config.noise_bound
    solver_params.estimate_scaling = True
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = config.rotation_max_iterations
    solver_params.rotation_cost_threshold = config.rotation_cost_threshold

    # Create the TEASER++ solver and solve the registration problem
    teaserpp_solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    teaserpp_solver.solve(src, dst)

    # Get the solution
    solution = teaserpp_solver.getSolution()
    inliers = teaserpp_solver.getRotationInliers()
    #print("Solution is:", solution)

    # Extract rotation, translation, and scale from the solution
    R = solution.rotation
    t = solution.translation
    s = solution.scale

    return R, t, s, len(inliers)

def backproject(depth, intrinsics, instance_mask=None):
    intrinsics = np.array([[intrinsics['fx'], 0, intrinsics['cx']], [0, intrinsics['fy'], intrinsics['cy']],[0,0,1]])
    intrinsics_inv = np.linalg.inv(intrinsics)

    #non_zero_mask = np.logical_and(depth > 0, depth < 5000)
    non_zero_mask = (depth > 0)

    if instance_mask is not None:
        instance_mask = np.squeeze(instance_mask)
        final_instance_mask = np.logical_and(instance_mask, non_zero_mask)
    else:
        final_instance_mask = np.ones_like(depth, dtype=bool)

    idxs = np.where(final_instance_mask)
    grid = np.array([idxs[1], idxs[0]])

    length = grid.shape[1]
    ones = np.ones([1, length])
    uv_grid = np.concatenate((grid, ones), axis=0) # [3, num_pixel]

    xyz = intrinsics_inv @ uv_grid # [3, num_pixel]
    xyz = np.transpose(xyz) #[num_pixel, 3]

    z = depth[idxs[0], idxs[1]]
    pts = xyz * z[:, np.newaxis]/xyz[:, -1:]
    # pts[:, 0] = -pts[:, 0]
    # pts[:, 1] = -pts[:, 1]

    return pts, idxs

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
    pcd.points = o3d.utility.Vector3dVector(points.T)
    pcd.paint_uniform_color(color)
    return pcd

def show_pointcloud(points, axis_size=1.0):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, axis])

def filter_points(src, dst, filter_value=0, tolerance=1/255):
    mask = ~(np.max(np.abs(src - filter_value), axis=0) <= tolerance)
    indices_removed = np.where(~mask)[0]
    src_filtered = src[:, mask]
    dst_filtered = dst[:, mask]
    
    return src_filtered, dst_filtered, indices_removed

def rotate_transform_matrix_180_z(transform_matrix):
    # rotation_matrix_180_z = np.array([
    #     [-1, 0, 0],
    #     [0, -1, 0],
    #     [0, 0, 1]
    # ])

    # rotation_part = transform_matrix[:3, :3]
    # translation_part = transform_matrix[:3, 3]

    # new_rotation = np.dot(rotation_matrix_180_z, rotation_part)

    # new_translation = np.dot(rotation_matrix_180_z, translation_part)

    # transformed_matrix = np.eye(4)
    # transformed_matrix[:3, :3] = new_rotation
    # transformed_matrix[:3, 3] = new_translation
    
    z_180_RT = np.zeros((4, 4), dtype=np.float32)
    z_180_RT[:3, :3] = np.diag([-1, -1, 1])
    z_180_RT[3, 3] = 1

    RT = np.matmul(z_180_RT,transform_matrix)
    
    return RT

def remove_duplicate_pixels(img_array):
    processed_array = np.zeros_like(img_array)
    
    flat_array = img_array.reshape(-1, 3)
    _, unique_indices = np.unique(flat_array, axis=0, return_index=True)

    processed_array.reshape(-1, 3)[unique_indices] = flat_array[unique_indices]
    
    return processed_array