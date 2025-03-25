# parts of the code from: https://github.com/hughw19/NOCS_CVPR2019/blob/master/detect_eval.py

import sys, os

from torch.utils.data import Dataset, DataLoader

from imgaug import augmenters as iaa
import imgaug.augmenters as iaa  # noqa
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch 
import webdataset as wds
import open3d as o3d
import io
import PIL.Image

from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale
from typing import List, Tuple

from transformers import AutoImageProcessor, AutoModel
from torchvision import transforms
import torch
import torch.nn.functional as F
import torchvision.transforms as T

import argparse
import importlib.util
from torchvision import transforms
import json
import imageio
from typing import List, Tuple
import pickle

from skimage.transform import SimilarityTransform

class DinoFeatures:
    def __init__(self) -> None:
        device: str = "cuda"
        self.device = device

        self.dino_model = AutoModel.from_pretrained("./dinov2-small")
        self.patch_size = self.dino_model.config.patch_size
        self._feature_dim = self.dino_model.config.hidden_size
        self.dino_model.to(device)
        self.dino_model.eval()    

        self.upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=False).to(device)

        # Normalization transform based on ImageNet
        # self._transform = transforms.Compose([
        #     self.norm
        # ])

        self.unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # PCA model from sklearn
        self.diffnocs_pca = pickle.load(open("../nocs_renderer/pca6.pkl", "rb"))
        self.torch_pca = TorchPCA(n_components=6, device=self.device)

        self._transform = transforms.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            self.norm,
        ])

    def get_featup_features(self, rgb_image, mask_image):
        image_tensor = self._transform(rgb_image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        hr_feats = self.upsampler(image_tensor)

        mask_tensor = F.interpolate(torch.Tensor(mask_image).unsqueeze(0).unsqueeze(0), size=(256, 256), mode='nearest')
        featup_pca_features = self.featup_pca(hr_feats[0].unsqueeze(0), mask_tensor, dim=6)
        featup_pca_features = np.clip(featup_pca_features, 0, 1)
        featup_pca_resized = cv2.resize(featup_pca_features, (160, 160), interpolation=cv2.INTER_NEAREST)

        return featup_pca_resized

    def featup_pca(self, image_feats, masks, dim=3):
        #pca = PCA(n_components=6)

        features = np.transpose(image_feats.detach().cpu().numpy(), (0, 2, 3, 1)) 
        num_maps, map_w, map_h = features.shape[0:3]
        masks = masks[0][0].detach().cpu().numpy()[None]
        masked_features = features[masks.astype(bool)]

        masked_features_pca = self.torch_pca.fit(torch.tensor(masked_features, dtype=torch.float32).to(self.device)).transform(torch.tensor(masked_features, dtype=torch.float32).to(self.device)).numpy()

        #masked_features_pca = pca.fit_transform(masked_features)
        masked_features_pca = minmax_scale(masked_features_pca)

        # Initialize images for PCA reduced features
        features_pca_reshaped = np.zeros((num_maps, map_w, map_h, masked_features_pca.shape[-1]))

        # Fill in the PCA results only at the masked regions
        features_pca_reshaped[masks.astype(bool)] = masked_features_pca

        return features_pca_reshaped[0]

    def get_features(
        self,
        images: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert len(set(image.shape for image in images)) == 1, "Not all images have same shape"
        im_w, im_h = images[0].shape[0:2]
        assert (
            im_w % self.patch_size == 0 and im_h % self.patch_size == 0
        ), "Both width and height of the image must be divisible by patch size"

        image_array = np.stack(images) / 255.0
        input_tensor = torch.Tensor(np.transpose(image_array, [0, 3, 2, 1]))
        input_tensor = self.norm(input_tensor).to(self.device)

        with torch.no_grad():
            outputs = self.dino_model(input_tensor, output_hidden_states=True)
        
            # CLS token is first then patch tokens
            class_tokens = outputs.last_hidden_state[:, 0, :]
            patch_tokens = outputs.last_hidden_state[:, 1:, :]

            if patch_tokens.is_cuda:
                patch_tokens = patch_tokens.cpu()
                class_tokens = class_tokens.cpu()

            patch_tokens = patch_tokens.detach().numpy()
            class_tokens = class_tokens.detach().numpy()

            all_patches = patch_tokens.reshape(
                [-1, im_w // self.patch_size, im_h // self.patch_size, self._feature_dim]
            )
        all_patches = np.transpose(all_patches, (0, 2, 1, 3))

        return all_patches, class_tokens

    def apply_pca(self, features: np.ndarray, masks: np.ndarray, diffnocs_fit=False) -> np.ndarray:
        
        num_maps, map_w, map_h = features.shape[0:3]
        masked_features = features[masks.astype(bool)]
        print("masked_features.shape: ", masked_features.shape)

        num_samples, num_features = masked_features.shape

        if num_samples < 6:
            print("Warning: Not enough coloumns for PCA. Returning zeros.")
            return np.zeros((num_maps, map_w, map_h, self.torch_pca.n_components))
        
        if diffnocs_fit:
            pca = self.diffnocs_pca
            masked_features_pca = pca.transform(masked_features)
        else:
            #pca = PCA(n_components=6)
            #masked_features_pca = pca.fit_transform(masked_features)
            masked_features_pca = self.torch_pca.fit(torch.tensor(masked_features, dtype=torch.float32).to(self.device)).transform(torch.tensor(masked_features, dtype=torch.float32).to(self.device)).numpy()

        masked_features_pca = minmax_scale(masked_features_pca)

        # Initialize images for PCA reduced features
        features_pca_reshaped = np.zeros((num_maps, map_w, map_h, masked_features_pca.shape[-1]))

        # Fill in the PCA results only at the masked regions
        features_pca_reshaped[masks.astype(bool)] = masked_features_pca

        return features_pca_reshaped

    def get_pca_features(
        self, rgb_image: np.ndarray, mask: np.ndarray, input_size: int = 448, diffnocs_fit=False
    ) -> np.ndarray:
        # Convert the mask to boolean type explicitly
        mask = mask.astype(bool)
        
        assert rgb_image.shape[:2] == mask.shape, "Image and mask dimensions must match"
        resized_rgb = cv2.resize(rgb_image * 255, (input_size, input_size))

        patch_size = self.patch_size
        resized_mask = cv2.resize(
            mask.astype(np.uint8),  # Convert back to uint8 for resize
            (input_size // patch_size, input_size // patch_size),
            interpolation=cv2.INTER_NEAREST,
        )

        # Get patch tokens from the model
        patch_tokens, _ = self.get_features([resized_rgb])

        # Apply PCA to the patch tokens
        pca_features = self.apply_pca(patch_tokens, resized_mask[None], diffnocs_fit=diffnocs_fit)

        if pca_features is None or pca_features.size == 0:
            print("Warning: PCA features are empty. Returning a zero-initialized output.")
            return np.zeros((160, 160, self.torch_pca.n_components))
        
        # Resize the PCA features for visualization
        resized_pca_features = transforms.functional.resize(
                torch.tensor(pca_features[0]).permute(2, 0, 1),
                size=(160, 160),
                interpolation=transforms.InterpolationMode.NEAREST,
            ).permute(1, 2, 0)

        resized_pca_features = resized_pca_features.detach().cpu().numpy()
        resized_pca_features = np.clip(resized_pca_features, 0, 1)

        return resized_pca_features

# from featUp github: https://github.com/mhamilton723/FeatUp
class TorchPCA(object):
    def __init__(self, n_components, device='cuda'):
        self.n_components = n_components
        self.device = device

    def fit(self, X):
        # Move data to the specified device (GPU or CPU)
        X = X.to(self.device)

        # Check if X has any elements
        if X.numel() == 0:
            raise ValueError("Input X is empty. PCA cannot be applied.")

        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)

        # Ensure that the number of components does not exceed data dimensions
        min_dim = min(unbiased.shape)
        if self.n_components > min_dim:
            self.n_components = min_dim
            print(f"Warning: Reducing n_components to {min_dim} due to small input size.")

        if min_dim == 0:
            raise ValueError("PCA cannot be applied on zero-dimensional input.")

        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        # Move data to the specified device (GPU or CPU)
        X = X.to(self.device)
        
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected.cpu()  # Move back to CPU for numpy conversion

# from featUp github: https://github.com/mhamilton723/FeatUp
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        if len(image2.shape) == 4:
            # batched
            image2 = image2.permute(1, 0, 2, 3)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2.permute(1, 0, 2, 3)
    
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
            category_names.append(self.categories[prediction['category_id']])
            category_ids.append(int(prediction['category_id']))
            scores.append(prediction['score'])

            bbox = np.array(prediction['bbox'], dtype=int)

            mask = self.decode_segmentation(prediction['segmentation'], image_data['width'], image_data['height'])
            mask = 1 - mask

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
            "rgb": transforms.ToTensor()(rgb_image),
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
        mask = np.zeros(width * height, dtype=np.uint8)

        rle_counts = rle['counts']
        current_position = 0

        for i in range(len(rle_counts)):
            run_length = rle_counts[i]
            if i % 2 == 0:
                mask[current_position:current_position + run_length] = 1
            current_position += run_length

        return mask.reshape((height, width), order="F")

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

        rgb_filename = img_info['color_file_name']
        depth_filename = img_info['depth_file_name']

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
        mask = np.zeros(width * height, dtype=np.uint8)

        rle_counts = rle['counts']
        current_position = 0

        for i in range(len(rle_counts)):
            run_length = rle_counts[i]
            if i % 2 == 0:
                mask[current_position:current_position + run_length] = 1
            current_position += run_length

        return mask.reshape((height, width), order="F")
    
def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('config', type=str, help="Path to the config file")
    return parser.parse_args()

def add_loss(est_points_batch, gt_points_batch):
    distances = torch.cdist(est_points_batch, gt_points_batch, p=2)
    min_distances, _ = distances.min(dim=2)
    add_scores = min_distances.mean(dim=1)

    return add_scores.mean()

def apply_rotation(points, rotation_matrix):
    points_transposed = points.transpose(1, 2)
    
    rotated_points = torch.bmm(rotation_matrix, points_transposed)

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

def scope_collate_fn(batch):
    rgb_batch = torch.stack([torch.tensor(item[0]) for item in batch])
    nocs_batch = torch.stack([torch.tensor(item[1]) for item in batch])
    normals_no_aug_batch = torch.stack([torch.tensor(item[2]) for item in batch])
    normals_with_aug_batch = torch.stack([torch.tensor(item[3]) for item in batch])
    mask_batch = torch.stack([torch.tensor(item[4]) for item in batch])
    dino_batch = torch.stack([torch.tensor(item[5]) for item in batch])
    info_batch = [item[6] for item in batch]

    return {
        'rgb': rgb_batch,
        'nocs': nocs_batch,
        'normals_no_aug': normals_no_aug_batch,
        'normals_with_aug': normals_with_aug_batch,
        'mask': mask_batch,
        'dino_pca': dino_batch,
        'info': info_batch,
    }

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

    return {
        'rgb': rgb_batch,
        'mask': mask_batch,
        'nocs': nocs_batch,
        'metric_depth': depth_batch,
        'info': info_batch,
    }

class WebDatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
        self.data = []
        
        for sample in self.dataset:
            self.data.append(sample)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
def load_depth_image(depth_bytes):
    depth_image = Image.open(io.BytesIO(depth_bytes)).convert("I;16")
    depth_array = np.array(depth_image, dtype=np.float32)
    return depth_array

def load_image(data):
    with io.BytesIO(data) as stream:
        img = PIL.Image.open(stream)
        img.load()
        img = img.convert("rgb".upper())
        img = img.convert("RGB")
        return img

def create_webdataset_v2(dataset_paths, size=128, shuffle_buffer=1000, dino_mode="", augment=False, center_crop=False, class_name=None):

    dataset = wds.WebDataset(dataset_paths, shardshuffle=True) \
        .decode() \
        .shuffle(shuffle_buffer, initial=size) \
        .to_tuple("rgb.png", "nocs.png", "normals_no_aug.png", "normals_with_aug.png", "mask_visib.png", dino_mode + ".npy", "info.json") \
        .map_tuple( 
            lambda rgb: preprocess(image=load_image(rgb), size=size, interpolation=Image.BICUBIC, augment=augment), 
            lambda nocs: preprocess(image=load_image(nocs), size=size, interpolation=Image.NEAREST), 
            lambda normals_no_aug: preprocess(image=load_image(normals_no_aug), size=size, interpolation=Image.NEAREST, augment=True), 
            lambda normals_with_aug: preprocess(image=load_image(normals_with_aug), size=size, interpolation=Image.NEAREST, augment=True), 
            lambda mask: preprocess(image=load_image(mask), size=size, interpolation=Image.NEAREST),
            lambda dino_pca: preprocess(image=dino_pca, size=size, interpolation=Image.NEAREST),
            lambda info: info)
    return dataset
    
def create_webdataset(dataset_paths, size=128, shuffle_buffer=1000, augment=False, class_name=None):

    dataset = wds.WebDataset(dataset_paths, shardshuffle=True) \
        .decode() \
        .shuffle(shuffle_buffer, initial=size) \
        .to_tuple("rgb.png", "normals.png", "mask_visib.png", "nocs.png", "info.json") \
        .map_tuple( 
            lambda rgb: preprocess(image=load_image(rgb), size=size, interpolation=Image.BICUBIC, augment=augment), 
            lambda normals: preprocess(image=load_image(normals), size=size, interpolation=Image.NEAREST, augment=True), 
            lambda mask: preprocess(image=load_image(mask), size=size, interpolation=Image.NEAREST), 
            lambda nocs: preprocess(image=load_image(nocs), size=size, interpolation=Image.NEAREST), 
            lambda info: info) \
        .select(lambda sample: (class_name is None) or (sample[3].get('category_id') == class_name)) # Adjust index for 'info.json'

    return dataset

# def create_webdataset_test(dataset_paths, size=128, shuffle_buffer=1000, augment=False, class_name=None):

#     dataset = wds.WebDataset(dataset_paths, shardshuffle=True) \
#         .decode() \
#         .shuffle(shuffle_buffer, initial=size) \
#         .to_tuple("rgb.png", "mask_visib.png", "nocs.png", "metric_depth.png", "info.json") \
#         .map_tuple( 
#             lambda rgb: preprocess(load_image(rgb), size, Image.BICUBIC, augment=augment), 
#             lambda mask: preprocess(load_image(mask), size, Image.NEAREST), 
#             lambda nocs: preprocess(load_image(nocs), size, Image.NEAREST), 
#             lambda metric_depth: preprocess(load_depth_image(metric_depth), size, Image.NEAREST, is_depth=True),  # Preprocess depth after loading
#             lambda info: info) \
#         .select(lambda sample: (class_name is None) or (sample[4].get('category_id') == class_name))  # Adjust index for 'info.json'

#     return dataset

def preprocess(image, size, interpolation, augment=False):
    
    img_array = np.array(image).astype(np.uint8)

    if img_array.shape[2] == 6:
        return img_array

    h, w = img_array.shape[0], img_array.shape[1]

    crop = min(h, w)
    img_array = img_array[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]

    if augment:
        seq_syn = iaa.Sequential([
                            iaa.Sometimes(0.3, iaa.CoarseDropout(p=0.2, size_percent=0.05)),
                            iaa.Sometimes(0.5, iaa.Dropout(p=(0.0, 0.1)))
                            ], random_order=True)

        img_array = seq_syn.augment_image(img_array)

    image = Image.fromarray(img_array)
    image = image.resize((size, size), resample=interpolation)
    img_array = np.array(image).astype(np.uint8)   

    return img_array

def normalize_quaternion(q):
    norm = torch.norm(q, dim=1, keepdim=True)
    return q / norm

def setup_environment(gpu_id):
    if len(sys.argv) != 2:
        print("Usage: python3 train.py <gpu_id>")
        sys.exit()

    if gpu_id == '-1':
        gpu_id = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

def plot_progress_imgs(imgfn, rgb_images, normal_images, nocs_images_normalized_gt, nocs_estimated, mask_images, config):
    _,ax = plt.subplots(config.num_imgs_log, 6, figsize=(10,20))
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

    plt.tight_layout()
    plt.savefig(imgfn, dpi=300)
    plt.close()

def plot_progress_imgs_v2(imgfn, rgb_images, normal_images, nocs_images_normalized_gt, nocs_estimated, mask_images, dino_images, config):
    _,ax = plt.subplots(config.num_imgs_log, 6, figsize=(10,20))
    col_titles = ['RGB Image', 'Normals GT', 'NOCS GT', 'NOCS Estimated', 'Mask GT', 'DINO GT']
    
    # Add column titles
    for i, title in enumerate(col_titles):
        ax[0, i].set_title(title, fontsize=12)

    for i in range(config.num_imgs_log):
        ax[i, 0].imshow(((rgb_images[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 1].imshow(((normal_images[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 2].imshow(((nocs_images_normalized_gt[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 3].imshow(((nocs_estimated[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 4].imshow(((mask_images[i])).detach().cpu().numpy().transpose(1, 2, 0))
        ax[i, 5].imshow(((dino_images[i] + 1) / 2).detach().cpu().numpy().transpose(1, 2, 0))

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

def crop_and_resize(img, enlarged_bbox, original_bbox, target_size=128, interpolation=Image.NEAREST):

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
    bbox_width = bbox[2] - bbox[0]
    bbox_height = bbox[3] - bbox[1]
    center_x = (bbox[2] + bbox[0]) // 2
    center_y = (bbox[3] + bbox[1]) // 2
        
    enlarged_size = int(max(bbox_width, bbox_height) * bbox_scaler)
    print("enlarged_size: ", enlarged_size)

    crop_xmin = max(center_x - enlarged_size // 2, 0)
    crop_xmax = min(center_x + enlarged_size // 2, img_shape[1])
    crop_ymin = max(center_y - enlarged_size // 2, 0)
    crop_ymax = min(center_y + enlarged_size // 2, img_shape[0])

    return np.array([crop_xmin, crop_ymin, crop_xmax, crop_ymax])

def restore_original_bbox_crop(cropped_resized_img, metadata, interpolation=Image.NEAREST):

    scale_factor = metadata['scale_factor']
    original_bbox_size = metadata['original_bbox_size']
    original_offset = metadata['original_offset']

    enlarged_size = int(cropped_resized_img.shape[1] / scale_factor)

    cropped_img_pil = Image.fromarray(cropped_resized_img)
    restored_enlarged_img = cropped_img_pil.resize((enlarged_size, enlarged_size), interpolation)
    restored_enlarged_img = np.array(restored_enlarged_img)

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
    binary_mask = (mask_image > 0).astype(np.uint8)
    masked_nocs = nocs_image.copy()
    masked_nocs[binary_mask == 0] = 0

    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

    overlay_image = full_scale_rgb.copy()
    
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
    R_inliers = teaserpp_solver.getRotationInliers()
    t_inliers = teaserpp_solver.getTranslationInliers()
    s_inliers = teaserpp_solver.getScaleInliers()
    #print("Solution is:", solution)

    # Extract rotation, translation, and scale from the solution
    R = solution.rotation
    t = solution.translation
    s = solution.scale

    return R, t, s, len(R_inliers), len(t_inliers), len(s_inliers)

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

    indices = np.random.choice(src.shape[1], num_samples, replace=False)

    return src[:, indices], dst[:, indices]

def create_line_set(src_points, dst_points, color=[0, 1, 0]):
    src_points = np.asarray(src_points, dtype=np.float64).T
    dst_points = np.asarray(dst_points, dtype=np.float64).T

    if src_points.shape[1] != 3 or dst_points.shape[1] != 3:
        raise ValueError("Points must have a shape of (N, 3)")

    lines = [[i, i + len(src_points)] for i in range(len(src_points))]

    line_set = o3d.geometry.LineSet()

    all_points = np.concatenate((src_points, dst_points), axis=0)
    line_set.points = o3d.utility.Vector3dVector(all_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)

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

def project_pointcloud_to_image(pointcloud, pointnormals, fx, fy, cx, cy, image_shape):
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

def create_pointnormals(dst):
    pcd_normals = create_open3d_point_cloud(dst, [1, 0, 0])
    o3d.geometry.PointCloud.estimate_normals(pcd_normals)
    pcd_normals.normalize_normals()
    pcd_normals.orient_normals_towards_camera_location()
    normals = np.asarray(pcd_normals.normals)

    return normals