import torch
import torch.nn.functional as F

from tqdm.auto import tqdm

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DDIMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    UNet2DConditionModel,
    UNet2DModel
)
from torchvision import transforms
import cv2
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, BartTokenizer, BartModel
from transformers import AutoImageProcessor, AutoModel
import pickle
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
# import featup
# #from featup.util import pca
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import minmax_scale
# from typing import List, Tuple
# import numpy as np
import torchvision.transforms as T

# class TorchPCA(object):

#     def __init__(self, n_components):
#         self.n_components = n_components

#     def fit(self, X):
#         self.mean_ = X.mean(dim=0)
#         unbiased = X - self.mean_.unsqueeze(0)
#         U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
#         self.components_ = V.T
#         self.singular_values_ = S
#         return self

#     def transform(self, X):
#         t0 = X - self.mean_.unsqueeze(0)
#         projected = t0 @ self.components_.T
#         return projected
    
# def pca(image_feats, mask_tensor, dim=3, fit_pca=None, use_torch_pca=True, max_samples=None):
#     device = image_feats.device
#     masks = mask_tensor[0][0].detach().cpu().numpy()[None]
#     features = np.transpose(image_feats.detach().cpu().numpy(), (0, 2, 3, 1)) 
#     print("features: ", features.shape)
#     print("masks: ", masks.shape)
#     pca = PCA(n_components=6)

#     num_maps, map_w, map_h = features.shape[0:3]
#     masked_features = features[masks.astype(bool)]

#     masked_features_pca = pca.fit_transform(masked_features)
#     masked_features_pca = minmax_scale(masked_features_pca)

#     # Initialize images for PCA reduced features
#     features_pca_reshaped = np.zeros((num_maps, map_w, map_h, masked_features_pca.shape[-1]))

#     # Fill in the PCA results only at the masked regions
#     features_pca_reshaped[masks.astype(bool)] = masked_features_pca

#     return features_pca_reshaped[0]

# class UnNormalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, image):
#         image2 = torch.clone(image)
#         if len(image2.shape) == 4:
#             # batched
#             image2 = image2.permute(1, 0, 2, 3)
#         for t, m, s in zip(image2, self.mean, self.std):
#             t.mul_(s).add_(m)
#         return image2.permute(1, 0, 2, 3)
    
# class DinoFeatures:
#     def __init__(self, upsampling=False) -> None:
#         device: str = "cuda"
#         self._device = device

#         # Load DINOv2 model from transformers
#         self.dino_model = AutoModel.from_pretrained("./dinov2-small")
#         self.patch_size = self.dino_model.config.patch_size
#         self._feature_dim = self.dino_model.config.hidden_size
#         self.dino_model.to(device)
#         self.dino_model.eval()    

#         # Normalization transform based on ImageNet
#         self._transform = transforms.Compose(
#             [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
#         )

#         # PCA model from sklearn
#         #self._pca = pickle.load(open("../nocs_renderer/pca6.pkl", "rb"))
#         self._pca = PCA(n_components=6)

#     def get_features(
#         self,
#         images: List[np.ndarray],
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         assert len(set(image.shape for image in images)) == 1, "Not all images have same shape"
#         im_w, im_h = images[0].shape[0:2]
#         assert (
#             im_w % self.patch_size == 0 and im_h % self.patch_size == 0
#         ), "Both width and height of the image must be divisible by patch size"

#         image_array = np.stack(images) / 255.0
#         input_tensor = torch.Tensor(np.transpose(image_array, [0, 3, 2, 1]))
#         input_tensor = self._transform(input_tensor).to(self._device)

#         with torch.no_grad():
#             outputs = self.dino_model(input_tensor, output_hidden_states=True)
        
#             # CLS token is first then patch tokens
#             class_tokens = outputs.last_hidden_state[:, 0, :]
#             patch_tokens = outputs.last_hidden_state[:, 1:, :]

#             if patch_tokens.is_cuda:
#                 patch_tokens = patch_tokens.cpu()
#                 class_tokens = class_tokens.cpu()

#             patch_tokens = patch_tokens.detach().numpy()
#             class_tokens = class_tokens.detach().numpy()

#             all_patches = patch_tokens.reshape(
#                 [-1, im_w // self.patch_size, im_h // self.patch_size, self._feature_dim]
#             )
#         all_patches = np.transpose(all_patches, (0, 2, 1, 3))

#         return all_patches, class_tokens

#     def apply_pca(self, features: np.ndarray, masks: np.ndarray) -> np.ndarray:
#         num_maps, map_w, map_h = features.shape[0:3]
#         masked_features = features[masks.astype(bool)]

#         masked_features_pca = self._pca.fit_transform(masked_features)
#         masked_features_pca = minmax_scale(masked_features_pca)

#         # Initialize images for PCA reduced features
#         features_pca_reshaped = np.zeros((num_maps, map_w, map_h, masked_features_pca.shape[-1]))

#         # Fill in the PCA results only at the masked regions
#         features_pca_reshaped[masks.astype(bool)] = masked_features_pca

#         return features_pca_reshaped

#     def get_pca_features(
#         self, rgb_image: np.ndarray, mask: np.ndarray, input_size: int = 448
#     ) -> np.ndarray:
#         # Convert the mask to boolean type explicitly
#         mask = mask.astype(bool)
        
#         assert rgb_image.shape[:2] == mask.shape, "Image and mask dimensions must match"
#         resized_rgb = cv2.resize(rgb_image * 255, (input_size, input_size))

#         patch_size = self.patch_size
#         resized_mask = cv2.resize(
#             mask.astype(np.uint8),  # Convert back to uint8 for resize
#             (input_size // patch_size, input_size // patch_size),
#             interpolation=cv2.INTER_NEAREST,
#         )

#         # Get patch tokens from the model
#         patch_tokens, _ = self.get_features([resized_rgb])

#         # Apply PCA to the patch tokens
#         pca_features = self.apply_pca(patch_tokens, resized_mask[None])

#         # Resize the PCA features for visualization
#         resized_pca_features: np.ndarray = (
#             transforms.functional.resize(
#                 torch.tensor(pca_features[0]).permute(2, 0, 1),
#                 size=(160, 160),
#                 interpolation=transforms.InterpolationMode.NEAREST,
#             )
#             .permute(1, 2, 0)
#         )

#         return resized_pca_features

# class DiffusionNOCSDino(nn.Module):
#     """U-Net generator with a shared encoder and multiple decoder heads."""
    
#     def __init__(self, input_nc = 6, output_nc = 3, image_size=128, num_training_steps=1000, num_inference_steps=200):
#         """
#         Parameters:
#             input_nc (int)      -- the number of channels in input images
#             output_nc (int)     -- the number of channels in output images per head
#             num_downs (int)     -- the number of downsamplings in UNet
#             num_heads (int)     -- number of decoder heads
#             ngf (int)           -- the number of filters in the last conv layer
#             norm_layer          -- normalization layer
#             use_dropout (bool)  -- whether to use dropout in intermediate layers
#         """
#         super(DiffusionNOCSDino, self).__init__()

#         self.model = UNet2DConditionModel(
#             sample_size=image_size,  # the target image resolution
#             in_channels=input_nc,  # the number of input channels, 3 for RGB images
#             out_channels=output_nc,  # the number of output channels
#             layers_per_block=2,  # how many ResNet layers to use per UNet block
#             block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
#             down_block_types=(
#                 "DownBlock2D",  # a regular ResNet downsampling block
#                 "DownBlock2D",
#                 "DownBlock2D",
#                 "DownBlock2D",
#                 "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
#                 "DownBlock2D",
#             ),
#             up_block_types=(
#                 "UpBlock2D",  # a regular ResNet upsampling block
#                 "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
#                 "UpBlock2D",
#                 "UpBlock2D",
#                 "UpBlock2D",
#                 "UpBlock2D",
#             ),
#             cross_attention_dim=768,
#         )

#         self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base')

#         # self.bart_model = BartModel.from_pretrained('facebook/bart-base')
#         # self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

#         self.noise_scheduler = DDPMScheduler(num_train_timesteps=num_training_steps)
#         self.num_training_steps = num_training_steps
#         self.num_inference_steps = num_inference_steps

#     def forward(self, rgb_image, nocs_gt):
#         # sample noise
#         noise = torch.randn(nocs_gt.shape, dtype=nocs_gt.dtype, device=nocs_gt.device)
#         # batch size of noise
#         bsz = nocs_gt.shape[0]
#         # Sample a random timestep for each image
#         timesteps = torch.randint(
#             0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=nocs_gt.device
#         ).long()

#         noisy_latents = self.noise_scheduler.add_noise(nocs_gt, noise, timesteps)

#         latents = torch.cat([rgb_image, noisy_latents], dim=1)

#         combined_embeddings = self.get_embeddings(rgb_image)

#         model_output = self.model(latents, timesteps, combined_embeddings).sample

#         loss = F.mse_loss(model_output.float(), noise.float(), reduction="mean")

#         return loss
    
#     def get_embeddings(self, rgb_image):

#         # DINO Embeddings
#         images_resized = F.interpolate(rgb_image, size=(224, 224), mode='bilinear', align_corners=False)
#         with torch.no_grad():
#             dino_embeddings = self.dino_model(images_resized)
#         dino_embeddings = dino_embeddings[0]

#         return dino_embeddings

#     def inference(self, rgb_image):
#         self.noise_scheduler.set_timesteps(self.num_inference_steps)

#         nocs_noise = torch.randn(rgb_image.shape, dtype=rgb_image.dtype, device=rgb_image.device)

#         combined_embeddings = self.get_embeddings(rgb_image)

#         for timestep in tqdm(self.noise_scheduler.timesteps):

#             input = torch.cat(
#                 [rgb_image, nocs_noise], dim=1
#             )  # this order is important

#             with torch.no_grad():
#                 noisy_residual = self.model(input, timestep, combined_embeddings).sample
#             previous_noisy_sample = self.noise_scheduler.step(noisy_residual, timestep, nocs_noise).prev_sample

#             nocs_noise = previous_noisy_sample

#         nocs_estimated = nocs_noise
#         nocs_estimated = ((nocs_estimated + 1 ) / 2)

#         return nocs_estimated
    
class DiffusionNOCSDinoBART(nn.Module):
    """U-Net generator with a shared encoder and multiple decoder heads."""
    
    def __init__(self, input_nc = 6, output_nc = 3, image_size=128, num_training_steps=1000, num_inference_steps=200):
        """
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images per head
            num_downs (int)     -- the number of downsamplings in UNet
            num_heads (int)     -- number of decoder heads
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- whether to use dropout in intermediate layers
        """
        super(DiffusionNOCSDinoBART, self).__init__()

        self.model = UNet2DConditionModel(
            sample_size=image_size,  # the target image resolution
            in_channels=input_nc,  # the number of input channels, 3 for RGB images
            out_channels=output_nc,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=768,
        )

        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base')

        self.bart_model = BartModel.from_pretrained('facebook/bart-base')
        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

        self.train_noise_scheduler = DDPMScheduler(num_train_timesteps=num_training_steps)
        self.inference_noise_scheduler = DPMSolverSinglestepScheduler(num_train_timesteps=num_training_steps, algorithm_type="dpmsolver++", thresholding=True)

        self.num_training_steps = num_training_steps
        self.num_inference_steps = num_inference_steps

    def forward(self, rgb_image, nocs_gt, obj_names):
        # sample noise
        noise = torch.randn(nocs_gt.shape, dtype=nocs_gt.dtype, device=nocs_gt.device)
        # batch size of noise
        bsz = nocs_gt.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.train_noise_scheduler.config.num_train_timesteps, (bsz,), device=nocs_gt.device
        ).long()

        noisy_latents = self.train_noise_scheduler.add_noise(nocs_gt, noise, timesteps)

        latents = torch.cat([rgb_image, noisy_latents], dim=1)

        combined_embeddings = self.get_embeddings(rgb_image, obj_names)

        model_output = self.model(latents, timesteps, combined_embeddings).sample

        loss = F.mse_loss(model_output.float(), noise.float(), reduction="mean")

        return loss
    
    def get_embeddings(self, rgb_image, obj_names):

        # DINO Embeddings
        images_resized = F.interpolate(rgb_image, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            dino_embeddings = self.dino_model(images_resized)
        dino_embeddings = dino_embeddings[0]

        # BART Language Embeddings
        tokens = self.bart_tokenizer(obj_names, return_tensors="pt", padding=True).to(rgb_image.device)
        with torch.no_grad():
            bart_embeddings = self.bart_model(**tokens)  
            bart_embeddings = bart_embeddings.last_hidden_state

        # Calculate the repeat factor dynamically
        repeat_factor = (dino_embeddings.size(1) + bart_embeddings.size(1) - 1) // bart_embeddings.size(1)  # ceil(257 / bart_embeddings.size(1))
        bart_repeated = bart_embeddings.repeat(1, repeat_factor, 1)  # Extend along sequence length
        bart_repeated = bart_repeated[:, :dino_embeddings.size(1), :]  # Trim to [1, 257, 768]
    
        combined_embeddings = torch.cat((dino_embeddings, bart_repeated), dim=1)

        return combined_embeddings

    def inference(self, rgb_image, combined_embeddings):
        self.inference_noise_scheduler.set_timesteps(self.num_inference_steps)

        nocs_noise = torch.randn(rgb_image.shape, dtype=rgb_image.dtype, device=rgb_image.device)

        #combined_embeddings = self.get_embeddings(rgb_image, obj_names)

        for timestep in tqdm(self.inference_noise_scheduler.timesteps):

            input = torch.cat(
                [rgb_image, nocs_noise], dim=1
            )  # this order is important

            with torch.no_grad():
                noisy_residual = self.model(input, timestep, combined_embeddings).sample
            previous_noisy_sample = self.inference_noise_scheduler.step(noisy_residual, timestep, nocs_noise).prev_sample

            nocs_noise = previous_noisy_sample

        nocs_estimated = nocs_noise
        nocs_estimated = ((nocs_estimated + 1 ) / 2)

        return nocs_estimated
    
class DiffusionNOCSDinoBARTNormals(nn.Module):
    """U-Net generator with a shared encoder and multiple decoder heads."""
    
    def __init__(self, input_nc = 9, output_nc = 3, image_size=128, num_training_steps=1000, num_inference_steps=50):
        """
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images per head
            num_downs (int)     -- the number of downsamplings in UNet
            num_heads (int)     -- number of decoder heads
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- whether to use dropout in intermediate layers
        """
        super(DiffusionNOCSDinoBARTNormals, self).__init__()

        self.model = UNet2DConditionModel(
            sample_size=image_size,  # the target image resolution
            in_channels=input_nc,  # the number of input channels, 3 for RGB images
            out_channels=output_nc,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=768,
        )

        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base')

        self.bart_model = BartModel.from_pretrained('facebook/bart-base')
        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

        self.train_noise_scheduler = DDPMScheduler(num_train_timesteps=num_training_steps)
        self.inference_noise_scheduler = DPMSolverSinglestepScheduler(num_train_timesteps=num_training_steps, algorithm_type="dpmsolver++", thresholding=True)

        self.num_training_steps = num_training_steps
        self.num_inference_steps = num_inference_steps

    def forward(self, rgb_image, normals_image, nocs_gt, obj_names):
        noise = torch.randn(nocs_gt.shape, dtype=nocs_gt.dtype, device=nocs_gt.device)

        bsz = nocs_gt.shape[0]

        timesteps = torch.randint(
            0, self.train_noise_scheduler.config.num_train_timesteps, (bsz,), device=nocs_gt.device
        ).long()

        noisy_latents = self.train_noise_scheduler.add_noise(nocs_gt, noise, timesteps)

        latents = torch.cat([rgb_image, normals_image, noisy_latents], dim=1)

        combined_embeddings = self.get_embeddings(rgb_image, obj_names)

        model_output = self.model(latents, timesteps, combined_embeddings).sample

        loss = F.mse_loss(model_output.float(), noise.float(), reduction="mean")

        return loss
    
    def get_embeddings(self, rgb_image, obj_names):

        images_resized = F.interpolate(rgb_image, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            dino_embeddings = self.dino_model(images_resized)
        dino_embeddings = dino_embeddings[0]

        tokens = self.bart_tokenizer(obj_names, return_tensors="pt", padding=True).to(rgb_image.device)
        with torch.no_grad():
            bart_embeddings = self.bart_model(**tokens)  
            bart_embeddings = bart_embeddings.last_hidden_state

        repeat_factor = (dino_embeddings.size(1) + bart_embeddings.size(1) - 1) // bart_embeddings.size(1)  # ceil(257 / bart_embeddings.size(1))
        bart_repeated = bart_embeddings.repeat(1, repeat_factor, 1)  # Extend along sequence length
        bart_repeated = bart_repeated[:, :dino_embeddings.size(1), :]  # Trim to [1, 257, 768]
    
        combined_embeddings = torch.cat((dino_embeddings, bart_repeated), dim=1)

        return combined_embeddings

    def inference(self, rgb_image, normals_image, combined_embeddings):
        self.inference_noise_scheduler.set_timesteps(self.num_inference_steps)

        nocs_noise = torch.randn(rgb_image.shape, dtype=rgb_image.dtype, device=rgb_image.device)

        for timestep in tqdm(self.inference_noise_scheduler.timesteps):

            input = torch.cat(
                [rgb_image, normals_image, nocs_noise], dim=1
            )  # this order is important

            with torch.no_grad():
                noisy_residual = self.model(input, timestep, combined_embeddings).sample
            previous_noisy_sample = self.inference_noise_scheduler.step(noisy_residual, timestep, nocs_noise).prev_sample

            nocs_noise = previous_noisy_sample

        nocs_estimated = ((nocs_noise + 1 ) / 2)

        return nocs_estimated

class DiffusionNOCSBARTPCA(nn.Module):
    """U-Net generator with a shared encoder and multiple decoder heads."""
    
    def __init__(self, input_nc = 9, output_nc = 3, image_size=128, num_training_steps=1000, num_inference_steps=10):
        """
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images per head
            num_downs (int)     -- the number of downsamplings in UNet
            num_heads (int)     -- number of decoder heads
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- whether to use dropout in intermediate layers
        """
        super(DiffusionNOCSBARTPCA, self).__init__()

        self.model = UNet2DConditionModel(
            sample_size=image_size,  # the target image resolution
            in_channels=input_nc,  # the number of input channels, 3 for RGB images
            out_channels=output_nc,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=768,
        )

        #self.dino_model = AutoModel.from_pretrained("./dinov2-small")
        self.pca = pickle.load(open(str("./pca6.pkl"), "rb"))

        self.bart_model = BartModel.from_pretrained('facebook/bart-base')
        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.dino_model = DinoFeatures()

        self.upsampler = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=False)

        self.train_noise_scheduler = DDPMScheduler(num_train_timesteps=num_training_steps)
        self.inference_noise_scheduler = DPMSolverSinglestepScheduler(num_train_timesteps=num_training_steps, algorithm_type="dpmsolver++", thresholding=True)

        self.num_training_steps = num_training_steps
        self.num_inference_steps = num_inference_steps

        self.unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, rgb_image, normals_image, nocs_gt, mask_images, obj_names):
        # sample noise
        noise = torch.randn(nocs_gt.shape, dtype=nocs_gt.dtype, device=nocs_gt.device)
        # batch size of noise
        bsz = nocs_gt.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.train_noise_scheduler.config.num_train_timesteps, (bsz,), device=nocs_gt.device
        ).long()

        noisy_latents = self.train_noise_scheduler.add_noise(nocs_gt, noise, timesteps)

        latents = torch.cat([rgb_image, normals_image, noisy_latents], dim=1)

        bart_embeddings = self.get_bart_embeddings(rgb_image, obj_names)
        pca_embeddings = self.get_featup_embeddings(rgb_image, mask_images)
        model_output = self.model(latents, timesteps, bart_embeddings).sample

        loss = F.mse_loss(model_output.float(), noise.float(), reduction="mean")

        return loss
    
    def get_bart_embeddings(self, rgb_image, obj_names):
        # BART Language Embeddings
        tokens = self.bart_tokenizer(obj_names, return_tensors="pt", padding=True).to(rgb_image.device)
        with torch.no_grad():
            bart_embeddings = self.bart_model(**tokens)  
            bart_embeddings = bart_embeddings.last_hidden_state

        return bart_embeddings

    def get_featup_embeddings(self, rgb_image, mask_images):

        diffnocs_pca_features = self.dino_model.get_pca_features(((rgb_image + 1) / 2)[0].permute(1, 2, 0).detach().cpu().numpy(), mask_images[0].permute(1, 2, 0).detach().cpu().numpy()[..., 0] * 255)
        print(diffnocs_pca_features.shape)

        self._transform = transforms.Compose([
            T.Resize((224, 224)),
            self.norm,
        ])

        image_tensor = self._transform(((rgb_image + 1) / 2))

        hr_feats = self.upsampler(image_tensor)
        lr_feats = self.upsampler.model(image_tensor)

        # Resize masks to match input size
        mask_tensor = F.interpolate(mask_images, size=(256, 256), mode='nearest')
        mask_tensor_full_dim = mask_tensor[:, :1, :, :].expand(-1, hr_feats.shape[1], -1, -1)

        hr_feats = hr_feats * mask_tensor_full_dim
        #lr_feats = lr_feats * mask_tensor

        featup_pca_features = pca(hr_feats[0].unsqueeze(0), mask_tensor, dim=6)

        image_tensor = self.unnorm(image_tensor)
        # mask_tensor_full_dim = mask_tensor[:, :1, :, :].expand(-1, hr_feats_pca.shape[1], -1, -1)
        # hr_feats_pca =  hr_feats_pca * mask_tensor_full_dim
        # hr_feats_pca = hr_feats_pca[0, :3, :, :]

        # lr_feats_pca = transforms.functional.resize(
        #         lr_feats_pca,
        #         size=(256, 256),
        #         interpolation=transforms.InterpolationMode.NEAREST
        #         )
        
        # print(lr_feats_pca.shape)
        # print(mask_tensor_full_dim.shape)

        # mask_tensor_full_dim = F.interpolate(mask_tensor_full_dim, size=(256, 256), mode='nearest')
        # # mask_tensor_full_dim = mask_tensor[:, :1, :, :].expand(-1, hr_feats_pca.shape[1], -1, -1)
        
        # lr_feats_pca =  lr_feats_pca * mask_tensor_full_dim
        # lr_feats_pca = lr_feats_pca[0, :3, :, :]

        # lr_feats_pca = lr_feats_pca.permute(1, 2, 0).detach().cpu()
        # hr_feats_pca = hr_feats_pca.permute(1, 2, 0).detach().cpu()

        # print(lr_feats_pca.shape)

        fig, ax = plt.subplots(1, 4, figsize=(15, 5))

        ax[0].imshow(image_tensor[0].permute(1, 2, 0).detach().cpu())
        ax[0].set_title("Image")
        # ax[1].imshow(lr_feats_pca)
        # ax[1].set_title("Original Features")
        ax[2].imshow(featup_pca_features[..., :3])
        ax[2].set_title("FeatUp Features")
        ax[3].imshow(diffnocs_pca_features[..., :3])
        ax[3].set_title("DINOv2 Features")
        #remove_axes(ax)
        plt.show()

        return hr_feats

    def inference(self, rgb_image, normals_image, obj_names):
        self.inference_noise_scheduler.set_timesteps(self.num_inference_steps)

        nocs_noise = torch.randn(rgb_image.shape, dtype=rgb_image.dtype, device=rgb_image.device)

        bart_embeddings = self.get_bart_embeddings(rgb_image, obj_names)
        pca_embeddings = self.get_featup_embeddings(rgb_image)

        images = torch.concat((rgb_image, normals_image), dim=1)

        inputs = torch.concat((images, nocs_noise), dim=1)
        
        for timestep in tqdm(self.inference_noise_scheduler.timesteps):
            with torch.no_grad():
                noisy_residual = self.model(inputs, timestep, bart_embeddings).sample
                previous_noisy_sample = self.inference_noise_scheduler.step(noisy_residual, timestep, inputs[:, -3:]).prev_sample
                inputs = torch.concat((images, previous_noisy_sample), dim=1)

        nocs_estimated = previous_noisy_sample
        nocs_estimated = ((nocs_estimated + 1 ) / 2)

        return nocs_estimated

class DiffusionNOCSDino(nn.Module):
    """U-Net generator with a shared encoder and multiple decoder heads."""
    
    def __init__(self, input_nc = 9, output_nc = 3, image_size=128, num_training_steps=1000, num_inference_steps=50):
        """
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images per head
            num_downs (int)     -- the number of downsamplings in UNet
            num_heads (int)     -- number of decoder heads
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- whether to use dropout in intermediate layers
        """
        super(DiffusionNOCSDinoBARTNormals, self).__init__()

        self.model = UNet2DConditionModel(
            sample_size=image_size,  # the target image resolution
            in_channels=input_nc,  # the number of input channels, 3 for RGB images
            out_channels=output_nc,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=768,
        )

        self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base')

        self.train_noise_scheduler = DDPMScheduler(num_train_timesteps=num_training_steps)
        self.inference_noise_scheduler = DPMSolverSinglestepScheduler(num_train_timesteps=num_training_steps, algorithm_type="dpmsolver++", thresholding=True)

        self.num_training_steps = num_training_steps
        self.num_inference_steps = num_inference_steps

        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._transform = transforms.Compose([
            T.Resize((224, 224)),
            self.norm,
        ])

    def forward(self, rgb_image, normals_image, nocs_gt, obj_names):
        noise = torch.randn(nocs_gt.shape, dtype=nocs_gt.dtype, device=nocs_gt.device)

        bsz = nocs_gt.shape[0]

        timesteps = torch.randint(
            0, self.train_noise_scheduler.config.num_train_timesteps, (bsz,), device=nocs_gt.device
        ).long()

        noisy_latents = self.train_noise_scheduler.add_noise(nocs_gt, noise, timesteps)

        latents = torch.cat([rgb_image, normals_image, noisy_latents], dim=1)

        combined_embeddings = self.get_embeddings(rgb_image, obj_names)

        model_output = self.model(latents, timesteps, combined_embeddings).sample

        loss = F.mse_loss(model_output.float(), noise.float(), reduction="mean")

        return loss
    
    def get_dino_embeddings(self, rgb_image):
        rgb_image = self._transform(((rgb_image + 1) / 2))

        with torch.no_grad():
            dino_embeddings = self.dino_model(rgb_image)

        dino_embeddings = dino_embeddings.last_hidden_state
        print("dino_embeddings: ", dino_embeddings.shape)

        return dino_embeddings

    def inference(self, rgb_image, normals_image, embeddings):
        self.inference_noise_scheduler.set_timesteps(self.num_inference_steps)

        nocs_noise = torch.randn(rgb_image.shape, dtype=rgb_image.dtype, device=rgb_image.device)

        for timestep in tqdm(self.inference_noise_scheduler.timesteps):

            input = torch.cat(
                [rgb_image, normals_image, nocs_noise], dim=1
            )  # this order is important

            with torch.no_grad():
                noisy_residual = self.model(input, timestep, embeddings).sample
            previous_noisy_sample = self.inference_noise_scheduler.step(noisy_residual, timestep, nocs_noise).prev_sample

            nocs_noise = previous_noisy_sample

        nocs_estimated = ((nocs_noise + 1 ) / 2)

        return nocs_estimated

class DiffusionNOCSBART(nn.Module):
    """U-Net generator with a shared encoder and multiple decoder heads."""
    
    def __init__(self, input_nc = 9, output_nc = 3, image_size=128, num_training_steps=1000, num_inference_steps=50):
        """
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images per head
            num_downs (int)     -- the number of downsamplings in UNet
            num_heads (int)     -- number of decoder heads
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- whether to use dropout in intermediate layers
        """
        super(DiffusionNOCSBART, self).__init__()

        self.model = UNet2DConditionModel(
            sample_size=image_size,  # the target image resolution
            in_channels=input_nc,  # the number of input channels, 3 for RGB images
            out_channels=output_nc,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=768,
        )

        self.bart_model = BartModel.from_pretrained('facebook/bart-base')
        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

        self.train_noise_scheduler = DDPMScheduler(num_train_timesteps=num_training_steps)
        self.inference_noise_scheduler = DPMSolverSinglestepScheduler(num_train_timesteps=num_training_steps, algorithm_type="dpmsolver++", thresholding=True)

        self.num_training_steps = num_training_steps
        self.num_inference_steps = num_inference_steps

    def forward(self, rgb_image, normals_image, nocs_gt, obj_names):
        noise = torch.randn(nocs_gt.shape, dtype=nocs_gt.dtype, device=nocs_gt.device)

        bsz = nocs_gt.shape[0]

        timesteps = torch.randint(
            0, self.train_noise_scheduler.config.num_train_timesteps, (bsz,), device=nocs_gt.device
        ).long()

        noisy_latents = self.train_noise_scheduler.add_noise(nocs_gt, noise, timesteps)

        latents = torch.cat([rgb_image, normals_image, noisy_latents], dim=1)

        bart_embeddings = self.get_bart_embeddings(obj_names)

        model_output = self.model(latents, timesteps, bart_embeddings).sample

        loss = F.mse_loss(model_output.float(), noise.float(), reduction="mean")

        return loss
    
    def get_bart_embeddings(self, obj_names):

        tokens = self.bart_tokenizer(obj_names, return_tensors="pt", padding=True).to(self.bart_tokenizer.device)
        with torch.no_grad():
            bart_embeddings = self.bart_model(**tokens)  

        bart_embeddings = bart_embeddings.last_hidden_state
        print("bart_embeddings: ", bart_embeddings.shape)
        return bart_embeddings

    def inference(self, rgb_image, normals_image, embeddings):
        self.inference_noise_scheduler.set_timesteps(self.num_inference_steps)

        nocs_noise = torch.randn(rgb_image.shape, dtype=rgb_image.dtype, device=rgb_image.device)

        for timestep in tqdm(self.inference_noise_scheduler.timesteps):

            input = torch.cat(
                [rgb_image, normals_image, nocs_noise], dim=1
            )  # this order is important

            with torch.no_grad():
                noisy_residual = self.model(input, timestep, embeddings).sample

            previous_noisy_sample = self.inference_noise_scheduler.step(noisy_residual, timestep, nocs_noise).prev_sample

            nocs_noise = previous_noisy_sample

        nocs_estimated = ((nocs_noise + 1 ) / 2)

        return nocs_estimated

class DiffusionNOCS(nn.Module):
    """U-Net generator with a shared encoder and multiple decoder heads."""
    
    def __init__(self, input_nc = 9, output_nc = 3, with_dino_feat = True, with_bart_feat = True, image_size=128, num_training_steps=1000, num_inference_steps=50):
        """
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images per head
            num_downs (int)     -- the number of downsamplings in UNet
            num_heads (int)     -- number of decoder heads
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- whether to use dropout in intermediate layers
        """
        super(DiffusionNOCS, self).__init__()

        self.model = UNet2DConditionModel(
            sample_size=image_size,  # the target image resolution
            in_channels=input_nc,  # the number of input channels, 3 for RGB images
            out_channels=output_nc,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            cross_attention_dim=768,
        )

        self.with_dino_feat = with_dino_feat
        self.with_bart_feat = with_bart_feat

        if self.with_dino_feat:
            self.dino_model = AutoModel.from_pretrained('facebook/dinov2-base')

        if self.with_bart_feat:
            self.bart_model = BartModel.from_pretrained('facebook/bart-base')
            self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

        self.train_noise_scheduler = DDPMScheduler(num_train_timesteps=num_training_steps)
        self.inference_noise_scheduler = DPMSolverSinglestepScheduler(num_train_timesteps=num_training_steps, algorithm_type="dpmsolver++", thresholding=True)

        self.num_training_steps = num_training_steps
        self.num_inference_steps = num_inference_steps

        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._transform = transforms.Compose([
            T.Resize((224, 224)),
            self.norm,
        ])

    def forward(self, rgb_image, normals_image, nocs_gt, obj_names):
        noise = torch.randn(nocs_gt.shape, dtype=nocs_gt.dtype, device=nocs_gt.device)

        bsz = nocs_gt.shape[0]

        timesteps = torch.randint(
            0, self.train_noise_scheduler.config.num_train_timesteps, (bsz,), device=nocs_gt.device
        ).long()

        noisy_latents = self.train_noise_scheduler.add_noise(nocs_gt, noise, timesteps)

        latents = torch.cat([rgb_image, normals_image, noisy_latents], dim=1)

        embeddings = self.get_embeddings(rgb_image, obj_names)

        model_output = self.model(latents, timesteps, embeddings).sample

        loss = F.mse_loss(model_output.float(), noise.float(), reduction="mean")

        return loss

    def get_bart_embeddings(self, obj_names):

        tokens = self.bart_tokenizer(obj_names, return_tensors="pt", padding=True).to(self.bart_tokenizer.device)
        with torch.no_grad():
            bart_embeddings = self.bart_model(**tokens)  

        bart_embeddings = bart_embeddings.last_hidden_state
        print("bart_embeddings: ", bart_embeddings.shape)
        return bart_embeddings

    def get_dino_embeddings(self, rgb_image):
        rgb_image = self._transform(((rgb_image + 1) / 2))

        with torch.no_grad():
            dino_embeddings = self.dino_model(rgb_image)

        dino_embeddings = dino_embeddings.last_hidden_state
        print("dino_embeddings: ", dino_embeddings.shape)

        return dino_embeddings

    def get_embeddings(self, rgb_image, obj_names):

        # combined embeddings
        if self.with_dino_feat and self.with_bart_feat:
            dino_embeddings = self.get_dino_embeddings(rgb_image)
            bart_embeddings = self.get_bart_embeddings(obj_names)

            repeat_factor = (dino_embeddings.size(1) + bart_embeddings.size(1) - 1) // bart_embeddings.size(1)
            bart_repeated = bart_embeddings.repeat(1, repeat_factor, 1)
            bart_repeated = bart_repeated[:, :dino_embeddings.size(1), :]
        
            embeddings = torch.cat((dino_embeddings, bart_repeated), dim=1)

        # dino only
        elif self.with_dino_feat and not self.with_bart_feat:
            embeddings = self.get_dino_embeddings(rgb_image)

        # bart only
        elif self.with_bart_feat and not self.with_dino_feat:
            embeddings = self.get_bart_embeddings(obj_names)
        
        else:
            print("pick either/or dino bart")

        return embeddings

    def inference(self, rgb_image, normals_image, embeddings):
        self.inference_noise_scheduler.set_timesteps(self.num_inference_steps)

        nocs_noise = torch.randn(rgb_image.shape, dtype=rgb_image.dtype, device=rgb_image.device)

        for timestep in tqdm(self.inference_noise_scheduler.timesteps):

            input = torch.cat(
                [rgb_image, normals_image, nocs_noise], dim=1
            )  # this order is important

            with torch.no_grad():
                noisy_residual = self.model(input, timestep, embeddings).sample
            previous_noisy_sample = self.inference_noise_scheduler.step(noisy_residual, timestep, nocs_noise).prev_sample

            nocs_noise = previous_noisy_sample

        nocs_estimated = ((nocs_noise + 1 ) / 2)

        return nocs_estimated
