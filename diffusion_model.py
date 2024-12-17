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
    UNet2DConditionModel,
    UNet2DModel
)

from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel, BartTokenizer, BartModel
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from transformers import AutoImageProcessor, AutoModel

import torch
import torch.nn as nn

class DiffusionNOCS(nn.Module):
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
        super(DiffusionNOCS, self).__init__()

        self.model = UNet2DModel(
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
        )

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=num_training_steps)
        self.num_training_steps = num_training_steps
        self.num_inference_steps = num_inference_steps

    def forward(self, rgb_image, nocs_gt):
        # sample noise
        noise = torch.randn(nocs_gt.shape, dtype=nocs_gt.dtype, device=nocs_gt.device)
        # batch size of noise
        bsz = nocs_gt.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=nocs_gt.device
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(nocs_gt, noise, timesteps)

        latents = torch.cat([rgb_image, noisy_latents], dim=1)

        model_output = self.model(latents, timesteps).sample

        loss = F.mse_loss(model_output.float(), noise.float(), reduction="mean")

        return loss
        
    def inference(self, rgb_image):
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        nocs_noise = torch.randn(rgb_image.shape, dtype=rgb_image.dtype, device=rgb_image.device)

        for timestep in tqdm(self.noise_scheduler.timesteps):

            input = torch.cat(
                [rgb_image, nocs_noise], dim=1
            )  # this order is important

            with torch.no_grad():
                noisy_residual = self.model(input, timestep).sample
            previous_noisy_sample = self.noise_scheduler.step(noisy_residual, timestep, nocs_noise).prev_sample

            nocs_noise = previous_noisy_sample

        nocs_estimated = nocs_noise

        return nocs_estimated
    
class DiffusionNOCSDino(nn.Module):
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
        super(DiffusionNOCSDino, self).__init__()

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

        # self.bart_model = BartModel.from_pretrained('facebook/bart-base')
        # self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=num_training_steps)
        self.num_training_steps = num_training_steps
        self.num_inference_steps = num_inference_steps

    def forward(self, rgb_image, nocs_gt):
        # sample noise
        noise = torch.randn(nocs_gt.shape, dtype=nocs_gt.dtype, device=nocs_gt.device)
        # batch size of noise
        bsz = nocs_gt.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=nocs_gt.device
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(nocs_gt, noise, timesteps)

        latents = torch.cat([rgb_image, noisy_latents], dim=1)

        combined_embeddings = self.get_embeddings(rgb_image)

        model_output = self.model(latents, timesteps, combined_embeddings).sample

        loss = F.mse_loss(model_output.float(), noise.float(), reduction="mean")

        return loss
    
    def get_embeddings(self, rgb_image):

        # DINO Embeddings
        images_resized = F.interpolate(rgb_image, size=(224, 224), mode='bilinear', align_corners=False)
        with torch.no_grad():
            dino_embeddings = self.dino_model(images_resized)
        dino_embeddings = dino_embeddings[0]

        return dino_embeddings

    def inference(self, rgb_image):
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        nocs_noise = torch.randn(rgb_image.shape, dtype=rgb_image.dtype, device=rgb_image.device)

        combined_embeddings = self.get_embeddings(rgb_image)

        for timestep in tqdm(self.noise_scheduler.timesteps):

            input = torch.cat(
                [rgb_image, nocs_noise], dim=1
            )  # this order is important

            with torch.no_grad():
                noisy_residual = self.model(input, timestep, combined_embeddings).sample
            previous_noisy_sample = self.noise_scheduler.step(noisy_residual, timestep, nocs_noise).prev_sample

            nocs_noise = previous_noisy_sample

        nocs_estimated = nocs_noise

        return nocs_estimated
    
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

        self.noise_scheduler = DDPMScheduler(num_train_timesteps=num_training_steps)
        self.num_training_steps = num_training_steps
        self.num_inference_steps = num_inference_steps

    def forward(self, rgb_image, nocs_gt, obj_names):
        # sample noise
        noise = torch.randn(nocs_gt.shape, dtype=nocs_gt.dtype, device=nocs_gt.device)
        # batch size of noise
        bsz = nocs_gt.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=nocs_gt.device
        ).long()

        noisy_latents = self.noise_scheduler.add_noise(nocs_gt, noise, timesteps)

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

    def inference(self, rgb_image, obj_names):
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        nocs_noise = torch.randn(rgb_image.shape, dtype=rgb_image.dtype, device=rgb_image.device)

        combined_embeddings = self.get_embeddings(rgb_image, obj_names)

        for timestep in tqdm(self.noise_scheduler.timesteps):

            input = torch.cat(
                [rgb_image, nocs_noise], dim=1
            )  # this order is important

            with torch.no_grad():
                noisy_residual = self.model(input, timestep, combined_embeddings).sample
            previous_noisy_sample = self.noise_scheduler.step(noisy_residual, timestep, nocs_noise).prev_sample

            nocs_noise = previous_noisy_sample

        nocs_estimated = nocs_noise

        return nocs_estimated