import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import datasets
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from packaging import version
from torchvision import transforms
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

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.embeddings import LabelEmbedding
from transformers import AutoImageProcessor, AutoModel

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

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