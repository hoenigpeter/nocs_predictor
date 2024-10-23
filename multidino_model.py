import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import ViTModel, Dinov2Model, Dinov2Config, DPTConfig, DPTModel, DPTPreTrainedModel
from transformers.models.dpt.modeling_dpt import DPTReassembleLayer, DPTFeatureFusionStage
from transformers.utils.backbone_utils import load_backbone
from transformers.utils import torch_int
from transformers.activations import ACT2FN 
from typing import List, Optional, Set, Tuple, Union

from sklearn.decomposition import PCA

def _get_backbone_hidden_size(config):
    if config.backbone_config is not None and config.is_hybrid is False:
        return config.backbone_config.hidden_size
    else:
        return config.hidden_size
    
class DPTReassembleStage(nn.Module):
    """
    This class reassembles the hidden states of the backbone into image-like feature representations at various
    resolutions.

    This happens in 3 stages:
    1. Map the N + 1 tokens to a set of N tokens, by taking into account the readout ([CLS]) token according to
       `config.readout_type`.
    2. Project the channel dimension of the hidden states according to `config.neck_hidden_sizes`.
    3. Resizing the spatial dimensions (height, width).

    Args:
        config (`[DPTConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.layers = nn.ModuleList()
        if config.is_hybrid:
            self._init_reassemble_dpt_hybrid(config)
        else:
            self._init_reassemble_dpt(config)

        self.neck_ignore_stages = config.neck_ignore_stages

    def _init_reassemble_dpt_hybrid(self, config):
        r""" "
        For DPT-Hybrid the first 2 reassemble layers are set to `nn.Identity()`, please check the official
        implementation: https://github.com/isl-org/DPT/blob/f43ef9e08d70a752195028a51be5e1aff227b913/dpt/vit.py#L438
        for more details.
        """
        for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
            if i <= 1:
                self.layers.append(nn.Identity())
            elif i > 1:
                self.layers.append(DPTReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))

        if config.readout_type != "project":
            raise ValueError(f"Readout type {config.readout_type} is not supported for DPT-Hybrid.")

        # When using DPT-Hybrid the readout type is set to "project". The sanity check is done on the config file
        self.readout_projects = nn.ModuleList()
        hidden_size = _get_backbone_hidden_size(config)
        for i in range(len(config.neck_hidden_sizes)):
            if i <= 1:
                self.readout_projects.append(nn.Sequential(nn.Identity()))
            elif i > 1:
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act])
                )

    def _init_reassemble_dpt(self, config):
        for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
            self.layers.append(DPTReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))
            print("poop")

        if config.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            hidden_size = _get_backbone_hidden_size(config)
            for _ in range(len(config.neck_hidden_sizes)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act])
                )

    def forward(self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length + 1, hidden_size)`):
                List of hidden states from the backbone.
        """
        out = []
        
        for i, hidden_state in enumerate(hidden_states):
            if i not in self.neck_ignore_stages:
                # reshape to (batch_size, num_channels, height, width)
                cls_token, hidden_state = hidden_state[:, 0], hidden_state[:, 1:]
                batch_size, sequence_length, num_channels = hidden_state.shape
                if patch_height is not None and patch_width is not None:
                    hidden_state = hidden_state.reshape(batch_size, patch_height, patch_width, num_channels)
                else:
                    size = torch_int(sequence_length**0.5)
                    hidden_state = hidden_state.reshape(batch_size, size, size, num_channels)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

                feature_shape = hidden_state.shape
                if self.config.readout_type == "project":
                    # reshape to (batch_size, height*width, num_channels)
                    hidden_state = hidden_state.flatten(2).permute((0, 2, 1))
                    readout = cls_token.unsqueeze(1).expand_as(hidden_state)
                    # concatenate the readout token to the hidden states and project
                    hidden_state = self.readout_projects[i](torch.cat((hidden_state, readout), -1))
                    # reshape back to (batch_size, num_channels, height, width)
                    hidden_state = hidden_state.permute(0, 2, 1).reshape(feature_shape)
                elif self.config.readout_type == "add":
                    hidden_state = hidden_state.flatten(2) + cls_token.unsqueeze(-1)
                    hidden_state = hidden_state.reshape(feature_shape)
                hidden_state = self.layers[i](hidden_state)
            out.append(hidden_state)

        return out
    
class DPTNeck(nn.Module):
    """
    DPTNeck. A neck is a module that is normally used between the backbone and the head. It takes a list of tensors as
    input and produces another list of tensors as output. For DPT, it includes 2 stages:

    * DPTReassembleStage
    * DPTFeatureFusionStage.

    Args:
        config (dict): config dict.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # postprocessing: only required in case of a non-hierarchical backbone (e.g. ViT, BEiT)
        if config.backbone_config is not None and config.backbone_config.model_type in ["swinv2"]:
            self.reassemble_stage = None
        else:
            self.reassemble_stage = DPTReassembleStage(config)

        self.convs = nn.ModuleList()
        for channel in config.neck_hidden_sizes:
            self.convs.append(nn.Conv2d(channel, config.fusion_hidden_size, kernel_size=3, padding=1, bias=False))

        # fusion
        self.fusion_stage = DPTFeatureFusionStage(config)

    def forward(self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, hidden_size, height, width)`):
                List of hidden states from the backbone.
        """
        if not isinstance(hidden_states, (tuple, list)):
            raise TypeError("hidden_states should be a tuple or list of tensors")

        if len(hidden_states) != len(self.config.neck_hidden_sizes):
            raise ValueError("The number of hidden states should be equal to the number of neck hidden sizes.")

        # postprocess hidden states
        if self.reassemble_stage is not None:
            hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]

        # fusion blocks
        output = self.fusion_stage(features)

        return output
    
class DPT(DPTPreTrainedModel):
    def __init__(self, config, freeze_backbone=False):
        super().__init__(config, freeze_backbone)

        self.backbone = None
        if config.is_hybrid is False and (config.backbone_config is not None or config.backbone is not None):
            self.backbone = load_backbone(config)

            for param in self.backbone.parameters():
                if freeze_backbone == True:
                    param.requires_grad = False
                print(param.requires_grad)


        else:
            self.dpt = DPTModel(config, add_pooling_layer=False)
        
        self.dpt = DPTModel(config, add_pooling_layer=False)

        # self.config.neck_hidden_sizes = [96, 192, 384, 384, 768, 768]
        # self.config.reassemble_factors = [4, 2, 1, 1, 0.5, 0.5]

        self.config.neck_hidden_sizes = [96, 192, 384, 768]
        self.config.reassemble_factors = [4, 2, 1, 0.5]

        self.neck = DPTNeck(config)

        self.post_init()

    def forward(
        self,
        pixel_values,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=False,
        return_dict=False,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        if self.backbone is not None:
            outputs = self.backbone.forward_with_filtered_kwargs(
                pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
            )
            hidden_states = outputs.feature_maps

        patch_height, patch_width = None, None
        if self.config.backbone_config is not None and self.config.is_hybrid is False:
            _, _, height, width = pixel_values.shape
            patch_size = self.config.backbone_config.patch_size
            patch_height = height // patch_size
            patch_width = width // patch_size

        hidden_states = self.neck(hidden_states, patch_height, patch_width)
        hidden_states = hidden_states[self.config.head_in_index]

        return hidden_states

class SimpleViT(nn.Module):
    def __init__(self, input_dim, patch_size, num_patches, embedding_dim=768, num_blocks=10):
        super(SimpleViT, self).__init__()
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.proj = nn.Linear(input_dim, embedding_dim)  # Project to embedding dimension
        self.positional_encodings = nn.Parameter(torch.randn(1, num_patches, embedding_dim))  # Learnable positional encodings
        # Transformer encoder layers
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8) for _ in range(num_blocks)
        ])

    def create_patches(self, x):
        # x shape: (batch_size, channels, height, width)
        batch_size, channels, height, width = x.shape
        
        # Calculate number of patches
        num_patches_height = height // self.patch_size
        num_patches_width = width // self.patch_size
        patch_dim = channels * self.patch_size * self.patch_size  # Flattened patch size

        # Reshape to (batch_size, num_patches_height, num_patches_width, channels, patch_size, patch_size)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)  # (batch_size, channels, num_patches_height, num_patches_width, patch_size, patch_size)
        
        # Rearrange dimensions and flatten patches
        x = x.contiguous().view(batch_size, channels, num_patches_height, num_patches_width, self.patch_size, self.patch_size)  # Ensure contiguous
        x = x.permute(0, 2, 3, 1, 4, 5)  # Rearrange to (batch_size, num_patches_height, num_patches_width, channels, patch_size, patch_size)
        x = x.reshape(batch_size, num_patches_height * num_patches_width, patch_dim)  # (batch_size, num_patches, patch_dim)
    
        return x

    def forward(self, x):
        # Create patches
        batch_size = x.size(0)
        x = self.create_patches(x)  # (batch_size, num_patches, patch_dim)

        x = self.proj(x)  # (batch_size, num_patches, embedding_dim)

        x = x + self.positional_encodings  # (batch_size, num_patches, embedding_dim)

        for block in self.transformer_blocks:
            x = block(x)

        x = x.view(batch_size, 32, 32, self.embedding_dim)

        x = x.permute(0, 3, 1, 2)

        return x

class UpsamplingNetwork(nn.Module):
    def __init__(self, input_channels=768, output_channels=3):
        super(UpsamplingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 512, kernel_size=3, padding=1)  # From 768 to 512 channels
        self.relu1 = nn.ReLU()
        
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # Upsample to 64x64
        self.relu2 = nn.ReLU()

        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Upsample to 128x128
        self.relu4 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, output_channels, kernel_size=3, padding=1)  # Final output to 50 channels

    def forward(self, x):
        # Reshape input from (batch_size, 32, 32, 768) to (batch_size, 768, 32, 32)
        x = self.conv1(x)  # Shape: (batch_size, 512, 32, 32)
        x = self.relu1(x)

        x = self.upsample1(x)  # Shape: (batch_size, 256, 64, 64)
        x = self.relu2(x)

        x = self.conv2(x)  # Shape: (batch_size, 128, 64, 64)
        x = self.relu3(x)

        x = self.upsample2(x)  # Shape: (batch_size, 64, 128, 128)
        x = self.relu4(x)

        x = self.conv3(x)  # Shape: (batch_size, 50, 128, 128)

        return x
        
class MultiDINO(nn.Module):
    def __init__(self, input_resolution=256, num_bins=256, freeze_backbone=False):
        super(MultiDINO, self).__init__()

        self.nhead = 4
        self.num_bins = num_bins
        self.input_resolution=input_resolution
        self.num_blocks = 1
        self.freeze_backbone = freeze_backbone

        backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-base", out_features=["stage2", "stage5", "stage8", "stage11"], reshape_hidden_states=False)
        config = DPTConfig(backbone_config=backbone_config, add_pooling_layer=False)

        self.dpt = DPT(config, self.freeze_backbone)
    
        input_channels = 256
        height = 128
        width = 128
        patch_size = 4  # Patch size of 16x16
        num_patches = (height // patch_size) * (width // patch_size)  # Total patches

        self.vit = SimpleViT(input_dim=input_channels * patch_size * patch_size,
                   patch_size=patch_size, 
                   num_patches=num_patches,
                   embedding_dim=768)
        
        self.x_head = UpsamplingNetwork(input_channels=768, output_channels=50)
        self.y_head = UpsamplingNetwork(input_channels=768, output_channels=50)
        self.z_head = UpsamplingNetwork(input_channels=768, output_channels=50)

        self.mask_head = nn.Sequential(
            UpsamplingNetwork(input_channels=768, output_channels=1),
            nn.Sigmoid()
        )

        # self.x_head = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Increase channels for more features
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Match the output channels to the existing heads
        #     nn.ReLU(),
        #     nn.Conv2d(64, self.num_bins, kernel_size=3, padding=1),  # Input channels from geometry head
        # )

        # self.y_head = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Increase channels for more features
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Match the output channels to the existing heads
        #     nn.ReLU(),
        #     nn.Conv2d(64, self.num_bins, kernel_size=3, padding=1),  # Input channels from geometry head
        # )
        
        # self.z_head = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Increase channels for more features
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Match the output channels to the existing heads
        #     nn.ReLU(),
        #     nn.Conv2d(64, self.num_bins, kernel_size=3, padding=1),  # Input channels from geometry head
        # )

        self.rotation_head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Adjust input channels to 3
            nn.ReLU(),  # Activation for non-linearity
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # More channels for complexity
            nn.ReLU(),
            nn.Conv2d(128, 6, kernel_size=3, stride=1, padding=1),  # Output 6 channels for the 6D rotation
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to (batch_size, 6, 1, 1)
            nn.Flatten()  # Flatten to shape (batch_size, 6)
        )

        # self.mask_head = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Dropout(0.1, False),
        #     nn.Conv2d(256, 1, kernel_size=1),
        #     nn.Sigmoid()
        # )

    def rot6d_to_rotmat(self, x):
        """Convert 6D rotation representation to 3x3 rotation matrix."""
        x = x.view(-1, 3, 2)  # Reshape into two 3D vectors
        a1 = x[:, :, 0]  # First 3D vector
        a2 = x[:, :, 1]  # Second 3D vector

        # Normalize a1 to get the first basis vector
        b1 = nn.functional.normalize(a1, dim=1)

        # Make a2 orthogonal to b1
        b2 = nn.functional.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1)

        # Compute the third basis vector by taking the cross product
        b3 = torch.cross(b1, b2, dim=1)

        # Form the rotation matrix
        rot_mat = torch.stack([b1, b2, b3], dim=-1)  # Shape: (batch_size, 3, 3)
        return rot_mat

    def forward(self, x):
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        outputs = self.dpt(
            x_resized,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        
        vit_features = self.vit(outputs)

        # Final MLP to produce classification logits over 256 bins
        x_logits = self.x_head(vit_features)  # Shape: [batch_size, num_bins, 128, 128]
        y_logits = self.y_head(vit_features)  # Shape: [batch_size, num_bins, 128, 128]
        z_logits = self.z_head(vit_features)  # Shape: [batch_size, num_bins, 128, 128]
        masks = self.mask_head(vit_features)

        # Softmax over the bin dimension for x, y, z logits
        x_bins = torch.softmax(x_logits, dim=1)  # Softmax over the bin dimension
        y_bins = torch.softmax(y_logits, dim=1)
        z_bins = torch.softmax(z_logits, dim=1)

        # Bin centers (shared for x, y, z dimensions)
        bin_centers = torch.linspace(-1, 1, self.num_bins).to(x_logits.device)  # Bin centers

        # Compute the estimated NOCS map for each dimension by multiplying with bin centers and summing over bins
        nocs_x_estimated = torch.sum(x_bins * bin_centers.view(1, self.num_bins, 1, 1), dim=1)
        nocs_y_estimated = torch.sum(y_bins * bin_centers.view(1, self.num_bins, 1, 1), dim=1)
        nocs_z_estimated = torch.sum(z_bins * bin_centers.view(1, self.num_bins, 1, 1), dim=1)

        # Combine the estimated NOCS map from x, y, and z dimensions
        nocs_estimated = torch.stack([nocs_x_estimated, nocs_y_estimated, nocs_z_estimated], dim=1)

        rotation_6d = self.rotation_head(nocs_estimated)
        rotation_matrix = self.rot6d_to_rotmat(rotation_6d)

        return x_logits, y_logits, z_logits, nocs_estimated, masks, rotation_matrix

class MultiDINORot(nn.Module):
    def __init__(self, input_resolution=256, num_bins=256, num_labels=10, freeze_backbone=False):
        super(MultiDINORot, self).__init__()

        self.nhead = 4
        self.num_bins = num_bins
        self.input_resolution=input_resolution
        self.num_blocks = 1
        self.freeze_backbone = freeze_backbone
        self.num_labels = num_labels

        backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-base", out_features=["stage2", "stage5", "stage8", "stage11"], reshape_hidden_states=False)
        config = DPTConfig(backbone_config=backbone_config, add_pooling_layer=False)

        self.dpt = DPT(config, self.freeze_backbone)

        # NOCS head
       # Learning spatial relationships for x, y, z
        self.geometry_head = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.x_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.num_bins, kernel_size=3, padding=1),  # Input channels from geometry head
        )

        self.y_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.num_bins, kernel_size=3, padding=1),  # Input channels from geometry head
        )
        
        self.z_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, self.num_bins, kernel_size=3, padding=1),  # Input channels from geometry head
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Rotation head
        self.rotation_head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Adjust input channels to 3
            nn.ReLU(),  # Activation for non-linearity
            nn.Conv2d(64, 6, kernel_size=3, stride=1, padding=1),  # Output 6 channels for the 6D rotation
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to (batch_size, 6, 1, 1)
            nn.Flatten()  # Flatten to shape (batch_size, 6)
        )

    def rot6d_to_rotmat(self, x):
        """Convert 6D rotation representation to 3x3 rotation matrix."""
        x = x.view(-1, 3, 2)  # Reshape into two 3D vectors
        a1 = x[:, :, 0]  # First 3D vector
        a2 = x[:, :, 1]  # Second 3D vector

        # Normalize a1 to get the first basis vector
        b1 = nn.functional.normalize(a1, dim=1)

        # Make a2 orthogonal to b1
        b2 = nn.functional.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1)

        # Compute the third basis vector by taking the cross product
        b3 = torch.cross(b1, b2, dim=1)

        # Form the rotation matrix
        rot_mat = torch.stack([b1, b2, b3], dim=-1)  # Shape: (batch_size, 3, 3)
        return rot_mat

    def forward(self, x):
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        outputs = self.dpt(
            x_resized,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )
        
        # Final MLP to produce classification logits over 256 bins
        geometry_features = self.geometry_head(outputs)
        x_logits = self.x_head(geometry_features)  # Shape: [batch_size, num_bins, 128, 128]
        y_logits = self.y_head(geometry_features)  # Shape: [batch_size, num_bins, 128, 128]
        z_logits = self.z_head(geometry_features)  # Shape: [batch_size, num_bins, 128, 128]
        masks = self.mask_head(geometry_features)

        # Softmax over the bin dimension for x, y, z logits
        x_bins = torch.softmax(x_logits, dim=1)  # Softmax over the bin dimension
        y_bins = torch.softmax(y_logits, dim=1)
        z_bins = torch.softmax(z_logits, dim=1)

        # Bin centers (shared for x, y, z dimensions)
        bin_centers = torch.linspace(-1, 1, self.num_bins).to(x_logits.device)  # Bin centers

        # Compute the estimated NOCS map for each dimension by multiplying with bin centers and summing over bins
        nocs_x_estimated = torch.sum(x_bins * bin_centers.view(1, self.num_bins, 1, 1), dim=1)
        nocs_y_estimated = torch.sum(y_bins * bin_centers.view(1, self.num_bins, 1, 1), dim=1)
        nocs_z_estimated = torch.sum(z_bins * bin_centers.view(1, self.num_bins, 1, 1), dim=1)
      
        # Combine the estimated NOCS map from x, y, and z dimensions
        nocs_estimated = torch.stack([nocs_x_estimated, nocs_y_estimated, nocs_z_estimated], dim=1)

        rotation_6d = self.rotation_head(nocs_estimated)
        rotation_matrix = self.rot6d_to_rotmat(rotation_6d)

        return x_logits, y_logits, z_logits, nocs_estimated, masks, rotation_matrix

# class MultiDINO(nn.Module):
#     def __init__(self, input_resolution=256, num_bins=50):
#         super(MultiDINO, self).__init__()

#         self.nhead = 8
#         self.num_bins = num_bins
#         self.input_resolution=input_resolution
#         self.num_blocks = 5

#         backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-base", out_features=["stage1", "stage2", "stage3", "stage4"], reshape_hidden_states=False)
#         config = DPTConfig(backbone_config=backbone_config, add_pooling_layer=False)

#         self.dpt = DPT(config)

#         features = 256

#         self.transformer_encoder = TransformerEncoder(num_blocks=self.num_blocks, feature_dim=768, nhead=self.nhead, patch_size=16)

#         self.geometry_neck = nn.Sequential(
#             nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
#             nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # Upsample to double the size
#             nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#         )

#         self.mask_head = nn.Sequential(
#             nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),  # Output a single channel mask
#             nn.Sigmoid(),  # Use sigmoid to squash output between 0 and 1
#         )

#         self.nocs_head = nn.Sequential(
#             nn.Conv2d(32, self.num_bins * 3, kernel_size=1, stride=1, padding=0),  # Output a single channel mask
#         )

#         self.rotation_head = nn.Sequential(
#             nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0),  # Output 4 components of the quaternion
#             nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to reduce to (batch_size, 4, 1, 1)
#             nn.Flatten()  # Flatten to shape (batch_size, 4)
#         )
       
#     def forward(self, x):
#         x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
#         outputs = self.dpt(
#             x_resized,
#             head_mask=None,
#             output_attentions=False,
#             output_hidden_states=False,
#             return_dict=False,
#         )

#         print(outputs.shape)
#         outputs = self.transformer_encoder(outputs)   
#         print(outputs.shape)

#         geometry_features = self.geometry_neck(outputs)

#         mask = self.mask_head(geometry_features)

#         nocs_logits = self.nocs_head(geometry_features)
#         rotation = self.rotation_head(geometry_features)

#         batch_size = nocs_logits.size(0)
#         nocs_logits = nocs_logits.view(batch_size, 3, self.num_bins, self.input_resolution, self.input_resolution)

#         return nocs_logits, mask, rotation



# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=768, device='cuda:0'):
#         super(PositionalEncoding, self).__init__()
#         self.encoding = torch.zeros(max_len, d_model, device=device)  # Create encoding on the specified device
#         position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(torch.log(torch.tensor(10000.0, device=device)) / d_model))
        
#         self.encoding[:, 0::2] = torch.sin(position * div_term)
#         self.encoding[:, 1::2] = torch.cos(position * div_term)


#     def forward(self, x):
#         # Ensure x has the correct shape for positional encoding
#         batch_size, seq_length, _ = x.size()  # Get batch size and sequence length (num_patches)
#         assert seq_length <= self.encoding.size(0), "Input sequence length exceeds encoding length"
        
#         # Get positional encoding for the current batch
#         pos_enc = self.encoding[:seq_length, :].unsqueeze(0)  # Shape: [1, seq_length, d_model]
        
#         return x + pos_enc  # Add positional encoding based on sequence length

# class TransformerEncoder(nn.Module):
#     def __init__(self, num_blocks=10, feature_dim=256, nhead=1, patch_size=16):
#         super(TransformerEncoder, self).__init__()
#         self.num_blocks = num_blocks
#         self.feature_dim = feature_dim
#         self.nhead = nhead
#         self.patch_size = patch_size
#         # self.positional_encoding = PositionalEncoding(feature_dim)

#        # Positional encoding as a learnable parameter, with shape [1, num_patches, embed_dim]
#         self.positional_encoding = nn.Parameter(torch.randn(1, 64, 256))

#         # Transformer encoder layers
#         self.attention_blocks = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=self.nhead) for _ in range(self.num_blocks)
#         ])

#     def forward(self, x):
#         # x is of shape [batch_size, channels, height, width]
#         batch_size, channels, height, width = x.size()

#         # Step 1: Patch extraction
#         num_patches_height = height // self.patch_size
#         num_patches_width = width // self.patch_size
#         num_patches = num_patches_height * num_patches_width
        
#         x = x.view(batch_size, channels, num_patches_height, self.patch_size, num_patches_width, self.patch_size)
#         x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(batch_size, num_patches, channels)

#         print()

#         # Step 2: Flatten patches to create patch embeddings
#         x = x.permute(0, 2, 1, 3, 4).contiguous()  # Rearrange to [batch_size, num_patches, channels, patch_size, patch_size]
#         x = x.view(batch_size, num_patches_height * num_patches_width, -1)  # Flatten patches to [batch_size, num_patches, feature_dim]

#         x = x + self.positional_encoding  # Positional encoding has shape [1, num_patches, embed_dim], broadcasted across batches
#         print(x.shape)

#         # Step 4: Feed into transformer blocks
#         for block in self.attention_blocks:
#             x = block(x)

#         # Reshape from [batch_size, num_patches, embed_dim] back to grid [batch_size, num_patches_h, num_patches_w, embed_dim]
#         x = x.view(batch_size, num_patches_height, num_patches_width, 256)

#         # Permute back to the original image-like shape [batch_size, embed_dim, num_patches_h, patch_size, num_patches_w, patch_size]
#         x = x.permute(0, 3, 1, 4, 2, 5).contiguous()

#         # Now reshape back to [batch_size, embed_dim, height, width]
#         x = x.view(batch_size, 256, height, width)

#         print(x.shape)

#         return x