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

class MultiDINO(nn.Module):
    def __init__(self, input_resolution=256, num_bins=50, num_labels=10, freeze_backbone=False):
        super(MultiDINO, self).__init__()

        self.nhead = 4
        self.num_bins = num_bins
        self.input_resolution=input_resolution
        self.num_blocks = 1
        self.freeze_backbone = freeze_backbone
        self.num_labels = num_labels

        backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-base", out_features=["stage2", "stage5", "stage8", "stage11"], reshape_hidden_states=False)
        config = DPTConfig(backbone_config=backbone_config, add_pooling_layer=False)

        self.dpt = DPT(config, self.freeze_backbone)

        #self.transformer_encoder = TransformerEncoder(num_blocks=self.num_blocks, feature_dim=256, nhead=self.nhead, patch_size=16)

        #self.geometry_head = UNetGeometryHead()

        # self.geometry_head = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Increase channels for more features
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Match the output channels to the existing heads
        #     nn.ReLU(),
        # )

        # Mask head
        # self.mask_head = nn.Sequential(
        #     nn.Conv2d(256, 128, kernel_size=3, padding=1),  # Increase channels for more features
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Match the output channels to the existing heads
        #     nn.ReLU(),
        #     nn.Conv2d(64, 1, kernel_size=3, padding=1),  # Input channels from geometry head
        #     nn.Sigmoid()
        # )

        # NOCS head
       # Learning spatial relationships for x, y, z
        self.x_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1, False),
            nn.Conv2d(256, self.num_bins, kernel_size=1),
        )

        self.y_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1, False),
            nn.Conv2d(256, self.num_bins, kernel_size=1),
        )
        
        self.z_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1, False),
            nn.Conv2d(256, self.num_bins, kernel_size=1),
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1, False),
            nn.Conv2d(256, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.cls_head = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.1, False),
            nn.Conv2d(256, 256, kernel_size=1),  # Intermediate conv layer
            nn.AdaptiveAvgPool2d((1, 1)),        # Global Average Pooling to reduce to [batch_size, 256, 1, 1]
            nn.Flatten(),                        # Flatten the output to [batch_size, 256]
            nn.Linear(256, self.num_labels)       # Fully connected layer for classification
        )

        # # Rotation head
        # self.rotation_head = nn.Sequential(
        #     nn.Conv2d(256, 4, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(),  # Activation for non-linearity
        #     nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        #     nn.Flatten()  # Flatten to shape (batch_size, 4)
        # )

        # Rotation head
        # self.rotation_head = nn.Sequential(
        #     nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(),  # Activation for non-linearity
        #     nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
        #     nn.Flatten()  # Flatten to shape (batch_size, 4)
        # )
       
    def forward(self, x):
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        outputs = self.dpt(
            x_resized,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

        #outputs = F.interpolate(outputs, size=(128, 128), mode='bilinear', align_corners=True)
        #outputs = self.geometry_head(outputs) 

        # mask = self.mask_head(outputs)

        #nocs_logits = self.nocs_head(outputs)
        # Apply spatial learning for each NOCS coordinate
        
        # Final MLP to produce classification logits over 256 bins
        x_logits = self.x_head(outputs)  # Shape: [batch_size, 256, 128, 128]
        y_logits = self.y_head(outputs)  # Shape: [batch_size, 256, 128, 128]
        z_logits = self.z_head(outputs)  # Shape: [batch_size, 256, 128, 128]
        cls_logits = self.cls_head(outputs)
        
        # Concatenate NOCS logits along the channel dimension
        # nocs_logits = torch.cat((x_logits, y_logits, z_logits), dim=1)

        quaternions = self.rotation_head(outputs)
        masks = self.mask_head(outputs)
        #rotation = self.rotation_head(outputs)

        #batch_size = nocs_logits.size(0)
        #nocs_logits = nocs_logits.view(batch_size, 3, self.num_bins, self.input_resolution, self.input_resolution)

        return x_logits, y_logits, z_logits, cls_logits, masks
    
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

# class UNetGeometryHead(nn.Module):
#     def __init__(self):
#         super(UNetGeometryHead, self).__init__()

#         # Encoding path
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(256, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.pool1 = nn.MaxPool2d(kernel_size=2)  # Output: (batch_size, 128, 128, 128)

#         self.enc2 = nn.Sequential(
#             nn.Conv2d(128, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.pool2 = nn.MaxPool2d(kernel_size=2)  # Output: (batch_size, 64, 64, 64)

#         self.enc3 = nn.Sequential(
#             nn.Conv2d(64, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )
#         self.pool3 = nn.MaxPool2d(kernel_size=2)  # Output: (batch_size, 32, 32, 32)

#         # Bottleneck
#         self.bottleneck = nn.Sequential(
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#         )

#         # Decoding path
#         self.upconv3 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)  # Upsampling
#         self.upconv2 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # Upsampling
#         self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # Upsampling

#         self.final_conv = nn.Conv2d(256, 128, kernel_size=1)  # Output: (batch_size, 32, 128, 128)

#     def forward(self, x):
#         enc1_out = self.enc1(x)  # (batch_size, 128, 128, 128)
#         pool1_out = self.pool1(enc1_out)

#         enc2_out = self.enc2(pool1_out)  # (batch_size, 64, 64, 64)
#         pool2_out = self.pool2(enc2_out)

#         enc3_out = self.enc3(pool2_out)  # (batch_size, 32, 32, 32)
#         pool3_out = self.pool3(enc3_out)

#         bottleneck_out = self.bottleneck(pool3_out)  # (batch_size, 32, 32, 32)

#         dec3_out = self.upconv3(bottleneck_out)  # (batch_size, 32, 64, 64)
#         dec3_out = torch.cat((dec3_out, enc3_out), dim=1)  # Concatenate along channel dimension

#         dec2_out = self.upconv2(dec3_out)  # (batch_size, 64, 128, 128)
#         dec2_out = torch.cat((dec2_out, enc2_out), dim=1)  # Concatenate along channel dimension
        
#         dec1_out = self.upconv1(dec2_out)  # (batch_size, 128, 256, 256)
#         dec1_out = torch.cat((dec1_out, enc1_out), dim=1)  # Concatenate along channel dimension

#         output = self.final_conv(dec1_out)  # Final output: (batch_size, 32, 128, 128)
#         return output

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