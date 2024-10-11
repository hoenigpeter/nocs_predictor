import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import ViTModel, Dinov2Model, Dinov2Config, DPTConfig, DPTModel, DPTPreTrainedModel
from transformers.models.dpt.modeling_dpt import DPTNeck
from transformers.utils.backbone_utils import load_backbone 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, device='cuda:0'):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)  # Create encoding on the specified device
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(torch.log(torch.tensor(10000.0, device=device)) / d_model))
        
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)


    def forward(self, x):
        # Ensure x has the correct shape for positional encoding
        batch_size, seq_length, _ = x.size()  # Get batch size and sequence length (num_patches)
        assert seq_length <= self.encoding.size(0), "Input sequence length exceeds encoding length"
        
        # Get positional encoding for the current batch
        pos_enc = self.encoding[:seq_length, :].unsqueeze(0)  # Shape: [1, seq_length, d_model]
        
        return x + pos_enc  # Add positional encoding based on sequence length

class TransformerEncoder(nn.Module):
    def __init__(self, num_blocks=10, feature_dim=4096, nhead=1, patch_size=16):
        super(TransformerEncoder, self).__init__()
        self.num_blocks = num_blocks
        self.feature_dim = feature_dim
        self.nhead = nhead
        self.patch_size = patch_size
        self.positional_encoding = PositionalEncoding(feature_dim)

        self.conv1x1 = nn.Conv2d(256, 16, kernel_size=1)

        # Transformer encoder layers
        self.attention_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=self.feature_dim, nhead=self.nhead) for _ in range(self.num_blocks)
        ])

    def forward(self, x):
        # x is of shape [batch_size, channels, height, width]
        x = self.conv1x1(x)

        batch_size, channels, height, width = x.size()

        # Step 1: Patch extraction
        num_patches_height = height // self.patch_size
        num_patches_width = width // self.patch_size
        
        # Unfold (extract patches)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(batch_size, channels, num_patches_height * num_patches_width, self.patch_size, self.patch_size)

        # Step 2: Flatten patches to create patch embeddings
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # Rearrange to [batch_size, num_patches, channels, patch_size, patch_size]
        x = x.view(batch_size, num_patches_height * num_patches_width, -1)  # Flatten patches to [batch_size, num_patches, feature_dim]

        # Step 3: Add positional encodings
        x = self.positional_encoding(x)

        # Step 4: Feed into transformer blocks
        for block in self.attention_blocks:
            x = block(x)

        # Step 5: Reshape back to original dimensions
        # Reshape x to [batch_size, num_patches, channels, patch_size, patch_size]
        x = x.view(batch_size, num_patches_height, num_patches_width, channels, -1)
        x = x.permute(0, 3, 1, 2, 4)  # Rearrange to [batch_size, channels, num_patches_height, num_patches_width, patch_size]
        
        # Step 6: Combine patches back into original image size
        # Reshape to [batch_size, channels, height, width]
        x = x.contiguous().view(batch_size, channels, height, width)
        
        return x
    
class DPT(DPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.backbone = None
        if config.is_hybrid is False and (config.backbone_config is not None or config.backbone is not None):
            self.backbone = load_backbone(config)
        else:
            self.dpt = DPTModel(config, add_pooling_layer=False)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.dpt = DPTModel(config, add_pooling_layer=False)
        self.neck = DPTNeck(config)
        # self.config.neck_hidden_sizes = [96, 192, 384, 768]

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

        else:
            outputs = self.dpt(
                pixel_values,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )

            hidden_states = outputs.hidden_states if return_dict else outputs[1]

            if not self.config.is_hybrid:
                hidden_states = [
                    feature for idx, feature in enumerate(hidden_states[1:]) if idx in self.config.backbone_out_indices
                ]
            else:
                backbone_hidden_states = outputs.intermediate_activations if return_dict else list(outputs[-1])
                backbone_hidden_states.extend(
                    feature
                    for idx, feature in enumerate(hidden_states[1:])
                    if idx in self.config.backbone_out_indices[2:]
                )

                hidden_states = backbone_hidden_states

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
    def __init__(self, input_resolution=256, num_bins=50):
        super(MultiDINO, self).__init__()

        self.nhead = 8
        self.num_bins = num_bins
        self.input_resolution=input_resolution
        self.num_blocks = 1

        backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-base", out_features=["stage1", "stage2", "stage3", "stage4"], reshape_hidden_states=False)
        config = DPTConfig(backbone_config=backbone_config, add_pooling_layer=False)

        self.dpt = DPT(config)

        features = 256

        #self.transformer_encoder = TransformerEncoder(num_blocks=self.num_blocks, feature_dim=4096, nhead=self.nhead, patch_size=16)

        self.geometry_neck = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # Upsample to double the size
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),  # Output a single channel mask
            nn.Sigmoid(),  # Use sigmoid to squash output between 0 and 1
        )

        self.nocs_head = nn.Sequential(
            nn.Conv2d(32, self.num_bins * 3, kernel_size=1, stride=1, padding=0),  # Output a single channel mask
        )

        self.rotation_head = nn.Sequential(
            nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0),  # Output 4 components of the quaternion
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to reduce to (batch_size, 4, 1, 1)
            nn.Flatten()  # Flatten to shape (batch_size, 4)
        )
       
    def forward(self, x):
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        outputs = self.dpt(
            x_resized,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

        #outputs = self.transformer_encoder(outputs)   

        geometry_features = self.geometry_neck(outputs)

        mask = self.mask_head(geometry_features)

        nocs_logits = self.nocs_head(geometry_features)
        rotation = self.rotation_head(geometry_features)

        batch_size = nocs_logits.size(0)
        nocs_logits = nocs_logits.view(batch_size, 3, self.num_bins, self.input_resolution, self.input_resolution)

        return nocs_logits, mask, rotation