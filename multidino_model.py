import torch
import torch.nn as nn
import numpy as np
from transformers import ViTModel, Dinov2Model, Dinov2Config, DPTConfig, DPTModel, DPTPreTrainedModel
from transformers.models.dpt.modeling_dpt import DPTNeck
from transformers.utils.backbone_utils import load_backbone 

class DPT(DPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.backbone = None
        if config.is_hybrid is False and (config.backbone_config is not None or config.backbone is not None):
            self.backbone = load_backbone(config)
        else:
            self.dpt = DPTModel(config, add_pooling_layer=False)

        self.dpt = DPTModel(config, add_pooling_layer=False)
        self.neck = DPTNeck(config)
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=True,
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
                output_hidden_states=True,  # we need the intermediate hidden states
                return_dict=return_dict,
            )

            # only keep certain features based on config.backbone_out_indices
            # note that the hidden_states also include the initial embeddings
            hidden_states = outputs.hidden_states if return_dict else outputs[1]
            # only keep certain features based on config.backbone_out_indices
            # note that the hidden_states also include the initial embeddings

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
    def __init__(self, input_resolution=224, feature_dim=768, num_blocks=5, num_bins=50):
        super(MultiDINO, self).__init__()

        self.input_resolution = input_resolution
        self.feature_dim = 768
        self.num_bins = num_bins
        
        backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-base", out_features=["stage1", "stage2", "stage3", "stage4"], reshape_hidden_states=False)
        config = DPTConfig(backbone_config=backbone_config, add_pooling_layer=False)
        config.fusion_hidden_size = 224
        self.dpt = DPT(config)

        features = config.fusion_hidden_size

        # Stack of 10 self-attention blocks
        # self.attention_blocks = nn.ModuleList([
        #     nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8) for _ in range(num_blocks)
        # ])

        self.geometry_neck = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # Upsample to double the size
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.mask_head = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),  # Output a single channel mask
            nn.Upsample(size=(224, 224), mode="bilinear", align_corners=True),  # Ensure output is 224x224
            nn.Sigmoid(),  # Use sigmoid to squash output between 0 and 1
        )

        self.nocs_head = nn.Sequential(
            nn.Conv2d(32, self.num_bins * 3, kernel_size=1, stride=1, padding=0),  # Output a single channel mask
            nn.Upsample(size=(224, 224), mode="bilinear", align_corners=True),  # Ensure output is 224x224
        )

        self.rotation_head = nn.Sequential(
            nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0),  # Output 4 components of the quaternion
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to reduce to (batch_size, 4, 1, 1)
            nn.Flatten()  # Flatten to shape (batch_size, 4)
        )

        # self.rotation_head = nn.Sequential(
        #     nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 4, kernel_size=1, stride=1, padding=0),  # Output 4 components of the quaternion
        #     nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to reduce to (batch_size, 4, 1, 1)
        #     nn.Flatten()  # Flatten to shape (batch_size, 4)
        # )
       
    def forward(self, x):
        outputs = self.dpt(
            x,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=False,
        )
      
        geometry_features = self.geometry_neck(outputs)
        # Instance mask prediction [batch_size, 1, 224, 224]
        mask = self.mask_head(geometry_features)
        # # NOCS map prediction [batch_size, 3, 224, 224]
        nocs_logits = self.nocs_head(geometry_features)
        
        rotation = self.rotation_head(geometry_features)

        batch_size = nocs_logits.size(0)
        nocs_logits = nocs_logits.view(batch_size, 3, self.num_bins, 224, 224)

        return nocs_logits, mask, rotation