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

# Star Head
class StarHead(nn.Module):
    def __init__(self):
        super(StarHead, self).__init__()
        # Conv layers to process the input feature map
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=1)  # Output 3D points (3 channels)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        star_output = self.conv3(x)  # Final output shape: (batch_size, 3, 256, 256)
        return star_output

# Dash Head
class DashHead(nn.Module):
    def __init__(self):
        super(DashHead, self).__init__()
        # Conv layers to process the input feature map
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=1)  # Output 3D vectors (3 channels)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        dash_output = self.conv3(x)  # Final output shape: (batch_size, 3, 256, 256)
        return dash_output

class SymmetryHead(nn.Module):
    def __init__(self, num_classes=12):
        super(SymmetryHead, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_classes)  # Output the number of symmetry classes

    def forward(self, x):
        x = self.global_avg_pool(x)  # Shape: (batch_size, 256, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, 256)
        x = F.relu(self.fc1(x))
        symmetry_order_logits = self.fc2(x)  # Shape: (batch_size, num_classes)

        # Apply softmax to get probabilities
        symmetry_order_probabilities = F.softmax(symmetry_order_logits, dim=1)

        # Get the predicted symmetry order (discrete integer value)
        predicted_order = torch.argmax(symmetry_order_probabilities, dim=1) + 1  # Add 1 to convert to 1-indexed

        #return predicted_order
        return torch.full((x.size(0),), 3, device=x.device)
    
class RotationHead(nn.Module):
    def __init__(self):
        super(RotationHead, self).__init__()
        # Simple CNN to predict quaternion (4 values)
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * 128 * 128, 4)  # Predict 4-dimensional quaternion


    def forward(self, nocs_map):
        x = F.relu(self.conv1(nocs_map))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        quaternion = self.fc(x)
        # Normalize to ensure unit quaternion
        quaternion = self.normalize_quaternion(quaternion)

        return quaternion
    
    def normalize_quaternion(self, q):
        """
        Normalize the quaternion to ensure it's a unit quaternion.
        Args:
        - q: (batch_size, 4), predicted quaternion

        Returns:
        - normalized_q: (batch_size, 4), normalized quaternion
        """
        norm = torch.norm(q, p=2, dim=1, keepdim=True)  # Calculate L2 norm
        normalized_q = q / norm  # Normalize the quaternion
        return normalized_q
    
class MultiDINO(nn.Module):
    def __init__(self, input_resolution=256, num_bins=50, num_labels=10, freeze_backbone=False):
        super(MultiDINO, self).__init__()

        self.num_bins = num_bins
        self.input_resolution=input_resolution
        self.freeze_backbone = freeze_backbone
        self.num_labels = num_labels

        backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-base", out_features=["stage2", "stage5", "stage8", "stage11"], reshape_hidden_states=False)
        config = DPTConfig(backbone_config=backbone_config, add_pooling_layer=False)

        self.dpt = DPT(config, self.freeze_backbone)
        self.star_head = StarHead()
        self.dash_head = DashHead()
        self.symmetry_head = SymmetryHead()
        self.rotation_head = RotationHead()

    def destarize(self, star_output, symmetry_order):
        """
        Reverse the star transformation to convert it back to object points.
        Args:
        - star_output: (batch_size, 3, 128, 128) from StarHead
        - symmetry_order: (batch_size, 1) from SymmetryHead
        
        Returns:
        - destarized_points: (batch_size, 3, 128, 128), object points in Cartesian coordinates
        """

        n = symmetry_order  # Remove unnecessary dimensions, now n.shape is (batch_size,)

        # Get cylindrical coordinates from star output
        rho = torch.sqrt(star_output[:, 0]**2 + star_output[:, 1]**2)  # Radius
        theta = torch.atan2(star_output[:, 1], star_output[:, 0]) / n.view(-1, 1, 1)  # Ensure n has shape (batch_size, 1, 1) for broadcasting
        z = star_output[:, 2]  # Z remains the same
        
        # Convert back to Cartesian coordinates
        x = rho * torch.cos(theta)
        y = rho * torch.sin(theta)
        
        # Ensure all components have shape (batch_size, 128, 128)
        return torch.stack([x, y, z], dim=1)  # Shape: (batch_size, 3, 128, 128)


    def dash_disambiguation(self, destarized_points, dash_output):
        """
        Use dash output to disambiguate points in the destarized points.
        Args:
        - destarized_points: (batch_size, 3, 256, 256), destarized points from star head
        - dash_output: (batch_size, 3, 256, 256) from DashHead
        
        Returns:
        - disambiguated_points: (batch_size, 3, 256, 256), disambiguated object points
        """
        # For now, we simply adjust the destarized points using the dash output
        # This step could be refined later to account for more complex disambiguation
        disambiguated_points = destarized_points + dash_output
        
        return disambiguated_points


    def predict_nocs(self, star_output, dash_output, symmetry_order):
        """
        Predict the NOCS map by applying the destarization and disambiguation process.
        Args:
        - star_output: Output of the StarHead
        - dash_output: Output of the DashHead
        - symmetry_order: Output of the SymmetryHead
        
        Returns:
        - nocs_map: Predicted NOCS map
        """
        # Step 1: Destarize the star output
        destarized_points = self.destarize(star_output, symmetry_order)
        
        # Step 2: Apply dash disambiguation
        disambiguated_points = self.dash_disambiguation(destarized_points, dash_output)
        
        # The disambiguated points are the final NOCS map
        return disambiguated_points

    def forward(self, x):
        x_resized = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=True)
        outputs = self.dpt(
            x_resized,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        )

        star_output = self.star_head(outputs)       # Shape: (batch_size, 3, 256, 256)
        dash_output = self.dash_head(outputs)       # Shape: (batch_size, 3, 256, 256)
        symmetry_order = self.symmetry_head(outputs)  # Shape: (batch_size, 1)

        nocs_output = self.predict_nocs(star_output, dash_output, symmetry_order)
        nocs_output = torch.tanh(nocs_output)

        # Predict rotation quaternion from NOCS map
        predicted_quaternion = self.rotation_head(nocs_output)
        
        return nocs_output, star_output, dash_output, symmetry_order, predicted_quaternion
