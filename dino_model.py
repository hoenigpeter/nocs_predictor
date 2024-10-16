import torch
import torch.nn as nn
import numpy as np
from transformers import ViTModel, Dinov2Model, Dinov2Config, DPTConfig, DPTModel, DPTPreTrainedModel
from transformers.models.dpt.modeling_dpt import DPTNeck
from transformers.utils.backbone_utils import load_backbone

class DinoViTEncoder(nn.Module):
    def __init__(self):
        super(DinoViTEncoder, self).__init__()
        #self.vit = ViTModel.from_pretrained('facebook/dino-vits16', output_hidden_states=True)
        self.vit = Dinov2Model.from_pretrained("facebook/dinov2-base", output_hidden_states=True, patch_size=16, ignore_mismatched_sizes=True)
        for param in self.vit.parameters():
            param.requires_grad = False  # Freeze the encoder parameters

    def forward(self, x):
        outputs = self.vit(pixel_values=x)
        return outputs.hidden_states  # Returning all hidden states

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding):
        super(DecoderBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))

class AutoencoderXYZHead(nn.Module):
    def __init__(self, input_resolution):
        super(AutoencoderXYZHead, self).__init__()

        self.input_resolution = input_resolution
        assert input_resolution == 224, "Input resolution must be 224"
        
        self.encoder = DinoViTEncoder()
        
        self.before_bottleneck_size = self.input_resolution // 16  # 224 // 16 = 14
        
        self.bottleneck = nn.Linear(197 * 384, 256)
        
        self.decoder = nn.ModuleList([
            nn.Linear(256, self.before_bottleneck_size * self.before_bottleneck_size * 256),
            nn.Unflatten(1, (256, self.before_bottleneck_size, self.before_bottleneck_size))
        ])

        self.x_head = nn.ModuleList([            
            DecoderBlock(256 + 64, 128, 3, 2, 1, 1),  # Adjusted input channels for concatenation
            DecoderBlock(128, 128, 3, 1, 1, 0),
            DecoderBlock(128 + 64, 64, 3, 2, 1, 1),
            DecoderBlock(64, 64, 3, 1, 1, 0),
            DecoderBlock(64 + 64, 32, 3, 2, 1, 1),
            DecoderBlock(32, 32, 3, 1, 1, 0)
        ])

        self.y_head = nn.ModuleList([            
            DecoderBlock(256 + 64, 128, 3, 2, 1, 1),  # Adjusted input channels for concatenation
            DecoderBlock(128, 128, 3, 1, 1, 0),
            DecoderBlock(128 + 64, 64, 3, 2, 1, 1),
            DecoderBlock(64, 64, 3, 1, 1, 0),
            DecoderBlock(64 + 64, 32, 3, 2, 1, 1),
            DecoderBlock(32, 32, 3, 1, 1, 0)
        ])

        self.z_head = nn.ModuleList([            
            DecoderBlock(256 + 64, 128, 3, 2, 1, 1),  # Adjusted input channels for concatenation
            DecoderBlock(128, 128, 3, 1, 1, 0),
            DecoderBlock(128 + 64, 64, 3, 2, 1, 1),
            DecoderBlock(64, 64, 3, 1, 1, 0),
            DecoderBlock(64 + 64, 32, 3, 2, 1, 1),
            DecoderBlock(32, 32, 3, 1, 1, 0)
        ])

        self.skip_upsamples = nn.ModuleList([
            nn.ConvTranspose2d(384, 64, kernel_size=1, stride=1),  # Upsample to 28x28
            nn.ConvTranspose2d(384, 64, kernel_size=2, stride=2),  # Upsample to 56x56
            nn.ConvTranspose2d(384, 64, kernel_size=4, stride=4),  # Upsample to 112x112
        ])

        # self.final_decoder = nn.Sequential(
        #     nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1),  # Output size: 224x224
        #     nn.Tanh()
        # )

        self.output_x = nn.Sequential(
            nn.ConvTranspose2d(32, 256, kernel_size=5, stride=2, padding=2, output_padding=1),  # Output size: 224x224
        )
        self.output_y = nn.Sequential(
            nn.ConvTranspose2d(32, 256, kernel_size=5, stride=2, padding=2, output_padding=1),  # Output size: 224x224
        )
        self.output_z = nn.Sequential(
            nn.ConvTranspose2d(32, 256, kernel_size=5, stride=2, padding=2, output_padding=1),  # Output size: 224x224
        )

    def forward(self, x):
        hidden_states = self.encoder(x)
        
        # Using the last hidden state for bottleneck
        f4 = hidden_states[-1]  # Last hidden state
        f4 = f4.view(f4.size(0), -1)
        x = self.bottleneck(f4)

        # Processing the decoder with skip connections
        d1 = self.decoder[0](x)
        d1 = self.decoder[1](d1)

        # Reshape and process skip connections
        def process_skip_connection(skip, upsample_layer=None):
            skip = skip[:, 1:, :].permute(0, 2, 1).contiguous()
            skip = skip.view(skip.size(0), 384, 14, 14)
            if upsample_layer:
                skip = upsample_layer(skip)
            return skip

        skip1 = process_skip_connection(hidden_states[-2], self.skip_upsamples[0])
        skip2 = process_skip_connection(hidden_states[-3], self.skip_upsamples[1])
        skip3 = process_skip_connection(hidden_states[-4], self.skip_upsamples[2])

        d1_x = torch.cat((d1, skip1), dim=1)
        d1_x = self.x_head[0](d1_x)
        d1_x = self.x_head[1](d1_x)
        
        d2_x = torch.cat((d1_x, skip2), dim=1)
        d2_x = self.x_head[2](d2_x)
        d2_x = self.x_head[3](d2_x)
        
        d3_x = torch.cat((d2_x, skip3), dim=1)
        d3_x = self.x_head[4](d3_x)
        d3_x = self.x_head[5](d3_x)

        d1_y = torch.cat((d1, skip1), dim=1)
        d1_y = self.y_head[0](d1_y)
        d1_y = self.y_head[1](d1_y)
        
        d2_y = torch.cat((d1_y, skip2), dim=1)
        d2_y = self.y_head[2](d2_y)
        d2_y = self.y_head[3](d2_y)
        
        d3_y = torch.cat((d2_y, skip3), dim=1)
        d3_y = self.y_head[4](d3_y)
        d3_y = self.y_head[5](d3_y)

        d1_z = torch.cat((d1, skip1), dim=1)
        d1_z = self.z_head[0](d1_z)
        d1_z = self.z_head[1](d1_z)
        
        d2_z = torch.cat((d1_z, skip2), dim=1)
        d2_z = self.z_head[2](d2_z)
        d2_z = self.z_head[3](d2_z)
        
        d3_z = torch.cat((d2_z, skip3), dim=1)
        d3_z = self.z_head[4](d3_z)
        d3_z = self.z_head[5](d3_z)

        x = self.output_x(d3_x)
        y = self.output_y(d3_y)
        z = self.output_z(d3_z)
        
        return x, y, z
    
class Autoencoder(nn.Module):
    def __init__(self, input_resolution):
        super(Autoencoder, self).__init__()

        self.input_resolution = input_resolution
        assert input_resolution == 224, "Input resolution must be 224"
        
        self.encoder = DinoViTEncoder()
        
        self.before_bottleneck_size = self.input_resolution // 16  # 224 // 16 = 14
        
        self.bottleneck = nn.Linear(197 * 768, 256)
        
        self.decoder = nn.ModuleList([
            nn.Linear(256, self.before_bottleneck_size * self.before_bottleneck_size * 256),
            nn.Unflatten(1, (256, self.before_bottleneck_size, self.before_bottleneck_size)),
            DecoderBlock(256 + 64, 128, 3, 2, 1, 1),  # Adjusted input channels for concatenation
            DecoderBlock(128, 128, 3, 1, 1, 0),
            DecoderBlock(128 + 64, 64, 3, 2, 1, 1),
            DecoderBlock(64, 64, 3, 1, 1, 0),
            DecoderBlock(64 + 64, 32, 3, 2, 1, 1),
            DecoderBlock(32, 32, 3, 1, 1, 0)
        ])

        self.skip_upsamples = nn.ModuleList([
            nn.ConvTranspose2d(768, 64, kernel_size=1, stride=1),  # Upsample to 28x28
            nn.ConvTranspose2d(768, 64, kernel_size=2, stride=2),  # Upsample to 56x56
            nn.ConvTranspose2d(768, 64, kernel_size=4, stride=4),  # Upsample to 112x112
        ])

        self.output = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1),  # Output size: 224x224
            nn.Tanh()
        )
    def forward(self, x):
        hidden_states = self.encoder(x)
        
        # Using the last hidden state for bottleneck
        f4 = hidden_states[-1]  # Last hidden state      
        f4 = f4.view(f4.size(0), -1)
        x = self.bottleneck(f4)

        # Processing the decoder with skip connections
        d1 = self.decoder[0](x)
        d1 = self.decoder[1](d1)

        # Reshape and process skip connections
        def process_skip_connection(skip, upsample_layer=None):
            skip = skip[:, 1:, :].permute(0, 2, 1).contiguous()
            skip = skip.view(skip.size(0), 768, 14, 14)
            if upsample_layer:
                skip = upsample_layer(skip)
            return skip

        skip1 = process_skip_connection(hidden_states[-2], self.skip_upsamples[0])
        skip2 = process_skip_connection(hidden_states[-3], self.skip_upsamples[1])
        skip3 = process_skip_connection(hidden_states[-4], self.skip_upsamples[2])

        d1 = torch.cat((d1, skip1), dim=1)
        d1 = self.decoder[2](d1)
        d1 = self.decoder[3](d1)
        
        d2 = torch.cat((d1, skip2), dim=1)
        d2 = self.decoder[4](d2)
        d2 = self.decoder[5](d2)
        
        d3 = torch.cat((d2, skip3), dim=1)
        d3 = self.decoder[6](d3)
        d3 = self.decoder[7](d3)

        out = self.output(d3)
        
        return out
    

class DPTForSemanticSegmentation(DPTPreTrainedModel):
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

        print(hidden_states.shape)
        return hidden_states
    
class MultiDINO(nn.Module):
    def __init__(self, input_resolution=224, feature_dim=768, num_blocks=5, num_bins=50):
        super(MultiDINO, self).__init__()

        self.input_resolution = input_resolution
        print("input_resolution: ", input_resolution)
        assert input_resolution == 224, "Input resolution must be 224"
        self.feature_dim = 768
        
        backbone_config = Dinov2Config.from_pretrained("facebook/dinov2-base", out_features=["stage1", "stage2", "stage3", "stage4"], reshape_hidden_states=False)
        config = DPTConfig(backbone_config=backbone_config, add_pooling_layer=False)
        self.dpt = DPTForSemanticSegmentation(config)

        #self.encoder = DinoViTEncoder()
        self.num_bins = num_bins
        self.num_blocks = num_blocks

        # Stack of 10 self-attention blocks
        self.attention_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=feature_dim, nhead=8) for _ in range(num_blocks)
        ])
        
        # Final convolution layer for instance mask (1x224x224)
        # self.mask_conv = nn.Sequential(
        #     nn.Conv2d(in_channels=feature_dim, out_channels=1, kernel_size=3, padding=1),
        #     nn.Upsample(size=(input_resolution, input_resolution), mode='bilinear', align_corners=False)
        # )
        
        # Final convolution layer for NOCS map (3x224x224)
        self.nocs_conv = nn.Sequential(
            nn.Conv2d(in_channels=feature_dim, out_channels=3, kernel_size=3, padding=1),
            nn.Upsample(size=(input_resolution, input_resolution), mode='bilinear', align_corners=False)
        )
        
        # Small MLP for predicting a 3x3 rotation matrix
        self.rotation_mlp = nn.Sequential(
            nn.Linear(196 * feature_dim, 128),  # Input is flattened (197 tokens, each with 768 features)
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 9),  # Output a 9-element vector that can be reshaped into a 3x3 matrix
            nn.Tanh()  # Keep the values bounded between -1 and 1
        )

        self.mask_conv = nn.ConvTranspose2d(
            in_channels=feature_dim,
            out_channels=1,
            kernel_size=16,
            stride=16,
            padding=0
        )

    def forward(self, x):
        outputs = self.dpt(
            x,
            head_mask=None,
            output_attentions=False,
            output_hidden_states=True,  # we need the intermediate hidden states
            return_dict=False,
        )
        print(outputs.shape)

        # Using the last hidden state for the segmentation head
        f4 = outputs[-1]  # Last hidden state: [batch_size, 197, 768]


        # Flatten f4 for the MLP: [batch_size, 197*768]
        f4_flattened = f4.view(f4.size(0), -1)

        x = f4
        for block in self.attention_blocks:
            x = block(x)
       
        # Exclude the [CLS] token, keeping only the patch tokens
        # [batch_size, 197, 768] -> [batch_size, 196, 768]
        x = x[:, 1:, :]  # Exclude the first token ([CLS])

        # Reshape x_patches to match the spatial grid (14x14 patches)
        batch_size = x.size(0)
        x_conv = x.transpose(1, 2).view(batch_size, 768, 14, 14)
        
        # Instance mask prediction [batch_size, 1, 224, 224]
        mask = self.mask_conv(x_conv)
        mask = torch.sigmoid(mask)  # Apply sigmoid to keep values in [0, 1]
        
        # NOCS map prediction [batch_size, 3, 224, 224]
        nocs_map = self.nocs_conv(x_conv)
        nocs_map = torch.tanh(nocs_map)  # Apply tanh to keep values in [-1, 1]
        
        # Flatten the output of the attention blocks [batch_size, 197 * 768]
        flattened_features = x.view(batch_size, -1)
        
        # Use the flattened features to predict the rotation matrix
        rotation_matrix = self.rotation_mlp(flattened_features).view(batch_size, 3, 3)
        
        return nocs_map, mask, rotation_matrix
    
class CustomEncoder(nn.Module):
    def __init__(self):
        super(CustomEncoder, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # 112x112
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 28x28
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 14x14
            nn.ReLU(),
            
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),  # 7x7
            nn.ReLU()
        )
    
    def forward(self, x):
        feature_maps = []
        x = self.layers[0](x)  # Conv1
        feature_maps.append(x)
        x = self.layers[1](x)  # MaxPool
        x = self.layers[2](x)  # Conv2
        feature_maps.append(x)
        x = self.layers[3](x)  # Conv3
        feature_maps.append(x)
        x = self.layers[4](x)  # Conv4
        feature_maps.append(x)
        return feature_maps

class CustomAutoencoder(nn.Module):
    def __init__(self, input_resolution):
        super(CustomAutoencoder, self).__init__()

        self.input_resolution = input_resolution
        assert input_resolution == 224, "Input resolution must be 224"
        
        self.encoder = CustomEncoder()
        
        self.before_bottleneck_size = self.input_resolution // 16  # 224 // 16 = 14
        
        self.bottleneck = nn.Linear(384 * 7 * 7, 256)  # Adjusted for new encoder
        
        self.decoder = nn.ModuleList([
            nn.Linear(256, self.before_bottleneck_size * self.before_bottleneck_size * 256),
            nn.Unflatten(1, (256, self.before_bottleneck_size, self.before_bottleneck_size)),
            DecoderBlock(256 + 64, 128, 3, 2, 1, 1),  # Adjusted input channels for concatenation
            DecoderBlock(128, 128, 3, 1, 1, 0),
            DecoderBlock(128 + 64, 64, 3, 2, 1, 1),
            DecoderBlock(64, 64, 3, 1, 1, 0),
            DecoderBlock(64 + 64, 32, 3, 2, 1, 1),
            DecoderBlock(32, 32, 3, 1, 1, 0)
        ])

        self.skip_upsamples = nn.ModuleList([
            nn.ConvTranspose2d(256, 64, kernel_size=1, stride=1),  # Upsample to 28x28
            nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2),  # Upsample to 56x56
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=4),  # Upsample to 112x112
        ])

        self.output = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2, output_padding=1),  # Output size: 224x224
            nn.Tanh()
        )
        
    def forward(self, x):
        hidden_states = self.encoder(x)
        
        # Using the last hidden state for bottleneck
        f4 = hidden_states[-1]  # Last hidden state
        f4 = f4.view(f4.size(0), -1)
        x = self.bottleneck(f4)

        # Processing the decoder with skip connections
        d1 = self.decoder[0](x)
        d1 = self.decoder[1](d1)

        # Reshape and process skip connections
        def process_skip_connection(skip, upsample_layer=None):
            skip = skip.view(skip.size(0), 384, 7, 7)
            if upsample_layer:
                skip = upsample_layer(skip)
            return skip

        skip1 = process_skip_connection(hidden_states[-2], self.skip_upsamples[0])
        skip2 = process_skip_connection(hidden_states[-3], self.skip_upsamples[1])
        skip3 = process_skip_connection(hidden_states[-4], self.skip_upsamples[2])

        d1 = torch.cat((d1, skip1), dim=1)
        d1 = self.decoder[2](d1)
        d1 = self.decoder[3](d1)
        
        d2 = torch.cat((d1, skip2), dim=1)
        d2 = self.decoder[4](d2)
        d2 = self.decoder[5](d2)
        
        d3 = torch.cat((d2, skip3), dim=1)
        d3 = self.decoder[6](d3)
        d3 = self.decoder[7](d3)

        out = self.output(d3)
        
        return out
