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