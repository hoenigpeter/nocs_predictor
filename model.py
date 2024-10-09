import torch
import torch.nn as nn
import numpy as np
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        #self.attention = AttentionModule(out_channels)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding):
        super(DecoderBlock, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=2, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        return self.relu(self.bn(self.deconv(x)))

class Autoencoder(nn.Module):
    def __init__(self, input_resolution):
        super(Autoencoder, self).__init__()

        self.input_resolution = input_resolution
        assert input_resolution in [128, 256, 512], "Input resolution must be one of 128, 256, 512"
        
        self.encoder = nn.ModuleList([
            EncoderBlock(3, 64),
            EncoderBlock(3, 64),            
            EncoderBlock(2*64, 128),
            EncoderBlock(2*64, 128),
            EncoderBlock(2*128, 128),
            EncoderBlock(2*128, 128),
            EncoderBlock(2*128, 256),
            EncoderBlock(2*128, 256)
        ])

        # Dynamically calculate the size before the bottleneck
        self.before_bottleneck_size = self.input_resolution // 16  # Assuming 4 pooling layers
        
        self.bottleneck = nn.Linear(self.before_bottleneck_size * self.before_bottleneck_size * 512, 256)
        
        self.decoder = nn.ModuleList([
            nn.Linear(256, self.before_bottleneck_size * self.before_bottleneck_size * 256),
            nn.Unflatten(1, (256, self.before_bottleneck_size, self.before_bottleneck_size)),
            DecoderBlock(256, 128, 5, 2, 1),   # fuer d1
            DecoderBlock(256, 256, 5, 1, 0),   # fuer d1_uni
            DecoderBlock(256, 128, 5, 2, 1),   # fuer d2
            DecoderBlock(256, 128, 5, 1, 0),   # fuer d2_uni           
            DecoderBlock(128, 64, 5, 2, 1),   # fuer d3
            DecoderBlock(128, 128, 5, 1, 0),   # fuer d3_uni   
        ])

        self.final_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        f1_1 = self.encoder[0](x)
        f1_2 = self.encoder[1](x)
        f1 = torch.cat([f1_1, f1_2], dim=1)
        
        f2_1 = self.encoder[2](f1)
        f2_2 = self.encoder[3](f1)
        f2 = torch.cat([f2_1, f2_2], dim=1)
        
        f3_1 = self.encoder[4](f2)
        f3_2 = self.encoder[5](f2)
        f3 = torch.cat([f3_1, f3_2], dim=1)
        
        f4_1 = self.encoder[6](f3)
        f4_2 = self.encoder[7](f3)
        f4 = torch.cat([f4_1, f4_2], dim=1)
        f4 = f4.contiguous()

        x = f4.view(f4.size(0), -1)
        x = self.bottleneck(x)

        d1 = self.decoder[0](x)
        d1 = self.decoder[1](d1)    

        d1 = self.decoder[2](d1)        
        d1_uni = torch.cat([d1, f3_2], dim=1)
        d1_uni = self.decoder[3](d1_uni)        

        d2 = self.decoder[4](d1_uni)      
        d2_uni = torch.cat([d2, f2_2], dim=1)
        d2_uni = self.decoder[5](d2_uni)
        
        d3 = self.decoder[6](d2_uni)     
        d3_uni = torch.cat([d3, f1_2], dim=1)    
        d3_uni = self.decoder[7](d3_uni)      

        decoded = self.final_decoder(d3_uni)

        return decoded
    
class AutoencoderXYZHead(nn.Module):
    def __init__(self, input_resolution):
        super(AutoencoderXYZHead, self).__init__()

        self.input_resolution = input_resolution
        assert input_resolution in [128, 256, 512], "Input resolution must be one of 128, 256, 512"
        
        self.encoder = nn.ModuleList([
            EncoderBlock(3, 64),
            EncoderBlock(3, 64),            
            EncoderBlock(2*64, 128),
            EncoderBlock(2*64, 128),
            EncoderBlock(2*128, 128),
            EncoderBlock(2*128, 128),
            EncoderBlock(2*128, 256),
            EncoderBlock(2*128, 256)
        ])

        # Dynamically calculate the size before the bottleneck
        self.before_bottleneck_size = self.input_resolution // 16  # Assuming 4 pooling layers
        
        self.bottleneck = nn.Linear(self.before_bottleneck_size * self.before_bottleneck_size * 512, 256)
        
        self.decoder = nn.ModuleList([
            nn.Linear(256, self.before_bottleneck_size * self.before_bottleneck_size * 256),
            nn.Unflatten(1, (256, self.before_bottleneck_size, self.before_bottleneck_size)),
        ])

        self.x_head = nn.ModuleList([
            DecoderBlock(256, 128, 5, 2, 1),   # fuer d1
            DecoderBlock(256, 256, 5, 1, 0),   # fuer d1_uni
            DecoderBlock(256, 128, 5, 2, 1),   # fuer d2
            DecoderBlock(256, 128, 5, 1, 0),   # fuer d2_uni           
            DecoderBlock(128, 64, 5, 2, 1),   # fuer d3
            DecoderBlock(128, 128, 5, 1, 0),   # fuer d3_uni  
            nn.ConvTranspose2d(128, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh() 
        ])

        self.y_head = nn.ModuleList([
            DecoderBlock(256, 128, 5, 2, 1),   # fuer d1
            DecoderBlock(256, 256, 5, 1, 0),   # fuer d1_uni
            DecoderBlock(256, 128, 5, 2, 1),   # fuer d2
            DecoderBlock(256, 128, 5, 1, 0),   # fuer d2_uni           
            DecoderBlock(128, 64, 5, 2, 1),   # fuer d3
            DecoderBlock(128, 128, 5, 1, 0),   # fuer d3_uni   
            nn.ConvTranspose2d(128, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        ])

        self.z_head = nn.ModuleList([
            DecoderBlock(256, 128, 5, 2, 1),   # fuer d1
            DecoderBlock(256, 256, 5, 1, 0),   # fuer d1_uni
            DecoderBlock(256, 128, 5, 2, 1),   # fuer d2
            DecoderBlock(256, 128, 5, 1, 0),   # fuer d2_uni           
            DecoderBlock(128, 64, 5, 2, 1),   # fuer d3
            DecoderBlock(128, 128, 5, 1, 0),   # fuer d3_uni   
            nn.ConvTranspose2d(128, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()            
        ])

        self.final_decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 256, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        f1_1 = self.encoder[0](x)
        f1_2 = self.encoder[1](x)
        f1 = torch.cat([f1_1, f1_2], dim=1)
        
        f2_1 = self.encoder[2](f1)
        f2_2 = self.encoder[3](f1)
        f2 = torch.cat([f2_1, f2_2], dim=1)
        
        f3_1 = self.encoder[4](f2)
        f3_2 = self.encoder[5](f2)
        f3 = torch.cat([f3_1, f3_2], dim=1)
        
        f4_1 = self.encoder[6](f3)
        f4_2 = self.encoder[7](f3)
        f4 = torch.cat([f4_1, f4_2], dim=1)
        f4 = f4.contiguous()

        x = f4.view(f4.size(0), -1)
        x = self.bottleneck(x)

        d1 = self.decoder[0](x)
        d1 = self.decoder[1](d1)    

        # XYZ Heads
        # x head
        d1_x = self.x_head[0](d1)        
        d1_x = torch.cat([d1_x, f3_2], dim=1)
        d1_x = self.x_head[1](d1_x)        

        d2_x = self.x_head[2](d1_x)      
        d2_x = torch.cat([d2_x, f2_2], dim=1)
        d2_x = self.x_head[3](d2_x)
        
        d3_x = self.x_head[4](d2_x)     
        d3_x = torch.cat([d3_x, f1_2], dim=1)    
        d3_x = self.x_head[5](d3_x)

        x_out = self.x_head[6](d3_x)
        x_out = self.x_head[7](x_out)

        # y head
        d1_y = self.y_head[0](d1)        
        d1_y = torch.cat([d1_y, f3_2], dim=1)
        d1_y = self.y_head[1](d1_y)        

        d2_y = self.y_head[2](d1_y)      
        d2_y = torch.cat([d2_y, f2_2], dim=1)
        d2_y = self.y_head[3](d2_y)
        
        d3_y = self.y_head[4](d2_x)     
        d3_y = torch.cat([d3_y, f1_2], dim=1)    
        d3_y = self.y_head[5](d3_y)

        y_out = self.y_head[6](d3_y)
        y_out = self.y_head[7](y_out)

        # z head
        d1_z = self.z_head[0](d1)        
        d1_z = torch.cat([d1_z, f3_2], dim=1)
        d1_z = self.z_head[1](d1_z)        

        d2_z = self.z_head[2](d1_z)      
        d2_z = torch.cat([d2_z, f2_2], dim=1)
        d2_z = self.z_head[3](d2_z)
        
        d3_z = self.z_head[4](d2_z)     
        d3_z = torch.cat([d3_z, f1_2], dim=1)    
        d3_z = self.z_head[5](d3_z)      

        z_out = self.z_head[6](d3_z)
        z_out = self.z_head[7](z_out)

        return x_out, y_out, z_out