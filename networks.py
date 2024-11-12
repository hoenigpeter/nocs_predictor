import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

class UnetGeneratoNOCSHead(nn.Module):
    """U-Net generator with a shared encoder and multiple decoder heads."""
    
    def __init__(self, input_nc, output_nc, num_downs, num_heads=3, ngf=64, num_bins=50, norm_layer=nn.BatchNorm2d, use_dropout=False):
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
        super(UnetGeneratoNOCSHead, self).__init__()
        
        # Construct the encoder with skip connections
        self.num_bins = num_bins

        self.encoder = nn.ModuleList()
        input_nc_curr = input_nc
        for i in range(num_downs):
            output_nc_curr = ngf * min(2 ** i, 8)
            self.encoder.append(UnetEncoderBlock(input_nc_curr, output_nc_curr, norm_layer=norm_layer))
            input_nc_curr = output_nc_curr
            
        # # Construct multiple decoder heads with independent weights
        # self.decoders = nn.ModuleList([
        #     UnetDecoder(num_downs, output_nc, ngf, norm_layer, use_dropout)
        #     for _ in range(num_heads)
        # ])
        self.nocs_head = UnetDecoder(num_downs, 3, ngf, norm_layer, use_dropout)

        # self.rotation_head = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Adjust input channels to 3
        #     nn.ReLU(),  # Activation for non-linearity
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # More channels for complexity
        #     nn.ReLU(),
        #     nn.Conv2d(128, 6, kernel_size=3, stride=1, padding=1),  # Output 6 channels for the 6D rotation
        #     nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to (batch_size, 6, 1, 1)
        #     nn.Flatten()  # Flatten to shape (batch_size, 6)
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
        # Encoder forward pass with skip connections
        encoder_features = []
        for layer in self.encoder:
            x = layer(x)
            encoder_features.append(x)
        encoder_features = encoder_features[::-1]  # reverse for correct skip connection order
        
        # Pass through each decoder head independently
        #outputs = [decoder(encoder_features) for decoder in self.decoders]

        nocs_estimated = torch.tanh(self.nocs_head(encoder_features))

        return nocs_estimated

class UnetGeneratoNOCSBinHead(nn.Module):
    """U-Net generator with a shared encoder and multiple decoder heads."""
    
    def __init__(self, input_nc, output_nc, num_downs, num_heads=3, ngf=64, num_bins=50, norm_layer=nn.BatchNorm2d, use_dropout=False):
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
        super(UnetGeneratoNOCSBinHead, self).__init__()
        
        # Construct the encoder with skip connections
        self.num_bins = num_bins

        self.encoder = nn.ModuleList()
        input_nc_curr = input_nc
        for i in range(num_downs):
            output_nc_curr = ngf * min(2 ** i, 8)
            self.encoder.append(UnetEncoderBlock(input_nc_curr, output_nc_curr, norm_layer=norm_layer))
            input_nc_curr = output_nc_curr
            
        self.x_head = UnetDecoder(num_downs, output_nc, ngf, norm_layer, use_dropout)
        self.y_head = UnetDecoder(num_downs, output_nc, ngf, norm_layer, use_dropout)
        self.z_head = UnetDecoder(num_downs, output_nc, ngf, norm_layer, use_dropout)

   
    def forward(self, x):
        # Encoder forward pass with skip connections
        encoder_features = []
        for layer in self.encoder:
            x = layer(x)
            encoder_features.append(x)
        encoder_features = encoder_features[::-1]  # reverse for correct skip connection order
        
        x_logits = self.x_head(encoder_features)
        y_logits = self.y_head(encoder_features)
        z_logits = self.z_head(encoder_features)

        return x_logits, y_logits, z_logits

class UnetGeneratorMultiHead(nn.Module):
    """U-Net generator with a shared encoder and multiple decoder heads."""
    
    def __init__(self, input_nc, output_nc, num_downs, num_heads=3, ngf=64, num_bins=50, norm_layer=nn.BatchNorm2d, use_dropout=False):
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
        super(UnetGeneratorMultiHead, self).__init__()
        
        # Construct the encoder with skip connections
        self.num_bins = num_bins

        self.encoder = nn.ModuleList()
        input_nc_curr = input_nc
        for i in range(num_downs):
            output_nc_curr = ngf * min(2 ** i, 8)
            self.encoder.append(UnetEncoderBlock(input_nc_curr, output_nc_curr, norm_layer=norm_layer))
            input_nc_curr = output_nc_curr
            
        # # Construct multiple decoder heads with independent weights
        # self.decoders = nn.ModuleList([
        #     UnetDecoder(num_downs, output_nc, ngf, norm_layer, use_dropout)
        #     for _ in range(num_heads)
        # ])
        self.x_head = UnetDecoder(num_downs, output_nc, ngf, norm_layer, use_dropout)
        self.y_head = UnetDecoder(num_downs, output_nc, ngf, norm_layer, use_dropout)
        self.z_head = UnetDecoder(num_downs, output_nc, ngf, norm_layer, use_dropout)
        self.mask_head = UnetDecoder(num_downs, 1, ngf, norm_layer, use_dropout)

        # self.rotation_head = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Adjust input channels to 3
        #     nn.ReLU(),  # Activation for non-linearity
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # More channels for complexity
        #     nn.ReLU(),
        #     nn.Conv2d(128, 6, kernel_size=3, stride=1, padding=1),  # Output 6 channels for the 6D rotation
        #     nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to (batch_size, 6, 1, 1)
        #     nn.Flatten()  # Flatten to shape (batch_size, 6)
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
        # Encoder forward pass with skip connections
        encoder_features = []
        for layer in self.encoder:
            x = layer(x)
            encoder_features.append(x)
        encoder_features = encoder_features[::-1]  # reverse for correct skip connection order
        
        # Pass through each decoder head independently
        #outputs = [decoder(encoder_features) for decoder in self.decoders]

        x_logits = self.x_head(encoder_features)
        y_logits = self.y_head(encoder_features)
        z_logits = self.z_head(encoder_features)

        mask_estimated = torch.sigmoid(self.mask_head(encoder_features))

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

        # rotation_6d = self.rotation_head(nocs_estimated)
        # rotation_matrix = self.rot6d_to_rotmat(rotation_6d)

        return x_logits, y_logits, z_logits, nocs_estimated, mask_estimated

class UnetGeneratorMultiHeadRot(nn.Module):
    """U-Net generator with a shared encoder and multiple decoder heads."""
    
    def __init__(self, input_nc, output_nc, num_downs, num_heads=3, ngf=64, num_bins=50, norm_layer=nn.BatchNorm2d, use_dropout=False):
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
        super(UnetGeneratorMultiHeadRot, self).__init__()
        
        # Construct the encoder with skip connections
        self.num_bins = num_bins

        self.encoder = nn.ModuleList()
        input_nc_curr = input_nc
        for i in range(num_downs):
            output_nc_curr = ngf * min(2 ** i, 8)
            self.encoder.append(UnetEncoderBlock(input_nc_curr, output_nc_curr, norm_layer=norm_layer))
            input_nc_curr = output_nc_curr
            
        # # Construct multiple decoder heads with independent weights
        # self.decoders = nn.ModuleList([
        #     UnetDecoder(num_downs, output_nc, ngf, norm_layer, use_dropout)
        #     for _ in range(num_heads)
        # ])
        self.x_head = UnetDecoder(num_downs, output_nc, ngf, norm_layer, use_dropout)
        self.y_head = UnetDecoder(num_downs, output_nc, ngf, norm_layer, use_dropout)
        self.z_head = UnetDecoder(num_downs, output_nc, ngf, norm_layer, use_dropout)
        self.mask_head = UnetDecoder(num_downs, 1, ngf, norm_layer, use_dropout)

        self.rotation_head = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # Adjust input channels to 3
            nn.ReLU(),  # Activation for non-linearity
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # More channels for complexity
            nn.ReLU(),
            nn.Conv2d(128, 6, kernel_size=3, stride=1, padding=1),  # Output 6 channels for the 6D rotation
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
        # Encoder forward pass with skip connections
        encoder_features = []
        for layer in self.encoder:
            x = layer(x)
            encoder_features.append(x)
        encoder_features = encoder_features[::-1]  # reverse for correct skip connection order
        
        # Pass through each decoder head independently
        #outputs = [decoder(encoder_features) for decoder in self.decoders]

        x_logits = self.x_head(encoder_features)
        y_logits = self.y_head(encoder_features)
        z_logits = self.z_head(encoder_features)

        mask_estimated = torch.sigmoid(self.mask_head(encoder_features))

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

        return x_logits, y_logits, z_logits, nocs_estimated, mask_estimated, rotation_matrix
    
class UnetEncoderBlock(nn.Module):
    """Single encoder block in U-Net, including downsampling."""
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(UnetEncoderBlock, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.block(x)

class UnetDecoder(nn.Module):
    """Decoder head in U-Net with skip connections from encoder."""

    def __init__(self, num_downs, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetDecoder, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(num_downs):
            inner_nc = ngf * min(2 ** (num_downs - i - 1), 8)
            outer_nc = ngf * min(2 ** (num_downs - i - 2), 8) if i < num_downs - 1 else output_nc
            is_outermost = i == num_downs - 1
            first_layer = i == 0  # Set the first layer flag
            self.layers.append(UnetDecoderBlock(inner_nc, outer_nc, is_outermost, norm_layer, use_dropout, first_layer))

    def forward(self, encoder_features):
        # Start with the bottleneck feature
        x = encoder_features[0]
        # Iterate through layers and skip connections
        for idx, (layer, skip_feature) in enumerate(zip(self.layers, encoder_features)):
            x = layer(x, skip_feature, first_layer=(idx == 0))  # Pass first_layer flag for the first layer
        return x

class UnetDecoderBlock(nn.Module):
    """Single decoder block in U-Net, including upsampling and skip connections."""
    
    def __init__(self, inner_nc, outer_nc, outermost, norm_layer, use_dropout, first_layer):
        super(UnetDecoderBlock, self).__init__()
        self.outermost = outermost
        use_bias = norm_layer == nn.InstanceNorm2d
        uprelu = nn.ReLU()
        upnorm = norm_layer(outer_nc)
        
        # Adjust input channels for the first layer as it won’t concatenate
        in_channels = inner_nc if first_layer else inner_nc * 2

        if outermost:
            upconv = nn.ConvTranspose2d(in_channels, outer_nc, kernel_size=4, stride=2, padding=1)
            self.block = nn.Sequential(uprelu, upconv)
        else:
            upconv = nn.ConvTranspose2d(in_channels, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            block = [uprelu, upconv, upnorm]
            if use_dropout:
                block.append(nn.Dropout(0.5))
            self.block = nn.Sequential(*block)

    def forward(self, x, skip_feature, first_layer=False):
        # Apply skip connection only if not the first layer

        if not first_layer:
            x = torch.cat([x, skip_feature], 1)
        return self.block(x)
