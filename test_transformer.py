import torch
import torch.nn as nn

class UpsamplingNetwork(nn.Module):
    def __init__(self, input_channels=512):
        super(UpsamplingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 512, kernel_size=3, padding=1)  # From 768 to 512 channels
        self.relu1 = nn.ReLU()
        
        self.upsample1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)  # Upsample to 64x64
        self.relu2 = nn.ReLU()

        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        self.upsample2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Upsample to 128x128
        self.relu4 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 50, kernel_size=3, padding=1)  # Final output to 50 channels

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
    
# Define a simple Vision Transformer model
class SimpleViT(nn.Module):
    def __init__(self, input_dim, patch_size, num_patches, embedding_dim=512, num_blocks=10):
        super(SimpleViT, self).__init__()
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
        x = x.contiguous().view(batch_size, channels, num_patches_height, num_patches_width, patch_size, patch_size)  # Ensure contiguous
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

        x = x.view(batch_size, 32, 32, 512)

        x = x.permute(0, 3, 1, 2)

        return x

# Example usage
batch_size = 16
input_channels = 256
height = 128
width = 128
patch_size = 4  # Patch size of 16x16
num_patches = (height // patch_size) * (width // patch_size)  # Total patches

# Create a random input tensor
input_tensor = torch.randn(batch_size, input_channels, height, width).to('cuda')  # Move input to GPU

# Create the model and move it to GPU
model = SimpleViT(input_dim=input_channels * patch_size * patch_size,
                   patch_size=patch_size, 
                   num_patches=num_patches).to('cuda')  # Move model to GPU

# Forward pass
output = model(input_tensor)

# Create the model
x_head = UpsamplingNetwork().to('cuda')
y_head = UpsamplingNetwork().to('cuda')
z_head = UpsamplingNetwork().to('cuda')

print("Output shape:", output.shape)

# Forward pass
final_output = x_head(output)
final_output = y_head(output)
final_output = z_head(output)

print("Output shape:", final_output.shape)  # Should be (batch_size, num_classes)
