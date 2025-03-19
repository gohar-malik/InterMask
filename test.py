import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_downsampling(input_size, target_size=5, num_layers=3):
    layers = []
    current_size = input_size
    
    for i in range(num_layers):
        stride = max(2, int((current_size / target_size) ** (1 / (num_layers - i))))
        filter_size = 2 * stride  # Typical choice to maintain spatial coherence
        layers.append((filter_size, stride))
        current_size = math.ceil(current_size / stride)
 
    return layers

def compute_upsampling(target_size, final_size, num_layers=3):
    factors = []
    current_size = target_size
    
    for i in range(num_layers):
        factor = round((final_size / current_size) ** (1 / (num_layers - i)), 1)
        factors.append(factor)
        current_size = math.floor(current_size*factor)
    
    return factors

def test_with_pytorch(spatial_size):
    down_layers = compute_downsampling(spatial_size)
    up_factors = compute_upsampling(5, spatial_size)
    
    x = torch.randn(1, 1, spatial_size)  # Simulate a 1D spatial input
    
    # Downsampling with Conv1d
    for i, (f, s) in enumerate(down_layers):
        conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=f, stride=s, padding=s//2)
        x = conv(x)
        print(f"After Downsampling Layer {i+1}: {x.shape[-1]}")
    
    # Upsampling with Interpolate
    for i, factor in enumerate(up_factors):
        x = F.interpolate(x, scale_factor=factor, mode='nearest')
        print(f"After Upsampling Layer {i+1}: {x.shape[-1]}")
    
# Test with spatial sizes 22 and 56
spatial_sizes = [22, 56]

def print_results(spatial):
    down_layers = compute_downsampling(spatial)
    up_factors = compute_upsampling(5, spatial)
    
    print(f"Spatial Size: {spatial}")
    print("Downsampling:")
    for i, (f, s) in enumerate(down_layers):
        print(f"  Conv Layer {i+1}: Filter Size = {f}, Stride = {s}")
    
    print("Upsampling:")
    for i, factor in enumerate(up_factors):
        print(f"  Upsample Layer {i+1}: Factor = {factor}")
    print()

test_with_pytorch(22)
test_with_pytorch(56)

for s in spatial_sizes:
    print_results(s)
