
import torch
import random
import numpy as np
import cv2
import os

# Import the function from the file (we need to mock or copy since imports might fail if context missing)
# For simplicity, I will copy the function here to test the logic specifically.
# If I import, I might hit issues with 'dataset' or 'model' imports in train_multimodal_v5 if I don't have them in path.
# But let's try to copy the logic to be safe and test the logic ITSELF.

def apply_random_augmentation(img):
    B, C, H, W = img.shape
    
    # 1. Random Gamma
    if random.random() < 0.5: 
        gamma = random.uniform(0.7, 1.5)
        img = img.pow(gamma)
        print(f"Applied Gamma: {gamma:.2f}")
        
    # 2. Random Contrast
    if random.random() < 0.5:
        contrast_factor = random.uniform(0.7, 1.3)
        mean = img.mean(dim=(2, 3), keepdim=True)
        img = (img - mean) * contrast_factor + mean
        print(f"Applied Contrast: {contrast_factor:.2f}")
        
    # 3. Random Brightness
    if random.random() < 0.5:
        brightness_offset = random.uniform(-0.1, 0.1)
        img = img + brightness_offset
        print(f"Applied Brightness: {brightness_offset:.2f}")
        
    img = torch.clamp(img, 0.0, 1.0)
    return img

def test():
    # Create a dummy image (gradient)
    img = torch.linspace(0, 1, 100).view(1, 1, 1, 100).expand(1, 1, 100, 100)
    
    print("Original stats:", img.mean().item(), img.min().item(), img.max().item())
    
    # Run a few times
    for i in range(5):
        print(f"\n--- Run {i+1} ---")
        aug = apply_random_augmentation(img.clone())
        print("Augmented stats:", aug.mean().item(), aug.min().item(), aug.max().item())
        
        # Visualize diff
        diff = (aug - img).abs().mean()
        print(f"Diff: {diff.item()}")
        
        if diff > 0:
            print("Augmentation applied successfully.")
        else:
            print("No augmentation applied (random chance).")

if __name__ == "__main__":
    test()
