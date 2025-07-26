import os
import torch
import cv2
import numpy as np
from torchvision.transforms import ToTensor
from model import U2NET

def normalize_pred(d):
    return (d - d.min()) / (d.max() - d.min())

def save_mask(mask, output_dir, img_name):
    mask = (mask * 255).astype(np.uint8)
    output_path = os.path.join(output_dir, f"mask_{img_name}")
    cv2.imwrite(output_path, mask)
    print(f"Saved: {output_path}")

def main():
    # Paths
    model_path = os.path.join("saved_models", "u2net_portrait", "u2net_portrait.pth")
    input_dir = os.path.join("test_data", "test_portrait_images")
    output_dir = os.path.join("test_data", "u2net_portrait_results")
    os.makedirs(output_dir, exist_ok=True)

    # Debug: Check input files
    print(f"\nInput files: {os.listdir(input_dir)}\n")

    # Load model
    net = U2NET(3, 1)
    net.load_state_dict(torch.load(model_path, map_location='cpu'))
    net.eval()

    # Process images
    for img_name in os.listdir(input_dir):
        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(input_dir, img_name)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"Error: Could not read {img_name}")
                continue

            img = cv2.resize(img, (512, 512))
            img_tensor = ToTensor()(img).unsqueeze(0)

            try:
                with torch.no_grad():
                    d1 = net(img_tensor)[0]
                pred = normalize_pred(d1)
                save_mask(pred, output_dir, img_name)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")

if __name__ == "__main__":
    main()