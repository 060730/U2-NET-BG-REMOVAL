import os
import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


from model.u2net import U2NET  # Ensure this file exists

# Set paths
input_folder = 'test_data/test_images'
output_folder = 'test_data/output_results'
model_path = 'saved_models/u2net/u2net.pth'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Load U2NET model
print("Loading U²-Net model...")
model = U2NET(3, 1)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor()
])

# Loop through all images
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]

        print(f"\nProcessing: {filename}")
        original = Image.open(image_path).convert('RGB')
        original_np = np.array(original)
        w, h = original.size

        # Preprocess for U2NET
        input_tensor = transform(original).unsqueeze(0)

        with torch.no_grad():
            d1, _, _, _, _, _, _ = model(input_tensor)
            pred = d1[:, 0, :, :]
            pred = (pred - pred.min()) / (pred.max() - pred.min())
            pred_np = pred.squeeze().cpu().numpy()

        # Resize and normalize mask
        mask = cv2.resize(pred_np, (w, h))
        mask = (mask * 255).astype(np.uint8)
        mask_path = os.path.join(output_folder, f'{base_name}_mask.png')
        cv2.imwrite(mask_path, mask)

        # Create 3-channel mask
        mask_3c = cv2.merge([mask, mask, mask])

        # Extract foreground using mask
        foreground_np = cv2.bitwise_and(original_np, mask_3c)
        foreground_img = Image.fromarray(foreground_np)
        foreground_path = os.path.join(output_folder, f'{base_name}_foreground.png')
        foreground_img.save(foreground_path)
        print(f"Saved foreground image: {foreground_path}")

        # OCR on masked foreground
        print("Extracting text from foreground...")
        ocr_text = pytesseract.image_to_string(foreground_img, lang='hin',config='--psm 6')


        text_path = os.path.join(output_folder, f'{base_name}_text.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(ocr_text)
        print(f"Saved OCR text: {text_path}")
