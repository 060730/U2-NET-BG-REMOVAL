import os
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from datetime import datetime
import numpy as np
from model.u2net import U2NET  # Ensure you have the correct model import

# Function to normalize and resize image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image, image_tensor

# Function to apply mask to remove background
def apply_mask(original, mask):
    original = original.resize(mask.size)
    original_np = np.array(original)
    mask_np = np.array(mask).astype(np.float32) / 255
    if mask_np.ndim == 2:
        mask_np = np.expand_dims(mask_np, axis=2)
    masked = (original_np * mask_np).astype(np.uint8)
    return Image.fromarray(masked)

# Paths
model_path = './saved_models/u2net/u2net.pth'
test_dir = './test_data/test_images'
output_dir = './test_data/output_images'
os.makedirs(output_dir, exist_ok=True)

# Track start time
start_time = datetime.now()
print("Start Time:", start_time.strftime("%H:%M:%S"))

# Load model
print("Loading model...")
net = U2NET(3, 1)
net.load_state_dict(torch.load(model_path, map_location='cpu'))
net.eval()

# Process images
print("Starting inference...")
for file_name in os.listdir(test_dir):
    if not file_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(test_dir, file_name)
    original_image, image_tensor = preprocess_image(image_path)

    image_tensor = Variable(image_tensor)
    d1, _, _, _, _, _, _ = net(image_tensor)
    pred = d1[:, 0, :, :]
    pred = pred.squeeze().cpu().data.numpy()

    # Normalize and create mask
    mask = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    mask_image = Image.fromarray((mask * 255).astype(np.uint8)).resize(original_image.size)

    # Apply mask to remove background
    result = apply_mask(original_image, mask_image)

    output_path = os.path.join(output_dir, file_name)
    result.save(output_path)
    print(f"Saved: {output_path}")

# Track end time
end_time = datetime.now()
print("End Time:", end_time.strftime("%H:%M:%S"))
print("Total Processing Time:", str(end_time - start_time))