# U²-Net Background Remover

This project uses the powerful U²-Net deep learning model to remove backgrounds from images. It supports transparent output and background replacement with a custom image (e.g., studio background). This tool is ideal for bulk image processing in educational, design, or e-commerce scenarios.

---

## Features
-  Background removal using U²-Net
-  Optional background replacement (e.g., white studio background)
-  Batch processing of multiple images
-  Outputs: Transparent PNG or background-composited JPEG
-  Processing time report available (CSV format)

---

##  Sample Results

| Input Image | Transparent Output | With Background |
|-------------|--------------------|------------------|
| [][test_images/sample1.jpg] | [][output_images/sample1_transparent.png] | [][output_images/sample1_with_bg.jpg] |

---

##  Folder Structure

```
background-removal-u2net/
├── test_images/             # Input images
├── output_images/           # Final output (replaced backgrounds only)
├── bg_images/               # Background to overlay
├── saved_models/            # Pretrained U²-Net weights
├── model/                   # U²-Net model definition
├── u2net_replace_bg.py      # Main script to process images
├── requirements.txt         # All Python dependencies
├── README.md                # Project documentation

```
---

##  Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

---

## How to Run

```bash
python u2net_replace_bg.py
```

Make sure your background image is placed inside the `bg_images/` folder.

---
## Model Info

- Paper: [U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection]
---
