import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def convert_yolo_to_coco(img_dir, label_dir, output_json, class_names):
    images = []
    annotations = []
    categories = []
    ann_id = 1
    image_id = 1

    # Create category entries
    for i, class_name in enumerate(class_names):
        categories.append({
            'id': i,
            'name': class_name,
            'supercategory': 'none'
        })

    img_paths = list(Path(img_dir).glob("*.jpg")) + list(Path(img_dir).glob("*.png"))

    for img_path in tqdm(img_paths, desc="Converting"):
        img = Image.open(img_path)
        width, height = img.size

        images.append({
            'file_name': img_path.name,
            'height': height,
            'width': width,
            'id': image_id
        })

        # Read corresponding YOLO label file
        label_path = Path(label_dir) / img_path.with_suffix('.txt').name
        if not label_path.exists():
            image_id += 1
            continue  # skip images with no labels

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, w, h = map(float, parts)
                x_min = (x_center - w / 2) * width
                y_min = (y_center - h / 2) * height
                bbox_width = w * width
                bbox_height = h * height

                annotations.append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': int(class_id),
                    'bbox': [x_min, y_min, bbox_width, bbox_height],
                    'area': bbox_width * bbox_height,
                    'iscrowd': 0,
                    'segmentation': []
                })
                ann_id += 1
        image_id += 1

    coco_dict = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(output_json, 'w') as f:
        json.dump(coco_dict, f, indent=4)

    print(f"✅ COCO JSON saved to {output_json} with {len(images)} images and {len(annotations)} annotations.")


# ===============================
# 💡 Example usage
# ===============================
if __name__ == "__main__":
    # Set your own paths below:
    img_dir = "/Data3/Sona/datasets/UIA_dataset/benchmark_train/images"        # path to validation images
    label_dir = "/Data3/Sona/datasets/UIA_dataset/benchmark_train/labels"      # path to YOLO .txt files
    output_json = "/Data3/Sona/datasets/UIA_dataset/annotations/instances_train.json"

    # List your class names in correct order
    class_names = ["DJMINI", "CODRONE", "TELLO"]

    convert_yolo_to_coco(img_dir, label_dir, output_json, class_names)
