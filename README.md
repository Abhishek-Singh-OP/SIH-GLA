# SIH-GLA
Smart India Hackathon GLA University
import cv2
import numpy as np

def load_annotations(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    annotations = []
    for line in lines:
        label, x_center, y_center, width, height = map(float, line.strip().split())
        annotations.append((label, x_center, y_center, width, height))
    return annotations

def convert_yolo_to_pixel_coords(image_shape, bbox):
    img_h, img_w = image_shape[:2]
    label, x_center, y_center, width, height = bbox
    x_center *= img_w
    y_center *= img_h
    width *= img_w
    height *= img_h
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    return label, x_min, y_min, x_max, y_max

def crop_labels_from_image(image_path, annotations, output_dir):
    image = cv2.imread(image_path)
    image_shape = image.shape

    for idx, bbox in enumerate(annotations):
        label, x_min, y_min, x_max, y_max = convert_yolo_to_pixel_coords(image_shape, bbox)
        cropped_image = image[y_min:y_max, x_min:x_max]
        output_path = f"{output_dir}/label_{int(label)}_{idx}.png"
        cv2.imwrite(output_path, cropped_image)

# Paths
image_path = "path_to_your_image.jpg"
annotations_path = "path_to_your_annotations.txt"
output_dir = "path_to_save_cropped_labels"

# Load annotations
annotations = load_annotations(annotations_path)

# Crop labels from image and save
crop_labels_from_image(image_path, annotations, output_dir)
