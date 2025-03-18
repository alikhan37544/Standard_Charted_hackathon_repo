import os
import cv2
import numpy as np
import random
from config import input_dir_aadhar, input_dir_control, output_dir_aadhar, output_dir_control

def load_file(input_dir):
    """Loads images from the given directory and returns a dictionary of filename:image."""
    images = {}
    for filename in os.listdir(input_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:
                images[filename] = image
            else:
                print(f"Could not read {filename}, skipping...")
    return images

def geometric_aug(image):
    """Applies multiple geometric transformations."""
    augmentations = []
    h, w = image.shape[:2]

    # Rotate
    for angle in [-60, -45, -30, -15, 15, 30, 45, 60]:
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        augmentations.append(cv2.warpAffine(image, M, (w, h)))

    # Flip & Translate
    augmentations.extend([cv2.flip(image, 1), cv2.flip(image, 0)])
    for tx, ty in [(-30, -30), (30, 30), (-40, 20), (20, -40)]:
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        augmentations.append(cv2.warpAffine(image, M, (w, h)))

    return augmentations

def combined_augmentations(image):
    """Generates multiple augmentations to reach 200 images per input."""
    augmentations = []
    augmentations.extend(geometric_aug(image))
    for geom in geometric_aug(image):
        augmentations.append(cv2.convertScaleAbs(geom, alpha=1.2, beta=30))  # Brightness
    return augmentations[:200]  # Limit to 200 images

def save_augmented_images(images, output_dir):
    """Saves augmented images."""
    os.makedirs(output_dir, exist_ok=True)
    for filename, image_list in images.items():
        for i, img in enumerate(image_list):
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_aug_{i}.jpg")
            cv2.imwrite(output_path, img)

def run_augmentation():
    """Runs the entire augmentation process."""
    print("Loading images...")
    aadhar_images = load_file(input_dir_aadhar)
    control_images = load_file(input_dir_control)

    print("Generating augmentations...")
    aadhar_augmented = {filename: combined_augmentations(img) for filename, img in aadhar_images.items()}
    control_augmented = {filename: combined_augmentations(img) for filename, img in control_images.items()}

    print("Saving augmented images...")
    save_augmented_images(aadhar_augmented, output_dir_aadhar)
    save_augmented_images(control_augmented, output_dir_control)
    print("Augmentation completed!")

if __name__ == "__main__":
    run_augmentation()
