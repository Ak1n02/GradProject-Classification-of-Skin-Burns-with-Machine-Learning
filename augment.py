import cv2
import numpy as np
import random
import os


def adjust_brightness(image, factor):
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def adjust_contrast(image, factor):
    mean = np.mean(image)
    return cv2.convertScaleAbs(image, alpha=factor, beta=(1 - factor) * mean)


def add_gaussian_blur(image, ksize):
    return cv2.GaussianBlur(image, (ksize, ksize), 0)


def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy_image = image.copy()
    total_pixels = image.size
    num_salt = int(total_pixels * salt_prob)
    num_pepper = int(total_pixels * pepper_prob)

    # Add salt (white pixels)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 255

    # Add pepper (black pixels)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    noisy_image[coords[0], coords[1]] = 0

    return noisy_image


def augment_image(image):
    augmented_images = [
        image,  # Original
        cv2.flip(image, 1),  # Flip horizontally
        cv2.flip(image, 0),  # Flip vertically
        cv2.flip(image, -1),  # Flip both axes
        adjust_brightness(image, 1.2),  # Increase brightness
        adjust_brightness(image, 0.8),  # Decrease brightness
        adjust_contrast(image, 1.5),  # Increase contrast
        adjust_contrast(image, 0.5),  # Decrease contrast
        add_gaussian_blur(image, 3),  # Gaussian blur
        add_gaussian_blur(image, 5),  # Stronger Gaussian blur
        add_salt_and_pepper_noise(image, 0.02, 0.02),  # Salt & pepper noise
        add_salt_and_pepper_noise(image, 0.05, 0.05),  # More noise
        cv2.GaussianBlur(cv2.flip(image, 1), (3, 3), 0),  # Flip + Blur
        adjust_brightness(cv2.flip(image, 0), 1.2),  # Flip + Brightness Increase
        adjust_contrast(cv2.flip(image, -1), 1.5),  # Flip + Contrast Increase
        add_salt_and_pepper_noise(cv2.flip(image, 1), 0.02, 0.02)  # Flip + Noise
    ]

    return augmented_images

def augment_dataset(dataset_dir, save_dir):
    dataset_dir = os.path.abspath(dataset_dir)
    save_dir = os.path.abspath(save_dir)

    print(f"Dataset directory: {dataset_dir}")  # Debugging
    print(f"Save directory: {save_dir}")  # Debugging

    # Create save directories for each burn degree if they don't exist
    for burn_degree in ['first_degree', 'second_degree', 'third_degree']:
        os.makedirs(os.path.normpath(os.path.join(save_dir, burn_degree)), exist_ok=True)

    # Loop through the dataset directory
    for burn_degree in ['first_degree', 'second_degree', 'third_degree']:
        burn_degree_path = os.path.normpath(os.path.join(dataset_dir, burn_degree))

        if not os.path.exists(burn_degree_path):  # Check if the directory exists
            print(f"Error: Directory '{burn_degree_path}' does not exist. Skipping.")
            continue

        print(f"Processing images in: {burn_degree_path}")  # Debugging

        # Loop through images in each burn degree folder
        for image_name in os.listdir(burn_degree_path):
            image_path = os.path.normpath(os.path.join(burn_degree_path, image_name))

            # Check if the file is an image
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Warning: Unable to read {image_path}. Skipping.")
                    continue

                augmented_images = augment_image(image)

                # Save augmented images
                for idx, img in enumerate(augmented_images):
                    save_path = os.path.normpath(os.path.join(save_dir, burn_degree, f"{os.path.splitext(image_name)[0]}_aug_{idx}.jpg"))
                    cv2.imwrite(save_path, img)


# Example usage
dataset_dir = 'final_data_set_no_bg'  # Path to your dataset
save_dir = 'final_data_set_no_bg_augmented'  # Path to save the augmented images
augment_dataset(dataset_dir, save_dir)
