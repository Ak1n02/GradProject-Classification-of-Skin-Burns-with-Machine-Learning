import cv2
import numpy as np
import os
import torch


def fuzzy_c_means_gpu_torch(data, n_clusters, m=2, error=0.005, maxiter=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = torch.tensor(data, dtype=torch.float32, device=device).unsqueeze(1)  # [n_samples, 1]

    n_samples = data.size(0)

    # Initialize random cluster centers
    centers = torch.rand(n_clusters, 1, device=device)  # [n_clusters, 1]

    # Initialize fuzzy membership matrix
    u = torch.rand(n_samples, n_clusters, device=device)
    u = u / u.sum(dim=1, keepdim=True)  # Normalize

    for iteration in range(maxiter):
        # Update cluster centers
        data = data.view(data.shape[0], -1)
        numerator = (u ** m).T @ data  # [n_clusters, 1]
        denominator = torch.sum(u ** m, dim=0).unsqueeze(1)  # [n_clusters, 1]
        centers = numerator / denominator  # [n_clusters, 1]

        # Calculate distances
        distances = torch.abs(data - centers.T)  # [n_samples, n_clusters]
        distances = torch.clamp(distances, min=1e-10)  # Avoid division by zero

        # Update membership matrix
        inv_distances = distances ** (-2 / (m - 1))  # [n_samples, n_clusters]
        u_new = inv_distances / torch.sum(inv_distances, dim=1, keepdim=True)  # [n_samples, n_clusters]

        # Check for convergence
        if torch.norm(u - u_new) < error:
            break

        u = u_new

    return centers.cpu().numpy().squeeze(), u.cpu().numpy()  # Move back to CPU for further processing


def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert image to YCrCb color space
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Extract Y (luminance) and Cr (chrominance) components
    Y, Cr, _ = cv2.split(image_ycrcb)

    # Normalize Cr and Y components
    Cr_normalized = Cr / 255.0
    Y_normalized = Y / 255.0

    # Identify bright background regions
    background_threshold = 0.8  # Adjust this threshold for your dataset
    background_mask = Y_normalized > background_threshold

    # Set background pixels to NaN
    Cr_normalized_no_background = Cr_normalized.copy()
    Cr_normalized_no_background[background_mask] = np.nan

    # **Step 1: Extract only valid pixels (non-NaN) for clustering**
    valid_pixels = Cr_normalized_no_background[~np.isnan(Cr_normalized_no_background)].reshape(-1, 1)
    valid_pixels_torch = torch.tensor(valid_pixels, dtype=torch.float32, device='cpu')

    if len(valid_pixels) == 0:
        print(f"Skipping {image_path}: No valid pixels for clustering.")
        return image, None, None  # No processing possible

    # **Step 2: Run Fuzzy C-Means on non-NaN pixels**
    n_clusters = 2
    centers, membership = fuzzy_c_means_gpu_torch(valid_pixels_torch, n_clusters)

    # **Step 3: Reconstruct the full image's segmented labels**
    segmented_image = np.full(Cr_normalized.shape, np.nan)  # Initialize with NaNs
    valid_indices = np.where(~np.isnan(Cr_normalized_no_background))  # Get non-NaN indices
    segmented_image[valid_indices] = np.argmax(membership, axis=1)  # Assign cluster labels

    # **Step 4: Assign fixed labels to clusters**
    cluster_means = [np.mean(Cr[segmented_image == i]) for i in range(n_clusters)]
    sorted_clusters = np.argsort(cluster_means)

    background_label = sorted_clusters[0]  # Darkest region (Background)
    healthy_skin_label = sorted_clusters[1]  # Mid-range (Healthy Skin)

    # **Step 5: Map clusters to fixed labels**
    fixed_labels = np.full(segmented_image.shape, np.nan)  # Keep NaNs for background
    fixed_labels[segmented_image == background_label] = 0  # Background
    fixed_labels[segmented_image == healthy_skin_label] = 1  # Healthy Skin

    # **Detect black/burn regions using HSV**
    # Convert image to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split into H, S, V channels
    H, S, V = cv2.split(image_hsv)

    # Normalize S and V components
    S_normalized = S / 255.0
    V_normalized = V / 255.0

    # Identify dark burn regions by low V and low S
    dark_mask = (V_normalized < 0.4) & (S_normalized < 0.5)  # Adjust thresholds if necessary

    # **Step 7: Combine the burn regions**
    final_burn_mask = np.zeros_like(fixed_labels)  # Start with a mask of zeros

    # Mark pixels from the Fuzzy C-Means segmentation as burn
    final_burn_mask[fixed_labels == 1] = 1  # Fuzzy C-Means burn areas

    # Mark pixels from the HSV-based dark burn region
    final_burn_mask[dark_mask] = 1  # Dark burn areas from HSV

    # **Create a white background mask**
    white_mask = np.where(final_burn_mask == 0, 255, 0).astype(np.uint8)
    white_background = np.full_like(image, 255)

    # **Final image processing: Keep burn areas intact, white out everything else**
    result_image = image.copy()
    result_image[white_mask == 255] = white_background[white_mask == 255]

    return result_image


def process_dataset(input_directory, output_directory):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(input_directory):
        for filename in files:
            # Check if file is an image
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                # Construct full input and output paths
                input_path = os.path.join(root, filename)

                # Create corresponding output subdirectory structure
                relative_path = os.path.relpath(root, input_directory)
                output_subdir = os.path.join(output_directory, relative_path)
                os.makedirs(output_subdir, exist_ok=True)

                # Construct output path
                output_path = os.path.join(output_subdir, f"processed_{filename}")

                try:
                    # Process the image
                    result_image = process_image(input_path)

                    # Save the processed image
                    cv2.imwrite(output_path, result_image)

                    print(f"Processed: {input_path} -> {output_path}")

                except Exception as e:
                    print(f"Error processing {input_path}: {e}")


def main():
    # Define input and output directories
    input_directory = 'test_akin_removed'  # Root directory containing images and subdirectories
    output_directory = 'test_akin_removed_onlyburn'  # Output root directory

    # Process the entire dataset
    process_dataset(input_directory, output_directory)


if __name__ == '__main__':
    main()