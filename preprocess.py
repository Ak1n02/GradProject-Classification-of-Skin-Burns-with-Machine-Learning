import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import torch

# Fuzzy C-Means implementation using PyTorch
def fuzzy_c_means_gpu_torch(data, n_clusters, m=2, error=0.005, maxiter=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(torch.__version__)
    print(torch.version.cuda)  # Should print a version number if CUDA is supported
    print(torch.cuda.is_available())  # Should return True if CUDA is available

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

# Path to the Dataset directory
dataset_directory = "Dataset_bgrem/"

# Function to process each image
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
    #burn_1st_label = sorted_clusters[2]  # Brightest (1st-degree burn)

    # **Step 5: Map clusters to fixed labels**
    fixed_labels = np.full(segmented_image.shape, np.nan)  # Keep NaNs for background
    fixed_labels[segmented_image == background_label] = 0  # Background
    fixed_labels[segmented_image == healthy_skin_label] = 1  # Healthy Skin
    #fixed_labels[segmented_image == burn_1st_label] = 2  # 1st Degree Burn

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

    # **Step 8: Create masks for burn and healthy skin**
    burn_mask = np.where(final_burn_mask == 1, 255, 0).astype(np.uint8)  # Final burn mask
    healthy_skin_mask = np.where(final_burn_mask == 0, 255, 0).astype(np.uint8)  # Healthy skin area

    # **Apply the masks to the original image**:
    burn_image = cv2.bitwise_and(image, image, mask=burn_mask)  # Only burn areas
    healthy_skin_image = cv2.bitwise_and(image, image, mask=healthy_skin_mask)  # Only healthy skin areas

    # **Apply the mask to the original image**: Mask the image by keeping only valid pixels
    mask = np.where(final_burn_mask == 0, 255, 0).astype(np.uint8)  # Set burn pixels to 255
    combined_image = cv2.add(burn_image, healthy_skin_image)  # Add burn and healthy skin areas

    result_image = cv2.bitwise_and(image, image, mask=mask)

    return image, segmented_image, combined_image, burn_image, healthy_skin_image


# Loop through all images in the dataset directory
for filename in os.listdir(dataset_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(dataset_directory, filename)

        # Process the image
        original_image, segmented_image, result_image, burn_image, healthy_skin_image  = process_image(image_path)

        # Display results for each image in separate plots
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Original Image: {filename}")
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.imshow(segmented_image, cmap='gray')
        plt.title(f"FCM Segmentation: {filename}")
        plt.show()

        # Display results
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Only Healhty skin and burn area: {filename}")
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(burn_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Burn Area Masked: {filename}")
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(healthy_skin_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Healthy Skin Masked: {filename}")
        plt.show()
