import cv2
import numpy as np
import torch  # For GPU acceleration
import os
import matplotlib.pyplot as plt

# Fuzzy C-Means implementation using PyTorch
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

    # Extract the Cr component
    _, Cr, _ = cv2.split(image_ycrcb)

    # Normalize the Cr component to [0, 1] for clustering
    Cr_normalized = Cr / 255.0

    # Reshape Cr into a 1D array for clustering
    Cr_reshaped = Cr_normalized.flatten()

    # Apply Fuzzy C-Means clustering
    n_clusters = 3
    centers, membership = fuzzy_c_means_gpu_torch(Cr_reshaped, n_clusters)

    # Assign pixels to the cluster with the highest membership
    segmented_image = np.argmax(membership, axis=1).reshape(Cr.shape)

    # Sort clusters by mean Cr value
    cluster_means = [np.mean(Cr[segmented_image == i]) for i in range(n_clusters)]
    sorted_clusters = np.argsort(cluster_means)
    background_label = sorted_clusters[0]
    healthy_skin_label = sorted_clusters[1]
    burn_1st_label = sorted_clusters[2]
    #burn_2nd_label = sorted_clusters[3]
    #burn_3rd_label = sorted_clusters[4]

    # Map the clusters to fixed labels (background, skin, burn)
    fixed_labels = np.copy(segmented_image)
    fixed_labels[segmented_image == background_label] = 0  # Background
    fixed_labels[segmented_image == healthy_skin_label] = 1  # Healthy Skin
    fixed_labels[segmented_image == burn_1st_label] = 2  # 1st Degree Burn
    #fixed_labels[segmented_image == burn_2nd_label] = 3  # 2nd Degree Burn
    #fixed_labels[segmented_image == burn_3rd_label] = 4  # 3rd Degree Burn

    # Create binary masks for each class
    mask_healthy_skin = (fixed_labels == 1)  # Healthy Skin
    mask_1st_degree = (fixed_labels == 2)  # 1st Degree Burn
    mask_2nd_degree = (fixed_labels == 3)  # 2nd Degree Burn
    mask_3rd_degree = (fixed_labels == 4)  # 3rd Degree Burn

    # Combine masks for visualization or processing
    combined_mask = np.logical_or.reduce(
        ( mask_1st_degree)
    )

    # Create a new image where the background is removed
    result_image = image.copy()
    result_image[mask_healthy_skin] = 0  # Set background pixels to black

    return image, segmented_image, result_image

# Loop through all images in the dataset directory
for filename in os.listdir(dataset_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(dataset_directory, filename)

        # Process the image
        original_image, segmented_image, result_image = process_image(image_path)

        # Display results for each image in separate plots
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Original Image: {filename}")
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.imshow(segmented_image, cmap='gray')
        plt.title(f"FCM Segmentation: {filename}")
        plt.show()

        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Foreground Only: {filename}")
        plt.show()