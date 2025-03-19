import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def get_hue(folder_index, img):

    # Load image and convert to LAB color space
    lab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab_img)

    # Mask out black pixels
    mask = ~((l == 0) & (a == 128) & (b == 128))
    a_valid = a[mask]
    b_valid = b[mask]

    # Remove zero a*,b* pairs to avoid division by zero in atan2
    valid_mask = (a_valid != 0) | (b_valid != 0)
    a_valid = a_valid[valid_mask]
    b_valid = b_valid[valid_mask]

    # Compute hue values from a*,b* pairs and normalize to [0, 1]
    hue_values = np.arctan2(b_valid, a_valid) * (180 / np.pi) # Convert from radians to degrees
    denom = np.max(hue_values) - np.min(hue_values)
    if denom == 0:
        return np.min(hue_values), 0
    else:
        hue_values = (hue_values - np.min(hue_values)) / denom  # Normalize to [0, 1]

    # Ensure there are valid values before computing mean and std
    if len(hue_values) == 0:
        hue_mean, hue_std = 0, 0
    else:
        hue_mean = np.mean(hue_values, dtype=np.float64)
        hue_std = np.std(hue_values, dtype=np.float64)
        print(f"Degree: {folder_index + 1} Hue Mean: {hue_mean:.2f}, Hue Std: {hue_std:.2f}")

    return hue_mean, hue_std

def plot_information(folder_index, folder_path):

    hue_data = [] # y-axis
    x_axis = [] # x-axis
    for index, filename in enumerate(os.listdir(folder_path)):
        if (filename.endswith(".jpg") or filename.endswith(".png")) and filename.startswith('burn'):
            image_path = os.path.join(folder_path, filename)
            image = cv.imread(image_path)
            hue_mean, hue_std = get_hue(folder_index, image)
            hue_data.append((hue_mean, hue_std))
            x_axis.append(index + 1)

    # Extract hue mean and std from hue_data
    hue_means = [data[0] for data in hue_data]
    hue_stds = [data[1] for data in hue_data]

    # Plot the hue mean and std for each image
    x = np.arange(len(x_axis))
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the bars for hue mean and hue std
    ax.bar(x - 0.2, hue_means, 0.4, label='Hue Mean', color='b')
    ax.bar(x + 0.2, hue_stds, 0.4, label='Hue Std', color='g')

    # Set the x-ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(x_axis, rotation=90, fontsize=10)
    ax.set_xlabel('Image')
    ax.set_ylabel('Value')
    ax.set_title(f'Hue Mean and Standard Deviation for Each Image {folder_index + 1}.Degree')

    # Show legend
    ax.legend()

    # Display the plot
    plt.tight_layout()
    plt.show()

def main():
    folder_paths = ['Dataset_Test_Eren/FirstDegreeSegmented', 'Dataset_Test_Eren/SecondDegreeSegmented', 'Dataset_Test_Eren/ThirdDegreeSegmented']
    for index, folder_path in enumerate(folder_paths):
        plot_information(index, folder_path)

if __name__ == '__main__':
    main()
