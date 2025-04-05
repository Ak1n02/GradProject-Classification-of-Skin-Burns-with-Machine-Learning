import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import kurtosis, skew


def get_features(folder_index, img, feature_type, type_number=1):

    # Save the statistics to text files
    text_files = ['FirstDegreeStatistics.txt', 'SecondDegreeStatistics.txt', 'ThirdDegreeStatistics.txt']
    folders = {'hue': '../Dataset_Test_Eren/Graphs/Hue', 'kurtosis_feature': ('../Dataset_Test_Eren/Graphs/Kurtosis/Fisher (Normal)', '../Dataset_Test_Eren/Graphs/Kurtosis/Article'), 'skewness': ('../Dataset_Test_Eren/Graphs/Skewness/Default', '../Dataset_Test_Eren/Graphs/Skewness/Article')}

    # Load image and convert to LAB color space
    lab_img = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab_img)

    # Mask out white pixels
    mask = ~((l == 255) & (a == 128) & (b == 128))
    a_valid = a[mask]
    b_valid = b[mask]

    # Remove zero a*,b* pairs to avoid division by zero in atan2
    valid_mask = (a_valid != 0) | (b_valid != 0)
    a_valid = a_valid[valid_mask]
    b_valid = b_valid[valid_mask]

    # Compute mean and std of a* and b* values
    mean_val_a, std_val_a = np.mean(a_valid, dtype=np.float64), np.std(a_valid, dtype=np.float64)
    mean_val_b, std_val_b = np.mean(b_valid, dtype=np.float64), np.std(b_valid, dtype=np.float64)

    statement = ''
    if feature_type == 'hue':
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

        statement = f"Degree: {folder_index + 1}, Hue Mean: {hue_mean:.2f}, Hue Std: {hue_std:.2f}"
        return os.path.join(folders[feature_type], text_files[folder_index]), statement


    elif feature_type == 'kurtosis_feature':
        if type_number == 1:
            kurtosis_a, kurtosis_b = kurtosis(a_valid, fisher=True), kurtosis(b_valid, fisher=True)
        else:
            fourth_moment_a, fourth_moment_b = np.mean((a_valid - mean_val_a) ** 4), np.mean((b_valid - mean_val_b) ** 4)
            kurtosis_a = (fourth_moment_a / (std_val_a * mean_val_a)) if std_val_a != 0 else 0
            kurtosis_b = (fourth_moment_b / (std_val_b * mean_val_b)) if std_val_b != 0 else 0

        statement = f"Degree: {folder_index + 1}, Kurtosis A: {kurtosis_a:.2f}, Kurtosis B: {kurtosis_b:.2f}"
        return os.path.join(folders[feature_type][type_number - 1], text_files[folder_index]), statement


    elif feature_type == 'skewness':
        if type_number == 1:
            if (std_val_a == 0 or len(a_valid) < 2) and (std_val_b == 0 or len(b_valid) < 2):
                skewness_a, skewness_b  = 0, 0
            elif std_val_a == 0 or len(a_valid) < 2:
                skewness_a, skewness_b  = 0, skew(b_valid)
            elif std_val_b == 0 or len(b_valid) < 2:
                skewness_a, skewness_b  = skew(a_valid), 0
            else:
                skewness_a, skewness_b  = skew(a_valid), skew(b_valid)
        else:
            skewness_a = np.mean((a_valid - mean_val_a) ** 3) / (std_val_a ** 3) if std_val_a != 0 and len(a_valid) >= 2 else 0
            skewness_b = np.mean((b_valid - mean_val_b) ** 3) / (std_val_b ** 3) if std_val_b != 0 and len(b_valid) >= 2 else 0
        statement = f"Degree: {folder_index + 1}, Skewness A: {skewness_a:.2f}, Skewness B: {skewness_b:.2f}"
        return os.path.join(folders[feature_type][type_number - 1], text_files[folder_index]), statement


def save_statistics(file_path, statement):
    with open(file_path, 'a') as f:
        f.write(statement + '\n')

def plot_information(folder_index, folder_path, feature_type):

#    x_axis = []  # x-axis
#    feature_data = [] # y-axis
#    labels = {'hue': ('Hue Mean', 'Hue Std'), 'kurtosis_feature': ('Kurtosis A', 'Kurtosis B'), 'skewness': ('Skewness A', 'Skewness B')}
#    colors = {'hue': ('b', 'g'), 'kurtosis_feature': ('b', 'g'), 'skewness': ('b', 'g')}

    for index, filename in enumerate(os.listdir(folder_path)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            image = cv.imread(image_path)
            file_path, statement = get_features(folder_index, image, feature_type, 1)
            save_statistics(file_path, statement)
#            feature_data.append(data)
#            x_axis.append(index + 1)

"""
        # Extract hue mean and std from hue_data
        feature_1 = [data[0] for data in feature_data]
        feature_2 = [data[1] for data in feature_data]
    
        # Plot the hue mean and std for each image
        x = np.arange(len(x_axis))
        fig, ax = plt.subplots(figsize=(10, 6))
    
        # Plot the bars for hue mean and hue std
        ax.bar(x - 0.2, feature_1, 0.4, label=labels[feature_type][0], color=colors[feature_type][0])
        ax.bar(x + 0.2, feature_2, 0.4, label=labels[feature_type][1], color=colors[feature_type][1])
    
        # Set the x-ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(x_axis, rotation=90, fontsize=10)
        ax.set_xlabel('Image')
        ax.set_ylabel('Value')
        ax.set_title(f'{labels[feature_type][0]} and {labels[feature_type][1]} for Each Image {folder_index + 1}.Degree')
        ax.legend()
    
        # Display the plot
        plt.tight_layout()
        plt.show() """


def main():
    folder_paths = ['../new_final_preprocessed_data_set/first_degree', '../new_final_preprocessed_data_set/second_degree', '../new_final_preprocessed_data_set/third_degree']
    for index, folder_path in enumerate(folder_paths):
        plot_information(index, folder_path, 'skewness')

if __name__ == '__main__':
    main()
