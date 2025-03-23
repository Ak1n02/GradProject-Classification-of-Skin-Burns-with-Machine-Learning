import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def normalize(channel):
    return (channel - np.min(channel)) / (np.max(channel) - np.min(channel)) if np.max(channel) - np.min(channel) != 0 else 0

def get_hsv_features(folder_index, img):

        file_names_first_degree = ['../Dataset_Test_Eren/Graphs/HSV/Hue/FirstDegreeStatistics.txt', '../Dataset_Test_Eren/Graphs/HSV/Saturation/FirstDegreeStatistics.txt', '../Dataset_Test_Eren/Graphs/HSV/Value//FirstDegreeStatistics.txt']
        file_names_second_degree = ['../Dataset_Test_Eren/Graphs/HSV/Hue/SecondDegreeStatistics.txt', '../Dataset_Test_Eren/Graphs/HSV/Saturation/SecondDegreeStatistics.txt', '../Dataset_Test_Eren/Graphs/HSV/Value/SecondDegreeStatistics.txt']
        file_names_third_degree = ['../Dataset_Test_Eren/Graphs/HSV/Hue/ThirdDegreeStatistics.txt', '../Dataset_Test_Eren/Graphs/HSV/Saturation/ThirdDegreeStatistics.txt', '../Dataset_Test_Eren/Graphs/HSV/Value/ThirdDegreeStatistics.txt']
        all_files = [file_names_first_degree, file_names_second_degree, file_names_third_degree]

        hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(hsv_img)

        mask = ~((h == 0) & (s == 0) & (v == 0))
        h_valid, s_valid, v_valid = h[mask], s[mask], v[mask]
        h_valid, s_valid, v_valid = normalize(h_valid), normalize(s_valid), normalize(v_valid) # normalize

        h_mean, s_mean, v_mean = np.mean(h_valid, dtype=np.float64), np.mean(s_valid, dtype=np.float64), np.mean(v_valid, dtype=np.float64)
        h_std, s_std, v_std = np.std(h_valid, dtype=np.float64), np.std(s_valid, dtype=np.float64), np.std(v_valid, dtype=np.float64)
        for index, file in enumerate(all_files[folder_index]):
            if index == 0:
                with open(file, 'a') as f:
                    f.write(f'Degree: {folder_index + 1}, Hue Mean: {h_mean:.2f}, Hue Std: {h_std:.2f}\n')
            elif index == 1:
                with open(file, 'a') as f:
                    f.write(f'Degree: {folder_index + 1}, Saturation Mean: {s_mean:.2f}, Saturation Std: {s_std:.2f}\n')
            elif index == 2:
                with open(file, 'a') as f:
                    f.write(f'Degree: {folder_index + 1}, Value Mean: {v_mean:.2f}, Value Std: {v_std:.2f}\n')

        print(f'Degree: {folder_index + 1}, Hue Mean: {h_mean:.2f}, Hue Std: {h_std:.2f}, Saturation Mean: {s_mean:.2f}, Saturation Std: {s_std:.2f}, Value Mean: {v_mean:.2f}, Value Std: {v_std:.2f}')
        return h_mean, h_std, s_mean, s_std, v_mean, v_std


def plot(folder_index, folder_path):
    x_axis = []  # x-axis
    feature_data = []  # y-axis
    labels = {'hsv': ('Hue Mean', 'Hue Std', 'Saturation Mean', 'Saturation Std', 'Value Mean', 'Value Std')}
    colors = {'hsv': ('b', 'g')}

    for index, filename in enumerate(os.listdir(folder_path)):
        if (filename.endswith(".jpg") or filename.endswith(".png")) and filename.startswith('burn'):
            image_path = os.path.join(folder_path, filename)
            img = cv.imread(image_path, cv.COLOR_BGR2RGB)
            data = get_hsv_features(folder_index, img)  # Assuming this function returns the feature data
            feature_data.append(data)
            x_axis.append(index + 1)

    # Transpose so that each feature is in a separate list
    feature_data = np.array(feature_data).T

    x = np.arange(len(x_axis))  # X-axis positions

    # Create subplots for each pair of features (e.g., Hue Mean and Hue Std, Saturation Mean and Saturation Std, etc.)
    num_features = len(labels["hsv"])
    num_pairs = num_features // 2

    # Loop through each feature pair and plot them on separate figures
    for i in range(num_pairs):
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot the feature pair (e.g., Hue Mean and Hue Std)
        ax.bar(x - 0.2, feature_data[2*i], 0.4, label=labels["hsv"][2*i], color=colors["hsv"][0])
        ax.bar(x + 0.2, feature_data[2*i + 1], 0.4, label=labels["hsv"][2*i + 1], color=colors["hsv"][1])

        # Set axis labels and title for each plot
        ax.set_xticks(x)
        ax.set_xticklabels(x_axis, rotation=45, fontsize=10)
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Feature Value')
        ax.set_title(f'{labels["hsv"][2*i]} and {labels["hsv"][2*i + 1]} for Images {folder_index + 1}.Degree')
        ax.legend()

        plt.tight_layout()
        plt.show()

def main():
    folder_paths = ['../Dataset_Test_Eren/FirstDegreeSegmented', '../Dataset_Test_Eren/SecondDegreeSegmented', '../Dataset_Test_Eren/ThirdDegreeSegmented']
    for index, folder_path in enumerate(folder_paths):
        plot(index, folder_path)

if __name__ == "__main__":
    main()