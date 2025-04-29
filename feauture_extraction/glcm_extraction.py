import os
import cv2 as cv
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def get_glcm_features(image, distances=[1, 2], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):

    gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = gray_img < 255

    # Apply ROI
    x, y = np.where(mask)
    roi = gray_img[np.min(x):np.max(x) + 1, np.min(y):np.max(y) + 1]

    # Compute GLCM
    glcm = graycomatrix(
        roi,
        distances=distances,
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True
    )
    # Compute GLCM features
    contrast = graycoprops(glcm, 'contrast').mean()
    contrast_norm = min(contrast / 3000.0, 1.0)
    homogeneity = graycoprops(glcm, 'homogeneity').mean()
    correlation = graycoprops(glcm, 'correlation').mean()

    return contrast_norm, homogeneity, correlation

def save_glcm_features(folder_index, folder_path):

    output_folder_paths = ['../Dataset_Test_Eren/Graphs/Contrast', '../Dataset_Test_Eren/Graphs/Homogeneity', '../Dataset_Test_Eren/Graphs/Correlation']
    text_files = ['FirstDegreeStatistics.txt', 'SecondDegreeStatistics.txt', 'ThirdDegreeStatistics.txt']
    for files in os.listdir(folder_path):
        if files.endswith('.jpg') or files.endswith('.png'):
            img = cv.imread(os.path.join(folder_path, files))
            contrast, homogeneity, correlation = get_glcm_features(img)
            for index, output_folder in enumerate(output_folder_paths):
                with open(os.path.join(output_folder, text_files[folder_index]), 'a') as f:
                    if index == 0:
                        f.write(f"Degree: {folder_index + 1}, Contrast: {contrast:.2f}\n")
                    elif index == 1:
                        f.write(f"Degree: {folder_index + 1}, Homogeneity: {homogeneity:.2f}\n")
                    else:
                        f.write(f"Degree: {folder_index + 1}, Correlation: {correlation:.2f}\n")

def main():
    folder_paths = ['../new_final_preprocessed_data_set/first_degree', '../new_final_preprocessed_data_set/second_degree', '../new_final_preprocessed_data_set/third_degree']
    for index, folder_path in enumerate(folder_paths):
        save_glcm_features(index, folder_path)


if __name__ == '__main__':
    main()