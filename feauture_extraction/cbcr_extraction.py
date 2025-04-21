import os
import cv2 as cv
import numpy as np
from hsv import normalize

def get_features(img):

    data = []
    y, cr, cb = cv.split(img)
    mask = ~((y == 255) & (cr == 128) & (cb == 128))
    y_valid, cr_valid, cb_valid = y[mask], cr[mask], cb[mask]
    y_valid_mean, y_valid_std = np.mean(normalize(y_valid), dtype=np.float64), np.std(normalize(y_valid), dtype=np.float64)
    cb_valid_mean, cb_valid_std = np.mean(normalize(cb_valid), dtype=np.float64), np.std(normalize(cb_valid), dtype=np.float64)
    cr_valid_mean, cr_valid_std = np.mean(normalize(cr_valid), dtype=np.float64), np.std(normalize(cr_valid), dtype=np.float64)
    data.extend([y_valid_mean, y_valid_std, cr_valid_mean, cr_valid_std, cb_valid_mean, cb_valid_std])
    return data

def save_features(folder_index, data):
    folders = ['../Dataset_Test_Eren/Graphs/YCrCb/Y', '../Dataset_Test_Eren/Graphs/YCrCb/Cr', '../Dataset_Test_Eren/Graphs/YCrCb/Cb']
    text_files = ['FirstDegreeStatistics.txt', 'SecondDegreeStatistics.txt', 'ThirdDegreeStatistics.txt']
    for index, folder in enumerate(folders):
        if index == 0:
            with open(os.path.join(folder, text_files[folder_index]), 'a') as f:
                f.write(f'Degree: {folder_index + 1}, Y Mean: {data[0]:.2f}, Y Std: {data[1]:.2f}\n')
        elif index == 1:
            with open(os.path.join(folder, text_files[folder_index]), 'a') as f:
                f.write(f'Degree: {folder_index + 1}, Cr Mean: {data[2]:.2f}, Cr Std: {data[3]:.2f}\n')
        elif index == 2:
            with open(os.path.join(folder, text_files[folder_index]), 'a') as f:
                f.write(f'Degree: {folder_index + 1}, Cb Mean: {data[4]:.2f}, Cb Std: {data[5]:.2f}\n')

def process_images(folder_index, folder_path):

    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = cv.imread(image_path, cv.COLOR_BGR2YCrCb)
            datas = get_features(img)
            save_features(folder_index, datas)

def main():
    folder_paths = ['../new_final_preprocessed_data_set/first_degree', '../new_final_preprocessed_data_set/second_degree', '../new_final_preprocessed_data_set/third_degree']
    for index, folder_path in enumerate(folder_paths):
        process_images(index, folder_path)




if __name__ == "__main__":
    main()