import os

import pandas as pd

def convert_to_csv(path, file_name, feature_type, type_number):
    labels = {'hue': ('Hue Mean', 'Hue Std'), 'kurtosis': ('Kurtosis A', 'Kurtosis B'), 'skewness': ('Skewness A', 'Skewness B')}
    data = []
    with open(path, 'r') as file:
        for line in file:
            parts = line.split(',')
            degree = parts[0].split(':')[1].strip()
            feature_one = parts[1].split(':')[1].strip()
            feature_two = parts[2].split(':')[1].strip()
            data.append([degree, feature_one, feature_two])

    df = pd.DataFrame(data, columns=['Degree', labels[feature_type][0], labels[feature_type][1]])
    if feature_type == 'hue':
        df.to_csv(f'../Dataset_Test_Eren/Graphs/{feature_type.capitalize()}/{file_name.split(".")[0]}', index=False)
    elif feature_type == 'kurtosis':
        if type_number == 1:
            df.to_csv(f'../Dataset_Test_Eren/Graphs/{feature_type.capitalize()}/Fisher (Normal)/{file_name.split(".")[0]}', index=False)
        else:
            df.to_csv(f'../Dataset_Test_Eren/Graphs/{feature_type.capitalize()}/Article/{file_name.split(".")[0]}', index=False)
    else:
        if type_number == 1:
            df.to_csv(f'../Dataset_Test_Eren/Graphs/{feature_type.capitalize()}/Default/{file_name.split(".")[0]}', index=False)
        else:
            df.to_csv(f'../Dataset_Test_Eren/Graphs/{feature_type.capitalize()}/Article/{file_name.split(".")[0]}', index=False)

def arrange_folders():
    folder_paths = ['../Dataset_Test_Eren/Graphs/Hue', '../Dataset_Test_Eren/Graphs/Kurtosis', '../Dataset_Test_Eren/Graphs/Skewness']
    feature_types = ['hue', 'kurtosis', 'skewness']
    for index, folder_path in enumerate(folder_paths):
        if index == 0:
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.txt'):
                    convert_to_csv(f'{folder_path}/{file_name}', file_name, feature_types[index], 0)
        elif index == 1:
            inner_folder = ['Fisher (Normal)', 'Article']
            for inner_index, folder in enumerate(inner_folder):
                for file_name in os.listdir(f'{folder_path}/{folder}'):
                    if file_name.endswith('.txt'):
                        convert_to_csv(f'{folder_path}/{folder}/{file_name}', file_name, feature_types[index], 1 if inner_index == 0 else 2)
        elif index == 2:
            inner_folder = ['Default', 'Article']
            for inner_index, folder in enumerate(inner_folder):
                for file_name in os.listdir(f'{folder_path}/{folder}'):
                    if file_name.endswith('.txt'):
                        convert_to_csv(f'{folder_path}/{folder}/{file_name}', file_name, feature_types[index], 1 if inner_index == 0 else 2)


