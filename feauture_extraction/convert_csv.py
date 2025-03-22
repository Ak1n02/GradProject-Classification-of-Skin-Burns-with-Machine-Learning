import os

import pandas as pd

def collect_all(folder_paths,type_kurtosis, type_skewness):

    hue_data, kurtosis_data, skewness_data = pd.read_csv(f'{folder_paths[0]}/General.csv'), [], []
    if type_kurtosis == 1:
        kurtosis_data = pd.read_csv(f'{folder_paths[1]}/Fisher (Normal)/General.csv')
    else:
        kurtosis_data = pd.read_csv(f'{folder_paths[1]}/Article/General.csv')
    if type_skewness == 1:
        skewness_data = pd.read_csv(f'{folder_paths[2]}/Default/General.csv')
    else:
        skewness_data = pd.read_csv(f'{folder_paths[2]}/Article/General.csv')

    kurtosis_data = kurtosis_data.loc[:, ~kurtosis_data.columns.isin(hue_data.columns)]
    skewness_data = skewness_data.loc[:, ~skewness_data.columns.isin(hue_data.columns)]

    data = pd.concat([hue_data, kurtosis_data, skewness_data], axis=1)
    data.to_csv(f'../Dataset_Test_Eren/Graphs/Dataset_Vector.csv', index=False)

def collect_degrees(folder_paths, type_kurtosis, type_skewness):
    for index,folder_path in enumerate(folder_paths):
        data = []
        if index == 0:
            for files in os.listdir(folder_path):
                if files.endswith('.csv') and files != 'General.csv':
                    hue_data = pd.read_csv(f'{folder_path}/{files}')
                    data.append(hue_data)
            combined_data = pd.concat(data, ignore_index=True)
            combined_data.to_csv(f'{folder_path}/General.csv', index=False)
        elif index == 1:
            if type_kurtosis == 1:
                for files in os.listdir(f'{folder_path}/Fisher (Normal)'):
                    if files.endswith('.csv') and files != 'General.csv':
                        kurtosis_data = pd.read_csv(f'{folder_path}/Fisher (Normal)/{files}')
                        data.append(kurtosis_data)
            else:
                for files in os.listdir(f'{folder_path}/Article'):
                    if files.endswith('.csv') and files != 'General.csv':
                        kurtosis_data = pd.read_csv(f'{folder_path}/Article/{files}')
                        data.append(kurtosis_data)

            combined_data = pd.concat(data, ignore_index=True)
            combined_data.to_csv(f'{folder_path}/{"Fisher (Normal)" if type_kurtosis == 1 else "Article"}/General.csv', index=False)
        else:
            if type_skewness == 1:
                for files in os.listdir(f'{folder_path}/Default'):
                    if files.endswith('.csv') and files != 'General.csv':
                        skewness_data = pd.read_csv(f'{folder_path}/Default/{files}')
                        data.append(skewness_data)
            else:
                for files in os.listdir(f'{folder_path}/Article'):
                    if files.endswith('.csv') and files != 'General.csv':
                        skewness_data = pd.read_csv(f'{folder_path}/Article/{files}')
                        data.append(skewness_data)
            combined_data = pd.concat(data, ignore_index=True)
            combined_data.to_csv(f'{folder_path}/{"Default" if type_kurtosis == 1 else "Article"}/General.csv', index=False)

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
        df.to_csv(f'../Dataset_Test_Eren/Graphs/{feature_type.capitalize()}/{file_name.split(".")[0]}.csv', index=False)
    elif feature_type == 'kurtosis':
        if type_number == 1:
            df.to_csv(f'../Dataset_Test_Eren/Graphs/{feature_type.capitalize()}/Fisher (Normal)/{file_name.split(".")[0]}.csv', index=False)
        else:
            df.to_csv(f'../Dataset_Test_Eren/Graphs/{feature_type.capitalize()}/Article/{file_name.split(".")[0]}.csv', index=False)
    else:
        if type_number == 1:
            df.to_csv(f'../Dataset_Test_Eren/Graphs/{feature_type.capitalize()}/Default/{file_name.split(".")[0]}.csv', index=False)
        else:
            df.to_csv(f'../Dataset_Test_Eren/Graphs/{feature_type.capitalize()}/Article/{file_name.split(".")[0]}.csv', index=False)

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


if __name__ == '__main__':
    folders = ['../Dataset_Test_Eren/Graphs/Hue', '../Dataset_Test_Eren/Graphs/Kurtosis', '../Dataset_Test_Eren/Graphs/Skewness']
#    collect_degrees(folders, 2, 2)
    collect_all(folders, 2, 2)
